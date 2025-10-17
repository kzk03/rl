# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Gerrit Retention IRL Project**: A research repository for developer retention prediction and task assignment optimization using Reinforcement Learning (RL) and Inverse Reinforcement Learning (IRL) on OpenStack Gerrit review history data.

The core innovation is **temporal IRL with LSTM** to predict whether reviewers will continue contributing to OSS projects, achieving AUC-ROC 0.868 on real OpenStack data (137,632 reviews, 13 years).

## Essential Commands

### Environment Setup
```bash
# Install all dependencies
uv sync

# Install dev dependencies (optional)
uv sync --all-extras
```

### Running Core Workflows

#### 0. Data Preprocessing (Recommended)

**IMPORTANT**: Filter bot accounts before training to improve data quality (removes 44% noise).

```bash
# Remove bot accounts (recommended)
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_no_bots.csv

# Filter by specific projects
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --output data/review_requests_nova_neutron.csv \
  --projects "openstack/nova" "openstack/neutron"

# Or extract top N projects automatically
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --output data/review_requests_top3.csv \
  --top 3

# Split by project (for automated per-project analysis)
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --split-by-project \
  --output-dir data/projects/
```

See [docs/DATA_FILTERING_GUIDE.md](docs/DATA_FILTERING_GUIDE.md) for details.

#### 1. Temporal IRL Training and Evaluation

**IMPORTANT**: All evaluation results should be saved to `importants/` directory for easy tracking.

```bash
# Quick test with sample data
uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
  --reviews data/sample_reviews.csv \
  --snapshot-date 2021-01-01 \
  --history-months 3 6 9 \
  --target-months 3 6 9 \
  --epochs 10 \
  --sequence \
  --seq-len 10 \
  --output importants/irl_test

# Production evaluation with real data
uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --snapshot-date 2020-01-01 \
  --history-months 3 6 9 12 \
  --target-months 3 6 9 12 \
  --epochs 20 \
  --sequence \
  --seq-len 15 \
  --output importants/irl_openstack_real
```

#### 2. Project-Aware IRL Evaluation
```bash
# Cross-project learning (recommended for best accuracy)
uv run python scripts/training/irl/train_temporal_irl_project_aware.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --snapshot-date 2023-01-01 \
  --history-months 6 12 \
  --target-months 6 12 \
  --mode cross-project \
  --sequence \
  --seq-len 15 \
  --epochs 30 \
  --output importants/irl_project_cross

# Project-specific learning
uv run python scripts/training/irl/train_temporal_irl_project_aware.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --mode project-specific \
  --projects "openstack/nova" "openstack/neutron" \
  --sequence \
  --output importants/irl_project_specific
```

#### 3. Traditional Workflows
```bash
# Build review request data
uv run python examples/build_eval_from_review_requests.py \
  --input data/raw/review_requests_openstack.json \
  --output data/review_requests_openstack_w14.csv

# Extract action matches
uv run python data_processing/extract_action_match.py \
  --input data/review_requests_openstack_w14.csv \
  --output data/processed/action_match.jsonl

# Train IRL model (offline RL)
uv run python scripts/training/offline_rl/train_retention_irl_from_offline.py \
  --config configs/rl_config.yaml \
  --output-dir outputs/irl_model_latest

# Task assignment replay evaluation
uv run python scripts/evaluation/task_assignment_replay_eval.py \
  --tasks outputs/task_assign_multilabel/2023_cutoff_full/tasks_eval.jsonl \
  --cutoff 2023-07-01T00:00:00 \
  --eval-mode irl \
  --irl-model outputs/task_assign_multilabel/2023_cutoff_full/irl_model.json \
  --windows "0m-3m,3m-6m,6m-9m,9m-12m" \
  --csv-dir outputs/task_assign_multilabel/2023_cutoff_full/eval_detail \
  --out outputs/task_assign_multilabel/2023_cutoff_full/replay_eval_eval.json
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=gerrit_retention --cov-report=html

# Run specific test file
uv run pytest tests/test_retention_irl.py

# Run tests by marker
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m integration  # Only integration tests
```

### Code Quality
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/gerrit_retention
```

### API Server
```bash
# Start FastAPI server
uv run python api/main.py

# Development mode with auto-reload
uv run uvicorn api.main:app --reload
```

## Architecture

### Core Components

#### 1. Temporal IRL System (`src/gerrit_retention/rl_prediction/retention_irl_system.py`)

**Key Innovation**: LSTM-based temporal learning for reviewer continuation prediction.

**Architecture**:
```
Input: Temporal trajectory [batch, seq_len, feature_dim]
  ↓
State Encoder (Linear → ReLU → Linear → ReLU)
  ↓
Action Encoder (Linear → ReLU → Linear → ReLU)
  ↓
Combined (Addition)
  ↓
LSTM (1-layer, hidden_size=128)  ← Core temporal component
  ↓
├─ Reward Predictor (Linear → ReLU → Linear)
└─ Continuation Predictor (Linear → ReLU → Linear → Sigmoid)
```

**Critical Features**:
- **Sequence Mode** (`self.sequence`): When True, processes entire trajectories with LSTM
- **Padding/Truncation**: Handles variable-length sequences
  - Padding: Repeats first action when `len(actions) < seq_len`
  - Truncation: Uses latest `seq_len` actions when `len(actions) >= seq_len`
- **State Features** (10-dim): Experience, activity frequency, collaboration score, etc.
- **Action Features** (5-dim): Action type, intensity, quality, collaboration

**Key Methods**:
- `train_irl()`: Trains model on expert trajectories (line 306)
- `predict_continuation_probability()`: Predicts if reviewer will continue (line 441)
- `save_model()` / `load_model()`: Model persistence (line 720, 732)

#### 2. Sliding Window Evaluation (`scripts/training/irl/train_temporal_irl_sliding_window.py`)

Automated evaluation across multiple learning/prediction period combinations.

**Workflow**:
1. Load review logs
2. For each (history_months, target_months) combination:
   - Extract trajectories with sliding window
   - Train IRL model
   - Evaluate on test set
   - Save model and metrics
3. Generate matrix-format reports

**Output Structure**:
```
importants/irl_openstack_real/
├── models/
│   ├── irl_h3m_t3m_seq.pth
│   ├── irl_h12m_t6m_seq.pth  ← Best model (AUC-ROC 0.855)
│   └── ... (16 models total)
├── sliding_window_results_seq.csv
├── evaluation_matrix_seq.txt
└── EVALUATION_REPORT.md
```

#### 3. Project-Aware Evaluation (`scripts/training/irl/train_temporal_irl_project_aware.py`)

Two modes for project-specific analysis:

**Cross-Project Mode** (Recommended):
- Trains on all projects combined
- Predicts continuation **within the same project**
- Creates trajectories per (reviewer, project) pair
- Achieves highest accuracy (AUC-ROC 0.868)

**Project-Specific Mode**:
- Trains on single project data only
- Useful for project-specific patterns
- Lower accuracy but captures project culture

**Continuation Label Logic**:
```python
# Critical: Continuation is project-specific
for reviewer in reviewers:
    for project in active_projects:
        # Only counts activity in THIS project
        continued = has_activity_in_same_project(
            reviewer, project, prediction_period
        )
```

### Data Flow

```
Raw Gerrit Data (JSON/CSV)
  ↓
data_processing/
  ├── gerrit_extraction/     # Extract from Gerrit API
  ├── preprocessing/         # Clean and normalize
  └── feature_engineering/   # Generate features
  ↓
Training Data (trajectories)
  ↓
scripts/training/
  ├── irl/                   # Temporal IRL training
  ├── rl_training/           # RL policy training
  └── offline_rl/            # Offline RL methods
  ↓
Trained Models (.pth files)
  ↓
scripts/evaluation/          # Replay evaluation
  ↓
Evaluation Results (JSON/CSV/figures)
  ↓
analysis/                    # Statistical analysis & visualization
```

### Module Organization

```
src/gerrit_retention/
├── rl_prediction/           # Core IRL/RL systems
│   ├── retention_irl_system.py    # Temporal IRL (★ most important)
│   └── retention_rl_system.py     # RL policy
├── irl/                     # IRL algorithms
│   └── maxent_binary_irl.py
├── rl_environment/          # Gymnasium environments
├── recommendation/          # Reviewer recommendation
├── behavior_analysis/       # Developer behavior analysis
├── visualization/           # Charts and dashboards
├── analysis/                # Reports and analytics
├── data_integration/        # Data loading
└── utils/                   # Shared utilities
```

## Critical Implementation Details

### 1. Sequence Length (`seq_len`)

**Default**: 15 (recommended for OpenStack data)

**Data Distribution** (OpenStack):
- 25th percentile: 3 actions
- 50th percentile: 7 actions
- 75th percentile: 15 actions
- 90th percentile: 31 actions

**Choosing seq_len**:
- Too short (< 10): Loses temporal context
- Optimal (10-15): Balances coverage and efficiency
- Too long (> 20): Diminishing returns, slower training

See `docs/seq_len_explanation.md` for detailed analysis.

### 2. Snapshot Date Selection

**Critical**: Must have sufficient data in both learning and prediction windows.

**Good Practice**:
```bash
# Check data availability first
python -c "
import pandas as pd
df = pd.read_csv('data/your_data.csv')
print(f'Date range: {df[\"request_time\"].min()} to {df[\"request_time\"].max()}')
print(f'Total reviews: {len(df)}')
"

# Use a date with rich history
--snapshot-date 2020-01-01  # For OpenStack data
```

**Common Issue**: `ZeroDivisionError` means insufficient data for that date/period combination.

### 3. Model Naming Convention

Models are saved with descriptive names:

```
# Temporal IRL sliding window
irl_h{history}m_t{target}m_{seq|nosq}.pth
Example: irl_h12m_t6m_seq.pth

# Project-aware cross-project
irl_h{history}m_t{target}m_cross_seq.pth

# Project-aware project-specific
irl_h{history}m_t{target}m_{project_name}_seq.pth
Example: irl_h12m_t6m_openstack_nova_seq.pth
```

### 4. CSV Data Requirements

**Required Columns**:
- `reviewer_email` or `email`: Reviewer identifier
- `request_time` or `created`: Timestamp
- `project`: Project name (for project-aware mode)

**Optional but Recommended**:
- `lines_added`, `lines_deleted`, `files_changed`: For intensity calculation
- `message`: For quality scoring

### 5. Evaluation Metrics

**AUC-ROC**: Overall discrimination ability (range 0-1, higher is better)
- 0.9-1.0: Excellent
- 0.8-0.9: Good
- 0.7-0.8: Fair
- 0.5: Random

**AUC-PR**: Precision-recall (important for imbalanced data)
- High imbalance (8.5% continuation rate in OpenStack)
- Better metric than AUC-ROC for rare events

**F1 Score**: Harmonic mean of precision and recall
- Balances false positives and false negatives

**Best Results** (OpenStack data):
- AUC-ROC: 0.868 (12m learning × 6m prediction)
- AUC-PR: 0.983 (3m learning × 12m prediction)
- F1: 0.978 (3m learning × 12m prediction)

## Common Workflows

### Adding a New IRL Training Script

1. Import core system:
```python
from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem
```

2. Configure model:
```python
config = {
    'state_dim': 10,
    'action_dim': 5,
    'hidden_dim': 128,
    'learning_rate': 0.001,
    'sequence': True,        # Enable LSTM
    'seq_len': 15           # Sequence length
}
```

3. Train:
```python
irl_system = RetentionIRLSystem(config)
result = irl_system.train_irl(trajectories, epochs=30)
```

4. Evaluate:
```python
prediction = irl_system.predict_continuation_probability(
    developer=developer_info,
    activity_history=recent_activities,
    context_date=datetime.now()
)
```

### Loading a Trained Model

```python
from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

# Load model from importants directory
model = RetentionIRLSystem.load_model(
    'importants/irl_openstack_real/models/irl_h12m_t6m_seq.pth'
)

# Check configuration
checkpoint = torch.load('model.pth')
print(f"Sequence mode: {checkpoint['config'].get('sequence', False)}")
print(f"Sequence length: {checkpoint['config'].get('seq_len', 0)}")

# Predict
result = model.predict_continuation_probability(
    developer=developer_data,
    activity_history=activities
)

print(f"Continuation probability: {result['continuation_probability']:.1%}")
print(f"Reasoning: {result['reasoning']}")
```

### Debugging Data Issues

```python
# Check trajectory structure
import pandas as pd

df = pd.read_csv('data/your_data.csv')

# Verify required columns
required_cols = ['reviewer_email', 'request_time', 'project']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")

# Check date range
df['request_time'] = pd.to_datetime(df['request_time'])
print(f"Date range: {df['request_time'].min()} to {df['request_time'].max()}")

# Check for nulls
print(df[required_cols].isnull().sum())

# Activity distribution per reviewer
reviewer_counts = df.groupby('reviewer_email').size()
print(f"Median activities per reviewer: {reviewer_counts.median()}")
print(f"75th percentile: {reviewer_counts.quantile(0.75)}")
```

## Performance Considerations

### Memory Management

**Large Datasets**:
- Default seq_len=15 is optimized for memory
- Reduce seq_len to 10 if memory issues occur
- Batch size is auto-managed in training loop

**Typical Memory Usage**:
- OpenStack dataset (137K reviews): ~2GB RAM
- Training 16 models: ~4-6GB RAM total
- GPU not required but will accelerate training

### Execution Time

**Reference** (CPU, OpenStack data):
- Single model training (30 epochs): ~2-3 minutes
- Sliding window evaluation (16 combinations): ~4 minutes
- Project-aware evaluation: ~3-4 minutes per mode

**Optimization Tips**:
- Use `--epochs 10` for quick tests
- Reduce `--history-months` and `--target-months` options
- Use sample data for development

## Known Issues and Solutions

### Issue: "軌跡が不足しているためスキップ"
**Cause**: Insufficient data for snapshot date + learning period combination
**Solution**:
- Adjust `--snapshot-date` to a data-rich period
- Reduce `--history-months` (e.g., remove 18, 15)
- Check data availability with date range query

### Issue: LSTM shape mismatch
**Cause**: Model saved without sequence mode, loaded with sequence=True
**Solution**:
```python
# Check model config before loading
checkpoint = torch.load('model.pth')
config = checkpoint['config']
assert config.get('sequence') == True, "Model not trained in sequence mode"
```

### Issue: Low accuracy on specific project
**Cause**: Project has too little data or extreme continuation rate
**Solution**:
- Use cross-project mode instead
- Adjust learning/prediction periods
- Combine with related projects

## Output Directory Convention

**IMPORTANT**: All evaluation results and trained models should be saved to the `importants/` directory instead of `outputs/` for easy tracking and organization.

**Standard Structure**:
```
importants/
├── irl_openstack_real/           # Production temporal IRL results
├── irl_project_cross/            # Cross-project evaluation
├── irl_project_specific/         # Project-specific evaluation
└── irl_test/                     # Quick tests and experiments
```

**Why `importants/`**:
- Centralized location for all important evaluation results
- Easy to track and compare experiments
- Separates production results from temporary outputs
- Consistent across all training scripts

## Documentation References

- **README.md**: Project overview and basic workflows
- **README_TEMPORAL_IRL.md**: Complete temporal IRL guide with results
- **docs/seq_len_explanation.md**: Detailed seq_len parameter analysis
- **docs/project_aware_irl_evaluation.md**: Project-specific evaluation guide
- **importants/irl_openstack_real/EVALUATION_REPORT.md**: Full experimental results

## Design Philosophy

1. **Temporal Learning First**: Always use `--sequence` flag for temporal IRL
2. **Sliding Window Evaluation**: Test multiple period combinations to find optimal settings
3. **Project-Aware Prediction**: Continuation is judged within the same project
4. **Reproducibility**: All experiments should be reproducible with one command
5. **Comprehensive Reporting**: Generate matrix-format results for easy interpretation

## uv-Specific Notes

This project uses [uv](https://github.com/astral-sh/uv) as the package manager:

- All dependencies are in `pyproject.toml`
- Always prefix commands with `uv run`
- `uv sync` installs dependencies (much faster than pip)
- No need for separate virtual environment activation

