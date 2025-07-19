# Technology Stack

##　　日本語にして

## Build System & Package Management

- **Package Manager**: `uv` (Python package manager)
- **Build Configuration**: `pyproject.toml` with uv package management
- **Python Version**: >= 3.11

## Core Dependencies

### Machine Learning & AI

- **PyTorch**: >= 2.3 (deep learning framework)
- **PyTorch Geometric**: >= 2.5.0 (graph neural networks)
- **Stable Baselines3**: >= 2.3.0 (reinforcement learning algorithms)
- **Scikit-learn**: >= 1.7.0 (traditional ML utilities)

### Reinforcement Learning Environment

- **Gymnasium**: >= 0.29 (RL environment interface)
- **PettingZoo**: >= 1.25 (multi-agent RL environments)
- **Shimmy**: >= 2.0.0 (environment wrappers)
- **SuperSuit**: >= 3.10.0 (multi-agent environment utilities)

### Configuration & Utilities

- **Hydra**: >= 1.3.0 (configuration management)
- **OmegaConf**: >= 2.3.0 (configuration framework)
- **PyYAML**: >= 6.0.2 (YAML parsing)
- **tqdm**: >= 4.65.0 (progress bars)
- **python-dotenv**: >= 1.0.0 (environment variables)

### Development Tools

- **pytest**: >= 8.3.5 (testing framework)
- **ruff**: >= 0.11.6 (linting and formatting)

## Common Commands

### Environment Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate
```

### Training Pipeline

```bash
# Full training pipeline (GAT → IRL → RL)
python scripts/full_training_pipeline.py

# Production training
python scripts/full_training_pipeline.py --production

# Skip specific components
python scripts/full_training_pipeline.py --skip-gnn    # Skip GNN training
python scripts/full_training_pipeline.py --skip-irl    # Skip IRL training
python scripts/full_training_pipeline.py --skip-rl     # Skip RL training
```

### Individual Component Training

```bash
# GAT training
python training/gat/train_gat.py

# IRL training
python training/irl/train_irl.py

# RL training
python training/rl/train_rl.py
```

### Data Processing

```bash
# Generate training data
python data_processing/generate_backlog.py
python data_processing/generate_profiles.py
python data_processing/generate_graph.py
python data_processing/generate_labels.py
```

### Evaluation

```bash
# Evaluate models
python evaluation/evaluate_models.py

# 2022 test evaluation
python scripts/evaluate_2022_test.py --config configs/base_test_2022.yaml
```

### Analysis & Reports

```bash
# Generate comprehensive analysis
python analysis/reports/summary_report.py

# Individual analyses
python analysis/reports/irl_analysis.py
python analysis/reports/gat_analysis.py
python analysis/reports/collaboration_analysis.py
```

### Code Quality

```bash
# Run linter and formatter
uv run ruff check .
uv run ruff format .

# Run tests
pytest tests/
```

## Configuration Files

- **Training**: `configs/base_training.yaml`
- **Testing**: `configs/base_test_2022.yaml`
- **Production**: `configs/production.yaml`
- **Developer Profiles**: `configs/dev_profiles*.yaml`

## Docker Support

Basic Dockerfile available for containerized deployment using Python 3.9 slim base image.
