# True IRL Comprehensive Evaluation Plan

**Created**: 2025-10-19
**Script**: `scripts/training/irl/evaluate_true_irl_comprehensive.py`

## Objective

Comprehensive evaluation of the **True IRL system** (EnhancedRetentionIRLSystem with LSTM) to understand:
1. How well temporal IRL predicts developer retention
2. Optimal learning and prediction period combinations
3. Differences between single-project and multi-project learning

## Key Innovation: Training Data Period Matches Learning Period

**Critical Design Decision**: Unlike previous approaches, this evaluation aligns the IRL training data period with the learning period length.

### Why This Matters

In previous scripts, IRL training might use all available historical data regardless of the learning period. This new approach ensures consistency:

- **3-month learning period** → IRL trains on 3 months of data before snapshot
- **6-month learning period** → IRL trains on 6 months of data before snapshot
- **12-month learning period** → IRL trains on 12 months of data before snapshot

### Example Timeline

**IMPORTANT**: Prediction checks if reviewer is active AFTER the waiting period (not within it).

```
Snapshot Date: 2023-01-01

3-month configuration:
├─────────────┼─────────────┼─────────→
2022-10-01   2023-01-01   2023-04-01  (future...)
  ↑              ↑              ↑
  │              │              └─ Check start (3m after snapshot)
  │              │                 Label = 1 if active after this date
  │              └─ Snapshot (dividing point)
  └─ IRL training start (3m before snapshot)

Meaning: "Will the reviewer still be active 3 months later?"

12-month configuration:
├─────────────────────────────┼─────────────────────────────┼─────────→
2022-01-01                   2023-01-01                   2024-01-01  (future...)
  ↑                              ↑                              ↑
  │                              │                              └─ Check start (12m after)
  │                              │                                 Label = 1 if active after this
  │                              └─ Snapshot (dividing point)
  └─ IRL training start (12m before snapshot)

Meaning: "Will the reviewer still be active 12 months later?"
```

### Rationale

1. **Temporal Consistency**: The model learns from a period that matches what it will predict
2. **Fair Comparison**: Different learning periods use proportional amounts of data
3. **Realistic Simulation**: Mimics real-world scenario where you have limited historical data
4. **Avoids Data Leakage**: Clear separation between training and evaluation periods

## Dataset

**File**: `data/review_requests_openstack_multi_5y_detail.csv`

### Dataset Statistics

From `importants/true_irl_comprehensive/data_analysis/dataset_stats.json`:

- **Size**: 78.75 MB
- **Total Reviews**: 137,632
- **Date Range**: 2012-06-20 to 2025-09-27 (13.3 years)
- **Unique Reviewers**: 1,379
- **Projects**: 5
  - openstack/cinder: 71,604 reviews (52.0%)
  - openstack/nova: 36,127 reviews (26.2%)
  - openstack/neutron: 16,915 reviews (12.3%)
  - openstack/glance: 11,203 reviews (8.1%)
  - openstack/keystone: 1,783 reviews (1.3%)

### Data Characteristics

- **Continuation Rate**: 42.9% (based on recent analysis)
- **Median Reviews per Reviewer**: 7
- **75th Percentile**: 15 reviews per reviewer
- **90th Percentile**: 31 reviews per reviewer

## Experimental Design

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Snapshot Date** | 2020-01-01 | Ensures sufficient data before (7.5 years) and after (5 years) |
| **Learning Periods** | 3, 6, 9, 12 months | Standard quarterly and semi-annual intervals |
| **Prediction Periods** | 3, 6, 9, 12 months | Same intervals for symmetric evaluation |
| **Sequence Length** | 15 | Matches 75th percentile of reviewer activity |
| **Training Epochs** | 30 | Balanced between convergence and computation time |

### Evaluation Modes

#### 1. Multi-Project Mode

- **Data**: All 5 projects combined
- **Training**: IRL learns from all projects
- **Prediction**: Tests on all reviewers across all projects
- **Use Case**: General retention prediction model

#### 2. Single-Project Mode

- **Data**: Each project evaluated separately
- **Training**: IRL learns from single project only
- **Prediction**: Tests on reviewers within that project
- **Use Case**: Project-specific retention patterns
- **Output**: 5 separate evaluations (one per project)

### Sliding Window Grid

4 × 4 = **16 configurations** per mode:

|   | 3m pred | 6m pred | 9m pred | 12m pred |
|---|---------|---------|---------|----------|
| **3m learn** | ✓ | ✓ | ✓ | ✓ |
| **6m learn** | ✓ | ✓ | ✓ | ✓ |
| **9m learn** | ✓ | ✓ | ✓ | ✓ |
| **12m learn** | ✓ | ✓ | ✓ | ✓ |

**Total evaluations**:
- Multi-project: 16 configurations
- Single-project: 16 × 5 projects = 80 configurations
- **Grand total**: 96 model training runs

## Trajectory Construction

### Input Data Structure

Each trajectory represents one reviewer's activity in the learning period:

```python
trajectory = {
    "states": np.array([...]),    # Shape: (seq_len, 10)
    "actions": np.array([...]),   # Shape: (seq_len, 5)
    "reviewer": "reviewer@email.com"
}
```

### Feature Extraction

**State Features (10-dimensional)**:
1. Experience (total reviews up to this point)
2. Active flag (binary)
3. Activity frequency (reviews per month)
4. Collaboration score
5. Domain expertise
6. Code quality
7. Time since last activity
8. Response time
9. Project diversity
10. Social network centrality

**Action Features (5-dimensional)**:
1. Action type (always 1.0 for review)
2. Intensity (based on lines changed)
3. Quality score
4. Collaboration indicator
5. Impact score

### Sequence Handling

- **Short sequences** (< 15 reviews): Pad by repeating first review
- **Long sequences** (≥ 15 reviews): Use most recent 15 reviews
- **Rationale**: Focus on recent behavior, ensure fixed input size for LSTM

### Labeling Strategy

```python
check_start_date = snapshot_date + prediction_months

if reviewer has activity AFTER check_start_date:
    label = 1  # Continued (still active n months later)
else:
    label = 0  # Churned (not active n months later)
```

**Important**:
- Check for activity **AFTER** the waiting period (not within it)
- Activity counts ANY review after check_start_date, regardless of volume
- Practical meaning: "Will the reviewer still be active n months later?"

## Model Architecture

### EnhancedRetentionIRLSystem with LSTM

```
Input: Trajectory [batch, seq_len, feature_dim]
  ↓
State Encoder:
  Linear(10, 128)
  → LayerNorm
  → ReLU
  → Dropout(0.3)
  → Linear(128, 64)
  → LayerNorm
  → ReLU
  ↓
Action Encoder:
  Linear(5, 128)
  → LayerNorm
  → ReLU
  → Dropout(0.3)
  → Linear(128, 64)
  → LayerNorm
  → ReLU
  ↓
Combined State + Action (Addition)
  ↓
LSTM (2-layer, hidden_size=128, dropout=0.3)
  ↓
Split into two heads:
  ├─ Reward Predictor: Linear(128, 64) → ReLU → Linear(64, 1)
  └─ Continuation Predictor: Linear(128, 64) → ReLU → Linear(64, 1) → Sigmoid
```

### Training Configuration

```python
config = {
    "state_dim": 10,
    "action_dim": 5,
    "hidden_dim": 128,
    "learning_rate": 0.001,
    "sequence": True,           # Enable LSTM
    "seq_len": 15,
    "use_layer_norm": True,     # Enhanced version
    "dropout_rate": 0.3,        # Regularization
}
```

### Loss Function

IRL uses Maximum Entropy framework:
- Learns reward function from expert trajectories
- Expert = reviewers who continued contributing
- Maximizes likelihood of observed behavior under learned reward

## Evaluation Metrics

### Primary Metrics

1. **AUC-ROC** (Area Under ROC Curve)
   - Range: 0.0 to 1.0
   - Interpretation:
     - 0.9-1.0: Excellent
     - 0.8-0.9: Good
     - 0.7-0.8: Fair
     - 0.5: Random
   - Measures overall discrimination ability

2. **AUC-PR** (Area Under Precision-Recall Curve)
   - Range: 0.0 to 1.0
   - More informative for imbalanced datasets
   - Critical for rare positive class (continuation)

3. **F1 Score**
   - Harmonic mean of precision and recall
   - Balances false positives and false negatives
   - Threshold: 0.5 for binary classification

### Secondary Metrics

4. **Precision**: True Positives / (True Positives + False Positives)
5. **Recall**: True Positives / (True Positives + False Negatives)
6. **Confusion Matrix**: TN, FP, FN, TP counts

## Expected Outputs

### Directory Structure

```
importants/true_irl_comprehensive/
├── data_analysis/                           # Pre-existing from data analysis
│   ├── dataset_stats.json
│   ├── project_stats.csv
│   └── ...
├── models/
│   ├── multi_project/
│   │   ├── irl_h3m_t3m.pth
│   │   ├── irl_h3m_t6m.pth
│   │   ├── ...
│   │   └── irl_h12m_t12m.pth              (16 models)
│   ├── project_openstack_cinder/
│   │   └── ... (16 models)
│   ├── project_openstack_nova/
│   │   └── ... (16 models)
│   ├── project_openstack_neutron/
│   │   └── ... (16 models)
│   ├── project_openstack_glance/
│   │   └── ... (16 models)
│   └── project_openstack_keystone/
│       └── ... (16 models)
├── sliding_window_results_multi_project.csv
├── sliding_window_results_project_openstack_cinder.csv
├── sliding_window_results_project_openstack_nova.csv
├── sliding_window_results_project_openstack_neutron.csv
├── sliding_window_results_project_openstack_glance.csv
├── sliding_window_results_project_openstack_keystone.csv
├── heatmaps_multi_project.png
├── heatmaps_project_openstack_cinder.png
├── heatmaps_project_openstack_nova.png
├── heatmaps_project_openstack_neutron.png
├── heatmaps_project_openstack_glance.png
├── heatmaps_project_openstack_keystone.png
├── EVALUATION_PLAN.md                      (this file)
└── EVALUATION_REPORT.md                    (generated after evaluation)
```

### CSV Results Format

Each `sliding_window_results_*.csv` contains:

| Column | Description |
|--------|-------------|
| learning_months | Learning period length |
| prediction_months | Prediction period length |
| project | Project name (or "all" for multi-project) |
| n_trajectories | Number of trajectories used |
| continuation_rate | % of reviewers who continued |
| auc_roc | AUC-ROC score |
| auc_pr | AUC-PR score |
| f1 | F1 score |
| precision | Precision |
| recall | Recall |
| tn | True negatives |
| fp | False positives |
| fn | False negatives |
| tp | True positives |
| model_path | Path to saved model |

### Heatmap Visualizations

Each heatmap file contains 5 subplots (2×3 grid):
1. AUC-ROC
2. AUC-PR
3. F1 Score
4. Precision
5. Recall

**Axes**:
- X-axis: Learning period (3, 6, 9, 12 months)
- Y-axis: Prediction period (3, 6, 9, 12 months)

**Color**: YlOrRd (Yellow-Orange-Red), range 0-1

## Execution Command

```bash
uv run python scripts/training/irl/evaluate_true_irl_comprehensive.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --snapshot-date 2020-01-01 \
  --learning-months 3 6 9 12 \
  --prediction-months 3 6 9 12 \
  --mode both \
  --epochs 30 \
  --seq-len 15 \
  --output importants/true_irl_comprehensive
```

### Estimated Runtime

Based on data size and model complexity:
- Single configuration: ~2-3 minutes
- Multi-project (16 configs): ~40-50 minutes
- Single-project per project (16 configs): ~30-40 minutes
- All 5 projects: ~2.5-3 hours
- **Total (96 configurations)**: ~3-4 hours

### Memory Requirements

- Peak memory: ~4-6 GB RAM
- GPU: Not required (CPU training is acceptable)
- Disk space: ~500 MB (for all models and outputs)

## Research Questions to Answer

### 1. Optimal Period Configuration

- **Q**: What combination of learning and prediction periods yields best performance?
- **Method**: Find highest AUC-ROC in heatmap
- **Expected**: Longer learning periods (9-12m) likely perform better

### 2. Prediction Horizon Limit

- **Q**: How far into the future can we reliably predict?
- **Method**: Compare AUC-ROC across prediction periods
- **Expected**: Performance degrades for longer prediction periods

### 3. Data Efficiency

- **Q**: Is 3 months of history sufficient for good predictions?
- **Method**: Compare 3m vs 12m learning periods
- **Expected**: Diminishing returns after 6-9 months

### 4. Project-Specific vs General Model

- **Q**: Do project-specific models outperform general model?
- **Method**: Compare single-project vs multi-project results
- **Expected**: Project-specific may excel for large projects (cinder, nova)

### 5. Cross-Project Generalization

- **Q**: Can a model trained on all projects generalize to individual projects?
- **Method**: Multi-project model vs single-project models
- **Insight**: If multi-project performs well, it can be deployed universally

### 6. Continuation Rate Impact

- **Q**: Does continuation rate affect prediction difficulty?
- **Method**: Compare performance across projects with different continuation rates
- **Expected**: More balanced rates (closer to 50%) are easier to predict

## Validation and Quality Checks

### During Execution

Script will print:
- ✓ Number of trajectories extracted
- ✓ Continuation rate (label balance)
- ✓ Training progress
- ✓ Evaluation metrics

### Post-Execution Checks

1. **Model files**: Verify all .pth files are saved and loadable
2. **CSV completeness**: Check for missing configurations
3. **Metric ranges**: Ensure all metrics are in valid ranges [0, 1]
4. **Heatmap quality**: Visual inspection for patterns and anomalies

### Expected Patterns

- **Diagonal trend**: Similar learning/prediction periods may perform best
- **Learning period effect**: Longer learning usually improves performance
- **Prediction period effect**: Performance may degrade for distant futures
- **Project size effect**: Larger projects (cinder, nova) likely have better performance

## Troubleshooting

### Potential Issues

1. **Insufficient data for configuration**
   - **Symptom**: "Skipping: No trajectories"
   - **Cause**: Not enough reviews in learning period
   - **Solution**: Use earlier snapshot date or shorter periods

2. **Low continuation rate**
   - **Symptom**: Imbalanced dataset warning
   - **Impact**: High AUC-PR but low F1
   - **Mitigation**: Focus on AUC-PR metric

3. **Memory issues**
   - **Symptom**: OOM error
   - **Solution**: Reduce seq_len or batch size

4. **Training instability**
   - **Symptom**: NaN or 0.5 AUC-ROC
   - **Cause**: Learning rate too high or gradient explosion
   - **Solution**: Check LayerNorm and Dropout are active

## Next Steps After Evaluation

1. **Analyze Results**
   - Identify best configuration
   - Understand project-specific patterns
   - Document insights in EVALUATION_REPORT.md

2. **Model Selection**
   - Choose production model(s)
   - Document selection criteria

3. **Further Research**
   - Test on additional projects
   - Experiment with feature engineering
   - Try ensemble methods

4. **Deployment Planning**
   - Create inference pipeline
   - Set up monitoring
   - Define retraining schedule

## References

- **EnhancedRetentionIRLSystem**: `src/gerrit_retention/rl_prediction/enhanced_retention_irl_system.py`
- **Data Analysis**: `importants/true_irl_comprehensive/data_analysis/ANALYSIS_REPORT.txt`
- **Project README**: `README_TEMPORAL_IRL.md`

---

**Document Status**: Ready for execution
**Next Action**: Run evaluation script
**Expected Completion**: 3-4 hours from start
