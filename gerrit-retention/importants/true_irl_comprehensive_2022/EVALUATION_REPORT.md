# True IRL Comprehensive Evaluation Report

**Generated**: 2025-10-19 22:13:35

## Executive Summary

This report presents a comprehensive evaluation of the **True IRL system** (EnhancedRetentionIRLSystem with LSTM) for developer retention prediction.

### Key Innovation

- **Temporal IRL with LSTM**: Full sequential modeling of developer activity trajectories
- **Training data period matches learning period**: 3-month learning uses 3 months of data, 12-month learning uses 12 months of data
- **Project-aware prediction**: Evaluates continuation within the same project

## Dataset Information

- **File**: `data/review_requests_openstack_multi_5y_detail.csv`
- **Size**: 78.75 MB
- **Total Reviews**: 137,632
- **Date Range**: 2012-06-20 07:20:24 to 2025-09-27 10:39:10 (13.3 years)
- **Unique Reviewers**: 1,379
- **Projects**: 5

### Project Distribution

- **openstack/cinder**: 71,604 reviews (52.0%)
- **openstack/neutron**: 32,888 reviews (23.9%)
- **openstack/nova**: 27,328 reviews (19.9%)
- **openstack/glance**: 3,273 reviews (2.4%)
- **openstack/keystone**: 2,539 reviews (1.8%)

## Experimental Setup

- **Snapshot Date**: 2022-01-01
- **Learning Periods**: 3, 6, 9, 12 months
- **Prediction Periods**: 3, 6, 9, 12 months
- **Sequence Length**: 15
- **Training Epochs**: 30
- **Evaluation Mode**: both

### Critical Design Decision: Training Data Period

**Unlike other approaches**, this evaluation matches the IRL training data period to the learning period:

- **3-month learning**: Uses data from 2021-10-03 to 2022-01-01 for IRL training
- **6-month learning**: Uses data from 2021-07-05 to 2022-01-01 for IRL training
- **9-month learning**: Uses data from 2021-04-06 to 2022-01-01 for IRL training
- **12-month learning**: Uses data from 2021-01-06 to 2022-01-01 for IRL training

This ensures that the model learns from a representative sample of the learning period it will be predicting for.

## Results Summary

### Multi Project

**Total Configurations**: 16

**Best Configurations**:

- **AUC-ROC**: 0.8712 (9m learning × 6m prediction)
- **AUC-PR**: 0.9323 (9m learning × 6m prediction)
- **F1**: 0.8409 (3m learning × 6m prediction)

**See heatmaps**: `heatmaps_multi_project.png`

**Full results**: `sliding_window_results_multi_project.csv`

### Project Openstack Nova

**Total Configurations**: 16

**Best Configurations**:

- **AUC-ROC**: 0.8542 (6m learning × 9m prediction)
- **AUC-PR**: 0.8867 (3m learning × 12m prediction)
- **F1**: 0.7200 (3m learning × 3m prediction)

**See heatmaps**: `heatmaps_project_openstack_nova.png`

**Full results**: `sliding_window_results_project_openstack_nova.csv`

### Project Openstack Cinder

**Total Configurations**: 16

**Best Configurations**:

- **AUC-ROC**: 0.9062 (3m learning × 6m prediction)
- **AUC-PR**: 0.9794 (3m learning × 6m prediction)
- **F1**: 0.9189 (3m learning × 3m prediction)

**See heatmaps**: `heatmaps_project_openstack_cinder.png`

**Full results**: `sliding_window_results_project_openstack_cinder.csv`

### Project Openstack Neutron

**Total Configurations**: 16

**Best Configurations**:

- **AUC-ROC**: 0.9231 (3m learning × 9m prediction)
- **AUC-PR**: 0.9760 (3m learning × 9m prediction)
- **F1**: 0.7586 (3m learning × 6m prediction)

**See heatmaps**: `heatmaps_project_openstack_neutron.png`

**Full results**: `sliding_window_results_project_openstack_neutron.csv`

### Project Openstack Glance

**Total Configurations**: 16

**Best Configurations**:

- **AUC-ROC**: 1.0000 (3m learning × 9m prediction)
- **AUC-PR**: 1.0000 (3m learning × 9m prediction)
- **F1**: 1.0000 (9m learning × 12m prediction)

**See heatmaps**: `heatmaps_project_openstack_glance.png`

**Full results**: `sliding_window_results_project_openstack_glance.csv`

### Project Openstack Keystone

**Total Configurations**: 16

**Best Configurations**:

- **AUC-ROC**: 1.0000 (6m learning × 6m prediction)
- **AUC-PR**: 1.0000 (6m learning × 6m prediction)
- **F1**: 0.5714 (3m learning × 3m prediction)

**See heatmaps**: `heatmaps_project_openstack_keystone.png`

**Full results**: `sliding_window_results_project_openstack_keystone.csv`

## Methodology

### Trajectory Construction

1. **Learning Period**: Reviews from [snapshot_date - learning_months, snapshot_date]
2. **Sequence Processing**:
   - Extract state (10-dim) and action (5-dim) features for each review
   - Pad sequences shorter than seq_len by repeating first review
   - Truncate sequences longer than seq_len to most recent reviews
3. **Labeling**: 1 if reviewer had activity in prediction period, 0 otherwise

### Model Architecture

```
Input: Trajectory [batch, seq_len, feature_dim]
  ↓
State Encoder (Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm → ReLU)
  ↓
Action Encoder (Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm → ReLU)
  ↓
Combined (Addition)
  ↓
LSTM (2-layer, hidden_size=128, dropout=0.3)
  ↓
├─ Reward Predictor (Linear → ReLU → Linear)
└─ Continuation Predictor (Linear → ReLU → Linear → Sigmoid)
```

### Evaluation Metrics

- **AUC-ROC**: Area under ROC curve (overall discrimination ability)
- **AUC-PR**: Area under precision-recall curve (important for imbalanced data)
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

## Interpretation Guidelines

### AUC-ROC Interpretation

- **0.9-1.0**: Excellent discrimination
- **0.8-0.9**: Good discrimination
- **0.7-0.8**: Fair discrimination
- **0.5**: Random guessing

### Period Selection Insights

Based on the heatmaps, you can identify:

1. **Optimal learning period**: How much history is needed for accurate prediction?
2. **Optimal prediction period**: What time horizon can we reliably predict?
3. **Trade-offs**: Longer periods may not always be better due to data staleness

## Files Generated

```
importants/true_irl_comprehensive_2022/
├── models/multi_project/
│   └── irl_hXm_tXm.pth  (trained model files)
├── sliding_window_results_multi_project.csv
├── heatmaps_multi_project.png
├── models/project_openstack_nova/
│   └── irl_hXm_tXm.pth  (trained model files)
├── sliding_window_results_project_openstack_nova.csv
├── heatmaps_project_openstack_nova.png
├── models/project_openstack_cinder/
│   └── irl_hXm_tXm.pth  (trained model files)
├── sliding_window_results_project_openstack_cinder.csv
├── heatmaps_project_openstack_cinder.png
├── models/project_openstack_neutron/
│   └── irl_hXm_tXm.pth  (trained model files)
├── sliding_window_results_project_openstack_neutron.csv
├── heatmaps_project_openstack_neutron.png
├── models/project_openstack_glance/
│   └── irl_hXm_tXm.pth  (trained model files)
├── sliding_window_results_project_openstack_glance.csv
├── heatmaps_project_openstack_glance.png
├── models/project_openstack_keystone/
│   └── irl_hXm_tXm.pth  (trained model files)
├── sliding_window_results_project_openstack_keystone.csv
├── heatmaps_project_openstack_keystone.png
└── EVALUATION_REPORT.md  (this file)
```

## Reproducibility

To reproduce these results, run:

```bash
uv run python scripts/training/irl/evaluate_true_irl_comprehensive.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --snapshot-date 2022-01-01 \
  --learning-months 3 6 9 12 \
  --prediction-months 3 6 9 12 \
  --mode both \
  --epochs 30 \
  --seq-len 15 \
  --output importants/true_irl_comprehensive_2022
```

---

*Report generated by evaluate_true_irl_comprehensive.py on 2025-10-19 22:13:35*