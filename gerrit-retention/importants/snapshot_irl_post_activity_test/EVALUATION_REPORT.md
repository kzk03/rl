# Snapshot-based IRL Evaluation Report

**Population Design**: Post-Activity (Active within 3 months after snapshot)

**Snapshot Date**: 2022-01-01

**Population Size**: 320 reviewers

## Results Summary

### Best Performances

- **Best AUC-ROC**: 0.7229 (12m learning × 6m prediction)
- **Best AUC-PR**: 0.8494 (12m learning × 6m prediction)
- **Best F1**: 0.8385 (3m learning × 6m prediction)

## All Results

```
 learning_months  prediction_months  n_population  n_train  continuation_rate  auc_roc   auc_pr       f1  precision   recall  tn  fp  fn  tp                                                                  model_path
               3                  6           320      171            72.1875 0.706746 0.839516 0.838475   0.721875 1.000000   0  89   0 231  importants/snapshot_irl_post_activity_test/models/reward_model_h3m_t6m.pth
              12                  6           320      214            72.1875 0.722919 0.849355 0.786517   0.817757 0.757576  50  39  56 175 importants/snapshot_irl_post_activity_test/models/reward_model_h12m_t6m.pth
```