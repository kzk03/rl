# quick_02_deep_seq12

## Config

{
  "hidden_dim": 192,
  "learning_rate": 0.0001,
  "dropout": 0.1,
  "seq_len": 12,
  "output_temperature": 1.0,
  "epochs": 40,
  "train_start": "2021-01-01T00:00:00",
  "train_end": "2023-01-01T00:00:00",
  "eval_start": "2023-01-01T00:00:00",
  "eval_end": "2024-01-01T00:00:00",
  "future_window_start": 0,
  "future_window_end": 3,
  "min_history_events": 3,
  "seed": 777
}

## Metrics

{
  "auc_roc": 0.36141636141636135,
  "auc_pr": 0.28253160330841487,
  "precision": 0.3559322033898305,
  "recall": 1.0,
  "f1_score": 0.525,
  "optimal_threshold": 0.470441609621048,
  "eval_optimal_threshold": 0.47057080268859863,
  "eval_optimal_f1": 0.5249999999612812,
  "positive_count": 21,
  "negative_count": 39,
  "total_count": 60,
  "prediction_stats": {
    "min": 0.46441414952278137,
    "max": 0.5545468330383301,
    "mean": 0.5245926032463709,
    "std": 0.01717477723957559,
    "median": 0.527426540851593
  }
}