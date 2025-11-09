# quick_03_no_drop_high_lr_clip9m

## Config

{
  "hidden_dim": 192,
  "learning_rate": 0.0005,
  "dropout": 0.0,
  "seq_len": 12,
  "output_temperature": 0.85,
  "epochs": 45,
  "train_start": "2021-01-01T00:00:00",
  "train_end": "2022-10-01T00:00:00",
  "eval_start": "2023-01-01T00:00:00",
  "eval_end": "2024-01-01T00:00:00",
  "future_window_start": 0,
  "future_window_end": 3,
  "min_history_events": 3,
  "seed": 777
}

## Metrics

{
  "auc_roc": 0.265625,
  "auc_pr": 0.22214641518566233,
  "precision": 0.29411764705882354,
  "recall": 0.9375,
  "f1_score": 0.44776119402985076,
  "optimal_threshold": 0.366556437529317,
  "eval_optimal_threshold": 0.3659306331939904,
  "eval_optimal_f1": 0.4705882352581315,
  "positive_count": 16,
  "negative_count": 36,
  "total_count": 52,
  "prediction_stats": {
    "min": 0.3659306331939904,
    "max": 0.6700884187066187,
    "mean": 0.5241957670731741,
    "std": 0.07931271701064369,
    "median": 0.5229186939567101
  }
}