# custom_001_hd128_lr5e-05_do0.30_seq0_temp0.60

## Config

{
  "hidden_dim": 128,
  "learning_rate": 5e-05,
  "dropout": 0.3,
  "seq_len": 0,
  "output_temperature": 0.6,
  "epochs": 70,
  "train_start": "2021-01-01T00:00:00",
  "train_end": "2023-01-01T00:00:00",
  "eval_start": "2023-01-01T00:00:00",
  "eval_end": "2024-01-01T00:00:00",
  "future_window_start": 0,
  "future_window_end": 3,
  "min_history_events": 0,
  "seed": 777
}

## Metrics

{
  "auc_roc": 0.6952861952861953,
  "auc_pr": 0.5062199774483216,
  "precision": 0.36538461538461536,
  "recall": 0.8636363636363636,
  "f1_score": 0.5135135135135135,
  "optimal_threshold": 0.4532371681548632,
  "eval_optimal_threshold": 0.4539886391863138,
  "eval_optimal_f1": 0.5428571428140408,
  "positive_count": 22,
  "negative_count": 54,
  "total_count": 76,
  "prediction_stats": {
    "min": 0.44698894489646596,
    "max": 0.492560747950249,
    "mean": 0.4612933045783939,
    "std": 0.011267263439647712,
    "median": 0.4563580902234437
  }
}