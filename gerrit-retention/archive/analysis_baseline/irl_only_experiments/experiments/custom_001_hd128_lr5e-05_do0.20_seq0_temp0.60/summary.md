# custom_001_hd128_lr5e-05_do0.20_seq0_temp0.60

## Config

{
  "hidden_dim": 128,
  "learning_rate": 5e-05,
  "dropout": 0.2,
  "seq_len": 0,
  "output_temperature": 0.6,
  "epochs": 60,
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
  "auc_pr": 0.5241896045271546,
  "precision": 0.43333333333333335,
  "recall": 0.5909090909090909,
  "f1_score": 0.5,
  "optimal_threshold": 0.47040797774453447,
  "eval_optimal_threshold": 0.4700974970775709,
  "eval_optimal_f1": 0.5555555555072703,
  "positive_count": 22,
  "negative_count": 54,
  "total_count": 76,
  "prediction_stats": {
    "min": 0.45120335877363515,
    "max": 0.5068499881832502,
    "mean": 0.4700479103425143,
    "std": 0.013752420299375734,
    "median": 0.46564797733248
  }
}