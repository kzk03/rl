# quick_01_baseline_cfg

## Config

{
  "hidden_dim": 128,
  "learning_rate": 0.0001,
  "dropout": 0.2,
  "seq_len": 0,
  "output_temperature": 1.0,
  "epochs": 30,
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
  "auc_roc": 0.7167277167277167,
  "auc_pr": 0.5949770804506123,
  "precision": 0.5652173913043478,
  "recall": 0.6190476190476191,
  "f1_score": 0.5909090909090909,
  "optimal_threshold": 0.4562135338783264,
  "eval_optimal_threshold": 0.444004625082016,
  "eval_optimal_f1": 0.6249999999559083,
  "positive_count": 21,
  "negative_count": 39,
  "total_count": 60,
  "prediction_stats": {
    "min": 0.4297329783439636,
    "max": 0.4932312071323395,
    "mean": 0.4547299270828565,
    "std": 0.01497129491974381,
    "median": 0.4546119421720505
  }
}