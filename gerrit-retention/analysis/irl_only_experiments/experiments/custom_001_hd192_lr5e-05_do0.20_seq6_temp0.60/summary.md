# custom_001_hd192_lr5e-05_do0.20_seq6_temp0.60

## Config

{
  "hidden_dim": 192,
  "learning_rate": 5e-05,
  "dropout": 0.2,
  "seq_len": 6,
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
  "auc_roc": 0.3181818181818181,
  "auc_pr": 0.2275650435003861,
  "precision": 0.2894736842105263,
  "recall": 1.0,
  "f1_score": 0.4489795918367347,
  "optimal_threshold": 0.41994230789222003,
  "eval_optimal_threshold": 0.421653456789014,
  "eval_optimal_f1": 0.4489795918019159,
  "positive_count": 22,
  "negative_count": 54,
  "total_count": 76,
  "prediction_stats": {
    "min": 0.421653456789014,
    "max": 0.5515619622630538,
    "mean": 0.5209536128146528,
    "std": 0.024015100818614406,
    "median": 0.5233850414466961
  }
}