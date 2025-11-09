# custom_001_hd96_lr5e-05_do0.20_seq0_temp0.60

## Config

{
  "hidden_dim": 96,
  "learning_rate": 5e-05,
  "dropout": 0.2,
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
  "auc_roc": 0.39983164983164987,
  "auc_pr": 0.29197671285260146,
  "precision": 0.2894736842105263,
  "recall": 1.0,
  "f1_score": 0.4489795918367347,
  "optimal_threshold": 0.40815812404515645,
  "eval_optimal_threshold": 0.40940074214374045,
  "eval_optimal_f1": 0.4489795918019159,
  "positive_count": 22,
  "negative_count": 54,
  "total_count": 76,
  "prediction_stats": {
    "min": 0.40940074214374045,
    "max": 0.46324307138698245,
    "mean": 0.44006183534368215,
    "std": 0.009441536814103079,
    "median": 0.44012308149463353
  }
}