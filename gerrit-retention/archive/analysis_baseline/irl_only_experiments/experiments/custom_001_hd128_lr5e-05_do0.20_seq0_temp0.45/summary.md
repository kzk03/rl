# custom_001_hd128_lr5e-05_do0.20_seq0_temp0.45

## Config

{
  "hidden_dim": 128,
  "learning_rate": 5e-05,
  "dropout": 0.2,
  "seq_len": 0,
  "output_temperature": 0.45,
  "epochs": 70,
  "train_start": "2021-01-01T00:00:00",
  "train_end": "2023-01-01T00:00:00",
  "eval_start": "2023-01-01T00:00:00",
  "eval_end": "2024-01-01T00:00:00",
  "future_window_start": 0,
  "future_window_end": 3,
  "min_history_events": 0,
  "seed": 777,
  "focal_alpha": 0.5,
  "focal_gamma": 1.0
}

## Metrics

{
  "auc_roc": 0.7163299663299664,
  "auc_pr": 0.5591684215292956,
  "precision": 0.4827586206896552,
  "recall": 0.6363636363636364,
  "f1_score": 0.5490196078431373,
  "optimal_threshold": 0.48314122213066485,
  "eval_optimal_threshold": 0.48091781446554394,
  "eval_optimal_f1": 0.6037735848571022,
  "positive_count": 22,
  "negative_count": 54,
  "total_count": 76,
  "prediction_stats": {
    "min": 0.4672685913511141,
    "max": 0.5230014587155558,
    "mean": 0.48278065404654424,
    "std": 0.013551957064413601,
    "median": 0.47773022094095663
  }
}