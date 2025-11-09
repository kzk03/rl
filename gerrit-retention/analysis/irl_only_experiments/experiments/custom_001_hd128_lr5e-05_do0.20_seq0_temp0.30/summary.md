# custom_001_hd128_lr5e-05_do0.20_seq0_temp0.30

## Config

{
  "hidden_dim": 128,
  "learning_rate": 5e-05,
  "dropout": 0.2,
  "seq_len": 0,
  "output_temperature": 0.3,
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
  "auc_roc": 0.691919191919192,
  "auc_pr": 0.5228038045737469,
  "precision": 0.41935483870967744,
  "recall": 0.5909090909090909,
  "f1_score": 0.49056603773584906,
  "optimal_threshold": 0.4389871548822743,
  "eval_optimal_threshold": 0.42812023649138115,
  "eval_optimal_f1": 0.5483870967284078,
  "positive_count": 22,
  "negative_count": 54,
  "total_count": 76,
  "prediction_stats": {
    "min": 0.39970158279104945,
    "max": 0.5155848007732641,
    "mean": 0.438852771193489,
    "std": 0.029206594783500132,
    "median": 0.4291186029920106
  }
}