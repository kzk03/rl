# custom_001_hd128_lr1e-04_do0.20_seq0_temp0.60

## Config

{
  "hidden_dim": 128,
  "learning_rate": 0.0001,
  "dropout": 0.2,
  "seq_len": 0,
  "output_temperature": 0.6,
  "epochs": 45,
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
  "auc_roc": 0.7203907203907204,
  "auc_pr": 0.6023416307427791,
  "precision": 0.5652173913043478,
  "recall": 0.6190476190476191,
  "f1_score": 0.5909090909090909,
  "optimal_threshold": 0.4135876682083783,
  "eval_optimal_threshold": 0.38913085686948445,
  "eval_optimal_f1": 0.6153846153408756,
  "positive_count": 21,
  "negative_count": 39,
  "total_count": 60,
  "prediction_stats": {
    "min": 0.3669086910735598,
    "max": 0.4877938256617901,
    "mean": 0.41077705855661845,
    "std": 0.028755113698780786,
    "median": 0.40940979517387843
  }
}