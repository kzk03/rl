# prediction ディレクトリ詳細

- 目的: 継続率/負荷/専門性などの各種予測・分析器。
- 主なファイル（一部）:
  - `retention_predictor.py`: 継続率の予測器。
  - `realtime_retention_scorer.py`: リアルタイム推定。
  - `historical_feature_builder.py`: 履歴特徴の構築。
  - `workload_aware_predictor.py`: 負荷考慮の予測。
  - `workload_expertise_analyzer.py`: 負荷と専門性の分析。
  - `stress_analyzer.py` / `stress_mitigation_advisor.py`: ストレス分析と提言。
  - `advanced_accuracy_improver.py`: 高度な精度向上パイプライン。
  - `ab_testing_system.py`: AB テスト連携。

## IRL 特徴の有効化（クイックスタート）

1. IRL モデルを用意（例: MaxEntBinaryIRL を学習して保存）

2. 予測器の設定で有効化:

```python
from gerrit_retention.prediction.retention_predictor import RetentionPredictor

config = {
    'feature_extraction': {
        'irl_features': {
            'enabled': True,
            'model_path': '/path/to/irl_model.joblib',  # 省略可（後から set_irl_model でも可）
            'idle_gap_threshold': 45,
        }
    }
}

predictor = RetentionPredictor(config)
# もしくは後から学習済IRLを注入
# predictor.set_irl_model(irl)

# 通常通り fit/predict
predictor.fit(developers, contexts, labels)
prob = predictor.predict_retention_probability(developer, context)
```

3. IRL の学習は `prediction/irl_training_utils.py` の `build_transitions_from_events` と `fit_maxent_irl` を参照。
