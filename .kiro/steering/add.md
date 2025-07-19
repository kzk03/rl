---
inclusion: always
---

# 時系列データ使用規則

## データ期間の厳格な分離

### 逆強化学習（IRL）データ期間

- **使用期間**: 2019 年〜2021 年のデータのみ
- **目的**: エキスパート軌跡の学習による報酬関数の推定
- **データソース**: `data/status/2019/`, `data/status/2020/`, `data/status/2021/`
- **出力ファイル**: `data/expert_trajectories*.pkl`

### 強化学習（RL）訓練データ期間

- **使用期間**: 2022 年のデータのみ
- **目的**: PPO エージェントのポリシー学習
- **データソース**: `data/status/2022/`
- **設定ファイル**: `configs/base_training.yaml`

### 強化学習（RL）テストデータ期間

- **使用期間**: 2023 年のデータのみ
- **目的**: 最終的なモデル性能評価
- **データソース**: `data/status/2023/`
- **設定ファイル**: `configs/base_test_2022.yaml`

## 時系列整合性の検証

### 必須チェック項目

- **データリーク防止**: 未来のデータが過去の予測に使用されていないことを確認
- **時系列順序**: IRL (2019-2021) → RL 訓練 (2022) → RL テスト (2023)
- **分割検証**: `scripts/split_temporal_data.py` で時系列分割を事前検証

### 設定ファイルでの時系列指定

```yaml
temporal_split:
  irl_end_date: "2021-12-31" # IRL用データの終了日
  train_start_date: "2022-01-01" # RL訓練データの開始日
  train_end_date: "2022-12-31" # RL訓練データの終了日
  test_start_date: "2023-01-01" # テストデータの開始日
```

## データファイル命名規則

### 期間別データファイル

- **IRL 用**: `expert_trajectories_2019_2021.pkl`
- **訓練用**: `backlog_training_2022.json`
- **テスト用**: `backlog_test_2023.json`

### 時系列検証の実行

```bash
# 時系列分割の検証
python scripts/split_temporal_data.py --verify

# 期間別データの整合性チェック
python scripts/validate_temporal_consistency.py
```

## 重要な注意事項

- **絶対禁止**: 2023 年のデータを 2022 年以前の学習に使用すること
- **必須**: 各コンポーネントの訓練前に時系列整合性を検証すること
- **推奨**: データ処理時に期間フィルタリングを明示的に実装すること
