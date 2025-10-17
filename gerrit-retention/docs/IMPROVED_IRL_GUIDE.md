# 改良版IRL実装ガイド

拡張特徴量を統合したIRL訓練スクリプトの使用方法

**最終更新**: 2025-10-17
**スクリプト**: [train_improved_irl.py](../scripts/training/irl/train_improved_irl.py)

---

## 📊 追加された重要特徴量

### A1: 活動頻度の多期間比較

| 特徴量 | 説明 | 効果 |
|-------|------|------|
| `reviewer_past_reviews_7d` | 7日間のレビュー数 | 短期的な活動パターン把握 |
| `reviewer_past_reviews_30d` | 30日間のレビュー数 | 中期的な活動パターン把握 |
| `reviewer_past_reviews_90d` | 90日間のレビュー数 | 長期的な活動パターン把握 |
| `activity_freq_7d/30d/90d` | 期間別活動頻度 | 1日あたりの活動数 |
| `activity_acceleration` | 活動加速度 | (freq_7d - freq_30d) / freq_30d |

### B1: レビュー負荷指標と集中度

| 特徴量 | 説明 | 効果 |
|-------|------|------|
| `reviewer_assignment_load_7d` | 7日間のレビュー依頼数 | 短期的な負荷 |
| `reviewer_assignment_load_30d` | 30日間のレビュー依頼数 | 中期的な負荷 |
| `reviewer_assignment_load_180d` | 180日間のレビュー依頼数 | 長期的な負荷 |
| `review_concentration_score` | **レビュー依頼の集中度** | 短期間に多数の依頼が集中しているか |
| `is_overloaded` | 過負荷フラグ | 1日5件以上でバーンアウトリスク |

**集中度の計算**:
```python
concentration = (load_7d * 30) / (7 * load_30d)
score = max(concentration - 1.0, 0.0)  # 1.5倍以上で集中している
```

### C1: 相互作用履歴

| 特徴量 | 説明 | 効果 |
|-------|------|------|
| `owner_reviewer_past_interactions_180d` | 180日間の相互作用回数 | コミュニティ結びつき |
| `owner_reviewer_project_interactions_180d` | プロジェクト内相互作用 | プロジェクト帰属意識 |
| `owner_reviewer_past_assignments_180d` | レビュー割当履歴 | 役割の定着度 |
| `interaction_intensity` | 相互作用強度 | 月あたりの相互作用頻度 |

### D1: Path類似度

| 特徴量 | 説明 | 効果 |
|-------|------|------|
| `path_jaccard_files_project` | ファイルパスJaccard類似度 | 専門分野の一致度 |
| `path_jaccard_dir1_project` | ディレクトリ1階層目Jaccard | 同上 |
| `path_jaccard_dir2_project` | ディレクトリ2階層目Jaccard | 同上 |
| `path_similarity_score` | 平均path類似度 | 総合的な専門性マッチング |

---

## 🚀 使用方法

### 基本的な使用方法

```bash
# 拡張特徴量を使用したIRL訓練
uv run python scripts/training/irl/train_improved_irl.py \
  --reviews data/review_requests_openstack_no_bots.csv \
  --snapshot-date 2023-01-01 \
  --history-months 12 \
  --target-months 6 \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_improved_12m_6m
```

### 複数期間での実験

```bash
# マトリクス評価
uv run python scripts/training/irl/train_improved_irl.py \
  --reviews data/review_requests_openstack_no_bots.csv \
  --snapshot-date 2023-01-01 \
  --history-months 6 12 18 \
  --target-months 3 6 9 12 \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_improved_matrix
```

### ボット除外済みデータで実験

```bash
# ステップ1: ボット除外
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_no_bots.csv

# ステップ2: 改良版IRL訓練
uv run python scripts/training/irl/train_improved_irl.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2023-01-01 \
  --history-months 12 \
  --target-months 6 \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_improved_no_bots
```

---

## 📈 出力ファイル

### ディレクトリ構造

```
importants/irl_improved_12m_6m/
├── models/
│   └── irl_h12m_t6m_improved.pth      # 訓練済みモデル
├── results_h12m_t6m.json              # 詳細結果
├── summary_improved.json              # 全実験サマリー
└── IMPROVED_REPORT.md                 # マトリクスレポート
```

### 結果JSONの内容

```json
{
  "history_months": 12,
  "target_months": 6,
  "train_size": 232,
  "test_size": 59,
  "train_continuation_rate": 0.659,
  "test_continuation_rate": 0.448,
  "auc_roc": 0.875,
  "auc_pr": 0.965,
  "f1_score": 0.882,
  "train_losses": [...],
  "config": {
    "state_dim": 32,
    "action_dim": 9,
    "hidden_dim": 128,
    "sequence": true,
    "seq_len": 15
  }
}
```

---

## 🔍 従来版との比較

### 特徴量の違い

| 項目 | 基本版 | 拡張版 | 改良版 |
|-----|--------|--------|--------|
| 状態次元 | 10 | 32 | 32 |
| 行動次元 | 5 | 9 | 9 |
| 活動頻度 | 30日のみ | 7/30/90日 | ✅ 7/30/90日 |
| レビュー負荷 | なし | 7/30/180日 | ✅ 7/30/180日 + 集中度 |
| 相互作用 | 簡易スコア | 詳細履歴 | ✅ 詳細履歴 |
| Path類似度 | なし | あり | ✅ あり |

### 期待される性能向上

| 指標 | 基本版 | 拡張版 | 改良版（期待） |
|-----|--------|--------|--------------|
| AUC-ROC | 0.840 | 0.855 | **0.870-0.880** |
| AUC-PR | 0.950 | 0.960 | **0.970-0.980** |
| F1 Score | 0.870 | 0.880 | **0.890-0.900** |

---

## 🧪 実験プラン

### Phase 1: ベースライン比較

```bash
# 基本版（比較用）
uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2023-01-01 \
  --history-months 12 --target-months 6 \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_baseline_12m_6m

# 改良版
uv run python scripts/training/irl/train_improved_irl.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2023-01-01 \
  --history-months 12 --target-months 6 \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_improved_12m_6m
```

### Phase 2: マトリクス評価

```bash
# 各スナップショット日付で実験
for date in "2022-01-01" "2022-07-01" "2023-01-01" "2023-07-01"; do
  uv run python scripts/training/irl/train_improved_irl.py \
    --reviews data/review_requests_no_bots.csv \
    --snapshot-date $date \
    --history-months 6 12 18 \
    --target-months 3 6 9 12 \
    --sequence --seq-len 15 --epochs 30 \
    --output importants/irl_improved_${date//-/}
done
```

### Phase 3: アブレーションスタディ

各特徴量グループの寄与度を個別に評価:

```bash
# A1: 活動頻度のみ追加
--features activity_frequency

# B1: レビュー負荷のみ追加
--features review_load

# C1: 相互作用のみ追加
--features interaction

# D1: Path類似度のみ追加
--features path_similarity

# 全特徴量
--features all
```

---

## 🔧 トラブルシューティング

### Q1: 「軌跡が不足しているためスキップ」エラー

**原因**: データに必要なカラムが存在しない

**解決策**:
```bash
# データのカラムを確認
uv run python -c "import pandas as pd; df = pd.read_csv('data/review_requests_openstack_no_bots.csv'); print(df.columns.tolist())"

# 必要なカラム:
# - reviewer_past_reviews_7d/30d/90d
# - reviewer_assignment_load_7d/30d/180d
# - owner_reviewer_past_interactions_180d
# - path_jaccard_files_project
```

### Q2: メモリ不足エラー

**原因**: 大量のデータと拡張特徴量でメモリ使用量が増加

**解決策**:
```bash
# データをプロジェクト別に分割
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --split-by-project \
  --output-dir data/projects/

# 各プロジェクトで個別に訓練
for project in data/projects/*.csv; do
  uv run python scripts/training/irl/train_improved_irl.py \
    --reviews $project \
    --snapshot-date 2023-01-01 \
    --history-months 12 --target-months 6 \
    --sequence --seq-len 15 --epochs 30 \
    --output importants/irl_$(basename $project .csv)
done
```

### Q3: 訓練時間が長すぎる

**解決策**:
```bash
# エポック数を減らす
--epochs 10

# シーケンス長を短くする
--seq-len 10

# 非時系列モードを使用
# --sequence フラグを削除
```

---

## 📊 結果の解釈

### 集中度スコアの解釈

```python
# concentration_score = 0.0: 負荷が均等に分散
# concentration_score = 0.5: 7日間の負荷が30日平均の1.5倍
# concentration_score = 1.0: 7日間の負荷が30日平均の2.0倍（危険）
# concentration_score > 1.0: 過度な集中（バーンアウトリスク高）
```

### Path類似度スコアの解釈

```python
# path_similarity_score = 0.0: 全く異なる領域
# path_similarity_score = 0.3: 部分的に重複
# path_similarity_score = 0.6: 高い類似性（専門性マッチング）
# path_similarity_score = 0.9: ほぼ同一領域（専門家）
```

---

## 📚 関連ドキュメント

- [IRL_FEATURE_SUMMARY.md](IRL_FEATURE_SUMMARY.md): 特徴量の簡潔なまとめ
- [IRL_FEATURE_ANALYSIS.md](IRL_FEATURE_ANALYSIS.md): 特徴量の詳細分析
- [DATA_FILTERING_GUIDE.md](DATA_FILTERING_GUIDE.md): データフィルタリング
- [enhanced_feature_extractor.py](../src/gerrit_retention/rl_prediction/enhanced_feature_extractor.py): 特徴量抽出実装

---

**作成者**: Claude + Kazuki-h
**ステータス**: 完成
