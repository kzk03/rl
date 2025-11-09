# Enhanced IRL 月次サンプリング実装メモ

## 2025-11-09 実装状況

### ✅ 実装完了

1. **月次サンプリング** (`run_multi_seed_eval_v2.py`)

   - Train: 月次スライディングウィンドウ
   - Eval: 1 スナップショット
   - レビュワーごとのデータ事前分割

2. **進捗表示**

   - Train: 6 ヶ月ごと
   - Eval: 500 人ごと

3. **データリーク防止**
   - Train 最後の N 月はラベル計算専用
   - Train の予測期間が train_end_dt を超えないように制限

### ⚠️ パフォーマンス問題

**症状**: `temporal_feature_extractor`の処理が非常に遅い

**原因**:

- `extract_all_temporal_features`内で毎回 reviewer_email でフィルタリング
- すでにレビュワーごとに分割済みの DataFrame を渡しているのに、内部で再度フィルタリング

**影響**:

- 1 サンプルあたり数秒かかる
- 数千サンプルで数時間〜数十時間

### 🔧 必要な最適化

#### Option 1: TemporalFeatureExtractor を修正

```python
# 修正案: すでにフィルタリング済みの場合はスキップ
def extract_acceptance_sequence(self, reviewer_email, period_requests, context_date, pre_filtered=False):
    if not pre_filtered:
        reviewer_requests = period_requests[
            period_requests['reviewer_email'] == reviewer_email
        ]
    else:
        reviewer_requests = period_requests  # すでにフィルタリング済み
```

#### Option 2: 呼び出し側で完全に前処理

```python
# レビュワーごとに全特徴量を事前計算してキャッシュ
temporal_cache = {}
for reviewer in all_reviewers:
    for month in all_months:
        key = (reviewer, month)
        temporal_cache[key] = extract_all_temporal_features(...)
```

#### Option 3: Numba/Cython で高速化

- 時系列特徴量計算を Numba でコンパイル
- ループを並列化

### 📝 今後の作業

1. [x] 月次サンプリング実装
2. [x] データリーク防止
3. [ ] パフォーマンス最適化（Option 1 or 2）
4. [ ] 全期間組み合わせで実行
5. [ ] 結果分析・可視化

### 🎯 実験設計の確認

**Train 期間**: 2021-01-01 〜 2023-01-01（24 ヶ月）

- 観測可能: 2021-01-01 〜 2022-07-01（18 ヶ月）← 月次サンプリング
- ラベル専用: 2022-07-01 〜 2023-01-01（6 ヶ月）

**Eval 期間**: 2023-01-01 〜 2024-01-01（12 ヶ月）

- 観測: 2023-01-01 時点のスナップショット（1 個のみ）
- ラベル計算: 2023-01-01 以降の将来期間

**期待サンプル数**:

- Train: 18 ヶ月 × ~200 人/月 = ~3,600 サンプル
- Eval: ~500 サンプル

**現在の問題**:

- 処理時間が長すぎて実験不可能
- 最適化が必須
