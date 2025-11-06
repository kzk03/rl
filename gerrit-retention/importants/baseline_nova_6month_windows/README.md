# Baseline Comparison - 6-Month Windows (IRL-Aligned)

**⚠️ このディレクトリは論文用に保存された6ヶ月幅バージョンです**

## 実験設定

### Future Windows（6ヶ月幅）
このバージョンは**IRL実験と完全に同じ設定**で実行されています：

| 訓練期間名 | Future Window | 使える月数 | 軌跡数 |
|-----------|--------------|-----------|--------|
| 0-3m | **0-6ヶ月** | 23ヶ月 | 905 |
| 3-6m | **6-12ヶ月** | 17ヶ月 | 540 |
| 6-9m | **12-18ヶ月** | 11ヶ月 | 268 |
| 9-12m | **18-24ヶ月** | 5ヶ月 | 102 |

**注意**: 訓練期間の名前（0-3m、3-6m等）は評価期間に合わせた名前ですが、実際のfuture windowは6ヶ月幅です。

### データセット
- プロジェクト: OpenStack Nova only
- レビュー数: 27,328件
- 訓練期間: 2021-01-01 ～ 2023-01-01 (24ヶ月)
- 評価期間: 2023-01-01 ～ 2024-01-01 (12ヶ月)

### 訓練方式
- 月次訓練（IRLと同一）
- max-date制約あり（データリーク防止）

## 主要結果

### 対角線+未来評価（10セル）
- IRL+LSTM: **0.801** (±0.068)
- Logistic Regression: 0.763 (±0.141)
- Random Forest: 0.693 (±0.100)

**IRL優位性**: +3.8% vs LR、+10.8% vs RF

### 9-12m訓練期間（102軌跡）での結果
- IRL: **0.693** ← 頑健
- LR: **0.361** ← 崩壊
- RF: 0.485

**IRL優位性**: +91.9% vs LR

## 実験コマンド

```bash
uv run python scripts/experiments/run_baseline_nova_fair_comparison.py \
  --reviews data/review_requests_nova.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_6month_windows/
```

## ファイル一覧

- `EXECUTIVE_SUMMARY.md` - 総括レポート
- `NOVA_ONLY_ANALYSIS.md` - 詳細分析
- `REVIEW_ACCEPTANCE_EXPERIMENT_GUIDE.md` - 実験ガイド
- `FUTURE_RESEARCH_IDEAS.md` - 今後の研究アイデア
- `comparison_heatmaps_full.png` - 全16セル比較
- `comparison_heatmaps_diagonal_future.png` - 実用的評価のみ
- `logistic_regression/` - LR結果
- `random_forest/` - RF結果

## 3ヶ月幅バージョンとの比較

3ヶ月幅に統一したバージョンは `importants/baseline_nova_3month_windows/` を参照してください。

---

**保存日**: 2024-11-06
**目的**: IRL実験との公平な比較（論文用）
