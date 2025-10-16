# 固定対象者アプローチ - 実装結果まとめ

## 問題の発見

学習期間によって予測対象となるレビュアーの人数が異なる問題を発見しました。

### 旧実装の問題点

**outputs/irl_project_cross/project_aware_results_cross_project_seq.csv より:**

| 学習期間 | 訓練サンプル数 | テストサンプル数 | 合計レビュアー数 |
|---------|--------------|----------------|-----------------|
| 6ヶ月   | 314          | 79             | 393             |
| 12ヶ月  | 488          | 122            | 610             |

**差分: 217人（55%増加）**

### 原因

`train_temporal_irl_sliding_window.py:97` および `train_temporal_irl_project_aware.py:120`:
```python
reviewers_in_history = set(history_df[reviewer_col].unique())
```

この実装では、学習期間中に活動があったレビュアーのみを対象とするため、学習期間が長いほど多くのレビュアーが含まれます。

## 解決策の実装

### アプローチ1: 固定基準期間方式（採用）

全ての学習期間実験で同じレビュアーセットを評価対象とする方式を実装しました。

**特徴:**
- 基準期間（デフォルト6ヶ月）で対象レビュアーを決定
- 全ての学習期間実験で同じレビュアーセットを使用
- 学習期間中に活動がないレビュアーはスキップするが、総予測対象数は一定
- 学習期間の効果を公平に比較可能

### 実装ファイル

**1. scripts/training/irl/train_temporal_irl_sliding_window_fixed_pop.py**
- 汎用的なスライディングウィンドウ評価（固定対象者版）
- 実装済み・動作確認済み ✅

**実行コマンド:**
```bash
uv run python scripts/training/irl/train_temporal_irl_sliding_window_fixed_pop.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --snapshot-date 2023-01-01 \
  --reference-period 6 \
  --history-months 6 12 \
  --target-months 6 12 \
  --sequence \
  --seq-len 15 \
  --epochs 30 \
  --output importants/irl_fixed_population
```

## 評価結果の比較

### 旧実装（Variable Population）

| 学習期間 | 予測期間 | 合計レビュアー数 | AUC-ROC | AUC-PR  |
|---------|---------|-----------------|---------|---------|
| 6ヶ月   | 6ヶ月   | 393             | 0.8323  | 0.9034  |
| 6ヶ月   | 12ヶ月  | 393             | 0.7957  | 0.8943  |
| 12ヶ月  | 6ヶ月   | 610             | 0.8679  | 0.8106  |
| 12ヶ月  | 12ヶ月  | 610             | 0.8511  | 0.8512  |

**問題点:** 6ヶ月と12ヶ月で異なるレビュアーセットを評価しているため、学習期間の効果を正確に比較できない。

### 新実装（Fixed Population）

| 学習期間 | 予測期間 | 合計レビュアー数 | AUC-ROC | AUC-PR  | F1      | Accuracy |
|---------|---------|-----------------|---------|---------|---------|----------|
| 6ヶ月   | 6ヶ月   | **291**         | 0.8221  | 0.9381  | 0.8049  | 0.7288   |
| 6ヶ月   | 12ヶ月  | **291**         | 0.7921  | 0.9333  | 0.8276  | 0.7458   |
| 12ヶ月  | 6ヶ月   | **291**         | 0.9244  | 0.9715  | 0.8837  | 0.8305   |
| 12ヶ月  | 12ヶ月  | **291**         | 0.8825  | 0.9634  | 0.8842  | 0.8136   |

**改善点:**
- ✅ すべての実験で同じ291人を予測対象としている
- ✅ 学習期間の効果を公平に比較可能
- ✅ 12ヶ月学習の方が明確に高性能（AUC-ROC: 0.92 vs 0.82）

## 重要な発見

### 学習期間の効果（Fixed Population で正確に測定）

**12ヶ月学習 vs 6ヶ月学習（予測期間6ヶ月の場合）:**
- AUC-ROC: 0.9244 vs 0.8221 (+12.5%)
- AUC-PR: 0.9715 vs 0.9381 (+3.6%)
- F1スコア: 0.8837 vs 0.8049 (+9.8%)
- Accuracy: 0.8305 vs 0.7288 (+14.0%)

**結論:** 長い学習期間（12ヶ月）は明確にモデル性能を向上させる。

## 今後の作業

### 1. プロジェクト別評価への適用（TODO）

`train_temporal_irl_project_aware_fixed_pop.py` の実装が必要です。

**課題:**
- レビュアー×プロジェクトの組み合わせを固定する必要がある
- project-specific モードと cross-project モードで異なる実装が必要

**実装方針:**
1. 基準期間でレビュアー×プロジェクトの組み合わせを抽出
2. すべての学習期間でその組み合わせを使用
3. 活動がない組み合わせはスキップするが総数は一定

### 2. 詳細な分析レポート作成

- [ ] 学習期間と予測期間の最適な組み合わせの分析
- [ ] プロジェクト別の特性分析
- [ ] 継続率と予測精度の関係分析

### 3. ドキュメント更新

- [ ] CLAUDE.md に固定対象者アプローチを追加
- [ ] README_TEMPORAL_IRL.md の更新
- [ ] 論文執筆用の図表作成

## ファイル構成

```
importants/
├── irl_fixed_population/              # 新実装の結果
│   ├── sliding_window_results_fixed_pop_seq.csv
│   ├── evaluation_matrix_fixed_pop_seq.txt
│   ├── evaluation_metadata.json
│   └── models/
│       ├── irl_h6m_t6m_fixed_seq.pth
│       ├── irl_h6m_t12m_fixed_seq.pth
│       ├── irl_h12m_t6m_fixed_seq.pth
│       └── irl_h12m_t12m_fixed_seq.pth
└── fixed_population_results_summary.md  # 本ドキュメント

outputs/
├── irl_project_cross/                 # 旧実装（比較用）
│   └── project_aware_results_cross_project_seq.csv
└── irl_project_specific/
    └── project_aware_results_project_specific_seq.csv

docs/
└── reviewer_count_variance_analysis.md  # 問題分析ドキュメント
```

## 参考情報

### 関連Issue/PR
- 問題発見: 学習期間によるレビュアー数の変動
- 解決策提案: docs/reviewer_count_variance_analysis.md
- 実装: scripts/training/irl/train_temporal_irl_sliding_window_fixed_pop.py

### 技術詳細
- 基準期間: 6ヶ月（デフォルト）
- シーケンス長: 15
- エポック数: 30
- 訓練/テスト分割: 80/20
- 乱数シード: 42（再現性のため固定）

---

作成日: 2025-10-17
最終更新: 2025-10-17
