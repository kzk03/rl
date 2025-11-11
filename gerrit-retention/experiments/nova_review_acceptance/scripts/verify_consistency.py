#!/usr/bin/env python3
"""全体評価と実用評価のデータ整合性確認"""
import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path("experiments/nova_review_acceptance")

# サマリー読み込み
comparison_summary = pd.read_csv(base_dir / "comparison_heatmaps/comparison_summary.csv")
practical_summary = pd.read_csv(base_dir / "practical_comparison_heatmaps/practical_comparison_summary.csv")

print("=" * 80)
print("全体評価（16セル全体の平均）")
print("=" * 80)
print(comparison_summary.to_string(index=False))
print()

print("=" * 80)
print("実用評価（有効10セルの平均）")
print("=" * 80)
print(practical_summary.to_string(index=False))
print()

# マトリクスデータ読み込みと検証
irl_auc = pd.read_csv(base_dir / "results_enhanced_irl/matrix_AUC_ROC.csv", index_col=0)
rf_auc = pd.read_csv(base_dir / "results_random_forest_corrected/matrix_AUC_ROC.csv", index_col=0)

print("=" * 80)
print("元データ（Enhanced IRL AUC-ROC）")
print("=" * 80)
print(irl_auc)
print()

# 転置・反転
irl_auc_transposed = np.flipud(irl_auc.values.T)
rf_auc_transposed = np.flipud(rf_auc.values.T)

# マスク作成
mask = np.zeros((4, 4), dtype=bool)
for i in range(4):
    for j in range(4):
        eval_period_idx = 3 - i
        train_period_idx = j
        if train_period_idx > eval_period_idx:
            mask[i, j] = True

print("=" * 80)
print("検証結果")
print("=" * 80)
print(f"16セル全体平均（IRL AUC-ROC）: {irl_auc_transposed.mean():.10f}")
print(f"comparison_summaryの値:       {comparison_summary.loc[0, 'Enhanced_IRL_mean']:.10f}")
print(f"一致: {abs(irl_auc_transposed.mean() - comparison_summary.loc[0, 'Enhanced_IRL_mean']) < 1e-9}")
print()

print(f"有効10セル平均（IRL AUC-ROC）: {np.nanmean(irl_auc_transposed[~mask]):.10f}")
print(f"practical_summaryの値:        {practical_summary.loc[0, 'Enhanced_IRL_mean']:.10f}")
print(f"一致: {abs(np.nanmean(irl_auc_transposed[~mask]) - practical_summary.loc[0, 'Enhanced_IRL_mean']) < 1e-9}")
print()

print("=" * 80)
print("マスクと有効セル")
print("=" * 80)
print("マスク（True=無効）:")
print(mask)
print()
print(f"有効セル数: {(~mask).sum()}")
print()

print("有効セルの位置（表示座標）:")
period_labels = ['0-3m', '3-6m', '6-9m', '9-12m']
for i in range(4):
    for j in range(4):
        if not mask[i, j]:
            eval_idx = 3 - i
            train_idx = j
            print(f"  [{i},{j}] Train={period_labels[train_idx]}, Eval={period_labels[eval_idx]}, Value={irl_auc_transposed[i,j]:.4f}")
