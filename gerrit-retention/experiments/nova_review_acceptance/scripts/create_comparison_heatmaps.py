#!/usr/bin/env python3
"""
Enhanced IRL vs Random Forest 比較ヒートマップ作成
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# データ読み込み
enhanced_irl_dir = Path("experiments/nova_review_acceptance/results_enhanced_irl")
rf_dir = Path("experiments/nova_review_acceptance/results_random_forest_corrected")
output_dir = Path("experiments/nova_review_acceptance/comparison_heatmaps")
output_dir.mkdir(exist_ok=True)

# AUC-ROC読み込み
irl_auc_roc = pd.read_csv(enhanced_irl_dir / "matrix_AUC_ROC.csv", index_col=0)
rf_auc_roc = pd.read_csv(rf_dir / "matrix_AUC_ROC.csv", index_col=0)

# AUC-PR読み込み
irl_auc_pr = pd.read_csv(enhanced_irl_dir / "matrix_AUC_PR.csv", index_col=0)
rf_auc_pr = pd.read_csv(rf_dir / "matrix_AUC_PR.csv", index_col=0)

# F1読み込み
irl_f1 = pd.read_csv(enhanced_irl_dir / "matrix_F1_SCORE.csv", index_col=0)
rf_f1 = pd.read_csv(rf_dir / "matrix_F1_SCORE.csv", index_col=0)

# 期間ラベル
period_labels = ['0-3m', '3-6m', '6-9m', '9-12m']

# データ変換: 左下を(train=0-3m, eval=0-3m)にする
# 元データ: 行=train, 列=eval
# 目標: 横軸=train(左から0-3m, 3-6m, 6-9m, 9-12m), 縦軸=eval(下から0-3m, 3-6m, 6-9m, 9-12m)
# → 行を反転するだけ（flipud）
irl_auc_roc_transposed = np.flipud(irl_auc_roc.values)
rf_auc_roc_transposed = np.flipud(rf_auc_roc.values)
irl_auc_pr_transposed = np.flipud(irl_auc_pr.values)
rf_auc_pr_transposed = np.flipud(rf_auc_pr.values)
irl_f1_transposed = np.flipud(irl_f1.values)
rf_f1_transposed = np.flipud(rf_f1.values)

# 差分計算 (Enhanced IRL - Random Forest)
diff_auc_roc = irl_auc_roc_transposed - rf_auc_roc_transposed
diff_auc_pr = irl_auc_pr_transposed - rf_auc_pr_transposed
diff_f1 = irl_f1_transposed - rf_f1_transposed

print("=" * 80)
print("Enhanced IRL vs Random Forest 比較")
print("=" * 80)
print(f"\nAUC-ROC平均: Enhanced IRL={irl_auc_roc_transposed.mean():.4f}, RF={rf_auc_roc_transposed.mean():.4f}, 差={diff_auc_roc.mean():.4f}")
print(f"AUC-PR平均:  Enhanced IRL={irl_auc_pr_transposed.mean():.4f}, RF={rf_auc_pr_transposed.mean():.4f}, 差={diff_auc_pr.mean():.4f}")
print(f"F1平均:      Enhanced IRL={irl_f1_transposed.mean():.4f}, RF={rf_f1_transposed.mean():.4f}, 差={diff_f1.mean():.4f}")

# 図1: 3つのメトリクス並べて比較 (Enhanced IRL vs RF)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Enhanced IRL (Attention) vs Random Forest - 4×4 Cross-Evaluation', fontsize=16, fontweight='bold')

# AUC-ROC
sns.heatmap(irl_auc_roc_transposed, annot=True, fmt='.3f', cmap='Blues', vmin=0.0, vmax=1.0,
            ax=axes[0, 0], cbar_kws={'label': 'AUC-ROC'})
axes[0, 0].set_title('Enhanced IRL (Attention) - AUC-ROC', fontweight='bold')
axes[0, 0].set_xlabel('Training Period')
axes[0, 0].set_ylabel('Evaluation Period')
axes[0, 0].set_xticklabels(period_labels)
axes[0, 0].set_yticklabels(period_labels[::-1])

sns.heatmap(rf_auc_roc_transposed, annot=True, fmt='.3f', cmap='Blues', vmin=0.0, vmax=1.0,
            ax=axes[1, 0], cbar_kws={'label': 'AUC-ROC'})
axes[1, 0].set_title('Random Forest - AUC-ROC', fontweight='bold')
axes[1, 0].set_xlabel('Training Period')
axes[1, 0].set_ylabel('Evaluation Period')
axes[1, 0].set_xticklabels(period_labels)
axes[1, 0].set_yticklabels(period_labels[::-1])

# AUC-PR
sns.heatmap(irl_auc_pr_transposed, annot=True, fmt='.3f', cmap='Blues', vmin=0.0, vmax=1.0,
            ax=axes[0, 1], cbar_kws={'label': 'AUC-PR'})
axes[0, 1].set_title('Enhanced IRL (Attention) - AUC-PR', fontweight='bold')
axes[0, 1].set_xlabel('Training Period')
axes[0, 1].set_ylabel('Evaluation Period')
axes[0, 1].set_xticklabels(period_labels)
axes[0, 1].set_yticklabels(period_labels[::-1])

sns.heatmap(rf_auc_pr_transposed, annot=True, fmt='.3f', cmap='Blues', vmin=0.0, vmax=1.0,
            ax=axes[1, 1], cbar_kws={'label': 'AUC-PR'})
axes[1, 1].set_title('Random Forest - AUC-PR', fontweight='bold')
axes[1, 1].set_xlabel('Training Period')
axes[1, 1].set_ylabel('Evaluation Period')
axes[1, 1].set_xticklabels(period_labels)
axes[1, 1].set_yticklabels(period_labels[::-1])

# F1 Score
sns.heatmap(irl_f1_transposed, annot=True, fmt='.3f', cmap='Blues', vmin=0.0, vmax=1.0,
            ax=axes[0, 2], cbar_kws={'label': 'F1 Score'})
axes[0, 2].set_title('Enhanced IRL (Attention) - F1 Score', fontweight='bold')
axes[0, 2].set_xlabel('Training Period')
axes[0, 2].set_ylabel('Evaluation Period')
axes[0, 2].set_xticklabels(period_labels)
axes[0, 2].set_yticklabels(period_labels[::-1])

sns.heatmap(rf_f1_transposed, annot=True, fmt='.3f', cmap='Blues', vmin=0.0, vmax=1.0,
            ax=axes[1, 2], cbar_kws={'label': 'F1 Score'})
axes[1, 2].set_title('Random Forest - F1 Score', fontweight='bold')
axes[1, 2].set_xlabel('Training Period')
axes[1, 2].set_ylabel('Evaluation Period')
axes[1, 2].set_xticklabels(period_labels)
axes[1, 2].set_yticklabels(period_labels[::-1])

plt.tight_layout()
plt.savefig(output_dir / 'comparison_all_metrics.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 保存: {output_dir / 'comparison_all_metrics.png'}")

# 図2: 差分ヒートマップ (Enhanced IRL - Random Forest)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Performance Difference: Enhanced IRL - Random Forest\n(Positive = Enhanced IRL Better)',
             fontsize=16, fontweight='bold')

# AUC-ROC差分
max_abs_diff_roc = max(abs(diff_auc_roc.min()), abs(diff_auc_roc.max()))
sns.heatmap(diff_auc_roc, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-max_abs_diff_roc, vmax=max_abs_diff_roc,
            ax=axes[0], cbar_kws={'label': 'Difference'})
axes[0].set_title(f'AUC-ROC Difference\n(Mean: {diff_auc_roc.mean():+.4f})', fontweight='bold')
axes[0].set_xlabel('Training Period')
axes[0].set_ylabel('Evaluation Period')
axes[0].set_xticklabels(period_labels)
axes[0].set_yticklabels(period_labels[::-1])

# AUC-PR差分
max_abs_diff_pr = max(abs(diff_auc_pr.min()), abs(diff_auc_pr.max()))
sns.heatmap(diff_auc_pr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-max_abs_diff_pr, vmax=max_abs_diff_pr,
            ax=axes[1], cbar_kws={'label': 'Difference'})
axes[1].set_title(f'AUC-PR Difference\n(Mean: {diff_auc_pr.mean():+.4f})', fontweight='bold')
axes[1].set_xlabel('Training Period')
axes[1].set_ylabel('Evaluation Period')
axes[1].set_xticklabels(period_labels)
axes[1].set_yticklabels(period_labels[::-1])

# F1差分
max_abs_diff_f1 = max(abs(diff_f1.min()), abs(diff_f1.max()))
sns.heatmap(diff_f1, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-max_abs_diff_f1, vmax=max_abs_diff_f1,
            ax=axes[2], cbar_kws={'label': 'Difference'})
axes[2].set_title(f'F1 Score Difference\n(Mean: {diff_f1.mean():+.4f})', fontweight='bold')
axes[2].set_xlabel('Training Period')
axes[2].set_ylabel('Evaluation Period')
axes[2].set_xticklabels(period_labels)
axes[2].set_yticklabels(period_labels[::-1])

plt.tight_layout()
plt.savefig(output_dir / 'difference_heatmaps.png', dpi=300, bbox_inches='tight')
print(f"✅ 保存: {output_dir / 'difference_heatmaps.png'}")

# 図3: サイドバイサイド比較 (AUC-ROCのみ、大きく)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('AUC-ROC Comparison: Enhanced IRL vs Random Forest', fontsize=16, fontweight='bold')

sns.heatmap(irl_auc_roc_transposed, annot=True, fmt='.3f', cmap='Blues', vmin=0.0, vmax=1.0,
            ax=axes[0], cbar_kws={'label': 'AUC-ROC'}, annot_kws={'size': 12})
axes[0].set_title(f'Enhanced IRL (Attention)\nMean: {irl_auc_roc_transposed.mean():.4f}', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Training Period', fontsize=12)
axes[0].set_ylabel('Evaluation Period', fontsize=12)
axes[0].set_xticklabels(period_labels)
axes[0].set_yticklabels(period_labels[::-1])

sns.heatmap(rf_auc_roc_transposed, annot=True, fmt='.3f', cmap='Blues', vmin=0.0, vmax=1.0,
            ax=axes[1], cbar_kws={'label': 'AUC-ROC'}, annot_kws={'size': 12})
axes[1].set_title(f'Random Forest\nMean: {rf_auc_roc_transposed.mean():.4f}', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Training Period', fontsize=12)
axes[1].set_ylabel('Evaluation Period', fontsize=12)
axes[1].set_xticklabels(period_labels)
axes[1].set_yticklabels(period_labels[::-1])

sns.heatmap(diff_auc_roc, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-0.1, vmax=0.1,
            ax=axes[2], cbar_kws={'label': 'Difference'}, annot_kws={'size': 12})
axes[2].set_title(f'Difference (IRL - RF)\nMean: {diff_auc_roc.mean():+.4f}', fontweight='bold', fontsize=14)
axes[2].set_xlabel('Training Period', fontsize=12)
axes[2].set_ylabel('Evaluation Period', fontsize=12)
axes[2].set_xticklabels(period_labels)
axes[2].set_yticklabels(period_labels[::-1])

plt.tight_layout()
plt.savefig(output_dir / 'auc_roc_comparison_large.png', dpi=300, bbox_inches='tight')
print(f"✅ 保存: {output_dir / 'auc_roc_comparison_large.png'}")

# 統計サマリー保存
summary = {
    'metric': ['AUC-ROC', 'AUC-PR', 'F1 Score'],
    'Enhanced_IRL_mean': [irl_auc_roc_transposed.mean(), irl_auc_pr_transposed.mean(), irl_f1_transposed.mean()],
    'Random_Forest_mean': [rf_auc_roc_transposed.mean(), rf_auc_pr_transposed.mean(), rf_f1_transposed.mean()],
    'Difference': [diff_auc_roc.mean(), diff_auc_pr.mean(), diff_f1.mean()],
    'IRL_better_cells': [(diff_auc_roc > 0).sum(), (diff_auc_pr > 0).sum(), (diff_f1 > 0).sum()],
    'RF_better_cells': [(diff_auc_roc < 0).sum(), (diff_auc_pr < 0).sum(), (diff_f1 < 0).sum()],
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(output_dir / 'comparison_summary.csv', index=False)
print(f"✅ 保存: {output_dir / 'comparison_summary.csv'}")

print("\n" + "=" * 80)
print("比較サマリー")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("完了！")
print(f"結果ディレクトリ: {output_dir}")
print("=" * 80)
