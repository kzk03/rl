#!/usr/bin/env python3
"""
Baseline IRL実用的な10段階評価ヒートマップ作成

訓練期間 ≤ 評価期間の組み合わせのみ表示
- 横軸: 訓練期間 (0-3m, 3-6m, 6-9m, 9-12m)
- 縦軸: 予測期間 (下から0-3m, 3-6m, 6-9m, 9-12m)
- 無効な組み合わせはグレーアウト
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
baseline_irl_dir = Path("experiments/nova_review_acceptance/results_importants")
rf_dir = Path("experiments/nova_review_acceptance/results_random_forest_corrected")
output_dir = Path("experiments/nova_review_acceptance/baseline_practical_comparison_heatmaps")
output_dir.mkdir(exist_ok=True)

# AUC-ROC読み込み
irl_auc_roc = pd.read_csv(baseline_irl_dir / "matrix_AUC_ROC.csv", index_col=0)
rf_auc_roc = pd.read_csv(rf_dir / "matrix_AUC_ROC.csv", index_col=0)

# AUC-PR読み込み
irl_auc_pr = pd.read_csv(baseline_irl_dir / "matrix_AUC_PR.csv", index_col=0)
rf_auc_pr = pd.read_csv(rf_dir / "matrix_AUC_PR.csv", index_col=0)

# F1読み込み
irl_f1 = pd.read_csv(baseline_irl_dir / "matrix_F1_SCORE.csv", index_col=0)
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

# マスク作成: 訓練期間 > 評価期間 の組み合わせをマスク
# flipud後の座標系: 行0=9-12m(上), 行3=0-3m(下), 列0=0-3m(左), 列3=9-12m(右)
# 有効: 訓練期間(列) ≤ 評価期間(行)
mask = np.zeros((4, 4), dtype=bool)
for i in range(4):  # 行（評価期間、flipud後）
    for j in range(4):  # 列（訓練期間）
        eval_period_idx = 3 - i  # 実際の期間インデックス（0=0-3m, 3=9-12m）
        train_period_idx = j
        if train_period_idx > eval_period_idx:  # 訓練期間 > 評価期間の場合マスク
            mask[i, j] = True

print("=" * 80)
print("Baseline IRL 実用的な10段階評価")
print("=" * 80)
print(f"\n有効な評価セル数: {(~mask).sum()} / 16")
print(f"\nマスクされるセル数: {mask.sum()} / 16")

# カスタムカラーマップ: グレーアウト用
def create_masked_heatmap(ax, data, mask, title, cmap='RdYlGn', vmin=0.5, vmax=1.0, annot_size=14):
    """マスク付きヒートマップ作成"""
    # データをマスク
    masked_data = data.copy()
    masked_data[mask] = np.nan

    # ヒートマップ描画
    sns.heatmap(
        masked_data,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={'label': 'Score'},
        annot_kws={'size': annot_size, 'weight': 'bold'},
        linewidths=2,
        linecolor='white',
        square=True,
        mask=mask,
        cbar=True
    )

    # グレーアウトセルを追加
    for i in range(4):
        for j in range(4):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='lightgray', alpha=0.5))
                ax.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center',
                       fontsize=12, color='gray', weight='bold')

    ax.set_title(title, fontweight='bold', fontsize=16, pad=15)
    ax.set_xlabel('Training Period', fontsize=14, fontweight='bold')
    ax.set_ylabel('Evaluation Period (Prediction Window)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(period_labels, fontsize=12)
    ax.set_yticklabels(period_labels[::-1], fontsize=12)

    # 平均値計算（有効セルのみ）
    valid_mean = np.nanmean(masked_data)
    ax.text(0.02, 0.98, f'Mean (valid): {valid_mean:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 図1: Baseline IRL vs Random Forest 比較 (AUC-ROC)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Practical 10-Level Evaluation: AUC-ROC Comparison (Baseline IRL)\n(Valid: Training Period ≤ Evaluation Period)',
             fontsize=18, fontweight='bold', y=0.98)

create_masked_heatmap(
    axes[0],
    irl_auc_roc_transposed,
    mask,
    'Baseline IRL (No Attention)',
    cmap='Blues',
    vmin=0.0,
    vmax=1.0
)

create_masked_heatmap(
    axes[1],
    rf_auc_roc_transposed,
    mask,
    'Random Forest',
    cmap='Blues',
    vmin=0.0,
    vmax=1.0
)

plt.tight_layout()
plt.savefig(output_dir / 'baseline_practical_auc_roc_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 保存: {output_dir / 'baseline_practical_auc_roc_comparison.png'}")

# 図2: Baseline IRL vs Random Forest 比較 (AUC-PR)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Practical 10-Level Evaluation: AUC-PR Comparison (Baseline IRL)\n(Valid: Training Period ≤ Evaluation Period)',
             fontsize=18, fontweight='bold', y=0.98)

create_masked_heatmap(
    axes[0],
    irl_auc_pr_transposed,
    mask,
    'Baseline IRL (No Attention)',
    cmap='Blues',
    vmin=0.0,
    vmax=1.0
)

create_masked_heatmap(
    axes[1],
    rf_auc_pr_transposed,
    mask,
    'Random Forest',
    cmap='Blues',
    vmin=0.0,
    vmax=1.0
)

plt.tight_layout()
plt.savefig(output_dir / 'baseline_practical_auc_pr_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ 保存: {output_dir / 'baseline_practical_auc_pr_comparison.png'}")

# 図3: Baseline IRL vs Random Forest 比較 (F1 Score)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Practical 10-Level Evaluation: F1 Score Comparison (Baseline IRL)\n(Valid: Training Period ≤ Evaluation Period)',
             fontsize=18, fontweight='bold', y=0.98)

create_masked_heatmap(
    axes[0],
    irl_f1_transposed,
    mask,
    'Baseline IRL (No Attention)',
    cmap='Blues',
    vmin=0.0,
    vmax=1.0
)

create_masked_heatmap(
    axes[1],
    rf_f1_transposed,
    mask,
    'Random Forest',
    cmap='Blues',
    vmin=0.0,
    vmax=1.0
)

plt.tight_layout()
plt.savefig(output_dir / 'baseline_practical_f1_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ 保存: {output_dir / 'baseline_practical_f1_comparison.png'}")

# 図4: 差分ヒートマップ (Baseline IRL - Random Forest)
diff_auc_roc = irl_auc_roc_transposed - rf_auc_roc_transposed
diff_auc_pr = irl_auc_pr_transposed - rf_auc_pr_transposed
diff_f1 = irl_f1_transposed - rf_f1_transposed

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle('Performance Difference: Baseline IRL - Random Forest\n(Positive = Baseline IRL Better, Valid Combinations Only)',
             fontsize=18, fontweight='bold', y=0.98)

# AUC-ROC差分
max_abs_diff_roc = max(abs(np.nanmin(diff_auc_roc[~mask])), abs(np.nanmax(diff_auc_roc[~mask])))
create_masked_heatmap(
    axes[0],
    diff_auc_roc,
    mask,
    f'AUC-ROC Difference\n(Mean: {np.nanmean(diff_auc_roc[~mask]):+.4f})',
    cmap='RdBu_r',
    vmin=-max_abs_diff_roc,
    vmax=max_abs_diff_roc
)

# AUC-PR差分
max_abs_diff_pr = max(abs(np.nanmin(diff_auc_pr[~mask])), abs(np.nanmax(diff_auc_pr[~mask])))
create_masked_heatmap(
    axes[1],
    diff_auc_pr,
    mask,
    f'AUC-PR Difference\n(Mean: {np.nanmean(diff_auc_pr[~mask]):+.4f})',
    cmap='RdBu_r',
    vmin=-max_abs_diff_pr,
    vmax=max_abs_diff_pr
)

# F1差分
max_abs_diff_f1 = max(abs(np.nanmin(diff_f1[~mask])), abs(np.nanmax(diff_f1[~mask])))
create_masked_heatmap(
    axes[2],
    diff_f1,
    mask,
    f'F1 Score Difference\n(Mean: {np.nanmean(diff_f1[~mask]):+.4f})',
    cmap='RdBu_r',
    vmin=-max_abs_diff_f1,
    vmax=max_abs_diff_f1
)

plt.tight_layout()
plt.savefig(output_dir / 'baseline_practical_difference_heatmaps.png', dpi=300, bbox_inches='tight')
print(f"✅ 保存: {output_dir / 'baseline_practical_difference_heatmaps.png'}")

# 統計サマリー保存（有効セルのみ）
summary = {
    'metric': ['AUC-ROC', 'AUC-PR', 'F1 Score'],
    'Baseline_IRL_mean': [
        np.nanmean(irl_auc_roc_transposed[~mask]),
        np.nanmean(irl_auc_pr_transposed[~mask]),
        np.nanmean(irl_f1_transposed[~mask])
    ],
    'Random_Forest_mean': [
        np.nanmean(rf_auc_roc_transposed[~mask]),
        np.nanmean(rf_auc_pr_transposed[~mask]),
        np.nanmean(rf_f1_transposed[~mask])
    ],
    'Difference': [
        np.nanmean(diff_auc_roc[~mask]),
        np.nanmean(diff_auc_pr[~mask]),
        np.nanmean(diff_f1[~mask])
    ],
    'IRL_better_cells': [
        (diff_auc_roc[~mask] > 0).sum(),
        (diff_auc_pr[~mask] > 0).sum(),
        (diff_f1[~mask] > 0).sum()
    ],
    'RF_better_cells': [
        (diff_auc_roc[~mask] < 0).sum(),
        (diff_auc_pr[~mask] < 0).sum(),
        (diff_f1[~mask] < 0).sum()
    ],
    'Valid_cells': [10, 10, 10]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(output_dir / 'baseline_practical_comparison_summary.csv', index=False)
print(f"✅ 保存: {output_dir / 'baseline_practical_comparison_summary.csv'}")

print("\n" + "=" * 80)
print("Baseline IRL 実用的な10段階評価サマリー")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("有効な評価組み合わせ:")
print("=" * 80)
# マスクが反転されているので、元のインデックスに戻して表示
for i in range(4):
    for j in range(4):
        # mask[i, j]はflipudされた後の座標
        # 元の評価期間インデックス = 3 - i（下から上に並んでいるため）
        eval_idx = 3 - i
        train_idx = j
        if not mask[i, j]:  # 有効なセル
            print(f"  Train: {period_labels[train_idx]} → Eval: {period_labels[eval_idx]}")

print("\n" + "=" * 80)
print("完了！")
print(f"結果ディレクトリ: {output_dir}")
print("=" * 80)
