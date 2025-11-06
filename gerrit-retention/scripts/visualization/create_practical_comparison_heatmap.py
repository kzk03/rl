"""
実用的評価（対角線+未来）のみを強調した比較ヒートマップを作成
過去期間の評価はグレーアウト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_matrix(csv_path):
    """CSVマトリクスを読み込み"""
    df = pd.read_csv(csv_path, index_col=0)
    return df


def create_practical_comparison(
    irl_matrix,
    lr_matrix,
    rf_matrix,
    output_path,
):
    """実用的評価のみを強調した比較ヒートマップを作成"""

    # 差分を計算
    diff_lr = irl_matrix.values - lr_matrix.values
    diff_rf = irl_matrix.values - rf_matrix.values

    # 実用的評価マスク（対角線+未来のみTrue）
    practical_mask = np.zeros((4, 4), dtype=bool)
    for i in range(4):
        for j in range(i, 4):  # 対角線+未来
            practical_mask[i, j] = True

    # 図を作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    # 左: IRL vs LR
    ax1 = axes[0]

    # グレーアウト用のマスク表示
    masked_diff_lr = np.ma.masked_where(~practical_mask, diff_lr)
    gray_diff_lr = np.ma.masked_where(practical_mask, diff_lr)

    # グレーセルを表示
    ax1.imshow(gray_diff_lr, cmap='gray', aspect='auto', vmin=-0.3, vmax=0.3, alpha=0.3)

    # 実用的評価セルを表示
    im1 = ax1.imshow(masked_diff_lr, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)

    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(periods)
    ax1.set_yticklabels(periods)
    ax1.set_xlabel('Training Period', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Evaluation Period', fontsize=11, fontweight='bold')
    ax1.set_title('IRL+LSTM - Logistic Regression\n(Practical Evaluation Only)',
                  fontsize=12, fontweight='bold', pad=10)

    # 値を表示（実用的評価セルのみ）
    for i in range(4):
        for j in range(4):
            if practical_mask[i, j]:
                text_color = 'white' if abs(diff_lr[i, j]) > 0.08 else 'black'
                sign = '+' if diff_lr[i, j] > 0 else ''
                ax1.text(j, i, f'{sign}{diff_lr[i, j]:.3f}',
                        ha='center', va='center', color=text_color,
                        fontsize=10, fontweight='bold')
            else:
                # グレーアウトセルは表示しない
                ax1.text(j, i, '—',
                        ha='center', va='center', color='gray',
                        fontsize=14, alpha=0.5)

    # カラーバー
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('AUC-ROC Difference', fontsize=10)

    # 右: IRL vs RF
    ax2 = axes[1]

    # グレーアウト用のマスク表示
    masked_diff_rf = np.ma.masked_where(~practical_mask, diff_rf)
    gray_diff_rf = np.ma.masked_where(practical_mask, diff_rf)

    # グレーセルを表示
    ax2.imshow(gray_diff_rf, cmap='gray', aspect='auto', vmin=-0.3, vmax=0.3, alpha=0.3)

    # 実用的評価セルを表示
    im2 = ax2.imshow(masked_diff_rf, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)

    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(periods)
    ax2.set_yticklabels(periods)
    ax2.set_xlabel('Training Period', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Evaluation Period', fontsize=11, fontweight='bold')
    ax2.set_title('IRL+LSTM - Random Forest\n(Practical Evaluation Only)',
                  fontsize=12, fontweight='bold', pad=10)

    # 値を表示（実用的評価セルのみ）
    for i in range(4):
        for j in range(4):
            if practical_mask[i, j]:
                text_color = 'white' if abs(diff_rf[i, j]) > 0.08 else 'black'
                sign = '+' if diff_rf[i, j] > 0 else ''
                ax2.text(j, i, f'{sign}{diff_rf[i, j]:.3f}',
                        ha='center', va='center', color=text_color,
                        fontsize=10, fontweight='bold')
            else:
                # グレーアウトセルは表示しない
                ax2.text(j, i, '—',
                        ha='center', va='center', color='gray',
                        fontsize=14, alpha=0.5)

    # カラーバー
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('AUC-ROC Difference', fontsize=10)

    # 全体タイトル
    fig.suptitle('Proposed Method (IRL+LSTM) vs Baselines: Practical Evaluation (3-Month Windows)',
                 fontsize=14, fontweight='bold', y=1.02)

    # 統計情報を追加（実用的評価セルのみ）
    practical_irl = np.array([irl_matrix.values[i, j] for i in range(4) for j in range(i, 4)])
    practical_lr = np.array([lr_matrix.values[i, j] for i in range(4) for j in range(i, 4)])
    practical_rf = np.array([rf_matrix.values[i, j] for i in range(4) for j in range(i, 4)])

    practical_diff_lr = practical_irl - practical_lr
    practical_diff_rf = practical_irl - practical_rf

    stats_text = (
        f'Practical Evaluation Statistics (10 cells - diagonal + future only):\n'
        f'IRL: {practical_irl.mean():.3f} ± {practical_irl.std():.3f}  |  '
        f'LR: {practical_lr.mean():.3f} ± {practical_lr.std():.3f}  |  '
        f'RF: {practical_rf.mean():.3f} ± {practical_rf.std():.3f}\n'
        f'IRL vs LR: {practical_diff_lr.mean():+.3f}  |  IRL vs RF: {practical_diff_rf.mean():+.3f}\n'
        f'Gray cells (past evaluation) are excluded from practical scenarios'
    )

    fig.text(0.5, -0.05, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Practical comparison heatmap saved: {output_path}")

    # 統計情報を表示
    print("\n=== Practical Evaluation Performance (Diagonal + Future Only) ===")
    print(f"IRL+LSTM:          {practical_irl.mean():.3f} ± {practical_irl.std():.3f}")
    print(f"Logistic Regression: {practical_lr.mean():.3f} ± {practical_lr.std():.3f}")
    print(f"Random Forest:       {practical_rf.mean():.3f} ± {practical_rf.std():.3f}")
    print(f"\nIRL vs LR: {practical_diff_lr.mean():+.3f} (average difference)")
    print(f"IRL vs RF: {practical_diff_rf.mean():+.3f} (average difference)")

    # 個別セルの表示
    print("\n=== Individual Cell Performance ===")
    print(f"{'Train':<8} {'Eval':<8} {'IRL':<8} {'LR':<8} {'RF':<8} {'IRL-LR':<10} {'IRL-RF':<10}")
    print("-" * 70)
    for i in range(4):
        for j in range(i, 4):
            irl_val = irl_matrix.values[i, j]
            lr_val = lr_matrix.values[i, j]
            rf_val = rf_matrix.values[i, j]
            print(f"{periods[i]:<8} {periods[j]:<8} {irl_val:.3f}    {lr_val:.3f}    "
                  f"{rf_val:.3f}    {irl_val-lr_val:+.3f}      {irl_val-rf_val:+.3f}")


def main():
    parser = argparse.ArgumentParser(description='Create practical comparison heatmap')
    parser.add_argument('--irl-matrix', type=str, required=True,
                       help='Path to IRL matrix CSV')
    parser.add_argument('--lr-matrix', type=str, required=True,
                       help='Path to LR matrix CSV')
    parser.add_argument('--rf-matrix', type=str, required=True,
                       help='Path to RF matrix CSV')
    parser.add_argument('--output', type=str, required=True,
                       help='Output PNG path')

    args = parser.parse_args()

    # マトリクスを読み込み
    irl_matrix = load_matrix(args.irl_matrix)
    lr_matrix = load_matrix(args.lr_matrix)
    rf_matrix = load_matrix(args.rf_matrix)

    # ヒートマップを作成
    create_practical_comparison(
        irl_matrix=irl_matrix,
        lr_matrix=lr_matrix,
        rf_matrix=rf_matrix,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
