"""
提案手法（IRL+LSTM）とベースライン手法の2×2シンプル比較ヒートマップを作成

左: IRL+LSTM vs Logistic Regression
右: IRL+LSTM vs Random Forest
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
    return df.values


def create_simple_comparison(
    irl_matrix,
    lr_matrix,
    rf_matrix,
    output_path,
):
    """シンプルな2×2比較ヒートマップを作成"""

    # 差分を計算
    diff_lr = irl_matrix - lr_matrix
    diff_rf = irl_matrix - rf_matrix

    # 図を作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    # 左: IRL vs LR
    ax1 = axes[0]
    im1 = ax1.imshow(diff_lr, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(periods)
    ax1.set_yticklabels(periods)
    ax1.set_xlabel('Training Period', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Evaluation Period', fontsize=11, fontweight='bold')
    ax1.set_title('IRL+LSTM - Logistic Regression', fontsize=12, fontweight='bold', pad=10)

    # 値を表示
    for i in range(4):
        for j in range(4):
            text_color = 'white' if abs(diff_lr[i, j]) > 0.08 else 'black'
            sign = '+' if diff_lr[i, j] > 0 else ''
            ax1.text(j, i, f'{sign}{diff_lr[i, j]:.3f}',
                    ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')

    # カラーバー
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('AUC-ROC Difference', fontsize=10)

    # 右: IRL vs RF
    ax2 = axes[1]
    im2 = ax2.imshow(diff_rf, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(periods)
    ax2.set_yticklabels(periods)
    ax2.set_xlabel('Training Period', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Evaluation Period', fontsize=11, fontweight='bold')
    ax2.set_title('IRL+LSTM - Random Forest', fontsize=12, fontweight='bold', pad=10)

    # 値を表示
    for i in range(4):
        for j in range(4):
            text_color = 'white' if abs(diff_rf[i, j]) > 0.08 else 'black'
            sign = '+' if diff_rf[i, j] > 0 else ''
            ax2.text(j, i, f'{sign}{diff_rf[i, j]:.3f}',
                    ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')

    # カラーバー
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('AUC-ROC Difference', fontsize=10)

    # 全体タイトル
    fig.suptitle('Proposed Method (IRL+LSTM) vs Baselines: Performance Comparison (3-Month Windows)',
                 fontsize=14, fontweight='bold', y=1.02)

    # 統計情報を追加
    stats_text = (
        f'Overall Statistics (16 cells):\n'
        f'IRL: {irl_matrix.mean():.3f} ± {irl_matrix.std():.3f}  |  '
        f'LR: {lr_matrix.mean():.3f} ± {lr_matrix.std():.3f}  |  '
        f'RF: {rf_matrix.mean():.3f} ± {rf_matrix.std():.3f}\n'
        f'IRL vs LR: {diff_lr.mean():+.3f}  |  IRL vs RF: {diff_rf.mean():+.3f}\n'
        f'Positive: IRL better (red)  |  Negative: Baseline better (blue)'
    )

    fig.text(0.5, -0.05, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Simple comparison heatmap saved: {output_path}")

    # 統計情報を表示
    print("\n=== Performance Summary ===")
    print(f"IRL+LSTM:          {irl_matrix.mean():.3f} ± {irl_matrix.std():.3f}")
    print(f"Logistic Regression: {lr_matrix.mean():.3f} ± {lr_matrix.std():.3f}")
    print(f"Random Forest:       {rf_matrix.mean():.3f} ± {rf_matrix.std():.3f}")
    print(f"\nIRL vs LR: {diff_lr.mean():+.3f} (average difference)")
    print(f"IRL vs RF: {diff_rf.mean():+.3f} (average difference)")

    # 対角線+未来のみ
    diagonal_future = []
    for i in range(4):
        for j in range(i, 4):
            diagonal_future.append((i, j))

    irl_diag = np.array([irl_matrix[i, j] for i, j in diagonal_future])
    lr_diag = np.array([lr_matrix[i, j] for i, j in diagonal_future])
    rf_diag = np.array([rf_matrix[i, j] for i, j in diagonal_future])

    print("\n=== Diagonal + Future (10 cells) ===")
    print(f"IRL+LSTM:          {irl_diag.mean():.3f} ± {irl_diag.std():.3f}")
    print(f"Logistic Regression: {lr_diag.mean():.3f} ± {lr_diag.std():.3f}")
    print(f"Random Forest:       {rf_diag.mean():.3f} ± {rf_diag.std():.3f}")
    print(f"\nIRL vs LR: {(irl_diag - lr_diag).mean():+.3f}")
    print(f"IRL vs RF: {(irl_diag - rf_diag).mean():+.3f}")


def main():
    parser = argparse.ArgumentParser(description='Create simple comparison heatmap')
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
    create_simple_comparison(
        irl_matrix=irl_matrix,
        lr_matrix=lr_matrix,
        rf_matrix=rf_matrix,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
