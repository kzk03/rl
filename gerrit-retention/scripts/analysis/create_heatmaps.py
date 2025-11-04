#!/usr/bin/env python3
"""
クロス評価結果からヒートマップを生成
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path("outputs/review_acceptance_cross_eval_nova")
train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

def collect_metrics():
    """全組み合わせのメトリクスを収集"""
    metrics_data = {}
    for metric_name in ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']:
        matrix = np.zeros((len(train_periods), len(eval_periods)))

        for i, train_period in enumerate(train_periods):
            for j, eval_period in enumerate(eval_periods):
                metrics_file = BASE_DIR / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"

                if metrics_file.exists():
                    with open(metrics_file) as f:
                        data = json.load(f)
                        matrix[i, j] = data.get(metric_name, 0.0)

        metrics_data[metric_name] = matrix

    return metrics_data

def create_heatmaps(metrics_data):
    """ヒートマップを作成"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric_name, matrix) in enumerate(metrics_data.items()):
        ax = axes[idx]

        # 最高値を見つける
        max_val = np.max(matrix)
        max_pos = np.unravel_index(np.argmax(matrix), matrix.shape)

        # カラーマップの範囲を設定
        if metric_name == 'auc_roc':
            vmin, vmax = 0.3, 0.9
        elif metric_name == 'auc_pr':
            vmin, vmax = 0.0, 0.8
        else:
            vmin, vmax = 0.0, 1.0

        # ヒートマップ作成
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=eval_periods, yticklabels=train_periods,
                    vmin=vmin, vmax=vmax,
                    ax=ax, cbar_kws={'label': metric_name.upper()})

        # 最高値にマーカー
        ax.add_patch(plt.Rectangle((max_pos[1], max_pos[0]), 1, 1,
                                   fill=False, edgecolor='red', lw=3))
        ax.text(max_pos[1] + 0.5, max_pos[0] - 0.15, '★ BEST',
                ha='center', va='top', color='red', fontsize=10, weight='bold')

        ax.set_title(f'{metric_name.upper()} Cross Evaluation\n(★ Best: {max_val:.3f})',
                    fontsize=12, weight='bold')
        ax.set_xlabel('Evaluation Period (Future Window)')
        ax.set_ylabel('Training Period (Future Window)')

    axes[-1].axis('off')  # 6番目は非表示
    plt.suptitle('Nova Project: Cross Evaluation Heatmaps\n(Improved Parameters: LR=0.0003, Dropout=0.1, Focal γ=1.0)',
                 fontsize=16, weight='bold')
    plt.tight_layout()

    output_file = BASE_DIR / 'all_metrics_heatmaps_improved.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")

    return output_file

def print_summary(metrics_data):
    """メトリクスサマリーを出力"""
    print()
    print("=" * 70)
    print("クロス評価サマリー（改善後）")
    print("=" * 70)

    for metric_name, matrix in metrics_data.items():
        max_val = np.max(matrix)
        max_pos = np.unravel_index(np.argmax(matrix), matrix.shape)
        train_period = train_periods[max_pos[0]]
        eval_period = eval_periods[max_pos[1]]

        print(f"\n{metric_name.upper()}:")
        print(f"  最高値: {max_val:.3f}")
        print(f"  組み合わせ: train_{train_period} → eval_{eval_period}")
        print(f"  平均値: {matrix.mean():.3f}")
        print(f"  標準偏差: {matrix.std():.3f}")

    print()
    print("=" * 70)
    print("改善前との比較")
    print("=" * 70)
    print("\n修正前（train_6-9m → eval_0-3m）:")
    print("  AUC-ROC: 0.453")
    print("  Precision: 0.488")
    print("  Recall: 1.000 (異常)")
    print("\n修正後（train_6-9m → eval_0-3m）:")
    auc_roc_69_03 = metrics_data['auc_roc'][2, 0]
    precision_69_03 = metrics_data['precision'][2, 0]
    recall_69_03 = metrics_data['recall'][2, 0]
    print(f"  AUC-ROC: {auc_roc_69_03:.3f} ({(auc_roc_69_03/0.453 - 1)*100:+.1f}%)")
    print(f"  Precision: {precision_69_03:.3f}")
    print(f"  Recall: {recall_69_03:.3f} (正常化)")
    print()
    print("=" * 70)

if __name__ == "__main__":
    print("メトリクス収集中...")
    metrics = collect_metrics()

    print("ヒートマップ作成中...")
    output_file = create_heatmaps(metrics)

    print_summary(metrics)

    print(f"\n✅ 完了！ヒートマップ: {output_file}")
