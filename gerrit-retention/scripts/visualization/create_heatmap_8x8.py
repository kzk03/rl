#!/usr/bin/env python3
"""
8×8行列のヒートマップ可視化スクリプト

学習期間（3-24ヶ月） × 予測期間（3-24ヶ月）の評価結果を
ヒートマップで可視化します。
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_heatmap(
    csv_path: Path,
    output_dir: Path,
    metric: str = "auc_roc",
    title_suffix: str = ""
):
    """
    CSVファイルから8×8ヒートマップを作成

    Args:
        csv_path: 評価結果CSVのパス
        output_dir: 出力ディレクトリ
        metric: 可視化するメトリクス（auc_roc, auc_pr, f1, precision, recall, accuracy）
        title_suffix: タイトルの接尾辞
    """
    # データ読み込み
    df = pd.read_csv(csv_path)

    # 行列形式に変換
    matrix = df.pivot(
        index='history_months',
        columns='target_months',
        values=metric
    )

    # メトリクス名の表示用マッピング
    metric_names = {
        'auc_roc': 'AUC-ROC',
        'auc_pr': 'AUC-PR',
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'accuracy': 'Accuracy'
    }

    metric_display = metric_names.get(metric, metric.upper())

    # ヒートマップ作成
    plt.figure(figsize=(12, 10))

    # カラーマップの設定（メトリクスに応じて調整）
    if metric in ['auc_roc', 'auc_pr', 'f1', 'accuracy']:
        vmin, vmax = 0.5, 1.0
        cmap = 'RdYlGn'
    elif metric == 'precision':
        vmin, vmax = 0.5, 1.0
        cmap = 'Blues'
    elif metric == 'recall':
        vmin, vmax = 0.0, 1.0
        cmap = 'Oranges'
    else:
        vmin, vmax = None, None
        cmap = 'viridis'

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': metric_display},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(f'{metric_display} - Learning Period × Target Period{title_suffix}',
              fontsize=16, pad=20)
    plt.xlabel('Prediction Period (months)', fontsize=14, labelpad=10)
    plt.ylabel('Learning Period (months)', fontsize=14, labelpad=10)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 保存
    output_path = output_dir / f'heatmap_{metric}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved: {output_path}")
    plt.close()


def create_all_heatmaps(csv_path: Path, output_dir: Path, title_suffix: str = ""):
    """全メトリクスのヒートマップを作成"""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'accuracy']

    print(f"\n{'='*60}")
    print(f"ヒートマップ作成中: {csv_path.name}")
    print(f"{'='*60}\n")

    for metric in metrics:
        try:
            create_heatmap(csv_path, output_dir, metric, title_suffix)
        except Exception as e:
            print(f"⚠️  {metric} のヒートマップ作成に失敗: {e}")

    print(f"\n{'='*60}")
    print(f"✅ 全てのヒートマップが作成されました！")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"{'='*60}\n")


def create_comparison_heatmap(csv_path: Path, output_dir: Path):
    """
    複数メトリクスを1つの図にまとめた比較ヒートマップを作成
    """
    df = pd.read_csv(csv_path)

    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'accuracy']
    metric_names = ['AUC-ROC', 'AUC-PR', 'F1', 'Precision', 'Recall', 'Accuracy']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        matrix = df.pivot(
            index='history_months',
            columns='target_months',
            values=metric
        )

        # カラーマップ設定
        if metric in ['auc_roc', 'auc_pr', 'f1', 'accuracy']:
            vmin, vmax = 0.5, 1.0
            cmap = 'RdYlGn'
        elif metric == 'precision':
            vmin, vmax = 0.5, 1.0
            cmap = 'Blues'
        elif metric == 'recall':
            vmin, vmax = 0.0, 1.0
            cmap = 'Oranges'
        else:
            vmin, vmax = None, None
            cmap = 'viridis'

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar_kws={'label': name},
            linewidths=0.3,
            linecolor='gray',
            annot_kws={'fontsize': 8}
        )

        ax.set_title(name, fontsize=12, pad=10)
        ax.set_xlabel('Prediction Period (months)', fontsize=10)
        ax.set_ylabel('Learning Period (months)', fontsize=10)
        ax.tick_params(labelsize=9)

    plt.suptitle('IRL Model Performance Matrix (8×8)', fontsize=16, y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'heatmap_comparison_all.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison heatmap saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create 8×8 heatmap visualizations')
    parser.add_argument('--csv', type=Path, required=True,
                       help='Path to evaluation results CSV')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for heatmaps')
    parser.add_argument('--title-suffix', type=str, default='',
                       help='Suffix to add to plot titles')

    args = parser.parse_args()

    # 個別ヒートマップ作成
    create_all_heatmaps(args.csv, args.output, args.title_suffix)

    # 比較ヒートマップ作成
    create_comparison_heatmap(args.csv, args.output)


if __name__ == '__main__':
    main()
