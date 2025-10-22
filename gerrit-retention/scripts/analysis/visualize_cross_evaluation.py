#!/usr/bin/env python3
"""
クロス評価結果の可視化

訓練ラベル × 評価範囲のヒートマップを生成
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio']
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    # データ読み込み
    df = pd.read_csv(args.results)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # メトリクス
    metrics = ['auc_pr', 'f1', 'precision', 'recall']
    metric_names = {
        'auc_pr': 'AUC-PR',
        'f1': 'F1スコア',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    # ヒートマップ作成
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        pivot = df.pivot(index='train_range', columns='eval_range', values=metric)
        
        ax = axes[idx]
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0.5,
            vmax=1.0,
            ax=ax,
            cbar_kws={'label': metric_names[metric]},
            linewidths=0.5
        )
        
        ax.set_title(f'{metric_names[metric]}のクロス評価', fontsize=14, fontweight='bold')
        ax.set_xlabel('評価範囲', fontsize=12)
        ax.set_ylabel('訓練ラベル', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_evaluation_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"ヒートマップを保存: {output_dir / 'cross_evaluation_heatmaps.png'}")
    plt.close()
    
    # サマリーテーブル
    print("\n" + "="*80)
    print("クロス評価結果サマリー")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()

