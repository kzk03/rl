#!/usr/bin/env python3
"""
Epoch 50 のクロス評価結果をヒートマップで可視化
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use('Agg')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic']
plt.rcParams['font.size'] = 10

def load_metrics(base_dir: Path, train_period: str, eval_period: str) -> dict:
    """メトリクスを読み込む"""
    metrics_path = base_dir / f'train_{train_period}' / f'eval_{eval_period}' / 'metrics.json'
    
    if not metrics_path.exists():
        print(f"❌ {metrics_path} が存在しません")
        return None
    
    with open(metrics_path) as f:
        return json.load(f)

def create_heatmap_matrix(base_dir: Path, periods: list, metric_key: str) -> np.ndarray:
    """ヒートマップ用のマトリックスを作成"""
    n = len(periods)
    matrix = np.zeros((n, n))
    
    for i, eval_period in enumerate(periods):
        for j, train_period in enumerate(periods):
            metrics = load_metrics(base_dir, train_period, eval_period)
            if metrics and metric_key in metrics:
                matrix[i, j] = metrics[metric_key]
            else:
                matrix[i, j] = np.nan
    
    return matrix

def create_single_heatmap(ax, matrix, periods, title, vmin=0.0, vmax=1.0):
    """個別のヒートマップを作成"""
    # ヒートマップ描画
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        vmin=vmin,
        vmax=vmax,
        xticklabels=periods,
        yticklabels=periods,
        ax=ax,
        cbar_kws={'label': title}
    )
    
    # 軸ラベル設定（y軸を下から上に：0-3m, 3-6m, 6-9m, 9-12m）
    ax.set_xlabel('訓練期間', fontsize=11, fontweight='bold')
    ax.set_ylabel('評価期間', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # y軸の順序を逆にする（下から上に 0-3m, 3-6m, 6-9m, 9-12m）
    ax.set_yticks(np.arange(len(periods)) + 0.5)
    ax.set_yticklabels(periods[::-1], rotation=0)
    
    # x軸はそのまま（左から右に 0-3m, 3-6m, 6-9m, 9-12m）
    ax.set_xticks(np.arange(len(periods)) + 0.5)
    ax.set_xticklabels(periods, rotation=45, ha='right')

def main():
    base_dir = Path('outputs/review_acceptance_cross_eval_nova')
    output_dir = base_dir / 'heatmaps'
    output_dir.mkdir(exist_ok=True)
    
    periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    # 評価指標
    metrics = {
        'auc_pr': 'AUC-PR',
        'auc_roc': 'AUC-ROC',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
    }
    
    print("## 📊 Epoch 50 ヒートマップ作成中...")
    print()
    
    # 個別ヒートマップ作成
    for metric_key, metric_name in metrics.items():
        matrix = create_heatmap_matrix(base_dir, periods, metric_key)
        
        # y軸を逆順にする（下から上に 0-3m, 3-6m, 6-9m, 9-12m）
        matrix_reversed = matrix[::-1, :]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        create_single_heatmap(ax, matrix_reversed, periods, metric_name, vmin=0.0, vmax=1.0)
        
        output_path = output_dir / f'heatmap_{metric_key}.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {output_path.name}")
    
    # 統合ヒートマップ作成（2行3列）
    fig = plt.figure(figsize=(18, 12))
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items(), 1):
        matrix = create_heatmap_matrix(base_dir, periods, metric_key)
        
        # y軸を逆順にする
        matrix_reversed = matrix[::-1, :]
        
        ax = fig.add_subplot(2, 3, idx)
        create_single_heatmap(ax, matrix_reversed, periods, metric_name, vmin=0.0, vmax=1.0)
    
    # 空白の6番目のサブプロットを削除
    fig.delaxes(fig.axes[5])
    
    output_path = output_dir / 'heatmap_combined_epoch50.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print()
    print(f"✅ {output_path.name} (統合版)")
    print()
    print(f"📂 出力先: {output_dir}/")

if __name__ == '__main__':
    main()

