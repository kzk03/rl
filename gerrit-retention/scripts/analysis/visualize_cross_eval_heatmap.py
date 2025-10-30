#!/usr/bin/env python3
"""
クロス評価結果のヒートマップ可視化
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_heatmap(csv_path: Path, title: str, output_path: Path, fmt: str = '.3f'):
    """ヒートマップを作成"""
    
    # データ読み込み
    df = pd.read_csv(csv_path, index_col=0)
    
    # パーセント表記の場合は数値に変換
    if df.applymap(lambda x: isinstance(x, str) and '%' in str(x)).any().any():
        df = df.applymap(lambda x: float(str(x).replace('%', '')) / 100 if isinstance(x, str) and '%' in str(x) else x)
        fmt = '.1%'
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        df.astype(float),
        annot=True,
        fmt=fmt,
        cmap='YlOrRd',
        cbar_kws={'label': title},
        linewidths=0.5,
        ax=ax,
        vmin=df.min().min() if 'AUC' in title or 'F1' in title else None,
        vmax=df.max().max() if 'AUC' in title or 'F1' in title else None
    )
    
    ax.set_title(f'{title}\n訓練ラベル × 評価期間', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('評価期間', fontsize=12, fontweight='bold')
    ax.set_ylabel('訓練ラベル', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()


def create_combined_heatmap(base_dir: Path):
    """複数メトリクスの組み合わせヒートマップを作成（2行3列レイアウト）"""
    
    metrics = ['AUC_ROC', 'AUC_PR', 'F1', 'Precision', 'Recall']
    metric_names = ['AUC-ROC', 'AUC-PR', 'F1-Score', 'Precision', 'Recall']
    csv_files = [base_dir / f'matrix_{metric}.csv' for metric in metrics]
    
    # 2行3列のレイアウト
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for i, (csv_file, metric, name) in enumerate(zip(csv_files, metrics, metric_names)):
        if not csv_file.exists():
            print(f"⚠️ ファイルが見つかりません: {csv_file}")
            continue
            
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # データ読み込み
        df = pd.read_csv(csv_file, index_col=0)
        
        # パーセント表記の場合は数値に変換
        if df.applymap(lambda x: isinstance(x, str) and '%' in str(x)).any().any():
            df = df.applymap(lambda x: float(str(x).replace('%', '')) / 100 if isinstance(x, str) and '%' in str(x) else x)
            fmt = '.1%'
        else:
            fmt = '.3f'
        
        # ヒートマップ作成（0-1.0の範囲で統一）
        sns.heatmap(
            df.astype(float),
            annot=True,
            fmt=fmt,
            cmap='Blues',
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': name}
        )
        
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('訓練期間', fontsize=11)
        ax.set_ylabel('評価期間', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    
    # 最後のサブプロット（右下）を非表示
    axes[1, 2].set_visible(False)
    
    plt.suptitle('スライディングウィンドウ クロス評価結果', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = base_dir / 'heatmap_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 組み合わせヒートマップ保存: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_cross_eval_heatmap.py <output_base_dir>")
        sys.exit(1)
    
    base_dir = Path(sys.argv[1])
    
    if not base_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {base_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("ヒートマップ可視化")
    print("=" * 80)
    print()
    
    # 可視化するメトリクス
    metrics = [
        ("AUC_ROC", "AUC-ROC", '.3f'),
        ("AUC_PR", "AUC-PR", '.3f'),
        ("F1", "F1スコア", '.3f'),
        ("Precision", "Precision", '.3f'),
        ("Recall", "Recall", '.3f'),
        ("継続rate", "継続率", '.1%'),
    ]
    
    for metric_file, metric_title, fmt in metrics:
        csv_path = base_dir / f"matrix_{metric_file}.csv"
        
        if not csv_path.exists():
            print(f"⚠️  スキップ: {csv_path} が存在しません")
            continue
        
        output_path = base_dir / f"heatmap_{metric_file}.png"
        create_heatmap(csv_path, metric_title, output_path, fmt)
    
    # 組み合わせヒートマップを作成
    print()
    print("組み合わせヒートマップを作成中...")
    create_combined_heatmap(base_dir)
    
    print()
    print("=" * 80)
    print("ヒートマップ生成完了！")
    print(f"出力先: {base_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

