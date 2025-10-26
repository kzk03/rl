#!/usr/bin/env python3
"""
クロス評価結果のヒートマップを生成
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_matrix(csv_path: Path) -> pd.DataFrame:
    """CSVからマトリクスを読み込み"""
    df = pd.read_csv(csv_path, index_col=0)
    return df


def create_heatmap(
    matrix: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    figsize: tuple = (10, 8)
):
    """
    単一のヒートマップを生成
    
    Args:
        matrix: データマトリクス
        metric: メトリクス名（AUC_ROC, F1など）
        title: グラフタイトル
        output_path: 出力パス
        figsize: 図のサイズ
    """
    # メトリクスに応じた設定
    if metric in ['AUC_ROC', 'AUC_PR']:
        vmin, vmax = 0, 1
        cmap = 'RdYlGn'
        fmt = '.3f'
        metric_label = metric.replace('_', '-')
    elif metric == 'F1':
        vmin, vmax = 0, 1
        cmap = 'RdYlGn'
        fmt = '.3f'
        metric_label = 'F1 Score'
    elif metric in ['Precision', 'Recall']:
        vmin, vmax = 0, 1
        cmap = 'RdYlGn'
        fmt = '.3f'
        metric_label = metric
    elif metric == '継続rate':
        vmin, vmax = 0, 100
        cmap = 'YlOrRd'
        fmt = '.1f'
        metric_label = 'Continuation Rate (%)'
    else:
        vmin, vmax = None, None
        cmap = 'RdYlGn'
        fmt = '.3f'
        metric_label = metric
    
    # 行列を転置（訓練ラベルをx軸、評価期間をy軸に）
    matrix = matrix.T
    
    # 図を作成
    fig, ax = plt.subplots(figsize=figsize)
    
    # ヒートマップを描画
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': metric_label},
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Training Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('Evaluation Period', fontsize=13, fontweight='bold')
    
    # y軸を反転（原点を左下に）
    ax.invert_yaxis()
    
    # 軸ラベルを少し大きく
    ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ ヒートマップ保存: {output_path}")
    plt.close()


def create_all_heatmaps(
    result_dir: Path,
    output_dir: Path,
    title_prefix: str = ''
):
    """
    全メトリクスのヒートマップを生成
    
    Args:
        result_dir: 結果ディレクトリ（matrix_*.csvがある場所）
        output_dir: 出力ディレクトリ
        title_prefix: タイトルのプレフィックス（例: "Normal IRL - "）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 利用可能なメトリクス
    metrics = ['AUC_ROC', 'AUC_PR', 'F1', 'Precision', 'Recall', '継続rate']
    
    for metric in metrics:
        matrix_path = result_dir / f'matrix_{metric}.csv'
        
        if not matrix_path.exists():
            print(f"⚠ {metric} のマトリクスが見つかりません: {matrix_path}")
            continue
        
        # マトリクスを読み込み
        matrix = load_matrix(matrix_path)
        
        # 継続率の場合は%に変換
        if metric == '継続rate':
            # 文字列の場合は数値に変換
            if matrix.dtypes.iloc[0] == 'object':
                # '%'を削除して数値に変換
                matrix = matrix.apply(lambda x: x.str.rstrip('%').astype(float) if x.dtype == 'object' else x)
            elif matrix.values.max() <= 1.0:
                matrix = matrix * 100
        
        # タイトルを作成
        metric_label = metric.replace('_', '-')
        title = f"{title_prefix}{metric_label}"
        
        # 出力パス
        output_path = output_dir / f'heatmap_{metric}.png'
        
        # ヒートマップを生成
        create_heatmap(matrix, metric, title, output_path)


def create_combined_view(
    result_dir: Path,
    output_dir: Path,
    title_prefix: str = ''
):
    """
    主要メトリクス（AUC-ROC, F1）を1つの図にまとめて表示
    
    Args:
        result_dir: 結果ディレクトリ
        output_dir: 出力ディレクトリ
        title_prefix: タイトルのプレフィックス
    """
    # AUC-ROCとF1を読み込み
    auc_path = result_dir / 'matrix_AUC_ROC.csv'
    f1_path = result_dir / 'matrix_F1.csv'
    
    if not auc_path.exists() or not f1_path.exists():
        print("⚠ AUC-ROCまたはF1のマトリクスが見つかりません")
        return
    
    auc_matrix = load_matrix(auc_path)
    f1_matrix = load_matrix(f1_path)
    
    # 行列を転置（訓練ラベルをx軸、評価期間をy軸に）
    auc_matrix = auc_matrix.T
    f1_matrix = f1_matrix.T
    
    # 図を作成（横並び）
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # AUC-ROC
    sns.heatmap(
        auc_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'AUC-ROC'},
        ax=axes[0],
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    axes[0].set_title(f'{title_prefix}AUC-ROC', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Training Label', fontsize=12)
    axes[0].set_ylabel('Evaluation Period', fontsize=12)
    axes[0].invert_yaxis()  # y軸を反転（原点を左下に）
    
    # F1
    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'F1 Score'},
        ax=axes[1],
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    axes[1].set_title(f'{title_prefix}F1 Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Training Label', fontsize=12)
    axes[1].set_ylabel('Evaluation Period', fontsize=12)
    axes[1].invert_yaxis()  # y軸を反転（原点を左下に）
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'heatmap_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 統合ヒートマップ保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='クロス評価結果のヒートマップを生成'
    )
    parser.add_argument(
        'result_dir',
        type=str,
        help='結果ディレクトリ（matrix_*.csvがある場所）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='出力ディレクトリ（デフォルト: <result_dir>/heatmaps）'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='',
        help='タイトルのプレフィックス（例: "Normal IRL - "）'
    )
    parser.add_argument(
        '--combined-only',
        action='store_true',
        help='統合ヒートマップ（AUC-ROC + F1）のみ生成'
    )
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    if not result_dir.exists():
        print(f"❌ 結果ディレクトリが見つかりません: {result_dir}")
        return
    
    # 出力ディレクトリを設定
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = result_dir / 'heatmaps'
    
    print("=" * 80)
    print("クロス評価結果ヒートマップ生成")
    print("=" * 80)
    print(f"結果ディレクトリ: {result_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"タイトル: {args.title}")
    print("=" * 80)
    
    if args.combined_only:
        # 統合ヒートマップのみ
        print("\n統合ヒートマップを生成中...")
        create_combined_view(result_dir, output_dir, args.title)
    else:
        # 全メトリクスのヒートマップを生成
        print("\n全メトリクスのヒートマップを生成中...")
        create_all_heatmaps(result_dir, output_dir, args.title)
        
        # 統合ヒートマップも生成
        print("\n統合ヒートマップを生成中...")
        create_combined_view(result_dir, output_dir, args.title)
    
    print("\n" + "=" * 80)
    print("✅ ヒートマップ生成完了！")
    print("=" * 80)
    print(f"出力先: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
