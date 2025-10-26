#!/usr/bin/env python3
"""
通常IRLと拡張IRLのクロス評価結果を比較し、ヒートマップを生成
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_matrix(csv_path: Path) -> pd.DataFrame:
    """CSVからマトリクスを読み込み"""
    df = pd.read_csv(csv_path, index_col=0)
    return df

def create_comparison_heatmaps(
    normal_dir: Path,
    enhanced_dir: Path,
    output_dir: Path,
    metric: str = 'AUC_ROC'
):
    """
    通常IRLと拡張IRLの比較ヒートマップを生成
    
    Args:
        normal_dir: 通常IRLの結果ディレクトリ
        enhanced_dir: 拡張IRLの結果ディレクトリ
        output_dir: 出力ディレクトリ
        metric: 比較するメトリクス（AUC_ROC, F1, Precision, Recall, AUC_PR）
    """
    # マトリクスを読み込み
    normal_matrix = load_matrix(normal_dir / f'matrix_{metric}.csv')
    enhanced_matrix = load_matrix(enhanced_dir / f'matrix_{metric}.csv')
    
    # 行列を転置（訓練ラベルをx軸、評価期間をy軸に）
    normal_matrix = normal_matrix.T
    enhanced_matrix = enhanced_matrix.T
    
    # 差分を計算（拡張 - 通常）
    diff_matrix = enhanced_matrix - normal_matrix
    
    # 図を作成（3つ並べる）
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # カラーマップとスケール設定
    if metric == 'AUC_ROC':
        vmin, vmax = 0, 1
        cmap = 'RdYlGn'
        fmt = '.3f'
        metric_label = 'AUC-ROC'
    elif metric == 'F1':
        vmin, vmax = 0, 1
        cmap = 'RdYlGn'
        fmt = '.3f'
        metric_label = 'F1 Score'
    else:
        vmin, vmax = 0, 1
        cmap = 'RdYlGn'
        fmt = '.3f'
        metric_label = metric.replace('_', '-')
    
    # 1. 通常IRL
    sns.heatmap(
        normal_matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': metric_label},
        ax=axes[0],
        square=True
    )
    axes[0].set_title(f'Normal IRL - {metric_label}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Training Label', fontsize=12)
    axes[0].set_ylabel('Evaluation Period', fontsize=12)
    axes[0].invert_yaxis()  # y軸を反転（原点を左下に）
    
    # 2. 拡張IRL
    sns.heatmap(
        enhanced_matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': metric_label},
        ax=axes[1],
        square=True
    )
    axes[1].set_title(f'Enhanced IRL - {metric_label}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Training Label', fontsize=12)
    axes[1].set_ylabel('Evaluation Period', fontsize=12)
    axes[1].invert_yaxis()  # y軸を反転（原点を左下に）
    
    # 3. 差分（拡張 - 通常）
    # 差分用のカラーマップ（赤=悪化、緑=改善）
    sns.heatmap(
        diff_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': f'Difference ({metric_label})'},
        ax=axes[2],
        square=True
    )
    axes[2].set_title(f'Difference (Enhanced - Normal)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Training Label', fontsize=12)
    axes[2].set_ylabel('Evaluation Period', fontsize=12)
    axes[2].invert_yaxis()  # y軸を反転（原点を左下に）
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / f'comparison_heatmap_{metric}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ ヒートマップ保存: {output_path}")
    plt.close()

def create_summary_comparison(
    normal_dir: Path,
    enhanced_dir: Path,
    output_dir: Path
):
    """
    通常IRLと拡張IRLの統計的比較を生成
    """
    metrics = ['AUC_ROC', 'F1', 'Precision', 'Recall', 'AUC_PR']
    
    comparison_data = []
    
    for metric in metrics:
        try:
            normal_matrix = load_matrix(normal_dir / f'matrix_{metric}.csv')
            enhanced_matrix = load_matrix(enhanced_dir / f'matrix_{metric}.csv')
            diff_matrix = enhanced_matrix - normal_matrix
            
            # 統計計算
            comparison_data.append({
                'Metric': metric.replace('_', '-'),
                'Normal_Mean': normal_matrix.values.mean(),
                'Normal_Std': normal_matrix.values.std(),
                'Enhanced_Mean': enhanced_matrix.values.mean(),
                'Enhanced_Std': enhanced_matrix.values.std(),
                'Diff_Mean': diff_matrix.values.mean(),
                'Diff_Std': diff_matrix.values.std(),
                'Better_Count': (diff_matrix.values > 0).sum(),
                'Worse_Count': (diff_matrix.values < 0).sum(),
                'Same_Count': (diff_matrix.values == 0).sum()
            })
        except FileNotFoundError:
            print(f"⚠ {metric} のマトリクスが見つかりません")
            continue
    
    # DataFrameに変換
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存
    csv_path = output_dir / 'comparison_summary.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ 比較サマリー保存: {csv_path}")
    
    # コンソールに表示
    print("\n" + "=" * 100)
    print("統計的比較サマリー")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100)
    
    return comparison_df

def create_per_model_comparison(
    normal_dir: Path,
    enhanced_dir: Path,
    output_dir: Path
):
    """
    各訓練モデル別の平均性能比較
    """
    normal_auc = load_matrix(normal_dir / 'matrix_AUC_ROC.csv')
    enhanced_auc = load_matrix(enhanced_dir / 'matrix_AUC_ROC.csv')
    normal_f1 = load_matrix(normal_dir / 'matrix_F1.csv')
    enhanced_f1 = load_matrix(enhanced_dir / 'matrix_F1.csv')
    
    # 各モデルの平均を計算
    model_comparison = pd.DataFrame({
        'Training_Label': normal_auc.index,
        'Normal_AUC_Mean': normal_auc.mean(axis=1).values,
        'Enhanced_AUC_Mean': enhanced_auc.mean(axis=1).values,
        'AUC_Diff': (enhanced_auc.mean(axis=1) - normal_auc.mean(axis=1)).values,
        'Normal_F1_Mean': normal_f1.mean(axis=1).values,
        'Enhanced_F1_Mean': enhanced_f1.mean(axis=1).values,
        'F1_Diff': (enhanced_f1.mean(axis=1) - normal_f1.mean(axis=1)).values
    })
    
    # 保存
    csv_path = output_dir / 'per_model_comparison.csv'
    model_comparison.to_csv(csv_path, index=False)
    print(f"✓ モデル別比較保存: {csv_path}")
    
    # コンソールに表示
    print("\n" + "=" * 100)
    print("訓練モデル別平均性能比較")
    print("=" * 100)
    print(model_comparison.to_string(index=False))
    print("=" * 100)
    
    # 棒グラフ作成
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(model_comparison))
    width = 0.35
    
    # AUC-ROC
    axes[0].bar(x - width/2, model_comparison['Normal_AUC_Mean'], width, 
                label='Normal IRL', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, model_comparison['Enhanced_AUC_Mean'], width,
                label='Enhanced IRL', alpha=0.8, color='coral')
    axes[0].set_xlabel('Training Label', fontsize=12)
    axes[0].set_ylabel('Mean AUC-ROC', fontsize=12)
    axes[0].set_title('Average AUC-ROC by Training Label', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_comparison['Training_Label'])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # F1
    axes[1].bar(x - width/2, model_comparison['Normal_F1_Mean'], width,
                label='Normal IRL', alpha=0.8, color='steelblue')
    axes[1].bar(x + width/2, model_comparison['Enhanced_F1_Mean'], width,
                label='Enhanced IRL', alpha=0.8, color='coral')
    axes[1].set_xlabel('Training Label', fontsize=12)
    axes[1].set_ylabel('Mean F1 Score', fontsize=12)
    axes[1].set_title('Average F1 Score by Training Label', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_comparison['Training_Label'])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    bar_path = output_dir / 'per_model_comparison_bars.png'
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"✓ モデル別比較グラフ保存: {bar_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='通常IRLと拡張IRLのクロス評価結果を比較'
    )
    parser.add_argument(
        '--normal',
        type=str,
        default='outputs/full_cross_eval',
        help='通常IRLの結果ディレクトリ'
    )
    parser.add_argument(
        '--enhanced',
        type=str,
        default='outputs/enhanced_cross_eval',
        help='拡張IRLの結果ディレクトリ'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/irl_comparison',
        help='出力ディレクトリ'
    )
    
    args = parser.parse_args()
    
    normal_dir = Path(args.normal)
    enhanced_dir = Path(args.enhanced)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("通常IRL vs 拡張IRL 比較分析")
    print("=" * 100)
    print(f"通常IRL: {normal_dir}")
    print(f"拡張IRL: {enhanced_dir}")
    print(f"出力先: {output_dir}")
    print("=" * 100)
    
    # 各メトリクスのヒートマップを生成
    metrics = ['AUC_ROC', 'F1', 'Precision', 'Recall', 'AUC_PR']
    
    for metric in metrics:
        try:
            print(f"\n{metric} のヒートマップを生成中...")
            create_comparison_heatmaps(normal_dir, enhanced_dir, output_dir, metric)
        except FileNotFoundError as e:
            print(f"⚠ {metric} のマトリクスが見つかりません: {e}")
            continue
    
    # 統計的比較
    print("\n統計的比較を生成中...")
    create_summary_comparison(normal_dir, enhanced_dir, output_dir)
    
    # モデル別比較
    print("\nモデル別比較を生成中...")
    create_per_model_comparison(normal_dir, enhanced_dir, output_dir)
    
    print("\n" + "=" * 100)
    print("✅ 比較分析完了！")
    print("=" * 100)
    print(f"結果: {output_dir}")
    print("=" * 100)

if __name__ == '__main__':
    main()

