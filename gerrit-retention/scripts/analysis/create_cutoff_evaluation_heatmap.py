#!/usr/bin/env python3
"""
Cutoff時点での評価結果のヒートマップ作成スクリプト
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_metrics_from_directory(base_dir: Path) -> pd.DataFrame:
    """ディレクトリからメトリクスを読み込み（完全な4x4クロス評価対応）"""
    results = []
    
    # 訓練期間
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    for train_period in train_periods:
        for eval_period in eval_periods:
            # メトリクスファイルのパスを決定
            metrics_file = base_dir / f'train_{train_period}' / f'eval_{eval_period}' / 'metrics.json'
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    results.append({
                        'train_period': train_period,
                        'eval_period': eval_period,
                        'auc_roc': metrics.get('auc_roc', 0.0),
                        'auc_pr': metrics.get('auc_pr', 0.0),
                        'f1': metrics.get('f1_score', 0.0),
                        'precision': metrics.get('precision', 0.0),
                        'recall': metrics.get('recall', 0.0),
                        'continuation_rate': metrics.get('continuation_rate', 0.0),
                        'sample_count': metrics.get('sample_count', 0)
                    })
                    logger.info(f"読み込み: {train_period} -> {eval_period}")
                    
                except Exception as e:
                    logger.warning(f"読み込みエラー {metrics_file}: {e}")
            else:
                logger.warning(f"ファイルが見つかりません: {metrics_file}")
    
    return pd.DataFrame(results)

def create_heatmap(df: pd.DataFrame, metric: str, title: str, output_path: Path):
    """ヒートマップを作成（縦軸：評価期間、横軸：訓練期間、左下から0-3m）"""
    # ピボットテーブルを作成（縦軸：評価期間、横軸：訓練期間）
    pivot_table = df.pivot(index='eval_period', columns='train_period', values=metric)
    
    # 期間をソート（評価期間は下から0-3m、訓練期間は左から0-3m）
    periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods_reversed = ['9-12m', '6-9m', '3-6m', '0-3m']  # 下から0-3mになるように逆順
    
    # ピボットテーブルを再構築して確実に順序を制御
    pivot_table = pivot_table.reindex(eval_periods_reversed, columns=periods)
    
    # デバッグ用：ピボットテーブルのインデックスを確認
    print(f"Pivot table index: {pivot_table.index.tolist()}")
    print(f"Pivot table columns: {pivot_table.columns.tolist()}")
    
    # ヒートマップを作成
    plt.figure(figsize=(10, 8))
    
    # カラーマップを選択（Blues、固定スケール0-1.0）
    if metric in ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']:
        cmap = 'Blues'  # 青のグラデーション
        vmin, vmax = 0.0, 1.0
    else:
        cmap = 'Blues'
        vmin, vmax = 0.0, 1.0
    
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': metric},
        square=True,
        linewidths=0.5,
        yticklabels=True,  # Y軸ラベルを明示的に表示
        xticklabels=True   # X軸ラベルを明示的に表示
    )
    
    # Y軸の順序を明示的に設定（下から0-3mになるように）
    ax = plt.gca()
    ax.set_yticks(range(len(eval_periods_reversed)))
    ax.set_yticklabels(eval_periods_reversed)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods)
    
    plt.title(f'{title}\n評価期間 vs 訓練期間', fontsize=16, fontweight='bold')
    plt.xlabel('訓練期間', fontsize=12)
    plt.ylabel('評価期間', fontsize=12)
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ヒートマップ保存: {output_path}")

def create_combined_heatmap(df: pd.DataFrame, output_dir: Path):
    """複数メトリクスの組み合わせヒートマップを作成（2行3列レイアウト）"""
    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
    metric_names = ['AUC-ROC', 'AUC-PR', 'F1-Score', 'Precision', 'Recall']
    
    # 2行3列のレイアウト
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # ピボットテーブルを作成（縦軸：評価期間、横軸：訓練期間）
        pivot_table = df.pivot(index='eval_period', columns='train_period', values=metric)
        
        # 期間をソート（評価期間は下から0-3m、訓練期間は左から0-3m）
        periods = ['0-3m', '3-6m', '6-9m', '9-12m']
        eval_periods_reversed = ['9-12m', '6-9m', '3-6m', '0-3m']  # 下から0-3mになるように逆順
        pivot_table = pivot_table.reindex(eval_periods_reversed, columns=periods)
        
        # ヒートマップを作成（0-1.0の範囲で統一）
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            linewidths=0.5,
            cbar_kws={'label': name},
            yticklabels=True,  # Y軸ラベルを明示的に表示
            xticklabels=True   # X軸ラベルを明示的に表示
        )
        
        # Y軸の順序を明示的に設定（下から0-3mになるように）
        ax.set_yticks(range(len(eval_periods_reversed)))
        ax.set_yticklabels(eval_periods_reversed)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods)
        
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('訓練期間', fontsize=11)
        ax.set_ylabel('評価期間', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    
    # 最後のサブプロット（右下）を非表示
    axes[1, 2].set_visible(False)
    
    plt.suptitle('拡張IRLモデル クロス評価結果（Cutoff評価）', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'heatmap_combined_cutoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"組み合わせヒートマップ保存: {output_path}")

def create_csv_summary(df: pd.DataFrame, output_dir: Path):
    """CSVサマリーを作成（縦軸：評価期間、横軸：訓練期間）"""
    periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
    
    for metric in metrics:
        pivot_table = df.pivot(index='eval_period', columns='train_period', values=metric)
        pivot_table = pivot_table.reindex(periods[::-1], columns=periods)  # 評価期間を逆順に
        
        output_path = output_dir / f'matrix_{metric}_cutoff.csv'
        pivot_table.to_csv(output_path)
        logger.info(f"CSV保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Cutoff時点での評価結果のヒートマップ作成')
    parser.add_argument('--input-dir', type=str, required=True, help='入力ディレクトリ')
    parser.add_argument('--output-dir', type=str, required=True, help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"入力ディレクトリ: {input_dir}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    
    # メトリクスを読み込み
    df = load_metrics_from_directory(input_dir)
    
    if df.empty:
        logger.error("メトリクスデータが見つかりません")
        return
    
    logger.info(f"読み込み完了: {len(df)}件")
    
    # 各メトリクスのヒートマップを作成
    metrics_config = [
        ('auc_roc', 'AUC-ROC'),
        ('auc_pr', 'AUC-PR'),
        ('f1', 'F1-Score'),
        ('precision', 'Precision'),
        ('recall', 'Recall')
    ]
    
    for metric, title in metrics_config:
        output_path = output_dir / f'heatmap_{title}_cutoff.png'
        create_heatmap(df, metric, title, output_path)
    
    # 組み合わせヒートマップを作成
    create_combined_heatmap(df, output_dir)
    
    # CSVサマリーを作成
    create_csv_summary(df, output_dir)
    
    logger.info("完了！")

if __name__ == '__main__':
    main()
