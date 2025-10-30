#!/usr/bin/env python3
"""
通常IRLモデルのクロス評価結果のヒートマップを作成
縦軸: 評価期間 (0-3m, 3-6m, 6-9m, 9-12m)
横軸: 訓練期間 (0-3m, 3-6m, 6-9m, 9-12m)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def load_cross_evaluation_results(base_dir: str) -> pd.DataFrame:
    """クロス評価結果を読み込み、DataFrameに変換"""
    
    metrics_data = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.warning(f"ディレクトリが存在しません: {base_dir}")
        return pd.DataFrame()
    
    # 各訓練期間ディレクトリを探索
    for train_dir in base_path.iterdir():
        if not train_dir.is_dir() or train_dir.name.startswith('.'):
            continue
            
        train_period = train_dir.name.replace('train_', '')
        logger.info(f"訓練期間ディレクトリを処理中: {train_period}")
        
        # 各評価期間ディレクトリを探索
        for eval_dir in train_dir.iterdir():
            if not eval_dir.is_dir() or eval_dir.name.startswith('.'):
                continue
                
            eval_period = eval_dir.name.replace('eval_', '')
            metrics_file = eval_dir / "metrics.json"
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    metrics_data.append({
                        'train_period': train_period,
                        'eval_period': eval_period,
                        'auc_roc': metrics.get('auc_roc', 0.0),
                        'auc_pr': metrics.get('auc_pr', 0.0),
                        'f1': metrics.get('f1', 0.0),
                        'precision': metrics.get('precision', 0.0),
                        'recall': metrics.get('recall', 0.0),
                        'continuation_rate': metrics.get('continuation_rate', 0.0),
                        'sample_count': metrics.get('sample_count', 0)
                    })
                    logger.info(f"メトリクス読み込み完了: {train_period}/{eval_period}")
                    
                except Exception as e:
                    logger.warning(f"メトリクス読み込みエラー {train_period}/{eval_period}: {e}")
            else:
                logger.warning(f"メトリクスファイルが見つかりません: {metrics_file}")
    
    if not metrics_data:
        logger.warning("メトリクスデータが見つかりませんでした")
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics_data)
    logger.info(f"メトリクス読み込み完了: {len(df)}件")
    return df

def create_heatmap(data: pd.DataFrame, metric: str, title: str, output_path: str):
    """ヒートマップを作成"""
    
    # データをピボット
    pivot_table = data.pivot_table(
        index='eval_period', 
        columns='train_period', 
        values=metric, 
        fill_value=0.0
    )
    
    # 期間の順序を定義（縦軸: 評価期間、横軸: 訓練期間）
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    # インデックスとカラムを再順序付け
    pivot_table = pivot_table.reindex(eval_periods[::-1], columns=train_periods)
    
    # ヒートマップ作成
    plt.figure(figsize=(10, 8))
    
    # カラーマップの範囲を設定
    vmin = pivot_table.min().min()
    vmax = pivot_table.max().max()
    
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': metric},
        linewidths=0.5
    )
    
    plt.title(f'{title} - 通常IRLクロス評価', fontsize=16, fontweight='bold')
    plt.xlabel('訓練期間', fontsize=12)
    plt.ylabel('評価期間', fontsize=12)
    
    # 軸ラベルの調整
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ヒートマップ保存完了: {output_path}")

def create_combined_heatmap(data: pd.DataFrame, output_dir: str):
    """複数メトリクスの結合ヒートマップを作成（2行3列レイアウト）"""
    
    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
    metric_names = ['AUC-ROC', 'AUC-PR', 'F1-Score', 'Precision', 'Recall']
    
    # 2行3列のレイアウト
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
            
        # データをピボット
        pivot_table = data.pivot_table(
            index='eval_period', 
            columns='train_period', 
            values=metric, 
            fill_value=0.0
        )
        
        # 期間の順序を定義
        eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
        train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
        
        # インデックスとカラムを再順序付け
        pivot_table = pivot_table.reindex(eval_periods, columns=train_periods)
        
        # ヒートマップ作成（0-1.0の範囲で統一）
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cbar_kws={'label': name},
            linewidths=0.5
        )
        
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('訓練期間', fontsize=11)
        ax.set_ylabel('評価期間', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    
    # 最後のサブプロット（右下）を非表示
    axes[1, 2].set_visible(False)
    
    plt.suptitle('通常IRLモデル クロス評価結果', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'heatmap_combined_normal_irl_cross.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"結合ヒートマップ保存完了: {output_path}")

def save_matrices_as_csv(data: pd.DataFrame, output_dir: str):
    """各メトリクスのマトリックスをCSVとして保存"""
    
    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
    
    for metric in metrics:
        pivot_table = data.pivot_table(
            index='eval_period', 
            columns='train_period', 
            values=metric, 
            fill_value=0.0
        )
        
        # 期間の順序を定義
        eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
        train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
        
        # インデックスとカラムを再順序付け
        pivot_table = pivot_table.reindex(eval_periods, columns=train_periods)
        
        output_path = os.path.join(output_dir, f'matrix_{metric}_normal_irl_cross.csv')
        pivot_table.to_csv(output_path)
        logger.info(f"マトリックス保存完了: {output_path}")

def main():
    """メイン処理"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    
    # 入力ディレクトリ
    input_dir = "outputs/normal_irl_cross_eval"
    
    # 出力ディレクトリ
    output_dir = "outputs/normal_irl_cross_eval/heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("通常IRLモデル クロス評価ヒートマップ作成開始")
    logger.info(f"入力ディレクトリ: {input_dir}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    logger.info("=" * 80)
    
    # メトリクスデータを読み込み
    data = load_cross_evaluation_results(input_dir)
    
    if data.empty:
        logger.error("メトリクスデータが見つかりませんでした")
        return
    
    logger.info(f"読み込み完了: {len(data)}件のメトリクス")
    logger.info(f"訓練期間: {sorted(data['train_period'].unique())}")
    logger.info(f"評価期間: {sorted(data['eval_period'].unique())}")
    
    # 各メトリクスのヒートマップを作成
    metrics_config = [
        ('auc_roc', 'AUC-ROC'),
        ('auc_pr', 'AUC-PR'),
        ('f1', 'F1-Score'),
        ('precision', 'Precision'),
        ('recall', 'Recall')
    ]
    
    for metric, name in metrics_config:
        output_path = os.path.join(output_dir, f'heatmap_{name}_normal_irl_cross.png')
        create_heatmap(data, metric, name, output_path)
    
    # 結合ヒートマップを作成
    create_combined_heatmap(data, output_dir)
    
    # マトリックスをCSVとして保存
    save_matrices_as_csv(data, output_dir)
    
    # 生データも保存
    raw_data_path = os.path.join(output_dir, 'raw_metrics_normal_irl_cross.csv')
    data.to_csv(raw_data_path, index=False)
    logger.info(f"生データ保存完了: {raw_data_path}")
    
    logger.info("=" * 80)
    logger.info("通常IRLモデル クロス評価ヒートマップ作成完了")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
