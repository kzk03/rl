#!/usr/bin/env python3
"""
レビュー承諾予測のクロス評価結果をヒートマップで可視化
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 日本語フォント設定
import matplotlib

matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_metrics_from_directory(base_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    ディレクトリからメトリクスを読み込む
    
    Args:
        base_dir: ベースディレクトリ
        
    Returns:
        {(train_period, eval_period): metrics} の辞書
    """
    metrics_dict = {}
    
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    for train_period in train_periods:
        for eval_period in eval_periods:
            metrics_path = base_dir / f'train_{train_period}' / f'eval_{eval_period}' / 'metrics.json'
            
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    metrics_dict[(train_period, eval_period)] = metrics
                    logger.info(f"読み込み: train_{train_period} -> eval_{eval_period}")
            else:
                logger.warning(f"メトリクスが見つかりません: {metrics_path}")
    
    return metrics_dict


def create_heatmap(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_name: str,
    output_path: Path,
    title: str = None
):
    """
    ヒートマップを作成
    
    Args:
        metrics_dict: メトリクス辞書
        metric_name: メトリクス名
        output_path: 出力パス
        title: タイトル
    """
    periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    # データフレームを作成
    data = []
    for train_period in periods:
        for eval_period in periods:
            key = (train_period, eval_period)
            if key in metrics_dict:
                value = metrics_dict[key].get(metric_name, np.nan)
                data.append({
                    'train_period': train_period,
                    'eval_period': eval_period,
                    'value': value
                })
    
    if not data:
        logger.warning(f"{metric_name} のデータがありません")
        return
    
    df = pd.DataFrame(data)
    pivot_table = df.pivot(index='eval_period', columns='train_period', values='value')
    
    # 軸の順序を設定
    pivot_table = pivot_table.reindex(periods, columns=periods)
    
    # ヒートマップを作成
    plt.figure(figsize=(10, 8))
    
    # 色スケールを0-1.0に固定
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': metric_name}
    )
    
    # y軸の順序を明示的に設定（下から上: 0-3m, 3-6m, 6-9m, 9-12m）
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(['0-3m', '3-6m', '6-9m', '9-12m'])
    
    if title is None:
        title = f'{metric_name} - レビュー承諾予測'
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('訓練期間', fontsize=12)
    plt.ylabel('評価期間', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ヒートマップを保存: {output_path}")


def create_combined_heatmap(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: Path
):
    """
    全メトリクスを1つのPNGファイルに統合
    
    Args:
        metrics_dict: メトリクス辞書
        output_path: 出力パス
    """
    periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    # 5つのメトリクスを表示（継続率を除く）
    metrics = ['auc_roc', 'auc_pr', 'precision', 'recall', 'f1_score']
    metric_titles = ['AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'F1 Score']
    
    # 2行3列のレイアウト（上段3つ、下段2つ）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric_name, metric_title) in enumerate(zip(metrics, metric_titles)):
        # データフレームを作成
        data = []
        for train_period in periods:
            for eval_period in periods:
                key = (train_period, eval_period)
                if key in metrics_dict:
                    value = metrics_dict[key].get(metric_name, np.nan)
                    data.append({
                        'train_period': train_period,
                        'eval_period': eval_period,
                        'value': value
                    })
        
        if not data:
            logger.warning(f"{metric_name} のデータがありません")
            continue
        
        df = pd.DataFrame(data)
        pivot_table = df.pivot(index='eval_period', columns='train_period', values='value')
        
        # 軸の順序を設定
        pivot_table = pivot_table.reindex(periods, columns=periods)
        
        # ヒートマップを作成
        ax = axes[idx]
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cbar_kws={'label': metric_title}
        )
        
        # y軸の順序を明示的に設定
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(['0-3m', '3-6m', '6-9m', '9-12m'])
        
        ax.set_title(metric_title, fontsize=12, pad=10)
        ax.set_xlabel('訓練期間', fontsize=10)
        ax.set_ylabel('評価期間', fontsize=10)
    
    # 最後のサブプロットを非表示
    axes[5].axis('off')
    
    plt.suptitle('レビュー承諾予測 - クロス評価結果', fontsize=16, y=0.995)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"統合ヒートマップを保存: {output_path}")


def main():
    base_dir = Path("outputs/review_acceptance_cross_eval_nova")
    
    if not base_dir.exists():
        logger.error(f"ベースディレクトリが存在しません: {base_dir}")
        return
    
    # メトリクスを読み込み
    logger.info("メトリクスを読み込み中...")
    metrics_dict = load_metrics_from_directory(base_dir)
    
    if not metrics_dict:
        logger.error("メトリクスが読み込めませんでした")
        return
    
    logger.info(f"読み込んだメトリクス数: {len(metrics_dict)}")
    
    # 個別のヒートマップを作成
    logger.info("個別ヒートマップを作成中...")
    
    metrics_to_plot = {
        'auc_roc': 'AUC-ROC',
        'auc_pr': 'AUC-PR',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score'
    }
    
    for metric_name, metric_title in metrics_to_plot.items():
        output_path = base_dir / f"heatmap_{metric_name}.png"
        create_heatmap(
            metrics_dict,
            metric_name,
            output_path,
            title=f'{metric_title} - レビュー承諾予測'
        )
    
    # 統合ヒートマップを作成
    logger.info("統合ヒートマップを作成中...")
    combined_output_path = base_dir / "heatmap_combined.png"
    create_combined_heatmap(metrics_dict, combined_output_path)
    
    # メトリクスCSVを作成
    logger.info("メトリクスCSVを作成中...")
    for metric_name in metrics_to_plot.keys():
        data = []
        periods = ['0-3m', '3-6m', '6-9m', '9-12m']
        
        for train_period in periods:
            row = {'train_period': train_period}
            for eval_period in periods:
                key = (train_period, eval_period)
                if key in metrics_dict:
                    value = metrics_dict[key].get(metric_name, np.nan)
                    row[f'eval_{eval_period}'] = value
                else:
                    row[f'eval_{eval_period}'] = np.nan
            data.append(row)
        
        df = pd.DataFrame(data)
        csv_path = base_dir / f"matrix_{metric_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSVを保存: {csv_path}")
    
    logger.info("=" * 80)
    logger.info("全てのヒートマップとCSVが作成されました！")
    logger.info(f"出力ディレクトリ: {base_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

