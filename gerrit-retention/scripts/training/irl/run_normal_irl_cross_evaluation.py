#!/usr/bin/env python3
"""
通常IRLモデルのクロス評価を実行
各訓練期間のモデルを異なる評価期間で評価
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 相対インポートを使用
from evaluate_normal_irl_snapshot import (
    evaluate_normal_irl_with_snapshot_features,
    extract_snapshot_evaluation_trajectories,
    load_review_logs,
)


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_cross_evaluation(
    reviews_file: str,
    base_model_dir: str,
    output_dir: str,
    cutoff_date: str = "2023-01-01",
    history_window: int = 12,
    min_history_events: int = 3,
    project: str = "openstack/nova"
):
    """クロス評価を実行"""
    
    logger = setup_logging()
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # データ読み込み
    logger.info(f"レビューログを読み込み中: {reviews_file}")
    df = load_review_logs(reviews_file)
    
    # カットオフ日を変換
    cutoff_date = pd.Timestamp(cutoff_date)
    
    # 訓練期間と評価期間の組み合わせ
    train_periods = [
        ("0-3m", 0, 3),
        ("3-6m", 3, 6),
        ("6-9m", 6, 9),
        ("9-12m", 9, 12)
    ]
    
    eval_periods = [
        ("0-3m", 0, 3),
        ("3-6m", 3, 6),
        ("6-9m", 6, 9),
        ("9-12m", 9, 12)
    ]
    
    results = []
    
    # 各訓練期間のモデルを読み込み
    models = {}
    for train_name, train_start, train_end in train_periods:
        model_path = os.path.join(base_model_dir, f"train_{train_name}", "irl_model.pt")
        if os.path.exists(model_path):
            logger.info(f"モデル読み込み中: {model_path}")
            models[train_name] = RetentionIRLSystem.load_model(model_path)
            logger.info(f"モデル読み込み完了: {train_name}")
        else:
            logger.warning(f"モデルファイルが見つかりません: {model_path}")
    
    # クロス評価実行
    for train_name, train_start, train_end in train_periods:
        if train_name not in models:
            logger.warning(f"モデルが見つからないためスキップ: {train_name}")
            continue
            
        logger.info(f"訓練期間 {train_name} のモデルで評価開始")
        
        for eval_name, eval_start, eval_end in eval_periods:
            logger.info(f"  評価期間 {eval_name} を評価中...")
            
            # 評価用軌跡を抽出
            trajectories = extract_snapshot_evaluation_trajectories(
                df,
                cutoff_date=cutoff_date,
                history_window_months=history_window,
                future_window_start_months=eval_start,
                future_window_end_months=eval_end,
                min_history_events=min_history_events,
                project=project
            )
            
            if not trajectories:
                logger.warning(f"軌跡が見つかりません: {train_name}/{eval_name}")
                continue
            
            # モデル評価
            metrics = evaluate_normal_irl_with_snapshot_features(
                models[train_name],
                trajectories,
                cutoff_date,
                history_window
            )
            
            # 結果保存
            result_dir = os.path.join(output_dir, f"train_{train_name}", f"eval_{eval_name}")
            os.makedirs(result_dir, exist_ok=True)
            
            result_file = os.path.join(result_dir, "metrics.json")
            with open(result_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"  結果保存完了: {result_file}")
            logger.info(f"    AUC-ROC: {metrics['auc_roc']:.3f}")
            logger.info(f"    AUC-PR: {metrics['auc_pr']:.3f}")
            logger.info(f"    F1: {metrics['f1']:.3f}")
            
            # 結果をリストに追加
            results.append({
                'train_period': train_name,
                'eval_period': eval_name,
                'auc_roc': metrics['auc_roc'],
                'auc_pr': metrics['auc_pr'],
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'continuation_rate': metrics['continuation_rate'],
                'sample_count': len(trajectories)
            })
    
    # 結果をCSVとして保存
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "cross_evaluation_results.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"クロス評価結果を保存: {results_file}")
    
    logger.info("=" * 80)
    logger.info("クロス評価完了")
    logger.info(f"総評価数: {len(results)}")
    logger.info("=" * 80)

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="通常IRLモデルのクロス評価")
    parser.add_argument("--reviews", required=True, help="レビューログファイル")
    parser.add_argument("--base-model-dir", required=True, help="ベースモデルディレクトリ")
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--cutoff-date", default="2023-01-01", help="カットオフ日")
    parser.add_argument("--history-window", type=int, default=12, help="履歴期間（月）")
    parser.add_argument("--min-history-events", type=int, default=3, help="最小履歴イベント数")
    parser.add_argument("--project", default="openstack/nova", help="プロジェクト名")
    
    args = parser.parse_args()
    
    run_cross_evaluation(
        reviews_file=args.reviews,
        base_model_dir=args.base_model_dir,
        output_dir=args.output,
        cutoff_date=args.cutoff_date,
        history_window=args.history_window,
        min_history_events=args.min_history_events,
        project=args.project
    )

if __name__ == "__main__":
    main()
