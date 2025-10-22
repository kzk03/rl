#!/usr/bin/env python3
"""
単一モデル・複数期間評価

1つのラベル期間で学習したモデルを、
複数の異なる期間で評価して汎化性能を測定する。

使用例:
  # 3ヶ月後のラベルで学習し、1m/3m/6m/12m後で評価
  uv run python scripts/training/irl/train_irl_single_model_multi_eval.py \\
    --reviews data/review_requests_openstack_multi_5y_detail.csv \\
    --train-start 2022-01-01 --train-end 2024-01-01 \\
    --eval-start 2024-01-01 --eval-end 2025-01-01 \\
    --history-window 12 \\
    --train-label-start 0 --train-label-end 3 \\
    --eval-periods 1 3 6 12 \\
    --output outputs/single_model_multi_eval
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# 既存の訓練・評価関数をインポート
import sys
sys.path.append(str(Path(__file__).parent))
from train_irl_within_training_period import (
    load_review_logs,
    extract_temporal_trajectories_within_training_period,
    train_irl_model,
    evaluate_irl_model
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='単一モデル・複数期間評価'
    )
    
    # データ設定
    parser.add_argument('--reviews', type=Path, required=True,
                        help='レビューログCSVファイル')
    
    # 期間設定
    parser.add_argument('--train-start', type=str, required=True,
                        help='訓練期間の開始日（例: 2022-01-01）')
    parser.add_argument('--train-end', type=str, required=True,
                        help='訓練期間の終了日（例: 2024-01-01）')
    parser.add_argument('--eval-start', type=str, required=True,
                        help='評価期間の開始日（例: 2024-01-01）')
    parser.add_argument('--eval-end', type=str, required=True,
                        help='評価期間の終了日（例: 2025-01-01）')
    
    # ウィンドウ設定
    parser.add_argument('--history-window', type=int, default=12,
                        help='履歴ウィンドウ（ヶ月、固定）')
    
    # 訓練ラベル設定
    parser.add_argument('--train-label-start', type=int, default=0,
                        help='訓練時のラベル期間開始（ヶ月）')
    parser.add_argument('--train-label-end', type=int, default=3,
                        help='訓練時のラベル期間終了（ヶ月）')
    
    # 評価期間設定
    parser.add_argument('--eval-periods', type=int, nargs='+', 
                        default=[1, 3, 6, 12],
                        help='評価する期間のリスト（ヶ月）')
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=30,
                        help='訓練エポック数')
    parser.add_argument('--seq-len', type=int, default=20,
                        help='シーケンス長')
    
    # 出力設定
    parser.add_argument('--output', type=Path, 
                        default=Path('outputs/single_model_multi_eval'),
                        help='出力ディレクトリ')
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 期間をパース
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    
    logger.info("=" * 80)
    logger.info("単一モデル・複数期間評価")
    logger.info("=" * 80)
    logger.info(f"訓練期間: {train_start} ～ {train_end}")
    logger.info(f"評価期間: {eval_start} ～ {eval_end}")
    logger.info(f"履歴窓: {args.history_window}ヶ月（固定）")
    logger.info(f"訓練ラベル: {args.train_label_start}-{args.train_label_end}ヶ月後")
    logger.info(f"評価期間: {args.eval_periods}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # ===== ステップ1: モデル訓練 =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("ステップ1: モデル訓練")
    logger.info(f"ラベル: {args.train_label_start}-{args.train_label_end}ヶ月後の継続")
    logger.info("=" * 80)
    
    # 訓練データ抽出
    train_trajectories = extract_temporal_trajectories_within_training_period(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=args.history_window,
        future_window_start_months=args.train_label_start,
        future_window_end_months=args.train_label_end,
        sampling_interval_months=1,
        min_history_events=3
    )
    
    if len(train_trajectories) == 0:
        logger.error("訓練データが見つかりません")
        return
    
    # モデル訓練
    config = {
        'use_sequences': True,
        'sequence_length': args.seq_len,
        'state_dim': 32,
        'action_dim': 9,
        'hidden_dim': 128,
        'learning_rate': 0.001
    }
    
    irl_system = train_irl_model(train_trajectories, config, args.epochs)
    
    # モデル保存
    model_path = args.output / "irl_model.pth"
    irl_system.save_model(model_path)
    logger.info(f"モデルを保存: {model_path}")
    
    # ===== ステップ2: 複数期間で評価 =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("ステップ2: 複数期間での評価")
    logger.info("=" * 80)
    
    all_results = []
    
    for eval_months in args.eval_periods:
        logger.info("")
        logger.info(f"--- 評価: {eval_months}ヶ月後の継続 ---")
        
        # 評価データ抽出
        eval_trajectories = extract_temporal_trajectories_within_training_period(
            df=df,
            train_start=eval_start,
            train_end=eval_end,
            history_window_months=args.history_window,
            future_window_start_months=0,
            future_window_end_months=eval_months,
            sampling_interval_months=1,
            min_history_events=3
        )
        
        if len(eval_trajectories) == 0:
            logger.warning(f"評価データなし（{eval_months}ヶ月）")
            continue
        
        # 評価実行
        metrics = evaluate_irl_model(irl_system, eval_trajectories)
        
        # 結果を保存
        result = {
            'eval_period_months': eval_months,
            'eval_period_label': f'{eval_months}m',
            'train_label': f'{args.train_label_start}-{args.train_label_end}m',
            **metrics
        }
        all_results.append(result)
        
        # 個別結果を保存
        result_file = args.output / f"eval_{eval_months}m.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"結果を保存: {result_file}")
    
    # ===== ステップ3: 結果サマリー =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("評価結果サマリー")
    logger.info("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    
    # テーブル表示
    print("\n")
    print("| 評価期間 | AUC-PR | F1 | Precision | Recall | 継続率 |")
    print("|---------|--------|-----|-----------|--------|--------|")
    for _, row in results_df.iterrows():
        print(f"| {row['eval_period_label']:7s} | {row['auc_pr']:.3f}  | {row['f1']:.3f} | "
              f"{row['precision']:.3f}     | {row['recall']:.3f}  | {row['positive_rate']:.1%}  |")
    print()
    
    # CSV保存
    csv_path = args.output / "all_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"全結果を保存: {csv_path}")
    
    # JSON保存
    json_path = args.output / "all_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"全結果を保存: {json_path}")
    
    logger.info("=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

