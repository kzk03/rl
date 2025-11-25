#!/usr/bin/env python3
"""
範囲ラベル・複数範囲評価

学習: 各ステップで0-3m範囲内に貢献があるかでラベル付け（固定）
評価: 0-2m, 3-5m, 6-8m など異なる範囲で評価（可変）

使用例:
  uv run python scripts/training/irl/train_range_label_multi_range_eval.py \\
    --reviews data/review_requests_openstack_multi_5y_detail.csv \\
    --train-start 2022-01-01 --train-end 2024-01-01 \\
    --eval-start 2024-01-01 --eval-end 2025-01-01 \\
    --history-window 12 \\
    --train-range "0-3" \\
    --eval-ranges "0-2" "3-5" "6-8" "9-11" \\
    --output outputs/range_label_multi_eval
"""

import argparse
import json
import logging

# 既存の関数をインポート
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

sys.path.append(str(Path(__file__).parent))
from train_irl_within_training_period import (
    evaluate_irl_model,
    extract_temporal_trajectories_within_training_period,
    load_review_logs,
    train_irl_model,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_range(range_str: str) -> Tuple[int, int]:
    """範囲文字列をパース (例: "0-3" -> (0, 3))"""
    parts = range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"無効な範囲形式: {range_str}")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description='範囲ラベル・複数範囲評価'
    )
    
    # データ設定
    parser.add_argument('--reviews', type=Path, required=True)
    
    # 期間設定
    parser.add_argument('--train-start', type=str, required=True)
    parser.add_argument('--train-end', type=str, required=True)
    parser.add_argument('--eval-start', type=str, required=True)
    parser.add_argument('--eval-end', type=str, required=True)
    
    # ウィンドウ設定
    parser.add_argument('--history-window', type=int, default=12)
    
    # 訓練ラベル設定（範囲）
    parser.add_argument('--train-range', type=str, default='0-3',
                        help='訓練時のラベル範囲（例: 0-3）')
    
    # 評価範囲設定
    parser.add_argument('--eval-ranges', type=str, nargs='+',
                        default=['0-2', '3-5', '6-8'],
                        help='評価する範囲のリスト（例: 0-2 3-5）')
    
    # その他
    parser.add_argument('--sampling-interval', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--output', type=Path, 
                        default=Path('outputs/range_label_multi_eval'))
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 期間をパース
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    
    # 訓練範囲をパース
    train_range_start, train_range_end = parse_range(args.train_range)
    
    logger.info("=" * 80)
    logger.info("範囲ラベル・複数範囲評価")
    logger.info("=" * 80)
    logger.info(f"訓練期間: {train_start} ～ {train_end}")
    logger.info(f"評価期間: {eval_start} ～ {eval_end}")
    logger.info(f"履歴窓: {args.history_window}ヶ月")
    logger.info(f"訓練ラベル: {args.train_range}範囲内に貢献")
    logger.info(f"評価範囲: {args.eval_ranges}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # ===== ステップ1: モデル訓練 =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("ステップ1: モデル訓練")
    logger.info(f"ラベル: {train_range_start}-{train_range_end}m範囲内に貢献")
    logger.info("=" * 80)
    
    # 訓練データ抽出（範囲ラベル）
    train_trajectories = extract_temporal_trajectories_within_training_period(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=args.history_window,
        future_window_start_months=train_range_start,  # 範囲開始
        future_window_end_months=train_range_end,      # 範囲終了
        sampling_interval_months=args.sampling_interval,
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
    
    # 訓練設定を保存
    train_config = {
        'history_window': args.history_window,
        'train_range': args.train_range,
        'train_range_start': train_range_start,
        'train_range_end': train_range_end,
        'train_samples': len(train_trajectories),
    }
    with open(args.output / "train_config.json", 'w') as f:
        json.dump(train_config, f, indent=2)
    
    # ===== ステップ2: 複数範囲で評価 =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("ステップ2: 複数範囲での評価")
    logger.info("=" * 80)
    
    all_results = []
    
    for eval_range_str in args.eval_ranges:
        try:
            eval_start_m, eval_end_m = parse_range(eval_range_str)
        except ValueError as e:
            logger.error(str(e))
            continue
        
        logger.info("")
        logger.info(f"--- 評価: {eval_range_str}範囲 ---")
        
        # 評価データ抽出
        eval_trajectories = extract_temporal_trajectories_within_training_period(
            df=df,
            train_start=eval_start,
            train_end=eval_end,
            history_window_months=args.history_window,
            future_window_start_months=eval_start_m,
            future_window_end_months=eval_end_m,
            sampling_interval_months=args.sampling_interval,
            min_history_events=3
        )
        
        if len(eval_trajectories) == 0:
            logger.warning(f"評価データなし（{eval_range_str}）")
            continue
        
        # 評価実行
        metrics = evaluate_irl_model(irl_system, eval_trajectories)
        
        # 結果を保存
        result = {
            'eval_range': eval_range_str,
            'eval_range_start': eval_start_m,
            'eval_range_end': eval_end_m,
            'train_range': args.train_range,
            **metrics
        }
        all_results.append(result)
        
        # 個別結果を保存
        result_file = args.output / f"eval_{eval_range_str.replace('-', '_')}m.json"
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
    print(f"訓練ラベル: {args.train_range}範囲")
    print()
    print("| 評価範囲 | AUC-PR | F1 | Precision | Recall | 継続率 | サンプル数 |")
    print("|---------|--------|-----|-----------|--------|--------|-----------|")
    for _, row in results_df.iterrows():
        print(f"| {row['eval_range']:7s} | {row['auc_pr']:.3f}  | {row['f1']:.3f} | "
              f"{row['precision']:.3f}     | {row['recall']:.3f}  | "
              f"{row['positive_rate']:6.1%}  | {row['test_samples']:9d} |")
    print()
    
    # CSV/JSON保存
    results_df.to_csv(args.output / "all_results.csv", index=False)
    with open(args.output / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"全結果を保存: {args.output}/all_results.csv")
    logger.info("=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

