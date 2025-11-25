#!/usr/bin/env python3
"""
固定訓練ラベル・複数評価期間

学習: 各ステップで3ヶ月後の貢献フラグで訓練
評価: 0-3m, 4-6m, 7-9m などの複数期間で評価

使用例:
  uv run python scripts/training/irl/train_irl_fixed_train_label_multi_eval.py \\
    --reviews data/review_requests_openstack_multi_5y_detail.csv \\
    --train-start 2022-01-01 --train-end 2024-01-01 \\
    --eval-start 2024-01-01 --eval-end 2025-01-01 \\
    --history-window 12 \\
    --train-label-months 3 \\
    --eval-windows "0-3" "4-6" "7-9" "10-12" \\
    --output outputs/fixed_train_label_multi_eval
"""

import argparse
import json
import logging

# 既存の関数をインポート
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def parse_window_range(window_str: str) -> Tuple[int, int]:
    """
    ウィンドウ範囲文字列をパース
    
    例: "0-3" -> (0, 3)
        "4-6" -> (4, 6)
    """
    parts = window_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"無効なウィンドウ形式: {window_str}。'start-end'形式で指定してください")
    
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description='固定訓練ラベル・複数評価期間'
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
    parser.add_argument('--train-label-months', type=int, default=3,
                        help='訓練時のラベル: nヶ月後に貢献があるか（デフォルト: 3）')
    
    # 評価ウィンドウ設定
    parser.add_argument('--eval-windows', type=str, nargs='+', 
                        default=['0-3', '4-6', '7-9'],
                        help='評価するウィンドウのリスト（例: 0-3 4-6 7-9）')
    
    # サンプリング設定
    parser.add_argument('--sampling-interval', type=int, default=1,
                        help='サンプリング間隔（ヶ月、デフォルト: 1）')
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=30,
                        help='訓練エポック数')
    parser.add_argument('--seq-len', type=int, default=20,
                        help='シーケンス長')
    
    # 出力設定
    parser.add_argument('--output', type=Path, 
                        default=Path('outputs/fixed_train_label_multi_eval'),
                        help='出力ディレクトリ')
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 期間をパース
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    
    logger.info("=" * 80)
    logger.info("固定訓練ラベル・複数評価期間")
    logger.info("=" * 80)
    logger.info(f"訓練期間: {train_start} ～ {train_end}")
    logger.info(f"評価期間: {eval_start} ～ {eval_end}")
    logger.info(f"履歴窓: {args.history_window}ヶ月（固定）")
    logger.info(f"訓練ラベル: {args.train_label_months}ヶ月後の貢献")
    logger.info(f"サンプリング間隔: {args.sampling_interval}ヶ月")
    logger.info(f"評価ウィンドウ: {args.eval_windows}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # ===== ステップ1: モデル訓練 =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("ステップ1: モデル訓練")
    logger.info(f"各ステップで{args.train_label_months}ヶ月後の貢献フラグを使用")
    logger.info("=" * 80)
    
    # 訓練データ抽出（nヶ月後の貢献をラベルとして使用）
    train_trajectories = extract_temporal_trajectories_within_training_period(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=args.history_window,
        future_window_start_months=args.train_label_months,  # nヶ月後
        future_window_end_months=args.train_label_months,    # nヶ月後（点）
        sampling_interval_months=args.sampling_interval,
        min_history_events=3
    )
    
    if len(train_trajectories) == 0:
        logger.error("訓練データが見つかりません")
        return
    
    logger.info(f"訓練サンプル数: {len(train_trajectories)}")
    
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
        'train_label_months': args.train_label_months,
        'sampling_interval': args.sampling_interval,
        'train_samples': len(train_trajectories),
        'epochs': args.epochs,
        'seq_len': args.seq_len,
        'train_start': str(train_start),
        'train_end': str(train_end),
    }
    
    with open(args.output / "train_config.json", 'w') as f:
        json.dump(train_config, f, indent=2)
    
    # ===== ステップ2: 複数ウィンドウで評価 =====
    logger.info("")
    logger.info("=" * 80)
    logger.info("ステップ2: 複数評価ウィンドウでの評価")
    logger.info("=" * 80)
    
    all_results = []
    
    for window_str in args.eval_windows:
        try:
            start_months, end_months = parse_window_range(window_str)
        except ValueError as e:
            logger.error(str(e))
            continue
        
        logger.info("")
        logger.info(f"--- 評価: {start_months}-{end_months}ヶ月後の貢献 ---")
        
        # 評価データ抽出
        eval_trajectories = extract_temporal_trajectories_within_training_period(
            df=df,
            train_start=eval_start,
            train_end=eval_end,
            history_window_months=args.history_window,
            future_window_start_months=start_months,
            future_window_end_months=end_months,
            sampling_interval_months=args.sampling_interval,
            min_history_events=3
        )
        
        if len(eval_trajectories) == 0:
            logger.warning(f"評価データなし（{window_str}）")
            continue
        
        logger.info(f"評価サンプル数: {len(eval_trajectories)}")
        
        # 評価実行
        metrics = evaluate_irl_model(irl_system, eval_trajectories)
        
        # 結果を保存
        result = {
            'eval_window': window_str,
            'eval_window_start': start_months,
            'eval_window_end': end_months,
            'train_label_months': args.train_label_months,
            **metrics
        }
        all_results.append(result)
        
        # 個別結果を保存
        result_file = args.output / f"eval_{window_str.replace('-', '_')}m.json"
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
    print(f"訓練ラベル: {args.train_label_months}ヶ月後")
    print()
    print("| 評価期間 | AUC-PR | F1 | Precision | Recall | 継続率 | サンプル数 |")
    print("|---------|--------|-----|-----------|--------|--------|-----------|")
    for _, row in results_df.iterrows():
        print(f"| {row['eval_window']:7s} | {row['auc_pr']:.3f}  | {row['f1']:.3f} | "
              f"{row['precision']:.3f}     | {row['recall']:.3f}  | "
              f"{row['positive_rate']:6.1%}  | {row['test_samples']:9d} |")
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
    
    # ===== ステップ4: 可視化 =====
    logger.info("")
    logger.info("結果を可視化中...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import rcParams

        # 日本語フォント設定
        rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio']
        rcParams['font.family'] = 'sans-serif'
        rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ('auc_pr', 'AUC-PR'),
            ('f1', 'F1スコア'),
            ('precision', 'Precision'),
            ('positive_rate', '継続率')
        ]
        
        for idx, (metric, name) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            colors = sns.color_palette('viridis', len(results_df))
            bars = ax.bar(
                results_df['eval_window'],
                results_df[metric],
                color=colors,
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5
            )
            
            # 値をバーの上に表示
            for bar in bars:
                height = bar.get_height()
                label = f'{height:.1%}' if metric == 'positive_rate' else f'{height:.3f}'
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            ax.set_xlabel('評価期間（ヶ月後）', fontsize=12)
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(
                f'{name} vs 評価期間\n（訓練ラベル: {args.train_label_months}ヶ月後）',
                fontsize=14,
                fontweight='bold'
            )
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        viz_path = args.output / "evaluation_comparison.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"可視化を保存: {viz_path}")
        plt.close()
        
    except Exception as e:
        logger.warning(f"可視化に失敗: {e}")
    
    logger.info("=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

