#!/usr/bin/env python3
"""
各タイムステップラベル付きIRL訓練スクリプト

重要な設計:
- 各タイムステップから将来の貢献をラベルとして計算
- seq_len個のラベルを生成（各ステップで1つ）
- すべてのステップで予測と損失を計算
- より豊富な学習信号を活用

目的:
- 各時点での継続予測を学習
- 時系列パターンをより詳細に学習
- 予測精度の向上
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

SCRIPTS_DIR = ROOT / "scripts" / "training" / "irl"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from train_irl_within_training_period import (
    extract_cutoff_evaluation_trajectories,
    extract_multi_step_label_trajectories,
    find_optimal_threshold,
    load_review_logs,
)

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_irl_model_multi_step(
    trajectories: List[Dict[str, Any]],
    config: Dict[str, Any],
    epochs: int = 30
) -> RetentionIRLSystem:
    """
    各ステップラベル付きIRLモデルを訓練
    
    Args:
        trajectories: 各ステップラベル付き軌跡データ
        config: モデル設定
        epochs: エポック数
        
    Returns:
        訓練済みモデル
    """
    logger.info("=" * 80)
    logger.info("IRL訓練開始")
    logger.info(f"軌跡数: {len(trajectories)}")
    logger.info(f"エポック数: {epochs}")
    logger.info(f"目標: 各時点での継続予測を学習")
    logger.info("=" * 80)
    
    # IRLシステムの初期化
    irl_system = RetentionIRLSystem(config)
    
    # 訓練
    result = irl_system.train_irl_multi_step_labels(
        expert_trajectories=trajectories,
        epochs=epochs
    )
    
    logger.info("=" * 80)
    logger.info(f"訓練完了: 最終損失 = {result['final_loss']:.4f}")
    logger.info("=" * 80)
    
    return irl_system


def evaluate_model(
    irl_system: RetentionIRLSystem,
    eval_trajectories: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """モデルを評価"""
    logger.info("=" * 80)
    logger.info("モデル評価開始")
    logger.info(f"評価サンプル数: {len(eval_trajectories)}")
    logger.info(f"閾値: {threshold}")
    logger.info("=" * 80)
    
    predictions = []
    true_labels = []
    
    for trajectory in eval_trajectories:
        developer = trajectory['developer']
        activity_history = trajectory['activity_history']
        context_date = trajectory.get('context_date', datetime.now())
        true_label = trajectory.get('future_contribution', False)
        
        # 予測
        result = irl_system.predict_continuation_probability(
            developer, activity_history, context_date
        )
        
        predictions.append(result['continuation_probability'])
        true_labels.append(1 if true_label else 0)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # 最適閾値を探索
    optimal_threshold, best_metrics = find_optimal_threshold(true_labels, predictions)
    
    # 評価指標を計算（最適閾値使用）
    pred_binary = (predictions >= optimal_threshold).astype(int)
    
    metrics = {
        'auc_roc': float(roc_auc_score(true_labels, predictions)),
        'precision': float(precision_score(true_labels, pred_binary, zero_division=0)),
        'recall': float(recall_score(true_labels, pred_binary, zero_division=0)),
        'f1': float(f1_score(true_labels, pred_binary, zero_division=0)),
        'optimal_threshold': float(optimal_threshold),
        'best_f1': float(best_metrics['f1']) if best_metrics else 0.0,
        'sample_count': len(eval_trajectories),
        'positive_count': int(true_labels.sum()),
        'negative_count': int((1 - true_labels).sum()),
        'continuation_rate': float(true_labels.mean()),
    }
    
    # AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, predictions)
    metrics['auc_pr'] = float(auc(recall_curve, precision_curve))
    
    logger.info(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    logger.info(f"AUC-PR: {metrics['auc_pr']:.3f}")
    logger.info(f"最適閾値: {metrics['optimal_threshold']:.3f} (F1={metrics['best_f1']:.3f})")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall: {metrics['recall']:.3f}")
    logger.info(f"F1: {metrics['f1']:.3f}")
    logger.info(f"継続率: {metrics['continuation_rate']:.1%}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='各ステップラベル付きIRL訓練')
    parser.add_argument('--reviews', type=str, required=True, help='レビューログCSV')
    parser.add_argument('--train-start', type=str, required=True, help='学習開始日')
    parser.add_argument('--train-end', type=str, required=True, help='学習終了日')
    parser.add_argument('--eval-start', type=str, required=True, help='評価開始日')
    parser.add_argument('--eval-end', type=str, required=True, help='評価終了日')
    parser.add_argument('--history-window', type=int, default=12, help='履歴ウィンドウ（ヶ月）')
    parser.add_argument('--future-window-start', type=int, default=0, help='将来窓開始（ヶ月）')
    parser.add_argument('--future-window-end', type=int, default=3, help='将来窓終了（ヶ月）')
    parser.add_argument('--seq-len', type=int, default=20, help='LSTMシーケンス長')
    parser.add_argument('--epochs', type=int, default=30, help='訓練エポック数')
    parser.add_argument('--eval-future-window-start', type=int, default=None, 
                        help='評価時の将来窓開始（ヶ月、デフォルト=future-window-start）')
    parser.add_argument('--eval-future-window-end', type=int, default=None, 
                        help='評価時の将来窓終了（ヶ月、デフォルト=future-window-end）')
    parser.add_argument('--output', type=str, required=True, help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    # 評価用の将来窓パラメータ（デフォルトは訓練時と同じ）
    eval_future_start = args.eval_future_window_start if args.eval_future_window_start is not None else args.future_window_start
    eval_future_end = args.eval_future_window_end if args.eval_future_window_end is not None else args.future_window_end
    
    logger.info("各タイムステップラベル付きIRL訓練")
    logger.info("=" * 80)
    logger.info(f"レビューログ: {args.reviews}")
    logger.info(f"学習期間: {args.train_start} ～ {args.train_end}")
    logger.info(f"評価期間: {args.eval_start} ～ {args.eval_end}")
    logger.info(f"履歴ウィンドウ: {args.history_window}ヶ月")
    logger.info(f"訓練ラベル: {args.future_window_start}～{args.future_window_end}ヶ月")
    logger.info(f"評価ラベル: {eval_future_start}～{eval_future_end}ヶ月")
    if eval_future_start != args.future_window_start or eval_future_end != args.future_window_end:
        logger.info("  ⚠️ 訓練と評価で異なるラベル期間を使用（クロス評価）")
    logger.info(f"LSTMシーケンス長: {args.seq_len}")
    logger.info(f"エポック数: {args.epochs}")
    logger.info(f"出力: {output_dir}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(Path(args.reviews))
    
    # 日付を解析
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    
    # 訓練データを抽出（各ステップラベル付き）
    logger.info("\n" + "=" * 80)
    logger.info("訓練データ抽出（各ステップラベル付き）")
    logger.info("=" * 80)
    
    train_trajectories = extract_multi_step_label_trajectories(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=args.history_window,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        sampling_interval_months=1,
        seq_len=args.seq_len,
        min_history_events=3,
    )
    
    logger.info(f"訓練サンプル数: {len(train_trajectories)}")
    
    if len(train_trajectories) == 0:
        logger.error("訓練データが抽出できませんでした")
        return
    
    # 評価データを抽出
    logger.info("\n" + "=" * 80)
    logger.info("評価データ抽出")
    logger.info("=" * 80)
    
    eval_trajectories = extract_cutoff_evaluation_trajectories(
        df=df,
        cutoff_date=train_end,  # 訓練終了日から評価
        history_window_months=args.history_window,
        future_window_start_months=eval_future_start,
        future_window_end_months=eval_future_end,
        min_history_events=3,
    )
    
    logger.info(f"評価サンプル数: {len(eval_trajectories)}")
    
    # モデル設定
    config = {
        'state_dim': 10,
        'action_dim': 5,
        'hidden_dim': 128,
        'learning_rate': 0.0001,
        'sequence': True,
        'seq_len': args.seq_len,
    }
    
    # モデル訓練
    irl_system = train_irl_model_multi_step(
        train_trajectories,
        config,
        epochs=args.epochs
    )
    
    # モデルを保存
    model_path = output_dir / 'irl_model.pt'
    torch.save(irl_system.network.state_dict(), model_path)
    logger.info(f"モデル保存: {model_path}")
    
    # 評価
    if len(eval_trajectories) > 0:
        metrics = evaluate_model(irl_system, eval_trajectories)
        
        # メトリクスを保存
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"メトリクス保存: {metrics_path}")
        
        # 軌跡を保存
        trajectories_path = output_dir / 'train_trajectories.pkl'
        with open(trajectories_path, 'wb') as f:
            pickle.dump(train_trajectories[:100], f)  # 最初の100個のみ保存
        logger.info(f"訓練軌跡保存: {trajectories_path}")
        
        eval_trajectories_path = output_dir / 'eval_trajectories.pkl'
        with open(eval_trajectories_path, 'wb') as f:
            pickle.dump(eval_trajectories, f)
        logger.info(f"評価軌跡保存: {eval_trajectories_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

