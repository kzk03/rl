#!/usr/bin/env python3
"""
Enhanced IRL (Attention) - Validation setで閾値決定版

訓練データをTrain/Valに分割し、Valで閾値を決定してデータリークを防ぐ
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# ランダムシード固定
RANDOM_SEED = 777
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from train_enhanced_irl_importants import (
    prepare_snapshot_evaluation_trajectories,
    prepare_trajectories_importants_style,
)

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_enhanced_irl_with_validation(
    train_trajectories: List[Dict],
    eval_trajectories: List[Dict],
    val_split: float = 0.2,
    epochs: int = 50,
    output_dir: Path = None
) -> Dict[str, float]:
    """
    Enhanced IRL訓練（Validation setで閾値決定）
    
    Args:
        train_trajectories: 訓練用軌跡（これをさらにtrain/valに分割）
        eval_trajectories: 評価用軌跡
        val_split: Validation分割率
        epochs: 訓練エポック数
        output_dir: 出力ディレクトリ
    """
    
    logger.info("=" * 80)
    logger.info("Enhanced IRL (Attention) 訓練 with Validation")
    logger.info("=" * 80)
    
    # Train/Val分割
    n_samples = len(train_trajectories)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    # シャッフル（再現性のため固定seed）
    indices = np.random.RandomState(RANDOM_SEED).permutation(n_samples)
    
    train_set = [train_trajectories[i] for i in indices[:n_train]]
    val_set = [train_trajectories[i] for i in indices[n_train:]]
    
    logger.info(f"データ分割:")
    logger.info(f"  Train: {n_train}サンプル")
    logger.info(f"  Validation: {n_val}サンプル")
    logger.info(f"  Eval: {len(eval_trajectories)}サンプル")
    
    # システム初期化
    config = {
        'state_dim': 10,
        'action_dim': 4,
        'hidden_dim': 128,
        'sequence': True,
        'seq_len': 10,
        'dropout': 0.2,
        'learning_rate': 0.001
    }
    system = RetentionIRLSystem(config)
    
    # Train setで訓練
    logger.info(f"\nTrain setで訓練開始...")
    training_results = system.train_irl_multi_step_labels(
        expert_trajectories=train_set,
        epochs=epochs
    )
    
    # Validation setで閾値決定
    logger.info(f"\nValidation setで閾値決定...")
    val_probs = []
    val_labels = []
    
    for traj in val_set:
        try:
            result = system.predict_continuation_probability(
                developer=traj['developer_info'],
                activity_history=traj['activity_history']
            )
            val_probs.append(result['continuation_probability'])
            
            if len(traj['step_labels']) > 0:
                val_labels.append(traj['step_labels'][-1])
            else:
                val_labels.append(0)
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"Validation軌跡スキップ: {e}")
            continue
    
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    
    # Validation setでF1最大の閾値を決定
    from sklearn.metrics import precision_recall_curve
    if len(set(val_labels)) > 1:
        precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        logger.info(f"最適閾値（Validation）: {optimal_threshold:.4f}")
        logger.info(f"  Validation F1: {f1_scores[optimal_idx]:.4f}")
        logger.info(f"  Validation Precision: {precision[optimal_idx]:.4f}")
        logger.info(f"  Validation Recall: {recall[optimal_idx]:.4f}")
    else:
        optimal_threshold = 0.5
        logger.warning("Validationのラベルが単一クラス → 閾値=0.5")
    
    # 確率分布の統計
    logger.info(f"\nValidation確率分布:")
    logger.info(f"  平均: {val_probs.mean():.4f}")
    logger.info(f"  中央値: {np.median(val_probs):.4f}")
    logger.info(f"  最小値: {val_probs.min():.4f}")
    logger.info(f"  最大値: {val_probs.max():.4f}")
    
    # 評価データで性能測定（閾値は変更しない）
    logger.info(f"\n評価データで性能測定...")
    eval_probs = []
    eval_labels = []
    
    for traj in eval_trajectories:
        try:
            result = system.predict_continuation_probability(
                developer=traj['developer_info'],
                activity_history=traj['activity_history']
            )
            eval_probs.append(result['continuation_probability'])
            
            if len(traj['step_labels']) > 0:
                eval_labels.append(traj['step_labels'][-1])
            else:
                eval_labels.append(0)
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"評価軌跡スキップ: {e}")
            continue
    
    # メトリクス計算
    eval_probs = np.array(eval_probs)
    eval_labels = np.array(eval_labels)
    
    if len(set(eval_labels)) > 1:
        from sklearn.metrics import auc as calc_auc
        
        auc_roc = roc_auc_score(eval_labels, eval_probs)
        precision, recall, _ = precision_recall_curve(eval_labels, eval_probs)
        auc_pr = calc_auc(recall, precision)
        
        # Validationで決めた閾値を使用
        eval_preds = (eval_probs >= optimal_threshold).astype(int)
        precision_val = precision_score(eval_labels, eval_preds, zero_division=0)
        recall_val = recall_score(eval_labels, eval_preds, zero_division=0)
        f1_val = f1_score(eval_labels, eval_preds, zero_division=0)
        
        # 参考: 閾値0.5での性能
        eval_preds_05 = (eval_probs >= 0.5).astype(int)
        f1_05 = f1_score(eval_labels, eval_preds_05, zero_division=0)
        precision_05 = precision_score(eval_labels, eval_preds_05, zero_division=0)
        recall_05 = recall_score(eval_labels, eval_preds_05, zero_division=0)
    else:
        auc_roc = 0.0
        auc_pr = 0.0
        precision_val = 0.0
        recall_val = 0.0
        f1_val = 0.0
        f1_05 = 0.0
        precision_05 = 0.0
        recall_05 = 0.0
    
    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'threshold': float(optimal_threshold),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1_score': float(f1_val),
        'positive_count': int(eval_labels.sum()),
        'negative_count': int((1 - eval_labels).sum()),
        'total_count': len(eval_labels),
        # 参考値
        'f1_at_0.5': float(f1_05),
        'precision_at_0.5': float(precision_05),
        'recall_at_0.5': float(recall_05),
        # データ分割情報
        'n_train': n_train,
        'n_val': n_val,
        'n_eval': len(eval_labels)
    }
    
    logger.info(f"\n評価結果:")
    logger.info(f"  AUC-ROC: {auc_roc:.4f}")
    logger.info(f"  AUC-PR: {auc_pr:.4f}")
    logger.info(f"  F1 (閾値={optimal_threshold:.4f}): {f1_val:.4f}")
    logger.info(f"  Precision: {precision_val:.4f}")
    logger.info(f"  Recall: {recall_val:.4f}")
    logger.info(f"\n  参考: F1 (閾値=0.5): {f1_05:.4f}")
    logger.info(f"  Precision (0.5): {precision_05:.4f}")
    logger.info(f"  Recall (0.5): {recall_05:.4f}")
    
    # 保存
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル保存
        system.save_model(str(output_dir / "enhanced_irl_model.pt"))
        
        # メトリクス保存
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Validation結果も保存
        val_metrics = {
            'val_threshold': float(optimal_threshold),
            'val_probs_mean': float(val_probs.mean()),
            'val_probs_std': float(val_probs.std()),
            'val_probs_min': float(val_probs.min()),
            'val_probs_max': float(val_probs.max()),
            'val_positive_count': int(val_labels.sum()),
            'val_negative_count': int((1 - val_labels).sum())
        }
        with open(output_dir / "validation_info.json", 'w') as f:
            json.dump(val_metrics, f, indent=2)
        
        logger.info(f"\n✅ 保存完了: {output_dir}")
    
    return metrics


def evaluate_with_existing_model(
    model_path: Path,
    eval_trajectories: List[Dict],
    threshold: float,
    output_dir: Path = None
) -> Dict[str, float]:
    """
    既存モデルで評価のみ実施（閾値は指定値を使用）
    
    Args:
        model_path: モデルファイルパス
        eval_trajectories: 評価軌跡
        threshold: 分類閾値（Validationで決定されたもの）
        output_dir: 出力ディレクトリ
    """

    logger.info("=" * 80)
    logger.info("Enhanced IRL (Attention) 評価のみ")
    logger.info("=" * 80)
    logger.info(f"モデル: {model_path}")
    logger.info(f"閾値（Validation決定）: {threshold:.4f}")

    # モデル読み込み
    system = RetentionIRLSystem.load_model(str(model_path))

    # 評価
    eval_probs = []
    eval_labels = []

    logger.info(f"評価軌跡: {len(eval_trajectories)}")

    for traj in eval_trajectories:
        try:
            result = system.predict_continuation_probability(
                developer=traj['developer_info'],
                activity_history=traj['activity_history']
            )
            eval_probs.append(result['continuation_probability'])

            if len(traj['step_labels']) > 0:
                eval_labels.append(traj['step_labels'][-1])
            else:
                eval_labels.append(0)
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"評価軌跡スキップ: {e}")
            continue

    # メトリクス計算
    eval_probs = np.array(eval_probs)
    eval_labels = np.array(eval_labels)

    if len(set(eval_labels)) > 1:
        from sklearn.metrics import auc as calc_auc
        from sklearn.metrics import precision_recall_curve
        
        auc_roc = roc_auc_score(eval_labels, eval_probs)
        precision, recall, _ = precision_recall_curve(eval_labels, eval_probs)
        auc_pr = calc_auc(recall, precision)

        # 指定された閾値を使用
        eval_preds = (eval_probs >= threshold).astype(int)
        precision_val = precision_score(eval_labels, eval_preds, zero_division=0)
        recall_val = recall_score(eval_labels, eval_preds, zero_division=0)
        f1_val = f1_score(eval_labels, eval_preds, zero_division=0)
        
        # 参考: 閾値0.5
        eval_preds_05 = (eval_probs >= 0.5).astype(int)
        f1_05 = f1_score(eval_labels, eval_preds_05, zero_division=0)
        precision_05 = precision_score(eval_labels, eval_preds_05, zero_division=0)
        recall_05 = recall_score(eval_labels, eval_preds_05, zero_division=0)
    else:
        auc_roc = 0.0
        auc_pr = 0.0
        precision_val = 0.0
        recall_val = 0.0
        f1_val = 0.0
        f1_05 = 0.0
        precision_05 = 0.0
        recall_05 = 0.0

    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'threshold': float(threshold),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1_score': float(f1_val),
        'positive_count': int(eval_labels.sum()),
        'negative_count': int((1 - eval_labels).sum()),
        'total_count': len(eval_labels),
        'f1_at_0.5': float(f1_05),
        'precision_at_0.5': float(precision_05),
        'recall_at_0.5': float(recall_05)
    }

    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"AUC-PR: {auc_pr:.4f}")
    logger.info(f"F1 (閾値={threshold:.4f}): {f1_val:.4f}")
    logger.info(f"Precision: {precision_val:.4f}")
    logger.info(f"Recall: {recall_val:.4f}")
    logger.info(f"\n参考: F1 (閾値=0.5): {f1_05:.4f}")

    # 保存
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✅ 保存完了: {output_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, help="レビューデータCSV")
    parser.add_argument("--train-start", required=True, help="訓練開始日 (YYYY-MM-DD)")
    parser.add_argument("--train-end", required=True, help="訓練終了日 (YYYY-MM-DD)")
    parser.add_argument("--eval-start", required=True, help="評価開始日 (YYYY-MM-DD)")
    parser.add_argument("--eval-end", required=True, help="評価終了日 (YYYY-MM-DD)")
    parser.add_argument("--future-window-start", type=int, required=True, help="Future Window開始(月)")
    parser.add_argument("--future-window-end", type=int, required=True, help="Future Window終了(月)")
    parser.add_argument("--epochs", type=int, default=50, help="訓練エポック数")
    parser.add_argument("--min-history-events", type=int, default=3, help="最小履歴イベント数")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation分割率")
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--project", default="openstack/nova", help="プロジェクト名")
    parser.add_argument("--model", type=str, default=None, help="既存モデルのパス（評価のみの場合）")
    parser.add_argument("--threshold", type=float, default=None, help="閾値（評価のみの場合に指定）")

    args = parser.parse_args()

    # データ読み込み
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])

    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    eval_start = pd.to_datetime(args.eval_start)
    eval_end = pd.to_datetime(args.eval_end)

    output_dir = Path(args.output)

    # モデルが指定されている場合は評価のみ
    if args.model:
        logger.info("既存モデルでスナップショット評価を実施")

        if args.threshold is None:
            logger.error("評価のみの場合は --threshold を指定してください")
            sys.exit(1)

        # スナップショット評価データ準備
        eval_trajectories = prepare_snapshot_evaluation_trajectories(
            df, eval_start,
            args.future_window_start, args.future_window_end,
            args.min_history_events, args.project
        )

        # モデルで評価
        model_path = Path(args.model)
        metrics = evaluate_with_existing_model(
            model_path, eval_trajectories, args.threshold, output_dir
        )
    else:
        logger.info("訓練と評価を実施（Validation使用）")

        # 訓練データ準備
        train_trajectories = prepare_trajectories_importants_style(
            df, train_start, train_end,
            args.future_window_start, args.future_window_end,
            args.min_history_events, args.project
        )

        # 評価データ準備
        eval_trajectories = prepare_trajectories_importants_style(
            df, eval_start, eval_end,
            args.future_window_start, args.future_window_end,
            args.min_history_events, args.project
        )

        # Enhanced IRL訓練（Validation使用）
        metrics = train_enhanced_irl_with_validation(
            train_trajectories, eval_trajectories,
            args.val_split, args.epochs, output_dir
        )

    logger.info("=" * 80)
    logger.info("完了!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
