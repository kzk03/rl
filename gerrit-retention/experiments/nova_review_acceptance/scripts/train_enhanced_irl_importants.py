#!/usr/bin/env python3
"""
Enhanced IRL (Attention) - importants準拠のデータ準備で訓練

importantsのtrain_irl_review_acceptance.pyのデータ準備ロジックを使用し、
モデルだけEnhanced IRL (Attention)に置き換える
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

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_trajectories_importants_style(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history_requests: int = 3,
    project: str = None
) -> List[Dict[str, Any]]:
    """
    importantsと同じデータ準備方式（データリークなし版）
    
    重要：訓練期間を分割してデータリークを防止
    - 特徴量計算期間: train_start ~ (train_end - future_window_end_months)
    - ラベル計算期間: (train_end - future_window_end_months) ~ train_end
    
    各レビュワーについて:
    - 特徴量計算期間内の履歴から月次の軌跡を生成
    - 各月末からfuture_window後の承諾有無をラベルとする（train_end内）
    """
    if project:
        df = df[df['project'] == project].copy()
    
    date_col = 'request_time'
    reviewer_col = 'reviewer_email'
    label_col = 'label'
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 特徴量計算期間の終了（ラベル期間の開始）
    feature_end = train_end - pd.DateOffset(months=future_window_end_months)
    
    # 特徴量計算期間のデータ
    feature_df = df[(df[date_col] >= train_start) & (df[date_col] < feature_end)]
    reviewers = feature_df[reviewer_col].unique()
    
    trajectories = []
    skipped_min_requests = 0
    
    logger.info(f"訓練期間全体: {train_start} ~ {train_end}")
    logger.info(f"特徴量計算期間: {train_start} ~ {feature_end}")
    logger.info(f"ラベル計算期間: {feature_end} ~ {train_end}")
    logger.info(f"Future Window: {future_window_start_months} ~ {future_window_end_months}ヶ月")
    logger.info(f"レビュワー候補: {len(reviewers)}")
    
    for reviewer in reviewers:
        reviewer_history = feature_df[feature_df[reviewer_col] == reviewer]
        
        # 最小依頼数チェック
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue
        
        # 月次の軌跡生成（特徴量計算期間内のみ）
        history_months = pd.date_range(
            start=train_start,
            end=feature_end,  # 特徴量計算期間の終了まで
            freq='MS'  # 月初
        )
        
        step_labels = []
        monthly_activity_histories = []
        
        for month_start in history_months[:-1]:
            month_end = month_start + pd.DateOffset(months=1)
            
            # この月からfuture_window後のラベル計算期間
            future_start = month_end + pd.DateOffset(months=future_window_start_months)
            future_end = month_end + pd.DateOffset(months=future_window_end_months)
            
            # train_endを超えないように制限（データリーク防止）
            if future_end > train_end:
                future_end = train_end
            
            # future_startがtrain_endを超える場合、この月のラベルは作成不可
            if future_start >= train_end:
                continue
            
            # 将来期間のレビュー依頼（train_end内）
            month_future_df = df[
                (df[date_col] >= future_start) &
                (df[date_col] < future_end) &
                (df[reviewer_col] == reviewer)
            ]
            
            # ラベル
            if len(month_future_df) == 0:
                month_label = 0
            else:
                month_accepted = month_future_df[month_future_df[label_col] == 1]
                month_label = 1 if len(month_accepted) > 0 else 0
            
            step_labels.append(month_label)
            
            # この月時点までの活動履歴（特徴量計算期間内）
            month_history = reviewer_history[reviewer_history[date_col] < month_end]
            monthly_activities = []
            for _, row in month_history.iterrows():
                activity = {
                    'timestamp': row[date_col],
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                    'request_time': row.get('request_time', row[date_col]),
                    'response_time': row.get('first_response_time'),
                    'accepted': row.get(label_col, 0) == 1,
                }
                monthly_activities.append(activity)
            monthly_activity_histories.append(monthly_activities)
        
        # 軌跡が生成されなかった場合はスキップ
        if len(step_labels) == 0:
            continue
        
        # 全期間の活動履歴
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col],
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'request_time': row.get('request_time', row[date_col]),
                'response_time': row.get('first_response_time'),
                'accepted': row.get(label_col, 0) == 1,
            }
            activity_history.append(activity)
        
        # 開発者情報
        developer_info = {
            'developer_email': reviewer,
            'first_seen': reviewer_history[date_col].min(),
            'changes_authored': 0,
            'changes_reviewed': len(reviewer_history[reviewer_history[label_col] == 1]),
            'requests_received': len(reviewer_history),
            'requests_accepted': len(reviewer_history[reviewer_history[label_col] == 1]),
            'requests_rejected': len(reviewer_history[reviewer_history[label_col] == 0]),
            'acceptance_rate': len(reviewer_history[reviewer_history[label_col] == 1]) / len(reviewer_history),
            'projects': reviewer_history['project'].unique().tolist() if 'project' in reviewer_history.columns else []
        }
        
        # 軌跡を作成
        trajectory = {
            'developer_info': developer_info,
            'activity_history': activity_history,
            'monthly_activity_histories': monthly_activity_histories,
            'context_date': train_end,
            'step_labels': step_labels,
            'seq_len': len(step_labels),
        }
        
        trajectories.append(trajectory)
    
    logger.info(f"軌跡生成完了: {len(trajectories)}サンプル")
    logger.info(f"スキップ（最小依頼数未満）: {skipped_min_requests}")
    
    return trajectories


def prepare_snapshot_evaluation_trajectories(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history_requests: int = 3,
    project: str = None
) -> List[Dict[str, Any]]:
    """
    スナップショット評価用の軌跡準備（単一時点評価）

    Args:
        df: 全レビューデータ
        snapshot_date: スナップショット日（例: 2023-01-01）
        future_window_start_months: Future Window開始（月）
        future_window_end_months: Future Window終了（月）
        min_history_requests: 最小履歴依頼数
        project: プロジェクト名フィルタ

    Returns:
        軌跡リスト
    """
    if project:
        df = df[df['project'] == project].copy()

    date_col = 'request_time'
    reviewer_col = 'reviewer_email'
    label_col = 'label'

    df[date_col] = pd.to_datetime(df[date_col])

    # スナップショット以前の履歴データ
    history_df = df[df[date_col] < snapshot_date]
    reviewers = history_df[reviewer_col].unique()

    # ラベル計算期間
    label_start = snapshot_date + pd.DateOffset(months=future_window_start_months)
    label_end = snapshot_date + pd.DateOffset(months=future_window_end_months)

    trajectories = []
    skipped_min_requests = 0

    logger.info(f"スナップショット日: {snapshot_date}")
    logger.info(f"ラベル期間: {label_start} ~ {label_end}")
    logger.info(f"レビュワー候補: {len(reviewers)}")

    for reviewer in reviewers:
        # スナップショット時点までの履歴
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]

        # 最小依頼数チェック
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue

        # ラベル期間のレビュー依頼
        reviewer_future = df[
            (df[date_col] >= label_start) &
            (df[date_col] < label_end) &
            (df[reviewer_col] == reviewer)
        ]

        # ラベル期間に依頼がない場合はスキップ
        if len(reviewer_future) == 0:
            continue

        # ラベル: 受理したレビューがあるか
        accepted_reviews = reviewer_future[reviewer_future[label_col] == 1]
        label = 1 if len(accepted_reviews) > 0 else 0

        # 活動履歴
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col],
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'request_time': row.get('request_time', row[date_col]),
                'response_time': row.get('first_response_time'),
                'accepted': row.get(label_col, 0) == 1,
            }
            activity_history.append(activity)

        # 開発者情報
        developer_info = {
            'developer_email': reviewer,
            'first_seen': reviewer_history[date_col].min(),
            'changes_authored': 0,
            'changes_reviewed': len(reviewer_history[reviewer_history[label_col] == 1]),
            'requests_received': len(reviewer_history),
            'requests_accepted': len(reviewer_history[reviewer_history[label_col] == 1]),
            'requests_rejected': len(reviewer_history[reviewer_history[label_col] == 0]),
            'acceptance_rate': len(reviewer_history[reviewer_history[label_col] == 1]) / len(reviewer_history),
            'projects': reviewer_history['project'].unique().tolist() if 'project' in reviewer_history.columns else []
        }

        # 軌跡を作成（単一ラベル）
        trajectory = {
            'developer_info': developer_info,
            'activity_history': activity_history,
            'context_date': snapshot_date,
            'step_labels': [label],  # 単一ラベル
            'seq_len': 1,
        }

        trajectories.append(trajectory)

    logger.info(f"軌跡生成完了: {len(trajectories)}サンプル")
    logger.info(f"スキップ（最小依頼数未満）: {skipped_min_requests}")

    positive = sum(1 for t in trajectories if t['step_labels'][0] == 1)
    negative = len(trajectories) - positive
    logger.info(f"ポジティブ: {positive}, ネガティブ: {negative}")

    return trajectories


def train_enhanced_irl(
    train_trajectories: List[Dict],
    eval_trajectories: List[Dict],
    epochs: int = 50,
    output_dir: Path = None
) -> Dict[str, float]:
    """Enhanced IRL (Attention)で訓練"""
    
    logger.info("=" * 80)
    logger.info("Enhanced IRL (Attention) 訓練開始")
    logger.info("=" * 80)
    
    # システム初期化（configベース）
    # importantsのデータ形式に合わせる: action_dim=4
    config = {
        'state_dim': 10,
        'action_dim': 4,  # importantsと同じ（強度、協力、応答速度、規模）
        'hidden_dim': 128,
        'sequence': True,  # LSTM使用
        'seq_len': 10,
        'dropout': 0.2,
        'learning_rate': 0.001
    }
    system = RetentionIRLSystem(config)
    
    # 訓練
    logger.info(f"訓練軌跡: {len(train_trajectories)}")
    logger.info(f"評価軌跡: {len(eval_trajectories)}")
    
    training_results = system.train_irl_multi_step_labels(
        expert_trajectories=train_trajectories,
        epochs=epochs
    )
    
    # 訓練データで最適閾値を決定
    logger.info("訓練データで最適閾値を決定...")
    train_results = []
    train_labels = []
    
    for traj in train_trajectories:
        try:
            result = system.predict_continuation_probability(
                developer=traj['developer_info'],
                activity_history=traj['activity_history']
            )
            train_results.append(result['continuation_probability'])
            
            # 最後のstep_labelを訓練ラベルとして使用
            if len(traj['step_labels']) > 0:
                train_labels.append(traj['step_labels'][-1])
            else:
                train_labels.append(0)
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"訓練軌跡スキップ: {e}")
            continue
    
    train_probs = np.array(train_results)
    train_labels = np.array(train_labels)
    
    # 訓練データでF1最大の閾値を決定
    from sklearn.metrics import precision_recall_curve
    if len(set(train_labels)) > 1:
        precision, recall, thresholds = precision_recall_curve(train_labels, train_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        logger.info(f"最適閾値（訓練データ）: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.5
        logger.warning("訓練データのラベルが単一クラス → 閾値=0.5")
    
    # 評価データで性能測定（閾値は訓練データで決定済み）
    eval_results = []
    eval_labels = []
    
    for traj in eval_trajectories:
        try:
            result = system.predict_continuation_probability(
                developer=traj['developer_info'],
                activity_history=traj['activity_history']
            )
            eval_results.append(result['continuation_probability'])
            
            # 最後のstep_labelを評価ラベルとして使用
            if len(traj['step_labels']) > 0:
                eval_labels.append(traj['step_labels'][-1])
            else:
                eval_labels.append(0)
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"評価軌跡スキップ: {e}")
            continue
    
    # メトリクス計算
    eval_probs = np.array(eval_results)
    eval_labels = np.array(eval_labels)
    
    if len(set(eval_labels)) > 1:
        from sklearn.metrics import auc as calc_auc
        
        auc_roc = roc_auc_score(eval_labels, eval_probs)
        precision, recall, _ = precision_recall_curve(eval_labels, eval_probs)
        auc_pr = calc_auc(recall, precision)
        
        # 訓練データで決めた閾値を使用
        eval_preds = (eval_probs >= optimal_threshold).astype(int)
        precision_val = precision_score(eval_labels, eval_preds, zero_division=0)
        recall_val = recall_score(eval_labels, eval_preds, zero_division=0)
        f1_val = f1_score(eval_labels, eval_preds, zero_division=0)
    else:
        auc_roc = 0.0
        auc_pr = 0.0
        precision_val = 0.0
        recall_val = 0.0
        f1_val = 0.0
    
    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'threshold': float(optimal_threshold),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1_score': float(f1_val),
        'positive_count': int(eval_labels.sum()),
        'negative_count': int((1 - eval_labels).sum()),
        'total_count': len(eval_labels)
    }
    
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"AUC-PR: {auc_pr:.4f}")
    logger.info(f"F1: {f1_val:.4f}")
    logger.info(f"閾値（訓練データで決定）: {optimal_threshold:.4f}")
    
    # 保存
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル保存
        system.save_model(str(output_dir / "enhanced_irl_model.pt"))
        
        # メトリクス保存
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✅ 保存完了: {output_dir}")
    
    return metrics


def evaluate_with_existing_model(
    model_path: Path,
    eval_trajectories: List[Dict],
    output_dir: Path = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    既存モデルで評価のみ実施
    
    Args:
        model_path: モデルファイルパス
        eval_trajectories: 評価軌跡
        output_dir: 出力ディレクトリ
        threshold: 分類閾値（訓練データで決定されたもの）
    """

    logger.info("=" * 80)
    logger.info("Enhanced IRL (Attention) 評価のみ")
    logger.info("=" * 80)
    logger.info(f"モデル: {model_path}")
    logger.info(f"閾値: {threshold:.4f}")

    # モデル読み込み
    system = RetentionIRLSystem.load_model(str(model_path))

    # 評価
    eval_results = []
    eval_labels = []

    logger.info(f"評価軌跡: {len(eval_trajectories)}")

    for traj in eval_trajectories:
        try:
            result = system.predict_continuation_probability(
                developer=traj['developer_info'],
                activity_history=traj['activity_history']
            )
            eval_results.append(result['continuation_probability'])

            # 最後のstep_labelを評価ラベルとして使用
            if len(traj['step_labels']) > 0:
                eval_labels.append(traj['step_labels'][-1])
            else:
                eval_labels.append(0)
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"評価軌跡スキップ: {e}")
            continue

    # メトリクス計算
    eval_probs = np.array(eval_results)
    eval_labels = np.array(eval_labels)

    if len(set(eval_labels)) > 1:
        from sklearn.metrics import auc as calc_auc
        from sklearn.metrics import precision_recall_curve
        
        auc_roc = roc_auc_score(eval_labels, eval_probs)
        precision, recall, _ = precision_recall_curve(eval_labels, eval_probs)
        auc_pr = calc_auc(recall, precision)

        # 引数で渡された閾値を使用（訓練データで決定済み）
        eval_preds = (eval_probs >= threshold).astype(int)
        precision_val = precision_score(eval_labels, eval_preds, zero_division=0)
        recall_val = recall_score(eval_labels, eval_preds, zero_division=0)
        f1_val = f1_score(eval_labels, eval_preds, zero_division=0)
    else:
        auc_roc = 0.0
        auc_pr = 0.0
        precision_val = 0.0
        recall_val = 0.0
        f1_val = 0.0

    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'threshold': float(threshold),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1_score': float(f1_val),
        'positive_count': int(eval_labels.sum()),
        'negative_count': int((1 - eval_labels).sum()),
        'total_count': len(eval_labels)
    }

    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"AUC-PR: {auc_pr:.4f}")
    logger.info(f"F1: {f1_val:.4f}")
    logger.info(f"閾値（訓練時決定）: {threshold:.4f}")

    # 保存
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # メトリクス保存
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
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--project", default="openstack/nova", help="プロジェクト名")
    parser.add_argument("--model", type=str, default=None, help="既存モデルのパス（評価のみの場合）")

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

        # スナップショット評価データ準備
        eval_trajectories = prepare_snapshot_evaluation_trajectories(
            df, eval_start,
            args.future_window_start, args.future_window_end,
            args.min_history_events, args.project
        )

        # モデルで評価
        model_path = Path(args.model)
        metrics = evaluate_with_existing_model(model_path, eval_trajectories, output_dir)
    else:
        logger.info("訓練と評価を実施")

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

        # Enhanced IRL訓練
        metrics = train_enhanced_irl(
            train_trajectories, eval_trajectories,
            args.epochs, output_dir
        )

    logger.info("=" * 80)
    logger.info("完了!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
