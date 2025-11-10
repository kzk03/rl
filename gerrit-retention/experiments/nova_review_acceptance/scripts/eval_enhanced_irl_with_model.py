#!/usr/bin/env python3
"""
Enhanced IRL (Attention) - 既存モデルで評価のみ実施

訓練済みモデルを読み込んで、指定された評価期間でメトリクスを計算
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
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history_requests: int = 3,
    project: str = None
) -> List[Dict[str, Any]]:
    """
    importantsと同じデータ準備方式（評価用）
    """
    if project:
        df = df[df['project'] == project].copy()

    date_col = 'request_time'
    reviewer_col = 'reviewer_email'
    label_col = 'label'

    df[date_col] = pd.to_datetime(df[date_col])

    # 評価期間のデータ
    eval_df = df[(df[date_col] >= eval_start) & (df[date_col] < eval_end)]
    reviewers = eval_df[reviewer_col].unique()

    trajectories = []
    skipped_min_requests = 0

    logger.info(f"評価期間: {eval_start} ~ {eval_end}")
    logger.info(f"Future Window: {future_window_start_months} ~ {future_window_end_months}ヶ月")
    logger.info(f"レビュワー候補: {len(reviewers)}")

    for reviewer in reviewers:
        reviewer_history = eval_df[eval_df[reviewer_col] == reviewer]

        # 最小依頼数チェック
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue

        # 月次の軌跡生成
        history_months = pd.date_range(
            start=eval_start,
            end=eval_end,
            freq='MS'  # 月初
        )

        step_labels = []
        monthly_activity_histories = []

        for month_start in history_months[:-1]:
            month_end = month_start + pd.DateOffset(months=1)

            # この月からfuture_window後のラベル計算期間
            future_start = month_end + pd.DateOffset(months=future_window_start_months)
            future_end = month_end + pd.DateOffset(months=future_window_end_months)

            # eval_endでクリップ（データリーク防止）
            if future_end > eval_end:
                future_end = eval_end

            # eval_endを超える場合はこの月のラベルは作成しない
            if future_start >= eval_end:
                continue

            # 将来期間のレビュー依頼
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

            # この月時点までの活動履歴
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
            'context_date': eval_end,
            'step_labels': step_labels,
            'seq_len': len(step_labels),
        }

        trajectories.append(trajectory)

    logger.info(f"軌跡生成完了: {len(trajectories)}サンプル")
    logger.info(f"スキップ（最小依頼数未満）: {skipped_min_requests}")

    return trajectories


def evaluate_with_model(
    model_path: Path,
    eval_trajectories: List[Dict],
    output_dir: Path = None
) -> Dict[str, float]:
    """既存モデルで評価"""

    logger.info("=" * 80)
    logger.info("Enhanced IRL (Attention) 評価")
    logger.info("=" * 80)
    logger.info(f"モデル: {model_path}")

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
            # エラーが発生した軌跡はスキップ
            logger.warning(f"評価軌跡スキップ: {e}")
            continue

    # メトリクス計算
    eval_probs = np.array(eval_results)
    eval_labels = np.array(eval_labels)

    if len(set(eval_labels)) > 1:
        auc_roc = roc_auc_score(eval_labels, eval_probs)

        # 最適閾値を探索
        from sklearn.metrics import auc as calc_auc
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(eval_labels, eval_probs)
        auc_pr = calc_auc(recall, precision)

        # F1最大の閾値
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        eval_preds = (eval_probs >= optimal_threshold).astype(int)
        precision_val = precision_score(eval_labels, eval_preds, zero_division=0)
        recall_val = recall_score(eval_labels, eval_preds, zero_division=0)
        f1_val = f1_score(eval_labels, eval_preds, zero_division=0)
    else:
        auc_roc = 0.0
        auc_pr = 0.0
        optimal_threshold = 0.5
        precision_val = 0.0
        recall_val = 0.0
        f1_val = 0.0

    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'optimal_threshold': float(optimal_threshold),
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
    parser.add_argument("--model", required=True, help="訓練済みモデルのパス")
    parser.add_argument("--reviews", required=True, help="レビューデータCSV")
    parser.add_argument("--eval-start", required=True, help="評価開始日 (YYYY-MM-DD)")
    parser.add_argument("--eval-end", required=True, help="評価終了日 (YYYY-MM-DD)")
    parser.add_argument("--future-window-start", type=int, required=True, help="Future Window開始(月)")
    parser.add_argument("--future-window-end", type=int, required=True, help="Future Window終了(月)")
    parser.add_argument("--min-history-events", type=int, default=3, help="最小履歴イベント数")
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--project", default="openstack/nova", help="プロジェクト名")

    args = parser.parse_args()

    # データ読み込み
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])

    eval_start = pd.to_datetime(args.eval_start)
    eval_end = pd.to_datetime(args.eval_end)

    # 評価データ準備
    eval_trajectories = prepare_trajectories_importants_style(
        df, eval_start, eval_end,
        args.future_window_start, args.future_window_end,
        args.min_history_events, args.project
    )

    # モデルで評価
    model_path = Path(args.model)
    output_dir = Path(args.output)
    metrics = evaluate_with_model(model_path, eval_trajectories, output_dir)

    logger.info("=" * 80)
    logger.info("完了!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
