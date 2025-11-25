#!/usr/bin/env python3
"""
通常IRLモデルのスナップショット時点評価

学習済みの通常IRLモデルを使用して、スナップショット時点での特徴量で評価を行う。
拡張IRLと同様の評価方法を適用する。
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem
from gerrit_retention.rl_prediction.snapshot_features import (
    compute_snapshot_action_features,
    compute_snapshot_state_features,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_review_logs(csv_path: str) -> pd.DataFrame:
    """レビューログを読み込み"""
    logger.info(f"レビューログを読み込み中: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 日付列を変換
    if 'request_time' in df.columns:
        df['request_time'] = pd.to_datetime(df['request_time'])
        df['timestamp'] = df['request_time']
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"読み込み完了: {len(df)}件")
    return df


def extract_snapshot_evaluation_trajectories(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    history_window_months: int = 12,
    future_window_start_months: int = 0,
    future_window_end_months: int = 3,
    min_history_events: int = 3,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time',
    project: str = None
) -> List[Dict[str, Any]]:
    """
    スナップショット時点での評価用軌跡を抽出（通常IRL用）
    
    Args:
        df: レビューデータ
        cutoff_date: カットオフ日（スナップショット時点）
        history_window_months: 履歴期間（月数）
        future_window_start_months: 将来窓開始（月数）
        future_window_end_months: 将来窓終了（月数）
        min_history_events: 最小活動数
        reviewer_col: レビュアー列名
        date_col: 日付列名
        project: プロジェクト名（指定時は単一プロジェクトのみ）
    
    Returns:
        各レビュアーの軌跡のリスト
    """
    logger.info("=" * 80)
    logger.info("通常IRL スナップショット時点評価用軌跡抽出を開始")
    logger.info(f"カットオフ日: {cutoff_date}")
    logger.info(f"履歴期間: {history_window_months}ヶ月")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project}")
    logger.info("=" * 80)
    
    # プロジェクトフィルタを適用
    if project and 'project' in df.columns:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")
    
    # 履歴期間のデータを取得
    history_start = cutoff_date - pd.DateOffset(months=history_window_months)
    history_df = df[(df[date_col] >= history_start) & (df[date_col] < cutoff_date)]
    
    # 将来期間のデータを取得
    future_start = cutoff_date + pd.DateOffset(months=future_window_start_months)
    future_end = cutoff_date + pd.DateOffset(months=future_window_end_months)
    future_df = df[(df[date_col] >= future_start) & (df[date_col] < future_end)]
    
    # 全レビュアーを取得
    all_reviewers = history_df[reviewer_col].unique()
    logger.info(f"レビュアー数: {len(all_reviewers)}")
    
    trajectories = []
    
    for idx, reviewer in enumerate(all_reviewers):
        if (idx + 1) % 100 == 0:
            logger.info(f"処理中: {idx+1}/{len(all_reviewers)}")
        
        # このレビュアーの履歴期間内の活動
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]
        
        # 最小イベント数を満たさない場合はスキップ
        if len(reviewer_history) < min_history_events:
            continue
        
        # このレビュアーの将来期間内の活動
        reviewer_future = future_df[future_df[reviewer_col] == reviewer]
        
        # 将来の貢献を判定
        future_contribution = len(reviewer_future) > 0
        
        # 開発者情報を構築
        developer_info = {
            'developer_email': reviewer,
            'first_seen': reviewer_history[date_col].min().isoformat(),
            'changes_authored': len(reviewer_history),
            'changes_reviewed': 0,  # 通常IRLでは簡略化
        }
        
        # 活動履歴を構築（スナップショット時点まで）
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col].isoformat(),
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'message': row.get('message', ''),
                'lines_added': row.get('lines_added', 0),
                'lines_deleted': row.get('lines_deleted', 0),
                'files_changed': row.get('files_changed', 1),
            }
            activity_history.append(activity)
        
        # 軌跡を作成
        trajectory = {
            'developer': developer_info,
            'activity_history': activity_history,
            'context_date': cutoff_date,
            'future_contribution': future_contribution,
            'sampling_point': cutoff_date,
        }
        
        trajectories.append(trajectory)
    
    # 統計情報
    positive_count = sum(1 for t in trajectories if t['future_contribution'])
    positive_rate = positive_count / len(trajectories) if trajectories else 0
    
    logger.info("=" * 80)
    logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル")
    logger.info(f"  継続率: {positive_rate:.1%} ({positive_count}/{len(trajectories)})")
    logger.info("=" * 80)
    
    return trajectories


def evaluate_normal_irl_with_snapshot_features(
    irl_system: RetentionIRLSystem,
    trajectories: List[Dict[str, Any]],
    cutoff_date: pd.Timestamp,
    history_window_months: int = 12
) -> Dict[str, Any]:
    """
    通常IRLモデルをスナップショット特徴量で評価
    
    Args:
        irl_system: 学習済み通常IRLシステム
        trajectories: 評価用軌跡データ
        cutoff_date: スナップショット日
        history_window_months: 履歴期間
    
    Returns:
        評価メトリクス
    """
    logger.info("=" * 80)
    logger.info("通常IRLモデル スナップショット特徴量評価開始")
    logger.info(f"評価サンプル数: {len(trajectories)}")
    logger.info("=" * 80)
    
    predictions = []
    true_labels = []
    
    for trajectory in trajectories:
        reviewer_email = trajectory['developer']['developer_email']
        
        # 通常IRLモデルで予測（スナップショット時点での軌跡データを使用）
        try:
            result = irl_system.predict_continuation_probability(
                developer=trajectory['developer'],
                activity_history=trajectory['activity_history'],
                context_date=cutoff_date.to_pydatetime()
            )
            
            prob = result.get('continuation_probability', 0.5)
            predictions.append(prob)
            true_labels.append(trajectory['future_contribution'])
            
        except Exception as e:
            logger.warning(f"予測エラー ({reviewer_email}): {e}")
            predictions.append(0.5)
            true_labels.append(trajectory['future_contribution'])
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # 継続率
    continuation_rate = np.mean(true_labels)
    logger.info(f"評価データの継続率: {continuation_rate:.1%}")
    
    # メトリクス計算
    if len(np.unique(true_labels)) < 2:
        logger.warning("評価データに正例または負例のみが含まれています")
        return {
            'auc_roc': 0.5,
            'auc_pr': continuation_rate,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'optimal_threshold': 0.5,
            'continuation_rate': continuation_rate,
            'sample_count': len(trajectories)
        }
    
    # AUC-ROC
    auc_roc = roc_auc_score(true_labels, predictions)
    
    # Precision-Recall曲線とAUC-PR
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, predictions)
    auc_pr = auc(recall_vals, precision_vals)
    
    # 最適閾値を見つける
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    for threshold in thresholds:
        y_pred_binary = (predictions >= threshold).astype(int)
        if len(np.unique(y_pred_binary)) > 1:
            f1 = f1_score(true_labels, y_pred_binary)
            precision = precision_score(true_labels, y_pred_binary, zero_division=0)
            recall = recall_score(true_labels, y_pred_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
    
    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'precision': float(best_precision),
        'recall': float(best_recall),
        'f1': float(best_f1),
        'optimal_threshold': float(best_threshold),
        'continuation_rate': float(continuation_rate),
        'sample_count': len(trajectories)
    }
    
    logger.info("=" * 80)
    logger.info("評価結果:")
    logger.info(f"  AUC-ROC: {auc_roc:.3f}")
    logger.info(f"  AUC-PR: {auc_pr:.3f}")
    logger.info(f"  最適閾値: {best_threshold:.3f}")
    logger.info(f"  Precision: {best_precision:.3f}")
    logger.info(f"  Recall: {best_recall:.3f}")
    logger.info(f"  F1スコア: {best_f1:.3f}")
    logger.info("=" * 80)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='通常IRLモデルのスナップショット時点評価')
    parser.add_argument('--model', type=str, required=True, help='学習済みモデルパス')
    parser.add_argument('--reviews', type=str, required=True, help='レビューデータCSV')
    parser.add_argument('--cutoff-date', type=str, required=True, help='カットオフ日 (YYYY-MM-DD)')
    parser.add_argument('--history-window', type=int, default=12, help='履歴期間（月数）')
    parser.add_argument('--future-window-start', type=int, default=0, help='将来窓開始（月数）')
    parser.add_argument('--future-window-end', type=int, default=3, help='将来窓終了（月数）')
    parser.add_argument('--min-history-events', type=int, default=3, help='最小活動数')
    parser.add_argument('--project', type=str, help='プロジェクト名（オプション）')
    parser.add_argument('--output', type=str, required=True, help='出力ファイルパス')
    
    args = parser.parse_args()
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # カットオフ日を変換
    cutoff_date = pd.Timestamp(args.cutoff_date)
    
    # 学習済みモデルを読み込み
    logger.info(f"学習済みモデルを読み込み: {args.model}")
    
    # モデルファイルから設定を読み取り
    checkpoint = torch.load(args.model, map_location='cpu')
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        model_config = checkpoint['config']
        logger.info(f"モデル設定: state_dim={model_config.get('state_dim')}, action_dim={model_config.get('action_dim')}")
    else:
        # デフォルト設定（古いモデル用）
        model_config = {
            'state_dim': 10,
            'action_dim': 5,
            'hidden_dim': 128,
            'learning_rate': 0.0001,
            'sequence': True,
            'seq_len': 0,
        }
        logger.info(f"デフォルト設定を使用: state_dim={model_config['state_dim']}, action_dim={model_config['action_dim']}")
    
    irl_system = RetentionIRLSystem.load_model(args.model)
    
    # 評価用軌跡を抽出
    trajectories = extract_snapshot_evaluation_trajectories(
        df,
        cutoff_date=cutoff_date,
        history_window_months=args.history_window,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        min_history_events=args.min_history_events,
        project=args.project,
    )
    
    if not trajectories:
        logger.error("評価データがありません")
        return
    
    # スナップショット特徴量で評価
    metrics = evaluate_normal_irl_with_snapshot_features(
        irl_system,
        trajectories,
        cutoff_date,
        args.history_window
    )
    
    # 結果保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"評価結果を保存: {output_path}")


if __name__ == "__main__":
    main()
