#!/usr/bin/env python3
"""
拡張IRL スライディングウィンドウ版訓練・評価スクリプト

train_irl_sliding_window.py の拡張IRL版
- 状態特徴量: 32次元
- 行動特徴量: 9次元
- スライディングウィンドウラベル（例: 0-3m, 3-6m, 6-9m, 9-12m）
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

from gerrit_retention.rl_prediction.enhanced_retention_irl_system import (
    EnhancedRetentionIRLSystem,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_review_logs(csv_path: str) -> pd.DataFrame:
    """レビューログをCSVから読み込み"""
    logger.info(f"レビューログを読み込み中: {csv_path}")
    df = pd.read_csv(csv_path)
    df['request_time'] = pd.to_datetime(df['request_time'])
    logger.info(f"レビューログ読み込み完了: {len(df)}件")
    logger.info(f"期間: {df['request_time'].min()} ～ {df['request_time'].max()}")
    return df


# def extract_sliding_window_trajectories(
#     df: pd.DataFrame,
#     train_start: pd.Timestamp,
#     train_end: pd.Timestamp,
#     future_window_start_months: int = 0,
#     future_window_end_months: int = 3,
#     min_history_events: int = 3,
#     reviewer_col: str = 'reviewer_email',
#     date_col: str = 'request_time',
#     project: str = None
# ) -> List[Dict[str, Any]]:
#     """
#     スライディングウィンドウ版：全シーケンス＋月次集約ラベル付き軌跡を抽出（拡張IRL用）
#     
#     train_irl_sliding_window.py の extract_sliding_window_trajectories と同じロジック
#     """
#    logger.info("=" * 80)
#    logger.info("拡張IRL スライディングウィンドウ版：全シーケンス＋月次集約ラベル付き軌跡抽出を開始")
#    logger.info(f"学習期間: {train_start} ～ {train_end}")
#    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月（スライディング）")
#    if project:
#        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
#    else:
#        logger.info("プロジェクト: 全プロジェクト")
#    logger.info("=" * 80)
#    
#    trajectories = []
#    
#    # プロジェクトフィルタを適用
#    if project and 'project' in df.columns:
#        df = df[df['project'] == project].copy()
#        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")
#    
#    # 評価期間内のデータを取得（評価時は評価期間を使用）
#    eval_end = pd.Timestamp('2024-04-01')  # 評価期間の終了日
#    train_df = df[(df[date_col] >= train_start) & (df[date_col] < eval_end)]
#    
#    # 全レビュアーを取得
#    all_reviewers = train_df[reviewer_col].unique()
#    logger.info(f"レビュアー数: {len(all_reviewers)}")
#    
#    reviewer_continuation_count = 0
#    
#    # 各レビュアーについて1サンプルを生成
#    for idx, reviewer in enumerate(all_reviewers):
#        if (idx + 1) % 100 == 0:
#            logger.info(f"処理中: {idx+1}/{len(all_reviewers)}")
#        
#        # このレビュアーの学習期間内の全活動
#        reviewer_history = train_df[train_df[reviewer_col] == reviewer]
#        
#        # 最小イベント数を満たさない場合はスキップ
#        if len(reviewer_history) < min_history_events:
#            continue
#        
#        # 活動履歴を時系列順に並べる
#        reviewer_history_sorted = reviewer_history.sort_values(date_col)
#        
#        # このレビュアーの全活動（学習期間外も含む）を時系列順に取得
#        reviewer_all_activities = df[df[reviewer_col] == reviewer].sort_values(date_col)
#        
#        # 履歴期間内で活動しているプロジェクトを取得
#        history_projects = set(reviewer_history_sorted['project'].dropna().unique())
#        
#        # 月ごとにラベルを計算
#        monthly_labels = {}
#        
#        for _, row in reviewer_history_sorted.iterrows():
#            activity_date = pd.Timestamp(row[date_col])
#            month_key = (activity_date.year, activity_date.month)
#            
#            if month_key not in monthly_labels:
#                # この月の最終日
#                month_end = (activity_date + pd.offsets.MonthEnd(0))
#                
#                # 将来窓の範囲（スライディング）
#                future_start = month_end + pd.DateOffset(months=future_window_start_months)
#                future_end = month_end + pd.DateOffset(months=future_window_end_months)
#                
#                # 将来窓が評価期間を超える場合はNone（学習期間ではなく評価期間で判定）
#                eval_end = pd.Timestamp('2024-04-01')  # 評価期間の終了日
#                if future_end > eval_end:
#                    monthly_labels[month_key] = None
#                else:
#                    # この月からスライディング窓内に活動があるか（履歴期間内のプロジェクトのみ）
#                    future_activities = reviewer_all_activities[
#                        (reviewer_all_activities[date_col] >= future_start) &
#                        (reviewer_all_activities[date_col] < future_end) &
#                        (reviewer_all_activities['project'].isin(history_projects))
#                    ]
#                    monthly_labels[month_key] = len(future_activities) > 0
#        
#        # 活動履歴を構築
#        activity_history = []
#        step_labels = []
#        
#        for _, row in reviewer_history_sorted.iterrows():
#            activity_date = pd.Timestamp(row[date_col])
#            month_key = (activity_date.year, activity_date.month)
#            
#            # この月のラベルがNoneの場合はスキップ
#            if monthly_labels[month_key] is None:
#                continue
#            
#            activity = {
#                'timestamp': row[date_col],
#                'action_type': 'review',
#                'project': row.get('project', 'unknown'),
#            }
#            activity_history.append(activity)
#            step_labels.append(monthly_labels[month_key])
#        
#        # ラベルがない場合はスキップ
#        if not step_labels:
#            continue
#        
#        # レビュアー単位の継続判定（最終月のラベル）
#        final_month_label = step_labels[-1]
#        if final_month_label:
#            reviewer_continuation_count += 1
#        
#        # 軌跡を作成
#        developer_info = {
#            'developer_email': reviewer
#        }
#        
#        trajectory = {
#            'developer_info': developer_info,
#            'activity_history': activity_history,
#            'context_date': train_end,  # 固定時点
#            'step_labels': step_labels,
#            'seq_len': len(step_labels)
#        }
#        
#        trajectories.append(trajectory)
#    
#    logger.info("=" * 80)
#    logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル（レビュアー）")
#    if trajectories:
#        total_steps = sum(t['seq_len'] for t in trajectories)
#        continued_steps = sum(sum(t['step_labels']) for t in trajectories)
#        logger.info(f"  総ステップ数: {total_steps}")
#        logger.info(f"  継続ステップ率: {continued_steps/total_steps*100:.1f}% ({continued_steps}/{total_steps})")
#        logger.info(f"  レビュアー単位継続率: {reviewer_continuation_count/len(trajectories)*100:.1f}% ({reviewer_continuation_count}/{len(trajectories)})")
#    logger.info("=" * 80)
#    
##     return trajectories


def train_irl_model_multi_step(
    trajectories: List[Dict[str, Any]],
    config: Dict[str, Any],
    epochs: int = 30
) -> EnhancedRetentionIRLSystem:
    """
    各ステップラベル付き拡張IRLモデルを訓練
    """
    logger.info("=" * 80)
    logger.info("拡張IRL訓練開始")
    logger.info(f"軌跡数: {len(trajectories)}")
    logger.info(f"エポック数: {epochs}")
    logger.info("=" * 80)
    
    # IRLシステムの初期化
    irl_system = EnhancedRetentionIRLSystem(config)
    
    # 訓練
    result = irl_system.train_irl_multi_step_labels(
        expert_trajectories=trajectories,
        epochs=epochs
    )
    
    logger.info("=" * 80)
    logger.info(f"訓練完了: 最終損失 = {result['final_loss']:.4f}")
    logger.info("=" * 80)
    
    return irl_system


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """最適な閾値を見つける（F1スコアを最大化）"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return {
        'threshold': float(best_threshold),
        'f1': float(f1_scores[best_idx]),
        'precision': float(precision[best_idx]),
        'recall': float(recall[best_idx])
    }


def evaluate_model(
    irl_system: EnhancedRetentionIRLSystem,
    trajectories: List[Dict[str, Any]],
    optimal_threshold: float = 0.5
) -> Dict[str, Any]:
    """モデルを評価"""
    logger.info(f"評価開始: {len(trajectories)}サンプル")
    
    y_true = []
    y_pred = []
    
    for traj in trajectories:
        developer = traj.get('developer_info', traj.get('developer', {}))
        activity_history = traj['activity_history']
        context_date = traj['context_date']
        step_labels = traj.get('step_labels', [])
        
        if not activity_history:
            continue
        
        # cutoff評価では future_contribution を使用
        future_contribution = traj.get('future_contribution', False)
        
        # 予測
        try:
            result = irl_system.predict_continuation_probability(
                developer=developer,
                activity_history=activity_history,
                context_date=context_date
            )
            prob = result['continuation_probability']
            
            y_true.append(1 if future_contribution else 0)
            y_pred.append(prob)
            
        except Exception as e:
            logger.warning(f"予測エラー: {e}")
            continue
    
    if len(y_true) == 0:
        logger.warning("評価サンプルが0件です")
        return {
            'auc_roc': 0.0,
            'auc_pr': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'optimal_threshold': 0.5,
            'sample_count': 0,
            'continuation_rate': 0.0
        }
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # メトリクス計算
    auc_roc = roc_auc_score(y_true, y_pred)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    
    # 閾値適用
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred_binary)
    precision_val = precision_score(y_true, y_pred_binary, zero_division=0)
    recall_val = recall_score(y_true, y_pred_binary, zero_division=0)
    
    continuation_rate = np.mean(y_true)
    
    logger.info(f"評価完了: AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}, F1={f1:.3f}, Precision={precision_val:.3f}, Recall={recall_val:.3f}")
    logger.info(f"継続率: {continuation_rate:.1%} ({np.sum(y_true)}/{len(y_true)})")
    
    return {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'f1': float(f1),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'optimal_threshold': float(optimal_threshold),
        'sample_count': int(len(y_true)),
        'continuation_rate': float(continuation_rate)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviews', required=True, help='レビューCSVファイル')
    parser.add_argument('--train-start', required=True, help='訓練開始日')
    parser.add_argument('--train-end', required=True, help='訓練終了日')
    parser.add_argument('--eval-start', required=True, help='評価開始日')
    parser.add_argument('--eval-end', required=True, help='評価終了日')
    parser.add_argument('--future-window-start', type=int, required=True)
    parser.add_argument('--future-window-end', type=int, required=True)
    parser.add_argument('--eval-future-window-start', type=int, default=None)
    parser.add_argument('--eval-future-window-end', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--min-history-events', type=int, default=3)
    parser.add_argument('--output', default='enhanced_irl_model.pt')
    parser.add_argument('--project', default=None, help='プロジェクト名（単一プロジェクト用）')
    parser.add_argument('--model', default=None, help='既存モデルのパス（評価のみ）')
    
    args = parser.parse_args()
    
    # 日付変換
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    
    # 評価用の将来窓（指定がなければ訓練と同じ）
    eval_future_start = args.eval_future_window_start if args.eval_future_window_start is not None else args.future_window_start
    eval_future_end = args.eval_future_window_end if args.eval_future_window_end is not None else args.future_window_end
    
    logger.info("=" * 80)
    logger.info("拡張IRL スライディングウィンドウ訓練・評価")
    logger.info("=" * 80)
    logger.info(f"レビューデータ: {args.reviews}")
    logger.info(f"学習期間: {train_start} ～ {train_end}")
    logger.info(f"評価期間: {eval_start} ～ {eval_end}")
    logger.info(f"訓練ラベル: {args.future_window_start}-{args.future_window_end}m")
    logger.info(f"評価期間: {eval_future_start}-{eval_future_end}m")
    if args.project:
        logger.info(f"プロジェクト: {args.project}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # モデル設定（拡張IRL）
    config = {
        'state_dim': 32,   # 拡張特徴量
        'action_dim': 9,   # 拡張特徴量
        'hidden_dim': 128,
        'learning_rate': 0.0001,
        'sequence': True,
        'seq_len': 0,  # 可変長
    }
    logger.info(f"🔧 Config: state_dim={config['state_dim']}, action_dim={config['action_dim']}")
    
    # モデルの訓練または読み込み
    if args.model and Path(args.model).exists():
        # 既存モデルをロード
        logger.info(f"既存モデルをロード: {args.model}")
        irl_system = EnhancedRetentionIRLSystem.load_model(args.model)
        logger.info("モデルロード完了")
    else:
        # 訓練データ抽出（cutoff時点での評価）
        from train_irl_within_training_period import (
            extract_cutoff_evaluation_trajectories,
        )
        train_trajectories = extract_cutoff_evaluation_trajectories(
            df,
            cutoff_date=train_end,
            history_window_months=12,
            future_window_start_months=args.future_window_start,
            future_window_end_months=args.future_window_end,
            min_history_events=args.min_history_events,
            project=args.project,
        )
        
        if len(train_trajectories) == 0:
            logger.error("訓練データが0件です")
            return
        
        # モデル訓練
        irl_system = train_irl_model_multi_step(
            trajectories=train_trajectories,
            config=config,
            epochs=args.epochs
        )
        
        # モデル保存
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        irl_system.save_model(str(output_path))
        logger.info(f"モデル保存: {output_path}")
    
    # 評価データ抽出（cutoff時点での評価）
    from train_irl_within_training_period import extract_cutoff_evaluation_trajectories
    eval_trajectories = extract_cutoff_evaluation_trajectories(
        df,
        cutoff_date=eval_start,
        history_window_months=12,
        future_window_start_months=eval_future_start,
        future_window_end_months=eval_future_end,
        min_history_events=args.min_history_events,
        project=args.project,
    )
    
    if len(eval_trajectories) == 0:
        logger.warning("評価データが0件です")
        return
    
    # 最適閾値を見つける
    logger.info("最適閾値を計算中...")
    y_true_thresh = []
    y_pred_thresh = []
    
    for traj in eval_trajectories:
        try:
            result = irl_system.predict_continuation_probability(
                developer=traj['developer_info'],
                activity_history=traj['activity_history'],
                context_date=traj['context_date']
            )
            y_true_thresh.append(1 if traj['step_labels'][-1] else 0)
            y_pred_thresh.append(result['continuation_probability'])
        except:
            continue
    
    if len(y_true_thresh) > 0:
        threshold_info = find_optimal_threshold(np.array(y_true_thresh), np.array(y_pred_thresh))
        optimal_threshold = threshold_info['threshold']
        logger.info(f"最適閾値: {optimal_threshold:.3f} (F1={threshold_info['f1']:.3f})")
    else:
        optimal_threshold = 0.5
    
    # 評価
    metrics = evaluate_model(irl_system, eval_trajectories, optimal_threshold)
    
    # 結果保存
    output_path = Path(args.output)
    if output_path.suffix == '.pt':
        metrics_path = output_path.parent / 'metrics.json'
    else:
        metrics_path = Path(args.output).with_suffix('.json') if not args.output.endswith('.json') else Path(args.output)
    
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"メトリクス保存: {metrics_path}")
    logger.info("=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


