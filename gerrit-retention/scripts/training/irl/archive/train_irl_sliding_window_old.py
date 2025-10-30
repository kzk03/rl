#!/usr/bin/env python3
"""
スライディングウィンドウIRL学習

訓練ラベルを独立したスライディングウィンドウで定義:
- 0-3m: 0～3ヶ月後に活動
- 3-6m: 3～6ヶ月後に活動（0-3mは除く）
- 6-9m: 6～9ヶ月後に活動（0-6mは除く）
- 9-12m: 9～12ヶ月後に活動（0-9mは除く）

これにより、各訓練ラベルが異なる時間スケールを学習します。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
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
    find_optimal_threshold,
    load_review_logs,
)

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_sliding_window_trajectories(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_window_start_months: int = 0,
    future_window_end_months: int = 3,
    min_history_events: int = 3,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time',
    project: str = None
) -> List[Dict[str, Any]]:
    """
    スライディングウィンドウ版：時系列訓練＋スナップショット予測軌跡を抽出
    
    特徴：
    - **訓練**：時系列データで学習（各ステップごとの特徴量）
    - **予測**：スナップショット特徴量で予測（特定時点での集約特徴量）
    - **ラベル**：月次集約ラベル（各月末からスライディングウィンドウ内に活動があるか）
    - **継続率**：レビュアー単位（最終月のラベルで判定）
    
    Args:
        df: レビューデータ
        train_start: 学習開始日
        train_end: 学習終了日
        future_window_start_months: 将来窓開始（月数）
        future_window_end_months: 将来窓終了（月数）
        min_history_events: 最小活動数
        reviewer_col: レビュアー列名
        date_col: 日付列名
        project: プロジェクト名（指定時は単一プロジェクトのみ）
    
    Returns:
        各レビュアーの軌跡のリスト（1レビュアー=1サンプル）
    """
    logger.info("=" * 80)
    logger.info("スライディングウィンドウ版：時系列訓練＋スナップショット予測軌跡抽出を開始")
    logger.info(f"学習期間: {train_start} ～ {train_end}")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月（スライディング）")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    logger.info("訓練: 時系列データ（各ステップごとの特徴量）")
    logger.info("予測: スナップショット特徴量（特定時点での集約特徴量）")
    logger.info("ラベル: 月次集約ラベル（各月末からスライディングウィンドウ内に活動があるか）")
    logger.info("継続判定: 履歴期間内のプロジェクトでの継続のみをカウント")
    if future_window_start_months > 0:
        logger.info(f"⚠️  スライディングウィンドウ: {future_window_start_months}ヶ月以内の活動は除外")
    logger.info("=" * 80)
    
    trajectories = []
    
    # プロジェクトフィルタを適用
    if project and 'project' in df.columns:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")
    
    # 学習期間内のデータを取得
    train_df = df[(df[date_col] >= train_start) & (df[date_col] < train_end)]
    
    # 全レビュアーを取得
    all_reviewers = train_df[reviewer_col].unique()
    logger.info(f"レビュアー数: {len(all_reviewers)}")
    
    reviewer_continuation_count = 0
    
    # 各レビュアーについて1サンプルを生成
    for idx, reviewer in enumerate(all_reviewers):
        if (idx + 1) % 100 == 0:
            logger.info(f"処理中: {idx+1}/{len(all_reviewers)}")
        
        # このレビュアーの学習期間内の全活動
        reviewer_history = train_df[train_df[reviewer_col] == reviewer]
        
        # 最小イベント数を満たさない場合はスキップ
        if len(reviewer_history) < min_history_events:
            continue
        
        # 活動履歴を時系列順に並べる
        reviewer_history_sorted = reviewer_history.sort_values(date_col)
        
        # このレビュアーの全活動（学習期間外も含む）を時系列順に取得
        reviewer_all_activities = df[df[reviewer_col] == reviewer].sort_values(date_col)
        
        # 履歴期間内で活動しているプロジェクトを取得
        history_projects = set(reviewer_history_sorted['project'].dropna().unique())
        
        # 月ごとにラベルを計算（月次集約ラベル）
        monthly_labels = {}
        
        for _, row in reviewer_history_sorted.iterrows():
            activity_date = pd.Timestamp(row[date_col])
            month_key = (activity_date.year, activity_date.month)
            
            if month_key not in monthly_labels:
                # この月の最終日
                month_end = (activity_date + pd.offsets.MonthEnd(0))
                
                # 将来窓の範囲（スライディング）
                future_start = month_end + pd.DateOffset(months=future_window_start_months)
                future_end = month_end + pd.DateOffset(months=future_window_end_months)
                
                # 将来窓が学習期間を超える場合はNone
                if future_end > train_end + pd.DateOffset(months=12):  # 最大1年先まで
                    monthly_labels[month_key] = None
                else:
                    # この月からスライディング窓内に活動があるか（履歴期間内のプロジェクトのみ）
                    future_activities = reviewer_all_activities[
                        (reviewer_all_activities[date_col] >= future_start) &
                        (reviewer_all_activities[date_col] < future_end) &
                        (reviewer_all_activities['project'].isin(history_projects))
                    ]
                    monthly_labels[month_key] = len(future_activities) > 0
        
        # 活動履歴を構築（時系列訓練用）
        activity_history = []
        step_labels = []
        
        for _, row in reviewer_history_sorted.iterrows():
            activity_date = pd.Timestamp(row[date_col])
            month_key = (activity_date.year, activity_date.month)
            
            # この月のラベルがNoneの場合はスキップ
            if monthly_labels[month_key] is None:
                continue
            
            activity = {
                'timestamp': row[date_col],
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'request_time': row.get('request_time', row[date_col]),  # レスポンス時間計算用
            }
            activity_history.append(activity)
            step_labels.append(monthly_labels[month_key])
        
        # ラベルがない場合はスキップ
        if not step_labels:
            continue
        
        # レビュアー単位の継続判定（最終月のラベル）
        final_month_label = step_labels[-1]
        if final_month_label:
            reviewer_continuation_count += 1
        
        # 軌跡を作成
        developer_info = {
            'developer_email': reviewer
        }
        
        trajectory = {
            'developer_info': developer_info,
            'activity_history': activity_history,
            'context_date': train_end,  # スナップショット日
            'step_labels': step_labels,  # 月次集約ラベル
            'seq_len': len(step_labels),
            'reviewer': reviewer,
            'history_count': len(reviewer_history)
        }
        
        trajectories.append(trajectory)
    
    logger.info("=" * 80)
    logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル（レビュアー）")
    if trajectories:
        total_steps = sum(t['seq_len'] for t in trajectories)
        continued_steps = sum(sum(t['step_labels']) for t in trajectories)
        logger.info(f"  総ステップ数: {total_steps}")
        logger.info(f"  継続ステップ率: {continued_steps/total_steps*100:.1f}% ({continued_steps}/{total_steps})")
        logger.info(f"  レビュアー単位継続率: {reviewer_continuation_count/len(trajectories)*100:.1f}% ({reviewer_continuation_count}/{len(trajectories)})")
    logger.info("=" * 80)
    
    return trajectories


def extract_cutoff_evaluation_trajectories(
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
    Cutoff時点での評価用軌跡を抽出（スナップショット特徴量用）
    
    Args:
        df: レビューログ
        cutoff_date: Cutoff日（通常は訓練終了日）
        history_window_months: 履歴ウィンドウ（ヶ月）
        future_window_start_months: 将来窓の開始（ヶ月）
        future_window_end_months: 将来窓の終了（ヶ月）
        min_history_events: 最小イベント数
        reviewer_col: レビュアーカラム名
        date_col: 日付カラム名
        project: プロジェクト名（指定時は単一プロジェクトのみ）
    
    Returns:
        軌跡リスト
    """
    logger.info("=" * 80)
    logger.info("Cutoff評価用軌跡抽出を開始（スナップショット特徴量用）")
    logger.info(f"Cutoff日: {cutoff_date}")
    logger.info(f"履歴ウィンドウ: {history_window_months}ヶ月")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    logger.info("継続判定: 履歴期間内のプロジェクトでの継続のみをカウント")
    logger.info("=" * 80)
    
    # プロジェクトフィルタを適用
    if project and 'project' in df.columns:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")
    
    trajectories = []
    
    # 履歴期間
    history_start = cutoff_date - pd.DateOffset(months=history_window_months)
    history_end = cutoff_date
    
    # 将来窓
    future_start = cutoff_date + pd.DateOffset(months=future_window_start_months)
    future_end = cutoff_date + pd.DateOffset(months=future_window_end_months)
    
    logger.info(f"履歴期間: {history_start} ～ {history_end}")
    logger.info(f"将来窓: {future_start} ～ {future_end}")
    
    # 履歴期間のデータ
    history_df = df[
        (df[date_col] >= history_start) &
        (df[date_col] < history_end)
    ]
    
    # 将来期間のデータ
    future_df = df[
        (df[date_col] >= future_start) &
        (df[date_col] < future_end)
    ]
    
    # 履歴期間内に活動があったレビュアーを対象
    active_reviewers = history_df[reviewer_col].unique()
    logger.info(f"履歴期間内の活動レビュアー数: {len(active_reviewers)}")
    
    for reviewer in active_reviewers:
        # 履歴期間の活動
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]
        
        # 最小イベント数を満たさない場合はスキップ
        if len(reviewer_history) < min_history_events:
            continue
        
        # 将来期間の活動
        reviewer_future = future_df[future_df[reviewer_col] == reviewer]
        
        # 継続ラベル
        future_contribution = len(reviewer_future) > 0
        
        # 活動履歴を構築
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col],
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'request_time': row.get('request_time', row[date_col]),  # レスポンス時間計算用
            }
            activity_history.append(activity)
        
        # 開発者情報
        developer_info = {
            'developer_email': reviewer,
            'first_seen': reviewer_history[date_col].min(),
            'changes_authored': 0,
            'changes_reviewed': len(reviewer_history),
            'projects': reviewer_history['project'].unique().tolist() if 'project' in reviewer_history.columns else []
        }
        
        # 軌跡を作成
        trajectory = {
            'developer': developer_info,
            'activity_history': activity_history,
            'context_date': cutoff_date,
            'future_contribution': future_contribution,
            'reviewer': reviewer,
            'history_count': len(reviewer_history),
            'future_count': len(reviewer_future)
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


def train_irl_model_temporal(
    trajectories: List[Dict[str, Any]],
    config: Dict[str, Any],
    epochs: int = 30
) -> RetentionIRLSystem:
    """
    時系列訓練版IRLモデルを訓練
    
    Args:
        trajectories: 時系列軌跡データ
        config: モデル設定
        epochs: エポック数
        
    Returns:
        訓練済みモデル
    """
    logger.info("=" * 80)
    logger.info("時系列訓練版IRLモデル訓練を開始")
    logger.info(f"軌跡数: {len(trajectories)}")
    logger.info(f"エポック数: {epochs}")
    logger.info(f"目標: 時系列データで継続予測を学習")
    logger.info("=" * 80)
    
    # IRLシステムの初期化
    irl_system = RetentionIRLSystem(config)
    
    # 時系列訓練
    result = irl_system.train_irl_temporal_trajectories(
        expert_trajectories=trajectories,
        epochs=epochs
    )
    
    logger.info("=" * 80)
    logger.info(f"訓練完了: 最終損失 = {result['final_loss']:.4f}")
    logger.info("=" * 80)
    
    return irl_system


def evaluate_model_snapshot(
    irl_system: RetentionIRLSystem,
    eval_trajectories: List[Dict[str, Any]]
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    スナップショット特徴量でモデルを評価
    
    Args:
        irl_system: 訓練済みIRLシステム
        eval_trajectories: 評価用軌跡データ
        
    Returns:
        (評価メトリクス, 予測詳細のリスト)
    """
    logger.info("=" * 80)
    logger.info("スナップショット特徴量でモデル評価開始")
    logger.info(f"評価サンプル数: {len(eval_trajectories)}")
    logger.info("=" * 80)
    
    # 予測
    predictions = []
    true_labels = []
    prediction_details = []
    
    for trajectory in eval_trajectories:
        # developer_info から developer_email を取得
        developer = trajectory.get('developer', trajectory.get('developer_info', {}))
        if isinstance(developer, dict):
            reviewer_email = developer.get('developer_email', 'unknown')
            activity_count = len(trajectory['activity_history'])
        else:
            reviewer_email = 'unknown'
            activity_count = len(trajectory['activity_history'])
        
        # スナップショット特徴量で予測確率を取得
        developer = trajectory.get('developer', {})
        activity_history = trajectory.get('activity_history', [])
        context_date = trajectory.get('context_date', None)
        
        result = irl_system.predict_continuation_probability_snapshot(
            developer=developer,
            activity_history=activity_history,
            context_date=context_date
        )
        prob = result.get('continuation_probability', 0.0)
        
        # 真のラベルを取得
        true_label = trajectory.get('future_contribution', False)
        
        predictions.append(prob)
        true_labels.append(1 if true_label else 0)
        
        prediction_details.append({
            'reviewer_email': reviewer_email,
            'predicted_prob': prob,
            'true_label': 1 if true_label else 0,
            'activity_count': activity_count,
            'reasoning': result.get('reasoning', ''),
            'confidence': result.get('confidence', 0.0)
        })
    
    # 評価メトリクス計算
    from sklearn.metrics import (
        auc,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    
    if len(set(true_labels)) > 1:  # 正例と負例が両方存在する場合
        auc_roc = roc_auc_score(true_labels, predictions)
        precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
        auc_pr = auc(recall, precision)
        
        # 最適閾値（F1スコアが最大）
        f1_scores = [f1_score(true_labels, [1 if p >= t else 0 for p in predictions]) for t in thresholds]
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        # 最適閾値でのメトリクス
        binary_predictions = [1 if p >= optimal_threshold else 0 for p in predictions]
        precision_val = precision_score(true_labels, binary_predictions)
        recall_val = recall_score(true_labels, binary_predictions)
        f1_val = f1_score(true_labels, binary_predictions)
        
        metrics = {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_val,
            'optimal_threshold': optimal_threshold,
            'continuation_rate': sum(true_labels) / len(true_labels)
        }
    else:
        metrics = {
            'auc_roc': 0.5,
            'auc_pr': sum(true_labels) / len(true_labels),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'optimal_threshold': 0.5,
            'continuation_rate': sum(true_labels) / len(true_labels)
        }
    
    logger.info("=" * 80)
    logger.info("評価完了")
    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"継続率: {metrics['continuation_rate']:.1%}")
    logger.info("=" * 80)
    
    return metrics, prediction_details


def main():
    parser = argparse.ArgumentParser(description='スライディングウィンドウIRL訓練・評価')
    parser.add_argument('--reviews', type=str, required=True,
                        help='レビューログCSVファイル')
    parser.add_argument('--train-start', type=str, required=True,
                        help='学習開始日 (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, required=True,
                        help='学習終了日 (YYYY-MM-DD)')
    parser.add_argument('--eval-start', type=str, required=True,
                        help='評価開始日 (YYYY-MM-DD)')
    parser.add_argument('--eval-end', type=str, required=True,
                        help='評価終了日 (YYYY-MM-DD)')
    parser.add_argument('--future-window-start', type=int, default=0,
                        help='将来窓開始（月数）')
    parser.add_argument('--future-window-end', type=int, default=3,
                        help='将来窓終了（月数）')
    parser.add_argument('--eval-future-window-start', type=int, default=None,
                        help='評価用将来窓開始（月数）')
    parser.add_argument('--eval-future-window-end', type=int, default=None,
                        help='評価用将来窓終了（月数）')
    parser.add_argument('--epochs', type=int, default=20,
                        help='訓練エポック数')
    parser.add_argument('--min-history-events', type=int, default=3,
                        help='最小履歴イベント数')
    parser.add_argument('--output', type=str, required=True,
                        help='出力ディレクトリ')
    parser.add_argument('--project', type=str, default=None,
                        help='プロジェクト名（指定時は単一プロジェクトのみ）')
    parser.add_argument('--model', type=str, default=None,
                        help='既存モデルファイル（指定時は評価のみ）')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 日付変換
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    
    # 評価用将来窓（指定がない場合は訓練と同じ）
    eval_future_window_start = args.eval_future_window_start if args.eval_future_window_start is not None else args.future_window_start
    eval_future_window_end = args.eval_future_window_end if args.eval_future_window_end is not None else args.future_window_end
    
    logger.info("=" * 80)
    logger.info("スライディングウィンドウIRL訓練・評価")
    logger.info("=" * 80)
    logger.info(f"レビューデータ: {args.reviews}")
    logger.info(f"学習期間: {train_start} ～ {train_end}")
    logger.info(f"評価期間: {eval_start} ～ {eval_end}")
    logger.info(f"訓練ラベル: {args.future_window_start}-{args.future_window_end}m")
    logger.info(f"評価期間: {eval_future_window_start}-{eval_future_window_end}m")
    if args.project:
        logger.info(f"プロジェクト: {args.project}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # モデル設定（時間特徴量除外版）
    config = {
        'state_dim': 9,   # 時間経過を除外（10→9）
        'action_dim': 4,  # レスポンス時間を含む（3→4）
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
        irl_system = RetentionIRLSystem.load_model(args.model)
        model_path = Path(args.model)
    else:
        # 訓練データ抽出（スライディングウィンドウ）
        train_trajectories = extract_sliding_window_trajectories(
            df,
            train_start=train_start,
            train_end=train_end,
            future_window_start_months=args.future_window_start,
            future_window_end_months=args.future_window_end,
            min_history_events=args.min_history_events,
            project=args.project,
        )
        
        if not train_trajectories:
            logger.error("訓練データがありません")
            return
        
        # モデル訓練（時系列）
        irl_system = train_irl_model_temporal(
            train_trajectories,
            config,
            epochs=args.epochs
        )
        
        # モデルを保存
        model_path = output_dir / 'irl_model.pt'
        irl_system.save_model(str(model_path))
        logger.info(f"モデル保存: {model_path}")
    
    # 評価データ抽出
    eval_trajectories = extract_cutoff_evaluation_trajectories(
        df,
        cutoff_date=eval_start,
        history_window_months=12,
        future_window_start_months=eval_future_window_start,
        future_window_end_months=eval_future_window_end,
        min_history_events=args.min_history_events,
        project=args.project,
    )
    
    if not eval_trajectories:
        logger.error("評価データがありません")
        return
    
    # モデル評価（スナップショット特徴量）
    metrics, prediction_details = evaluate_model_snapshot(irl_system, eval_trajectories)
    
    # 結果保存
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"メトリクス保存: {metrics_path}")
    
    # 予測詳細を保存
    if prediction_details:
        predictions_df = pd.DataFrame(prediction_details)
        predictions_df['predicted_binary'] = (predictions_df['predicted_prob'] >= metrics['optimal_threshold']).astype(int)
        predictions_path = output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"予測詳細保存: {predictions_path}")
    
    logger.info("=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
