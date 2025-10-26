#!/usr/bin/env python3
"""
学習期間内完結型IRL訓練スクリプト

重要な設計:
- 将来の貢献を「ラベル」として使用（状態特徴量には含めない）
- 学習期間内で複数時点をサンプリング
- すべてのデータが学習期間内で完結
- 評価期間とは cutoff で分離

目的:
- 続いた人と続かなかった人を区別する報酬関数を学習
- LSTMで時系列パターンを学習
"""
from __future__ import annotations

import argparse
import json
import logging
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

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_review_logs(csv_path: Path) -> pd.DataFrame:
    """レビューログを読み込む"""
    logger.info(f"レビューログを読み込み中: {csv_path}")
    df = pd.read_csv(csv_path)

    # 日付カラムをdatetimeに変換
    date_col = 'request_time' if 'request_time' in df.columns else 'created'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    logger.info(f"レビューログ読み込み完了: {len(df)}件")
    logger.info(f"期間: {df[date_col].min()} ～ {df[date_col].max()}")
    
    return df


def extract_temporal_trajectories_within_training_period(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    history_window_months: int = 6,
    future_window_start_months: int = 0,
    future_window_end_months: int = 1,
    sampling_interval_months: int = 1,
    min_history_events: int = 3,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time'
) -> List[Dict[str, Any]]:
    """
    学習期間内で完結する軌跡を抽出
    
    重要:
    - 将来の貢献は「ラベル」として使用（状態には含めない）
    - すべてのデータが学習期間内で完結
    
    Args:
        df: レビューログ
        train_start: 学習期間の開始日
        train_end: 学習期間の終了日
        history_window_months: 履歴ウィンドウ（ヶ月）
        future_window_start_months: 将来窓の開始（ヶ月）
        future_window_end_months: 将来窓の終了（ヶ月）
        sampling_interval_months: サンプリング間隔（ヶ月）
        min_history_events: 最小イベント数
        reviewer_col: レビュアーカラム名
        date_col: 日付カラム名
    
    Returns:
        軌跡リスト（各軌跡に future_contribution ラベルを含む）
    """
    logger.info("=" * 80)
    logger.info("学習期間内完結型の軌跡抽出を開始")
    logger.info(f"学習期間: {train_start} ～ {train_end}")
    logger.info(f"履歴ウィンドウ: {history_window_months}ヶ月")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    logger.info("=" * 80)
    
    trajectories = []
    
    # サンプリング可能範囲を計算
    min_sampling_date = train_start + pd.DateOffset(months=history_window_months)
    max_sampling_date = train_end - pd.DateOffset(months=future_window_end_months)
    
    logger.info(f"サンプリング可能範囲: {min_sampling_date} ～ {max_sampling_date}")
    
    # サンプリング時点を生成
    sampling_points = []
    current = min_sampling_date
    
    while current <= max_sampling_date:
        sampling_points.append(current)
        current += pd.DateOffset(months=sampling_interval_months)
    
    logger.info(f"サンプリング時点数: {len(sampling_points)}")
    
    # 全レビュアーを取得
    all_reviewers = df[reviewer_col].unique()
    logger.info(f"レビュアー数: {len(all_reviewers)}")
    
    # 各サンプリング時点でサンプルを生成
    for idx, sampling_point in enumerate(sampling_points):
        logger.info(f"サンプリング時点 {idx+1}/{len(sampling_points)}: {sampling_point}")
        
        # 履歴期間
        history_start = sampling_point - pd.DateOffset(months=history_window_months)
        history_end = sampling_point
        
        # 将来窓
        future_start = sampling_point + pd.DateOffset(months=future_window_start_months)
        future_end = sampling_point + pd.DateOffset(months=future_window_end_months)
        
        # 学習期間内で完結していることを確認
        assert future_end <= train_end, \
            f"将来窓が学習期間を超えています: {future_end} > {train_end}"
        
        # この時点の履歴データ
        history_df = df[
            (df[date_col] >= history_start) &
            (df[date_col] < history_end)
        ]
        
        # この時点の将来データ
        future_df = df[
            (df[date_col] >= future_start) &
            (df[date_col] < future_end)
        ]
        
        # レビュアーごとにサンプルを生成
        reviewers_with_history = history_df[reviewer_col].unique()
        
        for reviewer in reviewers_with_history:
            # このレビュアーの履歴
            reviewer_history = history_df[history_df[reviewer_col] == reviewer]
            
            # 最小イベント数を満たさない場合はスキップ
            if len(reviewer_history) < min_history_events:
                continue
            
            # 活動履歴を構築
            activity_history = []
            for _, row in reviewer_history.iterrows():
                activity = {
                    'timestamp': row[date_col],
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                }
                activity_history.append(activity)
            
            # 将来の貢献を計算（ラベルとして使用）
            reviewer_future = future_df[future_df[reviewer_col] == reviewer]
            future_contribution = len(reviewer_future) > 0
            
            # 開発者情報
            developer_info = {
                'developer_id': reviewer,
                'first_seen': reviewer_history[date_col].min(),
                'changes_reviewed': len(reviewer_history),
                'projects': (
                    reviewer_history['project'].unique().tolist()
                    if 'project' in reviewer_history.columns
                    else []
                ),
            }
            
            # 軌跡を作成
            trajectory = {
                'developer': developer_info,
                'activity_history': activity_history,
                'context_date': sampling_point,
                
                # 将来の貢献をラベルとして格納（状態特徴量ではない）
                'future_contribution': future_contribution,
                
                # メタデータ
                'future_window': {
                    'start_months': future_window_start_months,
                    'end_months': future_window_end_months,
                    'start_date': future_start,
                    'end_date': future_end,
                    'activity_count': len(reviewer_future),
                },
                'history_window': {
                    'months': history_window_months,
                    'start_date': history_start,
                    'end_date': history_end,
                    'activity_count': len(reviewer_history),
                },
                'sampling_point': sampling_point,
            }
            
            trajectories.append(trajectory)
    
    # 統計情報
    positive_count = sum(1 for t in trajectories if t['future_contribution'])
    positive_rate = positive_count / len(trajectories) if trajectories else 0
    
    logger.info("=" * 80)
    logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル")
    logger.info(f"  継続率: {positive_rate:.1%} ({positive_count}/{len(trajectories)})")
    logger.info(f"  すべてのデータが学習期間内で完結: ✅")
    logger.info("=" * 80)
    
    return trajectories


def extract_full_sequence_monthly_label_trajectories(
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
    全シーケンス＋月次集約ラベル付き軌跡を抽出（サンプリングなし）
    
    特徴：
    - **サンプリングなし**：各レビュアーから1サンプルのみ
    - **全シーケンス**：学習期間内の全活動履歴を使用
    - **ラベル**：月次集約（各月末から0-3m後に活動があるか）
    - **学習**：各活動（ステップ）単位で行う
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
    logger.info("全シーケンス＋月次集約ラベル付き軌跡抽出を開始（サンプリングなし）")
    logger.info(f"学習期間: {train_start} ～ {train_end}")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    logger.info("全シーケンス: 各レビュアーの学習期間内の全活動を使用")
    logger.info("ラベル: 月次集約（各月末から将来窓内に活動があるか）")
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
        
        # 月ごとにラベルを計算
        monthly_labels = {}
        
        for _, row in reviewer_history_sorted.iterrows():
            activity_date = pd.Timestamp(row[date_col])
            month_key = (activity_date.year, activity_date.month)
            
            if month_key not in monthly_labels:
                # この月の最終日
                month_end = (activity_date + pd.offsets.MonthEnd(0))
                
                # 将来窓の範囲
                future_start = month_end + pd.DateOffset(months=future_window_start_months)
                future_end = month_end + pd.DateOffset(months=future_window_end_months)
                
                # 将来窓が学習期間を超える場合はNone
                if future_end > train_end:
                    monthly_labels[month_key] = None
                else:
                    # この月から将来窓内に活動があるか
                    future_activities = reviewer_all_activities[
                        (reviewer_all_activities[date_col] >= future_start) &
                        (reviewer_all_activities[date_col] < future_end)
                    ]
                    monthly_labels[month_key] = len(future_activities) > 0
        
        # 活動履歴を構築
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
            'context_date': train_end,  # 固定時点
            'step_labels': step_labels,
            'seq_len': len(step_labels)
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


def extract_monthly_aggregated_label_trajectories(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    history_window_months: int = 12,
    future_window_start_months: int = 0,
    future_window_end_months: int = 3,
    sampling_interval_months: int = 1,
    seq_len: int = None,  # None = 可変長（全活動を使用）
    min_history_events: int = 3,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time',
    project: str = None
) -> List[Dict[str, Any]]:
    """
    サンプリング時点ベースのラベル付き軌跡を抽出
    
    特徴：
    - **ラベル**：サンプリング時点から0-3m後に活動があるか（レビュアー単位で1個）
    - **学習**：各活動（ステップ）単位で行う
    - 全ステップで同じラベルを持つ（レビュアー×サンプリング時点で固定）
    - 時系列学習は保たれる（LSTM入力は時系列順）
    
    Args:
        df: レビューデータ
        train_start: 学習開始日
        train_end: 学習終了日
        history_window_months: 履歴ウィンドウ（月数）
        future_window_start_months: 将来窓開始（月数）
        future_window_end_months: 将来窓終了（月数）
        sampling_interval_months: サンプリング間隔（月数）
        seq_len: シーケンス長（None=可変長）
        min_history_events: 最小活動数
        reviewer_col: レビュアー列名
        date_col: 日付列名
        project: プロジェクト名（指定時は単一プロジェクトのみ）
    
    Returns:
        各サンプルの軌跡のリスト
    """
    logger.info("=" * 80)
    logger.info("サンプリング時点ベースのラベル付き軌跡抽出を開始")
    logger.info(f"学習期間: {train_start} ～ {train_end}")
    logger.info(f"履歴ウィンドウ: {history_window_months}ヶ月")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    if seq_len is None:
        logger.info("seq_len: 可変長（全活動を使用）")
    else:
        logger.info(f"seq_len: {seq_len}（最新{seq_len}個の活動を使用）")
    logger.info("ラベル: サンプリング時点ベース（各サンプリング時点から将来窓内に活動があるか）")
    logger.info("学習: 各活動単位、全ステップで同じラベル（時系列学習は保持）")
    logger.info("=" * 80)
    
    # プロジェクトフィルタを適用
    if project and 'project' in df.columns:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")
    
    trajectories = []
    
    # サンプリング可能範囲を計算
    min_sampling_date = train_start + pd.DateOffset(months=history_window_months)
    max_sampling_date = train_end - pd.DateOffset(months=future_window_end_months)
    
    logger.info(f"サンプリング可能範囲: {min_sampling_date} ～ {max_sampling_date}")
    
    # サンプリング時点を生成
    sampling_points = []
    current = min_sampling_date
    
    while current <= max_sampling_date:
        sampling_points.append(current)
        current += pd.DateOffset(months=sampling_interval_months)
    
    logger.info(f"サンプリング時点数: {len(sampling_points)}")
    
    # 全レビュアーを取得
    all_reviewers = df[reviewer_col].unique()
    logger.info(f"レビュアー数: {len(all_reviewers)}")
    
    # 各サンプリング時点でサンプルを生成
    for idx, sampling_point in enumerate(sampling_points):
        if (idx + 1) % 5 == 0 or idx == 0:
            logger.info(f"サンプリング時点 {idx+1}/{len(sampling_points)}: {sampling_point}")
        
        # 履歴期間
        history_start = sampling_point - pd.DateOffset(months=history_window_months)
        history_end = sampling_point
        
        # サンプリング時点より前に活動があったすべての開発者（離脱者も含む）
        all_past_reviewers = df[df[date_col] < sampling_point][reviewer_col].unique()
        
        # 各レビュアーについてサンプルを生成
        for reviewer in all_past_reviewers:
            # このレビュアーの履歴期間内の活動
            reviewer_history = df[
                (df[reviewer_col] == reviewer) &
                (df[date_col] >= history_start) &
                (df[date_col] < history_end)
            ]
            
            # 最小イベント数を満たさない場合はスキップ
            if len(reviewer_history) < min_history_events:
                continue
            
            # 活動履歴を時系列順に並べる
            reviewer_history_sorted = reviewer_history.sort_values(date_col)
            
            # seq_lenがNoneでない場合のみ制限
            if seq_len is not None and len(reviewer_history_sorted) > seq_len:
                reviewer_history_sorted = reviewer_history_sorted.tail(seq_len)
            
            # サンプリング時点からの将来窓を計算
            future_start = sampling_point + pd.DateOffset(months=future_window_start_months)
            future_end = sampling_point + pd.DateOffset(months=future_window_end_months)
            
            # このレビュアーの将来窓内の活動を確認
            reviewer_all_activities = df[df[reviewer_col] == reviewer]
            future_activities = reviewer_all_activities[
                (reviewer_all_activities[date_col] >= future_start) &
                (reviewer_all_activities[date_col] < future_end)
            ]
            
            # サンプリング時点からの継続ラベル（レビュアー単位で1個）
            has_future_contribution = len(future_activities) > 0
            
            # 活動履歴を構築（全ステップで同じラベル）
            activity_history = []
            step_dates = []
            step_labels = []
            
            for _, row in reviewer_history_sorted.iterrows():
                activity = {
                    'timestamp': row[date_col],
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                }
                activity_history.append(activity)
                step_dates.append(row[date_col])
                # 全ステップで同じラベル（サンプリング時点からの継続）
                step_labels.append(has_future_contribution)
            
            # ラベルがない場合はスキップ
            if not step_labels:
                continue
            
            # 軌跡を作成
            developer_info = {
                'developer_email': reviewer
            }
            
            trajectory = {
                'developer_info': developer_info,
                'activity_history': activity_history,
                'context_date': sampling_point,
                'step_labels': step_labels,
                'seq_len': len(step_labels)
            }
            
            trajectories.append(trajectory)
    
    logger.info("=" * 80)
    logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル")
    if trajectories:
        total_steps = sum(t['seq_len'] for t in trajectories)
        continued_steps = sum(sum(t['step_labels']) for t in trajectories)
        logger.info(f"  総ステップ数: {total_steps}")
        logger.info(f"  継続ステップ率: {continued_steps/total_steps*100:.1f}% ({continued_steps}/{total_steps})")
    logger.info("=" * 80)
    
    return trajectories


def extract_multi_step_label_trajectories(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    history_window_months: int = 12,
    future_window_start_months: int = 0,
    future_window_end_months: int = 3,
    sampling_interval_months: int = 1,
    seq_len: int = None,  # None = 可変長（全活動を使用）
    min_history_events: int = 3,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time',
    project: str = None
) -> List[Dict[str, Any]]:
    """
    各タイムステップラベル付き軌跡を抽出
    
    重要:
    - seq_len個の実際の活動タイムスタンプを使用
    - 各タイムステップからの将来貢献をラベル付け
    - すべてのデータが学習期間内で完結
    
    Args:
        df: レビューログ
        train_start: 学習期間の開始日
        train_end: 学習期間の終了日
        history_window_months: 履歴ウィンドウ（ヶ月）
        future_window_start_months: 将来窓の開始（ヶ月）
        future_window_end_months: 将来窓の終了（ヶ月）
        sampling_interval_months: サンプリング間隔（ヶ月）
        seq_len: 使用する活動数（最新seq_len個）
        min_history_events: 最小イベント数
        reviewer_col: レビュアーカラム名
        date_col: 日付カラム名
        project: プロジェクト名（指定時は単一プロジェクトのみ）
    
    Returns:
        軌跡リスト（各軌跡に seq_len 個のラベルを含む）
    """
    logger.info("=" * 80)
    logger.info("各タイムステップラベル付き軌跡抽出を開始")
    logger.info(f"学習期間: {train_start} ～ {train_end}")
    logger.info(f"履歴ウィンドウ: {history_window_months}ヶ月")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    if seq_len is None:
        logger.info("seq_len: 可変長（全活動を使用）")
    else:
        logger.info(f"seq_len: {seq_len}（最新{seq_len}個の活動を使用）")
    logger.info("=" * 80)
    
    # プロジェクトフィルタを適用
    if project and 'project' in df.columns:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")
    
    trajectories = []
    
    # サンプリング可能範囲を計算
    min_sampling_date = train_start + pd.DateOffset(months=history_window_months)
    max_sampling_date = train_end - pd.DateOffset(months=future_window_end_months)
    
    logger.info(f"サンプリング可能範囲: {min_sampling_date} ～ {max_sampling_date}")
    
    # サンプリング時点を生成
    sampling_points = []
    current = min_sampling_date
    
    while current <= max_sampling_date:
        sampling_points.append(current)
        current += pd.DateOffset(months=sampling_interval_months)
    
    logger.info(f"サンプリング時点数: {len(sampling_points)}")
    
    # 全レビュアーを取得
    all_reviewers = df[reviewer_col].unique()
    logger.info(f"レビュアー数: {len(all_reviewers)}")
    
    # 各サンプリング時点でサンプルを生成
    for idx, sampling_point in enumerate(sampling_points):
        if (idx + 1) % 5 == 0 or idx == 0:
            logger.info(f"サンプリング時点 {idx+1}/{len(sampling_points)}: {sampling_point}")
        
        # 履歴期間
        history_start = sampling_point - pd.DateOffset(months=history_window_months)
        history_end = sampling_point
        
        # サンプリング時点より前に活動があったすべての開発者（離脱者も含む）
        all_past_reviewers = df[df[date_col] < sampling_point][reviewer_col].unique()
        
        # 各レビュアーについてサンプルを生成
        for reviewer in all_past_reviewers:
            # このレビュアーの履歴期間内の活動
            reviewer_history = df[
                (df[reviewer_col] == reviewer) &
                (df[date_col] >= history_start) &
                (df[date_col] < history_end)
            ]
            
            # 最小イベント数を満たさない場合はスキップ
            if len(reviewer_history) < min_history_events:
                continue
            
            # 活動履歴を時系列順に並べる
            reviewer_history_sorted = reviewer_history.sort_values(date_col)
            
            # seq_lenがNoneでない場合のみ制限
            if seq_len is not None and len(reviewer_history_sorted) > seq_len:
                reviewer_history_sorted = reviewer_history_sorted.tail(seq_len)
            
            # 活動履歴を構築
            activity_history = []
            step_dates = []
            
            for _, row in reviewer_history_sorted.iterrows():
                activity = {
                    'timestamp': row[date_col],
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                }
                activity_history.append(activity)
                step_dates.append(row[date_col])
            
            # 各ステップ（活動）から次の貢献までの期間を計算
            # このレビュアーの全活動を時系列順に取得
            reviewer_all_activities = df[df[reviewer_col] == reviewer].sort_values(date_col)
            reviewer_timestamps = reviewer_all_activities[date_col].values
            
            step_labels = []
            valid_indices = []
            
            for idx, step_date in enumerate(step_dates):
                # このステップより後の活動を取得
                future_activities = [ts for ts in reviewer_timestamps if pd.Timestamp(ts) > step_date]
                
                if len(future_activities) == 0:
                    # 次の活動がない → 離脱
                    has_near_future = False
                else:
                    # 次の活動までの期間を計算
                    next_activity_date = pd.Timestamp(future_activities[0])
                    
                    # 将来窓の範囲を計算
                    future_start = step_date + pd.DateOffset(months=future_window_start_months)
                    future_end = step_date + pd.DateOffset(months=future_window_end_months)
                    
                    # 次の活動が将来窓内にあるか
                    has_near_future = (future_start <= next_activity_date < future_end)
                
                # 将来窓が学習期間を超える場合はこのステップをスキップ
                future_window_end = step_date + pd.DateOffset(months=future_window_end_months)
                if future_window_end > train_end:
                    continue
                
                step_labels.append(has_near_future)
                valid_indices.append(idx)
            
            # 有効なステップがない場合はスキップ
            if not valid_indices:
                continue
            
            # 有効なステップのみを使用
            activity_history = [activity_history[i] for i in valid_indices]
            step_dates = [step_dates[i] for i in valid_indices]
            
            # ステップ数の確認
            if not step_labels:
                continue
            
            # 開発者情報
            developer_info = {
                'developer_id': reviewer,
                'first_seen': reviewer_history[date_col].min(),
                'changes_reviewed': len(reviewer_history),
                'projects': (
                    reviewer_history['project'].unique().tolist()
                    if 'project' in reviewer_history.columns
                    else []
                ),
            }
            
            # 軌跡を作成
            trajectory = {
                'developer': developer_info,
                'activity_history': activity_history,
                'context_date': sampling_point,
                
                # 各ステップのラベル
                'step_labels': step_labels,
                'seq_len': len(step_labels),
                
                # メタデータ
                'future_window': {
                    'start_months': future_window_start_months,
                    'end_months': future_window_end_months,
                },
                'history_window': {
                    'months': history_window_months,
                    'start_date': history_start,
                    'end_date': history_end,
                    'activity_count': len(reviewer_history),
                },
                'sampling_point': sampling_point,
            }
            
            trajectories.append(trajectory)
    
    # 統計情報
    if trajectories:
        total_steps = sum(t['seq_len'] for t in trajectories)
        positive_steps = sum(sum(1 for label in t['step_labels'] if label) for t in trajectories)
        positive_rate = positive_steps / total_steps if total_steps > 0 else 0
        
        logger.info("=" * 80)
        logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル")
        logger.info(f"  総ステップ数: {total_steps}")
        logger.info(f"  継続ステップ率: {positive_rate:.1%} ({positive_steps}/{total_steps})")
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
    Cutoff時点での評価用軌跡を抽出（ロングコントリビュータ予測スタイル）
    
    重要:
    - cutoff_date時点で履歴期間内に活動があった全開発者を対象
    - 将来貢献は cutoff_date から計算
    - サンプリングなし（全開発者を評価）
    
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
    logger.info("Cutoff評価用軌跡抽出を開始")
    logger.info(f"Cutoff日: {cutoff_date}")
    logger.info(f"履歴ウィンドウ: {history_window_months}ヶ月")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
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
        # このレビュアーの履歴
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]
        
        # 最小イベント数を満たさない場合はスキップ
        if len(reviewer_history) < min_history_events:
            continue
        
        # 活動履歴を構築
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col],
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
            }
            activity_history.append(activity)
        
        # 将来の貢献を計算
        reviewer_future = future_df[future_df[reviewer_col] == reviewer]
        future_contribution = len(reviewer_future) > 0
        
        # 開発者情報
        developer_info = {
            'developer_id': reviewer,
            'first_seen': reviewer_history[date_col].min(),
            'changes_reviewed': len(reviewer_history),
            'projects': (
                reviewer_history['project'].unique().tolist()
                if 'project' in reviewer_history.columns
                else []
            ),
        }
        
        # 軌跡を作成
        trajectory = {
            'developer': developer_info,
            'activity_history': activity_history,
            'context_date': cutoff_date,
            
            # 将来の貢献をラベルとして格納
            'future_contribution': future_contribution,
            
            # メタデータ
            'future_window': {
                'start_months': future_window_start_months,
                'end_months': future_window_end_months,
                'start_date': future_start,
                'end_date': future_end,
                'activity_count': len(reviewer_future),
            },
            'history_window': {
                'months': history_window_months,
                'start_date': history_start,
                'end_date': history_end,
                'activity_count': len(reviewer_history),
            },
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


def train_irl_model(
    trajectories: List[Dict[str, Any]],
    config: Dict[str, Any],
    epochs: int = 30
) -> RetentionIRLSystem:
    """
    続いた人と続かなかった人を区別する報酬関数を学習
    
    重要:
    - 状態には将来の情報を含めない
    - 将来の貢献をターゲット（ラベル）として使用
    - LSTMで時系列パターンを学習
    """
    logger.info("=" * 80)
    logger.info("IRL訓練開始")
    logger.info(f"軌跡数: {len(trajectories)}")
    logger.info(f"エポック数: {epochs}")
    logger.info("目標: 続いた人と続かなかった人を区別する報酬関数を学習")
    logger.info("=" * 80)
    
    irl_system = RetentionIRLSystem(config)
    
    training_result = irl_system.train_irl(trajectories, epochs=epochs)
    
    logger.info(f"訓練完了: 最終損失={training_result['final_loss']:.4f}")
    
    return irl_system


def find_optimal_threshold(y_true, y_pred_proba):
    """
    F1スコアを最大化する閾値を探索
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds:
        y_pred_binary = [1 if p >= threshold else 0 for p in y_pred_proba]
        
        try:
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'f1': f1,
                    'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                    'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                }
        except:
            continue
    
    return best_threshold, best_metrics


def evaluate_irl_model(
    irl_system: RetentionIRLSystem,
    test_trajectories: List[Dict[str, Any]],
    fixed_threshold: float = None
) -> Dict[str, float]:
    """
    IRLモデルを評価
    
    Args:
        irl_system: 評価するIRLシステム
        test_trajectories: テスト軌跡
        fixed_threshold: 固定閾値（Noneの場合は最適閾値を探索）
    """
    logger.info("=" * 80)
    logger.info("IRLモデル評価開始")
    logger.info(f"テストサンプル数: {len(test_trajectories)}")
    if fixed_threshold is not None:
        logger.info(f"固定閾値: {fixed_threshold:.2f}")
    logger.info("=" * 80)
    
    y_true = []
    y_pred = []
    
    for trajectory in test_trajectories:
        developer = trajectory['developer']
        activity_history = trajectory['activity_history']
        context_date = trajectory['context_date']
        true_label = trajectory['future_contribution']
        
        # 予測実行（状態には将来の情報を含めない）
        prediction = irl_system.predict_continuation_probability(
            developer, activity_history, context_date
        )
        
        y_true.append(1 if true_label else 0)
        y_pred.append(prediction['continuation_probability'])
    
    # 閾値を決定
    if fixed_threshold is not None:
        # 固定閾値を使用
        threshold = fixed_threshold
    else:
        # 最適閾値を探索
        threshold, _ = find_optimal_threshold(y_true, y_pred)
    
    # 閾値で2値予測
    y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]
    
    # メトリクス計算
    metrics = {}
    
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['auc_roc'] = 0.0
    
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
        metrics['auc_pr'] = auc(recall_curve, precision_curve)
    except:
        metrics['auc_pr'] = 0.0
    
    # メトリクス計算
    try:
        metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['optimal_threshold'] = threshold
    except:
        metrics['f1'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['optimal_threshold'] = 0.5
    
    # 比較用に閾値0.5でも計算（固定閾値使用時のみ）
    if fixed_threshold is not None:
        y_pred_binary_05 = [1 if p >= 0.5 else 0 for p in y_pred]
        metrics['f1_threshold_0.5'] = f1_score(y_true, y_pred_binary_05, zero_division=0)
    else:
        metrics['f1_threshold_0.5'] = metrics['f1']  # 最適閾値使用時は同じ
    
    # 予測確率の統計
    metrics['pred_mean'] = float(np.mean(y_pred))
    metrics['pred_std'] = float(np.std(y_pred))
    
    metrics['test_samples'] = len(test_trajectories)
    positive_count = sum(y_true)
    metrics['positive_samples'] = positive_count
    metrics['positive_rate'] = positive_count / len(y_true) if y_true else 0
    
    logger.info("評価結果:")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    logger.info(f"  AUC-PR: {metrics['auc_pr']:.3f}")
    logger.info(f"  予測確率: {metrics['pred_mean']:.3f} ± {metrics['pred_std']:.3f}")
    logger.info(f"  使用閾値: {metrics['optimal_threshold']:.2f}")
    logger.info(f"  F1: {metrics['f1']:.3f}")
    if fixed_threshold is not None:
        logger.info(f"  F1（参考: 閾値0.5）: {metrics['f1_threshold_0.5']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall: {metrics['recall']:.3f}")
    logger.info(f"  正例率: {metrics['positive_rate']:.1%}")
    logger.info("=" * 80)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='学習期間内完結型IRL訓練スクリプト'
    )
    
    # データ設定
    parser.add_argument(
        '--reviews',
        type=Path,
        required=True,
        help='レビューログCSVファイルのパス'
    )
    
    # 期間設定
    parser.add_argument(
        '--train-start',
        type=str,
        required=True,
        help='学習期間の開始日（例: 2019-01-01）'
    )
    parser.add_argument(
        '--train-end',
        type=str,
        required=True,
        help='学習期間の終了日（例: 2020-01-01）'
    )
    parser.add_argument(
        '--eval-start',
        type=str,
        help='評価期間の開始日（例: 2020-01-01、デフォルト=train-end）'
    )
    parser.add_argument(
        '--eval-end',
        type=str,
        help='評価期間の終了日（例: 2021-01-01、デフォルト=train-end+12m）'
    )
    
    # ウィンドウ設定
    parser.add_argument(
        '--history-window',
        type=int,
        default=6,
        help='履歴ウィンドウ（ヶ月、デフォルト: 6）'
    )
    parser.add_argument(
        '--future-window-start',
        type=int,
        default=0,
        help='将来窓の開始（ヶ月、デフォルト: 0）'
    )
    parser.add_argument(
        '--future-window-end',
        type=int,
        default=1,
        help='将来窓の終了（ヶ月、デフォルト: 1）'
    )
    parser.add_argument(
        '--sampling-interval',
        type=int,
        default=1,
        help='サンプリング間隔（ヶ月、デフォルト: 1）'
    )
    
    # 訓練設定
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='訓練エポック数（デフォルト: 30）'
    )
    parser.add_argument(
        '--sequence',
        action='store_true',
        help='時系列モード（LSTM）を有効化'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=15,
        help='シーケンス長（デフォルト: 15）'
    )
    
    # 出力設定
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/irl_within_training_period'),
        help='出力ディレクトリ'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 期間をパース
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    
    if args.eval_start:
        eval_start = pd.Timestamp(args.eval_start)
    else:
        eval_start = train_end  # デフォルト: 訓練終了日から
    
    if args.eval_end:
        eval_end = pd.Timestamp(args.eval_end)
    else:
        eval_end = eval_start + pd.DateOffset(months=12)  # デフォルト: +12ヶ月
    
    logger.info("=" * 80)
    logger.info("学習期間内完結型IRL訓練")
    logger.info("=" * 80)
    logger.info(f"訓練期間: {train_start} ～ {train_end}")
    logger.info(f"評価期間: {eval_start} ～ {eval_end}")
    logger.info(f"履歴ウィンドウ: {args.history_window}ヶ月")
    logger.info(f"将来窓: {args.future_window_start}～{args.future_window_end}ヶ月")
    logger.info(f"時系列モード: {'有効' if args.sequence else '無効'}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # 訓練データを抽出
    train_trajectories = extract_temporal_trajectories_within_training_period(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=args.history_window,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        sampling_interval_months=args.sampling_interval,
    )
    
    if not train_trajectories:
        logger.error("訓練データが見つかりません")
        return 1
    
    # 評価データを抽出
    eval_trajectories = extract_temporal_trajectories_within_training_period(
        df=df,
        train_start=eval_start,
        train_end=eval_end,
        history_window_months=args.history_window,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        sampling_interval_months=args.sampling_interval,
    )
    
    if not eval_trajectories:
        logger.warning("評価データが見つかりません")
    
    # IRL設定
    irl_config = {
        'state_dim': 10,  # 拡張IRL特徴量の次元
        'action_dim': 5,
        'hidden_dim': 64,
        'lstm_hidden': 128,
        'sequence': args.sequence,
        'seq_len': args.seq_len,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # 訓練
    irl_system = train_irl_model(
        trajectories=train_trajectories,
        config=irl_config,
        epochs=args.epochs
    )
    
    # モデルを保存
    model_path = args.output / 'irl_model.pth'
    irl_system.save_model(model_path)
    logger.info(f"モデルを保存: {model_path}")
    
    # 評価
    if eval_trajectories:
        metrics = evaluate_irl_model(irl_system, eval_trajectories)
        
        # 結果を保存
        results = {
            'train_period': {
                'start': str(train_start),
                'end': str(train_end),
            },
            'eval_period': {
                'start': str(eval_start),
                'end': str(eval_end),
            },
            'windows': {
                'history_months': args.history_window,
                'future_start_months': args.future_window_start,
                'future_end_months': args.future_window_end,
            },
            'training': {
                'epochs': args.epochs,
                'sequence': args.sequence,
                'seq_len': args.seq_len,
                'train_samples': len(train_trajectories),
            },
            'metrics': metrics,
        }
        
        results_path = args.output / 'evaluation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"評価結果を保存: {results_path}")
    
    logger.info("=" * 80)
    logger.info("完了")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

