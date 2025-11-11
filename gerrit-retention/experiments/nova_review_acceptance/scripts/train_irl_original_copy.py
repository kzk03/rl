#!/usr/bin/env python3
"""
レビュー承諾予測IRL学習（正しいロジック版）

目的：
- レビュー依頼を受けた開発者が、その依頼を承諾するかどうかを予測
- レビュー依頼を受けていない開発者は判定対象外として除外

継続判定ロジック：
- 評価期間内にレビュー依頼を受けていない → 除外
- 評価期間内にレビュー依頼を受けて、少なくとも1つ承諾 → 正例（継続）
- 評価期間内にレビュー依頼を受けたが、全て拒否 → 負例（離脱）

データ構造：
- label = 1: レビュー依頼に応答（承諾）
- label = 0: レビュー依頼に応答せず（拒否/無視）
"""
from __future__ import annotations

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

# ランダムシード固定
RANDOM_SEED = 777
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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


def load_review_requests(csv_path: str) -> pd.DataFrame:
    """
    レビュー依頼データを読み込む
    
    Args:
        csv_path: レビュー依頼CSVファイルのパス
        
    Returns:
        レビュー依頼データフレーム
    """
    logger.info(f"レビュー依頼データを読み込み: {csv_path}")
    df = pd.read_csv(csv_path)
    df['request_time'] = pd.to_datetime(df['request_time'])
    logger.info(f"総レビュー依頼数: {len(df)}")
    logger.info(f"承諾数: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")
    logger.info(f"拒否数: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    return df


def extract_review_acceptance_trajectories(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_window_start_months: int = 0,
    future_window_end_months: int = 3,
    min_history_requests: int = 3,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time',
    label_col: str = 'label',
    project: str = None,
    extended_label_window_months: int = 12
) -> List[Dict[str, Any]]:
    """
    レビュー承諾予測用の軌跡を抽出（データリークなし版）

    重要：訓練期間内で完結させるため、訓練期間を以下のように分割：
    - **特徴量計算期間**: train_start ～ (train_end - future_window_end_months)
    - **ラベル計算期間**: 特徴量計算期間終了後 ～ train_end

    これにより、訓練期間（train_end）を超えるデータを参照せずにラベル付けが可能。

    特徴：
    - **訓練**：特徴量計算期間内にレビュー依頼を受けた開発者のみを対象
    - **正例**：ラベル計算期間内に少なくとも1つのレビュー依頼を承諾した
    - **負例**：ラベル計算期間内にレビュー依頼を受けたが、全て拒否した
    - **負例（拡張）**：ラベル計算期間に依頼なし（この期間では活動なし）、拡張期間に依頼あり
    - **除外**：拡張期間内にもレビュー依頼を受けていない開発者（本当に離脱）

    Args:
        df: レビュー依頼データ
        train_start: 学習開始日
        train_end: 学習終了日
        future_window_start_months: 将来窓開始（月数）
        future_window_end_months: 将来窓終了（月数）
        min_history_requests: 最小履歴レビュー依頼数
        reviewer_col: レビュアー列名
        date_col: 日付列名
        label_col: ラベル列名（1=承諾, 0=拒否）
        project: プロジェクト名（指定時は単一プロジェクトのみ）
        extended_label_window_months: 拡張ラベル期間（月数、デフォルト12）

    Returns:
        各レビュアーの軌跡のリスト（1レビュアー=1サンプル）
    """
    logger.info("=" * 80)
    logger.info("レビュー承諾予測用軌跡抽出を開始（データリークなし版）")
    logger.info(f"訓練期間全体: {train_start} ～ {train_end}")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    logger.info(f"拡張ラベル期間: {future_window_start_months}～{extended_label_window_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    logger.info("ラベル定義: この期間で承諾=1、この期間で拒否=0、依頼なし→拡張期間チェック、訓練時は拡張期間にも依頼なしで除外")
    logger.info("データリーク防止: 訓練期間内でラベル計算")
    logger.info("=" * 80)
    
    # プロジェクトフィルタを適用
    if project and 'project' in df.columns:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")
    
    trajectories = []
    
    # 訓練期間全体を特徴量計算に使用（固定）
    history_start = train_start
    history_end = train_end
    
    # ラベル計算は各月末時点から将来窓を見る（月次ラベル用）
    # ここでは全体のポジティブ/ネガティブ判定用にtrain_end時点からのラベルを計算
    label_start = train_end + pd.DateOffset(months=future_window_start_months)
    label_end = train_end + pd.DateOffset(months=future_window_end_months)
    
    logger.info(f"特徴量計算期間（訓練全体）: {history_start} ～ {history_end}")
    logger.info(f"全体ラベル期間（train_end時点から）: {label_start} ～ {label_end}")
    
    # 特徴量計算期間のレビュー依頼データ
    history_df = df[
        (df[date_col] >= history_start) &
        (df[date_col] < history_end)
    ]
    
    # ラベル計算期間のレビュー依頼データ
    label_df = df[
        (df[date_col] >= label_start) &
        (df[date_col] < label_end)
    ]

    # 拡張ラベル計算期間のレビュー依頼データ
    extended_label_start = train_end + pd.DateOffset(months=future_window_start_months)
    extended_label_end = train_end + pd.DateOffset(months=extended_label_window_months)
    extended_label_df = df[
        (df[date_col] >= extended_label_start) &
        (df[date_col] < extended_label_end)
    ]

    # 特徴量計算期間内にレビュー依頼を受けたレビュアーを対象
    active_reviewers = history_df[reviewer_col].unique()
    logger.info(f"特徴量計算期間内のレビュアー数: {len(active_reviewers)}")

    skipped_min_requests = 0
    skipped_no_requests_until_end = 0  # 訓練期間末尾まで依頼がない（除外）
    positive_count = 0
    negative_count = 0
    negative_with_requests = 0  # 依頼あり→拒否
    negative_without_requests = 0  # 依頼なし（拡張期間に依頼あり）
    
    for reviewer in active_reviewers:
        # 特徴量計算期間のレビュー依頼
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]
        
        # 最小レビュー依頼数を満たさない場合はスキップ
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue
        
        # ラベル計算期間のレビュー依頼（訓練期間内）
        reviewer_label = label_df[label_df[reviewer_col] == reviewer]

        # ラベル計算期間にレビュー依頼を受けていない場合、拡張期間をチェック
        if len(reviewer_label) == 0:
            # 拡張期間のレビュー依頼をチェック
            reviewer_extended_label = extended_label_df[extended_label_df[reviewer_col] == reviewer]

            # 訓練時は拡張期間まで見て除外判定
            if len(reviewer_extended_label) == 0:
                # 拡張期間にも依頼がない → 訓練期間末尾までアサインがない → 除外（実質離脱者）
                skipped_no_requests_until_end += 1
                continue  # このレビュアーをスキップ
            
            # 拡張期間に依頼がある → 再びアサインされる可能性 → 重み付き負例
            future_acceptance = False  # この期間では活動なし
            accepted_requests = pd.DataFrame()  # 空
            rejected_requests = pd.DataFrame()  # 空（依頼自体がない）
            had_requests = False  # この期間に依頼がなかった
            sample_weight = 0.1  # 非常に低い重み（依頼なし）

            # 統計カウント
            negative_count += 1
            negative_without_requests += 1
        else:
            # 通常のラベル期間に依頼がある場合
            # 継続判定：ラベル計算期間内に少なくとも1つのレビュー依頼を承諾したか
            accepted_requests = reviewer_label[reviewer_label[label_col] == 1]
            rejected_requests = reviewer_label[reviewer_label[label_col] == 0]
            future_acceptance = len(accepted_requests) > 0
            had_requests = True  # この期間に依頼があった
            sample_weight = 1.0  # 通常の重み（依頼あり）

            if future_acceptance:
                positive_count += 1
            else:
                negative_count += 1
                negative_with_requests += 1
        
        # 特徴量計算期間の月次ラベルを計算（訓練用、データリークなし）
        history_months = pd.date_range(
            start=history_start,
            end=history_end,
            freq='MS'  # 月初
        )
        
        step_labels = []
        monthly_activity_histories = []  # 各月時点での活動履歴
        
        for month_start in history_months[:-1]:  # 最後の月を除く
            month_end = month_start + pd.DateOffset(months=1)
            
            # この月からfuture_window後のラベル計算期間
            future_start = month_end + pd.DateOffset(months=future_window_start_months)
            future_end = month_end + pd.DateOffset(months=future_window_end_months)
            
            # 重要：future_endがtrain_endを超えないようにクリップ（データリーク防止）
            if future_end > train_end:
                future_end = train_end
            
            # train_endを超える場合はこの月のラベルは作成しない
            if future_start >= train_end:
                continue
            
            # 将来期間のレビュー依頼（訓練期間内のみ）
            month_future_df = df[
                (df[date_col] >= future_start) &
                (df[date_col] < future_end) &
                (df[reviewer_col] == reviewer)
            ]
            
            # この月のラベル：将来期間にレビュー依頼を受けて承諾したか
            if len(month_future_df) == 0:
                # レビュー依頼なし → ラベル0
                month_label = 0
            else:
                # レビュー依頼あり → 承諾の有無
                month_accepted = month_future_df[month_future_df[label_col] == 1]
                month_label = 1 if len(month_accepted) > 0 else 0
            
            step_labels.append(month_label)
            
            # この月時点（month_end）までの活動履歴を保存（LSTM用）
            month_history = reviewer_history[reviewer_history[date_col] < month_end]
            monthly_activities = []
            for _, row in month_history.iterrows():
                activity = {
                    'timestamp': row[date_col],  # タイムスタンプは常にdate_col
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                    'request_time': row.get('request_time', row[date_col]),
                    'response_time': row.get('first_response_time'),  # response_time計算用
                    'accepted': row.get(label_col, 0) == 1,
                }
                monthly_activities.append(activity)
            monthly_activity_histories.append(monthly_activities)
        
        # 全期間の活動履歴も保持（評価用）
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col],  # タイムスタンプは常にdate_col
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'request_time': row.get('request_time', row[date_col]),
                'response_time': row.get('first_response_time'),  # response_time計算用
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
        
        # 軌跡を作成（LSTM用に月次活動履歴を追加）
        trajectory = {
            'developer_info': developer_info,
            'activity_history': activity_history,  # 全期間の活動履歴（評価用）
            'monthly_activity_histories': monthly_activity_histories,  # 各月時点の活動履歴（LSTM訓練用）
            'context_date': train_end,
            'step_labels': step_labels,
            'seq_len': len(step_labels),
            'reviewer': reviewer,
            'history_request_count': len(reviewer_history),
            'history_accepted_count': len(reviewer_history[reviewer_history[label_col] == 1]),
            'history_rejected_count': len(reviewer_history[reviewer_history[label_col] == 0]),
            'label_request_count': len(reviewer_label),
            'label_accepted_count': len(accepted_requests),
            'label_rejected_count': len(rejected_requests),
            'future_acceptance': future_acceptance,
            'had_requests': had_requests,  # この期間に依頼があったか
            'sample_weight': sample_weight  # サンプル重み（依頼なし=0.3、依頼あり=1.0）
        }
        
        trajectories.append(trajectory)
    
    logger.info("=" * 80)
    logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル（レビュアー）")
    logger.info(f"  スキップ（最小依頼数未満）: {skipped_min_requests}")
    logger.info(f"  スキップ（訓練期間末尾まで依頼なし）: {skipped_no_requests_until_end}")
    if trajectories:
        logger.info(f"  正例（この期間で承諾あり）: {positive_count} ({positive_count/len(trajectories)*100:.1f}%)")
        logger.info(f"  負例（この期間で承諾なし）: {negative_count} ({negative_count/len(trajectories)*100:.1f}%)")
        if negative_with_requests > 0:
            logger.info(f"    - 依頼あり→拒否（重み=1.0）: {negative_with_requests}")
        if negative_without_requests > 0:
            logger.info(f"    - 依頼なし（拡張期間に依頼あり、重み=0.1）: {negative_without_requests}")
        total_steps = sum(t['seq_len'] for t in trajectories)
        continued_steps = sum(sum(t['step_labels']) for t in trajectories)
        logger.info(f"  総ステップ数: {total_steps}")
        if total_steps > 0:
            logger.info(f"  継続ステップ率: {continued_steps/total_steps*100:.1f}% ({continued_steps}/{total_steps})")
    logger.info("=" * 80)
    
    return trajectories


def extract_evaluation_trajectories(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    history_window_months: int = 12,
    future_window_start_months: int = 0,
    future_window_end_months: int = 3,
    min_history_requests: int = 3,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time',
    label_col: str = 'label',
    project: str = None,
    extended_label_window_months: int = 12
) -> List[Dict[str, Any]]:
    """
    評価用軌跡を抽出（スナップショット特徴量用）

    Args:
        df: レビュー依頼データ
        cutoff_date: Cutoff日（通常は訓練終了日）
        history_window_months: 履歴ウィンドウ（ヶ月）
        future_window_start_months: 将来窓の開始（ヶ月）
        future_window_end_months: 将来窓の終了（ヶ月）
        min_history_requests: 最小履歴レビュー依頼数
        reviewer_col: レビュアー列名
        date_col: 日付列名
        label_col: ラベル列名
        project: プロジェクト名（指定時は単一プロジェクトのみ）
        extended_label_window_months: 拡張ラベル期間（月数、デフォルト12）

    Returns:
        軌跡リスト
    """
    logger.info("=" * 80)
    logger.info("評価用軌跡抽出を開始（スナップショット特徴量用）")
    logger.info(f"Cutoff日: {cutoff_date}")
    logger.info(f"履歴ウィンドウ: {history_window_months}ヶ月")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    logger.info(f"拡張ラベル期間: {future_window_start_months}～{extended_label_window_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    logger.info("継続判定: この期間で承諾=1、この期間で拒否=0、依頼なし→拡張期間チェック、評価時は拡張期間にも依頼なしで除外（予測の母集団に入れない）")
    logger.info("=" * 80)
    
    # プロジェクトフィルタを適用
    if project and 'project' in df.columns:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")
    
    trajectories = []
    
    # 履歴期間
    history_start = cutoff_date - pd.DateOffset(months=history_window_months)
    history_end = cutoff_date
    
    # 評価期間
    eval_start = cutoff_date + pd.DateOffset(months=future_window_start_months)
    eval_end = cutoff_date + pd.DateOffset(months=future_window_end_months)
    
    logger.info(f"履歴期間: {history_start} ～ {history_end}")
    logger.info(f"評価期間: {eval_start} ～ {eval_end}")
    
    # 履歴期間のレビュー依頼データ
    history_df = df[
        (df[date_col] >= history_start) &
        (df[date_col] < history_end)
    ]
    
    # 評価期間のレビュー依頼データ
    eval_df = df[
        (df[date_col] >= eval_start) &
        (df[date_col] < eval_end)
    ]

    # 拡張評価期間のレビュー依頼データ
    extended_eval_start = cutoff_date + pd.DateOffset(months=future_window_start_months)
    extended_eval_end = cutoff_date + pd.DateOffset(months=extended_label_window_months)
    extended_eval_df = df[
        (df[date_col] >= extended_eval_start) &
        (df[date_col] < extended_eval_end)
    ]

    # 履歴期間内にレビュー依頼を受けたレビュアーを対象
    active_reviewers = history_df[reviewer_col].unique()
    logger.info(f"履歴期間内のレビュアー数: {len(active_reviewers)}")

    skipped_min_requests = 0
    skipped_no_requests_until_end = 0  # 訓練期間末尾まで依頼がない（除外）
    positive_count = 0
    negative_count = 0
    negative_with_requests = 0  # 依頼あり→拒否
    negative_without_requests = 0  # 依頼なし（拡張期間に依頼あり）
    
    for reviewer in active_reviewers:
        # 履歴期間のレビュー依頼
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]
        
        # 最小レビュー依頼数を満たさない場合はスキップ
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue
        
        # 評価期間のレビュー依頼
        reviewer_eval = eval_df[eval_df[reviewer_col] == reviewer]

        # 評価期間にレビュー依頼を受けていない場合、拡張期間をチェック
        if len(reviewer_eval) == 0:
            # 拡張期間のレビュー依頼をチェック
            reviewer_extended_eval = extended_eval_df[extended_eval_df[reviewer_col] == reviewer]

            if len(reviewer_extended_eval) == 0:
                # 拡張期間にも依頼がない → 訓練期間末尾までアサインがない → 除外（実質離脱者）
                skipped_no_requests_until_end += 1
                continue  # このレビュアーをスキップ
            
            # 拡張期間に依頼がある → 訓練期間中に指定期間を超えて再びアサインされる可能性 → 重み付き負例
            future_acceptance = False  # この期間では活動なし
            accepted_requests = pd.DataFrame()  # 空
            rejected_requests = pd.DataFrame()  # 空（依頼自体がない）
            had_requests = False  # この期間に依頼がなかった
            sample_weight = 0.1  # 非常に低い重み（依頼なし）

            # 統計カウント
            negative_count += 1
            negative_without_requests += 1
        else:
            # 通常の評価期間に依頼がある場合
            # 継続判定：評価期間内に少なくとも1つのレビュー依頼を承諾したか
            accepted_requests = reviewer_eval[reviewer_eval[label_col] == 1]
            rejected_requests = reviewer_eval[reviewer_eval[label_col] == 0]
            future_acceptance = len(accepted_requests) > 0
            had_requests = True  # この期間に依頼があった
            sample_weight = 1.0  # 通常の重み（依頼あり）

            if future_acceptance:
                positive_count += 1
            else:
                negative_count += 1
                negative_with_requests += 1

        # 活動履歴を構築
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col],  # タイムスタンプは常にdate_col
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'request_time': row.get('request_time', row[date_col]),
                'response_time': row.get('first_response_time'),  # response_time計算用
                'accepted': row.get(label_col, 0) == 1,
                # IRL特徴量計算用のデータを追加
                'files_changed': row.get('change_files_count', 0),  # 強度計算用
                'change_files_count': row.get('change_files_count', 0),  # 強度計算用
                'lines_added': row.get('change_insertions', 0),  # 規模計算用
                'lines_deleted': row.get('change_deletions', 0),  # 規模計算用
                'change_insertions': row.get('change_insertions', 0),  # 規模計算用
                'change_deletions': row.get('change_deletions', 0),  # 規模計算用
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
            'developer': developer_info,
            'activity_history': activity_history,
            'context_date': cutoff_date,
            'future_acceptance': future_acceptance,
            'reviewer': reviewer,
            'history_request_count': len(reviewer_history),
            'history_accepted_count': len(reviewer_history[reviewer_history[label_col] == 1]),
            'history_rejected_count': len(reviewer_history[reviewer_history[label_col] == 0]),
            'eval_request_count': len(reviewer_eval),
            'eval_accepted_count': len(accepted_requests),
            'eval_rejected_count': len(rejected_requests),
            'had_requests': had_requests,  # この期間に依頼があったか
            'sample_weight': sample_weight  # サンプル重み（依頼なし=0.3、依頼あり=1.0）
        }
        
        trajectories.append(trajectory)
    
    logger.info("=" * 80)
    logger.info(f"評価用軌跡抽出完了: {len(trajectories)}サンプル")
    logger.info(f"  スキップ（最小依頼数未満）: {skipped_min_requests}")
    logger.info(f"  スキップ（訓練期間末尾まで依頼なし）: {skipped_no_requests_until_end}")
    if trajectories:
        logger.info(f"  正例（この期間で承諾あり）: {positive_count} ({positive_count/len(trajectories)*100:.1f}%)")
        logger.info(f"  負例（この期間で承諾なし）: {negative_count} ({negative_count/len(trajectories)*100:.1f}%)")
        if negative_with_requests > 0:
            logger.info(f"    - 依頼あり→拒否（重み=1.0）: {negative_with_requests}")
        if negative_without_requests > 0:
            logger.info(f"    - 依頼なし（拡張期間に依頼あり、重み=0.1）: {negative_without_requests}")
    logger.info("=" * 80)
    
    return trajectories


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    最適な閾値を探索
    
    Args:
        y_true: 真のラベル
        y_pred: 予測確率
        
    Returns:
        最適閾値と各メトリクス
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return {
        'threshold': float(best_threshold),
        'precision': float(precision[best_idx]),
        'recall': float(recall[best_idx]),
        'f1': float(f1_scores[best_idx])
    }


def main():
    parser = argparse.ArgumentParser(
        description="レビュー承諾予測IRL学習（正しいロジック版）"
    )
    parser.add_argument(
        "--reviews",
        type=str,
        default="data/review_requests_openstack_multi_5y_detail.csv",
        help="レビュー依頼CSVファイルのパス"
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="2021-01-01",
        help="訓練開始日 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2023-01-01",
        help="訓練終了日 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--eval-start",
        type=str,
        default="2023-01-01",
        help="評価開始日 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--eval-end",
        type=str,
        default="2024-01-01",
        help="評価終了日 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--future-window-start",
        type=int,
        default=0,
        help="将来窓開始（月数）"
    )
    parser.add_argument(
        "--future-window-end",
        type=int,
        default=3,
        help="将来窓終了（月数）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="訓練エポック数"
    )
    parser.add_argument(
        "--min-history-events",
        type=int,
        default=3,
        help="最小履歴イベント数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/review_acceptance_irl",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="プロジェクト名（単一プロジェクトのみ）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="既存モデルのパス（評価のみの場合）"
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # レビュー依頼データを読み込み
    df = load_review_requests(args.reviews)
    
    # 日付をパース
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    
    logger.info(f"将来窓: {args.future_window_start}～{args.future_window_end}ヶ月")
    
    # 訓練用軌跡を抽出
    if args.model is None:
        logger.info("訓練用軌跡を抽出...")
        train_trajectories = extract_review_acceptance_trajectories(
            df,
            train_start=train_start,
            train_end=train_end,
            future_window_start_months=args.future_window_start,
            future_window_end_months=args.future_window_end,
            min_history_requests=args.min_history_events,
            project=args.project
        )
        
        if not train_trajectories:
            logger.error("訓練用軌跡が抽出できませんでした")
            return
        
        # IRLシステムを初期化
        # バランスの取れた設定：
        # - hidden_dim=128: 適度な表現力（256だと過剰）
        # - dropout=0.2: 適度な正則化（0.0だと過学習、0.1だと不十分）
        # - learning_rate=0.0001: やや高めで局所最適を回避
        config = {
            'state_dim': 10,  # 最近の受諾率+レビュー負荷を追加
            'action_dim': 4,
            'hidden_dim': 128,  # 安定した表現力
            'sequence': True,
            'seq_len': 0,
            'learning_rate': 0.0001,  # 局所最適回避
            'dropout': 0.2,  # 適度な正則化
        }
        irl_system = RetentionIRLSystem(config)
        
        # 訓練データの正例率を計算して Focal Loss を自動調整
        positive_count = sum(1 for t in train_trajectories if t['future_acceptance'])
        positive_rate = positive_count / len(train_trajectories)
        logger.info(f"訓練データ正例率: {positive_rate:.1%} ({positive_count}/{len(train_trajectories)})")
        
        irl_system.auto_tune_focal_loss(positive_rate)
        
        # 訓練
        logger.info("IRLモデルを訓練...")
        irl_system.train_irl_temporal_trajectories(
            train_trajectories,
            epochs=args.epochs
        )
        
        # 訓練データ上で最適閾値を決定
        logger.info("訓練データ上で最適閾値を決定...")
        train_y_true = []
        train_y_pred = []
        
        for traj in train_trajectories:
            # キー名を確認して適切に処理
            developer = traj.get('developer', traj.get('developer_info', {}))
            result = irl_system.predict_continuation_probability_snapshot(
                developer,
                traj['activity_history'],
                traj['context_date']
            )
            train_y_true.append(1 if traj['future_acceptance'] else 0)
            train_y_pred.append(result['continuation_probability'])
        
        train_y_true = np.array(train_y_true)
        train_y_pred = np.array(train_y_pred)

        # F1スコアを最大化する閾値を訓練データで決定
        positive_rate = train_y_true.mean()
        logger.info(f"訓練データ正例率: {positive_rate:.1%}")

        # find_optimal_threshold を使用してF1スコアを最大化する閾値を探索
        train_optimal_threshold_info = find_optimal_threshold(train_y_true, train_y_pred)
        train_optimal_threshold = train_optimal_threshold_info['threshold']
        train_optimal_threshold_info['positive_rate'] = float(positive_rate)
        train_optimal_threshold_info['method'] = 'f1_maximization_on_train_data'

        logger.info(f"F1最大化閾値（訓練データ）: {train_optimal_threshold:.4f}")
        logger.info(f"訓練データ性能: Precision={train_optimal_threshold_info['precision']:.3f}, Recall={train_optimal_threshold_info['recall']:.3f}, F1={train_optimal_threshold_info['f1']:.3f}")
        
        # 訓練データの予測確率分布も保存（デバッグ用）
        train_optimal_threshold_info['train_prediction_stats'] = {
            'min': float(train_y_pred.min()),
            'max': float(train_y_pred.max()),
            'mean': float(train_y_pred.mean()),
            'std': float(train_y_pred.std()),
            'median': float(np.median(train_y_pred))
        }
        logger.info(f"訓練データ予測確率: [{train_optimal_threshold_info['train_prediction_stats']['min']:.4f}, {train_optimal_threshold_info['train_prediction_stats']['max']:.4f}]")
        
        # モデルを保存
        model_path = output_dir / "irl_model.pt"
        torch.save(irl_system.network.state_dict(), model_path)
        logger.info(f"モデルを保存: {model_path}")
        
        # 閾値を保存
        threshold_path = output_dir / "optimal_threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump(train_optimal_threshold_info, f, indent=2)
        logger.info(f"最適閾値を保存: {threshold_path}")
    else:
        # 既存モデルを読み込み
        logger.info(f"既存モデルを読み込み: {args.model}")
        config = {
            'state_dim': 10,  # 最近の受諾率+レビュー負荷を追加
            'action_dim': 4,
            'hidden_dim': 128,  # 訓練時と同じ設定
            'sequence': True,
            'seq_len': 0,
            'dropout': 0.2
        }
        irl_system = RetentionIRLSystem(config)
        irl_system.network.load_state_dict(torch.load(args.model))
        irl_system.network.eval()
        
        # 保存された閾値を読み込み
        threshold_path = Path(args.model).parent / "optimal_threshold.json"
        if threshold_path.exists():
            with open(threshold_path) as f:
                train_optimal_threshold_info = json.load(f)
                train_optimal_threshold = train_optimal_threshold_info['threshold']
            logger.info(f"保存された閾値を読み込み: {train_optimal_threshold:.4f}")
        else:
            train_optimal_threshold = 0.5
            logger.warning(f"閾値ファイルが見つからないため、デフォルト値 {train_optimal_threshold:.4f} を使用")
    
    # グローバル変数として閾値を設定
    global optimal_threshold_from_training
    if 'train_optimal_threshold' in locals():
        optimal_threshold_from_training = train_optimal_threshold
    else:
        optimal_threshold_from_training = 0.5
    
    # 評価用軌跡を抽出
    logger.info("評価用軌跡を抽出...")
    history_window_months = int((train_end - train_start).days / 30)
    
    # future_window_start_monthsとfuture_window_end_monthsを使用
    # これらは--future-window-startと--future-window-endから来る
    eval_trajectories = extract_evaluation_trajectories(
        df,
        cutoff_date=train_end,
        history_window_months=history_window_months,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        min_history_requests=args.min_history_events,
        project=args.project
    )
    
    if not eval_trajectories:
        logger.error("評価用軌跡が抽出できませんでした")
        return
    
    # 予測
    logger.info("予測を実行...")
    y_true = []
    y_pred = []
    predictions = []
    
    for traj in eval_trajectories:
        # スナップショット特徴量で予測
        result = irl_system.predict_continuation_probability_snapshot(
            traj['developer'],
            traj['activity_history'],
            traj['context_date']
        )
        prob = result['continuation_probability']
        true_label = 1 if traj['future_acceptance'] else 0
        
        y_true.append(true_label)
        y_pred.append(prob)
        
        predictions.append({
            'reviewer_email': traj['reviewer'],
            'predicted_prob': float(prob),
            'true_label': true_label,
            'history_request_count': traj['history_request_count'],
            'history_acceptance_rate': traj['developer']['acceptance_rate'],
            'eval_request_count': traj['eval_request_count'],
            'eval_accepted_count': traj['eval_accepted_count'],
            'eval_rejected_count': traj['eval_rejected_count']
        })
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # メトリクスを計算
    logger.info("メトリクスを計算...")

    # 訓練データで決定した閾値を使用（データリーク防止）
    optimal_threshold = train_optimal_threshold
    logger.info(f"訓練データで決定した閾値を使用: {optimal_threshold:.4f}")

    # 参考：評価データ上での最適閾値も計算（比較用）
    eval_optimal_threshold_info = find_optimal_threshold(y_true, y_pred)
    logger.info(f"参考：評価データ上での最適閾値: {eval_optimal_threshold_info['threshold']:.4f} (F1={eval_optimal_threshold_info['f1']:.3f})")

    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    auc_roc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    
    precision_at_threshold = precision_score(y_true, y_pred_binary)
    recall_at_threshold = recall_score(y_true, y_pred_binary)
    f1_at_threshold = f1_score(y_true, y_pred_binary)
    
    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'optimal_threshold': float(optimal_threshold),
        'threshold_source': 'train_data',  # 訓練データで決定した閾値を使用
        'precision': float(precision_at_threshold),
        'recall': float(recall_at_threshold),
        'f1_score': float(f1_at_threshold),
        'positive_count': int(y_true.sum()),
        'negative_count': int((1 - y_true).sum()),
        'total_count': int(len(y_true)),
        # 参考情報：評価データでの最適閾値
        'eval_optimal_threshold': float(eval_optimal_threshold_info['threshold']),
        'eval_optimal_f1': float(eval_optimal_threshold_info['f1']),
        # 予測確率の分布統計
        'prediction_stats': {
            'min': float(y_pred.min()),
            'max': float(y_pred.max()),
            'mean': float(y_pred.mean()),
            'std': float(y_pred.std()),
            'median': float(np.median(y_pred))
        }
    }
    
    logger.info("=" * 80)
    logger.info("評価結果:")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    logger.info(f"  最適閾値（訓練データ決定）: {metrics['optimal_threshold']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  正例数: {metrics['positive_count']}")
    logger.info(f"  負例数: {metrics['negative_count']}")
    logger.info("---")
    logger.info("予測確率の分布:")
    logger.info(f"  範囲: [{metrics['prediction_stats']['min']:.4f}, {metrics['prediction_stats']['max']:.4f}]")
    logger.info(f"  平均: {metrics['prediction_stats']['mean']:.4f}")
    logger.info(f"  標準偏差: {metrics['prediction_stats']['std']:.4f}")
    logger.info(f"  中央値: {metrics['prediction_stats']['median']:.4f}")
    logger.info("=" * 80)
    
    # 結果を保存
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df['predicted_binary'] = y_pred_binary
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # 評価軌跡データを保存（特徴量重要度分析用）
    import pickle
    trajectories_path = output_dir / "eval_trajectories.pkl"
    with open(trajectories_path, 'wb') as f:
        pickle.dump(eval_trajectories, f)
    logger.info(f"評価軌跡を保存: {trajectories_path}")
    
    logger.info(f"結果を保存: {output_dir}")


if __name__ == "__main__":
    main()

