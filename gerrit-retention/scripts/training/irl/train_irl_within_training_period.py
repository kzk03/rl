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


def evaluate_irl_model(
    irl_system: RetentionIRLSystem,
    test_trajectories: List[Dict[str, Any]]
) -> Dict[str, float]:
    """IRLモデルを評価"""
    logger.info("=" * 80)
    logger.info("IRLモデル評価開始")
    logger.info(f"テストサンプル数: {len(test_trajectories)}")
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
    
    # 2値予測（閾値0.5）
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
    
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
    
    try:
        metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    except:
        metrics['f1'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
    
    metrics['test_samples'] = len(test_trajectories)
    positive_count = sum(y_true)
    metrics['positive_samples'] = positive_count
    metrics['positive_rate'] = positive_count / len(y_true) if y_true else 0
    
    logger.info("評価結果:")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    logger.info(f"  AUC-PR: {metrics['auc_pr']:.3f}")
    logger.info(f"  F1: {metrics['f1']:.3f}")
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

