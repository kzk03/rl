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
    スライディングウィンドウ版：全シーケンス＋月次集約ラベル付き軌跡を抽出
    
    重要な違い：
    - future_window_start_months > 0 の場合、その期間は除外
    - 例: 3-6m の場合、0-3m に活動があっても False とする
    
    特徴：
    - **サンプリングなし**：各レビュアーから1サンプルのみ
    - **全シーケンス**：学習期間内の全活動履歴を使用
    - **ラベル**：月次集約（各月末からスライディングウィンドウ内に活動があるか）
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
    logger.info("スライディングウィンドウ版：全シーケンス＋月次集約ラベル付き軌跡抽出を開始")
    logger.info(f"学習期間: {train_start} ～ {train_end}")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月（スライディング）")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    logger.info("全シーケンス: 各レビュアーの学習期間内の全活動を使用")
    logger.info("ラベル: 月次集約（各月末からスライディング窓内に活動があるか）")
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
        
        # 月ごとにラベルを計算
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
                if future_end > train_end:
                    monthly_labels[month_key] = None
                else:
                    # この月からスライディング窓内に活動があるか（履歴期間内のプロジェクトのみ）
                    future_activities = reviewer_all_activities[
                        (reviewer_all_activities[date_col] >= future_start) &
                        (reviewer_all_activities[date_col] < future_end) &
                        (reviewer_all_activities['project'].isin(history_projects))
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
    eval_trajectories: List[Dict[str, Any]]
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    モデルを評価
    
    Args:
        irl_system: 訓練済みIRLシステム
        eval_trajectories: 評価用軌跡データ
        
    Returns:
        (評価メトリクス, 予測詳細のリスト)
    """
    logger.info("=" * 80)
    logger.info("モデル評価開始")
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
        
        # 予測確率を取得
        prob = irl_system.predict_continuation_probability(trajectory)
        
        # 真のラベルを取得
        true_label = trajectory.get('future_contribution', False)
        
        predictions.append(prob)
        true_labels.append(true_label)
        
        # 予測詳細を記録
        prediction_details.append({
            'reviewer_email': reviewer_email,
            'predicted_prob': float(prob),
            'true_label': int(true_label),
            'activity_count': activity_count
        })
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # 継続率
    continuation_rate = np.mean(true_labels)
    logger.info(f"評価データの継続率: {continuation_rate:.1%}")
    
    # メトリクス計算
    if len(np.unique(true_labels)) < 2:
        logger.warning("評価データに正例または負例のみが含まれています")
        metrics = {
            'auc_roc': 0.5,
            'auc_pr': continuation_rate,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'optimal_threshold': 0.5,
            'best_f1': 0.0,
            'continuation_rate': continuation_rate
        }
        return metrics, prediction_details
    
    # AUC-ROC
    auc_roc = roc_auc_score(true_labels, predictions)
    
    # Precision-Recall曲線とAUC-PR
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, predictions)
    auc_pr = auc(recall_vals, precision_vals)
    
    # 最適閾値を見つける
    best_threshold, best_metrics = find_optimal_threshold(true_labels, predictions)
    
    metrics = {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'optimal_threshold': best_threshold,
        'best_f1': float(best_metrics['f1']) if best_metrics else 0.0,
        'continuation_rate': continuation_rate
    }
    
    logger.info("=" * 80)
    logger.info("評価結果:")
    logger.info(f"  AUC-ROC: {auc_roc:.3f}")
    logger.info(f"  AUC-PR: {auc_pr:.3f}")
    logger.info(f"  最適閾値: {best_threshold:.3f}")
    logger.info(f"  Precision: {best_metrics['precision']:.3f}")
    logger.info(f"  Recall: {best_metrics['recall']:.3f}")
    logger.info(f"  F1スコア: {best_metrics['f1']:.3f}")
    logger.info("=" * 80)
    
    return metrics, prediction_details


def main():
    parser = argparse.ArgumentParser(description='スライディングウィンドウIRL訓練')
    parser.add_argument('--reviews', type=str, required=True,
                        help='レビューデータ（CSV）のパス')
    parser.add_argument('--train-start', type=str, required=True,
                        help='学習開始日（YYYY-MM-DD）')
    parser.add_argument('--train-end', type=str, required=True,
                        help='学習終了日（YYYY-MM-DD）')
    parser.add_argument('--eval-start', type=str, required=True,
                        help='評価開始日（YYYY-MM-DD）')
    parser.add_argument('--eval-end', type=str, required=True,
                        help='評価終了日（YYYY-MM-DD）')
    parser.add_argument('--future-window-start', type=int, required=True,
                        help='将来窓開始（月数）')
    parser.add_argument('--future-window-end', type=int, required=True,
                        help='将来窓終了（月数）')
    parser.add_argument('--eval-future-window-start', type=int, default=None,
                        help='評価用将来窓開始（月数、指定なしの場合は訓練と同じ）')
    parser.add_argument('--eval-future-window-end', type=int, default=None,
                        help='評価用将来窓終了（月数、指定なしの場合は訓練と同じ）')
    parser.add_argument('--epochs', type=int, default=20,
                        help='エポック数')
    parser.add_argument('--min-history-events', type=int, default=1,
                        help='最小活動回数')
    parser.add_argument('--output', type=str, default='outputs/irl_sliding',
                        help='出力ディレクトリ')
    parser.add_argument('--project', type=str, default=None,
                        help='プロジェクト名（指定時は単一プロジェクトのみ）')
    parser.add_argument('--model', type=str, default=None,
                        help='既存モデルのパス（評価のみの場合）')
    
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
    
    # モデル設定
    config = {
        'state_dim': 10,
        'action_dim': 5,
        'hidden_dim': 128,
        'learning_rate': 0.0001,
        'sequence': True,
        'seq_len': 0,  # 可変長
    }
    
    # モデルの訓練または読み込み
    if args.model and Path(args.model).exists():
        # 既存モデルをロード
        logger.info(f"既存モデルをロード: {args.model}")
        irl_system = RetentionIRLSystem(config)
        irl_system.load_model(args.model)
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
    
    # モデル評価
    metrics, prediction_details = evaluate_model(irl_system, eval_trajectories)
    
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
