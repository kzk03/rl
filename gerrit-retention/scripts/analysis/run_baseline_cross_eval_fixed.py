#!/usr/bin/env python3
"""
Baseline (RF & LR) Cross-Evaluation - 期間対応版

IRLと同じロジックで期間ごとのラベルを計算：
- 訓練期間末尾から将来窓での承諾有無をラベル化
- 評価期間末尾から将来窓での承諾有無をラベル化
- レビュアー単位で集約（最後の依頼のみ使用）
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_review_requests(csv_path: str, project: str = None) -> pd.DataFrame:
    """レビュー依頼データを読み込む"""
    df = pd.read_csv(csv_path)
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    if project:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクト '{project}' でフィルタ: {len(df)} リクエスト")
    
    logger.info(f"総リクエスト数: {len(df)}")
    logger.info(f"承諾: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")
    logger.info(f"拒否: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    
    return df


def create_reviewer_features_and_labels(
    reviews_df: pd.DataFrame,
    period_start: datetime,
    period_end: datetime,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history_events: int = 3
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    期間内の各レビュアーについて特徴量とラベルを作成
    
    ラベル: period_end時点から future_window 内に承諾があったか？
    特徴量: period_end時点までの最新のレビュー依頼の特徴量
    """
    # 期間内のレビュー依頼
    period_requests = reviews_df[
        (reviews_df['request_time'] >= period_start) &
        (reviews_df['request_time'] < period_end)
    ].copy()
    
    if len(period_requests) == 0:
        return np.array([]), np.array([]), []
    
    # レビュアーごとにグループ化
    reviewer_groups = period_requests.groupby('reviewer_email')
    
    # レビュアーごとの履歴イベント数
    reviewer_counts = reviewer_groups.size()
    valid_reviewers = reviewer_counts[reviewer_counts >= min_history_events].index
    
    if len(valid_reviewers) == 0:
        return np.array([]), np.array([]), []
    
    # 各レビュアーの最新レビュー依頼を取得（特徴量用）
    latest_requests = period_requests[
        period_requests['reviewer_email'].isin(valid_reviewers)
    ].sort_values('request_time').groupby('reviewer_email').tail(1)
    
    # 将来窓の定義
    future_start = period_end + timedelta(days=30 * future_window_start_months)
    future_end = period_end + timedelta(days=30 * future_window_end_months)
    
    # 各レビュアーの将来窓内での承諾有無をラベルとして計算
    future_requests = reviews_df[
        (reviews_df['request_time'] >= future_start) &
        (reviews_df['request_time'] < future_end)
    ]
    
    # レビュアーごとに将来窓内で承諾があったかチェック
    future_acceptance = future_requests[future_requests['label'] == 1].groupby('reviewer_email').size()
    
    labels = []
    features = []
    reviewer_emails = []
    
    # IRLの状態10次元に対応した特徴量（状態のみ、行動は含まない）
    # DeveloperState: experience_days, total_changes, total_reviews, 
    #                 recent_activity_frequency, avg_activity_gap, activity_trend,
    #                 collaboration_score, code_quality_score, 
    #                 recent_acceptance_rate, review_load
    feature_columns = [
        # 1. experience_days
        'reviewer_tenure_days',
        # 2-3. total_changes, total_reviews (近似: 長期活動量)
        'reviewer_past_reviews_180d',
        # 4. recent_activity_frequency (近似: 短期活動量)
        'reviewer_past_reviews_30d',
        # 5. avg_activity_gap (近似: 中期活動量から逆算)
        'reviewer_past_reviews_90d',
        # 6. activity_trend は計算不可のため除外
        # 7-8. collaboration_score, code_quality_score (近似: response rate)
        'reviewer_past_response_rate_180d',
        # 9. recent_acceptance_rate は直接データなし
        # 10. review_load
        'reviewer_assignment_load_30d',
    ]
    
    available_features = [col for col in feature_columns if col in latest_requests.columns]
    
    if len(available_features) == 0:
        logger.warning("利用可能な特徴量がないため、簡易特徴量を使用")
        # レビュー数のみ使用
        for reviewer in valid_reviewers:
            reviewer_data = latest_requests[latest_requests['reviewer_email'] == reviewer]
            if len(reviewer_data) > 0:
                features.append([reviewer_counts[reviewer]])
                labels.append(1 if reviewer in future_acceptance.index else 0)
                reviewer_emails.append(reviewer)
    else:
        for reviewer in valid_reviewers:
            reviewer_data = latest_requests[latest_requests['reviewer_email'] == reviewer]
            if len(reviewer_data) > 0:
                feature_values = reviewer_data[available_features].fillna(0).values[0]
                features.append(feature_values)
                labels.append(1 if reviewer in future_acceptance.index else 0)
                reviewer_emails.append(reviewer)
    
    return np.array(features), np.array(labels), reviewer_emails


def train_and_evaluate(
    reviews_df: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    train_future_start: int,
    train_future_end: int,
    eval_start: datetime,
    eval_end: datetime,
    eval_future_start: int,
    eval_future_end: int,
    model_type: str,
    min_history_events: int
) -> Dict[str, Any]:
    """モデルを訓練して評価"""
    
    # 訓練データの特徴量とラベルを作成
    X_train, y_train, train_reviewers = create_reviewer_features_and_labels(
        reviews_df, train_start, train_end,
        train_future_start, train_future_end,
        min_history_events
    )
    
    if len(X_train) == 0:
        logger.warning("訓練データが不足")
        return None
    
    logger.info(f"    訓練: {len(X_train)} レビュアー, 正例率 {y_train.mean()*100:.1f}%")
    
    # モデル訓練
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=777, n_jobs=-1, class_weight='balanced')
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=777, max_iter=1000, class_weight='balanced')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # 評価データの特徴量とラベルを作成
    X_eval, y_eval, eval_reviewers = create_reviewer_features_and_labels(
        reviews_df, eval_start, eval_end,
        eval_future_start, eval_future_end,
        min_history_events
    )
    
    if len(X_eval) == 0:
        logger.warning("評価データが不足")
        return None
    
    logger.info(f"    評価: {len(X_eval)} レビュアー, 正例率 {y_eval.mean()*100:.1f}%")
    
    # 予測
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    
    # メトリクス計算
    try:
        auc_roc = roc_auc_score(y_eval, y_pred_proba)
    except:
        auc_roc = float('nan')
    
    try:
        precision, recall, _ = precision_recall_curve(y_eval, y_pred_proba)
        auc_pr = auc(recall, precision)
    except:
        auc_pr = float('nan')
    
    # 最適閾値でF1計算
    fpr, tpr, thresholds = roc_curve(y_eval, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'optimal_threshold': float(optimal_threshold),
        'precision': precision_score(y_eval, y_pred, zero_division=0),
        'recall': recall_score(y_eval, y_pred, zero_division=0),
        'f1_score': f1_score(y_eval, y_pred, zero_division=0),
        'n_train': len(X_train),
        'n_eval': len(X_eval),
        'positive_count': int((y_eval == 1).sum()),
        'negative_count': int((y_eval == 0).sum()),
    }


def run_cross_evaluation(
    reviews_df: pd.DataFrame,
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    model_type: str,
    output_dir: Path,
    min_history_events: int = 3
):
    """クロス評価を実行"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 期間定義
    label_periods = [
        ('0-3m', 0, 3),
        ('3-6m', 3, 6),
        ('6-9m', 6, 9),
        ('9-12m', 9, 12),
    ]
    
    results = {}
    
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    eval_start_dt = pd.to_datetime(eval_start)
    eval_end_dt = pd.to_datetime(eval_end)
    
    for train_period_name, train_future_start, train_future_end in label_periods:
        logger.info(f"\n{'='*80}")
        logger.info(f"訓練期間: {train_period_name} (将来窓: {train_future_start}-{train_future_end}ヶ月)")
        logger.info(f"{'='*80}")
        
        for eval_period_name, eval_future_start, eval_future_end in label_periods:
            logger.info(f"  評価期間: {eval_period_name} (将来窓: {eval_future_start}-{eval_future_end}ヶ月)")
            
            # 訓練と評価
            metrics = train_and_evaluate(
                reviews_df,
                train_start_dt, train_end_dt,
                train_future_start, train_future_end,
                eval_start_dt, eval_end_dt,
                eval_future_start, eval_future_end,
                model_type,
                min_history_events
            )
            
            if metrics:
                key = f"{train_period_name}_{eval_period_name}"
                results[key] = metrics
                
                logger.info(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
                logger.info(f"    AUC-PR: {metrics['auc_pr']:.4f}")
                logger.info(f"    F1: {metrics['f1_score']:.4f}")
    
    # 結果を保存
    results_file = output_dir / f"{model_type}_cross_eval_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n結果を保存: {results_file}")
    
    # マトリクスCSVを作成
    create_matrix_csv(results, label_periods, output_dir, model_type)


def create_matrix_csv(results: dict, periods: list, output_dir: Path, model_type: str):
    """メトリクスのマトリクスCSVを作成"""
    period_names = [p[0] for p in periods]
    
    for metric_name in ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall']:
        matrix = []
        for train_period in period_names:
            row = []
            for eval_period in period_names:
                key = f"{train_period}_{eval_period}"
                if key in results and metric_name in results[key]:
                    value = results[key][metric_name]
                    row.append(f"{value:.4f}" if not np.isnan(value) else "NaN")
                else:
                    row.append("NaN")
            matrix.append(row)
        
        # CSV保存
        df = pd.DataFrame(matrix, index=period_names, columns=period_names)
        csv_file = output_dir / f"{model_type}_matrix_{metric_name.upper()}.csv"
        df.to_csv(csv_file)
        logger.info(f"マトリクス保存: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='Baseline Cross-Evaluation (期間対応版)')
    parser.add_argument('--reviews', required=True, help='レビューCSVファイル')
    parser.add_argument('--train-start', default='2021-01-01', help='訓練開始日')
    parser.add_argument('--train-end', default='2023-01-01', help='訓練終了日')
    parser.add_argument('--eval-start', default='2023-01-01', help='評価開始日')
    parser.add_argument('--eval-end', default='2024-01-01', help='評価終了日')
    parser.add_argument('--model', choices=['random_forest', 'logistic_regression', 'both'], 
                       default='both', help='使用するモデル')
    parser.add_argument('--min-history-events', type=int, default=3, 
                       help='最小履歴イベント数')
    parser.add_argument('--project', help='単一プロジェクトに限定')
    parser.add_argument('--output', required=True, help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Baseline Cross-Evaluation (RF & LR) - 期間対応版")
    logger.info("="*80)
    logger.info(f"レビューデータ: {args.reviews}")
    logger.info(f"訓練期間: {args.train_start} to {args.train_end}")
    logger.info(f"評価期間: {args.eval_start} to {args.eval_end}")
    logger.info(f"最小履歴イベント数: {args.min_history_events}")
    logger.info(f"プロジェクト: {args.project or '全プロジェクト'}")
    logger.info("="*80)
    
    # データ読み込み
    reviews_df = load_review_requests(args.reviews, args.project)
    
    output_dir = Path(args.output)
    
    # モデル実行
    models = ['random_forest', 'logistic_regression'] if args.model == 'both' else [args.model]
    
    for model_type in models:
        logger.info(f"\n{'#'*80}")
        logger.info(f"モデル: {model_type.upper()}")
        logger.info(f"{'#'*80}")
        
        model_output_dir = output_dir / model_type
        run_cross_evaluation(
            reviews_df,
            args.train_start,
            args.train_end,
            args.eval_start,
            args.eval_end,
            model_type,
            model_output_dir,
            args.min_history_events
        )
    
    logger.info(f"\n{'='*80}")
    logger.info("✓ 全ての処理が完了しました！")
    logger.info(f"結果: {output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
