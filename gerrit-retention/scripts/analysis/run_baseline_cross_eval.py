#!/usr/bin/env python3
"""
Baseline (RF & LR) Cross-Evaluation - IRL実験条件に合わせた版

IRLと同じ条件でベースライン（Random Forest, Logistic Regression）を実行：
- 固定の訓練期間: 2021-01-01 to 2023-01-01
- 4つのラベル期間: 0-3m, 3-6m, 6-9m, 9-12m
- min-history-events: 3 (ベースライン版) または 0 (強化版)
- project: openstack/nova (ベースライン版) または全プロジェクト (強化版)

出力: outputs/baseline_cross_eval_[baseline|enhanced]/
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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


def extract_features(df: pd.DataFrame, min_history_events: int = 3) -> tuple:
    """
    特徴量抽出（リクエスト単位）
    
    各リクエストごとに、そのレビュアーの過去の統計を特徴量として使用：
    - reviewer_past_reviews_30d: 過去30日のレビュー数
    - reviewer_past_reviews_90d: 過去90日のレビュー数
    - reviewer_past_reviews_180d: 過去180日のレビュー数
    - reviewer_past_response_rate_180d: 過去180日の応答率
    - reviewer_tenure_days: レビュアーの在籍日数
    など、データに含まれる特徴量を使用
    """
    # データをリクエスト時刻でソート
    df = df.sort_values('request_time').reset_index(drop=True)
    
    # レビュアーごとの履歴イベント数をカウント
    reviewer_counts = df.groupby('reviewer_email').cumcount()
    
    # min_history_events以上のリクエストのみ使用
    mask = reviewer_counts >= min_history_events
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        return np.array([]), np.array([])
    
    # 特徴量カラムを選択（数値型のみ）
    feature_columns = [
        'reviewer_past_reviews_30d',
        'reviewer_past_reviews_90d', 
        'reviewer_past_reviews_180d',
        'reviewer_past_response_rate_180d',
        'reviewer_tenure_days',
        'owner_tenure_days',
        'change_insertions',
        'change_deletions',
        'change_files_count',
        'reviewer_assignment_load_7d',
        'reviewer_assignment_load_30d',
        'reviewer_assignment_load_180d',
    ]
    
    # 利用可能な特徴量のみ使用
    available_features = [col for col in feature_columns if col in filtered_df.columns]
    
    if len(available_features) == 0:
        # 特徴量がない場合は簡易的な特徴量を作成
        logger.warning("利用可能な特徴量がないため、簡易特徴量を使用")
        X = filtered_df[['reviewer_email']].copy()
        # レビュアーごとの通し番号を特徴量として使用（最低限の情報）
        X['reviewer_count'] = filtered_df.groupby('reviewer_email').cumcount()
        X = X[['reviewer_count']].values
    else:
        X = filtered_df[available_features].fillna(0).values
    
    y = filtered_df['label'].values
    
    return X, y


def train_and_evaluate(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_type: str,
    min_history_events: int
) -> Dict[str, Any]:
    """モデルを訓練して評価"""
    
    # 訓練データから特徴量抽出
    X_train, y_train = extract_features(train_df, min_history_events)
    
    if len(X_train) == 0:
        logger.warning("訓練データが不足")
        return None
    
    # モデル訓練
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=777, n_jobs=-1)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=777, max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # 評価データから特徴量抽出
    X_eval, y_eval = extract_features(eval_df, min_history_events)
    
    if len(X_eval) == 0:
        logger.warning("評価データが不足")
        return None
    
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
    
    for train_period_name, train_future_start, train_future_end in label_periods:
        logger.info(f"\n{'='*80}")
        logger.info(f"訓練期間: {train_period_name}")
        logger.info(f"{'='*80}")
        
        # 訓練データ準備
        train_start_ts = pd.to_datetime(train_start)
        train_end_ts = pd.to_datetime(train_end)
        
        train_df = reviews_df[
            (reviews_df['request_time'] >= train_start_ts) &
            (reviews_df['request_time'] < train_end_ts)
        ].copy()
        
        for eval_period_name, eval_future_start, eval_future_end in label_periods:
            logger.info(f"  評価期間: {eval_period_name}")
            
            # 評価データ準備
            eval_start_ts = pd.to_datetime(eval_start)
            eval_end_ts = pd.to_datetime(eval_end)
            
            eval_df = reviews_df[
                (reviews_df['request_time'] >= eval_start_ts) &
                (reviews_df['request_time'] < eval_end_ts)
            ].copy()
            
            # 訓練と評価
            metrics = train_and_evaluate(
                train_df, eval_df, model_type, min_history_events
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
    parser = argparse.ArgumentParser(description='Baseline Cross-Evaluation')
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
    logger.info("Baseline Cross-Evaluation (RF & LR)")
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
