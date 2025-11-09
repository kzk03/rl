#!/usr/bin/env python3
"""
ベースライン（RF/LR）クロス評価 - IRL完全準拠版

IRLの特徴量計算ロジックを完全移植:
- 状態10次元: retention_irl_system.py の extract_developer_state() と同一
- 行動は使用しない（レビュー依頼データのため行動は不要）
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
from sklearn.metrics import classification_report, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_irl_state_features(
    reviewer_email: str,
    period_requests: pd.DataFrame,
    context_date: datetime
) -> np.ndarray:
    """
    IRLの状態10次元を完全再現
    
    DeveloperState:
    1. experience_days: 最初の活動から現在までの日数
    2. total_changes: 総変更数（レビュー依頼数で近似）
    3. total_reviews: 総レビュー数
    4. recent_activity_frequency: 直近30日の活動頻度（回/日）
    5. avg_activity_gap: 平均活動間隔（日）
    6. activity_trend: 活動トレンド（increasing=1.0, stable=0.5, decreasing=0.0）
    7. collaboration_score: 協力スコア（全活動に対するレビュー活動の割合）
    8. code_quality_score: コード品質スコア（簡易版: 0.5固定）
    9. recent_acceptance_rate: 直近30日のレビュー受諾率（データなし: 0.5）
    10. review_load: レビュー負荷（直近30日 / 全期間平均）
    """
    reviewer_data = period_requests[period_requests['reviewer_email'] == reviewer_email].copy()
    
    if len(reviewer_data) == 0:
        # データなしの場合はゼロベクトル
        return np.zeros(10)
    
    # タイムスタンプをdatetimeに変換（CSVのカラム名に合わせる）
    timestamp_col = 'request_time' if 'request_time' in reviewer_data.columns else 'timestamp'
    reviewer_data[timestamp_col] = pd.to_datetime(reviewer_data[timestamp_col])
    reviewer_data = reviewer_data.sort_values(timestamp_col)
    
    # 1. experience_days: 最初の活動から現在までの日数
    first_seen = reviewer_data[timestamp_col].min()
    experience_days = (context_date - first_seen).days
    experience_days_norm = min(experience_days / 730.0, 1.0)  # 2年でキャップ
    
    # 2. total_changes: 総変更数（レビュー依頼数で近似）
    total_changes = len(reviewer_data)
    total_changes_norm = min(total_changes / 500.0, 1.0)  # 500件でキャップ
    
    # 3. total_reviews: 総レビュー数（レビュアーなので同じ）
    total_reviews = len(reviewer_data)
    total_reviews_norm = min(total_reviews / 500.0, 1.0)  # 500件でキャップ
    
    # 4. recent_activity_frequency: 直近30日の活動頻度
    cutoff_30d = context_date - timedelta(days=30)
    recent_30d = reviewer_data[reviewer_data[timestamp_col] >= cutoff_30d]
    recent_activity_frequency = len(recent_30d) / 30.0
    recent_activity_frequency_norm = min(recent_activity_frequency, 1.0)
    
    # 5. avg_activity_gap: 平均活動間隔
    if len(reviewer_data) >= 2:
        timestamps = reviewer_data[timestamp_col].values
        gaps = np.diff(timestamps) / np.timedelta64(1, 'D')  # 日数に変換
        avg_activity_gap = float(np.mean(gaps))
    else:
        avg_activity_gap = 30.0
    avg_activity_gap_norm = min(avg_activity_gap / 60.0, 1.0)  # 60日でキャップ
    
    # 6. activity_trend: 活動トレンド（最近30日 vs 過去30-60日）
    cutoff_60d = context_date - timedelta(days=60)
    past_30_60d = reviewer_data[(reviewer_data[timestamp_col] >= cutoff_60d) & 
                                 (reviewer_data[timestamp_col] < cutoff_30d)]
    recent_count = len(recent_30d)
    past_count = len(past_30_60d)
    
    if past_count == 0:
        activity_trend = 0.25  # unknown
    else:
        ratio = recent_count / past_count
        if ratio > 1.2:
            activity_trend = 1.0  # increasing
        elif ratio < 0.8:
            activity_trend = 0.0  # decreasing
        else:
            activity_trend = 0.5  # stable
    
    # 7. collaboration_score: レビュー活動の割合（全部レビューなので1.0）
    collaboration_score = 1.0
    
    # 8. code_quality_score: 簡易版（データなし: 0.5固定）
    code_quality_score = 0.5
    
    # 9. recent_acceptance_rate: 直近30日の受諾率（データなし: 0.5）
    recent_acceptance_rate = 0.5
    
    # 10. review_load: レビュー負荷（直近30日 / 全期間平均）
    total_days = max((context_date - first_seen).days, 1)
    avg_per_day = total_reviews / total_days
    recent_per_day = len(recent_30d) / 30.0
    
    if avg_per_day > 0:
        review_load = recent_per_day / avg_per_day
    else:
        review_load = 0.0
    review_load_norm = min(review_load, 1.0)
    
    # 10次元ベクトル（全て0-1正規化済み）
    features = np.array([
        experience_days_norm,
        total_changes_norm,
        total_reviews_norm,
        recent_activity_frequency_norm,
        avg_activity_gap_norm,
        activity_trend,
        collaboration_score,
        code_quality_score,
        recent_acceptance_rate,
        review_load_norm
    ])
    
    return features


def create_reviewer_features_and_labels(
    reviews_df: pd.DataFrame,
    period_start: datetime,
    period_end: datetime,
    future_start_months: int,
    future_end_months: int,
    min_history_events: int = 3,
    project: str = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    レビュアーごとの特徴量とラベルを作成（IRL完全準拠版）
    
    Args:
        reviews_df: レビューデータ
        period_start: 期間開始
        period_end: 期間終了
        future_start_months: 将来窓開始（期間終了からの月数）
        future_end_months: 将来窓終了（期間終了からの月数）
        min_history_events: 最小履歴イベント数
        project: プロジェクトフィルタ
    
    Returns:
        (特徴量配列, ラベル配列, レビュアーメールリスト)
    """
    logger.info(f"特徴量作成: 期間={period_start.date()}～{period_end.date()}, "
                f"将来窓={future_start_months}～{future_end_months}ヶ月後")
    
    # プロジェクトフィルタ
    if project and 'project' in reviews_df.columns:
        reviews_df = reviews_df[reviews_df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用: {project}, {len(reviews_df)}件")
    
    # タイムスタンプ変換（CSVのカラム名に合わせる）
    timestamp_col = 'request_time' if 'request_time' in reviews_df.columns else 'timestamp'
    reviews_df[timestamp_col] = pd.to_datetime(reviews_df[timestamp_col])
    
    # 期間内のデータ
    period_mask = (reviews_df[timestamp_col] >= period_start) & (reviews_df[timestamp_col] < period_end)
    period_requests = reviews_df[period_mask].copy()
    
    logger.info(f"期間内レビュー依頼: {len(period_requests)}件")
    
    # レビュアーごとの件数
    reviewer_counts = period_requests['reviewer_email'].value_counts()
    
    # 最小履歴イベント数でフィルタ
    valid_reviewers = reviewer_counts[reviewer_counts >= min_history_events].index.tolist()
    
    logger.info(f"有効レビュアー数: {len(valid_reviewers)}人 (最小履歴={min_history_events})")
    
    # 将来窓の定義
    future_start = period_end + timedelta(days=30 * future_start_months)
    future_end = period_end + timedelta(days=30 * future_end_months)
    
    logger.info(f"将来窓: {future_start.date()}～{future_end.date()}")
    
    # 将来窓内でレビュー依頼があったか（ラベル=1）
    future_mask = (reviews_df[timestamp_col] >= future_start) & (reviews_df[timestamp_col] < future_end)
    future_requests = reviews_df[future_mask]
    
    # レビュアーごとに将来窓内で承諾があったかチェック
    future_acceptance = future_requests[future_requests['label'] == 1].groupby('reviewer_email').size()
    
    labels = []
    features = []
    reviewer_emails = []
    
    # 各レビュアーの特徴量を計算（IRLロジック完全再現）
    for reviewer in valid_reviewers:
        # IRLの状態10次元を計算
        state_features = calculate_irl_state_features(
            reviewer_email=reviewer,
            period_requests=period_requests,
            context_date=period_end  # 期間終了時点での状態
        )
        
        features.append(state_features)
        labels.append(1 if reviewer in future_acceptance.index else 0)
        reviewer_emails.append(reviewer)
    
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    logger.info(f"特徴量形状: {features_array.shape}, 正例率: {labels_array.mean():.3f}")
    
    return features_array, labels_array, reviewer_emails


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
    model_type: str = 'rf',
    min_history_events: int = 3,
    project: str = None
) -> Dict[str, Any]:
    """
    訓練と評価を実行
    
    Args:
        model_type: 'rf' (RandomForest) or 'lr' (LogisticRegression)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"モデル訓練・評価: {model_type.upper()}")
    logger.info(f"訓練: {train_start.date()}～{train_end.date()} → {train_future_start}～{train_future_end}ヶ月後")
    logger.info(f"評価: {eval_start.date()}～{eval_end.date()} → {eval_future_start}～{eval_future_end}ヶ月後")
    logger.info(f"{'='*60}")
    
    # 訓練データ作成
    X_train, y_train, train_reviewers = create_reviewer_features_and_labels(
        reviews_df, train_start, train_end, 
        train_future_start, train_future_end,
        min_history_events, project
    )
    
    # 評価データ作成
    X_eval, y_eval, eval_reviewers = create_reviewer_features_and_labels(
        reviews_df, eval_start, eval_end,
        eval_future_start, eval_future_end,
        min_history_events, project
    )
    
    if len(X_train) == 0 or len(X_eval) == 0:
        logger.warning("訓練または評価データが空です")
        return None
    
    # モデル選択
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=777,
            n_jobs=-1
        )
    elif model_type == 'lr':
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=777,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # 訓練
    logger.info(f"訓練データ: {len(X_train)}件, 正例率: {y_train.mean():.3f}")
    model.fit(X_train, y_train)
    
    # 評価
    logger.info(f"評価データ: {len(X_eval)}件, 正例率: {y_eval.mean():.3f}")
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = model.predict(X_eval)
    
    # メトリクス計算
    auc_roc = roc_auc_score(y_eval, y_pred_proba)
    
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_eval, y_pred, digits=3))
    
    return {
        'model_type': model_type,
        'train_period': f"{train_start.date()}～{train_end.date()}",
        'train_future': f"{train_future_start}～{train_future_end}m",
        'eval_period': f"{eval_start.date()}～{eval_end.date()}",
        'eval_future': f"{eval_future_start}～{eval_future_end}m",
        'train_samples': len(X_train),
        'eval_samples': len(X_eval),
        'train_positive_rate': float(y_train.mean()),
        'eval_positive_rate': float(y_eval.mean()),
        'auc_roc': float(auc_roc),
        'classification_report': classification_report(y_eval, y_pred, output_dict=True)
    }


def run_cross_evaluation(
    reviews_path: str,
    train_start: datetime,
    train_end: datetime,
    eval_start: datetime,
    eval_end: datetime,
    model_type: str = 'both',
    min_history_events: int = 3,
    project: str = None,
    output_dir: str = 'outputs/baseline_cross_eval_irl'
) -> None:
    """クロス評価を実行"""
    
    logger.info(f"レビューデータ読み込み: {reviews_path}")
    reviews_df = pd.read_csv(reviews_path)
    logger.info(f"総レコード数: {len(reviews_df)}")
    
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 将来窓の定義
    future_periods = [
        (0, 3, '0-3m'),
        (3, 6, '3-6m'),
        (6, 9, '6-9m'),
        (9, 12, '9-12m')
    ]
    
    # モデルタイプ
    if model_type == 'both':
        model_types = ['rf', 'lr']
    else:
        model_types = [model_type]
    
    all_results = []
    
    # クロス評価（時系列順序を考慮：評価期間 >= 訓練期間）
    # 訓練期間より過去の評価は時系列的に不自然なため除外
    for model_t in model_types:
        for train_start_m, train_end_m, train_label in future_periods:
            for eval_start_m, eval_end_m, eval_label in future_periods:
                # 評価期間は訓練期間と同一または未来のみ
                if eval_start_m < train_start_m:
                    continue
                
                result = train_and_evaluate(
                    reviews_df=reviews_df,
                    train_start=train_start,
                    train_end=train_end,
                    train_future_start=train_start_m,
                    train_future_end=train_end_m,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    eval_future_start=eval_start_m,
                    eval_future_end=eval_end_m,
                    model_type=model_t,
                    min_history_events=min_history_events,
                    project=project
                )
                
                if result:
                    result['train_label'] = train_label
                    result['eval_label'] = eval_label
                    all_results.append(result)
    
    # 結果保存
    results_file = output_path / 'cross_eval_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n結果保存: {results_file}")
    
    # サマリー作成
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("ベースライン（RF/LR）クロス評価結果サマリー - IRL完全準拠版")
    summary_lines.append("=" * 80)
    
    for model_t in model_types:
        summary_lines.append(f"\n## {model_t.upper()} モデル")
        summary_lines.append("-" * 80)
        summary_lines.append(f"{'訓練期間':>12} | {'評価期間':>12} | {'AUC-ROC':>10} | {'訓練N':>8} | {'評価N':>8}")
        summary_lines.append("-" * 80)
        
        model_results = [r for r in all_results if r['model_type'] == model_t]
        auc_scores = []
        
        for result in model_results:
            summary_lines.append(
                f"{result['train_label']:>12} | {result['eval_label']:>12} | "
                f"{result['auc_roc']:>10.4f} | {result['train_samples']:>8} | {result['eval_samples']:>8}"
            )
            auc_scores.append(result['auc_roc'])
        
        summary_lines.append("-" * 80)
        summary_lines.append(f"平均 AUC-ROC: {np.mean(auc_scores):.4f}")
        summary_lines.append(f"標準偏差: {np.std(auc_scores):.4f}")
    
    summary_text = '\n'.join(summary_lines)
    print(summary_text)
    
    # サマリーファイル保存
    summary_file = output_path / 'summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    logger.info(f"サマリー保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='ベースライン（RF/LR）クロス評価 - IRL完全準拠版')
    parser.add_argument('--reviews', type=str, required=True, help='レビューデータCSVパス')
    parser.add_argument('--train-start', type=str, required=True, help='訓練期間開始 (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, required=True, help='訓練期間終了 (YYYY-MM-DD)')
    parser.add_argument('--eval-start', type=str, required=True, help='評価期間開始 (YYYY-MM-DD)')
    parser.add_argument('--eval-end', type=str, required=True, help='評価期間終了 (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default='both', choices=['rf', 'lr', 'both'], 
                       help='モデルタイプ')
    parser.add_argument('--min-history-events', type=int, default=3, 
                       help='最小履歴イベント数')
    parser.add_argument('--project', type=str, default=None, help='プロジェクトフィルタ')
    parser.add_argument('--output', type=str, default='outputs/baseline_cross_eval_irl',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    train_start = datetime.fromisoformat(args.train_start)
    train_end = datetime.fromisoformat(args.train_end)
    eval_start = datetime.fromisoformat(args.eval_start)
    eval_end = datetime.fromisoformat(args.eval_end)
    
    run_cross_evaluation(
        reviews_path=args.reviews,
        train_start=train_start,
        train_end=train_end,
        eval_start=eval_start,
        eval_end=eval_end,
        model_type=args.model,
        min_history_events=args.min_history_events,
        project=args.project,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
