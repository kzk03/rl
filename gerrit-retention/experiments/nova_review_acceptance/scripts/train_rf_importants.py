#!/usr/bin/env python3
"""
Random Forest Baseline - importants準拠のデータ準備で訓練

train_enhanced_irl_importants.pyと同じデータ準備ロジックを使用し、
モデルだけRandom Forestに置き換える。
K-Fold Cross-Validationで閾値を決定する。
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold

# ランダムシード固定
RANDOM_SEED = 777
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_trajectories_importants_style(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history_requests: int = 3,
    project: str = None
) -> List[Dict[str, Any]]:
    """
    importantsと同じデータ準備方式（データリークなし版）
    
    重要：訓練期間を分割してデータリークを防止
    - 特徴量計算期間: train_start ~ (train_end - future_window_end_months)
    - ラベル計算期間: (train_end - future_window_end_months) ~ train_end
    
    各レビュワーについて:
    - 特徴量計算期間内の履歴から月次の軌跡を生成
    - 各月末からfuture_window後の承諾有無をラベルとする（train_end内）
    """
    if project:
        df = df[df['project'] == project].copy()
    
    date_col = 'request_time'
    reviewer_col = 'reviewer_email'
    label_col = 'label'
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 特徴量計算期間の終了（ラベル期間の開始）
    feature_end = train_end - pd.DateOffset(months=future_window_end_months)
    
    # 特徴量計算期間のデータ
    feature_df = df[(df[date_col] >= train_start) & (df[date_col] < feature_end)]
    reviewers = feature_df[reviewer_col].unique()
    
    trajectories = []
    skipped_min_requests = 0
    
    logger.info(f"訓練期間全体: {train_start} ~ {train_end}")
    logger.info(f"特徴量計算期間: {train_start} ~ {feature_end}")
    logger.info(f"ラベル計算期間: {feature_end} ~ {train_end}")
    logger.info(f"Future Window: {future_window_start_months} ~ {future_window_end_months}ヶ月")
    logger.info(f"レビュワー候補: {len(reviewers)}")
    
    for reviewer in reviewers:
        reviewer_history = feature_df[feature_df[reviewer_col] == reviewer]
        
        # 最小依頼数チェック
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue
        
        # 月次の軌跡生成（特徴量計算期間内のみ）
        history_months = pd.date_range(
            start=train_start,
            end=feature_end,
            freq='MS'
        )
        
        step_labels = []
        monthly_activity_histories = []
        future_request_flags = []  # 将来依頼があったかのフラグ
        
        for month_start in history_months[:-1]:
            month_end = month_start + pd.DateOffset(months=1)
            
            # この月からfuture_window後のラベル計算期間
            future_start = month_end + pd.DateOffset(months=future_window_start_months)
            future_end = month_end + pd.DateOffset(months=future_window_end_months)
            
            # train_endを超えないように制限
            if future_end > train_end:
                future_end = train_end
            
            if future_start >= train_end:
                continue
            
            # この月時点までの活動履歴
            month_history = reviewer_history[reviewer_history[date_col] < month_end]
            monthly_activities = []
            for _, row in month_history.iterrows():
                activity = {
                    'timestamp': row[date_col],
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                    'request_time': row.get('request_time', row[date_col]),
                    'response_time': row.get('first_response_time'),
                    'accepted': row.get(label_col, 0) == 1,
                }
                monthly_activities.append(activity)
            
            # 将来期間のレビュー依頼
            month_future_df = df[
                (df[date_col] >= future_start) &
                (df[date_col] < future_end) &
                (df[reviewer_col] == reviewer)
            ]
            
            # ラベル
            has_future_request = len(month_future_df) > 0
            if has_future_request:
                month_accepted = month_future_df[month_future_df[label_col] == 1]
                month_label = 1 if len(month_accepted) > 0 else 0
            else:
                month_label = 0
            
            step_labels.append(month_label)
            monthly_activity_histories.append(monthly_activities)
            future_request_flags.append(has_future_request)  # フラグを記録
        
        if len(step_labels) == 0:
            continue
        
        # 軌跡として保存
        traj = {
            'reviewer_email': reviewer,
            'step_labels': step_labels,
            'monthly_activity_histories': monthly_activity_histories,
            'future_request_flags': future_request_flags,  # フラグを含める
        }
        trajectories.append(traj)
    
    logger.info(f"生成された軌跡数: {len(trajectories)}")
    logger.info(f"最小依頼数でスキップ: {skipped_min_requests}")
    
    # サンプル数統計
    total_steps = sum(len(t['step_labels']) for t in trajectories)
    positive_steps = sum(sum(t['step_labels']) for t in trajectories)
    logger.info(f"総ステップ数: {total_steps}, Positive: {positive_steps}, Negative: {total_steps - positive_steps}")
    
    return trajectories


def prepare_snapshot_evaluation_trajectories(
    df: pd.DataFrame,
    eval_snapshot: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history_requests: int = 3,
    project: str = None
) -> List[Dict[str, Any]]:
    """
    スナップショット時点の単一評価軌跡を生成
    
    eval_snapshot時点までの全履歴を使って、
    future_window後の承諾有無を予測
    """
    if project:
        df = df[df['project'] == project].copy()
    
    date_col = 'request_time'
    reviewer_col = 'reviewer_email'
    label_col = 'label'
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # スナップショット時点までの履歴
    history_df = df[df[date_col] < eval_snapshot]
    reviewers = history_df[reviewer_col].unique()
    
    trajectories = []
    skipped_min_requests = 0
    
    logger.info(f"評価スナップショット: {eval_snapshot}")
    logger.info(f"Future Window: {future_window_start_months} ~ {future_window_end_months}ヶ月")
    logger.info(f"レビュワー候補: {len(reviewers)}")
    
    for reviewer in reviewers:
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]
        
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue
        
        # スナップショット時点までの全活動履歴
        activities = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col],
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'request_time': row.get('request_time', row[date_col]),
                'response_time': row.get('first_response_time'),
                'accepted': row.get(label_col, 0) == 1,
            }
            activities.append(activity)
        
        # 将来期間の計算
        future_start = eval_snapshot + pd.DateOffset(months=future_window_start_months)
        future_end = eval_snapshot + pd.DateOffset(months=future_window_end_months)
        
        # 将来期間のレビュー依頼
        future_df = df[
            (df[date_col] >= future_start) &
            (df[date_col] < future_end) &
            (df[reviewer_col] == reviewer)
        ]
        
        # 評価時: 将来依頼がない場合はスキップ（予測対象外）
        if len(future_df) == 0:
            continue
        
        # ラベル: 受理したレビューがあるか
        accepted_df = future_df[future_df[label_col] == 1]
        label = 1 if len(accepted_df) > 0 else 0
        
        # 単一ステップの軌跡
        traj = {
            'reviewer_email': reviewer,
            'step_labels': [label],
            'monthly_activity_histories': [activities],
        }
        trajectories.append(traj)
    
    logger.info(f"生成された評価軌跡数: {len(trajectories)}")
    logger.info(f"最小依頼数でスキップ: {skipped_min_requests}")
    
    total_steps = sum(len(t['step_labels']) for t in trajectories)
    positive_steps = sum(sum(t['step_labels']) for t in trajectories)
    logger.info(f"総ステップ数: {total_steps}, Positive: {positive_steps}, Negative: {total_steps - positive_steps}")
    
    return trajectories


def extract_features_from_activities(activities: List[Dict[str, Any]]) -> np.ndarray:
    """
    活動履歴から集約特徴量を抽出
    
    特徴量（10次元）:
    - 総レビュー依頼数
    - 承諾数
    - 承諾率
    - 平均応答時間（時間単位、NaNは0）
    - 最終活動からの経過日数
    - 最近30日の活動数
    - 最近90日の活動数
    - 最近180日の活動数
    - 最近30日の承諾率
    - 最近90日の承諾率
    """
    if len(activities) == 0:
        return np.zeros(10)
    
    # 基本統計
    total_reviews = len(activities)
    accepted_count = sum(1 for a in activities if a.get('accepted', False))
    acceptance_rate = accepted_count / total_reviews if total_reviews > 0 else 0.0
    
    # 応答時間（request_time -> response_timeの差分）
    response_times = []
    for a in activities:
        req_time = a.get('request_time')
        resp_time = a.get('response_time')
        if req_time and resp_time:
            try:
                if isinstance(req_time, str):
                    req_time = pd.to_datetime(req_time)
                if isinstance(resp_time, str):
                    resp_time = pd.to_datetime(resp_time)
                delta = (resp_time - req_time).total_seconds() / 3600.0  # 時間単位
                if delta >= 0:
                    response_times.append(delta)
            except:
                pass
    
    avg_response_hours = np.mean(response_times) if response_times else 0.0
    
    # 最終活動からの経過日数
    timestamps = [a['timestamp'] for a in activities if 'timestamp' in a]
    if timestamps:
        latest_timestamp = max(timestamps)
        if isinstance(latest_timestamp, str):
            latest_timestamp = pd.to_datetime(latest_timestamp)
        # 現在時刻を最新のタイムスタンプとして経過日数を計算
        days_since_last = 0.0  # スナップショット時点での計算なので0
    else:
        days_since_last = 0.0
    
    # 期間別活動数・承諾率
    def count_recent(days: int) -> Tuple[int, float]:
        if not timestamps:
            return 0, 0.0
        cutoff = latest_timestamp - pd.Timedelta(days=days)
        recent = [a for a in activities if a['timestamp'] >= cutoff]
        count = len(recent)
        accepted = sum(1 for a in recent if a.get('accepted', False))
        rate = accepted / count if count > 0 else 0.0
        return count, rate
    
    recent_30_count, recent_30_rate = count_recent(30)
    recent_90_count, recent_90_rate = count_recent(90)
    recent_180_count, _ = count_recent(180)
    
    features = np.array([
        total_reviews,
        accepted_count,
        acceptance_rate,
        avg_response_hours,
        days_since_last,
        recent_30_count,
        recent_90_count,
        recent_180_count,
        recent_30_rate,
        recent_90_rate,
    ], dtype=np.float32)
    
    return features


def trajectories_to_features(trajectories: List[Dict[str, Any]], 
                            include_weights: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    軌跡リストを特徴量行列とラベルベクトルに変換
    
    Args:
        trajectories: 軌跡リスト
        include_weights: sample_weightsを含めるか（訓練時のみTrue）
        
    Returns:
        X: 特徴量行列
        y: ラベルベクトル
        weights: サンプル重み（include_weights=Trueの場合のみ意味を持つ）
    """
    X_list = []
    y_list = []
    weight_list = []
    
    for traj in trajectories:
        step_labels = traj['step_labels']
        monthly_histories = traj['monthly_activity_histories']
        # future_request_flags: 将来依頼があったかのフラグ（あれば使う）
        future_flags = traj.get('future_request_flags', [True] * len(step_labels))
        
        for label, activities, has_future_request in zip(step_labels, monthly_histories, future_flags):
            features = extract_features_from_activities(activities)
            X_list.append(features)
            y_list.append(label)
            
            # IRLと同じ重み付け: 依頼なし=0.5, 依頼あり=1.0
            if include_weights:
                if has_future_request:
                    weight = 1.0  # 依頼あり（メイン）
                else:
                    weight = 0.5  # 依頼なし（サブ）
            else:
                weight = 1.0  # 評価時は均一
            weight_list.append(weight)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    weights = np.array(weight_list, dtype=np.float32)
    
    # IRLと同じクラスバランシングを追加
    if include_weights and len(y) > 0:
        positive_count = np.sum(y == 1)
        total_count = len(y)
        positive_rate = positive_count / total_count if total_count > 0 else 0.5
        positive_rate = max(0.01, min(0.99, positive_rate))  # クリップ
        neg_weight = positive_rate / (1 - positive_rate)
        
        # label=0の重みを調整（IRLと同じ）
        for i in range(len(y)):
            if y[i] == 0:
                weights[i] *= neg_weight
    
    return X, y, weights


def find_best_f1_threshold(y_proba: np.ndarray, y_true: np.ndarray) -> float:
    """
    F1スコアを最大化する閾値を探索
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    f1_scores = []
    for prec, rec in zip(precisions[:-1], recalls[:-1]):
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        f1_scores.append(f1)
    
    if len(f1_scores) == 0:
        return 0.5
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    return float(best_threshold)


def train_random_forest_with_kfold_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights_train: np.ndarray,
    n_folds: int = 5
) -> Tuple[RandomForestClassifier, float, float]:
    """
    K-Fold Cross-Validationで閾値を決定し、全訓練データでモデルを訓練
    
    Args:
        X_train: 訓練特徴量
        y_train: 訓練ラベル
        weights_train: サンプル重み（依頼あり=1.0, 依頼なし=0.5）
        n_folds: Fold数
    
    Returns:
        model: 訓練済みRandomForestモデル
        threshold: K-Fold CVで決定した最適閾値
        threshold_std: 閾値の標準偏差
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_thresholds = []
    
    logger.info(f"K-Fold CV (K={n_folds})で閾値を決定中...")
    logger.info(f"Sample weights: 依頼あり={weights_train.max():.1f}, 依頼なし={weights_train.min():.1f}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        w_fold_train = weights_train[train_idx]  # 重みも分割
        
        # Foldごとにモデル訓練（sample_weight使用）
        fold_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_SEED + fold_idx,
            n_jobs=-1
        )
        fold_model.fit(X_fold_train, y_fold_train, sample_weight=w_fold_train)
        
        # 検証データで閾値決定
        val_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        
        if len(set(y_fold_val)) > 1:
            optimal_threshold = find_best_f1_threshold(val_proba, y_fold_val)
        else:
            optimal_threshold = 0.5
        
        fold_thresholds.append(optimal_threshold)
        logger.info(f"  Fold {fold_idx}: 閾値 = {optimal_threshold:.4f}")
    
    # 最終閾値 = 平均
    final_threshold = float(np.mean(fold_thresholds))
    threshold_std = float(np.std(fold_thresholds))
    
    logger.info(f"最終閾値: {final_threshold:.4f} (±{threshold_std:.4f})")
    
    # 全訓練データで最終モデル訓練
    logger.info("全訓練データでモデル訓練中（sample_weight使用）...")
    final_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train, sample_weight=weights_train)
    
    return final_model, final_threshold, threshold_std


def evaluate_model(
    model: RandomForestClassifier,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    モデルを評価してメトリクスを計算
    """
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {}
    
    # AUC-ROC
    if len(set(y_eval)) > 1:
        metrics['auc_roc'] = float(roc_auc_score(y_eval, y_proba))
    else:
        metrics['auc_roc'] = 0.0
    
    # AUC-PR
    if len(set(y_eval)) > 1:
        metrics['auc_pr'] = float(average_precision_score(y_eval, y_proba))
    else:
        metrics['auc_pr'] = 0.0
    
    # F1, Precision, Recall
    metrics['f1'] = float(f1_score(y_eval, y_pred, zero_division=0))
    metrics['precision'] = float(precision_score(y_eval, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_eval, y_pred, zero_division=0))
    
    # サンプル数
    metrics['num_samples'] = int(len(y_eval))
    metrics['num_positive'] = int(sum(y_eval))
    metrics['num_negative'] = int(len(y_eval) - sum(y_eval))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Random Forest Baseline - importants準拠')
    parser.add_argument('--train_start', type=str, required=True, help='訓練開始日 (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, required=True, help='訓練終了日 (YYYY-MM-DD)')
    parser.add_argument('--eval_snapshot', type=str, required=True, help='評価スナップショット日 (YYYY-MM-DD)')
    parser.add_argument('--train_fw_start', type=int, required=True, help='訓練Future Window開始月数')
    parser.add_argument('--train_fw_end', type=int, required=True, help='訓練Future Window終了月数')
    parser.add_argument('--eval_fw_start', type=int, required=True, help='評価Future Window開始月数')
    parser.add_argument('--eval_fw_end', type=int, required=True, help='評価Future Window終了月数')
    parser.add_argument('--output_dir', type=str, required=True, help='出力ディレクトリ')
    parser.add_argument('--project', type=str, default='openstack/nova', help='プロジェクト名')
    parser.add_argument('--data_path', type=str, 
                        default='data/review_requests_openstack_multi_5y_detail.csv',
                        help='データファイルパス')
    parser.add_argument('--n_folds', type=int, default=5, help='K-Fold CVのFold数')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 日付パース
    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    eval_snapshot = pd.to_datetime(args.eval_snapshot)
    
    logger.info("=" * 80)
    logger.info("Random Forest Baseline - importants準拠")
    logger.info("=" * 80)
    logger.info(f"訓練期間: {train_start} ~ {train_end}")
    logger.info(f"訓練 Future Window: {args.train_fw_start} ~ {args.train_fw_end}ヶ月")
    logger.info(f"評価スナップショット: {eval_snapshot}")
    logger.info(f"評価 Future Window: {args.eval_fw_start} ~ {args.eval_fw_end}ヶ月")
    logger.info(f"K-Fold CV: {args.n_folds} folds")
    logger.info(f"出力ディレクトリ: {output_dir}")
    
    # データ読み込み
    data_path = ROOT / args.data_path
    logger.info(f"データ読み込み: {data_path}")
    df = pd.read_csv(data_path)
    df['request_time'] = pd.to_datetime(df['request_time'])
    
    # プロジェクトフィルタ
    if args.project:
        df = df[df['project'] == args.project].copy()
        logger.info(f"プロジェクト: {args.project}, レコード数: {len(df)}")
    
    # 訓練データ準備
    logger.info("\n訓練データ準備中...")
    train_trajectories = prepare_trajectories_importants_style(
        df,
        train_start,
        train_end,
        args.train_fw_start,
        args.train_fw_end,
        min_history_requests=3,
        project=None  # 既にフィルタ済み
    )
    
    if len(train_trajectories) == 0:
        logger.error("訓練軌跡が0件です。終了します。")
        return
    
    # 評価データ準備
    logger.info("\n評価データ準備中...")
    eval_trajectories = prepare_snapshot_evaluation_trajectories(
        df,
        eval_snapshot,
        args.eval_fw_start,
        args.eval_fw_end,
        min_history_requests=3,
        project=None
    )
    
    if len(eval_trajectories) == 0:
        logger.error("評価軌跡が0件です。終了します。")
        return
    
    # 特徴量抽出
    logger.info("\n特徴量抽出中...")
    X_train, y_train, weights_train = trajectories_to_features(train_trajectories, include_weights=True)
    X_eval, y_eval, _ = trajectories_to_features(eval_trajectories, include_weights=False)
    
    # 重み統計
    positive_count = sum(y_train == 1)
    total_count = len(y_train)
    positive_rate = positive_count / total_count
    neg_weight = positive_rate / (1 - positive_rate)
    
    logger.info(f"訓練データ: {X_train.shape}, Positive: {positive_count}/{total_count} ({positive_rate:.1%})")
    logger.info(f"  - 依頼あり (base_weight=1.0): {sum(weights_train >= 0.9)}")
    logger.info(f"  - 依頼なし (base_weight=0.5): {sum(weights_train < 0.9)}")
    logger.info(f"  - クラスバランス: label=0の重み係数 = {neg_weight:.2f}")
    logger.info(f"評価データ: {X_eval.shape}, Positive: {sum(y_eval)}/{len(y_eval)}")
    
    # モデル訓練（K-Fold CVで閾値決定、sample_weight使用）
    logger.info("\nモデル訓練開始...")
    model, threshold, threshold_std = train_random_forest_with_kfold_cv(
        X_train, y_train, weights_train, n_folds=args.n_folds
    )
    
    # 評価
    logger.info("\nモデル評価中...")
    metrics = evaluate_model(model, X_eval, y_eval, threshold)
    
    # 閾値情報追加
    metrics['threshold'] = float(threshold)
    metrics['threshold_std'] = float(threshold_std)
    
    # 結果表示
    logger.info("\n" + "=" * 80)
    logger.info("評価結果")
    logger.info("=" * 80)
    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR:  {metrics['auc_pr']:.4f}")
    logger.info(f"F1:      {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"閾値:    {metrics['threshold']:.4f} (±{metrics['threshold_std']:.4f})")
    logger.info(f"サンプル数: {metrics['num_samples']} (Pos: {metrics['num_positive']}, Neg: {metrics['num_negative']})")
    
    # 結果保存
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nメトリクス保存: {metrics_path}")
    
    # モデル保存
    import joblib
    model_path = output_dir / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"モデル保存: {model_path}")
    
    logger.info("\n完了!")


if __name__ == '__main__':
    main()
