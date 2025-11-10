#!/usr/bin/env python3
"""
Random Forest Baseline - 4×4クロス評価（修正版）

Enhanced IRLと同じロジックで実行:
- 訓練: 訓練期間を4分割して各期間でモデル訓練
- 評価: 評価開始時点(2023-01-01)のスナップショットから異なるFuture Windowで予測
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc as calc_auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_features(df: pd.DataFrame, reviewer: str) -> np.ndarray:
    """レビュワーの特徴量抽出"""
    reviewer_df = df[df['reviewer_email'] == reviewer]

    total_requests = len(reviewer_df)
    accepted = len(reviewer_df[reviewer_df['label'] == 1])
    rejected = len(reviewer_df[reviewer_df['label'] == 0])
    acceptance_rate = accepted / total_requests if total_requests > 0 else 0

    # 時系列特徴
    if len(reviewer_df) > 1:
        days_active = (reviewer_df['request_time'].max() - reviewer_df['request_time'].min()).days
        requests_per_day = total_requests / (days_active + 1)
    else:
        days_active = 0
        requests_per_day = 0

    features = np.array([
        total_requests,
        accepted,
        rejected,
        acceptance_rate,
        days_active,
        requests_per_day,
        len(reviewer_df['project'].unique()) if 'project' in reviewer_df.columns else 1,
        reviewer_df['request_time'].dt.hour.mean() if len(reviewer_df) > 0 else 0,
    ])

    return features


def prepare_snapshot_data(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history_requests: int = 3
) -> tuple:
    """スナップショット評価用データ準備"""

    # スナップショット以前の履歴
    history_df = df[df['request_time'] < snapshot_date]
    reviewers = history_df['reviewer_email'].unique()

    # ラベル期間
    label_start = snapshot_date + pd.DateOffset(months=future_window_start_months)
    label_end = snapshot_date + pd.DateOffset(months=future_window_end_months)

    logger.info(f"スナップショット: {snapshot_date.date()}")
    logger.info(f"ラベル期間: {label_start.date()} ~ {label_end.date()}")
    logger.info(f"候補レビュワー: {len(reviewers)}")

    features_list = []
    labels_list = []
    skipped = 0

    for reviewer in reviewers:
        reviewer_history = history_df[history_df['reviewer_email'] == reviewer]

        if len(reviewer_history) < min_history_requests:
            skipped += 1
            continue

        # ラベル期間の活動
        reviewer_future = df[
            (df['request_time'] >= label_start) &
            (df['request_time'] < label_end) &
            (df['reviewer_email'] == reviewer)
        ]

        if len(reviewer_future) == 0:
            continue

        # 特徴量抽出（履歴から）
        features = extract_features(history_df, reviewer)

        # ラベル（将来期間で受理があるか）
        label = 1 if len(reviewer_future[reviewer_future['label'] == 1]) > 0 else 0

        features_list.append(features)
        labels_list.append(label)

    logger.info(f"データ数: {len(features_list)} (スキップ: {skipped})")
    logger.info(f"ポジティブ: {sum(labels_list)}, ネガティブ: {len(labels_list) - sum(labels_list)}")

    return np.array(features_list), np.array(labels_list)


def prepare_train_data_for_period(
    df: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history_requests: int = 3
) -> tuple:
    """訓練期間用データ準備（複数スナップショット）"""

    logger.info(f"訓練期間: {period_start.date()} ~ {period_end.date()}")
    logger.info(f"Future Window: {future_window_start_months}-{future_window_end_months}ヶ月")

    period_df = df[(df['request_time'] >= period_start) & (df['request_time'] < period_end)]
    reviewers = period_df['reviewer_email'].unique()

    features_list = []
    labels_list = []
    skipped = 0

    for reviewer in reviewers:
        reviewer_period = period_df[period_df['reviewer_email'] == reviewer]

        if len(reviewer_period) < min_history_requests:
            skipped += 1
            continue

        # ラベル期間（訓練期間終了後）
        label_start = period_end + pd.DateOffset(months=future_window_start_months)
        label_end = period_end + pd.DateOffset(months=future_window_end_months)

        reviewer_future = df[
            (df['request_time'] >= label_start) &
            (df['request_time'] < label_end) &
            (df['reviewer_email'] == reviewer)
        ]

        if len(reviewer_future) == 0:
            continue

        features = extract_features(period_df, reviewer)
        label = 1 if len(reviewer_future[reviewer_future['label'] == 1]) > 0 else 0

        features_list.append(features)
        labels_list.append(label)

    logger.info(f"訓練データ数: {len(features_list)}")

    return np.array(features_list), np.array(labels_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--eval-start", required=True)
    parser.add_argument("--eval-end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-history-events", type=int, default=3)

    args = parser.parse_args()

    # データ読み込み
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])

    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    eval_start = pd.to_datetime(args.eval_start)
    eval_end = pd.to_datetime(args.eval_end)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 訓練期間を4分割
    train_periods = []
    period_names = []
    total_months = 24  # 2年
    for i in range(4):
        p_start = train_start + pd.DateOffset(months=i * 6)
        p_end = train_start + pd.DateOffset(months=(i + 1) * 6)
        train_periods.append((p_start, p_end))
        period_names.append(f"{i*3}-{(i+1)*3}m")

    # 評価Future Window定義
    eval_windows = [(0, 3), (3, 6), (6, 9), (9, 12)]

    logger.info("=" * 80)
    logger.info("Random Forest 4×4クロス評価（修正版）")
    logger.info("=" * 80)

    # 結果マトリクス
    auc_roc_matrix = np.zeros((4, 4))
    auc_pr_matrix = np.zeros((4, 4))
    f1_matrix = np.zeros((4, 4))

    # STEP 1: 各訓練期間でモデル訓練
    logger.info("\nSTEP 1: モデル訓練")
    models = {}

    for i, ((p_start, p_end), train_name) in enumerate(zip(train_periods, period_names)):
        # 訓練期間の中間Future Windowで訓練（例: 3-6m）
        train_fw_start, train_fw_end = eval_windows[i]

        X_train, y_train = prepare_train_data_for_period(
            df, p_start, p_end, train_fw_start, train_fw_end, args.min_history_events
        )

        if len(X_train) == 0:
            logger.warning(f"訓練データなし: {train_name}")
            continue

        # モデル訓練
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[train_name] = model

        logger.info(f"✅ 訓練完了: {train_name}")

    # STEP 2: クロス評価
    logger.info("\nSTEP 2: クロス評価")

    for i, train_name in enumerate(period_names):
        if train_name not in models:
            continue

        model = models[train_name]

        for j, (eval_fw_start, eval_fw_end) in enumerate(eval_windows):
            eval_name = period_names[j]

            # スナップショット評価データ準備
            X_eval, y_eval = prepare_snapshot_data(
                df, eval_start, eval_fw_start, eval_fw_end, args.min_history_events
            )

            if len(X_eval) == 0 or len(np.unique(y_eval)) < 2:
                logger.warning(f"評価データ不足: {train_name} -> {eval_name}")
                continue

            # 予測
            y_pred_proba = model.predict_proba(X_eval)[:, 1]

            # メトリクス計算
            auc_roc = roc_auc_score(y_eval, y_pred_proba)

            precision, recall, thresholds = precision_recall_curve(y_eval, y_pred_proba)
            auc_pr = calc_auc(recall, precision)

            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            f1 = f1_score(y_eval, y_pred)

            # 結果保存
            auc_roc_matrix[i, j] = auc_roc
            auc_pr_matrix[i, j] = auc_pr
            f1_matrix[i, j] = f1

            # 詳細結果保存
            result_dir = output_dir / f"train_{train_name}" / f"eval_{eval_name}"
            result_dir.mkdir(parents=True, exist_ok=True)

            metrics = {
                "auc_roc": float(auc_roc),
                "auc_pr": float(auc_pr),
                "optimal_threshold": float(optimal_threshold),
                "precision": float(precision_score(y_eval, y_pred)),
                "recall": float(recall_score(y_eval, y_pred)),
                "f1_score": float(f1),
                "positive_count": int(sum(y_eval)),
                "negative_count": int(len(y_eval) - sum(y_eval)),
                "total_count": int(len(y_eval))
            }

            with open(result_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"{train_name} -> {eval_name}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")

    # マトリクス保存
    for metric_name, matrix in [('AUC_ROC', auc_roc_matrix), ('AUC_PR', auc_pr_matrix), ('F1_SCORE', f1_matrix)]:
        df_matrix = pd.DataFrame(matrix, index=period_names, columns=period_names)
        df_matrix.to_csv(output_dir / f"matrix_{metric_name}.csv")
        logger.info(f"✅ 保存: matrix_{metric_name}.csv")

    # サマリー表示
    logger.info("\n" + "=" * 80)
    logger.info("Random Forest 4×4クロス評価結果")
    logger.info("=" * 80)
    logger.info(f"\n平均AUC-ROC: {np.mean(auc_roc_matrix[auc_roc_matrix > 0]):.4f}")
    logger.info(f"平均AUC-PR: {np.mean(auc_pr_matrix[auc_pr_matrix > 0]):.4f}")
    logger.info(f"平均F1: {np.mean(f1_matrix[f1_matrix > 0]):.4f}")

    logger.info("\n完了！")


if __name__ == "__main__":
    main()
