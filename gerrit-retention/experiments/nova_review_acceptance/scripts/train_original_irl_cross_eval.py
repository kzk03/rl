#!/usr/bin/env python3
"""
オリジナルIRLの完全クロス評価（4×4パターン）

オリジナルの実装（scripts/training/irl/train_irl_review_acceptance.py）の
ロジックを完全再現し、4期間×4期間のクロス評価を実行します。

主要な特徴:
1. **sample_weight = 0.1** (依頼なしケース)
2. **拡張期間ロジック** (12ヶ月後までチェック)
3. **F1スコア最大化** (訓練データ上で閾値決定)
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
import torch
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ランダムシード固定
RANDOM_SEED = 777
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    F1スコアを最大化する閾値を探索（オリジナル実装）

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
        "threshold": float(best_threshold),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1_scores[best_idx]),
    }


def prepare_trajectories_original_style(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    future_window_months: int = 3,
    min_history_requests: int = 3,
    project: str = None,
    extended_label_months: int = 12,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    オリジナル実装のデータ準備ロジック

    重要な特徴:
    1. train_end時点で評価（スナップショット）
    2. 拡張期間（12ヶ月）を使って真の離脱を判定
    3. 依頼なしケースに sample_weight=0.1 を適用

    Args:
        df: レビューデータ
        train_start: 訓練期間開始
        train_end: 訓練期間終了
        eval_start: 評価期間開始（future_window_months後）
        eval_end: 評価期間終了
        future_window_months: ラベル計算のFuture Window
        min_history_requests: 最小履歴依頼数
        project: プロジェクト名
        extended_label_months: 拡張期間（真の離脱判定用）

    Returns:
        (trajectories, sample_weights)
    """
    if project:
        df = df[df["project"] == project].copy()

    date_col = "request_time"
    reviewer_col = "reviewer_email"
    label_col = "label"

    df[date_col] = pd.to_datetime(df[date_col])

    # train_end時点までの履歴
    history_df = df[df[date_col] < train_end]
    reviewers = history_df[reviewer_col].unique()

    # 拡張期間の終了
    extended_label_end = eval_end + pd.DateOffset(months=extended_label_months)

    trajectories = []
    sample_weights = []
    skipped_min_requests = 0
    no_request_count = 0
    no_request_but_extended_count = 0
    request_count = 0

    logger.info(f"訓練期間: {train_start} ~ {train_end}")
    logger.info(f"評価期間（通常）: {eval_start} ~ {eval_end}")
    logger.info(f"評価期間（拡張）: {eval_start} ~ {extended_label_end}")
    logger.info(f"レビュワー候補: {len(reviewers)}")

    for reviewer in reviewers:
        # train_end時点までの履歴
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]

        # 最小依頼数チェック
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue

        # 評価期間（通常）のレビュー依頼
        reviewer_eval = df[
            (df[date_col] >= eval_start)
            & (df[date_col] < eval_end)
            & (df[reviewer_col] == reviewer)
        ]

        # 評価期間（拡張）のレビュー依頼
        reviewer_extended_eval = df[
            (df[date_col] >= eval_start)
            & (df[date_col] < extended_label_end)
            & (df[reviewer_col] == reviewer)
        ]

        # ラベルとサンプル重みの決定（オリジナルロジック）
        if len(reviewer_eval) == 0:
            # 通常期間に依頼なし
            if len(reviewer_extended_eval) == 0:
                # 拡張期間にも依頼なし → 真の離脱、スキップ
                continue
            else:
                # 拡張期間には依頼あり → 一時的離脱、低重み
                label = 0
                sample_weight = 0.1  # 重要: オリジナルの重み設定
                no_request_but_extended_count += 1
        else:
            # 通常期間に依頼あり
            accepted_reviews = reviewer_eval[reviewer_eval[label_col] == 1]
            label = 1 if len(accepted_reviews) > 0 else 0
            sample_weight = 1.0
            request_count += 1

        # 活動履歴
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                "timestamp": row[date_col],
                "action_type": "review",
                "project": row.get("project", "unknown"),
                "request_time": row.get("request_time", row[date_col]),
                "response_time": row.get("first_response_time"),
                "accepted": row.get(label_col, 0) == 1,
            }
            activity_history.append(activity)

        # 開発者情報
        developer_info = {
            "developer_email": reviewer,
            "first_seen": reviewer_history[date_col].min(),
            "changes_authored": 0,
            "changes_reviewed": len(
                reviewer_history[reviewer_history[label_col] == 1]
            ),
            "requests_received": len(reviewer_history),
            "requests_accepted": len(
                reviewer_history[reviewer_history[label_col] == 1]
            ),
            "requests_rejected": len(
                reviewer_history[reviewer_history[label_col] == 0]
            ),
            "acceptance_rate": len(
                reviewer_history[reviewer_history[label_col] == 1]
            )
            / len(reviewer_history),
            "projects": reviewer_history["project"].unique().tolist()
            if "project" in reviewer_history.columns
            else [],
        }

        # 軌跡を作成
        trajectory = {
            "developer_info": developer_info,
            "activity_history": activity_history,
            "context_date": train_end,
            "step_labels": [label],
            "seq_len": 1,
        }

        trajectories.append(trajectory)
        sample_weights.append(sample_weight)

    logger.info(f"軌跡生成完了: {len(trajectories)}サンプル")
    logger.info(f"  依頼あり（重み=1.0）: {request_count}")
    logger.info(
        f"  依頼なし（拡張期間に依頼あり、重み=0.1）: {no_request_but_extended_count}"
    )
    logger.info(f"  スキップ（最小依頼数未満）: {skipped_min_requests}")

    positive = sum(1 for t in trajectories if t["step_labels"][0] == 1)
    negative = len(trajectories) - positive
    logger.info(f"  ポジティブ: {positive}, ネガティブ: {negative}")

    return trajectories, sample_weights


def train_and_evaluate_original_irl(
    train_trajectories: List[Dict],
    train_sample_weights: List[float],
    eval_trajectories: List[Dict],
    eval_sample_weights: List[float],
    epochs: int = 50,
    output_dir: Path = None,
) -> Dict[str, float]:
    """オリジナルIRLで訓練・評価"""

    logger.info("=" * 80)
    logger.info("オリジナルIRL訓練開始")
    logger.info("=" * 80)

    # システム初期化
    config = {
        "state_dim": 10,
        "action_dim": 4,
        "hidden_dim": 128,
        "sequence": True,
        "seq_len": 10,
        "dropout": 0.2,
        "learning_rate": 0.001,
    }
    system = RetentionIRLSystem(config)

    logger.info(f"訓練軌跡: {len(train_trajectories)}")
    logger.info(f"評価軌跡: {len(eval_trajectories)}")

    # 訓練
    logger.info("訓練開始...")
    system.train(
        train_trajectories,
        epochs=epochs,
        verbose=True,
        save_path=str(output_dir / "irl_model.pth") if output_dir else None,
    )

    # ====================================================================
    # オリジナルの閾値決定方式: F1スコア最大化（訓練データ上）
    # ====================================================================
    logger.info("=" * 80)
    logger.info("訓練データ上でF1最大化による閾値決定")
    logger.info("=" * 80)

    # 訓練データで予測
    train_predictions = []
    train_labels = []
    for traj in train_trajectories:
        pred = system.predict_retention(traj)
        train_predictions.append(pred)
        train_labels.append(traj["step_labels"][0])

    train_y_true = np.array(train_labels)
    train_y_pred = np.array(train_predictions)

    # F1最大化閾値を探索
    threshold_info = find_optimal_threshold(train_y_true, train_y_pred)
    optimal_threshold = threshold_info["threshold"]

    logger.info(f"最適閾値: {optimal_threshold:.4f}")
    logger.info(f"  訓練F1: {threshold_info['f1']:.4f}")
    logger.info(f"  訓練Precision: {threshold_info['precision']:.4f}")
    logger.info(f"  訓練Recall: {threshold_info['recall']:.4f}")

    # 訓練データでのAUC-ROC
    train_auc = roc_auc_score(train_y_true, train_y_pred)
    logger.info(f"訓練AUC-ROC: {train_auc:.4f}")

    # ====================================================================
    # 評価データで評価
    # ====================================================================
    logger.info("=" * 80)
    logger.info("評価データでの性能評価")
    logger.info("=" * 80)

    eval_predictions = []
    eval_labels = []
    for traj in eval_trajectories:
        pred = system.predict_retention(traj)
        eval_predictions.append(pred)
        eval_labels.append(traj["step_labels"][0])

    eval_y_true = np.array(eval_labels)
    eval_y_pred = np.array(eval_predictions)
    eval_y_pred_binary = (eval_y_pred >= optimal_threshold).astype(int)

    # メトリクス計算
    eval_auc = roc_auc_score(eval_y_true, eval_y_pred)
    eval_f1 = f1_score(eval_y_true, eval_y_pred_binary)
    eval_precision = precision_score(eval_y_true, eval_y_pred_binary, zero_division=0)
    eval_recall = recall_score(eval_y_true, eval_y_pred_binary, zero_division=0)

    logger.info(f"評価AUC-ROC: {eval_auc:.4f}")
    logger.info(f"評価F1: {eval_f1:.4f}")
    logger.info(f"評価Precision: {eval_precision:.4f}")
    logger.info(f"評価Recall: {eval_recall:.4f}")

    # サンプル統計
    positive = sum(eval_y_true)
    negative = len(eval_y_true) - positive
    logger.info(f"評価データ: ポジティブ={positive}, ネガティブ={negative}")

    results = {
        "train_auc_roc": train_auc,
        "eval_auc_roc": eval_auc,
        "eval_f1": eval_f1,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "optimal_threshold": optimal_threshold,
        "threshold_method": "f1_maximization_on_train_data",
        "train_samples": len(train_trajectories),
        "eval_samples": len(eval_trajectories),
        "eval_positive": int(positive),
        "eval_negative": int(negative),
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="オリジナルIRLの完全クロス評価（4×4パターン）"
    )
    parser.add_argument(
        "--reviews",
        type=str,
        default="data/review_requests_openstack_pilot_w14.csv",
        help="レビューデータCSVパス",
    )
    parser.add_argument(
        "--project", type=str, default="nova", help="プロジェクト名（フィルタ用）"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="訓練エポック数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/nova_review_acceptance/outputs_original_irl_cross_eval",
        help="出力ディレクトリ",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # レビューデータ読み込み
    reviews_path = ROOT / args.reviews
    logger.info(f"レビューデータ読み込み: {reviews_path}")
    df = pd.read_csv(reviews_path)
    logger.info(f"総レビュー数: {len(df)}")

    # 4期間定義
    periods = [
        {
            "name": "0-3m",
            "train_start": pd.Timestamp("2022-01-01"),
            "train_end": pd.Timestamp("2022-04-01"),
            "eval_start": pd.Timestamp("2022-04-01"),
            "eval_end": pd.Timestamp("2022-07-01"),
        },
        {
            "name": "3-6m",
            "train_start": pd.Timestamp("2022-04-01"),
            "train_end": pd.Timestamp("2022-07-01"),
            "eval_start": pd.Timestamp("2022-07-01"),
            "eval_end": pd.Timestamp("2022-10-01"),
        },
        {
            "name": "6-9m",
            "train_start": pd.Timestamp("2022-07-01"),
            "train_end": pd.Timestamp("2022-10-01"),
            "eval_start": pd.Timestamp("2022-10-01"),
            "eval_end": pd.Timestamp("2023-01-01"),
        },
        {
            "name": "9-12m",
            "train_start": pd.Timestamp("2022-10-01"),
            "train_end": pd.Timestamp("2023-01-01"),
            "eval_start": pd.Timestamp("2023-01-01"),
            "eval_end": pd.Timestamp("2023-04-01"),
        },
    ]

    # 4×4クロス評価
    results_matrix = []
    auc_matrix = np.zeros((4, 4))

    logger.info("=" * 80)
    logger.info("4×4クロス評価開始")
    logger.info("=" * 80)

    for i, train_period in enumerate(periods):
        for j, eval_period in enumerate(periods):
            logger.info("=" * 80)
            logger.info(
                f"パターン {i+1}-{j+1}: 訓練={train_period['name']}, 評価={eval_period['name']}"
            )
            logger.info("=" * 80)

            # 訓練データ準備
            train_trajectories, train_weights = prepare_trajectories_original_style(
                df=df,
                train_start=train_period["train_start"],
                train_end=train_period["train_end"],
                eval_start=train_period["eval_start"],
                eval_end=train_period["eval_end"],
                future_window_months=3,
                min_history_requests=3,
                project=args.project,
                extended_label_months=12,
            )

            # 評価データ準備
            eval_trajectories, eval_weights = prepare_trajectories_original_style(
                df=df,
                train_start=eval_period["train_start"],
                train_end=eval_period["train_end"],
                eval_start=eval_period["eval_start"],
                eval_end=eval_period["eval_end"],
                future_window_months=3,
                min_history_requests=3,
                project=args.project,
                extended_label_months=12,
            )

            # 訓練・評価
            pattern_output_dir = output_dir / f"pattern_{i+1}_{j+1}"
            pattern_output_dir.mkdir(parents=True, exist_ok=True)

            results = train_and_evaluate_original_irl(
                train_trajectories=train_trajectories,
                train_sample_weights=train_weights,
                eval_trajectories=eval_trajectories,
                eval_sample_weights=eval_weights,
                epochs=args.epochs,
                output_dir=pattern_output_dir,
            )

            # 結果保存
            results["train_period"] = train_period["name"]
            results["eval_period"] = eval_period["name"]
            results_matrix.append(results)
            auc_matrix[i, j] = results["eval_auc_roc"]

            logger.info(f"✅ パターン {i+1}-{j+1} 完了: AUC-ROC = {results['eval_auc_roc']:.4f}")

    # 全結果保存
    results_df = pd.DataFrame(results_matrix)
    results_df.to_csv(output_dir / "cross_eval_results.csv", index=False)
    logger.info(f"結果保存: {output_dir / 'cross_eval_results.csv'}")

    # AUC-ROCマトリクス保存
    auc_df = pd.DataFrame(
        auc_matrix,
        index=[p["name"] for p in periods],
        columns=[p["name"] for p in periods],
    )
    auc_df.to_csv(output_dir / "matrix_AUC_ROC.csv")
    logger.info(f"AUC-ROCマトリクス保存: {output_dir / 'matrix_AUC_ROC.csv'}")

    # サマリー表示
    logger.info("=" * 80)
    logger.info("クロス評価完了")
    logger.info("=" * 80)
    logger.info("\nAUC-ROCマトリクス:")
    logger.info(auc_df.to_string())
    logger.info(f"\n平均AUC-ROC: {auc_matrix.mean():.4f}")
    logger.info(f"対角線平均（同期間）: {np.diag(auc_matrix).mean():.4f}")
    logger.info(f"非対角線平均（異期間）: {auc_matrix[~np.eye(4, dtype=bool)].mean():.4f}")


if __name__ == "__main__":
    main()
