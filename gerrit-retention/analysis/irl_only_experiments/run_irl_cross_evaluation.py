#!/usr/bin/env python3
"""IRL クロス評価スクリプト。

訓練データ期間は固定（train-start～train-end）で、
将来窓（future-window）を0-3m, 3-6m, 6-9m, 9-12mと変えて
4つのモデルを訓練。各モデルを異なる評価窓（eval-start以降）で評価し、
4x4のクロス評価行列を生成する。

重要：訓練データ自体は全モデルで共通。異なるのはラベル付する将来窓のみ。
max-date（train-end）以降はラベル計算のためだけに使用される。

使用例:
    uv run python analysis/irl_only_experiments/run_irl_cross_evaluation.py \
        --reviews data/review_requests_nova.csv \
        --train-start 2021-01-01 \
        --train-end 2023-01-01 \
        --eval-start 2023-01-01 \
        --eval-end 2024-01-01 \
        --output analysis/irl_only_experiments/cross_eval_min0 \
        --min-history-events 0 \
        --hidden-dim 128 \
        --learning-rate 5e-05 \
        --dropout 0.2 \
        --seq-len 0 \
        --output-temperature 0.45 \
        --epochs 70 \
        --focal-alpha 0.5 \
        --focal-gamma 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# リポジトリルートをパスに追加
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem
from scripts.training.irl.train_irl_review_acceptance import (  # type: ignore
    extract_evaluation_trajectories,
    extract_review_acceptance_trajectories,
    find_optimal_threshold,
    load_review_requests,
)


@dataclass
class LabelWindow:
    """ラベル付する将来窓の定義"""
    name: str
    start_months: int
    end_months: int


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_label_windows() -> List[LabelWindow]:
    """4つのラベル窓を定義"""
    return [
        LabelWindow(name="0-3m", start_months=0, end_months=3),
        LabelWindow(name="3-6m", start_months=3, end_months=6),
        LabelWindow(name="6-9m", start_months=6, end_months=9),
        LabelWindow(name="9-12m", start_months=9, end_months=12),
    ]


def build_irl_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "hidden_dim": args.hidden_dim,
        "sequence": True,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "output_temperature": args.output_temperature,
    }


def weighted_positive_rate(trajectories: Sequence[Dict[str, Any]]) -> float:
    if not trajectories:
        return 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    for traj in trajectories:
        weight = float(traj.get("sample_weight", 1.0))
        label = 1.0 if traj.get("future_acceptance") else 0.0
        weighted_sum += weight * label
        weight_total += weight
    if weight_total == 0:
        return 0.0
    return weighted_sum / weight_total


def safe_find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        default_threshold = 0.5
        return {
            "threshold": default_threshold,
            "precision": float(np.mean(y_true)),
            "recall": 1.0 if np.mean(y_true) == 1.0 else 0.0,
            "f1": 0.0,
        }
    try:
        return find_optimal_threshold(y_true, y_prob)
    except ValueError:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = int(np.argmax(f1_scores))
        threshold = 0.5
        if len(thresholds) > 0 and best_idx < len(thresholds):
            threshold = float(thresholds[best_idx])
        return {
            "threshold": threshold,
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1_scores[best_idx]),
        }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if len(y_true) == 0:
        return {
            "auc_roc": math.nan,
            "auc_pr": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
            "threshold": threshold,
            "positive_rate": math.nan,
            "prediction_stats": {},
        }

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        metrics["auc_roc"] = math.nan
        metrics["auc_pr"] = math.nan
    else:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["threshold"] = float(threshold)
    metrics["positive_rate"] = float(np.mean(y_true))
    metrics["prediction_stats"] = {
        "min": float(np.min(y_prob)),
        "max": float(np.max(y_prob)),
        "mean": float(np.mean(y_prob)),
        "std": float(np.std(y_prob)),
        "median": float(np.median(y_prob)),
    }
    return metrics


def clean_nans(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_nans(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_nans(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clean_nans(v) for v in value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric
    return value


def to_serializable_matrix(matrix: np.ndarray) -> List[List[Optional[float]]]:
    serializable: List[List[Optional[float]]] = []
    for row in matrix:
        row_values: List[Optional[float]] = []
        for value in row:
            numeric = float(value)
            if math.isnan(numeric):
                row_values.append(None)
            else:
                row_values.append(numeric)
        serializable.append(row_values)
    return serializable


def train_irl_on_quarter(
    df: pd.DataFrame,
    quarter: QuarterSlice,
    args: argparse.Namespace,
    irl_config: Dict[str, Any],
) -> Tuple[Optional[TrainResult], Dict[str, Any]]:
    summary: Dict[str, Any] = {
        "name": quarter.name,
        "start": quarter.start.isoformat(),
        "end": quarter.end.isoformat(),
        "span_months": quarter.span_months,
    }

    trajectories = extract_review_acceptance_trajectories(
        df,
        train_start=quarter.start,
        train_end=quarter.end,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        min_history_requests=args.min_history_events,
        project=args.project,
        extended_label_window_months=args.extended_label_months,
    )

    summary["n_trajectories"] = len(trajectories)

    if not trajectories:
        summary["status"] = "no_trajectories"
        return None, summary

    positive_rate = weighted_positive_rate(trajectories)
    summary["positive_rate"] = positive_rate

    irl_system = RetentionIRLSystem(irl_config)
    irl_system.auto_tune_focal_loss(positive_rate)

    if args.focal_alpha is not None or args.focal_gamma is not None:
        alpha = args.focal_alpha if args.focal_alpha is not None else irl_system.focal_alpha
        gamma = args.focal_gamma if args.focal_gamma is not None else irl_system.focal_gamma
        irl_system.set_focal_loss_params(alpha, gamma)
        summary["focal_override"] = {"alpha": alpha, "gamma": gamma}
    else:
        summary["focal_override"] = None

    train_result = irl_system.train_irl_temporal_trajectories(
        trajectories,
        epochs=args.epochs,
    )

    train_summary = {
        "epochs_trained": train_result.get("epochs_trained", args.epochs),
        "final_loss": float(train_result.get("final_loss", 0.0)),
    }
    summary["training"] = train_summary

    y_true: List[int] = []
    y_prob: List[float] = []
    for traj in trajectories:
        developer = traj.get("developer", traj.get("developer_info", {}))
        result = irl_system.predict_continuation_probability_snapshot(
            developer,
            traj["activity_history"],
            traj["context_date"],
        )
        y_true.append(1 if traj.get("future_acceptance") else 0)
        y_prob.append(float(result["continuation_probability"]))

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)

    threshold_info = safe_find_optimal_threshold(y_true_arr, y_prob_arr)
    threshold = float(threshold_info.get("threshold", 0.5))

    train_metrics = compute_metrics(y_true_arr, y_prob_arr, threshold)
    summary["train_metrics"] = clean_nans(train_metrics)

    summary["threshold"] = threshold
    summary["threshold_info"] = threshold_info
    summary["status"] = "trained"

    return (
        TrainResult(
            model=irl_system,
            threshold=threshold,
            threshold_info=threshold_info,
            train_metrics=train_metrics,
            summary=summary,
        ),
        summary,
    )


def evaluate_on_quarter(
    df: pd.DataFrame,
    quarter: QuarterSlice,
    span_months: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    trajectories = extract_evaluation_trajectories(
        df,
        cutoff_date=quarter.end,
        history_window_months=span_months,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        min_history_requests=args.min_history_events,
        project=args.project,
        extended_label_window_months=args.extended_label_months,
    )

    summary = {
        "name": quarter.name,
        "start": quarter.start.isoformat(),
        "end": quarter.end.isoformat(),
        "span_months": span_months,
        "n_trajectories": len(trajectories),
    }

    if trajectories:
        labels = [1 if traj.get("future_acceptance") else 0 for traj in trajectories]
        summary["positive_rate"] = float(np.mean(labels))
    else:
        summary["positive_rate"] = None

    return trajectories, summary


def run_cross_evaluation(args: argparse.Namespace) -> None:
    df = load_review_requests(args.reviews)

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)

    train_quarters, train_span = split_period_into_quarters(train_start, train_end)
    eval_quarters, eval_span = split_period_into_quarters(eval_start, eval_end)

    irl_config = build_irl_config(args)

    train_results: List[Optional[TrainResult]] = []

    print("\n" + "=" * 70)
    print("IRL Cross-Evaluation (Training)")
    print("=" * 70)

    for quarter in train_quarters:
        print(f"\n[Train] {quarter.name} ({quarter.start.date()} to {quarter.end.date()})")
        trained, summary = train_irl_on_quarter(df, quarter, args, irl_config)
        train_results.append(trained)
        if summary.get("status") != "trained":
            print("  -> Skipped (no trajectories)")
        else:
            print(
                "  -> Trained: {n} trajectories, pos_rate={pr:.2%}, threshold={th:.4f}".format(
                    n=summary.get("n_trajectories", 0),
                    pr=summary.get("positive_rate", 0.0),
                    th=summary.get("threshold", 0.5),
                )
            )

    eval_sets: List[Dict[str, Any]] = []
    eval_summaries: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("IRL Cross-Evaluation (Evaluation Sets)")
    print("=" * 70)

    for quarter in eval_quarters:
        print(f"\n[Eval] {quarter.name} ({quarter.start.date()} to {quarter.end.date()})")
        trajectories, summary = evaluate_on_quarter(df, quarter, eval_span, args)
        eval_sets.append({"trajectories": trajectories, "summary": summary})
        eval_summaries.append(summary)
        print(
            "  -> Prepared: {n} trajectories, pos_rate={pr}".format(
                n=summary.get("n_trajectories", 0),
                pr=(
                    f"{summary['positive_rate']:.2%}"
                    if summary.get("positive_rate") is not None
                    else "N/A"
                ),
            )
        )

    metric_names = ["auc_roc", "auc_pr", "f1", "precision", "recall"]
    matrices = {
        name: np.full((4, 4), np.nan, dtype=float)
        for name in metric_names
    }
    valid_mask = np.zeros((4, 4), dtype=int)

    matrix_details: Dict[str, Any] = {}

    print("\n" + "=" * 70)
    print("Evaluating train/eval combinations")
    print("=" * 70)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for train_idx, train_result in enumerate(train_results):
        train_name = train_quarters[train_idx].name
        train_dir = output_dir / f"train_{train_name}"
        train_dir.mkdir(parents=True, exist_ok=True)

        # 訓練期間ディレクトリにモデルと対角評価metricsを保存
        if train_result is not None:
            model_path = train_dir / "irl_model.pt"
            train_result.model.save_model(str(model_path))

            # 訓練期間自体のメトリクス（対角線評価）を保存
            train_metrics_for_file = clean_nans(train_result.train_metrics)
            train_metrics_for_file["optimal_threshold"] = train_result.threshold
            train_metrics_for_file["threshold_source"] = "train_data"
            with open(train_dir / "metrics.json", "w") as f:
                json.dump(train_metrics_for_file, f, indent=2, ensure_ascii=False)

            with open(train_dir / "optimal_threshold.json", "w") as f:
                json.dump(clean_nans(train_result.threshold_info), f, indent=2, ensure_ascii=False)

        for eval_idx, eval_entry in enumerate(eval_sets):
            eval_summary = eval_entry["summary"]
            eval_name = eval_summary["name"]
            key = f"{train_name}_{eval_name}"
            detail: Dict[str, Any] = {
                "train_label": train_name,
                "eval_quarter": eval_summary,
                "status": None,
            }

            eval_dir = train_dir / f"eval_{eval_name}"
            eval_dir.mkdir(parents=True, exist_ok=True)

            if eval_idx < train_idx:
                detail["status"] = "skipped_past_window"
                matrix_details[key] = detail
                print(f"- {key}: skipped (eval window precedes train window)")
                # 空のメトリクスを書き込み
                with open(eval_dir / "metrics.json", "w") as f:
                    json.dump({"status": "skipped_past_window"}, f, indent=2, ensure_ascii=False)
                continue

            if train_result is None:
                detail["status"] = "no_train_model"
                matrix_details[key] = detail
                print(f"- {key}: skipped (no train model)")
                with open(eval_dir / "metrics.json", "w") as f:
                    json.dump({"status": "no_train_model"}, f, indent=2, ensure_ascii=False)
                continue

            eval_trajectories = eval_entry["trajectories"]
            if not eval_trajectories:
                detail["status"] = "no_eval_samples"
                matrix_details[key] = detail
                print(f"- {key}: skipped (no eval trajectories)")
                with open(eval_dir / "metrics.json", "w") as f:
                    json.dump({"status": "no_eval_samples"}, f, indent=2, ensure_ascii=False)
                continue

            y_true: List[int] = []
            y_prob: List[float] = []
            predictions_records: List[Dict[str, Any]] = []

            for traj in eval_trajectories:
                developer = traj.get("developer", traj.get("developer_info", {}))
                result = train_result.model.predict_continuation_probability_snapshot(
                    developer,
                    traj["activity_history"],
                    traj["context_date"],
                )
                prob = float(result["continuation_probability"])
                label = 1 if traj.get("future_acceptance") else 0
                y_true.append(label)
                y_prob.append(prob)

                # predictions.csv用レコード
                predictions_records.append({
                    "reviewer_email": developer.get("reviewer_email", "unknown"),
                    "predicted_prob": prob,
                    "true_label": label,
                    "history_acceptance_rate": developer.get("history_acceptance_rate", 0.0),
                })

            y_true_arr = np.array(y_true)
            y_prob_arr = np.array(y_prob)

            metrics = compute_metrics(y_true_arr, y_prob_arr, train_result.threshold)

            if len(np.unique(y_true_arr)) >= 2:
                eval_threshold_info = safe_find_optimal_threshold(y_true_arr, y_prob_arr)
            else:
                eval_threshold_info = {
                    "threshold": train_result.threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }

            metrics_clean = clean_nans(metrics)
            eval_threshold_clean = clean_nans(eval_threshold_info)
            train_threshold_clean = clean_nans(train_result.threshold_info)

            detail.update(
                {
                    "status": "evaluated",
                    "metrics": metrics_clean,
                    "eval_threshold": eval_threshold_clean,
                    "train_threshold": train_threshold_clean,
                    "n_eval": int(len(y_true_arr)),
                    "positives": int(y_true_arr.sum()),
                    "negatives": int(len(y_true_arr) - y_true_arr.sum()),
                }
            )

            for metric_name in metric_names:
                value = metrics.get(metric_name)
                if value is None:
                    matrices[metric_name][train_idx, eval_idx] = np.nan
                else:
                    matrices[metric_name][train_idx, eval_idx] = value

            valid_mask[train_idx, eval_idx] = 1

            matrix_details[key] = detail

            # 評価期間ディレクトリにmetrics.jsonとpredictions.csvを保存
            eval_metrics_for_file = clean_nans(metrics)
            eval_metrics_for_file["optimal_threshold"] = train_result.threshold
            eval_metrics_for_file["threshold_source"] = "train_data"
            eval_metrics_for_file["eval_optimal_threshold"] = eval_threshold_info["threshold"]
            eval_metrics_for_file["eval_optimal_f1"] = eval_threshold_info.get("f1", metrics["f1"])
            eval_metrics_for_file["positive_count"] = int(y_true_arr.sum())
            eval_metrics_for_file["negative_count"] = int(len(y_true_arr) - y_true_arr.sum())
            eval_metrics_for_file["total_count"] = int(len(y_true_arr))

            with open(eval_dir / "metrics.json", "w") as f:
                json.dump(eval_metrics_for_file, f, indent=2, ensure_ascii=False)

            pd.DataFrame(predictions_records).to_csv(eval_dir / "predictions.csv", index=False)

            print(
                f"- {key}: AUC-ROC={metrics['auc_roc'] if metrics['auc_roc'] is not None else math.nan:.3f}, "
                f"AUC-PR={metrics['auc_pr'] if metrics['auc_pr'] is not None else math.nan:.3f}, "
                f"F1={metrics['f1'] if metrics['f1'] is not None else math.nan:.3f}"
            )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_names = [q.name for q in train_quarters]
    eval_names = [q.name for q in eval_quarters]

    for metric_name, matrix in matrices.items():
        df_matrix = pd.DataFrame(matrix, index=train_names, columns=eval_names)
        df_matrix.to_csv(output_dir / f"matrix_{metric_name.upper()}.csv")

    pd.DataFrame(valid_mask, index=train_names, columns=eval_names).to_csv(
        output_dir / "matrix_VALID_MASK.csv"
    )

    cross_eval_results = {
        "config": {
            "reviews": str(args.reviews),
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "eval_start": eval_start.isoformat(),
            "eval_end": eval_end.isoformat(),
            "min_history_events": args.min_history_events,
            "future_window_start": args.future_window_start,
            "future_window_end": args.future_window_end,
            "extended_label_months": args.extended_label_months,
            "hidden_dim": args.hidden_dim,
            "seq_len": args.seq_len,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "output_temperature": args.output_temperature,
            "epochs": args.epochs,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
        },
        "eval_quarters": eval_summaries,
        "matrix": matrix_details,
        "metrics": {
            metric: to_serializable_matrix(matrix)
            for metric, matrix in matrices.items()
        },
        "valid_mask": valid_mask.tolist(),
    }

    with open(output_dir / "cross_eval_results.json", "w") as f:
        json.dump(cross_eval_results, f, indent=2, ensure_ascii=False)

    print("\n結果を保存しました: " + str(output_dir))
    for metric_name in metric_names:
        print(f"  - matrix_{metric_name.upper()}.csv")
    print("  - cross_eval_results.json")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IRL クロス評価を実施する")
    parser.add_argument("--reviews", type=Path, required=True, help="レビュー依頼CSVのパス")
    parser.add_argument("--train-start", type=str, required=True, help="訓練期間の開始日 (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, required=True, help="訓練期間の終了日 (YYYY-MM-DD)")
    parser.add_argument("--eval-start", type=str, required=True, help="評価期間の開始日 (YYYY-MM-DD)")
    parser.add_argument("--eval-end", type=str, required=True, help="評価期間の終了日 (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, required=True, help="結果保存先ディレクトリ")
    parser.add_argument("--project", type=str, default=None, help="対象プロジェクト (任意)")
    parser.add_argument("--min-history-events", type=int, default=0, help="履歴として必要なレビュー依頼数")
    parser.add_argument("--future-window-start", type=int, default=0, help="将来窓の開始 (月)")
    parser.add_argument("--future-window-end", type=int, default=3, help="将来窓の終了 (月)")
    parser.add_argument(
        "--extended-label-months",
        type=int,
        default=12,
        help="ラベル判定時に参照する拡張期間 (月)",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="IRL MLP/LSTM の隠れ次元")
    parser.add_argument("--seq-len", type=int, default=0, help="LSTM の系列長 (0 で自動) またはタイル長")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学習率")
    parser.add_argument("--dropout", type=float, default=0.2, help="ドロップアウト率")
    parser.add_argument(
        "--output-temperature",
        type=float,
        default=1.0,
        help="出力ロジットに掛ける温度 (確率スケーリング)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="訓練エポック数")
    parser.add_argument("--focal-alpha", type=float, default=None, help="Focal Loss の alpha 上書き")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Focal Loss の gamma 上書き")
    parser.add_argument("--state-dim", type=int, default=10, help="状態ベクトル次元")
    parser.add_argument("--action-dim", type=int, default=4, help="アクション数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)
    run_cross_evaluation(args)


if __name__ == "__main__":
    main()
