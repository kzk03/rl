#!/usr/bin/env python3
"""IRL クロス評価スクリプト（固定訓練期間版）。

訓練データ期間は固定（train-start～train-end）で、
将来窓（future-window）を0-3m, 3-6m, 6-9m, 9-12mと変えて
4つのモデルを訓練。各モデルを異なる評価窓（eval-start以降）で評価し、
4x4のクロス評価行列を生成する。

重要：訓練データ自体は全モデルで共通。異なるのはラベル付する将来窓のみ。
max-date（train-end）以降はラベル計算のためだけに使用される。

使用例:
    uv run python analysis/irl_only_experiments/run_irl_cross_evaluation_fixed.py \
        --reviews data/review_requests_nova.csv \
        --train-start 2021-01-01 \
        --train-end 2023-01-01 \
        --eval-start 2023-01-01 \
        --eval-end 2024-01-01 \
        --output analysis/irl_only_experiments/cross_eval_min0_temp045 \
        --min-history-events 0 \
        --hidden-dim 128 \
        --learning-rate 5e-05 \
        --dropout 0.2 \
        --seq-len 0 \
        --output-temperature 0.45 \
        --epochs 70 \
        --focal-alpha 0.5 \
        --focal-gamma 1.0 \
        --seed 777
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

# リポジトリルートをパスに追加
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sklearn.metrics import (
    average_precision_score,
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


@dataclass
class TrainedModel:
    """訓練済みモデルとメタデータ"""
    model: RetentionIRLSystem
    threshold: float
    threshold_info: Dict[str, float]
    train_metrics: Dict[str, Any]
    window: LabelWindow
    n_trajectories: int
    positive_rate: float


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


def clean_nans(obj: Any) -> Any:
    """NaN/Infを持つ辞書・リストを再帰的にクリーンアップ"""
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj


def to_serializable_matrix(matrix: np.ndarray) -> List[List[Optional[float]]]:
    """NumPy配列をJSON用リストに変換"""
    result = []
    for row in matrix:
        result_row = []
        for val in row:
            if math.isnan(val) or math.isinf(val):
                result_row.append(None)
            else:
                result_row.append(float(val))
        result.append(result_row)
    return result


def build_irl_config(args: argparse.Namespace) -> Dict[str, Any]:
    """IRL設定を構築"""
    return {
        "hidden_dim": args.hidden_dim,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "output_temperature": args.output_temperature,
    }


def safe_find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """最適閾値を安全に計算"""
    try:
        return find_optimal_threshold(y_true, y_prob)
    except Exception:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    """評価メトリクスを計算"""
    metrics = {}
    
    if len(np.unique(y_true)) >= 2:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["auc_pr"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auc_roc"] = None
        metrics["auc_pr"] = None
    
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    metrics["precision"] = float(prec)
    metrics["recall"] = float(rec)
    metrics["f1"] = float(f1)
    metrics["threshold"] = float(threshold)
    metrics["positive_rate"] = float(y_true.mean())
    metrics["prediction_stats"] = {
        "min": float(y_prob.min()),
        "max": float(y_prob.max()),
        "mean": float(y_prob.mean()),
        "std": float(y_prob.std()),
        "median": float(np.median(y_prob)),
    }
    
    return metrics


def train_model_for_window(
    df: pd.DataFrame,
    window: LabelWindow,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    args: argparse.Namespace,
    irl_config: Dict[str, Any],
) -> TrainedModel:
    """指定した将来窓でモデルを訓練
    
    訓練データ期間は全モデル共通（train_start～train_end）
    将来窓（future_window_start/end）でラベル付する期間だけが異なる
    """
    
    print(f"\n[Train] {window.name} (label window: {window.start_months}-{window.end_months} months after {train_end.date()})")
    
    # 訓練軌跡を抽出（将来窓のみが異なる）
    trajectories = extract_review_acceptance_trajectories(
        df,
        train_start=train_start,
        train_end=train_end,
        future_window_start_months=window.start_months,
        future_window_end_months=window.end_months,
        min_history_requests=args.min_history_events,
        project=args.project,
        extended_label_window_months=args.extended_label_months,
    )
    
    print(f"  訓練軌跡数: {len(trajectories)}")
    
    # IRLモデルを訓練
    irl_system = RetentionIRLSystem(config=irl_config)
    
    if args.focal_alpha is not None or args.focal_gamma is not None:
        irl_system.update_focal_loss_params(
            alpha=args.focal_alpha if args.focal_alpha is not None else irl_system.focal_alpha,
            gamma=args.focal_gamma if args.focal_gamma is not None else irl_system.focal_gamma,
        )
    
    train_result = irl_system.train_continuation_model(
        trajectories,
        epochs=args.epochs,
        return_details=True,
    )
    
    # 訓練データでメトリクス計算
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
    
    positive_rate = float(y_true_arr.mean())
    
    print(f"  -> 正例率={positive_rate:.2%}, threshold={threshold:.4f}")
    
    return TrainedModel(
        model=irl_system,
        threshold=threshold,
        threshold_info=threshold_info,
        train_metrics=train_metrics,
        window=window,
        n_trajectories=len(trajectories),
        positive_rate=positive_rate,
    )


def evaluate_model_on_window(
    trained: TrainedModel,
    eval_window: LabelWindow,
    df: pd.DataFrame,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """訓練済みモデルを評価窓で評価"""
    
    # 評価軌跡を抽出
    # cutoff_date = eval_start（評価期間の開始点）
    # 履歴窓は eval_window の長さ（3ヶ月）
    history_months = eval_window.end_months - eval_window.start_months
    
    trajectories = extract_evaluation_trajectories(
        df,
        cutoff_date=eval_start,
        history_window_months=history_months,
        future_window_start_months=eval_window.start_months,
        future_window_end_months=eval_window.end_months,
        min_history_requests=args.min_history_events,
        project=args.project,
        extended_label_window_months=args.extended_label_months,
    )
    
    if not trajectories:
        return {"status": "no_eval_samples"}
    
    # 予測
    y_true: List[int] = []
    y_prob: List[float] = []
    predictions_records: List[Dict[str, Any]] = []
    
    for traj in trajectories:
        developer = traj.get("developer", traj.get("developer_info", {}))
        result = trained.model.predict_continuation_probability_snapshot(
            developer,
            traj["activity_history"],
            traj["context_date"],
        )
        prob = float(result["continuation_probability"])
        label = 1 if traj.get("future_acceptance") else 0
        y_true.append(label)
        y_prob.append(prob)
        
        predictions_records.append({
            "reviewer_email": developer.get("reviewer_email", "unknown"),
            "predicted_prob": prob,
            "true_label": label,
            "history_acceptance_rate": developer.get("history_acceptance_rate", 0.0),
        })
    
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    
    # メトリクス計算
    metrics = compute_metrics(y_true_arr, y_prob_arr, trained.threshold)
    
    if len(np.unique(y_true_arr)) >= 2:
        eval_threshold_info = safe_find_optimal_threshold(y_true_arr, y_prob_arr)
    else:
        eval_threshold_info = {
            "threshold": trained.threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
    
    return {
        "status": "evaluated",
        "metrics": clean_nans(metrics),
        "eval_threshold": clean_nans(eval_threshold_info),
        "train_threshold": clean_nans(trained.threshold_info),
        "n_eval": int(len(y_true_arr)),
        "positives": int(y_true_arr.sum()),
        "negatives": int(len(y_true_arr) - y_true_arr.sum()),
        "predictions": predictions_records,
    }


def run_cross_evaluation(args: argparse.Namespace) -> None:
    """クロス評価を実行"""
    
    df = load_review_requests(args.reviews)
    
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    
    label_windows = get_label_windows()
    irl_config = build_irl_config(args)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # 1. 各将来窓でモデルを訓練
    # ========================================
    print("\n" + "=" * 70)
    print("IRL Cross-Evaluation (Training)")
    print("=" * 70)
    print(f"訓練期間: {train_start.date()} ～ {train_end.date()} (固定)")
    print(f"評価期間: {eval_start.date()} ～ {eval_end.date()}")
    print("=" * 70)
    
    trained_models: List[TrainedModel] = []
    
    for window in label_windows:
        trained = train_model_for_window(
            df, window, train_start, train_end, args, irl_config
        )
        trained_models.append(trained)
        
        # モデルを保存
        train_dir = output_dir / f"train_{window.name}"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = train_dir / "irl_model.pt"
        trained.model.save_model(str(model_path))
        
        # 訓練メトリクスを保存
        train_metrics_file = clean_nans(trained.train_metrics)
        train_metrics_file["optimal_threshold"] = trained.threshold
        train_metrics_file["threshold_source"] = "train_data"
        
        with open(train_dir / "metrics.json", "w") as f:
            json.dump(train_metrics_file, f, indent=2, ensure_ascii=False)
        
        with open(train_dir / "optimal_threshold.json", "w") as f:
            json.dump(clean_nans(trained.threshold_info), f, indent=2, ensure_ascii=False)
    
    # ========================================
    # 2. クロス評価
    # ========================================
    print("\n" + "=" * 70)
    print("IRL Cross-Evaluation (Evaluation)")
    print("=" * 70)
    
    metric_names = ["auc_roc", "auc_pr", "f1", "precision", "recall"]
    matrices = {
        name: np.full((4, 4), np.nan, dtype=float)
        for name in metric_names
    }
    
    matrix_details: Dict[str, Any] = {}
    
    for train_idx, trained in enumerate(trained_models):
        train_window = trained.window
        train_dir = output_dir / f"train_{train_window.name}"
        
        for eval_idx, eval_window in enumerate(label_windows):
            key = f"{train_window.name}_{eval_window.name}"
            
            eval_dir = train_dir / f"eval_{eval_window.name}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"評価中: {key}")
            
            result = evaluate_model_on_window(
                trained, eval_window, df, eval_start, eval_end, args
            )
            
            # 結果を保存
            if result["status"] == "evaluated":
                # メトリクスファイル
                eval_metrics = clean_nans(result["metrics"])
                eval_metrics["optimal_threshold"] = trained.threshold
                eval_metrics["threshold_source"] = "train_data"
                eval_metrics["eval_optimal_threshold"] = result["eval_threshold"]["threshold"]
                eval_metrics["eval_optimal_f1"] = result["eval_threshold"].get("f1", result["metrics"]["f1"])
                eval_metrics["positive_count"] = result["positives"]
                eval_metrics["negative_count"] = result["negatives"]
                eval_metrics["total_count"] = result["n_eval"]
                
                with open(eval_dir / "metrics.json", "w") as f:
                    json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
                
                # 予測CSVを保存
                pd.DataFrame(result["predictions"]).to_csv(eval_dir / "predictions.csv", index=False)
                
                # マトリクスを更新
                for metric_name in metric_names:
                    value = result["metrics"].get(metric_name)
                    if value is not None:
                        matrices[metric_name][train_idx, eval_idx] = value
                
                matrix_details[key] = {
                    "train_label": train_window.name,
                    "eval_label": eval_window.name,
                    "status": "evaluated",
                    "metrics": result["metrics"],
                    "n_eval": result["n_eval"],
                }
                
                print(f"  -> AUC-ROC={result['metrics']['auc_roc']:.3f}, "
                      f"AUC-PR={result['metrics']['auc_pr']:.3f}, "
                      f"F1={result['metrics']['f1']:.3f}")
            else:
                with open(eval_dir / "metrics.json", "w") as f:
                    json.dump({"status": result["status"]}, f, indent=2, ensure_ascii=False)
                
                matrix_details[key] = {
                    "train_label": train_window.name,
                    "eval_label": eval_window.name,
                    "status": result["status"],
                }
                print(f"  -> スキップ ({result['status']})")
    
    # ========================================
    # 3. マトリクスを保存
    # ========================================
    window_names = [w.name for w in label_windows]
    
    for metric_name, matrix in matrices.items():
        df_matrix = pd.DataFrame(matrix, index=window_names, columns=window_names)
        df_matrix.to_csv(output_dir / f"matrix_{metric_name.upper()}.csv")
    
    # サマリーJSONを保存
    cross_eval_results = {
        "config": {
            "reviews": str(args.reviews),
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "eval_start": eval_start.isoformat(),
            "eval_end": eval_end.isoformat(),
            "min_history_events": args.min_history_events,
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
        "label_windows": [
            {
                "name": w.name,
                "start_months": w.start_months,
                "end_months": w.end_months,
            }
            for w in label_windows
        ],
        "matrix": matrix_details,
        "metrics": {
            metric: to_serializable_matrix(matrix)
            for metric, matrix in matrices.items()
        },
    }
    
    with open(output_dir / "cross_eval_results.json", "w") as f:
        json.dump(cross_eval_results, f, indent=2, ensure_ascii=False)
    
    print("\n結果を保存しました: " + str(output_dir))
    for metric_name in metric_names:
        print(f"  - matrix_{metric_name.upper()}.csv")
    print("  - cross_eval_results.json")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IRL クロス評価を実施する（固定訓練期間版）")
    parser.add_argument("--reviews", type=Path, required=True, help="レビュー依頼CSVのパス")
    parser.add_argument("--train-start", type=str, required=True, help="訓練期間の開始日 (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, required=True, help="訓練期間の終了日 (YYYY-MM-DD)")
    parser.add_argument("--eval-start", type=str, required=True, help="評価期間の開始日 (YYYY-MM-DD)")
    parser.add_argument("--eval-end", type=str, required=True, help="評価期間の終了日 (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, required=True, help="結果保存先ディレクトリ")
    parser.add_argument("--project", type=str, default=None, help="対象プロジェクト (任意)")
    parser.add_argument("--min-history-events", type=int, default=0, help="履歴として必要なレビュー依頼数")
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
