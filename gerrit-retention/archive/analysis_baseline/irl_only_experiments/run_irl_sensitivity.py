#!/usr/bin/env python3
"""IRLモデル専用のハイパーパラメータ感度分析スクリプト。

既存の Logistic Regression などには触れず、IRL 設定のみを変えながら
訓練と評価を自動で繰り返すための軽量ツール。
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

# リポジトリルートをパスに追加
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem
from scripts.training.irl.train_irl_review_acceptance import (  # type: ignore
    extract_evaluation_trajectories,
    extract_review_acceptance_trajectories,
    find_optimal_threshold,
    load_review_requests,
)

DEFAULT_DATASET = REPO_ROOT / "data/review_requests_nova.csv"
DEFAULT_OUTPUT = REPO_ROOT / "analysis/irl_only_experiments/experiments"


@dataclass
class ExperimentConfig:
    name: str
    hidden_dim: int
    learning_rate: float
    dropout: float
    seq_len: int
    output_temperature: float
    epochs: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    future_window_start: int
    future_window_end: int
    min_history_events: int
    seed: int
    focal_alpha: Optional[float] = None
    focal_gamma: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "output_temperature": self.output_temperature,
            "epochs": self.epochs,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "eval_start": self.eval_start.isoformat(),
            "eval_end": self.eval_end.isoformat(),
            "future_window_start": self.future_window_start,
            "future_window_end": self.future_window_end,
            "min_history_events": self.min_history_events,
            "seed": self.seed,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
        }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def month_offset(start: pd.Timestamp, months: int) -> pd.Timestamp:
    return start + pd.DateOffset(months=months)


def build_grid(args: argparse.Namespace) -> List[ExperimentConfig]:
    base_train_start = pd.Timestamp(args.train_start)
    base_train_end = pd.Timestamp(args.train_end)
    base_eval_start = pd.Timestamp(args.eval_start)
    base_eval_end = pd.Timestamp(args.eval_end)

    def quick_preset() -> List[ExperimentConfig]:
        configs = []
        patterns = [
            {
                "hidden_dim": 128,
                "learning_rate": 1e-4,
                "dropout": 0.2,
                "seq_len": 0,
                "output_temperature": 1.0,
                "epochs": 30,
                "tag": "baseline_cfg",
                "train_months": None,
            },
            {
                "hidden_dim": 192,
                "learning_rate": 1e-4,
                "dropout": 0.1,
                "seq_len": 12,
                "output_temperature": 1.0,
                "epochs": 40,
                "tag": "deep_seq12",
                "train_months": None,
            },
            {
                "hidden_dim": 192,
                "learning_rate": 5e-4,
                "dropout": 0.0,
                "seq_len": 12,
                "output_temperature": 0.85,
                "epochs": 45,
                "tag": "no_drop_high_lr_clip9m",
                "train_months": 21,
            },
        ]
        for idx, p in enumerate(patterns):
            train_end = (
                month_offset(base_train_start, p["train_months"]) if p["train_months"] else base_train_end
            )
            configs.append(
                ExperimentConfig(
                    name=f"quick_{idx+1:02d}_{p['tag']}",
                    hidden_dim=p["hidden_dim"],
                    learning_rate=p["learning_rate"],
                    dropout=p["dropout"],
                    seq_len=p["seq_len"],
                    output_temperature=p["output_temperature"],
                    epochs=p["epochs"],
                    train_start=base_train_start,
                    train_end=train_end,
                    eval_start=base_eval_start,
                    eval_end=base_eval_end,
                    future_window_start=args.future_window_start,
                    future_window_end=args.future_window_end,
                    min_history_events=args.min_history_events,
                    seed=args.seed,
                    focal_alpha=args.focal_alpha,
                    focal_gamma=args.focal_gamma,
                )
            )
        return configs

    def extended_preset() -> List[ExperimentConfig]:
        hd_list = [128, 192, 256]
        lr_list = [1e-4, 3e-4]
        dropout_list = [0.2, 0.1, 0.0]
        seq_list = [0, 12]
        temp_list = [1.0, 0.9]
        configs = []
        counter = 1
        for hd, lr, dp, seq, temp in itertools.product(hd_list, lr_list, dropout_list, seq_list, temp_list):
            train_months = 21 if dp == 0.0 else None
            epochs = 40 if hd >= 192 else 30
            configs.append(
                ExperimentConfig(
                    name=(
                        f"ext_{counter:03d}_hd{hd}_lr{lr:.0e}_do{dp:.2f}_seq{seq}_temp{temp:.2f}"
                    ),
                    hidden_dim=hd,
                    learning_rate=lr,
                    dropout=dp,
                    seq_len=seq,
                    output_temperature=temp,
                    epochs=epochs,
                    train_start=base_train_start,
                    train_end=(
                        month_offset(base_train_start, train_months)
                        if train_months
                        else base_train_end
                    ),
                    eval_start=base_eval_start,
                    eval_end=base_eval_end,
                    future_window_start=args.future_window_start,
                    future_window_end=args.future_window_end,
                    min_history_events=args.min_history_events,
                    seed=args.seed,
                    focal_alpha=args.focal_alpha,
                    focal_gamma=args.focal_gamma,
                )
            )
            counter += 1
        return configs

    def custom_grid() -> List[ExperimentConfig]:
        hidden_dims = args.hidden_dim or [128]
        learning_rates = args.learning_rate or [1e-4]
        dropouts = args.dropout or [0.2]
        seq_lens = args.seq_len or [0]
        temps = args.temperature or [1.0]
        train_months_list = args.train_months or [None]
        configs = []
        counter = 1
        for hd, lr, dp, seq, temp, tm in itertools.product(
            hidden_dims, learning_rates, dropouts, seq_lens, temps, train_months_list
        ):
            configs.append(
                ExperimentConfig(
                    name=(
                        f"custom_{counter:03d}_hd{hd}_lr{lr:.0e}_do{dp:.2f}_seq{seq}_temp{temp:.2f}"
                    ),
                    hidden_dim=hd,
                    learning_rate=lr,
                    dropout=dp,
                    seq_len=seq,
                    output_temperature=temp,
                    epochs=args.epochs,
                    train_start=base_train_start,
                    train_end=(
                        month_offset(base_train_start, tm) if tm else base_train_end
                    ),
                    eval_start=base_eval_start,
                    eval_end=base_eval_end,
                    future_window_start=args.future_window_start,
                    future_window_end=args.future_window_end,
                    min_history_events=args.min_history_events,
                    seed=args.seed,
                    focal_alpha=args.focal_alpha,
                    focal_gamma=args.focal_gamma,
                )
            )
            counter += 1
        return configs

    if args.preset == "quick":
        grid = quick_preset()
    elif args.preset == "extended":
        grid = extended_preset()
    else:
        grid = custom_grid()

    if args.limit is not None:
        grid = grid[: args.limit]
    return grid


def train_and_evaluate(
    df: pd.DataFrame,
    exp: ExperimentConfig,
    dataset_path: Path,
    output_root: Path,
    project: Optional[str],
    dry_run: bool,
) -> Dict[str, object]:
    set_global_seed(exp.seed)

    exp_dir = output_root / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary = exp.to_dict()
    summary["dataset"] = str(dataset_path)
    summary["project"] = project

    if dry_run:
        (exp_dir / "_DRY_RUN").touch()
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return {"status": "dry_run", "experiment": exp.name}

    # 訓練用軌跡
    train_trajectories = extract_review_acceptance_trajectories(
        df,
        train_start=exp.train_start,
        train_end=exp.train_end,
        future_window_start_months=exp.future_window_start,
        future_window_end_months=exp.future_window_end,
        min_history_requests=exp.min_history_events,
        project=project,
    )
    if not train_trajectories:
        summary["error"] = "no_train_trajectories"
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return {"status": "failed", "experiment": exp.name}

    # IRL設定
    irl_config = {
        "state_dim": 10,
        "action_dim": 4,
        "hidden_dim": exp.hidden_dim,
        "sequence": True,
        "seq_len": exp.seq_len,
        "learning_rate": exp.learning_rate,
        "dropout": exp.dropout,
        "output_temperature": exp.output_temperature,
    }
    irl_system = RetentionIRLSystem(irl_config)

    # Focal Loss 調整
    positive_count = sum(1 for t in train_trajectories if t.get("future_acceptance"))
    positive_rate = positive_count / len(train_trajectories)
    irl_system.auto_tune_focal_loss(positive_rate)

    if exp.focal_alpha is not None or exp.focal_gamma is not None:
        override_alpha = exp.focal_alpha if exp.focal_alpha is not None else irl_system.focal_alpha
        override_gamma = exp.focal_gamma if exp.focal_gamma is not None else irl_system.focal_gamma
        irl_system.set_focal_loss_params(override_alpha, override_gamma)
        summary.setdefault("focal_override", {})
        summary["focal_override"].update({
            "alpha": override_alpha,
            "gamma": override_gamma,
        })

    # 訓練
    train_result = irl_system.train_irl_temporal_trajectories(
        train_trajectories,
        epochs=exp.epochs,
    )
    summary["training"] = train_result

    # 訓練データで閾値決定
    train_y_true: List[int] = []
    train_y_pred: List[float] = []
    for traj in train_trajectories:
        developer = traj.get("developer", traj.get("developer_info", {}))
        result = irl_system.predict_continuation_probability_snapshot(
            developer,
            traj["activity_history"],
            traj["context_date"],
        )
        train_y_true.append(1 if traj.get("future_acceptance") else 0)
        train_y_pred.append(float(result["continuation_probability"]))

    train_y_true_arr = np.array(train_y_true)
    train_y_pred_arr = np.array(train_y_pred)
    threshold_info = find_optimal_threshold(train_y_true_arr, train_y_pred_arr)
    summary["train_threshold"] = threshold_info

    # 評価
    history_window_months = int((exp.train_end - exp.train_start).days / 30)
    eval_trajectories = extract_evaluation_trajectories(
        df,
        cutoff_date=exp.train_end,
        history_window_months=history_window_months,
        future_window_start_months=exp.future_window_start,
        future_window_end_months=exp.future_window_end,
        min_history_requests=exp.min_history_events,
        project=project,
    )
    if not eval_trajectories:
        summary["error"] = "no_eval_trajectories"
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return {"status": "failed", "experiment": exp.name}

    y_true: List[int] = []
    y_prob: List[float] = []
    prediction_rows: List[Dict[str, object]] = []
    for traj in eval_trajectories:
        result = irl_system.predict_continuation_probability_snapshot(
            traj["developer"],
            traj["activity_history"],
            traj["context_date"],
        )
        prob = float(result["continuation_probability"])
        label = 1 if traj.get("future_acceptance") else 0
        y_true.append(label)
        y_prob.append(prob)
        prediction_rows.append(
            {
                "reviewer_email": traj.get("reviewer", "unknown"),
                "predicted_prob": prob,
                "true_label": label,
                "history_request_count": traj.get("history_request_count"),
                "history_acceptance_rate": traj.get("developer", {}).get("acceptance_rate"),
                "eval_request_count": traj.get("eval_request_count"),
                "eval_accepted_count": traj.get("eval_accepted_count"),
                "eval_rejected_count": traj.get("eval_rejected_count"),
            }
        )

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    optimal_threshold = threshold_info["threshold"]
    y_pred_binary = (y_prob_arr >= optimal_threshold).astype(int)

    auc_roc = float(torchmetrics_auc_roc(y_true_arr, y_prob_arr))
    auc_pr = float(torchmetrics_auc_pr(y_true_arr, y_prob_arr))
    precision = safe_precision(y_true_arr, y_pred_binary)
    recall = safe_recall(y_true_arr, y_pred_binary)
    f1 = safe_f1(y_true_arr, y_pred_binary)

    eval_threshold_info = find_optimal_threshold(y_true_arr, y_prob_arr)

    metrics = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "optimal_threshold": optimal_threshold,
        "eval_optimal_threshold": eval_threshold_info["threshold"],
        "eval_optimal_f1": eval_threshold_info["f1"],
        "positive_count": int(y_true_arr.sum()),
        "negative_count": int((1 - y_true_arr).sum()),
        "total_count": int(len(y_true_arr)),
        "prediction_stats": {
            "min": float(y_prob_arr.min()),
            "max": float(y_prob_arr.max()),
            "mean": float(y_prob_arr.mean()),
            "std": float(y_prob_arr.std()),
            "median": float(np.median(y_prob_arr)),
        },
    }
    summary["metrics"] = metrics

    # 保存
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(exp_dir / "threshold.json", "w") as f:
        json.dump(threshold_info, f, indent=2)
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(prediction_rows).to_csv(exp_dir / "predictions.csv", index=False)

    model_path = exp_dir / "model_state.pt"
    torch.save(irl_system.network.state_dict(), model_path)

    summary_md_lines = [
        f"# {exp.name}",
        "",
        "## Config",
        "",
        json.dumps(exp.to_dict(), indent=2),
        "",
        "## Metrics",
        "",
        json.dumps(metrics, indent=2),
    ]
    (exp_dir / "summary.md").write_text("\n".join(summary_md_lines))

    return {
        "status": "ok",
        "experiment": exp.name,
        "metrics": metrics,
        "summary_path": str(exp_dir / "summary.json"),
    }


def torchmetrics_auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def torchmetrics_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def safe_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import precision_score

    return float(precision_score(y_true, y_pred, zero_division=0))


def safe_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import recall_score

    return float(recall_score(y_true, y_pred, zero_division=0))


def safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, zero_division=0))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IRL-only sensitivity runner")
    parser.add_argument("--preset", choices=["quick", "extended", "custom"], default="quick")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--train-start", type=str, default="2021-01-01")
    parser.add_argument("--train-end", type=str, default="2023-01-01")
    parser.add_argument("--eval-start", type=str, default="2023-01-01")
    parser.add_argument("--eval-end", type=str, default="2024-01-01")
    parser.add_argument("--future-window-start", type=int, default=0)
    parser.add_argument("--future-window-end", type=int, default=3)
    parser.add_argument("--min-history-events", type=int, default=3)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)

    # custom grid overrides
    parser.add_argument("--hidden-dim", nargs="*", type=int)
    parser.add_argument("--learning-rate", nargs="*", type=float)
    parser.add_argument("--dropout", nargs="*", type=float)
    parser.add_argument("--seq-len", nargs="*", type=int)
    parser.add_argument("--temperature", nargs="*", type=float)
    parser.add_argument("--train-months", nargs="*", type=int)

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    dataset_path = args.dataset if args.dataset.is_absolute() else (REPO_ROOT / args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    output_root = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_review_requests(str(dataset_path))

    grid = build_grid(args)
    print(f"Running {len(grid)} experiment(s) with preset '{args.preset}'")

    results: List[Dict[str, object]] = []
    for exp in grid:
        print(f"[START] {exp.name}")
        result = train_and_evaluate(
            df=df,
            exp=exp,
            dataset_path=dataset_path,
            output_root=output_root,
            project=args.project,
            dry_run=args.dry_run,
        )
        results.append(result)
        print(f"[DONE ] {exp.name} -> {result['status']}")

    summary_path = output_root / "latest_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
