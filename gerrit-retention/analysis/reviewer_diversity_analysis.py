"""Aggregate reviewer diversity metrics and hit rate trends for BI dashboards."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV_DIR = PROJECT_ROOT / "outputs" / "task_assign_cut2023" / "replay_csv_train"
EVAL_CSV_DIR = PROJECT_ROOT / "outputs" / "task_assign_cut2023" / "replay_csv"
TRAIN_WINDOWS_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "task_assign_cut2023"
    / "replay_eval_irl_train_windows.json"
)
EVAL_WINDOWS_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "task_assign_cut2023"
    / "replay_eval_irl_windows.json"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analytics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


WINDOW_ORDER = [
    "-24m--18m",
    "-18m--12m",
    "-12m--9m",
    "-9m--6m",
    "-6m--3m",
    "-3m-0m",
]

EVAL_WINDOW_ORDER = [
    "1m",
    "1m-3m",
    "3m-6m",
    "6m-9m",
    "9m-12m",
    "12m-18m",
    "18m-24m",
]

EVAL_TOP_N = 1


@dataclass
class DiversityMetrics:
    window: str
    total_assignments: int
    unique_reviewers: int
    effective_reviewers: float
    shannon_effective_reviewers: float
    top1_share: float
    top5_share: float
    top10_share: float
    gini: float


def _gini(counts: np.ndarray) -> float:
    if counts.size == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = counts.size
    cumulative = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def _counter_from_csv(csv_path: Path) -> tuple[Counter[str], int]:
    counter: Counter[str] = Counter()
    total = 0
    chunk_iter = pd.read_csv(
        csv_path,
        usecols=["reviewer_id", "selected"],
        dtype={"reviewer_id": "string", "selected": "int8"},
        chunksize=200_000,
    )
    for chunk in chunk_iter:
        mask = chunk["selected"] == 1
        if mask.any():
            reviewers = chunk.loc[mask, "reviewer_id"].dropna()
            counter.update(reviewers.tolist())
            total += reviewers.size
    return counter, total


def _counter_from_eval_csv(csv_path: Path, top_n: int) -> tuple[Counter[str], int]:
    counter: Counter[str] = Counter()
    total = 0
    chunk_iter = pd.read_csv(
        csv_path,
        usecols=["reviewer_id", "candidate_rank"],
        dtype={"reviewer_id": "string", "candidate_rank": "int32"},
        chunksize=200_000,
    )
    for chunk in chunk_iter:
        mask = chunk["candidate_rank"] <= top_n
        if mask.any():
            reviewers = chunk.loc[mask, "reviewer_id"].dropna()
            counter.update(reviewers.tolist())
            total += reviewers.size
    return counter, total


def compute_diversity_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[DiversityMetrics] = []
    top_reviewers_rows: list[dict[str, str | float | int]] = []

    for window in WINDOW_ORDER:
        csv_path = TRAIN_CSV_DIR / f"{window}.csv"
        if not csv_path.exists():
            # Some windows may have been renamed or removed; skip gracefully.
            continue
        counter, total = _counter_from_csv(csv_path)
        counts = np.array(list(counter.values()), dtype=np.float64)
        if total == 0:
            records.append(
                DiversityMetrics(
                    window=window,
                    total_assignments=0,
                    unique_reviewers=0,
                    effective_reviewers=0.0,
                    shannon_effective_reviewers=0.0,
                    top1_share=0.0,
                    top5_share=0.0,
                    top10_share=0.0,
                    gini=0.0,
                )
            )
            continue

        shares = counts / total
        hhi = float(np.sum(shares**2))
        effective_reviewers = 1.0 / hhi if hhi > 0 else 0.0
        entropy = -np.sum(np.where(shares > 0, shares * np.log(shares), 0.0))
        shannon_effective = float(np.exp(entropy))

        sorted_counts = np.sort(counts)[::-1]
        top1_share = float(sorted_counts[:1].sum() / total)
        top5_share = float(sorted_counts[:5].sum() / total)
        top10_share = float(sorted_counts[:10].sum() / total)
        gini = float(_gini(counts))

        records.append(
            DiversityMetrics(
                window=window,
                total_assignments=total,
                unique_reviewers=int(counts.size),
                effective_reviewers=effective_reviewers,
                shannon_effective_reviewers=shannon_effective,
                top1_share=top1_share,
                top5_share=top5_share,
                top10_share=top10_share,
                gini=gini,
            )
        )

        for rank, (reviewer_id, assignments) in enumerate(
            counter.most_common(10), start=1
        ):
            top_reviewers_rows.append(
                {
                    "window": window,
                    "rank": rank,
                    "reviewer_id": reviewer_id,
                    "assignments": assignments,
                    "share": assignments / total,
                }
            )

    diversity_df = pd.DataFrame([vars(record) for record in records])
    top_reviewers_df = pd.DataFrame(top_reviewers_rows)

    return diversity_df, top_reviewers_df


def compute_eval_diversity_metrics(top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[DiversityMetrics] = []
    top_reviewers_rows: list[dict[str, str | float | int]] = []

    for window in EVAL_WINDOW_ORDER:
        csv_path = EVAL_CSV_DIR / f"{window}.csv"
        if not csv_path.exists():
            continue
        counter, total = _counter_from_eval_csv(csv_path, top_n)
        counts = np.array(list(counter.values()), dtype=np.float64)
        if total == 0:
            records.append(
                DiversityMetrics(
                    window=window,
                    total_assignments=0,
                    unique_reviewers=0,
                    effective_reviewers=0.0,
                    shannon_effective_reviewers=0.0,
                    top1_share=0.0,
                    top5_share=0.0,
                    top10_share=0.0,
                    gini=0.0,
                )
            )
            continue

        shares = counts / total
        hhi = float(np.sum(shares**2))
        effective_reviewers = 1.0 / hhi if hhi > 0 else 0.0
        entropy = -np.sum(np.where(shares > 0, shares * np.log(shares), 0.0))
        shannon_effective = float(np.exp(entropy))

        sorted_counts = np.sort(counts)[::-1]
        top1_share = float(sorted_counts[:1].sum() / total)
        top5_share = float(sorted_counts[:5].sum() / total)
        top10_share = float(sorted_counts[:10].sum() / total)
        gini = float(_gini(counts))

        records.append(
            DiversityMetrics(
                window=window,
                total_assignments=total,
                unique_reviewers=int(counts.size),
                effective_reviewers=effective_reviewers,
                shannon_effective_reviewers=shannon_effective,
                top1_share=top1_share,
                top5_share=top5_share,
                top10_share=top10_share,
                gini=gini,
            )
        )

        for rank, (reviewer_id, assignments) in enumerate(
            counter.most_common(10), start=1
        ):
            top_reviewers_rows.append(
                {
                    "window": window,
                    "rank": rank,
                    "reviewer_id": reviewer_id,
                    "assignments": assignments,
                    "share": assignments / total,
                }
            )

    diversity_df = pd.DataFrame([vars(record) for record in records])
    top_reviewers_df = pd.DataFrame(top_reviewers_rows)

    return diversity_df, top_reviewers_df


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def collect_hit_rate_trends() -> pd.DataFrame:
    records: list[dict[str, str | float | int | None]] = []

    if TRAIN_WINDOWS_JSON.exists():
        train_data = _load_json(TRAIN_WINDOWS_JSON)
        train_results = train_data.get("results", {})
        for window in WINDOW_ORDER:
            metrics = train_results.get(window)
            if not metrics:
                continue
            records.append(
                {
                    "phase": "train",
                    "window": window,
                    "steps": metrics.get("steps", 0),
                    "top1_hit_rate": metrics.get("top1_hit_rate")
                    if metrics.get("top1_hit_rate") is not None
                    else metrics.get("action_match_rate"),
                    "top3_hit_rate": metrics.get("top3_hit_rate"),
                    "top5_hit_rate": metrics.get("top5_hit_rate"),
                    "mAP": metrics.get("mAP"),
                }
            )

    if EVAL_WINDOWS_JSON.exists():
        eval_data = _load_json(EVAL_WINDOWS_JSON)
        eval_results = eval_data.get("results", {})
        for window in EVAL_WINDOW_ORDER:
            metrics = eval_results.get(window)
            if not metrics:
                continue
            records.append(
                {
                    "phase": "eval",
                    "window": window,
                    "steps": metrics.get("steps", 0),
                    "top1_hit_rate": metrics.get("top1_hit_rate")
                    if metrics.get("top1_hit_rate") is not None
                    else metrics.get("action_match_rate"),
                    "top3_hit_rate": metrics.get("top3_hit_rate"),
                    "top5_hit_rate": metrics.get("top5_hit_rate"),
                    "mAP": metrics.get("mAP"),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    phase_order_map = {
        "train": WINDOW_ORDER,
        "eval": EVAL_WINDOW_ORDER,
    }

    def sort_key(row: pd.Series) -> tuple[int, int]:
        phase = row["phase"]
        window = row["window"]
        phase_rank = 0 if phase == "train" else 1
        order = phase_order_map.get(phase, [])
        window_rank = order.index(window) if window in order else len(order)
        return phase_rank, window_rank

    df["_sort"] = df.apply(sort_key, axis=1)
    df.sort_values(by="_sort", inplace=True)
    df.drop(columns="_sort", inplace=True)
    df = df[df["steps"].fillna(0) > 0].reset_index(drop=True)
    return df


def _ensure_window_category(windows: Iterable[str]) -> list[str]:
    order = {name: idx for idx, name in enumerate(WINDOW_ORDER)}
    return sorted(windows, key=lambda name: order.get(name, len(order)))


def plot_diversity(diversity_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = diversity_df.set_index("window").loc[
        _ensure_window_category(diversity_df["window"].unique())
    ]
    ax.plot(ordered.index, ordered["unique_reviewers"], marker="o", label="Unique reviewers")
    ax.plot(
        ordered.index,
        ordered["effective_reviewers"],
        marker="s",
        label="Effective reviewers (1/HHI)",
    )
    ax.plot(
        ordered.index,
        ordered["shannon_effective_reviewers"],
        marker="^",
        label="Entropy-effective reviewers",
    )
    ax.set_xlabel("Training window")
    ax.set_ylabel("Reviewer count")
    ax.set_title("Reviewer diversity over training windows")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    output_path = OUTPUT_DIR / "reviewer_diversity_train.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_hit_rates(hit_rate_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    eval_df = hit_rate_df[hit_rate_df["phase"] == "eval"].copy()
    if eval_df.empty:
        return OUTPUT_DIR / "hit_rate_eval.png"
    eval_df["window"] = pd.Categorical(
        eval_df["window"], categories=EVAL_WINDOW_ORDER, ordered=True
    )
    eval_df.sort_values(by="window", inplace=True)
    ax.plot(eval_df["window"], eval_df["top3_hit_rate"], marker="o", label="Top-3 hit")
    ax.plot(eval_df["window"], eval_df["top5_hit_rate"], marker="s", label="Top-5 hit")
    if eval_df["top1_hit_rate"].notna().any():
        ax.plot(eval_df["window"], eval_df["top1_hit_rate"], marker="^", label="Top-1 hit")
    ax.set_xlabel("Evaluation horizon")
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1)
    ax.set_title("Evaluation hit rate trends")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    output_path = OUTPUT_DIR / "hit_rate_eval.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_eval_diversity(diversity_df: pd.DataFrame, top_n: int) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    if diversity_df.empty:
        return OUTPUT_DIR / "reviewer_diversity_eval.png"
    ordered = diversity_df.set_index("window").loc[
        [w for w in EVAL_WINDOW_ORDER if w in diversity_df["window"].values]
    ]
    ax.plot(ordered.index, ordered["unique_reviewers"], marker="o", label="Unique reviewers")
    ax.plot(
        ordered.index,
        ordered["effective_reviewers"],
        marker="s",
        label="Effective reviewers (1/HHI)",
    )
    ax.plot(
        ordered.index,
        ordered["shannon_effective_reviewers"],
        marker="^",
        label="Entropy-effective reviewers",
    )
    ax.set_xlabel("Evaluation window")
    ax.set_ylabel("Reviewer count")
    ax.set_title(f"Evaluation reviewer diversity (Top-{top_n})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    output_path = OUTPUT_DIR / f"reviewer_diversity_eval_top{top_n}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    diversity_df, top_reviewers_df = compute_diversity_metrics()
    diversity_df.to_csv(OUTPUT_DIR / "reviewer_diversity_train.csv", index=False)
    top_reviewers_df.to_csv(OUTPUT_DIR / "top_reviewers_train.csv", index=False)

    eval_diversity_df, eval_top_reviewers_df = compute_eval_diversity_metrics(EVAL_TOP_N)
    eval_diversity_df.to_csv(
        OUTPUT_DIR / f"reviewer_diversity_eval_top{EVAL_TOP_N}.csv", index=False
    )
    eval_top_reviewers_df.to_csv(
        OUTPUT_DIR / f"top_reviewers_eval_top{EVAL_TOP_N}.csv", index=False
    )

    hit_rate_df = collect_hit_rate_trends()
    hit_rate_df.to_csv(OUTPUT_DIR / "hit_rate_trends.csv", index=False)

    diversity_plot = plot_diversity(diversity_df)
    eval_diversity_plot = plot_eval_diversity(eval_diversity_df, EVAL_TOP_N)
    hit_rate_plot = plot_hit_rates(hit_rate_df)

    print(f"Saved diversity metrics to {OUTPUT_DIR / 'reviewer_diversity_train.csv'}")
    print(f"Saved top reviewer table to {OUTPUT_DIR / 'top_reviewers_train.csv'}")
    print(
        "Saved evaluation diversity metrics to "
        f"{OUTPUT_DIR / f'reviewer_diversity_eval_top{EVAL_TOP_N}.csv'}"
    )
    print(
        "Saved evaluation top reviewer table to "
        f"{OUTPUT_DIR / f'top_reviewers_eval_top{EVAL_TOP_N}.csv'}"
    )
    print(f"Saved hit rate metrics to {OUTPUT_DIR / 'hit_rate_trends.csv'}")
    print(f"Saved diversity plot to {diversity_plot}")
    print(f"Saved evaluation diversity plot to {eval_diversity_plot}")
    print(f"Saved evaluation hit rate plot to {hit_rate_plot}")


if __name__ == "__main__":
    main()
