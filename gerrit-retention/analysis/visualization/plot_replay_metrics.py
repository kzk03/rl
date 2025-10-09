from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AnnualSpec:
    label: str
    json_path: Path
    window: str


@dataclass(frozen=True)
class QuarterlySpec:
    json_path: Path
    pattern: str = r"^(?P<start>\d+)m-(?P<end>\d+)m$"


def _default_annual_specs(repo_root: Path) -> List[AnnualSpec]:
    return [
        AnnualSpec(
            label="1y",
            json_path=repo_root
            / "outputs/task_assign_nova_botless_full_2y/replay_eval_irl_annual.json",
            window="12m",
        ),
        AnnualSpec(
            label="2y",
            json_path=repo_root
            / "outputs/task_assign_nova_botless_full_2y/replay_eval_irl_annual.json",
            window="24m",
        ),
        AnnualSpec(
            label="3y",
            json_path=repo_root
            / "outputs/task_assign_nova_botless_full_eval_3y/replay_eval_irl_annual.json",
            window="36m",
        ),
        AnnualSpec(
            label="4y",
            json_path=repo_root
            / "outputs/task_assign_nova_botless_full_4y/replay_eval_irl_annual.json",
            window="48m",
        ),
    ]


def _default_quarterly_spec(repo_root: Path) -> QuarterlySpec:
    return QuarterlySpec(
        json_path=repo_root
        / "outputs/task_assign_nova_botless_full_4y/replay_eval_irl_3m.json"
    )


def _load_json(json_path: Path) -> Dict[str, object]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def _collect_annual_data(
    specs: Sequence[AnnualSpec],
) -> Tuple[List[str], Dict[str, List[float]], List[str]]:
    metrics: Dict[str, List[float]] = {
        "top1": [],
        "top3": [],
        "top5": [],
        "mAP": [],
        "ECE": [],
        "avg_candidates": [],
    }
    labels: List[str] = []
    info_lines: List[str] = []
    for spec in specs:
        data = _load_json(spec.json_path)
        results = data.get("results", {})
        if spec.window not in results:
            raise KeyError(
                f"Window '{spec.window}' が {spec.json_path} に存在しません。"
            )
        entry = results[spec.window]
        cutoff = data.get("cutoff")
        model_dir = spec.json_path.parent.name
        labels.append(spec.label)
        metrics["top1"].append(float(entry.get("action_match_rate") or 0.0))
        metrics["top3"].append(float(entry.get("top3_hit_rate") or 0.0))
        metrics["top5"].append(float(entry.get("top5_hit_rate") or 0.0))
        metrics["mAP"].append(float(entry.get("mAP") or 0.0))
        metrics["ECE"].append(float(entry.get("ECE") or 0.0))
        metrics["avg_candidates"].append(float(entry.get("avg_candidates") or 0.0))
        info_lines.append(
            f"{spec.label}: cutoff={cutoff}, window={spec.window}, model_dir={model_dir}"
        )
    return labels, metrics, info_lines


def _parse_window_start(window: str, pattern: str) -> int:
    m = re.match(pattern, window)
    if not m:
        raise ValueError(f"ウィンドウ '{window}' がパターン '{pattern}' に一致しません。")
    return int(m.group("start"))


def _collect_quarterly_data(spec: QuarterlySpec) -> Tuple[List[int], Dict[str, List[float]]]:
    data = _load_json(spec.json_path)
    results = data.get("results", {})
    filtered = {
        window: entry
        for window, entry in results.items()
        if re.match(spec.pattern, window)
    }
    windows_sorted = sorted(filtered.keys(), key=lambda w: _parse_window_start(w, spec.pattern))
    months: List[int] = []
    metrics: Dict[str, List[float]] = {
        "top1": [],
        "top3": [],
        "top5": [],
        "mAP": [],
        "ECE": [],
    }
    for window in windows_sorted:
        entry = filtered[window]
        months.append(_parse_window_start(window, spec.pattern))
        metrics["top1"].append(float(entry.get("action_match_rate") or 0.0))
        metrics["top3"].append(float(entry.get("top3_hit_rate") or 0.0))
        metrics["top5"].append(float(entry.get("top5_hit_rate") or 0.0))
        metrics["mAP"].append(float(entry.get("mAP") or 0.0))
        metrics["ECE"].append(float(entry.get("ECE") or 0.0))
    cutoff = data.get("cutoff")
    meta = (
        f"cutoff={cutoff}, windows matching '{spec.pattern}', model_dir={spec.json_path.parent.name}"
    )
    return months, metrics, meta


def _plot_annual(
    labels: Sequence[str],
    metrics: Dict[str, List[float]],
    info_lines: Sequence[str],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(labels, metrics["top1"], marker="o", label="Top-1")
    axes[0].plot(labels, metrics["top3"], marker="o", label="Top-3")
    axes[0].plot(labels, metrics["top5"], marker="o", label="Top-5")
    axes[0].set_ylabel("Hit Rate")
    axes[0].set_title("IRL Replay Evaluation (Annual)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(labels, metrics["mAP"], marker="o", color="#2ca02c", label="mAP")
    axes[1].plot(labels, metrics["ECE"], marker="o", color="#d62728", label="ECE")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Evaluation Horizon")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0.18, 1, 1])
    if info_lines:
        fig.text(
            0.5,
            0.05,
            "\n".join(info_lines),
            ha="center",
            va="center",
            fontsize=9,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_quarterly(
    months: Sequence[int],
    metrics: Dict[str, List[float]],
    meta_info: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(months, metrics["top1"], marker="o", label="Top-1")
    axes[0].plot(months, metrics["top3"], marker="o", label="Top-3")
    axes[0].plot(months, metrics["top5"], marker="o", label="Top-5")
    axes[0].set_ylabel("Hit Rate")
    axes[0].set_title("IRL Replay Evaluation (Quarterly)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(months, metrics["mAP"], marker="o", color="#2ca02c", label="mAP")
    axes[1].plot(months, metrics["ECE"], marker="o", color="#d62728", label="ECE")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Elapsed Months (window start)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.text(0.5, 0.05, meta_info, ha="center", va="center", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _parse_annual_arg(entries: Iterable[str], repo_root: Path) -> List[AnnualSpec]:
    specs: List[AnnualSpec] = []
    for ent in entries:
        try:
            label, rest = ent.split("=", 1)
            json_path_str, window = rest.split(":", 1)
        except ValueError as exc:  # pragma: no cover - CLI validation
            raise ValueError(
                "--annual の値は 'label=path.json:window' 形式で指定してください"
            ) from exc
        specs.append(
            AnnualSpec(
                label=label,
                json_path=(repo_root / json_path_str).resolve(),
                window=window,
            )
        )
    return specs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="IRL リプレイ評価結果を可視化してPNGとして出力します。"
    )
    ap.add_argument(
        "--annual",
        nargs="*",
        default=None,
        help="年次比較に使うデータ。'label=path:window' をスペース区切りで指定。"
    )
    ap.add_argument(
        "--quarterly",
        type=str,
        default=None,
        help="3ヶ月刻み評価に使うJSONのパス。"
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="outputs/visualizations",
        help="生成画像の出力先ディレクトリ。",
    )
    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    annual_specs = (
        _parse_annual_arg(args.annual, repo_root)
        if args.annual
        else _default_annual_specs(repo_root)
    )
    quarterly_spec = (
        QuarterlySpec(json_path=Path(args.quarterly))
        if args.quarterly
        else _default_quarterly_spec(repo_root)
    )

    annual_labels, annual_metrics, annual_info = _collect_annual_data(annual_specs)
    quarterly_months, quarterly_metrics, quarterly_meta = _collect_quarterly_data(
        quarterly_spec
    )

    out_dir = (repo_root / args.out_dir).resolve()
    _plot_annual(
        annual_labels,
        annual_metrics,
        annual_info,
        out_dir / "replay_metrics_annual.png",
    )
    _plot_quarterly(
        quarterly_months,
        quarterly_metrics,
        quarterly_meta,
        out_dir / "replay_metrics_quarterly.png",
    )

    print(
        json.dumps(
            {
                "annual_labels": annual_labels,
                "annual_out": str(out_dir / "replay_metrics_annual.png"),
                "annual_info": annual_info,
                "quarterly_points": len(quarterly_months),
                "quarterly_meta": quarterly_meta,
                "quarterly_out": str(out_dir / "replay_metrics_quarterly.png"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
