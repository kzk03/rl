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


def _has_values(values: Sequence[float | None]) -> bool:
    return any(v is not None for v in values)


def _append_metric(store: Dict[str, List[float | None]], key: str, value) -> None:
    lst = store.setdefault(key, [])
    if value is None:
        lst.append(None)
    else:
        try:
            lst.append(float(value))
        except (TypeError, ValueError):
            lst.append(None)


def _collect_annual_data(
    specs: Sequence[AnnualSpec],
) -> Tuple[List[str], Dict[str, List[float | None]], List[str]]:
    metrics: Dict[str, List[float | None]] = {
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
        _append_metric(metrics, "top1", entry.get("action_match_rate"))
        _append_metric(metrics, "top3", entry.get("top3_hit_rate"))
        _append_metric(metrics, "top5", entry.get("top5_hit_rate"))
        _append_metric(metrics, "mAP", entry.get("mAP"))
        _append_metric(metrics, "ECE", entry.get("ECE"))
        _append_metric(metrics, "avg_candidates", entry.get("avg_candidates"))
        _append_metric(metrics, "precision_at_1", entry.get("precision_at_1"))
        _append_metric(metrics, "precision_at_3", entry.get("precision_at_3"))
        _append_metric(metrics, "precision_at_5", entry.get("precision_at_5"))
        _append_metric(metrics, "recall_at_1", entry.get("recall_at_1"))
        _append_metric(metrics, "recall_at_3", entry.get("recall_at_3"))
        _append_metric(metrics, "recall_at_5", entry.get("recall_at_5"))
        _append_metric(metrics, "positive_coverage", entry.get("positive_coverage"))
        info_lines.append(
            f"{spec.label}: cutoff={cutoff}, window={spec.window}, model_dir={model_dir}"
        )
    return labels, metrics, info_lines


def _parse_window_start(window: str, pattern: str) -> int:
    m = re.match(pattern, window)
    if not m:
        raise ValueError(f"ウィンドウ '{window}' がパターン '{pattern}' に一致しません。")
    return int(m.group("start"))


def _collect_quarterly_data(spec: QuarterlySpec) -> Tuple[List[int], Dict[str, List[float | None]], str]:
    data = _load_json(spec.json_path)
    results = data.get("results", {})
    filtered = {
        window: entry
        for window, entry in results.items()
        if re.match(spec.pattern, window)
    }
    use_simple_month = False
    if not filtered:
        simple_pattern = r"^(?P<start>\d+)m$"
        fallback = {
            window: entry
            for window, entry in results.items()
            if re.match(simple_pattern, window)
        }
        if fallback:
            filtered = fallback
            use_simple_month = True
    if not filtered:
        raise KeyError(
            f"指定パターン '{spec.pattern}' に一致するウィンドウが {spec.json_path} に存在しません。"
        )

    def _parse_window(window: str) -> int:
        if use_simple_month:
            return int(window.rstrip("m"))
        return _parse_window_start(window, spec.pattern)

    windows_sorted = sorted(filtered.keys(), key=_parse_window)
    months: List[int] = []
    metrics: Dict[str, List[float | None]] = {
        "top1": [],
        "top3": [],
        "top5": [],
        "mAP": [],
        "ECE": [],
    }
    for window in windows_sorted:
        entry = filtered[window]
        months.append(_parse_window(window))
        _append_metric(metrics, "top1", entry.get("action_match_rate"))
        _append_metric(metrics, "top3", entry.get("top3_hit_rate"))
        _append_metric(metrics, "top5", entry.get("top5_hit_rate"))
        _append_metric(metrics, "mAP", entry.get("mAP"))
        _append_metric(metrics, "ECE", entry.get("ECE"))
        _append_metric(metrics, "precision_at_1", entry.get("precision_at_1"))
        _append_metric(metrics, "precision_at_3", entry.get("precision_at_3"))
        _append_metric(metrics, "precision_at_5", entry.get("precision_at_5"))
        _append_metric(metrics, "recall_at_1", entry.get("recall_at_1"))
        _append_metric(metrics, "recall_at_3", entry.get("recall_at_3"))
        _append_metric(metrics, "recall_at_5", entry.get("recall_at_5"))
        _append_metric(metrics, "positive_coverage", entry.get("positive_coverage"))
    cutoff = data.get("cutoff")
    meta = (
        f"cutoff={cutoff}, windows matching '{spec.pattern}', model_dir={spec.json_path.parent.name}"
    )
    return months, metrics, meta


def _plot_metric_series(ax, x, y_values, label, **plot_kwargs):
    xs = [val for val, score in zip(x, y_values) if score is not None]
    ys = [score for score in y_values if score is not None]
    if not xs:
        return
    ax.plot(xs, ys, marker="o", label=label, **plot_kwargs)


def _plot_annual(
    labels: Sequence[str],
    metrics: Dict[str, List[float | None]],
    info_lines: Sequence[str],
    out_path: Path,
) -> None:
    multilabel_keys = (
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "positive_coverage",
    )
    has_multilabel = any(_has_values(metrics.get(key, [])) for key in multilabel_keys)
    n_rows = 3 if has_multilabel else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 10 if has_multilabel else 8), sharex=True)
    if n_rows == 1:
        axes = [axes]

    axes[0].plot(labels, [v or 0.0 for v in metrics["top1"]], marker="o", label="Top-1")
    axes[0].plot(labels, [v or 0.0 for v in metrics["top3"]], marker="o", label="Top-3")
    axes[0].plot(labels, [v or 0.0 for v in metrics["top5"]], marker="o", label="Top-5")
    axes[0].set_ylabel("Hit Rate")
    axes[0].set_title("IRL Replay Evaluation (Annual)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    _plot_metric_series(axes[1], labels, metrics["mAP"], "mAP", color="#2ca02c")
    _plot_metric_series(axes[1], labels, metrics["ECE"], "ECE", color="#d62728")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Evaluation Horizon")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    if has_multilabel:
        ax_extra = axes[2]
        _plot_metric_series(ax_extra, labels, metrics.get("precision_at_1", []), "Precision@1")
        _plot_metric_series(ax_extra, labels, metrics.get("precision_at_3", []), "Precision@3")
        _plot_metric_series(ax_extra, labels, metrics.get("precision_at_5", []), "Precision@5")
        _plot_metric_series(ax_extra, labels, metrics.get("recall_at_1", []), "Recall@1", linestyle="--")
        _plot_metric_series(ax_extra, labels, metrics.get("recall_at_3", []), "Recall@3", linestyle="--")
        _plot_metric_series(ax_extra, labels, metrics.get("recall_at_5", []), "Recall@5", linestyle="--")
        _plot_metric_series(ax_extra, labels, metrics.get("positive_coverage", []), "Positive Coverage", linestyle=":")
        ax_extra.set_ylabel("Multi-label Metrics")
        ax_extra.set_xlabel("Evaluation Horizon")
        ax_extra.legend(ncol=2)
        ax_extra.grid(True, linestyle="--", alpha=0.4)

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
    metrics: Dict[str, List[float | None]],
    meta_info: str,
    out_path: Path,
) -> None:
    multilabel_keys = (
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "positive_coverage",
    )
    has_multilabel = any(_has_values(metrics.get(key, [])) for key in multilabel_keys)
    n_rows = 3 if has_multilabel else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 10 if has_multilabel else 8), sharex=True)
    if n_rows == 1:
        axes = [axes]

    axes[0].plot(months, [v or 0.0 for v in metrics["top1"]], marker="o", label="Top-1")
    axes[0].plot(months, [v or 0.0 for v in metrics["top3"]], marker="o", label="Top-3")
    axes[0].plot(months, [v or 0.0 for v in metrics["top5"]], marker="o", label="Top-5")
    axes[0].set_ylabel("Hit Rate")
    axes[0].set_title("IRL Replay Evaluation (Quarterly)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    _plot_metric_series(axes[1], months, metrics["mAP"], "mAP", color="#2ca02c")
    _plot_metric_series(axes[1], months, metrics["ECE"], "ECE", color="#d62728")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Elapsed Months (window start)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    if has_multilabel:
        ax_extra = axes[2]
        _plot_metric_series(ax_extra, months, metrics.get("precision_at_1", []), "Precision@1")
        _plot_metric_series(ax_extra, months, metrics.get("precision_at_3", []), "Precision@3")
        _plot_metric_series(ax_extra, months, metrics.get("precision_at_5", []), "Precision@5")
        _plot_metric_series(ax_extra, months, metrics.get("recall_at_1", []), "Recall@1", linestyle="--")
        _plot_metric_series(ax_extra, months, metrics.get("recall_at_3", []), "Recall@3", linestyle="--")
        _plot_metric_series(ax_extra, months, metrics.get("recall_at_5", []), "Recall@5", linestyle="--")
        _plot_metric_series(ax_extra, months, metrics.get("positive_coverage", []), "Positive Coverage", linestyle=":")
        ax_extra.set_ylabel("Multi-label Metrics")
        ax_extra.set_xlabel("Elapsed Months (window start)")
        ax_extra.legend(ncol=2)
        ax_extra.grid(True, linestyle="--", alpha=0.4)

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
