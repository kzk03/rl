"""長期貢献者予測用のラベルを生成するスクリプト。

指定したスナップショット日時から target_window_months 先の活動を
確認し、開発者が長期貢献者に該当するかを判定する。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/retention_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contributor retention labels"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="設定ファイル (YAML)。省略時は configs/retention_config.yaml を利用",
    )
    parser.add_argument(
        "--review-requests",
        type=Path,
        required=True,
        help="レビューリクエストの CSV ファイルパス",
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        required=True,
        help="スナップショット日付 (YYYY-MM-DD)。この翌日以降の活動でラベルを作成",
    )
    parser.add_argument(
        "--target-window-months",
        type=int,
        default=None,
        help="予測対象期間 (設定ファイルを上書き)",
    )
    parser.add_argument(
        "--feature-history-months",
        type=int,
        default=None,
        help="既存特徴量との整合用に履歴期間を併記 (設定ファイルを上書き)",
    )
    parser.add_argument(
        "--activity-after-months",
        type=int,
        default=None,
        help="スナップショット後 n ヶ月以降に活動があるかでラベルを生成する際の n を上書き",
    )
    parser.add_argument(
        "--developer-list",
        type=Path,
        default=None,
        help="対象開発者IDを含むファイル (Parquet/CSV)。省略時は将来期間に登場する開発者のみ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="ラベルを出力するディレクトリ (Parquet)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存ファイルを上書きする",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを表示",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが存在しません: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def resolve_parameters(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    retention_cfg = config.get("retention_prediction", {})
    params = {
        "target_window_months": args.target_window_months
        or retention_cfg.get("target_window_months", 12),
        "feature_history_months": args.feature_history_months
        or retention_cfg.get("feature_history_months", 18),
        "min_reviews": retention_cfg.get("min_reviews", 1),
        "active_ratio": retention_cfg.get("active_ratio", 0.75),
        "grace_period_days": retention_cfg.get("grace_period_days", 60),
    }
    params["activity_after_months"] = args.activity_after_months or retention_cfg.get(
        "activity_after_months",
        params["target_window_months"],
    )
    column_cfg = retention_cfg.get("columns", {})
    params["columns"] = {
        "reviewer": column_cfg.get("reviewer", "reviewer_email"),
        "created": column_cfg.get("created", "created_at"),
        "project": column_cfg.get("project", "project"),
    }
    return params


def load_review_requests(path: Path, columns: Dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"レビューリクエストが存在しません: {path}")
    LOGGER.info("レビューリクエストを読み込み中: %s", path)
    df = pd.read_csv(path)
    created_col = columns["created"]
    reviewer_col = columns["reviewer"]
    missing = [col for col in (created_col, reviewer_col) if col not in df.columns]
    if missing:
        raise KeyError(f"必要な列が見つかりません: {missing}")
    df[created_col] = pd.to_datetime(df[created_col], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=[created_col, reviewer_col])
    return df


def load_developer_list(path: Optional[Path]) -> Optional[pd.Series]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"developer_list ファイルが存在しません: {path}")
    LOGGER.info("対象開発者リストを読み込み中: %s", path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "developer_id" not in df.columns:
        raise KeyError("developer_list には developer_id 列が必要です")
    return df["developer_id"].dropna().astype(str)


def compute_label_window(snapshot_date: pd.Timestamp, target_months: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = snapshot_date.normalize() + pd.Timedelta(seconds=1)
    end = start + pd.DateOffset(months=target_months) - pd.Timedelta(seconds=1)
    return start, end


def aggregate_future_activity(
    df: pd.DataFrame,
    columns: Dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_reviews: int,
    activity_after_months: int,
) -> Dict[str, pd.Series]:
    created_col = columns["created"]
    reviewer_col = columns["reviewer"]
    mask = df[created_col].between(start, end, inclusive="both")
    future_df = df.loc[mask].copy()
    future_df["_month"] = future_df[created_col].dt.to_period("M")

    monthly_counts = future_df.groupby([reviewer_col, "_month"]).size()
    observed_months = monthly_counts.groupby(level=0).size()
    active_months = (
        monthly_counts[monthly_counts >= min_reviews].groupby(level=0).size()
    )
    total_reviews = future_df.groupby(reviewer_col).size()
    last_activity = future_df.groupby(reviewer_col)[created_col].max()

    if activity_after_months is not None and activity_after_months >= 0:
        post_start = start + pd.DateOffset(months=activity_after_months)
        post_mask = df[created_col] >= post_start
        post_activity_df = df.loc[post_mask]
        post_activity_counts = post_activity_df.groupby(reviewer_col).size()
        post_first_activity = post_activity_df.groupby(reviewer_col)[created_col].min()
    else:
        post_activity_counts = pd.Series(dtype=int)
        post_first_activity = pd.Series(dtype="datetime64[ns]")

    return {
        "observed_months": observed_months,
        "active_months": active_months,
        "total_reviews": total_reviews,
        "last_activity": last_activity,
        "post_activity_counts": post_activity_counts,
        "post_first_activity": post_first_activity,
    }


def build_labels(
    developer_ids: Iterable[str],
    aggregates: Dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    params: Dict[str, Any],
) -> pd.DataFrame:
    target_months = params["target_window_months"]
    active_ratio_threshold = params["active_ratio"]
    grace_period_days = params["grace_period_days"]
    activity_after_months = params.get("activity_after_months")

    index = pd.Index(sorted(set(str(dev) for dev in developer_ids)), name="developer_id")

    def reindex(series: pd.Series, fill_value=0) -> pd.Series:
        if series is None or series.empty:
            return pd.Series(fill_value, index=index)
        return series.reindex(index, fill_value=fill_value)

    observed_months = reindex(aggregates["observed_months"])
    active_months = reindex(aggregates["active_months"])
    total_reviews = reindex(aggregates["total_reviews"])
    last_activity = aggregates["last_activity"].reindex(index) if not aggregates["last_activity"].empty else pd.Series(pd.NaT, index=index)

    labels = pd.DataFrame(index=index)
    labels["target_window_months"] = target_months
    labels["observed_months"] = observed_months.astype(int)
    labels["active_months"] = active_months.astype(int)
    labels["total_reviews"] = total_reviews.astype(int)

    window_months = target_months if target_months > 0 else 1
    labels["active_ratio"] = labels["active_months"] / window_months
    labels["observed_ratio"] = labels["active_months"] / labels["observed_months"].clip(lower=1)
    labels["meets_observation"] = labels["observed_months"] >= target_months
    labels["is_long_term"] = (
        (labels["active_ratio"] >= active_ratio_threshold)
        & labels["meets_observation"]
    ).astype(int)

    labels["label_period_start"] = start
    labels["label_period_end"] = end
    labels["last_activity_in_window"] = last_activity

    post_counts = aggregates.get("post_activity_counts")
    post_first_activity = aggregates.get("post_first_activity")
    post_counts = reindex(post_counts, fill_value=0).astype(int)
    if post_first_activity is None or post_first_activity.empty:
        post_first_activity = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    else:
        post_first_activity = post_first_activity.reindex(index)

    labels["activity_after_months"] = activity_after_months
    labels["first_activity_after_threshold"] = post_first_activity
    labels["is_active_after_threshold"] = (post_counts > 0).astype(int)

    window_days = (end - start).days
    labels["event_time_days"] = window_days
    labels["event_observed"] = 0

    inactive_mask = labels["is_long_term"] == 0
    last_activity_with_grace = (
        last_activity + pd.to_timedelta(grace_period_days, unit="D")
    )
    event_time = (last_activity_with_grace - start).dt.days.clip(lower=0)
    event_time = event_time.fillna(0).clip(upper=window_days)
    labels.loc[inactive_mask, "event_time_days"] = event_time[inactive_mask]
    labels.loc[inactive_mask, "event_observed"] = 1

    no_observation = labels["observed_months"] == 0
    labels.loc[no_observation, "event_time_days"] = 0
    labels.loc[no_observation, "event_observed"] = 1

    labels = labels.reset_index()
    return labels


def save_labels(df: pd.DataFrame, output_dir: Path, snapshot_date: pd.Timestamp, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"contributor_labels_{snapshot_date.strftime('%Y%m%d')}.parquet"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"既にファイルが存在します: {output_path} -- --overwrite を指定してください")
    df.to_parquet(output_path, index=False)
    LOGGER.info("ラベルを保存しました: %s (件数=%d)", output_path, len(df))
    return output_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = load_config(args.config)
    params = resolve_parameters(config, args)

    snapshot_date = pd.to_datetime(args.snapshot_date, utc=False)
    future_start, future_end = compute_label_window(snapshot_date, params["target_window_months"])

    review_df = load_review_requests(args.review_requests, params["columns"])
    developer_list = load_developer_list(args.developer_list)
    if developer_list is not None:
        base_ids = developer_list.tolist()
    else:
        base_ids = review_df[params["columns"]["reviewer"]].dropna().astype(str).unique()

    aggregates = aggregate_future_activity(
        review_df,
        params["columns"],
        future_start,
        future_end,
        params["min_reviews"],
        params.get("activity_after_months"),
    )

    labels = build_labels(
        base_ids,
        aggregates,
        future_start,
        future_end,
        params,
    )

    output_path = save_labels(labels, args.output_dir, snapshot_date, args.overwrite)
    summary = {
        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "records": int(len(labels)),
        "target_window_months": params["target_window_months"],
        "activity_after_months": params.get("activity_after_months"),
        "min_reviews": params["min_reviews"],
        "active_ratio": params["active_ratio"],
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
