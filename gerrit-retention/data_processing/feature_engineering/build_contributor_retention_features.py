"""長期貢献者予測向けの特徴量テーブルを生成するスクリプト。

レビューリクエストの生ログと IRL 由来のメトリクスを統合し、
スナップショット日時までの活動履歴から開発者単位の特徴量を
集計して Parquet で出力する。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/retention_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contributor retention feature table"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="設定ファイル (YAML)。指定しない場合は configs/retention_config.yaml を利用",
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
        help="特徴量を確定させるスナップショット日付 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--feature-history-months",
        type=int,
        default=None,
        help="過去何ヶ月分の活動を特徴量に含めるか (指定しない場合は設定ファイル値を利用)",
    )
    parser.add_argument(
        "--target-window-months",
        type=int,
        default=None,
        help="予測対象期間 (設定ファイルを上書き)",
    )
    parser.add_argument(
        "--irl-replay",
        type=Path,
        default=None,
        help="オプション: IRL リプレイ評価結果 (JSON/JSONL/CSV) へのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="特徴量を出力するディレクトリ (Parquet 形式)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存ファイルがあっても上書きする",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを有効化",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが存在しません: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config or {}


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
    column_cfg = retention_cfg.get("columns", {})
    params["columns"] = {
        "reviewer": column_cfg.get("reviewer", "reviewer_email"),
        "created": column_cfg.get("created", "created_at"),
        "project": column_cfg.get("project", "project"),
        "status": column_cfg.get("status", "status"),
        "approval_flag": column_cfg.get("approval_flag", "is_approval"),
        "comment_count": column_cfg.get("comment_count", "comment_count"),
    }
    irl_cfg = retention_cfg.get("irl_metrics", {})
    params["irl_columns"] = {
        "reviewer": irl_cfg.get("reviewer", "reviewer_email"),
        "score": irl_cfg.get("score", "score"),
        "coverage": irl_cfg.get("positive_coverage", "positive_coverage"),
    }
    return params


def load_review_requests(path: Path, columns: Dict[str, str]) -> pd.DataFrame:
    LOGGER.info("レビューリクエストを読み込み中: %s", path)
    if not path.exists():
        raise FileNotFoundError(f"レビューリクエストファイルが存在しません: {path}")

    df = pd.read_csv(path)
    for required in (columns["reviewer"], columns["created"]):
        if required not in df.columns:
            raise KeyError(f"必要な列が見つかりません: {required}")

    # 日付変換
    df[columns["created"]] = pd.to_datetime(
        df[columns["created"]], errors="coerce", utc=True
    ).dt.tz_localize(None)
    df = df.dropna(subset=[columns["created"], columns["reviewer"]])
    return df


def filter_history_window(
    df: pd.DataFrame,
    columns: Dict[str, str],
    snapshot_date: pd.Timestamp,
    history_months: int,
) -> pd.DataFrame:
    end_date = snapshot_date.normalize() + pd.DateOffset(days=1) - pd.Timedelta(seconds=1)
    start_date = snapshot_date.normalize() - pd.DateOffset(months=history_months) + pd.Timedelta(seconds=1)
    LOGGER.info(
        "対象期間: %s 〜 %s (%s ヶ月)", start_date.date(), end_date.date(), history_months
    )
    mask = df[columns["created"]].between(start_date, end_date, inclusive="both")
    return df.loc[mask].copy()


def aggregate_activity_features(
    df: pd.DataFrame,
    columns: Dict[str, str],
    snapshot_date: pd.Timestamp,
    history_months: int,
) -> pd.DataFrame:
    reviewer_col = columns["reviewer"]
    created_col = columns["created"]

    grouped = df.groupby(reviewer_col, dropna=False)

    features = pd.DataFrame(index=grouped.size().index)
    features["total_reviews"] = grouped.size()

    approval_col = columns.get("approval_flag")
    if approval_col and approval_col in df.columns:
        features["total_approvals"] = grouped[approval_col].sum(min_count=1).fillna(0).astype(int)

    comment_col = columns.get("comment_count")
    if comment_col and comment_col in df.columns:
        features["total_comments"] = grouped[comment_col].sum(min_count=1).fillna(0)

    project_col = columns.get("project")
    if project_col and project_col in df.columns:
        features["unique_projects"] = grouped[project_col].nunique(dropna=True)

    # 月別活動数からアクティブ月数を計算
    df["_activity_month"] = df[created_col].dt.to_period("M")
    month_counts = df.groupby([reviewer_col, "_activity_month"]).size()
    features["active_months"] = month_counts.groupby(level=0).size()

    features["reviews_per_month"] = (
        features["total_reviews"] / features["active_months"].clip(lower=1)
    )

    # 最新活動日と経過日数
    last_activity = grouped[created_col].max()
    features["last_activity_date"] = last_activity
    features["days_since_last_activity"] = (
        snapshot_date.normalize() - last_activity
    ).dt.days

    features["activity_density"] = (
        features["total_reviews"] / history_months
    )

    features = features.reset_index().rename(columns={reviewer_col: "developer_id"})
    return features


def load_irl_metrics(path: Path, irl_columns: Dict[str, str]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        LOGGER.warning("IRLメトリクスが見つかりません: %s", path)
        return pd.DataFrame()

    LOGGER.info("IRLメトリクスを読み込み中: %s", path)
    try:
        if path.suffix.lower() in {".json", ".jsonl"}:
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_csv(path)
    except ValueError as exc:
        LOGGER.warning("IRLメトリクスの読み込みに失敗しました: %s", exc)
        return pd.DataFrame()

    reviewer_col = irl_columns.get("reviewer", "reviewer_email")
    if reviewer_col not in df.columns:
        LOGGER.warning("IRLメトリクスにレビュア列が見当たりません: %s", reviewer_col)
        return pd.DataFrame()

    rename_map = {reviewer_col: "developer_id"}
    score_col = irl_columns.get("score")
    if score_col and score_col in df.columns:
        rename_map[score_col] = "irl_mean_score"
    coverage_col = irl_columns.get("coverage")
    if coverage_col and coverage_col in df.columns:
        rename_map[coverage_col] = "irl_positive_coverage"

    metrics = df.rename(columns=rename_map)[list(rename_map.values())]
    metrics = metrics.groupby("developer_id", as_index=False).mean()
    return metrics


def save_features(df: pd.DataFrame, output_dir: Path, snapshot_date: pd.Timestamp, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"contributor_features_{snapshot_date.strftime('%Y%m%d')}.parquet"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"既にファイルが存在します: {output_path} -- --overwrite を指定してください")
    df.to_parquet(output_path, index=False)
    LOGGER.info("特徴量を保存しました: %s (件数=%d)", output_path, len(df))
    return output_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = load_config(args.config)
    params = resolve_parameters(config, args)

    snapshot_date = pd.to_datetime(args.snapshot_date, utc=False)
    review_df = load_review_requests(args.review_requests, params["columns"])
    scoped_df = filter_history_window(
        review_df,
        params["columns"],
        snapshot_date,
        params["feature_history_months"],
    )
    if scoped_df.empty:
        LOGGER.warning("対象期間内のレビュー記録が見つかりませんでした")

    activity_features = aggregate_activity_features(
        scoped_df,
        params["columns"],
        snapshot_date,
        params["feature_history_months"],
    )

    irl_features = load_irl_metrics(args.irl_replay, params["irl_columns"])
    if not irl_features.empty:
        activity_features = activity_features.merge(
            irl_features, on="developer_id", how="left"
        )

    output_path = save_features(
        activity_features,
        args.output_dir,
        snapshot_date,
        args.overwrite,
    )

    summary = {
        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "records": int(len(activity_features)),
        "history_months": params["feature_history_months"],
        "target_window_months": params["target_window_months"],
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
