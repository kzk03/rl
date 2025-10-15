"""長期貢献者ベースラインのバッチ実行パイプライン。

以下のステップを連続実行する:
1. 特徴量生成スクリプトの起動
2. ラベル生成スクリプトの起動
3. ベースライン分類モデル(ロジスティック回帰)の学習
4. 任意で学習済みモデルの評価

各ステップは既存成果物がある場合にスキップすることも可能。
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger(__name__)

DEFAULT_FEATURE_DIR = Path("outputs/retention/features")
DEFAULT_LABEL_DIR = Path("outputs/retention/labels")
DEFAULT_MODEL_DIR = Path("outputs/retention/models")
DEFAULT_EVAL_DIR = Path("outputs/retention/eval")


@dataclass
class SnapshotArtifacts:
    snapshot_date: str
    stamp: str
    feature_path: Path
    label_path: Path
    feature_provided: bool = False
    label_provided: bool = False


@dataclass
class PipelinePaths:
    train: SnapshotArtifacts
    eval: SnapshotArtifacts
    model_dir: Path
    eval_output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline retention pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/retention_config.yaml"))
    parser.add_argument("--review-requests", type=Path, required=True)
    parser.add_argument(
        "--train-snapshot-date",
        "--snapshot-date",
        dest="train_snapshot_date",
        type=str,
        required=True,
        help="学習用のスナップショット日付 (YYYY-MM-DD)",
    )
    parser.add_argument("--feature-history-months", type=int, default=None)
    parser.add_argument("--target-window-months", type=int, default=None)
    parser.add_argument("--activity-after-months", type=int, default=None)
    parser.add_argument("--label-column", type=str, default=None)
    parser.add_argument("--irl-replay", type=Path, default=None)
    parser.add_argument("--features-output-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--labels-output-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument(
        "--model-output-dir",
        "--output-dir",
        dest="model_output_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="モデル成果物を保存するディレクトリ (旧仕様互換として --output-dir も利用可)",
    )
    parser.add_argument("--evaluation-output-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--eval-features", type=Path, default=None, help="評価用特徴量を個別指定する場合")
    parser.add_argument("--eval-labels", type=Path, default=None, help="評価用ラベルを個別指定する場合")
    parser.add_argument("--eval-snapshot-date", type=str, default=None, help="評価用データのスナップショット (YYYY-MM-DD)")
    parser.add_argument("--skip-features", action="store_true", help="特徴量生成をスキップ")
    parser.add_argument("--skip-labels", action="store_true", help="ラベル生成をスキップ")
    parser.add_argument("--skip-training", action="store_true", help="モデル学習をスキップ")
    parser.add_argument("--skip-evaluation", action="store_true", help="モデル評価をスキップ")
    parser.add_argument("--overwrite", action="store_true", help="既存成果物を上書き")
    parser.add_argument("--verbose", action="store_true", help="詳細ログを表示")
    return parser.parse_args()


def to_snapshot_stamp(snapshot_date: str) -> str:
    try:
        dt = datetime.strptime(snapshot_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"snapshot-date は YYYY-MM-DD 形式で指定してください: {snapshot_date}") from exc
    return dt.strftime("%Y%m%d")


def build_paths(args: argparse.Namespace) -> PipelinePaths:
    train_snapshot = args.train_snapshot_date
    train_stamp = to_snapshot_stamp(train_snapshot)
    train_feature_path = args.features_output_dir / f"contributor_features_{train_stamp}.parquet"
    train_label_path = args.labels_output_dir / f"contributor_labels_{train_stamp}.parquet"
    train_artifacts = SnapshotArtifacts(
        snapshot_date=train_snapshot,
        stamp=train_stamp,
        feature_path=train_feature_path,
        label_path=train_label_path,
    )

    eval_snapshot = args.eval_snapshot_date or train_snapshot
    eval_stamp = to_snapshot_stamp(eval_snapshot)

    if args.eval_features:
        eval_feature_path = args.eval_features
        eval_feature_provided = True
    else:
        eval_feature_path = args.features_output_dir / f"contributor_features_{eval_stamp}.parquet"
        eval_feature_provided = False

    if args.eval_labels:
        eval_label_path = args.eval_labels
        eval_label_provided = True
    else:
        eval_label_path = args.labels_output_dir / f"contributor_labels_{eval_stamp}.parquet"
        eval_label_provided = False

    eval_artifacts = SnapshotArtifacts(
        snapshot_date=eval_snapshot,
        stamp=eval_stamp,
        feature_path=eval_feature_path,
        label_path=eval_label_path,
        feature_provided=eval_feature_provided,
        label_provided=eval_label_provided,
    )

    model_dir = args.model_output_dir / train_stamp
    eval_output_path = args.evaluation_output_dir / f"baseline_eval_{eval_stamp}.json"

    return PipelinePaths(
        train=train_artifacts,
        eval=eval_artifacts,
        model_dir=model_dir,
        eval_output_path=eval_output_path,
    )


def run_step(command: List[str], description: str) -> None:
    LOGGER.info("実行開始: %s", description)
    LOGGER.debug("コマンド: %s", " ".join(command))
    subprocess.run(command, check=True)
    LOGGER.info("完了: %s", description)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_features(
    args: argparse.Namespace,
    snapshot: SnapshotArtifacts,
    skip: bool,
    label: str,
) -> None:
    if skip:
        LOGGER.info("特徴量生成をスキップ (%s, --skip-features)", label)
        return
    if snapshot.feature_provided:
        LOGGER.info("特徴量生成(%s)は指定済みファイルを利用: %s", label, snapshot.feature_path)
        return

    command = [
        sys.executable,
        "data_processing/feature_engineering/build_contributor_retention_features.py",
        "--review-requests",
        str(args.review_requests),
        "--snapshot-date",
        snapshot.snapshot_date,
        "--output-dir",
        str(snapshot.feature_path.parent),
        "--config",
        str(args.config),
    ]

    if args.feature_history_months is not None:
        command.extend(["--feature-history-months", str(args.feature_history_months)])
    if args.target_window_months is not None:
        command.extend(["--target-window-months", str(args.target_window_months)])
    if args.irl_replay is not None:
        command.extend(["--irl-replay", str(args.irl_replay)])
    if args.overwrite:
        command.append("--overwrite")
    if args.verbose:
        command.append("--verbose")

    run_step(command, f"特徴量生成 ({label})")


def build_labels(
    args: argparse.Namespace,
    snapshot: SnapshotArtifacts,
    feature_path: Path,
    skip: bool,
    label: str,
) -> None:
    if skip:
        LOGGER.info("ラベル生成をスキップ (%s, --skip-labels)", label)
        return
    if snapshot.label_provided:
        LOGGER.info("ラベル生成(%s)は指定済みファイルを利用: %s", label, snapshot.label_path)
        return
    if not feature_path.exists():
        raise FileNotFoundError(
            f"ラベル生成({label})に必要な特徴量が見つかりません: {feature_path}"
        )

    command = [
        sys.executable,
        "scripts/offline/build_contributor_retention_labels.py",
        "--review-requests",
        str(args.review_requests),
        "--snapshot-date",
        snapshot.snapshot_date,
        "--output-dir",
        str(snapshot.label_path.parent),
        "--config",
        str(args.config),
        "--developer-list",
        str(feature_path),
    ]

    if args.target_window_months is not None:
        command.extend(["--target-window-months", str(args.target_window_months)])
    if args.feature_history_months is not None:
        command.extend(["--feature-history-months", str(args.feature_history_months)])
    if args.activity_after_months is not None:
        command.extend(["--activity-after-months", str(args.activity_after_months)])
    if args.overwrite:
        command.append("--overwrite")
    if args.verbose:
        command.append("--verbose")

    run_step(command, f"ラベル生成 ({label})")


def train_model(args: argparse.Namespace, paths: PipelinePaths) -> Path:
    if args.skip_training:
        LOGGER.info("モデル学習をスキップ (--skip-training)")
        return paths.model_dir / "baseline_model.joblib"

    ensure_parent(paths.model_dir / "dummy")
    command = [
        sys.executable,
        "scripts/training/retention/train_retention_baseline.py",
        "--features",
        str(paths.train.feature_path),
        "--labels",
        str(paths.train.label_path),
        "--output-dir",
        str(paths.model_dir),
    ]

    if args.overwrite:
        command.append("--overwrite")
    if args.verbose:
        command.append("--verbose")
    if args.label_column:
        command.extend(["--label-column", args.label_column])

    if args.feature_history_months is not None:
        LOGGER.debug("feature_history_months は学習へ直接渡されません (特徴量生成側で利用)")

    run_step(command, "モデル学習")
    return paths.model_dir / "baseline_model.joblib"


def evaluate_model(args: argparse.Namespace, paths: PipelinePaths, model_path: Path) -> Optional[dict]:
    if args.skip_evaluation:
        LOGGER.info("モデル評価をスキップ (--skip-evaluation)")
        return None

    eval_features = paths.eval.feature_path
    eval_labels = paths.eval.label_path

    if not eval_features.exists():
        raise FileNotFoundError(f"評価用特徴量が見つかりません: {eval_features}")
    if not eval_labels.exists():
        raise FileNotFoundError(f"評価用ラベルが見つかりません: {eval_labels}")
    ensure_parent(paths.eval_output_path)

    command = [
        sys.executable,
        "scripts/evaluation/evaluate_retention_model.py",
        "--model",
        str(model_path),
        "--features",
        str(eval_features),
        "--labels",
        str(eval_labels),
        "--output",
        str(paths.eval_output_path),
    ]

    if args.verbose:
        command.append("--verbose")
    if args.label_column:
        command.extend(["--label-column", args.label_column])

    run_step(command, "モデル評価")
    with paths.eval_output_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    LOGGER.info("評価結果: %s", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    paths = build_paths(args)
    LOGGER.debug("実行パス: %s", paths)

    build_features(args, paths.train, args.skip_features, "学習")

    if (
        not paths.eval.feature_provided
        and paths.eval.feature_path != paths.train.feature_path
    ):
        build_features(args, paths.eval, args.skip_features, "検証")
    elif paths.eval.feature_path == paths.train.feature_path:
        LOGGER.debug("検証用特徴量は学習用と同一ファイルを利用")

    build_labels(args, paths.train, paths.train.feature_path, args.skip_labels, "学習")

    if (
        not paths.eval.label_provided
        and paths.eval.label_path != paths.train.label_path
    ):
        build_labels(args, paths.eval, paths.eval.feature_path, args.skip_labels, "検証")
    elif paths.eval.label_path == paths.train.label_path:
        LOGGER.debug("検証用ラベルは学習用と同一ファイルを利用")

    model_path = train_model(args, paths)
    metrics = evaluate_model(args, paths, model_path)

    summary = {
        "train_feature_path": str(paths.train.feature_path),
        "train_label_path": str(paths.train.label_path),
        "model_dir": str(paths.model_dir),
        "eval_feature_path": str(paths.eval.feature_path),
        "eval_label_path": str(paths.eval.label_path),
        "evaluation_path": str(paths.eval_output_path),
        "metrics": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
