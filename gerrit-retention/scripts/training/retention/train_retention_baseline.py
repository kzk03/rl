"""長期貢献者予測のベースライン分類モデルを学習するスクリプト。

特徴量テーブルとラベルテーブルを読み込み、開発者単位の
データセットを構築して scikit-learn のモデルで学習する。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import check_random_state

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline classifier for long-term contributor prediction"
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="特徴量 Parquet ファイルへのパス",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="ラベル Parquet ファイルへのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="モデルと評価結果を保存するディレクトリ",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="検証用に確保する割合 (0-1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="乱数シード",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の成果物を上書き",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="is_long_term",
        help="学習に利用するラベル列名 (デフォルト: is_long_term)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを表示",
    )
    return parser.parse_args()


def load_tables(feature_path: Path, label_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not feature_path.exists():
        raise FileNotFoundError(f"特徴量ファイルが存在しません: {feature_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"ラベルファイルが存在しません: {label_path}")

    LOGGER.info("特徴量を読み込み中: %s", feature_path)
    features = pd.read_parquet(feature_path)
    LOGGER.info("ラベルを読み込み中: %s", label_path)
    labels = pd.read_parquet(label_path)
    return features, labels


def prepare_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    # developer_idがない場合、インデックスで結合を想定
    if "developer_id" in features.columns and "developer_id" in labels.columns:
        merged = labels.merge(features, on="developer_id", how="left", suffixes=("_label", ""))
    else:
        # developer_idがない場合、行数を確認してconcat
        if len(features) != len(labels):
            raise ValueError(f"特徴量とラベルの行数が一致しません: {len(features)} vs {len(labels)}")
        merged = pd.concat([features, labels], axis=1)
    
    if label_column not in merged.columns:
        raise KeyError(f"ラベル列 {label_column} が見つかりません")

    merged = merged.dropna(subset=[label_column])  # ラベル欠損は除外

    y = merged[label_column].astype(int)

    # 訓練に不要な列を削除
    drop_columns = {
        "developer_id",
        "label_period_start",
        "label_period_end",
        "last_activity_in_window",
    }
    label_meta_columns = {
        col for col in merged.columns if col.endswith("_label") and col != label_column
    }
    drop_candidates = drop_columns.union(label_meta_columns)
    existing_drop = [col for col in drop_candidates if col in merged.columns]
    label_drop = [label_column] if label_column in merged.columns else []
    X = merged.drop(columns=existing_drop + label_drop)

    datetime_cols = [
        col for col in X.columns if np.issubdtype(X[col].dtype, np.datetime64)
    ]
    if datetime_cols:
        LOGGER.debug("日時型の列を除外します: %s", datetime_cols)
        X = X.drop(columns=datetime_cols)

    label_leak_cols = [col for col in X.columns if col.endswith("_label")]
    if label_leak_cols:
        LOGGER.debug("ラベル由来の列を除外します: %s", label_leak_cols)
        X = X.drop(columns=label_leak_cols)

    return X, y


def build_pipeline(categorical: List[str], numeric: List[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(transformers)
    clf = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", clf),
        ]
    )
    return pipeline


def split_features(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical = [col for col in X.columns if X[col].dtype == "object"]
    numeric = [col for col in X.columns if col not in categorical]
    return categorical, numeric


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[Pipeline, dict]:
    rng = check_random_state(random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    categorical, numeric = split_features(X)
    pipeline = build_pipeline(categorical, numeric)

    LOGGER.info("学習データ: %s 行, 特徴量: %s", X_train.shape[0], X_train.shape[1])
    pipeline.fit(X_train, y_train)

    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_pred_prob)),
        "average_precision": float(average_precision_score(y_test, y_pred_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }
    return pipeline, metrics


def save_artifacts(
    model: Pipeline,
    metrics: dict,
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "baseline_model.joblib"
    metrics_path = output_dir / "baseline_metrics.json"

    if not overwrite:
        for path in (model_path, metrics_path):
            if path.exists():
                raise FileExistsError(f"既にファイルが存在します: {path} -- --overwrite を指定してください")

    joblib.dump(model, model_path)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    LOGGER.info("モデルを保存しました: %s", model_path)
    LOGGER.info("評価指標を保存しました: %s", metrics_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    features, labels = load_tables(args.features, args.labels)
    X, y = prepare_dataset(features, labels, args.label_column)

    if len(X) == 0:
        raise ValueError("学習対象のデータが存在しません")

    model, metrics = train_and_evaluate(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    save_artifacts(model, metrics, args.output_dir, args.overwrite)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
