"""Retention モデルの評価スクリプト。

学習済みモデルと評価用データセットを読み込み、分類指標を
算出する。将来的な拡張として回帰・生存分析指標にも対応可能。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retention baseline model")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="学習済みモデル (joblib) のパス",
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="評価用特徴量 Parquet のパス",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="評価用ラベル Parquet のパス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="評価結果 JSON を保存するパス",
    )
    parser.add_argument(
        "--irl-direct",
        action="store_true",
        help="IRL モデルを直接使い、continuation_prob で予測",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="is_long_term",
        help="評価対象のラベル列名 (デフォルト: is_long_term)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="詳細ログを表示"
    )
    return parser.parse_args()


def load_resources(model_path: Path, feature_path: Path, label_path: Path) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"特徴量ファイルが見つかりません: {feature_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"ラベルファイルが見つかりません: {label_path}")

    LOGGER.info("モデルを読み込み中: %s", model_path)
    model = joblib.load(model_path)
    features = pd.read_parquet(feature_path)
    labels = pd.read_parquet(label_path)
    return model, features, labels


def align_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    if "developer_id" not in features.columns or "developer_id" not in labels.columns:
        raise KeyError("両テーブルとも developer_id 列を含む必要があります")
    if label_column not in labels.columns:
        raise KeyError(f"ラベルに {label_column} 列が必要です")

    merged = labels.merge(features, on="developer_id", how="left", suffixes=("_label", ""))
    merged = merged.dropna(subset=[label_column])

    y_true = merged[label_column].astype(int)
    drop_columns = {
        "developer_id",
        "label_period_start",
        "label_period_end",
        "last_activity_in_window",
    }
    label_meta_columns = {
        col for col in labels.columns if col not in {"developer_id", label_column}
    }
    drop_candidates = drop_columns.union(label_meta_columns, {label_column})
    existing_drop = [col for col in drop_candidates if col in merged.columns]
    X = merged.drop(columns=existing_drop)

    label_leak_cols = [col for col in X.columns if col.endswith("_label")]
    if label_leak_cols:
        LOGGER.debug("ラベル由来の列を除外します: %s", label_leak_cols)
        X = X.drop(columns=label_leak_cols)

    datetime_cols = [
        col for col in X.columns if np.issubdtype(X[col].dtype, np.datetime64)
    ]
    if datetime_cols:
        LOGGER.debug("日時型の列を除外します: %s", datetime_cols)
        X = X.drop(columns=datetime_cols)
    return X, y_true


def compute_metrics(model, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }
    return metrics


def save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    LOGGER.info("評価結果を書き出しました: %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.irl_direct:
        # IRL direct mode
        import torch

        from gerrit_retention.rl_prediction.retention_irl_system import (
            RetentionIRLSystem,
        )
        irl = RetentionIRLSystem.load_model(str(args.model))
        features = pd.read_parquet(args.features)
        labels = pd.read_parquet(args.labels)
        merged = pd.concat([features, labels], axis=1)
        merged = merged.dropna(subset=[args.label_column])
        y_true = merged[args.label_column].astype(int)
        
        # Simple prediction: assume state from features, dummy action
        preds = []
        probs = []
        seq_len = 12  # Match training seq_len
        
        # Get numeric columns from merged DataFrame
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        
        for _, row in merged.iterrows():
            # Create state with 20 dimensions (use numeric features)
            numeric_features = row[numeric_cols].values
            if len(numeric_features) < 4:
                # Pad with zeros if not enough features
                numeric_features = np.pad(numeric_features, (0, 4 - len(numeric_features)), 'constant')
            
            # Repeat features to fill 20 dimensions
            repeated = np.tile(numeric_features[:4], 5)[:20].astype(np.float32)  # Repeat 5 times, take first 20
            single_state = torch.tensor(repeated, dtype=torch.float32)
            state = single_state.unsqueeze(0).repeat(seq_len, 1).unsqueeze(0).to(irl.device)  # (1, 12, 20)
            action = torch.zeros(1, seq_len, 3).to(irl.device)  # Dummy action sequence
            _, cont_prob = irl.network(state, action)
            pred = int(cont_prob.item() > 0.5)
            preds.append(pred)
            probs.append(cont_prob.item())
        y_pred = np.array(preds)
        y_prob = np.array(probs)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
            "pr_auc": average_precision_score(y_true, y_prob),
        }
    else:
        model, features, labels = load_resources(args.model, args.features, args.labels)
        X, y_true = align_dataset(features, labels, args.label_column)
        if len(X) == 0:
            raise ValueError("評価対象データが空です")
        metrics = compute_metrics(model, X, y_true)
    
    save_metrics(metrics, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
