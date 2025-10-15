#!/usr/bin/env python3
"""
Training Window Analysis Script

異なる学習期間 (training_window_days) でデータを生成し、
IRL vs Baselineの性能比較を行う。
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

LOGGER = logging.getLogger(__name__)


def run_data_generation(
    sequences_path: Path,
    output_base: Path,
    prediction_window_days: int,
    training_window_days: int = None,
    snapshot_date: str = None,
) -> tuple[Path, Path]:
    """データ生成を実行"""
    stamp = f"pred{prediction_window_days}d"
    if training_window_days:
        stamp += f"_train{training_window_days}d"

    features_path = output_base / f"features_{stamp}.parquet"
    labels_path = output_base / f"labels_{stamp}.parquet"

    # データ生成スクリプト実行
    cmd = [
        sys.executable,
        "scripts/create_retention_data_from_sequences.py",
        "--input", str(sequences_path),
        "--features-output", str(features_path),
        "--labels-output", str(labels_path),
        "--prediction-window-days", str(prediction_window_days),
    ]

    if training_window_days:
        cmd.extend(["--training-window-days", str(training_window_days)])
    
    if snapshot_date:
        cmd.extend(["--snapshot-date", snapshot_date])

    LOGGER.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

    if result.returncode != 0:
        LOGGER.error("Data generation failed: %s", result.stderr)
        raise RuntimeError(f"Data generation failed: {result.stderr}")

    LOGGER.info("Generated features: %s", features_path)
    LOGGER.info("Generated labels: %s", labels_path)

    return features_path, labels_path


def run_irl_feature_extraction(
    irl_model_path: Path,
    sequences_path: Path,
    features_path: Path,
    labels_path: Path,
    output_path: Path,
    training_window_days: int = None,
    snapshot_date: str = None,
) -> Path:
    """IRL特徴量抽出を実行"""
    cmd = [
        sys.executable,
        "scripts/feature_engineering/extract_irl_features.py",
        "--irl-model", str(irl_model_path),
        "--sequences", str(sequences_path),
        "--original-features", str(features_path),
        "--labels", str(labels_path),
        "--output", str(output_path),
    ]

    if training_window_days:
        cmd.extend(["--training-window-days", str(training_window_days)])

    if snapshot_date:
        cmd.extend(["--snapshot-date", snapshot_date])

    LOGGER.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

    if result.returncode != 0:
        LOGGER.error("IRL feature extraction failed: %s", result.stderr)
        raise RuntimeError(f"IRL feature extraction failed: {result.stderr}")

    LOGGER.info("Generated enhanced features: %s", output_path)
    return output_path


def run_baseline_training(
    features_path: Path,
    labels_path: Path,
    output_dir: Path,
    model_name: str,
) -> dict:
    """ベースライン訓練を実行"""
    cmd = [
        sys.executable,
        "scripts/training/retention/train_retention_baseline.py",
        "--features", str(features_path),
        "--labels", str(labels_path),
        "--output-dir", str(output_dir),
        "--label-column", "label",
        "--overwrite",
    ]

    LOGGER.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

    if result.returncode != 0:
        LOGGER.error("Baseline training failed: %s", result.stderr)
        raise RuntimeError(f"Baseline training failed: {result.stderr}")

    # メトリクス読み込み
    metrics_path = output_dir / "baseline_metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    LOGGER.info("Baseline metrics: %s", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Training Window Analysis")
    parser.add_argument(
        "--sequences",
        type=Path,
        required=True,
        help="レビュワーシーケンスJSONファイル",
    )
    parser.add_argument(
        "--irl-model",
        type=Path,
        required=True,
        help="学習済みIRLモデルパス",
    )
    parser.add_argument(
        "--prediction-windows",
        type=int,
        nargs="+",
        default=[30, 90, 180, 270, 365],
        help="予測ウィンドウの日数リスト (1m,3m,6m,9m,12m)",
    )
    parser.add_argument(
        "--training-windows",
        type=int,
        nargs="+",
        default=[30, 90, 180, 270, 365],
        help="学習ウィンドウの日数リスト (1m,3m,6m,9m,12m)",
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        default="2021-06-01T00:00:00",
        help="スナップショット日時 (ISO format)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("outputs/training_window_analysis"),
        help="出力ベースディレクトリ",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    args.output_base.mkdir(parents=True, exist_ok=True)

    # training_windowsはそのまま使用（None変換不要）
    training_windows = args.training_windows

    results = []

    for pred_days in args.prediction_windows:
        for train_days in training_windows:
            LOGGER.info("=== Processing pred=%dd, train=%s ===", pred_days, train_days)

            try:
                # データ生成
                features_path, labels_path = run_data_generation(
                    args.sequences,
                    args.output_base,
                    pred_days,
                    train_days,
                    args.snapshot_date,
                )

                # ベースライン評価
                baseline_output_dir = args.output_base / f"baseline_pred{pred_days}d_train{train_days or 'all'}"
                baseline_metrics = run_baseline_training(
                    features_path,
                    labels_path,
                    baseline_output_dir,
                    f"baseline_pred{pred_days}d_train{train_days or 'all'}",
                )

                # IRL特徴量抽出
                irl_features_path = args.output_base / f"irl_features_pred{pred_days}d_train{train_days or 'all'}.parquet"
                run_irl_feature_extraction(
                    args.irl_model,
                    args.sequences,
                    features_path,
                    labels_path,
                    irl_features_path,
                    train_days,
                    args.snapshot_date,
                )

                # IRL拡張モデル評価
                irl_output_dir = args.output_base / f"irl_pred{pred_days}d_train{train_days or 'all'}"
                irl_metrics = run_baseline_training(
                    irl_features_path,
                    labels_path,
                    irl_output_dir,
                    f"irl_pred{pred_days}d_train{train_days or 'all'}",
                )

                # 結果保存
                result = {
                    "prediction_window_days": pred_days,
                    "training_window_days": train_days,
                    "baseline_metrics": baseline_metrics,
                    "irl_metrics": irl_metrics,
                    "irl_improvement_auc": irl_metrics["roc_auc"] - baseline_metrics["roc_auc"],
                    "irl_improvement_acc": irl_metrics["accuracy"] - baseline_metrics["accuracy"],
                }
                results.append(result)

                LOGGER.info("IRL improvement (AUC): %.4f, (ACC): %.4f", result["irl_improvement_auc"], result["irl_improvement_acc"])

            except Exception as e:
                LOGGER.error("Failed for pred=%dd, train=%s: %s", pred_days, train_days, e)
                continue

    # 結果集計
    results_df = pd.DataFrame(results)
    results_path = args.output_base / "training_window_analysis_results.csv"
    results_df.to_csv(results_path, index=False)

    # 集計表示
    print("\n=== Training Window Analysis Results ===")
    print(results_df.to_string(index=False))

    # 学習期間別の平均改善度
    print("\n=== Average IRL Improvement by Training Window ===")
    avg_by_train = results_df.groupby("training_window_days")[["irl_improvement_auc", "irl_improvement_acc"]].mean()
    print(avg_by_train)

    # AUC改善度マトリクス
    print("\n=== AUC Improvement Matrix (Prediction Window x Training Window) ===")
    auc_matrix = results_df.pivot(
        index="prediction_window_days", 
        columns="training_window_days", 
        values="irl_improvement_auc"
    )
    print(auc_matrix.to_string(float_format="%.4f"))

    # Accuracy改善度マトリクス
    print("\n=== Accuracy Improvement Matrix (Prediction Window x Training Window) ===")
    acc_matrix = results_df.pivot(
        index="prediction_window_days", 
        columns="training_window_days", 
        values="irl_improvement_acc"
    )
    print(acc_matrix.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()