"""
Utility functions for baseline models.

This module provides common functionality shared across all baseline implementations,
including feature extraction, data loading, evaluation, and result saving.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)
import json
from pathlib import Path


def extract_static_features(trajectory_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract static features from time-series trajectory data.

    Converts time-series activity history into aggregated static features
    suitable for traditional ML models.

    Args:
        trajectory_data: List of trajectories, each containing:
            - 'developer': developer info with experience, reviews, etc.
            - 'activity_history': list of activities
            - 'continued': whether developer continued (label)

    Returns:
        Tuple of (features array, feature names)
        - features: numpy array of shape [n_samples, n_features]
        - feature_names: list of feature names
    """
    features = []
    feature_names = [
        # Developer基本特徴量
        'experience_days_normalized',
        'total_reviews_normalized',
        'acceptance_rate',
        'recent_activity_frequency',
        'avg_activity_gap_normalized',
        'collaboration_score',
        'code_quality_score',

        # 時系列統計量（活動強度）
        'intensity_mean',
        'intensity_std',
        'intensity_max',
        'intensity_min',
        'intensity_trend',  # 直近vs過去の比較

        # 時系列統計量（コラボレーション）
        'collaboration_mean',
        'collaboration_std',
        'collaboration_ratio',  # 高コラボレーション活動の割合

        # 時系列統計量（品質）
        'quality_mean',
        'quality_std',

        # 活動パターン
        'activity_count',
        'activity_consistency',  # 活動間隔の標準偏差の逆数

        # 受諾率の時系列
        'accepted_mean',
        'accepted_std',
        'accepted_trend'
    ]

    for trajectory in trajectory_data:
        developer = trajectory['developer']
        history = trajectory.get('activity_history', [])

        # 基本特徴量
        feat = [
            developer.get('experience_days', 0) / 730.0,  # 正規化（2年で1.0）
            developer.get('total_reviews', 0) / 500.0,
            developer.get('acceptance_rate', 0.0),
            developer.get('recent_activity_frequency', 0.0),
            developer.get('avg_activity_gap', 60.0) / 60.0,
            developer.get('collaboration_score', 0.0),
            developer.get('code_quality_score', 0.5)
        ]

        # 時系列統計量（活動強度）
        if history:
            intensities = [a.get('intensity', 0.0) for a in history]
            feat.extend([
                np.mean(intensities),
                np.std(intensities),
                np.max(intensities),
                np.min(intensities),
                # トレンド：直近10個 vs それ以前の平均差
                np.mean(intensities[-10:]) - np.mean(intensities[:-10]) if len(intensities) > 10 else 0.0
            ])

            # コラボレーション統計
            collaborations = [a.get('collaboration', 0.0) for a in history]
            feat.extend([
                np.mean(collaborations),
                np.std(collaborations),
                len([c for c in collaborations if c > 0.5]) / len(collaborations)
            ])

            # 品質統計
            qualities = [a.get('quality', 0.5) for a in history]
            feat.extend([
                np.mean(qualities),
                np.std(qualities)
            ])

            # 活動パターン
            feat.append(len(history))

            # 活動の一貫性（間隔の標準偏差の逆数）
            if len(history) > 1:
                timestamps = [a.get('timestamp', 0) for a in history]
                gaps = np.diff(sorted(timestamps))
                consistency = 1.0 / (np.std(gaps) + 1e-6) if len(gaps) > 0 else 0.0
                feat.append(consistency)
            else:
                feat.append(0.0)

            # 受諾率の時系列
            accepted_list = [a.get('accepted', 0) for a in history]
            feat.extend([
                np.mean(accepted_list),
                np.std(accepted_list),
                np.mean(accepted_list[-10:]) - np.mean(accepted_list[:-10]) if len(accepted_list) > 10 else 0.0
            ])
        else:
            # 履歴がない場合はゼロ埋め
            feat.extend([0.0] * 14)

        features.append(feat)

    return np.array(features), feature_names


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate prediction performance with multiple metrics.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Threshold for binary classification

    Returns:
        Dictionary of evaluation metrics
    """
    # 確率を二値化
    y_pred_binary = (y_pred >= threshold).astype(int)

    # 各種指標を計算
    metrics = {
        'auc_pr': average_precision_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'threshold': threshold
    }

    # 混同行列
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    })

    # 陽性率
    metrics['positive_rate'] = float(np.mean(y_true))

    return metrics


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    model_name: str
) -> None:
    """
    Save baseline experiment results to disk.

    Args:
        results: Dictionary containing:
            - 'predictions': prediction array
            - 'metrics': evaluation metrics
            - 'feature_importance': feature importance dict
            - 'training_time': training time in seconds
        output_dir: Directory to save results
        model_name: Name of the model (e.g., 'logistic_regression')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 予測結果を保存
    if 'predictions' in results:
        predictions_df = pd.DataFrame({
            'prediction': results['predictions'],
            'true_label': results.get('true_labels', [])
        })
        predictions_df.to_csv(output_dir / 'predictions.csv', index=False)

    # メトリクスを保存
    if 'metrics' in results:
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)

    # 特徴量重要度を保存
    if 'feature_importance' in results:
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in results['feature_importance'].items()
        ]).sort_values('importance', ascending=False)
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    print(f"Results saved to {output_dir}")


def load_trajectory_data(csv_path: str, snapshot_date: str, history_months: int, target_months: int) -> Dict[str, List[Dict]]:
    """
    Load and prepare trajectory data from CSV.

    Args:
        csv_path: Path to review CSV file
        snapshot_date: Snapshot date (e.g., '2020-01-01')
        history_months: Number of months for learning period
        target_months: Number of months for prediction period

    Returns:
        Dictionary with 'train' and 'test' trajectory lists
    """
    # この関数は実際のデータ読み込みロジックに合わせて実装する
    # ここではプレースホルダーとして定義
    raise NotImplementedError(
        "load_trajectory_data should be implemented based on "
        "the actual data loading logic from retention_irl_system.py"
    )


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2m 30s", "1h 15m")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
