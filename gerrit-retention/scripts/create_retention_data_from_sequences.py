"""reviewer_sequencesからRetention予測用データを生成するスクリプト"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def create_retention_data_from_sequences(sequences_path: Path, output_features: Path, output_labels: Path, prediction_window_days: int = 90, training_window_days: int = None, snapshot_date: str = None):
    """reviewer_sequencesからRetention予測用データを生成
    
    Args:
        prediction_window_days: 予測期間（日数）。この期間後に活動があるかを予測
        training_window_days: 学習期間（日数）。Noneの場合は全ての過去データを使用
    """

    with open(sequences_path, 'r', encoding='utf-8') as f:
        sequences_data = json.load(f)

    # スナップショットベースの場合、固定予測時点を設定
    global_prediction_time = None
    if snapshot_date:
        global_prediction_time = datetime.fromisoformat(snapshot_date).replace(tzinfo=timezone.utc)
        print(f"Using snapshot date: {global_prediction_time}")

    features_data = []
    labels_data = []

    for seq in sequences_data:
        reviewer_id = seq['reviewer_id']
        transitions = seq['transitions']

        if not transitions:
            continue

        # transitionsを時系列順にソート
        transitions.sort(key=lambda x: x['t'])
        timestamps = [datetime.fromisoformat(t['t']) for t in transitions]

        # 予測時点を設定
        if global_prediction_time:
            # スナップショットベース: 固定予測時点
            prediction_time = global_prediction_time
        else:
            # 従来: レビュワーごとの予測時点
            start_time = min(timestamps)
            end_time = max(timestamps)
            prediction_time = start_time + (end_time - start_time) / 4
        
        # 活動期間が十分長いレビュワーのみ使用（スナップショットベースの場合はスキップ）
        if not global_prediction_time:
            activity_duration = (end_time - start_time).days
            if activity_duration < prediction_window_days * 2:
                continue        # 予測時点までの遷移のみを使用（学習期間制限）
        if training_window_days is not None:
            training_start = prediction_time - timedelta(days=training_window_days)
            past_transitions = [t for t in transitions if training_start <= datetime.fromisoformat(t['t']) <= prediction_time]
        else:
            past_transitions = [t for t in transitions if datetime.fromisoformat(t['t']) <= prediction_time]
            
        # 過去データがない場合はスキップ
        if not past_transitions:
            continue
            
        # 過去データからの特徴量計算
        past_timestamps = [datetime.fromisoformat(t['t']) for t in past_transitions]
        past_start_time = min(past_timestamps)
        past_total_days = (prediction_time - past_start_time).days if (prediction_time - past_start_time).days > 0 else 1

        # 活動統計（予測時点まで）
        past_total_transitions = len(past_transitions)
        past_avg_gap_days = sum(t['state'].get('gap_days', 0) for t in past_transitions) / len(past_transitions)
        past_avg_activity_30d = sum(t['state'].get('activity_30d', 0) for t in past_transitions) / len(past_transitions)
        past_avg_activity_90d = sum(t['state'].get('activity_90d', 0) for t in past_transitions) / len(past_transitions)
        past_avg_activity_180d = sum(t['state'].get('activity_180d', 0) for t in past_transitions) / len(past_transitions)

        # ラベル: 予測期間内に活動があるかどうか
        future_transitions = [t for t in transitions if prediction_time < datetime.fromisoformat(t['t']) <= prediction_time + timedelta(days=prediction_window_days)]
        has_future_activity = 1 if future_transitions else 0

        # 特徴量 (ラベル作成に使った特徴量は除外してリーク防止)
        features = {
            'developer_id': reviewer_id,
            'days_since_last_activity': (datetime.now(timezone.utc) - prediction_time).days,
            # 'total_changes': past_total_transitions,  # 除外（ラベルリーク）
            # 'avg_insertions': past_avg_activity_30d,  # 除外（ラベルリーク）
            'avg_deletions': past_avg_gap_days,  # 仮定
            # 'activity_period_months': past_total_days / 30,  # 除外（ラベルリーク）
            # 'avg_activity_30d': past_avg_activity_30d,  # 除外（ラベルリーク）
            'avg_activity_90d': past_avg_activity_90d,
            'avg_activity_180d': past_avg_activity_180d,
        }

        # ラベル
        labels = {
            'developer_id': reviewer_id,
            'label': has_future_activity,
            'activity_period_months': past_total_days / 30,
            'total_transitions': past_total_transitions,
            'prediction_time': prediction_time.isoformat(),
        }

        features_data.append(features)
        labels_data.append(labels)

    # DataFrameに変換
    features_df = pd.DataFrame(features_data)
    labels_df = pd.DataFrame(labels_data)

    # 保存
    output_features.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)

    features_df.to_parquet(output_features)
    labels_df.to_parquet(output_labels)

    print(f"Created retention data for {len(features_data)} reviewers")
    print(f"Label distribution: {labels_df['label'].value_counts().to_dict()}")
    print(f"Features saved to {output_features}")
    print(f"Labels saved to {output_labels}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create retention data from reviewer sequences')
    parser.add_argument('--input', type=Path, required=True, help='Input reviewer sequences JSON')
    parser.add_argument('--features-output', type=Path, required=True, help='Output features Parquet')
    parser.add_argument('--labels-output', type=Path, required=True, help='Output labels Parquet')
    parser.add_argument('--prediction-window-days', type=int, default=90, help='Prediction window in days (default: 90)')
    parser.add_argument('--training-window-days', type=int, default=None, help='Training window in days (None = use all history)')
    parser.add_argument('--snapshot-date', type=str, default=None, help='Snapshot date for fixed prediction time (ISO format)')

    args = parser.parse_args()
    create_retention_data_from_sequences(args.input, args.features_output, args.labels_output, args.prediction_window_days, args.training_window_days, args.snapshot_date)