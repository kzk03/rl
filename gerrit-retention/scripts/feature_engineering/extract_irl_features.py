"""IRL特徴量抽出スクリプト。

学習済みIRLモデルから特徴量を抽出し、Retention予測の入力として活用。
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

LOGGER = logging.getLogger(__name__)


def extract_irl_features(
    irl_model: RetentionIRLSystem,
    sequences: List[Dict],
    labels_df: pd.DataFrame,
    training_window_days: int = None,
    snapshot_date: str = None,
) -> np.ndarray:
    """IRLモデルからレビュワーごとの特徴量を抽出（予測時点まで限定）"""
    from collections import defaultdict
    from datetime import datetime

    # reviewer_idごとにsequencesをグループ化
    reviewer_sequences = defaultdict(list)
    for seq in sequences:
        reviewer_id = seq.get("reviewer_id", "unknown")
        reviewer_sequences[reviewer_id].append(seq)
    
    # prediction_timeを取得
    if snapshot_date:
        global_prediction_time = datetime.fromisoformat(snapshot_date).replace(tzinfo=timezone.utc)
        prediction_times = {row['developer_id']: global_prediction_time for _, row in labels_df.iterrows()}
    else:
        prediction_times = {}
        for _, row in labels_df.iterrows():
            prediction_times[row['developer_id']] = datetime.fromisoformat(row['prediction_time'])
    
    reviewer_features = {}
    
    for reviewer_id, seqs in reviewer_sequences.items():
        if reviewer_id not in prediction_times:
            continue
            
        prediction_time = prediction_times[reviewer_id]
        
        # sequences.jsonの構造: seqs = [{'reviewer_id': ..., 'transitions': [...]}]
        if not seqs or 'transitions' not in seqs[0]:
            continue
        transitions = seqs[0]['transitions']
        
        # prediction_timeまでのtransitionsのみ使用（学習期間制限）
        from datetime import timedelta
        if training_window_days is not None:
            training_start = prediction_time - timedelta(days=training_window_days)
            past_seqs = [seq for seq in transitions if training_start <= datetime.fromisoformat(seq.get('t', '2000-01-01')).replace(tzinfo=timezone.utc) <= prediction_time]
        else:
            past_seqs = [seq for seq in transitions if datetime.fromisoformat(seq.get('t', '2000-01-01')).replace(tzinfo=timezone.utc) <= prediction_time]
        
        if not past_seqs:
            continue
        
        # 各transitionからIRL featuresを抽出し平均を取る
        feature_list = []
        for seq in past_seqs:
            state_dict = seq.get("state", {})
            action_int = seq.get("action", 0)
            
            # stateをリストに変換（sequences.jsonの実際のstate構造に合わせる）
            state_vec = [
                state_dict.get('gap_days', 0),
                state_dict.get('activity_7d', 0),
                state_dict.get('activity_30d', 0),
                state_dict.get('activity_90d', 0),
                state_dict.get('activity_ratio_7_30', 0),
                state_dict.get('activity_ratio_30_90', 0),
                state_dict.get('avg_gap_recent5', 0),
                state_dict.get('workload_level', 0.5),
                state_dict.get('burnout_risk', 0),
                state_dict.get('expertise_recent', 0),
                # 残りを0.5で埋める
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ][:20]
            
            state_np = np.array(state_vec, dtype=np.float32)
            if state_np.shape[0] != 20:
                continue
            
            action_oh = np.zeros(3, dtype=np.float32)
            if 0 <= action_int < 3:
                action_oh[action_int] = 1.0
            
            # テンソル変換
            state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(irl_model.device)
            action_tensor = torch.tensor(action_oh, dtype=torch.float32).unsqueeze(0).to(irl_model.device)
            
            # 特徴量抽出
            with torch.no_grad():
                result = irl_model.network(state_tensor, action_tensor, return_hidden=True)
                if len(result) == 3:
                    reward_pred, cont_prob, hidden = result
                else:
                    reward_pred, cont_prob = result
                    hidden = reward_pred  # fallback
                
                feature_vec = np.concatenate([
                    hidden.cpu().numpy().flatten(),
                    [reward_pred.item()],
                    [cont_prob.item()]
                ])
            
            feature_list.append(feature_vec)
        
        if feature_list:
            # 平均を取る
            avg_features = np.mean(feature_list, axis=0)
            reviewer_features[reviewer_id] = avg_features
    
    # labels_dfの順に並べた配列を返す
    feature_list = []
    for _, row in labels_df.iterrows():
        reviewer_id = row['developer_id']
        if reviewer_id in reviewer_features:
            feature_list.append(reviewer_features[reviewer_id])
        else:
            # デフォルト値
            feature_list.append(np.zeros(128 + 2))  # hidden_dim + reward + cont_prob
    
    return np.array(feature_list)


def create_irl_enhanced_features(
    original_features: pd.DataFrame,
    irl_features: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """元の特徴量にIRL特徴量を追加"""
    irl_df = pd.DataFrame(
        irl_features,
        columns=feature_names,
        index=original_features.index
    )

    enhanced = pd.concat([original_features, irl_df], axis=1)
    return enhanced


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IRL features for retention prediction")
    parser.add_argument(
        "--irl-model",
        type=Path,
        required=True,
        help="学習済みIRLモデルパス",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        required=True,
        help="レビュワーシーケンスJSONLファイル",
    )
    parser.add_argument(
        "--original-features",
        type=Path,
        required=True,
        help="元の特徴量Parquetファイル",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="ラベルParquetファイル（prediction_time用）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="拡張特徴量出力パス",
    )
    parser.add_argument(
        "--training-window-days",
        type=int,
        default=None,
        help="学習期間の日数（Noneの場合は全期間使用）",
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        default=None,
        help="スナップショット日時 (ISO format)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # IRLモデル読み込み
    LOGGER.info("IRLモデルを読み込み中: %s", args.irl_model)
    irl_model = RetentionIRLSystem.load_model(str(args.irl_model))

    # シーケンスデータ読み込み
    LOGGER.info("シーケンスデータを読み込み中: %s", args.sequences)
    with args.sequences.open("r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON array format
            sequences = json.loads(content)
        else:
            # JSONL format
            sequences = []
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    sequences.append(json.loads(line))

    # IRL特徴量抽出
    LOGGER.info("IRL特徴量を抽出中...")
    labels_df = pd.read_parquet(args.labels)
    irl_features = extract_irl_features(irl_model, sequences, labels_df, training_window_days=args.training_window_days, snapshot_date=args.snapshot_date)

    # 元の特徴量読み込み
    LOGGER.info("元の特徴量を読み込み中: %s", args.original_features)
    original_features = pd.read_parquet(args.original_features)

    # 特徴量名生成
    hidden_dim = irl_model.config['hidden_dim']
    feature_names = [f"irl_hidden_{i}" for i in range(hidden_dim)]
    feature_names.extend(["irl_reward_pred", "irl_cont_prob"])

    # 特徴量統合
    LOGGER.info("特徴量を統合中...")
    enhanced_features = create_irl_enhanced_features(
        original_features, irl_features, feature_names
    )

    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enhanced_features.to_parquet(args.output)
    LOGGER.info("拡張特徴量を保存しました: %s", args.output)
    LOGGER.info("特徴量数: %d -> %d", len(original_features.columns), len(enhanced_features.columns))


if __name__ == "__main__":
    import json
    main()