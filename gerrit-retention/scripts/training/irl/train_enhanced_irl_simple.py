"""
拡張特徴量を使用したTemporal IRL訓練スクリプト（シンプル版）

正規化を最小限にして動作確認
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from gerrit_retention.rl_prediction.enhanced_feature_extractor import EnhancedFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleEnhancedIRLNetwork(nn.Module):
    """シンプルな拡張IRLネットワーク（正規化なし）"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 sequence: bool = False, seq_len: int = 15):
        super().__init__()
        self.sequence = sequence
        self.seq_len = seq_len

        # シンプルなエンコーダー
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        if self.sequence:
            self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim, num_layers=1, batch_first=True)

        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.continuation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        if self.sequence:
            batch_size, seq_len, _ = state.shape
            state_flat = state.view(-1, state.shape[-1])
            action_flat = action.view(-1, action.shape[-1])

            state_encoded = self.state_encoder(state_flat).view(batch_size, seq_len, -1)
            action_encoded = self.action_encoder(action_flat).view(batch_size, seq_len, -1)

            combined = state_encoded + action_encoded
            lstm_out, _ = self.lstm(combined)
            hidden = lstm_out[:, -1, :]
        else:
            state_encoded = self.state_encoder(state)
            action_encoded = self.action_encoder(action)
            hidden = torch.cat([state_encoded, action_encoded], dim=1)

        reward = self.reward_predictor(hidden)
        continuation_prob = self.continuation_predictor(hidden)

        return reward, continuation_prob


def extract_trajectories(df, snapshot_date, history_months, target_months, fixed_reviewers=None):
    """軌跡を抽出"""
    learning_start = snapshot_date - timedelta(days=history_months * 30)
    learning_end = snapshot_date
    target_start = snapshot_date
    target_end = snapshot_date + timedelta(days=target_months * 30)

    if fixed_reviewers:
        df_learning = df[
            (df['request_time'] >= learning_start) &
            (df['request_time'] < learning_end) &
            (df['reviewer_email'].isin(fixed_reviewers))
        ]
    else:
        df_learning = df[
            (df['request_time'] >= learning_start) &
            (df['request_time'] < learning_end)
        ]

    trajectories = []
    for reviewer_email, group in df_learning.groupby('reviewer_email'):
        activity_history = []
        for _, row in group.iterrows():
            activity = {
                'timestamp': row['request_time'],
                'type': 'review',
                'change_insertions': row.get('change_insertions', 0),
                'change_deletions': row.get('change_deletions', 0),
                'change_files_count': row.get('change_files_count', 1),
                'response_latency_days': row.get('response_latency_days', 0.0),
                'message': row.get('subject', ''),
                'project': row.get('project', '')
            }
            activity_history.append(activity)

        if not activity_history:
            continue

        latest_row = group.iloc[-1]
        developer = {
            'developer_id': reviewer_email,
            'reviewer_email': reviewer_email,
            'changes_authored': 0,
            'changes_reviewed': len(group),
            'projects': group['project'].unique().tolist(),
            'first_seen': group['request_time'].min().isoformat(),
            'last_activity': group['request_time'].max().isoformat(),
            'reviewer_assignment_load_7d': latest_row.get('reviewer_assignment_load_7d', 0),
            'reviewer_assignment_load_30d': latest_row.get('reviewer_assignment_load_30d', 0),
            'reviewer_assignment_load_180d': latest_row.get('reviewer_assignment_load_180d', 0),
            'reviewer_past_reviews_30d': latest_row.get('reviewer_past_reviews_30d', 0),
            'reviewer_past_reviews_90d': latest_row.get('reviewer_past_reviews_90d', 0),
            'reviewer_past_reviews_180d': latest_row.get('reviewer_past_reviews_180d', 0),
            'reviewer_past_response_rate_180d': latest_row.get('reviewer_past_response_rate_180d', 1.0),
            'reviewer_tenure_days': latest_row.get('reviewer_tenure_days', 0),
            'owner_reviewer_past_interactions_180d': latest_row.get('owner_reviewer_past_interactions_180d', 0),
            'owner_reviewer_project_interactions_180d': latest_row.get('owner_reviewer_project_interactions_180d', 0),
            'owner_reviewer_past_assignments_180d': latest_row.get('owner_reviewer_past_assignments_180d', 0),
            'path_jaccard_files_project': latest_row.get('path_jaccard_files_project', 0.0),
            'path_jaccard_dir1_project': latest_row.get('path_jaccard_dir1_project', 0.0),
            'path_jaccard_dir2_project': latest_row.get('path_jaccard_dir2_project', 0.0),
            'path_overlap_files_project': latest_row.get('path_overlap_files_project', 0.0),
            'path_overlap_dir1_project': latest_row.get('path_overlap_dir1_project', 0.0),
            'path_overlap_dir2_project': latest_row.get('path_overlap_dir2_project', 0.0),
            'response_latency_days': latest_row.get('response_latency_days', 0.0),
            'change_insertions': latest_row.get('change_insertions', 0),
            'change_deletions': latest_row.get('change_deletions', 0),
            'change_files_count': latest_row.get('change_files_count', 1)
        }

        continued = len(df[
            (df['reviewer_email'] == reviewer_email) &
            (df['request_time'] >= target_start) &
            (df['request_time'] < target_end)
        ]) > 0

        trajectories.append({
            'developer': developer,
            'activity_history': activity_history,
            'continued': continued,
            'context_date': learning_end
        })

    return trajectories


def main():
    parser = argparse.ArgumentParser(description='シンプル拡張IRL訓練')
    parser.add_argument('--reviews', required=True)
    parser.add_argument('--snapshot-date', required=True)
    parser.add_argument('--history-months', type=int, default=12)
    parser.add_argument('--target-months', type=int, default=6)
    parser.add_argument('--reference-period', type=int, default=6)
    parser.add_argument('--sequence', action='store_true')
    parser.add_argument('--seq-len', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)

    logger.info(f"レビューログ読み込み: {args.reviews}")
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])
    logger.info(f"レビューログ: {len(df)}件")

    snapshot_date = datetime.strptime(args.snapshot_date, '%Y-%m-%d')

    reference_start = snapshot_date - timedelta(days=args.reference_period * 30)
    reference_end = snapshot_date
    df_reference = df[
        (df['request_time'] >= reference_start) &
        (df['request_time'] < reference_end)
    ]
    fixed_reviewers = set(df_reference['reviewer_email'].unique())
    logger.info(f"固定対象レビュアー: {len(fixed_reviewers)}人")

    logger.info("軌跡抽出中...")
    trajectories = extract_trajectories(df, snapshot_date, args.history_months, args.target_months, fixed_reviewers)
    logger.info(f"軌跡数: {len(trajectories)}")

    if len(trajectories) == 0:
        logger.error("軌跡0件")
        return

    split_idx = int(len(trajectories) * 0.8)
    train_traj = trajectories[:split_idx]
    test_traj = trajectories[split_idx:]

    logger.info(f"訓練={len(train_traj)}, テスト={len(test_traj)}")

    # 特徴抽出器とネットワーク
    feature_extractor = EnhancedFeatureExtractor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = SimpleEnhancedIRLNetwork(32, 9, 256, args.sequence, args.seq_len).to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    logger.info("訓練開始...")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_count = 0

        for traj in train_traj:
            try:
                developer = traj['developer']
                activity_history = traj['activity_history']
                continued = traj['continued']
                context_date = traj['context_date']

                state = feature_extractor.extract_enhanced_state(developer, activity_history, context_date)
                actions = [feature_extractor.extract_enhanced_action(act, context_date) for act in activity_history]

                if not actions:
                    continue

                if args.sequence:
                    if len(actions) < args.seq_len:
                        padded_actions = [actions[0]] * (args.seq_len - len(actions)) + actions
                    else:
                        padded_actions = actions[-args.seq_len:]

                    state_arrays = [feature_extractor.state_to_array(state) for _ in range(args.seq_len)]
                    action_arrays = [feature_extractor.action_to_array(act) for act in padded_actions]

                    # Min-Max正規化（手動で0-1に）
                    state_tensor = torch.tensor(np.array(state_arrays), dtype=torch.float32, device=device).unsqueeze(0)
                    action_tensor = torch.tensor(np.array(action_arrays), dtype=torch.float32, device=device).unsqueeze(0)

                    # 簡易正規化（絶対値が大きすぎる場合のみ）
                    state_tensor = torch.clamp(state_tensor / 1000.0, -10, 10)
                    action_tensor = torch.clamp(action_tensor / 1000.0, -10, 10)
                else:
                    recent_action = actions[-1]
                    state_array = feature_extractor.state_to_array(state)
                    action_array = feature_extractor.action_to_array(recent_action)

                    state_tensor = torch.tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0)
                    action_tensor = torch.tensor(action_array, dtype=torch.float32, device=device).unsqueeze(0)

                    state_tensor = torch.clamp(state_tensor / 1000.0, -10, 10)
                    action_tensor = torch.clamp(action_tensor / 1000.0, -10, 10)

                # 予測
                pred_reward, pred_cont = network(state_tensor, action_tensor)

                # ターゲット
                target = torch.tensor([[1.0 if continued else 0.0]], dtype=torch.float32, device=device)

                # 損失（MSEのみ）
                loss = mse_loss(pred_reward, target) + mse_loss(pred_cont, target)

                # 最適化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            except Exception as e:
                logger.warning(f"エラー: {e}")
                continue

        avg_loss = epoch_loss / max(batch_count, 1)
        if epoch % 10 == 0:
            logger.info(f"エポック {epoch}: 損失={avg_loss:.4f}")

    logger.info("訓練完了")

    # 評価
    network.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for traj in test_traj:
            try:
                developer = traj['developer']
                activity_history = traj['activity_history']
                continued = traj['continued']
                context_date = traj['context_date']

                state = feature_extractor.extract_enhanced_state(developer, activity_history, context_date)
                actions = [feature_extractor.extract_enhanced_action(act, context_date) for act in activity_history]

                if not actions:
                    continue

                if args.sequence:
                    if len(actions) < args.seq_len:
                        padded_actions = [actions[0]] * (args.seq_len - len(actions)) + actions
                    else:
                        padded_actions = actions[-args.seq_len:]

                    state_arrays = [feature_extractor.state_to_array(state) for _ in range(args.seq_len)]
                    action_arrays = [feature_extractor.action_to_array(act) for act in padded_actions]

                    state_tensor = torch.tensor(np.array(state_arrays), dtype=torch.float32, device=device).unsqueeze(0)
                    action_tensor = torch.tensor(np.array(action_arrays), dtype=torch.float32, device=device).unsqueeze(0)

                    state_tensor = torch.clamp(state_tensor / 1000.0, -10, 10)
                    action_tensor = torch.clamp(action_tensor / 1000.0, -10, 10)
                else:
                    recent_action = actions[-1]
                    state_array = feature_extractor.state_to_array(state)
                    action_array = feature_extractor.action_to_array(recent_action)

                    state_tensor = torch.tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0)
                    action_tensor = torch.tensor(action_array, dtype=torch.float32, device=device).unsqueeze(0)

                    state_tensor = torch.clamp(state_tensor / 1000.0, -10, 10)
                    action_tensor = torch.clamp(action_tensor / 1000.0, -10, 10)

                _, pred_cont = network(state_tensor, action_tensor)

                prob = pred_cont.item()
                if not (np.isnan(prob) or np.isinf(prob)):
                    y_true.append(1 if continued else 0)
                    y_pred.append(prob)

            except Exception as e:
                continue

    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_pred)
        auc_pr = average_precision_score(y_true, y_pred)
        y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
        f1 = f1_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred_binary)
    else:
        auc_roc = auc_pr = f1 = precision = recall = accuracy = 0.0

    logger.info(f"AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, F1: {f1:.3f}")

    result = {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }

    with open(output_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("\n" + "="*80)
    print("拡張IRL訓練結果（シンプル版）")
    print("="*80)
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
