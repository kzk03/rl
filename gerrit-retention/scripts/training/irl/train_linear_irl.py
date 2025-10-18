#!/usr/bin/env python3
"""
線形IRLによる継続予測

報酬関数を線形モデルでパラメータ化することで、
ロジスティック回帰のように解釈可能なIRLを実現します。

報酬関数:
  r(s, a) = w_s^T s + w_a^T a + b

継続確率:
  P(継続) = sigmoid(r(s, a))

利点:
- IRL（逆強化学習）の枠組みを使える
- 重みの符号が解釈可能（ロジスティック回帰と同じ）
- 専門家の軌跡から学習
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# プロジェクトのモジュールをインポート
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


# 特徴量名の定義
EXTENDED_STATE_FEATURES = [
    'experience_days', 'experience_normalized', 'total_changes',
    'activity_freq_7d', 'activity_freq_30d', 'activity_freq_90d',
    'lines_changed_7d', 'review_load_7d', 'review_load_30d',
    'unique_collaborators', 'avg_interaction_strength', 'cross_project_ratio',
    'total_projects', 'total_files_touched', 'file_type_diversity',
    'avg_directory_depth', 'specialization_score', 'avg_files_per_change',
    'avg_lines_per_change', 'avg_code_complexity', 'avg_complexity_7d',
    'avg_complexity_30d',
    'code_churn_7d', 'code_churn_30d', 'review_participation_rate',
    'review_response_time', 'avg_review_depth', 'multi_file_change_ratio',
    'collaboration_diversity', 'peak_activity_hour', 'weekend_activity_ratio',
    'consecutive_active_days'
]

EXTENDED_ACTION_FEATURES = [
    'action_type', 'intensity', 'quality', 'collaboration', 'timestamp_age',
    'change_size', 'files_count', 'complexity', 'response_latency'
]


class LinearRewardFunction(nn.Module):
    """
    線形報酬関数

    r(s, a) = w_s^T s + w_a^T a + b
    """

    def __init__(self, state_dim: int, action_dim: int, use_lstm: bool = False, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm
        self.hidden_dim = hidden_dim

        if use_lstm:
            # LSTM for temporal pattern learning
            self.lstm = nn.LSTM(
                state_dim + action_dim,  # 状態と行動を結合
                hidden_dim,
                num_layers=1,
                batch_first=True
            )
            # LSTMの出力から報酬への線形写像
            self.output_weight = nn.Parameter(torch.zeros(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            # 状態の重み
            self.state_weight = nn.Parameter(torch.zeros(state_dim))

            # 行動の重み
            self.action_weight = nn.Parameter(torch.zeros(action_dim))

            # バイアス
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        報酬を計算

        Args:
            state: [batch, state_dim] or [batch, seq_len, state_dim]
            action: [batch, action_dim] or [batch, seq_len, action_dim]

        Returns:
            reward: [batch] or [batch, seq_len]
        """
        if self.use_lstm:
            # LSTM使用時
            if state.dim() == 2:
                # 単一タイムステップの場合、シーケンス次元を追加
                state = state.unsqueeze(1)  # [batch, 1, state_dim]
                action = action.unsqueeze(1)  # [batch, 1, action_dim]

            # 状態と行動を結合
            combined = torch.cat([state, action], dim=-1)  # [batch, seq_len, state_dim + action_dim]

            # LSTM
            lstm_out, _ = self.lstm(combined)  # [batch, seq_len, hidden_dim]
            hidden = lstm_out[:, -1, :]  # 最終タイムステップ [batch, hidden_dim]

            # 線形写像
            reward = torch.matmul(hidden, self.output_weight) + self.bias  # [batch]

        else:
            # LSTM なし（従来の線形）
            # 時系列の場合は最終タイムステップを使用
            if state.dim() == 3:
                state = state[:, -1, :]  # [batch, state_dim]
                action = action[:, -1, :]  # [batch, action_dim]

            # 線形結合
            state_contrib = torch.matmul(state, self.state_weight)  # [batch]
            action_contrib = torch.matmul(action, self.action_weight)  # [batch]

            reward = state_contrib + action_contrib + self.bias  # [batch]

        return reward

    def get_weights(self) -> Dict[str, np.ndarray]:
        """重みを辞書で返す"""
        if self.use_lstm:
            return {
                'output_weight': self.output_weight.detach().cpu().numpy(),
                'bias': self.bias.detach().cpu().numpy()
            }
        else:
            return {
                'state_weight': self.state_weight.detach().cpu().numpy(),
                'action_weight': self.action_weight.detach().cpu().numpy(),
                'bias': self.bias.detach().cpu().numpy()
            }


class LinearIRLSystem:
    """線形IRL システム"""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.01,
                 use_lstm: bool = False, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.use_lstm = use_lstm

        # 報酬関数（線形 or LSTM）
        self.reward_function = LinearRewardFunction(state_dim, action_dim, use_lstm, hidden_dim)

        # オプティマイザ
        self.optimizer = optim.Adam(self.reward_function.parameters(), lr=learning_rate)

        # 標準化用のスケーラー
        self.state_scaler = StandardScaler()
        self.action_scaler = StandardScaler()

    def fit_scalers(self, states: np.ndarray, actions: np.ndarray):
        """スケーラーをフィット"""
        # 時系列の場合は全タイムステップで統計量を計算
        if states.ndim == 3:
            states_flat = states.reshape(-1, states.shape[-1])
            actions_flat = actions.reshape(-1, actions.shape[-1])
        else:
            states_flat = states
            actions_flat = actions

        self.state_scaler.fit(states_flat)
        self.action_scaler.fit(actions_flat)

    def normalize(self, states: np.ndarray, actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """データを標準化してTensorに変換"""
        if states.ndim == 3:
            # 時系列
            batch, seq_len, state_dim = states.shape
            states_norm = self.state_scaler.transform(states.reshape(-1, state_dim)).reshape(batch, seq_len, state_dim)

            batch, seq_len, action_dim = actions.shape
            actions_norm = self.action_scaler.transform(actions.reshape(-1, action_dim)).reshape(batch, seq_len, action_dim)
        else:
            # 単一タイムステップ
            states_norm = self.state_scaler.transform(states)
            actions_norm = self.action_scaler.transform(actions)

        states_t = torch.FloatTensor(states_norm)
        actions_t = torch.FloatTensor(actions_norm)

        return states_t, actions_t

    def train_irl(self,
                  states: np.ndarray,
                  actions: np.ndarray,
                  labels: np.ndarray,
                  epochs: int = 100,
                  batch_size: int = 64) -> Dict:
        """
        IRLで報酬関数を学習

        Args:
            states: [N, state_dim] or [N, seq_len, state_dim]
            actions: [N, action_dim] or [N, seq_len, action_dim]
            labels: [N] - 継続ラベル (0 or 1)
            epochs: エポック数
            batch_size: バッチサイズ

        Returns:
            metrics: 訓練履歴
        """
        print(f"\n--- Training Linear IRL ---")
        print(f"States: {states.shape}, Actions: {actions.shape}, Labels: {labels.shape}")
        print(f"Continuation rate: {labels.mean():.1%}")

        # スケーラーをフィット
        self.fit_scalers(states, actions)

        # 標準化
        states_t, actions_t = self.normalize(states, actions)
        labels_t = torch.FloatTensor(labels)

        # データセット
        dataset = torch.utils.data.TensorDataset(states_t, actions_t, labels_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 損失関数（Binary Cross Entropy）
        criterion = nn.BCEWithLogitsLoss()

        # 訓練ループ
        history = {'loss': [], 'auc_roc': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_states, batch_actions, batch_labels in dataloader:
                # 報酬を計算
                rewards = self.reward_function(batch_states, batch_actions)  # [batch]

                # 損失計算（報酬を継続確率のロジットとして扱う）
                loss = criterion(rewards, batch_labels)

                # 逆伝播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)

            # 評価（全データ）
            with torch.no_grad():
                rewards_all = self.reward_function(states_t, actions_t)
                probs = torch.sigmoid(rewards_all).numpy()
                auc = roc_auc_score(labels, probs)
                history['auc_roc'].append(auc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, AUC-ROC: {auc:.4f}")

        # 最終評価
        with torch.no_grad():
            rewards_all = self.reward_function(states_t, actions_t)
            probs = torch.sigmoid(rewards_all).numpy()
            preds = (probs > 0.5).astype(int)

            metrics = {
                'accuracy': accuracy_score(labels, preds),
                'auc_roc': roc_auc_score(labels, probs),
                'auc_pr': average_precision_score(labels, probs),
                'f1': f1_score(labels, preds)
            }

        print(f"\nFinal Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

        return {'metrics': metrics, 'history': history}

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """継続確率を予測"""
        states_t, actions_t = self.normalize(states, actions)

        with torch.no_grad():
            rewards = self.reward_function(states_t, actions_t)
            probs = torch.sigmoid(rewards).numpy()

        return probs

    def get_feature_importance(self) -> Tuple[np.ndarray, np.ndarray]:
        """特徴量の重要度を取得"""
        weights = self.reward_function.get_weights()

        state_importance = np.abs(weights['state_weight'])
        action_importance = np.abs(weights['action_weight'])

        return state_importance, action_importance

    def analyze_weights(self, state_names: List[str], action_names: List[str]) -> pd.DataFrame:
        """重みを分析"""
        weights = self.reward_function.get_weights()

        if self.use_lstm:
            # LSTM使用時は隠れ層の重みを分析
            print("\nNote: Using LSTM - showing output layer weights (less interpretable)")
            output_weight = weights['output_weight']

            df = pd.DataFrame({
                'feature': [f'lstm_hidden_{i}' for i in range(len(output_weight))],
                'weight': output_weight,
                'abs_weight': np.abs(output_weight),
                'type': 'lstm'
            })
        else:
            # 線形モデル使用時は通常の重み分析
            # 状態の重み
            state_df = pd.DataFrame({
                'feature': state_names,
                'weight': weights['state_weight'],
                'abs_weight': np.abs(weights['state_weight']),
                'type': 'state'
            })

            # 行動の重み
            action_df = pd.DataFrame({
                'feature': action_names,
                'weight': weights['action_weight'],
                'abs_weight': np.abs(weights['action_weight']),
                'type': 'action'
            })

            # 結合
            df = pd.concat([state_df, action_df], ignore_index=True)

        df = df.sort_values('abs_weight', ascending=False)

        return df


def generate_realistic_data(n_samples: int = 2000, state_dim: int = 32, action_dim: int = 9,
                           seq_len: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """リアルな合成データを生成（ロジスティック回帰と同じ）"""

    print(f"\nGenerating realistic data: {n_samples} samples")

    # 状態特徴
    states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)

    states[:, :, 0] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))  # experience_days
    states[:, :, 1] = states[:, :, 0] / (states[:, :, 0].max() + 1)  # experience_normalized
    states[:, :, 2] = np.random.lognormal(mean=4, sigma=2, size=(n_samples, seq_len))  # total_changes
    states[:, :, 3] = np.random.beta(2, 5, size=(n_samples, seq_len))  # activity_freq_7d
    states[:, :, 4] = np.random.beta(2, 4, size=(n_samples, seq_len))  # activity_freq_30d
    states[:, :, 5] = np.random.beta(2, 3, size=(n_samples, seq_len))  # activity_freq_90d
    states[:, :, 6] = np.random.lognormal(mean=5, sigma=2, size=(n_samples, seq_len))  # lines_changed_7d
    states[:, :, 7] = np.random.gamma(2, 5, size=(n_samples, seq_len))  # review_load_7d
    states[:, :, 8] = np.random.gamma(2, 5, size=(n_samples, seq_len))  # review_load_30d
    states[:, :, 9] = np.random.beta(2, 3, size=(n_samples, seq_len))  # unique_collaborators
    states[:, :, 10] = np.random.beta(2, 5, size=(n_samples, seq_len))  # avg_interaction_strength
    states[:, :, 11] = np.random.beta(2, 4, size=(n_samples, seq_len))  # cross_project_ratio
    states[:, :, 12] = np.random.lognormal(mean=1, sigma=1, size=(n_samples, seq_len))  # total_projects
    states[:, :, 13] = np.random.lognormal(mean=3, sigma=1.5, size=(n_samples, seq_len))  # total_files_touched
    states[:, :, 14] = np.random.beta(3, 3, size=(n_samples, seq_len))  # file_type_diversity
    states[:, :, 15] = np.random.beta(3, 2, size=(n_samples, seq_len))  # avg_directory_depth
    states[:, :, 16] = np.random.beta(3, 3, size=(n_samples, seq_len))  # specialization_score
    states[:, :, 17] = np.random.lognormal(mean=1, sigma=1, size=(n_samples, seq_len))  # avg_files_per_change
    states[:, :, 18] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))  # avg_lines_per_change
    states[:, :, 19] = np.random.beta(3, 3, size=(n_samples, seq_len))  # avg_code_complexity
    states[:, :, 20] = np.random.beta(3, 3, size=(n_samples, seq_len))  # avg_complexity_7d
    states[:, :, 21] = np.random.beta(3, 3, size=(n_samples, seq_len))  # avg_complexity_30d

    if state_dim >= 32:
        states[:, :, 22] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))  # code_churn_7d
        states[:, :, 23] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))  # code_churn_30d
        states[:, :, 24] = np.random.beta(3, 3, size=(n_samples, seq_len))  # review_participation_rate
        states[:, :, 25] = np.random.gamma(2, 5, size=(n_samples, seq_len))  # review_response_time
        states[:, :, 26] = np.random.beta(2, 3, size=(n_samples, seq_len))  # avg_review_depth
        states[:, :, 27] = np.random.beta(3, 3, size=(n_samples, seq_len))  # multi_file_change_ratio
        states[:, :, 28] = np.random.beta(3, 2, size=(n_samples, seq_len))  # collaboration_diversity
        states[:, :, 29] = np.random.randint(0, 24, size=(n_samples, seq_len)).astype(float)  # peak_activity_hour
        states[:, :, 30] = np.random.beta(2, 5, size=(n_samples, seq_len))  # weekend_activity_ratio
        states[:, :, 31] = np.random.poisson(lam=5, size=(n_samples, seq_len)).astype(float)  # consecutive_active_days

    # 行動特徴
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)

    actions[:, :, 0] = 0.8  # action_type（定数）
    actions[:, :, 1] = np.random.beta(3, 2, size=(n_samples, seq_len))  # intensity
    actions[:, :, 2] = np.random.beta(4, 2, size=(n_samples, seq_len))  # quality
    actions[:, :, 3] = np.random.beta(3, 3, size=(n_samples, seq_len))  # collaboration
    actions[:, :, 4] = np.random.exponential(scale=5, size=(n_samples, seq_len))  # timestamp_age

    if action_dim >= 9:
        actions[:, :, 5] = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, seq_len))  # change_size
        actions[:, :, 6] = np.random.poisson(lam=3, size=(n_samples, seq_len)).astype(float)  # files_count
        actions[:, :, 7] = np.random.beta(3, 3, size=(n_samples, seq_len))  # complexity
        actions[:, :, 8] = np.random.exponential(scale=2, size=(n_samples, seq_len))  # response_latency

    # ラベル生成（リアルな相関）
    continuation_prob = np.zeros(n_samples)

    collaboration_diversity = states[:, -1, 28]
    continuation_prob += 0.3 * collaboration_diversity

    activity_freq_90d = states[:, -1, 5]
    continuation_prob += 0.2 * activity_freq_90d

    intensity = actions[:, -1, 1]
    continuation_prob += 0.2 * intensity

    timestamp_age = actions[:, -1, 4]
    continuation_prob -= 0.1 * (timestamp_age / 30.0)

    continuation_prob = continuation_prob - continuation_prob.mean() + 0.1
    continuation_prob = np.clip(continuation_prob, 0, 1)

    labels = (np.random.rand(n_samples) < continuation_prob).astype(np.float32)

    print(f"  Actual continuation rate: {labels.mean():.1%}")

    return states, actions, labels


def plot_analysis(df: pd.DataFrame, history: Dict, output_dir: Path, top_k: int = 30):
    """分析結果を可視化"""

    # 1. 重みの可視化
    fig, axes = plt.subplots(1, 2, figsize=(18, max(10, top_k * 0.3)))

    df_top = df.head(top_k).sort_values('weight', ascending=True)

    # 符号付き重み
    ax1 = axes[0]
    colors = ['green' if x > 0 else 'red' for x in df_top['weight']]
    y_pos = np.arange(len(df_top))
    ax1.barh(y_pos, df_top['weight'], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_top['feature'], fontsize=9)
    ax1.set_xlabel('Weight (Linear Reward)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Linear IRL Weights (Top {top_k})', fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)

    # 絶対値重み
    ax2 = axes[1]
    df_abs = df.nlargest(top_k, 'abs_weight').sort_values('abs_weight', ascending=True)
    y_pos = np.arange(len(df_abs))
    ax2.barh(y_pos, df_abs['abs_weight'], color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_abs['feature'], fontsize=9)
    ax2.set_xlabel('|Weight| (Importance)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Feature Importance (Top {top_k})', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'linear_irl_weights.png', dpi=150, bbox_inches='tight')
    print(f"Weights plot saved: {output_dir / 'linear_irl_weights.png'}")
    plt.close()

    # 2. 訓練履歴
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(history['loss'], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history['auc_roc'], linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AUC-ROC', fontsize=12)
    ax2.set_title('AUC-ROC', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'linear_irl_training.png', dpi=150, bbox_inches='tight')
    print(f"Training plot saved: {output_dir / 'linear_irl_training.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Linear IRL for developer continuation prediction')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--n-samples', type=int, default=2000, help='Number of samples')
    parser.add_argument('--state-dim', type=int, default=32, help='State dimension')
    parser.add_argument('--action-dim', type=int, default=9, help='Action dimension')
    parser.add_argument('--seq-len', type=int, default=15, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--use-lstm', action='store_true', help='Use LSTM for temporal modeling')
    parser.add_argument('--hidden-dim', type=int, default=64, help='LSTM hidden dimension')
    parser.add_argument('--top-k', type=int, default=30, help='Top features to display')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ生成
    states, actions, labels = generate_realistic_data(
        n_samples=args.n_samples,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        seq_len=args.seq_len
    )

    # 線形IRL システム
    print(f"\n{'='*80}")
    print(f"Configuration: {'LSTM' if args.use_lstm else 'Linear'} IRL")
    print(f"{'='*80}")

    irl = LinearIRLSystem(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        learning_rate=args.lr,
        use_lstm=args.use_lstm,
        hidden_dim=args.hidden_dim
    )

    # 訓練
    result = irl.train_irl(states, actions, labels, epochs=args.epochs)

    # 重み分析
    state_names = EXTENDED_STATE_FEATURES[:args.state_dim]
    action_names = EXTENDED_ACTION_FEATURES[:args.action_dim]

    df_weights = irl.analyze_weights(state_names, action_names)

    # 可視化
    plot_analysis(df_weights, result['history'], output_dir, top_k=args.top_k)

    # CSV保存
    df_weights.to_csv(output_dir / 'linear_irl_weights.csv', index=False)
    print(f"\nWeights saved: {output_dir / 'linear_irl_weights.csv'}")

    # メトリクス保存
    with open(output_dir / 'linear_irl_metrics.json', 'w') as f:
        json.dump(result['metrics'], f, indent=2)

    # レポート
    report_path = output_dir / 'linear_irl_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LINEAR IRL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  State dim: {args.state_dim}\n")
        f.write(f"  Action dim: {args.action_dim}\n")
        f.write(f"  Sequence length: {args.seq_len}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning rate: {args.lr}\n\n")

        f.write("Performance:\n")
        for k, v in result['metrics'].items():
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"TOP {args.top_k} FEATURES\n")
        f.write("="*80 + "\n\n")

        for _, row in df_weights.head(args.top_k).iterrows():
            sign = '+' if row['weight'] > 0 else '-'
            effect = 'INCREASE continuation' if row['weight'] > 0 else 'DECREASE continuation'
            f.write(f"{sign} {row['feature']:50s}  Weight: {row['weight']:8.4f}  ({effect})\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Report saved: {report_path}")
    print("\n" + "="*80)
    print("Linear IRL analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
