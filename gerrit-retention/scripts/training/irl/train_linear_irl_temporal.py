#!/usr/bin/env python3
"""
線形IRLで時系列を考慮した予測

LSTMを使わず、完全に線形のまま時系列パターンを捉える手法：
1. 時間重み付き平均（指数的減衰）
2. 複数時間窓の統計量（短期・中期・長期）
3. トレンド特徴（傾き・変化量・変動）

利点:
- 完全に線形 → 各特徴の重みが解釈可能
- 時系列パターンを考慮 → LSTMに近い性能
- 過学習しにくい
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
from scipy import stats as scipy_stats

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


def extract_temporal_features(sequences: np.ndarray, decay_rate: float = 0.1) -> np.ndarray:
    """
    線形モデルで時系列を考慮した特徴量を抽出

    Args:
        sequences: [N, seq_len, feature_dim]
        decay_rate: 指数的減衰率（大きいほど最近を重視）

    Returns:
        temporal_features: [N, feature_dim * n_aggregations]
    """
    N, seq_len, feature_dim = sequences.shape

    features_list = []

    # 時間重みの計算（指数的減衰）
    time_steps = np.arange(seq_len)
    exp_weights = np.exp(-decay_rate * (seq_len - 1 - time_steps))  # 最新が重み1
    exp_weights = exp_weights / exp_weights.sum()  # 正規化

    for i in range(feature_dim):
        feature_seq = sequences[:, :, i]  # [N, seq_len]

        # 1. 最終値（最新の値）
        last_value = feature_seq[:, -1]  # [N]

        # 2. 平均値
        mean_value = feature_seq.mean(axis=1)  # [N]

        # 3. 時間重み付き平均（最近を重視）
        weighted_mean = (feature_seq * exp_weights).sum(axis=1)  # [N]

        # 4. 短期平均（直近3ステップ）
        short_window = min(3, seq_len)
        short_mean = feature_seq[:, -short_window:].mean(axis=1)  # [N]

        # 5. 中期平均（直近7ステップ）
        mid_window = min(7, seq_len)
        mid_mean = feature_seq[:, -mid_window:].mean(axis=1)  # [N]

        # 6. トレンド（線形回帰の傾き）
        # 各サンプルごとに線形回帰
        slopes = []
        for n in range(N):
            y = feature_seq[n, :]
            x = np.arange(seq_len)
            slope, _, _, _, _ = scipy_stats.linregress(x, y)
            slopes.append(slope)
        trend = np.array(slopes)  # [N]

        # 7. 変化量（最終 - 最初）
        change = feature_seq[:, -1] - feature_seq[:, 0]  # [N]

        # 8. 変動（標準偏差）
        volatility = feature_seq.std(axis=1)  # [N]

        # 9. 最大値
        max_value = feature_seq.max(axis=1)  # [N]

        # 10. 最小値
        min_value = feature_seq.min(axis=1)  # [N]

        # 結合
        features_list.extend([
            last_value,
            mean_value,
            weighted_mean,
            short_mean,
            mid_mean,
            trend,
            change,
            volatility,
            max_value,
            min_value
        ])

    # [N, feature_dim * 10]
    temporal_features = np.stack(features_list, axis=1)

    return temporal_features


def create_temporal_feature_names(base_names: List[str]) -> List[str]:
    """時系列特徴量の名前を生成"""
    aggregations = [
        'last', 'mean', 'weighted_mean',
        'short_mean', 'mid_mean',
        'trend', 'change', 'volatility',
        'max', 'min'
    ]

    feature_names = []
    for name in base_names:
        for agg in aggregations:
            feature_names.append(f'{name}_{agg}')

    return feature_names


class LinearTemporalIRLSystem:
    """線形時系列IRL システム"""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.01,
                 decay_rate: float = 0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        # 時系列特徴抽出後の次元数
        self.temporal_state_dim = state_dim * 10
        self.temporal_action_dim = action_dim * 10
        self.total_dim = self.temporal_state_dim + self.temporal_action_dim

        # 線形重み
        self.weights = nn.Parameter(torch.zeros(self.total_dim))
        self.bias = nn.Parameter(torch.zeros(1))

        # オプティマイザ
        self.optimizer = optim.Adam([self.weights, self.bias], lr=learning_rate)

        # スケーラー
        self.scaler = StandardScaler()

    def extract_features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """時系列特徴を抽出"""
        # 状態特徴の抽出
        state_features = extract_temporal_features(states, self.decay_rate)  # [N, state_dim * 10]

        # 行動特徴の抽出
        action_features = extract_temporal_features(actions, self.decay_rate)  # [N, action_dim * 10]

        # 結合
        all_features = np.concatenate([state_features, action_features], axis=1)  # [N, total_dim]

        return all_features

    def train_irl(self, states: np.ndarray, actions: np.ndarray, labels: np.ndarray,
                  epochs: int = 100, batch_size: int = 64) -> Dict:
        """
        IRLで報酬関数を学習

        Args:
            states: [N, seq_len, state_dim]
            actions: [N, seq_len, action_dim]
            labels: [N] - 継続ラベル
        """
        print(f"\n--- Training Linear Temporal IRL ---")
        print(f"States: {states.shape}, Actions: {actions.shape}, Labels: {labels.shape}")
        print(f"Continuation rate: {labels.mean():.1%}")
        print(f"Decay rate: {self.decay_rate} (higher = more recent focus)")

        # 時系列特徴を抽出
        print("\nExtracting temporal features...")
        X = self.extract_features(states, actions)
        print(f"Temporal features shape: {X.shape}")

        # 標準化
        X_scaled = self.scaler.fit_transform(X)
        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(labels)

        # データセット
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 損失関数
        criterion = nn.BCEWithLogitsLoss()

        # 訓練ループ
        history = {'loss': [], 'auc_roc': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                # 報酬計算
                rewards = torch.matmul(batch_X, self.weights) + self.bias  # [batch]

                # 損失
                loss = criterion(rewards, batch_y)

                # 逆伝播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)

            # 評価
            with torch.no_grad():
                rewards_all = torch.matmul(X_t, self.weights) + self.bias
                probs = torch.sigmoid(rewards_all).numpy()
                auc = roc_auc_score(labels, probs)
                history['auc_roc'].append(auc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, AUC-ROC: {auc:.4f}")

        # 最終評価
        with torch.no_grad():
            rewards_all = torch.matmul(X_t, self.weights) + self.bias
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
        X = self.extract_features(states, actions)
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            rewards = torch.matmul(X_t, self.weights) + self.bias
            probs = torch.sigmoid(rewards).numpy()

        return probs

    def analyze_weights(self, state_names: List[str], action_names: List[str]) -> pd.DataFrame:
        """重みを分析"""
        # 時系列特徴名を生成
        state_temporal_names = create_temporal_feature_names(state_names)
        action_temporal_names = create_temporal_feature_names(action_names)
        all_names = state_temporal_names + action_temporal_names

        # 重みを取得
        weights_np = self.weights.detach().cpu().numpy()

        # データフレーム作成
        df = pd.DataFrame({
            'feature': all_names,
            'weight': weights_np,
            'abs_weight': np.abs(weights_np)
        })

        # タイプを分類
        df['type'] = ['state'] * len(state_temporal_names) + ['action'] * len(action_temporal_names)

        # 集約タイプを抽出
        df['aggregation'] = df['feature'].str.split('_').str[-1]

        df = df.sort_values('abs_weight', ascending=False)

        return df


def generate_realistic_data(n_samples: int = 2000, state_dim: int = 32, action_dim: int = 9,
                           seq_len: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """リアルな合成データを生成（他のスクリプトと同じ）"""
    print(f"\nGenerating realistic data: {n_samples} samples")

    states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)

    # 状態特徴（簡略版）
    for i in range(state_dim):
        if i < 10:
            states[:, :, i] = np.random.beta(2, 3, size=(n_samples, seq_len))
        else:
            states[:, :, i] = np.random.lognormal(mean=2, sigma=1, size=(n_samples, seq_len))

    # 行動特徴
    for i in range(action_dim):
        if i == 0:
            actions[:, :, i] = 0.8  # action_type（定数）
        elif i < 4:
            actions[:, :, i] = np.random.beta(3, 2, size=(n_samples, seq_len))
        else:
            actions[:, :, i] = np.random.exponential(scale=3, size=(n_samples, seq_len))

    # ラベル生成（時系列パターンを考慮）
    continuation_prob = np.zeros(n_samples)

    # トレンドが重要
    if state_dim >= 32:
        collab_trend = states[:, -1, 28] - states[:, 0, 28]  # collaboration_diversity のトレンド
        continuation_prob += 0.3 * collab_trend

    # 最近の活動が重要
    recent_activity = states[:, -3:, 5].mean(axis=1)  # activity_freq_90d の直近3ステップ
    continuation_prob += 0.2 * recent_activity

    intensity_trend = actions[:, -1, 1] - actions[:, 0, 1]  # intensity のトレンド
    continuation_prob += 0.2 * intensity_trend

    continuation_prob = continuation_prob - continuation_prob.mean() + 0.1
    continuation_prob = np.clip(continuation_prob, 0, 1)

    labels = (np.random.rand(n_samples) < continuation_prob).astype(np.float32)

    print(f"  Actual continuation rate: {labels.mean():.1%}")

    return states, actions, labels


def plot_analysis(df: pd.DataFrame, history: Dict, output_dir: Path, top_k: int = 30):
    """分析結果を可視化"""

    # 1. トップk個の重み
    fig, axes = plt.subplots(1, 2, figsize=(18, max(10, top_k * 0.3)))

    df_top = df.head(top_k).sort_values('weight', ascending=True)

    # 符号付き重み
    ax1 = axes[0]
    colors = ['green' if x > 0 else 'red' for x in df_top['weight']]
    y_pos = np.arange(len(df_top))
    ax1.barh(y_pos, df_top['weight'], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_top['feature'], fontsize=8)
    ax1.set_xlabel('Weight', fontsize=12, fontweight='bold')
    ax1.set_title(f'Linear Temporal IRL Weights (Top {top_k})', fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)

    # 絶対値重み
    ax2 = axes[1]
    df_abs = df.nlargest(top_k, 'abs_weight').sort_values('abs_weight', ascending=True)
    y_pos = np.arange(len(df_abs))
    ax2.barh(y_pos, df_abs['abs_weight'], color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_abs['feature'], fontsize=8)
    ax2.set_xlabel('|Weight| (Importance)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Feature Importance (Top {top_k})', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_weights.png', dpi=150, bbox_inches='tight')
    print(f"\nWeights plot saved: {output_dir / 'temporal_weights.png'}")
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
    plt.savefig(output_dir / 'temporal_training.png', dpi=150, bbox_inches='tight')
    print(f"Training plot saved: {output_dir / 'temporal_training.png'}")
    plt.close()

    # 3. 集約タイプ別の重要度
    fig, ax = plt.subplots(figsize=(12, 6))

    agg_importance = df.groupby('aggregation')['abs_weight'].sum().sort_values(ascending=True)
    y_pos = np.arange(len(agg_importance))
    ax.barh(y_pos, agg_importance.values, color='coral', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg_importance.index, fontsize=12)
    ax.set_xlabel('Total Importance', fontsize=12, fontweight='bold')
    ax.set_title('Importance by Aggregation Type', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'aggregation_importance.png', dpi=150, bbox_inches='tight')
    print(f"Aggregation importance plot saved: {output_dir / 'aggregation_importance.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Linear Temporal IRL')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--n-samples', type=int, default=2000, help='Number of samples')
    parser.add_argument('--state-dim', type=int, default=32, help='State dimension')
    parser.add_argument('--action-dim', type=int, default=9, help='Action dimension')
    parser.add_argument('--seq-len', type=int, default=15, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay-rate', type=float, default=0.1, help='Exponential decay rate for time weighting')
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

    # 線形時系列IRL
    irl = LinearTemporalIRLSystem(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        learning_rate=args.lr,
        decay_rate=args.decay_rate
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
    df_weights.to_csv(output_dir / 'temporal_weights.csv', index=False)
    print(f"\nWeights saved: {output_dir / 'temporal_weights.csv'}")

    # メトリクス保存
    with open(output_dir / 'temporal_metrics.json', 'w') as f:
        json.dump(result['metrics'], f, indent=2)

    # レポート
    report_path = output_dir / 'temporal_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LINEAR TEMPORAL IRL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  State dim: {args.state_dim}\n")
        f.write(f"  Action dim: {args.action_dim}\n")
        f.write(f"  Sequence length: {args.seq_len}\n")
        f.write(f"  Decay rate: {args.decay_rate}\n")
        f.write(f"  Temporal features: {irl.total_dim}\n\n")

        f.write("Performance:\n")
        for k, v in result['metrics'].items():
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"TOP {args.top_k} FEATURES\n")
        f.write("="*80 + "\n\n")

        for _, row in df_weights.head(args.top_k).iterrows():
            sign = '+' if row['weight'] > 0 else '-'
            effect = 'INCREASE continuation' if row['weight'] > 0 else 'DECREASE continuation'
            f.write(f"{sign} {row['feature']:60s}  Weight: {row['weight']:8.4f}  ({effect})\n")

        f.write("\n" + "="*80 + "\n")
        f.write("AGGREGATION TYPE IMPORTANCE\n")
        f.write("="*80 + "\n\n")

        agg_importance = df_weights.groupby('aggregation')['abs_weight'].sum().sort_values(ascending=False)
        for agg, importance in agg_importance.items():
            f.write(f"  {agg:20s}  {importance:8.4f}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Report saved: {report_path}")
    print("\n" + "="*80)
    print("Linear Temporal IRL analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
