#!/usr/bin/env python3
"""
線形IRLで時系列とステップ間相互作用を捉える

方法1: 特徴量間の相互作用（異なる特徴の変化の積）
方法2: ステップごとの差分（同一特徴のタイミング情報）

両方を組み合わせることで、LSTMが捉えるパターンを線形モデルで再現：
- 「協働が増えながら強度が減る」（方法1）
- 「ステップ5で急激に変化」（方法2）
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
from itertools import combinations

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


def extract_step_diff_features(sequences: np.ndarray, max_steps: int = 5) -> Tuple[np.ndarray, List[str]]:
    """
    方法2: ステップごとの差分特徴

    Args:
        sequences: [N, seq_len, feature_dim]
        max_steps: 最初のk個のステップ差分のみ使用（次元削減）

    Returns:
        diff_features: [N, min(seq_len-1, max_steps) * feature_dim]
        feature_names: 特徴量名のリスト
    """
    N, seq_len, feature_dim = sequences.shape

    diff_features = []
    feature_names = []

    # 各ステップの差分を計算
    n_diffs = min(seq_len - 1, max_steps)

    for step in range(n_diffs):
        diff = sequences[:, step + 1, :] - sequences[:, step, :]  # [N, feature_dim]
        diff_features.append(diff)

        for i in range(feature_dim):
            feature_names.append(f'feat{i}_step{step+1}_diff')

    diff_features = np.concatenate(diff_features, axis=1)  # [N, n_diffs * feature_dim]

    return diff_features, feature_names


def extract_interaction_features(sequences: np.ndarray,
                                 top_k_features: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    方法1: 特徴量間の相互作用（変化量の積）

    Args:
        sequences: [N, seq_len, feature_dim]
        top_k_features: 上位k個の特徴のみで相互作用を計算

    Returns:
        interaction_features: [N, k*(k-1)/2]
        feature_names: 特徴量名のリスト
    """
    N, seq_len, feature_dim = sequences.shape

    # 各特徴の変化量を計算
    changes = sequences[:, -1, :] - sequences[:, 0, :]  # [N, feature_dim]

    # 分散が大きい上位k個の特徴を選択（重要な特徴）
    # feature_dimより大きい場合は全特徴を使用
    k = min(top_k_features, feature_dim)

    variances = changes.var(axis=0)
    top_indices = np.argsort(variances)[-k:]

    changes_top = changes[:, top_indices]  # [N, k]

    # 全ペアの相互作用を計算
    interaction_features = []
    feature_names = []

    for i, j in combinations(range(k), 2):
        interaction = changes_top[:, i] * changes_top[:, j]  # [N]
        interaction_features.append(interaction)
        feature_names.append(f'feat{top_indices[i]}_x_feat{top_indices[j]}_interaction')

    interaction_features = np.stack(interaction_features, axis=1)  # [N, k*(k-1)/2]

    return interaction_features, feature_names


def extract_temporal_features(sequences: np.ndarray, decay_rate: float = 0.1) -> Tuple[np.ndarray, List[str]]:
    """
    基本的な時系列特徴（前回と同じ）

    Returns:
        temporal_features: [N, feature_dim * 10]
        feature_names: 特徴量名のリスト
    """
    N, seq_len, feature_dim = sequences.shape

    features_list = []
    feature_names = []

    # 時間重みの計算
    time_steps = np.arange(seq_len)
    exp_weights = np.exp(-decay_rate * (seq_len - 1 - time_steps))
    exp_weights = exp_weights / exp_weights.sum()

    aggregations = ['last', 'mean', 'weighted_mean', 'short_mean', 'mid_mean',
                   'trend', 'change', 'volatility', 'max', 'min']

    for i in range(feature_dim):
        feature_seq = sequences[:, :, i]

        # 1. 最終値
        last_value = feature_seq[:, -1]

        # 2. 平均値
        mean_value = feature_seq.mean(axis=1)

        # 3. 時間重み付き平均
        weighted_mean = (feature_seq * exp_weights).sum(axis=1)

        # 4. 短期平均
        short_window = min(3, seq_len)
        short_mean = feature_seq[:, -short_window:].mean(axis=1)

        # 5. 中期平均
        mid_window = min(7, seq_len)
        mid_mean = feature_seq[:, -mid_window:].mean(axis=1)

        # 6. トレンド
        slopes = []
        for n in range(N):
            y = feature_seq[n, :]
            x = np.arange(seq_len)
            slope, _, _, _, _ = scipy_stats.linregress(x, y)
            slopes.append(slope)
        trend = np.array(slopes)

        # 7. 変化量
        change = feature_seq[:, -1] - feature_seq[:, 0]

        # 8. 変動
        volatility = feature_seq.std(axis=1)

        # 9. 最大値
        max_value = feature_seq.max(axis=1)

        # 10. 最小値
        min_value = feature_seq.min(axis=1)

        features_list.extend([
            last_value, mean_value, weighted_mean, short_mean, mid_mean,
            trend, change, volatility, max_value, min_value
        ])

        for agg in aggregations:
            feature_names.append(f'feat{i}_{agg}')

    temporal_features = np.stack(features_list, axis=1)

    return temporal_features, feature_names


class LinearTemporalInteractionIRLSystem:
    """線形時系列＋相互作用IRL システム"""

    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 0.01,
                 decay_rate: float = 0.1,
                 use_step_diff: bool = True,
                 use_interaction: bool = True,
                 max_diff_steps: int = 5,
                 top_k_interaction: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.use_step_diff = use_step_diff
        self.use_interaction = use_interaction
        self.max_diff_steps = max_diff_steps
        self.top_k_interaction = top_k_interaction

        # 特徴量名を保存（後で使用）
        self.all_feature_names = []

        # 線形重み（後で初期化）
        self.weights = None
        self.bias = None
        self.optimizer = None

        # スケーラー
        self.scaler = StandardScaler()

    def extract_all_features(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """全ての特徴を抽出"""
        all_features = []
        all_names = []

        # 1. 基本的な時系列特徴
        state_temporal, state_temporal_names = extract_temporal_features(states, self.decay_rate)
        action_temporal, action_temporal_names = extract_temporal_features(actions, self.decay_rate)

        all_features.extend([state_temporal, action_temporal])
        all_names.extend(['state_' + n for n in state_temporal_names])
        all_names.extend(['action_' + n for n in action_temporal_names])

        print(f"  Temporal features: {state_temporal.shape[1] + action_temporal.shape[1]}")

        # 2. ステップごとの差分（方法2）
        if self.use_step_diff:
            state_diff, state_diff_names = extract_step_diff_features(states, self.max_diff_steps)
            action_diff, action_diff_names = extract_step_diff_features(actions, self.max_diff_steps)

            all_features.extend([state_diff, action_diff])
            all_names.extend(['state_' + n for n in state_diff_names])
            all_names.extend(['action_' + n for n in action_diff_names])

            print(f"  Step-diff features: {state_diff.shape[1] + action_diff.shape[1]}")

        # 3. 特徴量間の相互作用（方法1）
        if self.use_interaction:
            state_interaction, state_interaction_names = extract_interaction_features(
                states, self.top_k_interaction
            )
            action_interaction, action_interaction_names = extract_interaction_features(
                actions, self.top_k_interaction
            )

            all_features.extend([state_interaction, action_interaction])
            all_names.extend(['state_' + n for n in state_interaction_names])
            all_names.extend(['action_' + n for n in action_interaction_names])

            print(f"  Interaction features: {state_interaction.shape[1] + action_interaction.shape[1]}")

        # 結合
        X = np.concatenate(all_features, axis=1)

        return X, all_names

    def train_irl(self, states: np.ndarray, actions: np.ndarray, labels: np.ndarray,
                  epochs: int = 100, batch_size: int = 64) -> Dict:
        """IRLで報酬関数を学習"""
        print(f"\n--- Training Linear Temporal + Interaction IRL ---")
        print(f"States: {states.shape}, Actions: {actions.shape}, Labels: {labels.shape}")
        print(f"Continuation rate: {labels.mean():.1%}")
        print(f"\nConfiguration:")
        print(f"  Use step-diff: {self.use_step_diff}")
        print(f"  Use interaction: {self.use_interaction}")
        print(f"  Decay rate: {self.decay_rate}")

        # 全特徴を抽出
        print("\nExtracting features...")
        X, self.all_feature_names = self.extract_all_features(states, actions)
        print(f"\nTotal features: {X.shape[1]}")

        # 重みを初期化
        total_dim = X.shape[1]
        self.weights = nn.Parameter(torch.zeros(total_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        self.optimizer = optim.Adam([self.weights, self.bias], lr=self.learning_rate)

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
                rewards = torch.matmul(batch_X, self.weights) + self.bias

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

    def analyze_weights(self) -> pd.DataFrame:
        """重みを分析"""
        weights_np = self.weights.detach().cpu().numpy()

        df = pd.DataFrame({
            'feature': self.all_feature_names,
            'weight': weights_np,
            'abs_weight': np.abs(weights_np)
        })

        # タイプを分類
        df['type'] = df['feature'].apply(lambda x: 'state' if x.startswith('state_') else 'action')

        # 特徴カテゴリを分類
        df['category'] = 'temporal'
        df.loc[df['feature'].str.contains('_diff'), 'category'] = 'step_diff'
        df.loc[df['feature'].str.contains('_interaction'), 'category'] = 'interaction'

        df = df.sort_values('abs_weight', ascending=False)

        return df


def generate_realistic_data(n_samples: int = 2000, state_dim: int = 32, action_dim: int = 9,
                           seq_len: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    リアルな合成データを生成

    重要: 時系列パターンと特徴間の相互作用を含むデータ
    """
    print(f"\nGenerating realistic data with temporal patterns: {n_samples} samples")

    states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)

    # 状態特徴（時系列パターンあり）
    for i in range(state_dim):
        for n in range(n_samples):
            # ランダムウォーク的な時系列
            initial = np.random.rand()
            trend = np.random.randn() * 0.02
            noise = np.random.randn(seq_len) * 0.1

            time_series = initial + trend * np.arange(seq_len) + np.cumsum(noise)
            states[n, :, i] = np.clip(time_series, 0, 10)

    # 行動特徴（時系列パターンあり）
    for i in range(action_dim):
        for n in range(n_samples):
            if i == 0:
                actions[n, :, i] = 0.8  # action_type（定数）
            else:
                initial = np.random.rand()
                trend = np.random.randn() * 0.02
                noise = np.random.randn(seq_len) * 0.1

                time_series = initial + trend * np.arange(seq_len) + np.cumsum(noise)
                actions[n, :, i] = np.clip(time_series, 0, 10)

    # ラベル生成（複雑なパターン）
    continuation_prob = np.zeros(n_samples)

    if state_dim >= 32:
        # パターン1: collaboration_diversity が増加しながら intensity が減少 → 離脱
        collab_change = states[:, -1, 28] - states[:, 0, 28]
        intensity_change = actions[:, -1, 1] - actions[:, 0, 1]
        negative_pattern = (collab_change > 0) & (intensity_change < 0)
        continuation_prob[negative_pattern] -= 0.3

        # パターン2: 両方とも増加 → 継続
        positive_pattern = (collab_change > 0) & (intensity_change > 0)
        continuation_prob[positive_pattern] += 0.4

        # パターン3: ステップ中盤で急激な変化 → 離脱
        mid_step = seq_len // 2
        collab_mid_change = np.abs(states[:, mid_step, 28] - states[:, mid_step-1, 28])
        sudden_change = collab_mid_change > 0.5
        continuation_prob[sudden_change] -= 0.2

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

    ax1 = axes[0]
    colors = ['green' if x > 0 else 'red' for x in df_top['weight']]
    y_pos = np.arange(len(df_top))
    ax1.barh(y_pos, df_top['weight'], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_top['feature'], fontsize=7)
    ax1.set_xlabel('Weight', fontsize=12, fontweight='bold')
    ax1.set_title(f'Linear Temporal + Interaction IRL Weights (Top {top_k})',
                  fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)

    ax2 = axes[1]
    df_abs = df.nlargest(top_k, 'abs_weight').sort_values('abs_weight', ascending=True)
    y_pos = np.arange(len(df_abs))
    ax2.barh(y_pos, df_abs['abs_weight'], color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_abs['feature'], fontsize=7)
    ax2.set_xlabel('|Weight| (Importance)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Feature Importance (Top {top_k})', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'interaction_weights.png', dpi=150, bbox_inches='tight')
    print(f"\nWeights plot saved: {output_dir / 'interaction_weights.png'}")
    plt.close()

    # 2. カテゴリ別の重要度
    fig, ax = plt.subplots(figsize=(10, 6))

    category_importance = df.groupby('category')['abs_weight'].sum().sort_values(ascending=True)
    y_pos = np.arange(len(category_importance))
    colors_cat = ['coral', 'skyblue', 'lightgreen']
    ax.barh(y_pos, category_importance.values, color=colors_cat, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(category_importance.index, fontsize=12)
    ax.set_xlabel('Total Importance', fontsize=12, fontweight='bold')
    ax.set_title('Importance by Feature Category', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 各カテゴリの説明
    for i, (cat, val) in enumerate(category_importance.items()):
        if cat == 'temporal':
            desc = '(Basic temporal features)'
        elif cat == 'step_diff':
            desc = '(Step-by-step differences)'
        else:
            desc = '(Feature interactions)'
        ax.text(val, i, f' {desc}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'category_importance.png', dpi=150, bbox_inches='tight')
    print(f"Category importance plot saved: {output_dir / 'category_importance.png'}")
    plt.close()

    # 3. 訓練履歴
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
    plt.savefig(output_dir / 'interaction_training.png', dpi=150, bbox_inches='tight')
    print(f"Training plot saved: {output_dir / 'interaction_training.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Linear Temporal + Interaction IRL')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--n-samples', type=int, default=2000, help='Number of samples')
    parser.add_argument('--state-dim', type=int, default=32, help='State dimension')
    parser.add_argument('--action-dim', type=int, default=9, help='Action dimension')
    parser.add_argument('--seq-len', type=int, default=15, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay-rate', type=float, default=0.1, help='Decay rate')
    parser.add_argument('--no-step-diff', action='store_true', help='Disable step-diff features')
    parser.add_argument('--no-interaction', action='store_true', help='Disable interaction features')
    parser.add_argument('--max-diff-steps', type=int, default=5, help='Max diff steps')
    parser.add_argument('--top-k-interaction', type=int, default=10, help='Top k features for interaction')
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

    # 線形時系列+相互作用IRL
    irl = LinearTemporalInteractionIRLSystem(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        learning_rate=args.lr,
        decay_rate=args.decay_rate,
        use_step_diff=not args.no_step_diff,
        use_interaction=not args.no_interaction,
        max_diff_steps=args.max_diff_steps,
        top_k_interaction=args.top_k_interaction
    )

    # 訓練
    result = irl.train_irl(states, actions, labels, epochs=args.epochs)

    # 重み分析
    df_weights = irl.analyze_weights()

    # 可視化
    plot_analysis(df_weights, result['history'], output_dir, top_k=args.top_k)

    # CSV保存
    df_weights.to_csv(output_dir / 'interaction_weights.csv', index=False)
    print(f"\nWeights saved: {output_dir / 'interaction_weights.csv'}")

    # メトリクス保存
    with open(output_dir / 'interaction_metrics.json', 'w') as f:
        json.dump(result['metrics'], f, indent=2)

    # レポート
    report_path = output_dir / 'interaction_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LINEAR TEMPORAL + INTERACTION IRL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Total features: {len(df_weights)}\n")
        f.write(f"  Use step-diff: {irl.use_step_diff}\n")
        f.write(f"  Use interaction: {irl.use_interaction}\n\n")

        f.write("Performance:\n")
        for k, v in result['metrics'].items():
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"TOP {args.top_k} FEATURES\n")
        f.write("="*80 + "\n\n")

        for _, row in df_weights.head(args.top_k).iterrows():
            sign = '+' if row['weight'] > 0 else '-'
            effect = 'INCREASE' if row['weight'] > 0 else 'DECREASE'
            f.write(f"{sign} {row['feature']:70s}  Weight: {row['weight']:8.4f}  ({effect} continuation) [{row['category']}]\n")

        f.write("\n" + "="*80 + "\n")
        f.write("CATEGORY IMPORTANCE\n")
        f.write("="*80 + "\n\n")

        cat_importance = df_weights.groupby('category')['abs_weight'].sum().sort_values(ascending=False)
        for cat, val in cat_importance.items():
            f.write(f"  {cat:20s}  {val:8.4f}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Report saved: {report_path}")
    print("\n" + "="*80)
    print("Linear Temporal + Interaction IRL analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
