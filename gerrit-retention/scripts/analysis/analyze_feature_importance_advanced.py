#!/usr/bin/env python3
"""
高度な特徴量重要度分析

ReLUなどの非線形性を考慮した特徴量重要度を計算：
1. Permutation Importance - 特徴量をシャッフルして精度変化を測定
2. Integrated Gradients - 勾配ベースの重要度（SHAP的アプローチ）

これにより、第1層の重みだけでなく、モデル全体での実際の影響を測定できます。
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
from tqdm import tqdm

# プロジェクトのモジュールをインポート
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLNetwork
from src.gerrit_retention.rl_prediction.enhanced_retention_irl_system import EnhancedRetentionIRLNetwork


# 特徴量名の定義
STATE_FEATURE_NAMES = [
    'experience_days', 'experience_normalized', 'total_changes',
    'activity_freq_7d', 'activity_freq_30d', 'activity_freq_90d',
    'lines_changed_7d', 'review_load_7d', 'review_load_30d',
    'unique_collaborators', 'avg_interaction_strength', 'cross_project_ratio',
    'total_projects', 'total_files_touched', 'file_type_diversity',
    'avg_directory_depth', 'specialization_score', 'avg_files_per_change',
    'avg_lines_per_change', 'avg_code_complexity', 'avg_complexity_7d',
    'avg_complexity_30d'
]

# 拡張版の場合はさらに追加
EXTENDED_STATE_FEATURES = STATE_FEATURE_NAMES + [
    'code_churn_7d', 'code_churn_30d', 'review_participation_rate',
    'review_response_time', 'avg_review_depth', 'multi_file_change_ratio',
    'collaboration_diversity', 'peak_activity_hour', 'weekend_activity_ratio',
    'consecutive_active_days'
]

ACTION_FEATURE_NAMES = [
    'action_type', 'intensity', 'quality', 'collaboration', 'timestamp_age'
]

EXTENDED_ACTION_FEATURES = ACTION_FEATURE_NAMES + [
    'change_size', 'files_count', 'complexity', 'response_latency'
]


def load_model_and_data(model_path: str) -> Tuple[RetentionIRLNetwork, Dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    モデルと評価データを読み込む

    Returns:
        model: 読み込んだモデル
        config: モデル設定
        test_states: テストデータの状態 [N, seq_len, state_dim] or None
        test_actions: テストデータの行動 [N, seq_len, action_dim] or None
    """
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # 設定を取得
    config = checkpoint.get('config', {})
    print(f"Config: {config}")

    # モデルを初期化
    state_dim = config.get('state_dim', 10)
    action_dim = config.get('action_dim', 5)
    hidden_dim = config.get('hidden_dim', 128)
    sequence = config.get('sequence', False)
    seq_len = config.get('seq_len', 10)
    dropout = config.get('dropout', 0.0)

    # 拡張版かどうかを判定（dropoutがあれば拡張版）
    if dropout > 0 or state_dim > 22:
        print("Using EnhancedRetentionIRLNetwork")
        model = EnhancedRetentionIRLNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            sequence=sequence,
            seq_len=seq_len,
            dropout=dropout
        )
    else:
        print("Using RetentionIRLNetwork")
        model = RetentionIRLNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            sequence=sequence,
            seq_len=seq_len
        )

    # 重みを読み込み
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'network_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['network_state_dict'])
    else:
        raise KeyError("No model state dict found in checkpoint")

    model.eval()

    # 評価データを読み込み（もしあれば）
    test_states = None
    test_actions = None

    # チェックポイントにテストデータがあるか確認
    if 'test_states' in checkpoint and 'test_actions' in checkpoint:
        test_states = checkpoint['test_states']
        test_actions = checkpoint['test_actions']
        print(f"Loaded test data: states {test_states.shape}, actions {test_actions.shape}")

    return model, config, test_states, test_actions


def generate_synthetic_test_data(config: Dict, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    合成テストデータを生成（実データがない場合）

    Returns:
        states: [n_samples, seq_len, state_dim]
        actions: [n_samples, seq_len, action_dim]
        labels: [n_samples] - 継続ラベル（0 or 1）
    """
    state_dim = config.get('state_dim', 10)
    action_dim = config.get('action_dim', 5)
    seq_len = config.get('seq_len', 10)

    print(f"\nGenerating synthetic test data: {n_samples} samples")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}, Seq len: {seq_len}")

    # 正規分布からサンプリング（実際のデータ分布に近い範囲）
    states = np.random.randn(n_samples, seq_len, state_dim).astype(np.float32)
    actions = np.random.randn(n_samples, seq_len, action_dim).astype(np.float32)

    # 一部の特徴を非負にする（経験日数、活動頻度など）
    if state_dim >= 10:
        states[:, :, 0] = np.abs(states[:, :, 0])  # experience_days
        states[:, :, 3:6] = np.abs(states[:, :, 3:6])  # activity frequencies

    if action_dim >= 5:
        actions[:, :, 1] = np.abs(actions[:, :, 1])  # intensity
        actions[:, :, 4] = np.abs(actions[:, :, 4])  # timestamp_age

    # ラベルはランダム（継続率を約10%に設定）
    labels = (np.random.rand(n_samples) < 0.1).astype(np.float32)

    return states, actions, labels


def permutation_importance(
    model: RetentionIRLNetwork,
    states: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    n_repeats: int = 10,
    feature_type: str = 'state'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permutation Importance を計算

    特徴量をシャッフルして予測精度の変化を測定することで、
    各特徴量の重要度を推定する。

    Args:
        model: 学習済みモデル
        states: 状態データ [N, seq_len, state_dim]
        actions: 行動データ [N, seq_len, action_dim]
        labels: ラベル [N]
        n_repeats: シャッフルを繰り返す回数
        feature_type: 'state' or 'action'

    Returns:
        importance_mean: 各特徴量の重要度平均 [feature_dim]
        importance_std: 各特徴量の重要度標準偏差 [feature_dim]
    """
    print(f"\n--- Computing Permutation Importance ({feature_type}) ---")

    # ベースラインの精度を計算
    with torch.no_grad():
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        labels_t = torch.FloatTensor(labels)

        _, cont_prob = model(states_t, actions_t)
        cont_prob = cont_prob.squeeze()

        # AUC-ROCの代わりにMSEを使用（簡易版）
        baseline_loss = ((cont_prob - labels_t) ** 2).mean().item()

    print(f"Baseline loss (MSE): {baseline_loss:.6f}")

    # 各特徴量をシャッフルして重要度を計算
    if feature_type == 'state':
        n_features = states.shape[2]
        data_to_permute = states
    else:  # action
        n_features = actions.shape[2]
        data_to_permute = actions

    importances = []

    for feature_idx in tqdm(range(n_features), desc=f"Permuting {feature_type} features"):
        feature_importances = []

        for _ in range(n_repeats):
            # データをコピーして、この特徴量だけシャッフル
            if feature_type == 'state':
                permuted_states = states.copy()
                permuted_actions = actions.copy()

                # この特徴量をシャッフル（全タイムステップ）
                permuted_states[:, :, feature_idx] = np.random.permutation(
                    permuted_states[:, :, feature_idx].flatten()
                ).reshape(permuted_states[:, :, feature_idx].shape)
            else:
                permuted_states = states.copy()
                permuted_actions = actions.copy()

                permuted_actions[:, :, feature_idx] = np.random.permutation(
                    permuted_actions[:, :, feature_idx].flatten()
                ).reshape(permuted_actions[:, :, feature_idx].shape)

            # シャッフル後の精度を計算
            with torch.no_grad():
                states_t = torch.FloatTensor(permuted_states)
                actions_t = torch.FloatTensor(permuted_actions)

                _, cont_prob = model(states_t, actions_t)
                cont_prob = cont_prob.squeeze()

                permuted_loss = ((cont_prob - labels_t) ** 2).mean().item()

            # 重要度 = ベースライン精度 - シャッフル後精度
            # （精度が下がれば重要度が高い）
            importance = permuted_loss - baseline_loss
            feature_importances.append(importance)

        importances.append(feature_importances)

    importances = np.array(importances)  # [n_features, n_repeats]
    importance_mean = importances.mean(axis=1)
    importance_std = importances.std(axis=1)

    return importance_mean, importance_std


def integrated_gradients(
    model: RetentionIRLNetwork,
    states: np.ndarray,
    actions: np.ndarray,
    n_steps: int = 50,
    feature_type: str = 'state'
) -> np.ndarray:
    """
    Integrated Gradients を計算

    ベースライン（ゼロ）から実際の入力までのパスに沿って勾配を積分することで、
    各特徴量の寄与度を計算する。

    Args:
        model: 学習済みモデル
        states: 状態データ [N, seq_len, state_dim]
        actions: 行動データ [N, seq_len, action_dim]
        n_steps: 積分ステップ数
        feature_type: 'state' or 'action'

    Returns:
        attributions: 各特徴量の寄与度 [feature_dim]
    """
    print(f"\n--- Computing Integrated Gradients ({feature_type}) ---")

    model.eval()
    states_t = torch.FloatTensor(states).requires_grad_(True)
    actions_t = torch.FloatTensor(actions).requires_grad_(True)

    # ベースライン（ゼロ）
    baseline_states = torch.zeros_like(states_t)
    baseline_actions = torch.zeros_like(actions_t)

    # 積分パス上で勾配を計算
    all_grads = []

    for step in tqdm(range(n_steps), desc=f"Computing gradients ({feature_type})"):
        # 線形補間
        alpha = (step + 1) / n_steps

        if feature_type == 'state':
            interpolated_states = baseline_states + alpha * (states_t - baseline_states)
            interpolated_actions = actions_t.detach()
            interpolated_states = interpolated_states.clone().detach().requires_grad_(True)
        else:
            interpolated_states = states_t.detach()
            interpolated_actions = baseline_actions + alpha * (actions_t - baseline_actions)
            interpolated_actions = interpolated_actions.clone().detach().requires_grad_(True)

        # 順伝播
        _, cont_prob = model(interpolated_states, interpolated_actions)
        cont_prob = cont_prob.mean()  # スカラーに集約

        # 勾配計算
        cont_prob.backward()

        if feature_type == 'state':
            grads = interpolated_states.grad.clone().detach().cpu().numpy()
        else:
            grads = interpolated_actions.grad.clone().detach().cpu().numpy()

        all_grads.append(grads)

        # 勾配をクリア
        model.zero_grad()
        if feature_type == 'state':
            interpolated_states.grad.zero_()
        else:
            interpolated_actions.grad.zero_()

    # 勾配の平均
    avg_grads = np.mean(all_grads, axis=0)  # [N, seq_len, feature_dim]

    # 入力との積
    if feature_type == 'state':
        input_diff = states - baseline_states.detach().cpu().numpy()
    else:
        input_diff = actions - baseline_actions.detach().cpu().numpy()

    attributions = avg_grads * input_diff  # [N, seq_len, feature_dim]

    # 全サンプル、全タイムステップで平均
    attributions_mean = np.abs(attributions).mean(axis=(0, 1))  # [feature_dim]

    return attributions_mean


def plot_importance_comparison(
    perm_mean: np.ndarray,
    perm_std: np.ndarray,
    ig_importance: np.ndarray,
    feature_names: List[str],
    feature_type: str,
    output_dir: Path
):
    """
    Permutation ImportanceとIntegrated Gradientsを比較するプロット
    """
    n_features = len(feature_names)

    # データフレーム作成
    df = pd.DataFrame({
        'Feature': feature_names,
        'Permutation': perm_mean,
        'Permutation_Std': perm_std,
        'Integrated_Gradients': ig_importance
    })

    # 並び替え（Permutation Importanceでソート）
    df = df.sort_values('Permutation', ascending=True)

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, n_features * 0.3)))

    # Permutation Importance
    ax1 = axes[0]
    y_pos = np.arange(len(df))
    ax1.barh(y_pos, df['Permutation'], xerr=df['Permutation_Std'],
             color='steelblue', alpha=0.8, capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df['Feature'], fontsize=10)
    ax1.set_xlabel('Importance (Loss Increase)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Permutation Importance - {feature_type.upper()}',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Integrated Gradients
    ax2 = axes[1]
    colors = ['green' if x > 0 else 'orange' for x in df['Integrated_Gradients']]
    ax2.barh(y_pos, df['Integrated_Gradients'], color=colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df['Feature'], fontsize=10)
    ax2.set_xlabel('Attribution (|Gradient × Input|)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Integrated Gradients - {feature_type.upper()}',
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    output_path = output_dir / f'{feature_type}_importance_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved: {output_path}")
    plt.close()

    # CSV保存
    csv_path = output_dir / f'{feature_type}_importance_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Comparison data saved: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Advanced feature importance analysis')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pth)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of synthetic samples (if no test data)')
    parser.add_argument('--n-repeats', type=int, default=10,
                        help='Number of permutation repeats')
    parser.add_argument('--n-steps', type=int, default=50,
                        help='Number of integration steps for IG')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルとデータを読み込み
    model, config, test_states, test_actions = load_model_and_data(args.model)

    # テストデータがなければ合成データを生成
    if test_states is None or test_actions is None:
        states, actions, labels = generate_synthetic_test_data(config, args.n_samples)
    else:
        states = test_states
        actions = test_actions
        # ラベルが保存されていない場合は合成
        labels = (np.random.rand(len(states)) < 0.1).astype(np.float32)

    print(f"\nUsing data: states {states.shape}, actions {actions.shape}, labels {labels.shape}")

    # 特徴量名を取得
    state_dim = config.get('state_dim', 10)
    action_dim = config.get('action_dim', 5)

    if state_dim == 32:
        state_features = EXTENDED_STATE_FEATURES
    elif state_dim == 22:
        state_features = STATE_FEATURE_NAMES
    else:
        state_features = [f'state_{i}' for i in range(state_dim)]

    if action_dim == 9:
        action_features = EXTENDED_ACTION_FEATURES
    elif action_dim == 5:
        action_features = ACTION_FEATURE_NAMES
    else:
        action_features = [f'action_{i}' for i in range(action_dim)]

    # State features 分析
    print("\n" + "="*80)
    print("STATE FEATURES ANALYSIS")
    print("="*80)

    perm_mean_state, perm_std_state = permutation_importance(
        model, states, actions, labels,
        n_repeats=args.n_repeats, feature_type='state'
    )

    ig_state = integrated_gradients(
        model, states, actions,
        n_steps=args.n_steps, feature_type='state'
    )

    df_state = plot_importance_comparison(
        perm_mean_state, perm_std_state, ig_state,
        state_features[:state_dim], 'state', output_dir
    )

    # Action features 分析
    print("\n" + "="*80)
    print("ACTION FEATURES ANALYSIS")
    print("="*80)

    perm_mean_action, perm_std_action = permutation_importance(
        model, states, actions, labels,
        n_repeats=args.n_repeats, feature_type='action'
    )

    ig_action = integrated_gradients(
        model, states, actions,
        n_steps=args.n_steps, feature_type='action'
    )

    df_action = plot_importance_comparison(
        perm_mean_action, perm_std_action, ig_action,
        action_features[:action_dim], 'action', output_dir
    )

    # サマリーレポート作成
    report_path = output_dir / 'importance_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ADVANCED FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("Methods:\n")
        f.write("  1. Permutation Importance: Measures loss increase when feature is shuffled\n")
        f.write("  2. Integrated Gradients: Gradient-based attribution along baseline path\n\n")

        f.write("="*80 + "\n")
        f.write("STATE FEATURES (Top 10)\n")
        f.write("="*80 + "\n\n")

        top_state = df_state.nlargest(10, 'Permutation')
        f.write("By Permutation Importance:\n")
        for idx, row in top_state.iterrows():
            f.write(f"  {row['Feature']:30s}  Perm: {row['Permutation']:8.6f} ± {row['Permutation_Std']:8.6f}  IG: {row['Integrated_Gradients']:8.6f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("ACTION FEATURES (All)\n")
        f.write("="*80 + "\n\n")

        f.write("By Permutation Importance:\n")
        for idx, row in df_action.iterrows():
            f.write(f"  {row['Feature']:30s}  Perm: {row['Permutation']:8.6f} ± {row['Permutation_Std']:8.6f}  IG: {row['Integrated_Gradients']:8.6f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"Analysis complete! All outputs saved to: {output_dir}\n")
        f.write("="*80 + "\n")

    print(f"\nReport saved: {report_path}")
    print("\n" + "="*80)
    print(f"Advanced importance analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
