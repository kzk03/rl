#!/usr/bin/env python3
"""
IRL報酬関数の特徴量重み分析

学習済みIRLモデルから特徴量の重要度を分析します。

使用例:
    python scripts/analysis/analyze_reward_weights.py \
        --model importants/irl_matrix_2023q1/models/irl_h12m_t6m_fixed_seq.pth \
        --output importants/irl_matrix_2023q1/feature_importance.png
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


# 特徴量名の定義
STATE_FEATURE_NAMES = [
    # Experience (4)
    'experience_days',
    'experience_normalized',
    'total_changes',
    'total_projects',

    # Activity patterns (12)
    'activity_freq_7d',
    'activity_freq_30d',
    'activity_freq_90d',
    'review_load_7d',
    'review_load_30d',
    'review_load_90d',
    'lines_changed_7d',
    'lines_changed_30d',
    'lines_changed_90d',
    'concentration_score',
    'avg_complexity_7d',
    'avg_complexity_30d',

    # Collaboration (8)
    'unique_collaborators',
    'collaboration_score',
    'cross_project_ratio',
    'top_collaborator_strength',
    'avg_interaction_strength',
    'num_active_collaborations_30d',
    'collaboration_diversity',
    'recent_collaboration_trend',

    # Technical expertise (8)
    'path_similarity',
    'avg_lines_per_change',
    'avg_files_per_change',
    'avg_code_complexity',
    'file_type_diversity',
    'avg_directory_depth',
    'total_files_touched',
    'specialization_score'
]

ACTION_FEATURE_NAMES = [
    # Basic (5)
    'action_type',
    'intensity',
    'quality',
    'collaboration',
    'timestamp_age',

    # Extended (4)
    'change_size',
    'files_count',
    'complexity',
    'response_latency'
]


def load_model(model_path: str):
    """学習済みモデルを読み込み"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    print(f"Model loaded: {model_path}")
    print(f"Config: {checkpoint.get('config', {})}")

    return checkpoint


def extract_encoder_weights(checkpoint, encoder_name='state_encoder'):
    """エンコーダーの重みを抽出"""
    # モデルの保存形式に応じてキーを選択
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'network_state_dict' in checkpoint:
        state_dict = checkpoint['network_state_dict']
    else:
        raise KeyError(f"Cannot find model weights. Available keys: {list(checkpoint.keys())}")

    # 最初の層の重み（input_dim → hidden_dim）
    first_layer_key = f'{encoder_name}.0.weight'

    if first_layer_key in state_dict:
        weights = state_dict[first_layer_key].numpy()  # [hidden_dim, input_dim]
        return weights
    else:
        print(f"Warning: {first_layer_key} not found in state_dict")
        print(f"Available keys: {list(state_dict.keys())[:10]}")
        return None


def calculate_feature_importance(weights):
    """
    特徴量の重要度を計算

    各入力特徴量から隠れ層への重みの絶対値の平均を重要度とする
    """
    # weights: [hidden_dim, input_dim]
    # 各入力次元について、全隠れユニットへの重みの絶対値平均
    importance = np.abs(weights).mean(axis=0)  # [input_dim]

    return importance


def plot_feature_importance(importance, feature_names, title, output_path, top_k=20):
    """特徴量重要度をプロット"""

    # DataFrameに変換
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    # 重要度で降順ソート
    df = df.sort_values('importance', ascending=False)

    # Top-k を取得
    df_top = df.head(top_k)

    # プロット
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_top)))

    bars = ax.barh(range(len(df_top)), df_top['importance'], color=colors)
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['feature'])
    ax.invert_yaxis()

    ax.set_xlabel('Importance (Mean Absolute Weight)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # 値を表示
    for i, (idx, row) in enumerate(df_top.iterrows()):
        ax.text(
            row['importance'], i,
            f" {row['importance']:.4f}",
            va='center',
            fontsize=9
        )

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plot saved: {output_path}")

    return fig, df


def plot_combined_importance(state_importance, action_importance,
                             state_names, action_names,
                             output_path, top_k=15):
    """状態と行動の特徴量重要度を並べて表示"""

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # State features
    df_state = pd.DataFrame({
        'feature': state_names,
        'importance': state_importance
    }).sort_values('importance', ascending=False).head(top_k)

    colors_state = plt.cm.Blues(np.linspace(0.4, 0.9, len(df_state)))
    axes[0].barh(range(len(df_state)), df_state['importance'], color=colors_state)
    axes[0].set_yticks(range(len(df_state)))
    axes[0].set_yticklabels(df_state['feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance', fontsize=12, fontweight='bold')
    axes[0].set_title('State Feature Importance (Top 15)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')

    for i, (_, row) in enumerate(df_state.iterrows()):
        axes[0].text(row['importance'], i, f" {row['importance']:.4f}", va='center', fontsize=9)

    # Action features
    df_action = pd.DataFrame({
        'feature': action_names,
        'importance': action_importance
    }).sort_values('importance', ascending=False)

    colors_action = plt.cm.Oranges(np.linspace(0.4, 0.9, len(df_action)))
    axes[1].barh(range(len(df_action)), df_action['importance'], color=colors_action)
    axes[1].set_yticks(range(len(df_action)))
    axes[1].set_yticklabels(df_action['feature'])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Importance', fontsize=12, fontweight='bold')
    axes[1].set_title('Action Feature Importance (All)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3, linestyle='--')

    for i, (_, row) in enumerate(df_action.iterrows()):
        axes[1].text(row['importance'], i, f" {row['importance']:.4f}", va='center', fontsize=9)

    plt.suptitle('IRL Reward Function: Feature Importance Analysis',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined importance plot saved: {output_path}")

    return fig


def analyze_layer_by_layer(checkpoint, encoder_name='state_encoder'):
    """層ごとの重み分布を分析"""
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'network_state_dict' in checkpoint:
        state_dict = checkpoint['network_state_dict']
    else:
        state_dict = checkpoint

    layer_stats = []

    for key in state_dict.keys():
        if encoder_name in key and 'weight' in key:
            weights = state_dict[key].numpy()

            stats = {
                'layer': key,
                'shape': weights.shape,
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'mean_abs': np.mean(np.abs(weights))
            }
            layer_stats.append(stats)

    return pd.DataFrame(layer_stats)


def print_statistics(df, title):
    """統計情報を表示"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print()

    # Top-10
    print(f"Top 10 Most Important Features:")
    print("-" * 60)
    for i, row in df.head(10).iterrows():
        print(f"{i+1:2d}. {row['feature']:30s} {row['importance']:.6f}")


def plot_weight_distribution(checkpoint, output_path):
    """重みの分布をヒストグラムで表示"""
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'network_state_dict' in checkpoint:
        state_dict = checkpoint['network_state_dict']
    else:
        state_dict = checkpoint

    # state_encoder と action_encoder の重みを収集
    state_weights = []
    action_weights = []

    for key, value in state_dict.items():
        if 'weight' in key:
            weights = value.numpy().flatten()
            if 'state_encoder' in key:
                state_weights.extend(weights)
            elif 'action_encoder' in key:
                action_weights.extend(weights)

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # State encoder
    axes[0].hist(state_weights, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Weight Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('State Encoder Weight Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Statistics
    state_mean = np.mean(state_weights)
    state_std = np.std(state_weights)
    axes[0].text(
        0.02, 0.98,
        f'Mean: {state_mean:.4f}\nStd: {state_std:.4f}',
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # Action encoder
    axes[1].hist(action_weights, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Weight Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Action Encoder Weight Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # Statistics
    action_mean = np.mean(action_weights)
    action_std = np.std(action_weights)
    axes[1].text(
        0.02, 0.98,
        f'Mean: {action_mean:.4f}\nStd: {action_std:.4f}',
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nWeight distribution plot saved: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='IRL報酬関数の特徴量重み分析'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='学習済みモデルファイル (.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='出力ディレクトリ（指定しない場合はモデルと同じディレクトリ）'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='表示する上位特徴量の数（デフォルト: 20）'
    )

    args = parser.parse_args()

    # モデルを読み込み
    checkpoint = load_model(args.model)

    # 出力ディレクトリの設定
    if args.output is None:
        model_path = Path(args.model)
        output_dir = model_path.parent
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Config から次元数を取得
    config = checkpoint.get('config', {})
    state_dim = config.get('state_dim', 32)
    action_dim = config.get('action_dim', 9)

    print(f"\nModel dimensions:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")

    # 特徴量名を調整
    state_names = STATE_FEATURE_NAMES[:state_dim]
    action_names = ACTION_FEATURE_NAMES[:action_dim]

    # State encoder の重み分析
    print("\n--- Analyzing State Encoder ---")
    state_weights = extract_encoder_weights(checkpoint, 'state_encoder')

    if state_weights is not None:
        state_importance = calculate_feature_importance(state_weights)

        # DataFrame作成
        df_state = pd.DataFrame({
            'feature': state_names,
            'importance': state_importance
        }).sort_values('importance', ascending=False)

        # 統計情報を表示
        print_statistics(df_state, "State Feature Importance")

        # プロット
        plot_feature_importance(
            state_importance,
            state_names,
            'State Feature Importance for IRL Reward Function',
            output_dir / 'state_feature_importance.png',
            top_k=args.top_k
        )

        # CSV保存
        df_state.to_csv(output_dir / 'state_feature_importance.csv', index=False)
        print(f"State feature importance saved: {output_dir / 'state_feature_importance.csv'}")

    # Action encoder の重み分析
    print("\n--- Analyzing Action Encoder ---")
    action_weights = extract_encoder_weights(checkpoint, 'action_encoder')

    if action_weights is not None:
        action_importance = calculate_feature_importance(action_weights)

        # DataFrame作成
        df_action = pd.DataFrame({
            'feature': action_names,
            'importance': action_importance
        }).sort_values('importance', ascending=False)

        # 統計情報を表示
        print_statistics(df_action, "Action Feature Importance")

        # プロット
        plot_feature_importance(
            action_importance,
            action_names,
            'Action Feature Importance for IRL Reward Function',
            output_dir / 'action_feature_importance.png',
            top_k=min(args.top_k, action_dim)
        )

        # CSV保存
        df_action.to_csv(output_dir / 'action_feature_importance.csv', index=False)
        print(f"Action feature importance saved: {output_dir / 'action_feature_importance.csv'}")

    # 統合プロット
    if state_weights is not None and action_weights is not None:
        plot_combined_importance(
            state_importance,
            action_importance,
            state_names,
            action_names,
            output_dir / 'combined_feature_importance.png',
            top_k=15
        )

    # 層ごとの統計
    print("\n--- Layer-by-Layer Analysis ---")
    state_layer_stats = analyze_layer_by_layer(checkpoint, 'state_encoder')
    print("\nState Encoder Layers:")
    print(state_layer_stats.to_string(index=False))

    action_layer_stats = analyze_layer_by_layer(checkpoint, 'action_encoder')
    print("\nAction Encoder Layers:")
    print(action_layer_stats.to_string(index=False))

    # 重み分布
    plot_weight_distribution(
        checkpoint,
        output_dir / 'weight_distribution.png'
    )

    print(f"\n{'='*60}")
    print(f"Analysis complete! All outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
