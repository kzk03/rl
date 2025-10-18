#!/usr/bin/env python3
"""
IRL報酬関数の符号付き重み分析

絶対値ではなく、元の重み（正負の符号付き）を分析します。
正の重みは報酬を増加させ、負の重みは報酬を減少させます。

使用例:
    python scripts/analysis/analyze_signed_weights.py \
        --model importants/enhanced_irl_final_12m_6m/models/enhanced_irl_h12m_t6m_seq.pth \
        --output importants/enhanced_irl_final_12m_6m/signed_weights_analysis
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
        return None


def calculate_signed_importance(weights):
    """
    符号付き重要度を計算

    各入力特徴量について、全隠れユニットへの重みの「平均」を計算（絶対値を取らない）
    - 正の値：報酬を増加させる方向に作用
    - 負の値：報酬を減少させる方向に作用
    - 大きさ：影響力の強さ
    """
    # weights: [hidden_dim, input_dim]
    # 各入力次元について、全隠れユニットへの重みの平均（符号付き）
    signed_importance = weights.mean(axis=0)  # [input_dim]

    # 絶対値での重要度も計算（参考用）
    abs_importance = np.abs(weights).mean(axis=0)  # [input_dim]

    return signed_importance, abs_importance


def plot_signed_importance(signed_importance, abs_importance, feature_names,
                          title, output_path, top_k=30):
    """符号付き重要度をプロット"""

    # DataFrameに変換
    df = pd.DataFrame({
        'feature': feature_names,
        'signed_importance': signed_importance,
        'abs_importance': abs_importance
    })

    # 絶対値で降順ソート（影響力の大きさ順）
    df = df.sort_values('abs_importance', ascending=False)

    # Top-k を取得
    df_top = df.head(top_k)

    # プロット
    fig, ax = plt.subplots(figsize=(14, max(10, top_k * 0.4)))

    # 色分け：正は青、負は赤
    colors = ['steelblue' if x >= 0 else 'coral' for x in df_top['signed_importance']]

    bars = ax.barh(range(len(df_top)), df_top['signed_importance'], color=colors, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['feature'], fontsize=10)
    ax.invert_yaxis()

    # ゼロライン
    ax.axvline(0, color='black', linestyle='-', linewidth=2)

    ax.set_xlabel('Signed Weight (Mean across hidden units)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # 値を表示
    for i, (idx, row) in enumerate(df_top.iterrows()):
        x_pos = row['signed_importance']
        # テキストの配置を調整
        ha = 'left' if x_pos >= 0 else 'right'
        x_offset = 0.001 if x_pos >= 0 else -0.001
        ax.text(
            x_pos + x_offset, i,
            f" {row['signed_importance']:.5f}",
            va='center',
            ha=ha,
            fontsize=8
        )

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Positive (Increases reward)'),
        Patch(facecolor='coral', label='Negative (Decreases reward)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSigned importance plot saved: {output_path}")

    return fig, df


def plot_comparison_signed_abs(signed_importance, abs_importance, feature_names,
                                output_path, top_k=20):
    """符号付き vs 絶対値の比較プロット"""

    # DataFrameに変換
    df = pd.DataFrame({
        'feature': feature_names,
        'signed': signed_importance,
        'abs': abs_importance
    })

    # 絶対値で降順ソート
    df = df.sort_values('abs', ascending=False).head(top_k)

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Left: Signed importance
    colors = ['steelblue' if x >= 0 else 'coral' for x in df['signed']]
    axes[0].barh(range(len(df)), df['signed'], color=colors, edgecolor='black', linewidth=0.8)
    axes[0].set_yticks(range(len(df)))
    axes[0].set_yticklabels(df['feature'], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color='black', linestyle='-', linewidth=2)
    axes[0].set_xlabel('Signed Weight', fontsize=11, fontweight='bold')
    axes[0].set_title('Signed Importance (Direction matters)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # Right: Absolute importance
    axes[1].barh(range(len(df)), df['abs'], color='mediumpurple', edgecolor='black', linewidth=0.8)
    axes[1].set_yticks(range(len(df)))
    axes[1].set_yticklabels(df['feature'], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Absolute Weight', fontsize=11, fontweight='bold')
    axes[1].set_title('Absolute Importance (Magnitude only)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle(f'Feature Importance Comparison (Top {top_k})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: {output_path}")

    return fig


def create_detailed_report(df_state, df_action, output_dir):
    """詳細レポートを作成"""

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SIGNED WEIGHT ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Note: Positive weights increase reward, negative weights decrease reward.")
    report_lines.append("")

    # State features
    report_lines.append("="*80)
    report_lines.append("STATE FEATURES")
    report_lines.append("="*80)
    report_lines.append("")

    # Positive features
    df_positive = df_state[df_state['signed_importance'] > 0].sort_values('signed_importance', ascending=False)
    report_lines.append(f"Positive Features (Increase Reward): {len(df_positive)}")
    report_lines.append("-"*80)
    for i, (_, row) in enumerate(df_positive.head(15).iterrows(), 1):
        report_lines.append(
            f"{i:2d}. {row['feature']:30s} "
            f"Signed: +{row['signed_importance']:.6f}  "
            f"Abs: {row['abs_importance']:.6f}"
        )

    report_lines.append("")

    # Negative features
    df_negative = df_state[df_state['signed_importance'] < 0].sort_values('signed_importance', ascending=True)
    report_lines.append(f"Negative Features (Decrease Reward): {len(df_negative)}")
    report_lines.append("-"*80)
    for i, (_, row) in enumerate(df_negative.head(15).iterrows(), 1):
        report_lines.append(
            f"{i:2d}. {row['feature']:30s} "
            f"Signed: {row['signed_importance']:.6f}  "
            f"Abs: {row['abs_importance']:.6f}"
        )

    report_lines.append("")

    # Action features
    report_lines.append("="*80)
    report_lines.append("ACTION FEATURES")
    report_lines.append("="*80)
    report_lines.append("")

    # Positive features
    df_positive_action = df_action[df_action['signed_importance'] > 0].sort_values('signed_importance', ascending=False)
    report_lines.append(f"Positive Features (Increase Reward): {len(df_positive_action)}")
    report_lines.append("-"*80)
    for i, (_, row) in enumerate(df_positive_action.iterrows(), 1):
        report_lines.append(
            f"{i:2d}. {row['feature']:20s} "
            f"Signed: +{row['signed_importance']:.6f}  "
            f"Abs: {row['abs_importance']:.6f}"
        )

    report_lines.append("")

    # Negative features
    df_negative_action = df_action[df_action['signed_importance'] < 0].sort_values('signed_importance', ascending=True)
    report_lines.append(f"Negative Features (Decrease Reward): {len(df_negative_action)}")
    report_lines.append("-"*80)
    for i, (_, row) in enumerate(df_negative_action.iterrows(), 1):
        report_lines.append(
            f"{i:2d}. {row['feature']:20s} "
            f"Signed: {row['signed_importance']:.6f}  "
            f"Abs: {row['abs_importance']:.6f}"
        )

    # 統計サマリー
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("="*80)
    report_lines.append(f"State features: {len(df_state)} total")
    report_lines.append(f"  Positive: {len(df_positive)} ({len(df_positive)/len(df_state)*100:.1f}%)")
    report_lines.append(f"  Negative: {len(df_negative)} ({len(df_negative)/len(df_state)*100:.1f}%)")
    report_lines.append(f"  Zero/near-zero: {len(df_state) - len(df_positive) - len(df_negative)}")
    report_lines.append("")
    report_lines.append(f"Action features: {len(df_action)} total")
    report_lines.append(f"  Positive: {len(df_positive_action)} ({len(df_positive_action)/len(df_action)*100:.1f}%)")
    report_lines.append(f"  Negative: {len(df_negative_action)} ({len(df_negative_action)/len(df_action)*100:.1f}%)")

    # ファイルに保存
    report_text = "\n".join(report_lines)

    output_path = output_dir / 'signed_weights_report.txt'
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\nDetailed report saved: {output_path}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description='IRL報酬関数の符号付き重み分析'
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
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=30,
        help='表示する上位特徴量の数'
    )

    args = parser.parse_args()

    # モデルを読み込み
    checkpoint = load_model(args.model)

    # 出力ディレクトリの設定
    if args.output is None:
        model_path = Path(args.model)
        output_dir = model_path.parent / 'signed_weights_analysis'
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
    print("\n--- Analyzing State Encoder (Signed) ---")
    state_weights = extract_encoder_weights(checkpoint, 'state_encoder')

    if state_weights is not None:
        state_signed, state_abs = calculate_signed_importance(state_weights)

        # DataFrameを作成
        df_state = pd.DataFrame({
            'feature': state_names,
            'signed_importance': state_signed,
            'abs_importance': state_abs
        }).sort_values('abs_importance', ascending=False)

        # プロット（符号付き）
        plot_signed_importance(
            state_signed,
            state_abs,
            state_names,
            'State Feature Signed Importance for IRL Reward Function',
            output_dir / 'state_signed_importance.png',
            top_k=args.top_k
        )

        # 比較プロット
        plot_comparison_signed_abs(
            state_signed,
            state_abs,
            state_names,
            output_dir / 'state_signed_vs_abs_comparison.png',
            top_k=20
        )

        # CSV保存
        df_state.to_csv(output_dir / 'state_signed_importance.csv', index=False)
        print(f"State signed importance saved: {output_dir / 'state_signed_importance.csv'}")

    # Action encoder の重み分析
    print("\n--- Analyzing Action Encoder (Signed) ---")
    action_weights = extract_encoder_weights(checkpoint, 'action_encoder')

    if action_weights is not None:
        action_signed, action_abs = calculate_signed_importance(action_weights)

        # DataFrameを作成
        df_action = pd.DataFrame({
            'feature': action_names,
            'signed_importance': action_signed,
            'abs_importance': action_abs
        }).sort_values('abs_importance', ascending=False)

        # プロット（符号付き）
        plot_signed_importance(
            action_signed,
            action_abs,
            action_names,
            'Action Feature Signed Importance for IRL Reward Function',
            output_dir / 'action_signed_importance.png',
            top_k=min(args.top_k, action_dim)
        )

        # 比較プロット
        plot_comparison_signed_abs(
            action_signed,
            action_abs,
            action_names,
            output_dir / 'action_signed_vs_abs_comparison.png',
            top_k=action_dim
        )

        # CSV保存
        df_action.to_csv(output_dir / 'action_signed_importance.csv', index=False)
        print(f"Action signed importance saved: {output_dir / 'action_signed_importance.csv'}")

    # 詳細レポート
    if state_weights is not None and action_weights is not None:
        create_detailed_report(df_state, df_action, output_dir)

    print(f"\n{'='*80}")
    print(f"Signed weight analysis complete! All outputs saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
