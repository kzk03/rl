#!/usr/bin/env python3
"""
IRL全層の重み分析

学習済みIRLモデルのすべてのLinear層の重みを詳細に分析します。

使用例:
    python scripts/analysis/analyze_all_weights.py \
        --model importants/irl_matrix_2023q1/models/irl_h12m_t12m_fixed_seq.pth \
        --output importants/irl_matrix_2023q1/weights_analysis
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from matplotlib.gridspec import GridSpec


def load_model(model_path: str):
    """学習済みモデルを読み込み"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"Model loaded: {model_path}")
    print(f"Config: {checkpoint.get('config', {})}")
    return checkpoint


def extract_all_weights(checkpoint):
    """すべてのLinear層の重みを抽出"""
    state_dict = checkpoint['model_state_dict']

    weights_dict = {}

    # Linear層の重みのみを抽出（bias含む）
    for key, value in state_dict.items():
        if 'weight' in key or 'bias' in key:
            # LSTM以外
            if 'lstm' not in key:
                weights_dict[key] = value.numpy()

    return weights_dict


def categorize_layers(weights_dict):
    """層をカテゴリ分け"""
    categories = {
        'state_encoder': {},
        'action_encoder': {},
        'reward_head': {},
        'continuation_head': {}
    }

    for key, value in weights_dict.items():
        if 'state_encoder' in key:
            categories['state_encoder'][key] = value
        elif 'action_encoder' in key:
            categories['action_encoder'][key] = value
        elif 'reward_head' in key:
            categories['reward_head'][key] = value
        elif 'continuation_head' in key:
            categories['continuation_head'][key] = value

    return categories


def analyze_weight_matrix(weight, layer_name):
    """重み行列の詳細分析"""
    stats = {
        'layer': layer_name,
        'shape': weight.shape,
        'n_params': weight.size,
        'mean': np.mean(weight),
        'std': np.std(weight),
        'min': np.min(weight),
        'max': np.max(weight),
        'abs_mean': np.mean(np.abs(weight)),
        'abs_max': np.max(np.abs(weight)),
        'l2_norm': np.linalg.norm(weight),
        'sparsity': np.sum(np.abs(weight) < 0.01) / weight.size * 100,  # % of near-zero
    }

    return stats


def plot_all_weights_heatmap(categories, output_dir):
    """すべての重み行列をヒートマップで表示"""

    for category_name, layers in categories.items():
        if not layers:
            continue

        # weight層のみを抽出
        weight_layers = {k: v for k, v in layers.items() if 'weight' in k}

        if not weight_layers:
            continue

        n_layers = len(weight_layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 5))

        if n_layers == 1:
            axes = [axes]

        for idx, (layer_name, weight) in enumerate(weight_layers.items()):
            # ヒートマップ
            im = axes[idx].imshow(weight, aspect='auto', cmap='RdBu_r',
                                 vmin=-np.abs(weight).max(),
                                 vmax=np.abs(weight).max())

            axes[idx].set_title(f'{layer_name}\n{weight.shape}', fontsize=11)
            axes[idx].set_xlabel('Input Dim', fontsize=10)
            axes[idx].set_ylabel('Output Dim', fontsize=10)

            # カラーバー
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        plt.suptitle(f'{category_name.replace("_", " ").title()} - Weight Matrices',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f'{category_name}_weights_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def plot_weight_distributions(categories, output_dir):
    """各カテゴリの重み分布をヒストグラムで表示"""

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)

    category_list = ['state_encoder', 'action_encoder', 'reward_head', 'continuation_head']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']

    for (category, pos, color) in zip(category_list, positions, colors):
        if category not in categories or not categories[category]:
            continue

        ax = fig.add_subplot(gs[pos[0], pos[1]])

        # すべての重みを収集
        all_weights = []
        for key, value in categories[category].items():
            if 'weight' in key:
                all_weights.extend(value.flatten())

        if not all_weights:
            continue

        # ヒストグラム
        ax.hist(all_weights, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)

        ax.set_xlabel('Weight Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{category.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        # 統計情報
        mean_val = np.mean(all_weights)
        std_val = np.std(all_weights)
        ax.text(
            0.98, 0.98,
            f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nN: {len(all_weights)}',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    plt.suptitle('Weight Distributions Across All Layers',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'all_weights_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_layer_statistics(stats_df, output_dir):
    """層ごとの統計をプロット"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = ['mean', 'std', 'abs_mean', 'abs_max', 'l2_norm', 'sparsity']
    titles = ['Mean', 'Std Dev', 'Mean |W|', 'Max |W|', 'L2 Norm', 'Sparsity (%)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        # カテゴリごとに色分け
        colors = []
        for layer in stats_df['layer']:
            if 'state_encoder' in layer:
                colors.append('steelblue')
            elif 'action_encoder' in layer:
                colors.append('coral')
            elif 'reward_head' in layer:
                colors.append('mediumseagreen')
            elif 'continuation_head' in layer:
                colors.append('mediumpurple')
            else:
                colors.append('gray')

        bars = ax.barh(range(len(stats_df)), stats_df[metric], color=colors)
        ax.set_yticks(range(len(stats_df)))
        ax.set_yticklabels(stats_df['layer'], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(title, fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 値を表示
        for i, value in enumerate(stats_df[metric]):
            ax.text(value, i, f' {value:.3f}', va='center', fontsize=7)

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='State Encoder'),
        Patch(facecolor='coral', label='Action Encoder'),
        Patch(facecolor='mediumseagreen', label='Reward Head'),
        Patch(facecolor='mediumpurple', label='Continuation Head')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.suptitle('Layer-wise Weight Statistics',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'layer_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gradient_flow_proxy(categories, output_dir):
    """重みの大きさから勾配フローの代理指標をプロット"""

    layer_order = []
    norms = []

    # レイヤーを順序付け
    order = [
        'state_encoder.0.weight',
        'state_encoder.2.weight',
        'action_encoder.0.weight',
        'action_encoder.2.weight',
        'reward_head.0.weight',
        'reward_head.2.weight',
        'continuation_head.0.weight',
        'continuation_head.2.weight'
    ]

    all_layers = {}
    for cat_name, cat_layers in categories.items():
        all_layers.update(cat_layers)

    for layer_name in order:
        if layer_name in all_layers:
            weight = all_layers[layer_name]
            layer_order.append(layer_name.replace('.weight', ''))
            norms.append(np.linalg.norm(weight))

    # プロット
    fig, ax = plt.subplots(figsize=(12, 6))

    colors_map = {
        'state_encoder': 'steelblue',
        'action_encoder': 'coral',
        'reward_head': 'mediumseagreen',
        'continuation_head': 'mediumpurple'
    }

    colors = []
    for layer in layer_order:
        for key, color in colors_map.items():
            if key in layer:
                colors.append(color)
                break

    bars = ax.bar(range(len(layer_order)), norms, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels(layer_order, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('L2 Norm', fontsize=12, fontweight='bold')
    ax.set_title('Weight Magnitude Across Layers (Gradient Flow Proxy)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 値を表示
    for i, (bar, norm) in enumerate(zip(bars, norms)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{norm:.1f}',
               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = output_dir / 'gradient_flow_proxy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_report(stats_df, output_dir):
    """サマリーレポートを作成"""

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("IRL MODEL WEIGHT ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")

    # カテゴリごとに集計
    categories = {
        'State Encoder': [],
        'Action Encoder': [],
        'Reward Head': [],
        'Continuation Head': []
    }

    for _, row in stats_df.iterrows():
        layer = row['layer']
        if 'state_encoder' in layer:
            categories['State Encoder'].append(row)
        elif 'action_encoder' in layer:
            categories['Action Encoder'].append(row)
        elif 'reward_head' in layer:
            categories['Reward Head'].append(row)
        elif 'continuation_head' in layer:
            categories['Continuation Head'].append(row)

    for cat_name, rows in categories.items():
        if not rows:
            continue

        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"{cat_name}")
        report_lines.append(f"{'='*80}")

        for row in rows:
            report_lines.append(f"\nLayer: {row['layer']}")
            report_lines.append(f"  Shape: {row['shape']}")
            report_lines.append(f"  Parameters: {row['n_params']:,}")
            report_lines.append(f"  Mean: {row['mean']:.6f}")
            report_lines.append(f"  Std: {row['std']:.6f}")
            report_lines.append(f"  Range: [{row['min']:.6f}, {row['max']:.6f}]")
            report_lines.append(f"  Mean |W|: {row['abs_mean']:.6f}")
            report_lines.append(f"  Max |W|: {row['abs_max']:.6f}")
            report_lines.append(f"  L2 Norm: {row['l2_norm']:.6f}")
            report_lines.append(f"  Sparsity: {row['sparsity']:.2f}%")

    # 全体統計
    report_lines.append(f"\n{'='*80}")
    report_lines.append("OVERALL STATISTICS")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"Total layers: {len(stats_df)}")
    report_lines.append(f"Total parameters: {stats_df['n_params'].sum():,}")
    report_lines.append(f"Average sparsity: {stats_df['sparsity'].mean():.2f}%")
    report_lines.append(f"Average L2 norm: {stats_df['l2_norm'].mean():.2f}")

    # ファイルに保存
    report_text = "\n".join(report_lines)

    output_path = output_dir / 'weight_analysis_report.txt'
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\nSaved: {output_path}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description='IRL全層の重み分析'
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

    args = parser.parse_args()

    # モデルを読み込み
    checkpoint = load_model(args.model)

    # 出力ディレクトリの設定
    if args.output is None:
        model_path = Path(args.model)
        output_dir = model_path.parent / 'weights_analysis'
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # すべての重みを抽出
    print("\n--- Extracting all weights ---")
    weights_dict = extract_all_weights(checkpoint)
    print(f"Found {len(weights_dict)} weight/bias tensors")

    # カテゴリ分け
    categories = categorize_layers(weights_dict)

    for cat_name, layers in categories.items():
        print(f"  {cat_name}: {len(layers)} layers")

    # 統計分析
    print("\n--- Analyzing weight statistics ---")
    stats_list = []
    for key, weight in weights_dict.items():
        if 'weight' in key:  # weight層のみ（biasは除く）
            stats = analyze_weight_matrix(weight, key)
            stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    # CSV保存
    csv_path = output_dir / 'layer_statistics.csv'
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # 可視化
    print("\n--- Generating visualizations ---")

    # 1. ヒートマップ
    print("Plotting weight heatmaps...")
    plot_all_weights_heatmap(categories, output_dir)

    # 2. 分布
    print("Plotting weight distributions...")
    plot_weight_distributions(categories, output_dir)

    # 3. 統計
    print("Plotting layer statistics...")
    plot_layer_statistics(stats_df, output_dir)

    # 4. 勾配フロー代理指標
    print("Plotting gradient flow proxy...")
    plot_gradient_flow_proxy(categories, output_dir)

    # 5. サマリーレポート
    print("\n--- Creating summary report ---")
    create_summary_report(stats_df, output_dir)

    print(f"\n{'='*80}")
    print(f"Analysis complete! All outputs saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
