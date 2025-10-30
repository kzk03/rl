#!/usr/bin/env python3
"""
全訓練期間の特徴量重要度を1つのPNGにまとめる
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 特徴量名の定義（10次元版）
STATE_FEATURE_NAMES = [
    "経験日数", "総コミット数", "総レビュー数",
    "最近の活動頻度", "平均活動間隔", "活動トレンド",
    "協力スコア", "コード品質スコア",
    "最近の受諾率", "レビュー負荷"
]

ACTION_FEATURE_NAMES = ["強度（ファイル数）", "協力度", "応答速度", "レビュー規模（行数）"]

PERIODS = ['0-3m', '3-6m', '6-9m', '9-12m']
BASE_DIR = Path("outputs/review_acceptance_cross_eval_nova")


def load_importance_data():
    """全期間の特徴量重要度を読み込み"""
    data = {}
    
    for period in PERIODS:
        importance_path = BASE_DIR / f"train_{period}" / "feature_importance" / "gradient_importance.json"
        
        if not importance_path.exists():
            logger.warning(f"{period} のデータが存在しません")
            continue
        
        with open(importance_path) as f:
            importance = json.load(f)
        
        data[period] = importance
    
    return data


def create_state_heatmap(data, output_path: Path):
    """状態特徴の重要度をヒートマップで可視化"""
    
    # データを行列に変換
    state_matrix = np.zeros((len(STATE_FEATURE_NAMES), len(PERIODS)))
    
    for i, period in enumerate(PERIODS):
        if period not in data:
            continue
        
        for j, feature_name in enumerate(STATE_FEATURE_NAMES):
            state_matrix[j, i] = data[period]['state_importance'][feature_name]
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 状態特徴のヒートマップ
    im = ax.imshow(state_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.008, vmax=0.008)
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels(PERIODS, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(STATE_FEATURE_NAMES)))
    ax.set_yticklabels(STATE_FEATURE_NAMES, fontsize=13)
    ax.set_xlabel('訓練期間', fontsize=16, fontweight='bold')
    ax.set_ylabel('状態特徴量', fontsize=16, fontweight='bold')
    ax.set_title('状態特徴量の重要度（全訓練期間）', fontsize=18, fontweight='bold', pad=20)
    
    # 数値を表示
    for i in range(len(STATE_FEATURE_NAMES)):
        for j in range(len(PERIODS)):
            value = state_matrix[i, j]
            color = 'white' if abs(value) > 0.003 else 'black'
            ax.text(j, i, f'{value:.4f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='重要度', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('重要度', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"保存: {output_path}")
    plt.close()


def create_action_heatmap(data, output_path: Path):
    """行動特徴の重要度をヒートマップで可視化"""
    
    # データを行列に変換
    action_matrix = np.zeros((len(ACTION_FEATURE_NAMES), len(PERIODS)))
    
    for i, period in enumerate(PERIODS):
        if period not in data:
            continue
        
        for j, feature_name in enumerate(ACTION_FEATURE_NAMES):
            action_matrix[j, i] = data[period]['action_importance'][feature_name]
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 行動特徴のヒートマップ
    im = ax.imshow(action_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.008, vmax=0.008)
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels(PERIODS, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(ACTION_FEATURE_NAMES)))
    ax.set_yticklabels(ACTION_FEATURE_NAMES, fontsize=13)
    ax.set_xlabel('訓練期間', fontsize=16, fontweight='bold')
    ax.set_ylabel('行動特徴量', fontsize=16, fontweight='bold')
    ax.set_title('行動特徴量の重要度（全訓練期間）', fontsize=18, fontweight='bold', pad=20)
    
    # 数値を表示
    for i in range(len(ACTION_FEATURE_NAMES)):
        for j in range(len(PERIODS)):
            value = action_matrix[i, j]
            color = 'white' if abs(value) > 0.003 else 'black'
            ax.text(j, i, f'{value:.4f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='重要度', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('重要度', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"保存: {output_path}")
    plt.close()


def create_bar_comparison(data, output_path: Path):
    """全モデルの特徴量重要度を棒グラフで比較"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx, period in enumerate(PERIODS):
        ax = axes[idx]
        
        if period not in data:
            ax.text(0.5, 0.5, f'{period}\nデータなし', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue
        
        # 状態特徴と行動特徴を結合
        all_features = list(STATE_FEATURE_NAMES) + list(ACTION_FEATURE_NAMES)
        all_values = []
        
        for feature in STATE_FEATURE_NAMES:
            all_values.append(data[period]['state_importance'][feature])
        
        for feature in ACTION_FEATURE_NAMES:
            all_values.append(data[period]['action_importance'][feature])
        
        # ソート（重要度の絶対値で）
        sorted_indices = sorted(range(len(all_values)), 
                              key=lambda i: abs(all_values[i]), reverse=True)
        sorted_features = [all_features[i] for i in sorted_indices]
        sorted_values = [all_values[i] for i in sorted_indices]
        
        # 色分け（正=青、負=赤）
        colors = ['#2196F3' if v >= 0 else '#F44336' for v in sorted_values]
        
        # 棒グラフ
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.set_xlabel('重要度', fontsize=12, fontweight='bold')
        ax.set_title(f'train_{period}', fontsize=14, fontweight='bold', pad=10)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 数値を表示（上位3つのみ）
        for i in range(min(3, len(sorted_values))):
            value = sorted_values[i]
            x_pos = value + (0.0005 if value >= 0 else -0.0005)
            ha = 'left' if value >= 0 else 'right'
            ax.text(x_pos, i, f'{value:.4f}', 
                   va='center', ha=ha, fontsize=9, fontweight='bold')
    
    plt.suptitle('全訓練期間の特徴量重要度（上位から表示）', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"保存: {output_path}")
    plt.close()


def create_trend_plot(data, output_path: Path):
    """特徴量の重要度の時系列変化をプロット"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 状態特徴のトレンド
    for feature_name in STATE_FEATURE_NAMES:
        values = []
        for period in PERIODS:
            if period in data:
                values.append(data[period]['state_importance'][feature_name])
            else:
                values.append(np.nan)
        
        ax1.plot(PERIODS, values, marker='o', label=feature_name, linewidth=2, markersize=8)
    
    ax1.set_xlabel('訓練期間', fontsize=14, fontweight='bold')
    ax1.set_ylabel('重要度', fontsize=14, fontweight='bold')
    ax1.set_title('状態特徴量の重要度トレンド', fontsize=16, fontweight='bold', pad=15)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 行動特徴のトレンド
    for feature_name in ACTION_FEATURE_NAMES:
        values = []
        for period in PERIODS:
            if period in data:
                values.append(data[period]['action_importance'][feature_name])
            else:
                values.append(np.nan)
        
        ax2.plot(PERIODS, values, marker='s', label=feature_name, linewidth=2, markersize=8)
    
    ax2.set_xlabel('訓練期間', fontsize=14, fontweight='bold')
    ax2.set_ylabel('重要度', fontsize=14, fontweight='bold')
    ax2.set_title('行動特徴量の重要度トレンド', fontsize=16, fontweight='bold', pad=15)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    
    plt.suptitle('特徴量重要度の時系列変化', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"保存: {output_path}")
    plt.close()


def main():
    logger.info("=" * 80)
    logger.info("全モデルの特徴量重要度を統合可視化")
    logger.info("=" * 80)
    
    # データ読み込み
    data = load_importance_data()
    
    if not data:
        logger.error("データが見つかりません")
        return
    
    logger.info(f"読み込んだ期間: {list(data.keys())}")
    
    # 出力ディレクトリ
    output_dir = BASE_DIR / "combined_feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 状態特徴のヒートマップ
    logger.info("\n1. 状態特徴のヒートマップを作成中...")
    create_state_heatmap(data, output_dir / "heatmap_state_features.png")
    
    # 2. 行動特徴のヒートマップ
    logger.info("\n2. 行動特徴のヒートマップを作成中...")
    create_action_heatmap(data, output_dir / "heatmap_action_features.png")
    
    # 3. 棒グラフ比較
    logger.info("\n3. 棒グラフ比較を作成中...")
    create_bar_comparison(data, output_dir / "bar_comparison_all_models.png")
    
    # 4. トレンドプロット
    logger.info("\n4. トレンドプロットを作成中...")
    create_trend_plot(data, output_dir / "trend_all_models.png")
    
    logger.info("\n" + "=" * 80)
    logger.info("統合可視化が完了しました！")
    logger.info("=" * 80)
    logger.info(f"\n出力ディレクトリ: {output_dir}")
    logger.info("  - heatmap_state_features.png (状態特徴のヒートマップ)")
    logger.info("  - heatmap_action_features.png (行動特徴のヒートマップ)")
    logger.info("  - bar_comparison_all_models.png (棒グラフ比較)")
    logger.info("  - trend_all_models.png (時系列トレンド)")


if __name__ == "__main__":
    main()

