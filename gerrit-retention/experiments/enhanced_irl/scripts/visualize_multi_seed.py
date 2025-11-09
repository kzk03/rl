"""
複数シード結果のヒートマップ可視化

各期間組み合わせでRFと比較
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results():
    """結果読み込み"""
    # Enhanced IRL結果（複数シード）
    irl_path = Path(__file__).parent.parent / "results" / "multi_seed_results.csv"
    irl_df = pd.read_csv(irl_path)
    
    # シードごとに平均を計算
    irl_avg = irl_df.groupby(['train_start', 'train_end', 'eval_start', 'eval_end'])['auc'].agg(['mean', 'std', 'min', 'max']).reset_index()
    irl_avg.columns = ['train_start', 'train_end', 'eval_start', 'eval_end', 'auc_mean', 'auc_std', 'auc_min', 'auc_max']
    
    # RF結果（既知の値）
    rf_results = []
    rf_matrix = [
        [0.8575, 0.8514, 0.8467, 0.8378],
        [None, 0.8760, 0.8718, 0.8688],
        [None, None, 0.8613, 0.8551],
        [None, None, None, 0.8452]
    ]
    
    for i, train_start_m in enumerate([0, 3, 6, 9]):
        train_end_m = train_start_m + 3
        for j, eval_start_m in enumerate([0, 3, 6, 9]):
            eval_end_m = eval_start_m + 3
            if eval_start_m >= train_start_m:
                rf_results.append({
                    'train_start': train_start_m,
                    'train_end': train_end_m,
                    'eval_start': eval_start_m,
                    'eval_end': eval_end_m,
                    'auc': rf_matrix[i][j]
                })
    
    rf_df = pd.DataFrame(rf_results)
    
    return irl_avg, rf_df, irl_df


def create_matrix(df, value_col='auc_mean'):
    """期間組み合わせから4x4マトリクスを作成"""
    matrix = np.full((4, 4), np.nan)
    
    for _, row in df.iterrows():
        train_idx = int(row['train_start'] // 3)
        eval_idx = int(row['eval_start'] // 3)
        matrix[train_idx, eval_idx] = row[value_col]
    
    return matrix


def plot_multi_seed_heatmaps():
    """複数シード結果のヒートマップ作成"""
    irl_avg, rf_df, irl_all = load_results()
    
    # マトリクス作成
    irl_mean_matrix = create_matrix(irl_avg, 'auc_mean')
    irl_max_matrix = create_matrix(irl_avg, 'auc_max')
    rf_matrix = create_matrix(rf_df, 'auc')
    diff_mean_matrix = irl_mean_matrix - rf_matrix
    diff_max_matrix = irl_max_matrix - rf_matrix
    
    # プロット
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    labels = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    # 1. Enhanced IRL 平均
    ax1 = axes[0, 0]
    sns.heatmap(irl_mean_matrix, annot=True, fmt='.4f', cmap='YlOrRd', 
                vmin=0.83, vmax=0.88, ax=ax1, cbar_kws={'label': 'AUC-ROC'})
    ax1.set_title('Enhanced IRL (5シード平均)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Eval期間', fontsize=12)
    ax1.set_ylabel('Train期間', fontsize=12)
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    
    # 2. Enhanced IRL 最大
    ax2 = axes[0, 1]
    sns.heatmap(irl_max_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                vmin=0.83, vmax=0.88, ax=ax2, cbar_kws={'label': 'AUC-ROC'})
    ax2.set_title('Enhanced IRL (5シード最大)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Eval期間', fontsize=12)
    ax2.set_ylabel('Train期間', fontsize=12)
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    
    # 3. Random Forest
    ax3 = axes[0, 2]
    sns.heatmap(rf_matrix, annot=True, fmt='.4f', cmap='YlGnBu',
                vmin=0.83, vmax=0.88, ax=ax3, cbar_kws={'label': 'AUC-ROC'})
    ax3.set_title('Random Forest Baseline', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Eval期間', fontsize=12)
    ax3.set_ylabel('Train期間', fontsize=12)
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels(labels)
    
    # 4. 差分（平均 - RF）
    ax4 = axes[1, 0]
    sns.heatmap(diff_mean_matrix, annot=True, fmt='.4f', cmap='RdBu_r',
                center=0, vmin=-0.02, vmax=0.02, ax=ax4, cbar_kws={'label': 'AUC差分'})
    ax4.set_title('差分 (IRL平均 - RF)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Eval期間', fontsize=12)
    ax4.set_ylabel('Train期間', fontsize=12)
    ax4.set_xticklabels(labels)
    ax4.set_yticklabels(labels)
    
    # 5. 差分（最大 - RF）
    ax5 = axes[1, 1]
    sns.heatmap(diff_max_matrix, annot=True, fmt='.4f', cmap='RdBu_r',
                center=0, vmin=-0.02, vmax=0.02, ax=ax5, cbar_kws={'label': 'AUC差分'})
    ax5.set_title('差分 (IRL最大 - RF)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Eval期間', fontsize=12)
    ax5.set_ylabel('Train期間', fontsize=12)
    ax5.set_xticklabels(labels)
    ax5.set_yticklabels(labels)
    
    # 6. 勝敗マップ
    ax6 = axes[1, 2]
    win_matrix = np.where(np.isnan(diff_max_matrix), np.nan, (diff_max_matrix > 0).astype(float))
    
    sns.heatmap(win_matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax6, cbar=False,
                cbar_kws={'label': '勝敗 (1=IRL勝, 0=RF勝)'})
    ax6.set_title('勝敗マップ (IRL最大 vs RF)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Eval期間', fontsize=12)
    ax6.set_ylabel('Train期間', fontsize=12)
    ax6.set_xticklabels(labels)
    ax6.set_yticklabels(labels)
    
    # 統計情報追加
    irl_avg_auc = irl_avg['auc_mean'].mean()
    irl_max_auc = irl_avg['auc_max'].mean()
    rf_avg_auc = rf_df['auc'].mean()
    
    fig.suptitle(
        f'Enhanced IRL (複数シード) vs RF 比較\n'
        f'IRL平均: {irl_avg_auc:.4f} | IRL最大平均: {irl_max_auc:.4f} | RF: {rf_avg_auc:.4f}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    
    # 保存
    output_dir = Path(__file__).parent.parent / "results"
    output_path = output_dir / "multi_seed_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ヒートマップを保存: {output_path}")
    
    return fig


def analyze_wins():
    """勝っている期間を分析"""
    irl_avg, rf_df, irl_all = load_results()
    
    # マージ
    merged = pd.merge(irl_avg, rf_df, on=['train_start', 'train_end', 'eval_start', 'eval_end'],
                      suffixes=('_irl', '_rf'))
    merged['diff_mean'] = merged['auc_mean'] - merged['auc']
    merged['diff_max'] = merged['auc_max'] - merged['auc']
    
    print("\n" + "=" * 70)
    print("勝敗分析")
    print("=" * 70)
    
    # 平均で勝った期間
    wins_mean = merged[merged['diff_mean'] > 0]
    print(f"\n【IRL平均がRFに勝った組み合わせ】: {len(wins_mean)}/{len(merged)}")
    if len(wins_mean) > 0:
        for _, row in wins_mean.iterrows():
            print(f"  Train={int(row['train_start'])}-{int(row['train_end'])}m → Eval={int(row['eval_start'])}-{int(row['eval_end'])}m: "
                  f"IRL平均={row['auc_mean']:.4f} vs RF={row['auc']:.4f} "
                  f"(差分: {row['diff_mean']:+.4f})")
    else:
        print("  なし")
    
    # 最大で勝った期間
    wins_max = merged[merged['diff_max'] > 0]
    print(f"\n【IRL最大がRFに勝った組み合わせ】: {len(wins_max)}/{len(merged)}")
    if len(wins_max) > 0:
        for _, row in wins_max.iterrows():
            print(f"  Train={int(row['train_start'])}-{int(row['train_end'])}m → Eval={int(row['eval_start'])}-{int(row['eval_end'])}m: "
                  f"IRL最大={row['auc_max']:.4f} vs RF={row['auc']:.4f} "
                  f"(差分: {row['diff_max']:+.4f})")
    else:
        print("  なし")
    
    # 最も善戦した組み合わせ
    print(f"\n【最も善戦した組み合わせ（IRL平均）】")
    best_mean = merged.loc[merged['diff_mean'].idxmax()]
    print(f"  Train={int(best_mean['train_start'])}-{int(best_mean['train_end'])}m → "
          f"Eval={int(best_mean['eval_start'])}-{int(best_mean['eval_end'])}m")
    print(f"  IRL平均: {best_mean['auc_mean']:.4f} ± {best_mean['auc_std']:.4f}")
    print(f"  RF: {best_mean['auc']:.4f}")
    print(f"  差分: {best_mean['diff_mean']:+.4f} ({best_mean['diff_mean']*100:+.2f}%)")
    
    # 最も苦戦した組み合わせ
    print(f"\n【最も苦戦した組み合わせ（IRL平均）】")
    worst_mean = merged.loc[merged['diff_mean'].idxmin()]
    print(f"  Train={int(worst_mean['train_start'])}-{int(worst_mean['train_end'])}m → "
          f"Eval={int(worst_mean['eval_start'])}-{int(worst_mean['eval_end'])}m")
    print(f"  IRL平均: {worst_mean['auc_mean']:.4f} ± {worst_mean['auc_std']:.4f}")
    print(f"  RF: {worst_mean['auc']:.4f}")
    print(f"  差分: {worst_mean['diff_mean']:+.4f} ({worst_mean['diff_mean']*100:+.2f}%)")
    
    print("=" * 70 + "\n")


def main():
    """メイン実行"""
    print("複数シード結果の可視化\n")
    
    # 勝敗分析
    analyze_wins()
    
    # ヒートマップ作成
    print("ヒートマップ作成中...")
    plot_multi_seed_heatmaps()
    
    print("\n✅ 可視化が完了しました！")
    print(f"結果: experiments/enhanced_irl/results/multi_seed_heatmap.png")


if __name__ == "__main__":
    main()
