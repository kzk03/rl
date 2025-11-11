#!/usr/bin/env python3
"""
IRL vs Random Forest v3 結果比較とヒートマップ作成
RF v3: 完全なIRL重み付けスキーム（sample_weights + class balancing）
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_irl_metrics(irl_dir: Path, train_period: str, eval_period: str) -> dict:
    """IRLメトリクスをロード"""
    metrics_file = irl_dir / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def load_rf_metrics(rf_dir: Path, train_period: str, eval_period: str) -> dict:
    """RFメトリクスをロード"""
    metrics_file = rf_dir / f"train_{train_period}_eval_{eval_period}" / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def create_heatmap(data_matrix, title, output_path, vmin=0.0, vmax=1.0, cmap='RdYlGn'):
    """ヒートマップを作成（y軸を下から0-3m, 3-6m, 6-9m, 9-12mにする）"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # データを反転（下から0-3m, 3-6m, 6-9m, 9-12m）
    data_reversed = data_matrix[::-1]
    
    sns.heatmap(
        data_reversed,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=['0-3m', '3-6m', '6-9m', '9-12m'],
        yticklabels=['9-12m', '6-9m', '3-6m', '0-3m'],  # 反転したラベル
        cbar_kws={'label': 'AUC-ROC'},
        ax=ax
    )
    
    ax.set_xlabel('Evaluation Period', fontsize=12)
    ax.set_ylabel('Training Period', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ヒートマップ保存: {output_path}")


def main():
    base_dir = Path("experiments/nova_review_acceptance")
    irl_dir = base_dir / "results_enhanced_irl_cv"
    rf_dir = base_dir / "results_rf_cv_v3"
    
    train_periods = ["0-3m", "3-6m", "6-9m", "9-12m"]
    eval_periods = ["0-3m", "3-6m", "6-9m", "9-12m"]
    
    results = []
    irl_matrix = np.zeros((4, 4))
    rf_matrix = np.zeros((4, 4))
    diff_matrix = np.zeros((4, 4))
    
    for i, train_period in enumerate(train_periods):
        for j, eval_period in enumerate(eval_periods):
            irl_metrics = load_irl_metrics(irl_dir, train_period, eval_period)
            rf_metrics = load_rf_metrics(rf_dir, train_period, eval_period)
            
            if irl_metrics and rf_metrics:
                irl_auc = irl_metrics.get('auc_roc', 0.0)
                rf_auc = rf_metrics.get('auc_roc', 0.0)
                
                irl_matrix[i, j] = irl_auc
                rf_matrix[i, j] = rf_auc
                diff_matrix[i, j] = rf_auc - irl_auc
                
                result = {
                    'train': train_period,
                    'eval': eval_period,
                    'IRL_AUC_ROC': irl_auc,
                    'IRL_AUC_PR': irl_metrics.get('auc_pr', 0.0),
                    'IRL_F1': irl_metrics.get('f1', 0.0),
                    'IRL_threshold': irl_metrics.get('threshold', 0.0),
                    'RF_AUC_ROC': rf_auc,
                    'RF_AUC_PR': rf_metrics.get('auc_pr', 0.0),
                    'RF_F1': rf_metrics.get('f1', 0.0),
                    'RF_threshold': rf_metrics.get('threshold', 0.0),
                    'num_samples': rf_metrics.get('num_samples', 0),
                    'num_positive': rf_metrics.get('num_positive', 0),
                    'diff_AUC_ROC': rf_auc - irl_auc,
                }
                results.append(result)
    
    df = pd.DataFrame(results)
    
    print("=" * 100)
    print("IRL vs Random Forest v3 比較")
    print("RF v3: 完全なIRL重み付けスキーム（sample_weights + class balancing）")
    print("=" * 100)
    print()
    
    # 統計サマリー
    print("統計サマリー:")
    print("-" * 100)
    print(f"IRL  - AUC-ROC: {df['IRL_AUC_ROC'].mean():.4f} (±{df['IRL_AUC_ROC'].std():.4f}) [{df['IRL_AUC_ROC'].min():.4f} - {df['IRL_AUC_ROC'].max():.4f}]")
    print(f"RF v3 - AUC-ROC: {df['RF_AUC_ROC'].mean():.4f} (±{df['RF_AUC_ROC'].std():.4f}) [{df['RF_AUC_ROC'].min():.4f} - {df['RF_AUC_ROC'].max():.4f}]")
    print(f"差分 (RF - IRL): {df['diff_AUC_ROC'].mean():+.4f} (±{df['diff_AUC_ROC'].std():.4f})")
    print()
    
    # 詳細結果
    print("\n詳細結果:")
    print("-" * 100)
    print(f"{'Train':>8} | {'Eval':>8} | {'IRL AUC':>8} | {'RF AUC':>8} | {'Diff':>8} | {'Samples':>8}")
    print("-" * 100)
    for _, row in df.iterrows():
        print(f"{row['train']:>8} | {row['eval']:>8} | {row['IRL_AUC_ROC']:8.4f} | {row['RF_AUC_ROC']:8.4f} | {row['diff_AUC_ROC']:+8.4f} | {row['num_samples']:>8}")
    print()
    
    # RF優位性
    rf_wins = (df['diff_AUC_ROC'] > 0).sum()
    total = len(df)
    print(f"RFが優位なパターン数: {rf_wins}/{total} ({rf_wins/total*100:.1f}%)")
    print()
    
    # CSV保存
    output_csv = base_dir / "irl_vs_rf_v3_comparison.csv"
    df.to_csv(output_csv, index=False)
    print(f"CSV保存: {output_csv}")
    print()
    
    # ヒートマップ作成
    print("ヒートマップ作成中...")
    
    # 1. IRL AUC-ROC
    create_heatmap(
        irl_matrix,
        'IRL AUC-ROC (K-Fold CV)',
        base_dir / 'heatmap_irl_auc_roc.png',
        vmin=0.4,
        vmax=0.7,
        cmap='YlOrRd'
    )
    
    # 2. RF v3 AUC-ROC
    create_heatmap(
        rf_matrix,
        'Random Forest v3 AUC-ROC (Complete IRL Weighting)',
        base_dir / 'heatmap_rf_v3_auc_roc.png',
        vmin=0.4,
        vmax=1.0,
        cmap='RdYlGn'
    )
    
    # 3. 差分 (RF - IRL)
    create_heatmap(
        diff_matrix,
        'Performance Difference (RF v3 - IRL)',
        base_dir / 'heatmap_diff_rf_v3_minus_irl.png',
        vmin=0.0,
        vmax=0.5,
        cmap='RdYlGn'
    )
    
    print()
    print("=" * 100)
    print("完了！")
    print("=" * 100)


if __name__ == '__main__':
    main()
