#!/usr/bin/env python3
"""
クロス窓評価実験の結果をマトリクスで可視化
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(base_dir: Path):
    """結果を読み込む"""
    results = []
    
    for model_dir in sorted(base_dir.glob("train_*_eval_*")):
        metrics_file = model_dir / "metrics.json"
        if not metrics_file.exists():
            print(f"警告: {metrics_file} が見つかりません")
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # モデル名から訓練ラベルと評価期間を抽出
        model_name = model_dir.name
        # 例: train_0-1m_eval_0-3m
        parts = model_name.split('_')
        train_window = parts[1]  # 0-1m
        eval_window = parts[3]   # 0-3m
        
        results.append({
            'train_window': train_window,
            'eval_window': eval_window,
            'auc_roc': metrics.get('auc_roc', 0),
            'auc_pr': metrics.get('auc_pr', 0),
            'f1': metrics.get('f1', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'optimal_threshold': metrics.get('optimal_threshold', 0),
            'sample_count': metrics.get('sample_count', 0),
            'continuation_rate': metrics.get('continuation_rate', 0),
        })
    
    return pd.DataFrame(results)


def create_matrix(df: pd.DataFrame, metric: str):
    """指定されたメトリクスのマトリクスを作成"""
    # ピボットテーブルを作成
    matrix = df.pivot(index='train_window', columns='eval_window', values=metric)
    
    # 訓練ラベルの順序（累積）
    train_order = ['0-1m', '0-3m', '0-6m', '0-9m', '0-12m']
    # 評価期間の順序（スライディング）
    eval_order = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    existing_train = [o for o in train_order if o in matrix.index]
    existing_eval = [o for o in eval_order if o in matrix.columns]
    
    matrix = matrix.reindex(index=existing_train, columns=existing_eval)
    
    return matrix


def create_heatmap_visualizations(df: pd.DataFrame, output_dir: Path):
    """ヒートマップを作成"""
    # メトリクスのリスト
    metrics = {
        'auc_roc': 'AUC-ROC',
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall',
    }
    
    # スタイル設定
    sns.set_style("white")
    
    # 各メトリクスのヒートマップを作成
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('クロス窓評価実験 結果マトリクス\n訓練ラベル(累積) × 評価期間(スライディング)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    axes_flat = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes_flat[idx]
        
        # マトリクスを作成
        matrix = create_matrix(df, metric_key)
        
        # ヒートマップを描画
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            center=0.5 if metric_key == 'auc_roc' else None,
            cbar_kws={'label': metric_name},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        
        ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('評価期間（スライディング）', fontsize=12, fontweight='bold')
        ax.set_ylabel('訓練ラベル（累積）', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_file = output_dir / "cross_window_evaluation_heatmaps.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"ヒートマップを保存: {output_file}")
    
    plt.close()


def create_summary_table(df: pd.DataFrame):
    """サマリーテーブルを作成"""
    print("\n" + "=" * 100)
    print("クロス窓評価実験 結果サマリー")
    print("=" * 100)
    print()
    
    # 訓練ラベルごとにグループ化
    for train_window in sorted(df['train_window'].unique()):
        train_df = df[df['train_window'] == train_window].sort_values('eval_window')
        
        print(f"【訓練ラベル: {train_window}（累積）】")
        print(train_df[['eval_window', 'auc_roc', 'f1', 'precision', 'recall', 'continuation_rate']].to_string(index=False))
        print()
    
    # 最良の組み合わせ
    print("【最良の組み合わせ】")
    best_auc = df.loc[df['auc_roc'].idxmax()]
    best_f1 = df.loc[df['f1'].idxmax()]
    print(f"  最高AUC-ROC: 訓練={best_auc['train_window']}, 評価={best_auc['eval_window']} ({best_auc['auc_roc']:.3f})")
    print(f"  最高F1:      訓練={best_f1['train_window']}, 評価={best_f1['eval_window']} ({best_f1['f1']:.3f})")
    print()


def create_line_plots(df: pd.DataFrame, output_dir: Path):
    """各訓練ラベルごとの性能推移をプロット"""
    metrics = ['auc_roc', 'f1', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('各訓練ラベル(累積)での評価期間(スライディング)ごとの性能推移', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    # 評価期間の順序
    eval_order = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        
        for train_window in sorted(df['train_window'].unique()):
            train_df = df[df['train_window'] == train_window]
            
            # 評価期間順にソート
            train_df = train_df.set_index('eval_window').reindex(eval_order).reset_index()
            
            ax.plot(train_df['eval_window'], train_df[metric], 
                   marker='o', linewidth=2, markersize=8, 
                   label=f'訓練: {train_window}')
        
        ax.set_title(metric.upper().replace('_', '-'), fontsize=12, fontweight='bold')
        ax.set_xlabel('評価期間（スライディング）', fontsize=11)
        ax.set_ylabel(metric.upper().replace('_', '-'), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # 0.5のライン（AUC-ROCの場合）
        if metric == 'auc_roc':
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    
    plt.tight_layout()
    
    output_file = output_dir / "cross_window_evaluation_lines.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"ラインプロットを保存: {output_file}")
    
    plt.close()


def main():
    base_dir = Path("outputs/cross_window_evaluation")
    
    if not base_dir.exists():
        print(f"エラー: {base_dir} が見つかりません")
        sys.exit(1)
    
    # 結果を読み込み
    df = load_results(base_dir)
    
    if df.empty:
        print("エラー: 結果が見つかりません")
        sys.exit(1)
    
    # サマリーテーブルを作成
    create_summary_table(df)
    
    # ヒートマップを作成
    create_heatmap_visualizations(df, base_dir)
    
    # ラインプロットを作成
    create_line_plots(df, base_dir)
    
    # CSVに保存
    output_csv = base_dir / "cross_evaluation_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"結果をCSVに保存: {output_csv}")
    print()
    
    print("=" * 100)
    print("可視化完了！")
    print("=" * 100)
    print(f"ヒートマップ: {base_dir}/cross_window_evaluation_heatmaps.png")
    print(f"ラインプロット: {base_dir}/cross_window_evaluation_lines.png")
    print(f"CSV: {base_dir}/cross_evaluation_results.csv")


if __name__ == "__main__":
    main()

