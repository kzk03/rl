#!/usr/bin/env python3
"""
将来窓実験の結果を比較するスクリプト
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# スタイル設定
sns.set_style("whitegrid")
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(result_paths: List[Path]) -> pd.DataFrame:
    """結果ファイルを読み込む"""
    data = []
    
    for path in result_paths:
        with open(path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        windows = result['windows']
        metrics = result['metrics']
        
        data.append({
            'future_window': f"{windows['future_start_months']}-{windows['future_end_months']}m",
            'future_start': windows['future_start_months'],
            'future_end': windows['future_end_months'],
            'history_months': windows['history_months'],
            'auc_roc': metrics['auc_roc'],
            'auc_pr': metrics['auc_pr'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'positive_rate': metrics['positive_rate'],
            'test_samples': metrics['test_samples'],
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('future_start')
    
    return df


def create_comparison_plot(df: pd.DataFrame, output_path: Path):
    """比較プロットを作成"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('将来窓による予測性能の比較', fontsize=16, y=0.995)
    
    # AUC-ROC
    ax = axes[0, 0]
    ax.bar(df['future_window'], df['auc_roc'], color='steelblue', alpha=0.7)
    ax.set_title('AUC-ROC', fontsize=14)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # AUC-PR
    ax = axes[0, 1]
    ax.bar(df['future_window'], df['auc_pr'], color='green', alpha=0.7)
    ax.set_title('AUC-PR', fontsize=14)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # F1 Score
    ax = axes[1, 0]
    ax.bar(df['future_window'], df['f1'], color='orange', alpha=0.7)
    ax.set_title('F1 Score', fontsize=14)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Future Window', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Precision / Recall
    ax = axes[1, 1]
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df['precision'], width, label='Precision', alpha=0.7)
    ax.bar([i + width/2 for i in x], df['recall'], width, label='Recall', alpha=0.7)
    ax.set_title('Precision & Recall', fontsize=14)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Future Window', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['future_window'])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"プロットを保存: {output_path}")


def print_comparison_table(df: pd.DataFrame):
    """比較表を表示"""
    print("\n" + "=" * 80)
    print("将来窓による予測性能の比較")
    print("=" * 80)
    print(f"\n{'将来窓':<12} {'AUC-ROC':>10} {'AUC-PR':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'正例率':>10}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['future_window']:<12} "
              f"{row['auc_roc']:>10.3f} "
              f"{row['auc_pr']:>10.3f} "
              f"{row['f1']:>10.3f} "
              f"{row['precision']:>10.3f} "
              f"{row['recall']:>10.3f} "
              f"{row['positive_rate']:>9.1%}")
    
    print("-" * 80)
    
    # 最良の結果
    best_auc_roc = df.loc[df['auc_roc'].idxmax()]
    best_f1 = df.loc[df['f1'].idxmax()]
    
    print(f"\n最良 AUC-ROC: {best_auc_roc['future_window']} ({best_auc_roc['auc_roc']:.3f})")
    print(f"最良 F1:      {best_f1['future_window']} ({best_f1['f1']:.3f})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='将来窓実験の結果比較')
    parser.add_argument(
        '--results',
        type=Path,
        nargs='+',
        required=True,
        help='evaluation_results.json ファイルのパス（複数可）'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/future_window_comparison.png'),
        help='出力プロットのパス'
    )
    
    args = parser.parse_args()
    
    # 結果を読み込み
    df = load_results(args.results)
    
    # 比較表を表示
    print_comparison_table(df)
    
    # プロットを作成
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_comparison_plot(df, args.output)
    
    # CSVとして保存
    csv_path = args.output.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"結果CSVを保存: {csv_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

