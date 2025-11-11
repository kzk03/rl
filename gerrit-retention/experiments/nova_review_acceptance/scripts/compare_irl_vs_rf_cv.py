#!/usr/bin/env python3
"""
IRL vs Random Forest 結果比較
K-Fold CV版の両手法を比較
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_metrics(result_dir: Path, pattern_name: str) -> dict:
    """メトリクスをロード"""
    metrics_file = result_dir / pattern_name / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def main():
    irl_dir = Path("experiments/nova_review_acceptance/results_enhanced_irl_cv")
    rf_dir = Path("experiments/nova_review_acceptance/results_rf_cv")
    
    train_periods = ["0-3m", "3-6m", "6-9m", "9-12m"]
    eval_periods = ["0-3m", "3-6m", "6-9m", "9-12m"]
    
    results = []
    
    for train_period in train_periods:
        for eval_period in eval_periods:
            pattern = f"train_{train_period}/eval_{eval_period}"
            
            irl_metrics = load_metrics(irl_dir, pattern)
            rf_metrics = load_metrics(rf_dir, pattern)
            
            if irl_metrics and rf_metrics:
                result = {
                    'train': train_period,
                    'eval': eval_period,
                    'pattern': pattern,
                    'IRL_AUC_ROC': irl_metrics.get('auc_roc', 0.0),
                    'IRL_AUC_PR': irl_metrics.get('auc_pr', 0.0),
                    'IRL_F1': irl_metrics.get('f1', 0.0),
                    'IRL_threshold': irl_metrics.get('threshold', 0.0),
                    'RF_AUC_ROC': rf_metrics.get('auc_roc', 0.0),
                    'RF_AUC_PR': rf_metrics.get('auc_pr', 0.0),
                    'RF_F1': rf_metrics.get('f1', 0.0),
                    'RF_threshold': rf_metrics.get('threshold', 0.0),
                    'num_samples': rf_metrics.get('num_samples', 0),
                    'num_positive': rf_metrics.get('num_positive', 0),
                }
                
                # 差分計算（RF - IRL）
                result['diff_AUC_ROC'] = result['RF_AUC_ROC'] - result['IRL_AUC_ROC']
                result['diff_AUC_PR'] = result['RF_AUC_PR'] - result['IRL_AUC_PR']
                result['diff_F1'] = result['RF_F1'] - result['IRL_F1']
                
                results.append(result)
    
    df = pd.DataFrame(results)
    
    print("=" * 100)
    print("IRL vs Random Forest 比較 (K-Fold CV版)")
    print("=" * 100)
    print()
    
    # パターン別比較
    print("パターン別結果:")
    print("-" * 100)
    for _, row in df.iterrows():
        print(f"\n{row['pattern']}:")
        print(f"  サンプル数: {row['num_samples']} ({row['num_positive']} positive)")
        print(f"  AUC-ROC: IRL={row['IRL_AUC_ROC']:.4f}, RF={row['RF_AUC_ROC']:.4f}, Diff={row['diff_AUC_ROC']:+.4f}")
        print(f"  AUC-PR:  IRL={row['IRL_AUC_PR']:.4f}, RF={row['RF_AUC_PR']:.4f}, Diff={row['diff_AUC_PR']:+.4f}")
        print(f"  F1:      IRL={row['IRL_F1']:.4f}, RF={row['RF_F1']:.4f}, Diff={row['diff_F1']:+.4f}")
    
    print()
    print("=" * 100)
    print("全体統計")
    print("=" * 100)
    print()
    
    # 平均値
    print("平均値:")
    print(f"  IRL - AUC-ROC: {df['IRL_AUC_ROC'].mean():.4f} (±{df['IRL_AUC_ROC'].std():.4f})")
    print(f"  IRL - AUC-PR:  {df['IRL_AUC_PR'].mean():.4f} (±{df['IRL_AUC_PR'].std():.4f})")
    print(f"  IRL - F1:      {df['IRL_F1'].mean():.4f} (±{df['IRL_F1'].std():.4f})")
    print()
    print(f"  RF - AUC-ROC: {df['RF_AUC_ROC'].mean():.4f} (±{df['RF_AUC_ROC'].std():.4f})")
    print(f"  RF - AUC-PR:  {df['RF_AUC_PR'].mean():.4f} (±{df['RF_AUC_PR'].std():.4f})")
    print(f"  RF - F1:      {df['RF_F1'].mean():.4f} (±{df['RF_F1'].std():.4f})")
    print()
    
    # 差分統計
    print("差分 (RF - IRL):")
    print(f"  AUC-ROC: {df['diff_AUC_ROC'].mean():+.4f} (±{df['diff_AUC_ROC'].std():.4f})")
    print(f"  AUC-PR:  {df['diff_AUC_PR'].mean():+.4f} (±{df['diff_AUC_PR'].std():.4f})")
    print(f"  F1:      {df['diff_F1'].mean():+.4f} (±{df['diff_F1'].std():.4f})")
    print()
    
    # RFが勝っているパターン数
    rf_wins_auc_roc = (df['diff_AUC_ROC'] > 0).sum()
    rf_wins_f1 = (df['diff_F1'] > 0).sum()
    total = len(df)
    
    print(f"RFが優位なパターン数:")
    print(f"  AUC-ROC: {rf_wins_auc_roc}/{total} ({rf_wins_auc_roc/total*100:.1f}%)")
    print(f"  F1:      {rf_wins_f1}/{total} ({rf_wins_f1/total*100:.1f}%)")
    print()
    
    # CSV保存
    output_path = Path("experiments/nova_review_acceptance/irl_vs_rf_comparison_cv.csv")
    df.to_csv(output_path, index=False)
    print(f"結果を保存: {output_path}")
    print()


if __name__ == '__main__':
    main()
