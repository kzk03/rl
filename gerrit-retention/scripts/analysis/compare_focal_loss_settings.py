#!/usr/bin/env python3
"""
Focal Loss 設定の比較分析

Recall優先版（alpha=0.25, gamma=2.0）とバランス版（alpha=0.4, gamma=1.5）を比較
"""

import json
from pathlib import Path

import pandas as pd


def load_metrics(base_dir: Path, train_period: str, eval_period: str = None) -> dict:
    """メトリクスを読み込む"""
    if eval_period:
        metrics_path = base_dir / f'train_{train_period}' / f'eval_{eval_period}' / 'metrics.json'
    else:
        metrics_path = base_dir / f'train_{train_period}' / 'metrics.json'
    
    if not metrics_path.exists():
        return None
    
    with open(metrics_path) as f:
        return json.load(f)

def main():
    print("## 📊 Focal Loss 設定の比較")
    print()
    
    # ディレクトリ
    recall_dir = Path('outputs/review_acceptance_cross_eval_nova')
    balanced_dir = Path('outputs/review_acceptance_cross_eval_nova_balanced')
    
    periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    # 自己評価の比較
    print("### **自己評価（対角線）の比較**")
    print()
    print("| 訓練期間 | Recall優先 AUC-PR | バランス型 AUC-PR | 差分 | Recall優先 Recall | バランス型 Recall | 差分 | Recall優先 Precision | バランス型 Precision | 差分 |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    for period in periods:
        # Recall 優先版
        m_recall = load_metrics(recall_dir, period, period)
        
        # バランス版
        m_balanced = load_metrics(balanced_dir, period, period)
        
        if m_recall and m_balanced:
            auc_diff = m_balanced['auc_pr'] - m_recall['auc_pr']
            recall_diff = m_balanced['recall'] - m_recall['recall']
            precision_diff = m_balanced['precision'] - m_recall['precision']
            
            auc_arrow = "⬆️" if auc_diff > 0.01 else "⬇️" if auc_diff < -0.01 else "→"
            recall_arrow = "⬆️" if recall_diff > 0.01 else "⬇️" if recall_diff < -0.01 else "→"
            precision_arrow = "⬆️" if precision_diff > 0.01 else "⬇️" if precision_diff < -0.01 else "→"
            
            print(f"| **{period}** | {m_recall['auc_pr']:.3f} | {m_balanced['auc_pr']:.3f} | {auc_diff:+.3f} {auc_arrow} | {m_recall['recall']:.3f} | {m_balanced['recall']:.3f} | {recall_diff:+.3f} {recall_arrow} | {m_recall['precision']:.3f} | {m_balanced['precision']:.3f} | {precision_diff:+.3f} {precision_arrow} |")
        elif m_recall:
            print(f"| **{period}** | {m_recall['auc_pr']:.3f} | ⏳ 訓練中 | - | {m_recall['recall']:.3f} | - | - | {m_recall['precision']:.3f} | - | - |")
        else:
            print(f"| **{period}** | ⏳ 訓練中 | ⏳ 訓練中 | - | - | - | - | - | - | - |")
    
    print()
    print("### **設定パラメータ**")
    print()
    print("| 設定 | alpha | gamma | 特徴 |")
    print("|:---|:---:|:---:|:---|")
    print("| Recall優先 | 0.25 | 2.0 | 正例を見逃さない（FN↓）|")
    print("| バランス型 | 0.4 | 1.5 | Recall と Precision のバランス |")

if __name__ == '__main__':
    main()

