#!/usr/bin/env python3
"""
閾値の影響を分析するスクリプト
"""
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_threshold_impact(input_dir: Path, output_dir: Path):
    """
    各訓練期間のモデルの閾値による性能変化を分析
    """
    logger.info("=" * 80)
    logger.info("閾値影響分析を開始")
    logger.info("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    periods = ['train_0-3m', 'train_3-6m', 'train_6-9m', 'train_9-12m']
    
    results = {}
    
    for period in periods:
        period_dir = input_dir / period
        predictions_file = period_dir / 'predictions.csv'
        
        if not predictions_file.exists():
            logger.warning(f"予測ファイルが見つかりません: {predictions_file}")
            continue
        
        logger.info(f"\n分析中: {period}")
        
        # 予測を読み込み
        import pandas as pd
        df = pd.read_csv(predictions_file)
        
        y_true = df['true_label'].values
        y_score = df['predicted_prob'].values
        
        # Precision-Recall曲線
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        
        # ROC曲線
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        
        # AUC計算
        auc_pr = auc(recall, precision)
        auc_roc = auc(fpr, tpr)
        
        # 最適閾値を見つける（複数の基準）
        # 1. F1最大
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = pr_thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        
        # 2. Youden's J statistic (ROC)
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        best_j_threshold = roc_thresholds[best_j_idx]
        
        # 3. デフォルト閾値（0.5）での性能
        default_pred = (y_score >= 0.5).astype(int)
        default_tp = np.sum((y_true == 1) & (default_pred == 1))
        default_fp = np.sum((y_true == 0) & (default_pred == 1))
        default_fn = np.sum((y_true == 1) & (default_pred == 0))
        default_precision = default_tp / (default_tp + default_fp + 1e-10)
        default_recall = default_tp / (default_tp + default_fn + 1e-10)
        default_f1 = 2 * default_precision * default_recall / (default_precision + default_recall + 1e-10)
        
        # 4. 現在の最適閾値（metricsから）
        metrics_file = period_dir / 'metrics.json'
        with open(metrics_file) as f:
            metrics = json.load(f)
        current_threshold = metrics['optimal_threshold']
        
        # 現在の閾値での性能
        current_pred = (y_score >= current_threshold).astype(int)
        current_tp = np.sum((y_true == 1) & (current_pred == 1))
        current_fp = np.sum((y_true == 0) & (current_pred == 1))
        current_fn = np.sum((y_true == 1) & (current_pred == 0))
        current_precision = current_tp / (current_tp + current_fp + 1e-10)
        current_recall = current_tp / (current_tp + current_fn + 1e-10)
        current_f1 = 2 * current_precision * current_recall / (current_precision + current_recall + 1e-10)
        
        results[period] = {
            'auc_pr': auc_pr,
            'auc_roc': auc_roc,
            'best_f1_threshold': float(best_f1_threshold),
            'best_f1': float(best_f1),
            'best_j_threshold': float(best_j_threshold),
            'default_threshold': 0.5,
            'default_f1': float(default_f1),
            'default_precision': float(default_precision),
            'default_recall': float(default_recall),
            'current_threshold': float(current_threshold),
            'current_f1': float(current_f1),
            'current_precision': float(current_precision),
            'current_recall': float(current_recall),
            'positive_count': int(np.sum(y_true == 1)),
            'negative_count': int(np.sum(y_true == 0)),
        }
        
        logger.info(f"  AUC-PR: {auc_pr:.4f}")
        logger.info(f"  AUC-ROC: {auc_roc:.4f}")
        logger.info(f"  最適F1閾値: {best_f1_threshold:.4f} (F1={best_f1:.4f})")
        logger.info(f"  Youden's J閾値: {best_j_threshold:.4f}")
        logger.info(f"  デフォルト閾値(0.5): F1={default_f1:.4f}, P={default_precision:.4f}, R={default_recall:.4f}")
        logger.info(f"  現在の閾値({current_threshold:.4f}): F1={current_f1:.4f}, P={current_precision:.4f}, R={current_recall:.4f}")
        
        # PR曲線を描画
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Precision-Recall曲線
        ax1.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AUC = {auc_pr:.3f})')
        ax1.plot(recall[best_f1_idx], precision[best_f1_idx], 'ro', markersize=10, 
                label=f'Best F1 (th={best_f1_threshold:.3f})')
        ax1.plot(current_recall, current_precision, 'g^', markersize=10,
                label=f'Current (th={current_threshold:.3f})')
        ax1.plot(default_recall, default_precision, 'ks', markersize=10,
                label=f'Default (th=0.5)')
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title(f'{period}: Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # ROC曲線
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_roc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax2.plot(fpr[best_j_idx], tpr[best_j_idx], 'ro', markersize=10,
                label=f"Youden's J (th={best_j_threshold:.3f})")
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title(f'{period}: ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f'{period}_threshold_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  保存: {output_file}")
    
    # サマリーを保存
    summary_file = output_dir / 'threshold_analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nサマリーを保存: {summary_file}")
    
    # マークダウンレポートを生成
    md_file = output_dir / 'threshold_analysis_report.md'
    with open(md_file, 'w') as f:
        f.write("# 閾値影響分析レポート\n\n")
        f.write("## 各期間の最適閾値\n\n")
        f.write("| 訓練期間 | AUC-PR | 現在閾値 | 現在F1 | 最適F1閾値 | 最適F1 | F1改善 |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        
        for period in periods:
            if period not in results:
                continue
            r = results[period]
            f1_improvement = r['best_f1'] - r['current_f1']
            improvement_pct = (f1_improvement / r['current_f1'] * 100) if r['current_f1'] > 0 else 0
            
            emoji = "✅" if f1_improvement > 0.05 else ("🟡" if f1_improvement > 0.01 else "⚪")
            
            f.write(f"| {period.replace('train_', '')} | {r['auc_pr']:.4f} | "
                   f"{r['current_threshold']:.4f} | {r['current_f1']:.4f} | "
                   f"{r['best_f1_threshold']:.4f} | {r['best_f1']:.4f} | "
                   f"{emoji} {improvement_pct:+.1f}% |\n")
        
        f.write("\n## 閾値比較の詳細\n\n")
        
        for period in periods:
            if period not in results:
                continue
            r = results[period]
            
            f.write(f"### {period}\n\n")
            f.write("| 閾値設定 | 閾値 | Precision | Recall | F1 |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|\n")
            f.write(f"| **現在の設定** | {r['current_threshold']:.4f} | "
                   f"{r['current_precision']:.4f} | {r['current_recall']:.4f} | {r['current_f1']:.4f} |\n")
            f.write(f"| **F1最適** | {r['best_f1_threshold']:.4f} | - | - | {r['best_f1']:.4f} |\n")
            f.write(f"| デフォルト(0.5) | 0.5000 | "
                   f"{r['default_precision']:.4f} | {r['default_recall']:.4f} | {r['default_f1']:.4f} |\n")
            f.write(f"\n母集団: Positive={r['positive_count']}, Negative={r['negative_count']}\n\n")
    
    logger.info(f"レポートを保存: {md_file}")
    logger.info("=" * 80)
    logger.info("閾値影響分析が完了しました！")
    logger.info("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='閾値の影響を分析')
    parser.add_argument('--input', type=str, default='outputs/review_acceptance_cross_eval_nova',
                       help='入力ディレクトリ')
    parser.add_argument('--output', type=str, default='outputs/review_acceptance_cross_eval_nova/threshold_analysis',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    analyze_threshold_impact(Path(args.input), Path(args.output))

