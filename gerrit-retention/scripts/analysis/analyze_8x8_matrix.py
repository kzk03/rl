#!/usr/bin/env python3
"""
8×8行列の詳細分析スクリプト

最適な学習期間・予測期間の組み合わせを特定し、
パターンを分析します。
"""
import argparse
from pathlib import Path

import pandas as pd


def analyze_matrix(csv_path: Path, output_path: Path):
    """8×8行列の詳細分析"""
    df = pd.read_csv(csv_path)

    # メトリクスのリスト
    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'accuracy']

    report = []
    report.append("="*80)
    report.append("8×8 IRL評価行列 - 詳細分析レポート")
    report.append("="*80)
    report.append("")

    # 基本情報
    report.append("## 評価概要")
    report.append("")
    report.append(f"- 総実験数: {len(df)}")
    report.append(f"- 学習期間: {sorted(df['history_months'].unique())} ヶ月")
    report.append(f"- 予測期間: {sorted(df['target_months'].unique())} ヶ月")
    report.append(f"- 固定対象レビュアー数: {df['target_reviewer_count'].iloc[0]}人")
    report.append("")

    # 各メトリクスごとの最良組み合わせ
    report.append("="*80)
    report.append("## 最良の組み合わせ（メトリクス別）")
    report.append("="*80)
    report.append("")

    for metric in metrics:
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]

        report.append(f"### {metric.upper()}")
        report.append(f"- 最高値: {best_row[metric]:.4f}")
        report.append(f"- 学習期間: {best_row['history_months']}ヶ月")
        report.append(f"- 予測期間: {best_row['target_months']}ヶ月")
        report.append("")

    # 学習期間別の平均性能
    report.append("="*80)
    report.append("## 学習期間別の平均性能")
    report.append("="*80)
    report.append("")

    history_stats = df.groupby('history_months')[metrics].mean()

    report.append("| 学習期間 | AUC-ROC | AUC-PR | F1    | Precision | Recall | Accuracy |")
    report.append("|---------|---------|--------|-------|-----------|--------|----------|")

    for history in sorted(df['history_months'].unique()):
        stats = history_stats.loc[history]
        report.append(
            f"| {history:2d}ヶ月   | {stats['auc_roc']:.3f}   | "
            f"{stats['auc_pr']:.3f}  | {stats['f1']:.3f} | "
            f"{stats['precision']:.3f}     | {stats['recall']:.3f}  | "
            f"{stats['accuracy']:.3f}    |"
        )

    # 最良の学習期間
    best_history = history_stats['auc_roc'].idxmax()
    report.append("")
    report.append(f"**推奨学習期間**: {best_history}ヶ月 "
                 f"(平均AUC-ROC: {history_stats.loc[best_history, 'auc_roc']:.3f})")
    report.append("")

    # 予測期間別の平均性能
    report.append("="*80)
    report.append("## 予測期間別の平均性能")
    report.append("="*80)
    report.append("")

    target_stats = df.groupby('target_months')[metrics].mean()

    report.append("| 予測期間 | AUC-ROC | AUC-PR | F1    | Precision | Recall | Accuracy |")
    report.append("|---------|---------|--------|-------|-----------|--------|----------|")

    for target in sorted(df['target_months'].unique()):
        stats = target_stats.loc[target]
        report.append(
            f"| {target:2d}ヶ月   | {stats['auc_roc']:.3f}   | "
            f"{stats['auc_pr']:.3f}  | {stats['f1']:.3f} | "
            f"{stats['precision']:.3f}     | {stats['recall']:.3f}  | "
            f"{stats['accuracy']:.3f}    |"
        )

    # 最良の予測期間
    best_target = target_stats['auc_roc'].idxmax()
    report.append("")
    report.append(f"**推奨予測期間**: {best_target}ヶ月 "
                 f"(平均AUC-ROC: {target_stats.loc[best_target, 'auc_roc']:.3f})")
    report.append("")

    # トップ5の組み合わせ
    report.append("="*80)
    report.append("## トップ5の組み合わせ（AUC-ROC）")
    report.append("="*80)
    report.append("")

    top5 = df.nlargest(5, 'auc_roc')[
        ['history_months', 'target_months', 'auc_roc', 'auc_pr', 'f1',
         'precision', 'recall', 'accuracy']
    ]

    report.append("| 順位 | 学習 | 予測 | AUC-ROC | AUC-PR | F1    | Precision | Recall |")
    report.append("|-----|------|------|---------|--------|-------|-----------|--------|")

    for idx, (_, row) in enumerate(top5.iterrows(), 1):
        report.append(
            f"| {idx}   | {row['history_months']:2d}月 | {row['target_months']:2d}月 | "
            f"{row['auc_roc']:.3f}   | {row['auc_pr']:.3f}  | {row['f1']:.3f} | "
            f"{row['precision']:.3f}     | {row['recall']:.3f}  |"
        )

    report.append("")

    # ワースト5の組み合わせ
    report.append("="*80)
    report.append("## ワースト5の組み合わせ（AUC-ROC）")
    report.append("="*80)
    report.append("")

    bottom5 = df.nsmallest(5, 'auc_roc')[
        ['history_months', 'target_months', 'auc_roc', 'auc_pr', 'f1',
         'precision', 'recall', 'accuracy']
    ]

    report.append("| 順位 | 学習 | 予測 | AUC-ROC | AUC-PR | F1    | Precision | Recall |")
    report.append("|-----|------|------|---------|--------|-------|-----------|--------|")

    for idx, (_, row) in enumerate(bottom5.iterrows(), 1):
        report.append(
            f"| {idx}   | {row['history_months']:2d}月 | {row['target_months']:2d}月 | "
            f"{row['auc_roc']:.3f}   | {row['auc_pr']:.3f}  | {row['f1']:.3f} | "
            f"{row['precision']:.3f}     | {row['recall']:.3f}  |"
        )

    report.append("")

    # パターン分析
    report.append("="*80)
    report.append("## パターン分析")
    report.append("="*80)
    report.append("")

    # 学習期間 vs 予測期間の比率
    df['ratio'] = df['history_months'] / df['target_months']
    ratio_stats = df.groupby('ratio')['auc_roc'].agg(['mean', 'count'])

    report.append("### 学習期間/予測期間の比率と性能")
    report.append("")
    report.append("| 比率 | 平均AUC-ROC | サンプル数 |")
    report.append("|------|-------------|-----------|")

    for ratio in sorted(ratio_stats.index):
        stats = ratio_stats.loc[ratio]
        report.append(f"| {ratio:.2f} | {stats['mean']:.3f}       | {stats['count']:2d}        |")

    report.append("")

    # 短期 vs 中期 vs 長期
    report.append("### 期間カテゴリ別の性能")
    report.append("")

    def categorize_period(months):
        if months <= 6:
            return "短期"
        elif months <= 12:
            return "中期"
        else:
            return "長期"

    df['history_category'] = df['history_months'].apply(categorize_period)
    df['target_category'] = df['target_months'].apply(categorize_period)

    category_stats = df.groupby(['history_category', 'target_category'])['auc_roc'].mean().unstack()

    report.append("**学習期間（行） × 予測期間（列）の平均AUC-ROC**")
    report.append("")
    report.append(category_stats.to_markdown())
    report.append("")

    # 全体統計
    report.append("="*80)
    report.append("## 全体統計")
    report.append("="*80)
    report.append("")

    for metric in metrics:
        report.append(f"### {metric.upper()}")
        report.append(f"- 平均: {df[metric].mean():.4f}")
        report.append(f"- 標準偏差: {df[metric].std():.4f}")
        report.append(f"- 最小: {df[metric].min():.4f}")
        report.append(f"- 最大: {df[metric].max():.4f}")
        report.append("")

    # レポート保存
    report_text = "\n".join(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding='utf-8')

    print(f"✅ 分析レポートを保存しました: {output_path}")
    print("\n" + "="*80)
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description='Analyze 8×8 evaluation matrix')
    parser.add_argument('--csv', type=Path, required=True,
                       help='Path to evaluation results CSV')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for analysis report')

    args = parser.parse_args()

    analyze_matrix(args.csv, args.output)


if __name__ == '__main__':
    main()
