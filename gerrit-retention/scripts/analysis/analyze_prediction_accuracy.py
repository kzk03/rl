#!/usr/bin/env python3
"""
予測的中率の詳細分析スクリプト

各スライディングウィンドウ設定での：
- True Positive率（継続者を継続と予測）
- True Negative率（離脱者を離脱と予測）
- False Positive率（離脱者を継続と誤予測）
- False Negative率（継続者を離脱と誤予測）
- 的中率（全体の正解率）
- Sensitivity（真陽性率、Recall）
- Specificity（真陰性率）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def calculate_detailed_metrics(row):
    """混同行列から詳細メトリクスを計算"""
    tn = row.get('tn', 0)
    fp = row.get('fp', 0)
    fn = row.get('fn', 0)
    tp = row.get('tp', 0)

    total = tn + fp + fn + tp

    if total == 0:
        return {
            'accuracy': 0.0,
            'tp_rate': 0.0,
            'tn_rate': 0.0,
            'fp_rate': 0.0,
            'fn_rate': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'positive_predictive_value': 0.0,
            'negative_predictive_value': 0.0,
        }

    # 的中率（全体の正解率）
    accuracy = (tp + tn) / total

    # 各カテゴリの割合
    tp_rate = tp / total
    tn_rate = tn / total
    fp_rate = fp / total
    fn_rate = fn / total

    # Sensitivity（真陽性率、Recall）= TP / (TP + FN)
    # 「実際に継続した人のうち、継続と予測できた割合」
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity（真陰性率）= TN / (TN + FP)
    # 「実際に離脱した人のうち、離脱と予測できた割合」
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Positive Predictive Value (Precision)
    # 「継続と予測した人のうち、実際に継続した割合」
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Negative Predictive Value
    # 「離脱と予測した人のうち、実際に離脱した割合」
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'tp_rate': tp_rate,
        'tn_rate': tn_rate,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'positive_predictive_value': ppv,
        'negative_predictive_value': npv,
    }


def analyze_prediction_accuracy(csv_path: Path, output_dir: Path):
    """予測的中率を分析"""

    # データ読み込み
    df = pd.read_csv(csv_path)

    # プロジェクト名を取得（ファイル名から）
    if 'multi_project' in csv_path.name:
        project_name = 'Multi-Project'
    else:
        # "sliding_window_results_project_openstack_nova.csv" -> "openstack/nova"
        parts = csv_path.stem.replace('sliding_window_results_project_', '').split('_')
        project_name = '/'.join(parts)

    # 詳細メトリクスを計算
    metrics_list = []
    for idx, row in df.iterrows():
        metrics = calculate_detailed_metrics(row)
        metrics['learning_months'] = row['learning_months']
        metrics['prediction_months'] = row['prediction_months']
        metrics['n_trajectories'] = row['n_trajectories']
        metrics['continuation_rate'] = row['continuation_rate']
        metrics['auc_roc'] = row['auc_roc']
        metrics['f1'] = row['f1']
        metrics['tn'] = row.get('tn', 0)
        metrics['fp'] = row.get('fp', 0)
        metrics['fn'] = row.get('fn', 0)
        metrics['tp'] = row.get('tp', 0)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    # CSVに保存
    output_csv = output_dir / f"accuracy_analysis_{csv_path.stem}.csv"
    metrics_df.to_csv(output_csv, index=False)
    print(f"Saved detailed metrics to: {output_csv}")

    # サマリーレポート作成
    create_accuracy_report(metrics_df, project_name, output_dir, csv_path.stem)

    # ヒートマップ作成
    create_accuracy_heatmaps(metrics_df, project_name, output_dir, csv_path.stem)

    return metrics_df


def create_accuracy_report(df: pd.DataFrame, project_name: str, output_dir: Path, file_stem: str):
    """的中率の詳細レポートを作成"""

    report_path = output_dir / f"accuracy_report_{file_stem}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 予測的中率分析レポート: {project_name}\n\n")

        # トップ10設定（的中率順）
        f.write("## 1. 的中率トップ10設定\n\n")
        f.write("| 順位 | 学習期間 | 予測期間 | 的中率 | Sensitivity | Specificity | AUC-ROC | F1 |\n")
        f.write("|-----|---------|---------|--------|-------------|-------------|---------|----|\n")

        top_accuracy = df.nlargest(10, 'accuracy')
        for idx, (i, row) in enumerate(top_accuracy.iterrows(), 1):
            f.write(f"| {idx} | {row['learning_months']}m | {row['prediction_months']}m | "
                   f"{row['accuracy']:.3f} | {row['sensitivity']:.3f} | {row['specificity']:.3f} | "
                   f"{row['auc_roc']:.3f} | {row['f1']:.3f} |\n")

        # Sensitivityトップ10
        f.write("\n## 2. Sensitivity（継続者的中率）トップ10\n\n")
        f.write("**継続した人を正しく継続と予測できた割合**\n\n")
        f.write("| 順位 | 学習期間 | 予測期間 | Sensitivity | TP | FN | 継続者数 | 継続率 |\n")
        f.write("|-----|---------|---------|-------------|----|----|---------|--------|\n")

        top_sens = df.nlargest(10, 'sensitivity')
        for idx, (i, row) in enumerate(top_sens.iterrows(), 1):
            continued = row['tp'] + row['fn']
            f.write(f"| {idx} | {row['learning_months']}m | {row['prediction_months']}m | "
                   f"{row['sensitivity']:.3f} | {int(row['tp'])} | {int(row['fn'])} | "
                   f"{continued} | {row['continuation_rate']:.1f}% |\n")

        # Specificityトップ10
        f.write("\n## 3. Specificity（離脱者的中率）トップ10\n\n")
        f.write("**離脱した人を正しく離脱と予測できた割合**\n\n")
        f.write("| 順位 | 学習期間 | 予測期間 | Specificity | TN | FP | 離脱者数 |\n")
        f.write("|-----|---------|---------|-------------|----|----|--------|\n")

        top_spec = df.nlargest(10, 'specificity')
        for idx, (i, row) in enumerate(top_spec.iterrows(), 1):
            churned = row['tn'] + row['fp']
            f.write(f"| {idx} | {row['learning_months']}m | {row['prediction_months']}m | "
                   f"{row['specificity']:.3f} | {int(row['tn'])} | {int(row['fp'])} | {churned} |\n")

        # バランスの良い設定
        f.write("\n## 4. バランスの良い設定（Sensitivity & Specificity両方高い）\n\n")
        f.write("| 順位 | 学習期間 | 予測期間 | 的中率 | Sensitivity | Specificity | バランススコア |\n")
        f.write("|-----|---------|---------|--------|-------------|-------------|-------------|\n")

        df['balance_score'] = (df['sensitivity'] + df['specificity']) / 2
        top_balance = df.nlargest(10, 'balance_score')
        for idx, (i, row) in enumerate(top_balance.iterrows(), 1):
            f.write(f"| {idx} | {row['learning_months']}m | {row['prediction_months']}m | "
                   f"{row['accuracy']:.3f} | {row['sensitivity']:.3f} | {row['specificity']:.3f} | "
                   f"{row['balance_score']:.3f} |\n")

        # 混同行列の詳細
        f.write("\n## 5. 全設定の混同行列\n\n")
        f.write("| 学習 | 予測 | TP | FP | FN | TN | 継続率 | 的中率 | Sens | Spec |\n")
        f.write("|-----|-----|----|----|----|----|--------|--------|------|------|\n")

        for _, row in df.iterrows():
            f.write(f"| {row['learning_months']}m | {row['prediction_months']}m | "
                   f"{int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} | {int(row['tn'])} | "
                   f"{row['continuation_rate']:.1f}% | {row['accuracy']:.3f} | "
                   f"{row['sensitivity']:.3f} | {row['specificity']:.3f} |\n")

        # 統計サマリー
        f.write("\n## 6. 統計サマリー\n\n")
        f.write(f"**的中率（Accuracy）**:\n")
        f.write(f"- 平均: {df['accuracy'].mean():.3f}\n")
        f.write(f"- 最高: {df['accuracy'].max():.3f}\n")
        f.write(f"- 最低: {df['accuracy'].min():.3f}\n")
        f.write(f"- 標準偏差: {df['accuracy'].std():.3f}\n\n")

        f.write(f"**Sensitivity（継続者的中率）**:\n")
        f.write(f"- 平均: {df['sensitivity'].mean():.3f}\n")
        f.write(f"- 最高: {df['sensitivity'].max():.3f}\n")
        f.write(f"- 最低: {df['sensitivity'].min():.3f}\n\n")

        f.write(f"**Specificity（離脱者的中率）**:\n")
        f.write(f"- 平均: {df['specificity'].mean():.3f}\n")
        f.write(f"- 最高: {df['specificity'].max():.3f}\n")
        f.write(f"- 最低: {df['specificity'].min():.3f}\n\n")

        # 洞察
        f.write("\n## 7. 主要な洞察\n\n")

        # 継続率と的中率の関係
        high_cont = df[df['continuation_rate'] > 70]
        low_cont = df[df['continuation_rate'] <= 50]

        f.write(f"### 継続率の影響\n\n")
        if len(high_cont) > 0:
            f.write(f"**高継続率（>70%）の設定**:\n")
            f.write(f"- 平均的中率: {high_cont['accuracy'].mean():.3f}\n")
            f.write(f"- 平均Sensitivity: {high_cont['sensitivity'].mean():.3f}\n")
            f.write(f"- 平均Specificity: {high_cont['specificity'].mean():.3f}\n\n")

        if len(low_cont) > 0:
            f.write(f"**低継続率（≤50%）の設定**:\n")
            f.write(f"- 平均的中率: {low_cont['accuracy'].mean():.3f}\n")
            f.write(f"- 平均Sensitivity: {low_cont['sensitivity'].mean():.3f}\n")
            f.write(f"- 平均Specificity: {low_cont['specificity'].mean():.3f}\n\n")

        # 学習期間別
        f.write(f"### 学習期間別の傾向\n\n")
        for learning in sorted(df['learning_months'].unique()):
            subset = df[df['learning_months'] == learning]
            f.write(f"**{learning}ヶ月学習**:\n")
            f.write(f"- 平均的中率: {subset['accuracy'].mean():.3f}\n")
            f.write(f"- 最高的中率: {subset['accuracy'].max():.3f} ({int(subset.loc[subset['accuracy'].idxmax(), 'prediction_months'])}m予測)\n\n")

    print(f"Saved accuracy report to: {report_path}")


def create_accuracy_heatmaps(df: pd.DataFrame, project_name: str, output_dir: Path, file_stem: str):
    """的中率のヒートマップを作成"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Prediction Accuracy Analysis: {project_name}', fontsize=16, fontweight='bold')

    # ヒートマップ用のピボットテーブル作成
    metrics_to_plot = [
        ('accuracy', 'Overall Accuracy (的中率)'),
        ('sensitivity', 'Sensitivity (継続者的中率)'),
        ('specificity', 'Specificity (離脱者的中率)'),
        ('tp_rate', 'True Positive Rate'),
        ('tn_rate', 'True Negative Rate'),
        ('balance_score', 'Balance Score (Sens+Spec)/2'),
    ]

    df['balance_score'] = (df['sensitivity'] + df['specificity']) / 2

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        # ピボットテーブル作成
        pivot = df.pivot(
            index='learning_months',
            columns='prediction_months',
            values=metric
        )

        # ヒートマップ描画
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            center=0.5,
            ax=ax,
            cbar_kws={'label': metric}
        )

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Prediction Period (months)', fontsize=10)
        ax.set_ylabel('Learning Period (months)', fontsize=10)
        ax.invert_yaxis()

    plt.tight_layout()

    output_path = output_dir / f"accuracy_heatmaps_{file_stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved accuracy heatmaps to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction accuracy across sliding windows")
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing CSV results')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for analysis')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # すべてのCSVファイルを処理
    csv_files = list(input_dir.glob('sliding_window_results_*.csv'))

    print(f"Found {len(csv_files)} CSV files to analyze")

    all_results = {}
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Analyzing: {csv_file.name}")
        print(f"{'='*60}")

        result_df = analyze_prediction_accuracy(csv_file, output_dir)
        all_results[csv_file.stem] = result_df

    # 総合比較レポート
    create_comparison_report(all_results, output_dir)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")


def create_comparison_report(all_results: dict, output_dir: Path):
    """全プロジェクトの比較レポートを作成"""

    report_path = output_dir / "accuracy_comparison_all_projects.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 全プロジェクト予測的中率比較レポート\n\n")

        # プロジェクト別サマリー
        f.write("## 1. プロジェクト別平均的中率\n\n")
        f.write("| プロジェクト | 平均的中率 | 最高的中率 | 平均Sensitivity | 平均Specificity |\n")
        f.write("|------------|-----------|-----------|----------------|----------------|\n")

        project_summary = []
        for name, df in all_results.items():
            if 'multi_project' in name:
                display_name = 'Multi-Project'
            else:
                display_name = name.replace('sliding_window_results_project_', '').replace('_', '/')

            project_summary.append({
                'name': display_name,
                'avg_accuracy': df['accuracy'].mean(),
                'max_accuracy': df['accuracy'].max(),
                'avg_sensitivity': df['sensitivity'].mean(),
                'avg_specificity': df['specificity'].mean(),
            })

        # ソート
        project_summary.sort(key=lambda x: x['avg_accuracy'], reverse=True)

        for proj in project_summary:
            f.write(f"| {proj['name']} | {proj['avg_accuracy']:.3f} | {proj['max_accuracy']:.3f} | "
                   f"{proj['avg_sensitivity']:.3f} | {proj['avg_specificity']:.3f} |\n")

        # 全体統計
        f.write("\n## 2. 全プロジェクト統合統計\n\n")

        all_df = pd.concat(all_results.values(), ignore_index=True)

        f.write(f"- 総設定数: {len(all_df)}\n")
        f.write(f"- 全体平均的中率: {all_df['accuracy'].mean():.3f}\n")
        f.write(f"- 全体最高的中率: {all_df['accuracy'].max():.3f}\n")
        f.write(f"- 全体平均Sensitivity: {all_df['sensitivity'].mean():.3f}\n")
        f.write(f"- 全体平均Specificity: {all_df['specificity'].mean():.3f}\n\n")

        # トップ設定（全プロジェクト）
        f.write("## 3. 全プロジェクト中の的中率トップ20設定\n\n")
        f.write("| 順位 | プロジェクト | 学習 | 予測 | 的中率 | Sens | Spec | AUC |\n")
        f.write("|-----|------------|------|------|--------|------|------| ----|\n")

        # プロジェクト名を追加
        all_with_names = []
        for name, df in all_results.items():
            if 'multi_project' in name:
                display_name = 'Multi-Project'
            else:
                display_name = name.replace('sliding_window_results_project_', '').replace('_', '/')

            df_copy = df.copy()
            df_copy['project_name'] = display_name
            all_with_names.append(df_copy)

        combined = pd.concat(all_with_names, ignore_index=True)
        top20 = combined.nlargest(20, 'accuracy')

        for idx, (_, row) in enumerate(top20.iterrows(), 1):
            f.write(f"| {idx} | {row['project_name']} | {row['learning_months']}m | {row['prediction_months']}m | "
                   f"{row['accuracy']:.3f} | {row['sensitivity']:.3f} | {row['specificity']:.3f} | "
                   f"{row['auc_roc']:.3f} |\n")

    print(f"\nSaved comparison report to: {report_path}")


if __name__ == '__main__':
    main()
