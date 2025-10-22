#!/usr/bin/env python3
"""
固定履歴・可変ラベル実験の結果比較

履歴窓を固定して、ラベル（nヶ月後の貢献）だけを変えた実験の結果を比較する。
これにより、「予測の難易度」と「モデルの学習しやすさ」の関係を分析できる。
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic']
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experiment_results(base_dir: Path) -> pd.DataFrame:
    """各実験の結果を読み込む"""
    results = []
    
    for exp_dir in sorted(base_dir.glob("label_*")):
        if not exp_dir.is_dir():
            continue
        
        label_name = exp_dir.name.replace("label_", "")
        
        # 評価結果を読み込み
        eval_file = exp_dir / "evaluation_results.json"
        if not eval_file.exists():
            logger.warning(f"評価結果が見つかりません: {eval_file}")
            continue
        
        with open(eval_file) as f:
            metrics = json.load(f)
        
        results.append({
            'label_period': label_name,
            'label_months': int(label_name.replace('m', '')),
            **metrics
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('label_months')
    
    return df


def create_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """比較プロットを生成"""
    logger.info("比較プロットを生成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # メトリクスリスト
    metrics = [
        ('auc_pr', 'AUC-PR'),
        ('f1', 'F1スコア'),
        ('precision', 'Precision'),
        ('positive_rate', '正例率')
    ]
    
    for idx, (metric, name) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # 棒グラフ
        bars = ax.bar(
            df['label_period'],
            df[metric],
            color=sns.color_palette('viridis', len(df)),
            alpha=0.7,
            edgecolor='black',
            linewidth=1.5
        )
        
        # 値をバーの上に表示
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_xlabel('ラベル期間（n ヶ月後）', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} vs ラベル期間', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Y軸の範囲
        if metric == 'positive_rate':
            ax.set_ylim([0, 1.0])
        else:
            ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = output_dir / "fixed_history_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"比較プロットを保存: {output_path}")
    plt.close()


def create_detailed_comparison(df: pd.DataFrame, output_dir: Path):
    """詳細比較（複数メトリクス）"""
    logger.info("詳細比較プロットを生成中...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 複数メトリクスを折れ線グラフで
    metrics = [
        ('auc_pr', 'AUC-PR'),
        ('f1', 'F1'),
        ('precision', 'Precision'),
        ('recall', 'Recall')
    ]
    
    for metric, name in metrics:
        ax.plot(
            df['label_months'],
            df[metric],
            marker='o',
            linewidth=2.5,
            markersize=10,
            label=name,
            alpha=0.8
        )
    
    ax.set_xlabel('ラベル期間 [月]', fontsize=14)
    ax.set_ylabel('スコア', fontsize=14)
    ax.set_title('固定履歴（12ヶ月）・可変ラベル実験\n全メトリクス比較', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # X軸のティック
    ax.set_xticks(df['label_months'])
    ax.set_xticklabels([f"{m}m" for m in df['label_months']])
    
    plt.tight_layout()
    output_path = output_dir / "detailed_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"詳細比較プロットを保存: {output_path}")
    plt.close()


def create_sample_size_analysis(df: pd.DataFrame, output_dir: Path):
    """サンプルサイズと正例率の分析"""
    logger.info("サンプルサイズ分析を生成中...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # サンプル数
    ax = axes[0]
    ax.bar(
        df['label_period'],
        df['test_samples'],
        color='steelblue',
        alpha=0.7,
        edgecolor='black',
        label='評価サンプル数'
    )
    
    for i, (period, count) in enumerate(zip(df['label_period'], df['test_samples'])):
        ax.text(i, count, str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('ラベル期間', fontsize=12)
    ax.set_ylabel('サンプル数', fontsize=12)
    ax.set_title('評価サンプル数', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 正例率（継続率）
    ax = axes[1]
    colors = sns.color_palette('RdYlGn', len(df))
    bars = ax.bar(
        df['label_period'],
        df['positive_rate'],
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )
    
    for i, (period, rate) in enumerate(zip(df['label_period'], df['positive_rate'])):
        ax.text(i, rate, f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('ラベル期間', fontsize=12)
    ax.set_ylabel('正例率（継続率）', fontsize=12)
    ax.set_title('ラベル別の継続率', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = output_dir / "sample_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"サンプル分析を保存: {output_path}")
    plt.close()


def create_report(df: pd.DataFrame, output_dir: Path):
    """レポートをMarkdown形式で生成"""
    logger.info("レポートを生成中...")
    
    report = []
    report.append("# 固定履歴・可変ラベル実験 結果レポート\n\n")
    report.append(f"生成日時: {pd.Timestamp.now()}\n\n")
    report.append("=" * 80 + "\n\n")
    
    # 実験設定
    report.append("## 実験設定\n\n")
    report.append("- **履歴窓**: 12ヶ月（固定）\n")
    report.append("- **訓練期間**: 2022-01-01 ～ 2024-01-01\n")
    report.append("- **評価期間**: 2024-01-01 ～ 2025-01-01\n")
    report.append(f"- **実験数**: {len(df)}\n\n")
    
    # 結果テーブル
    report.append("## 全結果\n\n")
    report.append("| ラベル期間 | AUC-PR | F1 | Precision | Recall | 継続率 | サンプル数 |\n")
    report.append("|-----------|--------|-----|-----------|--------|--------|------------|\n")
    
    for _, row in df.iterrows():
        report.append(
            f"| {row['label_period']} | {row['auc_pr']:.3f} | {row['f1']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['positive_rate']:.1%} | {row['test_samples']} |\n"
        )
    
    # 最良結果
    report.append("\n## 最良結果\n\n")
    
    best_auc_pr_idx = df['auc_pr'].idxmax()
    best_f1_idx = df['f1'].idxmax()
    
    report.append(f"### AUC-PR\n\n")
    report.append(f"- **最良**: {df.loc[best_auc_pr_idx, 'label_period']} "
                  f"(AUC-PR: {df.loc[best_auc_pr_idx, 'auc_pr']:.3f})\n\n")
    
    report.append(f"### F1スコア\n\n")
    report.append(f"- **最良**: {df.loc[best_f1_idx, 'label_period']} "
                  f"(F1: {df.loc[best_f1_idx, 'f1']:.3f})\n\n")
    
    # トレンド分析
    report.append("## トレンド分析\n\n")
    
    # AUC-PRのトレンド
    auc_pr_trend = df['auc_pr'].diff().fillna(0)
    if auc_pr_trend.sum() > 0:
        report.append("- **AUC-PR**: ラベル期間が長くなるほど性能が向上する傾向\n")
    elif auc_pr_trend.sum() < 0:
        report.append("- **AUC-PR**: ラベル期間が長くなるほど性能が低下する傾向\n")
    else:
        report.append("- **AUC-PR**: ラベル期間による明確なトレンドなし\n")
    
    # 継続率のトレンド
    cont_rate_trend = df['positive_rate'].diff().fillna(0)
    if cont_rate_trend.sum() < -0.1:
        report.append("- **継続率**: ラベル期間が長くなるほど継続率が低下（予測が難しくなる）\n")
    else:
        report.append("- **継続率**: ラベル期間による継続率の変化は小さい\n")
    
    report.append("\n## 推奨設定\n\n")
    report.append(f"性能と継続率のバランスを考慮した推奨設定:\n\n")
    report.append(f"- **ラベル期間**: {df.loc[best_auc_pr_idx, 'label_period']}\n")
    report.append(f"- **期待性能**: AUC-PR {df.loc[best_auc_pr_idx, 'auc_pr']:.3f}, "
                  f"F1 {df.loc[best_auc_pr_idx, 'f1']:.3f}\n")
    report.append(f"- **継続率**: {df.loc[best_auc_pr_idx, 'positive_rate']:.1%}\n\n")
    
    # ファイルに保存
    output_path = output_dir / "fixed_history_report.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    logger.info(f"レポートを保存: {output_path}")
    
    # コンソールにも出力
    print("\n" + "".join(report))


def main():
    parser = argparse.ArgumentParser(
        description="固定履歴・可変ラベル実験の結果比較"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="結果ディレクトリのパス"
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        logger.error(f"結果ディレクトリが見つかりません: {results_dir}")
        return
    
    logger.info("=" * 80)
    logger.info("固定履歴・可変ラベル実験の結果比較")
    logger.info("=" * 80)
    
    # 結果を読み込み
    df = load_experiment_results(results_dir)
    
    if len(df) == 0:
        logger.error("実験結果が見つかりません")
        return
    
    logger.info(f"読み込んだ実験数: {len(df)}")
    
    # 可視化とレポート生成
    create_comparison_plots(df, results_dir)
    create_detailed_comparison(df, results_dir)
    create_sample_size_analysis(df, results_dir)
    create_report(df, results_dir)
    
    logger.info("=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

