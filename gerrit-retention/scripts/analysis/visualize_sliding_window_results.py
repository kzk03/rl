#!/usr/bin/env python3
"""
スライディングウィンドウ評価結果の包括的可視化スクリプト

このスクリプトは以下の可視化を生成します:
1. メトリクスヒートマップ（改善版）
2. 履歴窓・将来窓ごとの性能トレンド
3. サンプル数と性能の関係
4. メトリクス相関分析
5. 総合評価レポート
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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


class SlidingWindowVisualizer:
    """スライディングウィンドウ評価結果の可視化クラス"""
    
    def __init__(self, results_dir: Path):
        """
        Args:
            results_dir: 結果ディレクトリのパス
        """
        self.results_dir = Path(results_dir)
        self.results_data = self._load_results()
        self.df = self._create_dataframe()
        
    def _load_results(self) -> List[Dict]:
        """結果JSONファイルを読み込む"""
        results_file = self.results_dir / "all_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"結果ファイルが見つかりません: {results_file}")
        
        with open(results_file) as f:
            return json.load(f)
    
    def _create_dataframe(self) -> pd.DataFrame:
        """結果データからDataFrameを作成"""
        rows = []
        for result in self.results_data:
            row = {
                'history_window': result['history_window'],
                'future_window': result['future_window_label'],
                'future_start': result['future_window_start'],
                'future_end': result['future_window_end'],
                'train_samples': result['train_samples'],
                'eval_samples': result['eval_samples'],
                **result['metrics']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        # 履歴窓と将来窓でソート
        df = df.sort_values(['history_window', 'future_start'])
        return df
    
    def create_enhanced_heatmaps(self, output_path: Path):
        """
        改善版ヒートマップを生成
        
        - より大きく見やすいレイアウト
        - 数値を見やすく表示
        - カラーマップの最適化
        """
        logger.info("改善版ヒートマップを生成中...")
        
        metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
        metric_names = {
            'auc_roc': 'AUC-ROC',
            'auc_pr': 'AUC-PR',
            'f1': 'F1スコア',
            'precision': 'Precision',
            'recall': 'Recall'
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            # ピボットテーブル作成
            pivot = self.df.pivot_table(
                values=metric,
                index='history_window',
                columns='future_window',
                aggfunc='mean'
            )
            
            # ヒートマップ描画
            ax = axes[idx]
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0.4 if metric == 'auc_roc' else 0.0,
                vmax=1.0,
                cbar_kws={'label': metric_names[metric]},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )
            
            ax.set_title(f'{metric_names[metric]}の比較', fontsize=14, fontweight='bold')
            ax.set_xlabel('将来窓（予測期間）', fontsize=12)
            ax.set_ylabel('履歴窓（観測期間）[月]', fontsize=12)
            
            # 最大値をハイライト
            max_val = pivot.max().max()
            if not np.isnan(max_val):
                max_pos = np.where(pivot == max_val)
                if len(max_pos[0]) > 0:
                    ax.add_patch(plt.Rectangle(
                        (max_pos[1][0], max_pos[0][0]),
                        1, 1,
                        fill=False,
                        edgecolor='blue',
                        linewidth=3
                    ))
        
        # 最後のサブプロットにサマリー情報を表示
        ax = axes[5]
        ax.axis('off')
        
        summary_text = self._create_summary_text()
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"ヒートマップを保存: {output_path}")
        plt.close()
    
    def _create_summary_text(self) -> str:
        """サマリーテキストを生成"""
        best_results = {}
        metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
        
        for metric in metrics:
            best_idx = self.df[metric].idxmax()
            best_row = self.df.loc[best_idx]
            best_results[metric] = (
                f"{metric.upper()}: {best_row[metric]:.3f}\n"
                f"  履歴={best_row['history_window']}m, "
                f"将来={best_row['future_window']}"
            )
        
        summary = "=" * 40 + "\n"
        summary += "最良結果サマリー\n"
        summary += "=" * 40 + "\n\n"
        summary += "\n\n".join(best_results.values())
        summary += f"\n\n総実験数: {len(self.df)}"
        
        return summary
    
    def create_trend_plots(self, output_path: Path):
        """履歴窓・将来窓ごとの性能トレンドを可視化"""
        logger.info("トレンドプロットを生成中...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [('auc_pr', 'AUC-PR'), ('f1', 'F1スコア'),
                   ('precision', 'Precision'), ('recall', 'Recall')]
        
        for idx, (metric, name) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # 将来窓ごとにグループ化してプロット
            for future_window in self.df['future_window'].unique():
                subset = self.df[self.df['future_window'] == future_window]
                ax.plot(
                    subset['history_window'],
                    subset[metric],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=f'将来窓: {future_window}',
                    alpha=0.7
                )
            
            ax.set_xlabel('履歴窓 [月]', fontsize=12)
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name} vs 履歴窓', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Y軸の範囲を設定
            if metric == 'recall':
                ax.set_ylim([0.95, 1.01])
            else:
                ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"トレンドプロットを保存: {output_path}")
        plt.close()
    
    def create_sample_analysis(self, output_path: Path):
        """サンプル数と性能の関係を分析"""
        logger.info("サンプル数分析を生成中...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics = ['auc_pr', 'f1', 'precision']
        sample_types = [('train_samples', '訓練'), ('eval_samples', '評価')]
        
        plot_idx = 0
        for metric, metric_name in [('auc_pr', 'AUC-PR'), ('f1', 'F1'), ('precision', 'Precision')]:
            for sample_type, sample_name in sample_types:
                ax = axes[plot_idx]
                
                # 散布図
                scatter = ax.scatter(
                    self.df[sample_type],
                    self.df[metric],
                    c=self.df['history_window'],
                    s=100,
                    alpha=0.6,
                    cmap='viridis'
                )
                
                # トレンドライン
                z = np.polyfit(self.df[sample_type], self.df[metric], 1)
                p = np.poly1d(z)
                ax.plot(
                    self.df[sample_type],
                    p(self.df[sample_type]),
                    "r--",
                    alpha=0.5,
                    linewidth=2
                )
                
                # 相関係数
                corr = self.df[[sample_type, metric]].corr().iloc[0, 1]
                ax.text(
                    0.05, 0.95,
                    f'相関係数: {corr:.3f}',
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
                
                ax.set_xlabel(f'{sample_name}サンプル数', fontsize=11)
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(f'{metric_name} vs {sample_name}サンプル数', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # カラーバー
                if plot_idx % 2 == 1:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('履歴窓 [月]', fontsize=10)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"サンプル数分析を保存: {output_path}")
        plt.close()
    
    def create_metrics_correlation(self, output_path: Path):
        """メトリクス間の相関を可視化"""
        logger.info("メトリクス相関分析を生成中...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 相関行列
        metrics_cols = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall',
                        'train_samples', 'eval_samples']
        corr_matrix = self.df[metrics_cols].corr()
        
        # ヒートマップ
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            ax=axes[0],
            square=True,
            linewidths=0.5
        )
        axes[0].set_title('メトリクス相関行列', fontsize=14, fontweight='bold')
        
        # ペアプロット（主要メトリクスのみ）
        main_metrics = ['auc_pr', 'f1', 'precision']
        scatter_data = []
        
        for i, metric1 in enumerate(main_metrics):
            for j, metric2 in enumerate(main_metrics):
                if i < j:
                    scatter_data.append({
                        'x': metric1,
                        'y': metric2,
                        'corr': corr_matrix.loc[metric1, metric2]
                    })
        
        # 最も強い相関を表示
        axes[1].axis('off')
        summary_text = "主要メトリクス間の相関\n" + "=" * 40 + "\n\n"
        
        for item in sorted(scatter_data, key=lambda x: abs(x['corr']), reverse=True):
            summary_text += f"{item['x'].upper()} ↔ {item['y'].upper()}: {item['corr']:+.3f}\n"
        
        # 追加分析
        summary_text += "\n" + "=" * 40 + "\n"
        summary_text += "データ統計\n" + "=" * 40 + "\n\n"
        summary_text += f"平均AUC-PR: {self.df['auc_pr'].mean():.3f}\n"
        summary_text += f"平均F1: {self.df['f1'].mean():.3f}\n"
        summary_text += f"平均Precision: {self.df['precision'].mean():.3f}\n"
        summary_text += f"平均訓練サンプル: {self.df['train_samples'].mean():.0f}\n"
        summary_text += f"平均評価サンプル: {self.df['eval_samples'].mean():.0f}\n"
        
        axes[1].text(
            0.1, 0.9,
            summary_text,
            transform=axes[1].transAxes,
            fontsize=12,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"相関分析を保存: {output_path}")
        plt.close()
    
    def create_detailed_report(self, output_path: Path):
        """詳細レポートをMarkdown形式で生成"""
        logger.info("詳細レポートを生成中...")
        
        report = []
        report.append("# スライディングウィンドウ評価 詳細レポート\n")
        report.append(f"生成日時: {pd.Timestamp.now()}\n")
        report.append("=" * 80 + "\n\n")
        
        # 概要
        report.append("## 1. 実験概要\n\n")
        report.append(f"- 総実験数: {len(self.df)}\n")
        report.append(f"- 履歴窓: {sorted(self.df['history_window'].unique())} ヶ月\n")
        report.append(f"- 将来窓: {sorted(self.df['future_window'].unique())}\n\n")
        
        # 最良結果
        report.append("## 2. 最良結果\n\n")
        metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
        
        for metric in metrics:
            best_idx = self.df[metric].idxmax()
            best_row = self.df.loc[best_idx]
            
            report.append(f"### {metric.upper()}\n\n")
            report.append(f"- **スコア**: {best_row[metric]:.4f}\n")
            report.append(f"- **設定**: 履歴={best_row['history_window']}ヶ月, 将来={best_row['future_window']}\n")
            report.append(f"- **訓練サンプル**: {best_row['train_samples']}\n")
            report.append(f"- **評価サンプル**: {best_row['eval_samples']}\n")
            report.append(f"- **モデル**: `{best_row.get('model_path', 'N/A')}`\n\n")
        
        # 全結果テーブル
        report.append("## 3. 全実験結果\n\n")
        report.append("| 履歴窓 | 将来窓 | AUC-PR | F1 | Precision | Recall | 訓練 | 評価 |\n")
        report.append("|--------|--------|--------|-----|-----------|--------|------|------|\n")
        
        for _, row in self.df.iterrows():
            report.append(
                f"| {row['history_window']}m | {row['future_window']} | "
                f"{row['auc_pr']:.3f} | {row['f1']:.3f} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | {row['train_samples']} | {row['eval_samples']} |\n"
            )
        
        # 主な発見
        report.append("\n## 4. 主な発見\n\n")
        
        # 履歴窓の影響
        avg_by_history = self.df.groupby('history_window')['auc_pr'].mean()
        best_history = avg_by_history.idxmax()
        report.append(f"### 履歴窓の影響\n\n")
        report.append(f"- **最適な履歴窓**: {best_history}ヶ月 (平均AUC-PR: {avg_by_history[best_history]:.3f})\n")
        for hist, score in avg_by_history.items():
            report.append(f"  - {hist}ヶ月: {score:.3f}\n")
        report.append("\n")
        
        # 将来窓の影響
        avg_by_future = self.df.groupby('future_window')['auc_pr'].mean()
        best_future = avg_by_future.idxmax()
        report.append(f"### 将来窓の影響\n\n")
        report.append(f"- **最適な将来窓**: {best_future} (平均AUC-PR: {avg_by_future[best_future]:.3f})\n")
        for fut, score in avg_by_future.items():
            report.append(f"  - {fut}: {score:.3f}\n")
        report.append("\n")
        
        # 推奨設定
        report.append("## 5. 推奨設定\n\n")
        best_overall = self.df.loc[self.df['auc_pr'].idxmax()]
        report.append(f"### 総合的に最適な設定\n\n")
        report.append(f"- **履歴窓**: {best_overall['history_window']}ヶ月\n")
        report.append(f"- **将来窓**: {best_overall['future_window']}\n")
        report.append(f"- **期待性能**:\n")
        report.append(f"  - AUC-PR: {best_overall['auc_pr']:.3f}\n")
        report.append(f"  - F1: {best_overall['f1']:.3f}\n")
        report.append(f"  - Precision: {best_overall['precision']:.3f}\n")
        report.append(f"  - Recall: {best_overall['recall']:.3f}\n\n")
        
        # ファイルに保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        logger.info(f"詳細レポートを保存: {output_path}")
    
    def generate_all_visualizations(self, output_dir: Path = None):
        """全ての可視化を生成"""
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("スライディングウィンドウ評価の可視化を開始")
        logger.info("=" * 80)
        
        # 各種可視化を生成
        self.create_enhanced_heatmaps(output_dir / "enhanced_heatmaps.png")
        self.create_trend_plots(output_dir / "trend_analysis.png")
        self.create_sample_analysis(output_dir / "sample_analysis.png")
        self.create_metrics_correlation(output_dir / "metrics_correlation.png")
        self.create_detailed_report(output_dir / "detailed_report.md")
        
        logger.info("=" * 80)
        logger.info(f"全ての可視化が完了しました: {output_dir}")
        logger.info("=" * 80)
        
        # 生成されたファイル一覧
        files = [
            "enhanced_heatmaps.png",
            "trend_analysis.png",
            "sample_analysis.png",
            "metrics_correlation.png",
            "detailed_report.md"
        ]
        
        print("\n生成されたファイル:")
        for f in files:
            print(f"  ✓ {f}")


def main():
    parser = argparse.ArgumentParser(
        description="スライディングウィンドウ評価結果の包括的可視化"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="結果ディレクトリのパス"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ（デフォルト: results-dirと同じ）"
    )
    
    args = parser.parse_args()
    
    visualizer = SlidingWindowVisualizer(args.results_dir)
    visualizer.generate_all_visualizations(args.output_dir)


if __name__ == "__main__":
    main()

