#!/usr/bin/env python3
"""
開発者ごとの予測確率の変動を分析するスクリプト

予測期間が変わった時、各開発者の継続確率がどう変化するかを可視化する。
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_all_predictions(predictions_dir: Path) -> pd.DataFrame:
    """全ての予測結果CSVを読み込んで統合する"""
    all_dfs = []

    for csv_file in predictions_dir.glob("predictions_*.csv"):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No prediction files found in {predictions_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(all_dfs)} prediction files")
    print(f"Total predictions: {len(combined)}")

    return combined


def analyze_probability_transitions(df: pd.DataFrame, output_dir: Path):
    """予測期間による確率変動を分析"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各学習期間ごとに分析
    learning_periods = sorted(df['learning_months'].unique())

    for learning_months in learning_periods:
        df_learn = df[df['learning_months'] == learning_months].copy()

        print(f"\n{'='*60}")
        print(f"学習期間: {learning_months}ヶ月")
        print(f"{'='*60}")

        # ピボットテーブル作成: 各開発者の予測期間ごとの確率
        pivot = df_learn.pivot(
            index='reviewer_email',
            columns='prediction_months',
            values='predicted_probability'
        )

        print(f"開発者数: {len(pivot)}人")
        print(f"予測期間: {sorted(pivot.columns.tolist())}")

        # 統計情報
        print(f"\n各予測期間での平均確率:")
        for pred_months in sorted(pivot.columns):
            mean_prob = pivot[pred_months].mean()
            print(f"  {pred_months}ヶ月: {mean_prob:.4f}")

        # 確率変動の統計
        print(f"\n確率変動の分析:")
        pivot['std'] = pivot.std(axis=1)  # 各開発者の標準偏差
        pivot['range'] = pivot.max(axis=1) - pivot.min(axis=1)  # 範囲

        print(f"  確率変動（標準偏差）の平均: {pivot['std'].mean():.4f}")
        print(f"  確率変動（範囲）の平均: {pivot['range'].mean():.4f}")
        print(f"  変動が大きい開発者（std > 0.1）: {(pivot['std'] > 0.1).sum()}人")

        # 最も変動が大きい開発者トップ10
        top_volatile = pivot.nlargest(10, 'std')
        print(f"\n最も変動が大きい開発者トップ10:")
        pred_cols = [c for c in pivot.columns if isinstance(c, (int, float))]
        for idx, row in top_volatile.iterrows():
            probs = [f"{row[c]:.3f}" for c in sorted(pred_cols)]
            print(f"  {idx}: {' → '.join(probs)} (std={row['std']:.3f})")

        # ヒートマップ作成（変動が大きい開発者上位30人）
        top_30 = pivot.nlargest(30, 'std')
        heatmap_data = top_30[sorted(pred_cols)]

        plt.figure(figsize=(10, 12))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': '継続確率'},
            xticklabels=[f"{int(c)}m" for c in sorted(pred_cols)],
            yticklabels=[email[:30] for email in heatmap_data.index]
        )
        plt.title(f'開発者ごとの予測確率変動（学習期間 {learning_months}ヶ月）\n変動が大きい上位30人',
                  fontsize=14, fontweight='bold')
        plt.xlabel('予測期間', fontsize=12)
        plt.ylabel('レビュアー', fontsize=12)
        plt.tight_layout()

        heatmap_file = output_dir / f'probability_heatmap_h{learning_months}m.png'
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        print(f"\nヒートマップ保存: {heatmap_file}")
        plt.close()

        # 変動パターンの分類
        print(f"\n変動パターンの分類:")

        # パターン1: 単調増加（予測期間が長いほど確率上昇）
        monotonic_increase = 0
        # パターン2: 単調減少（予測期間が長いほど確率低下）
        monotonic_decrease = 0
        # パターン3: U字型（中間で低い）
        u_shaped = 0
        # パターン4: 逆U字型（中間で高い）
        inverse_u = 0
        # パターン5: その他
        other = 0

        for idx, row in pivot.iterrows():
            vals = [row[c] for c in sorted(pred_cols)]

            # 差分を計算
            diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]

            if all(d >= 0 for d in diffs):
                monotonic_increase += 1
            elif all(d <= 0 for d in diffs):
                monotonic_decrease += 1
            elif vals[0] > vals[len(vals)//2] < vals[-1]:  # U字
                u_shaped += 1
            elif vals[0] < vals[len(vals)//2] > vals[-1]:  # 逆U字
                inverse_u += 1
            else:
                other += 1

        print(f"  単調増加: {monotonic_increase}人 ({monotonic_increase/len(pivot)*100:.1f}%)")
        print(f"  単調減少: {monotonic_decrease}人 ({monotonic_decrease/len(pivot)*100:.1f}%)")
        print(f"  U字型: {u_shaped}人 ({u_shaped/len(pivot)*100:.1f}%)")
        print(f"  逆U字型: {inverse_u}人 ({inverse_u/len(pivot)*100:.1f}%)")
        print(f"  その他: {other}人 ({other/len(pivot)*100:.1f}%)")

        # 詳細CSV保存
        pivot_sorted = pivot.sort_values('std', ascending=False)
        csv_file = output_dir / f'probability_transitions_h{learning_months}m.csv'
        pivot_sorted.to_csv(csv_file)
        print(f"\n詳細データ保存: {csv_file}")


def plot_example_transitions(df: pd.DataFrame, output_dir: Path, n_examples: int = 10):
    """代表的な開発者の確率変動をプロット"""
    learning_periods = sorted(df['learning_months'].unique())

    for learning_months in learning_periods:
        df_learn = df[df['learning_months'] == learning_months].copy()

        # ピボット作成
        pivot = df_learn.pivot(
            index='reviewer_email',
            columns='prediction_months',
            values='predicted_probability'
        )

        pred_cols = sorted([c for c in pivot.columns if isinstance(c, (int, float))])
        pivot['std'] = pivot[pred_cols].std(axis=1)

        # 変動が大きい順に並べ替え
        pivot_sorted = pivot.sort_values('std', ascending=False)

        # 上位n人をプロット
        top_n = pivot_sorted.head(n_examples)

        plt.figure(figsize=(12, 8))

        for idx, (email, row) in enumerate(top_n.iterrows()):
            probs = [row[c] for c in pred_cols]

            # 実際のラベルを取得
            actual_label = df_learn[df_learn['reviewer_email'] == email]['true_label'].iloc[0]
            label_str = "継続" if actual_label == 1 else "離脱"

            plt.plot(pred_cols, probs, marker='o', label=f"{email[:25]}... (実際: {label_str})", linewidth=2)

        plt.xlabel('予測期間（ヶ月）', fontsize=12)
        plt.ylabel('継続確率', fontsize=12)
        plt.title(f'予測確率の変動パターン（学習期間 {learning_months}ヶ月）\n変動が大きい開発者トップ{n_examples}',
                  fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()

        plot_file = output_dir / f'probability_transitions_plot_h{learning_months}m.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"変動プロット保存: {plot_file}")
        plt.close()


def create_summary_report(df: pd.DataFrame, output_dir: Path):
    """サマリーレポート作成"""
    report_lines = []
    report_lines.append("# 予測確率の変動分析レポート\n")

    learning_periods = sorted(df['learning_months'].unique())

    for learning_months in learning_periods:
        df_learn = df[df['learning_months'] == learning_months].copy()

        pivot = df_learn.pivot(
            index='reviewer_email',
            columns='prediction_months',
            values='predicted_probability'
        )

        pred_cols = sorted([c for c in pivot.columns if isinstance(c, (int, float))])
        pivot['std'] = pivot[pred_cols].std(axis=1)
        pivot['range'] = pivot[pred_cols].max(axis=1) - pivot[pred_cols].min(axis=1)

        report_lines.append(f"\n## 学習期間: {learning_months}ヶ月\n")
        report_lines.append(f"- 開発者数: {len(pivot)}人\n")
        report_lines.append(f"- 予測期間: {pred_cols}\n")
        report_lines.append(f"\n### 確率変動の統計\n")
        report_lines.append(f"- 平均変動（標準偏差）: {pivot['std'].mean():.4f}\n")
        report_lines.append(f"- 平均変動（範囲）: {pivot['range'].mean():.4f}\n")
        report_lines.append(f"- 大きく変動する開発者（std > 0.1）: {(pivot['std'] > 0.1).sum()}人\n")

        report_lines.append(f"\n### 予測期間ごとの平均確率\n")
        for pred_months in pred_cols:
            mean_prob = pivot[pred_months].mean()
            report_lines.append(f"- {pred_months}ヶ月: {mean_prob:.4f}\n")

        # 最も変動が大きい開発者トップ5
        report_lines.append(f"\n### 最も変動が大きい開発者トップ5\n")
        top_5 = pivot.nlargest(5, 'std')

        for idx, (email, row) in enumerate(top_5.iterrows(), 1):
            probs_str = " → ".join([f"{row[c]:.3f}" for c in pred_cols])
            report_lines.append(f"{idx}. `{email}`\n")
            report_lines.append(f"   - 確率変動: {probs_str}\n")
            report_lines.append(f"   - 標準偏差: {row['std']:.4f}\n")

    report_file = output_dir / 'PROBABILITY_TRANSITION_REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"\nサマリーレポート保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction probability transitions")
    parser.add_argument(
        '--predictions-dir',
        type=str,
        required=True,
        help='Directory containing prediction CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--n-examples',
        type=int,
        default=10,
        help='Number of example developers to plot (default: 10)'
    )

    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    print("="*80)
    print("予測確率の変動分析")
    print("="*80)
    print(f"予測データディレクトリ: {predictions_dir}")
    print(f"出力ディレクトリ: {output_dir}\n")

    # データ読み込み
    df = load_all_predictions(predictions_dir)

    # 分析実行
    analyze_probability_transitions(df, output_dir)

    # プロット作成
    plot_example_transitions(df, output_dir, args.n_examples)

    # レポート作成
    create_summary_report(df, output_dir)

    print("\n" + "="*80)
    print("分析完了")
    print("="*80)
    print(f"全ての結果は以下に保存されました: {output_dir}")


if __name__ == '__main__':
    main()
