#!/usr/bin/env python3
"""
8×8 スライディングウィンドウ評価結果のヒートマップ生成

3ヶ月単位のスライディングウィンドウ評価結果（学習期間×予測期間）を
ヒートマップとして可視化します。

使用例:
    python scripts/visualization/plot_8x8_heatmap.py \
        --results importants/irl_matrix_8x8_2023q1/sliding_window_results_seq.csv \
        --output importants/irl_matrix_8x8_2023q1/heatmap.png \
        --metric auc_roc
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_path: str) -> pd.DataFrame:
    """評価結果CSVを読み込み"""
    df = pd.read_csv(results_path)
    print(f"結果を読み込みました: {results_path}")
    print(f"  行数: {len(df)}")
    print(f"  カラム: {df.columns.tolist()}")
    return df


def create_heatmap_matrix(df: pd.DataFrame, metric: str = 'auc_roc') -> np.ndarray:
    """
    評価結果から8×8行列を作成

    Args:
        df: 評価結果DataFrame
        metric: 可視化する指標 ('auc_roc', 'auc_pr', 'f1', など)

    Returns:
        8×8のnumpy配列
    """
    # 列名の互換性チェック（learning_quarters または history_months）
    if 'learning_quarters' in df.columns:
        learning_col = 'learning_quarters'
        prediction_col = 'prediction_quarters'
    elif 'history_months' in df.columns:
        # history_months を quarters に変換
        df = df.copy()
        df['learning_quarters'] = df['history_months'] // 3
        df['prediction_quarters'] = df['target_months'] // 3
        learning_col = 'learning_quarters'
        prediction_col = 'prediction_quarters'
    else:
        raise ValueError("学習期間の列が見つかりません（'learning_quarters' または 'history_months'）")

    # 学習期間と予測期間のユニーク値を取得（3ヶ月単位）
    learning_quarters = sorted(df[learning_col].unique())
    prediction_quarters = sorted(df[prediction_col].unique())

    print(f"\n学習期間（四半期）: {learning_quarters}")
    print(f"予測期間（四半期）: {prediction_quarters}")

    # 行列を初期化（NaNで埋める）
    matrix = np.full((len(learning_quarters), len(prediction_quarters)), np.nan)

    # 各組み合わせの指標値を行列に配置
    for _, row in df.iterrows():
        i = learning_quarters.index(row[learning_col])
        j = prediction_quarters.index(row[prediction_col])

        if metric in row:
            matrix[i, j] = row[metric]
        else:
            print(f"警告: 指標 '{metric}' が見つかりません。利用可能: {df.columns.tolist()}")
            return None

    return matrix, learning_quarters, prediction_quarters


def plot_heatmap(matrix: np.ndarray,
                 learning_quarters: list,
                 prediction_quarters: list,
                 metric: str,
                 output_path: str,
                 title: str = None):
    """
    ヒートマップをプロットして保存

    Args:
        matrix: 8×8のデータ行列
        learning_quarters: 学習期間のラベル
        prediction_quarters: 予測期間のラベル
        metric: 指標名
        output_path: 出力ファイルパス
        title: グラフタイトル（Noneの場合は自動生成）
    """
    # Use default font (no Japanese)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 図のサイズを設定
    fig, ax = plt.subplots(figsize=(12, 10))

    # ヒートマップを作成
    # 指標に応じてカラーマップを選択
    if metric in ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']:
        cmap = 'YlOrRd'  # 黄色→オレンジ→赤（高いほど良い）
        vmin, vmax = 0.5, 1.0
    else:
        cmap = 'viridis'
        vmin, vmax = None, None

    sns.heatmap(
        matrix,
        annot=True,  # 数値を表示
        fmt='.3f',   # 小数点3桁
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': metric.upper().replace('_', '-')},
        xticklabels=[f'{q}Q' for q in prediction_quarters],
        yticklabels=[f'{q}Q' for q in learning_quarters],
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    # ラベルとタイトル
    ax.set_xlabel('Prediction Period (Quarters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Learning Period (Quarters)', fontsize=14, fontweight='bold')

    if title is None:
        title = f'Temporal IRL Evaluation: {metric.upper().replace("_", "-")} Heatmap\n(3-month Sliding Window)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 最良値を強調表示
    max_val = np.nanmax(matrix)
    max_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)

    # 最良セルに枠を追加
    rect = plt.Rectangle(
        (max_idx[1], max_idx[0]), 1, 1,
        fill=False, edgecolor='blue', linewidth=3
    )
    ax.add_patch(rect)

    # 最良値の情報を追加
    best_learning = learning_quarters[max_idx[0]]
    best_prediction = prediction_quarters[max_idx[1]]
    info_text = (
        f'Best: {best_learning}Q learn × {best_prediction}Q predict\n'
        f'{metric.upper().replace("_", "-")}: {max_val:.3f}'
    )
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nヒートマップを保存しました: {output_path}")

    # 統計情報を出力
    print(f"\n=== {metric.upper().replace('_', '-')} Statistics ===")
    print(f"Min: {np.nanmin(matrix):.3f}")
    print(f"Max: {np.nanmax(matrix):.3f}")
    print(f"Mean: {np.nanmean(matrix):.3f}")
    print(f"Std: {np.nanstd(matrix):.3f}")
    print(f"Best: {best_learning}Q learn × {best_prediction}Q predict = {max_val:.3f}")

    return fig


def plot_multiple_metrics(df: pd.DataFrame,
                         output_dir: str,
                         metrics: list = None):
    """
    複数の指標についてヒートマップを生成

    Args:
        df: 評価結果DataFrame
        output_dir: 出力ディレクトリ
        metrics: 可視化する指標リスト（Noneの場合は主要指標すべて）
    """
    if metrics is None:
        metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']

    # 利用可能な指標のみフィルタ
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print(f"警告: 指定された指標がデータに存在しません。")
        print(f"  指定: {metrics}")
        print(f"  利用可能: {df.columns.tolist()}")
        return

    print(f"\n{len(available_metrics)}個の指標についてヒートマップを生成します: {available_metrics}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in available_metrics:
        print(f"\n--- {metric.upper().replace('_', '-')} ---")

        matrix, learning_quarters, prediction_quarters = create_heatmap_matrix(df, metric)

        if matrix is not None:
            output_path = output_dir / f'heatmap_{metric}.png'
            plot_heatmap(
                matrix,
                learning_quarters,
                prediction_quarters,
                metric,
                str(output_path)
            )


def plot_comparison_heatmap(df: pd.DataFrame,
                           output_path: str,
                           metrics: list = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall'],
                           grid_layout: bool = True):
    """
    複数指標を並べて比較表示

    Args:
        df: 評価結果DataFrame
        output_path: 出力ファイルパス
        metrics: 比較する指標リスト
        grid_layout: Trueの場合は2×3グリッド、Falseの場合は1行に並べる
    """
    # 利用可能な指標のみフィルタ
    available_metrics = [m for m in metrics if m in df.columns]

    if len(available_metrics) < 2:
        print("比較には2つ以上の指標が必要です。")
        return

    n_metrics = len(available_metrics)

    # グリッドレイアウトの設定
    if grid_layout:
        # 2行または3行のグリッド（最大6個まで）
        if n_metrics <= 3:
            nrows, ncols = 1, n_metrics
            figsize = (8*ncols, 8)
        elif n_metrics <= 6:
            nrows, ncols = 2, 3
            figsize = (24, 16)
        else:
            nrows = (n_metrics + 2) // 3
            ncols = 3
            figsize = (24, 8*nrows)
    else:
        # 1行に並べる
        nrows, ncols = 1, n_metrics
        figsize = (8*n_metrics, 8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # axesを1次元リストに変換
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for idx, metric in enumerate(available_metrics):
        matrix, learning_quarters, prediction_quarters = create_heatmap_matrix(df, metric)

        if matrix is not None:
            # カラーマップと範囲を設定
            if metric in ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']:
                cmap = 'YlOrRd'
                vmin, vmax = 0.5, 1.0
            else:
                cmap = 'viridis'
                vmin, vmax = None, None

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': metric.upper().replace('_', '-')},
                xticklabels=[f'{q}Q' for q in prediction_quarters],
                yticklabels=[f'{q}Q' for q in learning_quarters],
                ax=axes[idx],
                linewidths=0.5,
                linecolor='gray'
            )

            # 最良値を強調
            max_val = np.nanmax(matrix)
            max_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
            rect = plt.Rectangle(
                (max_idx[1], max_idx[0]), 1, 1,
                fill=False, edgecolor='blue', linewidth=3
            )
            axes[idx].add_patch(rect)

            axes[idx].set_title(
                f'{metric.upper().replace("_", "-")} (Best: {max_val:.3f})',
                fontsize=14,
                fontweight='bold'
            )
            axes[idx].set_xlabel('Prediction Period (Q)', fontsize=11)

            # Y軸ラベルは左端の列のみ
            if grid_layout and idx % ncols == 0:
                axes[idx].set_ylabel('Learning Period (Q)', fontsize=11)
            elif not grid_layout and idx == 0:
                axes[idx].set_ylabel('Learning Period (Q)', fontsize=11)
            else:
                axes[idx].set_ylabel('')

    # 余分なサブプロットを非表示
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(
        'Temporal IRL Evaluation Comparison (3-month Sliding Window)',
        fontsize=18,
        fontweight='bold',
        y=0.995 if grid_layout else 1.02
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n比較ヒートマップを保存しました: {output_path}")

    # 各指標の最良値を表示
    print(f"\n=== Best Values for Each Metric ===")
    for metric in available_metrics:
        matrix, learning_quarters, prediction_quarters = create_heatmap_matrix(df, metric)
        if matrix is not None:
            max_val = np.nanmax(matrix)
            max_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
            best_learning = learning_quarters[max_idx[0]]
            best_prediction = prediction_quarters[max_idx[1]]
            print(f"{metric.upper().replace('_', '-'):12s}: {max_val:.3f} ({best_learning}Q learn × {best_prediction}Q predict)")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='8×8スライディングウィンドウ評価結果のヒートマップ生成'
    )
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='評価結果CSVファイルのパス'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='出力画像ファイルのパス（指定しない場合は結果と同じディレクトリ）'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='auc_roc',
        choices=['auc_roc', 'auc_pr', 'f1', 'precision', 'recall'],
        help='可視化する指標（デフォルト: auc_roc）'
    )
    parser.add_argument(
        '--all-metrics',
        action='store_true',
        help='すべての主要指標についてヒートマップを生成'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='複数指標を並べて比較表示'
    )
    parser.add_argument(
        '--title',
        type=str,
        help='グラフタイトル（指定しない場合は自動生成）'
    )

    args = parser.parse_args()

    # 結果を読み込み
    df = load_results(args.results)

    # 出力パスの設定
    if args.output is None:
        results_path = Path(args.results)
        output_base = results_path.parent / results_path.stem
    else:
        output_base = Path(args.output).with_suffix('')

    # 比較モード
    if args.comparison:
        output_path = f"{output_base}_comparison.png"
        plot_comparison_heatmap(df, output_path)

    # 全指標モード
    elif args.all_metrics:
        output_dir = output_base.parent / f"{output_base.name}_heatmaps"
        plot_multiple_metrics(df, str(output_dir))

    # 単一指標モード
    else:
        output_path = f"{output_base}_{args.metric}.png"
        matrix, learning_quarters, prediction_quarters = create_heatmap_matrix(df, args.metric)

        if matrix is not None:
            plot_heatmap(
                matrix,
                learning_quarters,
                prediction_quarters,
                args.metric,
                output_path,
                title=args.title
            )


if __name__ == '__main__':
    main()
