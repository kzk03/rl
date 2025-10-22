#!/usr/bin/env python3
"""
スライディングウィンドウ評価: 履歴窓 × 将来窓の全組み合わせを評価

目的:
- 最適な履歴窓と将来窓の組み合わせを発見
- 結果を行列形式で出力・可視化
- 研究論文のRQ1に答える
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

# 既存の関数をインポート
sys.path.append(str(ROOT / "scripts" / "training" / "irl"))
from train_irl_within_training_period import (
    evaluate_irl_model,
    extract_temporal_trajectories_within_training_period,
    load_review_logs,
    train_irl_model,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_experiment(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    history_window: int,
    future_window_start: int,
    future_window_end: int,
    epochs: int,
    sequence: bool,
    seq_len: int,
    output_dir: Path
) -> Dict[str, Any]:
    """
    単一の実験を実行
    
    Returns:
        実験結果（メトリクスを含む）
    """
    logger.info("=" * 80)
    logger.info(f"実験: 履歴={history_window}m, 将来={future_window_start}-{future_window_end}m")
    logger.info("=" * 80)
    
    # 訓練データ抽出
    train_trajectories = extract_temporal_trajectories_within_training_period(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=history_window,
        future_window_start_months=future_window_start,
        future_window_end_months=future_window_end,
        sampling_interval_months=1,
    )
    
    if not train_trajectories:
        logger.warning("訓練データが見つかりません")
        return None
    
    # 評価データ抽出
    eval_trajectories = extract_temporal_trajectories_within_training_period(
        df=df,
        train_start=eval_start,
        train_end=eval_end,
        history_window_months=history_window,
        future_window_start_months=future_window_start,
        future_window_end_months=future_window_end,
        sampling_interval_months=1,
    )
    
    if not eval_trajectories:
        logger.warning("評価データが見つかりません")
        return None
    
    # IRL設定
    irl_config = {
        'state_dim': 10,
        'action_dim': 5,
        'hidden_dim': 64,
        'lstm_hidden': 128,
        'sequence': sequence,
        'seq_len': seq_len,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # 訓練
    irl_system = train_irl_model(
        trajectories=train_trajectories,
        config=irl_config,
        epochs=epochs
    )
    
    # モデル保存
    model_path = output_dir / f'irl_h{history_window}m_f{future_window_start}_{future_window_end}m.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    irl_system.save_model(model_path)
    
    # 評価
    metrics = evaluate_irl_model(irl_system, eval_trajectories)
    
    # 結果
    result = {
        'history_window': history_window,
        'future_window_start': future_window_start,
        'future_window_end': future_window_end,
        'future_window_label': f"{future_window_start}-{future_window_end}m",
        'train_samples': len(train_trajectories),
        'eval_samples': len(eval_trajectories),
        'metrics': metrics,
        'model_path': str(model_path),
    }
    
    return result


def create_result_matrices(results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """結果を行列形式に変換"""
    
    # ユニークな履歴窓と将来窓を取得
    history_windows = sorted(set(r['history_window'] for r in results))
    future_window_labels = sorted(
        set(r['future_window_label'] for r in results),
        key=lambda x: int(x.split('-')[0])
    )
    
    # メトリクスごとに行列を作成
    metrics_names = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
    matrices = {}
    
    for metric_name in metrics_names:
        matrix = pd.DataFrame(
            index=[f"{h}m" for h in history_windows],
            columns=future_window_labels
        )
        
        for result in results:
            h = f"{result['history_window']}m"
            f = result['future_window_label']
            value = result['metrics'][metric_name]
            matrix.loc[h, f] = value
        
        matrices[metric_name] = matrix.astype(float)
    
    return matrices


def plot_result_matrices(
    matrices: Dict[str, pd.DataFrame],
    output_path: Path
):
    """結果行列をヒートマップで可視化"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('スライディングウィンドウ評価: 履歴窓 × 将来窓', fontsize=16, y=0.995)
    
    metrics_info = [
        ('auc_roc', 'AUC-ROC', 0, 0),
        ('auc_pr', 'AUC-PR', 0, 1),
        ('f1', 'F1 Score', 0, 2),
        ('precision', 'Precision', 1, 0),
        ('recall', 'Recall', 1, 1),
    ]
    
    for metric_name, title, row, col in metrics_info:
        ax = axes[row, col]
        matrix = matrices[metric_name]
        
        # ヒートマップ
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': 'Score'}
        )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Future Window', fontsize=12)
        ax.set_ylabel('History Window', fontsize=12)
    
    # 最後のサブプロット（サンプル数）
    ax = axes[1, 2]
    ax.axis('off')
    
    # サマリーテキスト
    best_auc_roc = matrices['auc_roc'].max().max()
    best_idx = matrices['auc_roc'].stack().idxmax()
    best_f1 = matrices['f1'].max().max()
    best_f1_idx = matrices['f1'].stack().idxmax()
    
    summary_text = f"""
    Summary:
    
    Best AUC-ROC:
      {best_auc_roc:.3f}
      History: {best_idx[0]}
      Future: {best_idx[1]}
    
    Best F1:
      {best_f1:.3f}
      History: {best_f1_idx[0]}
      Future: {best_f1_idx[1]}
    
    Total Experiments:
      {len(matrices['auc_roc'].stack())}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"ヒートマップを保存: {output_path}")


def print_result_matrices(matrices: Dict[str, pd.DataFrame]):
    """結果行列をテキスト形式で表示"""
    
    print("\n" + "=" * 80)
    print("スライディングウィンドウ評価結果")
    print("=" * 80)
    
    for metric_name, matrix in matrices.items():
        print(f"\n{metric_name.upper()} 行列:")
        print("-" * 80)
        print(matrix.to_string())
        print()
        
        # 最良の組み合わせ
        best_value = matrix.max().max()
        best_idx = matrix.stack().idxmax()
        print(f"最良: {best_value:.3f} (履歴={best_idx[0]}, 将来={best_idx[1]})")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='スライディングウィンドウ評価: 履歴窓 × 将来窓の全組み合わせ'
    )
    
    # データ設定
    parser.add_argument(
        '--reviews',
        type=Path,
        required=True,
        help='レビューログCSVファイル'
    )
    
    # 期間設定
    parser.add_argument(
        '--train-start',
        type=str,
        required=True,
        help='学習期間の開始日'
    )
    parser.add_argument(
        '--train-end',
        type=str,
        required=True,
        help='学習期間の終了日'
    )
    parser.add_argument(
        '--eval-start',
        type=str,
        help='評価期間の開始日（デフォルト=train-end）'
    )
    parser.add_argument(
        '--eval-end',
        type=str,
        help='評価期間の終了日（デフォルト=eval-start+12m）'
    )
    
    # ウィンドウ設定
    parser.add_argument(
        '--history-windows',
        type=int,
        nargs='+',
        default=[3, 6, 9, 12],
        help='履歴ウィンドウのリスト（ヶ月、デフォルト: 3 6 9 12）'
    )
    parser.add_argument(
        '--future-windows',
        type=str,
        nargs='+',
        default=['0-1', '1-3', '3-6', '6-12'],
        help='将来窓のリスト（開始-終了、デフォルト: 0-1 1-3 3-6 6-12）'
    )
    
    # 訓練設定
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='訓練エポック数'
    )
    parser.add_argument(
        '--sequence',
        action='store_true',
        help='時系列モード（LSTM）を有効化'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=15,
        help='シーケンス長'
    )
    
    # 出力設定
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/sliding_window_evaluation'),
        help='出力ディレクトリ'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 期間をパース
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    
    if args.eval_start:
        eval_start = pd.Timestamp(args.eval_start)
    else:
        eval_start = train_end
    
    if args.eval_end:
        eval_end = pd.Timestamp(args.eval_end)
    else:
        eval_end = eval_start + pd.DateOffset(months=12)
    
    # 将来窓をパース
    future_windows = []
    for fw_str in args.future_windows:
        parts = fw_str.split('-')
        future_windows.append((int(parts[0]), int(parts[1])))
    
    logger.info("=" * 80)
    logger.info("スライディングウィンドウ評価")
    logger.info("=" * 80)
    logger.info(f"訓練期間: {train_start} ～ {train_end}")
    logger.info(f"評価期間: {eval_start} ～ {eval_end}")
    logger.info(f"履歴窓: {args.history_windows}")
    logger.info(f"将来窓: {future_windows}")
    logger.info(f"総実験数: {len(args.history_windows) * len(future_windows)}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # 全組み合わせで実験
    results = []
    total_experiments = len(args.history_windows) * len(future_windows)
    current_experiment = 0
    
    for history_window in args.history_windows:
        for future_start, future_end in future_windows:
            current_experiment += 1
            
            logger.info("")
            logger.info(f"実験 {current_experiment}/{total_experiments}")
            
            result = run_single_experiment(
                df=df,
                train_start=train_start,
                train_end=train_end,
                eval_start=eval_start,
                eval_end=eval_end,
                history_window=history_window,
                future_window_start=future_start,
                future_window_end=future_end,
                epochs=args.epochs,
                sequence=args.sequence,
                seq_len=args.seq_len,
                output_dir=args.output / 'models'
            )
            
            if result:
                results.append(result)
    
    # 結果を保存
    results_path = args.output / 'all_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"全結果を保存: {results_path}")
    
    # 行列形式に変換
    matrices = create_result_matrices(results)
    
    # テキスト形式で表示
    print_result_matrices(matrices)
    
    # CSVとして保存
    for metric_name, matrix in matrices.items():
        csv_path = args.output / f'{metric_name}_matrix.csv'
        matrix.to_csv(csv_path)
        logger.info(f"{metric_name}行列を保存: {csv_path}")
    
    # ヒートマップを作成
    heatmap_path = args.output / 'sliding_window_heatmaps.png'
    plot_result_matrices(matrices, heatmap_path)
    
    logger.info("=" * 80)
    logger.info("スライディングウィンドウ評価完了")
    logger.info(f"結果: {args.output}/")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

