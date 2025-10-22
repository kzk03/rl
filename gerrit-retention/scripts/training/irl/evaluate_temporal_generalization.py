#!/usr/bin/env python3
"""
時間的汎化性能評価: モデルの長期的な性能を評価

目的:
- 複数の評価期間でテスト
- 時間経過による性能劣化を分析
- 再訓練のタイミングを推奨
- 研究論文のRQ3に答える
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


def evaluate_on_multiple_periods(
    irl_system: RetentionIRLSystem,
    df: pd.DataFrame,
    eval_periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
    history_window: int,
    future_window_start: int,
    future_window_end: int,
) -> List[Dict[str, Any]]:
    """
    複数の評価期間でモデルを評価
    
    Args:
        irl_system: 訓練済みIRLシステム
        df: レビューログデータ
        eval_periods: 評価期間のリスト [(start, end), ...]
        history_window: 履歴ウィンドウ（ヶ月）
        future_window_start: 将来窓の開始（ヶ月）
        future_window_end: 将来窓の終了（ヶ月）
    
    Returns:
        各評価期間のメトリクスのリスト
    """
    results = []
    
    for i, (eval_start, eval_end) in enumerate(eval_periods, 1):
        logger.info("=" * 80)
        logger.info(f"評価期間 {i}/{len(eval_periods)}: {eval_start} ～ {eval_end}")
        logger.info("=" * 80)
        
        # 評価データを抽出
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
            logger.warning(f"評価データなし: {eval_start} ～ {eval_end}")
            continue
        
        # 評価
        metrics = evaluate_irl_model(irl_system, eval_trajectories)
        
        # 結果を記録
        result = {
            'eval_period': {
                'start': str(eval_start),
                'end': str(eval_end),
            },
            'eval_samples': len(eval_trajectories),
            'metrics': metrics,
        }
        
        results.append(result)
        
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        logger.info(f"サンプル数: {len(eval_trajectories)}")
    
    return results


def calculate_degradation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """性能劣化のメトリクスを計算"""
    
    auc_rocs = [r['metrics']['auc_roc'] for r in results]
    auc_prs = [r['metrics']['auc_pr'] for r in results]
    f1s = [r['metrics']['f1'] for r in results]
    
    # 初期性能（最初の評価期間）
    initial_auc_roc = auc_rocs[0] if auc_rocs else 0.0
    initial_auc_pr = auc_prs[0] if auc_prs else 0.0
    initial_f1 = f1s[0] if f1s else 0.0
    
    # 最終性能（最後の評価期間）
    final_auc_roc = auc_rocs[-1] if auc_rocs else 0.0
    final_auc_pr = auc_prs[-1] if auc_prs else 0.0
    final_f1 = f1s[-1] if f1s else 0.0
    
    # 劣化率
    degradation_auc_roc = ((initial_auc_roc - final_auc_roc) / initial_auc_roc * 100) if initial_auc_roc > 0 else 0.0
    degradation_auc_pr = ((initial_auc_pr - final_auc_pr) / initial_auc_pr * 100) if initial_auc_pr > 0 else 0.0
    degradation_f1 = ((initial_f1 - final_f1) / initial_f1 * 100) if initial_f1 > 0 else 0.0
    
    # 平均と標準偏差
    mean_auc_roc = np.mean(auc_rocs) if auc_rocs else 0.0
    std_auc_roc = np.std(auc_rocs) if auc_rocs else 0.0
    
    return {
        'initial': {
            'auc_roc': initial_auc_roc,
            'auc_pr': initial_auc_pr,
            'f1': initial_f1,
        },
        'final': {
            'auc_roc': final_auc_roc,
            'auc_pr': final_auc_pr,
            'f1': final_f1,
        },
        'degradation_percent': {
            'auc_roc': degradation_auc_roc,
            'auc_pr': degradation_auc_pr,
            'f1': degradation_f1,
        },
        'statistics': {
            'mean_auc_roc': mean_auc_roc,
            'std_auc_roc': std_auc_roc,
        },
    }


def plot_temporal_generalization(
    results: List[Dict[str, Any]],
    train_end: pd.Timestamp,
    output_path: Path
):
    """時間的汎化性能を可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('時間的汎化性能評価', fontsize=16, y=0.995)
    
    # 評価期間の中央日を計算（訓練終了からの月数）
    months_after_training = []
    for result in results:
        eval_start = pd.Timestamp(result['eval_period']['start'])
        eval_end = pd.Timestamp(result['eval_period']['end'])
        eval_mid = eval_start + (eval_end - eval_start) / 2
        months = (eval_mid.year - train_end.year) * 12 + (eval_mid.month - train_end.month)
        months_after_training.append(months)
    
    # AUC-ROC の推移
    ax = axes[0, 0]
    auc_rocs = [r['metrics']['auc_roc'] for r in results]
    ax.plot(months_after_training, auc_rocs, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.set_title('AUC-ROC の時系列推移', fontsize=14)
    ax.set_xlabel('訓練終了からの経過月数', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    ax.legend()
    
    # AUC-PR の推移
    ax = axes[0, 1]
    auc_prs = [r['metrics']['auc_pr'] for r in results]
    ax.plot(months_after_training, auc_prs, marker='s', linewidth=2, markersize=8, color='green')
    ax.set_title('AUC-PR の時系列推移', fontsize=14)
    ax.set_xlabel('訓練終了からの経過月数', fontsize=12)
    ax.set_ylabel('AUC-PR', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    
    # F1 Score の推移
    ax = axes[1, 0]
    f1s = [r['metrics']['f1'] for r in results]
    ax.plot(months_after_training, f1s, marker='^', linewidth=2, markersize=8, color='orange')
    ax.set_title('F1 Score の時系列推移', fontsize=14)
    ax.set_xlabel('訓練終了からの経過月数', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    
    # 劣化率
    ax = axes[1, 1]
    initial_auc_roc = auc_rocs[0]
    degradation_rates = [(initial_auc_roc - auc) / initial_auc_roc * 100 for auc in auc_rocs]
    ax.plot(months_after_training, degradation_rates, marker='D', linewidth=2, markersize=8, color='red')
    ax.set_title('性能劣化率（AUC-ROC 基準）', fontsize=14)
    ax.set_xlabel('訓練終了からの経過月数', fontsize=12)
    ax.set_ylabel('劣化率 (%)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% 劣化')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% 劣化')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"プロットを保存: {output_path}")


def print_degradation_summary(
    results: List[Dict[str, Any]],
    degradation_metrics: Dict[str, Any],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp
):
    """劣化のサマリーを表示"""
    
    print("\n" + "=" * 80)
    print("時間的汎化性能評価")
    print("=" * 80)
    print(f"\n訓練期間: {train_start} ～ {train_end}")
    print(f"評価期間数: {len(results)}")
    
    print("\n" + "-" * 80)
    print("各評価期間の性能")
    print("-" * 80)
    print(f"{'評価期間':<40} {'AUC-ROC':>10} {'AUC-PR':>10} {'F1':>10} {'サンプル':>10}")
    print("-" * 80)
    
    for result in results:
        period_str = f"{result['eval_period']['start'][:10]} ～ {result['eval_period']['end'][:10]}"
        metrics = result['metrics']
        print(f"{period_str:<40} "
              f"{metrics['auc_roc']:>10.3f} "
              f"{metrics['auc_pr']:>10.3f} "
              f"{metrics['f1']:>10.3f} "
              f"{result['eval_samples']:>10}")
    
    print("-" * 80)
    
    # 劣化サマリー
    print("\n" + "-" * 80)
    print("性能劣化のサマリー")
    print("-" * 80)
    
    deg = degradation_metrics
    print(f"\n初期性能（最初の評価期間）:")
    print(f"  AUC-ROC: {deg['initial']['auc_roc']:.3f}")
    print(f"  AUC-PR:  {deg['initial']['auc_pr']:.3f}")
    print(f"  F1:      {deg['initial']['f1']:.3f}")
    
    print(f"\n最終性能（最後の評価期間）:")
    print(f"  AUC-ROC: {deg['final']['auc_roc']:.3f}")
    print(f"  AUC-PR:  {deg['final']['auc_pr']:.3f}")
    print(f"  F1:      {deg['final']['f1']:.3f}")
    
    print(f"\n劣化率:")
    print(f"  AUC-ROC: {deg['degradation_percent']['auc_roc']:>6.1f}%")
    print(f"  AUC-PR:  {deg['degradation_percent']['auc_pr']:>6.1f}%")
    print(f"  F1:      {deg['degradation_percent']['f1']:>6.1f}%")
    
    print(f"\n統計:")
    print(f"  平均 AUC-ROC: {deg['statistics']['mean_auc_roc']:.3f} ± {deg['statistics']['std_auc_roc']:.3f}")
    
    # 推奨事項
    print("\n" + "-" * 80)
    print("推奨事項")
    print("-" * 80)
    
    deg_percent = deg['degradation_percent']['auc_roc']
    if deg_percent < 5:
        print("✅ 性能劣化は軽微（<5%）。現在のモデルを継続使用可能。")
    elif deg_percent < 10:
        print("⚠️  性能劣化が見られる（5-10%）。6ヶ月以内に再訓練を推奨。")
    else:
        print("❌ 顕著な性能劣化（>10%）。すぐに再訓練が必要。")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='時間的汎化性能評価: モデルの長期的な性能を評価'
    )
    
    # データ設定
    parser.add_argument(
        '--reviews',
        type=Path,
        required=True,
        help='レビューログCSVファイル'
    )
    
    # 訓練期間
    parser.add_argument(
        '--train-start',
        type=str,
        required=True,
        help='訓練期間の開始日'
    )
    parser.add_argument(
        '--train-end',
        type=str,
        required=True,
        help='訓練期間の終了日'
    )
    
    # 評価期間（複数）
    parser.add_argument(
        '--eval-periods',
        type=str,
        nargs='+',
        help='評価期間のリスト（形式: "開始日,終了日"）'
    )
    parser.add_argument(
        '--eval-interval-months',
        type=int,
        default=6,
        help='自動生成する評価期間の間隔（ヶ月、デフォルト: 6）'
    )
    parser.add_argument(
        '--num-eval-periods',
        type=int,
        default=4,
        help='自動生成する評価期間の数（デフォルト: 4）'
    )
    
    # ウィンドウ設定
    parser.add_argument(
        '--history-window',
        type=int,
        default=6,
        help='履歴ウィンドウ（ヶ月、デフォルト: 6）'
    )
    parser.add_argument(
        '--future-window-start',
        type=int,
        default=0,
        help='将来窓の開始（ヶ月、デフォルト: 0）'
    )
    parser.add_argument(
        '--future-window-end',
        type=int,
        default=1,
        help='将来窓の終了（ヶ月、デフォルト: 1）'
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
    
    # モデル読み込み（オプション）
    parser.add_argument(
        '--model',
        type=Path,
        help='訓練済みモデルのパス（指定しない場合は新規訓練）'
    )
    
    # 出力設定
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/temporal_generalization'),
        help='出力ディレクトリ'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 訓練期間をパース
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    
    # 評価期間をパース
    if args.eval_periods:
        # 手動指定
        eval_periods = []
        for period_str in args.eval_periods:
            start_str, end_str = period_str.split(',')
            eval_periods.append((pd.Timestamp(start_str), pd.Timestamp(end_str)))
    else:
        # 自動生成
        eval_periods = []
        for i in range(args.num_eval_periods):
            eval_start = train_end + pd.DateOffset(months=i * args.eval_interval_months)
            eval_end = eval_start + pd.DateOffset(months=args.eval_interval_months)
            eval_periods.append((eval_start, eval_end))
    
    logger.info("=" * 80)
    logger.info("時間的汎化性能評価")
    logger.info("=" * 80)
    logger.info(f"訓練期間: {train_start} ～ {train_end}")
    logger.info(f"評価期間数: {len(eval_periods)}")
    for i, (start, end) in enumerate(eval_periods, 1):
        logger.info(f"  {i}. {start} ～ {end}")
    logger.info("=" * 80)
    
    # データ読み込み
    df = load_review_logs(args.reviews)
    
    # モデルを訓練または読み込み
    if args.model and args.model.exists():
        logger.info(f"モデルを読み込み: {args.model}")
        irl_system = RetentionIRLSystem.load_model(args.model)
    else:
        logger.info("新規にモデルを訓練")
        
        # 訓練データ抽出
        train_trajectories = extract_temporal_trajectories_within_training_period(
            df=df,
            train_start=train_start,
            train_end=train_end,
            history_window_months=args.history_window,
            future_window_start_months=args.future_window_start,
            future_window_end_months=args.future_window_end,
            sampling_interval_months=1,
        )
        
        # IRL設定
        irl_config = {
            'state_dim': 10,
            'action_dim': 5,
            'hidden_dim': 64,
            'lstm_hidden': 128,
            'sequence': args.sequence,
            'seq_len': args.seq_len,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        
        # 訓練
        irl_system = train_irl_model(
            trajectories=train_trajectories,
            config=irl_config,
            epochs=args.epochs
        )
        
        # モデル保存
        model_path = args.output / 'irl_model.pth'
        irl_system.save_model(model_path)
        logger.info(f"モデルを保存: {model_path}")
    
    # 複数の評価期間で評価
    results = evaluate_on_multiple_periods(
        irl_system=irl_system,
        df=df,
        eval_periods=eval_periods,
        history_window=args.history_window,
        future_window_start=args.future_window_start,
        future_window_end=args.future_window_end,
    )
    
    # 劣化メトリクスを計算
    degradation_metrics = calculate_degradation_metrics(results)
    
    # 結果を保存
    output_data = {
        'train_period': {
            'start': str(train_start),
            'end': str(train_end),
        },
        'windows': {
            'history_months': args.history_window,
            'future_start_months': args.future_window_start,
            'future_end_months': args.future_window_end,
        },
        'eval_periods_count': len(eval_periods),
        'results': results,
        'degradation_metrics': degradation_metrics,
    }
    
    results_path = args.output / 'temporal_generalization_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"結果を保存: {results_path}")
    
    # サマリーを表示
    print_degradation_summary(results, degradation_metrics, train_start, train_end)
    
    # プロットを作成
    plot_path = args.output / 'temporal_generalization_plot.png'
    plot_temporal_generalization(results, train_end, plot_path)
    
    # CSVとして保存
    csv_data = []
    for result in results:
        csv_data.append({
            'eval_start': result['eval_period']['start'],
            'eval_end': result['eval_period']['end'],
            'auc_roc': result['metrics']['auc_roc'],
            'auc_pr': result['metrics']['auc_pr'],
            'f1': result['metrics']['f1'],
            'precision': result['metrics']['precision'],
            'recall': result['metrics']['recall'],
            'samples': result['eval_samples'],
        })
    
    csv_path = args.output / 'temporal_generalization_metrics.csv'
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    logger.info(f"CSVを保存: {csv_path}")
    
    logger.info("=" * 80)
    logger.info("時間的汎化性能評価完了")
    logger.info(f"結果: {args.output}/")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

