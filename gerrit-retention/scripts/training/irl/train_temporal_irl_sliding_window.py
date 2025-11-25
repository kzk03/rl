"""
時系列IRL学習とスライディングウィンドウ評価

3ヶ月単位で学習期間と予測期間を変えながらIRLモデルを訓練・評価し、
精度を行列形式で出力します。

学習期間候補: 3, 6, 9, 12, 15, 18ヶ月
予測期間候補: 3, 6, 9, 12ヶ月

出力:
- 各組み合わせでの訓練済みモデル
- 精度評価結果（AUC, PR-AUC, F1, Precision, Recall）
- 行列形式の可視化レポート
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_review_logs(csv_path: Path) -> pd.DataFrame:
    """レビューログを読み込む"""
    logger.info(f"レビューログを読み込み中: {csv_path}")
    df = pd.read_csv(csv_path)

    # 日付カラムをdatetimeに変換
    date_col = 'request_time' if 'request_time' in df.columns else 'created'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    logger.info(f"レビューログ読み込み完了: {len(df)}件")
    return df


def extract_trajectories_with_window(
    df: pd.DataFrame,
    snapshot_date: datetime,
    history_months: int,
    target_months: int,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    指定された学習期間と予測期間で軌跡を抽出

    Args:
        df: レビューログ
        snapshot_date: スナップショット日
        history_months: 学習期間（ヶ月）
        target_months: 予測期間（ヶ月）
        reviewer_col: レビュアーカラム名
        date_col: 日付カラム名

    Returns:
        train_trajectories: 訓練用軌跡リスト
        test_trajectories: テスト用軌跡リスト（予測期間のラベル付き）
    """
    history_start = snapshot_date - pd.DateOffset(months=history_months)
    target_end = snapshot_date + pd.DateOffset(months=target_months)

    # 学習期間のデータ
    history_df = df[(df[date_col] >= history_start) & (df[date_col] < snapshot_date)]

    # 予測期間のデータ
    target_df = df[(df[date_col] >= snapshot_date) & (df[date_col] < target_end)]

    # レビュアーごとにグループ化
    reviewers_in_history = set(history_df[reviewer_col].unique())

    trajectories = []

    for reviewer in reviewers_in_history:
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]
        reviewer_target = target_df[target_df[reviewer_col] == reviewer]

        # 継続ラベル: 予測期間中に活動があったかどうか
        continued = len(reviewer_target) > 0

        # 活動履歴を構築
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity_history.append({
                'type': 'review',
                'timestamp': row[date_col],
                'project': row.get('project', 'unknown'),
                'message': '',
                'lines_added': 0,
                'lines_deleted': 0,
                'files_changed': 1
            })

        # 開発者情報
        developer_info = {
            'developer_id': reviewer,
            'first_seen': reviewer_history[date_col].min(),
            'changes_authored': 0,
            'changes_reviewed': len(reviewer_history),
            'projects': reviewer_history['project'].unique().tolist() if 'project' in reviewer_history.columns else []
        }

        trajectories.append({
            'developer': developer_info,
            'activity_history': activity_history,
            'continued': continued,
            'context_date': snapshot_date,
            'reviewer': reviewer,
            'history_count': len(reviewer_history),
            'target_count': len(reviewer_target)
        })

    # 訓練とテストに分割（80/20）
    np.random.seed(42)
    np.random.shuffle(trajectories)
    split_idx = int(len(trajectories) * 0.8)

    train_trajectories = trajectories[:split_idx]
    test_trajectories = trajectories[split_idx:]

    logger.info(f"軌跡抽出完了: 訓練={len(train_trajectories)}, テスト={len(test_trajectories)}")
    logger.info(f"  継続率（訓練）: {sum(1 for t in train_trajectories if t['continued']) / len(train_trajectories):.1%}")
    logger.info(f"  継続率（テスト）: {sum(1 for t in test_trajectories if t['continued']) / len(test_trajectories):.1%}")

    return train_trajectories, test_trajectories


def train_irl_model(
    trajectories: List[Dict[str, Any]],
    config: Dict[str, Any],
    epochs: int = 30
) -> RetentionIRLSystem:
    """IRLモデルを訓練"""
    logger.info("IRLモデルを訓練中...")
    irl_system = RetentionIRLSystem(config)

    training_result = irl_system.train_irl(trajectories, epochs=epochs)

    logger.info(f"訓練完了: 最終損失={training_result['final_loss']:.4f}")
    return irl_system


def evaluate_irl_model(
    irl_system: RetentionIRLSystem,
    test_trajectories: List[Dict[str, Any]]
) -> Dict[str, float]:
    """IRLモデルを評価"""
    logger.info("IRLモデルを評価中...")

    y_true = []
    y_pred = []

    for trajectory in test_trajectories:
        developer = trajectory['developer']
        activity_history = trajectory['activity_history']
        context_date = trajectory['context_date']
        true_label = trajectory['continued']

        # 予測実行
        prediction = irl_system.predict_continuation_probability(
            developer, activity_history, context_date
        )

        y_true.append(1 if true_label else 0)
        y_pred.append(prediction['continuation_probability'])

    # 2値予測（閾値0.5）
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

    # メトリクス計算
    metrics = {}

    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred)
    except Exception:
        metrics['auc_roc'] = 0.5

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        metrics['auc_pr'] = auc(recall, precision)
    except Exception:
        metrics['auc_pr'] = 0.0

    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)

    logger.info(f"評価完了: AUC={metrics['auc_roc']:.3f}, PR-AUC={metrics['auc_pr']:.3f}, F1={metrics['f1']:.3f}")

    return metrics


def sliding_window_evaluation(
    df: pd.DataFrame,
    snapshot_date: datetime,
    history_months_list: List[int],
    target_months_list: List[int],
    output_dir: Path,
    sequence: bool = True,
    seq_len: int = 10,
    epochs: int = 30
) -> pd.DataFrame:
    """
    スライディングウィンドウ評価

    Args:
        df: レビューログ
        snapshot_date: スナップショット日
        history_months_list: 学習期間候補（ヶ月）
        target_months_list: 予測期間候補（ヶ月）
        output_dir: 出力ディレクトリ
        sequence: 時系列モードを使用するか
        seq_len: シーケンス長
        epochs: 訓練エポック数

    Returns:
        results_df: 評価結果のDataFrame
    """
    results = []

    reviewer_col = 'reviewer_email' if 'reviewer_email' in df.columns else 'email'
    date_col = 'request_time' if 'request_time' in df.columns else 'created'

    total_combinations = len(history_months_list) * len(target_months_list)
    current_idx = 0

    for history_months in history_months_list:
        for target_months in target_months_list:
            current_idx += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"組み合わせ {current_idx}/{total_combinations}: 学習={history_months}ヶ月, 予測={target_months}ヶ月")
            logger.info(f"{'='*80}")

            # 軌跡抽出
            train_traj, test_traj = extract_trajectories_with_window(
                df, snapshot_date, history_months, target_months,
                reviewer_col, date_col
            )

            if len(train_traj) == 0 or len(test_traj) == 0:
                logger.warning("軌跡が不足しているためスキップ")
                continue

            # IRL設定
            config = {
                'state_dim': 10,
                'action_dim': 4,  # 4次元: intensity, collaboration, response_speed, review_size
                'hidden_dim': 128,
                'learning_rate': 0.001,
                'sequence': sequence,
                'seq_len': seq_len
            }

            # モデル訓練
            irl_system = train_irl_model(train_traj, config, epochs=epochs)

            # モデル保存
            model_name = f"irl_h{history_months}m_t{target_months}m_{'seq' if sequence else 'nosq'}.pth"
            model_path = output_dir / "models" / model_name
            model_path.parent.mkdir(parents=True, exist_ok=True)
            irl_system.save_model(str(model_path))

            # 評価
            metrics = evaluate_irl_model(irl_system, test_traj)

            # 結果を記録
            results.append({
                'history_months': history_months,
                'target_months': target_months,
                'sequence_mode': sequence,
                'seq_len': seq_len if sequence else 0,
                'train_samples': len(train_traj),
                'test_samples': len(test_traj),
                'auc_roc': metrics['auc_roc'],
                'auc_pr': metrics['auc_pr'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'model_path': str(model_path)
            })

    # DataFrameに変換
    results_df = pd.DataFrame(results)

    # 結果を保存
    results_path = output_dir / f"sliding_window_results_{'seq' if sequence else 'nosq'}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n評価結果を保存: {results_path}")

    return results_df


def create_evaluation_matrix(results_df: pd.DataFrame, output_dir: Path, sequence: bool = True):
    """評価結果を行列形式で可視化"""
    logger.info("\n行列形式で結果を整理中...")

    # メトリクスごとに行列を作成
    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'accuracy']

    report_lines = []
    report_lines.append("="*80)
    report_lines.append(f"スライディングウィンドウ評価結果（時系列モード: {sequence}）")
    report_lines.append("="*80)
    report_lines.append("")

    for metric in metrics:
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"{metric.upper()} 行列")
        report_lines.append(f"{'='*80}")

        # ピボットテーブル作成
        pivot = results_df.pivot_table(
            values=metric,
            index='history_months',
            columns='target_months',
            aggfunc='mean'
        )

        report_lines.append("\n行: 学習期間（ヶ月）, 列: 予測期間（ヶ月）")
        report_lines.append(pivot.to_string())

        # 最良の組み合わせ
        best_idx = results_df[metric].idxmax()
        best_row = results_df.loc[best_idx]
        report_lines.append(f"\n最良の組み合わせ:")
        report_lines.append(f"  学習期間: {best_row['history_months']}ヶ月")
        report_lines.append(f"  予測期間: {best_row['target_months']}ヶ月")
        report_lines.append(f"  {metric}: {best_row[metric]:.4f}")

    report_lines.append(f"\n{'='*80}")
    report_lines.append("全体サマリー")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"総実験数: {len(results_df)}")
    report_lines.append(f"\n各メトリクスの平均値:")
    for metric in metrics:
        report_lines.append(f"  {metric}: {results_df[metric].mean():.4f} (±{results_df[metric].std():.4f})")

    # レポートを保存
    report_path = output_dir / f"evaluation_matrix_{'seq' if sequence else 'nosq'}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"\n行列形式のレポートを保存: {report_path}")

    # コンソールにも出力
    print('\n'.join(report_lines))

    return report_lines


def main():
    parser = argparse.ArgumentParser(
        description="時系列IRL学習とスライディングウィンドウ評価"
    )
    parser.add_argument(
        '--reviews',
        type=Path,
        default=ROOT / "data" / "processed" / "android_reviews_combined.csv",
        help="レビューログCSVファイル"
    )
    parser.add_argument(
        '--snapshot-date',
        type=str,
        default="2023-07-01",
        help="スナップショット日（YYYY-MM-DD）"
    )
    parser.add_argument(
        '--history-months',
        type=int,
        nargs='+',
        default=[3, 6, 9, 12, 15, 18],
        help="学習期間候補（ヶ月）"
    )
    parser.add_argument(
        '--target-months',
        type=int,
        nargs='+',
        default=[3, 6, 9, 12],
        help="予測期間候補（ヶ月）"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / "outputs" / "irl_sliding_window",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        '--sequence',
        action='store_true',
        help="時系列モードを有効化"
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=10,
        help="シーケンス長"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help="訓練エポック数"
    )

    args = parser.parse_args()

    # 出力ディレクトリ作成
    args.output.mkdir(parents=True, exist_ok=True)

    # スナップショット日をdatetimeに変換
    snapshot_date = pd.to_datetime(args.snapshot_date)

    # レビューログ読み込み
    df = load_review_logs(args.reviews)

    # スライディングウィンドウ評価
    results_df = sliding_window_evaluation(
        df,
        snapshot_date,
        args.history_months,
        args.target_months,
        args.output,
        sequence=args.sequence,
        seq_len=args.seq_len,
        epochs=args.epochs
    )

    # 行列形式で可視化
    create_evaluation_matrix(results_df, args.output, sequence=args.sequence)

    logger.info("\n✅ 全ての評価が完了しました！")


if __name__ == '__main__':
    main()
