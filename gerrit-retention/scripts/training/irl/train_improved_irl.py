#!/usr/bin/env python3
"""
改良版IRL訓練スクリプト

未活用の重要特徴量を統合:
- 活動頻度の多期間比較（7日/30日/90日）
- レビュー負荷指標と集中度
- path類似度と相互作用履歴

使用例:
    # 基本的な使用方法（拡張特徴量を使用）
    uv run python scripts/training/irl/train_improved_irl.py \
        --reviews data/review_requests_openstack_no_bots.csv \
        --snapshot-date 2023-01-01 \
        --history-months 12 \
        --target-months 6 \
        --sequence --seq-len 15 --epochs 30 \
        --output importants/irl_improved_12m_6m

    # 複数の期間で実験
    uv run python scripts/training/irl/train_improved_irl.py \
        --reviews data/review_requests_openstack_no_bots.csv \
        --snapshot-date 2023-01-01 \
        --history-months 6 12 \
        --target-months 3 6 \
        --sequence --seq-len 15 --epochs 30 \
        --output importants/irl_improved_matrix
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from gerrit_retention.rl_prediction.enhanced_feature_extractor import EnhancedFeatureExtractor
from gerrit_retention.rl_prediction.enhanced_retention_irl_system import EnhancedRetentionIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_concentration_score(load_7d: float, load_30d: float) -> float:
    """
    レビュー依頼の集中度を計算

    短期間に多数の依頼が集中している場合、スコアが高くなる
    """
    if load_30d == 0:
        return 0.0

    # 7日間の負荷が30日間平均より高い場合、集中度が高い
    concentration = (load_7d * 30) / (7 * load_30d)

    # 1.5倍以上なら集中している（スコア > 1.0）
    return max(concentration - 1.0, 0.0)


def extract_trajectories_with_enhanced_features(
    df: pd.DataFrame,
    snapshot_date: datetime,
    history_months: int,
    target_months: int,
    fixed_reviewers: set = None
) -> tuple:
    """拡張特徴量を含む軌跡を抽出"""

    # 学習期間と予測期間
    learning_start = snapshot_date - timedelta(days=history_months * 30)
    learning_end = snapshot_date
    target_start = snapshot_date
    target_end = snapshot_date + timedelta(days=target_months * 30)

    logger.info(f"学習期間: {learning_start.date()} ~ {learning_end.date()}")
    logger.info(f"予測期間: {target_start.date()} ~ {target_end.date()}")

    # 固定対象レビュアーでフィルタ
    if fixed_reviewers:
        df_learning = df[
            (df['request_time'] >= learning_start) &
            (df['request_time'] < learning_end) &
            (df['reviewer_email'].isin(fixed_reviewers))
        ]
    else:
        df_learning = df[
            (df['request_time'] >= learning_start) &
            (df['request_time'] < learning_end)
        ]

    df_target = df[
        (df['request_time'] >= target_start) &
        (df['request_time'] < target_end)
    ]

    # レビュアーごとにグループ化
    trajectories = []
    skipped_no_history = 0

    for reviewer_email, group in df_learning.groupby('reviewer_email'):
        # 活動履歴
        activity_history = []
        for _, row in group.iterrows():
            activity = {
                'timestamp': row['request_time'],
                'type': 'review',
                'change_insertions': row.get('change_insertions', 0),
                'change_deletions': row.get('change_deletions', 0),
                'change_files_count': row.get('change_files_count', 1),
                'response_latency_days': row.get('response_latency_days', 0.0),
                'message': row.get('subject', ''),
                'project': row.get('project', '')
            }
            activity_history.append(activity)

        if not activity_history:
            skipped_no_history += 1
            continue

        # 開発者情報（最新のレコードから）
        latest_row = group.iloc[-1]

        # 拡張特徴量の抽出
        developer = {
            'developer_id': reviewer_email,
            'reviewer_email': reviewer_email,
            'changes_authored': 0,
            'changes_reviewed': len(group),
            'projects': group['project'].unique().tolist(),
            'first_seen': group['request_time'].min().isoformat(),
            'last_activity': group['request_time'].max().isoformat(),

            # === A1: 活動頻度の多期間比較 ===
            'reviewer_past_reviews_7d': latest_row.get('reviewer_past_reviews_7d', 0),
            'reviewer_past_reviews_30d': latest_row.get('reviewer_past_reviews_30d', 0),
            'reviewer_past_reviews_90d': latest_row.get('reviewer_past_reviews_90d', 0),
            'reviewer_past_reviews_180d': latest_row.get('reviewer_past_reviews_180d', 0),

            # === B1: レビュー負荷指標 ===
            'reviewer_assignment_load_7d': latest_row.get('reviewer_assignment_load_7d', 0),
            'reviewer_assignment_load_30d': latest_row.get('reviewer_assignment_load_30d', 0),
            'reviewer_assignment_load_180d': latest_row.get('reviewer_assignment_load_180d', 0),

            # 集中度スコアを追加計算
            'review_concentration_score': calculate_concentration_score(
                latest_row.get('reviewer_assignment_load_7d', 0) / 7.0,
                latest_row.get('reviewer_assignment_load_30d', 0) / 30.0
            ),

            # === C1: 相互作用履歴 ===
            'owner_reviewer_past_interactions_180d': latest_row.get('owner_reviewer_past_interactions_180d', 0),
            'owner_reviewer_project_interactions_180d': latest_row.get('owner_reviewer_project_interactions_180d', 0),
            'owner_reviewer_past_assignments_180d': latest_row.get('owner_reviewer_past_assignments_180d', 0),

            # === D1: path類似度 ===
            'path_jaccard_files_project': latest_row.get('path_jaccard_files_project', 0.0),
            'path_jaccard_dir1_project': latest_row.get('path_jaccard_dir1_project', 0.0),
            'path_jaccard_dir2_project': latest_row.get('path_jaccard_dir2_project', 0.0),
            'path_overlap_files_project': latest_row.get('path_overlap_files_project', 0.0),
            'path_overlap_dir1_project': latest_row.get('path_overlap_dir1_project', 0.0),
            'path_overlap_dir2_project': latest_row.get('path_overlap_dir2_project', 0.0),

            # その他の有用な特徴量
            'response_latency_days': latest_row.get('response_latency_days', 0.0),
            'reviewer_past_response_rate_180d': latest_row.get('reviewer_past_response_rate_180d', 1.0),
            'reviewer_tenure_days': latest_row.get('reviewer_tenure_days', 0),
        }

        # 継続ラベル: 予測期間中に活動があったか
        target_activities = df_target[df_target['reviewer_email'] == reviewer_email]
        continued = len(target_activities) > 0

        trajectories.append({
            'developer': developer,
            'activity_history': activity_history,
            'continued': continued,
            'context_date': snapshot_date
        })

    logger.info(f"抽出した軌跡数: {len(trajectories)}")
    logger.info(f"学習期間に活動なしでスキップ: {skipped_no_history}")

    # 継続率を計算
    continuation_rate = sum(1 for t in trajectories if t['continued']) / len(trajectories) if trajectories else 0
    logger.info(f"継続率: {continuation_rate:.1%}")

    return trajectories, continuation_rate


def train_and_evaluate(
    trajectories: list,
    config: dict,
    output_dir: Path,
    history_months: int,
    target_months: int
):
    """訓練と評価"""

    # 訓練/テスト分割
    train_trajectories, test_trajectories = train_test_split(
        trajectories, test_size=0.2, random_state=42
    )

    logger.info(f"訓練データ: {len(train_trajectories)}件")
    logger.info(f"テストデータ: {len(test_trajectories)}件")

    # IRLシステムの初期化
    irl_system = EnhancedRetentionIRLSystem(config)

    # 訓練
    logger.info("IRL訓練を開始...")
    train_result = irl_system.train_irl(train_trajectories, epochs=config.get('epochs', 30))

    # テスト評価
    logger.info("テストデータで評価...")
    test_predictions = []
    test_labels = []

    for trajectory in test_trajectories:
        result = irl_system.predict_continuation_probability(
            developer=trajectory['developer'],
            activity_history=trajectory['activity_history'],
            context_date=trajectory['context_date']
        )

        test_predictions.append(result['continuation_probability'])
        test_labels.append(1 if trajectory['continued'] else 0)

    # メトリクス計算
    auc_roc = roc_auc_score(test_labels, test_predictions)
    auc_pr = average_precision_score(test_labels, test_predictions)

    # F1スコア（閾値0.5）
    binary_predictions = [1 if p >= 0.5 else 0 for p in test_predictions]
    f1 = f1_score(test_labels, binary_predictions)

    # 継続率
    train_continuation_rate = sum(1 for t in train_trajectories if t['continued']) / len(train_trajectories)
    test_continuation_rate = sum(1 for t in test_trajectories if t['continued']) / len(test_trajectories)

    results = {
        'history_months': history_months,
        'target_months': target_months,
        'train_size': len(train_trajectories),
        'test_size': len(test_trajectories),
        'train_continuation_rate': train_continuation_rate,
        'test_continuation_rate': test_continuation_rate,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1_score': f1,
        'train_losses': train_result.get('losses', []),
        'config': config
    }

    logger.info(f"=== 評価結果 ===")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"AUC-PR: {auc_pr:.4f}")
    logger.info(f"F1スコア: {f1:.4f}")
    logger.info(f"訓練継続率: {train_continuation_rate:.1%}")
    logger.info(f"テスト継続率: {test_continuation_rate:.1%}")

    # モデルと結果を保存
    model_filename = f"irl_h{history_months}m_t{target_months}m_improved.pth"
    irl_system.save_model(str(output_dir / 'models' / model_filename))

    results_filename = f"results_h{history_months}m_t{target_months}m.json"
    with open(output_dir / results_filename, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='改良版IRL訓練（拡張特徴量を使用）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--reviews', type=str, required=True,
                        help='レビューリクエストCSVファイル')
    parser.add_argument('--snapshot-date', type=str, required=True,
                        help='スナップショット日（YYYY-MM-DD）')
    parser.add_argument('--history-months', type=int, nargs='+', default=[12],
                        help='学習期間（月単位）')
    parser.add_argument('--target-months', type=int, nargs='+', default=[6],
                        help='予測期間（月単位）')
    parser.add_argument('--sequence', action='store_true',
                        help='時系列モード（LSTM使用）')
    parser.add_argument('--seq-len', type=int, default=15,
                        help='シーケンス長')
    parser.add_argument('--epochs', type=int, default=30,
                        help='訓練エポック数')
    parser.add_argument('--output', type=str, required=True,
                        help='出力ディレクトリ')

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)

    # データ読み込み
    logger.info(f"データ読み込み: {args.reviews}")
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])

    snapshot_date = datetime.strptime(args.snapshot_date, '%Y-%m-%d')
    logger.info(f"スナップショット日: {snapshot_date.date()}")

    # 全結果を保存
    all_results = []

    # 各期間組み合わせで実験
    for history_months in args.history_months:
        for target_months in args.target_months:
            logger.info(f"\n{'='*60}")
            logger.info(f"学習期間: {history_months}ヶ月, 予測期間: {target_months}ヶ月")
            logger.info(f"{'='*60}")

            # 軌跡抽出
            trajectories, continuation_rate = extract_trajectories_with_enhanced_features(
                df, snapshot_date, history_months, target_months
            )

            if len(trajectories) < 10:
                logger.warning(f"軌跡が不足しているためスキップ（{len(trajectories)}件）")
                continue

            # 設定
            config = {
                'state_dim': 32,  # 拡張特徴量
                'action_dim': 9,  # 拡張行動特徴量
                'hidden_dim': 128,
                'learning_rate': 0.001,
                'sequence': args.sequence,
                'seq_len': args.seq_len,
                'epochs': args.epochs
            }

            # 訓練と評価
            result = train_and_evaluate(
                trajectories, config, output_dir, history_months, target_months
            )
            all_results.append(result)

    # サマリーを保存
    summary_file = output_dir / 'summary_improved.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n全実験完了。結果を保存: {summary_file}")

    # マトリクス形式のレポート作成
    if len(args.history_months) > 1 or len(args.target_months) > 1:
        create_matrix_report(all_results, output_dir)


def create_matrix_report(results: list, output_dir: Path):
    """マトリクス形式のレポートを作成"""

    report_lines = []
    report_lines.append("# 改良版IRL評価結果マトリクス\n")
    report_lines.append(f"**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**特徴量**: 拡張版（32次元状態 + 9次元行動）\n")
    report_lines.append("\n## AUC-ROC マトリクス\n")
    report_lines.append("| 学習期間 \\ 予測期間 | " + " | ".join([f"{r['target_months']}m" for r in results[:len(set(r['target_months'] for r in results))]]) + " |")
    report_lines.append("|" + "----|" * (len(set(r['target_months'] for r in results)) + 1))

    # 学習期間ごとにグループ化
    from collections import defaultdict
    matrix = defaultdict(dict)
    for r in results:
        matrix[r['history_months']][r['target_months']] = r

    for h_months in sorted(matrix.keys()):
        row = [f"**{h_months}m**"]
        for t_months in sorted(set(r['target_months'] for r in results)):
            if t_months in matrix[h_months]:
                auc = matrix[h_months][t_months]['auc_roc']
                row.append(f"{auc:.3f}")
            else:
                row.append("-")
        report_lines.append("| " + " | ".join(row) + " |")

    report_file = output_dir / 'IMPROVED_REPORT.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"マトリクスレポートを作成: {report_file}")


if __name__ == '__main__':
    main()
