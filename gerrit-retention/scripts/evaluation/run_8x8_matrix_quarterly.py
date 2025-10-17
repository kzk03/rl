#!/usr/bin/env python3
"""
3ヶ月単位スライディングウィンドウで2年間の8×8マトリクス評価

学習期間・予測期間を3ヶ月単位で変化させ、2年間（8期間）の全組み合わせで評価

使用例:
    # 基本的な使用方法（2023年Q1を開始点として2年間評価）
    uv run python scripts/evaluation/run_8x8_matrix_quarterly.py \
        --reviews data/review_requests_openstack_no_bots.csv \
        --start-date 2023-01-01 \
        --sequence --seq-len 15 --epochs 30 \
        --output importants/irl_matrix_8x8_2023q1

    # 拡張特徴量を使用
    uv run python scripts/evaluation/run_8x8_matrix_quarterly.py \
        --reviews data/review_requests_openstack_no_bots.csv \
        --start-date 2023-01-01 \
        --use-enhanced-features \
        --sequence --seq-len 15 --epochs 30 \
        --output importants/irl_matrix_8x8_enhanced
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from gerrit_retention.rl_prediction.enhanced_feature_extractor import EnhancedFeatureExtractor
from gerrit_retention.rl_prediction.enhanced_retention_irl_system import EnhancedRetentionIRLSystem
from gerrit_retention.rl_prediction.path_similarity import hierarchical_cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_trajectories_quarterly(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_quarters: int,
    prediction_quarters: int,
    use_enhanced_features: bool = False
) -> tuple:
    """
    3ヶ月単位で軌跡を抽出

    Args:
        df: レビューデータ
        snapshot_date: スナップショット日
        learning_quarters: 学習期間（四半期数）
        prediction_quarters: 予測期間（四半期数）
        use_enhanced_features: 拡張特徴量を使用するか

    Returns:
        (trajectories, continuation_rate)
    """
    # 学習期間と予測期間（3ヶ月 = 1四半期）
    learning_months = learning_quarters * 3
    prediction_months = prediction_quarters * 3

    learning_start = snapshot_date - timedelta(days=learning_months * 30)
    learning_end = snapshot_date
    target_start = snapshot_date
    target_end = snapshot_date + timedelta(days=prediction_months * 30)

    logger.info(f"学習期間: {learning_start.date()} ~ {learning_end.date()} ({learning_quarters}Q)")
    logger.info(f"予測期間: {target_start.date()} ~ {target_end.date()} ({prediction_quarters}Q)")

    # データフィルタ
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

    for reviewer_email, group in df_learning.groupby('reviewer_email'):
        # 活動履歴
        activity_history = []
        changed_paths = []

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

            # パス情報を収集
            if 'changed_files' in row and pd.notna(row['changed_files']):
                changed_paths.append(row['changed_files'])

        if not activity_history:
            continue

        # 開発者情報
        latest_row = group.iloc[-1]

        developer = {
            'developer_id': reviewer_email,
            'reviewer_email': reviewer_email,
            'changes_authored': 0,
            'changes_reviewed': len(group),
            'projects': group['project'].unique().tolist(),
            'first_seen': group['request_time'].min().isoformat(),
            'last_activity': group['request_time'].max().isoformat(),
        }

        # 拡張特徴量の追加
        if use_enhanced_features:
            developer.update({
                # 活動頻度（多期間）
                'reviewer_past_reviews_7d': latest_row.get('reviewer_past_reviews_7d', 0),
                'reviewer_past_reviews_30d': latest_row.get('reviewer_past_reviews_30d', 0),
                'reviewer_past_reviews_90d': latest_row.get('reviewer_past_reviews_90d', 0),
                'reviewer_past_reviews_180d': latest_row.get('reviewer_past_reviews_180d', 0),

                # レビュー負荷
                'reviewer_assignment_load_7d': latest_row.get('reviewer_assignment_load_7d', 0),
                'reviewer_assignment_load_30d': latest_row.get('reviewer_assignment_load_30d', 0),
                'reviewer_assignment_load_180d': latest_row.get('reviewer_assignment_load_180d', 0),

                # 相互作用
                'owner_reviewer_past_interactions_180d': latest_row.get('owner_reviewer_past_interactions_180d', 0),
                'owner_reviewer_project_interactions_180d': latest_row.get('owner_reviewer_project_interactions_180d', 0),
                'owner_reviewer_past_assignments_180d': latest_row.get('owner_reviewer_past_assignments_180d', 0),

                # Path類似度（階層的コサイン）
                'path_jaccard_files_project': latest_row.get('path_jaccard_files_project', 0.0),
                'path_jaccard_dir1_project': latest_row.get('path_jaccard_dir1_project', 0.0),
                'path_jaccard_dir2_project': latest_row.get('path_jaccard_dir2_project', 0.0),

                # その他
                'response_latency_days': latest_row.get('response_latency_days', 0.0),
                'reviewer_past_response_rate_180d': latest_row.get('reviewer_past_response_rate_180d', 1.0),
                'reviewer_tenure_days': latest_row.get('reviewer_tenure_days', 0),
            })

        # 継続ラベル
        target_activities = df_target[df_target['reviewer_email'] == reviewer_email]
        continued = len(target_activities) > 0

        trajectories.append({
            'developer': developer,
            'activity_history': activity_history,
            'continued': continued,
            'context_date': snapshot_date
        })

    # 継続率
    continuation_rate = sum(1 for t in trajectories if t['continued']) / len(trajectories) if trajectories else 0

    logger.info(f"抽出した軌跡数: {len(trajectories)}")
    logger.info(f"継続率: {continuation_rate:.1%}")

    return trajectories, continuation_rate


def train_and_evaluate_quarterly(
    trajectories: list,
    config: dict,
    learning_quarters: int,
    prediction_quarters: int
) -> dict:
    """四半期単位で訓練・評価"""

    if len(trajectories) < 10:
        logger.warning(f"軌跡が不足: {len(trajectories)}件")
        return None

    # 訓練/テスト分割
    train_trajectories, test_trajectories = train_test_split(
        trajectories, test_size=0.2, random_state=42
    )

    logger.info(f"訓練データ: {len(train_trajectories)}件")
    logger.info(f"テストデータ: {len(test_trajectories)}件")

    # IRLシステムの初期化
    irl_system = EnhancedRetentionIRLSystem(config)

    # 訓練
    train_result = irl_system.train_irl(train_trajectories, epochs=config.get('epochs', 30))

    # テスト評価
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

    binary_predictions = [1 if p >= 0.5 else 0 for p in test_predictions]
    f1 = f1_score(test_labels, binary_predictions)

    # 継続率
    train_continuation_rate = sum(1 for t in train_trajectories if t['continued']) / len(train_trajectories)
    test_continuation_rate = sum(1 for t in test_trajectories if t['continued']) / len(test_trajectories)

    results = {
        'learning_quarters': learning_quarters,
        'prediction_quarters': prediction_quarters,
        'train_size': len(train_trajectories),
        'test_size': len(test_trajectories),
        'train_continuation_rate': train_continuation_rate,
        'test_continuation_rate': test_continuation_rate,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1_score': f1,
        'final_loss': train_result.get('losses', [0])[-1] if train_result.get('losses') else 0
    }

    logger.info(f"=== 評価結果 ({learning_quarters}Q学習 × {prediction_quarters}Q予測) ===")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"AUC-PR: {auc_pr:.4f}")
    logger.info(f"F1スコア: {f1:.4f}")

    return results


def create_8x8_matrix_report(all_results: list, output_dir: Path, use_enhanced: bool):
    """8×8マトリクスレポートを作成"""

    # マトリクス構築
    matrix_auc_roc = np.zeros((8, 8))
    matrix_auc_pr = np.zeros((8, 8))
    matrix_f1 = np.zeros((8, 8))

    for result in all_results:
        if result is None:
            continue
        i = result['learning_quarters'] - 1
        j = result['prediction_quarters'] - 1
        matrix_auc_roc[i][j] = result['auc_roc']
        matrix_auc_pr[i][j] = result['auc_pr']
        matrix_f1[i][j] = result['f1_score']

    # レポート作成
    report_lines = []
    report_lines.append("# 8×8マトリクス評価結果（3ヶ月単位スライディングウィンドウ）\n")
    report_lines.append(f"**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**特徴量**: {'拡張版（32次元）' if use_enhanced else '基本版（10次元）'}\n")
    report_lines.append(f"**期間単位**: 3ヶ月（四半期）\n")
    report_lines.append(f"**評価範囲**: 2年間（8四半期）\n")

    # AUC-ROCマトリクス
    report_lines.append("\n## AUC-ROC マトリクス\n")
    report_lines.append("| 学習期間 \\ 予測期間 | 1Q | 2Q | 3Q | 4Q | 5Q | 6Q | 7Q | 8Q |")
    report_lines.append("|" + "----|" * 9)

    for i in range(8):
        row = [f"**{i+1}Q**"]
        for j in range(8):
            if matrix_auc_roc[i][j] > 0:
                row.append(f"{matrix_auc_roc[i][j]:.3f}")
            else:
                row.append("-")
        report_lines.append("| " + " | ".join(row) + " |")

    # 最高値の強調
    max_auc_roc = np.max(matrix_auc_roc)
    max_i, max_j = np.unravel_index(np.argmax(matrix_auc_roc), matrix_auc_roc.shape)
    report_lines.append(f"\n**最高AUC-ROC**: {max_auc_roc:.4f} ({max_i+1}Q学習 × {max_j+1}Q予測)\n")

    # AUC-PRマトリクス
    report_lines.append("\n## AUC-PR マトリクス\n")
    report_lines.append("| 学習期間 \\ 予測期間 | 1Q | 2Q | 3Q | 4Q | 5Q | 6Q | 7Q | 8Q |")
    report_lines.append("|" + "----|" * 9)

    for i in range(8):
        row = [f"**{i+1}Q**"]
        for j in range(8):
            if matrix_auc_pr[i][j] > 0:
                row.append(f"{matrix_auc_pr[i][j]:.3f}")
            else:
                row.append("-")
        report_lines.append("| " + " | ".join(row) + " |")

    # F1スコアマトリクス
    report_lines.append("\n## F1 Score マトリクス\n")
    report_lines.append("| 学習期間 \\ 予測期間 | 1Q | 2Q | 3Q | 4Q | 5Q | 6Q | 7Q | 8Q |")
    report_lines.append("|" + "----|" * 9)

    for i in range(8):
        row = [f"**{i+1}Q**"]
        for j in range(8):
            if matrix_f1[i][j] > 0:
                row.append(f"{matrix_f1[i][j]:.3f}")
            else:
                row.append("-")
        report_lines.append("| " + " | ".join(row) + " |")

    # トレンド分析
    report_lines.append("\n## トレンド分析\n")
    report_lines.append("### 学習期間ごとの平均AUC-ROC\n")
    for i in range(8):
        avg = np.mean([matrix_auc_roc[i][j] for j in range(8) if matrix_auc_roc[i][j] > 0])
        report_lines.append(f"- {i+1}Q学習: {avg:.4f}")

    report_lines.append("\n### 予測期間ごとの平均AUC-ROC\n")
    for j in range(8):
        avg = np.mean([matrix_auc_roc[i][j] for i in range(8) if matrix_auc_roc[i][j] > 0])
        report_lines.append(f"- {j+1}Q予測: {avg:.4f}")

    # ファイル保存
    report_file = output_dir / 'MATRIX_8x8_REPORT.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"8×8マトリクスレポートを作成: {report_file}")

    # NumPy形式でも保存
    np.save(output_dir / 'matrix_auc_roc.npy', matrix_auc_roc)
    np.save(output_dir / 'matrix_auc_pr.npy', matrix_auc_pr)
    np.save(output_dir / 'matrix_f1.npy', matrix_f1)


def main():
    parser = argparse.ArgumentParser(
        description='3ヶ月単位スライディングウィンドウで8×8マトリクス評価',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--reviews', type=str, required=True,
                        help='レビューリクエストCSVファイル')
    parser.add_argument('--start-date', type=str, required=True,
                        help='開始日（YYYY-MM-DD）')
    parser.add_argument('--use-enhanced-features', action='store_true',
                        help='拡張特徴量を使用')
    parser.add_argument('--sequence', action='store_true',
                        help='時系列モード（LSTM使用）')
    parser.add_argument('--seq-len', type=int, default=15,
                        help='シーケンス長')
    parser.add_argument('--epochs', type=int, default=30,
                        help='訓練エポック数')
    parser.add_argument('--output', type=str, required=True,
                        help='出力ディレクトリ')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    logger.info(f"データ読み込み: {args.reviews}")
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    logger.info(f"開始日: {start_date.date()}")
    logger.info(f"拡張特徴量: {args.use_enhanced_features}")

    # 設定
    config = {
        'state_dim': 32 if args.use_enhanced_features else 10,
        'action_dim': 9 if args.use_enhanced_features else 5,
        'hidden_dim': 128,
        'learning_rate': 0.001,
        'sequence': args.sequence,
        'seq_len': args.seq_len,
        'epochs': args.epochs
    }

    # 全結果を保存
    all_results = []

    # 8×8マトリクス評価
    total_combinations = 64
    completed = 0

    for learning_quarters in range(1, 9):  # 1Q ~ 8Q
        for prediction_quarters in range(1, 9):  # 1Q ~ 8Q
            completed += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"進捗: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")
            logger.info(f"学習期間: {learning_quarters}Q, 予測期間: {prediction_quarters}Q")
            logger.info(f"{'='*70}")

            # 軌跡抽出
            trajectories, continuation_rate = extract_trajectories_quarterly(
                df, start_date, learning_quarters, prediction_quarters, args.use_enhanced_features
            )

            # 訓練と評価
            result = train_and_evaluate_quarterly(
                trajectories, config, learning_quarters, prediction_quarters
            )

            if result:
                all_results.append(result)

    # 結果保存
    results_file = output_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n全実験完了。結果を保存: {results_file}")

    # 8×8マトリクスレポート作成
    create_8x8_matrix_report(all_results, output_dir, args.use_enhanced_features)


if __name__ == '__main__':
    main()
