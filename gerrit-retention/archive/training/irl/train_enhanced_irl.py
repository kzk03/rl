"""
拡張特徴量を使用したTemporal IRL訓練スクリプト

高優先度特徴量を統合し、既存モデルと性能比較
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from gerrit_retention.rl_prediction.enhanced_retention_irl_system import EnhancedRetentionIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_trajectories(df: pd.DataFrame,
                         snapshot_date: datetime,
                         history_months: int,
                         target_months: int,
                         fixed_reviewers: set = None) -> tuple:
    """軌跡を抽出（既存のロジックを踏襲）"""

    # 学習期間と予測期間
    learning_start = snapshot_date - timedelta(days=history_months * 30)
    learning_end = snapshot_date
    target_start = snapshot_date
    target_end = snapshot_date + timedelta(days=target_months * 30)

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

    # レビュアーごとにグループ化
    trajectories = []
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
            continue

        # 開発者情報（最新のレコードから）
        latest_row = group.iloc[-1]
        developer = {
            'developer_id': reviewer_email,
            'reviewer_email': reviewer_email,
            'changes_authored': 0,  # 簡易化
            'changes_reviewed': len(group),
            'projects': group['project'].unique().tolist(),
            'first_seen': group['request_time'].min().isoformat(),
            'last_activity': group['request_time'].max().isoformat(),
            # OpenStack固有の特徴量
            'reviewer_assignment_load_7d': latest_row.get('reviewer_assignment_load_7d', 0),
            'reviewer_assignment_load_30d': latest_row.get('reviewer_assignment_load_30d', 0),
            'reviewer_assignment_load_180d': latest_row.get('reviewer_assignment_load_180d', 0),
            'reviewer_past_reviews_30d': latest_row.get('reviewer_past_reviews_30d', 0),
            'reviewer_past_reviews_90d': latest_row.get('reviewer_past_reviews_90d', 0),
            'reviewer_past_reviews_180d': latest_row.get('reviewer_past_reviews_180d', 0),
            'reviewer_past_response_rate_180d': latest_row.get('reviewer_past_response_rate_180d', 1.0),
            'reviewer_tenure_days': latest_row.get('reviewer_tenure_days', 0),
            'owner_reviewer_past_interactions_180d': latest_row.get('owner_reviewer_past_interactions_180d', 0),
            'owner_reviewer_project_interactions_180d': latest_row.get('owner_reviewer_project_interactions_180d', 0),
            'owner_reviewer_past_assignments_180d': latest_row.get('owner_reviewer_past_assignments_180d', 0),
            'path_jaccard_files_project': latest_row.get('path_jaccard_files_project', 0.0),
            'path_jaccard_dir1_project': latest_row.get('path_jaccard_dir1_project', 0.0),
            'path_jaccard_dir2_project': latest_row.get('path_jaccard_dir2_project', 0.0),
            'path_overlap_files_project': latest_row.get('path_overlap_files_project', 0.0),
            'path_overlap_dir1_project': latest_row.get('path_overlap_dir1_project', 0.0),
            'path_overlap_dir2_project': latest_row.get('path_overlap_dir2_project', 0.0),
            'response_latency_days': latest_row.get('response_latency_days', 0.0),
            'change_insertions': latest_row.get('change_insertions', 0),
            'change_deletions': latest_row.get('change_deletions', 0),
            'change_files_count': latest_row.get('change_files_count', 1)
        }

        # 継続ラベル（予測期間に活動があるか）
        continued = len(df[
            (df['reviewer_email'] == reviewer_email) &
            (df['request_time'] >= target_start) &
            (df['request_time'] < target_end)
        ]) > 0

        trajectories.append({
            'developer': developer,
            'activity_history': activity_history,
            'continued': continued,
            'context_date': learning_end
        })

    return trajectories


def main():
    parser = argparse.ArgumentParser(description='拡張特徴量を使用したIRL訓練')
    parser.add_argument('--reviews', required=True, help='レビューログCSVファイル')
    parser.add_argument('--snapshot-date', required=True, help='スナップショット日付 (YYYY-MM-DD)')
    parser.add_argument('--history-months', type=int, default=12, help='学習期間（月）')
    parser.add_argument('--target-months', type=int, default=6, help='予測期間（月）')
    parser.add_argument('--reference-period', type=int, default=6, help='固定対象者決定用の基準期間（月）')
    parser.add_argument('--sequence', action='store_true', help='時系列モードを使用')
    parser.add_argument('--seq-len', type=int, default=15, help='シーケンス長')
    parser.add_argument('--epochs', type=int, default=30, help='訓練エポック数')
    parser.add_argument('--hidden-dim', type=int, default=256, help='隠れ層次元')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--output', required=True, help='出力ディレクトリ')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)

    # レビューログ読み込み
    logger.info(f"レビューログを読み込み中: {args.reviews}")
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])
    logger.info(f"レビューログ読み込み完了: {len(df)}件")

    # スナップショット日付
    snapshot_date = datetime.strptime(args.snapshot_date, '%Y-%m-%d')

    # 固定対象レビュアーを決定
    reference_start = snapshot_date - timedelta(days=args.reference_period * 30)
    reference_end = snapshot_date
    df_reference = df[
        (df['request_time'] >= reference_start) &
        (df['request_time'] < reference_end)
    ]
    fixed_reviewers = set(df_reference['reviewer_email'].unique())
    logger.info(f"固定対象レビュアー数: {len(fixed_reviewers)}人")

    # 軌跡抽出
    logger.info(f"軌跡抽出中: 学習={args.history_months}ヶ月, 予測={args.target_months}ヶ月")
    trajectories = extract_trajectories(
        df, snapshot_date, args.history_months, args.target_months, fixed_reviewers
    )
    logger.info(f"軌跡数: {len(trajectories)}件")

    if len(trajectories) == 0:
        logger.error("軌跡が抽出できませんでした")
        return

    # Train/Testに分割
    split_idx = int(len(trajectories) * 0.8)
    train_trajectories = trajectories[:split_idx]
    test_trajectories = trajectories[split_idx:]

    train_continued = sum(1 for t in train_trajectories if t['continued'])
    test_continued = sum(1 for t in test_trajectories if t['continued'])

    logger.info(f"訓練={len(train_trajectories)}, テスト={len(test_trajectories)}")
    logger.info(f"継続率（訓練）: {train_continued / len(train_trajectories) * 100:.1f}%")
    logger.info(f"継続率（テスト）: {test_continued / len(test_trajectories) * 100:.1f}%")

    # 拡張IRLシステムを初期化
    config = {
        'state_dim': 32,  # 拡張特徴量
        'action_dim': 9,  # 拡張特徴量
        'hidden_dim': args.hidden_dim,
        'sequence': args.sequence,
        'seq_len': args.seq_len,
        'dropout': args.dropout,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }

    logger.info("拡張IRLシステムを初期化中...")
    irl_system = EnhancedRetentionIRLSystem(config)

    # 訓練
    logger.info("拡張IRLモデルを訓練中...")
    train_result = irl_system.train_irl(train_trajectories, epochs=args.epochs)
    logger.info(f"訓練完了: 最終損失={train_result['final_loss']:.4f}")

    # モデル保存
    model_filename = f"enhanced_irl_h{args.history_months}m_t{args.target_months}m_{'seq' if args.sequence else 'nosq'}.pth"
    model_path = output_dir / 'models' / model_filename
    irl_system.save_model(str(model_path))

    # 評価
    logger.info("拡張IRLモデルを評価中...")
    eval_result = irl_system.evaluate(test_trajectories)

    logger.info(f"評価完了: AUC-ROC={eval_result['auc_roc']:.3f}, "
                f"AUC-PR={eval_result['auc_pr']:.3f}, "
                f"F1={eval_result['f1']:.3f}")

    # 結果を保存
    result_summary = {
        'model': 'enhanced_irl',
        'features': 'B1+C1+A1+D1 (high priority)',
        'snapshot_date': args.snapshot_date,
        'history_months': args.history_months,
        'target_months': args.target_months,
        'reference_period': args.reference_period,
        'fixed_reviewers': len(fixed_reviewers),
        'train_trajectories': len(train_trajectories),
        'test_trajectories': len(test_trajectories),
        'train_continuation_rate': train_continued / len(train_trajectories),
        'test_continuation_rate': test_continued / len(test_trajectories),
        'sequence': args.sequence,
        'seq_len': args.seq_len,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'final_loss': train_result['final_loss'],
        'evaluation': eval_result,
        'model_path': str(model_path)
    }

    result_path = output_dir / f"enhanced_result_h{args.history_months}m_t{args.target_months}m.json"
    with open(result_path, 'w') as f:
        json.dump(result_summary, f, indent=2)

    logger.info(f"結果を保存: {result_path}")

    # 簡易レポート出力
    print("\n" + "=" * 80)
    print("拡張特徴量IRL訓練結果")
    print("=" * 80)
    print(f"学習期間: {args.history_months}ヶ月, 予測期間: {args.target_months}ヶ月")
    print(f"固定対象者: {len(fixed_reviewers)}人")
    print(f"訓練サンプル: {len(train_trajectories)}件, テストサンプル: {len(test_trajectories)}件")
    print(f"\n【評価結果】")
    print(f"  AUC-ROC:    {eval_result['auc_roc']:.4f}")
    print(f"  AUC-PR:     {eval_result['auc_pr']:.4f}")
    print(f"  F1 Score:   {eval_result['f1']:.4f}")
    print(f"  Precision:  {eval_result['precision']:.4f}")
    print(f"  Recall:     {eval_result['recall']:.4f}")
    print(f"  Accuracy:   {eval_result['accuracy']:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
