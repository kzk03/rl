#!/usr/bin/env python3
"""
プロジェクト別フィルタースクリプト

OpenStackのGerritデータを特定のプロジェクトで絞り込むスクリプト。
プロジェクトの指定、自動的な主要プロジェクト抽出、統計情報の表示に対応。

使用例:
    # 特定のプロジェクトでフィルタリング
    uv run python scripts/preprocessing/filter_by_project.py \
        --input data/review_requests_openstack_multi_5y_detail.csv \
        --output data/review_requests_nova_neutron.csv \
        --projects "openstack/nova" "openstack/neutron"

    # 上位N個のプロジェクトを自動抽出
    uv run python scripts/preprocessing/filter_by_project.py \
        --input data/review_requests_openstack_multi_5y_detail.csv \
        --output data/review_requests_top5.csv \
        --top 5

    # 各プロジェクトごとに個別ファイルを作成（自動化）
    uv run python scripts/preprocessing/filter_by_project.py \
        --input data/review_requests_openstack_multi_5y_detail.csv \
        --split-by-project \
        --output-dir data/projects/

    # 統計情報のみ表示
    uv run python scripts/preprocessing/filter_by_project.py \
        --input data/review_requests_openstack_multi_5y_detail.csv \
        --stats-only
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_projects(df: pd.DataFrame, project_column: str = 'project') -> pd.DataFrame:
    """
    プロジェクトごとの統計情報を分析

    Args:
        df: データフレーム
        project_column: プロジェクト名のカラム

    Returns:
        プロジェクト統計のデータフレーム
    """
    stats = df.groupby(project_column).agg({
        'change_id': 'count',
        'reviewer_email': 'nunique',
        'request_time': ['min', 'max']
    }).reset_index()

    stats.columns = ['project', 'review_count', 'reviewer_count', 'start_date', 'end_date']
    stats = stats.sort_values('review_count', ascending=False)

    return stats


def display_project_stats(stats: pd.DataFrame, limit: int = 20) -> None:
    """
    プロジェクト統計を表示

    Args:
        stats: プロジェクト統計のデータフレーム
        limit: 表示する最大プロジェクト数
    """
    logger.info("\n=== プロジェクト統計 ===")
    logger.info(f"総プロジェクト数: {len(stats)}")
    logger.info(f"総レビュー数: {stats['review_count'].sum():,}件")
    logger.info(f"総レビュアー数: {stats['reviewer_count'].sum():,}人（重複含む）")

    logger.info(f"\n上位{min(limit, len(stats))}プロジェクト:")
    logger.info(f"{'順位':<4} {'プロジェクト名':<40} {'レビュー数':>12} {'レビュアー数':>12} {'期間'}")
    logger.info("-" * 100)

    for idx, row in stats.head(limit).iterrows():
        logger.info(
            f"{idx+1:<4} {row['project']:<40} {row['review_count']:>12,} "
            f"{row['reviewer_count']:>12,} {row['start_date'][:10]} ~ {row['end_date'][:10]}"
        )


def filter_by_projects(
    df: pd.DataFrame,
    projects: List[str],
    project_column: str = 'project'
) -> pd.DataFrame:
    """
    指定されたプロジェクトでフィルタリング

    Args:
        df: データフレーム
        projects: プロジェクト名のリスト
        project_column: プロジェクト名のカラム

    Returns:
        フィルタリングされたデータフレーム
    """
    # 存在しないプロジェクトをチェック
    available_projects = set(df[project_column].unique())
    invalid_projects = set(projects) - available_projects

    if invalid_projects:
        logger.warning(f"⚠️ 存在しないプロジェクト: {invalid_projects}")

    valid_projects = set(projects) & available_projects
    if not valid_projects:
        logger.error("❌ 有効なプロジェクトが1つもありません")
        return pd.DataFrame()

    logger.info(f"フィルタリング対象プロジェクト: {sorted(valid_projects)}")

    df_filtered = df[df[project_column].isin(valid_projects)]

    return df_filtered


def filter_top_projects(
    df: pd.DataFrame,
    top_n: int,
    project_column: str = 'project'
) -> pd.DataFrame:
    """
    レビュー数上位N個のプロジェクトでフィルタリング

    Args:
        df: データフレーム
        top_n: 上位何個のプロジェクトを抽出するか
        project_column: プロジェクト名のカラム

    Returns:
        フィルタリングされたデータフレーム
    """
    stats = analyze_projects(df, project_column)
    top_projects = stats.head(top_n)['project'].tolist()

    logger.info(f"上位{top_n}プロジェクトを抽出: {top_projects}")

    return filter_by_projects(df, top_projects, project_column)


def split_by_project(
    df: pd.DataFrame,
    output_dir: str,
    project_column: str = 'project',
    min_reviews: int = 100
) -> None:
    """
    各プロジェクトごとに個別のCSVファイルを作成

    Args:
        df: データフレーム
        output_dir: 出力ディレクトリ
        project_column: プロジェクト名のカラム
        min_reviews: 最小レビュー数（これ以下のプロジェクトは除外）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = analyze_projects(df, project_column)
    stats_filtered = stats[stats['review_count'] >= min_reviews]

    logger.info(f"\n=== プロジェクト別ファイル作成 ===")
    logger.info(f"出力ディレクトリ: {output_dir}")
    logger.info(f"最小レビュー数: {min_reviews}件")
    logger.info(f"対象プロジェクト数: {len(stats_filtered)}/{len(stats)}")

    for _, row in stats_filtered.iterrows():
        project_name = row['project']
        project_df = df[df[project_column] == project_name]

        # ファイル名を安全な形式に変換
        safe_name = project_name.replace('/', '_').replace(' ', '_')
        output_file = output_path / f"{safe_name}.csv"

        project_df.to_csv(output_file, index=False)
        logger.info(
            f"  ✅ {project_name}: {len(project_df):,}件 "
            f"→ {output_file.name}"
        )

    logger.info(f"\n合計{len(stats_filtered)}個のファイルを作成しました")


def main():
    parser = argparse.ArgumentParser(
        description='OpenStack Gerritデータをプロジェクト別にフィルタリング',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 特定のプロジェクトでフィルタリング
  uv run python scripts/preprocessing/filter_by_project.py \\
      --input data/review_requests_openstack_multi_5y_detail.csv \\
      --output data/review_requests_nova_neutron.csv \\
      --projects "openstack/nova" "openstack/neutron"

  # 上位5個のプロジェクトを自動抽出
  uv run python scripts/preprocessing/filter_by_project.py \\
      --input data/review_requests_openstack_multi_5y_detail.csv \\
      --output data/review_requests_top5.csv \\
      --top 5

  # 各プロジェクトごとに個別ファイルを作成（自動化）
  uv run python scripts/preprocessing/filter_by_project.py \\
      --input data/review_requests_openstack_multi_5y_detail.csv \\
      --split-by-project \\
      --output-dir data/projects/ \\
      --min-reviews 500

  # 統計情報のみ表示
  uv run python scripts/preprocessing/filter_by_project.py \\
      --input data/review_requests_openstack_multi_5y_detail.csv \\
      --stats-only
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='入力CSVファイルパス'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='出力CSVファイルパス（--projectsまたは--topと併用）'
    )

    parser.add_argument(
        '--project-column',
        type=str,
        default='project',
        help='プロジェクト名のカラム名（デフォルト: project）'
    )

    # フィルタリング方法（3つのうち1つを選択）
    filter_group = parser.add_mutually_exclusive_group()

    filter_group.add_argument(
        '--projects',
        nargs='+',
        help='フィルタリングするプロジェクト名のリスト'
    )

    filter_group.add_argument(
        '--top',
        type=int,
        help='レビュー数上位N個のプロジェクトを抽出'
    )

    filter_group.add_argument(
        '--split-by-project',
        action='store_true',
        help='各プロジェクトごとに個別のCSVファイルを作成'
    )

    filter_group.add_argument(
        '--stats-only',
        action='store_true',
        help='統計情報のみ表示してファイルは保存しない'
    )

    # split-by-project用のオプション
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/projects',
        help='プロジェクト別ファイルの出力ディレクトリ（デフォルト: data/projects）'
    )

    parser.add_argument(
        '--min-reviews',
        type=int,
        default=100,
        help='プロジェクト別分割時の最小レビュー数（デフォルト: 100）'
    )

    args = parser.parse_args()

    # データ読み込み
    logger.info(f"入力ファイル: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"元データ: {len(df):,}件, {df[args.project_column].nunique():,}プロジェクト")

    # プロジェクト統計を表示
    stats = analyze_projects(df, args.project_column)
    display_project_stats(stats)

    # 統計情報のみ表示モード
    if args.stats_only:
        logger.info("\n✅ 統計情報の表示が完了しました")
        return

    # プロジェクト別分割モード
    if args.split_by_project:
        split_by_project(df, args.output_dir, args.project_column, args.min_reviews)
        return

    # フィルタリングモード
    if args.projects:
        df_filtered = filter_by_projects(df, args.projects, args.project_column)
    elif args.top:
        df_filtered = filter_top_projects(df, args.top, args.project_column)
    else:
        logger.error("❌ --projects, --top, --split-by-project, --stats-only のいずれかを指定してください")
        return

    if df_filtered.empty:
        logger.error("❌ フィルタリング結果が空です")
        return

    # フィルタリング結果の統計
    logger.info(f"\n=== フィルタリング結果 ===")
    logger.info(f"フィルタリング後: {len(df_filtered):,}件, {df_filtered[args.project_column].nunique():,}プロジェクト")
    logger.info(f"除外されたレビュー数: {len(df) - len(df_filtered):,}件 ({(len(df) - len(df_filtered)) / len(df) * 100:.2f}%)")

    # 保存
    if args.output:
        df_filtered.to_csv(args.output, index=False)
        logger.info(f"\n✅ フィルター済みデータを保存: {args.output}")
    else:
        logger.warning("\n⚠️ --outputが指定されていないため、ファイルは保存されません")


if __name__ == '__main__':
    main()
