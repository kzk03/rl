#!/usr/bin/env python3
"""
ボットアカウント除外スクリプト

OpenStackのGerritデータからボットアカウントを除外するフィルタリングスクリプト。

使用例:
    # 基本的な使用方法
    uv run python scripts/preprocessing/filter_bot_accounts.py \
        --input data/review_requests_openstack_multi_5y_detail.csv \
        --output data/review_requests_openstack_no_bots.csv

    # 追加のボットパターンを指定
    uv run python scripts/preprocessing/filter_bot_accounts.py \
        --input data/review_requests_openstack_multi_5y_detail.csv \
        --output data/review_requests_openstack_no_bots.csv \
        --additional-patterns "deploy" "release"

    # 統計情報のみ表示（dry-run）
    uv run python scripts/preprocessing/filter_bot_accounts.py \
        --input data/review_requests_openstack_multi_5y_detail.csv \
        --dry-run
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Set
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# OpenStackで推奨されるボットパターン
RECOMMENDED_BOT_PATTERNS = [
    'bot',
    'ci',
    'automation',
    'jenkins',
    'build',
    'deploy',
    'zuul',           # OpenStack CI system
    'gerrit',         # Gerrit automation
    'infra',          # Infrastructure automation
    'DL-ARC',         # DL-ARC-InfoScale-OpenStack-CI@arctera.io
    'openstack-ci',   # openstack-ci@sap.com
    'noreply',        # No-reply automation accounts
    'service',        # Service accounts
]


def detect_bot_accounts(
    df: pd.DataFrame,
    email_column: str = 'reviewer_email',
    bot_patterns: List[str] = None,
    case_sensitive: bool = False
) -> pd.Series:
    """
    ボットアカウントを検出する

    Args:
        df: データフレーム
        email_column: メールアドレスのカラム名
        bot_patterns: ボット検出パターンのリスト（Noneの場合は推奨パターンを使用）
        case_sensitive: 大文字小文字を区別するか

    Returns:
        ボットアカウントを示すブールマスク
    """
    if bot_patterns is None:
        bot_patterns = RECOMMENDED_BOT_PATTERNS

    logger.info(f"ボット検出パターン: {bot_patterns}")

    bot_mask = pd.Series([False] * len(df), index=df.index)

    # パターンマッチング
    for pattern in bot_patterns:
        pattern_mask = df[email_column].str.contains(
            pattern,
            case=case_sensitive,
            na=False,
            regex=False
        )
        matched_count = pattern_mask.sum()
        if matched_count > 0:
            logger.info(f"  '{pattern}' パターン: {matched_count}件マッチ")
        bot_mask |= pattern_mask

    return bot_mask


def analyze_bot_accounts(
    df: pd.DataFrame,
    bot_mask: pd.Series,
    email_column: str = 'reviewer_email'
) -> None:
    """
    ボットアカウントの統計情報を表示

    Args:
        df: データフレーム
        bot_mask: ボットアカウントを示すブールマスク
        email_column: メールアドレスのカラム名
    """
    bot_df = df[bot_mask]

    logger.info("\n=== ボットアカウント統計 ===")
    logger.info(f"ボットアカウント数: {bot_df[email_column].nunique()}人")
    logger.info(f"ボット関連レビュー数: {len(bot_df):,}件")
    logger.info(f"全レビューに占める割合: {len(bot_df) / len(df) * 100:.2f}%")

    # 上位ボットアカウント
    logger.info("\n上位10ボットアカウント:")
    top_bots = bot_df[email_column].value_counts().head(10)
    for email, count in top_bots.items():
        logger.info(f"  {email}: {count:,}件")


def filter_bot_accounts(
    input_path: str,
    output_path: str = None,
    email_column: str = 'reviewer_email',
    additional_patterns: List[str] = None,
    dry_run: bool = False
) -> None:
    """
    ボットアカウントを除外してCSVを保存

    Args:
        input_path: 入力CSVファイルパス
        output_path: 出力CSVファイルパス
        email_column: メールアドレスのカラム名
        additional_patterns: 追加のボット検出パターン
        dry_run: Trueの場合、統計情報のみ表示して保存しない
    """
    logger.info(f"入力ファイル: {input_path}")

    # データ読み込み
    df = pd.read_csv(input_path)
    logger.info(f"元データ: {len(df):,}件, {df[email_column].nunique():,}人")

    # ボットパターンの準備
    bot_patterns = RECOMMENDED_BOT_PATTERNS.copy()
    if additional_patterns:
        bot_patterns.extend(additional_patterns)
        logger.info(f"追加パターン: {additional_patterns}")

    # ボット検出
    bot_mask = detect_bot_accounts(df, email_column, bot_patterns)

    # 統計情報表示
    analyze_bot_accounts(df, bot_mask, email_column)

    # フィルタリング後のデータ
    df_filtered = df[~bot_mask]
    logger.info(f"\nフィルタリング後: {len(df_filtered):,}件, {df_filtered[email_column].nunique():,}人")
    logger.info(f"除外されたレビュー数: {len(df) - len(df_filtered):,}件 ({(len(df) - len(df_filtered)) / len(df) * 100:.2f}%)")
    logger.info(f"除外されたレビュアー数: {df[email_column].nunique() - df_filtered[email_column].nunique():,}人")

    # 保存
    if not dry_run and output_path:
        df_filtered.to_csv(output_path, index=False)
        logger.info(f"\n✅ フィルター済みデータを保存: {output_path}")
    elif dry_run:
        logger.info("\n⚠️ Dry-runモード: ファイルは保存されません")
    else:
        logger.info("\n⚠️ 出力パスが指定されていないため、ファイルは保存されません")


def main():
    parser = argparse.ArgumentParser(
        description='OpenStack Gerritデータからボットアカウントを除外',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  uv run python scripts/preprocessing/filter_bot_accounts.py \\
      --input data/review_requests_openstack_multi_5y_detail.csv \\
      --output data/review_requests_openstack_no_bots.csv

  # 追加のボットパターンを指定
  uv run python scripts/preprocessing/filter_bot_accounts.py \\
      --input data/review_requests_openstack_multi_5y_detail.csv \\
      --output data/review_requests_openstack_no_bots.csv \\
      --additional-patterns "deploy" "release"

  # 統計情報のみ表示（dry-run）
  uv run python scripts/preprocessing/filter_bot_accounts.py \\
      --input data/review_requests_openstack_multi_5y_detail.csv \\
      --dry-run

推奨ボットパターン:
  bot, ci, automation, jenkins, build, deploy, zuul, gerrit, infra,
  DL-ARC, openstack-ci, noreply, service
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
        help='出力CSVファイルパス（指定しない場合は保存されない）'
    )

    parser.add_argument(
        '--email-column',
        type=str,
        default='reviewer_email',
        help='メールアドレスのカラム名（デフォルト: reviewer_email）'
    )

    parser.add_argument(
        '--additional-patterns',
        nargs='+',
        help='推奨パターンに追加するボット検出パターン'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='統計情報のみ表示してファイルは保存しない'
    )

    args = parser.parse_args()

    filter_bot_accounts(
        input_path=args.input,
        output_path=args.output,
        email_column=args.email_column,
        additional_patterns=args.additional_patterns,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
