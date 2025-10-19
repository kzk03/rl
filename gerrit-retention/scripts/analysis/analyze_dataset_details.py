#!/usr/bin/env python3
"""
データセット詳細分析

OpenStackレビューデータの完全な統計情報を抽出：
- ファイルパスと内容
- プロジェクト別統計
- 時系列範囲
- レビュアー統計
- 継続率
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_dataset(csv_path: str, output_dir: Path):
    """データセットの詳細分析"""

    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80)

    # ファイル情報
    file_path = Path(csv_path)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    print(f"\nFile Information:")
    print(f"  Path: {file_path.absolute()}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Exists: {file_path.exists()}")

    # データ読み込み
    print(f"\nLoading data...")
    df = pd.read_csv(csv_path)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # カラム情報
    print(f"\nColumns:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"  - {col:30s} {str(dtype):15s} (null: {null_count:6d} = {null_pct:5.2f}%)")

    # 日付カラムを特定
    date_col = None
    for col in ['request_time', 'created', 'timestamp', 'date']:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])

        # 時系列範囲
        print(f"\nTime Range ({date_col}):")
        print(f"  Start: {df[date_col].min()}")
        print(f"  End: {df[date_col].max()}")
        print(f"  Duration: {(df[date_col].max() - df[date_col].min()).days} days")
        print(f"  Duration: {(df[date_col].max() - df[date_col].min()).days / 365.25:.1f} years")

    # レビュアー情報
    reviewer_col = None
    for col in ['reviewer_email', 'email', 'reviewer', 'user']:
        if col in df.columns:
            reviewer_col = col
            break

    if reviewer_col:
        print(f"\nReviewer Statistics ({reviewer_col}):")
        n_reviewers = df[reviewer_col].nunique()
        print(f"  Total reviewers: {n_reviewers:,}")

        reviews_per_reviewer = df.groupby(reviewer_col).size()
        print(f"  Reviews per reviewer:")
        print(f"    Mean: {reviews_per_reviewer.mean():.1f}")
        print(f"    Median: {reviews_per_reviewer.median():.0f}")
        print(f"    25th percentile: {reviews_per_reviewer.quantile(0.25):.0f}")
        print(f"    75th percentile: {reviews_per_reviewer.quantile(0.75):.0f}")
        print(f"    90th percentile: {reviews_per_reviewer.quantile(0.90):.0f}")
        print(f"    95th percentile: {reviews_per_reviewer.quantile(0.95):.0f}")
        print(f"    Max: {reviews_per_reviewer.max():,.0f}")

    # プロジェクト情報
    project_col = None
    for col in ['project', 'repo', 'repository']:
        if col in df.columns:
            project_col = col
            break

    project_stats = None
    if project_col:
        print(f"\nProject Statistics ({project_col}):")
        n_projects = df[project_col].nunique()
        print(f"  Total projects: {n_projects:,}")

        project_stats = df.groupby(project_col).agg({
            reviewer_col: 'nunique' if reviewer_col else 'count',
            project_col: 'count'
        }).rename(columns={
            reviewer_col: 'reviewers',
            project_col: 'reviews'
        }).sort_values('reviews', ascending=False)

        print(f"\n  Top 10 projects by review count:")
        for idx, (proj, row) in enumerate(project_stats.head(10).iterrows(), 1):
            reviewers_info = f", {row['reviewers']:,} reviewers" if reviewer_col else ""
            print(f"    {idx:2d}. {proj:40s} {row['reviews']:6,} reviews{reviewers_info}")

    # 継続率の計算（簡易版）
    if date_col and reviewer_col:
        print(f"\nContinuation Rate Analysis:")

        # 中間地点で分割
        mid_date = df[date_col].min() + (df[date_col].max() - df[date_col].min()) / 2

        first_half = df[df[date_col] < mid_date]
        second_half = df[df[date_col] >= mid_date]

        reviewers_first = set(first_half[reviewer_col].unique())
        reviewers_second = set(second_half[reviewer_col].unique())

        continued = reviewers_first & reviewers_second
        continuation_rate = len(continued) / len(reviewers_first) if len(reviewers_first) > 0 else 0

        print(f"  Period 1 ({df[date_col].min().date()} to {mid_date.date()}):")
        print(f"    Reviewers: {len(reviewers_first):,}")
        print(f"  Period 2 ({mid_date.date()} to {df[date_col].max().date()}):")
        print(f"    Reviewers: {len(reviewers_second):,}")
        print(f"  Continued (appeared in both periods): {len(continued):,}")
        print(f"  Continuation rate: {continuation_rate:.1%}")

    # データを保存
    output_dir.mkdir(parents=True, exist_ok=True)

    # 統計情報をJSON保存
    stats = {
        'file_info': {
            'path': str(file_path.absolute()),
            'size_mb': file_size_mb,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns)
        }
    }

    if date_col:
        stats['time_range'] = {
            'column': date_col,
            'start': str(df[date_col].min()),
            'end': str(df[date_col].max()),
            'duration_days': int((df[date_col].max() - df[date_col].min()).days),
            'duration_years': float((df[date_col].max() - df[date_col].min()).days / 365.25)
        }

    if reviewer_col:
        stats['reviewers'] = {
            'column': reviewer_col,
            'total': int(n_reviewers),
            'reviews_per_reviewer': {
                'mean': float(reviews_per_reviewer.mean()),
                'median': float(reviews_per_reviewer.median()),
                'p25': float(reviews_per_reviewer.quantile(0.25)),
                'p75': float(reviews_per_reviewer.quantile(0.75)),
                'p90': float(reviews_per_reviewer.quantile(0.90)),
                'max': int(reviews_per_reviewer.max())
            }
        }

    if project_col:
        stats['projects'] = {
            'column': project_col,
            'total': int(n_projects),
            'top_10': [
                {'project': proj, 'reviews': int(row['reviews']),
                 'reviewers': int(row['reviewers']) if reviewer_col else None}
                for proj, row in project_stats.head(10).iterrows()
            ]
        }

    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved: {stats_path}")

    # プロジェクト別統計をCSV保存
    if project_stats is not None:
        project_csv = output_dir / 'project_stats.csv'
        project_stats.to_csv(project_csv)
        print(f"Project statistics saved: {project_csv}")

    # 可視化
    plot_visualizations(df, output_dir, date_col, reviewer_col, project_col, project_stats)

    # レポート作成
    create_report(stats, output_dir)

    return stats


def plot_visualizations(df: pd.DataFrame, output_dir: Path,
                        date_col: str, reviewer_col: str,
                        project_col: str, project_stats: pd.DataFrame):
    """データ可視化"""

    print(f"\nGenerating visualizations...")

    # 1. 時系列プロット
    if date_col:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # レビュー数の月次推移
        df_monthly = df.set_index(date_col).resample('M').size()

        ax1 = axes[0]
        ax1.plot(df_monthly.index, df_monthly.values, linewidth=2, color='steelblue')
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Reviews per Month', fontsize=12, fontweight='bold')
        ax1.set_title('Review Activity Over Time', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)

        # 累積レビュー数
        ax2 = axes[1]
        cumulative = df_monthly.cumsum()
        ax2.plot(cumulative.index, cumulative.values, linewidth=2, color='green')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Reviews', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Review Count', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'time_series.png', dpi=150, bbox_inches='tight')
        print(f"  Time series plot saved")
        plt.close()

    # 2. レビュアー分布
    if reviewer_col:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        reviews_per_reviewer = df.groupby(reviewer_col).size()

        # ヒストグラム
        ax1 = axes[0]
        ax1.hist(reviews_per_reviewer.values, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Reviews per Reviewer', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Reviewers', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Reviews per Reviewer', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(alpha=0.3)

        # ロングテール分布
        ax2 = axes[1]
        sorted_reviews = sorted(reviews_per_reviewer.values, reverse=True)
        ax2.plot(range(len(sorted_reviews)), sorted_reviews, linewidth=2, color='purple')
        ax2.set_xlabel('Reviewer Rank', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
        ax2.set_title('Review Distribution (Ranked)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'reviewer_distribution.png', dpi=150, bbox_inches='tight')
        print(f"  Reviewer distribution plot saved")
        plt.close()

    # 3. プロジェクト別統計
    if project_stats is not None and len(project_stats) > 0:
        fig, ax = plt.subplots(figsize=(12, max(8, len(project_stats.head(15)) * 0.4)))

        top_projects = project_stats.head(15).sort_values('reviews', ascending=True)

        y_pos = np.arange(len(top_projects))
        ax.barh(y_pos, top_projects['reviews'], color='skyblue', alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_projects.index, fontsize=10)
        ax.set_xlabel('Number of Reviews', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Projects by Review Count', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'project_distribution.png', dpi=150, bbox_inches='tight')
        print(f"  Project distribution plot saved")
        plt.close()


def create_report(stats: Dict, output_dir: Path):
    """レポート作成"""

    report_path = output_dir / 'DATASET_ANALYSIS_REPORT.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DATASET ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("File Information\n")
        f.write("-"*80 + "\n")
        f.write(f"Path: {stats['file_info']['path']}\n")
        f.write(f"Size: {stats['file_info']['size_mb']:.2f} MB\n")
        f.write(f"Rows: {stats['file_info']['rows']:,}\n")
        f.write(f"Columns: {stats['file_info']['columns']}\n\n")

        if 'time_range' in stats:
            f.write("Time Range\n")
            f.write("-"*80 + "\n")
            f.write(f"Column: {stats['time_range']['column']}\n")
            f.write(f"Start: {stats['time_range']['start']}\n")
            f.write(f"End: {stats['time_range']['end']}\n")
            f.write(f"Duration: {stats['time_range']['duration_days']:,} days ({stats['time_range']['duration_years']:.1f} years)\n\n")

        if 'reviewers' in stats:
            f.write("Reviewer Statistics\n")
            f.write("-"*80 + "\n")
            f.write(f"Total reviewers: {stats['reviewers']['total']:,}\n")
            f.write(f"Reviews per reviewer:\n")
            f.write(f"  Mean: {stats['reviewers']['reviews_per_reviewer']['mean']:.1f}\n")
            f.write(f"  Median: {stats['reviewers']['reviews_per_reviewer']['median']:.0f}\n")
            f.write(f"  75th percentile: {stats['reviewers']['reviews_per_reviewer']['p75']:.0f}\n")
            f.write(f"  90th percentile: {stats['reviewers']['reviews_per_reviewer']['p90']:.0f}\n")
            f.write(f"  Max: {stats['reviewers']['reviews_per_reviewer']['max']:,}\n\n")

        if 'projects' in stats:
            f.write("Project Statistics\n")
            f.write("-"*80 + "\n")
            f.write(f"Total projects: {stats['projects']['total']:,}\n\n")
            f.write("Top 10 projects:\n")
            for i, proj in enumerate(stats['projects']['top_10'], 1):
                reviewers_str = f", {proj['reviewers']:,} reviewers" if proj['reviewers'] else ""
                f.write(f"  {i:2d}. {proj['project']:40s} {proj['reviews']:6,} reviews{reviewers_str}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Analysis complete\n")
        f.write("="*80 + "\n")

    print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset details')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output)

    stats = analyze_dataset(args.data, output_dir)

    print("\n" + "="*80)
    print("Dataset analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
