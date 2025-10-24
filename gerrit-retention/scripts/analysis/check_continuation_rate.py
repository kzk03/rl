#!/usr/bin/env python3
"""
継続率を調査するスクリプト
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# パスを追加
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "scripts" / "training" / "irl"))

from train_irl_within_training_period import load_review_logs


def check_continuation_rate(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    history_window_months: int,
    future_window_start_months: int,
    future_window_end_months: int,
    sampling_interval_months: int = 1,
):
    """継続率を調査"""
    
    print("=" * 80)
    print(f"継続率調査")
    print(f"学習期間: {train_start} ～ {train_end}")
    print(f"履歴窓: {history_window_months}ヶ月")
    print(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    print("=" * 80)
    
    date_col = 'request_time'
    reviewer_col = 'reviewer_email'
    
    # サンプリング範囲
    min_sampling_date = train_start + pd.DateOffset(months=history_window_months)
    max_sampling_date = train_end - pd.DateOffset(months=future_window_end_months)
    
    print(f"サンプリング範囲: {min_sampling_date} ～ {max_sampling_date}")
    
    # サンプリング時点を生成
    sampling_points = []
    current = min_sampling_date
    while current <= max_sampling_date:
        sampling_points.append(current)
        current += pd.DateOffset(months=sampling_interval_months)
    
    print(f"サンプリング時点数: {len(sampling_points)}")
    
    # 各サンプリング時点で継続率を調査
    total_continued = 0
    total_not_continued = 0
    
    for idx, sampling_point in enumerate(sampling_points):
        # 履歴期間
        history_start = sampling_point - pd.DateOffset(months=history_window_months)
        history_end = sampling_point
        
        # 将来窓
        future_start = sampling_point + pd.DateOffset(months=future_window_start_months)
        future_end = sampling_point + pd.DateOffset(months=future_window_end_months)
        
        # 履歴データ
        history_df = df[
            (df[date_col] >= history_start) &
            (df[date_col] < history_end)
        ]
        
        # 将来データ
        future_df = df[
            (df[date_col] >= future_start) &
            (df[date_col] < future_end)
        ]
        
        # 履歴期間に活動があったレビュアー
        active_reviewers = history_df[reviewer_col].unique()
        
        # 将来期間に活動があったレビュアー
        future_active_reviewers = set(future_df[reviewer_col].unique())
        
        # 継続・非継続をカウント
        continued = sum(1 for r in active_reviewers if r in future_active_reviewers)
        not_continued = len(active_reviewers) - continued
        
        total_continued += continued
        total_not_continued += not_continued
        
        if idx < 3 or idx == len(sampling_points) - 1:  # 最初の3つと最後を表示
            print(f"\nサンプリング時点 {idx+1}/{len(sampling_points)}: {sampling_point.date()}")
            print(f"  履歴期間: {history_start.date()} ～ {history_end.date()}")
            print(f"  将来窓: {future_start.date()} ～ {future_end.date()}")
            print(f"  アクティブレビュアー: {len(active_reviewers)}人")
            print(f"  継続: {continued}人 ({continued/len(active_reviewers)*100:.1f}%)")
            print(f"  非継続: {not_continued}人 ({not_continued/len(active_reviewers)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("全体の継続率")
    print("=" * 80)
    total = total_continued + total_not_continued
    print(f"総開発者数（延べ）: {total}人")
    print(f"継続: {total_continued}人 ({total_continued/total*100:.1f}%)")
    print(f"非継続: {total_not_continued}人 ({total_not_continued/total*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    # データ読み込み
    data_file = Path("data/review_requests_openstack_multi_5y_detail.csv")
    df = load_review_logs(data_file)
    
    train_start = pd.Timestamp("2021-01-01")
    train_end = pd.Timestamp("2023-01-01")
    
    print("\n" + "=" * 80)
    print("【1】訓練ラベル 0-1m の継続率")
    print("=" * 80)
    check_continuation_rate(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=12,
        future_window_start_months=0,
        future_window_end_months=1,
    )
    
    print("\n\n" + "=" * 80)
    print("【2】訓練ラベル 0-3m の継続率")
    print("=" * 80)
    check_continuation_rate(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=12,
        future_window_start_months=0,
        future_window_end_months=3,
    )
    
    print("\n\n" + "=" * 80)
    print("【3】訓練ラベル 0-6m の継続率")
    print("=" * 80)
    check_continuation_rate(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=12,
        future_window_start_months=0,
        future_window_end_months=6,
    )
    
    print("\n\n" + "=" * 80)
    print("【4】訓練ラベル 0-12m の継続率")
    print("=" * 80)
    check_continuation_rate(
        df=df,
        train_start=train_start,
        train_end=train_end,
        history_window_months=12,
        future_window_start_months=0,
        future_window_end_months=12,
    )

