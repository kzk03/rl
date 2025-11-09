"""
月次サンプリングのテスト

Train: 月次スライディングウィンドウ（最後のN月はラベル専用）
Eval: 評価開始時点の1スナップショット
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# run_multi_seed_eval.pyから関数をインポート
sys.path.insert(0, str(Path(__file__).parent))
from run_multi_seed_eval import prepare_data


def main():
    """テスト実行"""
    # データ読み込み
    data_path = project_root / "data" / "review_requests_openstack_multi_5y_detail.csv"
    df = pd.read_csv(data_path)
    
    print("="*70)
    print("月次サンプリングテスト")
    print("="*70)
    
    # パラメータ
    train_start = "2021-01-01"
    train_end = "2023-01-01"
    eval_start = "2023-01-01"
    eval_end = "2024-01-01"
    
    train_future_start = 3
    train_future_end = 6
    eval_future_start = 3
    eval_future_end = 6
    
    print(f"\n【設定】")
    print(f"  Train観測期間: {train_start} 〜 {train_end}")
    print(f"  Train予測期間: +{train_future_start}〜{train_future_end}ヶ月")
    print(f"  Eval基準日: {eval_start}")
    print(f"  Eval予測期間: +{eval_future_start}〜{eval_future_end}ヶ月")
    
    print(f"\n【期待される動作】")
    print(f"  Train: 月次サンプリング、最後の{train_future_end}月はラベル専用")
    print(f"  Eval: {eval_start}時点の1スナップショット")
    
    # データ準備実行
    X_train_state, X_train_temporal, y_train, X_test_state, X_test_temporal, y_test = prepare_data(
        df, train_start, train_end, eval_start, eval_end,
        train_future_start, train_future_end,
        eval_future_start, eval_future_end
    )
    
    # 結果確認
    print(f"\n{'='*70}")
    print("結果")
    print("="*70)
    print(f"\nTrainデータ:")
    print(f"  サンプル数: {len(y_train)}")
    print(f"  State特徴量: {X_train_state.shape}")
    print(f"  Temporal特徴量: {X_train_temporal.shape}")
    print(f"  ラベル分布: 正例={sum(y_train)}, 負例={len(y_train)-sum(y_train)}")
    print(f"  正例率: {sum(y_train)/len(y_train)*100:.1f}%")
    
    print(f"\nEvalデータ:")
    print(f"  サンプル数: {len(y_test)}")
    print(f"  State特徴量: {X_test_state.shape}")
    print(f"  Temporal特徴量: {X_test_temporal.shape}")
    print(f"  ラベル分布: 正例={sum(y_test)}, 負例={len(y_test)-sum(y_test)}")
    print(f"  正例率: {sum(y_test)/len(y_test)*100:.1f}%")
    
    print(f"\n✓ テスト完了")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.temporal_feature_extractor import TemporalFeatureExtractor

# データ読み込み
data_path = project_root / "data" / "review_requests_openstack_multi_5y_detail.csv"
df = pd.read_csv(data_path)
df['request_time'] = pd.to_datetime(df['request_time'])

print(f"データ読み込み: {len(df)} 行\n")

# テスト：Train期間で月次サンプリング
train_start_dt = pd.to_datetime("2021-01-01")
train_end_dt = pd.to_datetime("2023-01-01")
train_future_start_months = 0
train_future_end_months = 3

print(f"Train期間: {train_start_dt} 〜 {train_end_dt}")
print(f"予測期間: {train_future_start_months}-{train_future_end_months}ヶ月\n")

# 月次でスライディング
train_months = pd.date_range(start=train_start_dt, end=train_end_dt, freq='MS')
print(f"月数: {len(train_months)} ヶ月\n")

sample_count = 0
skipped_count = 0

for i, month_start in enumerate(train_months[:-1][:3]):  # 最初の3ヶ月だけテスト
    month_end = month_start + pd.DateOffset(months=1)
    
    # この月の将来期間
    future_start = month_end + pd.DateOffset(months=train_future_start_months)
    future_end = month_end + pd.DateOffset(months=train_future_end_months)
    
    # データリーク防止
    if future_start >= train_end_dt:
        print(f"[{i+1}] {month_start.date()} 〜 {month_end.date()}: スキップ（future_start >= train_end）")
        skipped_count += 1
        continue
    
    if future_end > train_end_dt:
        future_end = train_end_dt
        print(f"[{i+1}] {month_start.date()} 〜 {month_end.date()}: future_endをクリップ（{future_end.date()}）")
    
    # この月の観測期間内にアクティビティがあったレビュワー
    month_df = df[(df['request_time'] >= train_start_dt) & (df['request_time'] < month_end)]
    month_reviewers = month_df['reviewer_email'].unique()
    
    print(f"[{i+1}] {month_start.date()} 〜 {month_end.date()}")
    print(f"     観測終了: {month_end.date()}")
    print(f"     予測期間: {future_start.date()} 〜 {future_end.date()}")
    print(f"     レビュワー数: {len(month_reviewers)}")
    
    month_samples = 0
    for reviewer in list(month_reviewers)[:5]:  # 最初の5人だけテスト
        # 将来期間のデータ
        future_df = df[
            (df['reviewer_email'] == reviewer) &
            (df['request_time'] >= future_start) &
            (df['request_time'] < future_end)
        ]
        
        label = 1 if len(future_df) > 0 and (future_df['label'] == 1).any() else 0
        month_samples += 1
        sample_count += 1
    
    print(f"     サンプル数（最初の5人）: {month_samples}\n")

print(f"総サンプル数: {sample_count}")
print(f"スキップ数: {skipped_count}")
