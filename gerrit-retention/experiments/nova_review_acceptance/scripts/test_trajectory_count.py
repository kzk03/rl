#!/usr/bin/env python3
"""
importants準拠の設定でtrajectory数をテスト
期待: 0-3m → 約23軌跡, 3-6m → 約17軌跡, 6-9m → 約11軌跡, 9-12m → 約5軌跡
(importantsは905, 540, 268, 102だが、それはレビュワー数×月数の総軌跡数)
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "experiments" / "nova_review_acceptance" / "scripts"))

from run_cross_eval import prepare_period_data

# データ読み込み
data_path = project_root / "data" / "review_requests_openstack_multi_5y_detail.csv"
df = pd.read_csv(data_path)
df['request_time'] = pd.to_datetime(df['request_time'])
df = df[df['project'] == 'openstack/nova'].copy()

print(f"Nova: {len(df)} レコード, {df['reviewer_email'].nunique()} レビュワー\n")

# importants準拠設定
base_train_start = datetime(2021, 1, 1)
train_end = datetime(2023, 1, 1)
base_eval_start = datetime(2023, 1, 1)

train_configs = [
    {'name': '0-3m', 'fw_start': 0, 'fw_end': 3},
    {'name': '3-6m', 'fw_start': 3, 'fw_end': 6},
    {'name': '6-9m', 'fw_start': 6, 'fw_end': 9},
    {'name': '9-12m', 'fw_start': 9, 'fw_end': 12},
]

eval_config = {'name': '0-3m', 'offset': 0, 'months': 3}

print("=" * 80)
print("訓練軌跡数のテスト（importants準拠）")
print("=" * 80)

for train_cfg in train_configs:
    train_start = base_train_start
    eval_start = base_eval_start
    
    train_df, eval_df = prepare_period_data(
        df, train_start, train_end,
        eval_start, eval_config['months'],
        future_window_start_months=train_cfg['fw_start'],
        future_window_months=train_cfg['fw_end'] - train_cfg['fw_start']
    )
    
    print(f"\n{train_cfg['name']} (FW: {train_cfg['fw_start']}~{train_cfg['fw_end']}ヶ月):")
    print(f"  訓練軌跡: {len(train_df)} サンプル")
    print(f"  継続率: {train_df['label'].mean():.3f}")
    print(f"  評価サンプル: {len(eval_df)}")

print("\n" + "=" * 80)
print("期待値（importantsドキュメント記載）:")
print("  0-3m: 905軌跡 (23ヶ月 × 約39レビュワー)")
print("  3-6m: 540軌跡 (17ヶ月 × 約32レビュワー)")
print("  6-9m: 268軌跡 (11ヶ月 × 約24レビュワー)")
print("  9-12m: 102軌跡 (5ヶ月 × 約20レビュワー)")
print("=" * 80)
