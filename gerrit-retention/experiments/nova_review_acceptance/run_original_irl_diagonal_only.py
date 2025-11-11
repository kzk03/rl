#!/usr/bin/env python3
"""
オリジナルIRL実装で完全4×4クロス評価を実行

訓練済みモデルを使って異なる期間で評価
"""
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("/Users/kazuki-h/rl/gerrit-retention")
ORIGINAL_SCRIPT = BASE_DIR / "scripts/training/irl/train_irl_review_acceptance.py"
DATA_PATH = BASE_DIR / "data/review_requests_openstack_multi_5y_detail.csv"
OUTPUT_BASE = BASE_DIR / "experiments/nova_review_acceptance/results_original_irl_full_cv"

# 訓練期間と評価スナップショット
PERIODS = {
    "0-3m": ("2020-01-01", "2022-01-01"),
    "3-6m": ("2020-04-01", "2022-04-01"),
    "6-9m": ("2020-07-01", "2022-07-01"),
    "9-12m": ("2020-10-01", "2022-10-01"),
}

PROJECT = "openstack/nova"
EPOCHS = 10

print("=" * 80)
print("オリジナルIRL 完全4×4クロス評価")
print("=" * 80)
print()
print("ステップ1: 各訓練期間でモデル訓練（4パターン）")
print("ステップ2: 既存のモデルを使って全期間で評価予定（16パターン）")
print()
print("⚠️ 注意: オリジナルスクリプトは評価専用モードがないため、")
print("   現時点では対角線（4パターン）のみ実行可能")
print("=" * 80)
print()

# 各訓練期間でモデル訓練
results = {}

for period_key, (train_start, train_end) in PERIODS.items():
    print(f"\n{'='*80}")
    print(f"訓練期間: {period_key} ({train_start} ~ {train_end})")
    print(f"{'='*80}")
    
    output_dir = OUTPUT_BASE / f"train_{period_key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "uv", "run", "python", str(ORIGINAL_SCRIPT),
        "--reviews", str(DATA_PATH),
        "--train-start", train_start,
        "--train-end", train_end,
        "--future-window-start", "0",
        "--future-window-end", "3",
        "--output", str(output_dir),
        "--project", PROJECT,
        "--epochs", str(EPOCHS),
    ]
    
    print(f"実行中...")
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ エラー")
        print(result.stderr[:500])
        continue
    
    # 結果を読み込み
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        results[f"{period_key}_{period_key}"] = metrics['auc_roc']
        
        print(f"✓ 完了: AUC-ROC = {metrics['auc_roc']:.4f}")
    else:
        print(f"⚠️ メトリクスファイルが見つかりません")

print()
print("=" * 80)
print("実行完了サマリー")
print("=" * 80)
print()
print("対角線結果（訓練期間 = 評価期間）:")
for key, auc in results.items():
    train_period = key.split('_')[0] + "-" + key.split('_')[1]
    print(f"  {train_period}: AUC-ROC = {auc:.4f}")

if results:
    import numpy as np
    aucs = list(results.values())
    print()
    print(f"平均AUC-ROC: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")
    print(f"範囲: {min(aucs):.4f} - {max(aucs):.4f}")

print()
print("=" * 80)
print(f"結果保存先: {OUTPUT_BASE}/")
print("=" * 80)
