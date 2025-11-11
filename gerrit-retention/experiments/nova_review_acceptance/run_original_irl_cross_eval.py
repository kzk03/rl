#!/usr/bin/env python3
"""
オリジナルIRL実装（train_irl_review_acceptance.py）を使った4×4クロス評価
Attention機能なしで、importantsの結果を完全再現
"""
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("/Users/kazuki-h/rl/gerrit-retention")
SCRIPT_PATH = BASE_DIR / "scripts/training/irl/train_irl_review_acceptance.py"
DATA_PATH = BASE_DIR / "data/review_requests_openstack_multi_5y_detail.csv"
OUTPUT_BASE = BASE_DIR / "experiments/nova_review_acceptance/results_original_irl_cv"

# 訓練期間（24ヶ月固定）
TRAIN_PERIODS = {
    "0-3m": ("2020-01-01", "2022-01-01"),
    "3-6m": ("2020-04-01", "2022-04-01"),
    "6-9m": ("2020-07-01", "2022-07-01"),
    "9-12m": ("2020-10-01", "2022-10-01"),
}

# Future Window設定（0-3ヶ月固定）
FW_START = 0
FW_END = 3

PROJECT = "openstack/nova"
EPOCHS = 10  # 高速化のため10エポック（オリジナルは50だがテスト用）

print("=" * 80)
print("オリジナルIRL 4×4クロス評価")
print(f"スクリプト: {SCRIPT_PATH}")
print(f"データ: {DATA_PATH}")
print(f"出力: {OUTPUT_BASE}")
print("=" * 80)
print()

# 各訓練期間でモデルを訓練
for train_key, (train_start, train_end) in TRAIN_PERIODS.items():
    print()
    print("=" * 80)
    print(f"訓練期間: {train_key} ({train_start} ~ {train_end})")
    print("=" * 80)
    
    output_dir = OUTPUT_BASE / f"train_{train_key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練実行
    cmd = [
        "uv", "run", "python", str(SCRIPT_PATH),
        "--reviews", str(DATA_PATH),
        "--train-start", train_start,
        "--train-end", train_end,
        "--future-window-start", str(FW_START),
        "--future-window-end", str(FW_END),
        "--output", str(output_dir),
        "--project", PROJECT,
        "--epochs", str(EPOCHS),
    ]
    
    print(f"コマンド: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=BASE_DIR, check=True)
    
    print(f"✓ 訓練完了: {output_dir}")
    
    # メトリクスを読み込んで表示
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  F1: {metrics['f1_score']:.4f}")

print()
print("=" * 80)
print("全4パターン完了！")
print(f"結果: {OUTPUT_BASE}/")
print("=" * 80)
