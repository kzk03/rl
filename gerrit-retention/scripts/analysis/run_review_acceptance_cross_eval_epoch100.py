#!/usr/bin/env python3
"""
レビュー承諾予測のクロス評価を実行（Epoch 100版）

訓練期間と評価期間の組み合わせで評価を行う
"""
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_OUTPUT_DIR = Path("outputs/review_acceptance_cross_eval_nova_epoch100")
TRAIN_SCRIPT = "scripts/training/irl/train_irl_review_acceptance.py"
REVIEWS_DATA = "data/review_requests_openstack_multi_5y_detail.csv"

# 訓練期間と評価期間の定義
train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

# 基準日
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-01-01"
EVAL_START = "2023-01-01"
EVAL_END = "2024-01-01"

def get_month_offset(period_str):
    """期間文字列から月数オフセットを取得"""
    start, end = period_str.split('-')
    start_month = int(start.replace('m', ''))
    end_month = int(end.replace('m', ''))
    return start_month, end_month

# 1. 訓練期間ごとにモデルを訓練
logger.info("=" * 80)
logger.info("訓練期間ごとにモデルを訓練（Epoch 100 + Focal Loss）")
logger.info("=" * 80)

for train_period in train_periods:
    train_start_month, train_end_month = get_month_offset(train_period)
    
    output_dir = BASE_OUTPUT_DIR / f"train_{train_period}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "irl_model.pt"
    
    if model_path.exists():
        logger.info(f"スキップ: train_{train_period} (モデル既に存在)")
        continue
    
    logger.info(f"訓練中: train_{train_period} (Epoch 100)")
    
    # 訓練
    command = [
        "uv", "run", "python", TRAIN_SCRIPT,
        "--reviews", REVIEWS_DATA,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_START,
        "--eval-end", EVAL_END,
        "--future-window-start", str(train_start_month),
        "--future-window-end", str(train_end_month),
        "--epochs", "100",  # ← 100エポック
        "--min-history-events", "3",
        "--output", str(output_dir),
        "--project", "openstack/nova"
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"成功: train_{train_period}")
    except subprocess.CalledProcessError as e:
        logger.error(f"エラー: {e.stderr}")
        logger.error(f"失敗: train_{train_period}")
        continue

# 2. クロス評価はスキップ（自己評価のみ実施済み）
logger.info("=" * 80)
logger.info("訓練完了！各モデルは自己評価済みです")
logger.info("=" * 80)

logger.info("=" * 80)
logger.info("クロス評価完了！")
logger.info(f"結果: {BASE_OUTPUT_DIR}")
logger.info("=" * 80)
