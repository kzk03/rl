#!/usr/bin/env python3
"""
IRLモデルのクロス評価スクリプト（新規ディレクトリ版）

固定の訓練期間（2021-01-01 to 2023-01-01）で4つのラベル期間を使用してモデルを訓練し、
各モデルを全てのラベル期間で評価する。

出力: outputs/irl_cross_eval_new/
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設定
BASE_OUTPUT_DIR = Path("outputs/irl_cross_eval_new")
REVIEWS_DATA = "data/review_requests_openstack_multi_5y_detail.csv"

# 固定の訓練期間
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-01-01"

# 評価開始日
EVAL_START = "2023-01-01"

# ラベル期間の定義（future_start, future_end in months）
LABEL_PERIODS = [
    ("0-3m", 0, 3),    # 0-3ヶ月後
    ("3-6m", 3, 6),    # 3-6ヶ月後
    ("6-9m", 6, 9),    # 6-9ヶ月後
    ("9-12m", 9, 12),  # 9-12ヶ月後
]

def train_model(period_name: str, future_start: int, future_end: int):
    """指定されたラベル期間でモデルを訓練"""
    output_dir = BASE_OUTPUT_DIR / f"train_{period_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    command = [
        "uv", "run", "python",
        "scripts/training/irl/train_irl_review_acceptance.py",
        "--reviews", REVIEWS_DATA,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_START,
        "--eval-end", "2024-01-01",
        "--future-window-start", str(future_start),
        "--future-window-end", str(future_end),
        "--epochs", "30",
        "--min-history-events", "3",
        "--output", str(output_dir),
        "--project", "openstack/nova",
    ]
    
    logger.info(f"訓練中: train_{period_name}")
    logger.info(f"コマンド: {' '.join(command)}")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    logger.info(f"✓ 訓練完了: train_{period_name}")
    return output_dir

def evaluate_model(model_dir: Path, eval_period_name: str, future_start: int, future_end: int):
    """訓練済みモデルを指定されたラベル期間で評価"""
    eval_output_dir = model_dir / f"eval_{eval_period_name}"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "irl_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    
    command = [
        "uv", "run", "python",
        "scripts/training/irl/train_irl_review_acceptance.py",
        "--reviews", REVIEWS_DATA,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_START,
        "--eval-end", "2024-01-01",
        "--future-window-start", str(future_start),
        "--future-window-end", str(future_end),
        "--epochs", "20",
        "--min-history-events", "3",
        "--output", str(eval_output_dir),
        "--project", "openstack/nova",
        "--model", str(model_path),
    ]
    
    logger.info(f"  評価中: {model_dir.name} -> eval_{eval_period_name}")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    logger.info(f"  ✓ 評価完了: {model_dir.name} -> eval_{eval_period_name}")

def main():
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("訓練期間ごとにモデルを訓練")
    logger.info("=" * 60)
    
    # 1. 各ラベル期間でモデルを訓練
    trained_models = {}
    for period_name, future_start, future_end in LABEL_PERIODS:
        model_dir = train_model(period_name, future_start, future_end)
        trained_models[period_name] = model_dir
    
    logger.info("\n" + "=" * 60)
    logger.info("クロス評価")
    logger.info("=" * 60)
    
    # 2. 各モデルを全てのラベル期間で評価
    for train_period, model_dir in trained_models.items():
        logger.info(f"\nモデル: train_{train_period}")
        for eval_period, future_start, future_end in LABEL_PERIODS:
            evaluate_model(model_dir, eval_period, future_start, future_end)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ 全ての処理が完了しました！")
    logger.info(f"結果: {BASE_OUTPUT_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
