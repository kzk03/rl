#!/usr/bin/env python3
"""
IRLモデルのクロス評価スクリプト（強化版 - ベースライン対決用）

ベースラインとの違い:
1. min-history-events=0 (活動量制限なし)
2. epochs=70 (ベースライン30 vs 強化版70)
3. 全プロジェクト対象 (ベースライン: openstack/nova のみ)

出力: outputs/irl_cross_eval_enhanced/
"""

import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設定
BASE_OUTPUT_DIR = Path("outputs/irl_cross_eval_enhanced")
REVIEWS_DATA = "data/review_requests_openstack_multi_5y_detail.csv"

# 固定の訓練期間
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-01-01"

# 評価期間
EVAL_START = "2023-01-01"
EVAL_END = "2024-01-01"

# ラベル期間の定義（future_start, future_end in months）
LABEL_PERIODS = [
    ("0-3m", 0, 3),    # 0-3ヶ月後
    ("3-6m", 3, 6),    # 3-6ヶ月後
    ("6-9m", 6, 9),    # 6-9ヶ月後
    ("9-12m", 9, 12),  # 9-12ヶ月後
]

def train_model(period_name: str, future_start: int, future_end: int):
    """指定されたラベル期間でモデルを訓練（強化版パラメータ）"""
    output_dir = BASE_OUTPUT_DIR / f"train_{period_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    command = [
        "uv", "run", "python",
        "scripts/training/irl/train_irl_review_acceptance.py",
        "--reviews", REVIEWS_DATA,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_START,
        "--eval-end", EVAL_END,
        "--future-window-start", str(future_start),
        "--future-window-end", str(future_end),
        "--epochs", "70",           # ベースライン30 → 強化版70
        "--min-history-events", "0", # ベースライン3 → 強化版0（制限なし）
        "--output", str(output_dir),
        # --project を指定しない（全プロジェクト対象）
    ]
    
    logger.info(f"訓練中: train_{period_name} (epochs=70, min-history=0, 全プロジェクト)")
    logger.info(f"コマンド: {' '.join(command)}")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    logger.info(f"✓ 訓練完了: train_{period_name}")
    return output_dir

def evaluate_model(model_dir: Path, eval_period_name: str, future_start: int, future_end: int):
    """訓練済みモデルを指定されたラベル期間で評価（強化版パラメータ）"""
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
        "--eval-end", EVAL_END,
        "--future-window-start", str(future_start),
        "--future-window-end", str(future_end),
        "--epochs", "20",           # 評価時は20エポックで十分
        "--min-history-events", "0",
        "--output", str(eval_output_dir),
        "--model", str(model_path),
    ]
    
    logger.info(f"  評価中: {model_dir.name} -> eval_{eval_period_name}")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    logger.info(f"  ✓ 評価完了: {model_dir.name} -> eval_{eval_period_name}")

def main():
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("IRLクロス評価 - 強化版（ベースライン対決用）")
    logger.info("=" * 80)
    logger.info("パラメータ:")
    logger.info("  - min-history-events: 0 (活動量制限なし)")
    logger.info("  - epochs: 70 (ベースライン30の2.3倍)")
    logger.info("  - project: 全プロジェクト対象")
    logger.info("=" * 80)
    
    # 1. 各ラベル期間でモデルを訓練
    trained_models = {}
    for period_name, future_start, future_end in LABEL_PERIODS:
        model_dir = train_model(period_name, future_start, future_end)
        trained_models[period_name] = model_dir
    
    logger.info("\n" + "=" * 80)
    logger.info("クロス評価")
    logger.info("=" * 80)
    
    # 2. 各モデルを全てのラベル期間で評価
    for train_period, model_dir in trained_models.items():
        logger.info(f"\nモデル: train_{train_period}")
        for eval_period, future_start, future_end in LABEL_PERIODS:
            evaluate_model(model_dir, eval_period, future_start, future_end)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ 全ての処理が完了しました！")
    logger.info(f"結果: {BASE_OUTPUT_DIR}")
    logger.info("=" * 80)
    
    # ベースラインとの比較を促す
    logger.info("\n次のステップ:")
    logger.info("  ベースラインとの比較:")
    logger.info("  - ベースライン: importants/review_acceptance_cross_eval_nova/")
    logger.info("  - 強化版IRL: outputs/irl_cross_eval_enhanced/")

if __name__ == "__main__":
    main()
