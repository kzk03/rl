#!/usr/bin/env python3
"""
レビュー承諾予測のクロス評価を実行（6ヶ月幅バージョン）

訓練期間と評価期間の組み合わせで評価を行う
Future Windows: 0-6m, 6-12m, 12-18m, 18-24m（6ヶ月幅）
"""
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_OUTPUT_DIR = Path("importants/review_acceptance_cross_eval_nova_6month")
TRAIN_SCRIPT = "scripts/training/irl/train_irl_review_acceptance.py"
REVIEWS_DATA = "data/review_requests_nova.csv"

# 訓練期間と評価期間の定義（名前は3ヶ月刻みだが、future windowは6ヶ月幅）
train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

# 基準日
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-01-01"
EVAL_START = "2023-01-01"
EVAL_END = "2024-01-01"

def get_6month_window(period_str):
    """
    期間文字列から6ヶ月幅のfuture windowを取得

    0-3m → (0, 6)   # 0～6ヶ月
    3-6m → (6, 12)  # 6～12ヶ月
    6-9m → (12, 18) # 12～18ヶ月
    9-12m → (18, 24) # 18～24ヶ月
    """
    start, end = period_str.split('-')
    start_month = int(start.replace('m', ''))

    # 6ヶ月幅にマッピング
    window_start = start_month * 2  # 0→0, 3→6, 6→12, 9→18
    window_end = window_start + 6    # 0→6, 6→12, 12→18, 18→24

    return window_start, window_end

logger.info("=" * 80)
logger.info("6ヶ月幅 IRL クロス評価")
logger.info("=" * 80)
logger.info("Future Windows: 0-6m, 6-12m, 12-18m, 18-24m")
logger.info(f"Output: {BASE_OUTPUT_DIR}")
logger.info("=" * 80)

# 1. 訓練期間ごとにモデルを訓練
logger.info("=" * 80)
logger.info("訓練期間ごとにモデルを訓練（6ヶ月幅）")
logger.info("=" * 80)

for train_period in train_periods:
    train_start_month, train_end_month = get_6month_window(train_period)

    output_dir = BASE_OUTPUT_DIR / f"train_{train_period}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "irl_model.pt"

    if model_path.exists():
        logger.info(f"スキップ: train_{train_period} (モデル既に存在)")
        continue

    logger.info(f"訓練中: train_{train_period}")
    logger.info(f"  Future window: {train_start_month}～{train_end_month}ヶ月（6ヶ月幅）")

    # 訓練期間の評価も同時に実行（0-3mの評価）
    command = [
        "uv", "run", "python", TRAIN_SCRIPT,
        "--reviews", REVIEWS_DATA,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_START,
        "--eval-end", EVAL_END,
        "--future-window-start", str(train_start_month),
        "--future-window-end", str(train_end_month),
        "--epochs", "30",  # エポック数を30に設定
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

# 2. 各訓練モデルで全評価期間を評価
logger.info("=" * 80)
logger.info("クロス評価を実行")
logger.info("=" * 80)

for train_period in train_periods:
    train_model_path = BASE_OUTPUT_DIR / f"train_{train_period}" / "irl_model.pt"

    if not train_model_path.exists():
        logger.warning(f"モデルが存在しません: {train_model_path}")
        continue

    for eval_period in eval_periods:
        eval_start_month, eval_end_month = get_6month_window(eval_period)

        output_dir = BASE_OUTPUT_DIR / f"train_{train_period}" / f"eval_{eval_period}"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = output_dir / "metrics.json"

        if metrics_file.exists():
            logger.info(f"スキップ: train_{train_period} -> eval_{eval_period} (既に存在)")
            continue

        logger.info(f"評価中: train_{train_period} -> eval_{eval_period}")
        logger.info(f"  Eval window: {eval_start_month}～{eval_end_month}ヶ月")

        command = [
            "uv", "run", "python", TRAIN_SCRIPT,
            "--reviews", REVIEWS_DATA,
            "--train-start", TRAIN_START,
            "--train-end", TRAIN_END,
            "--eval-start", EVAL_START,
            "--eval-end", EVAL_END,
            "--future-window-start", str(eval_start_month),
            "--future-window-end", str(eval_end_month),
            "--model", str(train_model_path),
            "--output", str(output_dir),
            "--project", "openstack/nova"
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"成功: train_{train_period} -> eval_{eval_period}")
        except subprocess.CalledProcessError as e:
            logger.error(f"エラー: {e.stderr}")
            logger.error(f"失敗: train_{train_period} -> eval_{eval_period}")
            continue

logger.info("=" * 80)
logger.info("クロス評価完了（6ヶ月幅）")
logger.info("=" * 80)
logger.info(f"結果: {BASE_OUTPUT_DIR}")
logger.info("次のステップ: scripts/analysis/visualize_cross_evaluation.py でマトリクス作成")
