#!/usr/bin/env python3
"""
Enhanced IRL (Attention) - 4×4クロス評価（正しい方法）

各学習期間でモデルを訓練し、各評価期間の軌跡で評価
- 訓練: 訓練期間 + Future Window X
- 評価: 評価期間 + Future Window X（同じFW）
"""
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = Path(__file__).parent.parent / "results_enhanced_irl"
RESULT_DIR.mkdir(exist_ok=True)

# Enhanced IRLスクリプトのパス
TRAIN_SCRIPT = Path(__file__).parent / "train_enhanced_irl_importants.py"
REVIEWS_DATA = PROJECT_ROOT / "data" / "review_requests_openstack_multi_5y_detail.csv"

# 訓練期間と評価期間の定義（importantsと同一）
train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

# 基準日（importantsと同一）
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


def main():
    logger.info("=" * 80)
    logger.info("Enhanced IRL (Attention) - 4×4クロス評価（正しい方法）")
    logger.info("=" * 80)
    logger.info(f"訓練スクリプト: {TRAIN_SCRIPT}")
    logger.info(f"データ: {REVIEWS_DATA}")
    logger.info(f"訓練期間: {TRAIN_START} ~ {TRAIN_END}")
    logger.info(f"評価期間: {EVAL_START} ~ {EVAL_END}")

    # 4×4のクロス評価
    logger.info("")
    logger.info("=" * 80)
    logger.info("4×4クロス評価")
    logger.info("=" * 80)
    logger.info("各(訓練FW, 評価FW)の組み合わせで評価")
    logger.info("")

    total_combinations = len(train_periods) * len(eval_periods)
    current = 0

    for train_period in train_periods:
        train_start_month, train_end_month = get_month_offset(train_period)

        logger.info(f"\n📊 train_{train_period} (FW: {train_start_month}-{train_end_month}m)")

        for eval_period in eval_periods:
            eval_start_month, eval_end_month = get_month_offset(eval_period)
            current += 1

            output_dir = RESULT_DIR / f"train_{train_period}" / f"eval_{eval_period}"
            output_dir.mkdir(parents=True, exist_ok=True)

            metrics_file = output_dir / "metrics.json"

            if metrics_file.exists():
                logger.info(f"  [{current}/{total_combinations}] ✅ スキップ: eval_{eval_period}")
                continue

            logger.info(f"  [{current}/{total_combinations}] 🔄 評価中: eval_{eval_period} (FW: {eval_start_month}-{eval_end_month}m)")

            # 重要: 訓練FWと評価FWを個別に指定
            # 訓練データは訓練FWで作成、評価データは評価FWで作成
            command = [
                "uv", "run", "python", str(TRAIN_SCRIPT),
                "--reviews", str(REVIEWS_DATA),
                "--train-start", TRAIN_START,
                "--train-end", TRAIN_END,
                "--eval-start", EVAL_START,
                "--eval-end", EVAL_END,
                "--future-window-start", str(eval_start_month),  # 評価FWで軌跡生成
                "--future-window-end", str(eval_end_month),
                "--epochs", "50",
                "--min-history-events", "3",
                "--output", str(output_dir),
                "--project", "openstack/nova"
            ]

            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                logger.info(f"  [{current}/{total_combinations}] ✅ 成功: eval_{eval_period}")
            except subprocess.CalledProcessError as e:
                logger.error(f"  [{current}/{total_combinations}] ❌ 失敗: eval_{eval_period}")
                logger.error(f"  エラー: {e.stderr[-500:]}")
                continue

    logger.info("")
    logger.info("=" * 80)
    logger.info("全ての訓練・評価が完了しました！")
    logger.info("=" * 80)

    # 結果マトリクスの作成（4×4）
    logger.info("")
    logger.info("=" * 80)
    logger.info("結果サマリー（Enhanced IRL - 4×4クロス評価）")
    logger.info("=" * 80)

    import numpy as np

    # メトリクスマトリクスを作成
    auc_roc_matrix = np.zeros((len(train_periods), len(eval_periods)))
    auc_pr_matrix = np.zeros((len(train_periods), len(eval_periods)))
    f1_matrix = np.zeros((len(train_periods), len(eval_periods)))

    for i, train_period in enumerate(train_periods):
        for j, eval_period in enumerate(eval_periods):
            metrics_file = RESULT_DIR / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"

            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)

                auc_roc_matrix[i, j] = data.get('auc_roc', 0.0)
                auc_pr_matrix[i, j] = data.get('auc_pr', 0.0)
                f1_matrix[i, j] = data.get('f1_score', 0.0)

    # AUC-ROCマトリクス表示
    logger.info("\n【AUC-ROC マトリクス】")
    logger.info(f"{'Train \\ Eval':<15} " + " ".join([f"{p:>8}" for p in eval_periods]))
    for i, train_period in enumerate(train_periods):
        values = " ".join([f"{auc_roc_matrix[i, j]:8.4f}" for j in range(len(eval_periods))])
        logger.info(f"{train_period:<15} {values}")

    # AUC-PRマトリクス表示
    logger.info("\n【AUC-PR マトリクス】")
    logger.info(f"{'Train \\ Eval':<15} " + " ".join([f"{p:>8}" for p in eval_periods]))
    for i, train_period in enumerate(train_periods):
        values = " ".join([f"{auc_pr_matrix[i, j]:8.4f}" for j in range(len(eval_periods))])
        logger.info(f"{train_period:<15} {values}")

    # F1マトリクス表示
    logger.info("\n【F1 Score マトリクス】")
    logger.info(f"{'Train \\ Eval':<15} " + " ".join([f"{p:>8}" for p in eval_periods]))
    for i, train_period in enumerate(train_periods):
        values = " ".join([f"{f1_matrix[i, j]:8.4f}" for j in range(len(eval_periods))])
        logger.info(f"{train_period:<15} {values}")

    # 統計情報
    logger.info("\n【統計情報】")
    non_zero_roc = auc_roc_matrix[auc_roc_matrix > 0]
    non_zero_pr = auc_pr_matrix[auc_pr_matrix > 0]
    non_zero_f1 = f1_matrix[f1_matrix > 0]

    if len(non_zero_roc) > 0:
        logger.info(f"平均AUC-ROC: {np.mean(non_zero_roc):.4f}")
    if len(non_zero_pr) > 0:
        logger.info(f"平均AUC-PR: {np.mean(non_zero_pr):.4f}")
    if len(non_zero_f1) > 0:
        logger.info(f"平均F1: {np.mean(non_zero_f1):.4f}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("完了！")
    logger.info(f"結果ディレクトリ: {RESULT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
