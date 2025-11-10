#!/usr/bin/env python3
"""
Nova レビュー受諾予測 - 4×4クロス評価（importants完全準拠版）

importantsのスクリプトを直接使用して、Enhanced IRLとの公平な比較を実施
"""
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = Path(__file__).parent.parent / "results_importants"
RESULT_DIR.mkdir(exist_ok=True)

# importantsスクリプトのパス
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "training" / "irl" / "train_irl_review_acceptance.py"
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
    logger.info("Nova レビュー受諾予測 - 4×4クロス評価（importants準拠）")
    logger.info("=" * 80)
    logger.info(f"訓練スクリプト: {TRAIN_SCRIPT}")
    logger.info(f"データ: {REVIEWS_DATA}")
    logger.info(f"訓練期間: {TRAIN_START} ~ {TRAIN_END}")
    logger.info(f"評価期間: {EVAL_START} ~ {EVAL_END}")
    
    # 1. 訓練期間ごとにモデルを訓練（Attention-less IRL）
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 1: 訓練期間ごとにモデルを訓練（Attention-less IRL）")
    logger.info("=" * 80)
    
    for train_period in train_periods:
        train_start_month, train_end_month = get_month_offset(train_period)
        
        output_dir = RESULT_DIR / f"train_{train_period}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "irl_model.pt"
        
        if model_path.exists():
            logger.info(f"✅ スキップ: train_{train_period} (モデル既に存在)")
            continue
        
        logger.info(f"🔄 訓練中: train_{train_period} (FW: {train_start_month}~{train_end_month}ヶ月)")
        
        # importantsのスクリプトを実行（対角線評価も同時に実行）
        command = [
            "uv", "run", "python", str(TRAIN_SCRIPT),
            "--reviews", str(REVIEWS_DATA),
            "--train-start", TRAIN_START,
            "--train-end", TRAIN_END,
            "--eval-start", EVAL_START,
            "--eval-end", EVAL_END,
            "--future-window-start", str(train_start_month),
            "--future-window-end", str(train_end_month),
            "--epochs", "50",  # Enhanced IRLと同じエポック数
            "--min-history-events", "3",
            "--output", str(output_dir),
            "--project", "openstack/nova"
        ]
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"✅ 成功: train_{train_period}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ エラー: {e.stderr}")
            logger.error(f"失敗: train_{train_period}")
            continue
    
    # 2. 各訓練モデルで全評価期間を評価（クロス評価）
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: クロス評価を実行")
    logger.info("=" * 80)
    
    for train_period in train_periods:
        train_model_path = RESULT_DIR / f"train_{train_period}" / "irl_model.pt"
        
        if not train_model_path.exists():
            logger.warning(f"⚠️  モデルが存在しません: {train_model_path}")
            continue
        
        for eval_period in eval_periods:
            eval_start_month, eval_end_month = get_month_offset(eval_period)
            
            output_dir = RESULT_DIR / f"train_{train_period}" / f"eval_{eval_period}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = output_dir / "metrics.json"
            
            if metrics_file.exists():
                logger.info(f"✅ スキップ: train_{train_period} -> eval_{eval_period} (既に存在)")
                continue
            
            logger.info(f"🔄 評価中: train_{train_period} -> eval_{eval_period}")
            
            # 評価期間のFuture Windowで評価
            command = [
                "uv", "run", "python", str(TRAIN_SCRIPT),
                "--reviews", str(REVIEWS_DATA),
                "--train-start", TRAIN_START,
                "--train-end", TRAIN_END,
                "--eval-start", EVAL_START,
                "--eval-end", EVAL_END,
                "--future-window-start", str(eval_start_month),
                "--future-window-end", str(eval_end_month),
                "--epochs", "20",
                "--min-history-events", "3",
                "--output", str(output_dir),
                "--project", "openstack/nova",
                "--model", str(train_model_path)
            ]
            
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                logger.info(f"✅ 成功: train_{train_period} -> eval_{eval_period}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ エラー: {e.stderr}")
                logger.error(f"失敗: train_{train_period} -> eval_{eval_period}")
                continue
    
    # 3. 結果マトリクスの作成
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 3: 結果マトリクスの作成")
    logger.info("=" * 80)
    
    import numpy as np
    
    metrics = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall']
    matrices = {metric: np.zeros((4, 4)) for metric in metrics}
    
    for i, train_period in enumerate(train_periods):
        for j, eval_period in enumerate(eval_periods):
            metrics_file = RESULT_DIR / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                
                for metric in metrics:
                    value = data.get(metric, 0.0)
                    matrices[metric][i, j] = value
                    
                logger.info(f"train_{train_period} -> eval_{eval_period}: "
                           f"AUC-ROC={data.get('auc_roc', 0):.4f}, "
                           f"AUC-PR={data.get('auc_pr', 0):.4f}")
            else:
                logger.warning(f"⚠️  メトリクスなし: train_{train_period} -> eval_{eval_period}")
    
    # マトリクスをCSVで保存
    import pandas as pd
    
    for metric, matrix in matrices.items():
        df = pd.DataFrame(matrix, index=train_periods, columns=eval_periods)
        csv_path = RESULT_DIR / f"matrix_{metric.upper()}.csv"
        df.to_csv(csv_path)
        logger.info(f"✅ 保存: {csv_path}")
    
    # サマリー
    logger.info("")
    logger.info("=" * 80)
    logger.info("実験サマリー（Attention-less IRL）")
    logger.info("=" * 80)
    
    auc_roc_matrix = matrices['auc_roc']
    diagonal = np.diag(auc_roc_matrix)
    
    logger.info(f"対角線平均AUC-ROC: {diagonal.mean():.4f}")
    logger.info(f"全体平均AUC-ROC: {auc_roc_matrix.mean():.4f}")
    logger.info(f"最高AUC-ROC: {auc_roc_matrix.max():.4f}")
    logger.info(f"最低AUC-ROC: {auc_roc_matrix.min():.4f}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("全てのクロス評価が完了しました！")
    logger.info(f"結果ディレクトリ: {RESULT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
