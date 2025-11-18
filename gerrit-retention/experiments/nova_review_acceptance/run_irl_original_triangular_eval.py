#!/usr/bin/env python3
"""
オリジナルIRL - 三角クロス評価（訓練FW以降のみ評価）

各モデルは訓練したFW以降の期間のみで評価
- 0-3mモデル → 0-3m, 3-6m, 6-9m, 9-12m (4通り)
- 3-6mモデル → 3-6m, 6-9m, 9-12m (3通り)
- 6-9mモデル → 6-9m, 9-12m (2通り)
- 9-12mモデル → 9-12m (1通り)
合計: 10通り
"""
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトルート
ROOT = Path(__file__).resolve().parents[2]

# 学習期間: 2021-01-01 ～ 2023-01-01（24ヶ月、固定）
# 評価スナップショット: 2023-01-01（固定）
FUTURE_WINDOWS = [
    {"name": "0-3m", "fw_start": 0, "fw_end": 3},
    {"name": "3-6m", "fw_start": 3, "fw_end": 6},
    {"name": "6-9m", "fw_start": 6, "fw_end": 9},
    {"name": "9-12m", "fw_start": 9, "fw_end": 12},
]

# 固定期間
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-01-01"
EVAL_SNAPSHOT = "2023-01-01"

DATA_PATH = "data/review_requests_openstack_multi_5y_detail.csv"
PROJECT = "openstack/nova"
EPOCHS = 50
OUTPUT_BASE = ROOT / "experiments/nova_review_acceptance/outputs_irl_original_triangular_eval"


def run_training(fw_window, output_dir):
    """訓練を実行（特定のFuture Windowで）"""
    cmd = [
        "uv", "run", "python",
        str(ROOT / "scripts/training/irl/train_irl_review_acceptance.py"),
        "--reviews", DATA_PATH,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_SNAPSHOT,
        "--eval-end", "2024-01-01",
        "--future-window-start", str(fw_window["fw_start"]),
        "--future-window-end", str(fw_window["fw_end"]),
        "--epochs", str(EPOCHS),
        "--min-history-events", "3",
        "--project", PROJECT,
        "--output", str(output_dir),
    ]
    
    logger.info(f"実行コマンド: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"訓練失敗: {fw_window['name']}")
        return False
    
    return True


def evaluate_with_model(model_path, threshold_path, fw_window, output_dir):
    """既存モデルで評価（異なるFuture Windowで）"""
    cmd = [
        "uv", "run", "python",
        str(ROOT / "scripts/training/irl/train_irl_review_acceptance.py"),
        "--reviews", DATA_PATH,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_SNAPSHOT,
        "--eval-end", "2024-01-01",
        "--future-window-start", str(fw_window["fw_start"]),
        "--future-window-end", str(fw_window["fw_end"]),
        "--min-history-events", "3",
        "--project", PROJECT,
        "--model", str(model_path),
        "--output", str(output_dir),
    ]
    
    logger.info(f"評価コマンド: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"評価失敗: {fw_window['name']}")
        return False
    
    return True


def collect_triangular_eval_results(cross_eval_results):
    """三角クロス評価結果をマトリクス形式で保存"""
    if not cross_eval_results:
        logger.warning("クロス評価結果がありません")
        return
    
    # DataFrame作成
    results_df = pd.DataFrame(cross_eval_results)
    results_df.to_csv(OUTPUT_BASE / "triangular_eval_results.csv", index=False)
    logger.info(f"結果保存: {OUTPUT_BASE / 'triangular_eval_results.csv'}")
    
    # AUC-ROCマトリクス作成（三角行列）
    matrix_auc = results_df.pivot(
        index="train_fw",
        columns="eval_fw",
        values="auc_roc"
    )
    
    # 列の順序を固定
    fw_order = ["0-3m", "3-6m", "6-9m", "9-12m"]
    matrix_auc = matrix_auc.reindex(index=fw_order, columns=fw_order)
    
    matrix_auc.to_csv(OUTPUT_BASE / "matrix_AUC_ROC_triangular.csv")
    logger.info(f"AUC-ROCマトリクス保存: {OUTPUT_BASE / 'matrix_AUC_ROC_triangular.csv'}")
    
    # サマリー表示
    logger.info("\n" + "=" * 80)
    logger.info("🔥 オリジナルIRL - 三角クロス評価 AUC-ROCマトリクス")
    logger.info("=" * 80)
    logger.info("\n" + matrix_auc.to_string())
    
    # 対角線（同じFW）の平均
    diagonal_values = [matrix_auc.iloc[i, i] for i in range(len(fw_order)) if not pd.isna(matrix_auc.iloc[i, i])]
    if diagonal_values:
        logger.info(f"\n📊 対角線平均（訓練FW = 評価FW）: {np.mean(diagonal_values):.4f}")
    
    # 上三角（訓練FW < 評価FW: 将来予測）
    upper_triangle_values = []
    for i in range(len(fw_order)):
        for j in range(i+1, len(fw_order)):
            if not pd.isna(matrix_auc.iloc[i, j]):
                upper_triangle_values.append(matrix_auc.iloc[i, j])
    
    if upper_triangle_values:
        logger.info(f"📊 上三角平均（訓練FW < 評価FW: 将来予測）: {np.mean(upper_triangle_values):.4f}")
        logger.info(f"   件数: {len(upper_triangle_values)}通り")
    
    # 全体平均（三角部分のみ）
    all_values = results_df["auc_roc"].dropna()
    if len(all_values) > 0:
        logger.info(f"📊 全体平均（10通り）: {all_values.mean():.4f}")
        logger.info(f"📊 最高AUC-ROC: {all_values.max():.4f}")
        logger.info(f"📊 最低AUC-ROC: {all_values.min():.4f}")
    
    # FW別の平均性能
    logger.info("\n" + "=" * 80)
    logger.info("📊 訓練FW別の平均性能（そのFW以降の評価）")
    logger.info("=" * 80)
    for train_fw in fw_order:
        fw_results = results_df[results_df["train_fw"] == train_fw]
        if len(fw_results) > 0:
            avg_auc = fw_results["auc_roc"].mean()
            n_evals = len(fw_results)
            logger.info(f"{train_fw:10s}: 平均AUC-ROC {avg_auc:.4f} ({n_evals}個の評価FW)")


def main():
    logger.info("=" * 80)
    logger.info("🔥 オリジナルIRL（Attentionなし） - 三角クロス評価 🔥")
    logger.info("=" * 80)
    logger.info(f"学習期間: {TRAIN_START} ～ {TRAIN_END}（固定）")
    logger.info(f"評価スナップショット: {EVAL_SNAPSHOT}（固定）")
    logger.info(f"Future Window: 0-3m, 3-6m, 6-9m, 9-12m の4パターン")
    logger.info(f"評価パターン: 三角行列（訓練FW以降のみ）= 10通り")
    logger.info("=" * 80)
    
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # 各Future Windowでモデルを訓練
    logger.info("\n【ステップ1】4つのモデルを訓練")
    logger.info("=" * 80)
    
    trained_models = {}
    
    for fw_window in FUTURE_WINDOWS:
        logger.info(f"\n訓練開始: Future Window {fw_window['name']} ({fw_window['fw_start']}-{fw_window['fw_end']}ヶ月後)")
        
        output_dir = OUTPUT_BASE / f"train_{fw_window['name']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not run_training(fw_window, output_dir):
            logger.error(f"訓練失敗: {fw_window['name']}")
            continue
        
        # モデルパスを保存
        model_path = output_dir / "irl_model.pt"
        threshold_path = output_dir / "optimal_threshold.json"
        
        if model_path.exists() and threshold_path.exists():
            trained_models[fw_window['name']] = {
                "model_path": model_path,
                "threshold_path": threshold_path,
                "fw_window": fw_window,
                "fw_index": FUTURE_WINDOWS.index(fw_window)
            }
            logger.info(f"✅ 訓練完了: {fw_window['name']}")
        else:
            logger.warning(f"⚠️ モデルファイルが見つかりません: {fw_window['name']}")
    
    # 三角クロス評価（訓練FW以降のみ）
    logger.info("\n【ステップ2】三角クロス評価（10通り）")
    logger.info("=" * 80)
    
    cross_eval_results = []
    
    for train_fw_name, model_info in trained_models.items():
        train_fw_idx = model_info["fw_index"]
        
        # このモデルの訓練FW以降のFWで評価
        for eval_fw_idx in range(train_fw_idx, len(FUTURE_WINDOWS)):
            eval_fw = FUTURE_WINDOWS[eval_fw_idx]
            
            logger.info(f"\n評価: {train_fw_name}モデル → {eval_fw['name']}FW")
            
            eval_output_dir = OUTPUT_BASE / f"train_{train_fw_name}" / f"eval_{eval_fw['name']}"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            
            if not evaluate_with_model(
                model_info["model_path"],
                model_info["threshold_path"],
                eval_fw,
                eval_output_dir
            ):
                logger.error(f"評価失敗: {train_fw_name} → {eval_fw['name']}")
                continue
            
            # 結果を収集
            metrics_path = eval_output_dir / "metrics.json"
            if metrics_path.exists():
                import json
                with open(metrics_path) as f:
                    metrics = json.load(f)
                
                cross_eval_results.append({
                    "train_fw": train_fw_name,
                    "eval_fw": eval_fw["name"],
                    "auc_roc": metrics.get("auc_roc", 0.0),
                    "auc_pr": metrics.get("auc_pr", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                })
                logger.info(f"✅ AUC-ROC: {metrics.get('auc_roc', 0.0):.4f}")
    
    # 結果収集
    logger.info("\n【ステップ3】結果をマトリクス形式で保存")
    collect_triangular_eval_results(cross_eval_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 三角クロス評価が完了しました！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
