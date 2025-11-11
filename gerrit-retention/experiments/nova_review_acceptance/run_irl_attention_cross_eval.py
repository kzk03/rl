#!/usr/bin/env python3
"""
Attention付きIRL - 4×4クロス評価実行スクリプト

オリジナル完全版 + Attention機構の4×4クロス評価を実行
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
# Future Window: スナップショットから0-3m, 3-6m, 6-9m, 9-12m後の貢献を予測
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
OUTPUT_BASE = ROOT / "experiments/nova_review_acceptance/outputs_irl_attention_cross_eval"


def run_training(fw_window, output_dir):
    """訓練を実行（特定のFuture Windowで）"""
    cmd = [
        "uv", "run", "python",
        str(ROOT / "experiments/nova_review_acceptance/scripts/train_irl_original_with_attention.py"),
        "--reviews", DATA_PATH,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_SNAPSHOT,
        "--eval-end", "2024-01-01",  # 評価期間全体
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
        str(ROOT / "experiments/nova_review_acceptance/scripts/train_irl_original_with_attention.py"),
        "--reviews", DATA_PATH,
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--eval-start", EVAL_SNAPSHOT,
        "--eval-end", "2024-01-01",
        "--future-window-start", str(fw_window["fw_start"]),
        "--future-window-end", str(fw_window["fw_end"]),
        "--min-history-events", "3",
        "--project", PROJECT,
        "--model", str(model_path),  # 既存モデルを使用（訓練をスキップ）
        "--output", str(output_dir),
    ]
    
    logger.info(f"評価コマンド: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"評価失敗: {fw_window['name']}")
        return False
    
    return True

def collect_results():
    """結果を収集してマトリクス作成（削除：不要）"""
    pass


def collect_cross_eval_results(cross_eval_results):
    """4×4クロス評価結果をマトリクス形式で保存"""
    if not cross_eval_results:
        logger.warning("クロス評価結果がありません")
        return
    
    # DataFrame作成
    results_df = pd.DataFrame(cross_eval_results)
    results_df.to_csv(OUTPUT_BASE / "cross_eval_results.csv", index=False)
    logger.info(f"結果保存: {OUTPUT_BASE / 'cross_eval_results.csv'}")
    
    # AUC-ROCマトリクス作成
    matrix_auc = results_df.pivot(
        index="train_fw",
        columns="eval_fw",
        values="auc_roc"
    )
    
    # 列の順序を固定
    fw_order = ["0-3m", "3-6m", "6-9m", "9-12m"]
    matrix_auc = matrix_auc.reindex(index=fw_order, columns=fw_order)
    
    matrix_auc.to_csv(OUTPUT_BASE / "matrix_AUC_ROC.csv")
    logger.info(f"AUC-ROCマトリクス保存: {OUTPUT_BASE / 'matrix_AUC_ROC.csv'}")
    
    # サマリー表示
    logger.info("\n" + "=" * 80)
    logger.info("🔥 4×4クロス評価 - AUC-ROCマトリクス")
    logger.info("=" * 80)
    logger.info("\n" + matrix_auc.to_string())
    
    # 対角線（同じFW）の平均
    diagonal_values = [matrix_auc.iloc[i, i] for i in range(len(fw_order)) if not pd.isna(matrix_auc.iloc[i, i])]
    if diagonal_values:
        logger.info(f"\n📊 対角線平均（訓練FW = 評価FW）: {np.mean(diagonal_values):.4f}")
    
    # 非対角線（異なるFW）の平均
    off_diagonal_values = []
    for i in range(len(fw_order)):
        for j in range(len(fw_order)):
            if i != j and not pd.isna(matrix_auc.iloc[i, j]):
                off_diagonal_values.append(matrix_auc.iloc[i, j])
    
    if off_diagonal_values:
        logger.info(f"📊 非対角線平均（訓練FW ≠ 評価FW）: {np.mean(off_diagonal_values):.4f}")
    
    # 全体平均
    all_values = results_df["auc_roc"].dropna()
    if len(all_values) > 0:
        logger.info(f"📊 全体平均: {all_values.mean():.4f}")
        logger.info(f"📊 最高AUC-ROC: {all_values.max():.4f}")
        logger.info(f"📊 最低AUC-ROC: {all_values.min():.4f}")


def main():
    logger.info("=" * 80)
    logger.info("🔥 Attention付きIRL - 4×4完全クロス評価 🔥")
    logger.info("=" * 80)
    logger.info(f"学習期間: {TRAIN_START} ～ {TRAIN_END}（固定）")
    logger.info(f"評価スナップショット: {EVAL_SNAPSHOT}（固定）")
    logger.info(f"Future Window: 0-3m, 3-6m, 6-9m, 9-12m の4パターン")
    logger.info(f"評価パターン: 4モデル × 4FW = 16通り")
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
        model_path = output_dir / "irl_model.pth"
        threshold_path = output_dir / "threshold.txt"
        
        if model_path.exists() and threshold_path.exists():
            trained_models[fw_window['name']] = {
                "model_path": model_path,
                "threshold_path": threshold_path,
                "fw_window": fw_window
            }
            logger.info(f"✅ 訓練完了: {fw_window['name']}")
        else:
            logger.warning(f"⚠️ モデルファイルが見つかりません: {fw_window['name']}")
    
    # 4×4クロス評価
    logger.info("\n【ステップ2】4×4クロス評価（16通り）")
    logger.info("=" * 80)
    
    cross_eval_results = []
    
    for train_fw_name, model_info in trained_models.items():
        for eval_fw in FUTURE_WINDOWS:
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
    collect_cross_eval_results(cross_eval_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 4×4完全クロス評価が完了しました！")
    logger.info("=" * 80)


def collect_results_fw():
    """Future Window別の結果を収集"""
    results = []
    
    for fw_window in FUTURE_WINDOWS:
        fw_name = fw_window["name"]
        metrics_path = OUTPUT_BASE / f"fw_{fw_name}" / "metrics.json"
        
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            results.append({
                "future_window": fw_name,
                "fw_start": fw_window["fw_start"],
                "fw_end": fw_window["fw_end"],
                "auc_roc": metrics.get("auc_roc", 0.0),
                "auc_pr": metrics.get("auc_pr", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
            })
        else:
            logger.warning(f"結果ファイルなし: {metrics_path}")
    
    # 結果保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_BASE / "fw_results.csv", index=False)
    logger.info(f"結果保存: {OUTPUT_BASE / 'fw_results.csv'}")
    
    # サマリー表示
    logger.info("=" * 80)
    logger.info("🔥 Attention付きIRL - Future Window別結果")
    logger.info("=" * 80)
    logger.info("\nFuture Window別AUC-ROC:")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['future_window']:10s} ({row['fw_start']}-{row['fw_end']}m後): {row['auc_roc']:.4f}")
    
    if len(results_df) > 0:
        logger.info(f"\n平均AUC-ROC: {results_df['auc_roc'].mean():.4f}")
        logger.info(f"最高AUC-ROC: {results_df['auc_roc'].max():.4f} ({results_df.loc[results_df['auc_roc'].idxmax(), 'future_window']})")
        logger.info(f"最低AUC-ROC: {results_df['auc_roc'].min():.4f} ({results_df.loc[results_df['auc_roc'].idxmin(), 'future_window']})")


if __name__ == "__main__":
    main()
