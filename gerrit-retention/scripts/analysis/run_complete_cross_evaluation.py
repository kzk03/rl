#!/usr/bin/env python3
"""
完全な4x4クロス評価を実行するスクリプト（再実行版）
"""

import subprocess
import sys
from pathlib import Path

def run_evaluation(train_period, eval_period, train_model_path, output_dir):
    """単一の評価を実行"""
    cmd = [
        "uv", "run", "python", "scripts/training/irl/train_irl_sliding_window.py",
        "--reviews", "data/review_requests_openstack_multi_5y_detail.csv",
        "--train-start", "2021-01-01",
        "--train-end", "2023-01-01", 
        "--eval-start", "2023-01-01",
        "--eval-end", "2024-01-01",
        "--future-window-start", eval_period.split('-')[0].replace('m', ''),
        "--future-window-end", eval_period.split('-')[1].replace('m', ''),
        "--epochs", "20",
        "--min-history-events", "3",
        "--output", str(output_dir),
        "--project", "openstack/nova",
        "--model", str(train_model_path)
    ]
    
    print(f"実行中: {train_period} -> {eval_period}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"エラー: {result.stderr}")
        return False
    return True

def main():
    base_dir = Path("outputs/sliding_cross_eval_nova_monthly_labels_2021_2023")
    
    # 訓練期間とモデルパス
    train_models = {
        "0-3m": base_dir / "train_0-3m" / "irl_model.pt",
        "3-6m": base_dir / "train_3-6m" / "irl_model.pt", 
        "6-9m": base_dir / "train_6-9m" / "irl_model.pt",
        "9-12m": base_dir / "train_9-12m" / "irl_model.pt"
    }
    
    # 評価期間
    eval_periods = ["0-3m", "3-6m", "6-9m", "9-12m"]
    
    # 4x4マトリックスを完全に実行
    for train_period, model_path in train_models.items():
        for eval_period in eval_periods:
            # 出力ディレクトリ
            output_dir = base_dir / f"train_{train_period}" / f"eval_{eval_period}"
            
            # 既に存在する場合はスキップ
            if output_dir.exists() and (output_dir / "metrics.json").exists():
                print(f"スキップ: {train_period} -> {eval_period} (既に存在)")
                continue
            
            # 評価実行
            success = run_evaluation(train_period, eval_period, model_path, output_dir)
            if not success:
                print(f"失敗: {train_period} -> {eval_period}")
                return 1
    
    print("全てのクロス評価が完了しました！")
    return 0

if __name__ == "__main__":
    sys.exit(main())
