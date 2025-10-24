#!/usr/bin/env python3
"""
クロス評価結果の集計スクリプト

Usage:
    python summarize_cross_evaluation.py <output_base_dir>
"""

import json
import sys
from pathlib import Path
import pandas as pd


def load_metrics(metrics_path: Path) -> dict:
    """メトリクスJSONファイルを読み込む"""
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def summarize_results(output_base: Path):
    """クロス評価結果を集計"""
    
    print("=" * 80)
    print("クロス評価結果サマリー")
    print("=" * 80)
    print()
    
    # 各訓練ラベルのメトリクスを収集
    results = []
    
    for train_dir in sorted(output_base.glob("train_eval_*")):
        # ディレクトリ名から訓練期間を抽出（例: train_eval_0-3m -> 0-3m）
        train_label = train_dir.name.replace("train_eval_", "")
        
        metrics_path = train_dir / "metrics.json"
        metrics = load_metrics(metrics_path)
        
        if metrics:
            results.append({
                "訓練ラベル": train_label,
                "AUC-ROC": f"{metrics.get('auc_roc', 0):.3f}",
                "AUC-PR": f"{metrics.get('auc_pr', 0):.3f}",
                "Precision": f"{metrics.get('precision', 0):.3f}",
                "Recall": f"{metrics.get('recall', 0):.3f}",
                "F1": f"{metrics.get('f1', 0):.3f}",
                "最適閾値": f"{metrics.get('optimal_threshold', 0):.3f}",
                "継続率": f"{metrics.get('continuation_rate', 0):.1%}",
                "サンプル数": metrics.get('sample_count', 0),
            })
    
    if not results:
        print("結果が見つかりませんでした。")
        return
    
    # データフレームとして表示
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    
    # CSVとして保存
    csv_path = output_base / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ 結果を保存: {csv_path}")
    print()
    
    # 重要な発見をハイライト
    print("=" * 80)
    print("重要な発見")
    print("=" * 80)
    print()
    
    # AUC-ROCが最も高いモデル
    best_auc_idx = df['AUC-ROC'].astype(float).idxmax()
    best_model = df.iloc[best_auc_idx]
    print(f"✓ 最高AUC-ROC: {best_model['訓練ラベル']} ({best_model['AUC-ROC']})")
    
    # F1が最も高いモデル
    best_f1_idx = df['F1'].astype(float).idxmax()
    best_f1_model = df.iloc[best_f1_idx]
    print(f"✓ 最高F1: {best_f1_model['訓練ラベル']} ({best_f1_model['F1']})")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_cross_evaluation.py <output_base_dir>")
        sys.exit(1)
    
    output_base = Path(sys.argv[1])
    
    if not output_base.exists():
        print(f"エラー: ディレクトリが存在しません: {output_base}")
        sys.exit(1)
    
    summarize_results(output_base)


if __name__ == "__main__":
    main()

