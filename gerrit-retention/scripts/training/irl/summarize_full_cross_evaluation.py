#!/usr/bin/env python3
"""
完全クロス評価結果の集計スクリプト

訓練ラベル × 評価期間のマトリクスを生成

Usage:
    python summarize_full_cross_evaluation.py <output_base_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_metrics(metrics_path: Path) -> dict:
    """メトリクスJSONファイルを読み込む"""
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def summarize_results(output_base: Path):
    """完全クロス評価結果を集計"""
    
    print("=" * 100)
    print("完全クロス評価結果サマリー")
    print("=" * 100)
    print()
    
    # 訓練ラベルと評価期間の定義
    train_labels = ["0-1m", "0-3m", "0-6m", "0-9m", "0-12m"]
    eval_windows = ["0-3m", "3-6m", "6-9m", "9-12m"]
    
    # 各メトリクスごとにマトリクスを作成
    metrics_names = ["AUC-ROC", "AUC-PR", "Precision", "Recall", "F1", "継続率"]
    metrics_keys = ["auc_roc", "auc_pr", "precision", "recall", "f1", "continuation_rate"]
    
    matrices = {name: pd.DataFrame(index=train_labels, columns=eval_windows) 
                for name in metrics_names}
    
    # 詳細結果リスト
    detailed_results = []
    
    # データ収集
    for train_label in train_labels:
        train_dir = output_base / f"train_{train_label}"
        
        for eval_window in eval_windows:
            eval_dir = train_dir / f"eval_{eval_window}"
            metrics_path = eval_dir / "metrics.json"
            
            metrics = load_metrics(metrics_path)
            
            if metrics:
                # マトリクスに値を設定
                for metric_name, metric_key in zip(metrics_names, metrics_keys):
                    value = metrics.get(metric_key, np.nan)
                    if metric_name == "継続率":
                        matrices[metric_name].loc[train_label, eval_window] = f"{value:.1%}" if not np.isnan(value) else "N/A"
                    else:
                        matrices[metric_name].loc[train_label, eval_window] = f"{value:.3f}" if not np.isnan(value) else "N/A"
                
                # 詳細結果に追加
                detailed_results.append({
                    "訓練ラベル": train_label,
                    "評価期間": eval_window,
                    "AUC-ROC": metrics.get("auc_roc", 0),
                    "AUC-PR": metrics.get("auc_pr", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1": metrics.get("f1", 0),
                    "最適閾値": metrics.get("optimal_threshold", 0),
                    "継続率": metrics.get("continuation_rate", 0),
                    "サンプル数": metrics.get("sample_count", 0),
                })
    
    if not detailed_results:
        print("結果が見つかりませんでした。")
        return
    
    # マトリクス表示
    print("【AUC-ROCマトリクス】")
    print("訓練ラベル × 評価期間")
    print()
    print(matrices["AUC-ROC"].to_string())
    print()
    print()
    
    print("【F1スコアマトリクス】")
    print("訓練ラベル × 評価期間")
    print()
    print(matrices["F1"].to_string())
    print()
    print()
    
    print("【継続率マトリクス】")
    print("訓練ラベル × 評価期間")
    print()
    print(matrices["継続率"].to_string())
    print()
    print()
    
    # 詳細結果をDataFrameに変換
    df = pd.DataFrame(detailed_results)
    
    # CSVとして保存
    csv_path = output_base / "full_cross_evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ 詳細結果を保存: {csv_path}")
    print()
    
    # 各マトリクスもCSVとして保存
    for metric_name, matrix in matrices.items():
        matrix_csv = output_base / f"matrix_{metric_name.replace('-', '_').replace('率', 'rate')}.csv"
        matrix.to_csv(matrix_csv)
        print(f"✓ {metric_name}マトリクスを保存: {matrix_csv}")
    print()
    
    # 重要な発見をハイライト
    print("=" * 100)
    print("重要な発見")
    print("=" * 100)
    print()
    
    # 数値型に変換（文字列から）
    df_numeric = df.copy()
    for col in ["AUC-ROC", "AUC-PR", "F1", "Precision", "Recall"]:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # AUC-ROCが最も高い組み合わせ
    best_auc_idx = df_numeric['AUC-ROC'].idxmax()
    best_auc = df.iloc[best_auc_idx]
    print(f"✓ 最高AUC-ROC: {best_auc['訓練ラベル']} → {best_auc['評価期間']} (AUC-ROC: {best_auc['AUC-ROC']:.3f})")
    
    # F1が最も高い組み合わせ
    best_f1_idx = df_numeric['F1'].idxmax()
    best_f1 = df.iloc[best_f1_idx]
    print(f"✓ 最高F1: {best_f1['訓練ラベル']} → {best_f1['評価期間']} (F1: {best_f1['F1']:.3f})")
    
    print()
    
    # 訓練ラベル別の平均性能
    print("【訓練ラベル別の平均性能】")
    print()
    train_avg = df_numeric.groupby('訓練ラベル')[['AUC-ROC', 'F1']].mean()
    print(train_avg.to_string())
    print()
    
    # 評価期間別の平均性能
    print("【評価期間別の平均性能】")
    print()
    eval_avg = df_numeric.groupby('評価期間')[['AUC-ROC', 'F1']].mean()
    print(eval_avg.to_string())
    print()
    
    # 対角成分（訓練と評価が同じ期間）の性能
    print("【訓練期間と評価期間が一致する場合の性能】")
    print()
    # 0-3m → 0-3m, 0-6m → 3-6m（該当なし）などは一致しないため、
    # 累積窓での一致を確認
    diagonal_results = []
    for idx, row in df.iterrows():
        train = row['訓練ラベル']
        eval_period = row['評価期間']
        # 訓練が0-3mで評価が0-3mなど
        if train == eval_period or (train == "0-6m" and eval_period == "0-3m"):
            diagonal_results.append(row)
    
    if diagonal_results:
        diag_df = pd.DataFrame(diagonal_results)
        print(diag_df[['訓練ラベル', '評価期間', 'AUC-ROC', 'F1']].to_string(index=False))
    else:
        print("該当なし（訓練と評価の期間定義が異なるため）")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_full_cross_evaluation.py <output_base_dir>")
        sys.exit(1)
    
    output_base = Path(sys.argv[1])
    
    if not output_base.exists():
        print(f"エラー: ディレクトリが存在しません: {output_base}")
        sys.exit(1)
    
    summarize_results(output_base)


if __name__ == "__main__":
    main()

