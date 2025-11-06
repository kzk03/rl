#!/usr/bin/env python3
"""
6ヶ月幅IRLのマトリクスを作成
"""
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path("importants/review_acceptance_cross_eval_nova_6month")

train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

# メトリクスマトリクスを作成
metrics = ['AUC_ROC', 'AUC_PR', 'F1', 'PRECISION', 'RECALL']
matrices = {metric: pd.DataFrame(index=train_periods, columns=eval_periods, dtype=float) for metric in metrics}

print("=" * 80)
print("6ヶ月幅IRL マトリクス作成")
print("=" * 80)

for train_period in train_periods:
    for eval_period in eval_periods:
        metrics_file = BASE_DIR / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"

        if not metrics_file.exists():
            print(f"⚠️  Missing: {metrics_file}")
            continue

        with open(metrics_file, 'r') as f:
            data = json.load(f)

        matrices['AUC_ROC'].loc[train_period, eval_period] = data['auc_roc']
        matrices['AUC_PR'].loc[train_period, eval_period] = data['auc_pr']
        matrices['F1'].loc[train_period, eval_period] = data['f1_score']
        matrices['PRECISION'].loc[train_period, eval_period] = data['precision']
        matrices['RECALL'].loc[train_period, eval_period] = data['recall']

        print(f"✅ {train_period} -> {eval_period}: AUC-ROC {data['auc_roc']:.3f}")

# マトリクスを保存
for metric, matrix in matrices.items():
    output_file = BASE_DIR / f"matrix_{metric}.csv"
    matrix.to_csv(output_file)
    print(f"\n保存: {output_file}")
    print(matrix.to_string(float_format='%.3f'))

print("\n" + "=" * 80)
print("マトリクス作成完了")
print("=" * 80)
print(f"出力先: {BASE_DIR}")
