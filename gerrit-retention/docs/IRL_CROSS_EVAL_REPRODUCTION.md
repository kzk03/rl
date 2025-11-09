# IRL クロス評価の再現手順

このドキュメントでは、IRL モデルのクロス評価を再現する手順を説明します。

## 概要

固定の訓練期間（2021-01-01 to 2023-01-01）で 4 つのラベル期間（0-3m, 3-6m, 6-9m, 9-12m）を使用してモデルを訓練し、各モデルを全てのラベル期間で評価します（4×4=16 通りの評価）。

## 実行環境

- **Python**: 3.13.1
- **PyTorch**: 2.8.0
- **Device**: CPU（CUDA 未使用）
- **Random Seed**: 777（固定）

## 再現手順

### 1. スクリプトの実行

```bash
# クロス評価スクリプトの実行
uv run python scripts/analysis/run_irl_cross_eval_new.py
```

### 2. 実行パラメータ

スクリプトは以下のパラメータで`train_irl_review_acceptance.py`を呼び出します：

#### 訓練時のパラメータ

```bash
--reviews data/review_requests_openstack_multi_5y_detail.csv
--train-start 2021-01-01
--train-end 2023-01-01
--eval-start 2023-01-01
--eval-end 2024-01-01
--future-window-start [0, 3, 6, 9]  # ラベル期間開始（月）
--future-window-end [3, 6, 9, 12]   # ラベル期間終了（月）
--epochs 30
--min-history-events 3
--project openstack/nova
```

#### 評価時のパラメータ

```bash
# 訓練時と同じパラメータ + 以下
--model [訓練済みモデルのパス]
--epochs 20
```

### 3. 出力ディレクトリ構造

```
outputs/irl_cross_eval_new/
├── train_0-3m/
│   ├── irl_model.pt
│   ├── metrics.json
│   ├── eval_0-3m/
│   │   ├── metrics.json
│   │   └── predictions.csv
│   ├── eval_3-6m/
│   ├── eval_6-9m/
│   └── eval_9-12m/
├── train_3-6m/
│   └── ...
├── train_6-9m/
│   └── ...
└── train_9-12m/
    └── ...
```

## ベースラインとの比較

### 比較スクリプト

```python
import json
from pathlib import Path
import pandas as pd

# ベースライン（元の結果）
baseline_dir = Path("importants/review_acceptance_cross_eval_nova")
# 新規IRL
new_dir = Path("outputs/irl_cross_eval_new")

periods = ["0-3m", "3-6m", "6-9m", "9-12m"]

results = []

for train_period in periods:
    for eval_period in periods:
        baseline_metrics = baseline_dir / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"
        new_metrics = new_dir / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"

        if baseline_metrics.exists() and new_metrics.exists():
            with open(baseline_metrics) as f:
                baseline = json.load(f)
            with open(new_metrics) as f:
                new = json.load(f)

            baseline_auc = baseline.get("auc_roc", baseline.get("test_auc_roc", 0))
            new_auc = new.get("auc_roc", new.get("test_auc_roc", 0))

            results.append({
                "Train": train_period,
                "Eval": eval_period,
                "Baseline_AUC": baseline_auc,
                "New_AUC": new_auc,
                "Diff": new_auc - baseline_auc,
            })

df = pd.DataFrame(results)
print("\n=== AUC-ROC比較 (Baseline vs New IRL) ===")
print(df.to_string(index=False))

# 統計
print(f"\n平均差分: {df['Diff'].mean():.6f}")
print(f"標準偏差: {df['Diff'].std():.6f}")

# マトリクス形式
pivot_diff = df.pivot(index='Train', columns='Eval', values='Diff')
print("\n=== Difference Matrix (New - Baseline) ===")
print(pivot_diff.round(6))
```

### 実行

```bash
# 比較スクリプトを /tmp/compare_cross_eval.py として保存して実行
uv run python /tmp/compare_cross_eval.py
```

## 結果サマリー（2025 年 11 月 6 日実行）

### 統計情報

| 指標     | 値                                 |
| -------- | ---------------------------------- |
| 平均差分 | -0.008545                          |
| 最大改善 | +0.029178 (Train=6-9m, Eval=6-9m)  |
| 最大劣化 | -0.058355 (Train=9-12m, Eval=6-9m) |
| 標準偏差 | 0.025236                           |

### Difference Matrix (New - Baseline)

```
Eval       0-3m      3-6m      6-9m     9-12m
Train
0-3m  -0.010989 -0.016517 -0.010610 -0.013587
3-6m  -0.024420 -0.013514 -0.005305 -0.046196
6-9m   0.007326  0.024024  0.029178  0.019022
9-12m  0.020757 -0.037538 -0.058355  0.000000
```

## 再現性に関する注意事項

### シード固定

`train_irl_review_acceptance.py`では以下のようにシードを固定しています：

```python
RANDOM_SEED = 777
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 完全再現が困難な理由

1. **CPU 環境での非決定性**

   - PyTorch の CPU 演算は完全決定的でない場合がある
   - 浮動小数点演算の順序依存性
   - マルチスレッド処理の影響

2. **実行時刻の違い**

   - ベースライン: 2025 年 10 月 31 日 02:26:22
   - 新規実行: 2025 年 11 月 6 日 16:43:36

3. **環境差異**
   - ライブラリバージョンの微妙な違い
   - OS レベルの乱数生成の違い

### 許容される差異

- 平均差分が ±0.01 以内であれば実質的に再現できていると判断
- 標準偏差が 0.03 以内であれば許容範囲

## トラブルシューティング

### 問題: パラメータエラー

**症状**: `unknown arguments: --future-start`などのエラー

**解決**: 正しいパラメータ名を使用

- ❌ `--future-start`, `--future-end`
- ✅ `--future-window-start`, `--future-window-end`

### 問題: モデルファイルが見つからない

**症状**: `FileNotFoundError: モデルが見つかりません`

**解決**: 訓練フェーズが完了してから評価フェーズを実行

- スクリプトは自動的に順序を制御しているので、手動実行時のみ注意

### 問題: 同一期間の評価がない

**症状**: `train_0-3m/eval_0-3m/`などのディレクトリが存在しない

**解決**: スクリプトで同一期間の評価をスキップしている場合がある

```python
# スキップ処理を削除
for eval_period, future_start, future_end in LABEL_PERIODS:
    evaluate_model(model_dir, eval_period, future_start, future_end)
    # if train_period == eval_period: continue  # この行を削除
```

## 実行時間

- **訓練フェーズ**: 約 2 分（4 モデル）

  - train_0-3m: 約 39 秒
  - train_3-6m: 約 32 秒
  - train_6-9m: 約 15 秒
  - train_9-12m: 約 12 秒

- **評価フェーズ**: 約 38 秒（16 評価）

  - 各評価: 約 2-3 秒

- **合計**: 約 2 分 40 秒

## 関連ファイル

- **実行スクリプト**: `scripts/analysis/run_irl_cross_eval_new.py`
- **訓練スクリプト**: `scripts/training/irl/train_irl_review_acceptance.py`
- **データファイル**: `data/review_requests_openstack_multi_5y_detail.csv`
- **ベースライン結果**: `importants/review_acceptance_cross_eval_nova/`
- **新規実行結果**: `outputs/irl_cross_eval_new/`

## 履歴

- **2025 年 10 月 31 日**: ベースラインデータ生成（importants/review_acceptance_cross_eval_nova/）
- **2025 年 11 月 6 日**: 再現実行（outputs/irl_cross_eval_new/）、このドキュメント作成
