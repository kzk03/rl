# 学習期間内完結型 IRL 訓練

## 📋 概要

このスクリプトは、**学習期間内で完結する**逆強化学習（IRL）訓練を実行します。

### 重要な設計

✅ **将来の貢献をラベルとして使用**（状態特徴量には含めない）  
✅ **LSTM で時系列パターンを学習**  
✅ **学習期間内ですべて完結**  
✅ **評価期間とは cutoff で分離**

### 目的

**続いた人と続かなかった人を区別する報酬関数を学習**

---

## 🚀 クイックスタート

### 基本的な実行

```bash
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --eval-start 2020-01-01 \
  --eval-end 2021-01-01 \
  --future-window-start 0 \
  --future-window-end 1 \
  --history-window 6 \
  --epochs 30 \
  --sequence \
  --output outputs/irl_0_1m
```

### 複数の将来窓で実験

```bash
# 実験を一括実行
./scripts/training/irl/run_future_window_experiments.sh

# 結果を比較
uv run python scripts/analysis/compare_future_window_results.py \
  --results outputs/future_window_experiments/*/evaluation_results.json \
  --output outputs/future_window_comparison.png
```

---

## 📊 タイムライン

### 学習期間内で完結

```
学習期間: 2019-01-01 ～ 2020-01-01 (12ヶ月)
|------------------------------------------------|
[train_start]                          [train_end]

サンプリング可能範囲（future_window=0-1m の場合）:
|-----------------------------------|
     ↑                           ↑
  2019-07-01                  2019-12-01
  (最初)                       (最後)
                                  ↓
                                  +-- 将来窓 (1m)
                                  [2019-12-01 ～ 2020-01-01]
                                                   ↑
                                                   学習期間内で完結！

評価期間: 2020-01-01 ～ 2021-01-01
|------------------------------------------------|
     ↑
     cutoff で完全分離
```

---

## 🔧 パラメータ

### 必須パラメータ

| パラメータ      | 説明             | 例                                                   |
| --------------- | ---------------- | ---------------------------------------------------- |
| `--reviews`     | レビューログ CSV | `data/review_requests_openstack_multi_5y_detail.csv` |
| `--train-start` | 学習期間の開始日 | `2019-01-01`                                         |
| `--train-end`   | 学習期間の終了日 | `2020-01-01`                                         |

### オプションパラメータ

| パラメータ              | デフォルト                           | 説明                         |
| ----------------------- | ------------------------------------ | ---------------------------- |
| `--eval-start`          | `train-end`                          | 評価期間の開始日             |
| `--eval-end`            | `eval-start + 12m`                   | 評価期間の終了日             |
| `--history-window`      | `6`                                  | 履歴ウィンドウ（ヶ月）       |
| `--future-window-start` | `0`                                  | 将来窓の開始（ヶ月）         |
| `--future-window-end`   | `1`                                  | 将来窓の終了（ヶ月）         |
| `--sampling-interval`   | `1`                                  | サンプリング間隔（ヶ月）     |
| `--epochs`              | `30`                                 | 訓練エポック数               |
| `--sequence`            | `False`                              | 時系列モード（LSTM）を有効化 |
| `--seq-len`             | `15`                                 | シーケンス長                 |
| `--output`              | `outputs/irl_within_training_period` | 出力ディレクトリ             |

---

## 📝 実験例

### 実験 1: 短期予測（0-1 ヶ月後）

```bash
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --future-window-start 0 \
  --future-window-end 1 \
  --output outputs/irl_future_0_1m
```

**期待される結果:** 短期的なパターンを学習、即座の離脱を予測

### 実験 2: 中期予測（1-3 ヶ月後）

```bash
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --future-window-start 1 \
  --future-window-end 3 \
  --output outputs/irl_future_1_3m
```

**期待される結果:** 中期的なパターンを学習、バランス型

### 実験 3: 長期予測（3-6 ヶ月後）

```bash
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --future-window-start 3 \
  --future-window-end 6 \
  --output outputs/irl_future_3_6m
```

**期待される結果:** 長期的なパターンを学習、持続的な貢献者を識別

---

## 📂 出力

### ディレクトリ構造

```
outputs/irl_future_0_1m/
├── irl_model.pth              # 訓練済みモデル
└── evaluation_results.json    # 評価結果
```

### evaluation_results.json

```json
{
  "train_period": {
    "start": "2019-01-01 00:00:00",
    "end": "2020-01-01 00:00:00"
  },
  "eval_period": {
    "start": "2020-01-01 00:00:00",
    "end": "2021-01-01 00:00:00"
  },
  "windows": {
    "history_months": 6,
    "future_start_months": 0,
    "future_end_months": 1
  },
  "training": {
    "epochs": 30,
    "sequence": true,
    "seq_len": 15,
    "train_samples": 5432
  },
  "metrics": {
    "auc_roc": 0.852,
    "auc_pr": 0.789,
    "f1": 0.654,
    "precision": 0.721,
    "recall": 0.598,
    "test_samples": 1234,
    "positive_samples": 189,
    "positive_rate": 0.153
  }
}
```

---

## 🎓 理論的背景

### LSTM で時系列パターンを学習

```python
# 過去の活動履歴（時系列）
activity_history = [action_t0, action_t1, ..., action_t15]

# LSTMで時系列的な変化を捉える
state_seq = LSTM(activity_history)

# 続いた人と続かなかった人の違いを学習
# - 続いた人: 活動増加傾向、規則的なパターン
# - 続かなかった人: 活動減少傾向、不規則なパターン
```

### 因果関係が正しい

```python
# ✅ 正しい: 過去から将来を予測
state = extract_past_features_only(trajectory)  # 過去のみ
target = trajectory['future_contribution']       # ラベル
loss = MSE(predicted_reward, target)
```

### 学習期間内で完結

```python
# すべてのサンプルについて
max_sampling_date = train_end - future_window_end_months

# 最後のサンプリング時点での将来窓
future_end = max_sampling_date + future_window_end_months
assert future_end == train_end  # ✅ 学習期間内
```

---

## ✅ チェックリスト

実装を確認する際のポイント：

- [ ] 将来の貢献を状態特徴量に含めていない
- [ ] 将来の貢献を学習のターゲット（ラベル）として使用
- [ ] サンプリング時点の範囲が正しく制限されている
- [ ] すべてのデータが学習期間内で完結
- [ ] 訓練期間と評価期間が cutoff で分離
- [ ] LSTM で時系列パターンを処理
- [ ] 続いた人 → 高報酬、続かなかった人 → 低報酬 を学習

---

## 🐛 トラブルシューティング

### データが見つからない

```bash
# データファイルを確認
ls -lh data/review_requests_openstack_multi_5y_detail.csv
```

### サンプル数が少ない

```bash
# サンプリング間隔を短くする
--sampling-interval 1  # 1ヶ月ごと（デフォルト）
```

### メモリ不足

```bash
# シーケンス長を短くする
--seq-len 10  # デフォルト: 15
```

---

## 📚 関連ドキュメント

- **詳細設計**: `docs/IRL最終設計_学習期間内完結版.md`
- **要約版**: `docs/IRL設計変更の要約.md`
- **既存の IRL**: `README_TEMPORAL_IRL.md`

---

**作成日**: 2025-10-21  
**ステータス**: 実装済み
