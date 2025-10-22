# IRL 実験ガイド

## 📋 実験の全体像

学習期間内完結型 IRL でできる実験をまとめています。

---

## 🎯 実験の種類

### 1. 単一実験

**目的:** 特定の設定で訓練・評価

**スクリプト:** `train_irl_within_training_period.py`

**使用例:**

```bash
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --history-window 6 \
  --future-window-start 0 \
  --future-window-end 1 \
  --epochs 30 \
  --sequence \
  --output outputs/irl_single
```

**用途:**

- 特定の設定を試す
- デバッグ
- プロトタイプ開発

---

### 2. 将来窓の比較実験

**目的:** 異なる将来窓を比較（履歴窓は固定）

**スクリプト:** `run_future_window_experiments.sh`

**実験数:** 4（0-1m, 1-3m, 3-6m, 6-12m）

**使用例:**

```bash
./scripts/training/irl/run_future_window_experiments.sh

# 結果比較
uv run python scripts/analysis/compare_future_window_results.py \
  --results outputs/future_window_experiments/*/evaluation_results.json \
  --output outputs/future_window_comparison.png
```

**用途:**

- どの将来窓が最適か調査
- 短期 vs 中期 vs 長期予測の比較

---

### 3. スライディングウィンドウ評価 ⭐

**目的:** 履歴窓 × 将来窓の**全組み合わせ**を評価

**スクリプト:** `train_irl_sliding_window_evaluation.py`

**実験数:** 16（4 履歴窓 × 4 将来窓）

**使用例:**

```bash
./scripts/training/irl/run_sliding_window_evaluation.sh
```

**出力:**

- AUC-ROC 行列（CSV + ヒートマップ）
- AUC-PR 行列
- F1 行列
- 全結果 JSON

**用途:**

- 最適な組み合わせを発見（RQ1）
- 論文の主要な実験
- 体系的な性能比較

**詳細:** `docs/スライディングウィンドウ評価ガイド.md`

---

### 4. 時間的汎化性能評価 ⭐

**目的:** 時間経過による性能劣化を分析

**スクリプト:** `evaluate_temporal_generalization.py`

**実験数:** 4（6 ヶ月間隔の評価期間）

**使用例:**

```bash
./scripts/training/irl/run_temporal_generalization.sh
```

**出力:**

- 時系列プロット（AUC-ROC, AUC-PR, F1, 劣化率）
- 性能劣化メトリクス
- 再訓練推奨事項

**用途:**

- 時間的汎化性能の評価（RQ3）
- 再訓練戦略の提案
- モデルの長期的な性能を把握

**詳細:** `docs/時間的汎化性能評価ガイド.md`

---

## 📊 実験の比較

| 実験タイプ         | 実験数 | 所要時間 | 用途                   |
| ------------------ | ------ | -------- | ---------------------- |
| 単一実験           | 1      | 〜10 分  | プロトタイプ・デバッグ |
| 将来窓比較         | 4      | 〜40 分  | 将来窓の選択           |
| スライディング評価 | 16     | 〜3 時間 | 論文の主実験（RQ1）    |
| 時間的汎化評価     | 4      | 〜1 時間 | 性能劣化分析（RQ3）    |

_所要時間は 30 エポック、LSTM 有効の場合の目安_

---

## 🔬 研究課題（RQ）と実験の対応

### RQ1: 最適な履歴窓と将来窓の組み合わせは？

**実験:** スライディングウィンドウ評価

```bash
./scripts/training/irl/run_sliding_window_evaluation.sh
```

**分析:**

- ヒートマップで全体像を把握
- 最高 AUC-ROC の組み合わせを特定
- 統計的検定（ANOVA）で有意差を確認

---

### RQ2: 時系列モデル（LSTM）の効果は？

**実験:** LSTM 有無での比較

```bash
# LSTMあり
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --sequence --seq-len 15 \
  --output outputs/with_lstm

# LSTMなし
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --output outputs/no_lstm
```

**分析:**

- AUC-ROC の差を計算
- 時系列パターンの重要性を評価

---

### RQ3: モデルの時間的汎化性能は？

**実験:** 複数の評価期間でテスト（今後実装）

```bash
# 評価期間を変えて複数回実行
for eval_start in 2020-01-01 2020-07-01 2021-01-01; do
  uv run python scripts/training/irl/train_irl_within_training_period.py \
    --train-start 2019-01-01 \
    --train-end 2020-01-01 \
    --eval-start ${eval_start} \
    --output outputs/temporal_generalization_${eval_start}
done
```

**分析:**

- 時間経過による性能劣化を分析
- 再訓練のタイミングを推奨

---

### RQ4: サブグループごとの予測精度の違いは？

**実験:** 層別分析（今後実装）

**分析対象:**

- プロジェクト別（nova, neutron, etc.）
- 経験レベル別（新人, 中堅, ベテラン）
- 活動レベル別（高活動, 低活動）

---

## 🚀 推奨実験フロー

### Step 1: 動作確認

```bash
# 小規模データで高速実行
./scripts/training/irl/test_basic_run.sh
```

**所要時間:** 〜5 分

---

### Step 2: 将来窓の探索

```bash
# 固定履歴窓で将来窓を比較
./scripts/training/irl/run_future_window_experiments.sh
```

**所要時間:** 〜40 分  
**出力:** 将来窓ごとの性能比較グラフ

---

### Step 3: 全組み合わせ評価（論文の主実験）

```bash
# スライディングウィンドウ評価
./scripts/training/irl/run_sliding_window_evaluation.sh
```

**所要時間:** 〜3 時間  
**出力:** 16 組み合わせのヒートマップ

---

### Step 4: 時間的汎化評価

```bash
# 性能劣化を分析
./scripts/training/irl/run_temporal_generalization.sh
```

**所要時間:** 〜1 時間  
**出力:** 時系列プロット、劣化メトリクス

---

### Step 5: 結果分析

```bash
# スライディング評価のヒートマップを確認
open outputs/sliding_window_evaluation/sliding_window_heatmaps.png

# 時間的汎化のプロットを確認
open outputs/temporal_generalization/temporal_generalization_plot.png

# CSV行列を確認
cat outputs/sliding_window_evaluation/auc_roc_matrix.csv
cat outputs/temporal_generalization/temporal_generalization_metrics.csv
```

---

### Step 6: 統計分析・論文執筆

- 最良の組み合わせを特定（RQ1）
- 時間的汎化性能を分析（RQ3）
- 統計的検定を実施
- 再訓練戦略を提案
- Table/Figure を作成

---

## 📁 出力ディレクトリ構造

```
outputs/
├── test_basic_run/                   # 動作確認
│   ├── irl_model.pth
│   └── evaluation_results.json
│
├── future_window_experiments/        # 将来窓比較
│   ├── future_0_1m/
│   ├── future_1_3m/
│   ├── future_3_6m/
│   └── future_6_12m/
│
├── future_window_comparison.png      # 将来窓比較グラフ
│
├── sliding_window_evaluation/        # スライディング評価 ⭐
│   ├── models/                       # 16個のモデル
│   ├── all_results.json              # 全結果
│   ├── auc_roc_matrix.csv            # AUC-ROC行列
│   ├── auc_pr_matrix.csv
│   ├── f1_matrix.csv
│   └── sliding_window_heatmaps.png   # ヒートマップ ⭐
│
└── temporal_generalization/          # 時間的汎化 ⭐
    ├── irl_model.pth                 # 訓練済みモデル
    ├── temporal_generalization_results.json
    ├── temporal_generalization_metrics.csv
    └── temporal_generalization_plot.png  # 時系列プロット ⭐
```

---

## 🎓 論文での活用

### 実験セクションの構成例

**4. Experiments**

**4.1 実験設定**

- データセット: OpenStack 5 年分
- 訓練期間: 2019-01-01 ～ 2020-01-01
- 評価期間: 2020-01-01 ～ 2021-01-01
- 履歴窓: 3, 6, 9, 12 ヶ月
- 将来窓: 0-1, 1-3, 3-6, 6-12 ヶ月
- モデル: IRL with LSTM (seq_len=15)

**4.2 RQ1: 最適な組み合わせ**

- スライディングウィンドウ評価の結果
- Table 1: AUC-ROC 行列
- Figure 1: ヒートマップ
- 発見: 12m 履歴 × 1-3m 将来窓が最良（AUC=0.855）

**4.3 RQ2: LSTM の効果**

- LSTM 有無での比較
- Table 2: 性能差
- 発見: LSTM により+5.2%の改善

**4.4 RQ3: 時間的汎化**

- 4 評価期間（+0-6m, +6-12m, +12-18m, +18-24m）での結果
- Figure 2: 性能の時系列推移
- Table 3: 評価期間ごとの性能と劣化率
- 発見: 6 ヶ月後から緩やかに劣化、12 ヶ月後に加速
- 推奨: 6 ヶ月ごとの再訓練

---

## 📚 ドキュメント

### 設計書

- `docs/IRL最終設計_学習期間内完結版.md` - 詳細設計
- `docs/IRL設計変更の要約.md` - 要約版
- `docs/将来的な比較実験の設計.md` - 実験設計

### 使い方ガイド

- `README_WITHIN_TRAINING_PERIOD.md` - 基本的な使い方
- `README_EXPERIMENTS.md` - **このファイル**（実験ガイド）
- `docs/スライディングウィンドウ評価ガイド.md` - Phase 1: スライディング評価
- `docs/時間的汎化性能評価ガイド.md` - Phase 2: 時間的汎化評価

### 既存の IRL

- `README_TEMPORAL_IRL.md` - 既存の時系列 IRL 実装

---

## 🔧 カスタマイズ

### 実験範囲を広げる

```bash
# より細かい粒度
--history-windows 1 2 3 4 5 6 7 8 9 10 11 12
--future-windows 0-1 1-2 2-3 3-4 4-5 5-6

# 総実験数: 12 × 6 = 72
```

### 実験範囲を狭める

```bash
# 短期集中
--history-windows 3 6
--future-windows 0-1 1-3

# 総実験数: 2 × 2 = 4
```

### 訓練を高速化

```bash
--epochs 10  # デフォルト: 30
--seq-len 10  # デフォルト: 15
```

---

## ⚡ パフォーマンス Tips

### 並列実行

```python
# GNU Parallelを使用（今後実装可能）
parallel --jobs 4 \
  uv run python scripts/training/irl/train_irl_within_training_period.py \
  --history-window {} \
  ::: 3 6 9 12
```

### GPU の活用

```bash
# CUDAが利用可能な場合は自動的にGPUを使用
# PyTorchがGPUを検出: 約3-5倍高速化
```

---

## 🎉 まとめ

### 実装済み

✅ 単一実験（`train_irl_within_training_period.py`）  
✅ 将来窓比較（`run_future_window_experiments.sh`）  
✅ スライディング評価（`train_irl_sliding_window_evaluation.py`）  
✅ 時間的汎化評価（`evaluate_temporal_generalization.py`）  
✅ 結果比較・可視化（`compare_future_window_results.py`）

### 今後実装予定

🚀 ローリングウィンドウ評価（Phase 3）  
🚀 層別分析（Phase 4）

### 次のステップ

1. 動作確認: `./scripts/training/irl/test_basic_run.sh`
2. 将来窓比較: `./scripts/training/irl/run_future_window_experiments.sh`
3. **スライディング評価**: `./scripts/training/irl/run_sliding_window_evaluation.sh` ⭐
4. **時間的汎化評価**: `./scripts/training/irl/run_temporal_generalization.sh` ⭐
5. 結果分析・論文執筆

---

**作成日:** 2025-10-21  
**ステータス:** Phase 1-2 完了
