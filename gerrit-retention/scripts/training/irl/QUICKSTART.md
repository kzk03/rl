# IRL 実験 クイックスタートガイド

## 🚀 すぐに実験を開始

実際のデータを使って Phase 1-2 の実験を実行するためのガイドです。

---

## 📁 利用可能なデータ

### 推奨データセット

**`review_requests_openstack_multi_5y_detail.csv` (79MB)**

- OpenStack 5 年分の詳細データ
- 最も情報が豊富
- 全実験に対応

### その他のデータセット

| ファイル名                              | サイズ | 説明              | 用途             |
| --------------------------------------- | ------ | ----------------- | ---------------- |
| `review_requests_openstack_no_bots.csv` | 42MB   | ボット除外版      | クリーンな実験   |
| `review_requests_nova_neutron.csv`      | 34MB   | nova/neutron のみ | プロジェクト別   |
| `review_requests_sample_1000.csv`       | 578KB  | サンプル 1000 件  | デバッグ・テスト |

---

## 🎯 実験 1: 動作確認（5 分）

### 目的

データとスクリプトが正常に動作するか確認

### 実行

```bash
# サンプルデータで高速テスト
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_sample_1000.csv \
  --train-start 2019-01-01 \
  --train-end 2019-07-01 \
  --eval-start 2019-07-01 \
  --eval-end 2019-12-01 \
  --history-window 3 \
  --future-window-start 0 \
  --future-window-end 1 \
  --epochs 5 \
  --sequence \
  --seq-len 10 \
  --output outputs/quickstart_test
```

### 確認

```bash
# 結果を確認
cat outputs/quickstart_test/evaluation_results.json
```

---

## 📊 実験 2: スライディングウィンドウ評価（3 時間）

### 目的

履歴窓 × 将来窓の最適な組み合わせを発見（RQ1）

### 実行

```bash
# 本番データで実行
uv run python scripts/training/irl/train_irl_sliding_window_evaluation.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --eval-start 2020-01-01 \
  --eval-end 2021-01-01 \
  --history-windows 3 6 9 12 \
  --future-windows 0-1 1-3 3-6 6-12 \
  --epochs 30 \
  --sequence \
  --output outputs/sliding_window_openstack_5y
```

### 確認

```bash
# ヒートマップを確認
open outputs/sliding_window_openstack_5y/sliding_window_heatmaps.png

# AUC-ROC 行列を確認
cat outputs/sliding_window_openstack_5y/auc_roc_matrix.csv
```

---

## ⏱️ 実験 3: 時間的汎化性能評価（1 時間）

### 目的

時間経過による性能劣化を分析、再訓練タイミングを推奨（RQ3）

### 実行

```bash
# 本番データで実行
uv run python scripts/training/irl/evaluate_temporal_generalization.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --eval-interval-months 6 \
  --num-eval-periods 4 \
  --history-window 6 \
  --future-window-start 0 \
  --future-window-end 1 \
  --epochs 30 \
  --sequence \
  --output outputs/temporal_gen_openstack_5y
```

### 確認

```bash
# 時系列プロットを確認
open outputs/temporal_gen_openstack_5y/temporal_generalization_plot.png

# 劣化メトリクスを確認
cat outputs/temporal_gen_openstack_5y/temporal_generalization_metrics.csv
```

---

## 🔬 実験 4: プロジェクト別比較

### nova vs neutron の比較

#### nova プロジェクト

```bash
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --eval-start 2020-01-01 \
  --eval-end 2021-01-01 \
  --history-window 6 \
  --future-window-start 0 \
  --future-window-end 1 \
  --epochs 30 \
  --sequence \
  --output outputs/irl_nova \
  # TODO: プロジェクトフィルタを追加
```

---

## 📅 実験スケジュール例

### Day 1: 動作確認とセットアップ

```bash
# 朝（30分）
./scripts/training/irl/test_basic_run.sh

# データ確認
head -100 data/review_requests_openstack_multi_5y_detail.csv
```

---

### Day 2: スライディングウィンドウ評価

```bash
# 実行開始（3時間）
./scripts/training/irl/run_sliding_window_evaluation.sh

# 結果分析
open outputs/sliding_window_evaluation/sliding_window_heatmaps.png
```

---

### Day 3: 時間的汎化評価

```bash
# 実行開始（1時間）
./scripts/training/irl/run_temporal_generalization.sh

# 結果分析
open outputs/temporal_generalization/temporal_generalization_plot.png
```

---

### Day 4: 論文執筆

- 最適な組み合わせを特定（RQ1）
- 時間的汎化性能を分析（RQ3）
- 統計的検定を実施
- Table/Figure を作成

---

## 💡 データサイズ別の推奨設定

### 小規模データ（< 1MB）

```bash
--epochs 10
--history-windows 3 6
--future-windows 0-1 1-3
--sampling-interval 2
```

**所要時間:** 〜30 分

---

### 中規模データ（1-10MB）

```bash
--epochs 20
--history-windows 3 6 9
--future-windows 0-1 1-3 3-6
--sampling-interval 1
```

**所要時間:** 〜1.5 時間

---

### 大規模データ（> 10MB）- 推奨

```bash
--epochs 30
--history-windows 3 6 9 12
--future-windows 0-1 1-3 3-6 6-12
--sampling-interval 1
```

**所要時間:** 〜3 時間

---

## 🐛 トラブルシューティング

### データが読み込めない

```bash
# データの存在確認
ls -lh data/review_requests_openstack_multi_5y_detail.csv

# ヘッダーを確認
head -1 data/review_requests_openstack_multi_5y_detail.csv
```

---

### メモリ不足

```bash
# サンプルデータを使用
--reviews data/review_requests_sample_1000.csv

# または、シーケンス長を短縮
--seq-len 10  # デフォルト: 15
```

---

### 実行が遅い

```bash
# エポック数を減らす
--epochs 10  # デフォルト: 30

# 組み合わせを減らす
--history-windows 6 12  # 2つのみ
--future-windows 0-1 1-3  # 2つのみ
```

---

## 📊 結果の確認コマンド

### ヒートマップを開く

```bash
# スライディング評価
open outputs/sliding_window_evaluation/sliding_window_heatmaps.png

# 時間的汎化
open outputs/temporal_generalization/temporal_generalization_plot.png
```

---

### CSV を確認

```bash
# AUC-ROC 行列
cat outputs/sliding_window_evaluation/auc_roc_matrix.csv

# 時間的汎化メトリクス
cat outputs/temporal_generalization/temporal_generalization_metrics.csv
```

---

### JSON を整形して表示

```bash
# 評価結果
cat outputs/quickstart_test/evaluation_results.json | python -m json.tool

# 全結果
cat outputs/sliding_window_evaluation/all_results.json | python -m json.tool
```

---

## 🎯 推奨実験フロー（本番）

### Step 1: データ確認

```bash
# データサイズを確認
ls -lh data/review_requests_openstack_multi_5y_detail.csv

# データ期間を確認
head -1000 data/review_requests_openstack_multi_5y_detail.csv | \
  cut -d',' -f5 | sort | uniq
```

---

### Step 2: 小規模テスト

```bash
# サンプルデータで動作確認
uv run python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_sample_1000.csv \
  --train-start 2019-01-01 \
  --train-end 2019-07-01 \
  --epochs 5 \
  --output outputs/test
```

---

### Step 3: 本番実行

```bash
# スライディング評価
./scripts/training/irl/run_sliding_window_evaluation.sh

# 時間的汎化評価
./scripts/training/irl/run_temporal_generalization.sh
```

---

### Step 4: 結果分析

```bash
# すべてのヒートマップを開く
open outputs/*/**.png

# CSVを確認
cat outputs/*/*.csv
```

---

## 📚 関連ドキュメント

- **基本**: `README_WITHIN_TRAINING_PERIOD.md`
- **実験ガイド**: `README_EXPERIMENTS.md`
- **Phase 1**: `docs/スライディングウィンドウ評価ガイド.md`
- **Phase 2**: `docs/時間的汎化性能評価ガイド.md`

---

## ✅ チェックリスト

実験開始前の確認:

- [ ] データファイルが存在する
- [ ] Python 環境が整っている（`uv` が使える）
- [ ] 十分なディスク容量がある（最低 1GB）
- [ ] 実行時間を確保している（3-4 時間）

---

**作成日:** 2025-10-21  
**推奨データ:** `review_requests_openstack_multi_5y_detail.csv`
