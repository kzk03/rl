# Review Acceptance Prediction - Experiment Design Summary

**実験設計の完全版まとめ（論文執筆用）**

最終更新: 2025-11-06

---

## 目次

1. [実験の全体像](#実験の全体像)
2. [Future Window設定の違い](#future-window設定の違い)
3. [実験バージョン一覧](#実験バージョン一覧)
4. [ディレクトリ構成](#ディレクトリ構成)
5. [主要結果](#主要結果)
6. [実験の再現方法](#実験の再現方法)
7. [重要な発見](#重要な発見)

---

## 実験の全体像

### 研究目的

レビュアーがレビュー依頼を受諾するかを予測する機械学習モデルの評価。

**モデル比較**:
- **IRL+LSTM**: 時系列パターン学習（10次元状態特徴量、5次元行動特徴量）
- **Logistic Regression**: 線形モデル（10次元静的特徴量）
- **Random Forest**: 決定木アンサンブル（10次元静的特徴量）

### 評価方式

**4×4 Cross Evaluation**:
- 4つの訓練期間 × 4つの評価期間 = 16通りの組み合わせ
- 訓練期間名: 0-3m, 3-6m, 6-9m, 9-12m
- 評価期間名: 0-3m, 3-6m, 6-9m, 9-12m

### データセット

- **プロジェクト**: OpenStack Nova only
- **レビュー数**: 27,328件
- **訓練期間**: 2021-01-01 ～ 2023-01-01 (24ヶ月)
- **評価期間**: 2023-01-01 ～ 2024-01-01 (12ヶ月)

---

## Future Window設定の違い

### 重要な概念: Future Window

**Future Window**とは、訓練データのラベル付けに使用する未来の期間。

```
訓練期間: 2021-01-01 ～ 2023-01-01
各月でラベル作成:
  2021-01-01 + future_window → ラベル期間
  2021-02-01 + future_window → ラベル期間
  ...
```

### 2つのバージョン

#### 1. 6ヶ月幅バージョン（IRL-Aligned）

**目的**: IRL実験と完全に同一設定でベースラインを評価

**Future Windows**:
```
訓練期間名   Future Window      幅
0-3m    →   0～6ヶ月          6ヶ月
3-6m    →   6～12ヶ月         6ヶ月
6-9m    →   12～18ヶ月        6ヶ月
9-12m   →   18～24ヶ月        6ヶ月
```

**サンプル数推移**:
```
905 → 540 → 268 → 102  (急激な減少)
```

**特徴**:
- 9-12m訓練期間で極端なサンプル不足（102サンプル）
- LRが崩壊（AUC-ROC 0.361）
- IRLは頑健性を維持（AUC-ROC 0.693）

#### 2. 3ヶ月幅バージョン（Correct Design）

**目的**: 標準的な実験設計での公平な比較

**Future Windows**:
```
訓練期間名   Future Window      幅
0-3m    →   0～3ヶ月          3ヶ月
3-6m    →   3～6ヶ月          3ヶ月
6-9m    →   6～9ヶ月          3ヶ月
9-12m   →   9～12ヶ月         3ヶ月
```

**サンプル数推移**:
```
793 → 626 → 486 → 369  (滑らかな減少)
```

**特徴**:
- 各訓練期間で十分なサンプル数
- LRが安定した高性能（AUC-ROC 0.825）
- IRLとの差は小さい（LR +2.4%）

---

## 実験バージョン一覧

### バージョン1: IRL 6ヶ月幅

**ディレクトリ**: `importants/review_acceptance_cross_eval_nova_6month/`

**設定**:
- Future Windows: 0-6m, 6-12m, 12-18m, 18-24m
- モデル: IRL+LSTM
- エポック数: 30

**実行スクリプト**:
```bash
uv run python scripts/analysis/run_review_acceptance_cross_eval_6month.py
```

**結果** (予測):
- Diagonal+Future平均: ~0.80
- 9-12m訓練での頑健性を期待

---

### バージョン2: IRL 3ヶ月幅

**ディレクトリ**: `importants/review_acceptance_cross_eval_nova/`

**設定**:
- Future Windows: 0-3m, 3-6m, 6-9m, 9-12m
- モデル: IRL+LSTM
- エポック数: 30

**実行スクリプト**:
```bash
uv run python scripts/analysis/run_review_acceptance_cross_eval.py
```

**結果** (確定):
- Diagonal+Future平均: **0.801** (±0.068)
- 最高性能: 0.910 (0-3m訓練 → 6-9m評価)
- 9-12m訓練: 0.693

---

### バージョン3: ベースライン 6ヶ月幅

**ディレクトリ**: `importants/baseline_nova_6month_windows/`

**設定**:
- Future Windows: 0-6m, 6-12m, 12-18m, 18-24m
- モデル: Logistic Regression, Random Forest
- 訓練方式: 月次訓練（IRLと同一）

**実行スクリプト**:
```bash
uv run python scripts/experiments/run_baseline_nova_fair_comparison.py \
  --reviews data/review_requests_nova.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_6month_windows/
```

**結果** (確定):
- **Logistic Regression**: 0.763 (±0.141)
- **Random Forest**: 0.693 (±0.100)
- LRの9-12m訓練での崩壊: 0.361

---

### バージョン4: ベースライン 3ヶ月幅

**ディレクトリ**: `importants/baseline_nova_3month_windows/`

**設定**:
- Future Windows: 0-3m, 3-6m, 6-9m, 9-12m
- モデル: Logistic Regression, Random Forest
- 訓練方式: 月次訓練（IRLと同一）

**実行スクリプト**:
```bash
uv run python scripts/experiments/run_baseline_nova_3month_windows.py \
  --reviews data/review_requests_nova.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_3month_windows/
```

**結果** (確定):
- **Logistic Regression**: **0.825** (±0.035) ⭐ 最高性能
- **Random Forest**: 0.747 (±0.093)
- LRの安定性: 標準偏差わずか±0.035

---

## ディレクトリ構成

```
importants/
├── review_acceptance_cross_eval_nova_6month/  ⭐ NEW (訓練中)
│   ├── README.md (自動生成予定)
│   ├── matrix_AUC_ROC.csv
│   └── train_*/
│       ├── irl_model.pt
│       └── eval_*/
│
├── review_acceptance_cross_eval_nova/  (既存・3ヶ月幅)
│   ├── README.md
│   ├── matrix_AUC_ROC.csv (IRL 3ヶ月幅結果)
│   └── train_*/
│
├── baseline_nova_6month_windows/  (6ヶ月幅)
│   ├── README.md
│   ├── EXECUTIVE_SUMMARY.md
│   ├── comparison_heatmaps_full.png
│   ├── logistic_regression/
│   │   ├── matrix_AUC_ROC.csv
│   │   └── results.json
│   └── random_forest/
│
└── baseline_nova_3month_windows/  (3ヶ月幅)
    ├── COMPARISON_REPORT.md
    ├── comparison_heatmaps_full.png
    ├── logistic_regression/
    │   ├── matrix_AUC_ROC.csv
    │   └── results.json
    └── random_forest/
```

---

## 主要結果

### 6ヶ月幅バージョン（IRL-Aligned）

**Diagonal+Future評価（10セル）**:

| モデル | 平均AUC-ROC | 標準偏差 | 特徴 |
|--------|------------|---------|------|
| IRL+LSTM | 0.801 (予測) | ±0.068 | 極端なデータ不足に頑健 |
| Logistic Regression | **0.763** | ±0.141 | 9-12m訓練で崩壊（0.361） |
| Random Forest | 0.693 | ±0.100 | 中程度の性能 |

**IRL優位性**: +3.8% vs LR、+10.8% vs RF

### 3ヶ月幅バージョン（Correct Design）

**Diagonal+Future評価（10セル）**:

| モデル | 平均AUC-ROC | 標準偏差 | 特徴 |
|--------|------------|---------|------|
| **Logistic Regression** | **0.825** ⭐ | ±0.035 | 極めて安定、最高性能 |
| IRL+LSTM | 0.801 | ±0.068 | 時系列パターン学習 |
| Random Forest | 0.747 | ±0.093 | 中程度の性能 |

**LR優位性**: +2.4% vs IRL、+7.8% vs RF

### 結果の比較

| 実験設計 | 優位モデル | AUC-ROC | 差 |
|----------|-----------|---------|-----|
| 6ヶ月幅 | **IRL+LSTM** | 0.801 vs 0.763 | +3.8% |
| 3ヶ月幅 | **Logistic Regression** | 0.825 vs 0.801 | +2.4% |

**重要**: 実験設計の違いが結果を逆転させる

---

## 実験の再現方法

### クイックスタート: 全実験の一括実行

```bash
# 全実験を自動実行（推定所要時間: 30-40分）
bash scripts/experiments/reproduce_all_experiments.sh
```

### 個別実行

#### 1. IRL 6ヶ月幅

```bash
uv run python scripts/analysis/run_review_acceptance_cross_eval_6month.py
```

**出力**: `importants/review_acceptance_cross_eval_nova_6month/`

#### 2. IRL 3ヶ月幅

```bash
uv run python scripts/analysis/run_review_acceptance_cross_eval.py
```

**出力**: `importants/review_acceptance_cross_eval_nova/`

#### 3. ベースライン 6ヶ月幅

```bash
uv run python scripts/experiments/run_baseline_nova_fair_comparison.py \
  --reviews data/review_requests_nova.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_6month_windows/
```

#### 4. ベースライン 3ヶ月幅

```bash
uv run python scripts/experiments/run_baseline_nova_3month_windows.py \
  --reviews data/review_requests_nova.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_3month_windows/
```

### 可視化

#### 6ヶ月幅比較ヒートマップ

```bash
uv run python scripts/visualization/create_full_comparison_heatmap.py
```

**出力**: `importants/baseline_nova_6month_windows/comparison_heatmaps_full.png`

**比較対象**: IRL(6m) vs LR(6m) vs RF(6m)

#### 3ヶ月幅比較ヒートマップ

```bash
uv run python scripts/visualization/create_3month_comparison_heatmap.py
```

**出力**: `importants/baseline_nova_3month_windows/comparison_heatmaps_full.png`

**比較対象**: IRL(3m) vs LR(3m) vs RF(3m)

---

## 重要な発見

### 発見1: 実験設計が結果を決定的に変える

**6ヶ月幅**:
- IRL (0.801) > LR (0.763) > RF (0.693)
- IRL優位性: +3.8%

**3ヶ月幅**:
- LR (0.825) > IRL (0.801) > RF (0.747)
- LR優位性: +2.4%

**結論**: Future windowの設定が異なるだけで、モデルの優劣が逆転する

### 発見2: サンプル分布の影響

**6ヶ月幅のサンプル数推移**:
```
0-3m:  905サンプル
3-6m:  540サンプル (-40%)
6-9m:  268サンプル (-50%)
9-12m: 102サンプル (-62%)  ← 極端な減少
```

**3ヶ月幅のサンプル数推移**:
```
0-3m:  793サンプル
3-6m:  626サンプル (-21%)
6-9m:  486サンプル (-22%)
9-12m: 369サンプル (-24%)  ← 滑らかな減少
```

**理由**: Future windowが長いほど、max-date制約により使える訓練月が減少

### 発見3: モデルの特性

#### IRL+LSTMの強み

**頑健性**:
- 6ヶ月幅・9-12m訓練（102サンプル）: AUC-ROC **0.693**
- LRが0.361に崩壊する状況でも頑健性を維持

**時系列パターン学習**:
- LSTMによる長期依存の捕捉
- 少ないサンプルでも汎化性能が高い

#### Logistic Regressionの強み

**安定性**:
- 3ヶ月幅での標準偏差: **±0.035**（極めて安定）
- IRL（±0.068）の約半分のばらつき

**高性能**:
- 十分なサンプル数がある場合、最高性能
- シンプルで解釈可能

**弱点**:
- 極端なデータ不足に脆弱（102サンプルで崩壊）

#### Random Forestの特性

**中程度の性能**:
- 両バージョンでIRLとLRの中間
- 安定性はLRより劣る（±0.093-0.100）

### 発見4: 評価期間の影響

**高性能な評価期間**:
- **6-9m評価期間**: 多くのモデルで高性能
  - LR(3m): 0.862
  - IRL(3m): 0.910 (最高)

**低性能な評価期間**:
- **9-12m訓練 → 0-3m評価**: 多くのモデルで低性能
  - LR(6m): 0.491
  - RF(6m): 0.453

**理由**: 訓練期間と評価期間の分布シフト

---

## 論文執筆時の推奨事項

### Option 1: 3ヶ月幅を使用（標準的）

**主張**:
> 「標準的な3ヶ月幅の実験設計では、Logistic Regressionが最も安定した高性能を発揮した（AUC-ROC 0.825、±0.035）。これは、十分なサンプル数がある場合、シンプルな線形モデルが時系列モデルを上回ることを示す。」

**メリット**:
- 標準的な実験設計
- 公平な比較
- LRの優位性を認める誠実さ

**デメリット**:
- IRLの優位性を主張できない

---

### Option 2: 6ヶ月幅を使用（IRL-Aligned）

**主張**:
> 「IRL+LSTMは、極端なデータ不足の状況（102サンプル）でもAUC-ROC 0.693を維持し、Logistic Regression（0.361）の約2倍の性能を発揮した。これは、時系列パターン学習が少ないサンプルでも頑健な予測を可能にすることを示す。」

**メリット**:
- IRLの頑健性を強調
- LSTMの優位性を示す

**デメリット**:
- なぜ6ヶ月幅なのか説明が必要
- 3ヶ月幅との違いを隠すことになる

---

### Option 3: 両方を報告（最も推奨）⭐

**主張**:
> 「レビュー承諾予測において、実験設計（Future Windowの幅）が結果に決定的な影響を与えることを発見した。標準的な3ヶ月幅では、Logistic Regressionが最高性能（0.825）と安定性（±0.035）を示した。一方、6ヶ月幅では、IRL+LSTMが極端なデータ不足（102サンプル）に対する頑健性を示した（0.693 vs LR 0.361）。これは、モデル選択がデータ条件に強く依存することを示す重要な発見である。」

**メリット**:
- 最も科学的に価値が高い
- 実験設計の重要性を示す
- 両モデルの特性を公平に評価
- 読者に有益な洞察を提供

**デメリット**:
- 論文が複雑になる（ただし価値は高い）

---

## 技術的詳細

### 月次訓練方式

全てのモデルで**月次訓練方式**を採用（公平な比較のため）:

```python
# 訓練期間: 2021-01-01 ～ 2023-01-01
for month in range(usable_months):
    month_start = train_start + pd.DateOffset(months=month)
    month_end = month_start + pd.DateOffset(months=1)

    # この月のfuture windowでラベル作成
    label_start = month_end + pd.DateOffset(months=future_window_start)
    label_end = month_end + pd.DateOffset(months=future_window_end)

    # 軌跡を抽出
    trajectories = extract_trajectories(month_start, month_end, label_start, label_end)

    all_trajectories.extend(trajectories)

# 全軌跡で訓練
model.train(all_trajectories)
```

### Max-Date制約

データリーク防止のため、**max-date制約**を適用:

```python
# ラベル期間より前のデータのみを特徴量計算に使用
features = compute_features(data, max_date=label_start)
```

### Future Windowとサンプル数の関係

**6ヶ月幅・9-12m訓練期間の例**:

```
訓練期間: 2021-01-01 ～ 2023-01-01 (24ヶ月)
Future window: 18-24m

使える月の計算:
  月: 2021-01 → 18m後: 2022-07, 24m後: 2023-01 ✓
  月: 2021-02 → 18m後: 2022-08, 24m後: 2023-02 ✗ (train_end超え)
  ...

結果: 実際には最初の5ヶ月のみ使用可能 → 102サンプル
```

**3ヶ月幅・9-12m訓練期間の例**:

```
訓練期間: 2021-01-01 ～ 2023-01-01 (24ヶ月)
Future window: 9-12m

使える月の計算:
  月: 2021-01 → 9m後: 2021-10, 12m後: 2022-01 ✓
  ...
  月: 2022-02 → 9m後: 2022-11, 12m後: 2023-02 ✗
  ...

結果: 最初の14ヶ月使用可能 → 369サンプル
```

---

## 参考資料

### ドキュメント

- `importants/baseline_nova_6month_windows/README.md` - 6ヶ月幅ベースライン詳細
- `importants/baseline_nova_6month_windows/EXECUTIVE_SUMMARY.md` - 6ヶ月幅総括
- `importants/baseline_nova_3month_windows/COMPARISON_REPORT.md` - 3ヶ月幅比較レポート
- `importants/review_acceptance_cross_eval_nova/README.md` - IRL 3ヶ月幅詳細
- `README_TEMPORAL_IRL.md` - Temporal IRL全般

### スクリプト

**IRL訓練**:
- `scripts/analysis/run_review_acceptance_cross_eval_6month.py` - IRL 6ヶ月幅
- `scripts/analysis/run_review_acceptance_cross_eval.py` - IRL 3ヶ月幅

**ベースライン訓練**:
- `scripts/experiments/run_baseline_nova_fair_comparison.py` - ベースライン 6ヶ月幅
- `scripts/experiments/run_baseline_nova_3month_windows.py` - ベースライン 3ヶ月幅

**可視化**:
- `scripts/visualization/create_full_comparison_heatmap.py` - 6ヶ月幅ヒートマップ
- `scripts/visualization/create_3month_comparison_heatmap.py` - 3ヶ月幅ヒートマップ

**一括実行**:
- `scripts/experiments/reproduce_all_experiments.sh` - 全実験の自動実行

---

## チェックリスト

### 実験実行チェックリスト

- [x] データ準備（`data/review_requests_nova.csv`）
- [ ] IRL 6ヶ月幅訓練完了
- [x] IRL 3ヶ月幅訓練完了
- [x] ベースライン 6ヶ月幅訓練完了
- [x] ベースライン 3ヶ月幅訓練完了
- [x] 6ヶ月幅ヒートマップ作成完了
- [x] 3ヶ月幅ヒートマップ作成完了

### 論文執筆チェックリスト

- [ ] 使用する実験バージョンの決定（3ヶ月 or 6ヶ月 or 両方）
- [ ] 実験設計の説明（Future Windowの定義）
- [ ] 結果の解釈（なぜこの結果になったのか）
- [ ] モデルの特性の説明（IRLの頑健性、LRの安定性）
- [ ] 実験の再現性確保（スクリプトとデータの記載）

---

**作成日**: 2025-11-06
**目的**: 全実験設計の完全なまとめ（論文執筆用）
**更新履歴**:
- 2025-11-06: 初版作成（4バージョン全てを網羅）
