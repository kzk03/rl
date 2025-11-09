# 統一実験設計: Enhanced IRL vs RF Baseline

## 実験概要

レビュワーのレビュー承諾予測における、Enhanced IRL（時系列特徴量+逆強化学習）と RF Baseline（状態特徴量のみ）の性能比較。

**実施日**: 2025 年 11 月 9 日  
**データ**: OpenStack 5 年間レビューデータ（137,632 行、1,379 レビュワー）

---

## 1. タスク定義

### Train（学習）

- **母集団**: current_month に活動があったレビュワー
- **予測対象**: 3-6 ヶ月後に継続的にレビューを承諾するか
- **特徴量**: current_month 以前の履歴
- **ラベル**: future 期間（3-6m/6-9m/9-12m 後）でレビュー依頼があり、かつ受け入れたか（label=1）

### Eval（評価）

- **母集団**: 予測期間（future_start〜future_end）にレビュー依頼があったレビュワー
- **予測対象**: その依頼を承諾するかどうか
- **特徴量**: eval_start 以前の履歴
- **ラベル**: 予測期間でレビューを受け入れたか（label=1）

**重要**: Train と Eval で母集団選定ロジックが異なるが、これは意図的な設計。Train では「継続的にレビューを承諾する傾向」を学習し、Eval では「レビュー依頼に応じるか」を予測する。

---

## 2. 期間設定

### 観測期間

- **Train 期間**: 2021-01-01 〜 2023-01-01（2 年間）
  - **ラベル専用期間は予測ウィンドウにより変動**:
    - 0-3m 予測: 観測 21 ヶ月、ラベル専用 3 ヶ月
    - 3-6m 予測: 観測 18 ヶ月、ラベル専用 6 ヶ月
    - 6-9m 予測: 観測 15 ヶ月、ラベル専用 9 ヶ月
    - 9-12m 予測: 観測 12 ヶ月、ラベル専用 12 ヶ月
- **Eval 期間**: 2023-01-01 〜 2024-01-01（1 年間）
  - 基準日: 2023-01-01（スナップショット時点）

### 予測期間（Future Windows）

4 つの予測ウィンドウ:

- **0-3m**: 0-3 ヶ月後
- **3-6m**: 3-6 ヶ月後
- **6-9m**: 6-9 ヶ月後
- **9-12m**: 9-12 ヶ月後

### 評価組み合わせ

Train 期間 × Eval 期間の全組み合わせで、時系列的に妥当なもの（eval_start >= train_start）のみ評価。

**有効な組み合わせ**: 10 通り

```
0-3→0-3, 0-3→3-6, 0-3→6-9, 0-3→9-12
3-6→3-6, 3-6→6-9, 3-6→9-12
6-9→6-9, 6-9→9-12
9-12→9-12
```

---

## 3. データ準備

### Train: 月次サンプリング

```python
current_month = train_start_dt  # 2021-01-01
while current_month < label_start:  # 2022-07-01まで
    # この月に活動があったレビュワー
    month_reviewers = df[
        (df['request_time'] >= current_month) &
        (df['request_time'] < current_month + 1ヶ月)
    ]['reviewer_email'].unique()

    for reviewer in month_reviewers:
        # 特徴量: current_month以前の履歴
        history = df[df['request_time'] < current_month]

        # ラベル: future期間で活動したか
        future_start = current_month + train_future_start_months
        future_end = min(current_month + train_future_end_months, train_end_dt)
        label = has_activity_in(future_start, future_end)

    current_month += 1ヶ月
```

**結果**: 約 2,427 サンプル（18 ヶ月 × 平均 135 人/月）

### Eval: シングルスナップショット

```python
# 予測期間にレビュー依頼があったレビュワーを母集団に
future_start = eval_start_dt + eval_future_start_months  # 2023-04-01など
future_end = eval_start_dt + eval_future_end_months      # 2023-07-01など

eval_reviewers = df[
    (df['request_time'] >= future_start) &
    (df['request_time'] < future_end)
]['reviewer_email'].unique()

for reviewer in eval_reviewers:
    # 特徴量: eval_start_dt以前の履歴（2023-01-01まで）
    history = df[df['request_time'] < eval_start_dt]

    # ラベル: 予測期間でレビューを受け入れたか
    label = has_accepted_in(future_start, future_end)
```

**結果**: 予測期間により変動（168〜276 サンプル）

---

## 4. 特徴量設計

### 状態特徴量（State Features）- 10 次元

両モデル共通で使用:

1. **experience_days**: 最初の活動から現在までの日数（正規化: /730 日）
2. **total_changes**: 総変更数（レビュー依頼数で近似、正規化: /500）
3. **total_reviews**: 総レビュー数（正規化: /500）
4. **recent_activity_frequency**: 直近 30 日の活動頻度（回/日）
5. **avg_activity_gap**: 平均活動間隔（正規化: /60 日）
6. **activity_trend**: 活動トレンド（0.0=減少, 0.25=不明, 0.5=安定, 1.0=増加）
7. **collaboration_score**: 協力スコア（固定値 1.0）
8. **code_quality_score**: コード品質スコア（固定値 0.5）
9. **recent_acceptance_rate**: 直近 30 日の受諾率（固定値 0.5）
10. **review_load**: レビュー負荷（直近 30 日 / 全期間平均）

### 時系列特徴量（Temporal Features）- 97 次元

**Enhanced IRL のみ使用**:

- **acceptance_sequence**: 過去 30 日間の日次受諾率（30 次元）
- **load_sequence**: 過去 30 日間の日次レビュー数（30 次元、正規化: /10）
- **response_time_sequence**: 過去 30 日間の日次応答時間（30 次元、正規化: /7 日）
- **weekly_pattern**: 曜日別パターン（7 次元）

---

## 5. モデル設計

### Enhanced IRL

```
入力: 状態(10次元) + 時系列(97次元)
構造:
  - LSTM: 2層、hidden_dim=128、双方向
  - Attention機構
  - Dropout: 0.3
  - 出力: BCELoss
最適化:
  - Optimizer: Adam (lr=0.001)
  - Epochs: 50
  - Batch size: 16
  - Random seeds: [42, 123, 777, 2024, 9999]
```

**総実験数**: 10 組み合わせ × 5 シード = **50 実験**

### RF Baseline

```
入力: 状態(10次元)のみ
構造:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 10
  - min_samples_leaf: 5
  - n_jobs: -1
Random seed: 42
```

**総実験数**: 10 組み合わせ × 1 シード = **10 実験**

---

## 6. 評価指標

### メトリクス

- **AUC (Area Under the ROC Curve)**: 主要評価指標
- 各期間組み合わせごとに AUC を計算
- シード間の平均と標準偏差を報告

### 比較方法

1. **全体平均 AUC**: 全 50 実験（IRL）vs 全 10 実験（RF）の平均
2. **期間組み合わせごと**: 各組み合わせで IRL 平均 vs RF
3. **勝敗カウント**: 10 組み合わせ中、IRL が勝った数
4. **効果量**: (IRL 平均 - RF 平均) / 標準誤差

---

## 7. 実験結果

### 全体成績

| モデル           | 平均 AUC    | 標準偏差 | 最小   | 最大   |
| ---------------- | ----------- | -------- | ------ | ------ |
| **Enhanced IRL** | **0.7473**  | 0.0633   | 0.6606 | 0.8547 |
| **RF Baseline**  | 0.7199      | 0.0398   | 0.6630 | 0.7892 |
| **差分**         | **+0.0274** | -        | -      | -      |

**改善率**: +3.79%  
**効果量**: 1.77 標準誤差

### 期間組み合わせごとの比較

| 組み合わせ | IRL AUC | RF AUC | 差分    | 判定   |
| ---------- | ------- | ------ | ------- | ------ |
| 0-3→0-3    | 0.8478  | 0.7892 | +0.0587 | ✅ IRL |
| 0-3→3-6    | 0.8297  | 0.7477 | +0.0820 | ✅ IRL |
| 0-3→6-9    | 0.7614  | 0.7016 | +0.0598 | ✅ IRL |
| 0-3→9-12   | 0.6880  | 0.6869 | +0.0012 | ✅ IRL |
| 3-6→3-6    | 0.8250  | 0.7518 | +0.0732 | ✅ IRL |
| 3-6→6-9    | 0.7401  | 0.6817 | +0.0584 | ✅ IRL |
| 3-6→9-12   | 0.6990  | 0.6952 | +0.0038 | ✅ IRL |
| 6-9→6-9    | 0.6695  | 0.6630 | +0.0064 | ✅ IRL |
| 6-9→9-12   | 0.7209  | 0.7401 | -0.0192 | ❌ RF  |
| 9-12→9-12  | 0.6911  | 0.7423 | -0.0512 | ❌ RF  |

**勝敗**: IRL 8 勝 - 2 敗 RF（80%勝率）

### シード別統計（Enhanced IRL）

| Seed | 平均 AUC | 標準偏差 |
| ---- | -------- | -------- |
| 777  | 0.7493   | 0.0692   |
| 9999 | 0.7483   | 0.0592   |
| 123  | 0.7482   | 0.0666   |
| 2024 | 0.7455   | 0.0675   |
| 42   | 0.7449   | 0.0668   |

**シード間の安定性**: 標準偏差 < 0.002（非常に安定）

---

## 8. 主要な発見

### ✅ 成功点

1. **時系列特徴量の有効性を実証**

   - Temporal 特徴（97 次元）の追加により+3.79%改善
   - LSTM+Attention による時系列パターン学習が機能

2. **短期予測で特に効果的**

   - 0-3m 予測: +5.87%改善
   - 0-3→3-6m: +8.20%改善（最大改善幅）
   - 3-6→3-6m: +7.32%改善

3. **母集団の違いによる AUC 変化を確認**

   - Eval 期間が変わると予測対象が変わり、AUC も変化
   - 前回の「全て同じ AUC」問題は解決済み

4. **高い再現性**
   - 5 シード間の標準偏差 < 0.002
   - ランダム性の影響が小さい

### ⚠️ 課題点

1. **長期予測の精度不足**

   - 6-9→9-12m: RF 勝利（-1.92%）
   - 9-12→9-12m: RF 勝利（-5.12%）
   - 予測期間が遠いほど性能劣化

2. **予測期間による性能差が大きい**
   - 最高（0-3→0-3）: 0.8478
   - 最低（6-9→6-9）: 0.6695
   - 差: 0.1783（大きなばらつき）

---

## 9. 結論

### 主要な成果

1. **Enhanced IRL の有効性を実証**（平均+3.79%改善、80%勝率）
2. **時系列特徴量の重要性を確認**（LSTM+Attention が効果的）
3. **統一された実験設計による公正な比較**を実現

### 今後の改善方向

1. **長期予測の強化**

   - より長い時系列（60 日 →90 日）の利用
   - Transformer 等の別アーキテクチャの検討

2. **特徴量エンジニアリング**

   - プロジェクト間の関係性
   - レビュワー間のネットワーク特徴

3. **モデルアンサンブル**
   - IRL + RF のアンサンブル
   - 期間ごとの最適モデル選択

---

## 10. 再現手順

### 実験実行

```bash
cd experiments/enhanced_irl

# Enhanced IRL（約3-4時間）
nohup uv run python scripts/run_multi_seed_eval.py > run_multi_seed_v3.log 2>&1 &

# RF Baseline（約10-15分）
nohup uv run python scripts/run_rf_baseline_multi_seed.py > run_rf_baseline.log 2>&1 &
```

### 結果確認

```bash
# 比較分析
uv run python /tmp/compare_irl_vs_rf.py
```

---

**最終更新**: 2025 年 11 月 9 日
