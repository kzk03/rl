# レビュー承諾予測実験ガイド（Nova Only）

**完全再現可能な実験手順書**

本ドキュメントは、OpenStack Novaプロジェクトのレビュー承諾予測実験の完全な再現手順とベースライン比較結果を提供します。

---

## 目次

1. [実験概要](#1-実験概要)
2. [タスク定義](#2-タスク定義)
3. [データセット](#3-データセット)
4. [実験設計](#4-実験設計)
5. [IRL+LSTM実装](#5-irllstm実装)
6. [ベースライン実装](#6-ベースライン実装)
7. [再現手順](#7-再現手順)
8. [結果解釈](#8-結果解釈)
9. [評価方法の比較](#9-評価方法の比較)
10. [トラブルシューティング](#10-トラブルシューティング)
11. [参考資料](#11-参考資料)
12. [まとめ](#12-まとめ)

---

## 1. 実験概要

### 1.1 目的

レビュー依頼を受けた開発者が、その依頼を承諾するかどうかを予測するモデルを構築し、IRL+LSTMとベースライン（LR, RF）を公平に比較する。

### 1.2 主要な結果

**対角線+未来評価（実用的評価、10組合せ）**:
- 🥇 **IRL+LSTM: 0.801** (AUC-ROC)
- 🥈 Logistic Regression: 0.763
- 🥉 Random Forest: 0.693

**IRLの優位性**: +3.8% (vs LR)

### 1.3 実験の構成

```
データ: OpenStack Nova（27,328レビュー）
期間: 訓練2021-2023、評価2023-2024
評価: 4×4クロス評価（16組み合わせ）
方法: 月次訓練方式（IRLと同じ）
モデル: IRL+LSTM, Logistic Regression, Random Forest
```

---

## 2. タスク定義

### 2.1 予測対象

**Question**: 「レビュー依頼を受けた開発者が、評価期間内に少なくとも1件のレビューを承諾するか？」

**入力**:
- 開発者の過去の活動履歴（訓練期間内）
- 開発者の状態特徴量（経験、活動頻度、受諾率など）

**出力**:
- 承諾確率（0-1の連続値）
- 二値分類（承諾 or 拒否）

### 2.2 ラベル定義

**正例（label=1）**: 評価期間内に少なくとも1件のレビュー依頼を承諾
**負例（label=0）**: 評価期間内にレビュー依頼を受けたが、全て拒否

**除外**: 評価期間内および拡張期間（12ヶ月）までレビュー依頼を受けていない開発者

### 2.3 重み付けラベル

**通常の負例（重み=1.0）**: 評価期間内に依頼あり、承諾なし
**拡張負例（重み=0.1）**: 評価期間内に依頼なし、拡張期間（12ヶ月）に依頼あり

---

## 3. データセット

### 3.1 データソース

**ファイル**: `data/review_requests_nova.csv`

**プロジェクト**:
- openstack/nova のみ

**統計**:
- 総レビュー依頼数: 27,328件
- 期間: 2012-06-20 ～ 2025-09-27
- 受諾数: 8,860件（約32.4%）

### 3.2 必須カラム

```python
reviewer_email      # レビュアーのメールアドレス
request_time        # レビュー依頼時刻（ISO 8601）
label               # 承諾=1, 拒否=0
project             # プロジェクト名（"openstack/nova"）
```

### 3.3 オプションカラム（特徴量計算用）

```python
change_files_count  # 変更ファイル数（強度計算用）
change_insertions   # 追加行数（規模計算用）
change_deletions    # 削除行数（規模計算用）
first_response_time # 初回応答時刻（応答速度計算用）
```

### 3.4 データ抽出

```bash
# OpenStack全体データからNova onlyを抽出
uv run python -c "
import pandas as pd
df = pd.read_csv('data/review_requests_openstack_multi_5y_detail.csv')
nova_df = df[df['project'] == 'openstack/nova']
nova_df.to_csv('data/review_requests_nova.csv', index=False)
print(f'Nova only: {len(nova_df)} reviews')
"
```

---

## 4. 実験設計

### 4.1 4×4クロス評価

**訓練期間**: 2021-01-01 ～ 2023-01-01（24ヶ月）

| 訓練期間名 | 期間 | Future Window |
|----------|------|--------------|
| 0-3m | 0～6ヶ月 | 0～6ヶ月 |
| 3-6m | 6～12ヶ月 | 6～12ヶ月 |
| 6-9m | 12～18ヶ月 | 12～18ヶ月 |
| 9-12m | 18～24ヶ月 | 18～24ヶ月 |

**評価期間**: 2023-01-01 ～ 2024-01-01（12ヶ月）

| 評価期間名 | 期間 |
|----------|------|
| 0-3m | 2023-01-01 ～ 2023-04-01 |
| 3-6m | 2023-04-01 ～ 2023-07-01 |
| 6-9m | 2023-07-01 ～ 2023-10-01 |
| 9-12m | 2023-10-01 ～ 2024-01-01 |

**評価数**: 4（訓練）× 4（評価）= 16通り

### 4.2 月次訓練方式（重要）

**従来の方法（max-date方式）**:
- 訓練期間の最後の日付をmax-dateとして使用
- その日付以前のデータのみで特徴量計算
- 9-12m訓練期間でデータ不足（0ヶ月の特徴量期間）

**月次訓練方式（IRLと同じ）**:
- 訓練期間内の各月ごとにラベルを作成
- 各月の終了時点から将来窓を見てラベル付け
- 全月のデータを集約して訓練

**例（9-12m訓練期間）**:
```
訓練期間: 2021-01-01 ～ 2023-01-01
Future window: 18～24ヶ月

月次処理:
2021-01-01 → ラベル期間: 2022-07-01 ～ 2023-01-01
2021-02-01 → ラベル期間: 2022-08-01 ～ 2023-01-01（クリップ）
...
2021-05-01 → ラベル期間: 2022-11-01 ～ 2023-01-01（クリップ）

結果: 5ヶ月分のラベルを集約 → 102サンプル
```

---

## 5. IRL+LSTM実装

### 5.1 アーキテクチャ

```python
class TemporalIRLNetwork(nn.Module):
    def __init__(self):
        # State Encoder (10次元 → 128次元)
        self.state_encoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Action Encoder (4次元 → 128次元)
        self.action_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # LSTM (128次元 → 128次元)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

        # Reward Predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Continuation Predictor
        self.continuation_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
```

### 5.2 特徴量

**状態特徴量（10次元）**:
```python
1. experience_days            # 経験日数（2年でキャップ、0-1正規化）
2. total_changes              # 総変更数（500件でキャップ、0-1正規化）
3. total_reviews              # 総レビュー数（500件でキャップ、0-1正規化）
4. recent_activity_frequency  # 最近の活動頻度（直近30日、0-1）
5. avg_activity_gap           # 平均活動間隔（60日でキャップ、0-1正規化）
6. activity_trend             # 活動トレンド（increasing=1.0, stable=0.5, decreasing=0.0）
7. collaboration_score        # 協力スコア（0-1）
8. code_quality_score         # コード品質スコア（0-1）
9. recent_acceptance_rate     # 直近30日の受諾率（0-1）
10. review_load               # レビュー負荷（直近30日/平均、0-1正規化）
```

**行動特徴量（4次元）**:
```python
1. intensity          # 強度（変更ファイル数ベース、0-1）
2. collaboration      # 協力度（0-1）
3. response_speed     # レスポンス速度（素早いほど大きい、0-1）
4. review_size        # レビュー規模（変更行数ベース、0-1）
```

### 5.3 ハイパーパラメータ

**設定ファイル**: 実験スクリプト内にハードコード

```python
config = {
    'state_dim': 10,
    'action_dim': 4,           # 行動特徴量は4次元
    'hidden_dim': 128,
    'learning_rate': 0.0001,   # 0.001 → 0.0001（局所最適回避）
    'sequence': True,          # LSTM使用
    'seq_len': 0,              # 月次訓練では可変長
    'dropout': 0.2             # Dropout率
}

epochs = 20                    # 訓練エポック数
```

### 5.4 訓練スクリプト

**ファイル**: `scripts/training/irl/train_irl_review_acceptance.py`

**コマンド**:
```bash
uv run python scripts/training/irl/train_irl_review_acceptance.py \
  --reviews data/review_requests_nova.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --output importants/review_acceptance_cross_eval_nova
```

**出力**:
```
importants/review_acceptance_cross_eval_nova/
├── models/
│   ├── irl_h0m_t6m_seq.pth       # 0-3m訓練モデル
│   ├── irl_h6m_t12m_seq.pth      # 3-6m訓練モデル
│   ├── irl_h12m_t18m_seq.pth     # 6-9m訓練モデル
│   └── irl_h18m_t24m_seq.pth     # 9-12m訓練モデル
├── matrix_AUC_ROC.csv
├── matrix_AUC_PR.csv
├── matrix_F1.csv
├── matrix_PRECISION.csv
├── matrix_RECALL.csv
└── README.md
```

---

## 6. ベースライン実装

### 6.1 Logistic Regression

**実装**: `gerrit_retention.baselines.LogisticRegressionBaseline`

**アルゴリズム**: scikit-learn の `LogisticRegression`

**ハイパーパラメータ**:
```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

**特徴量（10次元、静的）**:
```python
1. total_reviews          # 総レビュー数
2. activity_frequency     # 活動頻度（1日あたり）
3. experience_days        # 経験日数
4. acceptance_rate        # 受諾率（全期間）
5. avg_response_time      # 平均応答時間（日）
6. review_load            # レビュー負荷（直近30日/平均）
7. avg_activity_gap       # 平均活動間隔（日）
8. collaboration_score    # 協力スコア
9. recent_acceptance_rate # 直近30日の受諾率
10. activity_trend_score  # 活動トレンドスコア（0-1）
```

### 6.2 Random Forest

**実装**: `gerrit_retention.baselines.RandomForestBaseline`

**アルゴリズム**: scikit-learn の `RandomForestClassifier`

**ハイパーパラメータ**:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

**特徴量**: Logistic Regressionと同じ10次元

### 6.3 ベースライン訓練スクリプト

**ファイル**: `scripts/experiments/run_baseline_nova_fair_comparison.py`

**コマンド**:
```bash
uv run python scripts/experiments/run_baseline_nova_fair_comparison.py \
  --reviews data/review_requests_nova.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_only/
```

**出力**:
```
importants/baseline_nova_only/
├── logistic_regression/
│   ├── matrix_AUC_ROC.csv
│   ├── matrix_AUC_PR.csv
│   ├── matrix_F1.csv
│   └── results.json
└── random_forest/
    ├── matrix_AUC_ROC.csv
    ├── matrix_AUC_PR.csv
    ├── matrix_F1.csv
    └── results.json
```

---

## 7. 再現手順

### 7.1 環境セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/your-org/gerrit-retention.git
cd gerrit-retention

# 2. 依存関係をインストール
uv sync

# 3. データファイルを確認
ls -lh data/review_requests_nova.csv
```

### 7.2 データ準備（必要に応じて）

```bash
# OpenStack全体データからNova onlyを抽出
uv run python -c "
import pandas as pd
df = pd.read_csv('data/review_requests_openstack_multi_5y_detail.csv')
nova_df = df[df['project'] == 'openstack/nova']
nova_df.to_csv('data/review_requests_nova.csv', index=False)
print(f'Nova only: {len(nova_df)} reviews')
"
```

### 7.3 IRL+LSTM訓練

```bash
# 4×4クロス評価でIRLモデルを訓練
uv run python scripts/training/irl/train_irl_review_acceptance.py \
  --reviews data/review_requests_nova.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --output importants/review_acceptance_cross_eval_nova
```

**実行時間**: 約10-15分（CPU、4×4=16モデル）

### 7.4 ベースライン訓練

```bash
# Logistic RegressionとRandom Forestを訓練
uv run python scripts/experiments/run_baseline_nova_fair_comparison.py \
  --reviews data/review_requests_nova.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_only/
```

**実行時間**: 約5-10分（CPU、両ベースライン合計）

### 7.5 結果確認

```bash
# 5. 結果確認
cat importants/review_acceptance_cross_eval_nova/matrix_AUC_ROC.csv
cat importants/baseline_nova_only/logistic_regression/matrix_AUC_ROC.csv
cat importants/baseline_nova_only/random_forest/matrix_AUC_ROC.csv

# 6. 詳細分析レポート確認
cat importants/baseline_nova_only/NOVA_ONLY_ANALYSIS.md
```

---

## 8. 結果解釈

### 8.1 メトリクス

**AUC-ROC** (Area Under ROC Curve):
- 範囲: 0-1（1に近いほど良い）
- 解釈:
  - 0.9-1.0: 極めて優秀
  - 0.8-0.9: 優秀
  - 0.7-0.8: 良好
  - 0.5: ランダム

**AUC-PR** (Area Under Precision-Recall Curve):
- 不均衡データに適したメトリクス
- ベースライン: 正例率（約32.4%）

**F1 Score**:
- Precision と Recall の調和平均
- 範囲: 0-1

### 8.2 マトリクスの読み方

```
         0-3m    3-6m    6-9m    9-12m
0-3m  │ 0.717   0.823   0.910   0.734
3-6m  │ 0.724   0.820   0.894   0.802
6-9m  │ 0.673   0.790   0.785   0.832
9-12m │ 0.565   0.715   0.655   0.693
```

- **行**: 訓練期間（どのデータで訓練したか）
- **列**: 評価期間（どのデータで評価したか）
- **対角線**: 同一期間での評価（最も重要）
- **右上**: 未来への予測（実用的）
- **左下**: 過去への予測（参考）

### 8.3 評価タイプ

**対角線（4組）**: 同一期間での評価
- (0-3m, 0-3m), (3-6m, 3-6m), (6-9m, 6-9m), (9-12m, 9-12m)

**未来（6組）**: 訓練期間より後の期間を評価
- (0-3m, 3-6m), (0-3m, 6-9m), (0-3m, 9-12m)
- (3-6m, 6-9m), (3-6m, 9-12m)
- (6-9m, 9-12m)

**過去（6組）**: 訓練期間より前の期間を評価（実用性なし）

---

## 9. 評価方法の比較

### 9.1 3つの評価方法

| 評価方法 | 組合せ数 | 実用性 | 推奨度 |
|---------|---------|--------|--------|
| **全体（16組）** | 16 | 低（過去含む） | ❌ 非推奨 |
| **対角線+未来（10組）** | 10 | **高** | ✅ **推奨** |
| **未来のみ（6組）** | 6 | 中 | ⚠️ 条件付き |

### 9.2 結果の違い

| 評価方法 | IRL+LSTM | LR | RF | IRL優位性 |
|---------|----------|----|----|----------|
| 全体（16組） | 0.758 | 0.698 | 0.660 | **+6.0%** |
| **対角線+未来（10組）** | **0.801** | 0.763 | 0.693 | **+3.8%** |
| 未来のみ（6組） | 0.832 | 0.809 | 0.727 | +2.3% |

### 9.3 推奨評価方法

**対角線+未来評価（10組）を推奨**

**理由**:
1. **実用的**: 過去への予測は不要
2. **即時予測を含む**: 対角線（同一期間）も評価
3. **バランスが良い**: 近未来と遠未来の両方を含む

---

## 10. トラブルシューティング

### 10.1 データ不足エラー

**症状**:
```
ValueError: Not enough samples for training period 9-12m
```

**原因**: 訓練期間が長すぎてデータ不足

**解決策**:
- 訓練期間を短縮（9-12m → 6-9m）
- より長い訓練データ期間を使用
- min_history_requestsを減らす（デフォルト3）

### 10.2 サンプル数の確認

```bash
# データ分布を確認
uv run python -c "
import pandas as pd
df = pd.read_csv('data/review_requests_nova.csv')
df['request_time'] = pd.to_datetime(df['request_time'])

# 訓練期間のデータ数
train_df = df[(df['request_time'] >= '2021-01-01') &
              (df['request_time'] < '2023-01-01')]
print(f'Train: {len(train_df)} requests')

# 評価期間のデータ数
eval_df = df[(df['request_time'] >= '2023-01-01') &
             (df['request_time'] < '2024-01-01')]
print(f'Eval: {len(eval_df)} requests')
"
```

### 10.3 メモリ不足

**症状**: `MemoryError` or `Killed`

**解決策**:
- seq_lenを減らす（デフォルト15 → 10）
- バッチサイズを減らす（コード内で調整）
- より小さいデータセットでテスト

### 10.4 モデル読み込みエラー

**症状**:
```
RuntimeError: Error(s) in loading state_dict
```

**原因**: モデル構造の不一致

**解決策**:
```python
# モデル設定を確認
import torch
checkpoint = torch.load('model.pth')
print(checkpoint['config'])

# 正しい設定でモデルを初期化
config = checkpoint['config']
model = RetentionIRLSystem(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 11. 参考資料

### 11.1 関連ドキュメント

- **`NOVA_ONLY_ANALYSIS.md`**: Nova only詳細分析レポート
- **`../review_acceptance_cross_eval_nova/README.md`**: IRL実験の詳細
- **`../review_acceptance_cross_eval_nova/エグゼクティブサマリー.md`**: 経営層向け要約
- **`../../README_TEMPORAL_IRL.md`**: Temporal IRL全般のガイド

### 11.2 コードファイル

**IRL実装**:
- `src/gerrit_retention/rl_prediction/retention_irl_system.py`
- `scripts/training/irl/train_irl_review_acceptance.py`

**ベースライン実装**:
- `src/gerrit_retention/baselines/logistic_regression_baseline.py`
- `src/gerrit_retention/baselines/random_forest_baseline.py`
- `scripts/experiments/run_baseline_nova_fair_comparison.py`

### 11.3 論文・文献

- **IRL**: Ng & Russell (2000) "Algorithms for inverse reinforcement learning"
- **LSTM**: Hochreiter & Schmidhuber (1997) "Long short-term memory"
- **レビュー予測**: 関連研究は`docs/references.md`参照

---

## 12. まとめ

### 12.1 重要なポイント

1. **データの正確性**: Nova only（27,328件）を使用
2. **公平な比較**: 月次訓練方式でIRLとベースラインを同一条件で評価
3. **実用的評価**: 対角線+未来評価（10組）を推奨
4. **IRLの優位性**: +3.8%（LRより）、+10.8%（RFより）

### 12.2 次のステップ

**短期**:
- [ ] ヒートマップ可視化（3モデル比較）
- [ ] 統計的有意性検定
- [ ] 他プロジェクト（Cinder, Glance）で検証

**中期**:
- [ ] Transformer導入実験
- [ ] 論文ドラフト執筆
- [ ] GitHub Actions統合

**長期**:
- [ ] トップ会議投稿（ICSE, FSE, ASE）
- [ ] プロダクト化
- [ ] オープンソース化

---

**作成日**: 2025-01-05
**データ**: OpenStack Nova（27,328レビュー）
**評価**: 4×4クロス評価（16組合せ）
**モデル**: IRL+LSTM, Logistic Regression, Random Forest
**結果**: IRL+LSTM 0.801 > LR 0.763 > RF 0.693（対角線+未来評価）
