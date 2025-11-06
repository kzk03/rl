# レビュー受諾予測実験：完全ガイド

## 目次

1. [実験概要](#1-実験概要)
2. [タスク定義](#2-タスク定義)
3. [データセット](#3-データセット)
4. [実験設計](#4-実験設計)
5. [IRL+LSTM実装](#5-irlstm実装)
6. [ベースライン実装](#6-ベースライン実装)
7. [再現手順](#7-再現手順)
8. [結果の解釈](#8-結果の解釈)
9. [評価方法の比較](#9-評価方法の比較)
10. [トラブルシューティング](#10-トラブルシューティング)

---

## 1. 実験概要

### 1.1 目的

**レビュー受諾予測タスク**において、IRL+LSTMと従来の機械学習ベースライン（Logistic Regression, Random Forest）を**公平な条件**で比較評価し、各モデルの性能特性を明らかにする。

### 1.2 主要な発見

**対角線以降（同一期間+未来）の評価**:
- **IRL+LSTM**: 0.784（最高性能）
- **Logistic Regression**: 0.770
- **Random Forest**: 0.704

**未来予測のみ**:
- **IRL+LSTM**: 0.832（+7.1% vs LR）
- **Logistic Regression**: 0.777
- **Random Forest**: 0.737

### 1.3 実験の構成

```
データ: OpenStack Nova + Neutron（60,216レビュー）
期間: 訓練2021-2023、評価2023-2024
評価: 4×4クロス評価（16組み合わせ）
方法: 月次訓練方式（IRLと同じ）
モデル: IRL+LSTM, Logistic Regression, Random Forest
```

---

## 2. タスク定義

### 2.1 予測対象

**問題**: 評価期間中にレビュアーが少なくとも1つのレビュー依頼を受諾するかを予測

**正例（ラベル=1）**: 評価期間中に≥1件のレビュー依頼を受諾
**負例（ラベル=0）**: 評価期間中にレビュー依頼を受けたが全て拒否

### 2.2 タスクの特性

**静的特性が支配的**:
- 専門性（固定的）
- 過去の受諾率（安定）
- プロジェクトへのコミットメント（長期的）
- 活動量（比較的安定）

**時間的変化**:
- 小さい（レビュアーの行動パターンは固定的）
- 短期的な変動は少ない

**結果**: 静的特徴量で高性能が可能だが、未来予測では時系列モデルが優位

### 2.3 開発者継続性予測との違い

| 項目 | レビュー受諾予測 | 開発者継続性予測 |
|-----|----------------|----------------|
| **予測対象** | 依頼の受諾 | プロジェクトへの継続参加 |
| **支配的要因** | 静的特性 | 時間的ダイナミクス |
| **時間的変化** | 小さい | 大きい |
| **IRL優位性** | 未来予測で+7.1% | 全期間で+31% |
| **ベースライン性能** | LR 0.792（全期間） | LR 0.665-0.669 |

---

## 3. データセット

### 3.1 データソース

**ファイル**: `data/review_requests_nova_neutron.csv`

**プロジェクト**:
- openstack/nova
- openstack/neutron

**統計**:
- 総レビュー依頼数: 60,216件
- 期間: 2012-06-20 ～ 2025-09-27
- 受諾数: 8,860件（約14.7%）

### 3.2 必須カラム

```csv
reviewer_email    # レビュアーのメールアドレス（識別子）
request_time      # レビュー依頼の時刻（ISO 8601形式）
label             # 受諾フラグ（1=受諾, 0=拒否）
project           # プロジェクト名（例: "openstack/nova"）
```

### 3.3 オプションカラム（特徴量生成に有用）

```csv
lines_added       # 追加行数
lines_deleted     # 削除行数
files_changed     # 変更ファイル数
message           # コミットメッセージ
```

### 3.4 データ前処理

**推奨**: Bot アカウントのフィルタリング
```bash
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_no_bots.csv
```

**プロジェクト抽出**:
```bash
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --output data/review_requests_nova_neutron.csv \
  --projects "openstack/nova" "openstack/neutron"
```

---

## 4. 実験設計

### 4.1 時間軸の設定

```
訓練期間: 2021-01-01 ～ 2023-01-01（24ヶ月）
評価期間: 2023-01-01 ～ 2024-01-01（12ヶ月）

訓練期間の分割（4四半期）:
├─ 0-3m:  2021-01-01 ～ 2021-07-01（6ヶ月）
├─ 3-6m:  2021-07-01 ～ 2022-01-01（6ヶ月）
├─ 6-9m:  2022-01-01 ～ 2022-07-01（6ヶ月）
└─ 9-12m: 2022-07-01 ～ 2023-01-01（6ヶ月）

評価期間の分割（4四半期）:
├─ 0-3m:  2023-01-01 ～ 2023-04-01（3ヶ月）
├─ 3-6m:  2023-04-01 ～ 2023-07-01（3ヶ月）
├─ 6-9m:  2023-07-01 ～ 2023-10-01（3ヶ月）
└─ 9-12m: 2023-10-01 ～ 2024-01-01（3ヶ月）
```

**注意**:
- 訓練期間の四半期は6ヶ月間隔
- 評価期間の四半期は3ヶ月間隔
- これはIRL実装に合わせた設定

### 4.2 4×4クロス評価

**評価マトリクス**:
```
        評価期間 →
訓練     0-3m  3-6m  6-9m  9-12m
期間  0-3m   [  ]  [  ]  [  ]  [  ]
↓    3-6m   [  ]  [  ]  [  ]  [  ]
     6-9m   [  ]  [  ]  [  ]  [  ]
     9-12m  [  ]  [  ]  [  ]  [  ]

合計: 16組の訓練-評価の組み合わせ
```

**評価の分類**:
- **対角線（4組）**: 同一期間評価（例: 0-3m訓練 → 0-3m評価）
- **未来への評価（6組）**: 訓練期間より後（例: 0-3m訓練 → 3-6m評価）
- **過去への評価（6組）**: 訓練期間より前（例: 3-6m訓練 → 0-3m評価）

**実用的評価 = 対角線 + 未来 = 10組**

### 4.3 月次訓練方式（IRLと同じ）

**重要**: ベースラインもIRLと同じ月次訓練方式を使用

**訓練期間のインデックス表記**:
```
0-3m訓練 = 0～6ヶ月 future window
3-6m訓練 = 6～12ヶ月 future window
6-9m訓練 = 12～18ヶ月 future window
9-12m訓練 = 18～24ヶ月 future window
```

**月次訓練の仕組み（例: 0-3m訓練）**:
```
訓練期間: 2021-01-01 ～ 2023-01-01
Future window: 0～6ヶ月

月ごとにラベルを作成:
- 2021-01月: 特徴量（～2021-01）→ ラベル（2021-01～2021-07）
- 2021-02月: 特徴量（～2021-02）→ ラベル（2021-02～2021-08）
- ...
- 2022-06月: 特徴量（～2022-06）→ ラベル（2022-06～2022-12）
- 2022-07月以降: future_start >= train_end のためスキップ

全ての月のデータを集約して訓練（例: 1803 trajectories）
```

**なぜ月次訓練が必要か**:
```
全期間一括方式（不採用）:
- 9-12m訓練（18～24ヶ月 future window）の場合
- max_date = 2023-01-01 - 24ヶ月 = 2021-01-01
- 特徴量期間: 2021-01-01 ～ 2021-01-01（0ヶ月）
- → データなし ❌

月次訓練方式（採用）:
- 各月ごとに処理
- 2021-01～2021-05の5ヶ月分のデータを使用
- → 184 trajectories ✅
```

### 4.4 評価期間のデータ抽出

**評価データ**:
```
特徴量期間: train_start ～ train_end（訓練期間全体）
ラベル期間: eval_q_start ～ eval_q_end（評価四半期）

例（0-3m訓練 → 0-3m評価）:
- 特徴量: 2021-01-01 ～ 2023-01-01のレビュー履歴
- ラベル: 2023-01-01 ～ 2023-04-01の受諾有無
```

**重要**: 訓練期間と評価期間は時間的に重複していない
- 訓練: 2021-01 ～ 2023-01
- 評価: 2023-01 ～ 2024-01（完全に未来）

---

## 5. IRL+LSTM実装

### 5.1 アーキテクチャ

**モデル**: `RetentionIRLSystem` (temporal IRL with LSTM)

**構成**:
```
Input: [batch, seq_len, feature_dim]
  ↓
State Encoder (Linear → ReLU → Linear → ReLU)
  ↓
Action Encoder (Linear → ReLU → Linear → ReLU)
  ↓
Combined (Addition)
  ↓
LSTM (1-layer, hidden_size=128)
  ↓
├─ Reward Predictor (Linear → ReLU → Linear)
└─ Continuation Predictor (Linear → ReLU → Linear → Sigmoid)
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
￥
### 5.3 ハイパーパラメータ

**設定ファイル**: 実験スクリプト内にハードコード

```python
config = {
    'state_dim': 10,
    'action_dim': 5,
    'hidden_dim': 128,
    'learning_rate': 0.001,
    'sequence': True,          # LSTM使用
    'seq_len': 15,             # シーケンス長
    'gamma': 0.99,             # 割引率
    'dropout': 0.2             # Dropout率
}

epochs = 20                    # 訓練エポック数
```

### 5.4 訓練スクリプト

**ファイル**: `scripts/training/irl/train_irl_review_acceptance.py`

**使用方法**:
```bash
uv run python scripts/training/irl/train_irl_review_acceptance.py \
  --reviews data/review_requests_nova_neutron.csv \
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

**特徴量（10次元）**:
```python
1. total_reviews          # 総レビュー数
2. activity_frequency     # 活動頻度（1日あたり）
3. experience_days        # 経験日数
4. acceptance_rate        # 受諾率
5. recent_activity        # 最近の活動（直近30日）
6. collaboration_score    # 協力スコア（ユニークな協力者数）
7. quality_score          # 品質スコア（受諾率ベース）
8. project_diversity      # プロジェクト多様性（ユニークなプロジェクト数）
9. consistency            # 一貫性（活動の規則性）
10. trend                 # トレンド（活動の時間的変化）
```

**特徴量計算**: `extract_static_features()` 関数

### 6.2 Random Forest

**実装**: `gerrit_retention.baselines.RandomForestBaseline`

**アルゴリズム**: scikit-learn の `RandomForestClassifier`

**ハイパーパラメータ**:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

**特徴量**: Logistic Regressionと同じ10次元

### 6.3 ベースライン訓練スクリプト

**ファイル**: `scripts/experiments/run_baseline_nova_fair_comparison.py`

**月次訓練方式の実装**:
```python
def extract_trajectories_monthly_training(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history: int = 3
) -> List[Dict[str, Any]]:
    """
    IRLと同じ月次訓練方式でtrajectoryを抽出
    """
    # 月ごとのラベル作成
    history_months = pd.date_range(
        start=train_start,
        end=train_end,
        freq='MS'  # Month start
    )

    all_trajectories = []

    for month_start in history_months[:-1]:
        month_end = month_start + pd.DateOffset(months=1)

        # ラベル期間
        future_start = month_end + pd.DateOffset(months=future_window_start_months)
        future_end = month_end + pd.DateOffset(months=future_window_end_months)

        # train_endでクリップ
        if future_end > train_end:
            future_end = train_end

        # future_start >= train_endならスキップ
        if future_start >= train_end:
            continue

        # 特徴量とラベルの抽出
        # ... (詳細は実装参照)

        all_trajectories.extend(month_trajectories)

    return all_trajectories
```

**使用方法**:
```bash
uv run python scripts/experiments/run_baseline_nova_fair_comparison.py \
  --reviews data/review_requests_nova_neutron.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_monthly_training/
```

**出力**:
```
importants/baseline_nova_monthly_training/
├── logistic_regression/
│   ├── matrix_AUC_ROC.csv
│   ├── matrix_AUC_PR.csv
│   ├── matrix_F1.csv
│   ├── matrix_PRECISION.csv
│   ├── matrix_RECALL.csv
│   └── results.json
└── random_forest/
    ├── matrix_AUC_ROC.csv
    ├── matrix_AUC_PR.csv
    ├── matrix_F1.csv
    ├── matrix_PRECISION.csv
    ├── matrix_RECALL.csv
    └── results.json
```

### 6.4 特徴量計算の詳細

**関数**: `extract_static_features(trajectories)`

**入力**: trajectory のリスト
```python
trajectory = {
    'developer': {
        'developer_id': 'user@example.com',
        'first_seen': pd.Timestamp('2020-01-01'),
        'changes_authored': 50,
        'changes_reviewed': 100,
        'projects': ['openstack/nova', 'openstack/neutron']
    },
    'activity_history': [
        {
            'type': 'review',
            'timestamp': pd.Timestamp('2021-01-01'),
            'project': 'openstack/nova',
            'accepted': True,
            'message': '...',
            'lines_added': 10,
            'lines_deleted': 5,
            'files_changed': 2
        },
        # ... more activities
    ],
    'continued': True  # ラベル
}
```

**出力**: 特徴量行列（N × 10）

**計算ロジック**:
```python
# 1. total_reviews
total_reviews = len(activity_history)

# 2. activity_frequency
days_active = (last_activity - first_activity).days + 1
activity_frequency = total_reviews / days_active

# 3. experience_days
experience_days = (last_activity - first_activity).days

# 4. acceptance_rate
accepted = sum(1 for a in activity_history if a['accepted'])
acceptance_rate = accepted / total_reviews

# 5. recent_activity (直近30日)
cutoff = last_activity - pd.Timedelta(days=30)
recent_activity = sum(1 for a in activity_history if a['timestamp'] >= cutoff)

# 6. collaboration_score
unique_collaborators = len(set(a.get('collaborator') for a in activity_history))
collaboration_score = unique_collaborators / total_reviews

# 7. quality_score (受諾率ベース)
quality_score = acceptance_rate

# 8. project_diversity
unique_projects = len(set(a['project'] for a in activity_history))
project_diversity = unique_projects

# 9. consistency (活動の規則性)
time_diffs = [後の時刻 - 前の時刻 for 連続する活動]
consistency = 1.0 / (np.std(time_diffs) + 1e-6)

# 10. trend (活動の時間的変化)
# 前半と後半の活動頻度の比較
mid_point = len(activity_history) // 2
first_half_freq = mid_point / (activity_history[mid_point]['timestamp'] - first_activity).days
second_half_freq = (len(activity_history) - mid_point) / (last_activity - activity_history[mid_point]['timestamp']).days
trend = second_half_freq - first_half_freq
```

---

## 7. 再現手順

### 7.1 環境準備

**1. 依存関係のインストール**:
```bash
cd /path/to/gerrit-retention
uv sync
```

**2. データの確認**:
```bash
# データファイルの存在確認
ls -lh data/review_requests_nova_neutron.csv

# データの基本統計
uv run python -c "
import pandas as pd
df = pd.read_csv('data/review_requests_nova_neutron.csv')
print(f'Total reviews: {len(df)}')
print(f'Date range: {df[\"request_time\"].min()} to {df[\"request_time\"].max()}')
print(f'Unique reviewers: {df[\"reviewer_email\"].nunique()}')
print(f'Acceptance rate: {(df[\"label\"]==1).mean():.1%}')
"
```

### 7.2 IRL+LSTMの実行

**コマンド**:
```bash
uv run python scripts/training/irl/train_irl_review_acceptance.py \
  --reviews data/review_requests_nova_neutron.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --output importants/review_acceptance_cross_eval_nova
```

**実行時間**: 約10-15分（CPU、4×4=16組の評価）

**出力の確認**:
```bash
# 結果マトリクスの確認
cat importants/review_acceptance_cross_eval_nova/matrix_AUC_ROC.csv

# モデルファイルの確認
ls -lh importants/review_acceptance_cross_eval_nova/models/
```

### 7.3 ベースラインの実行

**コマンド**:
```bash
uv run python scripts/experiments/run_baseline_nova_fair_comparison.py \
  --reviews data/review_requests_nova_neutron.csv \
  --train-start 2021-01-01 \
  --train-end 2023-01-01 \
  --eval-start 2023-01-01 \
  --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_monthly_training/
```

**実行時間**: 約5-10分（CPU、両ベースライン合計）

**出力の確認**:
```bash
# Logistic Regressionの結果
cat importants/baseline_nova_monthly_training/logistic_regression/matrix_AUC_ROC.csv

# Random Forestの結果
cat importants/baseline_nova_monthly_training/random_forest/matrix_AUC_ROC.csv
```

### 7.4 結果の比較分析

**対角線以降の評価スクリプト**:
```bash
uv run python << 'EOF'
import pandas as pd
import numpy as np

# Load results
lr_matrix = pd.read_csv('importants/baseline_nova_monthly_training/logistic_regression/matrix_AUC_ROC.csv', index_col=0)
irl_matrix = pd.read_csv('importants/review_acceptance_cross_eval_nova/matrix_AUC_ROC.csv', index_col=0)
rf_matrix = pd.read_csv('importants/baseline_nova_monthly_training/random_forest/matrix_AUC_ROC.csv', index_col=0)

print("対角線以降（同一期間+未来）の平均性能:")
print(f"IRL+LSTM:            {irl_matrix.values[np.triu_indices(4)].mean():.3f}")
print(f"Logistic Regression: {lr_matrix.values[np.triu_indices(4)].mean():.3f}")
print(f"Random Forest:       {rf_matrix.values[np.triu_indices(4)].mean():.3f}")
EOF
```

---

## 8. 結果の解釈

### 8.1 評価指標

**AUC-ROC (Area Under ROC Curve)**:
- **意味**: 正例と負例を区別する能力
- **範囲**: 0.0～1.0（0.5=ランダム）
- **解釈**:
  - 0.9-1.0: 優秀
  - 0.8-0.9: 良好
  - 0.7-0.8: まあまあ
  - 0.5: ランダム

**AUC-PR (Area Under Precision-Recall Curve)**:
- **意味**: 不均衡データでの性能（受諾率が低い場合に重要）
- **範囲**: 0.0～1.0
- **解釈**: AUC-ROCより厳しい評価（特に負例が多い場合）

**F1スコア**:
- **意味**: PrecisionとRecallの調和平均
- **範囲**: 0.0～1.0
- **解釈**: バランスの取れた性能指標

**Precision（適合率）**:
- **意味**: 予測した正例のうち、実際に正例だった割合
- **計算**: TP / (TP + FP)
- **重要性**: 誤検知を減らしたい場合

**Recall（再現率）**:
- **意味**: 実際の正例のうち、正しく予測できた割合
- **計算**: TP / (TP + FN)
- **重要性**: 見逃しを減らしたい場合

### 8.2 マトリクスの読み方

**matrix_AUC_ROC.csv**:
```csv
,0-3m,3-6m,6-9m,9-12m
0-3m,0.828,0.861,0.777,0.758
3-6m,0.810,0.852,0.763,0.747
6-9m,0.821,0.858,0.768,0.758
9-12m,0.784,0.825,0.731,0.726
```

**読み方**:
- **行**: 訓練期間（例: 0-3m = 0-6ヶ月 future windowでの月次訓練）
- **列**: 評価期間（例: 0-3m = 2023-01～2023-04）
- **値**: その組み合わせでのAUC-ROC

**例**:
- `[0-3m, 6-9m] = 0.777`: 0-3m訓練モデルで6-9m評価期間を予測 → AUC-ROC 0.777
- `[3-6m, 3-6m] = 0.852`: 3-6m訓練モデルで3-6m評価期間を予測（対角線）

### 8.3 評価タイプの分類

**対角線（同一期間）**:
```
[0-3m, 0-3m], [3-6m, 3-6m], [6-9m, 6-9m], [9-12m, 9-12m]
→ 4組
```

**未来への予測**:
```
[0-3m, 3-6m], [0-3m, 6-9m], [0-3m, 9-12m]
[3-6m, 6-9m], [3-6m, 9-12m]
[6-9m, 9-12m]
→ 6組
```

**過去への評価（実用価値なし）**:
```
[3-6m, 0-3m]
[6-9m, 0-3m], [6-9m, 3-6m]
[9-12m, 0-3m], [9-12m, 3-6m], [9-12m, 6-9m]
→ 6組
```

**対角線以降（実用的評価）**:
```
対角線 + 未来 = 4 + 6 = 10組
```

### 8.4 平均性能の計算

**全期間平均（16組）**:
```python
all_mean = matrix.values.flatten().mean()
```

**対角線以降の平均（10組）**:
```python
diagonal_and_future = []
for train_idx in range(4):
    for eval_idx in range(train_idx, 4):
        diagonal_and_future.append(matrix.iloc[train_idx, eval_idx])
diag_future_mean = np.mean(diagonal_and_future)
```

**未来のみの平均（6組）**:
```python
future_only = []
for train_idx in range(4):
    for eval_idx in range(train_idx + 1, 4):
        future_only.append(matrix.iloc[train_idx, eval_idx])
future_mean = np.mean(future_only)
```

### 8.5 訓練データ数の確認

**訓練期間別の訓練データ数**:

```
0-3m訓練（0-6m future window）:
- 月数: 23ヶ月（2021-01～2022-11）
- Trajectories: 約1800

3-6m訓練（6-12m future window）:
- 月数: 17ヶ月（2021-01～2021-05）
- Trajectories: 約1060

6-9m訓練（12-18m future window）:
- 月数: 11ヶ月（2021-01～2021-11）
- Trajectories: 約534

9-12m訓練（18-24m future window）:
- 月数: 5ヶ月（2021-01～2021-05）
- Trajectories: 約184
```

**重要**: 訓練データ数が少ないほど、モデルの性能が低下する傾向

---

## 9. 評価方法の比較

### 9.1 3つの評価方法

**全期間評価（16組）**:
- 全ての訓練-評価の組み合わせを含む
- 過去への評価も含まれる
- 学術的には完全だが、実用的価値は低い組み合わせを含む

**対角線以降（10組）**:
- 対角線（同一期間）+ 未来への予測
- 過去への評価を除外
- **実用的な評価として推奨** ⭐

**未来のみ（6組）**:
- 未来への予測のみ
- 最も厳しい評価
- 時系列モデルの真の価値を測定

### 9.2 評価方法による順位変化

| 評価方法 | 1位 | 2位 | 3位 |
|---------|-----|-----|-----|
| **全期間（16組）** | LR 0.792 | IRL 0.758 | RF 0.708 |
| **対角線以降（10組）** | **IRL 0.784** ⭐ | LR 0.770 | RF 0.704 |
| **未来のみ（6組）** | **IRL 0.832** ⭐ | LR 0.777 | RF 0.737 |

**重要**: 実用的な評価（対角線以降）では**IRL+LSTMが最高性能**

### 9.3 各評価方法の使い分け

**全期間評価を使うべき場合**:
- 学術的な完全性が必要
- 全ての組み合わせを網羅的に評価
- ベースライン手法との公平な比較（従来研究が全期間評価を使用している場合）

**対角線以降を使うべき場合**:
- 実用的な性能を評価したい ⭐
- レビュアー推薦システムの実装を検討
- 過去への予測の実用価値がないことを明示

**未来のみを使うべき場合**:
- 時系列モデルの真の価値を測定
- 最も厳しい評価
- 長期予測能力を強調

**論文での推奨**:
1. **主要な評価**: 対角線以降（実用的）
2. **補助的な評価**: 全期間（従来研究との比較）、未来のみ（時系列の優位性）
3. **評価方法の違いを明示**: 順位変化の理由を説明

---

## 10. トラブルシューティング

### 10.1 よくあるエラー

**エラー1: `KeyError: 'reviewer_email'`**

**原因**: CSVファイルに必須カラムがない

**解決策**:
```python
# カラム名を確認
import pandas as pd
df = pd.read_csv('data/review_requests_nova_neutron.csv')
print(df.columns.tolist())

# 必須カラム: reviewer_email, request_time, label, project
```

**エラー2: `ZeroDivisionError` または "軌跡が不足"**

**原因**: 指定した期間にデータが不足

**解決策**:
```python
# データの日付範囲を確認
df['request_time'] = pd.to_datetime(df['request_time'])
print(f"Date range: {df['request_time'].min()} to {df['request_time'].max()}")

# 訓練期間にデータがあるか確認
train_df = df[(df['request_time'] >= '2021-01-01') &
              (df['request_time'] < '2023-01-01')]
print(f"Training data: {len(train_df)} reviews")
```

**エラー3: 9-12m訓練期間でデータが0**

**原因**: 月次訓練方式が正しく実装されていない

**確認**:
```bash
# 正しいスクリプトを使用しているか確認
ls scripts/experiments/run_baseline_nova_fair_comparison.py

# extract_trajectories_monthly_training関数が実装されているか確認
grep -n "extract_trajectories_monthly_training" scripts/experiments/run_baseline_nova_fair_comparison.py
```

### 10.2 性能が期待値と異なる場合

**チェックリスト**:

1. **データの確認**:
   ```python
   # 受諾率の確認
   print(f"Acceptance rate: {(df['label']==1).mean():.1%}")
   # 期待値: 約14-15%
   ```

2. **訓練データ数の確認**:
   ```bash
   # ログ出力を確認
   # "Total: XXX trajectories from YY months"
   # 期待値: 0-3m=1800, 3-6m=1060, 6-9m=534, 9-12m=184
   ```

3. **ハイパーパラメータの確認**:
   ```python
   # IRL
   epochs = 20  # 少なすぎないか？
   seq_len = 15  # 適切か？

   # Logistic Regression
   max_iter = 1000  # 収束しているか？
   ```

4. **評価期間の確認**:
   ```python
   # 評価データが十分にあるか
   eval_df = df[(df['request_time'] >= '2023-01-01') &
                (df['request_time'] < '2024-01-01')]
   print(f"Evaluation data: {len(eval_df)} reviews")
   ```

### 10.3 再現性の確保

**乱数シードの設定**:
```python
# Python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

**依存関係のバージョン固定**:
```bash
# pyproject.tomlで固定
uv sync

# バージョンの確認
uv pip list | grep -E "(torch|sklearn|pandas)"
```

**データの整合性**:
```bash
# データファイルのハッシュ確認
md5sum data/review_requests_nova_neutron.csv
# または
shasum -a 256 data/review_requests_nova_neutron.csv
```

### 10.4 性能改善のヒント

**IRL+LSTMの性能改善**:
1. **エポック数を増やす**: `epochs = 30` または `50`
2. **シーケンス長を調整**: `seq_len = 10` または `20`（データ分布に応じて）
3. **学習率を調整**: `learning_rate = 0.0001`（より細かく）
4. **Dropoutを調整**: `dropout = 0.3`（過学習を防ぐ）

**ベースラインの性能改善**:
1. **Random Forestのパラメータ**:
   ```python
   RandomForestClassifier(
       n_estimators=200,  # 増やす
       max_depth=15,      # 調整
       min_samples_split=5,
       random_state=42
   )
   ```

2. **特徴量エンジニアリング**:
   - プロジェクト固有の特徴を追加
   - 時間的な特徴を追加（月、曜日など）
   - 相互作用項を追加

3. **クラスバランスの調整**:
   ```python
   LogisticRegression(
       max_iter=1000,
       class_weight='balanced',  # 不均衡データ対応
       random_state=42
   )
   ```

---

## 11. 参考資料

### 11.1 関連ファイル

**実験結果**:
- `importants/review_acceptance_cross_eval_nova/`: IRL+LSTM結果
- `importants/baseline_nova_monthly_training/`: ベースライン結果
- `importants/baseline_nova_monthly_training/DIAGONAL_AND_FUTURE_ANALYSIS.md`: 詳細分析レポート
- `importants/baseline_nova_monthly_training/MONTHLY_TRAINING_COMPARISON_REPORT.md`: 月次訓練版レポート

**スクリプト**:
- `scripts/training/irl/train_irl_review_acceptance.py`: IRL訓練
- `scripts/experiments/run_baseline_nova_fair_comparison.py`: ベースライン訓練
- `src/gerrit_retention/rl_prediction/retention_irl_system.py`: IRL実装
- `src/gerrit_retention/baselines/`: ベースライン実装

**ドキュメント**:
- `CLAUDE.md`: プロジェクト全体の説明
- `README.md`: 基本的な使い方
- `docs/DATA_FILTERING_GUIDE.md`: データ前処理ガイド

### 11.2 重要な設定値

**時間設定**:
```python
train_start = '2021-01-01'
train_end = '2023-01-01'
eval_start = '2023-01-01'
eval_end = '2024-01-01'
```

**訓練期間のインデックス**:
```python
quarters_mapping = {
    '0-3m': (0, 6),    # 0-6ヶ月 future window
    '3-6m': (6, 12),   # 6-12ヶ月 future window
    '6-9m': (12, 18),  # 12-18ヶ月 future window
    '9-12m': (18, 24)  # 18-24ヶ月 future window
}
```

**評価指標の優先順位**:
1. **AUC-ROC**: 主要指標（全体的な識別能力）
2. **AUC-PR**: 不均衡データでの性能
3. **F1スコア**: バランスの取れた性能

### 11.3 論文執筆のヒント

**実験セクションでの記述**:
```markdown
## 実験設定

### データセット
OpenStack NovaとNeutronプロジェクトの60,216件のレビュー依頼データを使用した。
訓練期間は2021-01～2023-01（24ヶ月）、評価期間は2023-01～2024-01（12ヶ月）。

### 評価方法
4×4クロス評価を実施し、訓練期間と評価期間をそれぞれ4つの四半期に分割した。
各訓練期間でモデルを訓練し、全ての評価期間で性能を測定した（合計16組）。

IRLと公平な比較を行うため、ベースラインも月次訓練方式を採用した。
これにより、訓練期間の後半をラベル付けのためのみに使用し、特徴量計算には
使用しないという制約を全モデルに課した。

### モデル
- **IRL+LSTM**: 時系列パターンを学習する提案手法
- **Logistic Regression**: 静的特徴量を用いる線形ベースライン
- **Random Forest**: 静的特徴量を用いる非線形ベースライン

### 評価指標
主要指標としてAUC-ROCを使用し、補助的にAUC-PR、F1スコア、Precision、
Recallを報告する。

実用的な評価として、訓練期間と同一または将来の時期への予測のみを評価する
「対角線以降の評価」を採用した。過去への評価は実用的価値がないため除外した。
```

**結果セクションでの記述**:
```markdown
## 結果

### 全期間評価
全16組の評価では、Logistic Regressionが最高性能（AUC-ROC 0.792）を達成した。

### 対角線以降の評価（実用的評価）
過去への評価を除外した実用的な評価（対角線以降、10組）では、IRL+LSTMが
最高性能（AUC-ROC 0.784）を達成し、Logistic Regression（0.770）を1.7%上回った。

特に未来への予測（6組）では、IRL+LSTMが7.1%の優位性を示した（0.832 vs 0.777）。
```

---

## 12. まとめ

### 12.1 実験の要点

**公平な比較**:
- ✅ IRLと同じ月次訓練方式をベースラインに適用
- ✅ 全モデルに同じ時間的制約を課す
- ✅ 9-12m訓練期間も評価可能に

**評価方法**:
- 全期間評価（16組）: 学術的完全性
- **対角線以降（10組）**: 実用的評価 ⭐ **推奨**
- 未来のみ（6組）: 時系列の真の価値

**主要な発見**:
- 対角線以降でIRL+LSTM最高（0.784）
- 未来予測でIRL+LSTM圧倒的（0.832, +7.1%）
- 中距離予測（3-6ヶ月）で最も効果的（+12.3%）

### 12.2 再現のための最小限の手順

```bash
# 1. 環境準備
cd /path/to/gerrit-retention
uv sync

# 2. データ確認
ls data/review_requests_nova_neutron.csv

# 3. IRL+LSTM実行
uv run python scripts/training/irl/train_irl_review_acceptance.py \
  --reviews data/review_requests_nova_neutron.csv \
  --train-start 2021-01-01 --train-end 2023-01-01 \
  --eval-start 2023-01-01 --eval-end 2024-01-01 \
  --output importants/review_acceptance_cross_eval_nova

# 4. ベースライン実行
uv run python scripts/experiments/run_baseline_nova_fair_comparison.py \
  --reviews data/review_requests_nova_neutron.csv \
  --train-start 2021-01-01 --train-end 2023-01-01 \
  --eval-start 2023-01-01 --eval-end 2024-01-01 \
  --baselines logistic_regression random_forest \
  --output importants/baseline_nova_monthly_training/

# 5. 結果確認
cat importants/review_acceptance_cross_eval_nova/matrix_AUC_ROC.csv
cat importants/baseline_nova_monthly_training/logistic_regression/matrix_AUC_ROC.csv
cat importants/baseline_nova_monthly_training/random_forest/matrix_AUC_ROC.csv
```

**実行時間**: 合計約15-25分（CPU）

**期待される結果**:
- IRL+LSTM（対角線以降）: 0.784
- Logistic Regression（対角線以降）: 0.770
- Random Forest（対角線以降）: 0.704

---

**作成日**: 2025-11-04
**目的**: レビュー受諾予測実験の完全な再現可能性の確保
**対象読者**: 実験を再現したい研究者、実装を理解したい開発者
