# Gerrit Retention IRL プロジェクト完全ガイド

## 📌 プロジェクト概要

このプロジェクトは、**OpenStack Gerrit のレビュー履歴データを活用し、逆強化学習(IRL)とLSTMを組み合わせた時系列学習により、OSSプロジェクトのレビュアーが長期的に貢献を続けるかを予測する研究**です。

### 主要な成果

- **データ規模**: 13年分（2012-2025年）、137,632件のレビューデータ
- **予測精度**: AUC-ROC 0.868、AUC-PR 0.983、F1スコア 0.978
- **技術革新**: LSTM×IRLによる時系列学習で、従来の静的特徴量ベースを超える精度を実現

---

## 🔄 データ収集からIRL予測までの完全な流れ

### フェーズ1: データ収集

#### 1.1 Gerrit APIからの生データ抽出

```bash
# Gerritレビューデータの取得
uv run python data_processing/gerrit_extraction/extract_reviews.py \
  --project openstack/nova \
  --output data/raw/review_requests_openstack.json
```

**取得データの内容**:
- レビューリクエスト情報（作成日時、変更内容、プロジェクト名）
- レビュアー情報（メールアドレス、アクション履歴）
- レビュー結果（承認/却下、コメント数、応答時間）
- 変更統計（追加/削除行数、変更ファイル数）

**データ形式例**:
```json
{
  "change_id": "I1234abcd",
  "project": "openstack/nova",
  "created": "2020-01-15T10:30:00Z",
  "reviewer_email": "reviewer@example.com",
  "status": "MERGED",
  "lines_added": 120,
  "lines_deleted": 45,
  "files_changed": 8,
  "response_time": "2020-01-16T14:20:00Z"
}
```

---

### フェーズ2: データ前処理

#### 2.1 ボットアカウントの除外

**重要**: ボットアカウントは全体の44%を占め、除外しないとノイズとなります。

```bash
# ボットアカウントの除外
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_no_bots.csv
```

**除外基準**:
- 自動CI/CDボット（Jenkins、Zuul等）
- システムアカウント（gerrit-review、infra-bot等）
- メールアドレスパターンマッチング（`*bot@*`, `*-ci@*`）

**効果**: データ品質が大幅に向上し、人間レビュアーのみを対象とした精度の高い分析が可能に。

#### 2.2 プロジェクトフィルタリング

```bash
# 特定プロジェクトの抽出
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --output data/review_requests_nova_neutron.csv \
  --projects "openstack/nova" "openstack/neutron"

# または上位N件のプロジェクトを自動抽出
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --output data/review_requests_top3.csv \
  --top 3
```

#### 2.3 データクレンジング

```bash
# データクリーニング
uv run python data_processing/preprocessing/data_cleaning.py \
  --input data/review_requests_no_bots.csv \
  --output data/processed/cleaned_reviews.csv
```

**クリーニング内容**:
- 欠損値の補完（デフォルト値: 応答時間14日、変更行数0行）
- 重複レコードの削除
- 日付フォーマットの統一（ISO 8601）
- 異常値の検出と修正（変更行数 > 100万行等）

---

### フェーズ3: 特徴量エンジニアリング

#### 3.1 状態特徴量（10次元）

開発者の「状態」を表現する特徴量:

```python
state_features = [
    experience_days / 730.0,           # 経験年数（2年でキャップ）
    total_changes / 500.0,             # 総変更数（500件でキャップ）
    total_reviews / 500.0,             # 総レビュー数（500件でキャップ）
    recent_activity_frequency,         # 最近30日の活動頻度（0-1）
    avg_activity_gap / 60.0,           # 平均活動間隔（60日でキャップ）
    activity_trend,                    # トレンド（増加=1.0, 安定=0.5, 減少=0.0）
    collaboration_score,               # 協力スコア（0-1）
    code_quality_score,                # コード品質スコア（0-1）
    recent_acceptance_rate,            # 直近30日のレビュー受諾率（0-1）
    review_load                        # レビュー負荷（0-1、1.0=平均）
]
```

**算出ロジックの詳細**:

- **経験年数**: `(context_date - first_seen).days / 730.0`
  - 初回活動日からの経過日数を2年スケールで正規化

- **活動頻度**: `len(recent_30days_activities) / 30.0`
  - 直近30日間の活動件数を日次換算

- **活動トレンド**: 直近30日と過去30-60日の活動数を比較
  ```python
  ratio = recent_count / past_count
  if ratio > 1.2: trend = 'increasing' (1.0)
  elif ratio < 0.8: trend = 'decreasing' (0.0)
  else: trend = 'stable' (0.5)
  ```

- **協力スコア**: レビュー系活動の割合
  ```python
  collaboration_score = count(review, merge, collaboration活動) / total_activities
  ```

- **コード品質スコア**: 品質キーワードの出現率
  ```python
  quality_keywords = ['test', 'documentation', 'refactor', 'fix']
  quality_score = count(含むコミット) / total_commits + 0.3
  ```

#### 3.2 行動特徴量（5次元）

開発者の「行動」を表現する特徴量:

```python
action_features = [
    intensity,                         # 行動の強度（変更ファイル数、0-1）
    collaboration,                     # 協力度（0-1）
    response_speed,                    # レスポンス速度（素早いほど大、0-1）
    review_size                        # レビュー規模（変更行数、0-1）
]
```

**算出ロジックの詳細**:

- **強度**: `min(files_changed / 20.0, 1.0)`
  - 20ファイル変更で最大値1.0

- **レスポンス速度**: `1.0 / (1.0 + response_time_days / 3.0)`
  - 即日応答で1.0に近づき、3日で約0.5
  - 遅延するほど0に近づく

- **レビュー規模**: `min((lines_added + lines_deleted) / 500.0, 1.0)`
  - 500行変更で最大値1.0

---

### フェーズ4: IRL学習プロセス

#### 4.1 軌跡データの構築

**スライディングウィンドウ方式**:

```
時間軸: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→
         ├─────────┤  ├──────┤
         学習期間      予測期間
        (history_months) (target_months)
              ↑
         snapshot_date
```

**コード例**:
```python
# 学習期間: snapshot_date - 12ヶ月 ～ snapshot_date
history_start = snapshot_date - pd.DateOffset(months=12)
history_df = df[(df['request_time'] >= history_start) &
                (df['request_time'] < snapshot_date)]

# 予測期間: snapshot_date ～ snapshot_date + 6ヶ月
target_end = snapshot_date + pd.DateOffset(months=6)
target_df = df[(df['request_time'] >= snapshot_date) &
               (df['request_time'] < target_end)]

# 継続ラベル
for reviewer in reviewers:
    continued = len(target_df[target_df['reviewer_email'] == reviewer]) > 0
    # continued=True なら継続、False なら離脱
```

**重要な設計判断**:

1. **プロジェクト別継続判定**:
   ```python
   # 同一プロジェクト内での継続のみをカウント
   for reviewer in reviewers:
       for project in active_projects:
           continued = has_activity_in_same_project(reviewer, project, target_period)
   ```
   理由: 開発者はプロジェクトAから離脱してもプロジェクトBで継続する可能性があるため

2. **ラベル付けロジック**:
   ```python
   # 依頼なしレビュアーの扱い
   if no_review_request_in_target_period:
       label = False  # 離脱とみなす
       sample_weight = 0.5  # ただし重みを下げる
   elif has_review_request_and_accepted:
       label = True
       sample_weight = 1.0
   elif has_review_request_but_not_accepted:
       label = False
       sample_weight = 1.0
   ```

#### 4.2 モデルアーキテクチャ

```
Input: 時系列軌跡 [batch, seq_len, feature_dim]
   ↓
State Encoder (10次元 → 128次元 → 64次元)
   Linear(10, 128) → ReLU → Dropout(0.1) → Linear(128, 64) → ReLU → Dropout(0.1)
   ↓ [batch, seq_len, 64]
Action Encoder (5次元 → 128次元 → 64次元)
   Linear(5, 128) → ReLU → Dropout(0.1) → Linear(128, 64) → ReLU → Dropout(0.1)
   ↓ [batch, seq_len, 64]
Combined (Addition)
   state_encoded + action_encoded
   ↓ [batch, seq_len, 64]
LSTM (1層, hidden_size=128)
   ↓ [batch, 128]  ※最終ステップのみ使用
   ├─ Reward Predictor
   │    Linear(128, 64) → ReLU → Dropout(0.1) → Linear(64, 1)
   │    ↓ [batch, 1]
   └─ Continuation Predictor
        Linear(128, 64) → ReLU → Dropout(0.1) → Linear(64, 1) → Sigmoid
        ↓ [batch, 1]  ※継続確率（0-1）
```

**重要なパラメータ**:

- **sequence length (`seq_len`)**: 15（推奨）
  - データ分析結果: OpenStackデータの75パーセンタイルが15アクション
  - 10未満: 時系列コンテキスト不足
  - 15-20: 最適範囲
  - 20以上: 収穫逓減、計算コスト増

- **Dropout**: 0.1
  - 過学習防止のため全レイヤーに適用
  - 0.3から0.1に削減して学習安定化

- **学習率**: 0.0003
  - 0.001から削減して収束安定化

#### 4.3 訓練プロセス

```python
# 損失関数
loss = focal_loss(predicted_continuation, target_continuation) +
       mse_loss(predicted_reward, target_reward)

# Focal Loss（不均衡データ対応）
FL(p) = -α * (1 - p)^γ * log(p) * sample_weight

# パラメータ（正例率に応じて自動調整）
if positive_rate >= 0.6:
    alpha = 0.4, gamma = 1.0  # バランス重視
elif positive_rate >= 0.3:
    alpha = 0.3, gamma = 1.0  # 標準
else:
    alpha = 0.25, gamma = 1.5  # Recall重視
```

**訓練コマンド**:
```bash
uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --snapshot-date 2020-01-01 \
  --history-months 3 6 9 12 \
  --target-months 3 6 9 12 \
  --epochs 20 \
  --sequence \
  --seq-len 15 \
  --output importants/irl_openstack_real
```

**出力**:
```
importants/irl_openstack_real/
├── models/
│   ├── irl_h3m_t3m_seq.pth
│   ├── irl_h6m_t6m_seq.pth
│   ├── irl_h12m_t6m_seq.pth  ← 最高AUC-ROC 0.855
│   └── ... (全16モデル)
├── sliding_window_results_seq.csv
├── evaluation_matrix_seq.txt
└── EVALUATION_REPORT.md
```

---

### フェーズ5: 評価手法

#### 5.1 スライディングウィンドウ評価

複数の学習期間×予測期間の組み合わせで評価し、最適な設定を探索:

```python
# 評価マトリクス例
                予測期間（ヶ月）
              3      6      9     12
学習期間 3   0.731  0.444  0.683  0.682
（ヶ月） 6   0.842  0.802  0.757  0.718
        9   0.853  0.750  0.727  0.762
       12   0.777  0.855* 0.799  0.791
                    ↑ 最高AUC-ROC
```

#### 5.2 評価メトリクス

| メトリクス | 説明 | 本プロジェクトの結果 |
|-----------|------|---------------------|
| AUC-ROC | 継続/離脱の識別能力（0-1、高いほど良い） | 平均0.748、最高**0.855** |
| AUC-PR | 不均衡データでの精度（Precision-Recall曲線下面積） | 平均0.830、最高**0.983** |
| F1スコア | Precision と Recall の調和平均 | 平均0.736、最高**0.978** |
| Precision | 継続予測の正解率 | 平均0.854、最高**1.000** |
| Recall | 実際の継続者を捕捉できる割合 | 平均0.697、最高**1.000** |

**最良の組み合わせ**:

- **総合精度**: 学習12ヶ月 × 予測6ヶ月 → AUC-ROC 0.855
- **早期発見**: 学習3ヶ月 × 予測12ヶ月 → Recall 1.000, F1 0.978
- **高精度短期**: 学習6ヶ月 × 予測3ヶ月 → Precision 1.000

---

### フェーズ6: 予測の実行

#### 6.1 訓練済みモデルの読み込み

```python
from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem

# 最良モデルの読み込み
model = RetentionIRLSystem.load_model(
    'importants/irl_openstack_real/models/irl_h12m_t6m_seq.pth'
)

# モデル設定の確認
checkpoint = torch.load('model.pth')
print(f"学習期間: {checkpoint['config'].get('history_months')}ヶ月")
print(f"予測期間: {checkpoint['config'].get('target_months')}ヶ月")
print(f"シーケンスモード: {checkpoint['config'].get('sequence')}")
print(f"シーケンス長: {checkpoint['config'].get('seq_len')}")
```

#### 6.2 継続確率の予測

```python
# 予測対象の開発者データ
developer = {
    'developer_id': 'reviewer@example.com',
    'first_seen': '2019-01-01T00:00:00Z',
    'changes_authored': 150,
    'changes_reviewed': 320,
    'projects': ['openstack/nova', 'openstack/neutron']
}

# 最近の活動履歴
activity_history = [
    {
        'type': 'review',
        'timestamp': '2020-12-15T10:00:00Z',
        'project': 'openstack/nova',
        'lines_added': 45,
        'lines_deleted': 12,
        'files_changed': 3,
        'response_time': '2020-12-16T09:00:00Z'
    },
    # ... 他の活動
]

# 予測実行
result = model.predict_continuation_probability(
    developer=developer,
    activity_history=activity_history,
    context_date=datetime(2021, 1, 1)
)

print(f"継続確率: {result['continuation_probability']:.1%}")
print(f"信頼度: {result['confidence']:.1%}")
print(f"理由: {result['reasoning']}")
```

**予測結果の例**:
```python
{
    'continuation_probability': 0.87,  # 87%の確率で継続
    'confidence': 0.74,                # 信頼度74%
    'reward_score': 0.82,              # IRL報酬スコア
    'reasoning': '豊富な経験により継続確率が向上。高い活動頻度により継続確率が向上。高い協力度により継続確率が向上。学習された報酬関数により高い継続価値を予測。IRL予測継続確率: 87.0%',
    'state_features': {
        'experience_days': 730,
        'recent_activity_frequency': 0.23,
        'collaboration_score': 0.68,
        'code_quality_score': 0.75
    }
}
```

#### 6.3 アクションの実行

予測結果に基づいたアクション例:

```python
if result['continuation_probability'] < 0.3:
    # 離脱リスク高 → サポート強化
    print("⚠️ 離脱リスク: サポート施策を実施")
    actions = [
        "メンタリング担当者をアサイン",
        "簡単なタスクから開始",
        "週次チェックイン実施"
    ]

elif result['continuation_probability'] > 0.7:
    # 継続確率高 → 積極的な依頼
    print("✅ 継続見込み: 積極的なタスク依頼")
    actions = [
        "重要タスクを優先的に依頼",
        "リードポジションへの昇格検討"
    ]

else:
    # 中程度 → 経過観察
    print("📊 経過観察: 定期モニタリング")
    actions = [
        "月次活動状況チェック",
        "必要に応じてサポート"
    ]
```

---

## 📊 主要な実験結果

### 時系列学習の効果

| 手法 | AUC-ROC | 説明 |
|------|---------|------|
| 非時系列（従来） | 0.620 | 最新5アクションのみ使用、順序無視 |
| 時系列（LSTM） | **0.855** | 全履歴をLSTMで学習、+23.5%向上 |

### 継続率の分析

- **全体の継続率**: 8.5%（高度に不均衡）
- **学習3ヶ月後の継続率**: 10.2%
- **学習12ヶ月後の継続率**: 15.8%

→ 長期学習データがあるレビュアーほど継続傾向

---

## 🏗️ プロジェクト構造

```
gerrit-retention/
├── data/                                 # データディレクトリ
│   ├── raw/                              # 生データ
│   │   └── review_requests_openstack.json
│   ├── processed/                        # 前処理済みデータ
│   │   └── cleaned_reviews.csv
│   └── review_requests_openstack_multi_5y_detail.csv  # メインデータ
│
├── src/gerrit_retention/                 # ソースコード
│   ├── rl_prediction/
│   │   └── retention_irl_system.py       # ★ 時系列IRLシステム（コア）
│   ├── irl/
│   │   └── maxent_binary_irl.py          # MaxEnt IRL実装
│   ├── data_integration/                 # データ統合
│   ├── prediction/                       # 予測モデル
│   ├── recommendation/                   # レビュアー推薦
│   └── utils/                            # ユーティリティ
│
├── scripts/                              # 実行スクリプト
│   ├── preprocessing/                    # 前処理
│   │   ├── filter_bot_accounts.py        # ボット除外
│   │   └── filter_by_project.py          # プロジェクトフィルタ
│   └── training/irl/
│       ├── train_temporal_irl_sliding_window.py  # ★ スライディング評価
│       └── train_temporal_irl_project_aware.py   # プロジェクト別学習
│
├── importants/                           # 重要な実験結果
│   └── irl_openstack_real/               # メイン実験結果
│       ├── models/                       # 訓練済みモデル（16個）
│       ├── sliding_window_results_seq.csv
│       ├── evaluation_matrix_seq.txt
│       └── EVALUATION_REPORT.md
│
├── docs/                                 # ドキュメント
│   ├── (多数の日本語ドキュメント)
│   └── archive/                          # アーカイブ
│
├── README.md                             # プロジェクト概要
├── README_TEMPORAL_IRL.md                # 時系列IRL詳細ガイド
├── CLAUDE.md                             # Claude Code用ガイド
└── PROJECT_OVERVIEW.md                   # 本ドキュメント
```

---

## 🚀 クイックスタート

### 1. 環境構築

```bash
# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

### 2. データの準備

```bash
# ボットアカウントの除外
uv run python scripts/preprocessing/filter_bot_accounts.py \
  --input data/review_requests_openstack_multi_5y_detail.csv \
  --output data/review_requests_no_bots.csv

# プロジェクトのフィルタリング（任意）
uv run python scripts/preprocessing/filter_by_project.py \
  --input data/review_requests_no_bots.csv \
  --output data/review_requests_filtered.csv \
  --top 3
```

### 3. IRL学習と評価

```bash
# スライディングウィンドウ評価（推奨）
uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
  --reviews data/review_requests_no_bots.csv \
  --snapshot-date 2020-01-01 \
  --history-months 3 6 9 12 \
  --target-months 3 6 9 12 \
  --epochs 20 \
  --sequence \
  --seq-len 15 \
  --output importants/irl_my_experiment
```

実行時間: 約4分（CPU環境、16組み合わせ）

### 4. 結果の確認

```bash
# 評価マトリクスの表示
cat importants/irl_my_experiment/evaluation_matrix_seq.txt

# 詳細レポート
cat importants/irl_my_experiment/EVALUATION_REPORT.md

# CSV結果
head importants/irl_my_experiment/sliding_window_results_seq.csv
```

### 5. モデルの利用

```python
from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem
from datetime import datetime

# 最良モデルの読み込み
model = RetentionIRLSystem.load_model(
    'importants/irl_my_experiment/models/irl_h12m_t6m_seq.pth'
)

# 予測
developer = {...}  # 開発者データ
activity_history = [...]  # 活動履歴

result = model.predict_continuation_probability(
    developer=developer,
    activity_history=activity_history,
    context_date=datetime.now()
)

print(f"継続確率: {result['continuation_probability']:.1%}")
```

---

## 📚 関連ドキュメント

- **README.md**: プロジェクト基本情報
- **README_TEMPORAL_IRL.md**: 時系列IRL詳細ガイド
- **CLAUDE.md**: Claude Code用の実装ガイド
- **docs/**: 詳細な実験記録と分析レポート

---

## 🔬 技術詳細リファレンス

### 重要なハイパーパラメータ

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `seq_len` | 15 | シーケンス長（データの75パーセンタイル） |
| `hidden_dim` | 128 | LSTM隠れ層次元数 |
| `learning_rate` | 0.0003 | 学習率 |
| `dropout` | 0.1 | Dropout率 |
| `epochs` | 20-30 | 訓練エポック数 |
| `history_months` | 12 | 学習期間（推奨） |
| `target_months` | 6 | 予測期間（推奨） |

### 特徴量の正規化範囲

全特徴量は0-1の範囲に正規化され、上限でクリップされます:

- 経験日数: 730日（2年）でキャップ
- 変更数/レビュー数: 500件でキャップ
- 活動間隔: 60日でキャップ
- 変更ファイル数: 20ファイルでキャップ
- 変更行数: 500行でキャップ

### データ要件

**最小データ量**:
- 軌跡数: 20以上（訓練16 + テスト4）
- 各レビュアーの最小活動数: 1件以上

**推奨データ量**:
- 軌跡数: 100以上
- 各レビュアーの平均活動数: 5-15件

**CSV必須カラム**:
- `reviewer_email` または `email`: レビュアー識別子
- `request_time` または `created`: タイムスタンプ
- `project`: プロジェクト名

---

## 💡 ベストプラクティス

### データ前処理

1. **必ずボットを除外する**: 精度向上に最も効果的
2. **プロジェクト別に分析する**: 同一プロジェクト内での継続判定が正確
3. **データ期間を確認する**: スナップショット日前後に十分なデータがあることを確認

### モデル訓練

1. **時系列モードを使用する**: `--sequence` フラグ必須
2. **seq_len=15を使用する**: データ分布に基づく最適値
3. **複数の期間組み合わせを評価する**: スライディングウィンドウ評価推奨

### 予測の活用

1. **信頼度を確認する**: `confidence < 0.5` の場合は慎重に判断
2. **理由を参照する**: `reasoning` フィールドで予測根拠を確認
3. **定期的に再訓練する**: データ更新に応じて3-6ヶ月ごとに再訓練

---

## 🤝 貢献・質問

このプロジェクトに関する質問や改善提案は、Issueまたはプルリクエストをお願いします。

---

**最終更新**: 2025-11-04
