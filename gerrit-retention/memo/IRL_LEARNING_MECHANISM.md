# IRL学習メカニズムの詳細解説

## 目次

1. [現在の学習方法](#現在の学習方法)
2. [タスク拒否機能追加後の学習](#タスク拒否機能追加後の学習)
3. [選択肢の比較](#選択肢の比較)
4. [なぜこの設計なのか](#なぜこの設計なのか)

---

## 現在の学習方法

### アーキテクチャ全体像

```
入力データ（軌跡）
  │
  ├─ 状態 (State) [10次元]
  │   └─ experience_days, total_changes, total_reviews,
  │      recent_activity_frequency, avg_activity_gap, activity_trend,
  │      collaboration_score, code_quality_score,
  │      recent_acceptance_rate, review_load
  │
  └─ 行動列 (Actions) [seq_len x 4次元]
      └─ 各行動: intensity, collaboration, response_time, review_size
      └─ 時系列: [行動1, 行動2, ..., 行動N]

  ↓ エンコーディング

State Encoder (Linear→ReLU→Linear) : [10] → [64]
Action Encoder (Linear→ReLU→Linear) : [4] → [64]

  ↓ 結合

Combined = State_Encoded + Action_Encoded : [seq_len x 64]

  ↓ 時系列学習

LSTM (1-layer, hidden=128) : [seq_len x 64] → [seq_len x 128]
  └─ 最終ステップの隠れ状態を取得: [128]

  ↓ 予測

├─ Reward Predictor : [128] → [1]
│   └─ 予測報酬（継続=1.0, 離脱=0.0）
│
└─ Continuation Predictor : [128] → [1] → Sigmoid
    └─ 継続確率（0.0〜1.0）
```

### 学習プロセス（コードレベル）

#### ステップ1: 軌跡から特徴量抽出

**ファイル**: `retention_irl_system.py:503-513`

```python
# 入力: 1つの軌跡（1人のレビュアーの履歴）
trajectory = {
    'developer': {
        'developer_id': 'reviewer1@example.com',
        # ...
    },
    'activity_history': [
        {'type': 'review', 'timestamp': '2020-01-01', ...},
        {'type': 'review', 'timestamp': '2020-01-15', ...},
        {'type': 'review', 'timestamp': '2020-02-01', ...},
        # ... 全活動履歴
    ],
    'continued': True,  # ★ラベル: 予測期間中に継続したか
    'context_date': datetime(2020, 3, 1)
}

# 抽出
state = extract_developer_state(...)  # 状態: 1つ（スナップショット時点）
actions = extract_developer_actions(...)  # 行動: 複数（時系列）
```

**重要**:
- **状態（State）**: context_date時点の**累積的な特性**（1つのベクトル）
- **行動（Actions）**: 各活動の**個別特性**（複数のベクトル、時系列）

---

#### ステップ2: シーケンス化（パディング/トランケート）

**ファイル**: `retention_irl_system.py:518-534`

```python
seq_len = 15  # 固定長

# ケース1: 行動数 < seq_len の場合（パディング）
actions = [行動1, 行動2, 行動3]  # 3個
padded_actions = [行動1, 行動1, 行動1, ..., 行動1, 行動2, 行動3]  # 15個に
                  └─────────── 12回繰り返し ──────────┘

# ケース2: 行動数 >= seq_len の場合（トランケート）
actions = [行動1, 行動2, ..., 行動20]  # 20個
padded_actions = [行動6, 行動7, ..., 行動20]  # 最新15個を使用
```

**設計意図**:
- LSTMは固定長入力を期待
- 最新の行動を優先（トランケート時）
- パディングは最初の行動を繰り返し（データ不足対策）

---

#### ステップ3: テンソル変換

**ファイル**: `retention_irl_system.py:528-534`

```python
# 状態: 全タイムステップで同じ値を使用
state_tensors = [state_to_tensor(state)] * seq_len
state_seq = torch.stack(state_tensors)  # [seq_len, 10]
# ↑ 例: [[0.5, 0.3, ...], [0.5, 0.3, ...], ..., [0.5, 0.3, ...]]

# 行動: 時系列順序を保持
action_tensors = [action_to_tensor(a) for a in padded_actions]
action_seq = torch.stack(action_tensors)  # [seq_len, 4]
# ↑ 例: [[0.2, 0.8, ...], [0.5, 0.6, ...], ..., [0.9, 0.3, ...]]
```

**重要な設計選択**:
- **状態は全ステップで同じ** ← これが今回の議論の核心！
- 行動は時系列で変化

---

#### ステップ4: ネットワークのforward pass

**ファイル**: `retention_irl_system.py:101-163`

```python
def forward(state, action):
    # state: [1, seq_len, 10]
    # action: [1, seq_len, 4]

    # 1. エンコーディング
    state_encoded = state_encoder(state)    # [1, seq_len, 64]
    action_encoded = action_encoder(action)  # [1, seq_len, 64]

    # 2. 結合（加算）
    combined = state_encoded + action_encoded  # [1, seq_len, 64]

    # 3. LSTM処理（★時系列学習の核心）
    lstm_out, _ = lstm(combined)  # [1, seq_len, 128]
    #   ↑ 各タイムステップで：
    #      h_t = LSTM(combined_t, h_{t-1})
    #      combined_t = state_encoded_t + action_encoded_t

    # 4. 最終ステップの隠れ状態を使用
    hidden = lstm_out[:, -1, :]  # [1, 128]

    # 5. 予測
    reward = reward_predictor(hidden)         # [1, 1]
    continuation_prob = continuation_predictor(hidden)  # [1, 1] → Sigmoid

    return reward, continuation_prob
```

**LSTMの動作**:
```
t=0: h_0 = LSTM([state_0 + action_0], h_init)
t=1: h_1 = LSTM([state_1 + action_1], h_0)     ← h_0の情報を引き継ぐ
t=2: h_2 = LSTM([state_2 + action_2], h_1)     ← h_1の情報を引き継ぐ
...
t=14: h_14 = LSTM([state_14 + action_14], h_13)

最終予測 = continuation_predictor(h_14)  ← 全時系列情報を集約
```

---

#### ステップ5: 損失計算とバックプロパゲーション

**ファイル**: `retention_irl_system.py:541-559`

```python
# 予測値
predicted_reward = 0.7          # ネットワークの出力
predicted_continuation = 0.8    # ネットワークの出力

# 正解ラベル
continuation_label = True       # この人は継続した
target_reward = 1.0            # 継続=1.0, 離脱=0.0
target_continuation = 1.0      # 継続=1.0, 離脱=0.0

# 損失計算
reward_loss = MSE(0.7, 1.0) = 0.09
continuation_loss = BCE(0.8, 1.0) = 0.223
total_loss = 0.09 + 0.223 = 0.313

# バックプロパゲーション
total_loss.backward()  # 勾配計算
optimizer.step()       # パラメータ更新
```

**学習される内容**:
- 「継続した人」の軌跡 → 高い継続確率を出力するように調整
- 「離脱した人」の軌跡 → 低い継続確率を出力するように調整

---

### 現在の学習が捉えているパターン

#### パターン1: 活動頻度の減少

```
Timeline:
月1: [行動, 行動, 行動, 行動, 行動]  ← 高頻度
月2: [行動, 行動, 行動]              ← 中頻度
月3: [行動]                         ← 低頻度
→ 離脱

LSTMが学習:
「行動間隔が広がる（response_time増加）→ 離脱」
```

#### パターン2: 協力度の低下

```
Timeline:
t=0: collaboration=0.9, intensity=0.8
t=1: collaboration=0.8, intensity=0.7
t=2: collaboration=0.5, intensity=0.3
→ 離脱

LSTMが学習:
「協力的な行動が減る → 離脱」
```

#### パターン3: レビュー規模の減少

```
Timeline:
t=0: review_size=0.9 (大規模レビュー)
t=1: review_size=0.5 (中規模)
t=2: review_size=0.2 (小規模)
→ 離脱

LSTMが学習:
「小さいレビューしかしなくなる → 離脱」
```

---

## タスク拒否機能追加後の学習

### 新しいアーキテクチャ

```
入力データ（軌跡）
  │
  ├─ 状態 (State) [14次元]  ← 4次元追加
  │   └─ ... (既存10次元) ...
  │      recent_response_rate,          ★NEW: 直近の応答率
  │      consecutive_rejections,         ★NEW: 連続拒否回数
  │      rejection_trend,                ★NEW: 拒否傾向
  │      avg_rejection_rate              ★NEW: 平均拒否率
  │
  └─ 行動列 (Actions) [seq_len x 5次元]  ← 1次元追加
      └─ 各行動: intensity, collaboration, response_time, review_size,
                responded  ★NEW: この依頼に応答したか（1.0/0.0）
```

### 学習プロセスの変化

#### ステップ1: より詳細な軌跡

```python
trajectory = {
    'activity_history': [
        {
            'type': 'review',
            'timestamp': '2020-01-01',
            'responded': 1,  ★NEW: 応答した
            # ...
        },
        {
            'type': 'review',
            'timestamp': '2020-01-15',
            'responded': 1,  ★NEW: 応答した
            # ...
        },
        {
            'type': 'review',
            'timestamp': '2020-02-01',
            'responded': 0,  ★NEW: 拒否した
            # ...
        },
        {
            'type': 'review',
            'timestamp': '2020-02-15',
            'responded': 0,  ★NEW: 拒否した
            # ...
        },
    ],
    'continued': False,  # この人は離脱した
}
```

#### ステップ2: 状態の計算が変わる

```python
# 既存の状態計算
state = {
    'experience_days': 365,
    'total_changes': 50,
    'recent_activity_frequency': 0.3,
    # ... 既存特徴量 ...
}

# ★NEW: 拒否パターンの計算
recent_activities = [
    {'responded': 1},  # 30日前: 応答
    {'responded': 1},  # 25日前: 応答
    {'responded': 0},  # 15日前: 拒否
    {'responded': 0},  # 10日前: 拒否
    {'responded': 0},  # 5日前: 拒否
]

state['recent_response_rate'] = 2/5 = 0.4      # 直近40%しか応答していない
state['consecutive_rejections'] = 3             # 連続3回拒否
state['rejection_trend'] = 'increasing'         # 悪化傾向
state['avg_rejection_rate'] = 0.3               # 全期間では30%拒否
```

#### ステップ3: LSTMの入力が変わる

**現在**:
```python
# 時系列入力
t=0: state=[0.5, 0.3, ...] + action=[0.8, 0.9, 0.5, 0.3]
t=1: state=[0.5, 0.3, ...] + action=[0.7, 0.8, 0.6, 0.4]
t=2: state=[0.5, 0.3, ...] + action=[0.6, 0.7, 0.7, 0.5]
```

**拡張後**:
```python
# 時系列入力（responded追加）
t=0: state=[0.5, 0.3, ..., 1.0, 0, 'stable', 0.2] + action=[0.8, 0.9, 0.5, 0.3, 1.0]
                          ^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^
                          拒否パターン（4次元）                responded

t=1: state=[0.5, 0.3, ..., 0.9, 0, 'stable', 0.2] + action=[0.7, 0.8, 0.6, 0.4, 1.0]
                          ^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^
                          応答率下がる                         応答

t=2: state=[0.5, 0.3, ..., 0.7, 1, 'increasing', 0.3] + action=[0.6, 0.7, 0.7, 0.5, 0.0]
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^                ^^^^
                          応答率さらに低下、拒否増加                拒否！

t=3: state=[0.5, 0.3, ..., 0.5, 2, 'increasing', 0.4] + action=[0.5, 0.6, 0.8, 0.6, 0.0]
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^                ^^^^
                          連続拒否2回                                拒否！
```

### 新しく学習できるパターン

#### パターン4: 拒否増加→離脱

```
Timeline:
t=0: responded=1.0, recent_response_rate=1.0, consecutive=0
t=1: responded=1.0, recent_response_rate=1.0, consecutive=0
t=2: responded=0.0, recent_response_rate=0.8, consecutive=1  ← 拒否開始
t=3: responded=0.0, recent_response_rate=0.6, consecutive=2  ← 拒否続く
t=4: responded=0.0, recent_response_rate=0.4, consecutive=3  ← さらに拒否
→ 離脱

LSTMが学習:
「応答 → 応答 → 拒否 → 拒否 → 拒否 → 離脱」
         ^^^^^^^^^^^^^^^^^^^^^^^^
         このパターンを検出
```

#### パターン5: 負荷過多→拒否増加→離脱

```
Timeline:
t=0: review_load=0.5, responded=1.0
t=1: review_load=1.2, responded=1.0  ← 負荷増加
t=2: review_load=1.8, responded=0.0  ← 負荷過多で拒否開始
t=3: review_load=2.0, responded=0.0  ← 拒否続く
→ 離脱

LSTMが学習:
「負荷 ↑ → 拒否開始 → 離脱」の因果関係
```

#### パターン6: 選択的拒否（特定条件で拒否）

```
Timeline:
大規模タスク: review_size=0.9, responded=0.0  ← 大きいタスクは拒否
小規模タスク: review_size=0.2, responded=1.0  ← 小さいタスクは応答
大規模タスク: review_size=0.8, responded=0.0  ← また拒否
→ 継続（まだ離脱しない）

LSTMが学習:
「大規模タスクの選択的拒否は離脱と相関が低い」
※ ただし、状態のrecent_response_rateは低下するため、総合的に判断
```

---

## 選択肢の比較

### 選択肢1: 拒否を「状態」に含める（★採用）

```python
state = {
    'recent_response_rate': 0.4,
    'consecutive_rejections': 3,
    # ...
}
action = {
    'intensity': 0.5,
    'collaboration': 0.7,
    # responded は含めない
}
```

**メリット**:
- ✅ 状態は「累積的な特性」なので、拒否パターンがよく馴染む
- ✅ 予測時に「最近の拒否傾向」だけで判断可能（タスク情報不要）
- ✅ 計算が軽い（毎タイムステップで状態を再計算しない）

**デメリット**:
- ❌ 個別の拒否イベントの詳細が失われる
- ❌ 「いつ拒否したか」の時系列情報が粗い

---

### 選択肢2: 拒否を「行動」に含める（★採用）

```python
state = {
    'experience_days': 365,
    # ... 既存特徴量のみ ...
}
action = {
    'intensity': 0.5,
    'collaboration': 0.7,
    'responded': 0.0,  ← NEW
}
```

**メリット**:
- ✅ 個別の拒否イベントを時系列で学習
- ✅ LSTMが「拒否→拒否→離脱」の細かいパターンを捉える
- ✅ タスクの特徴（intensity, review_size）と拒否の相関を学習

**デメリット**:
- ❌ 状態に比べて粒度が細かすぎる可能性
- ❌ パディング時に最初の行動を繰り返すため、拒否情報が歪む

---

### 選択肢3: 両方に含める（★★最強：今回の提案★★）

```python
state = {
    'recent_response_rate': 0.4,      # 集約された拒否パターン
    'consecutive_rejections': 3,      # 集約された拒否パターン
    # ...
}
action = {
    'intensity': 0.5,
    'collaboration': 0.7,
    'responded': 0.0,                 # 個別の拒否イベント
}
```

**メリット**:
- ✅✅ **状態で大局的なパターン**（「最近拒否が多い」）を捉える
- ✅✅ **行動で細かい時系列パターン**（「拒否→拒否→拒否」）を捉える
- ✅✅ **両方の情報が相互補完**
  - 状態: 「この人は最近40%しか応答していない」
  - 行動: 「直近3回連続で拒否している」
  - LSTMが両方を統合して「離脱リスク高」と判断

**デメリット**:
- ❌ 若干の情報重複（responded と recent_response_rate）
- ✅ **しかし、粒度が違うため問題ない**

---

### 選択肢4: 拒否を別タスクとして学習

```python
# モデル1: レビュー応答予測
input: task_features (変更規模、プロジェクトなど)
output: will_respond (0/1)

# モデル2: 継続予測（従来通り）
input: state + actions
output: will_continue (0/1)
```

**メリット**:
- ✅ タスク応答と継続予測を分離（クリーンな設計）
- ✅ タスク応答予測で「どんなタスクが拒否されるか」を直接学習

**デメリット**:
- ❌ 2つのモデルが必要（複雑）
- ❌ 「拒否→離脱」の因果関係を直接学習できない
- ❌ モデル間で情報共有がない

---

## なぜこの設計なのか

### 採用した設計：選択肢3（状態 + 行動の両方）

```
状態 (State) [14次元]
  └─ 拒否パターンの集約（4次元追加）
      ├─ recent_response_rate: 「最近どのくらい拒否しているか」
      ├─ consecutive_rejections: 「何回連続で拒否しているか」
      ├─ rejection_trend: 「拒否が増えているか」
      └─ avg_rejection_rate: 「全期間でどのくらい拒否しているか」

行動 (Action) [5次元]
  └─ 個別の拒否イベント（1次元追加）
      └─ responded: 「この依頼に応答したか」(1.0/0.0)
```

### 理由1: 粒度の違いを活用

| 特徴量 | 粒度 | 捉えるパターン |
|--------|------|----------------|
| **状態: recent_response_rate** | 30日単位 | 「最近1ヶ月の傾向」 |
| **行動: responded** | イベント単位 | 「この依頼への応答」 |

**例**:
```
状態: recent_response_rate = 0.4  ← 「1ヶ月で40%応答」
行動: [1.0, 1.0, 0.0, 0.0, 0.0]  ← 「前半応答、後半拒否」

→ LSTMが「応答率40%だが、最近は拒否が続いている」を検出
```

### 理由2: LSTMの時系列学習能力を最大活用

```
LSTMへの入力（各タイムステップ）:
  state_t + action_t

t=0: [response_rate=1.0, consecutive=0, ...] + [responded=1.0, ...]
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
     「過去の累積傾向」                         「今回の行動」

t=1: [response_rate=0.9, consecutive=0, ...] + [responded=1.0, ...]
t=2: [response_rate=0.8, consecutive=0, ...] + [responded=0.0, ...]  ← 拒否開始
t=3: [response_rate=0.7, consecutive=1, ...] + [responded=0.0, ...]  ← 拒否続く
t=4: [response_rate=0.6, consecutive=2, ...] + [responded=0.0, ...]  ← さらに拒否

→ LSTMが学習:
   「状態の悪化（response_rate↓, consecutive↑）」
   AND
   「行動の変化（responded: 1→0）」
   → 「離脱リスク高」
```

### 理由3: 予測時にタスク情報が不要

**予測シナリオ**:
```python
# 予測時点での状態（過去の履歴から計算）
current_state = {
    'recent_response_rate': 0.3,      # 集約情報
    'consecutive_rejections': 5,      # 集約情報
    'rejection_trend': 'increasing',  # 集約情報
    # ...
}

# 最近の行動履歴
recent_actions = [
    Action(responded=1.0, ...),  # 20日前: 応答
    Action(responded=0.0, ...),  # 15日前: 拒否
    Action(responded=0.0, ...),  # 10日前: 拒否
    Action(responded=0.0, ...),  # 5日前: 拒否
]

# 予測（タスク情報なし）
prediction = model.predict(current_state, recent_actions)
# → 継続確率: 0.15（離脱リスク高）
```

**なぜタスク情報が不要か**:
- **状態**が「過去の拒否パターン」を集約している
- **行動**が「個別の拒否イベント」を記録している
- この2つで「拒否傾向」が十分表現できる

---

### 理由4: 因果関係の学習

```
因果チェーン:
負荷増加 → 拒否増加 → 離脱
   ↓          ↓        ↓
状態特徴    行動特徴   ラベル
```

**学習例**:
```
サンプル1（離脱パターン）:
t=0: state=[load=0.5, response_rate=1.0], action=[responded=1.0] → continued=True
t=1: state=[load=1.5, response_rate=0.8], action=[responded=0.0] → continued=True
t=2: state=[load=2.0, response_rate=0.5], action=[responded=0.0] → continued=False

サンプル2（継続パターン）:
t=0: state=[load=0.5, response_rate=1.0], action=[responded=1.0] → continued=True
t=1: state=[load=0.6, response_rate=0.9], action=[responded=1.0] → continued=True
t=2: state=[load=0.7, response_rate=0.9], action=[responded=1.0] → continued=True

→ モデルが学習:
  「load↑ かつ response_rate↓ かつ responded=0.0 → 離脱」
```

---

## まとめ：なぜこの設計が最適か

### 選択した設計の強み

1. **二重の情報源**
   - 状態: マクロ視点（30日の傾向）
   - 行動: ミクロ視点（各イベント）

2. **LSTMの能力を最大活用**
   - 状態の変化を追跡（response_rate↓, consecutive↑）
   - 行動の時系列パターンを学習（応答→拒否→拒否）

3. **予測時の柔軟性**
   - タスク情報不要（拒否履歴だけで予測）
   - リアルタイム予測が可能

4. **因果関係の学習**
   - 負荷→拒否→離脱の連鎖を捉える

5. **実装コストが低い**
   - 既存の枠組みに自然に追加
   - 10次元→14次元、4次元→5次元の拡張のみ

### 他の選択肢を採用しない理由

| 選択肢 | 採用しない理由 |
|--------|----------------|
| **状態のみ** | 個別イベントの時系列が失われる |
| **行動のみ** | 大局的な傾向が見えにくい |
| **別モデル** | 複雑で、因果関係を直接学習できない |

---

## 実装時の注意点

### 1. 状態と行動の独立性を保つ

```python
# ✅ 良い例
state = {
    'recent_response_rate': 0.4,  # 過去30日の集約
}
action = {
    'responded': 0.0,  # この1回のイベント
}

# ❌ 悪い例（情報重複）
state = {
    'last_action_responded': 0.0,  # 最新の行動を状態に含める
}
action = {
    'responded': 0.0,  # 同じ情報
}
```

### 2. 時系列の整合性を保つ

```python
# 状態は context_date 時点の累積情報
state = extract_state(activity_history, context_date='2020-03-01')
# → recent_response_rate は 2020-02-01 〜 2020-03-01 の30日間

# 行動は context_date より前のイベント
actions = extract_actions(activity_history, context_date='2020-03-01')
# → 2020-03-01 以前のすべてのイベント
```

### 3. パディング時の注意

```python
# 行動が少ない場合
actions = [Action(responded=1.0, ...)]  # 1個だけ

# パディング
padded = [actions[0]] * 15  # 最初の行動を15回繰り返す

# ⚠️ responded=1.0 が15回繰り返される
# → recent_response_rate=1.0 なら整合性あり
# → recent_response_rate=0.4 なら不整合（要注意）
```

---

## 期待される学習結果

### 学習後のモデルの挙動

```python
# ケース1: 健全なレビュアー
state = {'recent_response_rate': 0.9, 'consecutive_rejections': 0}
actions = [responded=1.0, responded=1.0, responded=1.0]
→ prediction = 0.95 (継続確率高)

# ケース2: 拒否が始まったレビュアー
state = {'recent_response_rate': 0.7, 'consecutive_rejections': 2}
actions = [responded=1.0, responded=1.0, responded=0.0, responded=0.0]
→ prediction = 0.60 (やや離脱リスク)

# ケース3: 拒否が続くレビュアー
state = {'recent_response_rate': 0.3, 'consecutive_rejections': 5}
actions = [responded=0.0, responded=0.0, responded=0.0, responded=0.0]
→ prediction = 0.20 (離脱リスク高)

# ケース4: 負荷過多のレビュアー
state = {'review_load': 2.0, 'recent_response_rate': 0.4}
actions = [responded=1.0, responded=0.0, responded=0.0]
→ prediction = 0.30 (離脱リスク高、負荷が原因)
```

### 特徴量重要度の予想

```
予想される重要度ランキング:
1. consecutive_rejections       (連続拒否が最強のシグナル)
2. recent_response_rate         (直近の応答率)
3. rejection_trend              (悪化傾向の検出)
4. review_load                  (負荷との相関)
5. responded (行動)             (個別イベント)
6. avg_rejection_rate           (長期傾向)
7. ... 既存特徴量 ...
```

---

以上が、現在の学習方法とタスク拒否機能追加後の学習メカニズムの詳細解説です。
