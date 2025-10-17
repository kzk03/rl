# 行動（Action）特徴量の詳細解説

IRLにおける「行動」「ステップ」「タスク」の定義と特徴量の説明

**最終更新**: 2025-10-17

---

## 📚 目次

1. [用語の定義](#1-用語の定義)
2. [行動特徴量の詳細](#2-行動特徴量の詳細)
3. [データの流れ](#3-データの流れ)
4. [時系列処理](#4-時系列処理)
5. [実装例](#5-実装例)

---

## 1. 用語の定義

### 1.1 行動（Action）とは？

**行動 = 1件のレビューリクエスト**

OpenStackのGerritデータでは、1行のレコード（CSVの1行）が1つの行動に相当します。

```csv
change_id,reviewer_email,request_time,project,change_insertions,change_deletions,...
abc123,alice@example.com,2023-01-15,openstack/nova,45,12,...
def456,alice@example.com,2023-01-20,openstack/nova,120,35,...
```

↓

```python
行動1: {
    'type': 'review',
    'timestamp': '2023-01-15',
    'change_insertions': 45,
    'change_deletions': 12,
    ...
}

行動2: {
    'type': 'review',
    'timestamp': '2023-01-20',
    'change_insertions': 120,
    'change_deletions': 35,
    ...
}
```

### 1.2 ステップ（Step）とは？

**ステップ = 行動の別名（RL/IRLの文脈）**

強化学習（RL）では、エージェントが環境で1回行動することを「ステップ」と呼びます。
このプロジェクトでは：

```
ステップ = 1件のレビューリクエスト = 1つの行動（Action）
```

**重要**: ステップは「訓練のエポック」ではありません！

### 1.3 タスク（Task）とは？

このプロジェクトでは2つの意味があります：

#### パターンA: レビュータスク（データレベル）
```
タスク = 1件の変更（Change）に対するレビュー依頼
```

例:
```
タスク: "openstack/nova の auth.py を変更したレビュー"
  ↓ レビュアーに割り当て
行動: alice@example.com がこのタスクをレビュー
```

#### パターンB: 予測タスク（モデルレベル）
```
タスク = 開発者の継続予測という問題全体
```

例:
```
タスク: "レビュアーAliceが3ヶ月後も活動しているか予測する"
```

---

## 2. 行動特徴量の詳細

### 2.1 基本版（5次元）

| # | 特徴量 | 説明 | 計算方法 | 値の範囲 |
|---|--------|------|---------|---------|
| 1 | `action_type` | 行動タイプ | カテゴリカルエンコーディング | 0.1-1.0 |
| 2 | `intensity` | 行動の強度 | (追加行数 + 削除行数) / (ファイル数 × 50) | 0.1-1.0 |
| 3 | `quality` | 行動の質 | キーワードベース（fix, test等） | 0.5-1.0 |
| 4 | `collaboration` | 協力度 | 行動タイプから推定 | 0.3-1.0 |
| 5 | `timestamp_age` | 時間経過 | (現在 - 行動日時) / 365日 | 0.0- |

#### action_type（行動タイプ）の詳細

**エンコーディング**:
```python
type_encoding = {
    'commit': 1.0,       # コミット（最も重要）
    'review': 0.8,       # レビュー
    'merge': 0.9,        # マージ
    'documentation': 0.6, # ドキュメント
    'issue': 0.4,        # Issue対応
    'collaboration': 0.7, # 協力活動
    'unknown': 0.1       # 不明
}
```

**OpenStackデータでは**:
- ほとんどが `'review'` (0.8)
- CSVの `request_time` カラムから判断

#### intensity（強度）の詳細

**計算式**:
```python
intensity = min(
    (change_insertions + change_deletions) / (change_files_count * 50),
    1.0
)
intensity = max(intensity, 0.1)  # 最小値0.1
```

**例**:
```python
# 小規模変更
insertions=10, deletions=5, files=1
→ intensity = (10+5)/(1*50) = 0.3

# 大規模変更
insertions=500, deletions=200, files=5
→ intensity = (500+200)/(5*50) = 2.8 → 1.0（上限）

# 空のコミット
insertions=0, deletions=0, files=1
→ intensity = 0.0 → 0.1（最小値）
```

#### quality（質）の詳細

**計算式**:
```python
quality_keywords = ['fix', 'improve', 'optimize', 'test', 'document', 'refactor']
quality_score = 0.5  # ベーススコア

for keyword in quality_keywords:
    if keyword in commit_message.lower():
        quality_score += 0.1

quality = min(quality_score, 1.0)
```

**例**:
```python
message = "Fix authentication bug"
→ 'fix' が含まれる → quality = 0.5 + 0.1 = 0.6

message = "Improve test coverage and refactor auth module"
→ 'improve', 'test', 'refactor' → quality = 0.5 + 0.3 = 0.8

message = "Update README"
→ キーワードなし → quality = 0.5
```

#### collaboration（協力度）の詳細

**行動タイプから推定**:
```python
collaboration_types = {
    'review': 0.8,        # レビューは協力的
    'merge': 0.7,         # マージも協力的
    'collaboration': 1.0, # 明示的な協力
    'mentoring': 0.9,     # メンタリング
    'documentation': 0.6, # ドキュメントは中程度
    'commit': 0.3         # コミットは個人作業
}
```

### 2.2 拡張版（9次元）

基本5次元 + 以下4次元:

| # | 特徴量 | 説明 | 計算方法 |
|---|--------|------|---------|
| 6 | `change_size` | 変更サイズ | insertions + deletions |
| 7 | `files_count` | 変更ファイル数 | change_files_count |
| 8 | `complexity` | 複雑度 | change_size / files_count |
| 9 | `response_latency` | 応答遅延 | レビュー依頼から応答までの日数 |

**complexity（複雑度）の例**:
```python
# 1ファイルに集中した変更
change_size=500, files=1
→ complexity = 500/1 = 500（高複雑度）

# 多数のファイルに分散した変更
change_size=500, files=20
→ complexity = 500/20 = 25（低複雑度）
```

---

## 3. データの流れ

### 3.1 CSVからActionオブジェクトへ

**ステップ1: CSVレコード**
```csv
reviewer_email,request_time,change_insertions,change_deletions,change_files_count,subject
alice@ex.com,2023-01-15,45,12,3,"Fix auth bug in core module"
```

**ステップ2: activity_historyに格納**
```python
activity = {
    'type': 'review',
    'timestamp': datetime(2023, 1, 15),
    'change_insertions': 45,
    'change_deletions': 12,
    'change_files_count': 3,
    'message': 'Fix auth bug in core module'
}
```

**ステップ3: DeveloperActionオブジェクト**
```python
action = DeveloperAction(
    action_type='review',           # 0.8
    intensity=0.38,                  # (45+12)/(3*50) = 0.38
    quality=0.6,                     # 'fix' が含まれる
    collaboration=0.8,               # 'review' タイプ
    timestamp=datetime(2023, 1, 15)
)
```

**ステップ4: テンソル化**
```python
action_tensor = torch.tensor([
    0.8,   # action_type (review)
    0.38,  # intensity
    0.6,   # quality
    0.8,   # collaboration
    0.02   # timestamp_age (7日前 = 7/365 ≈ 0.02)
])
# shape: [5]（基本版）
```

### 3.2 軌跡（Trajectory）の構成

**1人の開発者の軌跡**:
```python
trajectory = {
    'developer': {
        'developer_id': 'alice@example.com',
        'experience_days': 730,
        'total_changes': 120,
        ...  # 状態特徴量（10-32次元）
    },
    'activity_history': [
        # 過去3ヶ月の全行動（時系列順）
        action1,  # 2022-10-15
        action2,  # 2022-10-20
        action3,  # 2022-11-05
        ...
        action_N  # 2023-01-15（最新）
    ],
    'continued': True,  # 継続ラベル
    'context_date': datetime(2023, 1, 1)  # スナップショット日
}
```

**行動の数**:
- 最小: 1個（データ不足）
- 中央値: 7個（OpenStackデータ）
- 75%値: 15個
- 最大: 数百個

---

## 4. 時系列処理

### 4.1 非時系列モード（sequence=False）

**処理**: 最新5個の行動のみを独立に学習

```python
# 軌跡に20個の行動がある場合
actions = [action1, action2, ..., action20]

# 最新5個のみ使用
recent_5 = actions[-5:]  # [action16, action17, action18, action19, action20]

# 各行動を独立に学習
for action in recent_5:
    state_tensor = [...]  # shape: [1, 10]
    action_tensor = [...]  # shape: [1, 5]

    predicted_reward, predicted_continuation = network(state_tensor, action_tensor)
```

**問題点**:
- ❌ 時系列順序を無視
- ❌ 行動間の依存関係を捉えられない

### 4.2 時系列モード（sequence=True）★推奨

**処理**: LSTMで全行動の時系列パターンを学習

```python
# 軌跡に20個の行動がある場合
actions = [action1, action2, ..., action20]

# シーケンス長15に調整
if len(actions) < 15:
    # パディング: 最初の行動を繰り返す
    padded = [actions[0]] * (15 - len(actions)) + actions
else:
    # トランケート: 最新15個を使用
    padded = actions[-15:]  # [action6, action7, ..., action20]

# 時系列テンソル化
state_seq = torch.stack([state_to_tensor(state)] * 15)    # shape: [1, 15, 10]
action_seq = torch.stack([action_to_tensor(a) for a in padded])  # shape: [1, 15, 5]

# LSTMで時系列学習
predicted_reward, predicted_continuation = network(state_seq, action_seq)
```

**利点**:
- ✅ 時系列順序を保持
- ✅ 行動パターンの変化を捉える（例: 活動が増加/減少）
- ✅ LSTMで長期依存性を学習

### 4.3 シーケンス長（seq_len）の役割

**seq_len = 15（デフォルト）**

```
行動の数が7個の場合:
  [action1] [action1] [action1] [action1] [action1] [action1] [action1] [action1] [action2] [action3] [action4] [action5] [action6] [action7]
  ↑ パディング（8個）                                                                  ↑ 実際の行動（7個）

行動の数が20個の場合:
  [action6] [action7] [action8] [action9] [action10] [action11] [action12] [action13] [action14] [action15] [action16] [action17] [action18] [action19] [action20]
  ↑ トランケート（最新15個のみ使用）
```

**なぜ15か？**:
- OpenStackデータの75%値が15個
- 計算効率とカバレッジのバランス

詳細: [seq_len_explanation.md](seq_len_explanation.md)

---

## 5. 実装例

### 5.1 行動の抽出

```python
def extract_developer_actions(activity_history, context_date):
    """開発者の行動を抽出"""
    actions = []

    for activity in activity_history:
        # 行動タイプ
        action_type = activity.get('type', 'unknown')

        # 強度
        lines_added = activity.get('change_insertions', 0)
        lines_deleted = activity.get('change_deletions', 0)
        files_changed = activity.get('change_files_count', 1)
        intensity = min((lines_added + lines_deleted) / (files_changed * 50), 1.0)
        intensity = max(intensity, 0.1)

        # 質
        message = activity.get('message', '').lower()
        quality_keywords = ['fix', 'improve', 'optimize', 'test', 'document', 'refactor']
        quality = 0.5
        for keyword in quality_keywords:
            if keyword in message:
                quality += 0.1
        quality = min(quality, 1.0)

        # 協力度
        collaboration = 0.8 if action_type == 'review' else 0.3

        # タイムスタンプ
        timestamp = activity.get('timestamp', context_date)

        actions.append(DeveloperAction(
            action_type=action_type,
            intensity=intensity,
            quality=quality,
            collaboration=collaboration,
            timestamp=timestamp
        ))

    return actions
```

### 5.2 テンソル化

```python
def action_to_tensor(action):
    """行動をテンソルに変換"""
    type_encoding = {
        'commit': 1.0,
        'review': 0.8,
        'merge': 0.9,
        'unknown': 0.1
    }

    features = [
        type_encoding.get(action.action_type, 0.1),  # action_type
        action.intensity,                             # intensity
        action.quality,                               # quality
        action.collaboration,                         # collaboration
        (datetime.now() - action.timestamp).days / 365.0  # timestamp_age
    ]

    return torch.tensor(features, dtype=torch.float32)
```

### 5.3 時系列処理

```python
def prepare_sequence(actions, seq_len=15):
    """シーケンス長に合わせて調整"""
    if len(actions) < seq_len:
        # パディング
        padded = [actions[0]] * (seq_len - len(actions)) + actions
    else:
        # トランケート
        padded = actions[-seq_len:]

    # テンソル化
    action_tensors = [action_to_tensor(a) for a in padded]
    action_seq = torch.stack(action_tensors).unsqueeze(0)  # [1, seq_len, 5]

    return action_seq
```

---

## 6. まとめ

### 用語の整理

| 用語 | 定義 | データ上の対応 | 数量 |
|-----|------|--------------|------|
| **行動（Action）** | 1件のレビューリクエスト | CSVの1行 | 1個 |
| **ステップ（Step）** | 行動の別名（RL用語） | 同上 | 1個 |
| **軌跡（Trajectory）** | 1人の開発者の全行動 | 1人分のCSV行群 | 1-数百個の行動 |
| **エピソード（Episode）** | 軌跡の別名（RL用語） | 同上 | 1-数百個の行動 |
| **エポック（Epoch）** | 訓練の1周 | 全軌跡を1回学習 | - |

### 行動特徴量の要点

1. **基本5次元**: type, intensity, quality, collaboration, timestamp_age
2. **拡張9次元**: 基本5次元 + change_size, files_count, complexity, response_latency
3. **時系列処理**: LSTMでseq_len個の行動を時系列学習（推奨）
4. **データソース**: CSVの1行 = 1行動

### 重要なポイント

- ✅ 行動 = レビューリクエスト（CSVの1行）
- ✅ ステップ = 行動（同じもの）
- ✅ タスク = レビュータスク or 継続予測問題全体
- ✅ 時系列モード（sequence=True）が推奨

---

## 📚 関連ドキュメント

- [IRL_FEATURE_SUMMARY.md](IRL_FEATURE_SUMMARY.md): 特徴量全体のまとめ
- [seq_len_explanation.md](seq_len_explanation.md): シーケンス長の詳細
- [IRL_COMPREHENSIVE_GUIDE.md](IRL_COMPREHENSIVE_GUIDE.md): IRL全体ガイド

---

**作成者**: Claude + Kazuki-h
**ステータス**: 完成
