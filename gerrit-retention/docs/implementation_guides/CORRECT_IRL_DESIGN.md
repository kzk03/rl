# 正しいIRL実装設計

## 問題点

現在の実装は「時系列分類器」であり、真のIRLではない。

### 現在の実装
- 訓練: LSTMに時系列全体を入力 → 継続確率を出力
- 予測: LSTMに時系列全体を入力 → 継続確率を出力
- 問題: **報酬関数を学習していない**

### 正しいIRL
- 訓練: 時系列トラジェクトリから報酬関数 R(s,a) を学習
- 予測: **スナップショット時点の(s,a)だけ**で報酬値を計算

## 正しい実装

### 1. 報酬関数ネットワーク

```python
class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 報酬値（スカラー）
        )

    def forward(self, state, action):
        # state: [batch, state_dim]  ← 1時点だけ！
        # action: [batch, action_dim] ← 1時点だけ！
        x = torch.cat([state, action], dim=-1)
        reward = self.net(x)
        return reward  # [batch, 1]
```

### 2. IRL訓練（時系列トラジェクトリから学習）

```python
def train_irl_reward_function(trajectories, labels, epochs):
    """
    時系列トラジェクトリから報酬関数を学習
    """
    reward_net = RewardNetwork(state_dim=32, action_dim=9, hidden_dim=128)
    optimizer = optim.Adam(reward_net.parameters(), lr=0.001)

    for epoch in range(epochs):
        for trajectory, label in zip(trajectories, labels):
            # trajectory['states']: [seq_len, 32]
            # trajectory['actions']: [seq_len, 9]

            # 各時点での報酬を計算
            rewards = []
            for t in range(seq_len):
                state_t = trajectory['states'][t]  # [32]
                action_t = trajectory['actions'][t]  # [9]

                reward_t = reward_net(state_t, action_t)  # スカラー
                rewards.append(reward_t)

            # トラジェクトリ全体の累積報酬
            cumulative_reward = sum(rewards)

            # 継続した人は高報酬、離脱した人は低報酬になるように学習
            # Binary Cross Entropy with reward as logit
            pred_logit = cumulative_reward
            loss = F.binary_cross_entropy_with_logits(
                pred_logit,
                torch.tensor([label], dtype=torch.float)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return reward_net
```

### 3. 予測（スナップショット時点の特徴量のみ）

```python
def predict_continuation_at_snapshot(
    reviewer,
    snapshot_date,
    reward_net,
    df,
    learning_months
):
    """
    スナップショット時点での特徴量で継続確率を予測
    """
    learning_start = snapshot_date - timedelta(days=learning_months * 30)

    # スナップショット時点までの全活動履歴
    reviewer_history = df[
        (df['reviewer'] == reviewer) &
        (df['timestamp'] >= learning_start) &
        (df['timestamp'] < snapshot_date)
    ]

    if len(reviewer_history) == 0:
        return None  # データなし

    # スナップショット時点での累積特徴量を計算
    snapshot_state = compute_snapshot_features(
        reviewer_history,
        snapshot_date,
        df
    )  # [32-dim]

    # 最も直近の行動
    last_action = extract_last_action(reviewer_history)  # [9-dim]

    # 報酬関数で評価（1時点だけ！）
    with torch.no_grad():
        reward = reward_net(
            torch.FloatTensor(snapshot_state),
            torch.FloatTensor(last_action)
        )

        # 報酬値を確率に変換
        continuation_prob = torch.sigmoid(reward).item()

    return continuation_prob
```

### 4. スナップショット時点の特徴量

```python
def compute_snapshot_features(reviewer_history, snapshot_date, all_data):
    """
    スナップショット時点での開発者の累積特徴量
    """
    features = []

    # 1. 経験値（累積レビュー数）
    features.append(len(reviewer_history))

    # 2. 在籍期間（日数）
    first_activity = reviewer_history['timestamp'].min()
    tenure_days = (snapshot_date - first_activity).days
    features.append(tenure_days)

    # 3. 最近の活動頻度（過去30日間）
    last_30_days = snapshot_date - timedelta(days=30)
    recent_activity = reviewer_history[reviewer_history['timestamp'] >= last_30_days]
    features.append(len(recent_activity))

    # 4. 最終活動からの経過日数
    last_activity = reviewer_history['timestamp'].max()
    days_since_last = (snapshot_date - last_activity).days
    features.append(days_since_last)

    # 5. 平均活動間隔
    if len(reviewer_history) > 1:
        timestamps = reviewer_history['timestamp'].sort_values()
        intervals = timestamps.diff().dt.days.dropna()
        avg_interval = intervals.mean()
    else:
        avg_interval = 0
    features.append(avg_interval)

    # 6-32. その他の特徴量
    # - プロジェクト数
    # - コラボレーションスコア
    # - レビューの質
    # - etc.

    return np.array(features, dtype=np.float32)  # [32-dim]
```

## メリット

### 1. **真のIRL実装**
- 時系列から報酬関数を学習
- スナップショット時点で予測

### 2. **解釈可能性**
```python
# どの特徴量が報酬（継続性）に寄与しているか分析可能
state = compute_snapshot_features(reviewer, snapshot_date)

# 各特徴を少し変化させて報酬の変化を見る
for i in range(32):
    state_perturbed = state.copy()
    state_perturbed[i] += 0.1

    reward_original = reward_net(state, action)
    reward_perturbed = reward_net(state_perturbed, action)

    sensitivity = reward_perturbed - reward_original
    print(f"Feature {i} sensitivity: {sensitivity}")
```

### 3. **予測が軽い**
- LSTMで15時点処理 → 1時点だけの特徴量計算
- 高速で効率的

### 4. **母集団問題の解決**
- スナップショット時点の特徴量さえあれば予測可能
- 学習期間中にデータがなくてもOK
  - 前の期間のデータから特徴量を計算できる
  - またはゼロパディング/平均値で対応

## 実装の流れ

```bash
# 1. 報酬関数を学習（時系列トラジェクトリから）
python scripts/training/irl/train_reward_function.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --snapshot-date 2022-01-01 \
  --learning-months 12 \
  --output models/reward_function.pth

# 2. スナップショット時点の特徴量で予測
python scripts/evaluation/predict_with_reward_function.py \
  --reward-model models/reward_function.pth \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --snapshot-date 2022-01-01 \
  --output predictions.csv
```

## 比較

| 項目 | 現在の実装（LSTM） | 正しいIRL実装 |
|------|-------------------|---------------|
| 訓練入力 | 時系列15時点 | 時系列15時点 |
| 訓練出力 | 継続確率 | 報酬関数 |
| 予測入力 | **時系列15時点** | **スナップショット1時点** |
| 予測出力 | 継続確率 | 報酬値→継続確率 |
| 母集団問題 | ❌ データ必須 | ✅ 特徴量あればOK |
| 解釈性 | ❌ ブラックボックス | ✅ 特徴量の寄与分析可能 |
| IRL的正しさ | ❌ | ✅ |

## 次のステップ

1. `RewardNetwork`クラスを実装
2. `train_irl_reward_function()`を実装
3. `compute_snapshot_features()`を実装
4. `predict_continuation_at_snapshot()`を実装
5. 全スライディングウィンドウで評価
