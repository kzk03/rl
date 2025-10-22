# IRL 設計変更の要約

## 🎯 核心的な変更点

### 従来のアプローチ（教師あり学習的 IRL）

```python
# データ構造
trajectory = {
    'developer': {...},
    'activity_history': [...],  # 過去の履歴（特徴量）
    'continued': True,  # ← 正解ラベル（教師信号）
}

# 訓練
predicted_prob = model(state, action)
loss = BCE(predicted_prob, trajectory['continued'])  # 正解ラベルを使用
```

**問題点:**

- 正解ラベルを明示的に使用 → 教師あり学習に近い
- 学習期間内の時間的変化を活用できない
- データ効率が悪い（学習期間 12 ヶ月 → 1 サンプル）

---

### 新しいアプローチ（特徴量ベース IRL）

```python
# データ構造
trajectory = {
    'developer': {...},
    'activity_history': [...],  # 過去の履歴

    # 将来の貢献を「状態特徴量」として含める（1次元のみ）
    'future_contribution': True,   # ← 特徴量（正解ラベルではない）

    # メタデータ（どの期間を見ているか）
    'future_window': {
        'start_months': 0,  # 0ヶ月後から
        'end_months': 1,    # 1ヶ月後まで
    },
    # 正解ラベルは含まない
}

# 訓練
# 状態の特徴量に将来の貢献が含まれている（1次元のみ）
state_features = [
    *past_features,  # 過去の履歴から計算（拡張IRL特徴）
    1.0,  # future_contribution (特徴量として組み込み、1次元のみ)
]

reward = model(state_features, action)

# 将来の貢献から目標報酬を自動生成（1次元なのでシンプル）
future_contribution = trajectory['future_contribution']
target_reward = 1.0 if future_contribution else 0.0

loss = MSE(reward, target_reward)  # 正解ラベルは使わない
```

**改善点:**

- ✅ 正解ラベルを使わない → より純粋な IRL
- ✅ 将来の貢献パターンを状態特徴量に組み込む
- ✅ 学習期間内の複数時点からサンプリング → データ効率向上
- ✅ 柔軟な学習方法（二値化、連続値、コントラスト学習など）

---

## 📊 具体例で理解する

### タイムライン

```
学習期間: 2019-01-01 ～ 2020-01-01 (12ヶ月)

サンプリング時点:
  t1 (2019-01-01)  t2 (2019-02-01)  ...  t12 (2019-12-01)
  |                |                      |
  +-- 過去6ヶ月の履歴                      +-- 過去6ヶ月の履歴
  +-- 将来の貢献:                           +-- 将来の貢献:
      1m後: ✓ (特徴量)                         1m後: ✗ (特徴量)
      2m後: ✓ (特徴量)                         2m後: ✓ (特徴量)
      3m後: ✗ (特徴量)                         3m後: ✗ (特徴量)
```

### データ生成の違い

**従来:**

```python
# スナップショット日1点のみ
snapshot_date = 2020-01-01

trajectory = {
    'activity_history': [過去6ヶ月の履歴],
    'continued': has_activity_at(2020-07-01),  # 6ヶ月後の貢献（正解ラベル）
}

# → 12ヶ月の学習期間から1サンプルのみ
```

**新設計:**

```python
# 学習期間内を1ヶ月間隔でサンプリング
trajectories = []

for t in [2019-01-01, 2019-02-01, ..., 2019-12-01]:  # 12時点
    trajectory = {
        'activity_history': [t-6ヶ月 ～ t の履歴],
        'temporal_contribution_features': {
            'contrib_1m': has_activity(t, t+1m),  # 特徴量
            'contrib_2m': has_activity(t+1m, t+2m),  # 特徴量
            'contrib_3m': has_activity(t+2m, t+3m),  # 特徴量
        },
    }
    trajectories.append(trajectory)

# → 12ヶ月の学習期間から12サンプル生成
```

---

## 🔧 実装の主要変更箇所

### 1. データ抽出関数

```python
def extract_temporal_trajectories_with_future_features(
    df, train_start, train_end,
    history_window_months=6,
    future_window_start_months=0,  # ← parserで指定
    future_window_end_months=1,    # ← parserで指定
    sampling_interval_months=1
):
    """
    重要: 将来の貢献を「正解ラベル」ではなく「状態特徴量」として抽出

    Args:
        future_window_start_months: 将来窓の開始（0なら現時点から）
        future_window_end_months: 将来窓の終了（1なら1ヶ月後まで）
    """
    # 学習期間内を1ヶ月間隔でサンプリング
    for sampling_point in sampling_points:
        for reviewer in reviewers:
            # 過去の履歴
            history = get_history(reviewer, sampling_point - 6m, sampling_point)

            # 将来の貢献（1次元のみ）
            future_start = sampling_point + pd.DateOffset(months=future_window_start_months)
            future_end = sampling_point + pd.DateOffset(months=future_window_end_months)
            future_contribution = has_contribution(reviewer, future_start, future_end)

            trajectory = {
                'activity_history': history,
                'future_contribution': future_contribution,  # ← 1次元の特徴量
                'future_window': {
                    'start_months': future_window_start_months,
                    'end_months': future_window_end_months,
                },
                # 'continued': ... ← 削除（正解ラベルは渡さない）
            }
```

### 2. 状態抽出関数（既存の拡張 IRL を活用）

```python
def extract_developer_state_with_future(self, trajectory):
    """
    状態ベクトルに将来の貢献を含める（1次元のみ追加）

    既存の EnhancedRetentionIRLSystem の状態抽出を活用
    """
    # 既存の拡張IRL実装で過去ベースの特徴量を取得
    from gerrit_retention.rl_prediction.enhanced_retention_irl_system import EnhancedRetentionIRLSystem

    developer = trajectory['developer']
    activity_history = trajectory['activity_history']
    context_date = trajectory['context_date']

    # 既存実装を使用（この部分は変更不要）
    past_state_features = EnhancedRetentionIRLSystem.extract_developer_state(
        developer, activity_history, context_date
    )  # N次元（既存の拡張IRLの次元）

    # 将来の貢献を特徴量として追加（1次元のみ）
    future_contribution = trajectory['future_contribution']
    future_feature = np.array([
        1.0 if future_contribution else 0.0
    ])  # 1次元

    # 結合（N+1次元）
    state_vector = np.concatenate([past_state_features, future_feature])
    return state_vector
```

### 3. モデル（既存の拡張 IRL ネットワークを活用）

```python
# 既存の拡張IRLネットワークをそのまま使用可能
from gerrit_retention.rl_prediction.enhanced_retention_irl_system import EnhancedRetentionIRLNetwork

class FeatureBasedIRLNetwork(nn.Module):
    """
    報酬のみを出力（継続確率の予測ヘッドは削除）

    既存の EnhancedRetentionIRLNetwork をベースにする
    """

    def __init__(self, state_dim, action_dim=5):  # state_dim = 拡張IRL次元 + 1
        super().__init__()
        # 既存の拡張IRL実装をベースにネットワークを構築
        # state_dim は自動的に拡張IRL次元 + 1次元になる

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # LSTM（時系列モード）
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)

        # 報酬予測ヘッドのみ
        self.reward_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        # state には拡張IRL特徴 + 将来の貢献（1次元）が含まれている
        state_enc = self.state_encoder(state)
        action_enc = self.action_encoder(action)
        combined = state_enc + action_enc

        lstm_out, _ = self.lstm(combined)
        final = lstm_out[:, -1, :]

        reward = self.reward_head(final)
        return reward  # 報酬のみ返す
```

### 4. 訓練ループ

```python
def train_irl_feature_based(self, trajectories, epochs=100):
    """正解ラベルを使わずに訓練"""

    for epoch in range(epochs):
        for trajectory in trajectories:
            # 状態を抽出（将来の貢献パターンを含む）
            state = self.extract_developer_state(trajectory)
            actions = self.extract_developer_actions(trajectory['activity_history'])

            # 報酬を予測
            predicted_reward = self.network(state, action)

            # 将来の貢献から目標報酬を生成（1次元なのでシンプル）
            future_contribution = trajectory['future_contribution']
            target_reward = 1.0 if future_contribution else 0.0

            # 損失計算（正解ラベルは使わない）
            loss = F.mse_loss(predicted_reward, torch.tensor([[target_reward]]))

            # 逆伝播
            loss.backward()
            self.optimizer.step()
```

---

## ⚠️ 重要な注意点

### データ分割: シンプルな時系列 Cutoff で OK

**方針:** 訓練と評価は時系列で cutoff するだけで十分

```python
# シンプルな時系列分割
train_period = [2019-01-01, 2020-01-01]
eval_period = [2020-01-01, 2021-01-01]  # cutoff で分離

# これで十分（複雑な対策は不要）
```

### 状態特徴量: 既存の拡張 IRL を活用

**方針:** 状態特徴量の抽出は既存の `EnhancedRetentionIRLSystem` を活用

```python
# 既存の拡張IRL実装を使用
from gerrit_retention.rl_prediction.enhanced_retention_irl_system import EnhancedRetentionIRLSystem

# 状態抽出は既存実装を使う
state_features = enhanced_irl.extract_developer_state(developer, activity_history, context_date)

# 将来の貢献を追加（1次元のみ）
future_feature = [
    1.0 if trajectory['future_contribution'] else 0.0
]

# 結合
state_vector = np.concatenate([state_features, future_feature])
```

### 推論時の扱い

```python
# 推論時は将来の貢献が未知 → ゼロ埋め
future_contribution = 0.0  # または False
```

---

## 📝 設定ファイル例

```yaml
# configs/irl_feature_based_config.yaml

irl_feature_based:
  # 学習期間
  training_period:
    start_date: "2019-01-01"
    end_date: "2020-01-01"

  # サンプリング設定
  sampling:
    interval_months: 1 # 1ヶ月ごとにサンプリング
    history_window_months: 6 # 各サンプルの履歴期間
    min_history_events: 3

  # 将来の貢献窓（parserで指定可能）
  future_window:
    start_months: 0 # 0ヶ月後から（デフォルト）
    end_months: 1 # 1ヶ月後まで（デフォルト）
    description: "将来の貢献を見る期間（状態特徴量として使用、1次元のみ）"

  # モデル設定
  model:
    # state_dim は自動計算: 拡張IRL次元 + 1次元
    use_enhanced_irl_features: true # 拡張IRL特徴量を使用
    action_dim: 5
    hidden_dim: 64
    lstm_hidden: 128
    sequence: true
    seq_len: 15

  # 訓練設定
  training:
    epochs: 30
    learning_rate: 0.001
    approach: "binary" # "binary", "continuous", "contrastive"

# コマンドライン引数での指定例
# python train_feature_based_irl.py \
#   --future-window-start 0 \   # 0ヶ月後から
#   --future-window-end 1 \     # 1ヶ月後まで
#   --epochs 30
```

---

## 🚀 実装手順（シンプル版）

### Step 1: データ抽出の変更

```bash
# 新しいデータ抽出関数を実装
# scripts/data_processing/extract_temporal_features.py

# 重要: 既存のデータ抽出をベースに、将来の貢献パターンを追加するだけ
```

### Step 2: 状態抽出のラッパー作成

```bash
# 既存の EnhancedRetentionIRLSystem をラップ
# src/gerrit_retention/rl_prediction/feature_based_irl_wrapper.py

# extract_developer_state_with_future() を追加
# → 既存の拡張IRL特徴 + 将来3次元を結合
```

### Step 3: 訓練スクリプトの作成

```bash
# 新しい訓練スクリプトを実装
# scripts/training/irl/train_feature_based_irl.py

# 既存の train_temporal_irl_sliding_window.py をベースに
# 将来の貢献パターンを状態特徴量に組み込むように変更
```

### Step 4: 評価（時系列 Cutoff で分割）

```bash
# シンプルな時系列分割で評価
# train: 2019-01-01 ~ 2020-01-01
# eval:  2020-01-01 ~ 2021-01-01
```

**実装のポイント:**

- 既存の拡張 IRL 実装を最大限活用
- データ抽出と状態結合の部分だけを変更
- 複雑な対策は不要（cutoff で十分）

---

## 🎓 理論的背景

### なぜこのアプローチが有効か？

1. **状態の豊かな表現:**
   - 過去の履歴だけでなく、将来の結果も状態の一部として扱うことで、より豊かな状態表現を学習
2. **自己教師あり学習:**
   - 明示的な教師ラベルではなく、データ内のパターン（将来の貢献）から学習
3. **IRL の本質に近い:**
   - 専門家（継続する開発者）の軌跡から報酬関数を学習するという、IRL の本来の目的に近い

### 従来の教師あり学習との違い

| 特徴             | 教師あり学習       | 特徴量ベース IRL             |
| ---------------- | ------------------ | ---------------------------- |
| **ラベルの扱い** | 明示的な正解ラベル | 特徴量の一部                 |
| **学習目標**     | ラベルを正しく予測 | 報酬関数を学習               |
| **柔軟性**       | 固定的             | 複数のアプローチ可能         |
| **データ効率**   | 低い               | 高い（複数時点サンプリング） |

---

## 📚 参考

- 詳細な実装: `docs/irl_temporal_window_redesign.md`
- 現在の IRL 実装: `src/gerrit_retention/rl_prediction/retention_irl_system.py`
- 時系列 IRL: `README_TEMPORAL_IRL.md`

---

---

## 🔬 実験例（将来窓を変えてモデルの変化を観察）

### 実験設計

```bash
# 実験1: 短期予測（0-1ヶ月後）
python scripts/training/irl/train_feature_based_irl.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --future-window-start 0 \
  --future-window-end 1 \
  --output outputs/irl_future_0_1m

# 実験2: 中期予測（1-3ヶ月後）
python scripts/training/irl/train_feature_based_irl.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --future-window-start 1 \
  --future-window-end 3 \
  --output outputs/irl_future_1_3m

# 実験3: 長期予測（3-6ヶ月後）
python scripts/training/irl/train_feature_based_irl.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --future-window-start 3 \
  --future-window-end 6 \
  --output outputs/irl_future_3_6m
```

### 期待される結果

| 実験       | 将来窓 | 期待される特徴                               |
| ---------- | ------ | -------------------------------------------- |
| **実験 1** | 0-1m   | 短期的なパターンを学習、即座の離脱を予測     |
| **実験 2** | 1-3m   | 中期的なパターンを学習、バランス型           |
| **実験 3** | 3-6m   | 長期的なパターンを学習、持続的な貢献者を識別 |

### 比較分析

```bash
# 結果を比較
python scripts/analysis/compare_future_window_results.py \
  --results outputs/irl_future_*/evaluation_results.json \
  --output outputs/future_window_comparison.png
```

**分析観点:**

- どの将来窓が最も予測精度が高いか？
- 短期 vs 長期で学習される特徴はどう違うか？
- 実用的にはどの窓が最適か？

---

**作成日:** 2025-10-21  
**ステータス:** 提案
