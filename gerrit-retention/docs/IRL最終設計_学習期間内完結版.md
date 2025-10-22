# IRL 最終設計: 学習期間内完結版

## 🎯 核心的なポイント

### ✅ やりたいこと

**続いた人と続かなかった人を区別する報酬関数を学習**

- LSTM で過去の時系列パターンを学習
- 将来の貢献（1m 後など）をラベルとして使用
- 学習期間内ですべて完結
- 評価期間とは cutoff で分離

---

## 📐 設計の全体像

### タイムライン

```
学習期間: 2019-01-01 ～ 2020-01-01 (12ヶ月)
|------------------------------------------------|
[train_start]                          [train_end]

サンプリング可能範囲（future_window=0-1m の場合）:
|-----------------------------------|
     ↑                           ↑
  2019-07-01                  2019-12-01
  (最初)                       (最後)
                                  ↓
                                  +-- 将来窓 (1m)
                                  [2019-12-01 ～ 2020-01-01]
                                                   ↑
                                                   学習期間内で完結！

評価期間: 2020-01-01 ～ 2021-01-01
|------------------------------------------------|
     ↑
     cutoff で完全分離
```

### 重要な制約

```python
# サンプリング時点の範囲
min_sampling_date = train_start + history_window_months
max_sampling_date = train_end - future_window_end_months  # ← これが重要！

# 例: train_end = 2020-01-01, future_window = 1m
max_sampling_date = 2020-01-01 - 1m = 2019-12-01

# 最後のサンプリング時点での将来窓
future_end = 2019-12-01 + 1m = 2020-01-01  # ← train_end に一致
```

---

## 🔧 具体的な設計

### 1. データ構造

```python
trajectory = {
    'developer': {...},
    'activity_history': [...],  # 過去の活動履歴（LSTM入力）
    'context_date': sampling_point,

    # 将来の貢献は「ラベル」として格納（状態特徴量ではない）
    'future_contribution': True,  # ← 学習のターゲット

    # メタデータ
    'future_window': {
        'start_months': 0,
        'end_months': 1,
    },
}
```

**重要な違い:**

- ❌ 将来の貢献を状態特徴量に含める → 因果関係が逆転
- ✅ 将来の貢献を学習のターゲットにする → 正しい因果関係

### 2. データ抽出（学習期間内で完結）

```python
def extract_temporal_trajectories_within_training_period(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    history_window_months: int = 6,
    future_window_start_months: int = 0,  # parserで指定
    future_window_end_months: int = 1,    # parserで指定
    sampling_interval_months: int = 1
):
    """
    学習期間内で完結する軌跡を抽出

    重要:
    - 将来の貢献は「ラベル」として使用（状態には含めない）
    - すべてのデータが学習期間内で完結
    """
    trajectories = []

    # サンプリング可能範囲を計算
    min_sampling_date = train_start + pd.DateOffset(months=history_window_months)
    max_sampling_date = train_end - pd.DateOffset(months=future_window_end_months)
    # ↑ これにより学習期間内で完結

    # サンプリング時点を生成
    sampling_points = []
    current = min_sampling_date

    while current <= max_sampling_date:
        sampling_points.append(current)
        current += pd.DateOffset(months=sampling_interval_months)

    logger.info(f"サンプリング可能範囲: {min_sampling_date} ～ {max_sampling_date}")
    logger.info(f"サンプリング時点数: {len(sampling_points)}")

    for sampling_point in sampling_points:
        for reviewer in df['reviewer_email'].unique():
            # 過去の履歴（LSTM入力）
            history_start = sampling_point - pd.DateOffset(months=history_window_months)
            history_end = sampling_point

            history_df = df[
                (df['reviewer_email'] == reviewer) &
                (df['request_time'] >= history_start) &
                (df['request_time'] < history_end)
            ]

            if len(history_df) < 3:  # 最小イベント数
                continue

            # 活動履歴を構築
            activity_history = []
            for _, row in history_df.iterrows():
                activity = {
                    'timestamp': row['request_time'],
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                }
                activity_history.append(activity)

            # 将来の貢献を計算（ラベルとして使用）
            future_start = sampling_point + pd.DateOffset(months=future_window_start_months)
            future_end = sampling_point + pd.DateOffset(months=future_window_end_months)

            # 学習期間内で完結していることを確認
            assert future_end <= train_end, \
                f"将来窓が学習期間を超えています: {future_end} > {train_end}"

            future_df = df[
                (df['reviewer_email'] == reviewer) &
                (df['request_time'] >= future_start) &
                (df['request_time'] < future_end)
            ]

            future_contribution = len(future_df) > 0

            # 開発者情報
            developer_info = {
                'developer_id': reviewer,
                'first_seen': history_df['request_time'].min(),
                'changes_reviewed': len(history_df),
            }

            # 軌跡を作成
            trajectory = {
                'developer': developer_info,
                'activity_history': activity_history,
                'context_date': sampling_point,

                # 将来の貢献はラベルとして格納
                'future_contribution': future_contribution,

                # メタデータ
                'future_window': {
                    'start_months': future_window_start_months,
                    'end_months': future_window_end_months,
                    'start_date': future_start,
                    'end_date': future_end,
                },
                'history_window': {
                    'start_date': history_start,
                    'end_date': history_end,
                },
            }

            trajectories.append(trajectory)

    # 統計情報
    positive_count = sum(1 for t in trajectories if t['future_contribution'])
    positive_rate = positive_count / len(trajectories) if trajectories else 0

    logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル")
    logger.info(f"  継続率: {positive_rate:.1%} ({positive_count}/{len(trajectories)})")
    logger.info(f"  すべてのデータが学習期間内で完結: ✅")

    return trajectories
```

### 3. 状態抽出（将来の情報を含めない）

```python
def extract_developer_state(self, trajectory):
    """
    状態ベクトルを抽出（将来の情報は含めない）

    重要: 将来の貢献は状態特徴量には含めず、
    学習のターゲットとして使用する
    """
    # 既存の拡張IRL実装で過去ベースの特徴量を取得
    from gerrit_retention.rl_prediction.enhanced_retention_irl_system import (
        EnhancedRetentionIRLSystem
    )

    developer = trajectory['developer']
    activity_history = trajectory['activity_history']
    context_date = trajectory['context_date']

    # 既存実装を使用（過去の情報のみ）
    state_features = EnhancedRetentionIRLSystem.extract_developer_state(
        developer, activity_history, context_date
    )  # N次元（既存の拡張IRLの次元）

    # ここで future_contribution は追加しない！
    # → 将来の情報を状態に含めない

    return state_features
```

### 4. 訓練（続いた人と続かなかった人を区別）

```python
def train_irl_to_distinguish_continued(
    self,
    trajectories: List[Dict[str, Any]],
    epochs: int = 30
) -> Dict[str, Any]:
    """
    続いた人と続かなかった人を区別する報酬関数を学習

    重要:
    - 状態には将来の情報を含めない
    - 将来の貢献をターゲット（ラベル）として使用
    - LSTMで時系列パターンを学習
    """
    logger.info(f"IRL訓練開始: {len(trajectories)}軌跡, {epochs}エポック")
    logger.info("目標: 続いた人と続かなかった人を区別する報酬関数を学習")

    training_losses = []

    # 継続率を確認
    positive_count = sum(1 for t in trajectories if t['future_contribution'])
    positive_rate = positive_count / len(trajectories)
    logger.info(f"継続率: {positive_rate:.1%}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for trajectory in trajectories:
            try:
                # 状態を抽出（将来の情報は含まない）
                state = self.extract_developer_state(trajectory)
                actions = self.extract_developer_actions(
                    trajectory['activity_history'],
                    trajectory['context_date']
                )

                if not actions:
                    continue

                # 時系列処理（LSTM）
                if self.sequence:
                    # シーケンス構築
                    if len(actions) < self.seq_len:
                        padded_actions = [actions[0]] * (self.seq_len - len(actions)) + actions
                    else:
                        padded_actions = actions[-self.seq_len:]

                    state_seq = torch.stack([
                        self.state_to_tensor(state) for _ in range(self.seq_len)
                    ]).unsqueeze(0)
                    action_seq = torch.stack([
                        self.action_to_tensor(action) for action in padded_actions
                    ]).unsqueeze(0)

                    # LSTMで時系列パターンを学習
                    predicted_reward = self.network(state_seq, action_seq)
                else:
                    state_tensor = self.state_to_tensor(state).unsqueeze(0)
                    action_tensor = self.action_to_tensor(actions[-1]).unsqueeze(0)
                    predicted_reward = self.network(state_tensor, action_tensor)

                # 将来の貢献から目標報酬を生成
                future_contribution = trajectory['future_contribution']

                # 続いた人 → 報酬 1.0
                # 続かなかった人 → 報酬 0.0
                target_reward = torch.tensor(
                    [[1.0 if future_contribution else 0.0]],
                    device=self.device
                )

                # 損失計算
                loss = F.mse_loss(predicted_reward, target_reward)

                # 逆伝播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            except Exception as e:
                logger.warning(f"軌跡処理エラー: {e}")
                continue

        if batch_count > 0:
            epoch_loss /= batch_count
            training_losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: 損失={epoch_loss:.4f}")

    return {
        'losses': training_losses,
        'final_loss': training_losses[-1] if training_losses else 0
    }
```

### 5. 評価（時系列 Cutoff で分離）

```python
# 訓練データ（学習期間内で完結）
train_trajectories = extract_temporal_trajectories_within_training_period(
    df=df,
    train_start=pd.Timestamp('2019-01-01'),
    train_end=pd.Timestamp('2020-01-01'),
    history_window_months=6,
    future_window_start_months=0,
    future_window_end_months=1,
)

# 評価データ（評価期間で抽出）
eval_trajectories = extract_temporal_trajectories_within_training_period(
    df=df,
    train_start=pd.Timestamp('2020-01-01'),  # cutoff
    train_end=pd.Timestamp('2021-01-01'),
    history_window_months=6,
    future_window_start_months=0,
    future_window_end_months=1,
)

# データリークなし ✅
```

---

## 🎓 重要な理論的ポイント

### 1. LSTM で時系列パターンを学習

```python
# 過去の活動履歴（時系列）
activity_history = [
    action_t0,   # 6ヶ月前
    action_t1,   # 5.5ヶ月前
    ...
    action_t15,  # 現在
]

# LSTMで処理
state_seq = LSTM(activity_history)
# ↑ 時系列的な変化を捉えられる

# 続いた人と続かなかった人の違いを学習
# - 続いた人: 活動が増加傾向、規則的なパターン
# - 続かなかった人: 活動が減少傾向、不規則なパターン
```

### 2. 因果関係が正しい

```python
# ❌ 間違い: 将来を見て過去を評価
state = [*past_features, future_contribution]  # 将来の情報を含む
reward = model(state)

# ✅ 正しい: 過去から将来を予測
state = [*past_features]  # 過去の情報のみ
reward = model(state)
target = future_contribution  # 将来の貢献をターゲットに
loss = MSE(reward, target)
```

### 3. 学習期間内で完結

```python
# すべてのサンプルについて
for trajectory in trajectories:
    history_start >= train_start  # ✅
    history_end <= train_end      # ✅
    future_start >= train_start   # ✅
    future_end <= train_end       # ✅ 学習期間内
```

### 4. 続いた人と続かなかった人を区別

```python
# 続いた人の軌跡 → 高い報酬を学習
if future_contribution:
    target_reward = 1.0
    # モデルは「続く人の過去パターン」を高報酬と学習

# 続かなかった人の軌跡 → 低い報酬を学習
else:
    target_reward = 0.0
    # モデルは「続かない人の過去パターン」を低報酬と学習
```

---

## 📝 設定ファイル例

```yaml
# configs/irl_within_training_period.yaml

irl_training:
  # 学習期間
  training_period:
    start_date: "2019-01-01"
    end_date: "2020-01-01"

  # 評価期間（cutoffで分離）
  evaluation_period:
    start_date: "2020-01-01"
    end_date: "2021-01-01"

  # サンプリング設定
  sampling:
    interval_months: 1
    history_window_months: 6
    min_history_events: 3

  # 将来窓（parserで指定可能）
  future_window:
    start_months: 0 # 0ヶ月後から
    end_months: 1 # 1ヶ月後まで
    description: "将来の貢献を見る期間（学習のターゲット、状態には含めない）"

  # モデル設定
  model:
    use_enhanced_irl_features: true
    state_dim: auto # 拡張IRL次元（将来の情報は含まない）
    action_dim: 5
    hidden_dim: 64
    lstm_hidden: 128
    sequence: true
    seq_len: 15

  # 訓練設定
  training:
    epochs: 30
    learning_rate: 0.001
```

---

## 🚀 実行例

### コマンドライン引数

```bash
# 基本的な実行
python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --eval-start 2020-01-01 \
  --eval-end 2021-01-01 \
  --future-window-start 0 \
  --future-window-end 1 \
  --history-window 6 \
  --epochs 30 \
  --output outputs/irl_0_1m

# 実験: 将来窓を変更
python scripts/training/irl/train_irl_within_training_period.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2019-01-01 \
  --train-end 2020-01-01 \
  --eval-start 2020-01-01 \
  --eval-end 2021-01-01 \
  --future-window-start 1 \
  --future-window-end 3 \
  --history-window 6 \
  --epochs 30 \
  --output outputs/irl_1_3m
```

### 期待される出力

```
INFO: サンプリング可能範囲: 2019-07-01 ～ 2019-12-01
INFO: サンプリング時点数: 6
INFO: 軌跡抽出完了: 5432サンプル
INFO:   継続率: 15.3% (831/5432)
INFO:   すべてのデータが学習期間内で完結: ✅
INFO: IRL訓練開始: 5432軌跡, 30エポック
INFO: 目標: 続いた人と続かなかった人を区別する報酬関数を学習
INFO: Epoch 10/30: 損失=0.1234
INFO: Epoch 20/30: 損失=0.0987
INFO: Epoch 30/30: 損失=0.0823
INFO: 訓練完了
INFO: 評価開始...
INFO: AUC-ROC: 0.852
INFO: AUC-PR: 0.789
INFO: F1: 0.654
```

---

## ✅ チェックリスト

実装する際の確認事項：

- [ ] 将来の貢献を状態特徴量に含めていない
- [ ] 将来の貢献を学習のターゲット（ラベル）として使用
- [ ] サンプリング時点の範囲が正しく制限されている
- [ ] すべてのデータが学習期間内で完結
- [ ] 訓練期間と評価期間が cutoff で分離
- [ ] LSTM で時系列パターンを処理
- [ ] 続いた人 → 高報酬、続かなかった人 → 低報酬 を学習

---

## 🎯 まとめ

### この設計ができること

✅ **続いた人と続かなかった人の過去パターンを区別**

- LSTM で時系列的な変化を学習
- 続いた人: 高い報酬を学習
- 続かなかった人: 低い報酬を学習

✅ **学習期間内で完結**

- すべてのデータが学習期間内
- 評価期間とは cutoff で完全分離
- データリークなし

✅ **因果関係が正しい**

- 過去の情報のみから将来を予測
- 将来の情報は状態に含めない

✅ **実験の自由度**

- parser で将来窓を指定
- 短期・中期・長期の予測を比較可能

### できないこと

❌ 複数の将来窓を同時に学習

- 現在は 1 つの将来窓のみ
- 複数の期間を見たい場合は別々に実験

❌ 推論時に将来の貢献を予測

- 訓練時は将来の貢献をラベルとして使用
- 推論時は過去のパターンから報酬を計算のみ

---

**作成日:** 2025-10-21  
**ステータス:** 最終版  
**重要な変更:** 将来の貢献を状態特徴量ではなく、学習のターゲットとして使用
