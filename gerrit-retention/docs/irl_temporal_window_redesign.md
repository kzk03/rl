# IRL 時系列ウィンドウ設計変更案

## 📋 概要

本ドキュメントでは、逆強化学習（IRL）による長期貢献者予測の設計変更について説明します。

### 変更の目的

現状の「n ヶ月後の単一時点での貢献予測」から、「特定期間内（0-2 ヶ月、3-5 ヶ月など）の貢献予測」へ変更し、より実用的な予測を実現します。

---

## 🔄 現状の設計

### 現在のアプローチ

```
タイムライン:
|-------- 学習期間 --------|------ 予測 ------|
[history_start]    [snapshot_date]    [target_end]
                          |                |
                          |                +-- target_months 後
                          +-- 基準日
```

**特徴:**

- スナップショット日（基準日）を固定
- 学習期間: `snapshot_date - history_months` ～ `snapshot_date`
- 予測時点: `snapshot_date + target_months`（単一時点）
- ラベル: 予測時点に 1 件でも貢献があれば `continued=True`

### 現在の実装（例）

```python
def extract_trajectories_with_window(df, snapshot_date, history_months, target_months):
    """
    例: snapshot_date=2020-01-01, history=6m, target=3m

    学習期間: [2019-07-01, 2020-01-01)  ← 過去6ヶ月
    予測期間: [2020-01-01, 2020-04-01)  ← 未来3ヶ月（単一期間）

    ラベル: 予測期間中に1件でも活動があれば continued=True
    """
    history_start = snapshot_date - pd.DateOffset(months=history_months)
    target_end = snapshot_date + pd.DateOffset(months=target_months)

    history_df = df[(df[date_col] >= history_start) &
                    (df[date_col] < snapshot_date)]
    target_df = df[(df[date_col] >= snapshot_date) &
                   (df[date_col] < target_end)]

    # レビュアーごとの継続判定
    for reviewer in reviewers:
        continued = len(target_df[target_df[reviewer_col] == reviewer]) > 0
```

### 現状の問題点

1. **単一時点の予測**: n ヶ月後という 1 つの期間でしか予測できない
2. **固定スナップショット**: 学習期間内の時間的な変化を活用できない
3. **期間の粒度**: 短期・中期・長期など、複数の時間スケールでの予測が困難
4. **学習データの非効率**: 学習期間内の各時点のデータを 1 つのラベルにまとめている

---

## 🎯 変更後の設計

### 重要な設計方針の転換

**従来（教師あり学習的 IRL）:**

```python
trajectory = {
    'developer': {...},
    'activity_history': [...],  # 特徴量
    'continued': True,  # ← 正解ラベル（損失計算に使用）
}

# 訓練時
loss = BCE(predicted_continuation, target_continuation)
# target_continuation は trajectory['continued'] から来る
```

**新設計（特徴量ベース IRL）:**

```python
trajectory = {
    'developer': {...},
    'activity_history': [...],
    # 状態の特徴量として「nヶ月後の貢献有無」を含める
    'temporal_contribution_features': {
        '1m_later': True,   # ← 特徴量の一部
        '2m_later': False,
        '3m_later': True,
    },
    # 正解ラベルは渡さない
}

# 訓練時
# 軌跡のパターン（特徴量の組み合わせ）から報酬関数を学習
# 明示的な教師ラベルは使わない
```

### 新しいアプローチ: 学習期間内動的ラベリング

```
タイムライン（1年間の学習期間の例）:
|-------------- 学習期間全体（12ヶ月） ---------------|
[train_start]                             [train_end]
     |      |      |      |      |      |      |
     t1     t2     t3     t4     t5     t6     t7   ← サンプリング時点
     |                                          |
     |                                          |
     +-- t1時点の状態特徴量:                    +-- t7時点の状態特徴量:
         - 過去の活動履歴                           - 過去の活動履歴
         - 1m後貢献: ✓ (特徴量)                     - 1m後貢献: ✗ (特徴量)
         - 2m後貢献: ✓ (特徴量)                     - 2m後貢献: ✓ (特徴量)
         - 3m後貢献: ✗ (特徴量)                     - 3m後貢献: ✗ (特徴量)

         ※ 正解ラベルは渡さない
```

### 主要な変更点

#### 1. 期間ベースのラベル定義

**従来:**

```python
# 単一時点でのラベル
label = has_activity_at(snapshot_date + n_months)
```

**変更後:**

```python
# 期間ベースのラベル
label_0_2m = has_activity_in_period(t, t + 2_months)
label_3_5m = has_activity_in_period(t + 3_months, t + 5_months)
label_6_8m = has_activity_in_period(t + 6_months, t + 8_months)
```

#### 2. 学習期間内の動的サンプリング

**従来:** スナップショット日 1 点のみ

**変更後:** 学習期間内の複数時点でサンプリング

```python
# 学習期間: 2019-01-01 ～ 2020-01-01 (12ヶ月)
# サンプリング間隔: 1ヶ月

sampling_points = [
    2019-01-01,  # t1
    2019-02-01,  # t2
    2019-03-01,  # t3
    ...
    2019-12-01   # t12
]

# 各時点で複数の予測期間のラベルを生成
for t in sampling_points:
    for period_def in prediction_periods:
        label = generate_label(t, period_def)
```

#### 3. 将来貢献パターンを特徴量として組み込む

**定義例:**

```yaml
# 状態特徴量に含める将来の貢献パターン
temporal_contribution_features:
  - name: "contrib_1m_later"
    offset_start_months: 0
    offset_end_months: 1
    description: "0-1ヶ月後の貢献有無（二値）"
    feature_type: "binary"

  - name: "contrib_2m_later"
    offset_start_months: 1
    offset_end_months: 2
    description: "1-2ヶ月後の貢献有無（二値）"
    feature_type: "binary"

  - name: "contrib_3m_later"
    offset_start_months: 2
    offset_end_months: 3
    description: "2-3ヶ月後の貢献有無（二値）"
    feature_type: "binary"
# これらは状態特徴量の一部として扱われ、
# 正解ラベルとしては使用しない
```

### 新しいデータ構造

#### 軌跡データ（特徴量ベース）

```python
trajectory = {
    'developer': developer_info,
    'activity_history': activity_history,  # サンプリング時点までの履歴
    'context_date': sampling_point,        # サンプリング時点

    # 将来の貢献パターンを「状態特徴量」として含める
    'temporal_contribution_features': {
        'contrib_1m_later': True,   # 0-1ヶ月後に貢献あり（特徴量）
        'contrib_2m_later': False,  # 1-2ヶ月後に貢献なし（特徴量）
        'contrib_3m_later': True,   # 2-3ヶ月後に貢献あり（特徴量）
    },

    # メタデータ
    'sampling_point': sampling_point,
    'history_window_months': 6,    # このサンプルの学習に使った期間

    # 正解ラベルは含めない（教師なし学習的アプローチ）
}
```

**重要な違い:**

- 従来: `'labels'` として正解ラベルを格納 → 損失計算に使用
- 新設計: `'temporal_contribution_features'` として状態特徴量に格納 → モデルへの入力に使用

---

## 💻 実装案

### 1. データ抽出関数の変更

```python
def extract_temporal_trajectories_with_future_features(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    history_window_months: int,
    temporal_contribution_periods: List[Dict[str, Any]],
    sampling_interval_months: int = 1,
    min_history_events: int = 1
) -> List[Dict[str, Any]]:
    """
    学習期間内の複数時点から、将来の貢献パターンを特徴量として持つ軌跡を抽出

    重要: この関数は将来の貢献情報を「正解ラベル」ではなく「状態特徴量」として扱います

    Args:
        df: レビューデータ
        train_start: 学習期間の開始日
        train_end: 学習期間の終了日
        history_window_months: 各サンプリング時点で使用する履歴の長さ（ヶ月）
        temporal_contribution_periods: 特徴量として含める将来の貢献期間の定義
            [{'name': 'contrib_1m', 'start_months': 0, 'end_months': 1}, ...]
        sampling_interval_months: サンプリング間隔（ヶ月）
        min_history_events: サンプルに必要な最小イベント数

    Returns:
        List[Dict]: 軌跡データのリスト
            各軌跡には 'temporal_contribution_features' が含まれる（正解ラベルではない）
    """
    logger.info(f"時系列軌跡を抽出中: {train_start} ～ {train_end}")
    logger.info(f"  履歴ウィンドウ: {history_window_months}ヶ月")
    logger.info(f"  サンプリング間隔: {sampling_interval_months}ヶ月")
    logger.info(f"  将来貢献特徴量: {len(temporal_contribution_periods)}種類")
    logger.info(f"  ※ これらは正解ラベルではなく状態特徴量として使用")

    trajectories = []
    reviewer_col = 'reviewer_email'
    date_col = 'request_time'

    # データの日付カラムを変換
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # サンプリング時点を生成
    sampling_points = []
    current = train_start

    # 最大特徴量期間を計算（将来の貢献データが必要な最大期間）
    max_feature_months = max(
        period['end_months'] for period in temporal_contribution_periods
    )

    # サンプリング時点は、将来の貢献データを確保できる範囲内で設定
    max_sampling_date = train_end - pd.DateOffset(months=max_feature_months)

    while current <= max_sampling_date:
        # 履歴を十分に確保できるか確認
        history_start = current - pd.DateOffset(months=history_window_months)
        if history_start >= df[date_col].min():
            sampling_points.append(current)
        current += pd.DateOffset(months=sampling_interval_months)

    logger.info(f"サンプリング時点: {len(sampling_points)}点")

    # 全レビュアーを取得
    reviewers = df[reviewer_col].unique()

    for sampling_point in sampling_points:
        # 履歴期間を定義
        history_start = sampling_point - pd.DateOffset(months=history_window_months)
        history_end = sampling_point

        for reviewer in reviewers:
            # このレビュアーの履歴データ
            history_df = df[
                (df[reviewer_col] == reviewer) &
                (df[date_col] >= history_start) &
                (df[date_col] < history_end)
            ]

            # 最小イベント数を満たさない場合はスキップ
            if len(history_df) < min_history_events:
                continue

            # 将来の貢献パターンを状態特徴量として計算
            temporal_contrib_features = {}
            temporal_contrib_metadata = {}

            for period in temporal_contribution_periods:
                feature_name = period['name']
                start_months = period['start_months']
                end_months = period['end_months']

                # 将来の期間範囲
                future_start = sampling_point + pd.DateOffset(months=start_months)
                future_end = sampling_point + pd.DateOffset(months=end_months)

                # この期間内の活動を確認
                future_df = df[
                    (df[reviewer_col] == reviewer) &
                    (df[date_col] >= future_start) &
                    (df[date_col] < future_end)
                ]

                # 特徴量: 期間内に1件でも活動があれば True
                # ※ これは正解ラベルではなく、状態の特徴量として扱う
                has_activity = len(future_df) > 0
                temporal_contrib_features[feature_name] = has_activity

                # メタデータ
                temporal_contrib_metadata[feature_name] = {
                    'future_start': future_start,
                    'future_end': future_end,
                    'activity_count': len(future_df),
                }

            # 活動履歴を構築
            activity_history = []
            for _, row in history_df.iterrows():
                activity = {
                    'timestamp': row[date_col],
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                }
                activity_history.append(activity)

            # 開発者情報
            developer_info = {
                'developer_id': reviewer,
                'first_seen': history_df[date_col].min(),
                'changes_reviewed': len(history_df),
                'projects': history_df['project'].unique().tolist() if 'project' in history_df.columns else []
            }

            # 軌跡を作成（特徴量ベース）
            trajectory = {
                'developer': developer_info,
                'activity_history': activity_history,
                'context_date': sampling_point,

                # 将来の貢献パターンを「状態特徴量」として格納
                # ※ これは正解ラベルではない
                'temporal_contribution_features': temporal_contrib_features,
                'temporal_contribution_metadata': temporal_contrib_metadata,

                # メタデータ
                'sampling_point': sampling_point,
                'history_start': history_start,
                'history_end': history_end,
                'history_window_months': history_window_months,
                'history_event_count': len(history_df),
            }

            trajectories.append(trajectory)

    logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル")

    # 各特徴量の分布を表示
    for period in temporal_contribution_periods:
        feature_name = period['name']
        positive_count = sum(
            1 for t in trajectories
            if t['temporal_contribution_features'][feature_name]
        )
        positive_rate = positive_count / len(trajectories) if trajectories else 0
        logger.info(f"  {feature_name}: True率 {positive_rate:.1%} "
                   f"({positive_count}/{len(trajectories)}) ※特徴量として")

    return trajectories
```

### 2. 状態特徴量抽出の変更（最重要）

**従来の状態抽出:**

```python
def extract_developer_state(self, developer, activity_history, context_date):
    """従来: 過去の履歴のみから特徴量を抽出"""
    state = {
        'experience_days': ...,
        'total_reviews': ...,
        # 過去の情報のみ
    }
    return state
```

**新しい状態抽出（将来の貢献パターンを含む）:**

```python
def extract_developer_state(self, trajectory):
    """
    新設計: 過去の履歴 + 将来の貢献パターンを特徴量として抽出

    Args:
        trajectory: 軌跡データ（temporal_contribution_features を含む）

    Returns:
        状態ベクトル（将来の貢献パターンを含む）
    """
    developer = trajectory['developer']
    activity_history = trajectory['activity_history']
    context_date = trajectory['context_date']

    # 従来の過去ベースの特徴量
    state_features = []

    # 1. 過去の活動から計算される特徴量
    experience_days = (context_date - developer['first_seen']).days
    state_features.append(experience_days / 365.0)
    state_features.append(developer.get('changes_reviewed', 0) / 100.0)
    state_features.append(len(activity_history) / 100.0)
    # ... 他の過去ベース特徴量

    # 2. 新規: 将来の貢献パターンを特徴量として追加
    # ※ これは正解ラベルではなく、状態を記述する特徴量の一部
    temporal_contrib_feats = trajectory.get('temporal_contribution_features', {})

    # 各期間の貢献有無を二値特徴量として追加
    for feature_name in sorted(temporal_contrib_feats.keys()):
        has_contribution = temporal_contrib_feats[feature_name]
        state_features.append(1.0 if has_contribution else 0.0)

    return np.array(state_features, dtype=np.float32)
```

**重要な変更点:**

- 状態ベクトルの次元が増加（将来の貢献パターン分だけ）
- 例: 従来 10 次元 → 新設計 13 次元（3 つの将来貢献パターンを追加した場合）
- これらは「状態を記述する特徴量」であり、「予測する対象（正解ラベル）」ではない

### 3. IRL ネットワークの変更

**現状:** 教師あり学習的に継続確率を出力

```python
class RetentionIRLNetwork(nn.Module):
    def forward(self, state, action):
        # ...
        reward = self.reward_head(combined)
        continuation = torch.sigmoid(self.continuation_head(combined))
        return reward, continuation  # ← 教師ラベルと比較するための出力
```

**変更案:** 報酬のみを出力（継続確率の予測は行わない）

```python
class FeatureBasedIRLNetwork(nn.Module):
    """
    特徴量ベースIRLネットワーク

    重要: 将来の貢献パターンは状態特徴量に含まれており、
    モデルは報酬関数のみを学習する（継続確率の予測ヘッドは削除）
    """
    def __init__(
        self,
        state_dim: int,  # ← 将来の貢献パターン分だけ増加
        action_dim: int,
        hidden_dim: int = 64,
        lstm_hidden: int = 128,
        sequence: bool = False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence = sequence

        # 状態・行動エンコーダー
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # LSTM（時系列モード）
        if sequence:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden,
                batch_first=True
            )
            final_dim = lstm_hidden
        else:
            final_dim = hidden_dim

        # 報酬予測ヘッドのみ（継続予測ヘッドは不要）
        self.reward_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        Args:
            state: [batch, (seq_len,) state_dim]
                   ※ state_dim には将来の貢献パターンが含まれる
            action: [batch, (seq_len,) action_dim]

        Returns:
            reward: [batch, 1] - 状態-行動ペアの報酬
        """
        state_enc = self.state_encoder(state)
        action_enc = self.action_encoder(action)

        combined = state_enc + action_enc

        if self.sequence:
            # LSTM処理
            lstm_out, _ = self.lstm(combined)
            # 最終ステップのみ使用
            if lstm_out.dim() == 3:
                final = lstm_out[:, -1, :]
            else:
                final = lstm_out
        else:
            final = combined

        # 報酬予測のみ
        reward = self.reward_head(final)

        return reward
```

### 4. 訓練ループの変更（重要）

**従来の教師あり学習的アプローチ:**

```python
# ❌ 従来: 正解ラベルを使用
for trajectory in trajectories:
    predicted_continuation = model(state, action)
    true_label = trajectory['continued']  # 正解ラベル
    loss = BCE(predicted_continuation, true_label)
```

**新設計: 特徴量ベースの自己教師あり学習:**

```python
def train_irl_feature_based(
    self,
    trajectories: List[Dict[str, Any]],
    epochs: int = 100
) -> Dict[str, Any]:
    """
    特徴量ベースIRL訓練

    重要: 正解ラベルは使用せず、将来の貢献パターン（特徴量）から
    報酬関数を学習する

    Args:
        trajectories: 軌跡データ
            各軌跡には 'temporal_contribution_features' が含まれる
        epochs: エポック数

    Returns:
        訓練結果
    """
    logger.info(f"特徴量ベースIRL訓練開始: {len(trajectories)}軌跡, {epochs}エポック")
    logger.info("※ 正解ラベルは使用せず、特徴量のパターンから学習")

    training_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for trajectory in trajectories:
            try:
                # 状態を抽出（将来の貢献パターンを含む）
                state = self.extract_developer_state(trajectory)
                actions = self.extract_developer_actions(
                    trajectory['activity_history'],
                    trajectory['context_date']
                )

                if not actions:
                    continue

                # 時系列処理
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

                    predicted_reward = self.network(state_seq, action_seq)
                else:
                    # 非時系列モード
                    state_tensor = self.state_to_tensor(state).unsqueeze(0)
                    action_tensor = self.action_to_tensor(actions[-1]).unsqueeze(0)
                    predicted_reward = self.network(state_tensor, action_tensor)

                # 損失計算（複数のアプローチが可能）
                # アプローチ1: MaxEnt IRL風 - 将来貢献がある状態の報酬を最大化
                temporal_features = trajectory['temporal_contribution_features']

                # 将来の貢献パターンから「望ましさ」を計算
                # 例: いずれかの期間で貢献があれば望ましい
                has_any_contribution = any(temporal_features.values())

                if has_any_contribution:
                    # 貢献がある状態 → 報酬を高くしたい
                    target_reward = torch.tensor([[1.0]], device=self.device)
                else:
                    # 貢献がない状態 → 報酬を低くしたい
                    target_reward = torch.tensor([[0.0]], device=self.device)

                # MSE損失
                loss = F.mse_loss(predicted_reward, target_reward)

                # 逆伝播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 損失を記録
                epoch_loss += loss.item()
                batch_count += 1

            except Exception as e:
                logger.warning(f"軌跡処理エラー: {e}")
                continue

        if batch_count > 0:
            # 平均損失を計算
            epoch_loss /= batch_count
            training_losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: 損失={epoch_loss:.4f}")

    return {
        'losses': training_losses,
        'final_loss': training_losses[-1] if training_losses else 0
    }
```

**重要なポイント:**

1. **正解ラベルは使わない**: `trajectory['labels']` の代わりに `trajectory['temporal_contribution_features']` を使用
2. **特徴量から目標を生成**: 将来の貢献パターンから報酬の目標値を自動的に設定
3. **シンプルな損失**: MSE 損失のみ（継続確率の予測ヘッドがないため）

**代替アプローチ（より洗練された方法）:**

```python
# アプローチ2: 継続的な報酬スケール
# 将来の貢献回数に応じて報酬を段階的に設定
def calculate_target_reward_from_features(temporal_features):
    """
    将来の貢献パターンから目標報酬を計算
    """
    contribution_count = sum(1 for v in temporal_features.values() if v)
    total_periods = len(temporal_features)

    # 貢献期間の割合を報酬に変換
    reward = contribution_count / total_periods
    return reward

# 訓練ループ内で使用
temporal_features = trajectory['temporal_contribution_features']
target_reward_value = calculate_target_reward_from_features(temporal_features)
target_reward = torch.tensor([[target_reward_value]], device=self.device)
loss = F.mse_loss(predicted_reward, target_reward)
```

**アプローチ 3: コントラスト学習**

```python
# 同じバッチ内で異なる貢献パターンを持つサンプルを対比
def train_irl_contrastive(self, trajectories, epochs=100):
    """
    コントラスト学習ベースのIRL訓練

    将来の貢献パターンが似ているサンプルは似た報酬を持ち、
    異なるサンプルは異なる報酬を持つように学習
    """
    for epoch in range(epochs):
        # バッチをサンプリング
        batch = random.sample(trajectories, min(32, len(trajectories)))

        # 各サンプルの報酬を予測
        rewards = []
        features_list = []

        for traj in batch:
            state = self.extract_developer_state(traj)
            actions = self.extract_developer_actions(traj['activity_history'], traj['context_date'])

            # ... state_tensor, action_tensorを作成
            reward = self.network(state_tensor, action_tensor)
            rewards.append(reward)

            # 将来の貢献パターンを記録
            temporal_feats = traj['temporal_contribution_features']
            features_list.append(temporal_feats)

        # コントラスト損失
        # 似た将来パターンを持つサンプルの報酬を近づける
        loss = contrastive_loss(rewards, features_list)

        # 逆伝播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 4. 予測関数の変更

```python
def predict_continuation_multi_period(
    self,
    developer: Dict[str, Any],
    activity_history: List[Dict[str, Any]],
    context_date: datetime
) -> Dict[str, Any]:
    """
    複数期間の継続確率を予測

    Returns:
        {
            'continuation_probabilities': {
                'short_term': 0.85,
                'mid_term': 0.62,
                'long_term': 0.43
            },
            'predictions': {
                'short_term': True,   # >= 0.5
                'mid_term': True,
                'long_term': False
            },
            'reasoning': "..."
        }
    """
    self.network.eval()

    with torch.no_grad():
        state = self.extract_developer_state(developer, activity_history, context_date)
        actions = self.extract_developer_actions(activity_history, context_date)

        if not actions:
            return {
                'continuation_probabilities': {
                    period: 0.0 for period in self.network.prediction_periods
                },
                'predictions': {
                    period: False for period in self.network.prediction_periods
                },
                'reasoning': '活動履歴が不足しています'
            }

        # 推論
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

            predicted_reward, predicted_continuations = self.network(state_seq, action_seq)
        else:
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            action_tensor = self.action_to_tensor(actions[-1]).unsqueeze(0)
            predicted_reward, predicted_continuations = self.network(state_tensor, action_tensor)

        # 結果を抽出
        continuation_probs = {
            period: float(pred.item())
            for period, pred in predicted_continuations.items()
        }

        predictions = {
            period: prob >= 0.5
            for period, prob in continuation_probs.items()
        }

        # 理由付け
        reasoning_parts = []
        for period, prob in continuation_probs.items():
            if prob >= 0.7:
                reasoning_parts.append(f"{period}: 高確率で継続 ({prob:.1%})")
            elif prob >= 0.5:
                reasoning_parts.append(f"{period}: 継続の可能性あり ({prob:.1%})")
            else:
                reasoning_parts.append(f"{period}: 離脱の可能性 ({prob:.1%})")

        reasoning = "; ".join(reasoning_parts)

        return {
            'continuation_probabilities': continuation_probs,
            'predictions': predictions,
            'reasoning': reasoning,
            'reward': float(predicted_reward.item())
        }
```

### 5. 評価関数の変更

```python
def evaluate_irl_model_multi_period(
    irl_system: 'MultiPeriodRetentionIRLSystem',
    test_trajectories: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    マルチ期間IRLモデルを評価

    Returns:
        {
            'short_term': {
                'auc_roc': 0.85,
                'auc_pr': 0.90,
                'f1': 0.78,
                ...
            },
            'mid_term': {...},
            'long_term': {...}
        }
    """
    logger.info("マルチ期間IRLモデルを評価中...")

    # 各期間ごとの予測結果を格納
    period_results = {
        period: {'y_true': [], 'y_pred': []}
        for period in irl_system.network.prediction_periods
    }

    for trajectory in test_trajectories:
        developer = trajectory['developer']
        activity_history = trajectory['activity_history']
        context_date = trajectory['context_date']
        true_labels = trajectory['labels']

        # 予測実行
        prediction = irl_system.predict_continuation_multi_period(
            developer, activity_history, context_date
        )

        # 各期間の結果を記録
        for period in irl_system.network.prediction_periods:
            period_results[period]['y_true'].append(
                1 if true_labels[period] else 0
            )
            period_results[period]['y_pred'].append(
                prediction['continuation_probabilities'][period]
            )

    # 各期間のメトリクスを計算
    evaluation_results = {}

    for period, data in period_results.items():
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

        metrics = {}

        # AUC-ROC
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['auc_roc'] = 0.0

        # AUC-PR
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
            metrics['auc_pr'] = auc(recall_curve, precision_curve)
        except:
            metrics['auc_pr'] = 0.0

        # F1, Precision, Recall
        try:
            metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
            metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        except:
            metrics['f1'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0

        # サンプル数
        metrics['test_samples'] = len(y_true)
        metrics['positive_samples'] = sum(y_true)
        metrics['positive_rate'] = sum(y_true) / len(y_true) if y_true else 0

        evaluation_results[period] = metrics

        logger.info(f"{period}: AUC-ROC={metrics['auc_roc']:.3f}, "
                   f"AUC-PR={metrics['auc_pr']:.3f}, F1={metrics['f1']:.3f}, "
                   f"正例率={metrics['positive_rate']:.1%}")

    return evaluation_results
```

---

## 📊 実験設定例

### 設定ファイル

```yaml
# configs/irl_multi_period_config.yaml

irl_multi_period:
  # 学習期間設定
  training_period:
    start_date: "2019-01-01"
    end_date: "2020-01-01"

  # 履歴ウィンドウ（各サンプリング時点で使用）
  history_window_months: 6

  # サンプリング設定
  sampling:
    interval_months: 1 # 1ヶ月ごとにサンプリング
    min_history_events: 3 # 最小イベント数

  # 予測期間定義
  prediction_periods:
    - name: "immediate"
      start_months: 0
      end_months: 2
      description: "即時（0-2ヶ月後）"
      loss_weight: 1.0

    - name: "short_term"
      start_months: 3
      end_months: 5
      description: "短期（3-5ヶ月後）"
      loss_weight: 1.0

    - name: "mid_term"
      start_months: 6
      end_months: 8
      description: "中期（6-8ヶ月後）"
      loss_weight: 1.0

    - name: "long_term"
      start_months: 9
      end_months: 12
      description: "長期（9-12ヶ月後）"
      loss_weight: 0.8 # 長期は不確実性が高いため重みを下げる

  # モデル設定
  model:
    state_dim: 10
    action_dim: 5
    hidden_dim: 64
    lstm_hidden: 128
    sequence: true
    seq_len: 15

  # 訓練設定
  training:
    epochs: 30
    learning_rate: 0.001
    test_split: 0.2
```

### 実行コマンド例

```bash
# マルチ期間IRL訓練と評価
uv run python scripts/training/irl/train_multi_period_irl.py \
  --config configs/irl_multi_period_config.yaml \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --output outputs/irl_multi_period
```

---

## 📈 期待される効果

### 1. より細かい粒度の予測

**従来:**

- 「6 ヶ月後に貢献しているか？」のみ

**変更後:**

- 「0-2 ヶ月後は継続 (90%確率)」
- 「3-5 ヶ月後は微妙 (55%確率)」
- 「6-8 ヶ月後は離脱 (30%確率)」

→ **時間的な変化を捉えられる**

### 2. 学習データの効率的活用

**従来:** 学習期間 12 ヶ月 → 1 サンプル

**変更後:** 学習期間 12 ヶ月 → 12 サンプル（1 ヶ月間隔の場合）

→ **データ効率が大幅に向上**

### 3. 実用的な介入タイミングの提案

```python
prediction = model.predict_continuation_multi_period(developer, history, date)

if prediction['predictions']['immediate']:  # 0-2ヶ月は継続
    if not prediction['predictions']['short_term']:  # 3-5ヶ月で離脱リスク
        print("⚠️ 3ヶ月以内に介入が必要")
        print("推奨アクション: メンタリング、適切なタスクの割り当て")
```

### 4. 時系列パターンの学習

LSTM により、以下のパターンを学習可能:

- 徐々に活動が減少 → 離脱リスク高
- 一時的な休止後に復帰 → 長期的には継続
- 短期的に活発 → その後燃え尽き

---

## 🚀 実装ロードマップ

### Phase 1: データ抽出の実装（1-2 日）

- [ ] `extract_temporal_trajectories_multi_window()` の実装
- [ ] 既存データでの動作確認
- [ ] サンプリング間隔・履歴ウィンドウのパラメータ調整

### Phase 2: モデル拡張（2-3 日）

- [ ] `MultiPeriodRetentionIRLNetwork` の実装
- [ ] マルチヘッド構造のテスト
- [ ] 既存モデルとの互換性確保

### Phase 3: 訓練・評価の実装（2-3 日）

- [ ] `train_irl_multi_period()` の実装
- [ ] 損失重み付けの調整機能
- [ ] `evaluate_irl_model_multi_period()` の実装
- [ ] 期間ごとのメトリクス可視化

### Phase 4: 実験と検証（3-5 日）

- [ ] サンプルデータでの動作確認
- [ ] OpenStack 実データでの評価
- [ ] ハイパーパラメータチューニング
- [ ] 従来手法との比較実験

### Phase 5: ドキュメント整備（1-2 日）

- [ ] 実装ドキュメントの更新
- [ ] 使用例の作成
- [ ] README の更新

**総所要時間: 約 10-15 日**

---

## 🔍 代替案・拡張案

### 案 1: ウィンドウ数の動的調整

```python
# データが豊富な期間は細かくサンプリング
if data_density_high:
    sampling_interval_months = 1  # 1ヶ月ごと
else:
    sampling_interval_months = 3  # 3ヶ月ごと
```

### 案 2: 重複期間の定義

```python
# 重複を許す（より密な予測）
prediction_periods = [
    {'name': '0-3m', 'start': 0, 'end': 3},
    {'name': '2-5m', 'start': 2, 'end': 5},  # 重複
    {'name': '4-7m', 'start': 4, 'end': 7},  # 重複
]
```

### 案 3: 連続値での貢献度予測

```python
# 二値分類ではなく、期間内の貢献量を予測
label = {
    'short_term': {
        'has_activity': True,
        'activity_count': 5,      # 期間内のレビュー数
        'activity_intensity': 0.7  # 平均活動強度
    }
}
```

---

## 💡 まとめ

### 主要な変更点

1. **期間ベースのラベル**: 単一時点 → 期間（0-2m, 3-5m など）
2. **動的サンプリング**: 固定スナップショット → 学習期間内の複数時点
3. **マルチタスク学習**: 単一予測 → 複数期間の同時予測
4. **データ効率**: 1 サンプル → N サンプル（学習期間に比例）

### メリット

✅ より実用的な予測（いつ介入すべきか分かる）  
✅ 学習データの効率的活用  
✅ 時系列パターンの詳細な学習  
✅ 予測の信頼性向上（複数期間で検証）

### デメリット・注意点

⚠️ 実装の複雑性が増加  
⚠️ 訓練時間の増加（サンプル数増加）  
⚠️ メモリ使用量の増加（マルチヘッド）  
⚠️ ハイパーパラメータ調整の難度上昇

---

## 📚 参考資料

- 現在の実装: `scripts/training/irl/train_temporal_irl_sliding_window.py`
- IRL システム: `src/gerrit_retention/rl_prediction/retention_irl_system.py`
- 現在の設計: `README_TEMPORAL_IRL.md`

---

**作成日**: 2025-10-21  
**バージョン**: 1.0  
**ステータス**: 提案
