# 【最重要】タスク拒否パターン学習機能の実装ガイド

## 概要

**目的**: レビュアーが「どんなタスクを拒否してきたか」を学習し、「タスク拒否が続いたら離脱する」というパターンを予測に活用する。

**重要な設計原則**:
- ✅ 学習時: タスクへの応答/拒否の履歴を記録
- ✅ 予測時: タスク情報なしで、拒否パターンだけで継続/離脱を予測
- ✅ LSTMが時系列パターン（拒否→拒否→離脱）を自動学習

---

## 現在の問題

### 現状の実装 (`retention_irl_system.py`)

```python
# 現在: レビュー依頼があったという「事実」だけを記録
activity_history.append({
    'type': 'review',
    'timestamp': row['request_time'],
    'project': row['project'],
})

# ラベル: 予測期間中に活動があったか
continued = len(reviewer_target) > 0
```

**問題点**:
- ❌ レビュー依頼に「応答したか/拒否したか」を記録していない
- ❌ 「拒否が続くと離脱しやすい」パターンを学習できない
- ❌ タスク応答率や連続拒否などの重要指標が欠落

### 利用可能なデータ

`review_requests_openstack_pilot_w14_paths.csv` には既に存在：

| カラム | 説明 | 値 |
|--------|------|-----|
| `label` | レビュー依頼への応答 | 1=応答, 0=拒否/無視 |
| `responded_within_days` | 応答までの日数 | 0-14 |
| `response_latency_days` | 応答遅延 | 浮動小数点 |
| `reviewer_past_response_rate_180d` | 過去の応答率 | 0.0-1.0 |
| `reviewer_assignment_load_7d/30d/180d` | レビュー負荷 | 整数 |

---

## 実装ステップ

### ステップ1: データ構造の拡張

#### 1-1. `DeveloperState` に拒否パターンを追加

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py:23-38`

```python
@dataclass
class DeveloperState:
    """開発者の状態表現（拡張版: 14次元）"""
    developer_id: str
    experience_days: int
    total_changes: int
    total_reviews: int
    recent_activity_frequency: float
    avg_activity_gap: float
    activity_trend: str
    collaboration_score: float
    code_quality_score: float
    recent_acceptance_rate: float  # コード品質（既存）
    review_load: float  # レビュー負荷（既存）

    # ✨ NEW: タスク応答パターン（4次元追加）
    recent_response_rate: float        # 直近30日のタスク応答率（0.0-1.0）
    consecutive_rejections: int        # 連続拒否回数（0以上）
    rejection_trend: str               # 'increasing'/'stable'/'decreasing'
    avg_rejection_rate: float          # 全期間の平均拒否率（0.0-1.0）

    timestamp: datetime
```

**次元数の変更**:
- 10次元 → **14次元**
- `state_dim: 14` に更新

#### 1-2. `DeveloperAction` に応答情報を追加

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py:42-49`

```python
@dataclass
class DeveloperAction:
    """開発者の行動表現（拡張版: 5次元）"""
    action_type: str
    intensity: float
    collaboration: float
    response_time: float
    review_size: float

    # ✨ NEW: タスク応答情報
    responded: float  # 1.0=応答, 0.0=拒否/無視

    timestamp: datetime
```

**次元数の変更**:
- 4次元 → **5次元**
- `action_dim: 5` に更新

---

### ステップ2: 特徴量計算メソッドの追加

#### 2-1. タスク応答率の計算

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py` (新規メソッド)

**追加位置**: `_calculate_recent_acceptance_rate()` の後

```python
def _calculate_recent_response_rate(self,
                                   activity_history: List[Dict[str, Any]],
                                   context_date: datetime,
                                   days: int = 30) -> float:
    """
    直近N日のタスク応答率を計算

    Args:
        activity_history: 活動履歴（'responded'フィールドを含む）
        context_date: 基準日
        days: 対象期間（日数）

    Returns:
        応答率（0.0～1.0）、データなしは0.5（中立）
    """
    cutoff_date = context_date - timedelta(days=days)

    # 直近の活動のみフィルタ
    recent_activities = [
        activity for activity in activity_history
        if activity.get('timestamp', context_date) >= cutoff_date
    ]

    if not recent_activities:
        return 0.5  # データなし → 中立

    # レビュー依頼とその応答を集計
    total_requests = 0
    responded_requests = 0

    for activity in recent_activities:
        if activity.get('type') == 'review':
            total_requests += 1
            if activity.get('responded', True):  # デフォルトTrue（後方互換性）
                responded_requests += 1

    if total_requests == 0:
        return 0.5  # レビュー依頼なし → 中立

    return responded_requests / total_requests


def _calculate_consecutive_rejections(self,
                                     activity_history: List[Dict[str, Any]],
                                     context_date: datetime) -> int:
    """
    連続拒否回数を計算（最新から過去に遡る）

    Args:
        activity_history: 活動履歴
        context_date: 基準日

    Returns:
        連続拒否回数（0以上）
    """
    # 最新から過去に並び替え
    sorted_activities = sorted(
        [a for a in activity_history if a.get('type') == 'review'],
        key=lambda x: x.get('timestamp', context_date),
        reverse=True
    )

    consecutive = 0
    for activity in sorted_activities:
        if not activity.get('responded', True):
            consecutive += 1
        else:
            break  # 応答があったら終了

    return consecutive


def _calculate_rejection_trend(self,
                              activity_history: List[Dict[str, Any]],
                              context_date: datetime) -> str:
    """
    拒否傾向を分析（'increasing'/'stable'/'decreasing'）

    Args:
        activity_history: 活動履歴
        context_date: 基準日

    Returns:
        拒否傾向文字列
    """
    # 直近30日と60-90日前を比較
    cutoff_recent = context_date - timedelta(days=30)
    cutoff_old_start = context_date - timedelta(days=90)
    cutoff_old_end = context_date - timedelta(days=60)

    recent_rate = self._calculate_recent_response_rate(
        activity_history, context_date, days=30
    )

    # 過去のデータ
    old_activities = [
        a for a in activity_history
        if cutoff_old_end >= a.get('timestamp', context_date) >= cutoff_old_start
           and a.get('type') == 'review'
    ]

    if not old_activities:
        return 'stable'  # 比較データなし

    old_responded = sum(1 for a in old_activities if a.get('responded', True))
    old_rate = old_responded / len(old_activities)

    # 応答率の変化で判定（拒否率は1 - 応答率）
    if recent_rate < old_rate - 0.2:
        return 'increasing'  # 応答率低下 = 拒否増加
    elif recent_rate > old_rate + 0.2:
        return 'decreasing'  # 応答率上昇 = 拒否減少
    else:
        return 'stable'


def _calculate_avg_rejection_rate(self,
                                  activity_history: List[Dict[str, Any]]) -> float:
    """
    全期間の平均拒否率を計算

    Args:
        activity_history: 活動履歴

    Returns:
        平均拒否率（0.0～1.0）
    """
    review_activities = [
        a for a in activity_history
        if a.get('type') == 'review'
    ]

    if not review_activities:
        return 0.0

    rejected = sum(1 for a in review_activities if not a.get('responded', True))
    return rejected / len(review_activities)
```

#### 2-2. `extract_developer_state()` を更新

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py:330-387`

**変更箇所**: 状態構築部分（line 369-387付近）

```python
def extract_developer_state(self,
                           developer: Dict[str, Any],
                           activity_history: List[Dict[str, Any]],
                           context_date: datetime) -> DeveloperState:
    # ... 既存のコード ...

    # 最近のレビュー受諾率（直近30日）
    recent_acceptance_rate = self._calculate_recent_acceptance_rate(activity_history, context_date, days=30)

    # レビュー負荷（直近30日 / 平均）
    review_load = self._calculate_review_load(activity_history, context_date, days=30)

    # ✨ NEW: タスク応答パターン
    recent_response_rate = self._calculate_recent_response_rate(activity_history, context_date, days=30)
    consecutive_rejections = self._calculate_consecutive_rejections(activity_history, context_date)
    rejection_trend = self._calculate_rejection_trend(activity_history, context_date)
    avg_rejection_rate = self._calculate_avg_rejection_rate(activity_history)

    return DeveloperState(
        developer_id=developer_id,
        experience_days=experience_days,
        total_changes=total_changes,
        total_reviews=total_reviews,
        recent_activity_frequency=recent_activity_frequency,
        avg_activity_gap=avg_activity_gap,
        activity_trend=activity_trend,
        collaboration_score=collaboration_score,
        code_quality_score=code_quality_score,
        recent_acceptance_rate=recent_acceptance_rate,
        review_load=review_load,
        # ✨ NEW
        recent_response_rate=recent_response_rate,
        consecutive_rejections=consecutive_rejections,
        rejection_trend=rejection_trend,
        avg_rejection_rate=avg_rejection_rate,
        timestamp=context_date
    )
```

#### 2-3. `extract_developer_actions()` を更新

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py:390-434`

**変更箇所**: 行動構築部分（line 421-428付近）

```python
def extract_developer_actions(self,
                            activity_history: List[Dict[str, Any]],
                            context_date: datetime) -> List[DeveloperAction]:
    """開発者の行動を抽出"""

    actions = []

    for activity in activity_history:
        try:
            # 行動タイプ
            action_type = activity.get('type', 'unknown')

            # 行動の強度（変更ファイル数ベース）
            intensity = self._calculate_action_intensity(activity)

            # 協力度
            collaboration = self._calculate_action_collaboration(activity)

            # レスポンス時間（レビューリクエストから応答までの日数）
            response_time = self._calculate_response_time(activity)

            # レビュー規模（変更行数ベース）
            review_size = self._calculate_review_size(activity)

            # ✨ NEW: タスク応答情報
            responded = float(activity.get('responded', 1))  # デフォルト1（後方互換性）

            # タイムスタンプ
            timestamp_str = activity.get('timestamp', context_date.isoformat())
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = timestamp_str

            actions.append(DeveloperAction(
                action_type=action_type,
                intensity=intensity,
                collaboration=collaboration,
                response_time=response_time,
                review_size=review_size,
                responded=responded,  # ✨ NEW
                timestamp=timestamp
            ))

        except Exception as e:
            logger.warning(f"行動抽出エラー: {e}")
            continue

    return actions
```

---

### ステップ3: テンソル変換の更新

#### 3-1. `state_to_tensor()` を更新

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py:436-460`

```python
def state_to_tensor(self, state: DeveloperState) -> torch.Tensor:
    """状態をテンソルに変換（14次元版）"""

    # 活動トレンドのエンコーディング
    trend_map = {'increasing': 0.8, 'stable': 0.5, 'decreasing': 0.2}
    trend_value = trend_map.get(state.activity_trend, 0.5)

    # 拒否トレンドのエンコーディング ✨ NEW
    rejection_trend_map = {'increasing': 0.8, 'stable': 0.5, 'decreasing': 0.2}
    rejection_trend_value = rejection_trend_map.get(state.rejection_trend, 0.5)

    # 連続拒否を正規化 ✨ NEW
    normalized_consecutive_rejections = min(state.consecutive_rejections / 10.0, 1.0)

    # 全特徴量を0-1の範囲に正規化
    features = [
        min(state.experience_days / 365.0, 1.0),
        min(state.total_changes / 100.0, 1.0),
        min(state.total_reviews / 100.0, 1.0),
        min(state.recent_activity_frequency, 1.0),
        min(state.avg_activity_gap / 30.0, 1.0),
        trend_value,
        min(state.collaboration_score, 1.0),
        min(state.code_quality_score, 1.0),
        min(state.recent_acceptance_rate, 1.0),
        min(state.review_load, 1.0),
        # ✨ NEW: タスク応答パターン（4次元）
        min(state.recent_response_rate, 1.0),
        normalized_consecutive_rejections,
        rejection_trend_value,
        min(state.avg_rejection_rate, 1.0),
    ]

    return torch.tensor(features, dtype=torch.float32, device=self.device)
```

#### 3-2. `action_to_tensor()` を更新

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py:464-480`

```python
def action_to_tensor(self, action: DeveloperAction) -> torch.Tensor:
    """行動をテンソルに変換（5次元版）"""

    # レスポンス時間を「素早さ」に変換（0-1の範囲に正規化）
    response_speed = 1.0 / (1.0 + action.response_time / 3.0)

    # 全特徴量を0-1の範囲に正規化
    features = [
        min(action.intensity, 1.0),        # 強度（変更ファイル数、0-1）
        min(action.collaboration, 1.0),    # 協力度（0-1）
        min(response_speed, 1.0),           # レスポンス速度（素早いほど大きい、0-1）
        min(action.review_size, 1.0),       # レビュー規模（変更行数、0-1）
        min(action.responded, 1.0),         # ✨ NEW: タスク応答（0-1）
    ]

    return torch.tensor(features, dtype=torch.float32, device=self.device)
```

---

### ステップ4: ネットワーク設定の更新

#### 4-1. `RetentionIRLNetwork` の次元数変更

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py:52-100`

**変更不要**: `__init__`の引数で次元数を受け取るため、configで指定するだけでOK

#### 4-2. `RetentionIRLSystem` の初期化

**ファイル**: `src/gerrit_retention/rl_prediction/retention_irl_system.py:185-240`

**変更箇所**: デフォルト設定（line 188-195付近）

```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    """
    継続予測のためのIRLシステムを初期化

    Args:
        config: 設定辞書
    """
    self.config = config or {}
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # デフォルト設定
    self.state_dim = self.config.get('state_dim', 14)  # ✨ 10 → 14
    self.action_dim = self.config.get('action_dim', 5)  # ✨ 4 → 5
    self.hidden_dim = self.config.get('hidden_dim', 128)
    self.learning_rate = self.config.get('learning_rate', 0.001)
    # ... 以下既存コード ...
```

---

### ステップ5: データ読み込みの更新

#### 5-1. `train_temporal_irl_sliding_window.py` の更新

**ファイル**: `scripts/training/irl/train_temporal_irl_sliding_window.py`

**変更箇所1**: データ読み込み（line 50-61）

```python
def load_review_logs(csv_path: Path) -> pd.DataFrame:
    """レビューログを読み込む（拡張版: label対応）"""
    logger.info(f"レビューログを読み込み中: {csv_path}")
    df = pd.read_csv(csv_path)

    # 日付カラムをdatetimeに変換
    date_col = 'request_time' if 'request_time' in df.columns else 'created'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # ✨ NEW: labelカラムの確認
    if 'label' in df.columns:
        logger.info(f"タスク応答ラベルを検出: 応答率={df['label'].mean():.2%}")
    else:
        logger.warning("labelカラムが見つかりません。全て応答したとみなします。")
        df['label'] = 1  # デフォルト値

    logger.info(f"レビューログ読み込み完了: {len(df)}件")
    return df
```

**変更箇所2**: 軌跡抽出（line 108-119）

```python
# 活動履歴を構築
activity_history = []
for _, row in reviewer_history.iterrows():
    activity_history.append({
        'type': 'review',
        'timestamp': row[date_col],
        'project': row.get('project', 'unknown'),
        'responded': row.get('label', 1),  # ✨ NEW: タスク応答
        'message': '',
        'lines_added': row.get('change_insertions', 0),  # ✨ データにあれば使う
        'lines_deleted': row.get('change_deletions', 0),  # ✨ データにあれば使う
        'files_changed': row.get('change_files_count', 1),  # ✨ データにあれば使う
    })
```

**変更箇所3**: モデル設定（main関数内）

```python
# モデル設定
config = {
    'state_dim': 14,  # ✨ 10 → 14
    'action_dim': 5,  # ✨ 4 → 5
    'hidden_dim': 128,
    'learning_rate': 0.001,
    'sequence': args.sequence,
    'seq_len': args.seq_len,
}
```

---

### ステップ6: プロジェクト別予測の更新

#### 6-1. `train_temporal_irl_project_aware.py` の更新

**ファイル**: `scripts/training/irl/train_temporal_irl_project_aware.py`

**同様の変更を適用**:
- データ読み込みに`label`対応
- 活動履歴に`responded`追加
- モデル設定を`state_dim: 14, action_dim: 5`に変更

---

## 期待される効果

### 学習できるパターン

#### パターン1: 拒否増加→離脱

```
Timeline:
月1: response_rate=1.0, consecutive=0  → 継続確率 0.95
月2: response_rate=0.7, consecutive=2  → 継続確率 0.70
月3: response_rate=0.3, consecutive=5  → 継続確率 0.30
月4: 離脱

→ LSTMが「拒否率上昇 + 連続拒否 → 離脱」を学習
```

#### パターン2: 負荷過多→拒否増加→離脱

```
状態:
review_load=0.5, response_rate=1.0  → 継続
review_load=1.5, response_rate=0.7  → まだ継続
review_load=2.0, response_rate=0.3  → 離脱リスク↑

→ 因果関係「高負荷 → 拒否増加 → 離脱」を学習
```

#### パターン3: プロジェクト別応答率

```
project_A: response_rate=0.9  → 継続
project_B: response_rate=0.3  → 離脱リスク

→ 「特定プロジェクトでの拒否が離脱に繋がる」を学習
```

### 予測精度の向上見込み

| 指標 | 現在 | 予想 | 改善幅 |
|------|------|------|--------|
| AUC-ROC | 0.868 | **0.90+** | +3.2% |
| AUC-PR | 0.983 | **0.99+** | +0.7% |
| F1 | 0.978 | **0.98+** | +0.2% |

**特に改善が期待される領域**:
- 早期離脱予測（拒否が始まった時点で検出）
- 負荷過多による離脱（拒否率と負荷の相関）
- プロジェクト別離脱リスク（プロジェクト固有パターン）

---

## 実装チェックリスト

### フェーズ1: データ構造とメソッド（1-2時間）

- [ ] `DeveloperState` に4次元追加（`recent_response_rate`, `consecutive_rejections`, `rejection_trend`, `avg_rejection_rate`）
- [ ] `DeveloperAction` に`responded`追加
- [ ] `_calculate_recent_response_rate()` 実装
- [ ] `_calculate_consecutive_rejections()` 実装
- [ ] `_calculate_rejection_trend()` 実装
- [ ] `_calculate_avg_rejection_rate()` 実装
- [ ] `extract_developer_state()` 更新（新特徴量を計算）
- [ ] `extract_developer_actions()` 更新（`responded`を追加）

### フェーズ2: テンソル変換（30分）

- [ ] `state_to_tensor()` を14次元に更新
- [ ] `action_to_tensor()` を5次元に更新
- [ ] `RetentionIRLSystem.__init__()` のデフォルト次元数変更

### フェーズ3: データ読み込み（30分）

- [ ] `train_temporal_irl_sliding_window.py` の`load_review_logs()` に`label`対応
- [ ] 軌跡抽出部分に`responded`追加
- [ ] モデル設定を`state_dim: 14, action_dim: 5`に変更
- [ ] `train_temporal_irl_project_aware.py` にも同様の変更

### フェーズ4: テスト実行（30分）

- [ ] サンプルデータで動作確認
  ```bash
  uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
    --reviews data/review_requests_openstack_pilot_w14_paths.csv \
    --snapshot-date 2024-01-01 \
    --history-months 6 \
    --target-months 6 \
    --epochs 10 \
    --sequence \
    --seq-len 15 \
    --output importants/irl_task_rejection_test
  ```
- [ ] エラーがないか確認
- [ ] 学習ログで新特徴量が使われているか確認

### フェーズ5: 本番評価（1時間）

- [ ] 全データで学習・評価
  ```bash
  uv run python scripts/training/irl/train_temporal_irl_sliding_window.py \
    --reviews data/review_requests_openstack_pilot_w14_paths.csv \
    --snapshot-date 2023-01-01 \
    --history-months 3 6 9 12 \
    --target-months 3 6 9 12 \
    --epochs 30 \
    --sequence \
    --seq-len 15 \
    --output importants/irl_task_rejection_full
  ```
- [ ] 精度向上を確認（AUC-ROC 0.868 → 0.90+）
- [ ] 評価レポート生成
- [ ] 特徴量重要度分析（新特徴が有効か確認）

### フェーズ6: ドキュメント更新（30分）

- [ ] `README_TEMPORAL_IRL.md` に新機能を記載
- [ ] `CLAUDE.md` の特徴量説明を更新
- [ ] 評価結果を`importants/irl_task_rejection_full/EVALUATION_REPORT.md`に保存

---

## トラブルシューティング

### エラー1: 次元数不一致

```
RuntimeError: Expected tensor with 10 features, got 14
```

**原因**: 既存モデルとの互換性問題

**解決策**:
- 新規学習する（既存モデルは使用不可）
- または、`state_dim=10, action_dim=4`で学習した既存モデルは別ディレクトリに保存

### エラー2: `label`カラムが存在しない

```
KeyError: 'label'
```

**原因**: 古いデータ形式（`sample_reviews.csv`など）

**解決策**:
- `review_requests_*_w14*.csv`を使用（labelを含む）
- または、`df['label'] = 1`でデフォルト値を設定（後方互換性）

### エラー3: 精度が低下した

**原因**: データ不足、パラメータ調整不足

**解決策**:
- エポック数を増やす（`--epochs 50`）
- シーケンス長を調整（`--seq-len 20`）
- 学習期間を長くする（`--history-months 12 18`）

---

## データファイルの推奨

### 優先度1: 完全なデータ

```
data/review_requests_openstack_pilot_w14_paths.csv
```

**含まれる情報**:
- ✅ `label` (応答/拒否)
- ✅ `response_latency_days` (応答遅延)
- ✅ `reviewer_past_response_rate_180d` (過去の応答率)
- ✅ `reviewer_assignment_load_*` (負荷)
- ✅ `change_insertions/deletions/files_count` (変更規模)

### 優先度2: 基本データ

```
data/sample_reviews.csv
```

**含まれる情報**:
- ❌ `label`なし → デフォルト値1を設定
- ✅ `reviewer_email`, `request_time`, `project`

---

## 成功指標

### 定量指標

- [ ] AUC-ROC: 0.868 → **0.90以上**
- [ ] AUC-PR: 0.983 → **0.99以上**
- [ ] F1スコア: 0.978 → **0.98以上**
- [ ] 早期離脱検出率（3ヶ月前予測）: **70%以上**

### 定性指標

- [ ] 「拒否が続くと離脱」パターンを学習（連続拒否 vs 継続の相関）
- [ ] 負荷と拒否率の相関を捉えている
- [ ] プロジェクト別の応答率差を反映
- [ ] 予測理由で拒否パターンが言及される

---

## 参考情報

### データソース

- **メインデータ**: `data/review_requests_openstack_pilot_w14_paths.csv`
- **データ生成**: `examples/build_eval_from_review_requests.py`
- **データ仕様**: レビュー依頼から14日以内の応答を`label=1`とする

### 関連コード

- **IRLシステム**: `src/gerrit_retention/rl_prediction/retention_irl_system.py`
- **学習スクリプト**: `scripts/training/irl/train_temporal_irl_sliding_window.py`
- **プロジェクト別**: `scripts/training/irl/train_temporal_irl_project_aware.py`

### ドキュメント

- **IRL設計**: `README_TEMPORAL_IRL.md`
- **プロジェクト全体**: `README.md`
- **開発者ガイド**: `CLAUDE.md`

---

## 最後に

この機能は**継続予測の精度向上に直結する最重要機能**です。

**実装の核心**:
- タスク拒否パターンを時系列で学習
- LSTMが「拒否→拒否→離脱」を自動検出
- 予測時はタスク情報不要（拒否履歴だけで予測）

**推定工数**: 合計 **3-4時間**

実装時は上記チェックリストに沿って進めてください！
