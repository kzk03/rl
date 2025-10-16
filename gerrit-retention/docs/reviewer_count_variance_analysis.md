# 学習期間による予測対象レビュアー数の変化 - 原因分析と解決策

## 問題の概要

現状の実装では、**学習期間が異なると予測対象のレビュアー数が変化する**という問題があります。

## 実データによる確認（OpenStackクロスプロジェクト評価）

| 学習期間 | 予測期間 | 訓練サンプル数 | テストサンプル数 |
|---------|---------|---------------|----------------|
| 6ヶ月   | 6ヶ月   | 314           | 79             |
| 6ヶ月   | 12ヶ月  | 314           | 79             |
| 12ヶ月  | 6ヶ月   | 488           | 122            |
| 12ヶ月  | 12ヶ月  | 488           | 122            |

**問題**: 学習期間6ヶ月では393人（314+79）、12ヶ月では610人（488+122）のレビュアーを対象としている。

---

## 原因分析

### 根本原因

現在の実装では、**学習期間中に活動があったレビュアー**のみを抽出しています。

#### コード分析（`train_temporal_irl_sliding_window.py`）

```python
def extract_trajectories_with_window(
    df: pd.DataFrame,
    snapshot_date: datetime,
    history_months: int,  # ← この期間が変わると...
    target_months: int,
    ...
):
    # 学習期間の定義
    history_start = snapshot_date - pd.DateOffset(months=history_months)
    history_df = df[(df[date_col] >= history_start) &
                    (df[date_col] < snapshot_date)]

    # ★ここが問題: 学習期間中に活動があったレビュアーのみを抽出
    reviewers_in_history = set(history_df[reviewer_col].unique())  # ← 人数が変わる

    # 予測期間のデータ
    target_df = df[(df[date_col] >= snapshot_date) &
                   (df[date_col] < target_end)]

    # レビュアーごとに軌跡を作成
    for reviewer in reviewers_in_history:  # ← ここで対象者が決まる
        ...
```

### 具体例

**スナップショット日**: 2023-01-01

#### 学習期間6ヶ月の場合
```
学習期間: 2022-07-01 ~ 2023-01-01
→ この期間に活動があったレビュアー: 393人
```

#### 学習期間12ヶ月の場合
```
学習期間: 2022-01-01 ~ 2023-01-01
→ この期間に活動があったレビュアー: 610人
```

**差分**: 217人のレビュアーが、2022年1月～6月には活動していたが、7月～12月には活動していなかった。

### 問題点

1. **評価の一貫性欠如**: 異なる学習期間で異なる対象者を予測しているため、公平な比較ができない
2. **サンプリングバイアス**: 短い学習期間では「最近活動的なレビュアー」に偏り、長い学習期間では「長期的に活動してきたレビュアー」も含まれる
3. **再現性の問題**: 学習期間によって予測対象が変わるため、モデルの性能比較が困難

---

## 解決策の提案

### アプローチ1: 固定基準日での対象者統一（推奨）

**概要**: すべての学習期間で、スナップショット日に「最近活動していたレビュアー」を共通の対象とする。

#### 実装方法

```python
def extract_trajectories_with_fixed_population(
    df: pd.DataFrame,
    snapshot_date: datetime,
    history_months: int,
    target_months: int,
    reference_period_months: int = 6,  # 基準期間（例: 6ヶ月）
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    固定基準期間で対象レビュアーを決定する軌跡抽出

    Args:
        reference_period_months: 対象者選定の基準期間（6ヶ月推奨）
    """
    # ★ステップ1: 基準期間で対象レビュアーを決定（全実験で共通）
    reference_start = snapshot_date - pd.DateOffset(months=reference_period_months)
    reference_df = df[(df[date_col] >= reference_start) &
                     (df[date_col] < snapshot_date)]
    target_reviewers = set(reference_df[reviewer_col].unique())

    logger.info(f"基準期間（{reference_period_months}ヶ月）で対象レビュアーを決定: {len(target_reviewers)}人")

    # ★ステップ2: 学習期間のデータを取得
    history_start = snapshot_date - pd.DateOffset(months=history_months)
    history_df = df[(df[date_col] >= history_start) &
                    (df[date_col] < snapshot_date)]

    # 予測期間のデータ
    target_end = snapshot_date + pd.DateOffset(months=target_months)
    target_df = df[(df[date_col] >= snapshot_date) &
                   (df[date_col] < target_end)]

    trajectories = []

    # ★ステップ3: 対象レビュアー全員について軌跡を作成
    for reviewer in target_reviewers:  # ← 固定の対象者リスト
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]
        reviewer_target = target_df[target_df[reviewer_col] == reviewer]

        # 学習期間に活動がない場合の処理
        if len(reviewer_history) == 0:
            # オプション1: スキップ
            continue
            # オプション2: ダミーアクションを作成（推奨）
            # activity_history = create_dummy_activity(reviewer, history_start)
        else:
            activity_history = create_activity_history(reviewer_history)

        continued = len(reviewer_target) > 0

        trajectories.append({
            'developer': create_developer_info(reviewer, reviewer_history),
            'activity_history': activity_history,
            'continued': continued,
            'context_date': snapshot_date,
            'reviewer': reviewer
        })

    logger.info(f"全対象レビュアーの軌跡作成完了: {len(trajectories)}件")

    # 訓練/テスト分割
    np.random.seed(42)
    np.random.shuffle(trajectories)
    split_idx = int(len(trajectories) * 0.8)

    return trajectories[:split_idx], trajectories[split_idx:]
```

**メリット**:
- ✅ すべての学習期間で同じ対象者を予測
- ✅ 評価の一貫性が保たれる
- ✅ 実装が比較的シンプル

**デメリット**:
- ⚠️ 学習期間が基準期間より長い場合、一部レビュアーの学習データが不足する可能性

---

### アプローチ2: 最小共通期間での対象者統一

**概要**: すべての学習期間候補の中で**最短の期間**に活動していたレビュアーを対象とする。

#### 実装方法

```python
def extract_common_reviewers(
    df: pd.DataFrame,
    snapshot_date: datetime,
    history_months_list: List[int],  # 例: [3, 6, 9, 12]
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time'
) -> Set[str]:
    """
    全学習期間で共通の対象レビュアーを抽出
    """
    # 最短学習期間を使用
    min_history_months = min(history_months_list)

    history_start = snapshot_date - pd.DateOffset(months=min_history_months)
    history_df = df[(df[date_col] >= history_start) &
                    (df[date_col] < snapshot_date)]

    common_reviewers = set(history_df[reviewer_col].unique())

    logger.info(f"共通対象レビュアー（最短期間{min_history_months}ヶ月基準）: {len(common_reviewers)}人")

    return common_reviewers


def sliding_window_evaluation_fixed_population(
    df: pd.DataFrame,
    snapshot_date: datetime,
    history_months_list: List[int],
    target_months_list: List[int],
    ...
):
    """
    固定対象者でのスライディングウィンドウ評価
    """
    # ★ステップ1: 共通対象レビュアーを決定
    common_reviewers = extract_common_reviewers(
        df, snapshot_date, history_months_list
    )

    results = []

    for history_months in history_months_list:
        for target_months in target_months_list:
            # ★ステップ2: 共通対象レビュアーのみで軌跡を抽出
            train_traj, test_traj = extract_trajectories_for_reviewers(
                df, snapshot_date, history_months, target_months,
                target_reviewers=common_reviewers  # ← 固定リスト
            )

            # モデル訓練と評価
            ...
```

**メリット**:
- ✅ 完全に同じ対象者で評価
- ✅ 最も公平な比較が可能

**デメリット**:
- ⚠️ 最短期間に活動していたレビュアーのみに限定されるため、サンプル数が最も少なくなる
- ⚠️ 長期的な貢献者の情報が活用されない

---

### アプローチ3: 積集合での対象者統一

**概要**: **すべての学習期間**に活動していたレビュアーのみを対象とする。

#### 実装方法

```python
def extract_intersection_reviewers(
    df: pd.DataFrame,
    snapshot_date: datetime,
    history_months_list: List[int],
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time'
) -> Set[str]:
    """
    全学習期間に共通して活動していたレビュアーを抽出
    """
    reviewer_sets = []

    for history_months in history_months_list:
        history_start = snapshot_date - pd.DateOffset(months=history_months)
        history_df = df[(df[date_col] >= history_start) &
                        (df[date_col] < snapshot_date)]
        reviewer_sets.append(set(history_df[reviewer_col].unique()))

    # 積集合を取得
    intersection_reviewers = set.intersection(*reviewer_sets)

    logger.info(f"全期間共通レビュアー: {len(intersection_reviewers)}人")
    logger.info(f"  各期間のレビュアー数: {[len(s) for s in reviewer_sets]}")

    return intersection_reviewers
```

**メリット**:
- ✅ すべての期間で活動している「真の継続的貢献者」のみを対象
- ✅ 最も厳密な評価

**デメリット**:
- ❌ サンプル数が大幅に減少（特に長い学習期間を含む場合）
- ❌ 新規貢献者や一時的な貢献者が除外される

---

### アプローチ4: 重み付きサンプリング

**概要**: 学習期間ごとに対象者が変わることを前提に、**サンプル重み**で調整する。

#### 実装方法

```python
def extract_trajectories_with_weights(
    df: pd.DataFrame,
    snapshot_date: datetime,
    history_months: int,
    target_months: int,
    reference_population: Set[str],  # 基準となる全対象レビュアー
    ...
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
    """
    サンプル重み付き軌跡抽出
    """
    # 現在の学習期間で活動があったレビュアー
    active_reviewers = extract_active_reviewers(df, snapshot_date, history_months)

    # 重みの計算: 基準集団に対する割合の逆数
    sampling_rate = len(active_reviewers) / len(reference_population)
    sample_weight = 1.0 / sampling_rate  # 少ないサンプルには大きな重み

    trajectories = []
    weights = []

    for reviewer in active_reviewers:
        trajectory = create_trajectory(...)
        trajectories.append(trajectory)
        weights.append(sample_weight)

    return train_traj, test_traj, weights


def train_irl_model_weighted(
    trajectories: List[Dict[str, Any]],
    weights: List[float],
    config: Dict[str, Any],
    epochs: int = 30
) -> RetentionIRLSystem:
    """
    重み付きIRLモデル訓練
    """
    irl_system = RetentionIRLSystem(config)

    # 重み付き訓練（損失関数に重みを適用）
    for epoch in range(epochs):
        for trajectory, weight in zip(trajectories, weights):
            loss = compute_loss(trajectory)
            weighted_loss = loss * weight  # 重みを適用
            weighted_loss.backward()
            ...

    return irl_system
```

**メリット**:
- ✅ サンプリングバイアスを統計的に補正
- ✅ データの無駄がない

**デメリット**:
- ⚠️ 実装が複雑
- ⚠️ 重みの設定方法によっては過学習のリスク

---

## 推奨アプローチ

### **アプローチ1: 固定基準日での対象者統一** を推奨

**理由**:
1. **実装の簡潔性**: コード変更が最小限
2. **実用性**: 実際の運用では「最近活動しているレビュアー」を対象とするのが自然
3. **バランス**: サンプル数を過度に減らさず、一貫性も保てる

### 推奨パラメータ

```python
reference_period_months = 6  # 基準期間: 6ヶ月
```

**選定理由**:
- OpenStackデータでは6ヶ月で393人のレビュアーを対象化
- 中央値的な学習期間（3, 6, 9, 12ヶ月の中で中庸）
- 「最近の活動」を捉えつつ、十分なサンプル数を確保

### 実装例

```bash
# 基準期間6ヶ月で対象者を統一して評価
uv run python scripts/training/irl/train_temporal_irl_sliding_window_fixed_pop.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --snapshot-date 2023-01-01 \
  --history-months 3 6 9 12 \
  --target-months 3 6 9 12 \
  --reference-period 6 \
  --sequence \
  --seq-len 15 \
  --epochs 30 \
  --output importants/irl_fixed_population
```

---

## 期待される効果

### Before（現状）
| 学習期間 | 対象レビュアー数 | AUC-ROC |
|---------|--------------|---------|
| 6ヶ月   | 393人        | 0.832   |
| 12ヶ月  | 610人        | 0.868   |

→ **異なる対象者を予測しているため、公平な比較ができない**

### After（改善後）
| 学習期間 | 対象レビュアー数 | AUC-ROC |
|---------|--------------|---------|
| 6ヶ月   | 393人（固定） | 0.832   |
| 12ヶ月  | 393人（固定） | 0.875?  |

→ **同じ対象者を予測しているため、学習期間の効果を正確に評価可能**

### 評価の改善点

1. **公平な比較**: すべての学習期間で同じ対象者を予測
2. **バイアス除去**: 「誰を予測するか」ではなく「どれだけ正確に予測できるか」を評価
3. **再現性**: 基準期間を固定することで、実験の再現性が向上
4. **実用性**: 実運用では「最近活動しているレビュアーの将来を予測」が自然

---

## 次のステップ

1. **アプローチ1を実装**: `train_temporal_irl_sliding_window_fixed_pop.py` を作成
2. **検証実験**: 固定対象者で再評価し、結果を比較
3. **ドキュメント更新**: 新しい評価手法をREADMEに記載
4. **論文・報告書**: 手法の違いとその影響を記載

---

## 参考: プロジェクト別評価での同様の問題

プロジェクト別評価（`train_temporal_irl_project_aware.py`）でも同様の問題が発生しています。

### クロスプロジェクトモード

```python
# 現状: レビュアー×プロジェクトの組み合わせごとに軌跡を作成
for reviewer in reviewers_in_history:  # ← 学習期間で変わる
    for project in active_projects:
        trajectories.append(...)
```

→ 学習期間12ヶ月では610人のレビュアー、6ヶ月では393人のレビュアーから軌跡を作成している。

### 解決策

同様に**基準期間での対象者統一**を適用すべきです。

---

## まとめ

### 問題
- 学習期間が変わると予測対象レビュアー数が変化
- 評価の一貫性が欠如

### 原因
- 「学習期間中に活動があったレビュアー」を対象としているため

### 推奨解決策
- **固定基準期間（6ヶ月）での対象者統一**
- すべての学習期間で同じレビュアーを予測対象とする

### 実装すべき項目
1. ✅ `train_temporal_irl_sliding_window_fixed_pop.py` の作成
2. ✅ `train_temporal_irl_project_aware_fixed_pop.py` の作成
3. ✅ 検証実験の実施
4. ✅ ドキュメントとCLAUDE.mdの更新

この改善により、**学習期間の効果を正確に評価**できるようになります。
