# データ処理の詳細：ステップ単位とフィルター条件

**最終更新**: 2025-10-17
**プロジェクト**: Gerrit Retention IRL
**データソース**: OpenStack Gerrit (review_requests_openstack_multi_5y_detail.csv)

---

## 📊 データの全体像

### データセットの基本情報

```
データファイル: data/review_requests_openstack_multi_5y_detail.csv
総レコード数: 137,632件
期間: 2012-06-20 ~ 2025-09-27 (約13年間)
ユニークレビュアー数: 1,379人
ユニークオーナー数: 1,034人
カラム数: 65
```

### データの構造

```
各レコード = 1つのレビューリクエスト（review request）

主要カラム:
- change_id: 変更ID
- project: プロジェクト名（例: openstack/nova）
- owner_email: 変更の作成者
- reviewer_email: レビュアー（評価対象）
- request_time: レビューリクエストの日時
- responded_within_days: 応答までの日数
- label: レビュー結果（承認/却下/その他）
```

---

## 🎯 ステップ（Step）の定義

### 「ステップ」とは何か？

**このプロジェクトには「ステップ」という概念は直接的には存在しません。**

代わりに、以下の**時系列単位**でデータを処理します：

### 1. レビューリクエスト（Review Request）

**最小単位**: 1件のレビューリクエスト

```
レコード例:
{
  'change_id': 'openstack%2Fnova~960130',
  'reviewer_email': 'herveberaud.pro@gmail.com',
  'request_time': '2025-09-09T11:48:13',
  'project': 'openstack/nova',
  'label': 0
}
```

**意味**:
- あるオーナーが変更を提出
- その変更に対してレビュアーが割り当てられた
- このイベントが1つのレビューリクエスト

### 2. 軌跡（Trajectory）

**集約単位**: 特定の開発者の一定期間の活動記録

```python
trajectory = {
    'developer': {
        'developer_id': 'alice@example.com',
        'experience_days': 730,
        'total_changes': 120,
        # ... 状態特徴量32次元
    },
    'activity_history': [
        # 最近15個のレビューリクエスト
        {'type': 'review', 'timestamp': '2022-12-20', ...},
        {'type': 'review', 'timestamp': '2022-12-15', ...},
        # ...
    ],
    'continued': True,  # 継続ラベル
    'context_date': datetime(2023, 1, 1)
}
```

**構成要素**:
- **学習期間（History Period）**: 過去N ヶ月の活動（例: 12ヶ月）
- **予測期間（Target Period）**: 未来Mヶ月の活動有無（例: 6ヶ月）
- **スナップショット日（Snapshot Date）**: 学習と予測の境界（例: 2023-01-01）

### 3. エポック（Epoch）

**訓練単位**: ニューラルネットワークの訓練ループ

```
Epoch 0:  全軌跡を1回学習  → 平均損失 = 0.527
Epoch 1:  全軌跡を1回学習  → 平均損失 = 0.510
...
Epoch 30: 全軌跡を1回学習  → 平均損失 = 0.336  (収束)
```

**デフォルト値**: 30エポック

---

## 🔍 データフィルタリング条件

### 現在の実装状況

**重要**: 現在のIRL訓練スクリプトでは、**明示的なフィルタリングは行われていません。**

```python
# train_temporal_irl_sliding_window_fixed_pop.py
def extract_target_reviewers(df, snapshot_date, reference_period_months):
    """基準期間で対象レビュアーを決定"""
    reference_start = snapshot_date - pd.DateOffset(months=reference_period_months)
    reference_df = df[(df['request_time'] >= reference_start) &
                      (df['request_time'] < snapshot_date)]

    # すべてのユニークレビュアーを対象とする
    target_reviewers = set(reference_df['reviewer_email'].unique())

    return target_reviewers  # フィルタリングなし
```

### ボットアカウント検出（実装済みだが未使用）

`data_processing/preprocessing/data_cleaning.py` にボット検出機能が実装されていますが、**IRL訓練では使用されていません**。

```python
class DataCleaner:
    def __init__(self):
        self.bot_patterns = [
            "bot", "automation", "jenkins", "ci", "build", "deploy"
        ]

    def detect_bot_accounts(self, df, email_column='email'):
        """ボットアカウントを検出"""
        bot_mask = pd.Series([False] * len(df), index=df.index)

        # 1. メールアドレスパターンマッチング
        for pattern in self.bot_patterns:
            pattern_mask = df[email_column].str.contains(
                pattern, case=False, na=False, regex=False
            )
            bot_mask |= pattern_mask

        # 2. 異常な高活動量（統計的外れ値）
        if 'changes_authored' in df.columns:
            changes_mean = df['changes_authored'].mean()
            changes_std = df['changes_authored'].std()
            high_activity_threshold = changes_mean + (3 * changes_std)

            high_activity_mask = df['changes_authored'] > high_activity_threshold
            bot_mask |= high_activity_mask

        # 3. 名前がメールアドレスと同じ（自動生成アカウント）
        if 'name' in df.columns:
            auto_name_mask = df['name'] == df[email_column]
            bot_mask |= auto_name_mask

        return bot_mask
```

### OpenStackデータに含まれる可能性のあるボット

```
サンプルメールアドレスから推測:
- DL-ARC-InfoScale-OpenStack-CI@arctera.io     ← CI関連（可能性高）
- openstack@lightbitslabs.com                   ← 組織メール（bot可能性）
- openstack-ci@sap.com                          ← CI関連（可能性高）
```

---

## 📈 軌跡抽出の詳細プロセス

### ステップ1: 対象レビュアーの決定

```python
# 基準期間で活動があったレビュアーを抽出
snapshot_date = datetime(2023, 1, 1)
reference_period = 6  # ヶ月

# 2022-07-01 ~ 2023-01-01 の期間で活動があったレビュアー
reference_start = datetime(2022, 7, 1)
reference_df = df[(df['request_time'] >= reference_start) &
                  (df['request_time'] < snapshot_date)]

target_reviewers = set(reference_df['reviewer_email'].unique())
# 結果: 291人（OpenStack 2023-01-01 基準）
```

**目的**:
- すべての学習期間で同じレビュアーを対象とする
- 公平な比較のため

### ステップ2: 軌跡の抽出

```python
def extract_trajectories_with_fixed_population(
    df, snapshot_date, history_months, target_months, target_reviewers
):
    """固定対象者での軌跡抽出"""

    # 学習期間のデータ
    history_start = snapshot_date - pd.DateOffset(months=history_months)
    history_df = df[(df['request_time'] >= history_start) &
                    (df['request_time'] < snapshot_date)]

    # 予測期間のデータ
    target_end = snapshot_date + pd.DateOffset(months=target_months)
    target_df = df[(df['request_time'] >= snapshot_date) &
                   (df['request_time'] < target_end)]

    trajectories = []
    skipped_no_history = 0

    for reviewer in target_reviewers:
        reviewer_history = history_df[history_df['reviewer_email'] == reviewer]
        reviewer_target = target_df[target_df['reviewer_email'] == reviewer]

        # フィルター条件1: 学習期間に活動なし → スキップ
        if len(reviewer_history) == 0:
            skipped_no_history += 1
            continue

        # 継続ラベル: 予測期間中に活動があったか
        continued = len(reviewer_target) > 0

        # 活動履歴を構築
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity_history.append({
                'type': 'review',
                'timestamp': row['request_time'],
                'project': row.get('project', 'unknown'),
                # ...
            })

        trajectories.append({
            'developer': {...},
            'activity_history': activity_history,
            'continued': continued,
            'context_date': snapshot_date
        })

    return trajectories
```

### ステップ3: 訓練/テスト分割

```python
# 80/20 分割（ランダムシャッフル）
np.random.seed(42)
np.random.shuffle(trajectories)
split_idx = int(len(trajectories) * 0.8)

train_trajectories = trajectories[:split_idx]
test_trajectories = trajectories[split_idx:]

# 例: 291人 → 232人（訓練）+ 59人（テスト）
```

---

## 🎛️ 現在のフィルター条件まとめ

### ✅ 実施されているフィルター

| # | フィルター条件 | タイミング | 除外数 | 理由 |
|---|--------------|-----------|-------|------|
| 1 | **基準期間に活動なし** | 対象レビュアー決定時 | 不明 | 固定母集団の定義 |
| 2 | **学習期間に活動なし** | 軌跡抽出時 | 例: 101人/291人 | 学習データなし |

### ❌ 実施されていないフィルター

| # | フィルター候補 | 現状 | 影響 |
|---|--------------|------|------|
| 1 | **ボットアカウント除外** | 未実施 | CI botなどが含まれる可能性 |
| 2 | **最小活動回数** | 未実施 | 1回のみの活動者も含む |
| 3 | **異常な高活動量** | 未実施 | 外れ値が含まれる可能性 |
| 4 | **プロジェクト限定** | 未実施 | 全プロジェクト対象 |

---

## 📊 実際のフィルター効果（例: 12m × 6m, 2023-01-01）

```
【ベースラインIRL実験結果】
1. 基準期間（6ヶ月）で対象レビュアー決定
   → 291人

2. 学習期間（12ヶ月）で軌跡抽出
   - 対象レビュアー総数: 291人
   - 学習期間に活動なしでスキップ: 0人
   - 作成された軌跡数: 291件

3. 訓練/テスト分割
   - 訓練: 232件
   - テスト: 59件

4. 継続率
   - 訓練: 65.9%
   - テスト: 44.8%
```

---

## 🔧 推奨されるフィルター改善

### 優先度★★★（実装推奨）

#### 1. ボットアカウント除外

```python
def filter_bot_accounts(df, reviewer_col='reviewer_email'):
    """ボットアカウントを除外"""
    bot_patterns = ['bot', 'ci', 'automation', 'jenkins', 'build']

    bot_mask = pd.Series([False] * len(df), index=df.index)
    for pattern in bot_patterns:
        bot_mask |= df[reviewer_col].str.contains(pattern, case=False, na=False)

    logger.info(f"ボットアカウント除外: {bot_mask.sum()}件")
    return df[~bot_mask]

# 使用例
df_filtered = filter_bot_accounts(df)
target_reviewers = extract_target_reviewers(df_filtered, ...)
```

**理由**:
- CI bot は人間の行動パターンと異なる
- 予測精度に悪影響を与える可能性
- OpenStackデータに CI関連メールアドレスが含まれている

#### 2. 最小活動回数フィルター

```python
def filter_min_activity(df, target_reviewers, min_count=5):
    """最小活動回数でフィルター"""
    activity_counts = df['reviewer_email'].value_counts()

    active_reviewers = set(
        activity_counts[activity_counts >= min_count].index
    )

    filtered_reviewers = target_reviewers & active_reviewers

    logger.info(f"最小活動回数フィルター（{min_count}件以上）:")
    logger.info(f"  除外前: {len(target_reviewers)}人")
    logger.info(f"  除外後: {len(filtered_reviewers)}人")

    return filtered_reviewers
```

**理由**:
- 1-2回のみの活動者は統計的に信頼性が低い
- 継続予測が困難

### 優先度★★（検討推奨）

#### 3. 異常値除外（外れ値）

```python
def filter_outliers(df, reviewer_col='reviewer_email', std_threshold=3):
    """異常な高活動量を除外"""
    activity_counts = df[reviewer_col].value_counts()

    mean = activity_counts.mean()
    std = activity_counts.std()
    threshold = mean + (std_threshold * std)

    outliers = set(activity_counts[activity_counts > threshold].index)

    logger.info(f"外れ値除外（{threshold:.0f}件以上）: {len(outliers)}人")

    return df[~df[reviewer_col].isin(outliers)]
```

**理由**:
- 極端に多い活動者（bot可能性、専任レビュアー）は一般的でない
- モデルが過学習する可能性

### 優先度★（オプション）

#### 4. プロジェクト限定

```python
def filter_by_project(df, projects=['openstack/nova', 'openstack/neutron']):
    """特定プロジェクトに限定"""
    return df[df['project'].isin(projects)]
```

**理由**:
- プロジェクトごとの文化・パターンが異なる
- プロジェクト特化型モデルを作成する場合

---

## 🔬 フィルター効果のシミュレーション

### ケース1: ボットアカウント除外

```
推定効果:
- 除外数: ~50-100人（総1,379人の5-10%）
- AUC-ROC への影響: +1-3%（ノイズ除去効果）
```

### ケース2: 最小活動回数（5件以上）

```
推定効果:
- 除外数: ~200-400人（総1,379人の15-30%）
- AUC-ROC への影響: +2-5%（低品質データ除去）
- サンプルサイズ減少: -20-30%
```

### ケース3: 両方適用

```
推定効果:
- 除外数: ~250-500人（総1,379人の18-35%）
- AUC-ROC への影響: +3-7%（複合効果）
- データ品質: 大幅改善
- サンプルサイズ: 879-1,129人（依然十分）
```

---

## 📝 実装ガイド

### フィルター適用スクリプト例

```python
# scripts/preprocessing/filter_openstack_data.py
import pandas as pd
from pathlib import Path

def apply_filters(input_csv, output_csv):
    """OpenStackデータにフィルターを適用"""
    df = pd.read_csv(input_csv)

    print(f"元データ: {len(df):,}件, {df['reviewer_email'].nunique()}人")

    # 1. ボットアカウント除外
    bot_patterns = ['bot', 'ci', 'automation', 'jenkins', 'build']
    bot_mask = pd.Series([False] * len(df), index=df.index)
    for pattern in bot_patterns:
        bot_mask |= df['reviewer_email'].str.contains(pattern, case=False, na=False)

    df = df[~bot_mask]
    print(f"ボット除外後: {len(df):,}件, {df['reviewer_email'].nunique()}人")

    # 2. 最小活動回数フィルター
    min_activity = 5
    activity_counts = df['reviewer_email'].value_counts()
    active_reviewers = set(activity_counts[activity_counts >= min_activity].index)

    df = df[df['reviewer_email'].isin(active_reviewers)]
    print(f"最小活動{min_activity}件フィルター後: {len(df):,}件, {df['reviewer_email'].nunique()}人")

    # 3. 保存
    df.to_csv(output_csv, index=False)
    print(f"フィルター済みデータを保存: {output_csv}")

if __name__ == '__main__':
    apply_filters(
        'data/review_requests_openstack_multi_5y_detail.csv',
        'data/review_requests_openstack_filtered.csv'
    )
```

### 使用方法

```bash
# フィルター適用
uv run python scripts/preprocessing/filter_openstack_data.py

# フィルター済みデータで訓練
uv run python scripts/training/irl/train_temporal_irl_sliding_window_fixed_pop.py \
  --reviews data/review_requests_openstack_filtered.csv \
  --snapshot-date 2023-01-01 \
  --reference-period 6 \
  --history-months 12 \
  --target-months 6 \
  --sequence --seq-len 15 --epochs 30 \
  --output importants/irl_filtered_12m_6m
```

---

## 🎯 まとめ

### 現状の理解

1. **ステップ単位**: レビューリクエスト（最小）→ 軌跡（集約）→ エポック（訓練）
2. **フィルター条件**:
   - 実施済み: 基準期間活動なし、学習期間活動なし
   - 未実施: ボット除外、最小活動回数、外れ値除外
3. **データ品質**: ボット混入の可能性あり

### 推奨アクション

#### 短期（優先度高）
1. ✅ ボットアカウント除外スクリプトを作成

#### 中期（優先度中）
5. プロジェクト別フィルターの検討



---

## 📚 関連ドキュメント

- [IRL_COMPREHENSIVE_GUIDE.md](IRL_COMPREHENSIVE_GUIDE.md): IRL全体の説明
- [seq_len_explanation.md](seq_len_explanation.md): シーケンス長の詳細
- [project_aware_irl_evaluation.md](project_aware_irl_evaluation.md): プロジェクト別評価

---

**最終更新**: 2025-10-17
**作成者**: Claude + Kazuki-h
**ステータス**: 初版完成、フィルター実装待ち
