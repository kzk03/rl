# 逆強化学習（IRL）の特徴量分析と改善提案

## 現在の特徴量概要

### 1. 状態特徴量（State Features）- 10次元

現在実装されている状態特徴量は以下の通りです（[retention_irl_system.py:267-278](../src/gerrit_retention/rl_prediction/retention_irl_system.py#L267-L278)）：

| No | 特徴量名 | 説明 | 正規化方法 | データソース |
|----|---------|------|-----------|------------|
| 1 | `experience_days` | 開発者の経験日数 | `/365.0` (年単位) | `first_seen` から算出 |
| 2 | `total_changes` | 累積変更数 | `/100.0` | `changes_authored` |
| 3 | `total_reviews` | 累積レビュー数 | `/100.0` | `changes_reviewed` |
| 4 | `project_count` | 参加プロジェクト数 | `/10.0` | `projects` のリスト長 |
| 5 | `recent_activity_frequency` | 最近30日の活動頻度 | そのまま (0-1) | 最近30日の活動数 / 30 |
| 6 | `avg_activity_gap` | 平均活動間隔 | `/30.0` (月単位) | 活動タイムスタンプの差分 |
| 7 | `activity_trend` | 活動トレンド | カテゴリ変換 (0-1) | 最近30日 vs 過去30-60日の比較 |
| 8 | `collaboration_score` | 協力スコア | そのまま (0-1) | レビュー・マージ等の比率 |
| 9 | `code_quality_score` | コード品質スコア | そのまま (0-1) | test/doc/refactor/fixの頻度 |
| 10 | `timestamp_age` | 時間経過 | `/365.0` (年単位) | 現在時刻 - context_date |

### 2. 行動特徴量（Action Features）- 5次元

現在実装されている行動特徴量は以下の通りです（[retention_irl_system.py:296-302](../src/gerrit_retention/rl_prediction/retention_irl_system.py#L296-L302)）：

| No | 特徴量名 | 説明 | 正規化方法 | データソース |
|----|---------|------|-----------|------------|
| 1 | `action_type` | 行動タイプ | カテゴリ変換 (0-1) | commit/review/merge/etc. |
| 2 | `intensity` | 行動の強度 | `min((lines_added + deleted) / (files * 50), 1.0)` | コード変更量 |
| 3 | `quality` | 行動の質 | キーワードベース (0-1) | commit messageの品質キーワード |
| 4 | `collaboration` | 協力度 | タイプベース (0-1) | 行動タイプから推定 |
| 5 | `timestamp_age` | 時間経過 | `/365.0` (年単位) | 現在時刻 - action時刻 |

## OpenStackデータの統計的特性

### データセット概要
- **総レコード数**: 137,632件
- **開発者数（レビュアー）**: 1,379人
- **プロジェクト数**: 5個

### 重要な統計値

#### 1. コード変更の特性
```
change_insertions:
  平均: 184行, 中央値: 27行
  25%: 6行, 75%: 90行
  → 長いテール分布（大規模変更が稀に存在）

change_deletions:
  平均: 88行, 中央値: 5行
  25%: 1行, 75%: 23行
  → 追加より削除が少ない（新機能開発が多い）

change_files_count:
  平均: 4.6ファイル, 中央値: 2ファイル
  25%: 1ファイル, 75%: 4ファイル
  → 小規模変更が主流
```

#### 2. レビュアーの活動特性
```
reviewer_past_reviews_30d:
  平均: 166件, 中央値: 108件
  → 活発なレビュアーが多い

reviewer_past_reviews_180d:
  平均: 821件, 中央値: 545件
  → 半年で500件以上のレビューが標準的

reviewer_assignment_load_30d:
  平均: 48件, 中央値: 39件
  → 月あたり約40件のレビュー依頼を受ける
```

#### 3. レビュアーの経験
```
reviewer_tenure_days:
  平均: 1,502日（約4.1年）, 中央値: 1,491日
  25%: 890日（2.4年）, 75%: 2,092日（5.7年）
  → 長期継続者が多い成熟したコミュニティ

days_since_last_activity:
  中央値: 0日（活動中）
  → データの75%が最近活動している
```

#### 4. レスポンス特性
```
response_latency_days:
  平均: 2.6日, 中央値: 1日
  → 迅速なレスポンスが標準

reviewer_past_response_rate_180d:
  平均: 0.98, 中央値: 1.00
  → ほぼ全てのレビュー依頼に応答している
```

## 問題点と改善機会

### 1. 現在の特徴量の問題点

#### 1.1 データの有効活用不足
現在のIRLシステムは**簡易的な特徴量**しか使用していません：
- OpenStackデータには豊富な特徴量が含まれている（60+カラム）
- しかし、実際に使われているのは基本的な統計量のみ
- **path類似度、相互作用履歴、負荷指標などが未活用**

#### 1.2 時間的文脈の不足
- `timestamp_age`は絶対時間からの経過しか見ていない
- **時間的変化率**（トレンド）が不十分
- 短期（1週間）・中期（1ヶ月）・長期（6ヶ月）の区別がない

#### 1.3 社会的要因の不足
- 協力スコアが簡易的すぎる
- レビュアー間の**相互作用の質**が考慮されていない
- **コミュニティ内での立ち位置**（中心性）が未考慮

#### 1.4 負荷・ストレス要因の不足
- 現在の実装では作業負荷が考慮されていない
- **バーンアウトリスク**が未考慮
- レビュー依頼の**集中度**（短期間に多数の依頼）が未考慮

#### 1.5 正規化の問題
- `/100.0`などの固定値による正規化は、実データの分布を反映していない
  - 例：`total_reviews`の75%値は既に100を超えている
  - 例：`reviewer_past_reviews_180d`の平均は821件（正規化すると8.21）
- **Min-Max正規化**や**標準化**の方が適切

## 追加・修正すべき特徴量の提案

### カテゴリA：時間的特徴（Temporal Features）

#### A1. 活動頻度の多期間比較
**目的**: 短期・中期・長期の活動パターンを捉える

```python
# 追加特徴量
'activity_freq_7d': recent_7days_count / 7.0,      # 週次活動頻度
'activity_freq_30d': recent_30days_count / 30.0,   # 月次活動頻度（既存）
'activity_freq_90d': recent_90days_count / 90.0,   # 四半期活動頻度

# トレンド特徴
'activity_acceleration': (freq_7d - freq_30d) / freq_30d,  # 加速度
'consistency_score': 1.0 - std(weekly_counts) / mean(weekly_counts)  # 一貫性
```

**根拠**:
- OpenStackデータでは`reviewer_past_reviews_30d`, `90d`, `180d`が利用可能
- 短期的な変化が継続予測に重要（論文：Developer Retention in OSS Projects）

#### A2. 活動間隔の分布特徴
**目的**: 活動の規則性を捉える

```python
'activity_gap_std': std(activity_gaps),            # 間隔の標準偏差
'activity_gap_cv': std(activity_gaps) / mean(activity_gaps),  # 変動係数
'max_gap_days': max(activity_gaps),                # 最大空白期間
'recent_gap_trend': recent_gap_avg - overall_gap_avg  # 最近の傾向
```

**根拠**:
- 規則的な活動パターンは継続の強い指標
- OpenStackデータの`days_since_last_activity`を活用可能

### カテゴリB：作業負荷・ストレス特徴（Workload & Stress Features）

#### B1. レビュー負荷指標
**目的**: 過負荷によるバーンアウトリスクを検出

```python
'review_load_7d': reviewer_assignment_load_7d / 7.0,       # 1日あたりレビュー数
'review_load_30d': reviewer_assignment_load_30d / 30.0,
'review_load_trend': (load_7d - load_30d) / load_30d,      # 負荷の増加傾向

# 負荷レベルの離散化
'is_overloaded': 1.0 if load_7d > 5.0 else 0.0,           # 1日5件以上
'is_high_load': 1.0 if load_7d > 2.0 else 0.0,            # 1日2件以上
```

**根拠**:
- OpenStackデータには`reviewer_assignment_load_7d/30d/180d`が利用可能
- 平均30日負荷は48件（1日1.6件）、過負荷の閾値設定が可能

#### B2. バーンアウトリスクスコア
**目的**: 複合的なストレス要因を統合

```python
'burnout_risk': weighted_sum([
    0.4 * normalized_workload,           # 作業負荷
    0.3 * (1.0 - response_rate),         # レスポンス率の低下
    0.3 * review_fragmentation           # レビューの分散度
])
```

**根拠**:
- `reviewer_past_response_rate_180d`が利用可能（平均0.98）
- レスポンス率の低下は継続意欲の低下の指標

#### B3. コード変更の規模・複雑度
**目的**: タスクの難易度を捉える

```python
'avg_change_size': mean(insertions + deletions),          # 平均変更行数
'avg_files_changed': mean(files_count),                   # 平均変更ファイル数
'complexity_score': avg_change_size / avg_files_changed,  # 1ファイルあたり変更量

# 最近のトレンド
'recent_complexity_trend': recent_complexity - overall_complexity
```

**根拠**:
- OpenStackデータ: 平均184行追加、4.6ファイル変更
- 複雑度の増加は負担増の指標

### カテゴリC：社会的・協力特徴（Social & Collaboration Features）

#### C1. 相互作用の深さ
**目的**: レビュアー-作成者間の関係性を捉える

```python
'interaction_count': owner_reviewer_past_interactions_180d,
'interaction_intensity': interactions / max(1, months_active),
'project_specific_interactions': owner_reviewer_project_interactions_180d,
'assignment_history': owner_reviewer_past_assignments_180d
```

**根拠**:
- OpenStackデータに`owner_reviewer_past_interactions_180d`など利用可能
- 強い相互作用は継続の予測因子（Social Capital Theory）

#### C2. コミュニティでの立ち位置
**目的**: 開発者の影響力・中心性を捉える

```python
'review_centrality': reviewer_past_reviews_180d / community_avg_reviews,
'response_reliability': reviewer_past_response_rate_180d,
'tenure_percentile': percentile_rank(reviewer_tenure_days),
'is_core_contributor': 1.0 if review_centrality > 1.5 else 0.0
```

**根拠**:
- OpenStackデータ: レビュー数の分布から中心性を算出可能
- コア貢献者は継続率が高い傾向

#### C3. 協力の質
**目的**: 単なる数ではなく協力の質を評価

```python
'avg_response_time': mean(response_latency_days),         # 平均応答時間
'response_consistency': 1.0 - std(response_times) / mean(response_times),
'collaboration_diversity': unique_collaborators / total_interactions
```

**根拠**:
- OpenStackデータ: `response_latency_days`平均2.6日
- 迅速で一貫した対応は高品質な協力の指標

### カテゴリD：専門性・成長特徴（Expertise & Growth Features）

#### D1. 専門性の一致度
**目的**: タスクと開発者のスキルマッチングを評価

```python
'path_similarity_score': mean([
    path_jaccard_files_project,
    path_jaccard_dir1_project,
    path_jaccard_dir2_project
]),
'path_overlap_score': mean([
    path_overlap_files_project,
    path_overlap_dir1_project,
    path_overlap_dir2_project
])
```

**根拠**:
- OpenStackデータに豊富なpath類似度指標が存在
- スキルマッチングは継続の重要な予測因子

#### D2. 成長指標
**目的**: 開発者の成長を捉える

```python
'review_growth_rate': (recent_reviews - past_reviews) / past_reviews,
'skill_expansion': recent_project_count - past_project_count,
'complexity_progression': recent_avg_complexity - past_avg_complexity
```

**根拠**:
- 成長機会の認識は継続のモチベーション要因

### カテゴリE：プロジェクト特性（Project Features）

#### E1. プロジェクトレベルの特徴
**目的**: プロジェクト固有の文化・特性を捉える

```python
'project_activity_level': project_total_reviews / project_age_days,
'project_contributor_count': unique_reviewers_in_project,
'project_review_load': project_avg_assignment_load,
'project_response_culture': project_avg_response_rate
```

**根拠**:
- プロジェクト文化が継続に影響
- OpenStackは5プロジェクトあり、各々の特性が異なる可能性

## 特徴量の優先順位付け

### 高優先度（すぐに実装すべき）

1. **B1. レビュー負荷指標** - データ利用可能、実装容易、影響大
2. **C1. 相互作用の深さ** - データ利用可能、理論的根拠強い
3. **A1. 活動頻度の多期間比較** - データ利用可能、時系列分析に重要
4. **D1. 専門性の一致度** - データ豊富、未活用領域

### 中優先度（次のフェーズで実装）

5. **A2. 活動間隔の分布特徴** - 規則性の捉え方を改善
6. **C2. コミュニティでの立ち位置** - 算出方法やや複雑
7. **B2. バーンアウトリスクスコア** - 複数特徴の統合が必要
8. **C3. 協力の質** - レスポンス時間の分析が必要

### 低優先度（長期的に検討）

9. **E1. プロジェクト特性** - プロジェクトレベルの集計が必要
10. **D2. 成長指標** - 時系列比較の実装が複雑

## 実装上の推奨事項

### 1. 正規化手法の改善

現在の固定値正規化から、データ駆動型正規化へ：

```python
# 現在（問題あり）
normalized_value = raw_value / 100.0

# 推奨（Min-Max正規化）
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_value = scaler.fit_transform(raw_values.reshape(-1, 1))

# または標準化（平均0、標準偏差1）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_value = scaler.fit_transform(raw_values.reshape(-1, 1))
```

### 2. 特徴量エンジニアリングのパイプライン化

```python
class FeatureExtractor:
    def __init__(self, config):
        self.scalers = {}
        self.config = config

    def extract_temporal_features(self, activity_history):
        """時間的特徴を抽出"""
        pass

    def extract_workload_features(self, developer_data):
        """作業負荷特徴を抽出"""
        pass

    def extract_social_features(self, interaction_data):
        """社会的特徴を抽出"""
        pass

    def extract_expertise_features(self, path_data):
        """専門性特徴を抽出"""
        pass

    def extract_all_features(self, data):
        """全特徴量を抽出して結合"""
        features = []
        features.extend(self.extract_temporal_features(data))
        features.extend(self.extract_workload_features(data))
        features.extend(self.extract_social_features(data))
        features.extend(self.extract_expertise_features(data))
        return np.array(features)
```

### 3. 特徴量の重要度分析

実装後は必ず特徴量の重要度を分析：

```python
# SHAPを使った特徴量重要度分析
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 重要度の可視化
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### 4. 次元数の調整

特徴量を追加すると次元が増えるため：

- **状態特徴量**: 10次元 → 20-25次元を推奨
- **行動特徴量**: 5次元 → 8-10次元を推奨
- LSTMの`hidden_dim`を128から256に増やすことを検討

## 期待される効果

### 定量的改善予測

現在のベスト結果（固定対象者評価）：
- AUC-ROC: 0.900（21ヶ月学習 × 15ヶ月予測）
- AUC-PR: 0.956（3ヶ月学習 × 15ヶ月予測）

特徴量改善後の期待値：
- **AUC-ROC: 0.920-0.940** （+2-4%改善）
- **AUC-PR: 0.965-0.975** （+1-2%改善）
- 特に**短期予測（3-6ヶ月）での改善**が期待される

### 定性的改善

1. **解釈可能性の向上**
   - 負荷・ストレス要因が明示的になる
   - 予測理由がより具体的になる

2. **早期警告の精度向上**
   - バーンアウトリスクの早期検出
   - 過負荷状態の識別

3. **個別化された介入**
   - 開発者ごとの離脱リスク要因が明確に
   - 的確な支援策の提案が可能

## 実装ロードマップ

### Phase 1: 基礎データ活用（2週間）
- [ ] OpenStackデータの全カラムを調査
- [ ] 高優先度特徴量（B1, C1, A1, D1）を実装
- [ ] 正規化手法をMin-Max/StandardScalerに変更
- [ ] 既存モデルと性能比較

### Phase 2: 複合特徴量（2週間）
- [ ] 中優先度特徴量（A2, C2, B2, C3）を実装
- [ ] 特徴量エンジニアリングパイプラインの構築
- [ ] SHAP分析で特徴量重要度を可視化
- [ ] 次元数の最適化（Grid Search）

### Phase 3: 高度な分析（2週間）
- [ ] プロジェクト特性の分析と実装
- [ ] 時系列での特徴量変化の追跡
- [ ] アブレーションスタディ（各特徴量の寄与度分析）
- [ ] 最終モデルの評価とドキュメント化

## 参考文献・理論的背景

1. **Developer Retention in OSS**
   - Schilling et al. (2021): "Predicting Developer Turnover in OSS Projects"
   - 活動頻度・相互作用が継続の主要因子

2. **Burnout in Software Development**
   - Forsgren et al. (2018): "The State of DevOps Report"
   - 過負荷・レスポンス率低下がバーンアウトの指標

3. **Social Capital Theory**
   - Coleman (1988): 社会的相互作用が継続のモチベーション

4. **Task-Person Fit Theory**
   - Edwards (1991): スキルとタスクのマッチングが満足度に影響

## まとめ

現在のIRLシステムは**基礎的な特徴量のみ**を使用しており、OpenStackデータの豊富な情報を活用しきれていません。

**特に重要な改善点**:
1. **作業負荷・ストレス指標の追加** - バーンアウト予測に不可欠
2. **相互作用・協力の深さ** - 社会的要因は継続の強い予測因子
3. **多期間の時間的特徴** - 短期・中期・長期のパターンを区別
4. **専門性の一致度** - 豊富なpath類似度データの活用

これらの改善により、**AUC-ROC 0.92-0.94の達成**が期待され、より**解釈可能で実用的な継続予測システム**が構築できます。
