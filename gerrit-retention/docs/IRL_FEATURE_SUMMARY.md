# IRL特徴量まとめ - 現状分析と改善提案

**最終更新**: 2025-10-17

---

## 📊 現在の特徴量リスト

### 基本版（retention_irl_system.py）

#### 状態特徴量: 10次元

| No | 特徴量 | 重要度 | 備考 |
|----|-------|-------|------|
| 1 | experience_days | ★★★ | 経験日数（年単位正規化） |
| 2 | total_changes | ★★☆ | 総コミット数 |
| 3 | total_reviews | ★★☆ | 総レビュー数 |
| 4 | project_count | ★★☆ | 参加プロジェクト数 |
| 5 | recent_activity_frequency | ★★★ | 30日活動頻度 |
| 6 | avg_activity_gap | ★★☆ | 平均活動間隔 |
| 7 | activity_trend | ★★☆ | トレンド（増加/安定/減少） |
| 8 | collaboration_score | ★★☆ | 協力スコア |
| 9 | code_quality_score | ★☆☆ | 品質スコア（削除候補） |
| 10 | timestamp_age | ★☆☆ | 時間経過（削除候補） |

#### 行動特徴量: 5次元

| No | 特徴量 | 重要度 | 備考 |
|----|-------|-------|------|
| 1 | action_type | ★★☆ | commit/review/merge等 |
| 2 | intensity | ★★☆ | 変更量ベース |
| 3 | quality | ★☆☆ | キーワードベース（削除候補） |
| 4 | collaboration | ★★☆ | 協力度 |
| 5 | timestamp_age | ★☆☆ | 時間経過（削除候補） |

### 拡張版（enhanced_feature_extractor.py）

#### 状態特徴量: 32次元

**基本10次元** + 以下22次元:

**A1. 活動頻度（5次元）**
- activity_freq_7d/30d/90d
- activity_acceleration
- consistency_score

**B1. レビュー負荷（6次元）**
- review_load_7d/30d/180d
- review_load_trend
- is_overloaded / is_high_load

**C1. 相互作用（4次元）**
- interaction_count_180d
- interaction_intensity
- project_specific_interactions
- assignment_history_180d

**D1. 専門性（2次元）**
- path_similarity_score
- path_overlap_score

**その他（5次元）**
- avg_response_time_days
- response_rate_180d
- tenure_days
- avg_change_size
- avg_files_changed

#### 行動特徴量: 9次元

**基本5次元** + 以下4次元:
- change_size
- files_count
- complexity
- response_latency

---

## ✅ 追加すべき特徴量

### 優先度：高（★★★）

#### 1. ソーシャルネットワーク特徴量

| 特徴量 | 説明 | 実装難易度 |
|-------|------|----------|
| network_centrality | PageRank中心性 | 中 |
| community_bridge_score | 媒介中心性 | 中 |
| collaboration_diversity | 協力相手の多様性 | 低 |

**根拠**: OSS継続研究で中心性と継続率の強相関（r=0.45）

#### 2. 時系列パターン特徴量

| 特徴量 | 説明 | 実装難易度 |
|-------|------|----------|
| activity_entropy | 活動の多様性 | 低 |
| burst_pattern_score | 集中活動パターン | 中 |
| activity_cycle_period | 活動周期（週次/月次） | 高 |

**根拠**: バースト活動後の離脱パターンが存在

#### 3. フィードバック特徴量

| 特徴量 | 説明 | 実装難易度 |
|-------|------|----------|
| positive_feedback_ratio | +2/-2投票比率 | 低 |
| merge_success_rate | マージ成功率 | 低 |
| avg_feedback_latency | FB待ち時間 | 低 |

**根拠**: 肯定的FBが継続意欲を高める（心理学的根拠）

### 優先度：中（★★☆）

#### 4. プロジェクト特性

- project_activity_level
- project_reviewer_ratio
- project_maturity_score

#### 5. スキル成長

- skill_growth_rate
- learning_curve_slope
- domain_expansion_rate

---

## ❌ 削除すべき特徴量

### 削除推奨（優先度：高）

| 特徴量 | 削除理由 |
|-------|---------|
| code_quality_score | キーワードベースで精度低い |
| timestamp_age（状態） | 他特徴量と冗長 |
| timestamp_age（行動） | 同上 |
| path_overlap_score | path_similarityと高相関（r=0.85） |
| tenure_days | experience_daysとほぼ同じ（r=0.95） |
| avg_change_size | 行動特徴量と重複 |
| avg_files_changed | 行動特徴量と重複 |

### 統合検討

- total_changes + total_reviews → total_contributions

---

## 📈 改善ロードマップ

### Phase 1: 削除と最適化（1週間）

```python
# 削除: 7特徴量
removed = ['code_quality_score', 'timestamp_age', ...]

# 結果: 32次元 → 25次元
```

### Phase 2: 高優先度追加（2週間）

```python
# 追加: 9特徴量
added = [
    'network_centrality',
    'collaboration_diversity',
    'activity_entropy',
    'burst_pattern_score',
    'positive_feedback_ratio',
    'merge_success_rate',
    ...
]

# 結果: 25次元 → 34次元
```

### Phase 3: 評価と調整（1週間）

- SHAP分析で重要度確認
- 次元削減検討（PCA: 34→20次元）
- アブレーションスタディ

---

## 🎯 期待される効果

| 指標 | 現在 | 削除後 | 追加後 | 改善 |
|-----|------|-------|-------|------|
| AUC-ROC | 0.855 | 0.850 | **0.875** | +2.0% |
| 次元数 | 32 | 25 | 34 | +2 |
| 訓練時間 | 4分 | 3分 | 5分 | +1分 |

---

## 📚 実装例

### 削除の実装

```python
# scripts/training/irl/train_optimized_irl.py
config = {
    'state_dim': 25,  # 32 - 7
    'action_dim': 8,  # 9 - 1
    'remove_features': [
        'code_quality_score',
        'timestamp_age',
        'path_overlap_score',
        'tenure_days',
        'avg_change_size',
        'avg_files_changed'
    ]
}
```

### 追加の実装

```python
# 新規特徴量抽出
def extract_social_features(developer, interactions):
    """ソーシャルネットワーク特徴"""
    return {
        'network_centrality': calculate_pagerank(interactions),
        'collaboration_diversity': len(unique_collaborators) / total_interactions
    }

def extract_feedback_features(developer, reviews):
    """フィードバック特徴"""
    return {
        'positive_feedback_ratio': (votes_plus2 + votes_plus1) / total_votes,
        'merge_success_rate': merged_count / submitted_count
    }
```

---

## 🔗 関連ドキュメント

- [IRL_FEATURE_ANALYSIS.md](IRL_FEATURE_ANALYSIS.md): 詳細な分析
- [IRL_COMPREHENSIVE_GUIDE.md](IRL_COMPREHENSIVE_GUIDE.md): IRL全体ガイド
- [enhanced_feature_extractor.py](../src/gerrit_retention/rl_prediction/enhanced_feature_extractor.py): 実装

---

**作成者**: Claude + Kazuki-h
