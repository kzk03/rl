# 逆強化学習特徴量見直し - 設計文書

## 概要

本設計文書では、Kazoo プロジェクトの逆強化学習（IRL）コンポーネントで使用する特徴量の包括的な見直しと改善を定義する。現在の特徴量構成を分析し、より効果的な開発者-タスクマッチングを実現するための新しい特徴量設計、最適化手法、および実装アーキテクチャを提案する。

## アーキテクチャ

### システム全体構成

```
┌─────────────────────────────────────────────────────────────┐
│                    IRL Feature Redesign System              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Feature Analysis│  │ Feature Design  │  │ Feature      │ │
│  │ Module          │  │ Module          │  │ Optimization │ │
│  │                 │  │                 │  │ Module       │ │
│  │ - Importance    │  │ - New Features  │  │ - Selection  │ │
│  │ - Correlation   │  │ - Improvement   │  │ - Scaling    │ │
│  │ - Distribution  │  │ - Combination   │  │ - Reduction  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ GAT Enhancement │  │ Feature Pipeline│  │ Evaluation   │ │
│  │ Module          │  │ Module          │  │ Module       │ │
│  │                 │  │                 │  │              │ │
│  │ - Embedding Opt │  │ - Automation    │  │ - A/B Test   │ │
│  │ - Interpretation│  │ - Configuration │  │ - Metrics    │ │
│  │ - Visualization │  │ - Quality Check │  │ - Validation │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### データフロー

```
Raw Data → Feature Extraction → Feature Analysis → Feature Design →
Feature Optimization → GAT Enhancement → Feature Pipeline →
IRL Training → Evaluation → Feedback Loop
```

## コンポーネントと インターフェース

### 1. Feature Analysis Module

#### 1.1 FeatureImportanceAnalyzer

```python
class FeatureImportanceAnalyzer:
    """特徴量重要度分析器"""

    def __init__(self, irl_weights_path: str, feature_names: List[str]):
        self.weights = np.load(irl_weights_path)
        self.feature_names = feature_names

    def analyze_importance(self) -> Dict[str, Any]:
        """重要度分析を実行"""
        return {
            'importance_ranking': self._rank_by_importance(),
            'category_importance': self._analyze_by_category(),
            'statistical_significance': self._test_significance()
        }

    def _rank_by_importance(self) -> List[Tuple[str, float]]:
        """重要度ランキング作成"""
        pass

    def _analyze_by_category(self) -> Dict[str, float]:
        """カテゴリ別重要度分析"""
        pass

    def _test_significance(self) -> Dict[str, float]:
        """統計的有意性検証"""
        pass
```

#### 1.2 FeatureCorrelationAnalyzer

```python
class FeatureCorrelationAnalyzer:
    """特徴量相関分析器"""

    def __init__(self, feature_data: np.ndarray, feature_names: List[str]):
        self.feature_data = feature_data
        self.feature_names = feature_names

    def analyze_correlations(self) -> Dict[str, Any]:
        """相関分析を実行"""
        return {
            'correlation_matrix': self._compute_correlation_matrix(),
            'high_correlation_pairs': self._find_high_correlations(),
            'redundant_features': self._identify_redundant_features()
        }

    def _compute_correlation_matrix(self) -> np.ndarray:
        """相関行列計算"""
        pass

    def _find_high_correlations(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """高相関ペア特定"""
        pass

    def _identify_redundant_features(self) -> List[str]:
        """冗長特徴量特定"""
        pass
```

#### 1.3 FeatureDistributionAnalyzer

```python
class FeatureDistributionAnalyzer:
    """特徴量分布分析器"""

    def __init__(self, feature_data: np.ndarray, feature_names: List[str]):
        self.feature_data = feature_data
        self.feature_names = feature_names

    def analyze_distributions(self) -> Dict[str, Any]:
        """分布分析を実行"""
        return {
            'distribution_stats': self._compute_distribution_stats(),
            'normality_tests': self._test_normality(),
            'outlier_detection': self._detect_outliers(),
            'scale_imbalance': self._detect_scale_imbalance()
        }

    def _compute_distribution_stats(self) -> Dict[str, Dict[str, float]]:
        """分布統計計算"""
        pass

    def _test_normality(self) -> Dict[str, float]:
        """正規性検定"""
        pass

    def _detect_outliers(self) -> Dict[str, List[int]]:
        """外れ値検出"""
        pass

    def _detect_scale_imbalance(self) -> Dict[str, float]:
        """スケール不均衡検出"""
        pass
```

### 2. Feature Design Module

#### 2.1 TaskFeatureDesigner

```python
class TaskFeatureDesigner:
    """タスク特徴量設計器"""

    def __init__(self, cfg):
        self.cfg = cfg

    def design_enhanced_task_features(self, task, env) -> Dict[str, float]:
        """強化されたタスク特徴量を設計"""
        features = {}

        # 既存特徴量の改良
        features.update(self._improve_existing_features(task, env))

        # 新規特徴量の追加
        features.update(self._add_urgency_features(task))
        features.update(self._add_complexity_features(task))
        features.update(self._add_social_attention_features(task))

        return features

    def _improve_existing_features(self, task, env) -> Dict[str, float]:
        """既存特徴量の改良"""
        return {
            'task_days_since_last_activity_log': np.log1p(self._get_days_since_activity(task, env)),
            'task_discussion_activity_normalized': self._normalize_discussion_activity(task),
            'task_text_complexity_score': self._compute_text_complexity(task),
            'task_code_complexity_score': self._compute_code_complexity(task)
        }

    def _add_urgency_features(self, task) -> Dict[str, float]:
        """緊急度特徴量追加"""
        return {
            'task_has_priority_label': float(self._has_priority_label(task)),
            'task_has_deadline': float(self._has_deadline(task)),
            'task_milestone_proximity': self._compute_milestone_proximity(task),
            'task_blocking_issues_count': float(self._count_blocking_issues(task))
        }

    def _add_complexity_features(self, task) -> Dict[str, float]:
        """複雑度特徴量追加"""
        return {
            'task_technical_term_density': self._compute_technical_term_density(task),
            'task_reference_links_count': float(self._count_reference_links(task)),
            'task_estimated_effort': self._estimate_effort(task),
            'task_dependency_count': float(self._count_dependencies(task))
        }

    def _add_social_attention_features(self, task) -> Dict[str, float]:
        """社会的注目度特徴量追加"""
        return {
            'task_watchers_count': float(getattr(task, 'watchers_count', 0)),
            'task_reactions_count': float(self._count_reactions(task)),
            'task_mentions_count': float(self._count_mentions(task)),
            'task_external_references': float(self._count_external_references(task))
        }
```

#### 2.2 DeveloperFeatureDesigner

```python
class DeveloperFeatureDesigner:
    """開発者特徴量設計器"""

    def __init__(self, cfg):
        self.cfg = cfg

    def design_enhanced_developer_features(self, developer, env) -> Dict[str, float]:
        """強化された開発者特徴量を設計"""
        features = {}

        # 既存特徴量の改良
        features.update(self._improve_existing_features(developer, env))

        # 新規特徴量の追加
        features.update(self._add_expertise_features(developer))
        features.update(self._add_activity_pattern_features(developer, env))
        features.update(self._add_quality_features(developer))

        return features

    def _improve_existing_features(self, developer, env) -> Dict[str, float]:
        """既存特徴量の改良"""
        profile = developer['profile']
        return {
            'dev_recent_activity_count_log': np.log1p(len(env.dev_action_history.get(developer['name'], []))),
            'dev_workload_ratio': self._compute_workload_ratio(developer, env),
            'dev_lines_changed_per_commit': self._compute_lines_per_commit(profile),
            'dev_collaboration_efficiency': self._compute_collaboration_efficiency(profile)
        }

    def _add_expertise_features(self, developer) -> Dict[str, float]:
        """専門性特徴量追加"""
        profile = developer['profile']
        return {
            'dev_primary_language_strength': self._compute_language_strength(profile),
            'dev_domain_expertise_score': self._compute_domain_expertise(profile),
            'dev_technology_diversity': self._compute_tech_diversity(profile),
            'dev_learning_velocity': self._compute_learning_velocity(profile)
        }

    def _add_activity_pattern_features(self, developer, env) -> Dict[str, float]:
        """活動パターン特徴量追加"""
        return {
            'dev_preferred_time_zone': self._compute_timezone_preference(developer, env),
            'dev_weekday_activity_ratio': self._compute_weekday_ratio(developer, env),
            'dev_response_time_avg': self._compute_avg_response_time(developer, env),
            'dev_consistency_score': self._compute_consistency_score(developer, env)
        }

    def _add_quality_features(self, developer) -> Dict[str, float]:
        """品質特徴量追加"""
        profile = developer['profile']
        return {
            'dev_pr_merge_rate': profile.get('pr_merge_rate', 0.0),
            'dev_review_approval_rate': profile.get('review_approval_rate', 0.0),
            'dev_bug_introduction_rate': profile.get('bug_introduction_rate', 0.0),
            'dev_code_review_quality': profile.get('code_review_quality', 0.0)
        }
```

#### 2.3 MatchingFeatureDesigner

```python
class MatchingFeatureDesigner:
    """マッチング特徴量設計器"""

    def __init__(self, cfg):
        self.cfg = cfg

    def design_enhanced_matching_features(self, task, developer, env) -> Dict[str, float]:
        """強化されたマッチング特徴量を設計"""
        features = {}

        # 既存特徴量の改良
        features.update(self._improve_existing_features(task, developer, env))

        # 新規特徴量の追加
        features.update(self._add_temporal_proximity_features(task, developer, env))
        features.update(self._add_technical_compatibility_features(task, developer))
        features.update(self._add_success_history_features(task, developer, env))

        return features

    def _improve_existing_features(self, task, developer, env) -> Dict[str, float]:
        """既存特徴量の改良"""
        return {
            'match_collaboration_strength': self._compute_collaboration_strength(task, developer, env),
            'match_skill_compatibility_score': self._compute_skill_compatibility(task, developer),
            'match_file_expertise_relevance': self._compute_file_expertise_relevance(task, developer),
            'match_workload_balance_score': self._compute_workload_balance(task, developer, env)
        }

    def _add_temporal_proximity_features(self, task, developer, env) -> Dict[str, float]:
        """時間的近接性特徴量追加"""
        return {
            'match_recent_collaboration_days': self._compute_recent_collaboration_days(task, developer, env),
            'match_activity_time_overlap': self._compute_activity_time_overlap(task, developer, env),
            'match_timezone_compatibility': self._compute_timezone_compatibility(task, developer),
            'match_response_time_prediction': self._predict_response_time(task, developer, env)
        }

    def _add_technical_compatibility_features(self, task, developer) -> Dict[str, float]:
        """技術的適合性特徴量追加"""
        return {
            'match_tech_stack_overlap': self._compute_tech_stack_overlap(task, developer),
            'match_language_proficiency': self._compute_language_proficiency(task, developer),
            'match_framework_experience': self._compute_framework_experience(task, developer),
            'match_architecture_familiarity': self._compute_architecture_familiarity(task, developer)
        }

    def _add_success_history_features(self, task, developer, env) -> Dict[str, float]:
        """成功履歴特徴量追加"""
        return {
            'match_past_success_rate': self._compute_past_success_rate(task, developer, env),
            'match_similar_task_completion': self._compute_similar_task_completion(task, developer, env),
            'match_collaboration_satisfaction': self._compute_collaboration_satisfaction(task, developer, env),
            'match_estimated_success_probability': self._estimate_success_probability(task, developer, env)
        }
```

### 3. Feature Optimization Module

#### 3.1 FeatureScaler

```python
class FeatureScaler:
    """特徴量スケーリング器"""

    def __init__(self, scaling_config: Dict[str, str]):
        self.scaling_config = scaling_config
        self.scalers = {}

    def fit_transform(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """スケーリングを適用"""
        scaled_features = features.copy()

        for i, name in enumerate(feature_names):
            scaling_method = self.scaling_config.get(name, 'standard')

            if scaling_method == 'log':
                scaled_features[:, i] = np.log1p(np.maximum(features[:, i], 0))
            elif scaling_method == 'standard':
                scaler = StandardScaler()
                scaled_features[:, i] = scaler.fit_transform(features[:, i].reshape(-1, 1)).flatten()
                self.scalers[name] = scaler
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
                scaled_features[:, i] = scaler.fit_transform(features[:, i].reshape(-1, 1)).flatten()
                self.scalers[name] = scaler
            elif scaling_method == 'robust':
                scaler = RobustScaler()
                scaled_features[:, i] = scaler.fit_transform(features[:, i].reshape(-1, 1)).flatten()
                self.scalers[name] = scaler

        return scaled_features

    def transform(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """学習済みスケーラーで変換"""
        pass
```

#### 3.2 FeatureSelector

```python
class FeatureSelector:
    """特徴量選択器"""

    def __init__(self, selection_config: Dict[str, Any]):
        self.selection_config = selection_config
        self.selected_features = None

    def select_features(self, features: np.ndarray, targets: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """特徴量選択を実行"""
        method = self.selection_config.get('method', 'importance')

        if method == 'univariate':
            return self._univariate_selection(features, targets, feature_names)
        elif method == 'rfe':
            return self._recursive_feature_elimination(features, targets, feature_names)
        elif method == 'l1':
            return self._l1_regularization_selection(features, targets, feature_names)
        elif method == 'importance':
            return self._importance_based_selection(features, targets, feature_names)
        else:
            return features, feature_names

    def _univariate_selection(self, features: np.ndarray, targets: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """単変量統計による選択"""
        pass

    def _recursive_feature_elimination(self, features: np.ndarray, targets: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """再帰的特徴量除去"""
        pass

    def _l1_regularization_selection(self, features: np.ndarray, targets: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """L1正則化による選択"""
        pass

    def _importance_based_selection(self, features: np.ndarray, targets: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """重要度ベース選択"""
        pass
```

#### 3.3 DimensionReducer

```python
class DimensionReducer:
    """次元削減器"""

    def __init__(self, reduction_config: Dict[str, Any]):
        self.reduction_config = reduction_config
        self.reducer = None

    def reduce_dimensions(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """次元削減を実行"""
        method = self.reduction_config.get('method', 'pca')

        if method == 'pca':
            return self._pca_reduction(features, target_dim)
        elif method == 'umap':
            return self._umap_reduction(features, target_dim)
        elif method == 'tsne':
            return self._tsne_reduction(features, target_dim)
        else:
            return features

    def _pca_reduction(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """PCA次元削減"""
        pass

    def _umap_reduction(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """UMAP次元削減"""
        pass

    def _tsne_reduction(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """t-SNE次元削減"""
        pass
```

### 4. GAT Enhancement Module

#### 4.1 GATOptimizer

```python
class GATOptimizer:
    """GAT最適化器"""

    def __init__(self, gat_config: Dict[str, Any]):
        self.gat_config = gat_config
        self.optimal_dim = None

    def optimize_gat_features(self, gat_extractor: GATFeatureExtractor) -> Dict[str, Any]:
        """GAT特徴量を最適化"""
        return {
            'optimal_dimensions': self._optimize_dimensions(gat_extractor),
            'attention_analysis': self._analyze_attention_weights(gat_extractor),
            'embedding_quality': self._evaluate_embedding_quality(gat_extractor)
        }

    def _optimize_dimensions(self, gat_extractor: GATFeatureExtractor) -> int:
        """最適次元数を決定"""
        pass

    def _analyze_attention_weights(self, gat_extractor: GATFeatureExtractor) -> Dict[str, Any]:
        """アテンション重み分析"""
        pass

    def _evaluate_embedding_quality(self, gat_extractor: GATFeatureExtractor) -> Dict[str, float]:
        """埋め込み品質評価"""
        pass
```

#### 4.2 GATInterpreter

```python
class GATInterpreter:
    """GAT解釈器"""

    def __init__(self, gat_extractor: GATFeatureExtractor):
        self.gat_extractor = gat_extractor

    def interpret_gat_features(self) -> Dict[str, Any]:
        """GAT特徴量を解釈"""
        return {
            'dimension_meanings': self._interpret_dimensions(),
            'important_patterns': self._identify_important_patterns(),
            'collaboration_insights': self._extract_collaboration_insights()
        }

    def _interpret_dimensions(self) -> Dict[int, str]:
        """各次元の意味を解釈"""
        pass

    def _identify_important_patterns(self) -> List[Dict[str, Any]]:
        """重要なパターンを特定"""
        pass

    def _extract_collaboration_insights(self) -> Dict[str, Any]:
        """協力関係の洞察を抽出"""
        pass
```

### 5. Feature Pipeline Module

#### 5.1 FeaturePipeline

```python
class FeaturePipeline:
    """特徴量パイプライン"""

    def __init__(self, pipeline_config: Dict[str, Any]):
        self.config = pipeline_config
        self.components = self._initialize_components()

    def _initialize_components(self) -> Dict[str, Any]:
        """コンポーネントを初期化"""
        return {
            'analyzer': FeatureImportanceAnalyzer(
                self.config['irl_weights_path'],
                self.config['feature_names']
            ),
            'designer': self._create_feature_designers(),
            'optimizer': self._create_optimizers(),
            'gat_enhancer': GATOptimizer(self.config.get('gat', {}))
        }

    def execute_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """パイプラインを実行"""
        results = {}

        # 1. 特徴量分析
        results['analysis'] = self._run_analysis(input_data)

        # 2. 特徴量設計
        results['design'] = self._run_design(input_data, results['analysis'])

        # 3. 特徴量最適化
        results['optimization'] = self._run_optimization(input_data, results['design'])

        # 4. GAT強化
        results['gat_enhancement'] = self._run_gat_enhancement(input_data, results['optimization'])

        # 5. 評価
        results['evaluation'] = self._run_evaluation(results)

        return results

    def _run_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析フェーズ実行"""
        pass

    def _run_design(self, input_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """設計フェーズ実行"""
        pass

    def _run_optimization(self, input_data: Dict[str, Any], design_results: Dict[str, Any]) -> Dict[str, Any]:
        """最適化フェーズ実行"""
        pass

    def _run_gat_enhancement(self, input_data: Dict[str, Any], optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """GAT強化フェーズ実行"""
        pass

    def _run_evaluation(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """評価フェーズ実行"""
        pass
```

## データモデル

### 特徴量メタデータ

```python
@dataclass
class FeatureMetadata:
    """特徴量メタデータ"""
    name: str
    category: str  # 'task', 'developer', 'matching', 'gat'
    data_type: str  # 'numerical', 'categorical', 'binary'
    scaling_method: str  # 'standard', 'minmax', 'log', 'robust'
    importance_score: float
    correlation_threshold: float
    description: str

@dataclass
class FeatureSet:
    """特徴量セット"""
    features: List[FeatureMetadata]
    version: str
    creation_date: datetime
    performance_metrics: Dict[str, float]
    config: Dict[str, Any]
```

### 分析結果データモデル

```python
@dataclass
class AnalysisResults:
    """分析結果"""
    importance_ranking: List[Tuple[str, float]]
    correlation_matrix: np.ndarray
    distribution_stats: Dict[str, Dict[str, float]]
    redundant_features: List[str]
    recommended_actions: List[str]

@dataclass
class OptimizationResults:
    """最適化結果"""
    selected_features: List[str]
    scaling_parameters: Dict[str, Any]
    dimension_reduction_params: Dict[str, Any]
    performance_improvement: float
    computational_efficiency: Dict[str, float]
```

## エラーハンドリング

### エラー分類

1. **データ品質エラー**

   - 欠損値の処理
   - 外れ値の検出と処理
   - データ型の不整合

2. **計算エラー**

   - 数値計算の安定性
   - メモリ不足の処理
   - 収束しない最適化

3. **設定エラー**
   - 無効な設定パラメータ
   - 互換性のない設定組み合わせ
   - リソース制約の違反

### エラー処理戦略

```python
class FeatureEngineeringError(Exception):
    """特徴量エンジニアリング関連エラー"""
    pass

class DataQualityError(FeatureEngineeringError):
    """データ品質エラー"""
    pass

class ComputationError(FeatureEngineeringError):
    """計算エラー"""
    pass

class ConfigurationError(FeatureEngineeringError):
    """設定エラー"""
    pass

def handle_feature_extraction_error(func):
    """特徴量抽出エラーハンドリングデコレータ"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DataQualityError as e:
            logger.warning(f"Data quality issue in {func.__name__}: {e}")
            return default_feature_values()
        except ComputationError as e:
            logger.error(f"Computation error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return fallback_feature_values()
    return wrapper
```

## テスト戦略

### 単体テスト

```python
class TestFeatureAnalysis(unittest.TestCase):
    """特徴量分析のテスト"""

    def setUp(self):
        self.analyzer = FeatureImportanceAnalyzer(
            weights_path="test_data/test_weights.npy",
            feature_names=["feature_1", "feature_2", "feature_3"]
        )

    def test_importance_ranking(self):
        """重要度ランキングのテスト"""
        ranking = self.analyzer._rank_by_importance()
        self.assertIsInstance(ranking, list)
        self.assertTrue(all(isinstance(item, tuple) for item in ranking))

    def test_category_analysis(self):
        """カテゴリ分析のテスト"""
        analysis = self.analyzer._analyze_by_category()
        self.assertIsInstance(analysis, dict)
        self.assertTrue(all(isinstance(v, float) for v in analysis.values()))

class TestFeatureDesign(unittest.TestCase):
    """特徴量設計のテスト"""

    def setUp(self):
        self.designer = TaskFeatureDesigner(cfg=mock_config)

    def test_urgency_features(self):
        """緊急度特徴量のテスト"""
        task = create_mock_task()
        features = self.designer._add_urgency_features(task)
        self.assertIn('task_has_priority_label', features)
        self.assertIn('task_has_deadline', features)
```

### 統合テスト

```python
class TestFeaturePipeline(unittest.TestCase):
    """特徴量パイプラインの統合テスト"""

    def setUp(self):
        self.pipeline = FeaturePipeline(test_config)

    def test_full_pipeline_execution(self):
        """フルパイプライン実行のテスト"""
        input_data = create_test_input_data()
        results = self.pipeline.execute_pipeline(input_data)

        self.assertIn('analysis', results)
        self.assertIn('design', results)
        self.assertIn('optimization', results)
        self.assertIn('gat_enhancement', results)
        self.assertIn('evaluation', results)

    def test_pipeline_error_handling(self):
        """パイプラインエラーハンドリングのテスト"""
        invalid_input = create_invalid_input_data()
        with self.assertRaises(FeatureEngineeringError):
            self.pipeline.execute_pipeline(invalid_input)
```

### パフォーマンステスト

```python
class TestPerformance(unittest.TestCase):
    """パフォーマンステスト"""

    def test_feature_extraction_speed(self):
        """特徴量抽出速度のテスト"""
        start_time = time.time()

        for _ in range(1000):
            features = extract_features(mock_task, mock_developer, mock_env)

        end_time = time.time()
        avg_time = (end_time - start_time) / 1000

        self.assertLess(avg_time, 0.01)  # 10ms以下

    def test_memory_usage(self):
        """メモリ使用量のテスト"""
        import psutil
        process = psutil.Process()

        initial_memory = process.memory_info().rss

        # 大量の特徴量抽出を実行
        for _ in range(10000):
            features = extract_features(mock_task, mock_developer, mock_env)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        self.assertLess(memory_increase, 100 * 1024 * 1024)  # 100MB以下
```

## 設定管理

### 特徴量設定ファイル

```yaml
# feature_config.yaml
feature_engineering:
  analysis:
    importance_threshold: 0.01
    correlation_threshold: 0.8
    significance_level: 0.05

  design:
    task_features:
      enable_urgency: true
      enable_complexity: true
      enable_social_attention: true

    developer_features:
      enable_expertise: true
      enable_activity_patterns: true
      enable_quality_metrics: true

    matching_features:
      enable_temporal_proximity: true
      enable_technical_compatibility: true
      enable_success_history: true

  optimization:
    scaling:
      default_method: "standard"
      log_transform_features:
        - "task_days_since_last_activity"
        - "dev_recent_activity_count"
        - "dev_total_lines_changed"

      minmax_features:
        - "task_watchers_count"
        - "task_reactions_count"

    selection:
      method: "importance" # "univariate", "rfe", "l1", "importance"
      max_features: 50
      min_importance: 0.001

    dimension_reduction:
      enable: false
      method: "pca" # "pca", "umap", "tsne"
      target_dimensions: 30

  gat_enhancement:
    optimize_dimensions: true
    target_embedding_dim: 32
    enable_interpretation: true

  pipeline:
    enable_caching: true
    cache_directory: "cache/features"
    parallel_processing: true
    max_workers: 4

  evaluation:
    enable_ab_testing: true
    baseline_feature_set: "current"
    metrics:
      - "irl_loss"
      - "prediction_accuracy"
      - "computational_time"
      - "memory_usage"
```

この設計文書は、逆強化学習の特徴量見直しプロジェクトの包括的な技術設計を提供し、実装チームが効率的に開発を進められるよう詳細な仕様を定義している。
