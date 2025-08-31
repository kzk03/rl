# 開発者定着予測システム - 設計文書

## 概要

本設計文書では、開発者の長期的な貢献を促進する「開発者定着予測システム」の包括的な技術設計を定義する。このシステムは、開発者の「沸点」（ストレス限界点）を予測し、レビュー受諾行動を最適化することで、持続可能な開発者コミュニティを構築することを目的とする。

## アーキテクチャ

### システム全体構成

```
┌─────────────────────────────────────────────────────────────────┐
│                Developer Retention Prediction System            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Retention       │  │ Stress & Boiling│  │ Review Behavior │  │
│  │ Prediction      │  │ Point Analysis  │  │ Analysis        │  │
│  │ Module          │  │ Module          │  │ Module          │  │
│  │                 │  │                 │  │                 │  │
│  │ - Probability   │  │ - Stress Calc   │  │ - Acceptance    │  │
│  │ - Factors       │  │ - Threshold     │  │ - Similarity    │  │
│  │ - Validation    │  │ - Mitigation    │  │ - Preferences   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ RL Environment  │  │ Visualization   │  │ Adaptive        │  │
│  │ Module          │  │ Module          │  │ Strategy Module │  │
│  │                 │  │                 │  │                 │  │
│  │ - State Space   │  │ - Heatmaps      │  │ - Dynamic Adj   │  │
│  │ - Action Space  │  │ - Dashboards    │  │ - Multi-obj Opt │  │
│  │ - Reward Design │  │ - Patterns      │  │ - Learning      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### データフロー

```
Gerrit Data → Feature Extraction → Retention Prediction → Stress Analysis →
Review Behavior Analysis → RL Environment → Action Selection →
Reward Calculation → Model Update → Visualization → Feedback Loop
```

## Gerrit データ統合

### Gerrit データの特徴

Gerrit は企業環境でのコードレビューに特化したシステムであり、GitHub と比較して以下の利点がある：

1. **構造化されたレビューフロー**: Change → Review → Approval/Rejection の明確なフロー
2. **詳細なレビューメタデータ**: レビュー時間、コメント数、承認者情報の詳細な記録
3. **企業環境での実データ**: 実際の開発チームでの長期的な協力関係データ
4. **レビュー品質指標**: +1/+2/-1/-2 の段階的評価システム

### Gerrit API データ取得

```python
class GerritDataExtractor:
    """Gerritデータ抽出器"""

    def __init__(self, gerrit_url: str, auth_config: Dict[str, str]):
        self.gerrit_url = gerrit_url
        self.auth = auth_config
        self.client = self._initialize_client()

    def extract_review_data(self, project: str, time_range: Tuple[str, str]) -> List[Dict[str, Any]]:
        """レビューデータを抽出"""
        changes = self._get_changes(project, time_range)
        review_data = []

        for change in changes:
            change_detail = self._get_change_detail(change['id'])
            reviews = self._get_reviews(change['id'])

            review_data.append({
                'change_id': change['id'],
                'project': project,
                'author': change_detail['owner']['email'],
                'created': change_detail['created'],
                'updated': change_detail['updated'],
                'status': change_detail['status'],
                'reviews': self._process_reviews(reviews),
                'files_changed': len(change_detail.get('files', {})),
                'lines_added': change_detail.get('insertions', 0),
                'lines_deleted': change_detail.get('deletions', 0)
            })

        return review_data

    def extract_developer_profiles(self, project: str) -> Dict[str, Dict[str, Any]]:
        """開発者プロファイルを抽出"""
        developers = {}
        changes = self._get_all_changes(project)

        for change in changes:
            author_email = change['owner']['email']
            if author_email not in developers:
                developers[author_email] = {
                    'email': author_email,
                    'name': change['owner'].get('name', ''),
                    'changes_authored': 0,
                    'reviews_given': 0,
                    'reviews_received': 0,
                    'approval_rate': 0.0,
                    'avg_review_time': 0.0,
                    'expertise_areas': set(),
                    'collaboration_network': set()
                }

            developers[author_email]['changes_authored'] += 1
            # さらなる統計情報を蓄積

        return developers
```

## コンポーネントとインターフェース

### 1. Retention Prediction Module

#### 1.1 RetentionPredictor

```python
class RetentionPredictor:
    """開発者定着予測器"""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model = self._initialize_model()
        self.feature_extractor = RetentionFeatureExtractor()

    def predict_retention_probability(self, developer: Dict[str, Any],
                                    context: Dict[str, Any]) -> float:
        """定着確率を予測"""
        features = self.feature_extractor.extract_features(developer, context)
        probability = self.model.predict_proba(features.reshape(1, -1))[0][1]
        return probability

    def analyze_retention_factors(self, developer: Dict[str, Any]) -> Dict[str, float]:
        """定着に影響する要因を分析"""
        return {
            'task_compatibility': self._calculate_task_compatibility(developer),
            'workload_stress': self._calculate_workload_stress(developer),
            'social_factors': self._calculate_social_factors(developer),
            'expertise_growth': self._calculate_expertise_growth(developer)
        }
```

### 2. Stress & Boiling Point Analysis Module

#### 2.1 StressAnalyzer

```python
class StressAnalyzer:
    """開発者ストレス分析器"""

    def __init__(self, stress_config: Dict[str, Any]):
        self.stress_config = stress_config
        self.stress_weights = stress_config.get('weights', {})

    def calculate_stress_indicators(self, developer: Dict[str, Any],
                                  context: Dict[str, Any]) -> Dict[str, float]:
        """ストレス指標を計算"""
        return {
            'task_compatibility_stress': self._calculate_task_compatibility_stress(developer, context),
            'workload_stress': self._calculate_workload_stress(developer, context),
            'social_stress': self._calculate_social_stress(developer, context),
            'temporal_stress': self._calculate_temporal_stress(developer, context)
        }

    def predict_boiling_point(self, developer: Dict[str, Any],
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """沸点を予測"""
        stress_indicators = self.calculate_stress_indicators(developer, context)
        current_stress = self._calculate_total_stress(stress_indicators)

        # 過去の離脱パターンから沸点閾値を推定
        boiling_threshold = self._estimate_boiling_threshold(developer)

        return {
            'current_stress': current_stress,
            'boiling_threshold': boiling_threshold,
            'stress_margin': boiling_threshold - current_stress,
            'risk_level': self._calculate_risk_level(current_stress, boiling_threshold),
            'time_to_boiling': self._estimate_time_to_boiling(developer, current_stress, boiling_threshold)
        }
```

### 3. RL Environment Module

#### 3.1 ReviewAcceptanceEnvironment

```python
class ReviewAcceptanceEnvironment(gym.Env):
    """レビュー受諾強化学習環境"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # 状態空間の定義
        self.observation_space = self._define_observation_space()

        # 行動空間の定義（レビュー受諾/拒否/待機）
        self.action_space = spaces.Discrete(3)  # 0: 拒否, 1: 受諾, 2: 待機

        # 環境の初期化
        self.reset()

    def reset(self) -> np.ndarray:
        """環境をリセット"""
        self.current_step = 0
        self.developer_state = self._initialize_developer_state()
        self.review_queue = self._initialize_review_queue()
        self.stress_accumulator = 0.0
        self.acceptance_history = []

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """1ステップ実行"""
        # 行動を実行
        reward = self._execute_action(action)

        # 状態を更新
        self._update_state(action)

        # 終了条件をチェック
        done = self._check_done()

        # 次の観測を取得
        next_observation = self._get_observation()

        # 情報を収集
        info = self._get_info()

        self.current_step += 1

        return next_observation, reward, done, info

    def _execute_action(self, action: int) -> float:
        """行動を実行して報酬を計算"""
        if not self.review_queue:
            return 0.0  # レビューがない場合は報酬なし

        current_review = self.review_queue[0]

        if action == 0:  # 拒否
            reward = self._calculate_rejection_reward(current_review)
            self.review_queue.pop(0)
            self.acceptance_history.append({'action': 'reject', 'review': current_review})

        elif action == 1:  # 受諾
            reward = self._calculate_acceptance_reward(current_review)
            self.review_queue.pop(0)
            self.acceptance_history.append({'action': 'accept', 'review': current_review})
            self._update_developer_state_after_acceptance(current_review)

        else:  # 待機
            reward = self._calculate_waiting_reward()
            # レビューはキューに残る

        return reward

    def _calculate_acceptance_reward(self, review: Dict[str, Any]) -> float:
        """受諾時の報酬を計算"""
        base_reward = 1.0  # 基本受諾報酬

        # 継続報酬
        recent_acceptances = sum(1 for h in self.acceptance_history[-5:] if h['action'] == 'accept')
        continuity_reward = 0.3 * recent_acceptances

        # ストレス報酬
        expertise_match = review.get('expertise_match', 0.5)
        if expertise_match > 0.7:
            stress_reward = 0.2  # 専門性に合う場合はストレス軽減
        else:
            stress_reward = -0.4  # 専門性に合わない場合はストレス増加

        # 品質報酬（予測）
        expected_quality = self._predict_review_quality(review)
        quality_reward = 0.1 * expected_quality

        # 協力報酬
        if review.get('requester_relationship', 0.0) < 0.3:
            collaboration_reward = 0.15  # 新しい協力関係
        else:
            collaboration_reward = 0.1  # 既存関係の強化

        total_reward = base_reward + continuity_reward + stress_reward + quality_reward + collaboration_reward
        return total_reward
```

## データモデル

### Gerrit 特化データモデル

```python
@dataclass
class GerritChange:
    """Gerrit Change（レビュー対象）"""
    change_id: str
    project: str
    branch: str
    author_email: str
    subject: str
    created: datetime
    updated: datetime
    status: str  # 'NEW', 'MERGED', 'ABANDONED'
    files_changed: int
    lines_added: int
    lines_deleted: int
    complexity_score: float
    technical_domain: str

@dataclass
class GerritReview:
    """Gerritレビュー"""
    change_id: str
    reviewer_email: str
    timestamp: datetime
    score: int  # -2, -1, 0, +1, +2
    message: str
    response_time_hours: float
    review_effort_estimated: float

@dataclass
class DeveloperState:
    """開発者状態（Gerrit版）"""
    developer_email: str
    name: str
    expertise_level: float
    stress_level: float
    activity_pattern: Dict[str, float]
    recent_review_acceptance_rate: float
    workload_ratio: float
    collaboration_quality: float
    learning_velocity: float
    satisfaction_level: float
    boiling_point_estimate: float

    # Gerrit特有の指標
    avg_review_score_given: float  # 与えたレビューの平均スコア
    avg_review_score_received: float  # 受けたレビューの平均スコア
    review_response_time_avg: float  # 平均レビュー応答時間
    code_review_thoroughness: float  # レビューの詳細度

    last_updated: datetime

@dataclass
class ReviewAssignmentRequest:
    """レビュー割り当て依頼（Gerrit版）"""
    change_id: str
    author_email: str
    potential_reviewers: List[str]
    change_complexity: float
    change_size: int
    technical_domain: str
    urgency_level: float
    estimated_review_effort: float
    required_expertise: List[str]
    created_at: datetime
    deadline: Optional[datetime]

@dataclass
class ReviewAcceptanceResponse:
    """レビュー受諾応答（Gerrit版）"""
    change_id: str
    reviewer_email: str
    action: str  # 'accept', 'decline', 'defer'
    response_time_hours: float
    confidence_level: float
    stress_impact: float
    expected_review_quality: float
    timestamp: datetime
```

## エラーハンドリング

```python
class DeveloperRetentionError(Exception):
    """開発者定着予測システム関連エラー"""
    pass

class StressCalculationError(DeveloperRetentionError):
    """ストレス計算エラー"""
    pass

class BoilingPointPredictionError(DeveloperRetentionError):
    """沸点予測エラー"""
    pass

def handle_prediction_error(func):
    """予測エラーハンドリングデコレータ"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except StressCalculationError as e:
            logger.warning(f"Stress calculation failed in {func.__name__}: {e}")
            return default_stress_values()
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return safe_fallback_values()
    return wrapper
```

## テスト戦略

### 単体テスト

```python
class TestRetentionPredictor(unittest.TestCase):
    """定着予測器のテスト"""

    def setUp(self):
        self.predictor = RetentionPredictor(test_config)

    def test_retention_probability_calculation(self):
        """定着確率計算のテスト"""
        developer = create_mock_developer()
        context = create_mock_context()

        probability = self.predictor.predict_retention_probability(developer, context)

        self.assertIsInstance(probability, float)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
```

## 設定管理

```yaml
# developer_retention_config.yaml
developer_retention_system:
  gerrit_integration:
    gerrit_url: "https://gerrit.example.com"
    auth:
      username: "${GERRIT_USERNAME}"
      password: "${GERRIT_PASSWORD}"
    projects:
      - "project-a"
      - "project-b"
    data_extraction:
      batch_size: 1000
      rate_limit_delay: 1.0  # seconds
      max_retries: 3

  retention_prediction:
    model_type: "random_forest"
    model_params:
      n_estimators: 100
      max_depth: 10
      random_state: 42

    # Gerrit特有の特徴量
    gerrit_features:
      use_review_scores: true
      use_response_times: true
      use_change_complexity: true
      use_collaboration_network: true

  stress_analysis:
    weights:
      review_compatibility_stress: 0.3  # レビュー適合度ストレス
      workload_stress: 0.4
      social_stress: 0.2
      temporal_stress: 0.1

    # Gerrit特有のストレス要因
    gerrit_stress_factors:
      high_complexity_threshold: 0.8
      review_queue_size_threshold: 5
      response_time_pressure_threshold: 24  # hours

  rl_environment:
    observation_space_dim: 20
    action_space_size: 3  # accept, decline, defer
    max_episode_length: 100

    reward_weights:
      acceptance_reward: 1.0
      decline_penalty: -0.5
      defer_penalty: -0.1
      continuity_bonus: 0.3
      stress_factor: 0.2
      quality_bonus: 0.1
      collaboration_bonus: 0.15

    # Gerrit特有の報酬設計
    gerrit_rewards:
      high_quality_review_bonus: 0.2  # +2スコアのレビュー
      thorough_review_bonus: 0.1
      quick_response_bonus: 0.05

  ppo_agent:
    policy_lr: 3e-4
    value_lr: 3e-4
    clip_epsilon: 0.2
    gamma: 0.99
    gae_lambda: 0.95

  temporal_consistency:
    enable_strict_validation: true
    train_end_date: "2022-12-31"
    test_start_date: "2023-01-01"

    # Gerrit特有の時系列設定
    gerrit_temporal:
      review_history_window: 90  # days
      developer_profile_update_frequency: 7  # days
      stress_calculation_window: 14  # days
    test_start_date: "2023-01-01"
```

## プロジェクト構造（Gerrit 版）

### 新しいディレクトリレイアウト

```
gerrit-retention/                    # Gerrit版開発者定着予測システム
├── src/gerrit_retention/           # コアソースコード（インストール可能パッケージ）
│   ├── __init__.py
│   ├── data_integration/           # Gerritデータ統合
│   │   ├── __init__.py
│   │   ├── gerrit_client.py       # Gerrit APIクライアント
│   │   ├── data_extractor.py      # データ抽出器
│   │   └── data_transformer.py    # データ変換器
│   ├── prediction/                 # 予測モデル
│   │   ├── __init__.py
│   │   ├── retention_predictor.py # 定着予測器
│   │   ├── stress_analyzer.py     # ストレス分析器
│   │   └── boiling_point_predictor.py # 沸点予測器
│   ├── behavior_analysis/          # 行動分析
│   │   ├── __init__.py
│   │   ├── review_behavior.py     # レビュー行動分析
│   │   ├── similarity_calculator.py # 類似度計算
│   │   └── preference_analyzer.py # 好み分析
│   ├── rl_environment/             # 強化学習環境
│   │   ├── __init__.py
│   │   ├── review_env.py          # レビュー受諾環境
│   │   ├── ppo_agent.py           # PPOエージェント
│   │   └── reward_calculator.py   # 報酬計算器
│   ├── visualization/              # 可視化
│   │   ├── __init__.py
│   │   ├── heatmap_generator.py   # ヒートマップ生成
│   │   ├── dashboard.py           # ダッシュボード
│   │   └── chart_generator.py     # チャート生成
│   ├── adaptive_strategy/          # 適応戦略
│   │   ├── __init__.py
│   │   ├── strategy_manager.py    # 戦略管理
│   │   └── multi_objective_optimizer.py # 多目的最適化
│   └── utils/                      # ユーティリティ
│       ├── __init__.py
│       ├── config_manager.py      # 設定管理
│       ├── logger.py              # ログ管理
│       └── time_utils.py          # 時系列ユーティリティ
├── training/                       # 訓練モジュール
│   ├── retention_training/         # 定着予測訓練
│   │   ├── train_retention_model.py
│   │   └── evaluate_retention.py
│   ├── stress_training/            # ストレス分析訓練
│   │   ├── train_stress_model.py
│   │   └── evaluate_stress.py
│   └── rl_training/                # 強化学習訓練
│       ├── train_ppo_agent.py
│       └── evaluate_rl_policy.py
├── data_processing/                # データ処理
│   ├── gerrit_extraction/          # Gerritデータ抽出
│   │   ├── extract_changes.py     # Change抽出
│   │   ├── extract_reviews.py     # Review抽出
│   │   └── extract_developers.py  # Developer抽出
│   ├── feature_engineering/        # 特徴量エンジニアリング
│   │   ├── developer_features.py  # 開発者特徴量
│   │   ├── review_features.py     # レビュー特徴量
│   │   └── temporal_features.py   # 時系列特徴量
│   └── preprocessing/              # 前処理
│       ├── data_cleaning.py       # データクリーニング
│       ├── normalization.py       # 正規化
│       └── temporal_split.py      # 時系列分割
├── analysis/                       # 分析・レポート
│   ├── reports/                    # レポート生成
│   │   ├── retention_analysis.py  # 定着分析レポート
│   │   ├── stress_analysis.py     # ストレス分析レポート
│   │   └── behavior_analysis.py   # 行動分析レポート
│   └── visualization/              # 可視化スクリプト
│       ├── retention_plots.py     # 定着可視化
│       ├── stress_plots.py        # ストレス可視化
│       └── behavior_plots.py      # 行動可視化
├── evaluation/                     # 評価・テスト
│   ├── model_evaluation/           # モデル評価
│   │   ├── retention_eval.py      # 定着予測評価
│   │   ├── stress_eval.py         # ストレス予測評価
│   │   └── rl_eval.py             # 強化学習評価
│   ├── ab_testing/                 # A/Bテスト
│   │   ├── experiment_design.py   # 実験設計
│   │   └── statistical_analysis.py # 統計分析
│   └── integration_tests/          # 統合テスト
│       ├── end_to_end_test.py     # エンドツーエンドテスト
│       └── performance_test.py    # パフォーマンステスト
├── pipelines/                      # パイプライン
│   ├── data_pipeline.py           # データパイプライン
│   ├── training_pipeline.py       # 訓練パイプライン
│   └── inference_pipeline.py      # 推論パイプライン
├── scripts/                        # 実行スクリプト
│   ├── setup_gerrit_connection.py # Gerrit接続設定
│   ├── run_full_pipeline.py       # フルパイプライン実行
│   └── deploy_model.py            # モデルデプロイ
├── configs/                        # 設定ファイル
│   ├── gerrit_config.yaml         # Gerrit接続設定
│   ├── retention_config.yaml      # 定着予測設定
│   ├── stress_config.yaml         # ストレス分析設定
│   ├── rl_config.yaml             # 強化学習設定
│   └── visualization_config.yaml  # 可視化設定
├── data/                           # データディレクトリ
│   ├── raw/                        # 生データ
│   │   ├── gerrit_changes/        # Gerrit Change データ
│   │   ├── gerrit_reviews/        # Gerrit Review データ
│   │   └── gerrit_developers/     # 開発者データ
│   ├── processed/                  # 処理済みデータ
│   │   ├── features/              # 特徴量データ
│   │   ├── labels/                # ラベルデータ
│   │   └── splits/                # 時系列分割データ
│   └── external/                   # 外部データ
├── models/                         # 保存済みモデル
│   ├── retention_models/          # 定着予測モデル
│   ├── stress_models/             # ストレス分析モデル
│   └── rl_models/                 # 強化学習モデル
├── logs/                           # ログファイル
│   ├── training/                  # 訓練ログ
│   ├── inference/                 # 推論ログ
│   └── system/                    # システムログ
├── outputs/                        # 出力ファイル
│   ├── predictions/               # 予測結果
│   ├── reports/                   # 生成レポート
│   └── visualizations/            # 可視化出力
├── tests/                          # テストファイル
│   ├── unit/                      # 単体テスト
│   ├── integration/               # 統合テスト
│   └── fixtures/                  # テストデータ
├── docs/                           # ドキュメント
│   ├── api/                       # API ドキュメント
│   ├── user_guide/                # ユーザーガイド
│   └── development/               # 開発ドキュメント
├── docker/                         # Docker関連
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── .env.example                    # 環境変数テンプレート
├── pyproject.toml                  # Python プロジェクト設定
├── README.md                       # プロジェクト説明
└── setup.py                        # セットアップスクリプト
```

### 主要ディレクトリの説明

#### コア実装 (`src/gerrit_retention/`)

- **data_integration/**: Gerrit API との統合、データ抽出・変換
- **prediction/**: 定着予測、ストレス分析、沸点予測の核心アルゴリズム
- **behavior_analysis/**: レビュー行動分析、類似度計算、好み分析
- **rl_environment/**: 強化学習環境、PPO エージェント、報酬計算
- **visualization/**: ヒートマップ、ダッシュボード、チャート生成
- **adaptive_strategy/**: 適応的戦略管理、多目的最適化

#### データ処理 (`data_processing/`)

- **gerrit_extraction/**: Gerrit 特有のデータ抽出ロジック
- **feature_engineering/**: Gerrit データに特化した特徴量エンジニアリング
- **preprocessing/**: データクリーニング、正規化、時系列分割

#### 設定管理 (`configs/`)

- Gerrit 接続、各モデル、可視化の個別設定ファイル
- 環境別設定の階層管理

### ファイル命名規則

#### Gerrit 特有の命名

- **Gerrit データ**: `gerrit_changes.json`, `gerrit_reviews.pkl`
- **モデル**: `retention_model_gerrit_20250714.pkl`
- **設定**: `gerrit_production.yaml`, `gerrit_development.yaml`

#### 機能別プレフィックス

- **定着関連**: `retention_*`
- **ストレス関連**: `stress_*`
- **レビュー関連**: `review_*`
- **可視化関連**: `viz_*`

この設計文書は、開発者定着予測システムの包括的な技術設計を提供し、特にレビュー受諾行動に焦点を当てた強化学習環境と、開発者の沸点予測に基づく持続可能な推薦システムの実現を目指している。
