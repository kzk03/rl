"""
開発者定着予測モジュール

開発者の長期的な定着確率を予測し、定着に影響する要因を分析する。
Random Forest、XGBoost、Neural Networkモデルを統合したアンサンブル予測器を提供する。
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # SHAPが利用できない場合のダミークラス
    class shap:
        class TreeExplainer:
            def __init__(self, model):
                self.model = model
            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))
        
        class KernelExplainer:
            def __init__(self, model, data):
                self.model = model
            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
    # XGBoostが利用できない場合はRandomForestで代替
    XGBClassifier = None

try:
    from ...data_processing.feature_engineering.feature_integration import (
        FeatureIntegrator,
        IntegratedFeatures,
    )
except ImportError:
    # フォールバック: 特徴量統合器が存在しない場合
    class FeatureIntegrator:
        def __init__(self, config):
            pass
        def extract_integrated_features(self, *args, **kwargs):
            # ダミーの統合特徴量を返す
            class DummyIntegratedFeatures:
                def __init__(self):
                    self.feature_vector = np.zeros(50)  # 50次元のダミー特徴量
            return DummyIntegratedFeatures()
    
    class IntegratedFeatures:
        pass

try:
    from ..data_integration.data_transformer import DataTransformer
except ImportError:
    # フォールバック: データ変換器が存在しない場合
    class DataTransformer:
        def __init__(self, config):
            pass

logger = logging.getLogger(__name__)


class RetentionFeatureExtractor:
    """定着予測用特徴量抽出器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.feature_integrator = FeatureIntegrator(config.get('feature_integration', {}))
        self.data_transformer = DataTransformer(config.get('data_transformation', {}))

        # IRL feature adapter (optional)
        self.irl_enabled = config.get('irl_features', {}).get('enabled', False)
        self.irl_adapter = None
        if self.irl_enabled:
            try:
                from ..irl.maxent_binary_irl import MaxEntBinaryIRL
                from .irl_feature_adapter import (
                    IRLFeatureAdapter,
                    IRLFeatureAdapterConfig,
                )
                irl_conf = config.get('irl_features', {})
                self._irl_model = MaxEntBinaryIRL()  # Expect caller to fit and set externally if needed
                self.irl_adapter = IRLFeatureAdapter(
                    irl_model=self._irl_model,
                    config=IRLFeatureAdapterConfig(
                        idle_gap_threshold=irl_conf.get('idle_gap_threshold', 45)
                    ),
                )
                # Optional: load a pre-trained IRL model from path
                model_path = irl_conf.get('model_path')
                if isinstance(model_path, str) and len(model_path) > 0:
                    try:
                        loaded = joblib.load(model_path)
                        self._irl_model = loaded
                        self.irl_adapter.irl_model = loaded
                        logger.info(f"IRLモデルをロードしました: {model_path}")
                    except Exception as e:
                        logger.warning(f"IRLモデルのロードに失敗しました（スキップ）: {e}")
            except Exception as e:
                logger.warning(f"IRL特徴の初期化に失敗しました: {e}")
                self.irl_enabled = False
        
        # 定着予測特有の設定
        self.retention_window_days = config.get('retention_window_days', 180)  # 6ヶ月
        self.activity_threshold = config.get('activity_threshold', 1)  # 最低活動回数
        self.time_decay_factor = config.get('time_decay_factor', 0.95)  # 時間減衰係数
        # ギャップ系特徴の有無（ラベルがギャップ由来のときのフェアネス向上のために無効化可能）
        self.include_gap_features = bool(config.get('include_gap_features', True))
        
    def extract_features(self, 
                        developer: Dict[str, Any], 
                        context: Dict[str, Any]) -> np.ndarray:
        """
        定着予測用特徴量を抽出
        
        Args:
            developer: 開発者データ
            context: コンテキストデータ（プロジェクト状況、時期など）
            
        Returns:
            np.ndarray: 特徴量ベクトル
        """
        developer_email = developer.get('email', developer.get('developer_email'))
        context_date = context.get('context_date', datetime.now())
        
        # 統合特徴量を抽出
        integrated_features = self.feature_integrator.extract_integrated_features(
            developer_email=developer_email,
            changes_data=context.get('changes_data', []),
            reviews_data=context.get('reviews_data', []),
            context_date=context_date
        )
        
        # 定着予測特有の特徴量を追加
        retention_specific_features = self._extract_retention_specific_features(
            developer, context, integrated_features
        )
        
        parts = [integrated_features.feature_vector, retention_specific_features]
        # IRL由来特徴の付与（有効化時）
        if self.irl_enabled and self.irl_adapter is not None:
            try:
                irl_feats, _ = self.irl_adapter.compute_features(developer, context)
                parts.append(irl_feats)
            except Exception as e:
                logger.debug(f"IRL特徴の抽出に失敗（スキップ）: {e}")

        # 特徴量を結合
        combined_features = np.concatenate(parts)
        
        return combined_features
    
    def _extract_retention_specific_features(self, 
                                           developer: Dict[str, Any],
                                           context: Dict[str, Any],
                                           integrated_features: IntegratedFeatures) -> np.ndarray:
        """定着予測特有の特徴量を抽出"""
        
        features = []
        
        # 過去の定着パターン
        historical_retention = self._calculate_historical_retention_pattern(developer, context)
        features.extend(historical_retention)
        
        # プロジェクト関与度
        project_engagement = self._calculate_project_engagement(developer, context)
        features.extend(project_engagement)
        
        # 成長・学習指標
        growth_indicators = self._calculate_growth_indicators(developer, context)
        features.extend(growth_indicators)
        
        # 社会的結合度
        social_bonds = self._calculate_social_bonds(developer, context)
        features.extend(social_bonds)
        
        # 時期・季節性要因
        temporal_factors = self._calculate_temporal_factors(context)
        features.extend(temporal_factors)

        # ギャップ系特徴（最終活動からの経過日数）
        # ラベルがギャップしきい値由来の場合にリーク的になるため、設定で無効化できるようにする
        if self.include_gap_features:
            gap_days = float(developer.get('days_since_last_activity', 0.0) or 0.0)
            gap_days = max(0.0, min(gap_days, 3650.0))  # 10年でクリップ
            # 単純な非線形変換も併せて付与（モデル依存度を下げるための基底関数）
            inv_gap = 1.0 / (1.0 + gap_days)
            exp_decay_30 = float(np.exp(-gap_days / 30.0))  # 30日スケールの減衰
            features.extend([gap_days, inv_gap, exp_decay_30])

        return np.array(features)
    
    def _calculate_historical_retention_pattern(self, 
                                              developer: Dict[str, Any],
                                              context: Dict[str, Any]) -> List[float]:
        """過去の定着パターンを計算"""
        
        # 過去のプロジェクト参加期間
        past_project_durations = developer.get('past_project_durations', [])
        avg_project_duration = np.mean(past_project_durations) if past_project_durations else 0.0
        max_project_duration = max(past_project_durations) if past_project_durations else 0.0
        
        # 離脱・復帰パターン
        departure_count = developer.get('departure_count', 0)
        return_count = developer.get('return_count', 0)
        return_rate = return_count / max(departure_count, 1)
        
        # 活動の継続性
        activity_gaps = developer.get('activity_gaps', [])
        avg_gap_duration = np.mean(activity_gaps) if activity_gaps else 0.0
        max_gap_duration = max(activity_gaps) if activity_gaps else 0.0
        
        return [
            avg_project_duration,
            max_project_duration,
            departure_count,
            return_count,
            return_rate,
            avg_gap_duration,
            max_gap_duration
        ]
    
    def _calculate_project_engagement(self, 
                                    developer: Dict[str, Any],
                                    context: Dict[str, Any]) -> List[float]:
        """プロジェクト関与度を計算"""
        
        # コア機能への関与度
        core_feature_involvement = developer.get('core_feature_involvement', 0.0)
        
        # 重要な意思決定への参加
        decision_participation = developer.get('decision_participation', 0.0)
        
        # ドキュメント・テスト貢献
        documentation_contribution = developer.get('documentation_contribution', 0.0)
        test_contribution = developer.get('test_contribution', 0.0)
        
        # プロジェクト固有知識
        project_specific_knowledge = developer.get('project_specific_knowledge', 0.0)
        
        # リーダーシップ・メンタリング
        leadership_score = developer.get('leadership_score', 0.0)
        mentoring_activity = developer.get('mentoring_activity', 0.0)
        
        return [
            core_feature_involvement,
            decision_participation,
            documentation_contribution,
            test_contribution,
            project_specific_knowledge,
            leadership_score,
            mentoring_activity
        ]
    
    def _calculate_growth_indicators(self, 
                                   developer: Dict[str, Any],
                                   context: Dict[str, Any]) -> List[float]:
        """成長・学習指標を計算"""
        
        # 技術スキルの成長率
        skill_growth_rate = developer.get('skill_growth_rate', 0.0)
        
        # 新技術への適応性
        technology_adaptability = developer.get('technology_adaptability', 0.0)
        
        # 学習機会の活用度
        learning_opportunity_utilization = developer.get('learning_opportunity_utilization', 0.0)
        
        # 挑戦的タスクへの取り組み
        challenging_task_engagement = developer.get('challenging_task_engagement', 0.0)
        
        # 知識共有活動
        knowledge_sharing_activity = developer.get('knowledge_sharing_activity', 0.0)
        
        return [
            skill_growth_rate,
            technology_adaptability,
            learning_opportunity_utilization,
            challenging_task_engagement,
            knowledge_sharing_activity
        ]
    
    def _calculate_social_bonds(self, 
                              developer: Dict[str, Any],
                              context: Dict[str, Any]) -> List[float]:
        """社会的結合度を計算"""
        
        # チーム内での信頼度
        team_trust_level = developer.get('team_trust_level', 0.0)
        
        # 協力関係の質
        collaboration_quality = developer.get('collaboration_quality', 0.0)
        
        # コミュニティ参加度
        community_participation = developer.get('community_participation', 0.0)
        
        # 友好関係の数
        friendship_count = developer.get('friendship_count', 0)
        
        # チーム外との連携
        external_collaboration = developer.get('external_collaboration', 0.0)
        
        return [
            team_trust_level,
            collaboration_quality,
            community_participation,
            friendship_count,
            external_collaboration
        ]
    
    def _calculate_temporal_factors(self, context: Dict[str, Any]) -> List[float]:
        """時期・季節性要因を計算"""
        
        context_date = context.get('context_date', datetime.now())
        
        # 年内の時期（0-1）
        day_of_year = context_date.timetuple().tm_yday / 365.0
        
        # 月（1-12を0-1に正規化）
        month_normalized = (context_date.month - 1) / 11.0
        
        # 四半期（1-4を0-1に正規化）
        quarter = ((context_date.month - 1) // 3 + 1)
        quarter_normalized = (quarter - 1) / 3.0
        
        # プロジェクトフェーズ
        project_phase = context.get('project_phase', 'development')  # development, testing, release
        phase_encoding = {
            'development': 0.0,
            'testing': 0.5,
            'release': 1.0
        }
        phase_value = phase_encoding.get(project_phase, 0.0)
        
        # リリースまでの期間（日数を正規化）
        release_date = context.get('next_release_date')
        days_to_release = 0.0
        if release_date:
            if isinstance(release_date, str):
                release_date = datetime.fromisoformat(release_date)
            days_diff = (release_date - context_date).days
            days_to_release = max(0, min(days_diff / 365.0, 1.0))  # 1年で正規化
        
        return [
            day_of_year,
            month_normalized,
            quarter_normalized,
            phase_value,
            days_to_release
        ]


class RetentionPredictor:
    """開発者定着予測器"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        初期化
        
        Args:
            model_config: モデル設定辞書
        """
        self.model_config = model_config
        self.feature_extractor = RetentionFeatureExtractor(
            model_config.get('feature_extraction', {})
        )

        # 外部からIRLモデルを差し込むためのフック
        self.irl_adapter = getattr(self.feature_extractor, 'irl_adapter', None)
        
        # アンサンブルモデルの初期化
        self.models = {}
        self.model_weights = {}
        self.is_fitted = False
        
        # SHAP説明器
        self.shap_explainers = {}
        
        # 特徴量名
        self.feature_names = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """モデルを初期化"""
        
        # Random Forest
        rf_config = self.model_config.get('random_forest', {})
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 10),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            random_state=rf_config.get('random_state', 42),
            n_jobs=rf_config.get('n_jobs', -1)
        )
        
        # XGBoost (または代替のRandomForest)
        xgb_config = self.model_config.get('xgboost', {})
        if XGBOOST_AVAILABLE and XGBClassifier is not None:
            try:
                self.models['xgboost'] = XGBClassifier(
                    n_estimators=xgb_config.get('n_estimators', 100),
                    max_depth=xgb_config.get('max_depth', 6),
                    learning_rate=xgb_config.get('learning_rate', 0.1),
                    subsample=xgb_config.get('subsample', 0.8),
                    colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
                    random_state=xgb_config.get('random_state', 42),
                    n_jobs=xgb_config.get('n_jobs', -1)
                )
            except Exception as e:
                logger.warning(f"XGBoostの初期化に失敗しました: {e}")
                # XGBoostが利用できない場合はRandomForestで代替
                self.models['xgboost'] = RandomForestClassifier(
                    n_estimators=xgb_config.get('n_estimators', 100),
                    max_depth=xgb_config.get('max_depth', 6),
                    random_state=xgb_config.get('random_state', 42),
                    n_jobs=xgb_config.get('n_jobs', -1)
                )
        else:
            # XGBoostが利用できない場合はRandomForestで代替
            self.models['xgboost'] = RandomForestClassifier(
                n_estimators=xgb_config.get('n_estimators', 100),
                max_depth=xgb_config.get('max_depth', 6),
                random_state=xgb_config.get('random_state', 42),
                n_jobs=xgb_config.get('n_jobs', -1)
            )
        
        # Neural Network
        nn_config = self.model_config.get('neural_network', {})
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=nn_config.get('hidden_layer_sizes', (100, 50)),
            activation=nn_config.get('activation', 'relu'),
            solver=nn_config.get('solver', 'adam'),
            alpha=nn_config.get('alpha', 0.001),
            learning_rate=nn_config.get('learning_rate', 'constant'),
            max_iter=nn_config.get('max_iter', 500),
            random_state=nn_config.get('random_state', 42)
        )
        
        # デフォルトの重み
        self.model_weights = self.model_config.get('ensemble_weights', {
            'random_forest': 0.4,
            'xgboost': 0.4,
            'neural_network': 0.2
        })
    
    def fit(self, 
            developers: List[Dict[str, Any]], 
            contexts: List[Dict[str, Any]], 
            labels: List[int]) -> None:
        """
        モデルを訓練
        
        Args:
            developers: 開発者データのリスト
            contexts: コンテキストデータのリスト
            labels: 定着ラベル（1: 定着, 0: 離脱）
        """
        logger.info("定着予測モデルの訓練を開始...")
        
        # 特徴量を抽出
        X = []
        for developer, context in zip(developers, contexts):
            features = self.feature_extractor.extract_features(developer, context)
            X.append(features)
        
        X = np.array(X)
        y = np.array(labels)
        
        logger.info(f"訓練データ: {X.shape[0]}サンプル, {X.shape[1]}特徴量")
        
        # 各モデルを訓練
        for model_name, model in self.models.items():
            logger.info(f"{model_name}モデルを訓練中...")
            
            try:
                model.fit(X, y)
                
                # クロスバリデーションで性能評価（少数データでは分割数を調整）
                try:
                    cv = 5
                    n_samples = X.shape[0]
                    if n_samples < 5:
                        cv = max(2, n_samples)
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
                    logger.info(f"{model_name} CV AUC (cv={cv}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                except Exception as cv_err:
                    logger.info(f"{model_name} CV評価をスキップ: {cv_err}")
                
                # SHAP説明器を初期化
                if model_name == 'random_forest':
                    self.shap_explainers[model_name] = shap.TreeExplainer(model)
                elif model_name == 'xgboost':
                    self.shap_explainers[model_name] = shap.TreeExplainer(model)
                else:
                    # Neural Networkの場合はKernelExplainerを使用
                    self.shap_explainers[model_name] = shap.KernelExplainer(
                        model.predict_proba, X[:100]  # サンプルを使用
                    )
                
            except Exception as e:
                logger.error(f"{model_name}モデルの訓練でエラー: {e}")
                continue
        
        self.is_fitted = True
        logger.info("定着予測モデルの訓練が完了しました")

    def set_irl_model(self, irl_model: Any) -> None:
        """外部で学習済みのIRLモデルを差し込む。

        例:
            from src.gerrit_retention.irl.maxent_binary_irl import MaxEntBinaryIRL
            irl = MaxEntBinaryIRL(...); irl.fit(transitions)
            predictor.set_irl_model(irl)
        """
        try:
            if hasattr(self.feature_extractor, 'irl_adapter') and self.feature_extractor.irl_adapter is not None:
                self.feature_extractor.irl_adapter.irl_model = irl_model
                self.irl_adapter = self.feature_extractor.irl_adapter
                logger.info("IRLモデルを予測器へ適用しました（IRL特徴有効）")
            else:
                logger.warning("IRL特徴が無効化されています。model_config.feature_extraction.irl_features.enabled を True にしてください。")
        except Exception as e:
            logger.error(f"IRLモデルの適用でエラー: {e}")

    def set_irl_model_from_path(self, model_path: str) -> None:
        """事前学習済みIRLモデルをファイルから読み込み、適用する。"""
        try:
            irl_model = joblib.load(model_path)
            self.set_irl_model(irl_model)
            logger.info(f"IRLモデルをファイルから適用しました: {model_path}")
        except Exception as e:
            logger.error(f"IRLモデルの読み込みでエラー: {e}")
    
    def predict_retention_probability(self, 
                                    developer: Dict[str, Any], 
                                    context: Dict[str, Any]) -> float:
        """
        定着確率を予測
        
        Args:
            developer: 開発者データ
            context: コンテキストデータ
            
        Returns:
            float: 定着確率（0-1）
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません。先にfit()を呼び出してください。")
        
        # 特徴量を抽出
        features = self.feature_extractor.extract_features(developer, context)
        X = features.reshape(1, -1)
        
        # 各モデルで予測
        predictions = {}
        for model_name, model in self.models.items():
            try:
                proba = model.predict_proba(X)
                # 予測確率の形状に応じてクラス1の確率を取り出す
                if hasattr(model, "classes_"):
                    classes = getattr(model, "classes_", None)
                    if classes is not None:
                        classes = np.array(classes)
                        if classes.ndim == 1:
                            # クラス1の列インデックスを特定
                            idx_ones = np.where(classes == 1)[0]
                            if idx_ones.size == 1 and proba.shape[1] > idx_ones[0]:
                                prob = float(proba[0, idx_ones[0]])
                            elif classes.size == 1:
                                # 単一クラスで学習された場合
                                only_cls = int(classes[0])
                                prob = 1.0 if only_cls == 1 else 0.0
                            else:
                                # 想定外の形状
                                prob = float(proba[0, -1]) if proba.ndim == 2 else float(proba[0])
                        else:
                            prob = float(proba[0, -1])
                    else:
                        prob = float(proba[0, -1])
                else:
                    # classes_が無い場合は最後の列をクラス1相当とみなす
                    prob = float(proba[0, -1]) if proba.ndim == 2 else float(proba[0])
                predictions[model_name] = prob
            except Exception as e:
                logger.warning(f"{model_name}での予測でエラー: {e}")
                predictions[model_name] = 0.5  # デフォルト値
        
        # アンサンブル予測
        ensemble_prob = sum(
            predictions[model_name] * self.model_weights.get(model_name, 0.0)
            for model_name in predictions
        )
        
        return float(ensemble_prob)
    
    def analyze_retention_factors(self, 
                                developer: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, float]:
        """
        定着に影響する要因を分析
        
        Args:
            developer: 開発者データ
            context: コンテキストデータ
            
        Returns:
            Dict[str, float]: 要因別影響度
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません。先にfit()を呼び出してください。")
        
        # 基本的な要因分析
        basic_factors = {
            'task_compatibility': self._calculate_task_compatibility(developer, context),
            'workload_stress': self._calculate_workload_stress(developer, context),
            'social_factors': self._calculate_social_factors(developer, context),
            'expertise_growth': self._calculate_expertise_growth(developer, context)
        }
        
        # SHAP値による詳細分析
        shap_factors = self._calculate_shap_factors(developer, context)
        
        # 結果を統合
        factors = {**basic_factors, **shap_factors}
        
        return factors
    
    def _calculate_task_compatibility(self, 
                                    developer: Dict[str, Any], 
                                    context: Dict[str, Any]) -> float:
        """タスク適合度を計算"""
        
        # 開発者の専門性
        expertise_areas = developer.get('expertise_areas', set())
        
        # 現在のタスクの技術領域
        current_tasks = context.get('current_tasks', [])
        task_domains = set()
        for task in current_tasks:
            task_domains.update(task.get('technical_domains', []))
        
        # 適合度計算
        if not expertise_areas or not task_domains:
            return 0.5  # デフォルト値
        
        overlap = len(expertise_areas.intersection(task_domains))
        compatibility = overlap / len(task_domains) if task_domains else 0.0
        
        return min(compatibility, 1.0)
    
    def _calculate_workload_stress(self, 
                                 developer: Dict[str, Any], 
                                 context: Dict[str, Any]) -> float:
        """ワークロードストレスを計算"""
        
        # 現在の負荷
        current_workload = developer.get('current_workload', 0.0)
        
        # 理想的な負荷
        ideal_workload = developer.get('ideal_workload', 1.0)
        
        # ストレス計算（負荷が理想から離れるほど高い）
        if ideal_workload > 0:
            stress = abs(current_workload - ideal_workload) / ideal_workload
        else:
            stress = current_workload
        
        return min(stress, 1.0)
    
    def _calculate_social_factors(self, 
                                developer: Dict[str, Any], 
                                context: Dict[str, Any]) -> float:
        """社会的要因を計算"""
        
        # 協力関係の質
        collaboration_quality = developer.get('collaboration_quality', 0.0)
        
        # コミュニティ参加度
        community_participation = developer.get('community_participation', 0.0)
        
        # チーム内での地位
        team_status = developer.get('team_status', 0.0)
        
        # 総合的な社会的要因
        social_score = (collaboration_quality + community_participation + team_status) / 3.0
        
        return social_score
    
    def _calculate_expertise_growth(self, 
                                  developer: Dict[str, Any], 
                                  context: Dict[str, Any]) -> float:
        """専門性成長を計算"""
        
        # 過去の成長率
        historical_growth = developer.get('expertise_growth_rate', 0.0)
        
        # 学習機会
        learning_opportunities = context.get('learning_opportunities', 0.0)
        
        # 挑戦的タスクの割合
        challenging_task_ratio = developer.get('challenging_task_ratio', 0.0)
        
        # 成長ポテンシャル
        growth_potential = (historical_growth + learning_opportunities + challenging_task_ratio) / 3.0
        
        return growth_potential
    
    def _calculate_shap_factors(self, 
                              developer: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, float]:
        """SHAP値による要因分析"""
        
        try:
            # 特徴量を抽出
            features = self.feature_extractor.extract_features(developer, context)
            X = features.reshape(1, -1)
            
            # Random ForestのSHAP値を計算（最も信頼性が高い）
            if 'random_forest' in self.shap_explainers:
                explainer = self.shap_explainers['random_forest']
                shap_values = explainer.shap_values(X)
                
                # クラス1（定着）のSHAP値を使用
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # 定着クラス
                
                shap_values = shap_values[0]  # 最初のサンプル
                
                # 上位の重要な特徴量を抽出
                feature_importance = {}
                if self.feature_names:
                    for i, importance in enumerate(shap_values):
                        if i < len(self.feature_names):
                            feature_importance[self.feature_names[i]] = float(importance)
                
                # 上位5つの要因を返す
                sorted_factors = sorted(
                    feature_importance.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:5]
                
                return {f"shap_{name}": value for name, value in sorted_factors}
            
        except Exception as e:
            logger.warning(f"SHAP分析でエラー: {e}")
        
        return {}
    
    def predict_batch(self, 
                     developers: List[Dict[str, Any]], 
                     contexts: List[Dict[str, Any]]) -> List[float]:
        """
        バッチ予測
        
        Args:
            developers: 開発者データのリスト
            contexts: コンテキストデータのリスト
            
        Returns:
            List[float]: 定着確率のリスト
        """
        predictions = []
        
        for developer, context in zip(developers, contexts):
            try:
                prob = self.predict_retention_probability(developer, context)
                predictions.append(prob)
            except Exception as e:
                logger.error(f"予測エラー (開発者: {developer.get('email', 'unknown')}): {e}")
                predictions.append(0.5)  # デフォルト値
        
        return predictions
    
    def evaluate_model(self, 
                      developers: List[Dict[str, Any]], 
                      contexts: List[Dict[str, Any]], 
                      true_labels: List[int]) -> Dict[str, float]:
        """
        モデル性能を評価
        
        Args:
            developers: 開発者データのリスト
            contexts: コンテキストデータのリスト
            true_labels: 真の定着ラベル
            
        Returns:
            Dict[str, float]: 評価メトリクス
        """
        # 予測
        predicted_probs = self.predict_batch(developers, contexts)
        predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probs]
        
        # メトリクス計算
        metrics = {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'precision': precision_score(true_labels, predicted_labels, zero_division=0),
            'recall': recall_score(true_labels, predicted_labels, zero_division=0),
            'f1_score': f1_score(true_labels, predicted_labels, zero_division=0),
            'auc_score': roc_auc_score(true_labels, predicted_probs)
        }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """モデルを保存"""
        
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'model_config': self.model_config,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"定着予測モデルを保存しました: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """モデルを読み込み"""
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.model_weights = model_data['model_weights']
        self.model_config = model_data['model_config']
        self.feature_names = model_data.get('feature_names')
        self.is_fitted = model_data['is_fitted']
        
        # SHAP説明器を再初期化
        self._reinitialize_shap_explainers()
        
        logger.info(f"定着予測モデルを読み込みました: {filepath}")
    
    def _reinitialize_shap_explainers(self):
        """SHAP説明器を再初期化"""
        
        self.shap_explainers = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name in ['random_forest', 'xgboost']:
                    self.shap_explainers[model_name] = shap.TreeExplainer(model)
                # Neural Networkは再初期化時にはスキップ（計算コストが高いため）
            except Exception as e:
                logger.warning(f"{model_name}のSHAP説明器初期化でエラー: {e}")


def create_retention_predictor(config_path: str) -> RetentionPredictor:
    """
    設定ファイルから定着予測器を作成
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        RetentionPredictor: 設定済みの予測器
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return RetentionPredictor(config.get('retention_prediction', {}))


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_developers = [
        {
            'email': 'dev1@example.com',
            'expertise_areas': {'python', 'machine_learning'},
            'current_workload': 0.8,
            'ideal_workload': 1.0,
            'collaboration_quality': 0.7,
            'community_participation': 0.6,
            'team_status': 0.8,
            'expertise_growth_rate': 0.1,
            'challenging_task_ratio': 0.3
        },
        {
            'email': 'dev2@example.com',
            'expertise_areas': {'java', 'backend'},
            'current_workload': 1.2,
            'ideal_workload': 1.0,
            'collaboration_quality': 0.9,
            'community_participation': 0.8,
            'team_status': 0.6,
            'expertise_growth_rate': 0.05,
            'challenging_task_ratio': 0.2
        }
    ]
    
    sample_contexts = [
        {
            'context_date': datetime(2023, 6, 1),
            'current_tasks': [
                {'technical_domains': ['python', 'machine_learning']}
            ],
            'learning_opportunities': 0.7,
            'project_phase': 'development',
            'changes_data': [],
            'reviews_data': []
        },
        {
            'context_date': datetime(2023, 6, 1),
            'current_tasks': [
                {'technical_domains': ['java', 'database']}
            ],
            'learning_opportunities': 0.4,
            'project_phase': 'testing',
            'changes_data': [],
            'reviews_data': []
        }
    ]
    
    sample_labels = [1, 0]  # dev1は定着、dev2は離脱
    
    # 予測器のテスト
    config = {
        'feature_extraction': {
            'retention_window_days': 180,
            'activity_threshold': 1,
            'feature_integration': {}
        },
        'random_forest': {
            'n_estimators': 50,
            'max_depth': 5
        },
        'xgboost': {
            'n_estimators': 50,
            'max_depth': 3
        },
        'neural_network': {
            'hidden_layer_sizes': (50, 25),
            'max_iter': 200
        }
    }
    
    predictor = RetentionPredictor(config)
    
    # モデル訓練
    predictor.fit(sample_developers, sample_contexts, sample_labels)
    
    # 予測テスト
    for i, (developer, context) in enumerate(zip(sample_developers, sample_contexts)):
        prob = predictor.predict_retention_probability(developer, context)
        factors = predictor.analyze_retention_factors(developer, context)
        
        print(f"\n開発者 {i+1} ({developer['email']}):")
        print(f"定着確率: {prob:.3f}")
        print("主要要因:")
        for factor, value in factors.items():
            print(f"  {factor}: {value:.3f}")
    
    # モデル評価
    metrics = predictor.evaluate_model(sample_developers, sample_contexts, sample_labels)
    print(f"\nモデル性能:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")