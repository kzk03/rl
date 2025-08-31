"""
継続学習システム

オンライン学習による推薦改善、概念ドリフト検出・対応、
新規開発者への迅速適応を行うシステムです。
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LearningInstance:
    """学習インスタンス"""
    features: np.ndarray
    target: float
    timestamp: datetime
    developer_id: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptDriftDetection:
    """概念ドリフト検出結果"""
    is_drift_detected: bool
    drift_magnitude: float
    drift_type: str  # 'gradual', 'sudden', 'recurring'
    affected_features: List[str]
    detection_timestamp: datetime
    confidence: float


class ContinualLearner:
    """
    継続学習システム
    
    オンライン学習による推薦改善、概念ドリフト検出・対応、
    新規開発者への迅速適応を行います。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 継続学習設定
        """
        self.config = config
        
        # 学習データバッファ
        self.buffer_size = config.get('buffer_size', 1000)
        self.learning_buffer: deque = deque(maxlen=self.buffer_size)
        
        # モデル管理
        self.models: Dict[str, BaseEstimator] = {}
        self.model_performance: Dict[str, List[float]] = {}
        
        # 概念ドリフト検出
        self.drift_detection_window = config.get('drift_detection_window', 100)
        self.drift_threshold = config.get('drift_threshold', 0.1)
        self.performance_history: deque = deque(maxlen=self.drift_detection_window)
        
        # 新規開発者適応
        self.new_developer_threshold = config.get('new_developer_threshold', 10)
        self.developer_data: Dict[str, List[LearningInstance]] = {}
        
        # 学習スケジュール
        self.learning_frequency = config.get('learning_frequency', 10)
        self.update_counter = 0
        
        logger.info("継続学習システムを初期化しました")
    
    def add_learning_instance(self, 
                            features: np.ndarray,
                            target: float,
                            developer_id: str,
                            context: Dict[str, Any] = None) -> None:
        """
        学習インスタンスを追加
        
        Args:
            features: 特徴量
            target: 目標値
            developer_id: 開発者ID
            context: コンテキスト情報
        """
        instance = LearningInstance(
            features=features,
            target=target,
            timestamp=datetime.now(),
            developer_id=developer_id,
            context=context or {}
        )
        
        # バッファに追加
        self.learning_buffer.append(instance)
        
        # 開発者別データに追加
        if developer_id not in self.developer_data:
            self.developer_data[developer_id] = []
        self.developer_data[developer_id].append(instance)
        
        # 定期的な学習実行
        self.update_counter += 1
        if self.update_counter >= self.learning_frequency:
            self._perform_incremental_learning()
            self.update_counter = 0
        
        logger.debug(f"学習インスタンスを追加: 開発者={developer_id}, 目標値={target}")
    
    def _perform_incremental_learning(self) -> None:
        """増分学習を実行"""
        if len(self.learning_buffer) < self.config.get('min_learning_samples', 10):
            return
        
        logger.info("増分学習を実行中...")
        
        # 最新データを取得
        recent_instances = list(self.learning_buffer)[-self.learning_frequency:]
        
        if not recent_instances:
            return
        
        # 特徴量とターゲットを抽出
        X = np.array([instance.features for instance in recent_instances])
        y = np.array([instance.target for instance in recent_instances])
        
        # モデル更新
        self._update_models(X, y)
        
        # 概念ドリフト検出
        drift_result = self._detect_concept_drift(recent_instances)
        if drift_result.is_drift_detected:
            self._handle_concept_drift(drift_result)
        
        logger.info("増分学習が完了しました")    
 
   def _update_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """モデルを更新"""
        for model_name in self.config.get('model_types', ['retention', 'stress', 'satisfaction']):
            if model_name not in self.models:
                # 新しいモデルを初期化
                self.models[model_name] = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )
                self.model_performance[model_name] = []
            
            model = self.models[model_name]
            
            try:
                # 既存モデルがある場合は部分的に更新
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X, y)
                else:
                    # Random Forestの場合は再訓練
                    if len(self.learning_buffer) > 50:
                        # 最新のデータで再訓練
                        recent_data = list(self.learning_buffer)[-200:]
                        X_retrain = np.array([inst.features for inst in recent_data])
                        y_retrain = np.array([inst.target for inst in recent_data])
                        model.fit(X_retrain, y_retrain)
                
                # 性能評価
                if len(X) > 1:
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    self.model_performance[model_name].append(mse)
                    
                    # 性能履歴を制限
                    if len(self.model_performance[model_name]) > 100:
                        self.model_performance[model_name] = \
                            self.model_performance[model_name][-100:]
                
            except Exception as e:
                logger.warning(f"モデル {model_name} の更新中にエラー: {e}")
    
    def _detect_concept_drift(self, recent_instances: List[LearningInstance]) -> ConceptDriftDetection:
        """概念ドリフトを検出"""
        if len(self.performance_history) < self.drift_detection_window // 2:
            return ConceptDriftDetection(
                is_drift_detected=False,
                drift_magnitude=0.0,
                drift_type="none",
                affected_features=[],
                detection_timestamp=datetime.now(),
                confidence=0.0
            )
        
        # 最新の性能を計算
        if recent_instances:
            X = np.array([inst.features for inst in recent_instances])
            y = np.array([inst.target for inst in recent_instances])
            
            current_performance = []
            for model_name, model in self.models.items():
                try:
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    current_performance.append(mse)
                except:
                    current_performance.append(1.0)  # デフォルト値
            
            avg_performance = np.mean(current_performance)
            self.performance_history.append(avg_performance)
        
        # ドリフト検出アルゴリズム
        if len(self.performance_history) < self.drift_detection_window:
            return ConceptDriftDetection(
                is_drift_detected=False,
                drift_magnitude=0.0,
                drift_type="none",
                affected_features=[],
                detection_timestamp=datetime.now(),
                confidence=0.0
            )
        
        # 性能の変化を分析
        recent_perf = list(self.performance_history)[-self.drift_detection_window//2:]
        older_perf = list(self.performance_history)[-self.drift_detection_window:-self.drift_detection_window//2]
        
        recent_mean = np.mean(recent_perf)
        older_mean = np.mean(older_perf)
        
        # ドリフトの大きさを計算
        drift_magnitude = abs(recent_mean - older_mean) / (older_mean + 1e-8)
        
        # ドリフト検出
        is_drift = drift_magnitude > self.drift_threshold
        
        # ドリフトタイプの判定
        drift_type = self._classify_drift_type(list(self.performance_history))
        
        # 影響を受けた特徴量の特定
        affected_features = self._identify_affected_features(recent_instances)
        
        return ConceptDriftDetection(
            is_drift_detected=is_drift,
            drift_magnitude=drift_magnitude,
            drift_type=drift_type,
            affected_features=affected_features,
            detection_timestamp=datetime.now(),
            confidence=min(drift_magnitude / self.drift_threshold, 1.0)
        ) 
   
    def _classify_drift_type(self, performance_history: List[float]) -> str:
        """ドリフトタイプを分類"""
        if len(performance_history) < 20:
            return "unknown"
        
        # 最近の傾向を分析
        recent_trend = np.polyfit(range(len(performance_history[-20:])), 
                                performance_history[-20:], 1)[0]
        
        # 変動の大きさを分析
        recent_std = np.std(performance_history[-20:])
        overall_std = np.std(performance_history)
        
        if abs(recent_trend) > 0.01:
            return "gradual"
        elif recent_std > overall_std * 1.5:
            return "sudden"
        else:
            return "recurring"
    
    def _identify_affected_features(self, instances: List[LearningInstance]) -> List[str]:
        """影響を受けた特徴量を特定"""
        # 簡略化された実装
        # 実際の実装では特徴量の重要度変化を分析
        feature_names = [f"feature_{i}" for i in range(len(instances[0].features))]
        
        # ランダムに一部の特徴量を選択（実際の実装では統計的分析を使用）
        affected_count = min(3, len(feature_names))
        affected_indices = np.random.choice(len(feature_names), affected_count, replace=False)
        
        return [feature_names[i] for i in affected_indices]
    
    def _handle_concept_drift(self, drift_result: ConceptDriftDetection) -> None:
        """概念ドリフトに対応"""
        logger.warning(f"概念ドリフトを検出: タイプ={drift_result.drift_type}, "
                      f"大きさ={drift_result.drift_magnitude:.3f}")
        
        if drift_result.drift_type == "sudden":
            # 急激なドリフト: モデルをリセット
            self._reset_models()
            logger.info("急激なドリフトによりモデルをリセットしました")
            
        elif drift_result.drift_type == "gradual":
            # 段階的なドリフト: 学習率を調整
            self._adjust_learning_parameters(increase_rate=True)
            logger.info("段階的なドリフトにより学習率を調整しました")
            
        elif drift_result.drift_type == "recurring":
            # 周期的なドリフト: アンサンブル手法を使用
            self._enable_ensemble_learning()
            logger.info("周期的なドリフトによりアンサンブル学習を有効化しました")
    
    def _reset_models(self) -> None:
        """モデルをリセット"""
        for model_name in self.models:
            self.models[model_name] = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            self.model_performance[model_name] = []
        
        # 最新データで再訓練
        if len(self.learning_buffer) > 20:
            recent_data = list(self.learning_buffer)[-100:]
            X = np.array([inst.features for inst in recent_data])
            y = np.array([inst.target for inst in recent_data])
            
            for model in self.models.values():
                try:
                    model.fit(X, y)
                except Exception as e:
                    logger.warning(f"モデル再訓練中にエラー: {e}")
    
    def _adjust_learning_parameters(self, increase_rate: bool = True) -> None:
        """学習パラメータを調整"""
        if increase_rate:
            self.learning_frequency = max(5, self.learning_frequency // 2)
        else:
            self.learning_frequency = min(50, self.learning_frequency * 2)
        
        logger.info(f"学習頻度を {self.learning_frequency} に調整しました")
    
    def _enable_ensemble_learning(self) -> None:
        """アンサンブル学習を有効化"""
        # 簡略化された実装
        # 実際の実装では複数のモデルを組み合わせる
        logger.info("アンサンブル学習を有効化しました（実装予定）") 
   
    def adapt_to_new_developer(self, developer_id: str, 
                             initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        新規開発者に迅速適応
        
        Args:
            developer_id: 新規開発者ID
            initial_context: 初期コンテキスト情報
            
        Returns:
            Dict[str, Any]: 適応戦略
        """
        logger.info(f"新規開発者 {developer_id} への適応を開始")
        
        if initial_context is None:
            initial_context = {}
        
        # 類似開発者を検索
        similar_developers = self._find_similar_developers(developer_id, initial_context)
        
        # 類似開発者のデータから初期モデルを構築
        adaptation_strategy = self._create_adaptation_strategy(
            developer_id, similar_developers, initial_context
        )
        
        # 新規開発者用の専用バッファを作成
        if developer_id not in self.developer_data:
            self.developer_data[developer_id] = []
        
        logger.info(f"新規開発者 {developer_id} への適応が完了")
        
        return adaptation_strategy
    
    def _find_similar_developers(self, new_developer_id: str, 
                               context: Dict[str, Any]) -> List[str]:
        """類似開発者を検索"""
        similar_developers = []
        
        # 簡略化された類似度計算
        new_dev_features = self._extract_developer_features(context)
        
        for dev_id, instances in self.developer_data.items():
            if len(instances) < self.new_developer_threshold:
                continue
            
            # 開発者の平均的な特徴量を計算
            dev_features = np.mean([inst.features for inst in instances[-20:]], axis=0)
            
            # コサイン類似度を計算
            similarity = self._calculate_cosine_similarity(new_dev_features, dev_features)
            
            if similarity > 0.7:  # 閾値
                similar_developers.append(dev_id)
        
        return similar_developers[:5]  # 上位5人
    
    def _extract_developer_features(self, context: Dict[str, Any]) -> np.ndarray:
        """開発者の特徴量を抽出"""
        # 簡略化された特徴量抽出
        features = [
            context.get('expertise_level', 0.5),
            context.get('activity_level', 0.5),
            context.get('collaboration_score', 0.5),
            context.get('domain_experience', 0.5),
            context.get('review_quality', 0.5)
        ]
        
        return np.array(features)
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """コサイン類似度を計算"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _create_adaptation_strategy(self, 
                                  new_developer_id: str,
                                  similar_developers: List[str],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """適応戦略を作成"""
        strategy = {
            'developer_id': new_developer_id,
            'similar_developers': similar_developers,
            'learning_rate_multiplier': 2.0,  # 新規開発者は学習率を上げる
            'exploration_bonus': 0.3,  # 探索を促進
            'safety_threshold': 0.8,  # 安全な推薦の閾値を上げる
            'feedback_weight': 1.5,  # フィードバックの重みを増加
            'adaptation_period_days': 30,  # 適応期間
            'created_at': datetime.now()
        }
        
        # 類似開発者のパターンを分析
        if similar_developers:
            strategy['recommended_patterns'] = self._analyze_similar_patterns(similar_developers)
        
        return strategy    

    def _analyze_similar_patterns(self, similar_developers: List[str]) -> Dict[str, Any]:
        """類似開発者のパターンを分析"""
        patterns = {
            'preferred_complexity': [],
            'optimal_workload': [],
            'collaboration_preference': [],
            'learning_velocity': []
        }
        
        for dev_id in similar_developers:
            if dev_id in self.developer_data:
                instances = self.developer_data[dev_id]
                
                # パターンを抽出（簡略化）
                if instances:
                    avg_target = np.mean([inst.target for inst in instances])
                    patterns['preferred_complexity'].append(avg_target)
                    
                    # コンテキストから他の情報を抽出
                    for inst in instances[-10:]:  # 最新10件
                        context = inst.context
                        patterns['optimal_workload'].append(
                            context.get('workload', 0.5)
                        )
                        patterns['collaboration_preference'].append(
                            context.get('collaboration_score', 0.5)
                        )
                        patterns['learning_velocity'].append(
                            context.get('learning_rate', 0.5)
                        )
        
        # 平均値を計算
        analyzed_patterns = {}
        for key, values in patterns.items():
            if values:
                analyzed_patterns[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values)
                }
        
        return analyzed_patterns
    
    def predict_with_uncertainty(self, 
                               features: np.ndarray,
                               model_name: str = 'retention') -> Tuple[float, float]:
        """
        不確実性を含む予測
        
        Args:
            features: 特徴量
            model_name: モデル名
            
        Returns:
            Tuple[float, float]: (予測値, 不確実性)
        """
        if model_name not in self.models:
            return 0.5, 1.0  # デフォルト値
        
        model = self.models[model_name]
        
        try:
            # 基本予測
            prediction = model.predict(features.reshape(1, -1))[0]
            
            # 不確実性の推定（簡略化）
            if hasattr(model, 'estimators_'):
                # Random Forestの場合、各木の予測のばらつきから不確実性を推定
                tree_predictions = [
                    tree.predict(features.reshape(1, -1))[0] 
                    for tree in model.estimators_
                ]
                uncertainty = np.std(tree_predictions)
            else:
                # 最近の性能から不確実性を推定
                recent_performance = self.model_performance.get(model_name, [0.1])
                uncertainty = np.mean(recent_performance[-10:]) if recent_performance else 0.1
            
            return prediction, uncertainty
            
        except Exception as e:
            logger.warning(f"予測中にエラー: {e}")
            return 0.5, 1.0
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """学習統計を取得"""
        stats = {
            'buffer_size': len(self.learning_buffer),
            'total_developers': len(self.developer_data),
            'models_count': len(self.models),
            'update_counter': self.update_counter,
            'learning_frequency': self.learning_frequency
        }
        
        # モデル性能統計
        for model_name, performance_history in self.model_performance.items():
            if performance_history:
                stats[f'{model_name}_performance'] = {
                    'mean_mse': np.mean(performance_history),
                    'recent_mse': np.mean(performance_history[-10:]) if len(performance_history) >= 10 else np.mean(performance_history),
                    'performance_trend': self._calculate_performance_trend(performance_history)
                }
        
        # 開発者別統計
        developer_stats = {}
        for dev_id, instances in self.developer_data.items():
            developer_stats[dev_id] = {
                'instance_count': len(instances),
                'latest_update': instances[-1].timestamp if instances else None,
                'avg_target': np.mean([inst.target for inst in instances]) if instances else 0.0
            }
        
        stats['developer_statistics'] = developer_stats
        
        return stats
    
    def _calculate_performance_trend(self, performance_history: List[float]) -> str:
        """性能トレンドを計算"""
        if len(performance_history) < 5:
            return "insufficient_data"
        
        recent = performance_history[-5:]
        older = performance_history[-10:-5] if len(performance_history) >= 10 else performance_history[:-5]
        
        if not older:
            return "insufficient_data"
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg < older_avg * 0.95:
            return "improving"
        elif recent_avg > older_avg * 1.05:
            return "degrading"
        else:
            return "stable"