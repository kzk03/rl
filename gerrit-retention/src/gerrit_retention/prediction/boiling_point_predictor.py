"""
開発者沸点予測器

開発者のストレス限界点（沸点）を予測し、離脱リスクを評価するモジュール。
SVR（Support Vector Regression）ベースの予測モデルを使用して、
過去の離脱パターンから個々の開発者の沸点を学習・予測する。

主要機能:
- 過去の離脱パターン学習
- 沸点閾値の推定
- リスクレベル分類
- 早期警告システム
- 沸点到達時間予測
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from gerrit_retention.prediction.stress_analyzer import (
        StressAnalyzer,
        StressIndicators,
    )
    from gerrit_retention.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class BoilingPointPrediction:
    """沸点予測結果データクラス"""
    developer_email: str
    current_stress: float
    predicted_boiling_point: float
    stress_margin: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    time_to_boiling_days: Optional[float]
    confidence_score: float
    contributing_factors: Dict[str, float]
    predicted_at: datetime


@dataclass
class DeveloperExitPattern:
    """開発者離脱パターンデータクラス"""
    developer_email: str
    exit_date: datetime
    stress_history: List[float]
    final_stress_level: float
    exit_trigger: str  # 'workload', 'social', 'compatibility', 'temporal'
    days_before_exit: int


class BoilingPointPredictor:
    """
    開発者沸点予測器
    
    SVRベースの機械学習モデルを使用して、開発者の
    ストレス限界点（沸点）を予測し、離脱リスクを評価する。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        沸点予測器を初期化
        
        Args:
            config: 沸点予測の設定辞書
        """
        self.config = config
        self.model_config = config.get('boiling_point_model', {})
        
        # SVRモデルの設定
        self.svr_params = self.model_config.get('svr_params', {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'epsilon': 0.1
        })
        
        # リスク分類の閾値
        self.risk_thresholds = config.get('risk_thresholds', {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        })
        
        # 予測モデルとスケーラー
        self.model: Optional[SVR] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        
        # ストレス分析器
        self.stress_analyzer = StressAnalyzer(config.get('stress_config', {}))
        
        logger.info("沸点予測器が初期化されました")
    
    def train_model(self, exit_patterns: List[DeveloperExitPattern], 
                   developer_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        離脱パターンデータからモデルを訓練
        
        Args:
            exit_patterns: 過去の離脱パターンデータ
            developer_profiles: 開発者プロファイル情報
            
        Returns:
            Dict[str, float]: 訓練結果のメトリクス
        """
        try:
            if not exit_patterns:
                logger.warning("訓練データが空です")
                return {'error': 1.0}
            
            # 特徴量とターゲットを準備
            X, y = self._prepare_training_data(exit_patterns, developer_profiles)
            
            if len(X) < 10:  # 最小データ数チェック
                logger.warning(f"訓練データが不足しています: {len(X)}件")
                return {'error': 1.0, 'data_count': len(X)}
            
            # データを訓練・テスト用に分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 特徴量を正規化
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # SVRモデルを訓練
            self.model = SVR(**self.svr_params)
            self.model.fit(X_train_scaled, y_train)
            
            # モデル性能を評価
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            metrics = {
                'mse': mse,
                'r2_score': r2,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logger.info(f"沸点予測モデルの訓練完了: MSE={mse:.4f}, R2={r2:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"モデル訓練中にエラーが発生: {e}")
            return {'error': 1.0, 'message': str(e)}
    
    def predict_boiling_point(self, developer: Dict[str, Any], 
                            context: Dict[str, Any]) -> BoilingPointPrediction:
        """
        開発者の沸点を予測
        
        Args:
            developer: 開発者情報
            context: コンテキスト情報
            
        Returns:
            BoilingPointPrediction: 沸点予測結果
        """
        try:
            if not self.is_trained:
                logger.warning("モデルが訓練されていません。デフォルト予測を返します")
                return self._get_default_prediction(developer)
            
            # 現在のストレス状態を分析
            stress_indicators = self.stress_analyzer.calculate_stress_indicators(developer, context)
            
            # 予測用特徴量を準備
            features = self._extract_prediction_features(developer, stress_indicators)
            features_scaled = self.scaler.transform([features])
            
            # 沸点を予測
            predicted_boiling_point = self.model.predict(features_scaled)[0]
            predicted_boiling_point = max(0.0, min(1.0, predicted_boiling_point))  # 0-1に制限
            
            # ストレス余裕度を計算
            current_stress = stress_indicators.total_stress
            stress_margin = predicted_boiling_point - current_stress
            
            # リスクレベルを判定
            risk_level = self._calculate_risk_level(current_stress, predicted_boiling_point)
            
            # 沸点到達時間を推定
            time_to_boiling = self._estimate_time_to_boiling(
                current_stress, predicted_boiling_point, developer
            )
            
            # 信頼度スコアを計算
            confidence_score = self._calculate_confidence_score(features, developer)
            
            # 寄与要因を分析
            contributing_factors = self._analyze_contributing_factors(stress_indicators)
            
            prediction = BoilingPointPrediction(
                developer_email=developer.get('email', 'unknown'),
                current_stress=current_stress,
                predicted_boiling_point=predicted_boiling_point,
                stress_margin=stress_margin,
                risk_level=risk_level,
                time_to_boiling_days=time_to_boiling,
                confidence_score=confidence_score,
                contributing_factors=contributing_factors,
                predicted_at=datetime.now()
            )
            
            logger.debug(f"開発者 {developer.get('email', 'unknown')} の沸点予測完了: {risk_level}")
            return prediction
            
        except Exception as e:
            logger.error(f"沸点予測中にエラーが発生: {e}")
            return self._get_default_prediction(developer)
    
    def _prepare_training_data(self, exit_patterns: List[DeveloperExitPattern], 
                             developer_profiles: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        訓練データを準備
        
        Args:
            exit_patterns: 離脱パターンデータ
            developer_profiles: 開発者プロファイル
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 特徴量とターゲット
        """
        features = []
        targets = []
        
        for pattern in exit_patterns:
            developer_profile = developer_profiles.get(pattern.developer_email, {})
            
            # 特徴量を抽出
            feature_vector = [
                # ストレス履歴の統計
                np.mean(pattern.stress_history) if pattern.stress_history else 0.5,
                np.std(pattern.stress_history) if len(pattern.stress_history) > 1 else 0.0,
                np.max(pattern.stress_history) if pattern.stress_history else 0.5,
                
                # 開発者特性
                developer_profile.get('collaboration_quality', 0.5),
                developer_profile.get('expertise_level', 0.5),
                developer_profile.get('workload_ratio', 0.5),
                developer_profile.get('recent_rejection_rate', 0.0),
                
                # 活動パターン
                len(developer_profile.get('expertise_areas', [])) / 10.0,  # 正規化
                developer_profile.get('avg_response_time_hours', 24.0) / 48.0,  # 正規化
                
                # 離脱前の期間
                min(pattern.days_before_exit / 30.0, 1.0),  # 30日で正規化
                
                # 離脱トリガーのワンホットエンコーディング
                1.0 if pattern.exit_trigger == 'workload' else 0.0,
                1.0 if pattern.exit_trigger == 'social' else 0.0,
                1.0 if pattern.exit_trigger == 'compatibility' else 0.0,
                1.0 if pattern.exit_trigger == 'temporal' else 0.0,
            ]
            
            features.append(feature_vector)
            targets.append(pattern.final_stress_level)
        
        return np.array(features), np.array(targets)
    
    def _extract_prediction_features(self, developer: Dict[str, Any], 
                                   stress_indicators: StressIndicators) -> List[float]:
        """
        予測用特徴量を抽出
        
        Args:
            developer: 開発者情報
            stress_indicators: ストレス指標
            
        Returns:
            List[float]: 特徴量ベクトル
        """
        return [
            # 現在のストレス状態
            stress_indicators.total_stress,
            stress_indicators.task_compatibility_stress,
            stress_indicators.workload_stress,
            
            # 開発者特性
            developer.get('collaboration_quality', 0.5),
            developer.get('expertise_level', 0.5),
            developer.get('workload_ratio', 0.5),
            developer.get('recent_rejection_rate', 0.0),
            
            # 活動パターン
            len(developer.get('expertise_areas', [])) / 10.0,
            developer.get('avg_response_time_hours', 24.0) / 48.0,
            
            # 時間的要因（仮想的な離脱前期間として現在の状態継続期間を使用）
            0.5,  # デフォルト値
            
            # 主要ストレス要因の推定
            1.0 if stress_indicators.workload_stress > 0.6 else 0.0,
            1.0 if stress_indicators.social_stress > 0.6 else 0.0,
            1.0 if stress_indicators.task_compatibility_stress > 0.6 else 0.0,
            1.0 if stress_indicators.temporal_stress > 0.6 else 0.0,
        ]
    
    def _calculate_risk_level(self, current_stress: float, predicted_boiling_point: float) -> str:
        """
        リスクレベルを計算
        
        Args:
            current_stress: 現在のストレス
            predicted_boiling_point: 予測沸点
            
        Returns:
            str: リスクレベル
        """
        if predicted_boiling_point <= 0:
            return 'critical'
        
        stress_ratio = current_stress / predicted_boiling_point
        
        if stress_ratio < self.risk_thresholds['low']:
            return 'low'
        elif stress_ratio < self.risk_thresholds['medium']:
            return 'medium'
        elif stress_ratio < self.risk_thresholds['high']:
            return 'high'
        else:
            return 'critical'
    
    def _estimate_time_to_boiling(self, current_stress: float, predicted_boiling_point: float, 
                                developer: Dict[str, Any]) -> Optional[float]:
        """
        沸点到達時間を推定
        
        Args:
            current_stress: 現在のストレス
            predicted_boiling_point: 予測沸点
            developer: 開発者情報
            
        Returns:
            Optional[float]: 沸点到達までの日数（None if already at boiling point）
        """
        if current_stress >= predicted_boiling_point:
            return 0.0
        
        stress_margin = predicted_boiling_point - current_stress
        
        # ストレス増加率を推定（開発者の活動パターンに基づく）
        workload_ratio = developer.get('workload_ratio', 0.5)
        stress_increase_rate = 0.01 * (1 + workload_ratio)  # 1日あたりのストレス増加率
        
        if stress_increase_rate <= 0:
            return None  # ストレスが増加しない場合
        
        days_to_boiling = stress_margin / stress_increase_rate
        return min(days_to_boiling, 365.0)  # 最大1年に制限
    
    def _calculate_confidence_score(self, features: List[float], developer: Dict[str, Any]) -> float:
        """
        予測の信頼度スコアを計算
        
        Args:
            features: 特徴量ベクトル
            developer: 開発者情報
            
        Returns:
            float: 信頼度スコア (0.0-1.0)
        """
        # 基本信頼度
        base_confidence = 0.7 if self.is_trained else 0.3
        
        # データ品質による調整
        data_completeness = sum(1 for f in features if f != 0.5) / len(features)
        
        # 開発者履歴の豊富さによる調整
        history_richness = min(len(developer.get('expertise_areas', [])) / 5.0, 1.0)
        
        confidence = base_confidence * 0.6 + data_completeness * 0.2 + history_richness * 0.2
        return min(confidence, 1.0)
    
    def _analyze_contributing_factors(self, stress_indicators: StressIndicators) -> Dict[str, float]:
        """
        沸点予測に寄与する要因を分析
        
        Args:
            stress_indicators: ストレス指標
            
        Returns:
            Dict[str, float]: 寄与要因とその重要度
        """
        # 各ストレス要素の重みを使用して寄与度を計算
        weights = self.stress_analyzer.stress_weights
        
        return {
            'task_compatibility': weights.get('task_compatibility_stress', 0.3),
            'workload': weights.get('workload_stress', 0.4),
            'social': weights.get('social_stress', 0.2),
            'temporal': weights.get('temporal_stress', 0.1)
        }
    
    def _get_default_prediction(self, developer: Dict[str, Any]) -> BoilingPointPrediction:
        """
        デフォルトの沸点予測を返す（モデル未訓練時のフォールバック）
        
        Args:
            developer: 開発者情報
            
        Returns:
            BoilingPointPrediction: デフォルト予測
        """
        return BoilingPointPrediction(
            developer_email=developer.get('email', 'unknown'),
            current_stress=0.5,
            predicted_boiling_point=0.7,
            stress_margin=0.2,
            risk_level='medium',
            time_to_boiling_days=30.0,
            confidence_score=0.3,
            contributing_factors={
                'task_compatibility': 0.25,
                'workload': 0.25,
                'social': 0.25,
                'temporal': 0.25
            },
            predicted_at=datetime.now()
        )
    
    def save_model(self, filepath: str) -> bool:
        """
        訓練済みモデルを保存
        
        Args:
            filepath: 保存先ファイルパス
            
        Returns:
            bool: 保存成功の場合True
        """
        try:
            if not self.is_trained:
                logger.warning("訓練されていないモデルは保存できません")
                return False
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config,
                'svr_params': self.svr_params,
                'risk_thresholds': self.risk_thresholds
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"沸点予測モデルを保存しました: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"モデル保存中にエラーが発生: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        保存済みモデルを読み込み
        
        Args:
            filepath: モデルファイルパス
            
        Returns:
            bool: 読み込み成功の場合True
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.svr_params = model_data.get('svr_params', self.svr_params)
            self.risk_thresholds = model_data.get('risk_thresholds', self.risk_thresholds)
            self.is_trained = True
            
            logger.info(f"沸点予測モデルを読み込みました: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"モデル読み込み中にエラーが発生: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得
        
        Returns:
            Dict[str, Any]: モデル情報
        """
        return {
            'is_trained': self.is_trained,
            'svr_params': self.svr_params,
            'risk_thresholds': self.risk_thresholds,
            'model_type': 'SVR',
            'feature_count': 14 if self.is_trained else None
        }