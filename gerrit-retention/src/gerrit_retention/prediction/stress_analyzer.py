"""
開発者ストレス分析器

開発者の多次元ストレス指標を計算し、総合ストレススコアを算出するモジュール。
Gerrit環境での開発者のストレス状態を定量化し、沸点予測の基盤を提供する。

主要機能:
- レビュー適合度ストレスの計算
- ワークロードストレスの評価
- 社会的ストレスの分析
- 時間的ストレスの測定
- 総合ストレススコアの算出
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from gerrit_retention.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class StressIndicators:
    """ストレス指標データクラス"""
    task_compatibility_stress: float
    workload_stress: float
    social_stress: float
    temporal_stress: float
    total_stress: float
    stress_level: str  # 'low', 'medium', 'high', 'critical'
    calculated_at: datetime


class StressAnalyzer:
    """
    開発者ストレス分析器
    
    Gerrit環境での開発者のストレス状態を多次元で分析し、
    定量的なストレス指標を提供する。
    """
    
    def __init__(self, stress_config: Dict[str, Any]):
        """
        ストレス分析器を初期化
        
        Args:
            stress_config: ストレス分析の設定辞書
        """
        self.stress_config = stress_config
        self.stress_weights = stress_config.get('weights', {
            'task_compatibility_stress': 0.3,
            'workload_stress': 0.4,
            'social_stress': 0.2,
            'temporal_stress': 0.1
        })
        
        # Gerrit特有のストレス閾値
        self.gerrit_thresholds = stress_config.get('gerrit_stress_factors', {
            'high_complexity_threshold': 0.8,
            'review_queue_size_threshold': 5,
            'response_time_pressure_threshold': 24  # hours
        })
        
        logger.info("ストレス分析器が初期化されました")
    
    def calculate_stress_indicators(self, developer: Dict[str, Any], 
                                  context: Dict[str, Any]) -> StressIndicators:
        """
        開発者のストレス指標を計算
        
        Args:
            developer: 開発者情報辞書
            context: コンテキスト情報（プロジェクト状態、時間等）
            
        Returns:
            StressIndicators: 計算されたストレス指標
        """
        try:
            # 各次元のストレスを計算
            task_stress = self._calculate_task_compatibility_stress(developer, context)
            workload_stress = self._calculate_workload_stress(developer, context)
            social_stress = self._calculate_social_stress(developer, context)
            temporal_stress = self._calculate_temporal_stress(developer, context)
            
            # 総合ストレススコアを計算
            total_stress = self._calculate_total_stress({
                'task_compatibility_stress': task_stress,
                'workload_stress': workload_stress,
                'social_stress': social_stress,
                'temporal_stress': temporal_stress
            })
            
            # ストレスレベルを判定
            stress_level = self._determine_stress_level(total_stress)
            
            indicators = StressIndicators(
                task_compatibility_stress=task_stress,
                workload_stress=workload_stress,
                social_stress=social_stress,
                temporal_stress=temporal_stress,
                total_stress=total_stress,
                stress_level=stress_level,
                calculated_at=datetime.now()
            )
            
            logger.debug(f"開発者 {developer.get('email', 'unknown')} のストレス指標を計算: {stress_level}")
            return indicators
            
        except Exception as e:
            logger.error(f"ストレス指標計算中にエラーが発生: {e}")
            # フォールバック値を返す
            return self._get_default_stress_indicators()
    
    def _calculate_task_compatibility_stress(self, developer: Dict[str, Any], 
                                           context: Dict[str, Any]) -> float:
        """
        タスク適合度ストレスを計算
        
        専門外タスクの割合、レビュー対象の複雑度、技術領域の適合度を評価
        
        Args:
            developer: 開発者情報
            context: コンテキスト情報
            
        Returns:
            float: タスク適合度ストレス (0.0-1.0)
        """
        try:
            # 開発者の専門領域を取得
            expertise_areas = set(developer.get('expertise_areas', []))
            
            # 現在のレビューキューを分析
            review_queue = context.get('review_queue', [])
            if not review_queue:
                return 0.0
            
            mismatch_count = 0
            high_complexity_count = 0
            
            for review in review_queue:
                # 技術領域の不適合をチェック
                review_domain = review.get('technical_domain', '')
                if review_domain and review_domain not in expertise_areas:
                    mismatch_count += 1
                
                # 高複雑度タスクをチェック
                complexity = review.get('complexity_score', 0.0)
                if complexity > self.gerrit_thresholds['high_complexity_threshold']:
                    high_complexity_count += 1
            
            # 不適合率を計算
            mismatch_ratio = mismatch_count / len(review_queue) if review_queue else 0.0
            complexity_ratio = high_complexity_count / len(review_queue) if review_queue else 0.0
            
            # 適合度ストレスを算出 (不適合率と複雑度の重み付き平均)
            compatibility_stress = (mismatch_ratio * 0.7 + complexity_ratio * 0.3)
            
            return min(compatibility_stress, 1.0)
            
        except Exception as e:
            logger.warning(f"タスク適合度ストレス計算エラー: {e}")
            return 0.5  # デフォルト値
    
    def _calculate_workload_stress(self, developer: Dict[str, Any], 
                                 context: Dict[str, Any]) -> float:
        """
        ワークロードストレスを計算
        
        同時進行タスク数、締切プレッシャー、レビュー負荷を評価
        
        Args:
            developer: 開発者情報
            context: コンテキスト情報
            
        Returns:
            float: ワークロードストレス (0.0-1.0)
        """
        try:
            # レビューキューサイズ
            review_queue_size = len(context.get('review_queue', []))
            queue_stress = min(review_queue_size / self.gerrit_thresholds['review_queue_size_threshold'], 1.0)
            
            # 進行中のタスク数
            active_tasks = developer.get('active_tasks_count', 0)
            task_stress = min(active_tasks / 3.0, 1.0)  # 3つ以上で最大ストレス
            
            # 締切プレッシャー
            urgent_reviews = 0
            review_queue = context.get('review_queue', [])
            current_time = datetime.now()
            
            for review in review_queue:
                deadline = review.get('deadline')
                if deadline:
                    if isinstance(deadline, str):
                        deadline = datetime.fromisoformat(deadline)
                    
                    time_to_deadline = (deadline - current_time).total_seconds() / 3600  # hours
                    if time_to_deadline < self.gerrit_thresholds['response_time_pressure_threshold']:
                        urgent_reviews += 1
            
            deadline_stress = min(urgent_reviews / max(len(review_queue), 1), 1.0)
            
            # ワークロードストレスの統合計算
            workload_stress = (queue_stress * 0.4 + task_stress * 0.3 + deadline_stress * 0.3)
            
            return min(workload_stress, 1.0)
            
        except Exception as e:
            logger.warning(f"ワークロードストレス計算エラー: {e}")
            return 0.5  # デフォルト値
    
    def _calculate_social_stress(self, developer: Dict[str, Any], 
                               context: Dict[str, Any]) -> float:
        """
        社会的ストレスを計算
        
        協力関係の質、コミュニケーション負荷、チーム内での立場を評価
        
        Args:
            developer: 開発者情報
            context: コンテキスト情報
            
        Returns:
            float: 社会的ストレス (0.0-1.0)
        """
        try:
            # 協力関係の質
            collaboration_quality = developer.get('collaboration_quality', 0.5)
            relationship_stress = 1.0 - collaboration_quality
            
            # レビュー拒否率（社会的プレッシャーの指標）
            recent_rejection_rate = developer.get('recent_rejection_rate', 0.0)
            rejection_stress = recent_rejection_rate
            
            # コミュニケーション負荷
            review_queue = context.get('review_queue', [])
            communication_load = 0.0
            
            for review in review_queue:
                # 新しい協力相手との作業はコミュニケーション負荷が高い
                requester_relationship = review.get('requester_relationship', 0.0)
                if requester_relationship < 0.3:  # 新しい関係
                    communication_load += 0.2
            
            communication_stress = min(communication_load, 1.0)
            
            # 社会的ストレスの統合計算
            social_stress = (relationship_stress * 0.4 + rejection_stress * 0.3 + communication_stress * 0.3)
            
            return min(social_stress, 1.0)
            
        except Exception as e:
            logger.warning(f"社会的ストレス計算エラー: {e}")
            return 0.5  # デフォルト値
    
    def _calculate_temporal_stress(self, developer: Dict[str, Any], 
                                 context: Dict[str, Any]) -> float:
        """
        時間的ストレスを計算
        
        作業時間パターン、応答時間プレッシャー、時間的制約を評価
        
        Args:
            developer: 開発者情報
            context: コンテキスト情報
            
        Returns:
            float: 時間的ストレス (0.0-1.0)
        """
        try:
            current_time = datetime.now()
            
            # 作業時間パターンのストレス
            activity_pattern = developer.get('activity_pattern', {})
            current_hour = current_time.hour
            
            # 通常の活動時間外での作業要求
            normal_hours = activity_pattern.get('peak_hours', [9, 10, 11, 14, 15, 16])
            if current_hour not in normal_hours:
                time_pattern_stress = 0.3
            else:
                time_pattern_stress = 0.0
            
            # 応答時間プレッシャー
            avg_response_time = developer.get('avg_response_time_hours', 24.0)
            expected_response_time = self.gerrit_thresholds['response_time_pressure_threshold']
            
            if avg_response_time < expected_response_time:
                response_pressure = (expected_response_time - avg_response_time) / expected_response_time
            else:
                response_pressure = 0.0
            
            # 連続作業時間のストレス
            continuous_work_hours = context.get('continuous_work_hours', 0)
            continuous_stress = min(continuous_work_hours / 8.0, 1.0)  # 8時間以上で最大
            
            # 時間的ストレスの統合計算
            temporal_stress = (time_pattern_stress * 0.3 + response_pressure * 0.4 + continuous_stress * 0.3)
            
            return min(temporal_stress, 1.0)
            
        except Exception as e:
            logger.warning(f"時間的ストレス計算エラー: {e}")
            return 0.5  # デフォルト値
    
    def _calculate_total_stress(self, stress_components: Dict[str, float]) -> float:
        """
        総合ストレススコアを計算
        
        Args:
            stress_components: 各次元のストレス値
            
        Returns:
            float: 総合ストレススコア (0.0-1.0)
        """
        total_stress = 0.0
        
        for component, value in stress_components.items():
            weight = self.stress_weights.get(component, 0.25)
            total_stress += value * weight
        
        return min(total_stress, 1.0)
    
    def _determine_stress_level(self, total_stress: float) -> str:
        """
        総合ストレススコアからストレスレベルを判定
        
        Args:
            total_stress: 総合ストレススコア
            
        Returns:
            str: ストレスレベル ('low', 'medium', 'high', 'critical')
        """
        if total_stress < 0.3:
            return 'low'
        elif total_stress < 0.6:
            return 'medium'
        elif total_stress < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _get_default_stress_indicators(self) -> StressIndicators:
        """
        デフォルトのストレス指標を返す（エラー時のフォールバック）
        
        Returns:
            StressIndicators: デフォルト値のストレス指標
        """
        return StressIndicators(
            task_compatibility_stress=0.5,
            workload_stress=0.5,
            social_stress=0.5,
            temporal_stress=0.5,
            total_stress=0.5,
            stress_level='medium',
            calculated_at=datetime.now()
        )
    
    def get_stress_breakdown(self, indicators: StressIndicators) -> Dict[str, Any]:
        """
        ストレス指標の詳細分解を取得
        
        Args:
            indicators: ストレス指標
            
        Returns:
            Dict[str, Any]: ストレス分解情報
        """
        return {
            'total_stress': indicators.total_stress,
            'stress_level': indicators.stress_level,
            'components': {
                'task_compatibility': {
                    'value': indicators.task_compatibility_stress,
                    'weight': self.stress_weights.get('task_compatibility_stress', 0.3),
                    'contribution': indicators.task_compatibility_stress * self.stress_weights.get('task_compatibility_stress', 0.3)
                },
                'workload': {
                    'value': indicators.workload_stress,
                    'weight': self.stress_weights.get('workload_stress', 0.4),
                    'contribution': indicators.workload_stress * self.stress_weights.get('workload_stress', 0.4)
                },
                'social': {
                    'value': indicators.social_stress,
                    'weight': self.stress_weights.get('social_stress', 0.2),
                    'contribution': indicators.social_stress * self.stress_weights.get('social_stress', 0.2)
                },
                'temporal': {
                    'value': indicators.temporal_stress,
                    'weight': self.stress_weights.get('temporal_stress', 0.1),
                    'contribution': indicators.temporal_stress * self.stress_weights.get('temporal_stress', 0.1)
                }
            },
            'calculated_at': indicators.calculated_at.isoformat()
        }
    
    def is_stress_critical(self, indicators: StressIndicators) -> bool:
        """
        ストレスが危険レベルかどうかを判定
        
        Args:
            indicators: ストレス指標
            
        Returns:
            bool: 危険レベルの場合True
        """
        return indicators.stress_level in ['high', 'critical'] or indicators.total_stress > 0.75