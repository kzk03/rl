"""
適応戦略管理システム

開発者の状態（ストレスレベル、専門性成長段階、活動パターン）に基づいて
推薦戦略を動的に調整するシステムです。
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..behavior_analysis.preference_analyzer import PreferenceAnalyzer
from ..prediction.stress_analyzer import StressAnalyzer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StressLevel(Enum):
    """ストレスレベル分類"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ExpertiseStage(Enum):
    """専門性成長段階"""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ActivityPattern(Enum):
    """活動パターン分類"""
    REGULAR = "regular"
    SPORADIC = "sporadic"
    INTENSIVE = "intensive"
    DECLINING = "declining"


@dataclass
class RecommendationStrategy:
    """推薦戦略設定"""
    name: str
    stress_weight: float
    expertise_weight: float
    diversity_weight: float
    workload_limit: float
    review_complexity_threshold: float
    collaboration_bonus: float
    learning_opportunity_weight: float
    description: str
@data
class
class DeveloperState:
    """開発者状態情報"""
    developer_id: str
    stress_level: StressLevel
    expertise_stage: ExpertiseStage
    activity_pattern: ActivityPattern
    current_workload: float
    recent_performance: float
    collaboration_score: float
    learning_velocity: float
    satisfaction_trend: float
    last_updated: datetime


class StrategyManager:
    """
    適応戦略管理システム
    
    開発者の状態に応じて推薦戦略を動的に調整します。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 戦略設定
        """
        self.config = config
        self.stress_analyzer = StressAnalyzer(config.get('stress_config', {}))
        self.preference_analyzer = PreferenceAnalyzer(config.get('preference_config', {}))
        
        # 戦略定義の初期化
        self.strategies = self._initialize_strategies()
        
        # 開発者状態キャッシュ
        self.developer_states: Dict[str, DeveloperState] = {}
        
        # 戦略適用履歴
        self.strategy_history: Dict[str, List[Tuple[datetime, str]]] = {}
        
        logger.info("適応戦略管理システムを初期化しました")
    
    def _initialize_strategies(self) -> Dict[str, RecommendationStrategy]:
        """推薦戦略を初期化"""
        strategies = {
            # 低ストレス戦略
            "low_stress_growth": RecommendationStrategy(
                name="低ストレス成長戦略",
                stress_weight=0.1,
                expertise_weight=0.4,
                diversity_weight=0.3,
                workload_limit=0.8,
                review_complexity_threshold=0.7,
                collaboration_bonus=0.2,
                learning_opportunity_weight=0.4,
                description="ストレスが低い開発者向けの成長重視戦略"
            ),    
        
            # 中程度ストレス戦略
            "moderate_stress_balance": RecommendationStrategy(
                name="中程度ストレスバランス戦略",
                stress_weight=0.3,
                expertise_weight=0.3,
                diversity_weight=0.2,
                workload_limit=0.6,
                review_complexity_threshold=0.5,
                collaboration_bonus=0.15,
                learning_opportunity_weight=0.2,
                description="中程度ストレス開発者向けのバランス戦略"
            ),
            
            # 高ストレス戦略
            "high_stress_relief": RecommendationStrategy(
                name="高ストレス軽減戦略",
                stress_weight=0.6,
                expertise_weight=0.2,
                diversity_weight=0.1,
                workload_limit=0.4,
                review_complexity_threshold=0.3,
                collaboration_bonus=0.1,
                learning_opportunity_weight=0.1,
                description="高ストレス開発者向けの負荷軽減戦略"
            ),
            
            # 危機的ストレス戦略
            "critical_stress_emergency": RecommendationStrategy(
                name="危機的ストレス緊急戦略",
                stress_weight=0.8,
                expertise_weight=0.1,
                diversity_weight=0.05,
                workload_limit=0.2,
                review_complexity_threshold=0.2,
                collaboration_bonus=0.05,
                learning_opportunity_weight=0.0,
                description="危機的ストレス開発者向けの緊急対応戦略"
            ),
            
            # 初心者戦略
            "novice_learning": RecommendationStrategy(
                name="初心者学習戦略",
                stress_weight=0.2,
                expertise_weight=0.1,
                diversity_weight=0.4,
                workload_limit=0.5,
                review_complexity_threshold=0.3,
                collaboration_bonus=0.3,
                learning_opportunity_weight=0.5,
                description="初心者開発者向けの学習重視戦略"
            ), 
           
            # エキスパート戦略
            "expert_mentoring": RecommendationStrategy(
                name="エキスパートメンタリング戦略",
                stress_weight=0.2,
                expertise_weight=0.5,
                diversity_weight=0.2,
                workload_limit=0.7,
                review_complexity_threshold=0.8,
                collaboration_bonus=0.4,
                learning_opportunity_weight=0.2,
                description="エキスパート開発者向けのメンタリング重視戦略"
            )
        }
        
        return strategies
    
    def update_developer_state(self, developer_id: str, 
                             context: Dict[str, Any]) -> DeveloperState:
        """
        開発者状態を更新
        
        Args:
            developer_id: 開発者ID
            context: コンテキスト情報
            
        Returns:
            DeveloperState: 更新された開発者状態
        """
        # ストレス分析
        stress_indicators = self.stress_analyzer.calculate_stress_indicators(
            {'developer_id': developer_id}, context
        )
        stress_level = self._classify_stress_level(stress_indicators)
        
        # 専門性段階の判定
        expertise_stage = self._classify_expertise_stage(developer_id, context)
        
        # 活動パターンの判定
        activity_pattern = self._classify_activity_pattern(developer_id, context)
        
        # 開発者状態の作成
        state = DeveloperState(
            developer_id=developer_id,
            stress_level=stress_level,
            expertise_stage=expertise_stage,
            activity_pattern=activity_pattern,
            current_workload=context.get('current_workload', 0.0),
            recent_performance=context.get('recent_performance', 0.5),
            collaboration_score=context.get('collaboration_score', 0.5),
            learning_velocity=context.get('learning_velocity', 0.5),
            satisfaction_trend=context.get('satisfaction_trend', 0.5),
            last_updated=datetime.now()
        )
        
        self.developer_states[developer_id] = state
        logger.debug(f"開発者 {developer_id} の状態を更新: {state}")
        
        return state    
 
   def select_strategy(self, developer_id: str, 
                       context: Dict[str, Any]) -> RecommendationStrategy:
        """
        開発者に最適な戦略を選択
        
        Args:
            developer_id: 開発者ID
            context: コンテキスト情報
            
        Returns:
            RecommendationStrategy: 選択された戦略
        """
        # 開発者状態を更新
        state = self.update_developer_state(developer_id, context)
        
        # 戦略選択ロジック
        strategy_name = self._determine_strategy_name(state)
        strategy = self.strategies[strategy_name]
        
        # 戦略適用履歴を記録
        if developer_id not in self.strategy_history:
            self.strategy_history[developer_id] = []
        self.strategy_history[developer_id].append((datetime.now(), strategy_name))
        
        logger.info(f"開発者 {developer_id} に戦略 '{strategy_name}' を選択")
        
        return strategy
    
    def _determine_strategy_name(self, state: DeveloperState) -> str:
        """戦略名を決定"""
        # ストレスレベル優先の戦略選択
        if state.stress_level == StressLevel.CRITICAL:
            return "critical_stress_emergency"
        elif state.stress_level == StressLevel.HIGH:
            return "high_stress_relief"
        elif state.expertise_stage == ExpertiseStage.NOVICE:
            return "novice_learning"
        elif state.expertise_stage == ExpertiseStage.EXPERT:
            return "expert_mentoring"
        elif state.stress_level == StressLevel.MODERATE:
            return "moderate_stress_balance"
        else:
            return "low_stress_growth"
    
    def _classify_stress_level(self, stress_indicators: Dict[str, float]) -> StressLevel:
        """ストレスレベルを分類"""
        total_stress = sum(stress_indicators.values()) / len(stress_indicators)
        
        if total_stress >= 0.8:
            return StressLevel.CRITICAL
        elif total_stress >= 0.6:
            return StressLevel.HIGH
        elif total_stress >= 0.4:
            return StressLevel.MODERATE
        else:
            return StressLevel.LOW    
    
def _classify_expertise_stage(self, developer_id: str, 
                                context: Dict[str, Any]) -> ExpertiseStage:
        """専門性段階を分類"""
        expertise_score = context.get('expertise_score', 0.0)
        experience_months = context.get('experience_months', 0)
        
        if expertise_score >= 0.8 and experience_months >= 24:
            return ExpertiseStage.EXPERT
        elif expertise_score >= 0.6 and experience_months >= 12:
            return ExpertiseStage.ADVANCED
        elif expertise_score >= 0.4 and experience_months >= 6:
            return ExpertiseStage.INTERMEDIATE
        else:
            return ExpertiseStage.NOVICE
    
    def _classify_activity_pattern(self, developer_id: str, 
                                 context: Dict[str, Any]) -> ActivityPattern:
        """活動パターンを分類"""
        recent_activity = context.get('recent_activity_trend', 0.0)
        activity_consistency = context.get('activity_consistency', 0.0)
        
        if recent_activity < -0.3:
            return ActivityPattern.DECLINING
        elif activity_consistency < 0.3:
            return ActivityPattern.SPORADIC
        elif recent_activity > 0.5:
            return ActivityPattern.INTENSIVE
        else:
            return ActivityPattern.REGULAR
    
    def adjust_recommendation_weights(self, developer_id: str, 
                                    base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        戦略に基づいて推薦重みを調整
        
        Args:
            developer_id: 開発者ID
            base_weights: ベース重み
            
        Returns:
            Dict[str, float]: 調整された重み
        """
        if developer_id not in self.developer_states:
            return base_weights
        
        state = self.developer_states[developer_id]
        strategy_name = self._determine_strategy_name(state)
        strategy = self.strategies[strategy_name]
        
        # 戦略に基づく重み調整
        adjusted_weights = base_weights.copy()
        adjusted_weights['stress_factor'] = strategy.stress_weight
        adjusted_weights['expertise_match'] = strategy.expertise_weight
        adjusted_weights['diversity_bonus'] = strategy.diversity_weight
        adjusted_weights['collaboration_bonus'] = strategy.collaboration_bonus
        adjusted_weights['learning_opportunity'] = strategy.learning_opportunity_weight
        
        logger.debug(f"開発者 {developer_id} の推薦重みを調整: {adjusted_weights}")
        
        return adjusted_weights   
 
    def get_workload_limit(self, developer_id: str) -> float:
        """
        開発者のワークロード制限を取得
        
        Args:
            developer_id: 開発者ID
            
        Returns:
            float: ワークロード制限値
        """
        if developer_id not in self.developer_states:
            return 0.6  # デフォルト値
        
        state = self.developer_states[developer_id]
        strategy_name = self._determine_strategy_name(state)
        strategy = self.strategies[strategy_name]
        
        return strategy.workload_limit
    
    def should_recommend_review(self, developer_id: str, 
                              review_complexity: float) -> bool:
        """
        レビューを推薦すべきかを判定
        
        Args:
            developer_id: 開発者ID
            review_complexity: レビューの複雑度
            
        Returns:
            bool: 推薦すべきかどうか
        """
        if developer_id not in self.developer_states:
            return True  # デフォルトは推薦
        
        state = self.developer_states[developer_id]
        strategy_name = self._determine_strategy_name(state)
        strategy = self.strategies[strategy_name]
        
        # 複雑度閾値チェック
        if review_complexity > strategy.review_complexity_threshold:
            return False
        
        # ワークロード制限チェック
        if state.current_workload > strategy.workload_limit:
            return False
        
        return True
    
    def get_strategy_effectiveness(self, developer_id: str, 
                                 time_window_days: int = 30) -> Dict[str, float]:
        """
        戦略の効果を評価
        
        Args:
            developer_id: 開発者ID
            time_window_days: 評価期間（日数）
            
        Returns:
            Dict[str, float]: 効果指標
        """
        if developer_id not in self.strategy_history:
            return {}
        
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_strategies = [
            (timestamp, strategy) for timestamp, strategy in self.strategy_history[developer_id]
            if timestamp >= cutoff_date
        ]
        
        if not recent_strategies:
            return {}
        
        # 戦略別の使用頻度
        strategy_counts = {}
        for _, strategy in recent_strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        total_count = len(recent_strategies)
        strategy_ratios = {
            strategy: count / total_count 
            for strategy, count in strategy_counts.items()
        }
        
        return {
            'strategy_distribution': strategy_ratios,
            'strategy_switches': len(set(s for _, s in recent_strategies)),
            'most_used_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0]
        }