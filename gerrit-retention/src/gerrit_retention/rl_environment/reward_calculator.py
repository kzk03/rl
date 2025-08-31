"""
報酬計算システム

このモジュールは、レビュー受諾行動に対する報酬を計算する。
長期的な開発者定着を重視した報酬設計を実装し、Gerrit特有の報酬要素を含む。
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RewardComponents:
    """報酬構成要素"""
    base_reward: float = 0.0
    continuity_reward: float = 0.0
    stress_reward: float = 0.0
    quality_reward: float = 0.0
    collaboration_reward: float = 0.0
    urgency_reward: float = 0.0
    expertise_reward: float = 0.0
    workload_reward: float = 0.0
    gerrit_specific_reward: float = 0.0
    total_reward: float = 0.0


@dataclass
class ActionContext:
    """行動コンテキスト"""
    action: str  # 'accept', 'reject', 'wait'
    review_request: Dict[str, Any]
    developer_state: Dict[str, Any]
    environment_state: Dict[str, Any]
    history: List[Dict[str, Any]]
    timestamp: datetime


class RewardCalculator:
    """
    報酬計算器
    
    レビュー受諾行動に対する包括的な報酬計算を実行。
    長期定着重視の報酬設計とGerrit特有の要素を組み合わせる。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        報酬計算器を初期化
        
        Args:
            config: 報酬計算設定
        """
        self.config = config
        
        # 基本報酬重み
        self.weights = config.get('reward_weights', {
            'acceptance_reward': 1.0,
            'decline_penalty': -0.5,
            'defer_penalty': -0.1,
            'continuity_bonus': 0.3,
            'stress_factor': 0.2,
            'quality_bonus': 0.1,
            'collaboration_bonus': 0.15,
            'urgency_factor': 0.1,
            'expertise_factor': 0.2,
            'workload_factor': 0.3
        })
        
        # Gerrit特有の報酬重み
        self.gerrit_weights = config.get('gerrit_rewards', {
            'high_quality_review_bonus': 0.2,  # +2スコアのレビュー
            'thorough_review_bonus': 0.1,     # 詳細なレビュー
            'quick_response_bonus': 0.05,     # 迅速な応答
            'complex_change_bonus': 0.15,     # 複雑な変更のレビュー
            'cross_team_collaboration': 0.1,  # チーム間協力
            'mentoring_bonus': 0.2,           # 新人指導
            'critical_fix_bonus': 0.25        # 重要な修正
        })
        
        # 閾値設定
        self.thresholds = config.get('thresholds', {
            'stress_threshold': 0.8,
            'expertise_match_threshold': 0.7,
            'urgency_threshold': 0.8,
            'quality_threshold': 0.8,
            'workload_threshold': 0.9,
            'response_time_threshold': 4.0  # hours
        })
        
        logger.info("報酬計算器を初期化しました")
    
    def calculate_reward(self, context: ActionContext) -> RewardComponents:
        """
        行動に対する報酬を計算
        
        Args:
            context: 行動コンテキスト
            
        Returns:
            RewardComponents: 計算された報酬構成要素
        """
        components = RewardComponents()
        
        if context.action == 'accept':
            components = self._calculate_acceptance_reward(context)
        elif context.action == 'reject':
            components = self._calculate_rejection_reward(context)
        elif context.action == 'wait':
            components = self._calculate_waiting_reward(context)
        else:
            logger.warning(f"未知の行動: {context.action}")
            return components
        
        # 総報酬を計算
        components.total_reward = (
            components.base_reward +
            components.continuity_reward +
            components.stress_reward +
            components.quality_reward +
            components.collaboration_reward +
            components.urgency_reward +
            components.expertise_reward +
            components.workload_reward +
            components.gerrit_specific_reward
        )
        
        logger.debug(f"報酬計算完了: {context.action} -> {components.total_reward:.3f}")
        
        return components
    
    def _calculate_acceptance_reward(self, context: ActionContext) -> RewardComponents:
        """
        受諾時の報酬を計算
        
        Args:
            context: 行動コンテキスト
            
        Returns:
            RewardComponents: 受諾報酬構成要素
        """
        components = RewardComponents()
        
        # 基本受諾報酬
        components.base_reward = self.weights['acceptance_reward']
        
        # 継続報酬の計算
        components.continuity_reward = self._calculate_continuity_reward(context)
        
        # ストレス報酬の計算
        components.stress_reward = self._calculate_stress_reward(context, 'accept')
        
        # 品質報酬の計算
        components.quality_reward = self._calculate_quality_reward(context)
        
        # 協力報酬の計算
        components.collaboration_reward = self._calculate_collaboration_reward(context)
        
        # 緊急度報酬の計算
        components.urgency_reward = self._calculate_urgency_reward(context, 'accept')
        
        # 専門性報酬の計算
        components.expertise_reward = self._calculate_expertise_reward(context, 'accept')
        
        # ワークロード報酬の計算
        components.workload_reward = self._calculate_workload_reward(context, 'accept')
        
        # Gerrit特有報酬の計算
        components.gerrit_specific_reward = self._calculate_gerrit_specific_reward(context, 'accept')
        
        return components
    
    def _calculate_rejection_reward(self, context: ActionContext) -> RewardComponents:
        """
        拒否時の報酬を計算
        
        Args:
            context: 行動コンテキスト
            
        Returns:
            RewardComponents: 拒否報酬構成要素
        """
        components = RewardComponents()
        
        # 基本拒否ペナルティ
        components.base_reward = self.weights['decline_penalty']
        
        # ストレス軽減報酬（適切な拒否の場合）
        components.stress_reward = self._calculate_stress_reward(context, 'reject')
        
        # 専門性不適合による正当化報酬
        components.expertise_reward = self._calculate_expertise_reward(context, 'reject')
        
        # ワークロード保護報酬
        components.workload_reward = self._calculate_workload_reward(context, 'reject')
        
        # 緊急度による調整
        components.urgency_reward = self._calculate_urgency_reward(context, 'reject')
        
        return components
    
    def _calculate_waiting_reward(self, context: ActionContext) -> RewardComponents:
        """
        待機時の報酬を計算
        
        Args:
            context: 行動コンテキスト
            
        Returns:
            RewardComponents: 待機報酬構成要素
        """
        components = RewardComponents()
        
        # 基本待機ペナルティ
        components.base_reward = self.weights['defer_penalty']
        
        # 情報収集価値（複雑なレビューの場合）
        review = context.review_request
        if review.get('complexity_score', 0.0) > 0.7:
            components.quality_reward = 0.05  # 情報収集価値
        
        # 緊急度による調整
        components.urgency_reward = self._calculate_urgency_reward(context, 'wait')
        
        # 戦略的待機報酬（高負荷時）
        developer_state = context.developer_state
        if developer_state.get('workload_ratio', 0.0) > self.thresholds['workload_threshold']:
            components.workload_reward = 0.02  # 戦略的待機
        
        return components
    
    def _calculate_continuity_reward(self, context: ActionContext) -> float:
        """
        継続報酬を計算
        
        Args:
            context: 行動コンテキスト
            
        Returns:
            float: 継続報酬
        """
        # 最近の受諾履歴を分析
        recent_history = context.history[-5:] if len(context.history) >= 5 else context.history
        recent_acceptances = sum(1 for h in recent_history if h.get('action') == 'accept')
        
        # 連続受諾ボーナス
        continuity_bonus = self.weights['continuity_bonus'] * recent_acceptances
        
        # 長期関係構築ボーナス
        if recent_acceptances >= 3:
            long_term_bonus = 0.1 * (recent_acceptances - 2)
            continuity_bonus += long_term_bonus
        
        return continuity_bonus
    
    def _calculate_stress_reward(self, context: ActionContext, action: str) -> float:
        """
        ストレス関連報酬を計算
        
        Args:
            context: 行動コンテキスト
            action: 実行された行動
            
        Returns:
            float: ストレス報酬
        """
        developer_state = context.developer_state
        review = context.review_request
        
        current_stress = developer_state.get('stress_level', 0.5)
        stress_threshold = self.thresholds['stress_threshold']
        expertise_match = review.get('expertise_match', 0.5)
        
        if action == 'accept':
            if expertise_match > self.thresholds['expertise_match_threshold']:
                # 専門性に合う受諾はストレス軽減
                return self.weights['stress_factor']
            else:
                # 専門性に合わない受諾はストレス増加
                stress_penalty = -0.4 * self.weights['stress_factor']
                # 高ストレス時はさらにペナルティ
                if current_stress > stress_threshold:
                    stress_penalty *= 1.5
                return stress_penalty
        
        elif action == 'reject':
            if current_stress > stress_threshold:
                # 高ストレス時の適切な拒否は報酬
                return 0.3 * self.weights['stress_factor']
            elif expertise_match < 0.3:
                # 専門性不適合による正当な拒否
                return 0.2 * self.weights['stress_factor']
            else:
                # 通常の拒否は小さなペナルティ
                return -0.1 * self.weights['stress_factor']
        
        return 0.0
    
    def _calculate_quality_reward(self, context: ActionContext) -> float:
        """
        品質報酬を計算
        
        Args:
            context: 行動コンテキスト
            
        Returns:
            float: 品質報酬
        """
        developer_state = context.developer_state
        review = context.review_request
        
        # 開発者の専門性レベル
        expertise_level = developer_state.get('expertise_level', 0.5)
        
        # レビューとの専門性適合度
        expertise_match = review.get('expertise_match', 0.5)
        
        # 予測品質スコア
        base_quality = expertise_match * expertise_level
        
        # 複雑度による調整（適度な複雑度が最適）
        complexity = review.get('complexity_score', 0.5)
        complexity_factor = 1.0 - abs(complexity - 0.6)  # 0.6が最適複雑度
        
        # 開発者の過去のレビュー品質
        past_quality = developer_state.get('code_review_thoroughness', 0.7)
        
        predicted_quality = min(1.0, base_quality * complexity_factor * past_quality)
        
        return self.weights['quality_bonus'] * predicted_quality
    
    def _calculate_collaboration_reward(self, context: ActionContext) -> float:
        """
        協力報酬を計算
        
        Args:
            context: 行動コンテキスト
            
        Returns:
            float: 協力報酬
        """
        review = context.review_request
        developer_state = context.developer_state
        
        # 依頼者との関係性
        requester_relationship = review.get('requester_relationship', 0.5)
        
        # 新しい協力関係の形成
        if requester_relationship < 0.3:
            new_collaboration_bonus = 0.15
        else:
            # 既存関係の強化
            new_collaboration_bonus = 0.1
        
        # チーム間協力ボーナス
        if review.get('cross_team', False):
            new_collaboration_bonus += self.gerrit_weights['cross_team_collaboration']
        
        # 協力品質による調整
        collaboration_quality = developer_state.get('collaboration_quality', 0.7)
        
        return self.weights['collaboration_bonus'] * new_collaboration_bonus * collaboration_quality
    
    def _calculate_urgency_reward(self, context: ActionContext, action: str) -> float:
        """
        緊急度報酬を計算
        
        Args:
            context: 行動コンテキスト
            action: 実行された行動
            
        Returns:
            float: 緊急度報酬
        """
        review = context.review_request
        urgency_level = review.get('urgency_level', 0.5)
        
        if action == 'accept':
            # 緊急度が高い場合の受諾ボーナス
            if urgency_level > self.thresholds['urgency_threshold']:
                return self.weights['urgency_factor'] * urgency_level
            else:
                return 0.0
        
        elif action == 'reject':
            # 緊急度が高い場合の拒否ペナルティ
            if urgency_level > self.thresholds['urgency_threshold']:
                return -self.weights['urgency_factor'] * urgency_level
            else:
                return 0.0
        
        elif action == 'wait':
            # 緊急度が高い場合の待機ペナルティ
            return -self.weights['urgency_factor'] * urgency_level * 0.5
        
        return 0.0
    
    def _calculate_expertise_reward(self, context: ActionContext, action: str) -> float:
        """
        専門性報酬を計算
        
        Args:
            context: 行動コンテキスト
            action: 実行された行動
            
        Returns:
            float: 専門性報酬
        """
        review = context.review_request
        developer_state = context.developer_state
        
        expertise_match = review.get('expertise_match', 0.5)
        expertise_level = developer_state.get('expertise_level', 0.5)
        
        if action == 'accept':
            # 専門性適合度による報酬
            if expertise_match > self.thresholds['expertise_match_threshold']:
                # 高適合度の受諾は学習機会
                learning_bonus = self.weights['expertise_factor'] * expertise_match
                
                # 専門性向上の可能性
                if expertise_match > expertise_level:
                    learning_bonus *= 1.2  # 学習機会ボーナス
                
                return learning_bonus
            else:
                # 低適合度の受諾はペナルティ
                return -self.weights['expertise_factor'] * (1.0 - expertise_match)
        
        elif action == 'reject':
            # 専門性不適合による正当な拒否
            if expertise_match < 0.3:
                return self.weights['expertise_factor'] * 0.5
            else:
                # 適合度が高いのに拒否する場合は小さなペナルティ
                return -self.weights['expertise_factor'] * 0.2
        
        return 0.0
    
    def _calculate_workload_reward(self, context: ActionContext, action: str) -> float:
        """
        ワークロード報酬を計算
        
        Args:
            context: 行動コンテキスト
            action: 実行された行動
            
        Returns:
            float: ワークロード報酬
        """
        developer_state = context.developer_state
        review = context.review_request
        
        current_workload = developer_state.get('workload_ratio', 0.5)
        estimated_effort = review.get('estimated_review_effort', 2.0)
        workload_threshold = self.thresholds['workload_threshold']
        
        if action == 'accept':
            # 過負荷状態での受諾はペナルティ
            if current_workload > workload_threshold:
                overload_penalty = -self.weights['workload_factor'] * (current_workload - workload_threshold)
                # 高負荷作業の場合はさらにペナルティ
                if estimated_effort > 3.0:
                    overload_penalty *= 1.5
                return overload_penalty
            else:
                # 適切な負荷での受諾は小さなボーナス
                return self.weights['workload_factor'] * 0.1
        
        elif action == 'reject':
            # 過負荷状態での適切な拒否は報酬
            if current_workload > workload_threshold:
                return self.weights['workload_factor'] * 0.5
            else:
                return 0.0
        
        elif action == 'wait':
            # 高負荷時の戦略的待機
            if current_workload > workload_threshold:
                return self.weights['workload_factor'] * 0.1
            else:
                return 0.0
        
        return 0.0
    
    def _calculate_gerrit_specific_reward(self, context: ActionContext, action: str) -> float:
        """
        Gerrit特有の報酬を計算
        
        Args:
            context: 行動コンテキスト
            action: 実行された行動
            
        Returns:
            float: Gerrit特有報酬
        """
        if action != 'accept':
            return 0.0
        
        review = context.review_request
        developer_state = context.developer_state
        total_bonus = 0.0
        
        # 高品質レビューボーナス（予測）
        expected_score = self._predict_review_score(context)
        if expected_score >= 2.0:  # +2スコア予測
            total_bonus += self.gerrit_weights['high_quality_review_bonus']
        
        # 詳細レビューボーナス
        thoroughness = developer_state.get('code_review_thoroughness', 0.7)
        if thoroughness > 0.8:
            total_bonus += self.gerrit_weights['thorough_review_bonus']
        
        # 迅速応答ボーナス
        response_time = self._estimate_response_time(context)
        if response_time < self.thresholds['response_time_threshold']:
            total_bonus += self.gerrit_weights['quick_response_bonus']
        
        # 複雑な変更レビューボーナス
        complexity = review.get('complexity_score', 0.5)
        if complexity > 0.8:
            total_bonus += self.gerrit_weights['complex_change_bonus']
        
        # メンタリングボーナス（新人の変更をレビュー）
        if review.get('author_experience_level', 'senior') == 'junior':
            total_bonus += self.gerrit_weights['mentoring_bonus']
        
        # 重要な修正ボーナス
        if review.get('is_critical_fix', False):
            total_bonus += self.gerrit_weights['critical_fix_bonus']
        
        return total_bonus
    
    def _predict_review_score(self, context: ActionContext) -> float:
        """
        レビュースコアを予測
        
        Args:
            context: 行動コンテキスト
            
        Returns:
            float: 予測レビュースコア (-2 to +2)
        """
        developer_state = context.developer_state
        review = context.review_request
        
        # 基本スコア（開発者の過去の平均スコア）
        base_score = developer_state.get('avg_review_score_given', 1.0)
        
        # 専門性適合度による調整
        expertise_match = review.get('expertise_match', 0.5)
        expertise_adjustment = (expertise_match - 0.5) * 2.0  # -1 to +1
        
        # 複雑度による調整（適度な複雑度で最高スコア）
        complexity = review.get('complexity_score', 0.5)
        complexity_adjustment = 1.0 - abs(complexity - 0.6)  # 0.6が最適
        
        predicted_score = base_score + expertise_adjustment * complexity_adjustment
        
        # -2 to +2 の範囲にクリップ
        return np.clip(predicted_score, -2.0, 2.0)
    
    def _estimate_response_time(self, context: ActionContext) -> float:
        """
        応答時間を推定
        
        Args:
            context: 行動コンテキスト
            
        Returns:
            float: 推定応答時間（時間）
        """
        developer_state = context.developer_state
        review = context.review_request
        
        # 開発者の平均応答時間
        base_response_time = developer_state.get('review_response_time_avg', 8.0)
        
        # 複雑度による調整
        complexity = review.get('complexity_score', 0.5)
        complexity_factor = 1.0 + complexity  # 複雑度が高いほど時間がかかる
        
        # 現在のワークロードによる調整
        workload = developer_state.get('workload_ratio', 0.5)
        workload_factor = 1.0 + workload  # 負荷が高いほど時間がかかる
        
        # 緊急度による調整
        urgency = review.get('urgency_level', 0.5)
        urgency_factor = 1.0 - urgency * 0.5  # 緊急度が高いほど早く対応
        
        estimated_time = base_response_time * complexity_factor * workload_factor * urgency_factor
        
        return max(0.5, estimated_time)  # 最低30分
    
    def get_reward_breakdown(self, components: RewardComponents) -> Dict[str, float]:
        """
        報酬の内訳を取得
        
        Args:
            components: 報酬構成要素
            
        Returns:
            Dict[str, float]: 報酬内訳辞書
        """
        return {
            'base_reward': components.base_reward,
            'continuity_reward': components.continuity_reward,
            'stress_reward': components.stress_reward,
            'quality_reward': components.quality_reward,
            'collaboration_reward': components.collaboration_reward,
            'urgency_reward': components.urgency_reward,
            'expertise_reward': components.expertise_reward,
            'workload_reward': components.workload_reward,
            'gerrit_specific_reward': components.gerrit_specific_reward,
            'total_reward': components.total_reward
        }
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        報酬重みを更新
        
        Args:
            new_weights: 新しい重み辞書
        """
        self.weights.update(new_weights)
        logger.info(f"報酬重みを更新しました: {new_weights}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        現在の報酬重みを取得
        
        Returns:
            Dict[str, float]: 現在の重み辞書
        """
        return {**self.weights, **self.gerrit_weights}
    
    def validate_reward_components(self, components: RewardComponents) -> bool:
        """
        報酬構成要素を検証
        
        Args:
            components: 報酬構成要素
            
        Returns:
            bool: 検証結果
        """
        # 報酬の範囲チェック
        if abs(components.total_reward) > 10.0:
            logger.warning(f"報酬が異常に大きいです: {components.total_reward}")
            return False
        
        # NaN チェック
        if np.isnan(components.total_reward):
            logger.error("報酬がNaNです")
            return False
        
        return True