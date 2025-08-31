"""
ストレス軽減策提案システム

開発者のストレス状態を分析し、具体的な軽減策を提案するモジュール。
ストレス要因別に対策を提案し、実装難易度と効果予測を算出して
優先度付きの改善提案を提供する。

主要機能:
- ストレス要因別対策提案
- 実装難易度の評価
- 効果予測の算出
- 提案優先度付け
- 開発者マッチング提案
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from gerrit_retention.prediction.boiling_point_predictor import (
        BoilingPointPredictor,
    )
    from gerrit_retention.prediction.stress_analyzer import (
        StressAnalyzer,
        StressIndicators,
    )
    from gerrit_retention.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class MitigationCategory(Enum):
    """軽減策カテゴリ"""
    TASK_REALLOCATION = "task_reallocation"  # タスク再配分
    WORKLOAD_ADJUSTMENT = "workload_adjustment"  # ワークロード調整
    COLLABORATION_IMPROVEMENT = "collaboration_improvement"  # 協力関係改善
    SKILL_DEVELOPMENT = "skill_development"  # スキル開発
    PROCESS_OPTIMIZATION = "process_optimization"  # プロセス最適化


class ImplementationDifficulty(Enum):
    """実装難易度"""
    EASY = "easy"  # 簡単（即座に実行可能）
    MEDIUM = "medium"  # 中程度（数日〜1週間）
    HARD = "hard"  # 困難（数週間〜1ヶ月）
    VERY_HARD = "very_hard"  # 非常に困難（1ヶ月以上）


@dataclass
class MitigationProposal:
    """軽減策提案データクラス"""
    proposal_id: str
    category: MitigationCategory
    title: str
    description: str
    target_stress_factor: str  # 'task_compatibility', 'workload', 'social', 'temporal'
    expected_stress_reduction: float  # 期待されるストレス軽減効果 (0.0-1.0)
    implementation_difficulty: ImplementationDifficulty
    estimated_effort_hours: float
    priority_score: float  # 優先度スコア (0.0-1.0)
    required_resources: List[str]
    success_probability: float  # 成功確率 (0.0-1.0)
    side_effects: List[str]  # 副作用・リスク
    timeline_days: int  # 実装期間（日数）
    created_at: datetime


@dataclass
class DeveloperMatchingProposal:
    """開発者マッチング提案データクラス"""
    target_developer: str
    recommended_collaborators: List[Dict[str, Any]]
    collaboration_type: str  # 'mentoring', 'pair_programming', 'knowledge_sharing'
    expected_benefit: str
    matching_score: float


class StressMitigationAdvisor:
    """
    ストレス軽減策提案システム
    
    開発者のストレス状態を分析し、具体的で実行可能な
    軽減策を優先度付きで提案する。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        軽減策提案システムを初期化
        
        Args:
            config: システム設定辞書
        """
        self.config = config
        self.mitigation_config = config.get('mitigation_config', {})
        
        # 効果予測の重み
        self.effect_weights = self.mitigation_config.get('effect_weights', {
            'immediate_impact': 0.4,
            'long_term_benefit': 0.3,
            'implementation_feasibility': 0.3
        })
        
        # 難易度別の実装時間（時間）
        self.difficulty_hours = {
            ImplementationDifficulty.EASY: 2,
            ImplementationDifficulty.MEDIUM: 16,
            ImplementationDifficulty.HARD: 80,
            ImplementationDifficulty.VERY_HARD: 200
        }
        
        # ストレス分析器と沸点予測器
        self.stress_analyzer = StressAnalyzer(config.get('stress_config', {}))
        self.boiling_point_predictor = BoilingPointPredictor(config)
        
        logger.info("ストレス軽減策提案システムが初期化されました")
    
    def generate_mitigation_proposals(self, developer: Dict[str, Any], 
                                    context: Dict[str, Any],
                                    stress_indicators: StressIndicators) -> List[MitigationProposal]:
        """
        ストレス軽減策を生成
        
        Args:
            developer: 開発者情報
            context: コンテキスト情報
            stress_indicators: ストレス指標
            
        Returns:
            List[MitigationProposal]: 優先度順の軽減策提案リスト
        """
        try:
            proposals = []
            
            # 各ストレス要因に対する提案を生成
            if stress_indicators.task_compatibility_stress > 0.5:
                proposals.extend(self._generate_task_compatibility_proposals(
                    developer, context, stress_indicators
                ))
            
            if stress_indicators.workload_stress > 0.5:
                proposals.extend(self._generate_workload_proposals(
                    developer, context, stress_indicators
                ))
            
            if stress_indicators.social_stress > 0.5:
                proposals.extend(self._generate_social_proposals(
                    developer, context, stress_indicators
                ))
            
            if stress_indicators.temporal_stress > 0.5:
                proposals.extend(self._generate_temporal_proposals(
                    developer, context, stress_indicators
                ))
            
            # 優先度スコアを計算
            for proposal in proposals:
                proposal.priority_score = self._calculate_priority_score(
                    proposal, stress_indicators
                )
            
            # 優先度順にソート
            proposals.sort(key=lambda p: p.priority_score, reverse=True)
            
            logger.info(f"開発者 {developer.get('email', 'unknown')} に対して {len(proposals)} 件の軽減策を生成")
            return proposals
            
        except Exception as e:
            logger.error(f"軽減策生成中にエラーが発生: {e}")
            return []
    
    def _generate_task_compatibility_proposals(self, developer: Dict[str, Any], 
                                             context: Dict[str, Any],
                                             stress_indicators: StressIndicators) -> List[MitigationProposal]:
        """タスク適合度ストレス軽減策を生成"""
        proposals = []
        
        # 専門領域に合うタスクへの再配分
        proposals.append(MitigationProposal(
            proposal_id=f"task_realloc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            category=MitigationCategory.TASK_REALLOCATION,
            title="専門領域に適合するタスクへの再配分",
            description="開発者の専門領域に合致するレビュータスクを優先的に割り当て、専門外タスクを他の適切な開発者に再配分する",
            target_stress_factor="task_compatibility",
            expected_stress_reduction=0.6,
            implementation_difficulty=ImplementationDifficulty.MEDIUM,
            estimated_effort_hours=self.difficulty_hours[ImplementationDifficulty.MEDIUM],
            priority_score=0.0,  # 後で計算
            required_resources=["プロジェクトマネージャー", "タスク管理システム"],
            success_probability=0.8,
            side_effects=["他の開発者の負荷増加の可能性"],
            timeline_days=3,
            created_at=datetime.now()
        ))
        
        # スキル開発支援
        expertise_areas = developer.get('expertise_areas', [])
        if len(expertise_areas) < 3:  # 専門領域が少ない場合
            proposals.append(MitigationProposal(
                proposal_id=f"skill_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category=MitigationCategory.SKILL_DEVELOPMENT,
                title="専門領域拡張のための学習支援",
                description="現在のタスクに関連する技術領域の学習時間を確保し、メンタリング支援を提供する",
                target_stress_factor="task_compatibility",
                expected_stress_reduction=0.4,
                implementation_difficulty=ImplementationDifficulty.HARD,
                estimated_effort_hours=self.difficulty_hours[ImplementationDifficulty.HARD],
                priority_score=0.0,
                required_resources=["メンター", "学習リソース", "学習時間"],
                success_probability=0.7,
                side_effects=["短期的な生産性低下"],
                timeline_days=21,
                created_at=datetime.now()
            ))
        
        return proposals
    
    def _generate_workload_proposals(self, developer: Dict[str, Any], 
                                   context: Dict[str, Any],
                                   stress_indicators: StressIndicators) -> List[MitigationProposal]:
        """ワークロードストレス軽減策を生成"""
        proposals = []
        
        # レビューキューサイズの調整
        review_queue_size = len(context.get('review_queue', []))
        if review_queue_size > 5:
            proposals.append(MitigationProposal(
                proposal_id=f"workload_reduce_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category=MitigationCategory.WORKLOAD_ADJUSTMENT,
                title="レビューキューサイズの制限",
                description=f"同時レビュー数を{review_queue_size}件から3件に制限し、新規レビューの割り当てを一時停止する",
                target_stress_factor="workload",
                expected_stress_reduction=0.5,
                implementation_difficulty=ImplementationDifficulty.EASY,
                estimated_effort_hours=self.difficulty_hours[ImplementationDifficulty.EASY],
                priority_score=0.0,
                required_resources=["レビュー管理システム"],
                success_probability=0.9,
                side_effects=["プロジェクト進行の遅延可能性"],
                timeline_days=1,
                created_at=datetime.now()
            ))
        
        # 締切プレッシャーの軽減
        urgent_reviews = sum(1 for review in context.get('review_queue', []) 
                           if review.get('deadline') and 
                           (datetime.fromisoformat(review['deadline']) - datetime.now()).days < 1)
        
        if urgent_reviews > 2:
            proposals.append(MitigationProposal(
                proposal_id=f"deadline_adjust_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category=MitigationCategory.PROCESS_OPTIMIZATION,
                title="緊急レビューの締切調整",
                description="緊急度の高いレビューの締切を再評価し、可能な限り期限を延長または他の開発者に再配分する",
                target_stress_factor="workload",
                expected_stress_reduction=0.4,
                implementation_difficulty=ImplementationDifficulty.MEDIUM,
                estimated_effort_hours=self.difficulty_hours[ImplementationDifficulty.MEDIUM],
                priority_score=0.0,
                required_resources=["プロジェクトマネージャー", "ステークホルダー"],
                success_probability=0.6,
                side_effects=["プロジェクト計画の調整が必要"],
                timeline_days=2,
                created_at=datetime.now()
            ))
        
        return proposals
    
    def _generate_social_proposals(self, developer: Dict[str, Any], 
                                 context: Dict[str, Any],
                                 stress_indicators: StressIndicators) -> List[MitigationProposal]:
        """社会的ストレス軽減策を生成"""
        proposals = []
        
        # 協力関係の改善
        collaboration_quality = developer.get('collaboration_quality', 0.5)
        if collaboration_quality < 0.5:
            proposals.append(MitigationProposal(
                proposal_id=f"collab_improve_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category=MitigationCategory.COLLABORATION_IMPROVEMENT,
                title="チーム内コミュニケーション改善",
                description="定期的な1on1ミーティングの設定と、チーム内での知識共有セッションを実施する",
                target_stress_factor="social",
                expected_stress_reduction=0.5,
                implementation_difficulty=ImplementationDifficulty.MEDIUM,
                estimated_effort_hours=self.difficulty_hours[ImplementationDifficulty.MEDIUM],
                priority_score=0.0,
                required_resources=["チームリーダー", "ミーティング時間"],
                success_probability=0.7,
                side_effects=["ミーティング時間の増加"],
                timeline_days=7,
                created_at=datetime.now()
            ))
        
        # レビュー拒否率が高い場合の対策
        rejection_rate = developer.get('recent_rejection_rate', 0.0)
        if rejection_rate > 0.3:
            proposals.append(MitigationProposal(
                proposal_id=f"rejection_support_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category=MitigationCategory.COLLABORATION_IMPROVEMENT,
                title="レビュー負荷軽減とサポート強化",
                description="レビュー依頼の事前フィルタリングと、レビュー品質向上のためのガイドライン提供",
                target_stress_factor="social",
                expected_stress_reduction=0.3,
                implementation_difficulty=ImplementationDifficulty.EASY,
                estimated_effort_hours=self.difficulty_hours[ImplementationDifficulty.EASY],
                priority_score=0.0,
                required_resources=["レビューガイドライン", "フィルタリングツール"],
                success_probability=0.8,
                side_effects=["初期設定の手間"],
                timeline_days=2,
                created_at=datetime.now()
            ))
        
        return proposals
    
    def _generate_temporal_proposals(self, developer: Dict[str, Any], 
                                   context: Dict[str, Any],
                                   stress_indicators: StressIndicators) -> List[MitigationProposal]:
        """時間的ストレス軽減策を生成"""
        proposals = []
        
        # 連続作業時間の制限
        continuous_hours = context.get('continuous_work_hours', 0)
        if continuous_hours > 6:
            proposals.append(MitigationProposal(
                proposal_id=f"work_break_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category=MitigationCategory.PROCESS_OPTIMIZATION,
                title="作業時間制限と休憩時間の確保",
                description="連続作業時間を6時間に制限し、定期的な休憩時間を強制的に設ける",
                target_stress_factor="temporal",
                expected_stress_reduction=0.4,
                implementation_difficulty=ImplementationDifficulty.EASY,
                estimated_effort_hours=self.difficulty_hours[ImplementationDifficulty.EASY],
                priority_score=0.0,
                required_resources=["時間管理ツール"],
                success_probability=0.9,
                side_effects=["短期的な作業効率の低下"],
                timeline_days=1,
                created_at=datetime.now()
            ))
        
        # 応答時間プレッシャーの軽減
        avg_response_time = developer.get('avg_response_time_hours', 24.0)
        if avg_response_time < 12.0:  # 12時間以内の応答が期待されている場合
            proposals.append(MitigationProposal(
                proposal_id=f"response_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category=MitigationCategory.PROCESS_OPTIMIZATION,
                title="レビュー応答時間期待値の調整",
                description="レビュー応答時間の期待値を24時間に延長し、緊急時のエスカレーション手順を明確化する",
                target_stress_factor="temporal",
                expected_stress_reduction=0.3,
                implementation_difficulty=ImplementationDifficulty.MEDIUM,
                estimated_effort_hours=self.difficulty_hours[ImplementationDifficulty.MEDIUM],
                priority_score=0.0,
                required_resources=["チーム合意", "プロセス文書更新"],
                success_probability=0.7,
                side_effects=["レビューサイクルの延長"],
                timeline_days=5,
                created_at=datetime.now()
            ))
        
        return proposals
    
    def _calculate_priority_score(self, proposal: MitigationProposal, 
                                stress_indicators: StressIndicators) -> float:
        """
        提案の優先度スコアを計算
        
        Args:
            proposal: 軽減策提案
            stress_indicators: ストレス指標
            
        Returns:
            float: 優先度スコア (0.0-1.0)
        """
        # 即効性スコア（実装の容易さ）
        difficulty_score = {
            ImplementationDifficulty.EASY: 1.0,
            ImplementationDifficulty.MEDIUM: 0.7,
            ImplementationDifficulty.HARD: 0.4,
            ImplementationDifficulty.VERY_HARD: 0.2
        }[proposal.implementation_difficulty]
        
        # 効果スコア
        effect_score = proposal.expected_stress_reduction * proposal.success_probability
        
        # 緊急度スコア（対象ストレス要因の深刻度）
        target_stress_value = {
            'task_compatibility': stress_indicators.task_compatibility_stress,
            'workload': stress_indicators.workload_stress,
            'social': stress_indicators.social_stress,
            'temporal': stress_indicators.temporal_stress
        }.get(proposal.target_stress_factor, 0.5)
        
        urgency_score = min(target_stress_value * 1.5, 1.0)
        
        # 重み付き合計
        priority_score = (
            difficulty_score * self.effect_weights['implementation_feasibility'] +
            effect_score * self.effect_weights['immediate_impact'] +
            urgency_score * self.effect_weights['long_term_benefit']
        )
        
        return min(priority_score, 1.0)
    
    def generate_developer_matching_proposals(self, target_developer: Dict[str, Any],
                                            team_members: List[Dict[str, Any]]) -> List[DeveloperMatchingProposal]:
        """
        開発者マッチング提案を生成
        
        Args:
            target_developer: 対象開発者
            team_members: チームメンバーリスト
            
        Returns:
            List[DeveloperMatchingProposal]: マッチング提案リスト
        """
        try:
            proposals = []
            target_expertise = set(target_developer.get('expertise_areas', []))
            target_collaboration = target_developer.get('collaboration_quality', 0.5)
            
            for member in team_members:
                if member.get('email') == target_developer.get('email'):
                    continue  # 自分自身は除外
                
                member_expertise = set(member.get('expertise_areas', []))
                member_collaboration = member.get('collaboration_quality', 0.5)
                
                # 専門領域の重複度
                expertise_overlap = len(target_expertise & member_expertise) / max(len(target_expertise), 1)
                
                # 協力関係の質
                collaboration_compatibility = (target_collaboration + member_collaboration) / 2
                
                # マッチングスコア
                matching_score = (expertise_overlap * 0.6 + collaboration_compatibility * 0.4)
                
                if matching_score > 0.4:  # 閾値以上の場合のみ提案
                    # 協力タイプを決定
                    if target_collaboration < 0.4:
                        collaboration_type = "mentoring"
                        expected_benefit = "協力関係の質向上とコミュニケーションスキル向上"
                    elif expertise_overlap > 0.5:
                        collaboration_type = "pair_programming"
                        expected_benefit = "技術的知識の共有と問題解決能力向上"
                    else:
                        collaboration_type = "knowledge_sharing"
                        expected_benefit = "新しい技術領域の学習と視野拡大"
                    
                    proposals.append(DeveloperMatchingProposal(
                        target_developer=target_developer.get('email', 'unknown'),
                        recommended_collaborators=[{
                            'email': member.get('email'),
                            'name': member.get('name', ''),
                            'expertise_areas': list(member_expertise),
                            'collaboration_quality': member_collaboration
                        }],
                        collaboration_type=collaboration_type,
                        expected_benefit=expected_benefit,
                        matching_score=matching_score
                    ))
            
            # マッチングスコア順にソート
            proposals.sort(key=lambda p: p.matching_score, reverse=True)
            
            logger.info(f"開発者 {target_developer.get('email', 'unknown')} に対して {len(proposals)} 件のマッチング提案を生成")
            return proposals[:3]  # 上位3件のみ返す
            
        except Exception as e:
            logger.error(f"開発者マッチング提案生成中にエラーが発生: {e}")
            return []
    
    def evaluate_proposal_effectiveness(self, proposal: MitigationProposal,
                                      developer: Dict[str, Any],
                                      stress_indicators: StressIndicators) -> Dict[str, float]:
        """
        提案の効果を評価
        
        Args:
            proposal: 軽減策提案
            developer: 開発者情報
            stress_indicators: 現在のストレス指標
            
        Returns:
            Dict[str, float]: 効果評価結果
        """
        # 予想されるストレス軽減効果
        current_stress = getattr(stress_indicators, f"{proposal.target_stress_factor}_stress", 0.5)
        expected_new_stress = max(0.0, current_stress - proposal.expected_stress_reduction)
        
        # ROI計算（効果 / コスト）
        roi = proposal.expected_stress_reduction / max(proposal.estimated_effort_hours / 40, 0.1)  # 週単位
        
        # リスク評価
        risk_score = 1.0 - proposal.success_probability
        
        return {
            'current_stress_level': current_stress,
            'expected_stress_after': expected_new_stress,
            'stress_reduction_amount': proposal.expected_stress_reduction,
            'roi': roi,
            'risk_score': risk_score,
            'cost_benefit_ratio': proposal.expected_stress_reduction / max(proposal.estimated_effort_hours, 1),
            'implementation_urgency': min(current_stress * 2, 1.0)
        }