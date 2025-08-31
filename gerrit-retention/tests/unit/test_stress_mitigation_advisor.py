"""
ストレス軽減策提案システムのテスト

StressMitigationAdvisorクラスの機能をテストし、
軽減策提案とマッチング機能が正しく動作することを確認する。
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.gerrit_retention.prediction.stress_analyzer import StressIndicators
from src.gerrit_retention.prediction.stress_mitigation_advisor import (
    DeveloperMatchingProposal,
    ImplementationDifficulty,
    MitigationCategory,
    MitigationProposal,
    StressMitigationAdvisor,
)


class TestStressMitigationAdvisor(unittest.TestCase):
    """ストレス軽減策提案システムのテストクラス"""
    
    def setUp(self):
        """テスト用の設定を初期化"""
        self.config = {
            'mitigation_config': {
                'effect_weights': {
                    'immediate_impact': 0.4,
                    'long_term_benefit': 0.3,
                    'implementation_feasibility': 0.3
                }
            },
            'stress_config': {
                'weights': {
                    'task_compatibility_stress': 0.3,
                    'workload_stress': 0.4,
                    'social_stress': 0.2,
                    'temporal_stress': 0.1
                }
            }
        }
        
        self.advisor = StressMitigationAdvisor(self.config)
        
        # テスト用の開発者データ
        self.test_developer = {
            'email': 'test@example.com',
            'name': 'Test Developer',
            'expertise_areas': ['python', 'web'],
            'collaboration_quality': 0.4,
            'recent_rejection_rate': 0.4,
            'avg_response_time_hours': 8.0
        }
        
        # テスト用のコンテキスト
        self.test_context = {
            'review_queue': [
                {
                    'technical_domain': 'java',
                    'complexity_score': 0.9,
                    'deadline': (datetime.now() + timedelta(hours=6)).isoformat()
                },
                {
                    'technical_domain': 'rust',
                    'complexity_score': 0.8,
                    'deadline': (datetime.now() + timedelta(hours=12)).isoformat()
                },
                {
                    'technical_domain': 'go',
                    'complexity_score': 0.7,
                    'deadline': (datetime.now() + timedelta(days=2)).isoformat()
                }
            ],
            'continuous_work_hours': 8
        }
        
        # テスト用のストレス指標
        self.high_stress_indicators = StressIndicators(
            task_compatibility_stress=0.8,
            workload_stress=0.7,
            social_stress=0.6,
            temporal_stress=0.5,
            total_stress=0.7,
            stress_level='high',
            calculated_at=datetime.now()
        )
        
        # テスト用のチームメンバー
        self.team_members = [
            {
                'email': 'member1@example.com',
                'name': 'Member 1',
                'expertise_areas': ['python', 'api', 'database'],
                'collaboration_quality': 0.8
            },
            {
                'email': 'member2@example.com',
                'name': 'Member 2',
                'expertise_areas': ['web', 'frontend', 'ui'],
                'collaboration_quality': 0.7
            },
            {
                'email': 'member3@example.com',
                'name': 'Member 3',
                'expertise_areas': ['java', 'backend'],
                'collaboration_quality': 0.6
            }
        ]
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsInstance(self.advisor, StressMitigationAdvisor)
        self.assertEqual(self.advisor.config, self.config)
        self.assertIn('immediate_impact', self.advisor.effect_weights)
    
    def test_generate_mitigation_proposals(self):
        """軽減策提案生成のテスト"""
        proposals = self.advisor.generate_mitigation_proposals(
            self.test_developer, self.test_context, self.high_stress_indicators
        )
        
        # 提案が生成されることを確認
        self.assertGreater(len(proposals), 0)
        
        # 各提案の基本構造を確認
        for proposal in proposals:
            self.assertIsInstance(proposal, MitigationProposal)
            self.assertIsInstance(proposal.category, MitigationCategory)
            self.assertIsInstance(proposal.implementation_difficulty, ImplementationDifficulty)
            self.assertGreaterEqual(proposal.expected_stress_reduction, 0.0)
            self.assertLessEqual(proposal.expected_stress_reduction, 1.0)
            self.assertGreaterEqual(proposal.priority_score, 0.0)
            self.assertLessEqual(proposal.priority_score, 1.0)
        
        # 優先度順にソートされていることを確認
        for i in range(len(proposals) - 1):
            self.assertGreaterEqual(proposals[i].priority_score, proposals[i + 1].priority_score)
    
    def test_task_compatibility_proposals(self):
        """タスク適合度軽減策のテスト"""
        proposals = self.advisor._generate_task_compatibility_proposals(
            self.test_developer, self.test_context, self.high_stress_indicators
        )
        
        # タスク再配分提案が含まれることを確認
        task_realloc_proposals = [p for p in proposals if p.category == MitigationCategory.TASK_REALLOCATION]
        self.assertGreater(len(task_realloc_proposals), 0)
        
        # スキル開発提案が含まれることを確認（専門領域が少ない場合）
        skill_dev_proposals = [p for p in proposals if p.category == MitigationCategory.SKILL_DEVELOPMENT]
        self.assertGreater(len(skill_dev_proposals), 0)
    
    def test_workload_proposals(self):
        """ワークロード軽減策のテスト"""
        # レビューキューが多い状況を作成
        high_workload_context = {
            **self.test_context,
            'review_queue': [{'id': i} for i in range(8)]  # 8件のレビュー
        }
        
        proposals = self.advisor._generate_workload_proposals(
            self.test_developer, high_workload_context, self.high_stress_indicators
        )
        
        # ワークロード調整提案が含まれることを確認
        workload_proposals = [p for p in proposals if p.category == MitigationCategory.WORKLOAD_ADJUSTMENT]
        self.assertGreater(len(workload_proposals), 0)
        
        # 提案内容の妥当性を確認
        for proposal in workload_proposals:
            self.assertIn('workload', proposal.target_stress_factor)
            self.assertGreater(proposal.expected_stress_reduction, 0.0)
    
    def test_social_proposals(self):
        """社会的ストレス軽減策のテスト"""
        # 協力関係が悪い開発者を設定
        poor_social_developer = {
            **self.test_developer,
            'collaboration_quality': 0.3,
            'recent_rejection_rate': 0.5
        }
        
        proposals = self.advisor._generate_social_proposals(
            poor_social_developer, self.test_context, self.high_stress_indicators
        )
        
        # 協力関係改善提案が含まれることを確認
        collab_proposals = [p for p in proposals if p.category == MitigationCategory.COLLABORATION_IMPROVEMENT]
        self.assertGreater(len(collab_proposals), 0)
        
        # 提案内容の妥当性を確認
        for proposal in collab_proposals:
            self.assertIn('social', proposal.target_stress_factor)
            self.assertGreater(proposal.expected_stress_reduction, 0.0)
    
    def test_temporal_proposals(self):
        """時間的ストレス軽減策のテスト"""
        # 長時間作業の状況を作成
        high_temporal_context = {
            **self.test_context,
            'continuous_work_hours': 10
        }
        
        # 応答時間プレッシャーが高い開発者を設定
        high_pressure_developer = {
            **self.test_developer,
            'avg_response_time_hours': 6.0
        }
        
        proposals = self.advisor._generate_temporal_proposals(
            high_pressure_developer, high_temporal_context, self.high_stress_indicators
        )
        
        # プロセス最適化提案が含まれることを確認
        process_proposals = [p for p in proposals if p.category == MitigationCategory.PROCESS_OPTIMIZATION]
        self.assertGreater(len(process_proposals), 0)
        
        # 提案内容の妥当性を確認
        for proposal in process_proposals:
            self.assertIn('temporal', proposal.target_stress_factor)
            self.assertGreater(proposal.expected_stress_reduction, 0.0)
    
    def test_priority_score_calculation(self):
        """優先度スコア計算のテスト"""
        # テスト用の提案を作成
        test_proposal = MitigationProposal(
            proposal_id="test_proposal",
            category=MitigationCategory.WORKLOAD_ADJUSTMENT,
            title="Test Proposal",
            description="Test Description",
            target_stress_factor="workload",
            expected_stress_reduction=0.6,
            implementation_difficulty=ImplementationDifficulty.EASY,
            estimated_effort_hours=2,
            priority_score=0.0,
            required_resources=["test"],
            success_probability=0.8,
            side_effects=[],
            timeline_days=1,
            created_at=datetime.now()
        )
        
        priority_score = self.advisor._calculate_priority_score(
            test_proposal, self.high_stress_indicators
        )
        
        # スコアが適切な範囲内であることを確認
        self.assertGreaterEqual(priority_score, 0.0)
        self.assertLessEqual(priority_score, 1.0)
        
        # 簡単で効果的な提案は高いスコアを持つはず
        self.assertGreater(priority_score, 0.5)
    
    def test_developer_matching_proposals(self):
        """開発者マッチング提案のテスト"""
        proposals = self.advisor.generate_developer_matching_proposals(
            self.test_developer, self.team_members
        )
        
        # マッチング提案が生成されることを確認
        self.assertGreater(len(proposals), 0)
        self.assertLessEqual(len(proposals), 3)  # 最大3件
        
        # 各提案の構造を確認
        for proposal in proposals:
            self.assertIsInstance(proposal, DeveloperMatchingProposal)
            self.assertEqual(proposal.target_developer, 'test@example.com')
            self.assertGreater(len(proposal.recommended_collaborators), 0)
            self.assertIn(proposal.collaboration_type, ['mentoring', 'pair_programming', 'knowledge_sharing'])
            self.assertGreaterEqual(proposal.matching_score, 0.4)  # 閾値以上
        
        # マッチングスコア順にソートされていることを確認
        for i in range(len(proposals) - 1):
            self.assertGreaterEqual(proposals[i].matching_score, proposals[i + 1].matching_score)
    
    def test_proposal_effectiveness_evaluation(self):
        """提案効果評価のテスト"""
        # テスト用の提案を作成
        test_proposal = MitigationProposal(
            proposal_id="test_proposal",
            category=MitigationCategory.WORKLOAD_ADJUSTMENT,
            title="Test Proposal",
            description="Test Description",
            target_stress_factor="workload",
            expected_stress_reduction=0.4,
            implementation_difficulty=ImplementationDifficulty.MEDIUM,
            estimated_effort_hours=16,
            priority_score=0.0,
            required_resources=["test"],
            success_probability=0.7,
            side_effects=[],
            timeline_days=5,
            created_at=datetime.now()
        )
        
        evaluation = self.advisor.evaluate_proposal_effectiveness(
            test_proposal, self.test_developer, self.high_stress_indicators
        )
        
        # 評価結果の構造を確認
        expected_keys = [
            'current_stress_level', 'expected_stress_after', 'stress_reduction_amount',
            'roi', 'risk_score', 'cost_benefit_ratio', 'implementation_urgency'
        ]
        
        for key in expected_keys:
            self.assertIn(key, evaluation)
            self.assertIsInstance(evaluation[key], (int, float))
        
        # 論理的な値の確認
        self.assertEqual(evaluation['stress_reduction_amount'], 0.4)
        self.assertLess(evaluation['expected_stress_after'], evaluation['current_stress_level'])
        self.assertGreaterEqual(evaluation['risk_score'], 0.0)
        self.assertLessEqual(evaluation['risk_score'], 1.0)
    
    def test_low_stress_no_proposals(self):
        """低ストレス時の提案生成テスト"""
        # 低ストレス指標を作成
        low_stress_indicators = StressIndicators(
            task_compatibility_stress=0.2,
            workload_stress=0.3,
            social_stress=0.1,
            temporal_stress=0.2,
            total_stress=0.2,
            stress_level='low',
            calculated_at=datetime.now()
        )
        
        proposals = self.advisor.generate_mitigation_proposals(
            self.test_developer, self.test_context, low_stress_indicators
        )
        
        # 低ストレス時は提案が少ないか、ない場合がある
        self.assertLessEqual(len(proposals), 2)
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 不正なデータでもクラッシュしないことを確認
        invalid_developer = {}
        invalid_context = {}
        invalid_stress = StressIndicators(
            task_compatibility_stress=0.5,
            workload_stress=0.5,
            social_stress=0.5,
            temporal_stress=0.5,
            total_stress=0.5,
            stress_level='medium',
            calculated_at=datetime.now()
        )
        
        proposals = self.advisor.generate_mitigation_proposals(
            invalid_developer, invalid_context, invalid_stress
        )
        
        # エラーが発生してもリストが返されることを確認
        self.assertIsInstance(proposals, list)
        
        # マッチング提案でも同様
        matching_proposals = self.advisor.generate_developer_matching_proposals(
            invalid_developer, []
        )
        
        self.assertIsInstance(matching_proposals, list)


if __name__ == '__main__':
    unittest.main()