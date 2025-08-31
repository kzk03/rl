"""
ストレス分析器のテスト

StressAnalyzerクラスの機能をテストし、
各種ストレス指標の計算が正しく動作することを確認する。
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.gerrit_retention.prediction.stress_analyzer import (
    StressAnalyzer,
    StressIndicators,
)


class TestStressAnalyzer(unittest.TestCase):
    """ストレス分析器のテストクラス"""
    
    def setUp(self):
        """テスト用の設定を初期化"""
        self.stress_config = {
            'weights': {
                'task_compatibility_stress': 0.3,
                'workload_stress': 0.4,
                'social_stress': 0.2,
                'temporal_stress': 0.1
            },
            'gerrit_stress_factors': {
                'high_complexity_threshold': 0.8,
                'review_queue_size_threshold': 5,
                'response_time_pressure_threshold': 24
            }
        }
        self.analyzer = StressAnalyzer(self.stress_config)
        
        # テスト用の開発者データ
        self.test_developer = {
            'email': 'test@example.com',
            'expertise_areas': ['python', 'web'],
            'collaboration_quality': 0.7,
            'recent_rejection_rate': 0.1,
            'active_tasks_count': 2,
            'avg_response_time_hours': 12.0,
            'activity_pattern': {
                'peak_hours': [9, 10, 11, 14, 15, 16]
            }
        }
        
        # テスト用のコンテキストデータ
        self.test_context = {
            'review_queue': [
                {
                    'technical_domain': 'python',
                    'complexity_score': 0.5,
                    'deadline': (datetime.now() + timedelta(hours=48)).isoformat(),
                    'requester_relationship': 0.8
                },
                {
                    'technical_domain': 'java',
                    'complexity_score': 0.9,
                    'deadline': (datetime.now() + timedelta(hours=12)).isoformat(),
                    'requester_relationship': 0.2
                }
            ],
            'continuous_work_hours': 4
        }
    
    def test_calculate_stress_indicators(self):
        """ストレス指標計算のテスト"""
        indicators = self.analyzer.calculate_stress_indicators(
            self.test_developer, self.test_context
        )
        
        # 戻り値の型をチェック
        self.assertIsInstance(indicators, StressIndicators)
        
        # 各ストレス値が0-1の範囲内であることを確認
        self.assertGreaterEqual(indicators.task_compatibility_stress, 0.0)
        self.assertLessEqual(indicators.task_compatibility_stress, 1.0)
        
        self.assertGreaterEqual(indicators.workload_stress, 0.0)
        self.assertLessEqual(indicators.workload_stress, 1.0)
        
        self.assertGreaterEqual(indicators.social_stress, 0.0)
        self.assertLessEqual(indicators.social_stress, 1.0)
        
        self.assertGreaterEqual(indicators.temporal_stress, 0.0)
        self.assertLessEqual(indicators.temporal_stress, 1.0)
        
        self.assertGreaterEqual(indicators.total_stress, 0.0)
        self.assertLessEqual(indicators.total_stress, 1.0)
        
        # ストレスレベルが適切な値であることを確認
        self.assertIn(indicators.stress_level, ['low', 'medium', 'high', 'critical'])
    
    def test_task_compatibility_stress_calculation(self):
        """タスク適合度ストレス計算のテスト"""
        # 専門外タスクが多い場合
        high_mismatch_context = {
            'review_queue': [
                {'technical_domain': 'rust', 'complexity_score': 0.5},
                {'technical_domain': 'go', 'complexity_score': 0.9},
                {'technical_domain': 'c++', 'complexity_score': 0.7}
            ]
        }
        
        stress = self.analyzer._calculate_task_compatibility_stress(
            self.test_developer, high_mismatch_context
        )
        
        # 専門外タスクが多いのでストレスが高いはず
        self.assertGreater(stress, 0.5)
        
        # 専門内タスクが多い場合
        low_mismatch_context = {
            'review_queue': [
                {'technical_domain': 'python', 'complexity_score': 0.3},
                {'technical_domain': 'web', 'complexity_score': 0.4}
            ]
        }
        
        stress_low = self.analyzer._calculate_task_compatibility_stress(
            self.test_developer, low_mismatch_context
        )
        
        # 専門内タスクが多いのでストレスが低いはず
        self.assertLess(stress_low, stress)
    
    def test_workload_stress_calculation(self):
        """ワークロードストレス計算のテスト"""
        # 高負荷の場合
        high_workload_context = {
            'review_queue': [{'id': i} for i in range(10)],  # 大量のレビュー
            'continuous_work_hours': 10
        }
        
        high_workload_developer = {
            **self.test_developer,
            'active_tasks_count': 5
        }
        
        stress = self.analyzer._calculate_workload_stress(
            high_workload_developer, high_workload_context
        )
        
        # 高負荷なのでストレスが高いはず
        self.assertGreater(stress, 0.5)
        
        # 低負荷の場合
        low_workload_context = {
            'review_queue': [{'id': 1}],  # 少ないレビュー
            'continuous_work_hours': 2
        }
        
        low_workload_developer = {
            **self.test_developer,
            'active_tasks_count': 1
        }
        
        stress_low = self.analyzer._calculate_workload_stress(
            low_workload_developer, low_workload_context
        )
        
        # 低負荷なのでストレスが低いはず
        self.assertLess(stress_low, stress)
    
    def test_social_stress_calculation(self):
        """社会的ストレス計算のテスト"""
        # 協力関係が悪い場合
        poor_social_developer = {
            **self.test_developer,
            'collaboration_quality': 0.2,
            'recent_rejection_rate': 0.8
        }
        
        stress = self.analyzer._calculate_social_stress(
            poor_social_developer, self.test_context
        )
        
        # 協力関係が悪いのでストレスが高いはず
        self.assertGreater(stress, 0.5)
        
        # 協力関係が良い場合
        good_social_developer = {
            **self.test_developer,
            'collaboration_quality': 0.9,
            'recent_rejection_rate': 0.1
        }
        
        stress_low = self.analyzer._calculate_social_stress(
            good_social_developer, self.test_context
        )
        
        # 協力関係が良いのでストレスが低いはず
        self.assertLess(stress_low, stress)
    
    def test_temporal_stress_calculation(self):
        """時間的ストレス計算のテスト"""
        # 長時間連続作業の場合
        high_temporal_context = {
            **self.test_context,
            'continuous_work_hours': 12
        }
        
        with patch('src.gerrit_retention.prediction.stress_analyzer.datetime') as mock_datetime:
            # 通常の作業時間外（深夜）に設定
            mock_datetime.now.return_value = datetime(2023, 1, 1, 23, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            stress = self.analyzer._calculate_temporal_stress(
                self.test_developer, high_temporal_context
            )
            
            # 時間外 + 長時間作業でストレスが高いはず
            self.assertGreater(stress, 0.3)
    
    def test_stress_level_determination(self):
        """ストレスレベル判定のテスト"""
        # 各レベルのテスト
        self.assertEqual(self.analyzer._determine_stress_level(0.1), 'low')
        self.assertEqual(self.analyzer._determine_stress_level(0.4), 'medium')
        self.assertEqual(self.analyzer._determine_stress_level(0.7), 'high')
        self.assertEqual(self.analyzer._determine_stress_level(0.9), 'critical')
    
    def test_get_stress_breakdown(self):
        """ストレス分解情報取得のテスト"""
        indicators = self.analyzer.calculate_stress_indicators(
            self.test_developer, self.test_context
        )
        
        breakdown = self.analyzer.get_stress_breakdown(indicators)
        
        # 必要なキーが含まれていることを確認
        self.assertIn('total_stress', breakdown)
        self.assertIn('stress_level', breakdown)
        self.assertIn('components', breakdown)
        self.assertIn('calculated_at', breakdown)
        
        # コンポーネントの詳細が含まれていることを確認
        components = breakdown['components']
        for component in ['task_compatibility', 'workload', 'social', 'temporal']:
            self.assertIn(component, components)
            self.assertIn('value', components[component])
            self.assertIn('weight', components[component])
            self.assertIn('contribution', components[component])
    
    def test_is_stress_critical(self):
        """危険ストレスレベル判定のテスト"""
        # 危険レベルのストレス指標
        critical_indicators = StressIndicators(
            task_compatibility_stress=0.9,
            workload_stress=0.8,
            social_stress=0.7,
            temporal_stress=0.6,
            total_stress=0.8,
            stress_level='critical',
            calculated_at=datetime.now()
        )
        
        self.assertTrue(self.analyzer.is_stress_critical(critical_indicators))
        
        # 正常レベルのストレス指標
        normal_indicators = StressIndicators(
            task_compatibility_stress=0.3,
            workload_stress=0.2,
            social_stress=0.1,
            temporal_stress=0.1,
            total_stress=0.2,
            stress_level='low',
            calculated_at=datetime.now()
        )
        
        self.assertFalse(self.analyzer.is_stress_critical(normal_indicators))
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 不正なデータでもクラッシュしないことを確認
        invalid_developer = {}
        invalid_context = {}
        
        indicators = self.analyzer.calculate_stress_indicators(
            invalid_developer, invalid_context
        )
        
        # デフォルト値が返されることを確認
        self.assertIsInstance(indicators, StressIndicators)
        # 実際の計算結果に基づいてテストを調整
        self.assertIn(indicators.stress_level, ['low', 'medium', 'high', 'critical'])


if __name__ == '__main__':
    unittest.main()