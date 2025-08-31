"""
適応的戦略システムのテスト
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
from gerrit_retention.adaptive_strategy import (
    ContinualLearner,
    MultiObjectiveOptimizer,
    StrategyManager,
)


class TestStrategyManager(unittest.TestCase):
    """戦略管理システムのテスト"""
    
    def setUp(self):
        """テスト前の準備"""
        self.config = {
            'stress_config': {},
            'preference_config': {}
        }
        
        # モックの設定
        with patch('gerrit_retention.adaptive_strategy.strategy_manager.StressAnalyzer'), \
             patch('gerrit_retention.adaptive_strategy.strategy_manager.PreferenceAnalyzer'):
            self.strategy_manager = StrategyManager(self.config)
    
    def test_strategy_initialization(self):
        """戦略初期化のテスト"""
        self.assertIsNotNone(self.strategy_manager.strategies)
        self.assertIn('low_stress_growth', self.strategy_manager.strategies)
        self.assertIn('high_stress_relief', self.strategy_manager.strategies)
    
    def test_developer_state_update(self):
        """開発者状態更新のテスト"""
        developer_id = "test_developer"
        context = {
            'current_workload': 0.6,
            'recent_performance': 0.8,
            'collaboration_score': 0.7
        }
        
        with patch.object(self.strategy_manager.stress_analyzer, 'calculate_stress_indicators') as mock_stress:
            mock_stress.return_value = {'total_stress': 0.3}
            
            state = self.strategy_manager.update_developer_state(developer_id, context)
            
            self.assertEqual(state.developer_id, developer_id)
            self.assertIsNotNone(state.stress_level)
            self.assertIsNotNone(state.expertise_stage)


class TestMultiObjectiveOptimizer(unittest.TestCase):
    """多目的最適化システムのテスト"""
    
    def setUp(self):
        """テスト前の準備"""
        self.config = {
            'task_completion_weight': 0.3,
            'retention_weight': 0.4,
            'optimization_runs': 2
        }
        self.optimizer = MultiObjectiveOptimizer(self.config)
    
    def test_objectives_initialization(self):
        """目的関数初期化のテスト"""
        self.assertIsNotNone(self.optimizer.objectives)
        self.assertIn('task_completion_rate', self.optimizer.objectives)
        self.assertIn('developer_retention', self.optimizer.objectives)
    
    def test_optimization_execution(self):
        """最適化実行のテスト"""
        current_state = {
            'historical_completion_rate': 0.7,
            'historical_retention_rate': 0.8,
            'current_stress_level': 0.4
        }
        
        result = self.optimizer.optimize_recommendation_strategy(current_state)
        
        self.assertIsInstance(result, dict)
        self.assertIn('efficiency_factor', result)


class TestContinualLearner(unittest.TestCase):
    """継続学習システムのテスト"""
    
    def setUp(self):
        """テスト前の準備"""
        self.config = {
            'buffer_size': 100,
            'learning_frequency': 5,
            'drift_threshold': 0.1
        }
        self.learner = ContinualLearner(self.config)
    
    def test_learning_instance_addition(self):
        """学習インスタンス追加のテスト"""
        features = np.array([0.5, 0.6, 0.7])
        target = 0.8
        developer_id = "test_dev"
        
        initial_buffer_size = len(self.learner.learning_buffer)
        
        self.learner.add_learning_instance(features, target, developer_id)
        
        self.assertEqual(len(self.learner.learning_buffer), initial_buffer_size + 1)
        self.assertIn(developer_id, self.learner.developer_data)
    
    def test_new_developer_adaptation(self):
        """新規開発者適応のテスト"""
        developer_id = "new_developer"
        context = {
            'expertise_level': 0.3,
            'activity_level': 0.5
        }
        
        strategy = self.learner.adapt_to_new_developer(developer_id, context)
        
        self.assertIsInstance(strategy, dict)
        self.assertEqual(strategy['developer_id'], developer_id)
        self.assertIn('learning_rate_multiplier', strategy)
    
    def test_prediction_with_uncertainty(self):
        """不確実性を含む予測のテスト"""
        features = np.array([0.5, 0.6, 0.7])
        
        prediction, uncertainty = self.learner.predict_with_uncertainty(features)
        
        self.assertIsInstance(prediction, float)
        self.assertIsInstance(uncertainty, float)
        self.assertGreaterEqual(uncertainty, 0.0)


if __name__ == '__main__':
    unittest.main()