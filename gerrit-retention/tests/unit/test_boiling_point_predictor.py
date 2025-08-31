"""
沸点予測器のテスト

BoilingPointPredictorクラスの機能をテストし、
沸点予測とリスク評価が正しく動作することを確認する。
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
from src.gerrit_retention.prediction.boiling_point_predictor import (
    BoilingPointPrediction,
    BoilingPointPredictor,
    DeveloperExitPattern,
)
from src.gerrit_retention.prediction.stress_analyzer import StressIndicators


class TestBoilingPointPredictor(unittest.TestCase):
    """沸点予測器のテストクラス"""
    
    def setUp(self):
        """テスト用の設定を初期化"""
        self.config = {
            'boiling_point_model': {
                'svr_params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'epsilon': 0.1
                }
            },
            'risk_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
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
        
        self.predictor = BoilingPointPredictor(self.config)
        
        # テスト用の開発者データ
        self.test_developer = {
            'email': 'test@example.com',
            'collaboration_quality': 0.7,
            'expertise_level': 0.8,
            'workload_ratio': 0.6,
            'recent_rejection_rate': 0.1,
            'expertise_areas': ['python', 'web', 'api'],
            'avg_response_time_hours': 12.0
        }
        
        # テスト用のコンテキスト
        self.test_context = {
            'review_queue': [
                {'technical_domain': 'python', 'complexity_score': 0.5},
                {'technical_domain': 'web', 'complexity_score': 0.7}
            ],
            'continuous_work_hours': 6
        }
        
        # テスト用の離脱パターンデータ（訓練に十分な数）
        self.exit_patterns = []
        for i in range(15):  # 15個のパターンを生成
            self.exit_patterns.append(
                DeveloperExitPattern(
                    developer_email=f'dev{i}@example.com',
                    exit_date=datetime.now() - timedelta(days=30 + i*10),
                    stress_history=[0.2 + i*0.05, 0.4 + i*0.03, 0.6 + i*0.02, 0.8 + i*0.01],
                    final_stress_level=0.7 + i*0.02,
                    exit_trigger=['workload', 'social', 'compatibility', 'temporal'][i % 4],
                    days_before_exit=15 + i*2
                )
            )
        
        # テスト用の開発者プロファイル
        self.developer_profiles = {}
        for i in range(15):
            self.developer_profiles[f'dev{i}@example.com'] = {
                'collaboration_quality': 0.3 + i*0.04,
                'expertise_level': 0.5 + i*0.03,
                'workload_ratio': 0.4 + i*0.03,
                'recent_rejection_rate': 0.1 + i*0.02,
                'expertise_areas': ['python', 'web', 'api'][:(i % 3) + 1],
                'avg_response_time_hours': 12.0 + i*2
            }
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsInstance(self.predictor, BoilingPointPredictor)
        self.assertFalse(self.predictor.is_trained)
        self.assertIsNone(self.predictor.model)
        self.assertIsNone(self.predictor.scaler)
    
    def test_train_model(self):
        """モデル訓練のテスト"""
        metrics = self.predictor.train_model(self.exit_patterns, self.developer_profiles)
        
        # 訓練が成功したことを確認
        self.assertIn('mse', metrics)
        self.assertIn('r2_score', metrics)
        self.assertTrue(self.predictor.is_trained)
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)
    
    def test_train_model_insufficient_data(self):
        """データ不足時の訓練テスト"""
        # データが少ない場合
        small_patterns = self.exit_patterns[:2]  # 2個だけ
        small_profiles = {k: v for k, v in list(self.developer_profiles.items())[:2]}
        metrics = self.predictor.train_model(small_patterns, small_profiles)
        
        # エラーが返されることを確認
        self.assertIn('error', metrics)
        self.assertFalse(self.predictor.is_trained)
    
    @patch('src.gerrit_retention.prediction.boiling_point_predictor.StressAnalyzer')
    def test_predict_boiling_point_untrained(self, mock_stress_analyzer):
        """未訓練モデルでの予測テスト"""
        # モックのストレス分析器を設定
        mock_indicators = StressIndicators(
            task_compatibility_stress=0.4,
            workload_stress=0.5,
            social_stress=0.3,
            temporal_stress=0.2,
            total_stress=0.4,
            stress_level='medium',
            calculated_at=datetime.now()
        )
        mock_stress_analyzer.return_value.calculate_stress_indicators.return_value = mock_indicators
        
        prediction = self.predictor.predict_boiling_point(self.test_developer, self.test_context)
        
        # デフォルト予測が返されることを確認
        self.assertIsInstance(prediction, BoilingPointPrediction)
        self.assertEqual(prediction.developer_email, 'test@example.com')
        self.assertEqual(prediction.risk_level, 'medium')
        self.assertLess(prediction.confidence_score, 0.5)  # 低い信頼度
    
    def test_predict_boiling_point_trained(self):
        """訓練済みモデルでの予測テスト"""
        # まずモデルを訓練
        self.predictor.train_model(self.exit_patterns, self.developer_profiles)
        
        prediction = self.predictor.predict_boiling_point(self.test_developer, self.test_context)
        
        # 予測結果の検証
        self.assertIsInstance(prediction, BoilingPointPrediction)
        self.assertEqual(prediction.developer_email, 'test@example.com')
        self.assertIn(prediction.risk_level, ['low', 'medium', 'high', 'critical'])
        self.assertGreaterEqual(prediction.current_stress, 0.0)
        self.assertLessEqual(prediction.current_stress, 1.0)
        self.assertGreaterEqual(prediction.predicted_boiling_point, 0.0)
        self.assertLessEqual(prediction.predicted_boiling_point, 1.0)
        self.assertGreaterEqual(prediction.confidence_score, 0.3)  # 訓練済みなので高い信頼度
    
    def test_risk_level_calculation(self):
        """リスクレベル計算のテスト"""
        # 低リスク
        risk = self.predictor._calculate_risk_level(0.2, 0.8)
        self.assertEqual(risk, 'low')
        
        # 中リスク
        risk = self.predictor._calculate_risk_level(0.4, 0.8)
        self.assertEqual(risk, 'medium')
        
        # 高リスク
        risk = self.predictor._calculate_risk_level(0.6, 0.8)
        self.assertEqual(risk, 'high')
        
        # 危険リスク
        risk = self.predictor._calculate_risk_level(0.8, 0.8)
        self.assertEqual(risk, 'critical')
    
    def test_time_to_boiling_estimation(self):
        """沸点到達時間推定のテスト"""
        # 正常ケース
        time_to_boiling = self.predictor._estimate_time_to_boiling(0.5, 0.8, self.test_developer)
        self.assertIsInstance(time_to_boiling, float)
        self.assertGreater(time_to_boiling, 0)
        
        # 既に沸点に達している場合
        time_to_boiling = self.predictor._estimate_time_to_boiling(0.9, 0.8, self.test_developer)
        self.assertEqual(time_to_boiling, 0.0)
    
    def test_confidence_score_calculation(self):
        """信頼度スコア計算のテスト"""
        features = [0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.2, 0.9, 0.1, 0.5, 1.0, 0.0, 0.0, 1.0]
        
        # 未訓練モデル
        confidence = self.predictor._calculate_confidence_score(features, self.test_developer)
        self.assertLess(confidence, 0.5)
        
        # 訓練済みモデル
        self.predictor.is_trained = True
        confidence = self.predictor._calculate_confidence_score(features, self.test_developer)
        self.assertGreater(confidence, 0.5)
    
    def test_contributing_factors_analysis(self):
        """寄与要因分析のテスト"""
        stress_indicators = StressIndicators(
            task_compatibility_stress=0.4,
            workload_stress=0.6,
            social_stress=0.2,
            temporal_stress=0.1,
            total_stress=0.5,
            stress_level='medium',
            calculated_at=datetime.now()
        )
        
        factors = self.predictor._analyze_contributing_factors(stress_indicators)
        
        # 各要因の寄与度が計算されていることを確認
        self.assertIn('task_compatibility', factors)
        self.assertIn('workload', factors)
        self.assertIn('social', factors)
        self.assertIn('temporal', factors)
        
        # 寄与度の合計が1.0であることを確認（重みの合計）
        total_contribution = sum(factors.values())
        self.assertAlmostEqual(total_contribution, 1.0, places=1)
    
    def test_prepare_training_data(self):
        """訓練データ準備のテスト"""
        X, y = self.predictor._prepare_training_data(self.exit_patterns, self.developer_profiles)
        
        # データ形状の確認
        self.assertEqual(len(X), len(self.exit_patterns))
        self.assertEqual(len(y), len(self.exit_patterns))
        self.assertEqual(X.shape[1], 14)  # 特徴量数
        
        # ターゲット値の確認（生成されたデータに基づく）
        self.assertAlmostEqual(y[0], 0.7)  # 最初のパターンの最終ストレス
        self.assertAlmostEqual(y[1], 0.72)  # 二番目のパターンの最終ストレス
    
    def test_extract_prediction_features(self):
        """予測用特徴量抽出のテスト"""
        stress_indicators = StressIndicators(
            task_compatibility_stress=0.4,
            workload_stress=0.5,
            social_stress=0.3,
            temporal_stress=0.2,
            total_stress=0.4,
            stress_level='medium',
            calculated_at=datetime.now()
        )
        
        features = self.predictor._extract_prediction_features(self.test_developer, stress_indicators)
        
        # 特徴量数の確認
        self.assertEqual(len(features), 14)
        
        # 特徴量の範囲確認
        for feature in features:
            self.assertGreaterEqual(feature, 0.0)
            self.assertLessEqual(feature, 1.0)
    
    def test_model_save_load(self):
        """モデル保存・読み込みのテスト"""
        # 未訓練モデルの保存は失敗するはず
        self.assertFalse(self.predictor.save_model('/tmp/test_model.pkl'))
        
        # モデルを訓練
        self.predictor.train_model(self.exit_patterns, self.developer_profiles)
        
        # 保存のテスト（実際のファイル操作はモック）
        with patch('joblib.dump') as mock_dump:
            result = self.predictor.save_model('/tmp/test_model.pkl')
            self.assertTrue(result)
            mock_dump.assert_called_once()
        
        # 読み込みのテスト
        with patch('joblib.load') as mock_load:
            mock_load.return_value = {
                'model': MagicMock(),
                'scaler': MagicMock(),
                'config': self.config,
                'svr_params': self.predictor.svr_params,
                'risk_thresholds': self.predictor.risk_thresholds
            }
            
            new_predictor = BoilingPointPredictor(self.config)
            result = new_predictor.load_model('/tmp/test_model.pkl')
            self.assertTrue(result)
            self.assertTrue(new_predictor.is_trained)
    
    def test_get_model_info(self):
        """モデル情報取得のテスト"""
        # 未訓練モデル
        info = self.predictor.get_model_info()
        self.assertFalse(info['is_trained'])
        self.assertIsNone(info['feature_count'])
        
        # 訓練済みモデル
        self.predictor.train_model(self.exit_patterns, self.developer_profiles)
        info = self.predictor.get_model_info()
        self.assertTrue(info['is_trained'])
        self.assertEqual(info['feature_count'], 14)
        self.assertEqual(info['model_type'], 'SVR')


if __name__ == '__main__':
    unittest.main()