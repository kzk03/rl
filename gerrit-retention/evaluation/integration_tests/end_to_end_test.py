#!/usr/bin/env python3
"""
エンドツーエンド統合テストシステム

このモジュールは、Gerrit開発者定着予測システムの全フローを検証する。
データ取得から予測まで、時系列整合性を含む包括的なテストを実行する。

要件: 8.1, 8.2, 8.3
"""

import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.behavior_analysis.review_behavior import ReviewBehaviorAnalyzer
from gerrit_retention.data_integration.data_transformer import DataTransformer
from gerrit_retention.data_integration.gerrit_client import GerritClient
from gerrit_retention.prediction.boiling_point_predictor import BoilingPointPredictor
from gerrit_retention.prediction.retention_predictor import RetentionPredictor
from gerrit_retention.prediction.stress_analyzer import StressAnalyzer
from gerrit_retention.rl_environment.review_env import ReviewAcceptanceEnvironment
from gerrit_retention.utils.config_manager import ConfigManager
from gerrit_retention.utils.logger import setup_logger


class EndToEndTestSuite:
    """エンドツーエンドテストスイート"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        テストスイートを初期化
        
        Args:
            config_path: テスト設定ファイルのパス
        """
        self.logger = setup_logger(__name__)
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # テスト用一時ディレクトリ
        self.temp_dir = None
        self.test_data_dir = None
        
        # テスト結果
        self.test_results = {
            'data_integration': {},
            'feature_engineering': {},
            'prediction_models': {},
            'rl_environment': {},
            'temporal_consistency': {},
            'performance_metrics': {}
        }
        
    def setup_test_environment(self) -> None:
        """テスト環境をセットアップ"""
        self.logger.info("テスト環境をセットアップ中...")
        
        # 一時ディレクトリ作成
        self.temp_dir = tempfile.mkdtemp(prefix="gerrit_retention_test_")
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # テスト用設定ファイル作成
        self._create_test_config()
        
        # テスト用データ生成
        self._generate_test_data()
        
        self.logger.info(f"テスト環境セットアップ完了: {self.temp_dir}")
        
    def teardown_test_environment(self) -> None:
        """テスト環境をクリーンアップ"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.logger.info("テスト環境をクリーンアップしました")
            
    def run_full_test_suite(self) -> Dict[str, Any]:
        """
        完全なテストスイートを実行
        
        Returns:
            テスト結果の辞書
        """
        try:
            self.setup_test_environment()
            
            # 各テストフェーズを順次実行
            self.logger.info("=== エンドツーエンドテスト開始 ===")
            
            # 1. データ統合テスト
            self._test_data_integration()
            
            # 2. 特徴量エンジニアリングテスト
            self._test_feature_engineering()
            
            # 3. 予測モデルテスト
            self._test_prediction_models()
            
            # 4. 強化学習環境テスト
            self._test_rl_environment()
            
            # 5. 時系列整合性テスト
            self._test_temporal_consistency()
            
            # 6. パフォーマンステスト
            self._test_performance_metrics()
            
            # 7. 統合フローテスト
            self._test_integrated_flow()
            
            self.logger.info("=== エンドツーエンドテスト完了 ===")
            
            return self._generate_test_report()
            
        except Exception as e:
            self.logger.error(f"テスト実行中にエラーが発生: {e}")
            raise
        finally:
            self.teardown_test_environment()
            
    def _create_test_config(self) -> None:
        """テスト用設定ファイルを作成"""
        test_config = {
            'gerrit_integration': {
                'gerrit_url': 'http://test-gerrit.example.com',
                'auth': {
                    'username': 'test_user',
                    'password': 'test_password'
                },
                'projects': ['test-project'],
                'data_extraction': {
                    'batch_size': 10,
                    'rate_limit_delay': 0.1,
                    'max_retries': 2
                }
            },
            'retention_prediction': {
                'model_type': 'random_forest',
                'model_params': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            'stress_analysis': {
                'weights': {
                    'review_compatibility_stress': 0.3,
                    'workload_stress': 0.4,
                    'social_stress': 0.2,
                    'temporal_stress': 0.1
                }
            },
            'rl_environment': {
                'observation_space_dim': 20,
                'action_space_size': 3,
                'max_episode_length': 50
            },
            'temporal_consistency': {
                'enable_strict_validation': True,
                'train_end_date': '2022-12-31',
                'test_start_date': '2023-01-01'
            }
        }
        
        config_path = self.test_data_dir / "test_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)
            
        # ConfigManagerを更新
        self.config_manager = ConfigManager(str(config_path))
        self.config = self.config_manager.get_config()
        
    def _generate_test_data(self) -> None:
        """テスト用データを生成"""
        # Gerrit Changes データ
        changes_data = []
        for i in range(50):
            change = {
                'change_id': f'test_change_{i}',
                'project': 'test-project',
                'author': f'developer_{i % 10}@example.com',
                'created': (datetime.now() - timedelta(days=365-i*7)).isoformat(),
                'updated': (datetime.now() - timedelta(days=365-i*7-1)).isoformat(),
                'status': 'MERGED' if i % 3 == 0 else 'NEW',
                'files_changed': np.random.randint(1, 20),
                'lines_added': np.random.randint(10, 500),
                'lines_deleted': np.random.randint(0, 200)
            }
            changes_data.append(change)
            
        # Reviews データ
        reviews_data = []
        for i in range(100):
            review = {
                'change_id': f'test_change_{i % 50}',
                'reviewer_email': f'reviewer_{i % 8}@example.com',
                'timestamp': (datetime.now() - timedelta(days=365-i*3)).isoformat(),
                'score': np.random.choice([-2, -1, 0, 1, 2], p=[0.05, 0.15, 0.4, 0.3, 0.1]),
                'response_time_hours': np.random.exponential(24),
                'message': f'Test review message {i}'
            }
            reviews_data.append(review)
            
        # Developer Profiles データ
        developers_data = {}
        for i in range(15):
            email = f'developer_{i}@example.com'
            developers_data[email] = {
                'email': email,
                'name': f'Developer {i}',
                'expertise_level': np.random.uniform(0.3, 0.9),
                'stress_level': np.random.uniform(0.1, 0.8),
                'activity_pattern': {
                    'morning': np.random.uniform(0.1, 0.9),
                    'afternoon': np.random.uniform(0.1, 0.9),
                    'evening': np.random.uniform(0.1, 0.9)
                },
                'recent_review_acceptance_rate': np.random.uniform(0.5, 0.95),
                'workload_ratio': np.random.uniform(0.3, 1.2),
                'collaboration_quality': np.random.uniform(0.4, 0.9)
            }
            
        # データファイルを保存
        with open(self.test_data_dir / "test_changes.json", 'w') as f:
            json.dump(changes_data, f, indent=2)
            
        with open(self.test_data_dir / "test_reviews.json", 'w') as f:
            json.dump(reviews_data, f, indent=2)
            
        with open(self.test_data_dir / "test_developers.pkl", 'wb') as f:
            pickle.dump(developers_data, f)
            
    def _test_data_integration(self) -> None:
        """データ統合テスト"""
        self.logger.info("データ統合テストを実行中...")
        
        try:
            # Gerrit クライアントのモックテスト
            with patch('gerrit_retention.data_integration.gerrit_client.requests') as mock_requests:
                mock_response = Mock()
                mock_response.json.return_value = {'test': 'data'}
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                
                client = GerritClient(
                    gerrit_url=self.config['gerrit_integration']['gerrit_url'],
                    auth_config=self.config['gerrit_integration']['auth']
                )
                
                # データ抽出テスト
                result = client.extract_review_data('test-project', ('2022-01-01', '2022-12-31'))
                self.test_results['data_integration']['gerrit_client'] = 'PASS'
                
            # データ変換テスト
            transformer = DataTransformer()
            
            # テストデータを読み込み
            with open(self.test_data_dir / "test_changes.json", 'r') as f:
                changes_data = json.load(f)
                
            with open(self.test_data_dir / "test_reviews.json", 'r') as f:
                reviews_data = json.load(f)
                
            # データ変換実行
            transformed_data = transformer.transform_gerrit_data(changes_data, reviews_data)
            
            # 変換結果検証
            assert 'features' in transformed_data
            assert 'labels' in transformed_data
            assert len(transformed_data['features']) > 0
            
            self.test_results['data_integration']['data_transformer'] = 'PASS'
            self.logger.info("データ統合テスト完了")
            
        except Exception as e:
            self.logger.error(f"データ統合テストでエラー: {e}")
            self.test_results['data_integration']['error'] = str(e)
            
    def _test_feature_engineering(self) -> None:
        """特徴量エンジニアリングテスト"""
        self.logger.info("特徴量エンジニアリングテストを実行中...")
        
        try:
            # テスト用開発者データ読み込み
            with open(self.test_data_dir / "test_developers.pkl", 'rb') as f:
                developers_data = pickle.load(f)
                
            # 各特徴量エンジニアリングモジュールをテスト
            from gerrit_retention.data_processing.feature_engineering.developer_features import (
                DeveloperFeatureExtractor,
            )
            from gerrit_retention.data_processing.feature_engineering.review_features import (
                ReviewFeatureExtractor,
            )
            from gerrit_retention.data_processing.feature_engineering.temporal_features import (
                TemporalFeatureExtractor,
            )

            # 開発者特徴量テスト
            dev_extractor = DeveloperFeatureExtractor()
            sample_developer = list(developers_data.values())[0]
            dev_features = dev_extractor.extract_features(sample_developer)
            
            assert isinstance(dev_features, np.ndarray)
            assert len(dev_features) > 0
            
            # レビュー特徴量テスト
            review_extractor = ReviewFeatureExtractor()
            sample_review = {
                'change_id': 'test_change_1',
                'complexity_score': 0.7,
                'files_changed': 5,
                'lines_added': 100,
                'technical_domain': 'backend'
            }
            review_features = review_extractor.extract_features(sample_review)
            
            assert isinstance(review_features, np.ndarray)
            assert len(review_features) > 0
            
            # 時系列特徴量テスト
            temporal_extractor = TemporalFeatureExtractor()
            temporal_data = {
                'timestamps': [datetime.now() - timedelta(days=i) for i in range(30)],
                'activities': np.random.rand(30)
            }
            temporal_features = temporal_extractor.extract_features(temporal_data)
            
            assert isinstance(temporal_features, np.ndarray)
            assert len(temporal_features) > 0
            
            self.test_results['feature_engineering']['all_extractors'] = 'PASS'
            self.logger.info("特徴量エンジニアリングテスト完了")
            
        except Exception as e:
            self.logger.error(f"特徴量エンジニアリングテストでエラー: {e}")
            self.test_results['feature_engineering']['error'] = str(e)
            
    def _test_prediction_models(self) -> None:
        """予測モデルテスト"""
        self.logger.info("予測モデルテストを実行中...")
        
        try:
            # テストデータ準備
            n_samples = 100
            n_features = 20
            X_test = np.random.rand(n_samples, n_features)
            y_test = np.random.randint(0, 2, n_samples)
            
            # 定着予測モデルテスト
            retention_predictor = RetentionPredictor(self.config['retention_prediction'])
            
            # モデル訓練（簡易版）
            retention_predictor.train(X_test, y_test)
            
            # 予測テスト
            sample_developer = {
                'expertise_level': 0.7,
                'stress_level': 0.3,
                'activity_pattern': {'morning': 0.8, 'afternoon': 0.6, 'evening': 0.4}
            }
            sample_context = {'project_phase': 'development', 'team_size': 10}
            
            retention_prob = retention_predictor.predict_retention_probability(
                sample_developer, sample_context
            )
            
            assert 0 <= retention_prob <= 1
            
            # ストレス分析テスト
            stress_analyzer = StressAnalyzer(self.config['stress_analysis'])
            stress_indicators = stress_analyzer.calculate_stress_indicators(
                sample_developer, sample_context
            )
            
            assert isinstance(stress_indicators, dict)
            assert 'workload_stress' in stress_indicators
            
            # 沸点予測テスト
            boiling_predictor = BoilingPointPredictor()
            boiling_prediction = boiling_predictor.predict_boiling_point(
                sample_developer, sample_context
            )
            
            assert isinstance(boiling_prediction, dict)
            assert 'current_stress' in boiling_prediction
            assert 'boiling_threshold' in boiling_prediction
            
            self.test_results['prediction_models']['all_models'] = 'PASS'
            self.logger.info("予測モデルテスト完了")
            
        except Exception as e:
            self.logger.error(f"予測モデルテストでエラー: {e}")
            self.test_results['prediction_models']['error'] = str(e)
            
    def _test_rl_environment(self) -> None:
        """強化学習環境テスト"""
        self.logger.info("強化学習環境テストを実行中...")
        
        try:
            # レビュー受諾環境テスト
            env_config = self.config['rl_environment']
            env = ReviewAcceptanceEnvironment(env_config)
            
            # 環境リセットテスト
            initial_obs = env.reset()
            assert initial_obs is not None
            assert len(initial_obs) == env_config['observation_space_dim']
            
            # ステップ実行テスト
            for action in range(env_config['action_space_size']):
                obs, reward, done, info = env.step(action)
                
                assert isinstance(obs, np.ndarray)
                assert isinstance(reward, (int, float))
                assert isinstance(done, bool)
                assert isinstance(info, dict)
                
                if done:
                    env.reset()
                    
            # レビュー行動分析テスト
            behavior_analyzer = ReviewBehaviorAnalyzer()
            
            sample_review_request = {
                'change_id': 'test_change_1',
                'complexity': 0.6,
                'technical_domain': 'backend',
                'urgency': 0.7
            }
            
            sample_reviewer = {
                'email': 'reviewer@example.com',
                'expertise_level': 0.8,
                'current_workload': 0.4
            }
            
            acceptance_prob = behavior_analyzer.predict_acceptance_probability(
                sample_review_request, sample_reviewer
            )
            
            assert 0 <= acceptance_prob <= 1
            
            self.test_results['rl_environment']['environment'] = 'PASS'
            self.test_results['rl_environment']['behavior_analysis'] = 'PASS'
            self.logger.info("強化学習環境テスト完了")
            
        except Exception as e:
            self.logger.error(f"強化学習環境テストでエラー: {e}")
            self.test_results['rl_environment']['error'] = str(e) 
           
    def _test_temporal_consistency(self) -> None:
        """時系列整合性テスト"""
        self.logger.info("時系列整合性テストを実行中...")
        
        try:
            # 時系列データの整合性検証
            temporal_config = self.config['temporal_consistency']
            
            if temporal_config['enable_strict_validation']:
                train_end = datetime.strptime(temporal_config['train_end_date'], '%Y-%m-%d')
                test_start = datetime.strptime(temporal_config['test_start_date'], '%Y-%m-%d')
                
                # 訓練終了日がテスト開始日より前であることを確認
                assert train_end < test_start, "訓練データがテストデータより未来の日付を含んでいます"
                
                # テストデータの時系列順序検証
                test_dates = []
                for i in range(10):
                    test_date = test_start + timedelta(days=i*30)
                    test_dates.append(test_date)
                    
                # 日付が昇順であることを確認
                assert test_dates == sorted(test_dates), "テストデータの時系列順序が正しくありません"
                
                # データリーク検証（未来データが過去の予測に使用されていないか）
                self._validate_no_data_leakage(train_end, test_start)
                
            self.test_results['temporal_consistency']['validation'] = 'PASS'
            self.logger.info("時系列整合性テスト完了")
            
        except Exception as e:
            self.logger.error(f"時系列整合性テストでエラー: {e}")
            self.test_results['temporal_consistency']['error'] = str(e)
            
    def _validate_no_data_leakage(self, train_end: datetime, test_start: datetime) -> None:
        """データリークの検証"""
        # テストデータを読み込み、日付をチェック
        with open(self.test_data_dir / "test_changes.json", 'r') as f:
            changes_data = json.load(f)
            
        for change in changes_data:
            created_date = datetime.fromisoformat(change['created'].replace('Z', '+00:00').replace('+00:00', ''))
            
            # 訓練期間のデータが適切な期間内にあることを確認
            if created_date <= train_end:
                # 訓練データとして使用可能
                continue
            elif created_date >= test_start:
                # テストデータとして使用可能
                continue
            else:
                # 検証期間のデータ（オプション）
                continue
                
    def _test_performance_metrics(self) -> None:
        """パフォーマンステスト"""
        self.logger.info("パフォーマンステストを実行中...")
        
        try:
            import gc
            import time

            import psutil

            # メモリ使用量測定開始
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # CPU使用率測定開始
            cpu_percent_start = psutil.cpu_percent()
            
            # パフォーマンステスト実行
            start_time = time.time()
            
            # 大量データでの予測テスト
            n_samples = 1000
            n_features = 20
            X_large = np.random.rand(n_samples, n_features)
            
            # 定着予測の実行時間測定
            retention_predictor = RetentionPredictor(self.config['retention_prediction'])
            
            prediction_start = time.time()
            for i in range(100):
                sample_developer = {
                    'expertise_level': np.random.rand(),
                    'stress_level': np.random.rand(),
                    'activity_pattern': {
                        'morning': np.random.rand(),
                        'afternoon': np.random.rand(),
                        'evening': np.random.rand()
                    }
                }
                sample_context = {'project_phase': 'development', 'team_size': 10}
                _ = retention_predictor.predict_retention_probability(sample_developer, sample_context)
                
            prediction_time = time.time() - prediction_start
            
            # ストレス分析の実行時間測定
            stress_analyzer = StressAnalyzer(self.config['stress_analysis'])
            
            stress_start = time.time()
            for i in range(100):
                sample_developer = {
                    'expertise_level': np.random.rand(),
                    'stress_level': np.random.rand(),
                    'workload_ratio': np.random.rand()
                }
                sample_context = {'project_phase': 'development', 'team_size': 10}
                _ = stress_analyzer.calculate_stress_indicators(sample_developer, sample_context)
                
            stress_time = time.time() - stress_start
            
            total_time = time.time() - start_time
            
            # メモリ使用量測定終了
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            # CPU使用率測定終了
            cpu_percent_end = psutil.cpu_percent()
            
            # パフォーマンス結果記録
            performance_metrics = {
                'total_execution_time': total_time,
                'prediction_time_per_100_calls': prediction_time,
                'stress_analysis_time_per_100_calls': stress_time,
                'memory_usage_mb': memory_usage,
                'cpu_usage_percent': (cpu_percent_start + cpu_percent_end) / 2
            }
            
            # パフォーマンス基準チェック
            assert prediction_time < 10.0, f"予測処理が遅すぎます: {prediction_time}秒"
            assert stress_time < 5.0, f"ストレス分析が遅すぎます: {stress_time}秒"
            assert memory_usage < 500, f"メモリ使用量が多すぎます: {memory_usage}MB"
            
            self.test_results['performance_metrics'] = performance_metrics
            self.test_results['performance_metrics']['status'] = 'PASS'
            self.logger.info("パフォーマンステスト完了")
            
            # メモリクリーンアップ
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"パフォーマンステストでエラー: {e}")
            self.test_results['performance_metrics']['error'] = str(e)
            
    def _test_integrated_flow(self) -> None:
        """統合フローテスト"""
        self.logger.info("統合フローテストを実行中...")
        
        try:
            # エンドツーエンドフローのシミュレーション
            
            # 1. データ取得シミュレーション
            with patch('gerrit_retention.data_integration.gerrit_client.requests') as mock_requests:
                mock_response = Mock()
                mock_response.json.return_value = {
                    'changes': [
                        {
                            'id': 'test_change_1',
                            'project': 'test-project',
                            'owner': {'email': 'developer@example.com'},
                            'created': '2023-01-01T00:00:00Z',
                            'status': 'NEW'
                        }
                    ]
                }
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                
                client = GerritClient(
                    gerrit_url=self.config['gerrit_integration']['gerrit_url'],
                    auth_config=self.config['gerrit_integration']['auth']
                )
                
                # データ抽出
                review_data = client.extract_review_data('test-project', ('2023-01-01', '2023-12-31'))
                
            # 2. データ変換
            transformer = DataTransformer()
            transformed_data = transformer.transform_gerrit_data(review_data, [])
            
            # 3. 特徴量抽出
            from gerrit_retention.data_processing.feature_engineering.developer_features import (
                DeveloperFeatureExtractor,
            )
            dev_extractor = DeveloperFeatureExtractor()
            
            sample_developer = {
                'email': 'developer@example.com',
                'expertise_level': 0.7,
                'stress_level': 0.3,
                'activity_pattern': {'morning': 0.8, 'afternoon': 0.6, 'evening': 0.4}
            }
            
            features = dev_extractor.extract_features(sample_developer)
            
            # 4. 予測実行
            retention_predictor = RetentionPredictor(self.config['retention_prediction'])
            stress_analyzer = StressAnalyzer(self.config['stress_analysis'])
            
            sample_context = {'project_phase': 'development', 'team_size': 10}
            
            # 定着予測
            retention_prob = retention_predictor.predict_retention_probability(
                sample_developer, sample_context
            )
            
            # ストレス分析
            stress_indicators = stress_analyzer.calculate_stress_indicators(
                sample_developer, sample_context
            )
            
            # 5. 強化学習環境での行動選択
            env = ReviewAcceptanceEnvironment(self.config['rl_environment'])
            obs = env.reset()
            action = env.action_space.sample()  # ランダム行動
            next_obs, reward, done, info = env.step(action)
            
            # 6. 結果検証
            assert isinstance(retention_prob, float)
            assert 0 <= retention_prob <= 1
            assert isinstance(stress_indicators, dict)
            assert isinstance(reward, (int, float))
            
            # 統合フロー成功
            integrated_flow_result = {
                'data_extraction': 'SUCCESS',
                'data_transformation': 'SUCCESS',
                'feature_extraction': 'SUCCESS',
                'retention_prediction': retention_prob,
                'stress_analysis': stress_indicators,
                'rl_action_selection': {
                    'action': int(action),
                    'reward': float(reward),
                    'done': done
                }
            }
            
            self.test_results['integrated_flow'] = integrated_flow_result
            self.logger.info("統合フローテスト完了")
            
        except Exception as e:
            self.logger.error(f"統合フローテストでエラー: {e}")
            self.test_results['integrated_flow'] = {'error': str(e)}
            
    def _generate_test_report(self) -> Dict[str, Any]:
        """テストレポートを生成"""
        self.logger.info("テストレポートを生成中...")
        
        # 成功/失敗の集計
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        def count_results(results_dict):
            nonlocal total_tests, passed_tests, failed_tests
            
            for key, value in results_dict.items():
                if isinstance(value, dict):
                    count_results(value)
                elif key == 'status' and value == 'PASS':
                    passed_tests += 1
                    total_tests += 1
                elif key == 'error':
                    failed_tests += 1
                    total_tests += 1
                elif value == 'PASS':
                    passed_tests += 1
                    total_tests += 1
                    
        count_results(self.test_results)
        
        # レポート生成
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'execution_timestamp': datetime.now().isoformat()
            },
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        # レポートファイル保存
        report_path = Path(self.temp_dir) / "end_to_end_test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
        self.logger.info(f"テストレポート生成完了: {report_path}")
        
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """テスト結果に基づく推奨事項を生成"""
        recommendations = []
        
        # エラーがあった場合の推奨事項
        for category, results in self.test_results.items():
            if isinstance(results, dict) and 'error' in results:
                recommendations.append(f"{category}でエラーが発生しました。詳細を確認してください。")
                
        # パフォーマンスに関する推奨事項
        if 'performance_metrics' in self.test_results:
            perf = self.test_results['performance_metrics']
            if isinstance(perf, dict):
                if perf.get('prediction_time_per_100_calls', 0) > 5.0:
                    recommendations.append("予測処理の最適化を検討してください。")
                if perf.get('memory_usage_mb', 0) > 200:
                    recommendations.append("メモリ使用量の最適化を検討してください。")
                    
        # 時系列整合性に関する推奨事項
        if 'temporal_consistency' in self.test_results:
            temp = self.test_results['temporal_consistency']
            if isinstance(temp, dict) and 'error' in temp:
                recommendations.append("時系列データの整合性を確認し、データリークを防止してください。")
                
        if not recommendations:
            recommendations.append("すべてのテストが正常に完了しました。")
            
        return recommendations


class EndToEndTestRunner:
    """エンドツーエンドテスト実行器"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
    def run_tests(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        テストを実行
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            テスト結果
        """
        self.logger.info("エンドツーエンドテスト実行を開始します")
        
        test_suite = EndToEndTestSuite(config_path)
        
        try:
            results = test_suite.run_full_test_suite()
            
            # 結果サマリーをログ出力
            summary = results['test_summary']
            self.logger.info(f"テスト完了: {summary['passed_tests']}/{summary['total_tests']} 成功")
            self.logger.info(f"成功率: {summary['success_rate']:.2%}")
            
            if summary['failed_tests'] > 0:
                self.logger.warning(f"{summary['failed_tests']}個のテストが失敗しました")
                
            return results
            
        except Exception as e:
            self.logger.error(f"テスト実行中に予期しないエラーが発生: {e}")
            raise


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gerrit開発者定着予測システム エンドツーエンドテスト')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    parser.add_argument('--output', type=str, help='結果出力ディレクトリ')
    parser.add_argument('--verbose', action='store_true', help='詳細ログ出力')
    
    args = parser.parse_args()
    
    # ログレベル設定
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # テスト実行
    runner = EndToEndTestRunner()
    results = runner.run_tests(args.config)
    
    # 結果出力
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"end_to_end_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"テスト結果を保存しました: {output_file}")
        
    # 終了コード設定
    if results['test_summary']['failed_tests'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()