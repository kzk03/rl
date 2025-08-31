"""
可視化システムのテスト
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.gerrit_retention.visualization import (
    ChartGenerator,
    DeveloperDashboard,
    HeatmapGenerator,
)


class TestVisualizationSystem(unittest.TestCase):
    """可視化システムのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'output_dir': self.temp_dir,
            'font_family': 'DejaVu Sans',
            'figure_size': (10, 8),
            'colormap': 'YlOrRd'
        }
        
        # テスト用データ
        self.developer_email = 'test@example.com'
        self.developer_data = {
            self.developer_email: {
                'stress_level': 0.7,
                'task_compatibility_stress': 0.6,
                'workload_stress': 0.8,
                'social_stress': 0.5,
                'temporal_stress': 0.4,
                'boiling_point_estimate': 0.9,
                'recent_acceptance_rate': 0.75,
                'avg_response_time': 2.5,
                'collaboration_quality': 0.8,
                'python_expertise': 0.9,
                'java_expertise': 0.6,
                'javascript_expertise': 0.4,
                'file_history': {
                    'src/main.py': 0.9,
                    'src/utils.py': 0.7,
                    'tests/test_main.py': 0.5
                },
                'stress_history': [
                    {'date': '2023-01-01', 'stress_level': 0.5},
                    {'date': '2023-01-02', 'stress_level': 0.6},
                    {'date': '2023-01-03', 'stress_level': 0.7}
                ]
            }
        }
        
        self.review_data = [
            {
                'reviewer_email': self.developer_email,
                'timestamp': '2023-01-01 10:00:00',
                'response_time_hours': 2.5
            },
            {
                'reviewer_email': self.developer_email,
                'timestamp': '2023-01-01 14:00:00',
                'response_time_hours': 1.5
            }
        ]
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 一時ディレクトリを削除
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_heatmap_generator_initialization(self):
        """ヒートマップ生成器の初期化テスト"""
        generator = HeatmapGenerator(self.config)
        
        self.assertEqual(generator.config, self.config)
        self.assertEqual(str(generator.output_dir), self.temp_dir)
        self.assertEqual(generator.colormap, 'YlOrRd')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_response_time_heatmap_generation(self, mock_close, mock_savefig):
        """レスポンス時間ヒートマップ生成テスト"""
        generator = HeatmapGenerator(self.config)
        
        result_path = generator.generate_response_time_heatmap(
            self.review_data, self.developer_email
        )
        
        # ファイルパスが返されることを確認
        self.assertTrue(result_path.endswith('.png'))
        self.assertIn('response_time_heatmap', result_path)
        
        # matplotlib関数が呼ばれることを確認
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_chart_generator_initialization(self):
        """チャート生成器の初期化テスト"""
        generator = ChartGenerator(self.config)
        
        self.assertEqual(generator.config, self.config)
        self.assertEqual(str(generator.output_dir), self.temp_dir)
        self.assertEqual(generator.color_palette, 'Set2')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_expertise_radar_chart_generation(self, mock_close, mock_savefig):
        """専門性レーダーチャート生成テスト"""
        generator = ChartGenerator(self.config)
        
        result_path = generator.generate_expertise_radar_chart(
            self.developer_data, self.developer_email
        )
        
        # ファイルパスが返されることを確認
        self.assertTrue(result_path.endswith('.png'))
        self.assertIn('expertise_radar', result_path)
        
        # matplotlib関数が呼ばれることを確認
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_dashboard_initialization(self):
        """ダッシュボードの初期化テスト"""
        dashboard = DeveloperDashboard(self.config)
        
        self.assertEqual(dashboard.config, self.config)
        self.assertEqual(str(dashboard.output_dir), self.temp_dir)
        self.assertIn('low', dashboard.stress_thresholds)
        self.assertIn('critical', dashboard.stress_thresholds)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_stress_dashboard_generation(self, mock_close, mock_savefig):
        """ストレスダッシュボード生成テスト"""
        dashboard = DeveloperDashboard(self.config)
        
        result_path = dashboard.generate_realtime_stress_dashboard(
            self.developer_data, self.developer_email
        )
        
        # ファイルパスが返されることを確認
        self.assertTrue(result_path.endswith('.png'))
        self.assertIn('stress_dashboard', result_path)
        
        # matplotlib関数が呼ばれることを確認
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_stress_data_extraction(self):
        """ストレスデータ抽出テスト"""
        dashboard = DeveloperDashboard(self.config)
        
        stress_data = dashboard._extract_current_stress_data(
            self.developer_data, self.developer_email
        )
        
        # 必要なキーが存在することを確認
        required_keys = [
            'total_stress', 'stress_factors', 'boiling_point',
            'stress_margin', 'risk_level', 'detailed_metrics'
        ]
        
        for key in required_keys:
            self.assertIn(key, stress_data)
        
        # 値の範囲チェック
        self.assertGreaterEqual(stress_data['total_stress'], 0.0)
        self.assertLessEqual(stress_data['total_stress'], 1.0)
        self.assertIn(stress_data['risk_level'], ['low', 'medium', 'high', 'critical'])
    
    def test_comprehensive_reports_generation(self):
        """包括的レポート生成テスト"""
        # ヒートマップレポート
        heatmap_generator = HeatmapGenerator(self.config)
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            heatmap_paths = heatmap_generator.generate_comprehensive_heatmap_report(
                self.review_data, self.developer_email
            )
            self.assertIsInstance(heatmap_paths, dict)
        
        # チャートレポート
        chart_generator = ChartGenerator(self.config)
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            chart_paths = chart_generator.generate_comprehensive_chart_report(
                self.developer_data, [], self.developer_email
            )
            self.assertIsInstance(chart_paths, dict)
        
        # ダッシュボードレポート
        dashboard = DeveloperDashboard(self.config)
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            dashboard_paths = dashboard.generate_comprehensive_dashboard_report(
                self.developer_data, [], self.developer_email
            )
            self.assertIsInstance(dashboard_paths, dict)
    
    def test_risk_level_calculation(self):
        """リスクレベル計算テスト"""
        dashboard = DeveloperDashboard(self.config)
        
        # 各リスクレベルのテスト
        self.assertEqual(dashboard._calculate_risk_level(0.95, 1.0), 'critical')
        self.assertEqual(dashboard._calculate_risk_level(0.85, 1.0), 'high')
        self.assertEqual(dashboard._calculate_risk_level(0.65, 1.0), 'medium')
        self.assertEqual(dashboard._calculate_risk_level(0.45, 1.0), 'low')
    
    def test_stress_color_mapping(self):
        """ストレス色マッピングテスト"""
        dashboard = DeveloperDashboard(self.config)
        
        # 各ストレスレベルに対応する色が返されることを確認
        colors = [
            dashboard._get_stress_color(0.2),  # low
            dashboard._get_stress_color(0.7),  # medium
            dashboard._get_stress_color(0.85), # high
            dashboard._get_stress_color(0.95)  # critical
        ]
        
        # すべて異なる色が返されることを確認
        self.assertEqual(len(set(colors)), 4)


if __name__ == '__main__':
    unittest.main()