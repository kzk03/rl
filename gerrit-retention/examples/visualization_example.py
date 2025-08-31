#!/usr/bin/env python3
"""
可視化システムの使用例

開発者定着予測システムの可視化機能の使用方法を示すサンプルスクリプト。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from gerrit_retention.visualization import (
    ChartGenerator,
    DeveloperDashboard,
    HeatmapGenerator,
)


def main():
    """メイン実行関数"""
    
    # 設定
    config = {
        'output_dir': 'outputs/visualizations',
        'font_family': 'DejaVu Sans',
        'figure_size': (12, 8),
        'colormap': 'YlOrRd',
        'color_palette': 'Set2',
        'stress_thresholds': {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
    }
    
    # サンプル開発者データ
    developer_email = 'sample_developer@example.com'
    
    # レビューデータ（ヒートマップ用）
    review_data = [
        {
            'reviewer_email': developer_email,
            'timestamp': '2023-01-01 09:00:00',
            'response_time_hours': 2.5
        },
        {
            'reviewer_email': developer_email,
            'timestamp': '2023-01-01 14:00:00',
            'response_time_hours': 1.5
        },
        {
            'reviewer_email': developer_email,
            'timestamp': '2023-01-02 10:30:00',
            'response_time_hours': 3.0
        },
        {
            'potential_reviewer': developer_email,
            'timestamp': '2023-01-01 11:00:00',
            'action': 'accept'
        },
        {
            'potential_reviewer': developer_email,
            'timestamp': '2023-01-01 15:00:00',
            'action': 'decline'
        }
    ]
    
    # 開発者プロファイルデータ
    developer_data = {
        developer_email: {
            # ストレス関連
            'stress_level': 0.7,
            'task_compatibility_stress': 0.6,
            'workload_stress': 0.8,
            'social_stress': 0.5,
            'temporal_stress': 0.4,
            'boiling_point_estimate': 0.9,
            
            # パフォーマンス指標
            'recent_acceptance_rate': 0.75,
            'avg_response_time': 2.5,
            'collaboration_quality': 0.8,
            
            # 専門性データ
            'python_expertise': 0.9,
            'java_expertise': 0.6,
            'javascript_expertise': 0.4,
            'frontend_expertise': 0.3,
            'backend_expertise': 0.8,
            'database_expertise': 0.7,
            'devops_expertise': 0.5,
            'testing_expertise': 0.6,
            
            # ファイル経験度
            'file_history': {
                'src/main.py': 0.9,
                'src/utils.py': 0.7,
                'src/models/predictor.py': 0.8,
                'tests/test_main.py': 0.5,
                'docs/README.md': 0.3
            },
            
            # ストレス履歴
            'stress_history': [
                {'date': '2023-01-01', 'stress_level': 0.5},
                {'date': '2023-01-02', 'stress_level': 0.6},
                {'date': '2023-01-03', 'stress_level': 0.65},
                {'date': '2023-01-04', 'stress_level': 0.7},
                {'date': '2023-01-05', 'stress_level': 0.7}
            ],
            
            # スキル成長履歴
            'skill_history': {
                '技術的専門性': [
                    {'date': '2023-01-01', 'value': 0.7},
                    {'date': '2023-01-15', 'value': 0.75},
                    {'date': '2023-02-01', 'value': 0.8}
                ],
                'レビュー品質': [
                    {'date': '2023-01-01', 'value': 0.6},
                    {'date': '2023-01-15', 'value': 0.65},
                    {'date': '2023-02-01', 'value': 0.7}
                ]
            }
        }
    }
    
    # 協力関係データ
    collaboration_data = [
        {
            'developer1': developer_email,
            'developer2': 'colleague1@example.com',
            'timestamp': '2023-01-01 10:00:00',
            'interaction_count': 5
        },
        {
            'developer1': developer_email,
            'developer2': 'colleague2@example.com',
            'timestamp': '2023-01-01 14:00:00',
            'interaction_count': 3
        }
    ]
    
    print("=== 開発者定着予測システム 可視化デモ ===\n")
    
    # 1. ヒートマップ生成
    print("1. ヒートマップ生成中...")
    heatmap_generator = HeatmapGenerator(config)
    
    try:
        # レスポンス時間ヒートマップ
        response_heatmap = heatmap_generator.generate_response_time_heatmap(
            review_data, developer_email
        )
        if response_heatmap:
            print(f"   ✓ レスポンス時間ヒートマップ: {response_heatmap}")
        
        # 受諾率ヒートマップ
        acceptance_heatmap = heatmap_generator.generate_acceptance_rate_heatmap(
            review_data, developer_email
        )
        if acceptance_heatmap:
            print(f"   ✓ 受諾率ヒートマップ: {acceptance_heatmap}")
        
        # 包括的ヒートマップレポート
        heatmap_report = heatmap_generator.generate_comprehensive_heatmap_report(
            review_data, developer_email
        )
        print(f"   ✓ 包括的ヒートマップレポート: {len(heatmap_report)}個のファイル生成")
        
    except Exception as e:
        print(f"   ✗ ヒートマップ生成エラー: {e}")
    
    # 2. チャート生成
    print("\n2. チャート生成中...")
    chart_generator = ChartGenerator(config)
    
    try:
        # 専門性レーダーチャート
        expertise_radar = chart_generator.generate_expertise_radar_chart(
            developer_data, developer_email
        )
        if expertise_radar:
            print(f"   ✓ 専門性レーダーチャート: {expertise_radar}")
        
        # ファイル経験度マップ
        file_experience_map = chart_generator.generate_file_experience_map(
            developer_data, developer_email
        )
        if file_experience_map:
            print(f"   ✓ ファイル経験度マップ: {file_experience_map}")
        
        # スキル成長チャート
        skill_progression = chart_generator.generate_skill_progression_chart(
            developer_data, developer_email
        )
        if skill_progression:
            print(f"   ✓ スキル成長チャート: {skill_progression}")
        
        # 協力関係ネットワークチャート
        collaboration_network = chart_generator.generate_collaboration_network_chart(
            collaboration_data, developer_email
        )
        if collaboration_network:
            print(f"   ✓ 協力関係ネットワークチャート: {collaboration_network}")
        
        # 包括的チャートレポート
        chart_report = chart_generator.generate_comprehensive_chart_report(
            developer_data, collaboration_data, developer_email
        )
        print(f"   ✓ 包括的チャートレポート: {len(chart_report)}個のファイル生成")
        
    except Exception as e:
        print(f"   ✗ チャート生成エラー: {e}")
    
    # 3. ダッシュボード生成
    print("\n3. ダッシュボード生成中...")
    dashboard = DeveloperDashboard(config)
    
    try:
        # リアルタイムストレスダッシュボード
        stress_dashboard = dashboard.generate_realtime_stress_dashboard(
            developer_data, developer_email
        )
        if stress_dashboard:
            print(f"   ✓ リアルタイムストレスダッシュボード: {stress_dashboard}")
        
        # チーム概要ダッシュボード（サンプルデータ）
        team_data = [developer_data[developer_email]]  # 簡略化
        team_dashboard = dashboard.generate_team_overview_dashboard(team_data)
        if team_dashboard:
            print(f"   ✓ チーム概要ダッシュボード: {team_dashboard}")
        
        # 包括的ダッシュボードレポート
        dashboard_report = dashboard.generate_comprehensive_dashboard_report(
            developer_data, team_data, developer_email
        )
        print(f"   ✓ 包括的ダッシュボードレポート: {len(dashboard_report)}個のファイル生成")
        
    except Exception as e:
        print(f"   ✗ ダッシュボード生成エラー: {e}")
    
    print("\n=== 可視化デモ完了 ===")
    print(f"生成されたファイルは '{config['output_dir']}' ディレクトリに保存されました。")


if __name__ == '__main__':
    main()