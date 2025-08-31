#!/usr/bin/env python3
"""
A/Bテストシステム使用例

このスクリプトは、開発者定着予測システムのA/Bテスト機能の使用方法を示す。
実験設計から統計分析まで、完全なワークフローを実行する。
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.evaluation.ab_testing import (
    ExperimentDesigner,
    ParticipantAllocator,
    StatisticalAnalyzer,
)


def create_sample_experiment():
    """サンプル実験を作成"""
    print("=== A/Bテスト実験設計例 ===")
    
    # 実験設計器を初期化
    designer = ExperimentDesigner()
    
    # 定着戦略の比較実験を設計
    strategies = [
        {
            'name': 'Current Strategy',
            'description': '現在の推薦戦略（ベースライン）',
            'stress_weight': 0.3,
            'expertise_weight': 0.4,
            'workload_weight': 0.3,
            'similarity_threshold': 0.7
        },
        {
            'name': 'Stress-Optimized Strategy',
            'description': 'ストレス軽減に最適化された戦略',
            'stress_weight': 0.5,
            'expertise_weight': 0.3,
            'workload_weight': 0.2,
            'similarity_threshold': 0.8
        },
        {
            'name': 'Expertise-Focused Strategy',
            'description': '専門性マッチングに重点を置いた戦略',
            'stress_weight': 0.2,
            'expertise_weight': 0.6,
            'workload_weight': 0.2,
            'similarity_threshold': 0.75
        }
    ]
    
    # 実験設計
    experiment_config = designer.design_retention_strategy_experiment(
        experiment_name="Developer Retention Strategy Comparison",
        strategies=strategies,
        duration_days=30,
        target_participants=1500
    )
    
    print(f"実験ID: {experiment_config.experiment_id}")
    print(f"実験名: {experiment_config.name}")
    print(f"バリアント数: {len(experiment_config.variants)}")
    print(f"メトリクス数: {len(experiment_config.metrics)}")
    print(f"必要サンプルサイズ: {experiment_config.minimum_sample_size}")
    
    # 実験設定を保存
    config_file = designer.save_experiment_config(experiment_config)
    print(f"設定ファイル: {config_file}")
    
    return experiment_config


def simulate_experiment_data(experiment_config):
    """実験データをシミュレート"""
    print("\n=== 実験データシミュレーション ===")
    
    # 参加者割り当て器を初期化
    allocator = ParticipantAllocator(experiment_config)
    
    # シミュレーションデータ生成
    np.random.seed(42)  # 再現性のため
    
    participants_data = []
    
    # 各バリアントの真の効果を設定
    variant_effects = {
        'variant_0': {'retention_rate': 0.70, 'review_acceptance_rate': 0.75, 'stress_level': 0.45},
        'variant_1': {'retention_rate': 0.75, 'review_acceptance_rate': 0.80, 'stress_level': 0.35},  # ストレス最適化
        'variant_2': {'retention_rate': 0.73, 'review_acceptance_rate': 0.78, 'stress_level': 0.40}   # 専門性重視
    }
    
    total_participants = 1200
    
    for i in range(total_participants):
        participant_id = f"developer_{i:04d}"
        
        # 参加者属性をランダム生成
        attributes = {
            'expertise_level': np.random.uniform(0.2, 0.9),
            'project_type': np.random.choice(['web', 'mobile', 'backend', 'ml']),
            'activity_days': np.random.randint(10, 365),
            'stress_level': np.random.uniform(0.1, 0.8)
        }
        
        # バリアントに割り当て
        variant_id = allocator.allocate_participant(participant_id, attributes)
        
        if variant_id is None:
            continue  # 除外された参加者
            
        # バリアントの効果に基づいてメトリクス値を生成
        effects = variant_effects[variant_id]
        
        # ノイズを追加してリアルなデータを生成
        retention_rate = np.random.binomial(1, effects['retention_rate'])
        review_acceptance_rate = np.clip(
            np.random.normal(effects['review_acceptance_rate'], 0.1), 0, 1
        )
        stress_level = np.clip(
            np.random.normal(effects['stress_level'], 0.1), 0, 1
        )
        task_completion_rate = np.clip(
            np.random.normal(0.85 + (effects['retention_rate'] - 0.7) * 0.2, 0.05), 0, 1
        )
        collaboration_score = np.clip(
            np.random.normal(0.6 + (effects['review_acceptance_rate'] - 0.75) * 0.4, 0.1), 0, 1
        )
        
        # データレコード作成
        record = {
            'participant_id': participant_id,
            'variant_id': variant_id,
            'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
            'retention_rate': retention_rate,
            'review_acceptance_rate': review_acceptance_rate,
            'stress_level': stress_level,
            'task_completion_rate': task_completion_rate,
            'collaboration_score': collaboration_score,
            **attributes
        }
        
        participants_data.append(record)
    
    # DataFrameに変換
    experiment_data = pd.DataFrame(participants_data)
    
    print(f"生成された参加者数: {len(experiment_data)}")
    print(f"バリアント別分布:")
    print(experiment_data['variant_id'].value_counts())
    
    # 割り当てサマリー表示
    allocation_summary = allocator.get_allocation_summary()
    print(f"割り当てサマリー: {allocation_summary}")
    
    return experiment_data


def analyze_experiment_results(experiment_config, experiment_data):
    """実験結果を分析"""
    print("\n=== 統計分析実行 ===")
    
    # 統計分析器を初期化
    analyzer = StatisticalAnalyzer()
    
    # 分析実行
    analysis_result = analyzer.analyze_experiment(experiment_config, experiment_data)
    
    # 結果表示
    print(f"分析完了: {analysis_result.experiment_id}")
    print(f"総参加者数: {analysis_result.total_participants:,}")
    print(f"実験期間: {analysis_result.experiment_duration_days}日")
    
    print(f"\n=== 全体結論 ===")
    print(analysis_result.overall_conclusion)
    
    print(f"\n=== バリアント性能 ===")
    for performance in analysis_result.variant_performances:
        print(f"\n{performance.variant_name} (n={performance.sample_size}):")
        for metric_id, value in performance.metric_values.items():
            ci = performance.confidence_intervals.get(metric_id, (value, value))
            print(f"  {metric_id}: {value:.4f} [95%CI: {ci[0]:.4f}, {ci[1]:.4f}]")
    
    print(f"\n=== 統計検定結果 ===")
    for test in analysis_result.statistical_tests:
        print(f"\n{test.metric_id}:")
        print(f"  検定方法: {test.test_type}")
        print(f"  p値: {test.p_value:.6f}")
        print(f"  効果サイズ: {test.effect_size:.4f}")
        print(f"  結果: {test.result.value}")
        print(f"  検出力: {test.power:.3f}")
        print(f"  解釈: {test.interpretation}")
    
    print(f"\n=== 推奨事項 ===")
    for i, recommendation in enumerate(analysis_result.recommendations, 1):
        print(f"{i}. {recommendation}")
    
    if analysis_result.data_quality_issues:
        print(f"\n=== データ品質の問題 ===")
        for issue in analysis_result.data_quality_issues:
            print(f"- {issue}")
    
    # 結果保存
    results_dir = Path("example_results")
    results_dir.mkdir(exist_ok=True)
    
    # JSON結果保存
    json_file = analyzer.save_analysis_results(
        analysis_result, results_dir / "analysis_results.json"
    )
    print(f"\nJSON結果保存: {json_file}")
    
    # HTMLレポート生成
    html_file = analyzer.generate_analysis_report(
        analysis_result, results_dir / "analysis_report.html"
    )
    print(f"HTMLレポート生成: {html_file}")
    
    # 可視化プロット作成
    plot_files = analyzer.create_visualization_plots(
        analysis_result, experiment_data, results_dir / "plots"
    )
    print(f"可視化プロット: {len(plot_files)}個作成")
    for plot_file in plot_files:
        print(f"  - {plot_file}")
    
    return analysis_result


def demonstrate_review_strategy_experiment():
    """レビュー戦略実験のデモンストレーション"""
    print("\n\n=== レビュー戦略A/Bテスト例 ===")
    
    designer = ExperimentDesigner()
    
    # レビュー戦略定義
    review_strategies = [
        {
            'name': 'Current Review Assignment',
            'description': '現在のレビュー割り当て戦略',
            'similarity_threshold': 0.7,
            'workload_factor': 0.3,
            'expertise_matching_weight': 0.4
        },
        {
            'name': 'High-Similarity Assignment',
            'description': '高類似度優先の割り当て戦略',
            'similarity_threshold': 0.85,
            'workload_factor': 0.2,
            'expertise_matching_weight': 0.6
        }
    ]
    
    # レビュー戦略実験設計
    review_experiment = designer.design_review_strategy_experiment(
        experiment_name="Review Assignment Strategy Comparison",
        review_strategies=review_strategies,
        duration_days=21,
        target_participants=800
    )
    
    print(f"レビュー実験ID: {review_experiment.experiment_id}")
    print(f"実験名: {review_experiment.name}")
    print(f"プライマリメトリクス: {[m.name for m in review_experiment.metrics if m.primary][0]}")
    print(f"セカンダリメトリクス: {[m.name for m in review_experiment.metrics if not m.primary]}")
    
    # 設定保存
    config_file = designer.save_experiment_config(review_experiment)
    print(f"レビュー実験設定保存: {config_file}")


def main():
    """メイン実行関数"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Gerrit開発者定着予測システム A/Bテスト使用例")
    print("=" * 60)
    
    try:
        # 1. 実験設計
        experiment_config = create_sample_experiment()
        
        # 2. 実験データシミュレーション
        experiment_data = simulate_experiment_data(experiment_config)
        
        # 3. 統計分析
        analysis_result = analyze_experiment_results(experiment_config, experiment_data)
        
        # 4. レビュー戦略実験のデモ
        demonstrate_review_strategy_experiment()
        
        print("\n" + "=" * 60)
        print("A/Bテストシステムのデモンストレーション完了！")
        print("生成されたファイルを確認してください:")
        print("- experiments/ : 実験設定ファイル")
        print("- example_results/ : 分析結果とレポート")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()