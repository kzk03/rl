#!/usr/bin/env python3
"""
最終的な強化学習デモンストレーション
改良されたモデルを使用した完全なタスク割り当て最適化システム
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.append('scripts')

from gerrit_retention.utils.logger import get_logger
from rl_task_optimizer import RLTaskOptimizer

logger = get_logger(__name__)

def comprehensive_rl_demonstration():
    """包括的な強化学習デモンストレーション"""
    logger.info("=== 最終強化学習システムデモ ===")
    
    # データの読み込み
    with open('data/processed/unified/all_developers.json', 'r') as f:
        developers_data = json.load(f)
    
    with open('data/processed/unified/all_reviews.json', 'r') as f:
        reviews_data = json.load(f)
    
    # 人間の開発者のみを抽出
    human_developers = []
    for dev in developers_data:
        name_lower = dev['name'].lower()
        email_lower = dev['developer_id'].lower()
        
        is_bot = any(keyword in name_lower for keyword in ['bot', 'robot', 'lint', 'presubmit', 'treehugger'])
        is_bot = is_bot or any(keyword in email_lower for keyword in ['bot', 'robot', 'system.gserviceaccount', 'presubmit'])
        
        if not is_bot and (dev['changes_authored'] > 0 or dev['changes_reviewed'] > 5):
            human_developers.append(dev)
    
    # 上位開発者を選択
    top_developers = sorted(
        human_developers, 
        key=lambda x: x['changes_authored'] + x['changes_reviewed'], 
        reverse=True
    )[:10]
    
    # 全てのタスクを使用
    all_tasks = reviews_data[:10]
    
    print(f"\\n🎯 最終デモンストレーション")
    print("=" * 70)
    print(f"📊 データ概要:")
    print(f"   総開発者数: {len(developers_data)}名")
    print(f"   人間開発者: {len(human_developers)}名")
    print(f"   選択開発者: {len(top_developers)}名")
    print(f"   総タスク数: {len(reviews_data)}件")
    print(f"   選択タスク: {len(all_tasks)}件")
    
    # 改良モデルを使用した最適化器
    optimizer = RLTaskOptimizer(model_path='models/improved_ppo_agent.pth')
    
    # 1. 個別推奨のデモ
    print(f"\\n🔍 1. 個別推奨分析")
    print("-" * 50)
    
    sample_developer = top_developers[0]
    sample_task = all_tasks[0]
    
    print(f"開発者: {sample_developer['name']} ({sample_developer['developer_id']})")
    print(f"  活動実績: 作成{sample_developer['changes_authored']}件, レビュー{sample_developer['changes_reviewed']}件")
    print(f"  プロジェクト: {len(sample_developer['projects'])}個")
    
    print(f"\\nタスク: {sample_task.get('change_id', 'unknown')}")
    print(f"  サイズ: {sample_task.get('lines_added', 0)}行追加, {sample_task.get('files_changed', 0)}ファイル変更")
    print(f"  状態: {sample_task.get('status', 'unknown')}")
    
    # 異なるコンテキストでの推奨
    contexts = [
        {'stress_level': -0.5, 'workload': -0.3, 'expertise_match': 0.8, 'name': '理想的条件'},
        {'stress_level': 0.7, 'workload': 0.8, 'expertise_match': 0.2, 'name': '高負荷条件'},
        {'stress_level': 0.0, 'workload': 0.0, 'expertise_match': 0.5, 'name': '標準条件'}
    ]
    
    print(f"\\n📋 コンテキスト別推奨:")
    for ctx in contexts:
        recommendation = optimizer.get_recommendation(sample_developer, sample_task, ctx)
        
        print(f"\\n  {ctx['name']}:")
        print(f"    推奨: {recommendation.action} (信頼度: {recommendation.confidence:.1%})")
        print(f"    期待価値: {recommendation.expected_value:.2f}")
        print(f"    確率分布: assign={recommendation.action_probabilities['assign']:.1%}, "
              f"reject={recommendation.action_probabilities['reject']:.1%}, "
              f"defer={recommendation.action_probabilities['defer']:.1%}")
        print(f"    主な理由: {recommendation.reasoning[0] if recommendation.reasoning else 'なし'}")
    
    # 2. チーム最適化のデモ
    print(f"\\n🚀 2. チーム全体最適化")
    print("-" * 50)
    
    optimization_result = optimizer.optimize_team_assignments(
        top_developers, all_tasks, max_assignments_per_developer=3
    )
    
    assignments = optimization_result['assignments']
    stats = optimization_result['statistics']
    
    print(f"\\n📊 最適化結果:")
    print(f"   総タスク数: {stats['total_tasks']}件")
    print(f"   割り当て成功: {stats['assigned_tasks']}件")
    print(f"   成功率: {stats['assignment_rate']:.1%}")
    print(f"   平均信頼度: {stats['average_confidence']:.1%}")
    
    print(f"\\n👥 開発者別割り当て:")
    for dev_id, dev_stats in stats['developer_stats'].items():
        print(f"   {dev_stats['name']}: {dev_stats['assignments']}件 "
              f"(平均信頼度: {dev_stats['avg_confidence']:.1%})")
    
    # 3. 詳細な割り当て結果
    print(f"\\n📋 3. 詳細割り当て結果")
    print("-" * 50)
    
    for i, assignment in enumerate(assignments[:5], 1):  # 上位5件を表示
        print(f"\\n  {i}. {assignment.task_id}")
        print(f"     → {assignment.developer_name}")
        print(f"     推奨: {assignment.action} (信頼度: {assignment.confidence:.1%})")
        print(f"     期待価値: {assignment.expected_value:.2f}")
        print(f"     理由:")
        for reason in assignment.reasoning[:2]:  # 主要な理由2つ
            print(f"       • {reason}")
    
    # 4. パフォーマンス分析
    print(f"\\n📈 4. システムパフォーマンス分析")
    print("-" * 50)
    
    # 信頼度分布
    confidences = [a.confidence for a in assignments]
    if confidences:
        print(f"   信頼度統計:")
        print(f"     最高: {max(confidences):.1%}")
        print(f"     最低: {min(confidences):.1%}")
        print(f"     平均: {np.mean(confidences):.1%}")
        print(f"     標準偏差: {np.std(confidences):.1%}")
    
    # 期待価値分析
    values = [a.expected_value for a in assignments]
    if values:
        print(f"\\n   期待価値統計:")
        print(f"     最高: {max(values):.2f}")
        print(f"     最低: {min(values):.2f}")
        print(f"     平均: {np.mean(values):.2f}")
    
    # 行動分布
    action_counts = {'assign': 0, 'reject': 0, 'defer': 0}
    for assignment in assignments:
        action_counts[assignment.action] += 1
    
    print(f"\\n   行動分布:")
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        for action, count in action_counts.items():
            percentage = count / total_actions * 100
            print(f"     {action}: {count}件 ({percentage:.1f}%)")
    
    # 5. 実用性評価
    print(f"\\n🎯 5. 実用性評価")
    print("-" * 50)
    
    # 負荷分散の評価
    developer_loads = {}
    for assignment in assignments:
        dev_id = assignment.developer_id
        developer_loads[dev_id] = developer_loads.get(dev_id, 0) + 1
    
    if developer_loads:
        load_values = list(developer_loads.values())
        load_balance_score = 1 - (np.std(load_values) / (np.mean(load_values) + 1e-8))
        print(f"   負荷分散スコア: {load_balance_score:.2f} (1.0が完全均等)")
    
    # 高信頼度割り当ての割合
    high_confidence_assignments = [a for a in assignments if a.confidence > 0.7]
    high_conf_rate = len(high_confidence_assignments) / len(assignments) if assignments else 0
    print(f"   高信頼度割り当て率: {high_conf_rate:.1%} (信頼度>70%)")
    
    # 経験豊富な開発者への割り当て率
    experienced_assignments = []
    for assignment in assignments:
        dev = next(d for d in top_developers if d['developer_id'] == assignment.developer_id)
        if (dev['changes_authored'] + dev['changes_reviewed']) > 50:
            experienced_assignments.append(assignment)
    
    exp_rate = len(experienced_assignments) / len(assignments) if assignments else 0
    print(f"   経験豊富な開発者への割り当て率: {exp_rate:.1%}")
    
    # 6. 推奨事項
    print(f"\\n💡 6. システム推奨事項")
    print("-" * 50)
    
    recommendations = []
    
    if stats['assignment_rate'] > 0.8:
        recommendations.append("✅ 高い割り当て成功率を達成")
    elif stats['assignment_rate'] > 0.5:
        recommendations.append("⚠️  割り当て成功率は中程度 - 条件調整を検討")
    else:
        recommendations.append("❌ 割り当て成功率が低い - モデル再訓練を推奨")
    
    if stats['average_confidence'] > 0.7:
        recommendations.append("✅ 高い平均信頼度を維持")
    else:
        recommendations.append("⚠️  信頼度向上のため特徴量エンジニアリングを検討")
    
    if load_balance_score > 0.8:
        recommendations.append("✅ 良好な負荷分散を実現")
    else:
        recommendations.append("⚠️  負荷分散の改善が必要")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print(f"\\n" + "=" * 70)
    print(f"🎉 強化学習システムデモンストレーション完了！")
    print(f"   実際のGerritデータで動作する完全なタスク割り当て最適化システムが稼働中です。")
    
    logger.info("最終デモンストレーション完了")
    
    return optimization_result


if __name__ == "__main__":
    comprehensive_rl_demonstration()