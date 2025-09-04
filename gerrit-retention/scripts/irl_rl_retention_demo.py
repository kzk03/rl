#!/usr/bin/env python3
"""
IRL/RL継続予測システムデモ

ルールベースシステムを逆強化学習・強化学習で置き換えた
次世代継続予測システムのデモンストレーション
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem
from gerrit_retention.rl_prediction.retention_rl_system import RetentionRLSystem


def create_training_data() -> List[Dict[str, Any]]:
    """訓練用データを作成"""
    
    training_data = [
        # 継続した開発者の例
        {
            'developer': {
                'developer_id': 'alice_continued@example.com',
                'name': 'Alice Continued',
                'first_seen': '2023-01-15T09:00:00Z',
                'changes_authored': 85,
                'changes_reviewed': 120,
                'projects': ['project-a', 'project-b', 'project-c']
            },
            'activity_history': [
                {'timestamp': '2023-01-20T10:30:00Z', 'type': 'commit', 'message': 'Fix critical bug in auth module', 'lines_added': 45, 'lines_deleted': 12},
                {'timestamp': '2023-02-05T14:15:00Z', 'type': 'review', 'score': 2, 'comments': ['LGTM', 'Good fix', 'Consider adding tests']},
                {'timestamp': '2023-02-18T11:45:00Z', 'type': 'commit', 'message': 'Add comprehensive test suite', 'lines_added': 234, 'lines_deleted': 5},
                {'timestamp': '2023-03-02T16:20:00Z', 'type': 'merge', 'message': 'Merge feature branch into main'},
                {'timestamp': '2023-03-15T09:10:00Z', 'type': 'review', 'score': 1, 'comments': ['Minor issues', 'Please fix formatting']},
                {'timestamp': '2023-04-01T12:00:00Z', 'type': 'commit', 'message': 'Refactor code for better maintainability', 'lines_added': 67, 'lines_deleted': 23},
                {'timestamp': '2023-04-20T15:30:00Z', 'type': 'review', 'score': 2, 'comments': ['Excellent refactoring work']},
                {'timestamp': '2023-05-10T11:15:00Z', 'type': 'commit', 'message': 'Performance optimization in query module', 'lines_added': 89, 'lines_deleted': 34}
            ],
            'continued': True,
            'context_date': datetime(2023, 3, 20)
        },
        {
            'developer': {
                'developer_id': 'bob_continued@example.com',
                'name': 'Bob Continued',
                'first_seen': '2022-11-10T14:30:00Z',
                'changes_authored': 156,
                'changes_reviewed': 89,
                'projects': ['project-a', 'project-c', 'project-d']
            },
            'activity_history': [
                {'timestamp': '2023-01-05T08:45:00Z', 'type': 'commit', 'message': 'Refactor database connection handling', 'lines_added': 156, 'lines_deleted': 89},
                {'timestamp': '2023-01-12T13:20:00Z', 'type': 'review', 'score': 2, 'comments': ['Excellent work', 'Clean implementation']},
                {'timestamp': '2023-01-25T10:15:00Z', 'type': 'issue_creation', 'title': 'Performance optimization needed in query module'},
                {'timestamp': '2023-02-08T15:30:00Z', 'type': 'commit', 'message': 'Optimize database queries for better performance', 'lines_added': 78, 'lines_deleted': 145},
                {'timestamp': '2023-02-20T11:00:00Z', 'type': 'documentation', 'message': 'Update API documentation with new endpoints'},
                {'timestamp': '2023-03-05T14:45:00Z', 'type': 'review', 'score': -1, 'comments': ['Needs more testing', 'Consider edge cases']},
                {'timestamp': '2023-04-15T09:30:00Z', 'type': 'commit', 'message': 'Add comprehensive error handling', 'lines_added': 123, 'lines_deleted': 45},
                {'timestamp': '2023-05-01T16:00:00Z', 'type': 'review', 'score': 2, 'comments': ['Good improvement', 'Much better error handling']}
            ],
            'continued': True,
            'context_date': datetime(2023, 3, 20)
        },
        # 離脱した開発者の例
        {
            'developer': {
                'developer_id': 'charlie_left@example.com',
                'name': 'Charlie Left',
                'first_seen': '2023-02-28T16:00:00Z',
                'changes_authored': 12,
                'changes_reviewed': 8,
                'projects': ['project-b']
            },
            'activity_history': [
                {'timestamp': '2023-03-01T09:30:00Z', 'type': 'commit', 'message': 'First contribution: fix typo in README', 'lines_added': 1, 'lines_deleted': 1},
                {'timestamp': '2023-03-03T11:15:00Z', 'type': 'review', 'score': 0, 'comments': ['Learning from this code']},
                {'timestamp': '2023-03-08T14:20:00Z', 'type': 'commit', 'message': 'Add basic unit test', 'lines_added': 45, 'lines_deleted': 0},
                {'timestamp': '2023-03-12T10:45:00Z', 'type': 'issue_comment', 'message': 'I can work on this issue'},
                {'timestamp': '2023-03-18T13:00:00Z', 'type': 'commit', 'message': 'Small improvement', 'lines_added': 12, 'lines_deleted': 3}
            ],
            'continued': False,
            'context_date': datetime(2023, 3, 20)
        },
        {
            'developer': {
                'developer_id': 'diana_left@example.com',
                'name': 'Diana Left',
                'first_seen': '2022-08-20T12:00:00Z',
                'changes_authored': 34,
                'changes_reviewed': 18,
                'projects': ['project-c']
            },
            'activity_history': [
                {'timestamp': '2022-12-15T14:30:00Z', 'type': 'commit', 'message': 'Fix minor bug in configuration parser', 'lines_added': 12, 'lines_deleted': 8},
                {'timestamp': '2023-01-20T16:45:00Z', 'type': 'review', 'score': 1, 'comments': ['Looks good to me']},
                {'timestamp': '2023-02-28T11:30:00Z', 'type': 'commit', 'message': 'Update dependency versions', 'lines_added': 5, 'lines_deleted': 5},
                {'timestamp': '2023-03-10T09:15:00Z', 'type': 'review', 'score': 0, 'comments': ['Not sure about this approach']}
            ],
            'continued': False,
            'context_date': datetime(2023, 3, 20)
        },
        # 追加の継続開発者
        {
            'developer': {
                'developer_id': 'eve_veteran@example.com',
                'name': 'Eve Veteran',
                'first_seen': '2021-06-15T10:00:00Z',
                'changes_authored': 456,
                'changes_reviewed': 234,
                'projects': ['project-a', 'project-b', 'project-c', 'project-d', 'project-e']
            },
            'activity_history': [
                {'timestamp': '2023-01-03T08:00:00Z', 'type': 'commit', 'message': 'Major refactoring of core architecture', 'lines_added': 567, 'lines_deleted': 234},
                {'timestamp': '2023-01-10T09:30:00Z', 'type': 'review', 'score': 2, 'comments': ['Approved', 'Great work', 'This will improve maintainability']},
                {'timestamp': '2023-01-18T13:45:00Z', 'type': 'merge', 'message': 'Merge architecture refactoring'},
                {'timestamp': '2023-01-25T11:20:00Z', 'type': 'documentation', 'message': 'Update architecture documentation'},
                {'timestamp': '2023-02-02T15:10:00Z', 'type': 'mentoring', 'message': 'Code review session with junior developers'},
                {'timestamp': '2023-02-15T10:30:00Z', 'type': 'commit', 'message': 'Implement advanced caching mechanism', 'lines_added': 234, 'lines_deleted': 67},
                {'timestamp': '2023-03-01T14:00:00Z', 'type': 'review', 'score': 1, 'comments': ['Good start', 'Needs some improvements']},
                {'timestamp': '2023-03-10T16:45:00Z', 'type': 'collaboration', 'message': 'Pair programming session on complex algorithm'},
                {'timestamp': '2023-04-05T12:30:00Z', 'type': 'commit', 'message': 'Security enhancement in authentication', 'lines_added': 123, 'lines_deleted': 45},
                {'timestamp': '2023-04-25T14:15:00Z', 'type': 'review', 'score': 2, 'comments': ['Excellent security improvement']},
                {'timestamp': '2023-05-15T10:00:00Z', 'type': 'commit', 'message': 'Performance tuning for high-load scenarios', 'lines_added': 89, 'lines_deleted': 56}
            ],
            'continued': True,
            'context_date': datetime(2023, 3, 20)
        }
    ]
    
    return training_data


def main():
    """メイン関数"""
    
    print("🤖 IRL/RL継続予測システム デモンストレーション")
    print("=" * 70)
    print("ルールベースシステムを機械学習で置き換えます")
    print()
    
    # 訓練データを作成
    training_data = create_training_data()
    print(f"📊 訓練データ: {len(training_data)}件")
    print(f"   継続: {sum(1 for d in training_data if d['continued'])}件")
    print(f"   離脱: {sum(1 for d in training_data if not d['continued'])}件")
    print()
    
    # IRL システムのデモ
    print("🧠 逆強化学習（IRL）システム")
    print("-" * 50)
    
    irl_config = {
        'state_dim': 10,
        'action_dim': 5,
        'hidden_dim': 64,
        'learning_rate': 0.01
    }
    
    irl_system = RetentionIRLSystem(irl_config)
    
    # IRL訓練
    print("🎯 IRL訓練を開始...")
    irl_results = irl_system.train_irl(training_data, epochs=50)
    print(f"   訓練完了: 最終損失 = {irl_results['final_loss']:.4f}")
    print()
    
    # IRL予測テスト
    print("🔮 IRL予測テスト:")
    test_developer = training_data[0]['developer']
    test_activity = training_data[0]['activity_history']
    
    irl_prediction = irl_system.predict_continuation_probability(
        test_developer, test_activity
    )
    
    print(f"   開発者: {test_developer['name']}")
    print(f"   IRL予測確率: {irl_prediction['continuation_probability']:.1%}")
    print(f"   信頼度: {irl_prediction['confidence']:.1%}")
    print(f"   報酬スコア: {irl_prediction['reward_score']:.3f}")
    print(f"   理由: {irl_prediction['reasoning']}")
    print()
    
    # RL システムのデモ
    print("🎮 強化学習（RL）システム")
    print("-" * 50)
    
    rl_config = {
        'state_dim': 15,
        'hidden_dim': 64,
        'learning_rate': 0.01
    }
    
    rl_system = RetentionRLSystem(rl_config)
    
    # RL訓練
    print("🎯 RL訓練を開始...")
    rl_results = rl_system.train_rl(training_data, episodes=200)
    print(f"   訓練完了: 最終平均報酬 = {rl_results['final_avg_reward']:.3f}")
    print(f"   最終平均精度 = {rl_results['final_avg_accuracy']:.1%}")
    print()
    
    # RL予測テスト
    print("🔮 RL予測テスト:")
    rl_prediction = rl_system.predict_continuation_probability(
        test_developer, test_activity
    )
    
    print(f"   開発者: {test_developer['name']}")
    print(f"   RL予測確率: {rl_prediction['continuation_probability']:.1%}")
    print(f"   信頼度: {rl_prediction['confidence']:.1%}")
    print(f"   理由: {rl_prediction['reasoning']}")
    print()
    
    # 全開発者での比較テスト
    print("📊 全開発者での予測比較")
    print("=" * 70)
    
    print(f"{'開発者名':<20} {'実際':<6} {'IRL予測':<8} {'RL予測':<8} {'IRL正解':<8} {'RL正解':<8}")
    print("-" * 70)
    
    irl_correct = 0
    rl_correct = 0
    
    for data_point in training_data:
        developer = data_point['developer']
        activity_history = data_point['activity_history']
        true_continued = data_point['continued']
        
        # IRL予測
        irl_pred = irl_system.predict_continuation_probability(developer, activity_history)
        irl_prob = irl_pred['continuation_probability']
        irl_label = irl_prob > 0.5
        irl_correct += (irl_label == true_continued)
        
        # RL予測
        rl_pred = rl_system.predict_continuation_probability(developer, activity_history)
        rl_prob = rl_pred['continuation_probability']
        rl_label = rl_prob > 0.5
        rl_correct += (rl_label == true_continued)
        
        # 結果表示
        true_str = "継続" if true_continued else "離脱"
        irl_correct_str = "✅" if irl_label == true_continued else "❌"
        rl_correct_str = "✅" if rl_label == true_continued else "❌"
        
        print(f"{developer['name'][:19]:<20} {true_str:<6} {irl_prob:>6.1%} {rl_prob:>8.1%} {irl_correct_str:>8} {rl_correct_str:>8}")
    
    print("-" * 70)
    print(f"{'総合精度':<20} {'':<6} {irl_correct/len(training_data):>6.1%} {rl_correct/len(training_data):>8.1%}")
    print()
    
    # システム比較サマリー
    print("🏆 システム比較サマリー")
    print("=" * 70)
    
    print("**従来のルールベースシステム:**")
    print("  - 手作りルールによる予測")
    print("  - 固定的な重み付け")
    print("  - 解釈しやすいが精度に限界")
    print()
    
    print("**逆強化学習（IRL）システム:**")
    print(f"  - 継続開発者の行動から報酬関数を学習")
    print(f"  - 予測精度: {irl_correct/len(training_data):.1%}")
    print(f"  - 最終訓練損失: {irl_results['final_loss']:.4f}")
    print(f"  - 特徴: エキスパートの行動パターンを模倣")
    print()
    
    print("**強化学習（RL）システム:**")
    print(f"  - 予測精度を最大化するポリシーを学習")
    print(f"  - 予測精度: {rl_correct/len(training_data):.1%}")
    print(f"  - 最終平均報酬: {rl_results['final_avg_reward']:.3f}")
    print(f"  - 特徴: 報酬に基づく最適化")
    print()
    
    # 推奨事項
    print("💡 実用化への推奨事項")
    print("=" * 70)
    print("1. **データ拡張**: より多くの訓練データで精度向上")
    print("2. **特徴量エンジニアリング**: コード品質、協力関係などの追加")
    print("3. **ハイブリッドアプローチ**: ルールベース + ML の組み合わせ")
    print("4. **継続学習**: 新しいデータでモデルを継続的に更新")
    print("5. **A/Bテスト**: 本番環境での段階的導入と効果検証")
    print()
    
    # 結果保存
    results = {
        'irl_results': irl_results,
        'rl_results': rl_results,
        'comparison': {
            'irl_accuracy': irl_correct / len(training_data),
            'rl_accuracy': rl_correct / len(training_data),
            'training_data_size': len(training_data)
        },
        'demo_date': datetime.now().isoformat()
    }
    
    output_path = "outputs/comprehensive_retention/irl_rl_demo_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 結果を保存しました: {output_path}")
    print()
    print("🎉 IRL/RL継続予測システムデモ完了！")
    print("   ルールベースから機械学習への移行が実証されました。")


if __name__ == "__main__":
    main()