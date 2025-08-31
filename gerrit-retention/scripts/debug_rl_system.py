#!/usr/bin/env python3
"""
強化学習システムのデバッグ
モデルの動作を詳細に分析
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.append('scripts')

from advanced_rl_system import AdvancedActorCritic
from gerrit_retention.utils.logger import get_logger
from rl_task_optimizer import RLTaskOptimizer

logger = get_logger(__name__)

def debug_model_predictions():
    """モデル予測のデバッグ"""
    logger.info("=== 強化学習モデルデバッグ ===")
    
    # データの読み込み
    with open('data/processed/unified/all_developers.json', 'r') as f:
        developers_data = json.load(f)
    
    with open('data/processed/unified/all_reviews.json', 'r') as f:
        reviews_data = json.load(f)
    
    # 人間の開発者を抽出
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
    )[:3]
    
    # タスクを選択
    tasks = reviews_data[:3]
    
    print(f"\\n🔍 デバッグ対象:")
    print(f"   開発者: {len(top_developers)}名")
    print(f"   タスク: {len(tasks)}件")
    
    # 最適化器の初期化
    optimizer = RLTaskOptimizer()
    
    print(f"\\n📊 詳細な予測分析:")
    
    for i, task in enumerate(tasks):
        print(f"\\n--- タスク {i+1}: {task.get('change_id', 'unknown')} ---")
        print(f"    サイズ: {task.get('lines_added', 0)}行追加, {task.get('files_changed', 0)}ファイル変更")
        
        for j, developer in enumerate(top_developers):
            print(f"\\n  開発者 {j+1}: {developer['name']}")
            print(f"    活動: 作成{developer['changes_authored']}件, レビュー{developer['changes_reviewed']}件")
            
            # 観測ベクトルの構築
            obs = optimizer._build_observation(developer, task)
            print(f"    観測ベクトル (最初の10次元): {obs[:10]}")
            
            # モデル予測
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, value = optimizer.actor_critic(obs_tensor)
                probs = torch.softmax(action_logits, dim=-1)
                
                print(f"    行動ロジット: {action_logits[0].numpy()}")
                print(f"    行動確率: assign={probs[0,0]:.3f}, reject={probs[0,1]:.3f}, defer={probs[0,2]:.3f}")
                print(f"    状態価値: {value.item():.3f}")
                
                # 最適行動
                best_action_idx = torch.argmax(probs, dim=-1).item()
                action_names = ['assign', 'reject', 'defer']
                best_action = action_names[best_action_idx]
                confidence = probs[0, best_action_idx].item()
                
                print(f"    → 推奨: {best_action} (信頼度: {confidence:.1%})")
    
    # 統計分析
    print(f"\\n📈 統計分析:")
    
    all_predictions = []
    for task in tasks:
        for developer in top_developers:
            obs = optimizer._build_observation(developer, task)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, value = optimizer.actor_critic(obs_tensor)
                probs = torch.softmax(action_logits, dim=-1)
                best_action_idx = torch.argmax(probs, dim=-1).item()
                
                all_predictions.append({
                    'action': best_action_idx,
                    'probs': probs[0].numpy(),
                    'value': value.item()
                })
    
    # 行動分布
    action_counts = [0, 0, 0]
    for pred in all_predictions:
        action_counts[pred['action']] += 1
    
    total_predictions = len(all_predictions)
    print(f"   総予測数: {total_predictions}")
    print(f"   割り当て (assign): {action_counts[0]}件 ({action_counts[0]/total_predictions*100:.1f}%)")
    print(f"   拒否 (reject): {action_counts[1]}件 ({action_counts[1]/total_predictions*100:.1f}%)")
    print(f"   延期 (defer): {action_counts[2]}件 ({action_counts[2]/total_predictions*100:.1f}%)")
    
    # 平均確率
    avg_probs = np.mean([pred['probs'] for pred in all_predictions], axis=0)
    print(f"   平均確率: assign={avg_probs[0]:.3f}, reject={avg_probs[1]:.3f}, defer={avg_probs[2]:.3f}")
    
    # 平均価値
    avg_value = np.mean([pred['value'] for pred in all_predictions])
    print(f"   平均状態価値: {avg_value:.3f}")
    
    # モデルの重みを確認
    print(f"\\n🔧 モデル構造分析:")
    total_params = sum(p.numel() for p in optimizer.actor_critic.parameters())
    trainable_params = sum(p.numel() for p in optimizer.actor_critic.parameters() if p.requires_grad)
    print(f"   総パラメータ数: {total_params:,}")
    print(f"   訓練可能パラメータ数: {trainable_params:,}")
    
    # 最終層の重みを確認
    actor_final_layer = optimizer.actor_critic.actor_head[-1]
    print(f"   Actor最終層の重み範囲: [{actor_final_layer.weight.min():.3f}, {actor_final_layer.weight.max():.3f}]")
    print(f"   Actor最終層のバイアス: {actor_final_layer.bias.data}")
    
    logger.info("デバッグ完了")


def create_improved_model():
    """改良されたモデルの作成"""
    logger.info("=== 改良モデル作成 ===")
    
    # より積極的な割り当てを行うモデルを作成
    model = AdvancedActorCritic(obs_dim=20, action_dim=3)
    
    # Actor最終層のバイアスを調整（assignを優遇）
    with torch.no_grad():
        model.actor_head[-1].bias[0] = 1.0   # assign
        model.actor_head[-1].bias[1] = -0.5  # reject
        model.actor_head[-1].bias[2] = 0.0   # defer
    
    # モデルの保存
    torch.save({
        'actor_critic_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'training_stats': {}
    }, 'models/improved_ppo_agent.pth')
    
    logger.info("改良モデルを保存: models/improved_ppo_agent.pth")
    
    # 改良モデルでテスト
    print(f"\\n🚀 改良モデルのテスト:")
    
    optimizer = RLTaskOptimizer(model_path='models/improved_ppo_agent.pth')
    
    # サンプルデータでテスト
    sample_developer = {
        'developer_id': 'test@example.com',
        'name': 'Test Developer',
        'changes_authored': 50,
        'changes_reviewed': 100,
        'projects': ['project1', 'project2']
    }
    
    sample_task = {
        'change_id': 'test_change_001',
        'lines_added': 100,
        'files_changed': 3,
        'status': 'NEW',
        'score': 1
    }
    
    recommendation = optimizer.get_recommendation(sample_developer, sample_task)
    
    print(f"   開発者: {recommendation.developer_name}")
    print(f"   タスク: {recommendation.task_id}")
    print(f"   推奨: {recommendation.action} (信頼度: {recommendation.confidence:.1%})")
    print(f"   確率分布: {recommendation.action_probabilities}")
    print(f"   理由: {recommendation.reasoning}")


if __name__ == "__main__":
    debug_model_predictions()
    print("\\n" + "="*60)
    create_improved_model()