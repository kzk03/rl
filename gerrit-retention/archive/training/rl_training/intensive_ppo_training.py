#!/usr/bin/env python3
"""
本格的PPOエージェント訓練スクリプト（時間をかけた深い学習）
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gerrit_retention.rl_environment.review_env import ReviewAcceptanceEnvironment
from gerrit_retention.rl_environment.ppo_agent import PPOAgent, PPOConfig
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import json
from datetime import datetime

def main():
    """本格的PPO訓練（時間をかけた深い学習）"""
    print('=== 🎓 本格的gerrit-retention強化学習システム実行 ===')
    print('（時間をかけた深い学習版）')
    
    # 本格的設定（時間をかける）
    config = PPOConfig(
        hidden_size=512,        # 大規模ネットワーク
        num_layers=4,           # 深いネットワーク
        learning_rate=5e-5,     # 低学習率（慎重な学習）
        gamma=0.995,            # 長期報酬重視
        gae_lambda=0.98,        # 高精度GAE
        clip_epsilon=0.1,       # 保守的クリッピング
        value_loss_coef=1.0,    # 価値学習重視
        entropy_coef=0.02,      # 探索促進
        batch_size=256,         # 大規模バッチ
        mini_batch_size=32,     # 細かい更新
        ppo_epochs=20,          # 十分な更新
        buffer_size=4096        # 大容量バッファ
    )
    
    # 複雑な環境設定
    env_config = {
        'max_episode_length': 500,  # 長いエピソード
        'max_queue_size': 30,       # 大きなキュー
        'stress_threshold': 0.9     # 厳しい閾値
    }
    
    print(f'📊 本格的訓練設定:')
    print(f'  隠れ層サイズ: {config.hidden_size}')
    print(f'  ネットワーク層数: {config.num_layers}')
    print(f'  学習率: {config.learning_rate}')
    print(f'  バッファサイズ: {config.buffer_size}')
    print(f'  バッチサイズ: {config.batch_size}')
    print(f'  PPOエポック: {config.ppo_epochs}')
    print(f'  最大エピソード長: {env_config[\"max_episode_length\"]}')
    
    # 環境とエージェント初期化
    env = ReviewAcceptanceEnvironment(env_config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(obs_dim, action_dim, config)
    
    # パラメータ数計算
    total_params = sum(p.numel() for p in agent.policy_net.parameters()) + sum(p.numel() for p in agent.value_net.parameters())
    
    print(f'\\n🏗️ システム初期化完了:')
    print(f'  観測次元: {obs_dim}')
    print(f'  行動次元: {action_dim}')
    print(f'  総パラメータ数: {total_params:,}')
    print(f'  デバイス: {agent.device}')
    
    # 訓練統計
    episode_rewards = []
    episode_lengths = []
    update_count = 0
    action_counts = {'reject': 0, 'accept': 0, 'wait': 0}
    action_names = ['reject', 'accept', 'wait']
    
    print(f'\\n=== 🎯 本格的PPO訓練開始（深い学習版） ===')
    start_time = time.time()
    best_avg_reward = float('-inf')
    
    total_episodes = 2000  # 大量のエピソード
    
    # 学習率スケジューラー
    scheduler = torch.optim.lr_scheduler.StepLR(
        agent.policy_optimizer, step_size=500, gamma=0.8
    )
    
    # 探索率スケジューラー
    initial_exploration = 0.3
    final_exploration = 0.05
    exploration_decay = (initial_exploration - final_exploration) / total_episodes
    
    try:
        for episode in tqdm(range(total_episodes), desc='本格的PPO訓練'):
            # 探索率調整
            current_exploration = max(
                final_exploration, 
                initial_exploration - episode * exploration_decay
            )
            
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            # 長いエピソード実行
            for step in range(env_config['max_episode_length']):
                try:
                    # 探索的行動選択
                    if np.random.random() < current_exploration:
                        action = np.random.randint(action_dim)
                        log_prob = np.log(1.0 / action_dim)
                        value = 0.0
                    else:
                        action, log_prob, value = agent.select_action(obs, training=True)
                    
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    
                    # 統計記録
                    action_counts[action_names[action]] += 1
                    
                    # 経験を保存
                    agent.store_experience(obs, action, reward, value, log_prob, terminated or truncated)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    obs = next_obs
                    
                    if terminated or truncated:
                        break
                        
                except Exception as e:
                    print(f'ステップエラー（スキップ）: {e}')
                    break
            
            # PPO更新（大容量バッファが溜まったら）
            if agent.buffer.size >= config.buffer_size:
                try:
                    # 複数回更新
                    for _ in range(3):  # 追加更新
                        losses = agent.update()
                        if losses:
                            update_count += 1
                    
                    # 学習率調整
                    scheduler.step()
                    
                except Exception as e:
                    print(f'更新エラー（スキップ）: {e}')
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 定期的な評価
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                
                print(f'\\n📊 エピソード {episode + 1:,}/{total_episodes:,}:')
                print(f'  平均報酬: {avg_reward:.3f}')
                print(f'  平均長: {avg_length:.1f}')
                print(f'  更新回数: {update_count:,}')
                print(f'  探索率: {current_exploration:.3f}')
                print(f'  学習率: {scheduler.get_last_lr()[0]:.6f}')
                
                # 行動分布
                total_actions = sum(action_counts.values())
                if total_actions > 0:
                    reject_pct = action_counts['reject'] / total_actions * 100
                    accept_pct = action_counts['accept'] / total_actions * 100
                    wait_pct = action_counts['wait'] / total_actions * 100
                    print(f'  行動分布: Reject {reject_pct:.1f}%, Accept {accept_pct:.1f}%, Wait {wait_pct:.1f}%')
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    print(f'  🏆 新記録! ベスト平均報酬: {best_avg_reward:.3f}')
                
                # 中間保存
                if (episode + 1) % 500 == 0:
                    torch.save(agent.state_dict(), f'models/intensive_ppo_checkpoint_{episode+1}.pth')
                    print(f'  💾 チェックポイント保存: episode_{episode+1}')
    
    except KeyboardInterrupt:
        print('\\n⚠️ 訓練が中断されました')
    except Exception as e:
        print(f'\\n❌ 予期しないエラー: {e}')
    
    # 最終統計
    training_time = time.time() - start_time
    completed_episodes = len(episode_rewards)
    
    print(f'\\n=== ✅ 本格的PPO訓練完了 ===')
    print(f'完了エピソード数: {completed_episodes:,}/{total_episodes:,}')
    print(f'総訓練時間: {training_time:.2f}秒 ({training_time/60:.1f}分)')
    print(f'総ステップ数: {sum(episode_lengths):,}')
    print(f'総更新回数: {update_count:,}')
    print(f'総パラメータ数: {total_params:,}')
    
    if episode_rewards:
        print(f'平均エピソード報酬: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}')
        print(f'ベスト平均報酬: {best_avg_reward:.3f}')
        print(f'平均エピソード長: {np.mean(episode_lengths):.1f}')
        
        # 学習効果分析
        if len(episode_rewards) >= 200:
            early_avg = np.mean(episode_rewards[:100])
            late_avg = np.mean(episode_rewards[-100:])
            improvement = late_avg - early_avg
            improvement_pct = improvement/abs(early_avg)*100 if early_avg != 0 else 0
            print(f'\\n📈 学習効果:')
            print(f'初期100エピソード平均: {early_avg:.3f}')
            print(f'最終100エピソード平均: {late_avg:.3f}')
            print(f'改善度: {improvement:.3f} ({improvement_pct:.1f}%)')
    
    # 最終モデル保存
    torch.save(agent.state_dict(), 'models/intensive_ppo_final.pth')
    
    # 結果保存
    results = {
        'timestamp': datetime.now().isoformat(),
        'completed_episodes': completed_episodes,
        'total_steps': sum(episode_lengths),
        'total_updates': update_count,
        'total_params': int(total_params),
        'training_time': training_time,
        'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'best_reward': float(best_avg_reward) if best_avg_reward != float('-inf') else 0.0,
        'action_distribution': {k: int(v) for k, v in action_counts.items()},
        'status': 'intensive_training_completed'
    }
    
    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'outputs/intensive_rl_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\\n💾 結果保存: outputs/intensive_rl_results_{timestamp}.json')
    print(f'\\n✅ 本格的gerrit-retention強化学習システム実行完了!')
    print(f'深い学習で{completed_episodes:,}エピソード完了しました！')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())