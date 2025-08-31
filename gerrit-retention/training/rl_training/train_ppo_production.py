#!/usr/bin/env python3
"""
本格的なPPOエージェント訓練スクリプト
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# プロジェクトパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gerrit_retention.rl_environment.ppo_agent import PPOAgent, PPOConfig
from gerrit_retention.rl_environment.review_env import ReviewAcceptanceEnvironment
from gerrit_retention.utils.logger import get_logger

logger = get_logger(__name__)


class ProductionPPOTrainer:
    """本格的なPPO訓練クラス"""
    
    def __init__(self, config: Dict):
        """
        訓練器を初期化
        
        Args:
            config: 訓練設定辞書
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用デバイス: {self.device}")
        
        # 環境設定
        self.env_config = config['environment']
        self.training_config = config['training']
        
        # 環境とエージェントの初期化
        self.env = ReviewAcceptanceEnvironment(self.env_config)
        
        # PPO設定
        ppo_config = PPOConfig(
            hidden_size=config['agent']['hidden_size'],
            num_layers=config['agent']['num_layers'],
            learning_rate=config['agent']['learning_rate'],
            gamma=config['agent']['gamma'],
            gae_lambda=config['agent']['gae_lambda'],
            clip_epsilon=config['agent']['clip_epsilon'],
            value_loss_coef=config['agent']['value_loss_coef'],
            entropy_coef=config['agent']['entropy_coef'],
            max_grad_norm=config['agent']['max_grad_norm'],
            batch_size=config['agent']['batch_size'],
            mini_batch_size=config['agent']['mini_batch_size'],
            ppo_epochs=config['agent']['ppo_epochs'],
            buffer_size=config['agent']['buffer_size']
        )
        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.agent = PPOAgent(obs_dim, action_dim, ppo_config, device=str(self.device))
        
        # 訓練統計
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        logger.info(f"PPO訓練器を初期化: 観測次元={obs_dim}, 行動次元={action_dim}")
    
    def train(self) -> Dict:
        """
        本格的な訓練を実行
        
        Returns:
            Dict: 訓練結果統計
        """
        total_episodes = self.training_config['total_episodes']
        max_steps_per_episode = self.training_config['max_steps_per_episode']
        save_interval = self.training_config['save_interval']
        eval_interval = self.training_config['eval_interval']
        
        logger.info(f"本格的PPO訓練開始: {total_episodes}エピソード")
        logger.info(f"最大ステップ/エピソード: {max_steps_per_episode}")
        
        start_time = time.time()
        best_avg_reward = float('-inf')
        
        # 訓練ループ
        for episode in tqdm(range(total_episodes), desc="PPO訓練"):
            episode_reward, episode_length, losses = self._train_episode(max_steps_per_episode)
            
            # 統計記録
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if losses:
                self.policy_losses.append(losses['policy_loss'])
                self.value_losses.append(losses['value_loss'])
                self.entropy_losses.append(losses['entropy_loss'])
            
            # 定期的な評価とログ出力
            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                avg_length = np.mean(self.episode_lengths[-eval_interval:])
                
                logger.info(f"エピソード {episode + 1}/{total_episodes}")
                logger.info(f"  平均報酬: {avg_reward:.3f}")
                logger.info(f"  平均長: {avg_length:.1f}")
                
                if losses:
                    logger.info(f"  政策損失: {losses['policy_loss']:.6f}")
                    logger.info(f"  価値損失: {losses['value_loss']:.6f}")
                    logger.info(f"  エントロピー損失: {losses['entropy_loss']:.6f}")
                
                # ベストモデル保存
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self._save_model(f"models/ppo_agent_best.pth", episode + 1, avg_reward)
                    logger.info(f"  新しいベストモデル保存: 報酬={avg_reward:.3f}")
            
            # 定期的なモデル保存
            if (episode + 1) % save_interval == 0:
                self._save_model(f"models/ppo_agent_episode_{episode + 1}.pth", episode + 1)
        
        # 最終モデル保存
        self._save_model("models/ppo_agent_final.pth", total_episodes)
        
        training_time = time.time() - start_time
        
        # 訓練統計
        training_stats = {
            'total_episodes': total_episodes,
            'total_training_time': training_time,
            'average_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'average_episode_length': np.mean(self.episode_lengths),
            'best_average_reward': best_avg_reward,
            'final_average_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'convergence_episode': self._find_convergence_episode(),
            'device_used': str(self.device)
        }
        
        # 統計保存
        self._save_training_stats(training_stats)
        
        logger.info(f"PPO訓練完了!")
        logger.info(f"総訓練時間: {training_time:.2f}秒")
        logger.info(f"平均エピソード報酬: {training_stats['average_episode_reward']:.3f}")
        logger.info(f"ベスト平均報酬: {best_avg_reward:.3f}")
        
        return training_stats
    
    def _train_episode(self, max_steps: int) -> Tuple[float, int, Dict]:
        """
        1エピソードの訓練を実行
        
        Args:
            max_steps: 最大ステップ数
            
        Returns:
            Tuple[float, int, Dict]: エピソード報酬、長さ、損失
        """
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # エピソード実行
        for step in range(max_steps):
            action, log_prob, value = self.agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # 経験を保存
            self.agent.store_experience(obs, action, reward, value, log_prob, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # PPO更新
        losses = None
        if self.agent.buffer.is_full() or (episode_length > 0 and episode_length % self.agent.config.batch_size == 0):
            losses = self.agent.update()
        
        return episode_reward, episode_length, losses
    
    def _save_model(self, filepath: str, episode: int, avg_reward: float = None):
        """
        モデルを保存
        
        Args:
            filepath: 保存パス
            episode: エピソード数
            avg_reward: 平均報酬
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.agent.state_dict(),
            'config': self.config,
            'avg_reward': avg_reward,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
    
    def _save_training_stats(self, stats: Dict):
        """
        訓練統計を保存
        
        Args:
            stats: 統計辞書
        """
        os.makedirs("models", exist_ok=True)
        
        # 詳細統計
        detailed_stats = {
            **stats,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open("models/training_stats_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
        
        # サマリー統計
        with open("models/training_stats_summary.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("訓練統計を保存しました")
    
    def _find_convergence_episode(self) -> int:
        """
        収束エピソードを推定
        
        Returns:
            int: 収束エピソード番号
        """
        if len(self.episode_rewards) < 100:
            return len(self.episode_rewards)
        
        # 移動平均を計算
        window_size = 50
        moving_avg = []
        for i in range(window_size, len(self.episode_rewards)):
            avg = np.mean(self.episode_rewards[i-window_size:i])
            moving_avg.append(avg)
        
        # 収束判定（移動平均の変化が小さくなった点）
        if len(moving_avg) < 50:
            return len(self.episode_rewards)
        
        threshold = 0.01  # 1%の変化
        for i in range(50, len(moving_avg)):
            recent_change = abs(moving_avg[i] - moving_avg[i-50]) / abs(moving_avg[i-50])
            if recent_change < threshold:
                return i + window_size
        
        return len(self.episode_rewards)


def main():
    """メイン関数"""
    # 本格的な訓練設定
    config = {
        'environment': {
            'max_episode_length': 200,
            'max_queue_size': 20,
            'stress_threshold': 0.8,
            'stress_config': {},
            'behavior_config': {},
            'similarity_config': {}
        },
        'agent': {
            'hidden_size': 256,
            'num_layers': 3,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'batch_size': 128,
            'mini_batch_size': 32,
            'ppo_epochs': 10,
            'buffer_size': 4096
        },
        'training': {
            'total_episodes': 5000,  # 本格的な訓練エピソード数
            'max_steps_per_episode': 200,
            'save_interval': 500,
            'eval_interval': 100
        }
    }
    
    logger.info("=== 本格的PPO訓練開始 ===")
    logger.info(f"総エピソード数: {config['training']['total_episodes']}")
    logger.info(f"バッファサイズ: {config['agent']['buffer_size']}")
    logger.info(f"バッチサイズ: {config['agent']['batch_size']}")
    logger.info(f"PPOエポック: {config['agent']['ppo_epochs']}")
    
    try:
        trainer = ProductionPPOTrainer(config)
        training_stats = trainer.train()
        
        logger.info("=== 訓練完了 ===")
        logger.info(f"最終平均報酬: {training_stats['final_average_reward']:.3f}")
        logger.info(f"収束エピソード: {training_stats['convergence_episode']}")
        logger.info(f"総訓練時間: {training_stats['total_training_time']:.2f}秒")
        
        return 0
        
    except Exception as e:
        logger.error(f"訓練エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())