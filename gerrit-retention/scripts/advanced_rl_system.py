#!/usr/bin/env python3
"""
高度な強化学習システム
安定した数値計算と本格的なPPO実装
"""

import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Categorical

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gerrit_retention.utils.logger import get_logger

logger = get_logger(__name__)

class AdvancedActorCritic(nn.Module):
    """高度なActor-Criticネットワーク"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # 共有特徴抽出器
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor（ポリシー）ヘッド
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic（価値）ヘッド
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 重み初期化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """重みの初期化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播"""
        shared_features = self.shared_layers(obs)
        
        # ポリシー（行動確率）
        action_logits = self.actor_head(shared_features)
        
        # 価値関数
        value = self.critic_head(shared_features)
        
        return action_logits, value.squeeze(-1)
    
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        """行動と価値の取得"""
        action_logits, value = self.forward(obs)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


class AdvancedPPOAgent:
    """高度なPPOエージェント"""
    
    def __init__(self, obs_dim: int, action_dim: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # ハイパーパラメータ
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        
        # ネットワーク
        self.actor_critic = AdvancedActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        
        # 統計情報
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'explained_variance': []
        }
    
    def get_action_and_value(self, obs: np.ndarray, deterministic: bool = False):
        """行動と価値の取得"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.actor_critic(obs_tensor)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                probs = Categorical(logits=action_logits)
                action = probs.sample()
            
            log_prob = F.log_softmax(action_logits, dim=-1)[0, action]
            
        return action.item(), log_prob.item(), value.item()
    
    def update(self, rollout_buffer):
        """エージェントの更新"""
        # データの準備
        obs = torch.FloatTensor(rollout_buffer['observations']).to(self.device)
        actions = torch.LongTensor(rollout_buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer['log_probs']).to(self.device)
        returns = torch.FloatTensor(rollout_buffer['returns']).to(self.device)
        advantages = torch.FloatTensor(rollout_buffer['advantages']).to(self.device)
        
        # アドバンテージの正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 複数エポック更新
        for _ in range(self.k_epochs):
            # 現在のポリシーでの評価
            action_logits, values = self.actor_critic(obs)
            probs = Categorical(logits=action_logits)
            
            new_log_probs = probs.log_prob(actions)
            entropy = probs.entropy()
            
            # 重要度比
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPOクリッピング
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 価値関数損失
            value_loss = F.mse_loss(values, returns)
            
            # エントロピー損失
            entropy_loss = -entropy.mean()
            
            # 総損失
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 勾配更新
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # 統計情報の記録
        with torch.no_grad():
            explained_var = 1 - torch.var(returns - values) / (torch.var(returns) + 1e-8)
            
            stats = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'total_loss': total_loss.item(),
                'explained_variance': explained_var.item()
            }
            
            for key, value in stats.items():
                self.training_stats[key].append(value)
        
        return stats
    
    def save(self, path: str):
        """モデルの保存"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
        logger.info(f"モデルを保存: {path}")
    
    def load(self, path: str):
        """モデルの読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', {})
        logger.info(f"モデルを読み込み: {path}")


class RolloutBuffer:
    """ロールアウトバッファ"""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, obs, action, log_prob, reward, value, done):
        """データの追加"""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """リターンとアドバンテージの計算"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        # GAE計算
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values[:-1]
        
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'returns': returns,
            'advantages': advantages
        }
    
    def clear(self):
        """バッファのクリア"""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


class AdvancedTaskEnvironment(gym.Env):
    """高度なタスク割り当て環境"""
    
    def __init__(self, developers_data: List[Dict], reviews_data: List[Dict]):
        super().__init__()
        
        self.developers_data = developers_data
        self.reviews_data = reviews_data
        
        # 状態空間: 20次元の特徴量
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(20,), dtype=np.float32
        )
        
        # 行動空間: [割り当て, 拒否, 延期]
        self.action_space = spaces.Discrete(3)
        
        # 環境状態
        self.reset()
        
        logger.info(f"高度な環境を初期化: {len(developers_data)}名の開発者, {len(reviews_data)}件のタスク")
    
    def reset(self, seed=None, options=None):
        """環境のリセット"""
        super().reset(seed=seed)
        
        self.current_developer_idx = 0
        self.current_task_idx = 0
        self.episode_step = 0
        self.max_episode_steps = min(100, len(self.developers_data) * 2)
        
        # 統計情報
        self.episode_stats = {
            'assignments': 0,
            'rejections': 0,
            'deferrals': 0,
            'total_reward': 0.0
        }
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """現在の観測を取得"""
        if (self.current_developer_idx >= len(self.developers_data) or 
            self.current_task_idx >= len(self.reviews_data)):
            return np.zeros(20, dtype=np.float32)
        
        developer = self.developers_data[self.current_developer_idx]
        task = self.reviews_data[self.current_task_idx]
        
        obs = np.zeros(20, dtype=np.float32)
        
        # 開発者特徴量 (0-9)
        obs[0] = np.tanh((developer['changes_authored'] + developer['changes_reviewed']) / 100.0)
        obs[1] = np.tanh(developer['changes_authored'] / 50.0)
        obs[2] = np.tanh(developer['changes_reviewed'] / 100.0)
        obs[3] = np.tanh(len(developer['projects']) / 10.0)
        obs[4] = np.random.normal(0, 0.3)  # ストレスレベル（シミュレート）
        obs[5] = np.random.uniform(-0.5, 0.5)  # 作業負荷
        obs[6] = np.random.uniform(-0.3, 0.7)  # 専門性マッチ
        obs[7] = np.random.uniform(0.2, 1.0)   # 可用性
        obs[8] = np.random.uniform(-0.2, 0.8)  # コラボレーションスコア
        obs[9] = np.random.uniform(-0.3, 0.7)  # 最近のパフォーマンス
        
        # タスク特徴量 (10-14)
        obs[10] = np.tanh(task.get('lines_added', 0) / 500.0)
        obs[11] = np.tanh(task.get('files_changed', 0) / 10.0)
        obs[12] = 1.0 if task.get('status') == 'NEW' else -1.0
        obs[13] = np.tanh(task.get('score', 0) / 2.0)
        obs[14] = np.random.uniform(-0.5, 0.5)  # タスク優先度
        
        # 環境状態 (15-19)
        obs[15] = np.random.uniform(-0.3, 0.3)  # チーム作業負荷
        obs[16] = np.random.uniform(-0.5, 0.5)  # 締切プレッシャー
        obs[17] = np.random.uniform(-0.3, 0.3)  # チーム平均ストレス
        obs[18] = np.random.uniform(-0.2, 0.8)  # スプリント進捗
        obs[19] = np.random.uniform(0.0, 1.0)   # リソース可用性
        
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """環境ステップ"""
        # 報酬計算
        reward = self._calculate_reward(action)
        
        # 統計更新
        if action == 0:  # 割り当て
            self.episode_stats['assignments'] += 1
        elif action == 1:  # 拒否
            self.episode_stats['rejections'] += 1
        else:  # 延期
            self.episode_stats['deferrals'] += 1
        
        self.episode_stats['total_reward'] += reward
        
        # 次の状態へ
        self.episode_step += 1
        self.current_developer_idx = (self.current_developer_idx + 1) % len(self.developers_data)
        if self.current_developer_idx == 0:
            self.current_task_idx = (self.current_task_idx + 1) % len(self.reviews_data)
        
        # 終了条件
        terminated = self.episode_step >= self.max_episode_steps
        truncated = False
        
        next_obs = self._get_observation()
        
        return next_obs, reward, terminated, truncated, self.episode_stats
    
    def _calculate_reward(self, action: int) -> float:
        """報酬の計算"""
        if (self.current_developer_idx >= len(self.developers_data) or 
            self.current_task_idx >= len(self.reviews_data)):
            return -1.0
        
        developer = self.developers_data[self.current_developer_idx]
        task = self.reviews_data[self.current_task_idx]
        
        # 基本報酬
        base_reward = 0.0
        
        # 開発者の経験レベル
        experience_score = (developer['changes_authored'] + developer['changes_reviewed']) / 100.0
        
        # タスクの複雑さ
        complexity_score = (task.get('lines_added', 0) + task.get('files_changed', 0) * 10) / 100.0
        
        if action == 0:  # 割り当て
            # 経験豊富な開発者に適度なタスクを割り当てると高報酬
            if experience_score > 0.5 and complexity_score < 2.0:
                base_reward = 1.0 + np.random.normal(0, 0.1)
            elif experience_score > 0.2:
                base_reward = 0.5 + np.random.normal(0, 0.1)
            else:
                base_reward = -0.5 + np.random.normal(0, 0.1)
        
        elif action == 1:  # 拒否
            # 経験不足の開発者や複雑すぎるタスクを拒否すると報酬
            if experience_score < 0.3 or complexity_score > 3.0:
                base_reward = 0.3 + np.random.normal(0, 0.1)
            else:
                base_reward = -0.3 + np.random.normal(0, 0.1)
        
        else:  # 延期
            # 中程度の報酬
            base_reward = 0.1 + np.random.normal(0, 0.05)
        
        return np.clip(base_reward, -2.0, 2.0)


def train_advanced_rl_system():
    """高度な強化学習システムの訓練"""
    logger.info("=== 高度な強化学習システム訓練開始 ===")
    
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
    
    logger.info(f"訓練データ: {len(human_developers)}名の開発者, {len(reviews_data)}件のレビュー")
    
    # 環境とエージェントの初期化
    env = AdvancedTaskEnvironment(human_developers, reviews_data)
    agent = AdvancedPPOAgent(obs_dim=20, action_dim=3)
    
    # 訓練パラメータ
    total_episodes = 200
    rollout_length = 64
    update_frequency = 4
    
    # 統計情報
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    
    # 訓練ループ
    for episode in range(total_episodes):
        rollout_buffer = RolloutBuffer()
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # ロールアウト収集
        for step in range(rollout_length):
            action, log_prob, value = agent.get_action_and_value(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            rollout_buffer.add(obs, action, log_prob, reward, value, terminated or truncated)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        # 最後の価値推定
        _, _, last_value = agent.get_action_and_value(obs)
        
        # リターンとアドバンテージの計算
        rollout_data = rollout_buffer.compute_returns_and_advantages(last_value)
        
        # エージェント更新
        if episode % update_frequency == 0:
            update_stats = agent.update(rollout_data)
        
        # 統計情報の記録
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        assignments = info.get('assignments', 0)
        total_actions = assignments + info.get('rejections', 0) + info.get('deferrals', 0)
        success_rate = assignments / max(total_actions, 1)
        success_rates.append(success_rate)
        
        # ログ出力
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            avg_success_rate = np.mean(success_rates[-20:]) if success_rates else 0
            logger.info(f"エピソード {episode}: 平均報酬 {avg_reward:.3f}, 成功率 {avg_success_rate:.3f}")
    
    # モデルの保存
    model_path = 'models/advanced_ppo_agent.pth'
    agent.save(model_path)
    
    # 訓練統計の保存
    training_stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rates': success_rates,
        'final_avg_reward': np.mean(episode_rewards[-20:]),
        'final_success_rate': np.mean(success_rates[-20:])
    }
    
    with open('models/advanced_training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info(f"高度な強化学習訓練完了: 最終平均報酬 {training_stats['final_avg_reward']:.3f}")
    
    return agent, training_stats


if __name__ == "__main__":
    train_advanced_rl_system()