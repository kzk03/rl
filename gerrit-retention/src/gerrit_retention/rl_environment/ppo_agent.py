"""
PPO エージェント実装

このモジュールは、開発者定着予測に最適化されたPPO（Proximal Policy Optimization）エージェントを実装する。
長期定着重視の学習アルゴリズムと、開発者ごとの個別価値関数学習を含む。
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PPOConfig:
    """PPO設定"""
    # ネットワーク設定
    hidden_size: int = 256
    num_layers: int = 3
    
    # 学習設定
    learning_rate: float = 3e-4
    gamma: float = 0.99  # 長期報酬重視
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # 訓練設定
    batch_size: int = 64
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    buffer_size: int = 2048
    
    # 適応的設定
    adaptive_lr: bool = True
    adaptive_clip: bool = True
    individual_value_functions: bool = True
    
    # 探索設定
    exploration_schedule: str = "linear"  # "linear", "exponential", "adaptive"
    initial_exploration: float = 0.3
    final_exploration: float = 0.05
    exploration_decay_steps: int = 100000


class PolicyNetwork(nn.Module):
    """
    ポリシーネットワーク
    
    状態から行動確率分布を出力する。
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256, num_layers: int = 3):
        """
        ポリシーネットワークを初期化
        
        Args:
            state_dim: 状態次元数
            action_dim: 行動次元数
            hidden_size: 隠れ層サイズ
            num_layers: 層数
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # ネットワーク構築
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_size
        
        # 出力層
        layers.append(nn.Linear(hidden_size, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 重み初期化
        self._initialize_weights()
        
        logger.debug(f"ポリシーネットワークを初期化: {state_dim} -> {action_dim}")
    
    def _initialize_weights(self):
        """重みを初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            state: 状態テンソル
            
        Returns:
            torch.Tensor: 行動ロジット
        """
        return self.network(state)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        行動確率を取得
        
        Args:
            state: 状態テンソル
            
        Returns:
            torch.Tensor: 行動確率
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_action_distribution(self, state: torch.Tensor) -> Categorical:
        """
        行動分布を取得
        
        Args:
            state: 状態テンソル
            
        Returns:
            Categorical: 行動分布
        """
        logits = self.forward(state)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """
    価値ネットワーク
    
    状態から価値を推定する。開発者ごとの個別価値関数をサポート。
    """
    
    def __init__(self, state_dim: int, hidden_size: int = 256, num_layers: int = 3, 
                 num_developers: int = 1):
        """
        価値ネットワークを初期化
        
        Args:
            state_dim: 状態次元数
            hidden_size: 隠れ層サイズ
            num_layers: 層数
            num_developers: 開発者数（個別価値関数用）
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_developers = num_developers
        
        # 共通特徴抽出層
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 個別価値関数ヘッド
        if num_developers > 1:
            self.value_heads = nn.ModuleList([
                nn.Linear(hidden_size, 1) for _ in range(num_developers)
            ])
        else:
            self.value_head = nn.Linear(hidden_size, 1)
        
        # 重み初期化
        self._initialize_weights()
        
        logger.debug(f"価値ネットワークを初期化: {state_dim} -> 1 (開発者数: {num_developers})")
    
    def _initialize_weights(self):
        """重みを初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, developer_id: Optional[int] = None) -> torch.Tensor:
        """
        順伝播
        
        Args:
            state: 状態テンソル
            developer_id: 開発者ID（個別価値関数用）
            
        Returns:
            torch.Tensor: 価値推定
        """
        features = self.feature_extractor(state)
        
        if self.num_developers > 1 and developer_id is not None:
            return self.value_heads[developer_id](features)
        else:
            return self.value_head(features)


class ExperienceBuffer:
    """
    経験バッファ
    
    PPO学習用の経験を蓄積する。
    """
    
    def __init__(self, buffer_size: int, state_dim: int):
        """
        経験バッファを初期化
        
        Args:
            buffer_size: バッファサイズ
            state_dim: 状態次元数
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        
        # バッファの初期化
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
        logger.debug(f"経験バッファを初期化: サイズ={buffer_size}")
    
    def store(self, state: np.ndarray, action: int, reward: float, value: float, 
              log_prob: float, done: bool):
        """
        経験を保存
        
        Args:
            state: 状態
            action: 行動
            reward: 報酬
            value: 価値推定
            log_prob: 行動対数確率
            done: 終了フラグ
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_advantages(self, gamma: float, gae_lambda: float, last_value: float = 0.0):
        """
        GAE（Generalized Advantage Estimation）でアドバンテージを計算
        
        Args:
            gamma: 割引率
            gae_lambda: GAEパラメータ
            last_value: 最後の価値推定
        """
        # リターンとアドバンテージの計算
        advantages = np.zeros(self.size, dtype=np.float32)
        last_gae_lam = 0
        
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            
            delta = (self.rewards[step] + gamma * next_values * next_non_terminal - 
                    self.values[step])
            advantages[step] = last_gae_lam = (delta + gamma * gae_lambda * 
                                              next_non_terminal * last_gae_lam)
        
        self.advantages[:self.size] = advantages
        self.returns[:self.size] = advantages + self.values[:self.size]
        
        # アドバンテージの正規化
        self.advantages[:self.size] = ((self.advantages[:self.size] - 
                                       self.advantages[:self.size].mean()) / 
                                      (self.advantages[:self.size].std() + 1e-8))
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        バッチデータを取得
        
        Args:
            batch_size: バッチサイズ
            
        Returns:
            Dict[str, torch.Tensor]: バッチデータ
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.LongTensor(self.actions[indices]),
            'old_log_probs': torch.FloatTensor(self.log_probs[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices]),
            'values': torch.FloatTensor(self.values[indices])
        }
    
    def clear(self):
        """バッファをクリア"""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """
    PPO エージェント
    
    開発者定着予測に最適化されたPPOアルゴリズムを実装。
    長期定着重視の学習と適応的パラメータ調整を含む。
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig, 
                 device: str = "cpu"):
        """
        PPOエージェントを初期化
        
        Args:
            state_dim: 状態次元数
            action_dim: 行動次元数
            config: PPO設定
            device: 計算デバイス
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)
        
        # ネットワークの初期化
        self.policy_net = PolicyNetwork(
            state_dim, action_dim, config.hidden_size, config.num_layers
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            state_dim, config.hidden_size, config.num_layers
        ).to(self.device)
        
        # オプティマイザーの初期化
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=config.learning_rate
        )
        
        # 経験バッファ
        self.buffer = ExperienceBuffer(config.buffer_size, state_dim)
        
        # 学習統計
        self.training_step = 0
        self.episode_count = 0
        self.total_rewards = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        
        # 適応的パラメータ
        self.current_lr = config.learning_rate
        self.current_clip_epsilon = config.clip_epsilon
        self.current_exploration = config.initial_exploration
        
        logger.info(f"PPOエージェントを初期化: 状態次元={state_dim}, 行動次元={action_dim}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        行動を選択
        
        Args:
            state: 現在の状態
            training: 訓練モードフラグ
            
        Returns:
            Tuple[int, float, float]: (行動, 対数確率, 価値推定)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # ポリシーから行動分布を取得
            action_dist = self.policy_net.get_action_distribution(state_tensor)
            
            # 価値推定
            value = self.value_net(state_tensor).item()
            
            if training:
                # 探索的行動選択
                if np.random.random() < self.current_exploration:
                    action = np.random.randint(self.action_dim)
                    log_prob = action_dist.log_prob(torch.tensor(action)).item()
                else:
                    action = action_dist.sample().item()
                    log_prob = action_dist.log_prob(torch.tensor(action)).item()
            else:
                # 貪欲行動選択
                action_probs = action_dist.probs
                action = torch.argmax(action_probs).item()
                log_prob = action_dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob, value
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float, done: bool):
        """
        経験を保存
        
        Args:
            state: 状態
            action: 行動
            reward: 報酬
            value: 価値推定
            log_prob: 行動対数確率
            done: 終了フラグ
        """
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def update(self) -> Dict[str, float]:
        """
        ポリシーと価値関数を更新
        
        Returns:
            Dict[str, float]: 学習統計
        """
        if self.buffer.size < self.config.batch_size:
            return {}
        
        # アドバンテージ計算
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.states[self.buffer.size - 1]).unsqueeze(0).to(self.device)
            last_value = self.value_net(last_state).item()
        
        self.buffer.compute_advantages(self.config.gamma, self.config.gae_lambda, last_value)
        
        # 学習統計
        policy_losses = []
        value_losses = []
        entropies = []
        
        # PPO更新
        for epoch in range(self.config.ppo_epochs):
            # ミニバッチでの更新
            num_batches = self.buffer.size // self.config.mini_batch_size
            
            for _ in range(num_batches):
                batch = self.buffer.get_batch(self.config.mini_batch_size)
                
                # ポリシー更新
                policy_loss, entropy = self._update_policy(batch)
                policy_losses.append(policy_loss)
                entropies.append(entropy)
                
                # 価値関数更新
                value_loss = self._update_value_function(batch)
                value_losses.append(value_loss)
        
        # バッファクリア
        self.buffer.clear()
        
        # 適応的パラメータ更新
        self._update_adaptive_parameters()
        
        # 統計更新
        self.training_step += 1
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'learning_rate': self.current_lr,
            'clip_epsilon': self.current_clip_epsilon,
            'exploration_rate': self.current_exploration
        }
    
    def _update_policy(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        ポリシーを更新
        
        Args:
            batch: バッチデータ
            
        Returns:
            Tuple[float, float]: (ポリシー損失, エントロピー)
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        
        # 現在のポリシーから行動分布を取得
        action_dist = self.policy_net.get_action_distribution(states)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        # 重要度比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPOクリップ損失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.current_clip_epsilon, 
                           1.0 + self.current_clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # エントロピー正則化
        total_loss = policy_loss - self.config.entropy_coef * entropy
        
        # 勾配更新
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 
                                      self.config.max_grad_norm)
        self.policy_optimizer.step()
        
        return policy_loss.item(), entropy.item()
    
    def _update_value_function(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        価値関数を更新
        
        Args:
            batch: バッチデータ
            
        Returns:
            float: 価値損失
        """
        states = batch['states'].to(self.device)
        returns = batch['returns'].to(self.device)
        old_values = batch['values'].to(self.device)
        
        # 現在の価値推定
        new_values = self.value_net(states).squeeze()
        
        # クリップされた価値損失
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, -self.current_clip_epsilon, self.current_clip_epsilon
        )
        
        value_loss1 = (new_values - returns).pow(2)
        value_loss2 = (value_pred_clipped - returns).pow(2)
        value_loss = torch.max(value_loss1, value_loss2).mean()
        
        # 勾配更新
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 
                                      self.config.max_grad_norm)
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def _update_adaptive_parameters(self):
        """適応的パラメータを更新"""
        # 学習率の適応的調整
        if self.config.adaptive_lr:
            if len(self.policy_losses) >= 10:
                recent_loss_trend = np.mean(self.policy_losses[-5:]) - np.mean(self.policy_losses[-10:-5])
                if recent_loss_trend > 0:  # 損失が増加傾向
                    self.current_lr *= 0.99
                else:  # 損失が減少傾向
                    self.current_lr *= 1.001
                
                # 学習率の範囲制限
                self.current_lr = np.clip(self.current_lr, 1e-5, 1e-2)
                
                # オプティマイザーの学習率更新
                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                for param_group in self.value_optimizer.param_groups:
                    param_group['lr'] = self.current_lr
        
        # クリップ率の適応的調整
        if self.config.adaptive_clip:
            if len(self.policy_losses) >= 5:
                loss_variance = np.var(self.policy_losses[-5:])
                if loss_variance > 0.1:  # 損失が不安定
                    self.current_clip_epsilon *= 0.99
                else:  # 損失が安定
                    self.current_clip_epsilon *= 1.001
                
                # クリップ率の範囲制限
                self.current_clip_epsilon = np.clip(self.current_clip_epsilon, 0.1, 0.3)
        
        # 探索率の更新
        if self.config.exploration_schedule == "linear":
            progress = min(1.0, self.training_step / self.config.exploration_decay_steps)
            self.current_exploration = (self.config.initial_exploration * (1 - progress) + 
                                      self.config.final_exploration * progress)
        elif self.config.exploration_schedule == "exponential":
            decay_rate = 0.995
            self.current_exploration = max(self.config.final_exploration,
                                         self.current_exploration * decay_rate)
    
    def save_model(self, filepath: str):
        """
        モデルを保存
        
        Args:
            filepath: 保存先パス
        """
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }, filepath)
        
        logger.info(f"モデルを保存しました: {filepath}")
    
    def load_model(self, filepath: str):
        """
        モデルを読み込み
        
        Args:
            filepath: 読み込み元パス
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        
        logger.info(f"モデルを読み込みました: {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        訓練統計を取得
        
        Returns:
            Dict[str, Any]: 訓練統計
        """
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'avg_policy_loss': np.mean(self.policy_losses) if self.policy_losses else 0.0,
            'avg_value_loss': np.mean(self.value_losses) if self.value_losses else 0.0,
            'avg_episode_reward': np.mean(self.total_rewards) if self.total_rewards else 0.0,
            'current_lr': self.current_lr,
            'current_clip_epsilon': self.current_clip_epsilon,
            'current_exploration': self.current_exploration
        }
    
    def set_training_mode(self, training: bool = True):
        """
        訓練モードを設定
        
        Args:
            training: 訓練モードフラグ
        """
        self.policy_net.train(training)
        self.value_net.train(training)
    
    def episode_finished(self, total_reward: float):
        """
        エピソード終了時の処理
        
        Args:
            total_reward: エピソード総報酬
        """
        self.episode_count += 1
        self.total_rewards.append(total_reward)
        
        logger.debug(f"エピソード {self.episode_count} 終了: 総報酬={total_reward:.3f}")


def create_ppo_agent(env: gym.Env, config: Optional[Dict[str, Any]] = None, 
                    device: str = "cpu") -> PPOAgent:
    """
    PPOエージェントを作成
    
    Args:
        env: 環境
        config: 設定辞書
        device: 計算デバイス
        
    Returns:
        PPOAgent: 作成されたPPOエージェント
    """
    # デフォルト設定
    default_config = PPOConfig()
    
    # 設定をマージ
    if config:
        for key, value in config.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
    
    # 環境から状態・行動次元を取得
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim, default_config, device)
    
    logger.info(f"PPOエージェントを作成しました: 状態次元={state_dim}, 行動次元={action_dim}")
    
    return agent