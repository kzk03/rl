#!/usr/bin/env python3
"""
実際の強化学習システム実装
開発者タスク割り当て最適化のためのPPOエージェント
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gerrit_retention.utils.logger import get_logger

logger = get_logger(__name__)

class DeveloperTaskEnvironment(gym.Env):
    """
    開発者タスク割り当て環境
    実際のGerritデータを使用した強化学習環境
    """
    
    def __init__(self, developers_data: List[Dict], reviews_data: List[Dict]):
        super().__init__()
        
        self.developers_data = developers_data
        self.reviews_data = reviews_data
        
        # 状態空間: [開発者特徴量(10) + タスク特徴量(5) + 環境状態(5)] = 20次元
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )
        
        # 行動空間: [割り当て, 拒否, 延期] = 3つの行動
        self.action_space = spaces.Discrete(3)
        
        # 環境状態
        self.current_developer_idx = 0
        self.current_task_idx = 0
        self.episode_step = 0
        self.max_episode_steps = 100
        
        # 統計情報
        self.total_assignments = 0
        self.successful_assignments = 0
        self.developer_stress_levels = {}
        
        logger.info(f"強化学習環境を初期化: {len(developers_data)}名の開発者, {len(reviews_data)}件のタスク")
    
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        super().reset(seed=seed)
        
        self.current_developer_idx = 0
        self.current_task_idx = 0
        self.episode_step = 0
        self.total_assignments = 0
        self.successful_assignments = 0
        self.developer_stress_levels = {i: 0.0 for i in range(len(self.developers_data))}
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """環境でアクションを実行"""
        reward = 0.0
        
        current_dev = self.developers_data[self.current_developer_idx]
        current_task = self.reviews_data[self.current_task_idx] if self.current_task_idx < len(self.reviews_data) else None
        
        if current_task is None:
            # タスクがない場合は終了
            return self._get_observation(), 0.0, True, False, {}
        
        # アクション実行
        if action == 0:  # 割り当て
            reward = self._calculate_assignment_reward(current_dev, current_task)
            self.total_assignments += 1
            if reward > 0:
                self.successful_assignments += 1
            
            # 開発者のストレスレベルを更新
            stress_increase = self._calculate_stress_increase(current_dev, current_task)
            self.developer_stress_levels[self.current_developer_idx] += stress_increase
            
        elif action == 1:  # 拒否
            reward = -0.1  # 小さなペナルティ
            
        elif action == 2:  # 延期
            reward = -0.05  # より小さなペナルティ
        
        # 次の状態に移行
        self.current_task_idx += 1
        if self.current_task_idx >= len(self.reviews_data):
            self.current_developer_idx += 1
            self.current_task_idx = 0
        
        self.episode_step += 1
        
        # 終了条件
        done = (self.episode_step >= self.max_episode_steps or 
                self.current_developer_idx >= len(self.developers_data))
        
        return self._get_observation(), reward, done, False, self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """現在の観測を取得"""
        obs = np.zeros(20, dtype=np.float32)
        
        if self.current_developer_idx < len(self.developers_data):
            dev = self.developers_data[self.current_developer_idx]
            
            # 開発者特徴量 (10次元)
            obs[0] = min(1.0, (dev['changes_authored'] + dev['changes_reviewed']) / 100.0)  # 活動レベル
            obs[1] = min(1.0, dev['changes_authored'] / 50.0)  # 作成経験
            obs[2] = min(1.0, dev['changes_reviewed'] / 100.0)  # レビュー経験
            obs[3] = len(dev['projects']) / 10.0  # プロジェクト多様性
            obs[4] = self.developer_stress_levels.get(self.current_developer_idx, 0.0)  # 現在のストレス
            obs[5:10] = np.random.random(5) * 0.1  # その他の特徴量（ランダム）
        
        if (self.current_task_idx < len(self.reviews_data)):
            task = self.reviews_data[self.current_task_idx]
            
            # タスク特徴量 (5次元)
            obs[10] = min(1.0, task.get('lines_added', 0) / 1000.0)  # タスクサイズ
            obs[11] = min(1.0, task.get('files_changed', 0) / 20.0)  # 複雑度
            obs[12] = 1.0 if task.get('status') == 'NEW' else 0.0  # 緊急度
            obs[13] = abs(task.get('score', 0)) / 2.0  # 重要度
            obs[14] = np.random.random() * 0.1  # その他
        
        # 環境状態 (5次元)
        obs[15] = self.episode_step / self.max_episode_steps  # 進行度
        obs[16] = self.successful_assignments / max(1, self.total_assignments)  # 成功率
        obs[17] = np.mean(list(self.developer_stress_levels.values()))  # 平均ストレス
        obs[18] = self.current_developer_idx / len(self.developers_data)  # 開発者進行度
        obs[19] = self.current_task_idx / max(1, len(self.reviews_data))  # タスク進行度
        
        return obs
    
    def _calculate_assignment_reward(self, developer: Dict, task: Dict) -> float:
        """割り当て報酬を計算"""
        reward = 0.0
        
        # 専門性マッチ報酬
        dev_activity = developer['changes_authored'] + developer['changes_reviewed']
        if dev_activity > 10:
            reward += 0.3
        
        # ワークロードバランス報酬
        current_stress = self.developer_stress_levels.get(self.current_developer_idx, 0.0)
        if current_stress < 0.7:
            reward += 0.2
        else:
            reward -= 0.3  # 高ストレス時はペナルティ
        
        # タスク適合性報酬
        task_size = task.get('lines_added', 0)
        if task_size < 100:  # 小さなタスク
            reward += 0.1
        elif task_size > 500:  # 大きなタスク
            if dev_activity > 20:  # 経験豊富な開発者のみ
                reward += 0.2
            else:
                reward -= 0.2
        
        # 協力関係報酬
        if developer['changes_reviewed'] > 0:
            reward += 0.1
        
        return reward
    
    def _calculate_stress_increase(self, developer: Dict, task: Dict) -> float:
        """ストレス増加量を計算"""
        base_stress = 0.1
        
        # タスクサイズによるストレス
        task_size = task.get('lines_added', 0)
        if task_size > 500:
            base_stress += 0.2
        
        # 開発者の経験によるストレス軽減
        dev_activity = developer['changes_authored'] + developer['changes_reviewed']
        if dev_activity > 50:
            base_stress *= 0.7
        
        return base_stress
    
    def _get_info(self) -> Dict:
        """追加情報を取得"""
        return {
            'total_assignments': self.total_assignments,
            'successful_assignments': self.successful_assignments,
            'success_rate': self.successful_assignments / max(1, self.total_assignments),
            'average_stress': np.mean(list(self.developer_stress_levels.values())),
            'episode_step': self.episode_step
        }

class SimplePPOAgent:
    """
    簡易PPOエージェント実装
    """
    
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        self.observation_space = observation_space
        self.action_space = action_space
        
        # 簡易ポリシーネットワーク（線形モデル）
        self.policy_weights = np.random.randn(observation_space.shape[0], action_space.n) * 0.1
        self.value_weights = np.random.randn(observation_space.shape[0]) * 0.1
        
        # ハイパーパラメータ
        self.learning_rate = 0.001
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        
        logger.info(f"PPOエージェントを初期化: 観測次元{observation_space.shape[0]}, 行動数{action_space.n}")
    
    def get_action(self, observation: np.ndarray) -> Tuple[int, float]:
        """行動を選択"""
        # ポリシー計算（ソフトマックス）
        logits = np.dot(observation, self.policy_weights)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # 行動をサンプリング
        action = np.random.choice(self.action_space.n, p=probs)
        action_prob = probs[action]
        
        return action, action_prob
    
    def get_value(self, observation: np.ndarray) -> float:
        """状態価値を計算"""
        return np.dot(observation, self.value_weights)
    
    def update(self, observations: List[np.ndarray], actions: List[int], 
               rewards: List[float], action_probs: List[float]):
        """ポリシーを更新（簡易版）"""
        
        # 報酬の割引計算
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        # 正規化
        discounted_rewards = np.array(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # 簡易ポリシー更新
        for i, (obs, action, reward, old_prob) in enumerate(zip(observations, actions, discounted_rewards, action_probs)):
            # 現在のポリシー確率
            logits = np.dot(obs, self.policy_weights)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            new_prob = probs[action]
            
            # PPOクリッピング
            ratio = new_prob / (old_prob + 1e-8)
            clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            # ポリシー勾配
            advantage = reward - self.get_value(obs)
            policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
            
            # 重み更新（簡易版）
            grad = np.outer(obs, np.zeros(self.action_space.n))
            grad[:, action] = obs * policy_loss * self.learning_rate
            self.policy_weights -= grad
            
            # 価値関数更新
            value_loss = (reward - self.get_value(obs)) ** 2
            value_grad = obs * value_loss * self.learning_rate
            self.value_weights -= value_grad

def train_real_rl_agent():
    """実際の強化学習エージェントを訓練"""
    
    logger.info("=== 実際の強化学習訓練開始 ===")
    
    # 実際のデータを読み込み
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
        
        if not is_bot:
            human_developers.append(dev)
    
    logger.info(f"訓練データ: {len(human_developers)}名の開発者, {len(reviews_data)}件のレビュー")
    
    # 環境とエージェントを作成
    env = DeveloperTaskEnvironment(human_developers, reviews_data)
    agent = SimplePPOAgent(env.observation_space, env.action_space)
    
    # 訓練ループ
    num_episodes = 50
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        observations = []
        actions = []
        rewards = []
        action_probs = []
        
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            # 行動選択
            action, action_prob = agent.get_action(obs)
            
            # 環境でステップ実行
            next_obs, reward, done, _, info = env.step(action)
            
            # データを記録
            observations.append(obs.copy())
            actions.append(action)
            rewards.append(reward)
            action_probs.append(action_prob)
            
            episode_reward += reward
            obs = next_obs
            step_count += 1
        
        # エージェントを更新
        if len(observations) > 0:
            agent.update(observations, actions, rewards, action_probs)
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"エピソード {episode}: 平均報酬 {avg_reward:.3f}, 成功率 {info.get('success_rate', 0):.3f}")
    
    # 訓練結果を保存
    training_results = {
        'agent_type': 'real_ppo_agent',
        'version': '2.0.0',
        'trained_at': str(np.datetime64('now')),
        'training_episodes': num_episodes,
        'final_average_reward': np.mean(episode_rewards[-10:]),
        'episode_rewards': episode_rewards,
        'hyperparameters': {
            'learning_rate': agent.learning_rate,
            'clip_epsilon': agent.clip_epsilon,
            'gamma': agent.gamma
        },
        'environment_info': {
            'num_developers': len(human_developers),
            'num_tasks': len(reviews_data),
            'observation_space_dim': env.observation_space.shape[0],
            'action_space_size': env.action_space.n
        }
    }
    
    # モデルを保存
    model_data = {
        'policy_weights': agent.policy_weights.tolist(),
        'value_weights': agent.value_weights.tolist(),
        'training_results': training_results
    }
    
    with open('models/real_ppo_agent.json', 'w') as f:
        json.dump(model_data, f, indent=2)
    
    logger.info(f"実際の強化学習訓練完了: 最終平均報酬 {training_results['final_average_reward']:.3f}")
    
    return training_results

if __name__ == "__main__":
    train_real_rl_agent()