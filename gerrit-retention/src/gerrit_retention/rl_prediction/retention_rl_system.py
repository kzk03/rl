"""
継続予測のための強化学習システム

継続予測を強化学習問題として定式化し、
予測精度を最大化するポリシーを学習する。
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

logger = logging.getLogger(__name__)


class RetentionPredictionEnv(gym.Env):
    """継続予測環境"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # 状態空間: 開発者の特徴量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(config.get('state_dim', 15),), 
            dtype=np.float32
        )
        
        # 行動空間: 継続確率の予測 [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # 環境状態
        self.current_developer = None
        self.current_activity_history = None
        self.true_continuation = None
        self.prediction_horizon = 30  # デフォルト30日後
        
        # データセット
        self.training_data = []
        self.current_episode = 0
        
        logger.info("継続予測環境を初期化しました")
    
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        
        super().reset(seed=seed)
        
        if not self.training_data:
            # デフォルトの開発者データ
            self.current_developer = {
                'developer_id': 'default@example.com',
                'changes_authored': 50,
                'changes_reviewed': 30,
                'projects': ['project-a']
            }
            self.current_activity_history = []
            self.true_continuation = True
        else:
            # ランダムに開発者を選択
            data_point = self.np_random.choice(self.training_data)
            self.current_developer = data_point['developer']
            self.current_activity_history = data_point['activity_history']
            self.true_continuation = data_point.get('continued', True)
        
        # 状態を計算
        state = self._compute_state()
        
        info = {
            'developer_id': self.current_developer.get('developer_id', 'unknown'),
            'true_continuation': self.true_continuation
        }
        
        return state, info
    
    def step(self, action):
        """ステップ実行"""
        
        # 行動（予測確率）を取得
        predicted_prob = float(action[0])
        predicted_prob = np.clip(predicted_prob, 0.0, 1.0)
        
        # 報酬計算
        reward = self._calculate_reward(predicted_prob, self.true_continuation)
        
        # エピソード終了
        terminated = True
        truncated = False
        
        # 次の状態（エピソード終了なので意味なし）
        next_state = self._compute_state()
        
        info = {
            'predicted_probability': predicted_prob,
            'true_continuation': self.true_continuation,
            'prediction_error': abs(predicted_prob - float(self.true_continuation)),
            'reward_components': self._get_reward_components(predicted_prob, self.true_continuation)
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _compute_state(self) -> np.ndarray:
        """現在の状態を計算"""
        
        if not self.current_developer:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # 基本特徴量
        features = []
        
        # 開発者基本情報
        features.append(self.current_developer.get('changes_authored', 0) / 100.0)  # 正規化
        features.append(self.current_developer.get('changes_reviewed', 0) / 100.0)  # 正規化
        features.append(len(self.current_developer.get('projects', [])) / 10.0)     # 正規化
        
        # 経験日数
        first_seen = self.current_developer.get('first_seen', datetime.now().isoformat())
        if isinstance(first_seen, str):
            first_date = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
        else:
            first_date = first_seen
        experience_days = (datetime.now() - first_date).days
        features.append(experience_days / 365.0)  # 正規化（年単位）
        
        # 活動履歴の特徴量
        if self.current_activity_history:
            # 活動数
            features.append(len(self.current_activity_history) / 50.0)  # 正規化
            
            # 最近の活動頻度
            recent_activities = self._get_recent_activities(30)
            features.append(len(recent_activities) / 30.0)  # 日次頻度
            
            # 活動の多様性
            activity_types = set(a.get('type', 'unknown') for a in self.current_activity_history)
            features.append(len(activity_types) / 10.0)  # 正規化
            
            # 活動間隔の統計
            gaps = self._calculate_activity_gaps()
            if gaps:
                features.append(np.mean(gaps) / 30.0)  # 平均間隔（月単位）
                features.append(np.std(gaps) / 30.0)   # 間隔の標準偏差
                features.append(max(gaps) / 90.0)      # 最大間隔（四半期単位）
            else:
                features.extend([1.0, 0.0, 1.0])  # デフォルト値
            
            # 協力関連活動の割合
            collaboration_types = ['review', 'merge', 'collaboration', 'mentoring']
            collaboration_count = sum(
                1 for a in self.current_activity_history 
                if a.get('type', '').lower() in collaboration_types
            )
            features.append(collaboration_count / len(self.current_activity_history))
            
            # コード品質関連活動の割合
            quality_keywords = ['fix', 'test', 'refactor', 'improve', 'optimize']
            quality_count = 0
            for activity in self.current_activity_history:
                message = activity.get('message', '').lower()
                if any(keyword in message for keyword in quality_keywords):
                    quality_count += 1
            features.append(quality_count / len(self.current_activity_history))
            
        else:
            # 活動履歴がない場合のデフォルト値
            features.extend([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        
        # 時期的要因
        current_month = datetime.now().month
        features.append(current_month / 12.0)  # 月の正規化
        features.append(np.sin(2 * np.pi * current_month / 12))  # 季節性（sin）
        features.append(np.cos(2 * np.pi * current_month / 12))  # 季節性（cos）
        
        # 特徴量数を調整
        while len(features) < self.observation_space.shape[0]:
            features.append(0.0)
        
        features = features[:self.observation_space.shape[0]]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, predicted_prob: float, true_continuation: bool) -> float:
        """報酬を計算"""
        
        true_value = 1.0 if true_continuation else 0.0
        
        # 基本的な予測精度報酬
        prediction_error = abs(predicted_prob - true_value)
        accuracy_reward = 1.0 - prediction_error
        
        # 校正報酬（確率の校正精度）
        if true_continuation:
            # 継続した場合、高い確率を予測していれば高報酬
            calibration_reward = predicted_prob
        else:
            # 離脱した場合、低い確率を予測していれば高報酬
            calibration_reward = 1.0 - predicted_prob
        
        # 信頼度報酬（極端な予測を奨励）
        confidence_reward = abs(predicted_prob - 0.5) * 2  # 0.5から離れるほど高報酬
        
        # 総合報酬
        total_reward = (
            accuracy_reward * 0.5 +      # 精度重視
            calibration_reward * 0.3 +   # 校正重視
            confidence_reward * 0.2      # 信頼度重視
        )
        
        return total_reward
    
    def _get_reward_components(self, predicted_prob: float, true_continuation: bool) -> Dict[str, float]:
        """報酬の内訳を取得"""
        
        true_value = 1.0 if true_continuation else 0.0
        prediction_error = abs(predicted_prob - true_value)
        
        return {
            'accuracy_reward': 1.0 - prediction_error,
            'calibration_reward': predicted_prob if true_continuation else 1.0 - predicted_prob,
            'confidence_reward': abs(predicted_prob - 0.5) * 2,
            'prediction_error': prediction_error
        }
    
    def _get_recent_activities(self, days: int) -> List[Dict[str, Any]]:
        """最近の活動を取得"""
        
        if not self.current_activity_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_activities = []
        
        for activity in self.current_activity_history:
            try:
                timestamp_str = activity.get('timestamp', datetime.now().isoformat())
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                if timestamp >= cutoff_date:
                    recent_activities.append(activity)
            except:
                continue
        
        return recent_activities
    
    def _calculate_activity_gaps(self) -> List[float]:
        """活動間隔を計算"""
        
        if not self.current_activity_history:
            return []
        
        timestamps = []
        for activity in self.current_activity_history:
            try:
                timestamp_str = activity.get('timestamp')
                if timestamp_str:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = timestamp_str
                    timestamps.append(timestamp)
            except:
                continue
        
        if len(timestamps) < 2:
            return []
        
        timestamps.sort()
        gaps = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).days
            gaps.append(gap)
        
        return gaps
    
    def set_training_data(self, training_data: List[Dict[str, Any]]) -> None:
        """訓練データを設定"""
        self.training_data = training_data
        logger.info(f"訓練データを設定しました: {len(training_data)}件")


class RetentionPolicyNetwork(nn.Module):
    """継続予測ポリシーネットワーク"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 確率出力
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RetentionRLSystem:
    """継続予測強化学習システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 環境設定
        self.env = RetentionPredictionEnv(config)
        
        # ネットワーク設定
        self.state_dim = config.get('state_dim', 15)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ポリシーネットワーク
        self.policy_network = RetentionPolicyNetwork(
            self.state_dim, self.hidden_dim
        ).to(self.device)
        
        # オプティマイザー
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        # 訓練統計
        self.training_stats = {
            'episode_rewards': [],
            'prediction_errors': [],
            'accuracy_scores': []
        }
        
        logger.info("継続予測RLシステムを初期化しました")
    
    def train_rl(self, 
                 training_data: List[Dict[str, Any]], 
                 episodes: int = 1000) -> Dict[str, Any]:
        """
        強化学習で訓練
        
        Args:
            training_data: 訓練データ
            episodes: 訓練エピソード数
            
        Returns:
            Dict[str, Any]: 訓練結果
        """
        logger.info(f"RL訓練開始: {len(training_data)}データ, {episodes}エピソード")
        
        # 環境に訓練データを設定
        self.env.set_training_data(training_data)
        
        for episode in range(episodes):
            # エピソード開始
            state, info = self.env.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # ポリシーで行動選択
            with torch.no_grad():
                predicted_prob = self.policy_network(state_tensor)
                action = predicted_prob.cpu().numpy()
            
            # ステップ実行
            next_state, reward, terminated, truncated, step_info = self.env.step(action)
            
            # 損失計算（ポリシー勾配）
            predicted_prob_tensor = self.policy_network(state_tensor)
            
            # 報酬に基づく損失（高報酬の行動を強化）
            loss = -torch.log(predicted_prob_tensor) * reward
            if not step_info['true_continuation']:
                # 離脱の場合は (1-p) の対数尤度
                loss = -torch.log(1 - predicted_prob_tensor) * reward
            
            # バックプロパゲーション
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 統計更新
            self.training_stats['episode_rewards'].append(reward)
            self.training_stats['prediction_errors'].append(step_info['prediction_error'])
            
            # 精度計算
            predicted_label = predicted_prob_tensor.item() > 0.5
            true_label = step_info['true_continuation']
            accuracy = 1.0 if predicted_label == true_label else 0.0
            self.training_stats['accuracy_scores'].append(accuracy)
            
            # ログ出力
            if episode % 100 == 0:
                recent_rewards = self.training_stats['episode_rewards'][-100:]
                recent_accuracy = self.training_stats['accuracy_scores'][-100:]
                avg_reward = np.mean(recent_rewards)
                avg_accuracy = np.mean(recent_accuracy)
                
                logger.info(f"エピソード {episode}: 平均報酬={avg_reward:.3f}, 平均精度={avg_accuracy:.3f}")
        
        logger.info("RL訓練完了")
        
        return {
            'final_avg_reward': np.mean(self.training_stats['episode_rewards'][-100:]),
            'final_avg_accuracy': np.mean(self.training_stats['accuracy_scores'][-100:]),
            'training_stats': self.training_stats
        }
    
    def predict_continuation_probability(self, 
                                       developer: Dict[str, Any], 
                                       activity_history: List[Dict[str, Any]],
                                       context_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        継続確率を予測
        
        Args:
            developer: 開発者データ
            activity_history: 活動履歴
            context_date: 基準日
            
        Returns:
            Dict[str, Any]: 予測結果
        """
        if context_date is None:
            context_date = datetime.now()
        
        self.policy_network.eval()
        
        # 環境に開発者データを設定
        self.env.current_developer = developer
        self.env.current_activity_history = activity_history
        
        # 状態を計算
        state = self.env._compute_state()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # 予測実行
            predicted_prob = self.policy_network(state_tensor)
            continuation_prob = predicted_prob.item()
            
            # 信頼度計算
            confidence = abs(continuation_prob - 0.5) * 2
            
            # 理由生成
            reasoning = self._generate_rl_reasoning(developer, activity_history, continuation_prob)
            
            return {
                'continuation_probability': continuation_prob,
                'confidence': confidence,
                'reasoning': reasoning,
                'method': 'reinforcement_learning',
                'state_features': state.tolist()
            }
    
    def _generate_rl_reasoning(self, 
                             developer: Dict[str, Any], 
                             activity_history: List[Dict[str, Any]], 
                             continuation_prob: float) -> str:
        """RL予測の理由を生成"""
        
        reasoning_parts = []
        
        # 基本情報
        total_activity = developer.get('changes_authored', 0) + developer.get('changes_reviewed', 0)
        if total_activity > 100:
            reasoning_parts.append("高い活動量により継続確率が向上")
        elif total_activity < 20:
            reasoning_parts.append("低い活動量により継続確率が低下")
        
        # プロジェクト関与
        project_count = len(developer.get('projects', []))
        if project_count > 3:
            reasoning_parts.append("複数プロジェクトへの関与により継続確率が向上")
        elif project_count == 0:
            reasoning_parts.append("プロジェクト関与なしにより継続確率が低下")
        
        # 活動履歴
        if activity_history:
            recent_activities = len([
                a for a in activity_history[-10:] 
                if 'timestamp' in a
            ])
            if recent_activities > 5:
                reasoning_parts.append("最近の活発な活動により継続確率が向上")
            elif recent_activities < 2:
                reasoning_parts.append("最近の活動不足により継続確率が低下")
        
        # 強化学習の判定
        if continuation_prob > 0.7:
            reasoning_parts.append("強化学習モデルが高い継続確率を予測")
        elif continuation_prob < 0.3:
            reasoning_parts.append("強化学習モデルが低い継続確率を予測")
        else:
            reasoning_parts.append("強化学習モデルが中程度の継続確率を予測")
        
        reasoning_parts.append(f"RL予測継続確率: {continuation_prob:.1%}")
        
        return "。".join(reasoning_parts)
    
    def save_model(self, filepath: str) -> None:
        """モデルを保存"""
        
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
        
        logger.info(f"RLモデルを保存しました: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """モデルを読み込み"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        logger.info(f"RLモデルを読み込みました: {filepath}")


if __name__ == "__main__":
    # テスト用の設定
    config = {
        'state_dim': 15,
        'hidden_dim': 128,
        'learning_rate': 0.001
    }
    
    # RLシステムを初期化
    rl_system = RetentionRLSystem(config)
    
    print("継続予測RLシステムのテスト完了")
    print(f"ポリシーネットワーク: {rl_system.policy_network}")
    print(f"環境: {rl_system.env}")
    print(f"デバイス: {rl_system.device}")