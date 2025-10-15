"""
継続予測のための逆強化学習システム

優秀な開発者（継続した開発者）の行動パターンから
継続に寄与する要因の報酬関数を学習し、
それを基に継続確率を予測する。
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


@dataclass
class DeveloperState:
    """開発者の状態表現"""
    developer_id: str
    experience_days: int
    total_changes: int
    total_reviews: int
    project_count: int
    recent_activity_frequency: float
    avg_activity_gap: float
    activity_trend: str
    collaboration_score: float
    code_quality_score: float
    timestamp: datetime


@dataclass
class DeveloperAction:
    """開発者の行動表現"""
    action_type: str  # 'commit', 'review', 'merge', 'documentation', etc.
    intensity: float  # 行動の強度（コード行数、レビューコメント数など）
    quality: float    # 行動の質
    collaboration: float  # 協力度
    timestamp: datetime


class RetentionIRLNetwork(nn.Module):
    """継続予測のためのIRLネットワーク (拡張: 時系列対応)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, sequence: bool = False, seq_len: int = 10):
        super().__init__()
        self.sequence = sequence
        self.seq_len = seq_len
        
        # 状態エンコーダー
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 行動エンコーダー
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        if self.sequence:
            # LSTM for sequence
            self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim, num_layers=1, batch_first=True)
        
        # 報酬予測器
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 継続確率予測器
        self.continuation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, return_hidden: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向き計算 (拡張: 時系列対応)
        
        Args:
            state: 開発者状態 [batch_size, seq_len, state_dim] if sequence else [batch_size, state_dim]
            action: 開発者行動 [batch_size, seq_len, action_dim] if sequence else [batch_size, action_dim]
            return_hidden: 隠れ状態も返すかどうか
            
        Returns:
            reward: 予測報酬 [batch_size, 1]
            continuation_prob: 継続確率 [batch_size, 1]
            hidden: 隠れ状態 [batch_size, hidden_dim] (return_hidden=Trueの場合)
        """
        if self.sequence:
            # Sequence mode: (batch, seq, dim)
            batch_size, seq_len, _ = state.shape
            state_encoded = self.state_encoder(state.view(-1, state.shape[-1])).view(batch_size, seq_len, -1)
            action_encoded = self.action_encoder(action.view(-1, action.shape[-1])).view(batch_size, seq_len, -1)
            
            combined = state_encoded + action_encoded  # Simple addition
            lstm_out, _ = self.lstm(combined)
            hidden = lstm_out[:, -1, :]  # Last timestep
        else:
            # Single step mode
            state_encoded = self.state_encoder(state)
            action_encoded = self.action_encoder(action)
            combined = torch.cat([state_encoded, action_encoded], dim=1)  # Concat for full hidden_dim
            hidden = combined
        
        reward = self.reward_predictor(hidden)
        continuation_prob = self.continuation_predictor(hidden)
        
        if return_hidden:
            return reward, continuation_prob, hidden
        else:
            return reward, continuation_prob


class RetentionIRLSystem:
    """継続予測IRL システム (拡張: 時系列対応)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # ネットワーク設定
        self.state_dim = config.get('state_dim', 10)
        self.action_dim = config.get('action_dim', 5)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.sequence = config.get('sequence', False)
        self.seq_len = config.get('seq_len', 10)
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ネットワーク初期化
        self.network = RetentionIRLNetwork(
            self.state_dim, self.action_dim, self.hidden_dim, self.sequence, self.seq_len
        ).to(self.device)
        
        # オプティマイザー
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        
        # 損失関数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        logger.info("継続予測IRLシステムを初期化しました")
    
    def extract_developer_state(self, 
                               developer: Dict[str, Any], 
                               activity_history: List[Dict[str, Any]],
                               context_date: datetime) -> DeveloperState:
        """開発者の状態を抽出"""
        
        # 経験日数
        first_seen = developer.get('first_seen', context_date.isoformat())
        if isinstance(first_seen, str):
            first_date = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
        else:
            first_date = first_seen
        experience_days = (context_date - first_date).days
        
        # 活動統計
        total_changes = developer.get('changes_authored', 0)
        total_reviews = developer.get('changes_reviewed', 0)
        projects = developer.get('projects', [])
        project_count = len(projects) if isinstance(projects, list) else 0
        
        # 最近の活動パターン
        recent_activities = self._get_recent_activities(activity_history, context_date, days=30)
        recent_activity_frequency = len(recent_activities) / 30.0
        
        # 活動間隔
        activity_gaps = self._calculate_activity_gaps(activity_history)
        avg_activity_gap = np.mean(activity_gaps) if activity_gaps else 30.0
        
        # 活動トレンド
        activity_trend = self._analyze_activity_trend(activity_history, context_date)
        
        # 協力スコア（簡易版）
        collaboration_score = self._calculate_collaboration_score(activity_history)
        
        # コード品質スコア（簡易版）
        code_quality_score = self._calculate_code_quality_score(activity_history)
        
        return DeveloperState(
            developer_id=developer.get('developer_id', 'unknown'),
            experience_days=experience_days,
            total_changes=total_changes,
            total_reviews=total_reviews,
            project_count=project_count,
            recent_activity_frequency=recent_activity_frequency,
            avg_activity_gap=avg_activity_gap,
            activity_trend=activity_trend,
            collaboration_score=collaboration_score,
            code_quality_score=code_quality_score,
            timestamp=context_date
        )
    
    def extract_developer_actions(self, 
                                activity_history: List[Dict[str, Any]],
                                context_date: datetime) -> List[DeveloperAction]:
        """開発者の行動を抽出"""
        
        actions = []
        
        for activity in activity_history:
            try:
                # 行動タイプ
                action_type = activity.get('type', 'unknown')
                
                # 行動の強度
                intensity = self._calculate_action_intensity(activity)
                
                # 行動の質
                quality = self._calculate_action_quality(activity)
                
                # 協力度
                collaboration = self._calculate_action_collaboration(activity)
                
                # タイムスタンプ
                timestamp_str = activity.get('timestamp', context_date.isoformat())
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                actions.append(DeveloperAction(
                    action_type=action_type,
                    intensity=intensity,
                    quality=quality,
                    collaboration=collaboration,
                    timestamp=timestamp
                ))
                
            except Exception as e:
                logger.warning(f"行動抽出エラー: {e}")
                continue
        
        return actions
    
    def state_to_tensor(self, state: DeveloperState) -> torch.Tensor:
        """状態をテンソルに変換"""
        
        # 活動トレンドのエンコーディング
        trend_encoding = {
            'increasing': 1.0,
            'stable': 0.5,
            'decreasing': 0.0,
            'unknown': 0.25
        }
        
        features = [
            state.experience_days / 365.0,  # 正規化（年単位）
            state.total_changes / 100.0,    # 正規化
            state.total_reviews / 100.0,    # 正規化
            state.project_count / 10.0,     # 正規化
            state.recent_activity_frequency,
            state.avg_activity_gap / 30.0,  # 正規化（月単位）
            trend_encoding.get(state.activity_trend, 0.25),
            state.collaboration_score,
            state.code_quality_score,
            (datetime.now() - state.timestamp).days / 365.0  # 時間経過
        ]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def action_to_tensor(self, action: DeveloperAction) -> torch.Tensor:
        """行動をテンソルに変換"""
        
        # 行動タイプのエンコーディング
        type_encoding = {
            'commit': 1.0,
            'review': 0.8,
            'merge': 0.9,
            'documentation': 0.6,
            'issue': 0.4,
            'collaboration': 0.7,
            'unknown': 0.1
        }
        
        features = [
            type_encoding.get(action.action_type, 0.1),
            action.intensity,
            action.quality,
            action.collaboration,
            (datetime.now() - action.timestamp).days / 365.0  # 時間経過
        ]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def train_irl(self, 
                  expert_trajectories: List[Dict[str, Any]], 
                  epochs: int = 100) -> Dict[str, Any]:
        """
        IRLモデルを訓練
        
        Args:
            expert_trajectories: エキスパート軌跡データ
            epochs: 訓練エポック数
            
        Returns:
            Dict[str, Any]: 訓練結果
        """
        logger.info(f"IRL訓練開始: {len(expert_trajectories)}軌跡, {epochs}エポック")
        
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for trajectory in expert_trajectories:
                try:
                    # 軌跡から状態と行動を抽出
                    developer = trajectory['developer']
                    activity_history = trajectory['activity_history']
                    continuation_label = trajectory.get('continued', True)  # 継続ラベル
                    context_date = trajectory.get('context_date', datetime.now())
                    
                    # 状態と行動を抽出
                    state = self.extract_developer_state(developer, activity_history, context_date)
                    actions = self.extract_developer_actions(activity_history, context_date)
                    
                    if not actions:
                        continue
                    
                    # テンソルに変換
                    state_tensor = self.state_to_tensor(state).unsqueeze(0)
                    
                    # 各行動に対して学習
                    for action in actions[-5:]:  # 最近の5つの行動
                        action_tensor = self.action_to_tensor(action).unsqueeze(0)
                        
                        # 前向き計算
                        predicted_reward, predicted_continuation = self.network(
                            state_tensor, action_tensor
                        )
                        
                        # 損失計算
                        # 継続した開発者の行動には高い報酬を与える
                        target_reward = torch.tensor(
                            [[1.0 if continuation_label else 0.0]], 
                            device=self.device
                        )
                        target_continuation = torch.tensor(
                            [[1.0 if continuation_label else 0.0]], 
                            device=self.device
                        )
                        
                        reward_loss = self.mse_loss(predicted_reward, target_reward)
                        continuation_loss = self.bce_loss(predicted_continuation, target_continuation)
                        
                        total_loss = reward_loss + continuation_loss
                        
                        # バックプロパゲーション
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        self.optimizer.step()
                        
                        epoch_loss += total_loss.item()
                        batch_count += 1
                
                except Exception as e:
                    logger.warning(f"軌跡処理エラー: {e}")
                    continue
            
            avg_loss = epoch_loss / max(batch_count, 1)
            training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"エポック {epoch}: 平均損失 = {avg_loss:.4f}")
        
        logger.info("IRL訓練完了")
        
        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1] if training_losses else 0.0,
            'epochs_trained': epochs
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
        
        self.network.eval()
        
        with torch.no_grad():
            # 状態と行動を抽出
            state = self.extract_developer_state(developer, activity_history, context_date)
            actions = self.extract_developer_actions(activity_history, context_date)
            
            if not actions:
                return {
                    'continuation_probability': 0.5,
                    'confidence': 0.0,
                    'reasoning': '活動履歴が不足しているため、デフォルト確率を返します'
                }
            
            # 最近の行動を使用
            recent_action = actions[-1]
            
            # テンソルに変換
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            action_tensor = self.action_to_tensor(recent_action).unsqueeze(0)
            
            # 予測実行
            predicted_reward, predicted_continuation = self.network(
                state_tensor, action_tensor
            )
            
            continuation_prob = predicted_continuation.item()
            reward_score = predicted_reward.item()
            
            # 信頼度計算（簡易版）
            confidence = min(abs(continuation_prob - 0.5) * 2, 1.0)
            
            # 理由生成
            reasoning = self._generate_irl_reasoning(
                state, recent_action, continuation_prob, reward_score
            )
            
            return {
                'continuation_probability': continuation_prob,
                'reward_score': reward_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'state_features': {
                    'experience_days': state.experience_days,
                    'recent_activity_frequency': state.recent_activity_frequency,
                    'collaboration_score': state.collaboration_score,
                    'code_quality_score': state.code_quality_score
                }
            }
    
    def _get_recent_activities(self, 
                             activity_history: List[Dict[str, Any]], 
                             context_date: datetime, 
                             days: int = 30) -> List[Dict[str, Any]]:
        """最近の活動を取得"""
        
        cutoff_date = context_date - timedelta(days=days)
        recent_activities = []
        
        for activity in activity_history:
            try:
                timestamp_str = activity.get('timestamp', context_date.isoformat())
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                if timestamp >= cutoff_date:
                    recent_activities.append(activity)
            except:
                continue
        
        return recent_activities
    
    def _calculate_activity_gaps(self, activity_history: List[Dict[str, Any]]) -> List[float]:
        """活動間隔を計算"""
        
        timestamps = []
        for activity in activity_history:
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
    
    def _analyze_activity_trend(self, 
                              activity_history: List[Dict[str, Any]], 
                              context_date: datetime) -> str:
        """活動トレンドを分析"""
        
        # 最近30日と過去30-60日を比較
        recent_activities = self._get_recent_activities(activity_history, context_date, 30)
        past_activities = self._get_recent_activities(activity_history, context_date - timedelta(days=30), 30)
        
        recent_count = len(recent_activities)
        past_count = len(past_activities)
        
        if past_count == 0:
            return 'unknown'
        
        ratio = recent_count / past_count
        
        if ratio > 1.2:
            return 'increasing'
        elif ratio < 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_collaboration_score(self, activity_history: List[Dict[str, Any]]) -> float:
        """協力スコアを計算（簡易版）"""
        
        collaboration_activities = ['review', 'merge', 'collaboration', 'mentoring']
        total_activities = len(activity_history)
        
        if total_activities == 0:
            return 0.0
        
        collaboration_count = sum(
            1 for activity in activity_history 
            if activity.get('type', '').lower() in collaboration_activities
        )
        
        return collaboration_count / total_activities
    
    def _calculate_code_quality_score(self, activity_history: List[Dict[str, Any]]) -> float:
        """コード品質スコアを計算（簡易版）"""
        
        quality_indicators = ['test', 'documentation', 'refactor', 'fix']
        total_activities = len(activity_history)
        
        if total_activities == 0:
            return 0.5
        
        quality_count = 0
        for activity in activity_history:
            message = activity.get('message', '').lower()
            if any(indicator in message for indicator in quality_indicators):
                quality_count += 1
        
        return min(quality_count / total_activities + 0.3, 1.0)
    
    def _calculate_action_intensity(self, activity: Dict[str, Any]) -> float:
        """行動の強度を計算"""
        
        lines_added = activity.get('lines_added', 0)
        lines_deleted = activity.get('lines_deleted', 0)
        files_changed = activity.get('files_changed', 1)
        
        # 正規化された強度
        intensity = min((lines_added + lines_deleted) / (files_changed * 50), 1.0)
        return max(intensity, 0.1)
    
    def _calculate_action_quality(self, activity: Dict[str, Any]) -> float:
        """行動の質を計算"""
        
        message = activity.get('message', '').lower()
        quality_keywords = ['fix', 'improve', 'optimize', 'test', 'document', 'refactor']
        
        quality_score = 0.5  # ベーススコア
        
        for keyword in quality_keywords:
            if keyword in message:
                quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _calculate_action_collaboration(self, activity: Dict[str, Any]) -> float:
        """行動の協力度を計算"""
        
        action_type = activity.get('type', '').lower()
        collaboration_types = {
            'review': 0.8,
            'merge': 0.7,
            'collaboration': 1.0,
            'mentoring': 0.9,
            'documentation': 0.6
        }
        
        return collaboration_types.get(action_type, 0.3)
    
    def _generate_irl_reasoning(self, 
                              state: DeveloperState, 
                              action: DeveloperAction, 
                              continuation_prob: float,
                              reward_score: float) -> str:
        """IRL予測の理由を生成"""
        
        reasoning_parts = []
        
        # 経験レベル
        if state.experience_days > 365:
            reasoning_parts.append("豊富な経験により継続確率が向上")
        elif state.experience_days < 90:
            reasoning_parts.append("経験が浅いため継続確率がやや低下")
        
        # 活動パターン
        if state.recent_activity_frequency > 0.1:
            reasoning_parts.append("高い活動頻度により継続確率が向上")
        elif state.recent_activity_frequency < 0.03:
            reasoning_parts.append("低い活動頻度により継続確率が低下")
        
        # 協力度
        if state.collaboration_score > 0.5:
            reasoning_parts.append("高い協力度により継続確率が向上")
        
        # 最近の行動
        if action.quality > 0.7:
            reasoning_parts.append("高品質な最近の行動により継続確率が向上")
        
        # 報酬スコア
        if reward_score > 0.7:
            reasoning_parts.append("学習された報酬関数により高い継続価値を予測")
        elif reward_score < 0.3:
            reasoning_parts.append("学習された報酬関数により低い継続価値を予測")
        
        reasoning_parts.append(f"IRL予測継続確率: {continuation_prob:.1%}")
        
        return "。".join(reasoning_parts)
    
    def save_model(self, filepath: str) -> None:
        """モデルを保存"""
        
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        
        logger.info(f"IRLモデルを保存しました: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RetentionIRLSystem':
        """モデルを読み込み (クラスメソッド)"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(filepath, map_location=device)
        
        config = checkpoint.get('config', {
            'state_dim': 20,
            'action_dim': 3,
            'hidden_dim': 128,
            'learning_rate': 0.001
        })
        
        instance = cls(config)
        instance.network.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"IRLモデルを読み込みました: {filepath}")
        return instance


if __name__ == "__main__":
    # テスト用の設定
    config = {
        'state_dim': 10,
        'action_dim': 5,
        'hidden_dim': 128,
        'learning_rate': 0.001
    }
    
    # IRLシステムを初期化
    irl_system = RetentionIRLSystem(config)
    
    print("継続予測IRLシステムのテスト完了")
    print(f"ネットワーク: {irl_system.network}")
    print(f"デバイス: {irl_system.device}")