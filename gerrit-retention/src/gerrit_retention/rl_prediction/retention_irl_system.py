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
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


@dataclass
class DeveloperState:
    """開発者の状態表現（10次元版）"""
    developer_id: str
    experience_days: int
    total_changes: int
    total_reviews: int
    # project_count: int  # マルチプロジェクト対応時に有効化（現状はプロジェクトフィルタにより常に1）
    recent_activity_frequency: float
    avg_activity_gap: float
    activity_trend: str
    collaboration_score: float
    code_quality_score: float
    recent_acceptance_rate: float  # 直近30日のレビュー受諾率
    review_load: float  # レビュー負荷（直近30日 / 平均）
    timestamp: datetime


@dataclass
class DeveloperAction:
    """開発者の行動表現（4次元版 - quality除外）"""
    action_type: str  # 'commit', 'review', 'merge', 'documentation', etc.
    intensity: float  # 行動の強度（変更ファイル数ベース）
    collaboration: float  # 協力度
    response_time: float   # レスポンス時間（日数）
    review_size: float  # レビュー規模（変更行数）✨ NEW
    timestamp: datetime


class RetentionIRLNetwork(nn.Module):
    """継続予測のためのIRLネットワーク (拡張: 時系列対応)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, sequence: bool = False, seq_len: int = 10, dropout: float = 0.1):
        super().__init__()
        self.sequence = sequence
        self.seq_len = seq_len

        # 状態エンコーダー（Dropout削減: 0.3 → 0.1）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 行動エンコーダー（Dropout削減: 0.3 → 0.1）
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.sequence:
            # LSTM for sequence（dropout付き）
            self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim, num_layers=1, batch_first=True, dropout=0.0 if dropout == 0 else dropout)

        # 報酬予測器（Dropout削減: 0.3 → 0.1）
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 継続確率予測器（Dropout削減: 0.3 → 0.1）
        self.continuation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, lengths: Optional[torch.Tensor] = None, return_hidden: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向き計算 (拡張: 時系列対応、可変長対応)
        
        Args:
            state: 開発者状態 [batch_size, seq_len, state_dim] if sequence else [batch_size, state_dim]
            action: 開発者行動 [batch_size, seq_len, action_dim] if sequence else [batch_size, action_dim]
            lengths: 各シーケンスの実際の長さ [batch_size] (可変長の場合)
            return_hidden: 隠れ状態も返すかどうか
            
        Returns:
            reward: 予測報酬 [batch_size, 1]
            continuation_prob: 継続確率 [batch_size, 1]
            hidden: 隠れ状態 [batch_size, hidden_dim] (return_hidden=Trueの場合)
        """
        if self.sequence and len(state.shape) == 3:
            # Sequence mode: (batch, seq, dim)
            batch_size, seq_len, _ = state.shape
            state_encoded = self.state_encoder(state.view(-1, state.shape[-1])).view(batch_size, seq_len, -1)
            action_encoded = self.action_encoder(action.view(-1, action.shape[-1])).view(batch_size, seq_len, -1)
            
            combined = state_encoded + action_encoded  # Simple addition
            
            if lengths is not None:
                # 可変長シーケンスの処理
                # 長さでソートして pack_padded_sequence を使用
                lengths_cpu = lengths.cpu()
                sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
                _, unsort_idx = sorted_idx.sort()
                
                combined_sorted = combined[sorted_idx]
                packed = nn.utils.rnn.pack_padded_sequence(
                    combined_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
                )
                lstm_out_packed, _ = self.lstm(packed)
                lstm_out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
                    lstm_out_packed, batch_first=True
                )
                lstm_out = lstm_out_sorted[unsort_idx]
                
                # 各シーケンスの実際の最終ステップを取得
                hidden = torch.zeros(batch_size, lstm_out.size(-1), device=state.device)
                for i in range(batch_size):
                    actual_len = lengths[i].item()
                    hidden[i] = lstm_out[i, actual_len - 1, :]
            else:
                # 固定長シーケンスの処理
                lstm_out, _ = self.lstm(combined)
                hidden = lstm_out[:, -1, :]  # Last timestep
        else:
            # Single step mode (スナップショット評価用)
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
    
    def forward_all_steps(self, state: torch.Tensor, action: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        全ステップで継続確率を予測（可変長対応）
        
        Args:
            state: [batch_size, max_seq_len, state_dim]
            action: [batch_size, max_seq_len, action_dim]
            lengths: [batch_size] 各シーケンスの実際の長さ
            
        Returns:
            predictions: [batch_size, max_seq_len] 各ステップの継続確率
        """
        batch_size, max_seq_len, _ = state.shape
        state_encoded = self.state_encoder(state.view(-1, state.shape[-1])).view(batch_size, max_seq_len, -1)
        action_encoded = self.action_encoder(action.view(-1, action.shape[-1])).view(batch_size, max_seq_len, -1)
        
        combined = state_encoded + action_encoded
        
        if lengths is not None:
            # 可変長シーケンスの処理
            lengths_cpu = lengths.cpu()
            sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
            _, unsort_idx = sorted_idx.sort()
            
            combined_sorted = combined[sorted_idx]
            packed = nn.utils.rnn.pack_padded_sequence(
                combined_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
            )
            lstm_out_packed, _ = self.lstm(packed)
            lstm_out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True, total_length=max_seq_len
            )
            lstm_out = lstm_out_sorted[unsort_idx]
        else:
            # 固定長シーケンスの処理
            lstm_out, _ = self.lstm(combined)
        
        # 各ステップで予測
        lstm_out_flat = lstm_out.reshape(-1, lstm_out.size(-1))
        predictions_flat = self.continuation_predictor(lstm_out_flat).squeeze(-1)
        predictions = predictions_flat.view(batch_size, max_seq_len)
        
        return predictions


class RetentionIRLSystem:
    """継続予測IRL システム (拡張: 時系列対応)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # ネットワーク設定（10次元状態、5次元行動）
        self.state_dim = config.get('state_dim', 10)
        self.action_dim = config.get('action_dim', 5)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.sequence = config.get('sequence', False)
        self.seq_len = config.get('seq_len', 10)
        self.dropout = config.get('dropout', 0.1)
        # 予測確率の温度スケーリング（1.0で無効、<1でシャープ、>1でフラット）
        self.output_temperature = float(config.get('output_temperature', 1.0))
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ネットワーク初期化
        self.network = RetentionIRLNetwork(
            self.state_dim, self.action_dim, self.hidden_dim, self.sequence, self.seq_len, self.dropout
        ).to(self.device)
        
        # オプティマイザー（学習率を増加: 0.001 → 0.0003がデフォルト、さらに調整可能）
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.get('learning_rate', 0.0003)
        )
        
        # 損失関数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # Focal Loss のパラメータ（デフォルト値、調整: gamma 2.0 → 1.0）
        self.focal_alpha = 0.25  # クラス重み（ポジティブクラスの重み）
        self.focal_gamma = 1.0   # フォーカスパラメータ（学習安定化のため削減）
        
        logger.info("継続予測IRLシステムを初期化しました")
    
    def set_focal_loss_params(self, alpha: float, gamma: float):
        """
        Focal Loss のパラメータを動的に設定
        
        Args:
            alpha: クラス重み（0～1、小さいほど正例重視）
            gamma: フォーカスパラメータ（0～5、大きいほど難しい例重視）
        """
        self.focal_alpha = alpha
        self.focal_gamma = gamma
        logger.info(f"Focal Loss パラメータ更新: alpha={alpha:.3f}, gamma={gamma:.3f}")
    
    def auto_tune_focal_loss(self, positive_rate: float):
        """
        正例率に応じて Focal Loss パラメータを自動調整
        
        Args:
            positive_rate: 訓練データの正例率（0～1）
        
        調整ロジック（gamma削減: 学習安定化のため）:
        - 正例率が高い（≥0.6）: alpha=0.4, gamma=1.0（バランス重視）
        - 正例率が中程度（0.3～0.6）: alpha=0.3, gamma=1.0（標準）
        - 正例率が低い（<0.3）: alpha=0.25, gamma=1.5（Recall 重視）
        """
        if positive_rate >= 0.6:
            # 正例が多い区間ではほぼバランスに近い重み
            alpha = 0.40
            gamma = 1.0
            strategy = "バランス重視（正例率≥60%・軽い正例優先)"
        elif positive_rate >= 0.3:
            # 標準帯域では適度に正例へ重みを寄せ、precision低下を防ぐ
            alpha = 0.45
            gamma = 1.0
            strategy = "継続重視（正例率30-60%・適度な正例ウェイト)"
        else:
            # 正例が希少な期間は追加で正例を持ち上げるがgammaは抑える
            alpha = 0.55
            gamma = 1.1
            strategy = "継続重視（正例率<30%・正例ウェイト中)"
        
        self.set_focal_loss_params(alpha, gamma)
        logger.info(f"正例率 {positive_rate:.1%} に基づき自動調整: {strategy}")
    
    def focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                   sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Focal Loss の計算（重み付きバイナリラベル対応）

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) * sample_weight

        Args:
            predictions: 予測確率 [batch_size] or [batch_size, 1]
            targets: ターゲットラベル [batch_size] or [batch_size, 1]
            sample_weights: サンプル重み [batch_size] or None（依頼なし=0.5、依頼あり=1.0）

        Returns:
            Focal Loss
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        # BCE loss
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')

        # p_t の計算
        p_t = predictions * targets + (1 - predictions) * (1 - targets)

        # alpha_t の計算
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)

        # Focal Loss
        focal_weight = alpha_t * torch.pow(1 - p_t, self.focal_gamma)
        focal_loss = focal_weight * bce_loss

        # Sample weightを適用
        if sample_weights is not None:
            sample_weights = sample_weights.squeeze()
            focal_loss = focal_loss * sample_weights

        return focal_loss.mean()
    
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
        # projects = developer.get('projects', [])  # マルチプロジェクト対応時に有効化
        # project_count = len(projects) if isinstance(projects, list) else 0
        
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
        
        # 最近のレビュー受諾率（直近30日）
        recent_acceptance_rate = self._calculate_recent_acceptance_rate(activity_history, context_date, days=30)
        
        # レビュー負荷（直近30日 / 平均）
        review_load = self._calculate_review_load(activity_history, context_date, days=30)
        
        return DeveloperState(
            developer_id=developer.get('developer_id', 'unknown'),
            experience_days=experience_days,
            total_changes=total_changes,
            total_reviews=total_reviews,
            # project_count=project_count,  # マルチプロジェクト対応時に有効化
            recent_activity_frequency=recent_activity_frequency,
            avg_activity_gap=avg_activity_gap,
            activity_trend=activity_trend,
            collaboration_score=collaboration_score,
            code_quality_score=code_quality_score,
            recent_acceptance_rate=recent_acceptance_rate,
            review_load=review_load,
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
                
                # 行動の強度（変更ファイル数ベース）
                intensity = self._calculate_action_intensity(activity)
                
                # 協力度
                collaboration = self._calculate_action_collaboration(activity)
                
                # レスポンス時間（レビューリクエストから応答までの日数）
                response_time = self._calculate_response_time(activity)
                
                # レビュー規模（変更行数ベース）
                review_size = self._calculate_review_size(activity)
                
                # タイムスタンプ
                timestamp_str = activity.get('timestamp', context_date.isoformat())
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                actions.append(DeveloperAction(
                    action_type=action_type,
                    intensity=intensity,
                    collaboration=collaboration,
                    response_time=response_time,
                    review_size=review_size,  # quality → review_sizeに変更
                    timestamp=timestamp
                ))
                
            except Exception as e:
                logger.warning(f"行動抽出エラー: {e}")
                continue
        
        return actions
    
    def state_to_tensor(self, state: DeveloperState) -> torch.Tensor:
        """状態をテンソルに変換（10次元版 - 全特徴量を0-1に正規化）"""
        
        # 活動トレンドのエンコーディング
        trend_encoding = {
            'increasing': 1.0,
            'stable': 0.5,
            'decreasing': 0.0,
            'unknown': 0.25
        }
        
        # 全特徴量を0-1の範囲に正規化（上限でクリップ）
        features = [
            min(state.experience_days / 730.0, 1.0),  # 2年でキャップ
            min(state.total_changes / 500.0, 1.0),    # 500件でキャップ
            min(state.total_reviews / 500.0, 1.0),    # 500件でキャップ
            # min(state.project_count / 5.0, 1.0),    # マルチプロジェクト対応時に有効化
            min(state.recent_activity_frequency, 1.0), # 既に0-1
            min(state.avg_activity_gap / 60.0, 1.0),  # 60日でキャップ
            trend_encoding.get(state.activity_trend, 0.25), # 既に0-1
            min(state.collaboration_score, 1.0),      # 既に0-1
            min(state.code_quality_score, 1.0),       # 既に0-1
            min(state.recent_acceptance_rate, 1.0),   # 既に0-1（直近30日の受諾率）
            min(state.review_load, 1.0)               # 既に0-1（負荷比率、正規化済み）
        ]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def action_to_tensor(self, action: DeveloperAction) -> torch.Tensor:
        """行動をテンソルに変換（4次元版 - quality除外、review_size追加）"""
        
        # レスポンス時間を「素早さ」に変換（0-1の範囲に正規化）
        # response_time が短い（素早い）ほど値が大きくなる
        # 3日でおよそ0.5、即日で1.0に近い値
        response_speed = 1.0 / (1.0 + action.response_time / 3.0)
        
        # 全特徴量を0-1の範囲に正規化
        features = [
            min(action.intensity, 1.0),        # 強度（変更ファイル数、0-1）
            min(action.collaboration, 1.0),    # 協力度（0-1）
            min(response_speed, 1.0),           # レスポンス速度（素早いほど大きい、0-1）
            min(action.review_size, 1.0)        # レビュー規模（変更行数、0-1）✨ NEW
        ]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def train_irl(self,
                  expert_trajectories: List[Dict[str, Any]],
                  epochs: int = 100) -> Dict[str, Any]:
        """
        IRLモデルを訓練（時系列対応）

        Args:
            expert_trajectories: エキスパート軌跡データ
            epochs: 訓練エポック数

        Returns:
            Dict[str, Any]: 訓練結果
        """
        logger.info(f"IRL訓練開始: {len(expert_trajectories)}軌跡, {epochs}エポック (時系列モード: {self.sequence})")

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

                    if self.sequence:
                        # 時系列モード: 軌跡全体を使用
                        # シーケンス長に合わせてパディングまたはトランケート
                        if len(actions) < self.seq_len:
                            # パディング: 最初のアクションを繰り返す
                            padded_actions = [actions[0]] * (self.seq_len - len(actions)) + actions
                        else:
                            # トランケート: 最新のseq_lenアクションを使用
                            padded_actions = actions[-self.seq_len:]

                        # 状態テンソル: 各タイムステップで同じ状態を使用（簡略化）
                        state_tensors = [self.state_to_tensor(state) for _ in range(self.seq_len)]
                        state_seq = torch.stack(state_tensors).unsqueeze(0)  # [1, seq_len, state_dim]

                        # 行動テンソル: 時系列順序を保持
                        action_tensors = [self.action_to_tensor(action) for action in padded_actions]
                        action_seq = torch.stack(action_tensors).unsqueeze(0)  # [1, seq_len, action_dim]

                        # 前向き計算
                        predicted_reward, predicted_continuation = self.network(
                            state_seq, action_seq
                        )

                        # 損失計算
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
                    else:
                        # 非時系列モード: 各行動を独立に学習（従来の方式）
                        state_tensor = self.state_to_tensor(state).unsqueeze(0)

                        # 最近の5つの行動のみ使用
                        for action in actions[-5:]:
                            action_tensor = self.action_to_tensor(action).unsqueeze(0)

                            # 前向き計算
                            predicted_reward, predicted_continuation = self.network(
                                state_tensor, action_tensor
                            )

                            # 損失計算
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
    
    def train_irl_multi_step_labels(self,
                                    expert_trajectories: List[Dict[str, Any]],
                                    epochs: int = 50) -> Dict[str, Any]:
        """
        各タイムステップラベル付きIRLモデルを訓練
        
        Args:
            expert_trajectories: エキスパート軌跡データ（各軌跡に step_labels を含む）
            epochs: 訓練エポック数
            
        Returns:
            Dict[str, Any]: 訓練結果
        """
        logger.info(f"各タイムステップラベル付きIRL訓練開始: {len(expert_trajectories)}軌跡, {epochs}エポック")
        
        training_losses = []
        
        # 正例と負例をカウント
        total_steps = sum(t.get('seq_len', len(t.get('step_labels', []))) for t in expert_trajectories)
        positive_steps = sum(sum(1 for label in t.get('step_labels', []) if label) for t in expert_trajectories)
        positive_rate = positive_steps / total_steps if total_steps > 0 else 0.5
        
        # クラスバランスを考慮
        positive_rate = max(0.01, min(0.99, positive_rate))  # 0や1にならないようクリップ
        neg_weight = positive_rate / (1 - positive_rate)
        
        logger.info(f"総ステップ数: {total_steps}, 継続ステップ: {positive_steps} ({positive_rate:.1%})")
        logger.info(f"負例の重み: {neg_weight:.2f}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for trajectory in expert_trajectories:
                try:
                    # 軌跡から状態と行動を抽出
                    developer = trajectory.get('developer', trajectory.get('developer_info', {}))
                    activity_history = trajectory['activity_history']
                    monthly_activity_histories = trajectory.get('monthly_activity_histories', None)
                    step_labels = trajectory.get('step_labels', [])
                    context_date = trajectory.get('context_date', datetime.now())
                    
                    if not step_labels or not activity_history:
                        continue
                    
                    # 実際のシーケンス長（step_labelsの長さを使用）
                    seq_len_actual = len(step_labels)
                    
                    # monthly_activity_historiesがある場合はそれを使用
                    if monthly_activity_histories is not None:
                        if len(monthly_activity_histories) != seq_len_actual:
                            logger.warning(f"長さ不一致: monthly_histories={len(monthly_activity_histories)}, step_labels={seq_len_actual}")
                            continue
                        
                        # 各月の履歴から状態と行動を抽出
                        state_sequence = []
                        action_sequence = []
                        
                        for step_idx, monthly_history in enumerate(monthly_activity_histories):
                            step_state = self.extract_developer_state(developer, monthly_history, context_date)
                            step_actions = self.extract_developer_actions(monthly_history, context_date)
                            
                            if not step_actions:
                                # 行動がない場合はデフォルト行動を使用
                                step_action = DeveloperAction(
                                    action_type='review',
                                    intensity=0.0,
                                    collaboration=0.0,
                                    response_time=999.0,
                                    review_size=0.0,
                                    timestamp=context_date
                                )
                            else:
                                # 最後の行動を使用
                                step_action = step_actions[-1]
                            
                            state_sequence.append(self.state_to_tensor(step_state))
                            action_sequence.append(self.action_to_tensor(step_action))
                    
                    else:
                        # 従来の方法（activity_history全体から抽出）
                        state = self.extract_developer_state(developer, activity_history, context_date)
                        actions = self.extract_developer_actions(activity_history, context_date)
                        
                        if not actions:
                            continue
                        
                        # actionsの長さがstep_labelsと一致しない場合はスキップ
                        if len(actions) != seq_len_actual:
                            logger.warning(f"長さ不一致: actions={len(actions)}, step_labels={seq_len_actual}")
                            continue
                        
                        # 状態と行動のシーケンスを構築
                        state_sequence = []
                        action_sequence = []
                        
                        for step_idx in range(len(actions)):
                            # このステップまでの履歴で状態を構築
                            history_up_to_step = activity_history[:step_idx+1]
                            step_state = self.extract_developer_state(developer, history_up_to_step, context_date)
                            state_sequence.append(self.state_to_tensor(step_state))
                            action_sequence.append(self.action_to_tensor(actions[step_idx]))
                    
                    # シーケンスの長さが一致しているか確認
                    if len(state_sequence) != seq_len_actual or len(action_sequence) != seq_len_actual:
                        logger.warning(f"シーケンス長不一致: state={len(state_sequence)}, action={len(action_sequence)}, labels={seq_len_actual}")
                        continue
                    
                    # テンソルに変換 [1, seq_len, dim]
                    state_tensor = torch.stack(state_sequence).unsqueeze(0)  # [1, seq_len, state_dim]
                    action_tensor = torch.stack(action_sequence).unsqueeze(0)  # [1, seq_len, action_dim]
                    
                    # 実際の長さ
                    lengths = torch.tensor([seq_len_actual], dtype=torch.long, device=self.device)
                    
                    # lengthsがテンソルのサイズを超えないかチェック
                    if seq_len_actual > state_tensor.size(1):
                        logger.warning(f"長さエラー: seq_len={seq_len_actual} > tensor_size={state_tensor.size(1)}")
                        continue
                    
                    # 全ステップで予測
                    predictions = self.network.forward_all_steps(
                        state_tensor, action_tensor, lengths
                    )  # [1, seq_len]
                    
                    # ターゲット
                    targets = torch.tensor(
                        [[1.0 if label else 0.0 for label in step_labels]],
                        device=self.device
                    )  # [1, seq_len]
                    
                    # 重み付き損失の計算
                    weights = torch.ones_like(targets)
                    for i, label in enumerate(step_labels):
                        if not label:
                            weights[0, i] = neg_weight
                    
                    # BCE損失（実際の長さまでのみ計算）
                    loss_per_step = nn.functional.binary_cross_entropy(
                        predictions[:, :seq_len_actual],
                        targets[:, :seq_len_actual],
                        weight=weights[:, :seq_len_actual],
                        reduction='mean'
                    )
                    
                    # バックプロパゲーション
                    self.optimizer.zero_grad()
                    loss_per_step.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss_per_step.item()
                    batch_count += 1
                
                except Exception as e:
                    logger.warning(f"軌跡処理エラー: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    continue
            
            avg_loss = epoch_loss / max(batch_count, 1)
            training_losses.append(avg_loss)
            
            # 学習率スケジューラ（あれば）
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step(avg_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                if epoch % 10 == 0:
                    logger.info(f"エポック {epoch}: 平均損失 = {avg_loss:.4f}, LR = {current_lr:.6f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"エポック {epoch}: 平均損失 = {avg_loss:.4f}")
        
        logger.info("各タイムステップラベル付きIRL訓練完了")
        
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
        継続確率を予測（時系列対応）

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

            if self.sequence:
                # 時系列モード: シーケンスとして予測
                # seq_lenが0（可変長）の場合は全活動を使用
                if self.seq_len == 0 or self.seq_len is None:
                    # 可変長：全活動を使用
                    padded_actions = actions
                    actual_seq_len = len(actions)
                elif len(actions) < self.seq_len:
                    # パディング: 最初のアクションを繰り返す
                    padded_actions = [actions[0]] * (self.seq_len - len(actions)) + actions
                    actual_seq_len = self.seq_len
                else:
                    # トランケート: 最新のseq_lenアクションを使用
                    padded_actions = actions[-self.seq_len:]
                    actual_seq_len = self.seq_len

                # 状態テンソル: 各タイムステップで状態を構築
                state_tensors = []
                for i in range(len(padded_actions)):
                    # 各ステップまでの履歴を使用
                    step_history = activity_history[:i+1]
                    step_state = self.extract_developer_state(developer, step_history, context_date)
                    state_tensors.append(self.state_to_tensor(step_state))
                
                state_seq = torch.stack(state_tensors).unsqueeze(0)  # [1, seq_len, state_dim]

                # 行動テンソル: 時系列順序を保持
                action_tensors = [self.action_to_tensor(action) for action in padded_actions]
                action_seq = torch.stack(action_tensors).unsqueeze(0)  # [1, seq_len, action_dim]

                # 可変長の場合はlengthsを渡す
                if self.seq_len == 0 or self.seq_len is None:
                    lengths = torch.tensor([actual_seq_len], dtype=torch.long, device=self.device)
                    predicted_reward, predicted_continuation = self.network(
                        state_seq, action_seq, lengths
                    )
                else:
                    # 予測実行
                    predicted_reward, predicted_continuation = self.network(
                        state_seq, action_seq
                    )

                recent_action = actions[-1]
            else:
                # 非時系列モード: 最近の行動を使用
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
    
    def _calculate_recent_acceptance_rate(self, activity_history: List[Dict[str, Any]], 
                                         context_date: datetime, days: int = 30) -> float:
        """
        直近N日のレビュー受諾率を計算
        
        Args:
            activity_history: 活動履歴
            context_date: 基準日
            days: 対象期間（日数）
        
        Returns:
            受諾率（0.0～1.0）、依頼がない場合は0.5（中立）
        """
        cutoff_date = context_date - timedelta(days=days)
        
        # 直近の活動のみフィルタ
        recent_activities = [
            activity for activity in activity_history
            if activity.get('timestamp', context_date) >= cutoff_date
        ]
        
        if not recent_activities:
            return 0.5  # データなし → 中立
        
        # レビュー依頼とその受諾を集計
        review_requests = 0
        accepted_reviews = 0
        
        for activity in recent_activities:
            # レビュー関連の活動かチェック
            if activity.get('action_type') == 'review' or 'review' in activity.get('type', '').lower():
                review_requests += 1
                # 受諾したかチェック
                if activity.get('accepted', False):
                    accepted_reviews += 1
        
        if review_requests == 0:
            return 0.5  # レビュー依頼なし → 中立
        
        return accepted_reviews / review_requests
    
    def _calculate_review_load(self, activity_history: List[Dict[str, Any]], 
                              context_date: datetime, days: int = 30) -> float:
        """
        レビュー負荷を計算（直近N日の依頼数 / 平均依頼数）
        
        Args:
            activity_history: 活動履歴
            context_date: 基準日
            days: 対象期間（日数）
        
        Returns:
            負荷比率（0.0～）、1.0が平均、>1.0が過負荷
        """
        cutoff_date = context_date - timedelta(days=days)
        
        # 直近の活動のみフィルタ
        recent_activities = [
            activity for activity in activity_history
            if activity.get('timestamp', context_date) >= cutoff_date
        ]
        
        # 直近のレビュー依頼数
        recent_requests = sum(
            1 for activity in recent_activities
            if activity.get('action_type') == 'review' or 'review' in activity.get('type', '').lower()
        )
        
        # 全期間の平均依頼数を計算（月あたり）
        if not activity_history:
            return 0.0
        
        total_requests = sum(
            1 for activity in activity_history
            if activity.get('action_type') == 'review' or 'review' in activity.get('type', '').lower()
        )
        
        # 活動履歴の期間を計算
        timestamps = [activity.get('timestamp', context_date) for activity in activity_history]
        if len(timestamps) < 2:
            return 1.0  # データ不足 → 標準負荷
        
        period_days = (max(timestamps) - min(timestamps)).days + 1
        avg_requests_per_period = total_requests / max(period_days / days, 1.0)
        
        if avg_requests_per_period == 0:
            return 0.0
        
        # 負荷比率を正規化（0-1の範囲にクリップ）
        load_ratio = recent_requests / avg_requests_per_period
        return min(load_ratio / 2.0, 1.0)  # 2倍の負荷を1.0にマッピング
    
    def _calculate_action_intensity(self, activity: Dict[str, Any]) -> float:
        """行動の強度を計算（変更ファイル数ベース）"""
        
        files_changed = activity.get('files_changed', 0)
        
        # 変更ファイル数で強度を計算（正規化）
        intensity = min(files_changed / 20.0, 1.0)  # 20ファイルで1.0
        return max(intensity, 0.0)
    
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
    
    def _calculate_review_size(self, activity: Dict[str, Any]) -> float:
        """レビュー規模を計算（変更行数ベース）"""
        
        lines_added = activity.get('lines_added', 0)
        lines_deleted = activity.get('lines_deleted', 0)
        total_lines = lines_added + lines_deleted
        
        # 変更行数で規模を計算（正規化）
        review_size = min(total_lines / 500.0, 1.0)  # 500行で1.0
        return max(review_size, 0.0)
    
    def _calculate_response_time(self, activity: Dict[str, Any]) -> float:
        """レスポンス時間を計算（日数）"""
        
        # レビューリクエストの作成日時と応答日時の差を計算
        request_time = activity.get('request_time')
        response_time = activity.get('response_time')  # 変更: timestamp → response_time
        
        if request_time and response_time:
            try:
                if isinstance(request_time, str):
                    request_dt = datetime.fromisoformat(request_time.replace('Z', '+00:00'))
                else:
                    request_dt = request_time
                    
                if isinstance(response_time, str):
                    response_dt = datetime.fromisoformat(response_time.replace('Z', '+00:00'))
                else:
                    response_dt = response_time
                
                # 日数で計算
                days_diff = (response_dt - request_dt).days
                return max(0.0, days_diff)  # 負の値は0に
                
            except Exception:
                return 14.0  # 最大レスポンス時間（未応答/非アクティブ）
        
        # デフォルト値（データがない場合）: 未応答を表す最大値を使用
        return 14.0  # 最大レスポンス時間（未応答/非アクティブ）
    
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
        
        # 最近の行動（属性チェック）
        if hasattr(action, 'quality') and action.quality > 0.7:
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
        
        # checkpointがstate_dict直接かdictかを判定
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 古い形式（dictに'model_state_dict'キーがある）
            config = checkpoint.get('config', {
                'state_dim': 20,
                'action_dim': 3,
                'hidden_dim': 128,
                'learning_rate': 0.001
            })
            
            instance = cls(config)
            instance.network.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # 新しい形式（state_dict直接保存）
            # デフォルトconfigで初期化
            config = {
                'state_dim': 10,
                'action_dim': 5,
                'hidden_dim': 128,
                'learning_rate': 0.0001,
                'sequence': True,
                'seq_len': 0,
            }
            
            instance = cls(config)
            instance.network.load_state_dict(checkpoint)
        
        logger.info(f"IRLモデルを読み込みました: {filepath}")
        return instance
    
    def train_irl_temporal_trajectories(self,
                                       expert_trajectories: List[Dict[str, Any]],
                                       epochs: int = 50) -> Dict[str, Any]:
        """
        時系列軌跡でIRLモデルを訓練
        
        Args:
            expert_trajectories: 時系列軌跡データ
            epochs: エポック数
            
        Returns:
            訓練結果
        """
        logger.info("時系列IRL訓練開始")
        logger.info(f"軌跡数: {len(expert_trajectories)}")
        
        self.network.train()
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for trajectory in expert_trajectories:
                try:
                    # 開発者情報と活動履歴を取得
                    developer = trajectory.get('developer', trajectory.get('developer_info', {}))
                    activity_history = trajectory.get('activity_history', [])
                    context_date = trajectory.get('context_date', datetime.now())
                    
                    if not activity_history:
                        continue
                    
                    # 状態と行動を抽出
                    state = self.extract_developer_state(developer, activity_history, context_date)
                    actions = self.extract_developer_actions(activity_history, context_date)
                    
                    if not actions:
                        continue
                    
                    # 時系列データとして処理
                    if self.sequence:
                        # 各ステップで損失を計算（月次集約ラベル）
                        step_labels = trajectory.get('step_labels', [])
                        monthly_histories = trajectory.get('monthly_activity_histories', [])
                        
                        if not step_labels or not monthly_histories:
                            continue
                        
                        # 各月の時点での状態を計算（LSTM用）
                        state_tensors = []
                        action_tensors = []
                        
                        min_len = min(len(monthly_histories), len(step_labels))
                        
                        for i in range(min_len):
                            month_history = monthly_histories[i]
                            if not month_history:
                                # 活動履歴がない場合はゼロ状態
                                state_tensors.append(torch.zeros(self.state_dim, device=self.device))
                                action_tensors.append(torch.zeros(self.action_dim, device=self.device))
                                continue
                            
                            # この月時点での状態を計算
                            month_context_date = month_history[-1]['timestamp']  # 最新の活動時刻
                            month_state = self.extract_developer_state(developer, month_history, month_context_date)
                            month_actions = self.extract_developer_actions(month_history, month_context_date)
                            
                            state_tensors.append(self.state_to_tensor(month_state))
                            
                            # 行動は最新のものを使用
                            if month_actions:
                                action_tensors.append(self.action_to_tensor(month_actions[-1]))
                            else:
                                action_tensors.append(torch.zeros(self.action_dim, device=self.device))
                        
                        if not state_tensors or not action_tensors:
                            continue
                        
                        # 3Dテンソルとして構築（LSTM用: [batch=1, sequence_length, features]）
                        state_sequence = torch.stack(state_tensors).unsqueeze(0)  # [1, seq_len, state_dim]
                        action_sequence = torch.stack(action_tensors).unsqueeze(0)  # [1, seq_len, action_dim]
                        
                        # LSTM全ステップの予測（明示的に全ステップ処理）
                        batch_size, seq_len, _ = state_sequence.shape
                        state_encoded = self.network.state_encoder(state_sequence.view(-1, state_sequence.shape[-1])).view(batch_size, seq_len, -1)
                        action_encoded = self.network.action_encoder(action_sequence.view(-1, action_sequence.shape[-1])).view(batch_size, seq_len, -1)
                        
                        combined = state_encoded + action_encoded
                        lstm_out, _ = self.network.lstm(combined)  # [1, seq_len, hidden_dim]
                        
                        # 各ステップで報酬と継続確率を予測
                        lstm_out_flat = lstm_out.reshape(-1, lstm_out.size(-1))  # [seq_len, hidden_dim]
                        predicted_reward_flat = self.network.reward_predictor(lstm_out_flat).squeeze(-1)  # [seq_len]
                        predicted_continuation_flat = self.network.continuation_predictor(lstm_out_flat).squeeze(-1)  # [seq_len]
                        
                        # 損失計算（月次集約ラベルを使用）
                        targets = torch.tensor([1.0 if label else 0.0 for label in step_labels[:min_len]], device=self.device)

                        # サンプル重みを取得（依頼なし=0.5、依頼あり=1.0）
                        sample_weight = trajectory.get('sample_weight', 1.0)
                        sample_weights = torch.full([min_len], sample_weight, device=self.device)

                        # 継続予測損失（Focal Loss を使用、重み付き）
                        continuation_loss = self.focal_loss(
                            predicted_continuation_flat, targets, sample_weights
                        )
                        
                        # 報酬損失（月次集約ラベルから逆算）
                        reward_loss = F.mse_loss(
                            predicted_reward_flat, targets * 2.0 - 1.0  # -1 to 1
                        )
                        
                        loss_per_step = continuation_loss + reward_loss

                    # 単一ステップモードは使用しない（LSTMモードのみ使用）
                    # else:
                    #     # 単一ステップ処理（最新のラベルを使用）
                    #     recent_action = actions[-1]
                    #     state_tensor = self.state_to_tensor(state)
                    #     action_tensor = self.action_to_tensor(recent_action)
                    #
                    #     predicted_reward, predicted_continuation = self.network(
                    #         state_tensor.unsqueeze(0), action_tensor.unsqueeze(0)
                    #     )
                    #
                    #     # 損失計算（最新の月次集約ラベルを使用）
                    #     step_labels = trajectory.get('step_labels', [])
                    #     if step_labels:
                    #         latest_label = step_labels[-1]
                    #         target = torch.tensor([1.0 if latest_label else 0.0], device=self.device)
                    #     else:
                    #         target = torch.tensor([0.0], device=self.device)
                    #
                    #     continuation_loss = self.focal_loss(
                    #         predicted_continuation.squeeze(), target
                    #     )
                    #
                    #     reward_loss = F.mse_loss(
                    #         predicted_reward.squeeze(), target * 2.0 - 1.0
                    #     )
                    #
                    #     loss_per_step = continuation_loss + reward_loss
                    
                    # バックプロパゲーション
                    self.optimizer.zero_grad()
                    loss_per_step.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss_per_step.item()
                    batch_count += 1
                
                except Exception as e:
                    logger.warning(f"軌跡処理エラー: {e}")
                    continue
            
            avg_loss = epoch_loss / max(batch_count, 1)
            training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"エポック {epoch}: 平均損失 = {avg_loss:.4f}")
        
        logger.info("時系列IRL訓練完了")
        
        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1] if training_losses else 0.0,
            'epochs_trained': epochs
        }
    
    def predict_continuation_probability_snapshot(self,
                                                developer: Dict[str, Any],
                                                activity_history: List[Dict[str, Any]],
                                                context_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        スナップショット特徴量で継続確率を予測
        
        Args:
            developer: 開発者データ
            activity_history: 活動履歴
            context_date: 基準日
            
        Returns:
            予測結果
        """
        if context_date is None:
            context_date = datetime.now()
        
        self.network.eval()
        
        with torch.no_grad():
            # スナップショット時点での状態と行動を抽出
            state = self.extract_developer_state(developer, activity_history, context_date)
            actions = self.extract_developer_actions(activity_history, context_date)
            
            if not actions:
                return {
                    'continuation_probability': 0.5,
                    'confidence': 0.0,
                    'reasoning': '活動履歴が不足しているため、デフォルト確率を返します'
                }
            
            # 最新の行動を使用（スナップショット特徴量）
            latest_action = actions[-1]
            state_tensor = self.state_to_tensor(state)
            action_tensor = self.action_to_tensor(latest_action)
            
            # 予測
            predicted_reward, predicted_continuation = self.network(
                state_tensor.unsqueeze(0), action_tensor.unsqueeze(0)
            )
            
            continuation_prob = predicted_continuation.item()
            # 温度スケーリングで確率分布の広がりを調整
            if self.output_temperature and abs(self.output_temperature - 1.0) > 1e-6:
                p = min(max(continuation_prob, 1e-6), 1.0 - 1e-6)
                # ロジットに変換して温度で割る（T<1でシャープ、T>1でフラット）
                import math
                logit = math.log(p / (1.0 - p))
                scaled_logit = logit / self.output_temperature
                continuation_prob = 1.0 / (1.0 + math.exp(-scaled_logit))
            confidence = abs(continuation_prob - 0.5) * 2
            
            # 理由生成
            reasoning = self._generate_snapshot_reasoning(
                developer, activity_history, continuation_prob, context_date
            )
            
            return {
                'continuation_probability': continuation_prob,
                'confidence': confidence,
                'reasoning': reasoning,
                'method': 'snapshot_features',
                'state_features': state_tensor.tolist(),
                'action_features': action_tensor.tolist()
            }
    
    def _generate_snapshot_reasoning(self,
                                   developer: Dict[str, Any],
                                   activity_history: List[Dict[str, Any]],
                                   continuation_prob: float,
                                   context_date: datetime) -> str:
        """スナップショット特徴量に基づく理由生成"""
        reasoning_parts = []
        
        # 活動履歴の分析
        if len(activity_history) > 10:
            reasoning_parts.append("豊富な活動履歴")
        elif len(activity_history) > 5:
            reasoning_parts.append("適度な活動履歴")
        else:
            reasoning_parts.append("限定的な活動履歴")
        
        # 継続確率に基づく判断
        if continuation_prob > 0.7:
            reasoning_parts.append("高い継続可能性")
        elif continuation_prob > 0.3:
            reasoning_parts.append("中程度の継続可能性")
        else:
            reasoning_parts.append("低い継続可能性")
        
        return f"スナップショット特徴量分析: {', '.join(reasoning_parts)}"


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