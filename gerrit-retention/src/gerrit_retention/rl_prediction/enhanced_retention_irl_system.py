"""
拡張特徴量を使用した継続予測IRLシステム

高優先度特徴量を統合:
- B1: レビュー負荷指標
- C1: 相互作用の深さ
- A1: 活動頻度の多期間比較
- D1: 専門性の一致度
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .enhanced_feature_extractor import (
    EnhancedDeveloperAction,
    EnhancedDeveloperState,
    EnhancedFeatureExtractor,
)

logger = logging.getLogger(__name__)


class EnhancedRetentionIRLNetwork(nn.Module):
    """拡張特徴量対応の継続予測IRLネットワーク"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 sequence: bool = False, seq_len: int = 15, dropout: float = 0.2):
        super().__init__()
        self.sequence = sequence
        self.seq_len = seq_len

        # 状態エンコーダー（より深いネットワーク + Dropout）
        #  BatchNorm1dをLayerNormに変更（シーケンスモードで安定）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 行動エンコーダー（より深いネットワーク + Dropout）
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.sequence:
            # LSTM for sequence (2層に拡張)
            self.lstm = nn.LSTM(
                hidden_dim // 2,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0
            )

        # 報酬予測器
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 継続確率予測器
        self.continuation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                return_hidden: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        前向き計算

        Args:
            state: 開発者状態 [batch_size, seq_len, state_dim] if sequence else [batch_size, state_dim]
            action: 開発者行動 [batch_size, seq_len, action_dim] if sequence else [batch_size, action_dim]
            return_hidden: 隠れ状態も返すかどうか

        Returns:
            reward, continuation_prob, (hidden)
        """
        if self.sequence:
            # Sequence mode
            batch_size, seq_len, _ = state.shape

            # Encode state and action for each timestep
            state_flat = state.view(-1, state.shape[-1])
            action_flat = action.view(-1, action.shape[-1])

            state_encoded = self.state_encoder(state_flat).view(batch_size, seq_len, -1)
            action_encoded = self.action_encoder(action_flat).view(batch_size, seq_len, -1)

            # Combine (addition)
            combined = state_encoded + action_encoded

            # LSTM
            lstm_out, _ = self.lstm(combined)
            hidden = lstm_out[:, -1, :]  # Last timestep
        else:
            # Single step mode
            state_encoded = self.state_encoder(state)
            action_encoded = self.action_encoder(action)
            combined = torch.cat([state_encoded, action_encoded], dim=1)
            hidden = combined

        # Predict
        reward = self.reward_predictor(hidden)
        continuation_prob = self.continuation_predictor(hidden)

        if return_hidden:
            return reward, continuation_prob, hidden
        else:
            return reward, continuation_prob


class EnhancedRetentionIRLSystem:
    """拡張特徴量対応の継続予測IRLシステム"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 特徴抽出器
        self.feature_extractor = EnhancedFeatureExtractor(config)

        # ネットワーク設定
        self.state_dim = self.feature_extractor.get_state_dim()  # 32
        self.action_dim = self.feature_extractor.get_action_dim()  # 9
        self.hidden_dim = config.get('hidden_dim', 256)
        self.sequence = config.get('sequence', False)
        self.seq_len = config.get('seq_len', 15)
        self.dropout = config.get('dropout', 0.2)

        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ネットワーク初期化
        self.network = EnhancedRetentionIRLNetwork(
            self.state_dim, self.action_dim, self.hidden_dim,
            self.sequence, self.seq_len, self.dropout
        ).to(self.device)

        # オプティマイザー
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # 損失関数（MSELossのみ使用）
        self.mse_loss = nn.MSELoss()

        logger.info(f"拡張IRLシステムを初期化: state_dim={self.state_dim}, action_dim={self.action_dim}")

    def train_irl(self, expert_trajectories: List[Dict[str, Any]], epochs: int = 30) -> Dict[str, Any]:
        """
        IRLモデルを訓練

        Args:
            expert_trajectories: エキスパート軌跡データ
            epochs: 訓練エポック数

        Returns:
            訓練結果
        """
        logger.info(f"拡張IRL訓練開始: {len(expert_trajectories)}軌跡, {epochs}エポック")

        # Scalerのフィット（最初の1回のみ）
        if not self.feature_extractor.scaler_fitted:
            logger.info("特徴量Scalerをフィット中...")
            all_states = []
            all_actions = []

            for traj in expert_trajectories:
                developer = traj['developer']
                activity_history = traj['activity_history']
                context_date = traj.get('context_date', datetime.now())

                state = self.feature_extractor.extract_enhanced_state(
                    developer, activity_history, context_date
                )
                all_states.append(self.feature_extractor.state_to_array(state))

                for activity in activity_history:
                    action = self.feature_extractor.extract_enhanced_action(activity, context_date)
                    all_actions.append(self.feature_extractor.action_to_array(action))

            self.feature_extractor.fit_scalers(all_states, all_actions)

        # 訓練ループ
        training_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for trajectory in expert_trajectories:
                try:
                    developer = trajectory['developer']
                    activity_history = trajectory['activity_history']
                    continuation_label = trajectory.get('continued', True)
                    context_date = trajectory.get('context_date', datetime.now())

                    # 状態と行動を抽出
                    state = self.feature_extractor.extract_enhanced_state(
                        developer, activity_history, context_date
                    )
                    actions = [self.feature_extractor.extract_enhanced_action(act, context_date)
                              for act in activity_history]

                    if not actions:
                        continue

                    if self.sequence:
                        # 時系列モード
                        if len(actions) < self.seq_len:
                            padded_actions = [actions[0]] * (self.seq_len - len(actions)) + actions
                        else:
                            padded_actions = actions[-self.seq_len:]

                        # テンソル化 + 正規化
                        state_arrays = [self.feature_extractor.state_to_array(state)
                                       for _ in range(self.seq_len)]
                        action_arrays = [self.feature_extractor.action_to_array(act)
                                        for act in padded_actions]

                        # 正規化
                        state_arrays_norm = [self.feature_extractor.normalize_state(s)
                                            for s in state_arrays]
                        action_arrays_norm = [self.feature_extractor.normalize_action(a)
                                             for a in action_arrays]

                        state_tensor = torch.tensor(
                            np.array(state_arrays_norm), dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        action_tensor = torch.tensor(
                            np.array(action_arrays_norm), dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                    else:
                        # 単一ステップモード
                        recent_action = actions[-1]

                        state_array = self.feature_extractor.state_to_array(state)
                        action_array = self.feature_extractor.action_to_array(recent_action)

                        state_array_norm = self.feature_extractor.normalize_state(state_array)
                        action_array_norm = self.feature_extractor.normalize_action(action_array)

                        state_tensor = torch.tensor(
                            state_array_norm, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        action_tensor = torch.tensor(
                            action_array_norm, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)

                    # 予測
                    predicted_reward, predicted_continuation = self.network(state_tensor, action_tensor)

                    # 損失計算
                    target_reward = torch.tensor(
                        [[1.0 if continuation_label else 0.0]], dtype=torch.float32, device=self.device
                    )
                    target_continuation = torch.tensor(
                        [[1.0 if continuation_label else 0.0]], dtype=torch.float32, device=self.device
                    )

                    loss_reward = self.mse_loss(predicted_reward, target_reward)
                    loss_continuation = self.mse_loss(predicted_continuation, target_continuation)
                    total_loss = loss_reward + loss_continuation

                    # 最適化
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
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

        logger.info("拡張IRL訓練完了")

        return {
            'final_loss': training_losses[-1] if training_losses else 0.0,
            'training_losses': training_losses,
            'epochs': epochs
        }

    def evaluate(self, test_trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        テストデータで評価

        Args:
            test_trajectories: テスト軌跡データ

        Returns:
            評価指標
        """
        self.network.eval()

        y_true = []
        y_pred = []
        y_pred_binary = []

        with torch.no_grad():
            for trajectory in test_trajectories:
                try:
                    developer = trajectory['developer']
                    activity_history = trajectory['activity_history']
                    continuation_label = trajectory.get('continued', True)
                    context_date = trajectory.get('context_date', datetime.now())

                    # 状態と行動を抽出
                    state = self.feature_extractor.extract_enhanced_state(
                        developer, activity_history, context_date
                    )
                    actions = [self.feature_extractor.extract_enhanced_action(act, context_date)
                              for act in activity_history]

                    if not actions:
                        continue

                    if self.sequence:
                        if len(actions) < self.seq_len:
                            padded_actions = [actions[0]] * (self.seq_len - len(actions)) + actions
                        else:
                            padded_actions = actions[-self.seq_len:]

                        state_arrays = [self.feature_extractor.state_to_array(state)
                                       for _ in range(self.seq_len)]
                        action_arrays = [self.feature_extractor.action_to_array(act)
                                        for act in padded_actions]

                        state_arrays_norm = [self.feature_extractor.normalize_state(s)
                                            for s in state_arrays]
                        action_arrays_norm = [self.feature_extractor.normalize_action(a)
                                             for a in action_arrays]

                        state_tensor = torch.tensor(
                            np.array(state_arrays_norm), dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        action_tensor = torch.tensor(
                            np.array(action_arrays_norm), dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                    else:
                        recent_action = actions[-1]

                        state_array = self.feature_extractor.state_to_array(state)
                        action_array = self.feature_extractor.action_to_array(recent_action)

                        state_array_norm = self.feature_extractor.normalize_state(state_array)
                        action_array_norm = self.feature_extractor.normalize_action(action_array)

                        state_tensor = torch.tensor(
                            state_array_norm, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        action_tensor = torch.tensor(
                            action_array_norm, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)

                    # 予測
                    _, predicted_continuation = self.network(state_tensor, action_tensor)

                    continuation_prob = predicted_continuation.item()

                    # NaNチェック
                    if not np.isnan(continuation_prob) and not np.isinf(continuation_prob):
                        y_true.append(1 if continuation_label else 0)
                        y_pred.append(continuation_prob)
                        y_pred_binary.append(1 if continuation_prob >= 0.5 else 0)

                except Exception as e:
                    logger.warning(f"評価エラー: {e}")
                    continue

        # メトリクス計算
        if len(y_true) == 0:
            logger.warning("評価可能なサンプルが0件です")
            return {
                'auc_roc': 0.0,
                'auc_pr': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0
            }

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_binary = np.array(y_pred_binary)

        # NaN/Inf再チェック
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            logger.error(f"予測値にNaN/Infが含まれています: {np.sum(np.isnan(y_pred))} NaN, {np.sum(np.isinf(y_pred))} Inf")
            # NaN/Infを除去
            valid_mask = ~(np.isnan(y_pred) | np.isinf(y_pred))
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            y_pred_binary = y_pred_binary[valid_mask]

        if len(y_true) == 0:
            logger.warning("有効なサンプルが0件です")
            return {
                'auc_roc': 0.0,
                'auc_pr': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0
            }

        # AUCは少なくとも両クラスが存在する必要がある
        if len(np.unique(y_true)) > 1:
            try:
                auc_roc = roc_auc_score(y_true, y_pred)
                auc_pr = average_precision_score(y_true, y_pred)
            except Exception as e:
                logger.error(f"AUC計算エラー: {e}")
                auc_roc = 0.0
                auc_pr = 0.0
        else:
            auc_roc = 0.0
            auc_pr = 0.0

        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred_binary)

        self.network.train()

        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }

    def save_model(self, filepath: str):
        """モデルを保存"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'state_scaler_min': self.feature_extractor.state_scaler.data_min_,
            'state_scaler_max': self.feature_extractor.state_scaler.data_max_,
            'state_scaler_scale': self.feature_extractor.state_scaler.scale_,
            'action_scaler_min': self.feature_extractor.action_scaler.data_min_,
            'action_scaler_max': self.feature_extractor.action_scaler.data_max_,
            'action_scaler_scale': self.feature_extractor.action_scaler.scale_,
            'scaler_fitted': self.feature_extractor.scaler_fitted
        }, filepath)
        logger.info(f"拡張IRLモデルを保存: {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'EnhancedRetentionIRLSystem':
        """モデルを読み込み"""
        checkpoint = torch.load(filepath, map_location='cpu')

        config = checkpoint['config']
        system = cls(config)

        system.network.load_state_dict(checkpoint['network_state_dict'])
        system.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Scalerの復元 (MinMaxScaler用)
        if checkpoint.get('scaler_fitted', False):
            system.feature_extractor.state_scaler.data_min_ = checkpoint['state_scaler_min']
            system.feature_extractor.state_scaler.data_max_ = checkpoint['state_scaler_max']
            system.feature_extractor.state_scaler.scale_ = checkpoint['state_scaler_scale']
            system.feature_extractor.action_scaler.data_min_ = checkpoint['action_scaler_min']
            system.feature_extractor.action_scaler.data_max_ = checkpoint['action_scaler_max']
            system.feature_extractor.action_scaler.scale_ = checkpoint['action_scaler_scale']
            system.feature_extractor.scaler_fitted = True

        logger.info(f"拡張IRLモデルを読み込み: {filepath}")
        return system

    def train_irl_multi_step_labels(self,
                                    expert_trajectories: List[Dict[str, Any]],
                                    epochs: int = 50) -> Dict[str, Any]:
        """
        各タイムステップラベル付きIRLモデルを訓練（拡張特徴量版）
        
        Args:
            expert_trajectories: エキスパート軌跡データ（各軌跡に step_labels を含む）
            epochs: 訓練エポック数
            
        Returns:
            Dict[str, Any]: 訓練結果
        """
        logger.info(f"各タイムステップラベル付き拡張IRL訓練開始: {len(expert_trajectories)}軌跡, {epochs}エポック")
        
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
        
        # Scalerのfitting（初回のみ）
        if not self.feature_extractor.scaler_fitted:
            logger.info("特徴量のスケーラーをフィット中...")
            all_states = []
            all_actions = []
            
            for trajectory in expert_trajectories:
                developer = trajectory.get('developer', trajectory.get('developer_info', {}))
                activity_history = trajectory['activity_history']
                context_date = trajectory.get('context_date', datetime.now())
                
                state = self.feature_extractor.extract_enhanced_state(developer, activity_history, context_date)
                all_states.append(self.feature_extractor.state_to_array(state))
                
                for activity in activity_history:
                    action = self.feature_extractor.extract_enhanced_action(activity, context_date)
                    all_actions.append(self.feature_extractor.action_to_array(action))
            
            self.feature_extractor.fit_scalers(all_states, all_actions)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for trajectory in expert_trajectories:
                try:
                    # 軌跡から状態と行動を抽出
                    developer = trajectory.get('developer', trajectory.get('developer_info', {}))
                    activity_history = trajectory['activity_history']
                    step_labels = trajectory.get('step_labels', [])
                    context_date = trajectory.get('context_date', datetime.now())
                    seq_len_actual = trajectory.get('seq_len', len(step_labels))
                    
                    if not step_labels or not activity_history:
                        continue
                    
                    # 各ステップごとに状態を動的に構築
                    state_sequence = []
                    action_sequence = []
                    
                    for step_idx in range(len(activity_history)):
                        # このステップまでの履歴で状態を抽出
                        history_up_to_step = activity_history[:step_idx+1]
                        step_state = self.feature_extractor.extract_enhanced_state(
                            developer, history_up_to_step, context_date
                        )
                        step_action = self.feature_extractor.extract_enhanced_action(
                            activity_history[step_idx], context_date
                        )
                        
                        # 配列化 + 正規化
                        state_array = self.feature_extractor.state_to_array(step_state)
                        action_array = self.feature_extractor.action_to_array(step_action)
                        
                        state_array_norm = self.feature_extractor.normalize_state(state_array)
                        action_array_norm = self.feature_extractor.normalize_action(action_array)
                        
                        state_sequence.append(state_array_norm)
                        action_sequence.append(action_array_norm)
                    
                    # テンソル化
                    state_tensor = torch.tensor(
                        np.array(state_sequence), dtype=torch.float32, device=self.device
                    ).unsqueeze(0)  # [1, seq_len, state_dim]
                    action_tensor = torch.tensor(
                        np.array(action_sequence), dtype=torch.float32, device=self.device
                    ).unsqueeze(0)  # [1, seq_len, action_dim]
                    
                    # 各ステップで予測（LSTMを使用）
                    batch_size, seq_len, _ = state_tensor.shape
                    
                    # Encode
                    state_flat = state_tensor.view(-1, state_tensor.shape[-1])
                    action_flat = action_tensor.view(-1, action_tensor.shape[-1])
                    
                    state_encoded = self.network.state_encoder(state_flat).view(batch_size, seq_len, -1)
                    action_encoded = self.network.action_encoder(action_flat).view(batch_size, seq_len, -1)
                    
                    # Combine
                    combined = state_encoded + action_encoded
                    
                    # LSTM
                    lstm_out, _ = self.network.lstm(combined)
                    
                    # 各ステップで継続予測
                    lstm_out_flat = lstm_out.view(-1, lstm_out.shape[-1])
                    predictions_flat = self.network.continuation_predictor(lstm_out_flat).squeeze(-1)
                    predictions = predictions_flat.view(batch_size, seq_len)
                    
                    # ラベル
                    targets = torch.tensor(
                        [[1.0 if label else 0.0 for label in step_labels]],
                        dtype=torch.float32,
                        device=self.device
                    )
                    
                    # 重み付き損失
                    weights = torch.ones_like(targets)
                    for i, label in enumerate(step_labels):
                        if not label:
                            weights[0, i] = neg_weight
                    
                    loss_per_step = nn.functional.binary_cross_entropy(
                        predictions[:, :seq_len_actual],
                        targets[:, :seq_len_actual],
                        weight=weights[:, :seq_len_actual],
                        reduction='mean'
                    )
                    
                    # 最適化
                    self.optimizer.zero_grad()
                    loss_per_step.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
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
            
            if epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"エポック {epoch}: 平均損失 = {avg_loss:.4f}, LR = {current_lr:.6f}")
        
        logger.info("拡張IRL訓練完了")
        
        result = {
            'training_losses': training_losses,
            'final_loss': training_losses[-1] if training_losses else 0.0,
            'epochs_trained': epochs
        }
        
        return result
    
    def predict_continuation_probability(self, developer: Dict[str, Any], 
                                        activity_history: List[Dict[str, Any]], 
                                        context_date: datetime) -> Dict[str, Any]:
        """継続確率を予測（拡張特徴量版）"""
        
        self.network.eval()
        
        with torch.no_grad():
            # 状態と行動を抽出
            state = self.feature_extractor.extract_enhanced_state(
                developer, activity_history, context_date
            )
            actions = [self.feature_extractor.extract_enhanced_action(act, context_date)
                      for act in activity_history]
            
            if not actions:
                return {
                    'continuation_probability': 0.5,
                    'predicted_reward': 0.0,
                    'reasoning': 'No actions available'
                }
            
            if self.sequence:
                # 可変長対応
                recent_actions = actions
                
                # 状態シーケンス（各ステップで履歴が増える）
                state_sequence = []
                action_sequence = []
                
                for step_idx in range(len(actions)):
                    history_up_to_step = activity_history[:step_idx+1]
                    step_state = self.feature_extractor.extract_enhanced_state(
                        developer, history_up_to_step, context_date
                    )
                    
                    state_array = self.feature_extractor.state_to_array(step_state)
                    action_array = self.feature_extractor.action_to_array(actions[step_idx])
                    
                    state_array_norm = self.feature_extractor.normalize_state(state_array)
                    action_array_norm = self.feature_extractor.normalize_action(action_array)
                    
                    state_sequence.append(state_array_norm)
                    action_sequence.append(action_array_norm)
                
                state_tensor = torch.tensor(
                    np.array(state_sequence), dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                action_tensor = torch.tensor(
                    np.array(action_sequence), dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                
                predicted_reward, predicted_continuation = self.network(
                    state_tensor, action_tensor
                )
            else:
                # 単一ステップ
                recent_action = actions[-1]
                
                state_array = self.feature_extractor.state_to_array(state)
                action_array = self.feature_extractor.action_to_array(recent_action)
                
                state_array_norm = self.feature_extractor.normalize_state(state_array)
                action_array_norm = self.feature_extractor.normalize_action(action_array)
                
                state_tensor = torch.tensor(
                    state_array_norm, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                action_tensor = torch.tensor(
                    action_array_norm, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                
                predicted_reward, predicted_continuation = self.network(
                    state_tensor, action_tensor
                )
        
        self.network.train()
        
        return {
            'continuation_probability': predicted_continuation.item(),
            'predicted_reward': predicted_reward.item(),
            'reasoning': f'Based on {len(actions)} actions'
        }
