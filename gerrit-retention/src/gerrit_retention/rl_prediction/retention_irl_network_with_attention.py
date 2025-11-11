"""
Attention付きRetentionIRLNetwork

既存のRetentionIRLNetworkにAttention機構を追加
データ準備・訓練ロジックは既存と完全に同じ
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetentionIRLNetworkWithAttention(nn.Module):
    """継続予測のためのIRLネットワーク (Attention付き)"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        sequence: bool = False,
        seq_len: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.sequence = sequence
        self.seq_len = seq_len

        # 状態エンコーダー（既存と同じ）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 行動エンコーダー（既存と同じ）
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.sequence:
            # LSTM for sequence（既存と同じ）
            self.lstm = nn.LSTM(
                hidden_dim // 2,
                hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0 if dropout == 0 else dropout
            )
            
            # ★ Attention層 ★ (新規追加)
            self.attention_weights = nn.Linear(hidden_dim, 1)

        # 報酬予測器（既存と同じ）
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 継続確率予測器（既存と同じ）
        self.continuation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def apply_attention(
        self,
        lstm_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attentionメカニズムを適用
        
        Args:
            lstm_out: LSTMの出力 [batch_size, seq_len, hidden_dim]
            
        Returns:
            context: Attention適用後のコンテキストベクトル [batch_size, hidden_dim]
            attention_weights: Attentionの重み [batch_size, seq_len]
        """
        # Attentionスコア計算
        # shape: (batch, seq_len, 1)
        attention_scores = self.attention_weights(lstm_out)
        
        # Softmaxで正規化
        # shape: (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 重み付け和
        # shape: (batch, hidden_dim)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        return context, attention_weights.squeeze(-1)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向き計算 (Attention付き)
        
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
            
            combined = state_encoded + action_encoded  # Simple addition（既存と同じ）
            
            if lengths is not None:
                # 可変長シーケンスの処理（既存と同じ）
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
            else:
                # 固定長シーケンスの処理（既存と同じ）
                lstm_out, _ = self.lstm(combined)
            
            # ★ Attention適用 ★（新規）
            hidden, attention_weights = self.apply_attention(lstm_out)
        else:
            # Single step mode (スナップショット評価用、既存と同じ)
            state_encoded = self.state_encoder(state)
            action_encoded = self.action_encoder(action)
            combined = torch.cat([state_encoded, action_encoded], dim=1)
            hidden = combined
        
        reward = self.reward_predictor(hidden)
        continuation_prob = self.continuation_predictor(hidden)
        
        if return_hidden:
            return reward, continuation_prob, hidden
        else:
            return reward, continuation_prob
    
    def forward_all_steps(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
            lstm_out, _ = self.lstm(combined)
        
        # ★ Attention適用（全ステップ） ★
        # 各ステップでAttentionを計算するのではなく、全体のコンテキストを使用
        context, _ = self.apply_attention(lstm_out)
        
        # 各ステップで予測（既存と同じ）
        predictions = []
        for t in range(max_seq_len):
            step_hidden = lstm_out[:, t, :]
            step_prob = self.continuation_predictor(step_hidden)
            predictions.append(step_prob)
        
        predictions = torch.cat(predictions, dim=1)
        return predictions
