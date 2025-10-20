"""
Reward Network for IRL-based developer continuation prediction.

This module implements a neural network that learns a reward function R(s,a)
from expert trajectories (time series of developer activities).

Training: Learn R(s,a) from time series trajectories
Prediction: Use R(s,a) with snapshot-time features only
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class RewardNetwork(nn.Module):
    """
    Neural network that learns reward function R(state, action).

    Unlike LSTM-based temporal models, this network:
    1. Takes a SINGLE time point (state, action) as input
    2. Outputs a scalar reward value
    3. Can be trained from time series but used with single snapshots

    Architecture:
        Input: concatenate(state, action) -> [state_dim + action_dim]
        Hidden layers with ReLU
        Output: scalar reward value
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize reward network.

        Args:
            state_dim: Dimension of state features (default: 32)
            action_dim: Dimension of action features (default: 9)
            hidden_dim: Hidden layer dimension (default: 128)
            dropout: Dropout rate for regularization (default: 0.3)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Network layers
        input_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Output: scalar reward
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward for (state, action) pair.

        Args:
            state: State features [batch, state_dim] or [state_dim]
            action: Action features [batch, action_dim] or [action_dim]

        Returns:
            reward: Scalar reward [batch, 1] or [1]
        """
        # Handle single sample (no batch dimension)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)  # [batch, state_dim + action_dim]

        # Compute reward
        reward = self.net(x)  # [batch, 1]

        return reward

    def compute_trajectory_reward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cumulative reward for a trajectory.

        This is used during TRAINING to learn from time series data.

        Args:
            states: State sequence [batch, seq_len, state_dim]
            actions: Action sequence [batch, seq_len, action_dim]

        Returns:
            cumulative_reward: Sum of rewards over trajectory [batch, 1]
        """
        batch_size, seq_len, _ = states.shape

        # Compute reward at each time step
        rewards = []
        for t in range(seq_len):
            state_t = states[:, t, :]  # [batch, state_dim]
            action_t = actions[:, t, :]  # [batch, action_dim]

            reward_t = self.forward(state_t, action_t)  # [batch, 1]
            rewards.append(reward_t)

        # Sum rewards over trajectory
        cumulative_reward = torch.stack(rewards, dim=1).sum(dim=1)  # [batch, 1]

        return cumulative_reward


class RewardNetworkTrainer:
    """
    Trainer for learning reward function from expert trajectories.
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 9,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        dropout: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            state_dim: State feature dimension
            action_dim: Action feature dimension
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for optimizer
            dropout: Dropout rate
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create reward network
        self.reward_net = RewardNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.reward_net.parameters(),
            lr=learning_rate
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def train(
        self,
        trajectories: List[Dict],
        labels: List[int],
        epochs: int = 30,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train reward network from trajectories.

        Args:
            trajectories: List of trajectory dicts with 'states' and 'actions'
            labels: List of continuation labels (1=continued, 0=churned)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print progress

        Returns:
            training_info: Dict with training loss history
        """
        self.reward_net.train()

        # Convert to tensors
        states_list = [torch.FloatTensor(t['states']) for t in trajectories]
        actions_list = [torch.FloatTensor(t['actions']) for t in trajectories]
        labels_tensor = torch.FloatTensor(labels).unsqueeze(1)  # [N, 1]

        n_samples = len(trajectories)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle data
            indices = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]

                # Get batch
                batch_states = torch.stack([states_list[idx] for idx in batch_indices]).to(self.device)
                batch_actions = torch.stack([actions_list[idx] for idx in batch_indices]).to(self.device)
                batch_labels = labels_tensor[batch_indices].to(self.device)

                # Forward pass: compute cumulative reward
                cumulative_rewards = self.reward_net.compute_trajectory_reward(
                    batch_states,
                    batch_actions
                )

                # Loss: BCEWithLogitsLoss (cumulative reward as logit)
                # High reward -> high continuation probability
                loss = self.criterion(cumulative_rewards, batch_labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return {'losses': losses, 'final_loss': losses[-1]}

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Predict continuation probability from snapshot-time features.

        This is the KEY DIFFERENCE from LSTM approach:
        - Takes SINGLE time point (state, action)
        - Not a sequence!

        Args:
            state: State features at snapshot time [state_dim]
            action: Action features at snapshot time [action_dim]

        Returns:
            probability: Continuation probability (0-1)
        """
        self.reward_net.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_tensor = torch.FloatTensor(action).to(self.device)

            # Compute reward
            reward = self.reward_net(state_tensor, action_tensor)

            # Convert to probability
            probability = torch.sigmoid(reward).item()

        return probability

    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.reward_net.state_dict(),
            'config': {
                'state_dim': self.reward_net.state_dim,
                'action_dim': self.reward_net.action_dim,
                'hidden_dim': self.reward_net.hidden_dim
            }
        }, path)

    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None) -> 'RewardNetworkTrainer':
        """Load trained model."""
        checkpoint = torch.load(path, map_location=device or 'cpu')
        config = checkpoint['config']

        trainer = cls(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            hidden_dim=config['hidden_dim'],
            device=device
        )

        trainer.reward_net.load_state_dict(checkpoint['model_state_dict'])

        return trainer
