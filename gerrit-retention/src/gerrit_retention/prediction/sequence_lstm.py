"""
Sequence model (LSTM) skeleton for time-series retention prediction.

Optional dependency: torch
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class SimpleLSTM(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is not installed; install to use SimpleLSTM")
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (B, T, D)
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # last time step
        logits = self.head(h)
        return logits.squeeze(-1)


class LSTMRetentionPredictor:
    """Minimal LSTM-based predictor interface for sequences.

    Expects sequences as np.ndarray of shape (N, T, D).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, lr: float = 1e-3):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is not installed; install to use LSTMRetentionPredictor")
        self.model = SimpleLSTM(input_dim, hidden_dim, num_layers)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        self.model.train()
        ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                self.opt.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.opt.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
