from __future__ import annotations

"""Simplified MaxEnt IRL for binary engage/idle actions.

We model P(a=1|s) = sigmoid(w · φ(s)). Idle (a=0) is complementary.
This reduces to logistic regression but exposed as IRL-style interface.

Features φ(s): configurable function; default uses gap statistics.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np


@dataclass
class IRLConfig:
    l2: float = 1e-2          # 強めの正則化で係数安定
    lr: float = 0.02          # 低め学習率でオーバーフロー低減
    epochs: int = 300
    early_stopping_patience: int = 30
    min_delta: float = 1e-5
    feature_standardize: bool = True  # 特徴量標準化
    clip_z: float = 50.0              # exp 安定化用クリップ


class MaxEntBinaryIRL:
    def __init__(self, feature_fn: Callable[[Dict[str, Any]], np.ndarray] | None = None, config: IRLConfig | None = None):
        self.config = config or IRLConfig()
        self.feature_fn = feature_fn or self.default_feature_fn
        self.w = None  # type: ignore[assignment]
        self.feature_names = ["bias", "gap_norm", "long_gap", "gap_log"]
        self.extra_feature_names = []  # dynamically discovered numeric state features
        self.max_gap_seen = 1.0
        self._feat_mean = None
        self._feat_std = None

    # ------------------------------------------------------------
    def default_feature_fn(self, s: Dict[str, Any]) -> np.ndarray:
        gap = float(s.get("gap_days", 0))
        self.max_gap_seen = max(self.max_gap_seen, gap)
        gap_norm = gap / (self.max_gap_seen or 1.0)
        long_gap = 1.0 if gap > s.get("idle_gap_threshold", 45) else 0.0
        gap_log = math.log1p(gap)
        base = [1.0, gap_norm, long_gap, gap_log]
        # collect extra numeric features (exclude already used keys)
        exclude = {"gap_days", "idle_gap_threshold"}
        numeric_candidates = []
        for k, v in s.items():
            if k in exclude:
                continue
            if k.startswith("prev_"):
                continue
            if isinstance(v, (int, float)) and k not in self.feature_names and k not in self.extra_feature_names:
                numeric_candidates.append(k)
        if numeric_candidates:
            numeric_candidates.sort()
            # freeze order once added
            self.extra_feature_names.extend([k for k in numeric_candidates if k not in self.extra_feature_names])
            self.feature_names = ["bias", "gap_norm", "long_gap", "gap_log"] + self.extra_feature_names
        extras = []
        for k in self.extra_feature_names:
            v = s.get(k, 0.0)
            if not isinstance(v, (int, float)):
                v = 0.0
            extras.append(float(v))
        return np.array(base + extras, dtype=float)

    # ------------------------------------------------------------
    def _loss_and_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ w
        if self.config.clip_z is not None:
            z = np.clip(z, -self.config.clip_z, self.config.clip_z)
        p = 1.0 / (1.0 + np.exp(-z))
        # logistic loss
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        # L2
        loss += 0.5 * self.config.l2 * np.sum(w * w)
        grad = (X.T @ (p - y)) / X.shape[0] + self.config.l2 * w
        return float(loss), grad

    # ------------------------------------------------------------
    def fit(self, transitions: Sequence[Dict[str, Any]]):
        # Two-pass: first collect all candidate extra features to freeze dimensionality
        collected_states: List[Dict[str, Any]] = []
        actions: List[int] = []
        # save current dynamic extras then reset for clean aggregation
        self.extra_feature_names = []
        self.feature_names = ["bias", "gap_norm", "long_gap", "gap_log"]
        # pass 1: gather feature names
        for tr in transitions:
            s = dict(tr.get("state", {}))
            s.setdefault("idle_gap_threshold", 45)
            collected_states.append(s)
            actions.append(int(tr.get("action", 0)))
            _ = self.feature_fn(s)  # this updates extra_feature_names
        # freeze ordering after pass1
        frozen_extra = list(self.extra_feature_names)
        self.extra_feature_names = frozen_extra
        self.feature_names = ["bias", "gap_norm", "long_gap", "gap_log"] + frozen_extra
        # pass 2: build feature matrix with fixed feature list
        feats = []
        for s in collected_states:
            # recompute with fixed extra_feature_names (no new additions)
            gap = float(s.get("gap_days", 0))
            self.max_gap_seen = max(self.max_gap_seen, gap)
            gap_norm = gap / (self.max_gap_seen or 1.0)
            long_gap = 1.0 if gap > s.get("idle_gap_threshold", 45) else 0.0
            gap_log = math.log1p(gap)
            base = [1.0, gap_norm, long_gap, gap_log]
            extras = []
            for k in frozen_extra:
                v = s.get(k, 0.0)
                if not isinstance(v, (int, float)):
                    v = 0.0
                extras.append(float(v))
            feats.append(base + extras)
        if not feats:
            raise ValueError("No transitions provided")
        X = np.array(feats, dtype=float)
        y = np.array(actions, dtype=float)
        # 標準化 (bias列=0番目は除外)
        if self.config.feature_standardize:
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-8
            mean[0] = 0.0
            std[0] = 1.0
            self._feat_mean = mean
            self._feat_std = std
            X = (X - mean) / std
        else:
            self._feat_mean = None
            self._feat_std = None

        self.w = np.zeros(X.shape[1], dtype=float)

        best_loss = float("inf")
        best_w = self.w.copy()
        patience = 0
        for epoch in range(self.config.epochs):
            loss, grad = self._loss_and_grad(X, y, self.w)
            self.w -= self.config.lr * grad
            if loss + self.config.min_delta < best_loss:
                best_loss = loss
                best_w = self.w.copy()
                patience = 0
            else:
                patience += 1
            if patience >= self.config.early_stopping_patience:
                break
        self.w = best_w
        return {"loss": best_loss, "epochs": epoch + 1}

    # ------------------------------------------------------------
    def predict_proba(self, states: Sequence[Dict[str, Any]]) -> List[float]:
        if self.w is None:
            raise RuntimeError("Model not fitted")
        probs = []
        for s in states:
            s2 = dict(s)
            s2.setdefault("idle_gap_threshold", 45)
            # 再構築 (fit の pass2 ロジックを簡易再現)
            gap = float(s2.get("gap_days", 0))
            self.max_gap_seen = max(self.max_gap_seen, gap)
            gap_norm = gap / (self.max_gap_seen or 1.0)
            long_gap = 1.0 if gap > s2.get("idle_gap_threshold", 45) else 0.0
            gap_log = math.log1p(gap)
            base = [1.0, gap_norm, long_gap, gap_log]
            extras = []
            for k in self.extra_feature_names:
                v = s2.get(k, 0.0)
                if not isinstance(v, (int, float)):
                    v = 0.0
                extras.append(float(v))
            f = np.array(base + extras, dtype=float)
            if self._feat_mean is not None and self._feat_std is not None and f.shape[0] == self._feat_mean.shape[0]:
                f = (f - self._feat_mean) / self._feat_std
            z = float(f @ self.w)
            if self.config.clip_z is not None:
                z = max(-self.config.clip_z, min(self.config.clip_z, z))
            p = 1.0 / (1.0 + math.exp(-z))
            probs.append(p)
        return probs

    # ------------------------------------------------------------
    def explain_weights(self) -> Dict[str, float]:
        if self.w is None:
            return {}
        # 逆標準化後の解釈 (標準化時は w/std で元スケール影響)
        if self._feat_std is not None:
            adjusted = []
            for i, val in enumerate(self.w):
                if i == 0:  # bias はそのまま
                    adjusted.append(float(val))
                else:
                    adjusted.append(float(val / self._feat_std[i]))
            return {name: v for name, v in zip(self.feature_names, adjusted)}
        return {name: float(v) for name, v in zip(self.feature_names, self.w)}


__all__ = ["MaxEntBinaryIRL", "IRLConfig"]
