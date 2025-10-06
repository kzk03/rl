"""
Multi-reviewer assignment environment

This Gymnasium environment presents a single change/task with up to K candidate
reviewers at each step. The agent selects one reviewer (Discrete[K]).

Reward options (default):
- match_gt: +1.0 if the selected reviewer is in the ground-truth set (who actually
  participated/accepted), else 0.0. Optional small penalties for invalid actions.

Observation:
- Concatenation of per-candidate feature vectors (fixed order) for up to K candidates
  (zero-padded if fewer). Optionally followed by a binary mask of valid candidates
  (length K) when use_action_mask=True.

This environment is stateless across tasks by default (each step is one task),
but supports episode of multiple tasks sequentially.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Candidate:
    reviewer_id: Any
    features: Dict[str, float]


@dataclass
class AssignmentTask:
    change_id: Any
    candidates: List[Candidate]
    # Ground-truth: reviewers who actually participated/accepted
    positive_reviewer_ids: List[Any]
    # Optional timestamp for splitting
    timestamp: Optional[str] = None


class MultiReviewerAssignmentEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, tasks: Sequence[AssignmentTask], feature_order: Sequence[str], config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.tasks = list(tasks)
        self.feature_order = list(feature_order)
        self.config = config or {}
        self.max_candidates = int(self.config.get("max_candidates", 8))
        self.use_action_mask = bool(self.config.get("use_action_mask", True))
        self.invalid_action_penalty = float(self.config.get("invalid_action_penalty", -0.1))
        self.reward_mode = str(self.config.get("reward_mode", "match_gt"))
        self.temperature = float(self.config.get("temperature", 1.0))  # for IRL softmax scaling
        self.entropy_coef = float(self.config.get("entropy_coef", 0.0))  # optional entropy bonus
        self.hybrid_alpha = float(self.config.get("hybrid_alpha", 0.5))  # weight for hybrid_match_irl
        # IRL scorer: expects dict with 'theta' (np.ndarray [D+1]) and optional 'scaler' (sklearn StandardScaler-like)
        self.irl_model = self.config.get("irl_model")
        # Continuity bonus for same reviewer being repeatedly selected across steps (decay by gap)
        self.continuity_weight = float(self.config.get("continuity_weight", 0.0))
        self.continuity_tau = float(self.config.get("continuity_tau", 2.0))
        self._last_selected_step_by_reviewer = {}
        self.shuffle_tasks_each_reset = bool(self.config.get("shuffle_tasks_each_reset", False))

        self._task_idx = 0

        # Define observation space: K * feature_dim (+ optional K mask)
        self.feature_dim = len(self.feature_order)
        obs_dim = self.max_candidates * self.feature_dim + (self.max_candidates if self.use_action_mask else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_candidates)

        # Precompute zero vectors
        self._zero_feat = np.zeros(self.feature_dim, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._task_idx = 0
        if self.shuffle_tasks_each_reset and len(self.tasks) > 1:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.tasks)
        return self._make_observation(), {}

    # Gymnasium step signature: (obs, reward, terminated, truncated, info)
    def step(self, action: int):
        cur = self._current_task()
        cand_count = min(len(cur.candidates), self.max_candidates)

        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        if action < 0 or action >= self.max_candidates or action >= cand_count:
            # invalid action (selecting a padded slot)
            reward = self.invalid_action_penalty
            info["invalid_action"] = True
        else:
            chosen = cur.candidates[action]
            info["chosen_reviewer_id"] = chosen.reviewer_id
            # Ground-truth match flag (used for evaluation independently of reward mode)
            pos_set = set(cur.positive_reviewer_ids or [])
            is_match = chosen.reviewer_id in pos_set
            info["is_match"] = bool(is_match)
            if self.reward_mode == "match_gt":
                reward = 1.0 if is_match else 0.0
            elif self.reward_mode in ("irl_softmax", "irl_logprob", "hybrid_match_irl", "irl_entropy_bonus"):
                # Utilities
                utils = [self._irl_utility(c.features) for c in cur.candidates[:cand_count]]
                if not utils:
                    utils = [0.0]
                temp = max(1e-6, self.temperature)
                scaled = [(u / temp) for u in utils]
                maxu = float(np.max(scaled))
                exps = [float(np.exp(u - maxu)) for u in scaled]
                denom = max(1e-9, float(sum(exps)))
                probs = [e / denom for e in exps]
                p = probs[action] if action < len(probs) else 0.0
                logp = float(np.log(max(p, 1e-9)))
                if self.reward_mode == "irl_softmax":
                    reward = float(p)
                elif self.reward_mode == "irl_logprob":
                    reward = logp
                elif self.reward_mode == "hybrid_match_irl":
                    base = 1.0 if is_match else 0.0
                    reward = self.hybrid_alpha * base + (1.0 - self.hybrid_alpha) * p
                elif self.reward_mode == "irl_entropy_bonus":
                    entropy = -sum(q * np.log(max(q, 1e-9)) for q in probs)
                    reward = float(p + self.entropy_coef * entropy)
                # Attach debug info
                info["irl_p"] = p
                info["irl_logp"] = logp
                info["irl_entropy"] = float(-sum(q * np.log(max(q, 1e-9)) for q in probs))
            elif self.reward_mode == "accept_prob":
                # Use logistic(sigmoid) of linear utility as acceptance probability
                u = self._irl_utility(chosen.features)
                reward = float(1.0 / (1.0 + np.exp(-u)))
            # Note: if continuity is disabled, do nothing

            # Continuity bonus: if same reviewer selected again after gap
            if self.continuity_weight > 0.0 and chosen.reviewer_id is not None:
                last = self._last_selected_step_by_reviewer.get(chosen.reviewer_id)
                if isinstance(last, int):
                    gap = max(1, self._task_idx - last)
                    cont = self.continuity_weight * float(np.exp(-gap / max(1e-6, self.continuity_tau)))
                    reward += cont
                    info["continuity_bonus"] = cont
                self._last_selected_step_by_reviewer[chosen.reviewer_id] = self._task_idx
            # Note: if continuity is disabled, we do not modify the reward

        # advance to next task
        self._task_idx += 1
        if self._task_idx >= len(self.tasks):
            terminated = True
        obs = self._make_observation()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _current_task(self) -> AssignmentTask:
        # clamp
        idx = min(self._task_idx, len(self.tasks) - 1)
        return self.tasks[idx]

    def _vec_candidate(self, c: Candidate) -> np.ndarray:
        # Order features deterministically, fill missing with 0.0
        return np.array([float(c.features.get(f, 0.0)) for f in self.feature_order], dtype=np.float32)

    def _make_observation(self) -> np.ndarray:
        cur = self._current_task()
        feats: List[np.ndarray] = []
        mask: List[float] = []
        for i in range(self.max_candidates):
            if i < len(cur.candidates):
                feats.append(self._vec_candidate(cur.candidates[i]))
                mask.append(1.0)
            else:
                feats.append(self._zero_feat)
                mask.append(0.0)
        flat = np.concatenate(feats, axis=0).astype(np.float32)
        if self.use_action_mask:
            flat = np.concatenate([flat, np.array(mask, dtype=np.float32)], axis=0)
        return flat

    def _irl_utility(self, feats: Dict[str, float]) -> float:
        if not self.irl_model:
            return 0.0
        theta = self.irl_model.get('theta')
        scaler = self.irl_model.get('scaler')
        if theta is None:
            return 0.0
        x = np.array([float(feats.get(f, 0.0)) for f in self.feature_order], dtype=np.float32)
        if scaler is not None:
            # Accept either a transformer with .transform or a dict with mean/scale
            try:
                if hasattr(scaler, 'transform'):
                    x = scaler.transform([x])[0]
                elif isinstance(scaler, dict):
                    mean = np.array(scaler.get('mean'), dtype=np.float32) if 'mean' in scaler else None
                    scale = np.array(scaler.get('scale'), dtype=np.float32) if 'scale' in scaler else None
                    if mean is not None and scale is not None and mean.shape[0] == x.shape[0] and scale.shape[0] == x.shape[0]:
                        x = (x - mean) / np.maximum(scale, 1e-8)
            except Exception:
                pass
        x_ext = np.concatenate([x.astype(np.float64), np.array([1.0], dtype=np.float64)])  # intercept
        try:
            return float(np.dot(x_ext, theta))
        except Exception:
            return 0.0

    def render(self):  # type: ignore[override]
        cur = self._current_task()
        print(f"Task {self._task_idx+1}/{len(self.tasks)} | candidates={len(cur.candidates)} | positives={len(cur.positive_reviewer_ids)}")

    def close(self):
        pass
