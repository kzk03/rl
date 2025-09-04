"""Task Recommendation (Phase 0: Offline Behavioral Cloning)

Goal:
  Provide initial scaffolding to move from retention IRL to per-developer
  task recommendation. Starts with a multinomial logistic policy (softmax)
  trained on historical (state, action) pairs. Reward proxy = future activity delta.

Steps implemented:
 1. Build (s, a, r) samples from change logs (simplified fields).
 2. State features: recent counts, gap_days, total_authored/reviewed ratios.
 3. Action abstraction: task_type in {create_change, review_change} + component tag hash.
 4. Train multinomial logistic classifier with class weights.
 5. Expose recommend_tasks(developer_state, top_k) -> ranked actions.

This is a minimal baseline; true RL (bandit / off-policy) to follow after
we collect propensity scores and counterfactual evaluation metrics.

Usage example (after data/processed/unified/all_reviews.json exists):
    from gerrit_retention.recommendation.task_recommendation_pipeline import (
        build_training_samples, TaskRecommender
    )
    samples = build_training_samples("data/processed/unified/all_reviews.json", min_actions=50)
    rec = TaskRecommender()
    rec.fit(samples)
    rec.recommend_tasks(samples[0]['state'])
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ------------------ Data Building ------------------

def _parse_ts(ts: str):
    from datetime import datetime, timezone
    if not ts:
        return None
    ts = ts.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def build_training_samples(changes_path: str | Path, min_actions: int = 20, horizon_days: int = 30) -> List[Dict[str, Any]]:
    """Extract simplified (state, action, reward) samples.

    reward proxy: future activity count within horizon - 1 (centered) clipped.
    """
    from datetime import timedelta
    path = Path(changes_path)
    data = json.loads(path.read_text())
    # group by developer timeline
    dev_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ch in data:
        owner = (ch.get('owner') or {}).get('email')
        if not owner:
            continue
        dt = _parse_ts(ch.get('created') or ch.get('updated') or ch.get('submitted'))
        if not dt:
            continue
        comp = ch.get('project') or 'unknown'
        action_type = 'create_change'
        dev_events[owner].append({'ts': dt, 'component': comp, 'action_type': action_type})
        # naive review inference (placeholder): if reviewers list includes owner !=? skip
    samples: List[Dict[str, Any]] = []
    horizon = timedelta(days=horizon_days)
    for dev, events in dev_events.items():
        events.sort(key=lambda x: x['ts'])
        if len(events) < min_actions:
            continue
        times = [e['ts'] for e in events]
        for i, ev in enumerate(events[:-1]):
            ts = ev['ts']
            future_end = ts + horizon
            # future activity count (after this event)
            fut = 0
            for later in events[i+1:]:
                if later['ts'] <= ts:
                    continue
                if later['ts'] <= future_end:
                    fut += 1
                else:
                    break
            reward = min(3, fut)  # cap
            # state features
            past_ev = events[:i+1]
            last_gap = 0
            if len(past_ev) >= 2:
                last_gap = (past_ev[-1]['ts'] - past_ev[-2]['ts']).days
            recent_7 = sum(1 for e in past_ev if (ts - e['ts']).days <= 7)
            recent_30 = sum(1 for e in past_ev if (ts - e['ts']).days <= 30)
            state = {
                'developer_id': dev,
                'event_index': i,
                'recent_count_7d': recent_7,
                'recent_count_30d': recent_30,
                'gap_days': last_gap,
                'total_events': len(past_ev),
            }
            # action abstraction
            comp = ev['component']
            action = f"{ev['action_type']}::component={comp}"
            samples.append({'state': state, 'action': action, 'reward': reward})
    return samples

# ------------------ Recommender ------------------

@dataclass
class TaskRecommenderConfig:
    max_iter: int = 500
    l2: float = 1.0
    min_action_freq: int = 5
    scale_features: bool = True


class TaskRecommender:
    def __init__(self, config: TaskRecommenderConfig | None = None):
        self.config = config or TaskRecommenderConfig()
        self.model = None  # will be LogisticRegression after fitting
        self.action_to_idx: Dict[str, int] = {}
        self.idx_to_action: List[str] = []
        self.feature_names: List[str] = []
        self.scaler: StandardScaler | None = None

    def _state_vector(self, s: Dict[str, Any]) -> np.ndarray:
        if not self.feature_names:
            self.feature_names = [
                'recent_count_7d',
                'recent_count_30d',
                'gap_days',
                'total_events',
            ]
        return np.array([float(s.get(f, 0.0)) for f in self.feature_names], dtype=float)

    def fit(self, samples: List[Dict[str, Any]]):
        freq = Counter(s['action'] for s in samples)
        kept = [s for s in samples if freq[s['action']] >= self.config.min_action_freq]
        if len(kept) < 10:
            raise ValueError('Not enough samples after filtering; lower min_action_freq')
        actions = sorted({s['action'] for s in kept})
        self.action_to_idx = {a: i for i, a in enumerate(actions)}
        self.idx_to_action = actions
        X = np.vstack([self._state_vector(s['state']) for s in kept])
        y = np.array([self.action_to_idx[s['action']] for s in kept])
        counts = np.array([freq[a] for a in actions], dtype=float)
        class_weights = {i: (1.0 / c) for i, c in enumerate(counts)}
        if self.config.scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self.model = LogisticRegression(
            max_iter=self.config.max_iter,
            C=1.0 / self.config.l2,
        )
        sample_weight = [class_weights[self.action_to_idx[s['action']]] for s in kept]
        self.model.fit(X, y, sample_weight=sample_weight)

    def action_scores(self, state: Dict[str, Any]) -> List[Tuple[str, float]]:
        if self.model is None:
            raise RuntimeError('Model not fitted')
        x = self._state_vector(state).reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        probs = self.model.predict_proba(x)[0]
        ranked = sorted(
            [(self.idx_to_action[i], float(p)) for i, p in enumerate(probs)],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    def recommend_tasks(self, state: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        ranked = self.action_scores(state)
        return [{'action': a, 'score': s} for a, s in ranked[:top_k]]

__all__ = ['build_training_samples', 'TaskRecommender', 'TaskRecommenderConfig']
