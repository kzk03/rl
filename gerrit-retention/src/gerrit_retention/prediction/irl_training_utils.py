"""Utilities to train and persist MaxEntBinaryIRL models.

These helpers make it easy to build transitions from raw logs,
fit the IRL, and save/load with joblib.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import joblib

try:
    from ..irl.maxent_binary_irl import IRLConfig, MaxEntBinaryIRL
except Exception:  # pragma: no cover
    IRLConfig = Any  # type: ignore
    MaxEntBinaryIRL = Any  # type: ignore


@dataclass
class Transition:
    state: Dict[str, Any]
    action: int  # 1 engage / 0 idle


def build_transitions_from_events(
    events: Sequence[Dict[str, Any]],
    idle_gap_threshold: int = 45,
) -> List[Dict[str, Any]]:
    """Minimal example: build state/action from event list.

    Expects events sorted by time and containing developer-centric fields.
    State includes gap_days estimated from timestamps.
    Action is derived as engage(1) if a review/commit/comment occurs, else 0.
    """
    import math
    from datetime import datetime

    def parse_dt(x: Any) -> datetime | None:
        if isinstance(x, str):
            try:
                return datetime.fromisoformat(x.replace("Z", "+00:00"))
            except Exception:
                return None
        return x if isinstance(x, datetime) else None

    transitions: List[Dict[str, Any]] = []
    prev_ts: datetime | None = None
    for ev in events:
        ts = parse_dt(ev.get("timestamp") or ev.get("date") or ev.get("created"))
        if ts is None:
            continue
        gap_days = float((ts - prev_ts).days) if prev_ts is not None else 0.0
        prev_ts = ts
        state = {
            "gap_days": max(0.0, gap_days),
            "idle_gap_threshold": idle_gap_threshold,
        }
        # pass-through numeric fields as potential extras
        for k, v in ev.items():
            if isinstance(v, (int, float)):
                state.setdefault(k, float(v))
        # action: treat known event types as engage=1
        t = (ev.get("type") or ev.get("action") or "").lower()
        engage = 1 if t in {"commit", "review", "comment", "merge", "push", "change"} else 0
        transitions.append({"state": state, "action": engage})
    return transitions


def fit_maxent_irl(
    transitions: Sequence[Dict[str, Any]],
    config: Any = None,
) -> Any:
    irl = MaxEntBinaryIRL(config=config)
    irl.fit(transitions)
    return irl


def save_irl(irl_model: Any, path: str) -> None:
    joblib.dump(irl_model, path)


def load_irl(path: str) -> Any:
    return joblib.load(path)
