"""IRL Feature Adapter

This adapter lets you inject IRL-derived features (probability/logit of engage)
into prediction models without coupling them to IRL implementation details.

Usage:
    from ..irl.maxent_binary_irl import MaxEntBinaryIRL
    from .irl_feature_adapter import IRLFeatureAdapter

    irl = MaxEntBinaryIRL(...)
    irl.fit(transitions)  # fit elsewhere
    adapter = IRLFeatureAdapter(irl)
    # Then attach to RetentionPredictor (see retention_predictor.py)

Notes:
    - This adapter only requires developer/context to build a minimal IRL state.
    - If richer state is available, pass a custom state_builder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    # Local import; avoid hard dependency at import time in callers
    from ..irl.maxent_binary_irl import MaxEntBinaryIRL
except Exception:  # pragma: no cover - optional at runtime
    MaxEntBinaryIRL = object  # type: ignore


StateBuilder = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


@dataclass
class IRLFeatureAdapterConfig:
    idle_gap_threshold: int = 45


class IRLFeatureAdapter:
    """Compute IRL-derived features from developer/context.

    Features returned (in order):
      - irl_p_engage:     sigmoid(logit)
      - irl_logit_engage: log(p/(1-p))
      - irl_entropy:      -(p*log p + (1-p)*log(1-p))
    """

    def __init__(
        self,
        irl_model: Any,
        config: Optional[IRLFeatureAdapterConfig] = None,
        state_builder: Optional[StateBuilder] = None,
    ) -> None:
        self.irl_model = irl_model
        self.config = config or IRLFeatureAdapterConfig()
        self.state_builder = state_builder or self._default_state_builder
        self.feature_names: List[str] = [
            "irl_p_engage",
            "irl_logit_engage",
            "irl_entropy",
        ]

    # ------------------------------------------------------------
    def _default_state_builder(self, developer: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Build a minimal IRL state using best-effort fields.

        Priority for gap_days:
          1) developer['days_since_last_activity'] or context['days_since_last_activity']
          2) Compute from developer['last_activity_date'] or context['last_activity_date'] vs context_date
          3) fallback 0
        """
        gap_days: float = 0.0

        # Direct field if available
        d_gap = developer.get("days_since_last_activity")
        c_gap = context.get("days_since_last_activity")
        if isinstance(d_gap, (int, float)):
            gap_days = float(d_gap)
        elif isinstance(c_gap, (int, float)):
            gap_days = float(c_gap)
        else:
            # Try compute from last_activity_date
            ctx_date = context.get("context_date")
            last_act = developer.get("last_activity_date") or context.get("last_activity_date")
            try:
                if isinstance(ctx_date, str):
                    ctx_date = datetime.fromisoformat(ctx_date.replace("Z", "+00:00"))
                if isinstance(last_act, str):
                    last_act = datetime.fromisoformat(last_act.replace("Z", "+00:00"))
                if isinstance(ctx_date, datetime) and isinstance(last_act, datetime):
                    gap_days = max(0.0, float((ctx_date - last_act).days))
            except Exception:
                gap_days = 0.0

        state = {
            "gap_days": gap_days,
            "idle_gap_threshold": self.config.idle_gap_threshold,
        }

        # Optionally pass through extra numeric fields if present
        # e.g., developer/context numeric keys can give IRL extra features
        for src in (developer, context):
            for k, v in src.items():
                if k in ("gap_days", "idle_gap_threshold", "last_activity_date", "context_date"):
                    continue
                if isinstance(v, (int, float)):
                    # Namespacing: prefer flat keys; IRL will filter/standardize
                    state.setdefault(k, float(v))

        return state

    # ------------------------------------------------------------
    def compute_features(
        self,
        developer: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[str]]:
        """Return IRL-derived features and names.

        Requires a fitted IRL model. If not fitted, will raise at predict time.
        """
        state = self.state_builder(developer, context)
        p_list = self.irl_model.predict_proba([state])  # may raise if not fitted
        p = float(p_list[0]) if p_list else 0.5

        # Compute logit robustly
        eps = 1e-12
        p_clip = max(eps, min(1 - eps, p))
        logit = math.log(p_clip / (1.0 - p_clip))
        entropy = -(p_clip * math.log(p_clip) + (1.0 - p_clip) * math.log(1.0 - p_clip))

        feats = np.array([p, logit, entropy], dtype=float)
        return feats, list(self.feature_names)


__all__ = ["IRLFeatureAdapter", "IRLFeatureAdapterConfig", "StateBuilder"]
