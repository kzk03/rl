"""
Cox proportional hazards survival head (skeleton).

Optional dependency: lifelines
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except Exception:
    CoxPHFitter = None
    LIFELINES_AVAILABLE = False


class CoxSurvivalHead:
    """Thin wrapper around lifelines CoxPHFitter for retention hazard modeling.

    Expected training data: duration (time-to-event), event_observed (1 if churn, 0 censored), features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = CoxPHFitter() if LIFELINES_AVAILABLE else None
        self.is_fitted = False

    def fit(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray, feature_names: Optional[List[str]] = None):
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines is not installed; install to use CoxSurvivalHead")
        n = X.shape[0]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df["duration"] = duration
        df["event"] = event
        self.model.fit(df, duration_col="duration", event_col="event")
        self.is_fitted = True

    def predict_partial_hazard(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Cox model not fitted")
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        ph = self.model.predict_partial_hazard(df).values.reshape(-1)
        return ph
