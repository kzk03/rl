from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.gerrit_retention.irl.maxent_binary_irl import IRLConfig, MaxEntBinaryIRL


def model_to_dict(model: MaxEntBinaryIRL) -> Dict[str, Any]:
    weights = model.w.tolist() if model.w is not None else None
    feat_mean = model._feat_mean.tolist() if getattr(model, "_feat_mean", None) is not None else None
    feat_std = model._feat_std.tolist() if getattr(model, "_feat_std", None) is not None else None
    return {
        "weights": weights,
        "feature_names": list(model.feature_names),
        "extra_feature_names": list(getattr(model, "extra_feature_names", [])),
        "max_gap_seen": float(getattr(model, "max_gap_seen", 0.0)),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "config": asdict(model.config),
    }


def save_model(model: MaxEntBinaryIRL, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_to_dict(model), ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(path: Path) -> MaxEntBinaryIRL:
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = IRLConfig(**data.get("config", {}))
    model = MaxEntBinaryIRL(config=cfg)
    weights = data.get("weights")
    if weights is None:
        raise ValueError("Model file missing 'weights'.")
    model.feature_names = list(data.get("feature_names", model.feature_names))
    model.extra_feature_names = list(data.get("extra_feature_names", getattr(model, "extra_feature_names", [])))
    model.max_gap_seen = float(data.get("max_gap_seen", 1.0))
    feat_mean = data.get("feat_mean")
    feat_std = data.get("feat_std")
    model.w = np.array(weights, dtype=float)
    if feat_mean is not None:
        model._feat_mean = np.array(feat_mean, dtype=float)
    if feat_std is not None:
        model._feat_std = np.array(feat_std, dtype=float)
    return model


def ensure_numeric_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert bool to float and drop non-numeric fields to stabilize feature extraction."""
    cleaned: Dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, bool):
            cleaned[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            cleaned[key] = float(value)
        # ignore strings like prev_review, project namesなど
    return cleaned
