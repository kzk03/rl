#!/usr/bin/env python3
"""
IRL feature integration example

This script shows how to train a simple MaxEntBinaryIRL from event logs,
plug it into RetentionPredictor, and produce retention probabilities.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project src to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.prediction.irl_training_utils import (
    build_transitions_from_events,
    fit_maxent_irl,
    save_irl,
)
from gerrit_retention.prediction.retention_predictor import RetentionPredictor


def main():
    # 1) Mock event sequence for a developer
    now = datetime.now()
    events = [
        {"timestamp": (now - timedelta(days=45)).isoformat(), "type": "commit"},
        {"timestamp": (now - timedelta(days=30)).isoformat(), "type": "review"},
        {"timestamp": (now - timedelta(days=7)).isoformat(), "type": "comment"},
        {"timestamp": (now - timedelta(days=1)).isoformat(), "type": "commit"},
    ]

    # 2) Build IRL transitions and fit
    transitions = build_transitions_from_events(events)
    irl = fit_maxent_irl(transitions)

    # 3) Configure predictor with IRL features enabled
    config = {
        "feature_extraction": {
            "irl_features": {
                "enabled": True,
                "idle_gap_threshold": 45,
            }
        },
        "random_forest": {"n_estimators": 10, "max_depth": 3},
        "xgboost": {"n_estimators": 10, "max_depth": 3},
        "neural_network": {"hidden_layer_sizes": (10,), "max_iter": 50},
    }

    predictor = RetentionPredictor(config)
    predictor.set_irl_model(irl)

    # 4) Train toy model with two developers
    developers = [
        {"email": "dev1@example.com", "days_since_last_activity": 1.0},
        {"email": "dev2@example.com", "days_since_last_activity": 40.0},
    ]
    contexts = [
        {"context_date": now},
        {"context_date": now},
    ]
    labels = [1, 0]

    predictor.fit(developers, contexts, labels)

    # 5) Predict
    for dev in developers:
        prob = predictor.predict_retention_probability(dev, {"context_date": now})
        print(dev["email"], f" -> retention_prob: {prob:.3f}")


if __name__ == "__main__":
    main()
