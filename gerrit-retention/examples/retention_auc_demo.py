#!/usr/bin/env python3
"""
Retention AUC demo (IRL features vs baseline)

This script trains RetentionPredictor on synthetic data twice:
- Baseline: without IRL-derived features
- IRL+: with IRL-derived features (p/logit/entropy)

Then it prints ROC-AUC for both to illustrate potential benefit.
"""
from __future__ import annotations

import random

# Add src to path if needed
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.prediction.irl_training_utils import (
    build_transitions_from_events,
    fit_maxent_irl,
)
from gerrit_retention.prediction.retention_predictor import RetentionPredictor


def make_synthetic_dataset(n: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    now = datetime.now()

    developers: List[Dict[str, Any]] = []
    contexts: List[Dict[str, Any]] = []
    labels: List[int] = []

    for _ in range(n):
        # days_since_last_activity is the main driver (smaller gap => more likely to retain)
        gap = float(rng.integers(0, 90))  # 0..89 days
        noise = float(rng.normal(0, 1))
        # True propensity via logistic on -(gap) + small noise
        z = -0.08 * gap + 0.5 * noise
        p = 1.0 / (1.0 + np.exp(-z))
        y = 1 if rng.random() < p else 0

        dev = {
            "email": f"dev_{rng.integers(0, 1_000_000)}@example.com",
            "days_since_last_activity": gap,
            # extra optional numeric fields (could be picked up by IRL)
            "community_participation": float(np.clip(rng.beta(2, 5), 0, 1)),
        }
        ctx = {"context_date": now}

        developers.append(dev)
        contexts.append(ctx)
        labels.append(y)

    return developers, contexts, labels


def fit_simple_irl(seed: int = 7):
    rng = np.random.default_rng(seed)
    now = datetime.now()

    # Build a simple event sequence where small gaps imply engage=1 more often
    events = []
    day = now - timedelta(days=120)
    for _ in range(300):
        # sample gap days: short 0-10 or long 20-80
        if rng.random() < 0.6:
            step = int(rng.integers(1, 6))  # short gap
            engage = 1
        else:
            step = int(rng.integers(15, 40))  # long gap
            engage = 0 if rng.random() < 0.8 else 1
        day = day + timedelta(days=step)
        t = "commit" if engage == 1 else "idle"
        events.append({"timestamp": day.isoformat(), "type": t})

    transitions = build_transitions_from_events(events)
    irl = fit_maxent_irl(transitions)
    return irl


def run_demo():
    # Data
    developers, contexts, labels = make_synthetic_dataset()
    idx = np.arange(len(labels))
    idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=123, stratify=labels)

    dev_tr = [developers[i] for i in idx_tr]
    ctx_tr = [contexts[i] for i in idx_tr]
    y_tr = [labels[i] for i in idx_tr]

    dev_te = [developers[i] for i in idx_te]
    ctx_te = [contexts[i] for i in idx_te]
    y_te = [labels[i] for i in idx_te]

    # Baseline predictor (no IRL features)
    base_cfg = {
        "feature_extraction": {
            # IRL disabled by default
        },
        "random_forest": {"n_estimators": 120, "max_depth": 8},
        "xgboost": {"n_estimators": 80, "max_depth": 4},
        "neural_network": {"hidden_layer_sizes": (64, 16), "max_iter": 200},
    }

    base_pred = RetentionPredictor(base_cfg)
    base_pred.fit(dev_tr, ctx_tr, y_tr)
    base_probs = base_pred.predict_batch(dev_te, ctx_te)
    base_auc = roc_auc_score(y_te, base_probs)
    base_ap = average_precision_score(y_te, base_probs)
    base_brier = brier_score_loss(y_te, base_probs)

    # IRL-enhanced predictor
    irl_cfg = {
        "feature_extraction": {
            "irl_features": {
                "enabled": True,
                "idle_gap_threshold": 45,
            }
        },
        "random_forest": {"n_estimators": 120, "max_depth": 8},
        "xgboost": {"n_estimators": 80, "max_depth": 4},
        "neural_network": {"hidden_layer_sizes": (64, 16), "max_iter": 200},
    }

    irl_model = fit_simple_irl()
    irl_pred = RetentionPredictor(irl_cfg)
    irl_pred.set_irl_model(irl_model)
    irl_pred.fit(dev_tr, ctx_tr, y_tr)
    irl_probs = irl_pred.predict_batch(dev_te, ctx_te)
    irl_auc = roc_auc_score(y_te, irl_probs)
    irl_ap = average_precision_score(y_te, irl_probs)
    irl_brier = brier_score_loss(y_te, irl_probs)

    # Expected Calibration Error (ECE)
    def ece(probs, labels, n_bins: int = 10):
        probs = np.asarray(probs)
        labels = np.asarray(labels)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        inds = np.digitize(probs, bins) - 1
        ece_val = 0.0
        total = len(probs)
        for b in range(n_bins):
            mask = inds == b
            if not np.any(mask):
                continue
            conf = probs[mask].mean()
            acc = labels[mask].mean()
            w = mask.mean()
            ece_val += w * abs(acc - conf)
        return float(ece_val)

    base_ece = ece(base_probs, y_te, n_bins=10)
    irl_ece = ece(irl_probs, y_te, n_bins=10)

    print("Baseline AUC:", f"{base_auc:.4f}")
    print("IRL+ AUC:    ", f"{irl_auc:.4f}")
    print("Delta AUC:   ", f"{(irl_auc - base_auc):+.4f}")
    print()
    print("Baseline AP:", f"{base_ap:.4f}", " Brier:", f"{base_brier:.4f}", " ECE:", f"{base_ece:.4f}")
    print("IRL+ AP:    ", f"{irl_ap:.4f}",  " Brier:", f"{irl_brier:.4f}",  " ECE:", f"{irl_ece:.4f}")


if __name__ == "__main__":
    run_demo()
