"""
Reusable metrics for retention prediction.

Includes ROC-AUC, Average Precision (PR-AUC), Brier score, and Expected Calibration Error (ECE).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> float:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(probs, bins) - 1
    ece_val = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        w = mask.mean()
        ece_val += w * abs(acc - conf)
    return float(ece_val)


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out: Dict[str, float] = {}
    # ROC-AUC
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    # Average Precision
    try:
        out["average_precision"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["average_precision"] = float("nan")
    # Brier
    try:
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        out["brier"] = float("nan")
    # ECE (well-defined even with single-class but we guard anyway)
    try:
        out["ece"] = float(expected_calibration_error(y_prob, y_true, n_bins=10))
    except Exception:
        out["ece"] = float("nan")
    return out
