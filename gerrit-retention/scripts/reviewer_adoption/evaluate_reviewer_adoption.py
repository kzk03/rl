#!/usr/bin/env python3
"""学習済みレビュアー adoption IRL モデルを評価します。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.reviewer_adoption.common import ensure_numeric_state, load_model


def _read_jsonl(path: Path) -> Tuple[List[Dict[str, object]], np.ndarray, List[Dict[str, float]]]:
    records: List[Dict[str, object]] = []
    states: List[Dict[str, float]] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = ensure_numeric_state(rec.get("state", {}) or {})
            if not state:
                continue
            action = int(rec.get("action", 0))
            records.append(rec)
            states.append(state)
            labels.append(action)
    return records, np.array(labels, dtype=int), states


def _ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> Dict[str, object]:
    if probs.size == 0:
        return {"ece": None, "bins": []}
    bins = max(1, bins)
    bin_indices = np.clip((probs * bins).astype(int), 0, bins - 1)
    total = len(probs)
    entries = []
    ece = 0.0
    for b in range(bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        prob_bin = float(np.mean(probs[mask]))
        acc_bin = float(np.mean(labels[mask]))
        weight = float(np.sum(mask)) / total
        ece += weight * abs(acc_bin - prob_bin)
        entries.append({
            "bin": b,
            "count": int(np.sum(mask)),
            "avg_prob": prob_bin,
            "avg_label": acc_bin,
        })
    return {"ece": ece, "bins": entries}


def main() -> int:
    ap = argparse.ArgumentParser(description="レビュアー adoption IRL モデルの評価指標を計算します。")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/reviewer_adoption/eval_metrics.json"))
    ap.add_argument("--ece-bins", type=int, default=10)
    args = ap.parse_args()

    records, labels, states = _read_jsonl(args.data)
    model = load_model(args.model)
    probs = np.array(model.predict_proba(states))
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "count": int(len(records)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "positive_rate": float(np.mean(labels)),
        "brier": float(brier_score_loss(labels, probs)),
    }
    if len(np.unique(labels)) > 1:
        metrics["auc"] = float(roc_auc_score(labels, probs))

    cal = _ece(probs, labels, bins=args.ece_bins)
    metrics["ece"] = cal["ece"]
    metrics["calibration_bins"] = cal["bins"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
