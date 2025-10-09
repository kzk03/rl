#!/usr/bin/env python3
"""MaxEntBinaryIRL を用いてレビュアー着手確率モデルを学習します。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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

from scripts.reviewer_adoption.common import ensure_numeric_state, save_model
from src.gerrit_retention.irl.maxent_binary_irl import IRLConfig, MaxEntBinaryIRL


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = ensure_numeric_state(rec.get("state", {}) or {})
            if not state:
                continue
            rec["state"] = state
            records.append(rec)
    return records


def train_model(train_path: Path, config: IRLConfig) -> Dict[str, object]:
    records = _read_jsonl(train_path)
    transitions = [{"state": r["state"], "action": int(r.get("action", 0))} for r in records]
    model = MaxEntBinaryIRL(config=config)
    info = model.fit(transitions)

    states = [r["state"] for r in records]
    y_true = np.array([int(r.get("action", 0)) for r in records])
    probs = np.array(model.predict_proba(states))
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, probs)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, probs))

    return {
        "model": model,
        "metrics": metrics,
        "train_info": info,
        "records": len(records),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="レビュアー adoption IRL モデルを学習します。")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/reviewer_adoption/model"))
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    cfg = IRLConfig(l2=args.l2, lr=args.lr, epochs=args.epochs, early_stopping_patience=args.patience)
    out = train_model(args.train, cfg)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.json"
    metrics_path = outdir / "metrics.json"

    save_model(out["model"], model_path)
    metrics_data = {
        "train_metrics": out["metrics"],
        "train_info": out["train_info"],
        "records": out["records"],
    }
    metrics_path.write_text(json.dumps(metrics_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(model_path), "metrics": str(metrics_path), "records": out["records"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
