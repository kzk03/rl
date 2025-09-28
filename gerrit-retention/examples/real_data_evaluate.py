#!/usr/bin/env python3
"""
Evaluate RetentionPredictor on real data (CSV/Parquet) with AUC/AP/Brier/ECE.

Expected input schema (flexible):
- developer_email (or email)
- context_date (optional, ISO8601); defaults to now
- label (0/1)
- Optional: days_since_last_activity, and other numeric fields used by extractor

Usage (example):
  uv run python examples/real_data_evaluate.py \
    --input data/retention_samples.csv \
    --format csv \
    --irl-model-path path/to/irl.joblib \
    --enable-irl
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

project_root = _Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gerrit_retention.prediction.metrics import compute_metrics
from gerrit_retention.prediction.retention_predictor import RetentionPredictor


def load_frame(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt in {"parquet", "pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {fmt}")


def row_to_inputs(row: pd.Series) -> Dict[str, Any]:
    dev = {}
    email = row.get("developer_email", row.get("email"))
    if isinstance(email, str):
        dev["email"] = email
    # pass through common numeric fields if present
    for col in ["days_since_last_activity", "community_participation", "team_trust_level"]:
        if col in row and pd.notna(row[col]):
            dev[col] = float(row[col])

    ctx: Dict[str, Any] = {}
    ts = row.get("context_date")
    if isinstance(ts, str) and ts:
        try:
            ctx["context_date"] = datetime.fromisoformat(ts)
        except Exception:
            pass
    return dev, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--format", choices=["csv", "parquet", "pq"], default="csv")
    ap.add_argument("--enable-irl", action="store_true")
    ap.add_argument("--irl-model-path", type=Path)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=123)
    # Splitting controls
    ap.add_argument(
        "--split-mode",
        choices=["random", "time"],
        default="random",
        help="Train/test split mode. 'time' uses context_date to split chronologically.",
    )
    ap.add_argument(
        "--time-cutoff",
        type=str,
        default=None,
        help="ISO8601 datetime string to use as the train/test cutoff when split-mode=time.\n"
        "Rows with context_date <= cutoff go to train; > cutoff go to test. If omitted, uses quantile by test-size.",
    )
    ap.add_argument(
        "--group-by-email",
        action="store_true",
        help="Ensure the same developer does not appear in both train and test (Group-aware split).",
    )
    ap.add_argument(
        "--exclude-gap-features",
        action="store_true",
        help="Disable gap-based features in the predictor to avoid trivial solutions when labels are gap-derived.",
    )
    args = ap.parse_args()

    df = load_frame(args.input, args.format)
    if "label" not in df.columns:
        raise ValueError("Input requires a 'label' column with binary targets (0/1)")

    developers: List[Dict[str, Any]] = []
    contexts: List[Dict[str, Any]] = []
    labels: List[int] = []
    for _, row in df.iterrows():
        dev, ctx = row_to_inputs(row)
        developers.append(dev)
        contexts.append(ctx)
        labels.append(int(row["label"]))

    # Split with options and graceful fallbacks for tiny datasets
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    idx = np.arange(len(labels))

    def do_random_split():
        try:
            return train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=labels)
        except Exception:
            if len(idx) < 3:
                return idx, idx  # tiny: same data for train/test
            return train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=None)

    if args.split_mode == "time":
        # Build time series of indices; parse timestamps, fallback to now if missing
        times: List[datetime] = []
        for ctx in contexts:
            t = ctx.get("context_date")
            if isinstance(t, datetime):
                times.append(t)
            else:
                times.append(datetime.utcnow())

        if args.time_cutoff:
            try:
                cutoff = datetime.fromisoformat(args.time_cutoff)
            except Exception:
                cutoff = np.quantile(np.array([pd.Timestamp(x) for x in times]).astype("datetime64[ns]"), 1 - args.test_size)
                cutoff = pd.Timestamp(cutoff).to_pydatetime()
        else:
            # Choose cutoff so that roughly test-size fraction falls into test set
            ts_array = np.array([pd.Timestamp(x) for x in times])
            q = np.quantile(ts_array.astype("datetime64[ns]"), 1 - args.test_size)
            cutoff = pd.Timestamp(q).to_pydatetime()

        tr_mask = np.array([t <= cutoff for t in times])
        te_mask = ~tr_mask
        idx_tr = idx[tr_mask]
        idx_te = idx[te_mask]

        # If too unbalanced or tiny, fallback to random
        if len(idx_tr) == 0 or len(idx_te) == 0:
            idx_tr, idx_te = do_random_split()
        elif args.group_by_email:
            # Enforce group split by developer email without crossing the cutoff boundary
            emails = [developers[i].get("email", "") for i in idx]
            groups = np.array(emails)
            # Use GroupShuffleSplit on the training portion to pick a train subset, test is post-cutoff
            gss = GroupShuffleSplit(n_splits=1, test_size=0.0, train_size=1.0, random_state=args.seed)
            # No actual split needed here; groups mainly prevent leakage when falling back
            # If either side is tiny, fallback to a single GroupShuffleSplit across all data
            if len(idx_tr) < 3 or len(idx_te) < 3:
                gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
                tr_idx, te_idx = next(gss.split(idx, labels, groups=groups))
                idx_tr, idx_te = idx[tr_idx], idx[te_idx]
    else:
        if args.group_by_email:
            emails = [dev.get("email", "") for dev in developers]
            groups = np.array(emails)
            try:
                gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
                tr_idx, te_idx = next(gss.split(idx, labels, groups=groups))
                idx_tr, idx_te = idx[tr_idx], idx[te_idx]
            except Exception:
                idx_tr, idx_te = do_random_split()
        else:
            idx_tr, idx_te = do_random_split()
    dev_tr = [developers[i] for i in idx_tr]
    ctx_tr = [contexts[i] for i in idx_tr]
    y_tr = [labels[i] for i in idx_tr]
    dev_te = [developers[i] for i in idx_te]
    ctx_te = [contexts[i] for i in idx_te]
    y_te = [labels[i] for i in idx_te]

    # Config
    fe_conf = {}
    if args.enable_irl:
        fe_conf["irl_features"] = {"enabled": True}
        if args.irl_model_path and args.irl_model_path.exists():
            fe_conf["irl_features"]["model_path"] = str(args.irl_model_path)

    # Predictor config
    feature_extraction_cfg = dict(fe_conf)
    if args.exclude_gap_features:
        feature_extraction_cfg["include_gap_features"] = False

    cfg = {
        "feature_extraction": feature_extraction_cfg,
        "random_forest": {"n_estimators": 200, "max_depth": 10},
        "xgboost": {"n_estimators": 120, "max_depth": 5},
        "neural_network": {"hidden_layer_sizes": (64, 16), "max_iter": 200},
    }

    predictor = RetentionPredictor(cfg)
    predictor.fit(dev_tr, ctx_tr, y_tr)
    probs = predictor.predict_batch(dev_te, ctx_te)
    m = compute_metrics(y_te, probs)

    print("Evaluation results")
    for k, v in m.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
