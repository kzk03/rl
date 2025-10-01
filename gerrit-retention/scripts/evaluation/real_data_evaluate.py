#!/usr/bin/env python3
"""
Evaluate RetentionPredictor on real data (CSV/Parquet) with AUC/AP/Brier/ECE.

Moved from examples/real_data_evaluate.py to scripts/evaluation/real_data_evaluate.py
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

project_root = _Path(__file__).parent.parent.parent
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
    # pass through numeric fields if present (from per-request builder)
    passthrough_cols = [
        # gap and generic
        "days_since_last_activity", "community_participation", "team_trust_level",
        # reviewer activity windows
        "reviewer_past_reviews_30d", "reviewer_past_reviews_90d", "reviewer_past_reviews_180d",
        # owner activity windows
        "owner_past_messages_30d", "owner_past_messages_90d", "owner_past_messages_180d",
        # owner↔reviewer interactions
        "owner_reviewer_past_interactions_180d",
        "owner_reviewer_project_interactions_180d",
        "owner_reviewer_past_assignments_180d",
        # assignment load
        "reviewer_assignment_load_7d", "reviewer_assignment_load_30d", "reviewer_assignment_load_180d",
        # response rate proxy
        "reviewer_past_response_rate_180d",
        # tenure
        "reviewer_tenure_days", "owner_tenure_days",
        # change complexity and flags
        "change_insertions", "change_deletions", "change_files_count",
        "work_in_progress", "subject_len",
    ]
    for col in passthrough_cols:
        if col in row and pd.notna(row[col]):
            try:
                dev[col] = float(row[col])
            except Exception:
                pass
    # auto-pass all path_* similarity/overlap features
    for col in row.index:
        if isinstance(col, str) and col.startswith("path_") and pd.notna(row[col]):
            try:
                dev[col] = float(row[col])
            except Exception:
                pass

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
        help=(
            "ISO8601 datetime string to use as the train/test cutoff when split-mode=time.\n"
            "Rows with context_date <= cutoff go to train; > cutoff go to test. If omitted, uses quantile by test-size."
        ),
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
                import pandas as pd
                cutoff = np.quantile(np.array([pd.Timestamp(x) for x in times]).astype("datetime64[ns]"), 1 - args.test_size)
                cutoff = pd.Timestamp(cutoff).to_pydatetime()
        else:
            import pandas as pd
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
            from sklearn.model_selection import GroupShuffleSplit
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
