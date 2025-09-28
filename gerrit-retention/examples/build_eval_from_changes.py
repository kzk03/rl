#!/usr/bin/env python3
"""
Build a developer-level evaluation CSV from Gerrit changes JSON exported by extract_changes.py.

Input JSON schema example (from extract_changes.py):
{
  "metadata": {"extraction_timestamp": "20250101_120000", ...},
  "data": {"project-a": [ {change}, {change}, ... ], "project-b": [...]}
}

For each developer (owner.email), we compute last activity timestamp and derive:
- days_since_last_activity (gap)
- label: 1 if gap <= label_gap_days, else 0  (simple proxy for "currently retained")
- context_date: extraction timestamp

Output CSV columns:
    developer_email, label, days_since_last_activity, context_date,
    total_changes, avg_insertions, avg_deletions, extraction_date
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_changes(paths: List[Path]) -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Support two shapes: {metadata, data: {project: [..]}} or a raw list [..]
        if isinstance(obj, dict) and "data" in obj:
            data = obj.get("data", {})
            for project, changes in data.items():
                for ch in changes:
                    ch = dict(ch)
                    ch["_project"] = project
                    all_records.append(ch)
        elif isinstance(obj, list):
            for ch in obj:
                ch = dict(ch)
                ch["_project"] = ch.get("project", "unknown")
                all_records.append(ch)
        else:
            # Unknown shape; try best-effort: if it looks like a dict of projects -> list
            if isinstance(obj, dict):
                for project, changes in obj.items():
                    if isinstance(changes, list):
                        for ch in changes:
                            ch = dict(ch)
                            ch["_project"] = project
                            all_records.append(ch)
    return all_records


def parse_ts(x: Any) -> datetime | None:
    """Parse various Gerrit timestamp shapes to datetime.

    Handles strings with optional fractional seconds (micro/nano), ISO8601 with or without 'T',
    and numeric epoch seconds.
    """
    if not x:
        return None
    if isinstance(x, (int, float)):
        try:
            return datetime.fromtimestamp(float(x))
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip().rstrip("Z")  # drop trailing Z if present
        # Drop fractional part if present (e.g., .000000000)
        if "." in s:
            s_main = s.split(".", 1)[0]
            # Try with seconds precision first
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(s_main, fmt)
                except Exception:
                    continue
        # Try common patterns without fractional seconds
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        # Fallback to fromisoformat (handles microseconds)
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def load_extraction_context(paths: List[Path]) -> datetime:
    # Use the newest metadata.extraction_timestamp found; fallback to now.
    ts_candidates: List[datetime] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            meta = obj.get("metadata", {})
            ts = meta.get("extraction_timestamp")
            if ts:
                # Try common formats: "%Y%m%d_%H%M%S" or ISO
                try:
                    ts_candidates.append(datetime.strptime(ts, "%Y%m%d_%H%M%S"))
                except Exception:
                    try:
                        ts_candidates.append(datetime.fromisoformat(ts))
                    except Exception:
                        pass
        except Exception:
            pass
    return max(ts_candidates) if ts_candidates else datetime.now()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", type=Path, required=True, help="Path(s) to changes_*.json")
    ap.add_argument("--output", type=Path, default=Path("data/retention_samples.csv"))
    ap.add_argument("--label-gap-days", type=int, default=30, help="Gap threshold for label=1")
    ap.add_argument(
        "--mode",
        choices=["last_activity", "per_change"],
        default="last_activity",
        help=(
            "Row generation mode: "
            "'last_activity' (one row per developer, label based on extraction date gap) or "
            "'per_change' (one row per developer change, label=1 if another activity occurs within label_gap_days after context_date)."
        ),
    )
    args = ap.parse_args()

    paths = [p for p in args.input if p.exists()]
    if not paths:
        raise FileNotFoundError("No valid input JSON files found")

    changes = load_changes(paths)
    extraction_ts = load_extraction_context(paths)

    rows: Dict[str, List[Dict[str, Any]]] = {}
    for ch in changes:
        owner = ch.get("owner") or ch.get("_owner")
        if isinstance(owner, dict):
            email = owner.get("email")
        else:
            email = None
        if not email:
            continue
        created = parse_ts(ch.get("created"))
        updated = parse_ts(ch.get("updated"))
        ts = updated or created
        row = {
            "ts": ts,
            "insertions": float(ch.get("insertions", 0) or 0.0),
            "deletions": float(ch.get("deletions", 0) or 0.0),
        }
        rows.setdefault(email, []).append(row)

    out_records: List[Dict[str, Any]] = []
    for email, recs in rows.items():
        # Filter records with valid timestamps and sort chronologically
        recs_sorted = [r for r in recs if r["ts"] is not None]
        recs_sorted.sort(key=lambda r: r["ts"])

        if args.mode == "per_change" and recs_sorted:
            # Precompute cumulative features up to each change (inclusive)
            ins = np.array([r["insertions"] for r in recs_sorted], dtype=float)
            dels = np.array([r["deletions"] for r in recs_sorted], dtype=float)
            ts = [r["ts"] for r in recs_sorted]

            cum_ins = np.cumsum(ins)
            cum_dels = np.cumsum(dels)

            n = len(ts)
            for i in range(n):
                context_t = ts[i]
                # Look ahead for any activity within label_gap_days strictly after context_t
                horizon = context_t + pd.Timedelta(days=args.label_gap_days)
                has_future = any((t > context_t) and (t <= horizon) for t in ts[i + 1 :])
                label = 1 if has_future else 0

                # Features up to context (inclusive)
                total_changes = i + 1
                avg_ins = float(cum_ins[i] / total_changes)
                avg_del = float(cum_dels[i] / total_changes)
                if i == 0:
                    days_since_last = 10_000  # no prior activity
                else:
                    days_since_last = (context_t - ts[i - 1]).days

                out_records.append({
                    "developer_email": email,
                    "label": label,
                    "days_since_last_activity": days_since_last,
                    "context_date": context_t.isoformat(),
                    "total_changes": total_changes,
                    "avg_insertions": avg_ins,
                    "avg_deletions": avg_del,
                    "extraction_date": extraction_ts.isoformat(),
                })
        else:
            # last_activity mode: one row per developer, label based on extraction date gap
            if recs_sorted:
                last_ts = recs_sorted[-1]["ts"]
                gap_days = (extraction_ts - last_ts).days
            else:
                last_ts = None
                gap_days = 10_000  # effectively very large

            total_changes = len(recs)
            avg_ins = float(np.mean([r["insertions"] for r in recs])) if recs else 0.0
            avg_del = float(np.mean([r["deletions"] for r in recs])) if recs else 0.0

            label = 1 if gap_days <= args.label_gap_days else 0
            # context_date is the developer's last activity (or extraction_ts if unknown)
            context_dt = (last_ts or extraction_ts).isoformat()

            out_records.append({
                "developer_email": email,
                "label": label,
                "days_since_last_activity": gap_days,
                "context_date": context_dt,
                "total_changes": total_changes,
                "avg_insertions": avg_ins,
                "avg_deletions": avg_del,
                "extraction_date": extraction_ts.isoformat(),
            })

    out_df = pd.DataFrame(out_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
