#!/usr/bin/env python3
"""
Build a per-review-request evaluation CSV from Gerrit changes JSON exported by extract_changes.py.

Row = (change_id, assigned_reviewer_email, request_time)
Label = 1 if the reviewer leaves a message/vote within W days after request_time, else 0.

Outputs columns:
    change_id, project, owner_email, reviewer_email, request_time,
    developer_email (alias of reviewer_email for evaluator compatibility),
    context_date (alias of request_time), responded_within_days,
    label, first_response_time, response_latency_days,
    days_since_last_activity (reviewer last message across any change),
    reviewer_past_reviews_30d, reviewer_past_reviews_90d, reviewer_past_reviews_180d,
    owner_reviewer_past_interactions_180d,
    extraction_date

Note: Requires JSON created with include_details=True so that messages, reviewers, and reviewer_updates exist.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def parse_ts(x: Any) -> datetime | None:
    if not x:
        return None
    if isinstance(x, (int, float)):
        try:
            return datetime.fromtimestamp(float(x))
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip().rstrip("Z")
        # Try common formats
        fmts = (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        )
        for fmt in fmts:
            try:
                return datetime.strptime(s.split(".", 1)[0], fmt)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def load_changes(paths: List[Path]) -> Tuple[List[Dict[str, Any]], datetime]:
    records: List[Dict[str, Any]] = []
    extraction_ts: datetime | None = None
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "data" in obj:
            meta = obj.get("metadata", {})
            ts = meta.get("extraction_timestamp")
            if ts and extraction_ts is None:
                try:
                    extraction_ts = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                except Exception:
                    try:
                        extraction_ts = datetime.fromisoformat(ts)
                    except Exception:
                        extraction_ts = None
            for project, changes in obj.get("data", {}).items():
                for ch in changes:
                    ch = dict(ch)
                    ch.setdefault("_project", project)
                    records.append(ch)
        elif isinstance(obj, list):
            for ch in obj:
                ch = dict(ch)
                ch.setdefault("_project", ch.get("project", "unknown"))
                records.append(ch)
    return records, (extraction_ts or datetime.utcnow())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", type=Path, required=True, help="Path(s) to detailed changes JSON")
    ap.add_argument("--output", type=Path, default=Path("data/review_requests.csv"))
    ap.add_argument("--response-window-days", type=int, default=14, help="W days to consider as responded")
    args = ap.parse_args()

    changes, extraction_ts = load_changes([p for p in args.input if p.exists()])

    rows: List[Dict[str, Any]] = []

    # Precompute global indexes for features
    # - Per-reviewer message index (for gap, past reviews, tenure)
    # - Per-reviewer assignment times (for load windows)
    # We'll do a single pass per change to emit per-request rows after building indexes

    # Build global per-reviewer message index for quick gap computation and response detection
    # messages: list of {author, date, message, _revision_number, ...}
    msgs_by_reviewer: Dict[str, List[datetime]] = defaultdict(list)
    first_msg_time: Dict[str, datetime] = {}

    # Build global per-reviewer assignment index (from reviewer_updates; fallback to created-time reviewers)
    assignments_by_reviewer: Dict[str, List[datetime]] = defaultdict(list)
    for ch in changes:
        for m in (ch.get("messages") or []):
            a = (m.get("author") or {}).get("email")
            dt = parse_ts(m.get("date"))
            if a and dt:
                msgs_by_reviewer[a].append(dt)
                if a not in first_msg_time or dt < first_msg_time[a]:
                    first_msg_time[a] = dt
        # accumulate assignments for index
        reviewer_updates = ch.get("reviewer_updates") or []
        created = parse_ts(ch.get("created"))
        reviewers_info = ch.get("reviewers") or {}
        assigned_events_tmp: List[Tuple[str, datetime]] = []
        for ru in reviewer_updates:
            rv = ru.get("reviewer") or {}
            email = rv.get("email")
            ts = parse_ts(ru.get("updated"))
            state = ru.get("state")
            if email and ts and state in ("REVIEWER", "CC"):
                assigned_events_tmp.append((email, ts))
        if not assigned_events_tmp and created:
            # fallback
            for group in reviewers_info.values():
                if isinstance(group, list):
                    for rv in group:
                        email = (rv or {}).get("email")
                        if email:
                            assigned_events_tmp.append((email, created))
        for email, ts in assigned_events_tmp:
            assignments_by_reviewer[email].append(ts)
    for a, lst in msgs_by_reviewer.items():
        lst.sort()
    for a, lst in assignments_by_reviewer.items():
        lst.sort()

    for ch in changes:
        change_id = ch.get("id") or ch.get("change_id")
        project = ch.get("_project") or ch.get("project")
        owner = ch.get("owner") or {}
        owner_email = (owner or {}).get("email")
        created = parse_ts(ch.get("created"))
        updated = parse_ts(ch.get("updated"))
        messages = ch.get("messages") or []
        reviewer_updates = ch.get("reviewer_updates") or []
        reviewers_info = ch.get("reviewers") or {}
        insertions = ch.get("insertions")
        deletions = ch.get("deletions")
        subject = ch.get("subject") or ""
        is_wip = bool(ch.get("work_in_progress") or False)
        # files count from current revision if available
        files_count = 0
        revisions = ch.get("revisions") or {}
        cur = ch.get("current_revision")
        if isinstance(revisions, dict) and cur and cur in revisions:
            files = (revisions.get(cur, {}) or {}).get("files") or {}
            if isinstance(files, dict):
                files_count = len(files.keys())

        # Index first response time per reviewer
        first_response: Dict[str, datetime] = {}
        for m in messages:
            author = (m.get("author") or {}).get("email")
            dt = parse_ts(m.get("date"))
            if author and dt:
                if author not in first_response or dt < first_response[author]:
                    first_response[author] = dt

        # Build assigned reviewer events: reviewer_updates contains state changes
        # reviewer_updates entries like {"reviewer": {account fields}, "updated": ts, "state": "REVIEWER"/"CC"/"REMOVED"}
        assigned_events: List[Tuple[str, datetime]] = []
        for ru in reviewer_updates:
            rv = ru.get("reviewer") or {}
            email = rv.get("email")
            state = ru.get("state")
            ts = parse_ts(ru.get("updated"))
            if not email or not ts:
                continue
            if state in ("REVIEWER", "CC"):
                assigned_events.append((email, ts))

        # Fallback: if reviewer_updates is empty, use current reviewers list as assigned at created time
        if not assigned_events:
            created_ts = created or (updated or None)
            if created_ts:
                for group in reviewers_info.values():
                    if isinstance(group, list):
                        for rv in group:
                            email = (rv or {}).get("email")
                            if email:
                                assigned_events.append((email, created_ts))

        # For each assignment, determine response within window
        W = timedelta(days=args.response_window_days)

        # Past activity features: simple counts in windows based on messages
        # Build per-reviewer message times for this change (crude; better to build global index if needed)
        # For a lighter pilot, we only compute counts relative to request_time within same change (often 0),
        # but it's still indicative for multi-comment reviewers.

        for reviewer_email, req_time in assigned_events:
            # label: has message or vote by this reviewer within window after req_time
            resp_time = first_response.get(reviewer_email)
            responded = 0
            response_latency_days = None
            if resp_time and resp_time > req_time and (resp_time - req_time) <= W:
                responded = 1
                response_latency_days = (resp_time - req_time).days

            # global reviewer past activity counts via index (before request time)
            owner_reviewer_interactions_180d = 0
            cutoff_30 = req_time - timedelta(days=30)
            cutoff_90 = req_time - timedelta(days=90)
            cutoff_180 = req_time - timedelta(days=180)

            # Count owner-reviewer interactions on this change (weak proxy)
            for m in messages:
                a = (m.get("author") or {}).get("email")
                dt = parse_ts(m.get("date"))
                if not a or not dt or dt >= req_time:
                    continue
                if owner_email and a in (reviewer_email, owner_email) and dt >= cutoff_180:
                    owner_reviewer_interactions_180d += 1

            # Reviewer activity windows using global message times
            r_times = msgs_by_reviewer.get(reviewer_email, [])
            import bisect
            j = bisect.bisect_left(r_times, req_time)
            # slice of messages strictly before req_time
            before = r_times[:j]
            # use bisect to find leftmost indices for cutoffs
            i30 = bisect.bisect_left(before, cutoff_30)
            i90 = bisect.bisect_left(before, cutoff_90)
            i180 = bisect.bisect_left(before, cutoff_180)
            reviewer_30 = len(before) - i30
            reviewer_90 = len(before) - i90
            reviewer_180 = len(before) - i180

            # Owner past message counts
            owner_30 = owner_90 = owner_180 = 0
            if owner_email:
                o_times = msgs_by_reviewer.get(owner_email, [])
                jo = bisect.bisect_left(o_times, req_time)
                obefore = o_times[:jo]
                oi30 = bisect.bisect_left(obefore, cutoff_30)
                oi90 = bisect.bisect_left(obefore, cutoff_90)
                oi180 = bisect.bisect_left(obefore, cutoff_180)
                owner_30 = len(obefore) - oi30
                owner_90 = len(obefore) - oi90
                owner_180 = len(obefore) - oi180

            # Reviewer assignment load windows
            load7 = load30 = 0
            a_times = assignments_by_reviewer.get(reviewer_email, [])
            j2 = bisect.bisect_left(a_times, req_time)
            abefore = a_times[:j2]
            cutoff_7 = req_time - timedelta(days=7)
            ia7 = bisect.bisect_left(abefore, cutoff_7)
            ia30 = bisect.bisect_left(abefore, cutoff_30)
            load7 = len(abefore) - ia7
            load30 = len(abefore) - ia30

            # Tenure and days since last activity (global, any change)
            gap_days = None
            lst = msgs_by_reviewer.get(reviewer_email, [])
            if lst:
                # find rightmost dt < req_time
                import bisect
                i = bisect.bisect_left(lst, req_time) - 1
                if i >= 0:
                    gap_days = (req_time - lst[i]).days
            if gap_days is None:
                gap_days = 10_000
            reviewer_tenure = None
            if reviewer_email in first_msg_time:
                reviewer_tenure = max(0, (req_time - first_msg_time[reviewer_email]).days)
            owner_tenure = None
            if owner_email and owner_email in first_msg_time:
                owner_tenure = max(0, (req_time - first_msg_time[owner_email]).days)

            rows.append({
                "change_id": change_id,
                "project": project,
                "owner_email": owner_email,
                "reviewer_email": reviewer_email,
                "request_time": req_time.isoformat(),
                "developer_email": reviewer_email,
                "context_date": req_time.isoformat(),
                "responded_within_days": args.response_window_days,
                "label": responded,
                "first_response_time": (resp_time.isoformat() if resp_time else None),
                "response_latency_days": response_latency_days,
                "days_since_last_activity": gap_days,
                "reviewer_past_reviews_30d": reviewer_30,
                "reviewer_past_reviews_90d": reviewer_90,
                "reviewer_past_reviews_180d": reviewer_180,
                "owner_past_messages_30d": owner_30,
                "owner_past_messages_90d": owner_90,
                "owner_past_messages_180d": owner_180,
                "owner_reviewer_past_interactions_180d": owner_reviewer_interactions_180d,
                "reviewer_assignment_load_7d": load7,
                "reviewer_assignment_load_30d": load30,
                "reviewer_tenure_days": reviewer_tenure,
                "owner_tenure_days": owner_tenure,
                "change_insertions": insertions,
                "change_deletions": deletions,
                "change_files_count": files_count,
                "work_in_progress": int(1 if is_wip else 0),
                "subject_len": len(subject),
                "extraction_date": extraction_ts.isoformat(),
            })

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
