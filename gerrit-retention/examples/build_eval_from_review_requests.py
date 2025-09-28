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


def _path_sets_from_files(files: Dict[str, Any]) -> Tuple[set, set, set]:
    """Derive file path sets and directory prefix sets (1- and 2-level) from a files dict.
    Returns: (files_set, dir1_set, dir2_set)
    """
    files_set = set()
    dir1_set = set()
    dir2_set = set()
    if isinstance(files, dict):
        for p in files.keys():
            if not isinstance(p, str):
                continue
            # skip Gerrit pseudo files
            if p.startswith("/"):
                continue
            files_set.add(p)
            parts = p.split("/")
            if len(parts) >= 1:
                dir1_set.add(parts[0])
            if len(parts) >= 2:
                dir2_set.add("/".join(parts[:2]))
    return files_set, dir1_set, dir2_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", type=Path, required=True, help="Path(s) to detailed changes JSON")
    ap.add_argument("--output", type=Path, default=Path("data/review_requests.csv"))
    ap.add_argument("--response-window-days", type=int, default=14, help="W days to consider as responded")
    ap.add_argument(
        "--response-type",
        choices=["message", "vote-only", "vote-or-message"],
        default="message",
        help="Define response as message only, vote only, or either (vote or message)",
    )
    ap.add_argument(
        "--interaction-window-days",
        type=int,
        default=14,
        help="±D-day proximity window to count owner↔reviewer interactions in the past 180 days",
    )
    args = ap.parse_args()

    changes, extraction_ts = load_changes([p for p in args.input if p.exists()])

    rows: List[Dict[str, Any]] = []

    # Precompute per-change path sets (files/dir1/dir2)
    change_paths: Dict[str, Tuple[set, set, set]] = {}
    for ch in changes:
        change_id = ch.get("id") or ch.get("change_id")
        revisions = ch.get("revisions") or {}
        cur = ch.get("current_revision")
        if isinstance(revisions, dict) and cur and cur in revisions:
            files = (revisions.get(cur, {}) or {}).get("files") or {}
            if isinstance(files, dict):
                change_paths[change_id] = _path_sets_from_files(files)

    # Precompute global indexes for features
    # - Per-reviewer message index (for gap, past reviews, tenure)
    # - Per-reviewer assignment times (for load windows)
    # We'll do a single pass per change to emit per-request rows after building indexes

    # Build global per-reviewer message index for quick gap computation and response detection
    # messages: list of {author, date, message, _revision_number, ...}
    msgs_by_reviewer: Dict[str, List[datetime]] = defaultdict(list)
    # Project-scoped message index: key = (email, project)
    msgs_by_email_project: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)
    first_msg_time: Dict[str, datetime] = {}

    # Build global per-reviewer assignment index (from reviewer_updates; fallback to created-time reviewers)
    assignments_by_reviewer: Dict[str, List[datetime]] = defaultdict(list)
    # Build owner->reviewer pair assignment history
    assignments_by_pair: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)
    # Reviewer past assigned changes: reviewer -> List[Tuple[datetime, change_id, project]]
    assignments_by_reviewer_changes: Dict[str, List[Tuple[datetime, str, str]]] = defaultdict(list)
    for ch in changes:
        project = ch.get("_project") or ch.get("project") or "unknown"
        change_id = ch.get("id") or ch.get("change_id")
        for m in (ch.get("messages") or []):
            a = (m.get("author") or {}).get("email")
            dt = parse_ts(m.get("date"))
            if a and dt:
                msgs_by_reviewer[a].append(dt)
                msgs_by_email_project[(a, project)].append(dt)
                if a not in first_msg_time or dt < first_msg_time[a]:
                    first_msg_time[a] = dt
        # accumulate assignments for index
        reviewer_updates = ch.get("reviewer_updates") or []
        created = parse_ts(ch.get("created"))
        reviewers_info = ch.get("reviewers") or {}
        owner = ch.get("owner") or {}
        owner_email = (owner or {}).get("email")
        assigned_events_tmp: List[Tuple[str, datetime]] = []
        for ru in reviewer_updates:
            rv = ru.get("reviewer") or {}
            email = rv.get("email")
            ts = parse_ts(ru.get("updated"))
            state = ru.get("state")
            if email and ts and state in ("REVIEWER", "CC"):
                assigned_events_tmp.append((email, ts))
        # Fallback 1: use current reviewers (if present) at created time
        if not assigned_events_tmp and created:
            # fallback
            for group in reviewers_info.values():
                if isinstance(group, list):
                    for rv in group:
                        email = (rv or {}).get("email")
                        if email:
                            assigned_events_tmp.append((email, created))
        # Fallback 2: derive from attention_set entries (if available)
        if not assigned_events_tmp:
            att = ch.get("attention_set") or {}
            if isinstance(att, dict):
                for entry in att.values():
                    try:
                        acc = (entry or {}).get("account") or {}
                        email = acc.get("email")
                        ts = parse_ts((entry or {}).get("last_update"))
                        reason = (entry or {}).get("reason") or ""
                        rl = reason.lower()
                        is_added = (
                            "reviewer was added" in rl or
                            "added as reviewer" in rl or
                            "added to reviewers" in rl or
                            "cc was added" in rl or
                            "added as cc" in rl or
                            "added to cc" in rl
                        )
                        if email and ts and email != owner_email and is_added:
                            assigned_events_tmp.append((email, ts))
                    except Exception:
                        continue
        for email, ts in assigned_events_tmp:
            assignments_by_reviewer[email].append(ts)
            assignments_by_reviewer_changes[email].append((ts, change_id, project))
            if owner_email:
                assignments_by_pair[(owner_email, email)].append(ts)
    for a, lst in msgs_by_reviewer.items():
        lst.sort()
    for k, lst in msgs_by_email_project.items():
        lst.sort()
    for a, lst in assignments_by_reviewer.items():
        lst.sort()
    for k, lst in assignments_by_pair.items():
        lst.sort()
    for a, lst in assignments_by_reviewer_changes.items():
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

        # Index first response time per reviewer (messages)
        first_response: Dict[str, datetime] = {}
        for m in messages:
            author = (m.get("author") or {}).get("email")
            dt = parse_ts(m.get("date"))
            if author and dt:
                if author not in first_response or dt < first_response[author]:
                    first_response[author] = dt

        # Index first vote time per reviewer (from labels)
        first_vote: Dict[str, datetime] = {}
        labels = ch.get("labels") or {}
        if isinstance(labels, dict):
            for lbl, info in labels.items():
                if not isinstance(info, dict):
                    continue
                all_votes = info.get("all") or []
                if not isinstance(all_votes, list):
                    continue
                for ent in all_votes:
                    try:
                        email = (ent or {}).get("email") or (ent or {}).get("username") or (ent or {}).get("name")
                        # Gerrit returns 0 for no vote; ignore zeros and null
                        val = (ent or {}).get("value")
                        dt = parse_ts((ent or {}).get("date"))
                        if email and dt and isinstance(val, (int, float)) and val != 0:
                            if email not in first_vote or dt < first_vote[email]:
                                first_vote[email] = dt
                    except Exception:
                        continue

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
        # Fallback: attention_set-based approximation
        if not assigned_events:
            att = ch.get("attention_set") or {}
            if isinstance(att, dict):
                for entry in att.values():
                    try:
                        acc = (entry or {}).get("account") or {}
                        email = acc.get("email")
                        ts = parse_ts((entry or {}).get("last_update"))
                        reason = (entry or {}).get("reason") or ""
                        rl = reason.lower()
                        is_added = (
                            "reviewer was added" in rl or
                            "added as reviewer" in rl or
                            "added to reviewers" in rl or
                            "cc was added" in rl or
                            "added as cc" in rl or
                            "added to cc" in rl
                        )
                        if email and ts and email != owner_email and is_added:
                            assigned_events.append((email, ts))
                    except Exception:
                        continue
        # Fallback: message text heuristic (very conservative)
        if not assigned_events and messages:
            import re
            rx_rev = re.compile(r"\badded\b.*\breviewer(s)?\b", re.I)
            rx_cc = re.compile(r"\badded\b.*\bcc\b", re.I)
            any_added = False
            for m in messages:
                txt = (m.get("message") or "").strip()
                if rx_rev.search(txt) or rx_cc.search(txt):
                    any_added = True
                    break
            if any_added and created:
                # As last resort, assume attention_set accounts at created time are candidates
                att = ch.get("attention_set") or {}
                if isinstance(att, dict):
                    for entry in att.values():
                        acc = (entry or {}).get("account") or {}
                        email = acc.get("email")
                        if email and email != owner_email:
                            assigned_events.append((email, created))

    # For each assignment, determine response within window
        W = timedelta(days=args.response_window_days)

    # Past activity features: use global indexes for reviewer/owner activity and interactions

        for reviewer_email, req_time in assigned_events:
            # label: based on response-type -> message, vote-only, or vote-or-message
            msg_time = first_response.get(reviewer_email)
            vote_time = first_vote.get(reviewer_email)
            first_vote_time_iso = None
            vote_latency_days = None
            resp_time = None
            responded = 0
            response_latency_days = None
            # Check window membership helpers
            def in_window(t: datetime | None) -> bool:
                return bool(t and t > req_time and (t - req_time) <= W)
            msg_ok = in_window(msg_time)
            vote_ok = in_window(vote_time)
            if args.response_type == "message":
                if msg_ok:
                    responded = 1
                    resp_time = msg_time
            elif args.response_type == "vote-only":
                if vote_ok:
                    responded = 1
                    resp_time = vote_time
            else:  # vote-or-message
                if msg_ok or vote_ok:
                    responded = 1
                    # choose earliest within window
                    candidates = [t for t in [msg_time, vote_time] if in_window(t)]
                    if candidates:
                        resp_time = min(candidates)
            if resp_time:
                response_latency_days = (resp_time - req_time).days
            if vote_time and in_window(vote_time):
                first_vote_time_iso = vote_time.isoformat()
                vote_latency_days = (vote_time - req_time).days

            # global reviewer past activity counts via index (before request time)
            owner_reviewer_interactions_180d = 0
            cutoff_30 = req_time - timedelta(days=30)
            cutoff_90 = req_time - timedelta(days=90)
            cutoff_180 = req_time - timedelta(days=180)

            # Global owner↔reviewer interactions (messages proximity) in past 180d, within ±interaction_window_days
            # Idea: for each owner message time t_o in [req_time-180d, req_time),
            # count reviewer message times t_r such that |t_r - t_o| <= D days.
            # Both sides strictly before the assignment time (to avoid leakage).
            import bisect
            D = timedelta(days=int(max(0, args.interaction_window_days)))
            if owner_email and reviewer_email and owner_email != reviewer_email:
                o_times = msgs_by_reviewer.get(owner_email, [])
                r_times = msgs_by_reviewer.get(reviewer_email, [])
                # Clip to [cutoff_180, req_time)
                jo_l = bisect.bisect_left(o_times, cutoff_180)
                jo_r = bisect.bisect_left(o_times, req_time)
                jr_l = bisect.bisect_left(r_times, cutoff_180)
                jr_r = bisect.bisect_left(r_times, req_time)
                o_slice = o_times[jo_l:jo_r]
                r_slice = r_times[jr_l:jr_r]
                # For each owner time, count reviewers within [t-D, t+D]
                for t in o_slice:
                    lo = bisect.bisect_left(r_slice, t - D)
                    hi = bisect.bisect_right(r_slice, t + D)
                    if hi > lo:
                        owner_reviewer_interactions_180d += (hi - lo)

            # Project-scoped owner↔reviewer interactions within the same project
            owner_reviewer_project_interactions_180d = 0
            if owner_email and reviewer_email and owner_email != reviewer_email and project:
                o_times_p = msgs_by_email_project.get((owner_email, project), [])
                r_times_p = msgs_by_email_project.get((reviewer_email, project), [])
                po_l = bisect.bisect_left(o_times_p, cutoff_180)
                po_r = bisect.bisect_left(o_times_p, req_time)
                pr_l = bisect.bisect_left(r_times_p, cutoff_180)
                pr_r = bisect.bisect_left(r_times_p, req_time)
                o_p_slice = o_times_p[po_l:po_r]
                r_p_slice = r_times_p[pr_l:pr_r]
                for t in o_p_slice:
                    lo = bisect.bisect_left(r_p_slice, t - D)
                    hi = bisect.bisect_right(r_p_slice, t + D)
                    if hi > lo:
                        owner_reviewer_project_interactions_180d += (hi - lo)

            # Reviewer activity windows using global message times
            r_times = msgs_by_reviewer.get(reviewer_email, [])
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
            load7 = load30 = load180 = 0
            a_times = assignments_by_reviewer.get(reviewer_email, [])
            j2 = bisect.bisect_left(a_times, req_time)
            abefore = a_times[:j2]
            cutoff_7 = req_time - timedelta(days=7)
            ia7 = bisect.bisect_left(abefore, cutoff_7)
            ia30 = bisect.bisect_left(abefore, cutoff_30)
            ia180 = bisect.bisect_left(abefore, cutoff_180)
            load7 = len(abefore) - ia7
            load30 = len(abefore) - ia30
            load180 = len(abefore) - ia180

            # Owner->Reviewer pair past assignments (global)
            pair_assign_180 = 0
            if owner_email and reviewer_email:
                pr_times = assignments_by_pair.get((owner_email, reviewer_email), [])
                k_hi = bisect.bisect_left(pr_times, req_time)
                k_lo = bisect.bisect_left(pr_times, cutoff_180)
                pair_assign_180 = max(0, k_hi - k_lo)

            # Path familiarity using reviewer's past assigned changes in last 180d
            files_set = set()
            dir1_set = set()
            dir2_set = set()
            if change_id in change_paths:
                files_set, dir1_set, dir2_set = change_paths[change_id]
            past_files_global: set = set()
            past_dir1_global: set = set()
            past_dir2_global: set = set()
            past_files_proj: set = set()
            past_dir1_proj: set = set()
            past_dir2_proj: set = set()
            past_assigned = assignments_by_reviewer_changes.get(reviewer_email, [])
            for ts, ch_id, prj in past_assigned:
                if ts < cutoff_180 or ts >= req_time:
                    continue
                if ch_id in change_paths:
                    pf, pd1, pd2 = change_paths[ch_id]
                    past_files_global |= pf
                    past_dir1_global |= pd1
                    past_dir2_global |= pd2
                    if prj == project:
                        past_files_proj |= pf
                        past_dir1_proj |= pd1
                        past_dir2_proj |= pd2
            # overlap counts and similarity ratios
            def jaccard(a:set,b:set)->float:
                return (len(a & b) / len(a | b)) if (a or b) else 0.0
            def dice(a:set,b:set)->float:
                inter = len(a & b)
                return (2*inter / (len(a)+len(b))) if (a or b) else 0.0
            def overlap_coeff(a:set,b:set)->float:
                inter = len(a & b)
                denom = min(len(a), len(b))
                return (inter / denom) if denom > 0 else 0.0
            def cosine_sim(a:set,b:set)->float:
                # cosine on binary vectors reduces to |A∩B| / sqrt(|A|*|B|)
                inter = len(a & b)
                denom = (len(a)*len(b))**0.5
                return (inter / denom) if denom > 0 else 0.0
            overlap_files_global = len(files_set & past_files_global)
            overlap_dir1_global = len(dir1_set & past_dir1_global)
            overlap_dir2_global = len(dir2_set & past_dir2_global)
            jacc_files_global = jaccard(files_set, past_files_global)
            jacc_dir1_global = jaccard(dir1_set, past_dir1_global)
            jacc_dir2_global = jaccard(dir2_set, past_dir2_global)
            dice_files_global = dice(files_set, past_files_global)
            dice_dir1_global = dice(dir1_set, past_dir1_global)
            dice_dir2_global = dice(dir2_set, past_dir2_global)
            ovl_files_global = overlap_coeff(files_set, past_files_global)
            ovl_dir1_global = overlap_coeff(dir1_set, past_dir1_global)
            ovl_dir2_global = overlap_coeff(dir2_set, past_dir2_global)
            cos_files_global = cosine_sim(files_set, past_files_global)
            cos_dir1_global = cosine_sim(dir1_set, past_dir1_global)
            cos_dir2_global = cosine_sim(dir2_set, past_dir2_global)
            overlap_files_proj = len(files_set & past_files_proj)
            overlap_dir1_proj = len(dir1_set & past_dir1_proj)
            overlap_dir2_proj = len(dir2_set & past_dir2_proj)
            jacc_files_proj = jaccard(files_set, past_files_proj)
            jacc_dir1_proj = jaccard(dir1_set, past_dir1_proj)
            jacc_dir2_proj = jaccard(dir2_set, past_dir2_proj)
            dice_files_proj = dice(files_set, past_files_proj)
            dice_dir1_proj = dice(dir1_set, past_dir1_proj)
            dice_dir2_proj = dice(dir2_set, past_dir2_proj)
            ovl_files_proj = overlap_coeff(files_set, past_files_proj)
            ovl_dir1_proj = overlap_coeff(dir1_set, past_dir1_proj)
            ovl_dir2_proj = overlap_coeff(dir2_set, past_dir2_proj)
            cos_files_proj = cosine_sim(files_set, past_files_proj)
            cos_dir1_proj = cosine_sim(dir1_set, past_dir1_proj)
            cos_dir2_proj = cosine_sim(dir2_set, past_dir2_proj)

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
                "first_vote_time": first_vote_time_iso,
                "vote_latency_days": vote_latency_days,
                "days_since_last_activity": gap_days,
                "reviewer_past_reviews_30d": reviewer_30,
                "reviewer_past_reviews_90d": reviewer_90,
                "reviewer_past_reviews_180d": reviewer_180,
                "owner_past_messages_30d": owner_30,
                "owner_past_messages_90d": owner_90,
                "owner_past_messages_180d": owner_180,
                "owner_reviewer_past_interactions_180d": owner_reviewer_interactions_180d,
                "owner_reviewer_project_interactions_180d": owner_reviewer_project_interactions_180d,
                "owner_reviewer_past_assignments_180d": pair_assign_180,
                "reviewer_assignment_load_7d": load7,
                "reviewer_assignment_load_30d": load30,
                "reviewer_assignment_load_180d": load180,
                # Proxy response rate in 180d = past reviews (messages) / assignments (clip to [0,1])
                "reviewer_past_response_rate_180d": float(min(1.0, (reviewer_180 / load180))) if load180 > 0 else 0.0,
                "reviewer_tenure_days": reviewer_tenure,
                "owner_tenure_days": owner_tenure,
                "change_insertions": insertions,
                "change_deletions": deletions,
                "change_files_count": files_count,
                "work_in_progress": int(1 if is_wip else 0),
                "subject_len": len(subject),
                # path familiarity (proxy-based for pilot)
                "path_overlap_files_global": overlap_files_global,
                "path_overlap_dir1_global": overlap_dir1_global,
                "path_overlap_dir2_global": overlap_dir2_global,
                "path_jaccard_files_global": jacc_files_global,
                "path_jaccard_dir1_global": jacc_dir1_global,
                "path_jaccard_dir2_global": jacc_dir2_global,
                "path_overlap_files_project": overlap_files_proj,
                "path_overlap_dir1_project": overlap_dir1_proj,
                "path_overlap_dir2_project": overlap_dir2_proj,
                "path_jaccard_files_project": jacc_files_proj,
                "path_jaccard_dir1_project": jacc_dir1_proj,
                "path_jaccard_dir2_project": jacc_dir2_proj,
                # additional similarity metrics
                "path_dice_files_global": dice_files_global,
                "path_dice_dir1_global": dice_dir1_global,
                "path_dice_dir2_global": dice_dir2_global,
                "path_overlap_coeff_files_global": ovl_files_global,
                "path_overlap_coeff_dir1_global": ovl_dir1_global,
                "path_overlap_coeff_dir2_global": ovl_dir2_global,
                "path_cosine_files_global": cos_files_global,
                "path_cosine_dir1_global": cos_dir1_global,
                "path_cosine_dir2_global": cos_dir2_global,
                "path_dice_files_project": dice_files_proj,
                "path_dice_dir1_project": dice_dir1_proj,
                "path_dice_dir2_project": dice_dir2_proj,
                "path_overlap_coeff_files_project": ovl_files_proj,
                "path_overlap_coeff_dir1_project": ovl_dir1_proj,
                "path_overlap_coeff_dir2_project": ovl_dir2_proj,
                "path_cosine_files_project": cos_files_proj,
                "path_cosine_dir1_project": cos_dir1_proj,
                "path_cosine_dir2_project": cos_dir2_proj,
                "extraction_date": extraction_ts.isoformat(),
            })

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
