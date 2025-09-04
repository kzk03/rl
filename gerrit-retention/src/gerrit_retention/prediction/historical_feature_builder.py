"""Historical feature reconstruction utilities.

Rebuild per-snapshot developer aggregate features strictly using changes
up to (and excluding future beyond) the snapshot date to avoid leakage.

Provided features (subset expected by WorkloadAwarePredictor / analyzers):
 - changes_authored
 - changes_reviewed (appearances as reviewer/CC)
 - total_insertions / total_deletions (from authored changes)
 - projects (list of distinct authored project ids up to snapshot)
 - last_activity (snapshot date ISO)
 - first_seen (earliest authored or master first_seen)

Review scores: Gerrit raw change objects do not easily expose per-review numeric
scores in a uniform array; we omit or leave empty to avoid fabricating data.
If later a message or label parser is added, integrate here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class DevChangeIndex:
    authored: List[Tuple[datetime, int, int, str]]  # (created, insertions, deletions, project)
    reviews: List[datetime]  # times where dev appears as reviewer/CC
    first_seen_master: datetime | None


def build_change_index(changes: List[Dict[str, Any]]) -> Dict[str, DevChangeIndex]:
    """Pre-index changes by developer id for authored and review appearances."""
    from .historical_feature_builder import (
        parse_dt,  # type: ignore  # forward reference
    )
    from .workload_aware_predictor import logger  # reuse logger
    index: Dict[str, DevChangeIndex] = {}

    def ensure(dev_id: str) -> DevChangeIndex:
        if dev_id not in index:
            index[dev_id] = DevChangeIndex([], [], None)
        return index[dev_id]

    for ch in changes:
        created_raw = ch.get('created') or ch.get('updated') or ch.get('submitted')
        created = parse_dt(created_raw)
        if not created:
            continue
        owner = ch.get('owner') or {}
        dev_id = owner.get('email') or owner.get('username')
        if dev_id:
            dc = ensure(dev_id)
            dc.authored.append((created, int(ch.get('insertions', 0)), int(ch.get('deletions', 0)), ch.get('project', '')))
            if dc.first_seen_master is None or created < dc.first_seen_master:
                dc.first_seen_master = created
        reviewers = ch.get('reviewers') or {}
        for role in ('REVIEWER', 'CC'):
            for rv in reviewers.get(role, []) or []:
                rid = rv.get('email') or rv.get('username')
                if not rid:
                    continue
                dt = created  # approximate invitation time
                ensure(rid).reviews.append(dt)

    # sort lists
    for dc in index.values():
        dc.authored.sort(key=lambda x: x[0])
        dc.reviews.sort()
    return index


def parse_dt(ts: str | None):
    from datetime import timezone
    if not ts:
        return None
    ts = ts.replace('Z', '+00:00').replace(' ', 'T')
    try:
        # truncate microseconds if needed
        if '.' in ts:
            head, tail = ts.split('.', 1)
            if '+' in tail:
                frac, tz = tail.split('+', 1)
                if len(frac) > 6:
                    frac = frac[:6]
                ts = f"{head}.{frac}+{tz}"
            elif '-' in tail and tail.count('-') > 1:
                frac, tz = tail.split('-', 1)
                if len(frac) > 6:
                    frac = frac[:6]
                ts = f"{head}.{frac}-{tz}"
            else:
                if len(tail) > 6:
                    ts = f"{head}.{tail[:6]}"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _count_until(sorted_list: List, key_fn, cutoff: datetime) -> int:
    # binary search manual to avoid bisect import overhead (micro-opt not critical)
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if key_fn(sorted_list[mid]) <= cutoff:
            lo = mid + 1
        else:
            hi = mid
    return lo


def build_snapshot_features(dev_id: str, snapshot: datetime, dev_master: Dict[str, Any], index: DevChangeIndex) -> Dict[str, Any]:
    authored = index.authored
    reviews = index.reviews
    # counts up to snapshot (inclusive)
    authored_n = _count_until(authored, lambda x: x[0], snapshot)
    review_n = _count_until(reviews, lambda x: x, snapshot)
    total_insertions = sum(a[1] for a in authored[:authored_n])
    total_deletions = sum(a[2] for a in authored[:authored_n])
    projects = list({a[3] for a in authored[:authored_n] if a[3]})
    first_seen_master = index.first_seen_master
    master_first_seen = dev_master.get('first_seen')
    if isinstance(master_first_seen, str):
        fs_parsed = parse_dt(master_first_seen)
    else:
        fs_parsed = None
    first_seen = min([d for d in [first_seen_master, fs_parsed] if d] or [snapshot])
    feat = {
        'developer_id': dev_id,
        'changes_authored': authored_n,
        'changes_reviewed': review_n,
        'total_insertions': total_insertions,
        'total_deletions': total_deletions,
        'projects': projects,
        'last_activity': snapshot.isoformat(),
        'first_seen': first_seen.isoformat() if first_seen else snapshot.isoformat(),
        # keep any stable meta fields
        'sources': dev_master.get('sources', []),
    }
    return feat
