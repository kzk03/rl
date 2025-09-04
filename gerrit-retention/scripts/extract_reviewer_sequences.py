#!/usr/bin/env python3
"""Extract reviewer engagement sequences (appearances as reviewer/CC).

Enhancements (Option A - stronger negatives):
 1. Configurable shorter idle gap threshold via --idle-gap (default 40 instead of 60).
 2. Idle insertion mode: by default we insert ONLY the idle transition (0) when gap > threshold and do NOT also append an immediate engage; instead the engage is represented by the next real event naturally (reduces positive inflation).
 3. Synthetic tail negatives: if the final event is older than --tail-inactive-days (e.g. 90), append one extra synthetic idle transition to represent churn risk.
 4. Optional max positives cap per reviewer to prevent extreme imbalance (--max-positives-per-user).
 5. All numeric dynamic activity features added (activity_7d,30d,90d etc.) reused by IRL.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_dt(ts: str):
    if not ts: return None
    ts = ts.replace('Z','+00:00').replace(' ', 'T')
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except: return None

def load_changes(path: Path):
    with path.open('r', encoding='utf-8') as f: return json.load(f)

def build_sequences(
    changes: List[Dict[str, Any]],
    idle_gap_threshold: int = 40,
    insert_idle_only: bool = True,
    tail_inactive_days: int = 90,
    max_positives_per_user: int | None = None,
):
    review_map: Dict[str, List[datetime]] = {}
    for ch in changes:
        created = parse_dt(ch.get('created'))
        if not created: continue
        reviewers = ch.get('reviewers') or {}
        for role in ('REVIEWER','CC'):
            for rv in reviewers.get(role, []) or []:
                dev = rv.get('email') or rv.get('username')
                if not dev: continue
                review_map.setdefault(dev, []).append(created)
    for v in review_map.values(): v.sort()
    sequences = []
    for dev, times in review_map.items():
        if len(times) < 2: continue
        ts_list = times  # sorted
        ts_seconds = [int(t.timestamp()) for t in ts_list]
        trans = []
        gaps_so_far: List[int] = []
        positive_count = 0
        for i in range(1, len(ts_list)):
            prev = ts_list[i-1]
            t = ts_list[i]
            gap = (t - prev).days
            gaps_so_far.append(gap)
            ref_time = prev
            def count_within(days: int) -> int:
                start_ts = int((ref_time - timedelta(days=days)).timestamp())
                left = bisect.bisect_left(ts_seconds, start_ts, 0, i)
                return i - left
            act_7 = count_within(7)
            act_30 = count_within(30)
            act_90 = count_within(90)
            ratio_7_30 = act_7 / act_30 if act_30 else 0.0
            ratio_30_90 = act_30 / act_90 if act_90 else 0.0
            avg_gap_recent = (sum(gaps_so_far[-5:]) / min(len(gaps_so_far),5)) if gaps_so_far else gap
            workload_level = act_30 / 30.0
            burnout_risk = 1.0 if (act_30 >= 50 and avg_gap_recent < 2) or (act_7 >= 20 and avg_gap_recent < 1.5) else 0.0
            unique_days_90 = len({d.date() for d in ts_list[:i] if ref_time - d <= timedelta(days=90)})
            expertise_recent = unique_days_90 / 90.0
            state = {
                'prev_review': prev.isoformat(),
                'gap_days': gap,
                'idle_gap_threshold': idle_gap_threshold,
                'activity_7d': float(act_7),
                'activity_30d': float(act_30),
                'activity_90d': float(act_90),
                'activity_ratio_7_30': float(ratio_7_30),
                'activity_ratio_30_90': float(ratio_30_90),
                'avg_gap_recent5': float(avg_gap_recent),
                'workload_level': float(workload_level),
                'burnout_risk': float(burnout_risk),
                'expertise_recent': float(expertise_recent),
            }
            if gap > idle_gap_threshold:
                # Insert idle transition only (option A) to avoid immediate positive inflation
                trans.append({'t': prev.isoformat(),'gap_days': gap,'action':0,'state':state})
            # Add engage if not exceeding positive cap
            if max_positives_per_user is None or positive_count < max_positives_per_user:
                trans.append({'t': t.isoformat(),'gap_days': gap,'action':1,'state':state})
                positive_count += 1
        # Tail synthetic negative if last activity is stale
        last_time = ts_list[-1]
        now_ref = ts_list[-1]  # context: dataset upper bound approximated by last event
        tail_gap = (now_ref - last_time).days  # always 0, so use dataset max time for better horizon
        dataset_max = ts_list[-1]
        horizon_days = (dataset_max - last_time).days
        if (datetime.now(timezone.utc) - last_time).days >= tail_inactive_days:
            # synthetic state based on tail inactivity
            state_tail = {
                'prev_review': last_time.isoformat(),
                'gap_days': (datetime.now(timezone.utc) - last_time).days,
                'idle_gap_threshold': idle_gap_threshold,
                'activity_7d': 0.0,
                'activity_30d': 0.0,
                'activity_90d': 0.0,
                'activity_ratio_7_30': 0.0,
                'activity_ratio_30_90': 0.0,
                'avg_gap_recent5': float(gaps_so_far[-1] if gaps_so_far else 0.0),
                'workload_level': 0.0,
                'burnout_risk': 0.0,
                'expertise_recent': float(len({d.date() for d in ts_list if last_time - d <= timedelta(days=90)})/90.0),
                'synthetic_tail_negative': 1.0,
            }
            trans.append({'t': datetime.now(timezone.utc).isoformat(), 'gap_days': state_tail['gap_days'], 'action':0,'state':state_tail})
        if trans:
            engage_rate = sum(1 for tr in trans if tr['action']==1)/len(trans)
            avg_gap = sum(tr['gap_days'] for tr in trans)/len(trans)
            sequences.append({'reviewer_id': dev,'transitions': trans,'summary':{'engage_rate':engage_rate,'avg_gap':avg_gap,'count':len(trans)}})
    return sequences


def parse_args():
    ap = argparse.ArgumentParser(description='Extract reviewer IRL sequences with stronger negatives')
    ap.add_argument('--idle-gap', type=int, default=40)
    ap.add_argument('--tail-inactive-days', type=int, default=90)
    ap.add_argument('--no-idle-only', action='store_true', help='Also add immediate engage after idle (legacy behavior)')
    ap.add_argument('--max-positives-per-user', type=int, default=None)
    ap.add_argument('--output', default='outputs/irl/reviewer_sequences.json')
    return ap.parse_args()

def main():
    args = parse_args()
    changes_path = Path('data/processed/unified/all_reviews.json')
    if not changes_path.exists():
        print('missing changes'); return 1
    changes = load_changes(changes_path)
    seqs = build_sequences(
        changes,
        idle_gap_threshold=args.idle_gap,
        insert_idle_only=not args.no_idle_only,
        tail_inactive_days=args.tail_inactive_days,
        max_positives_per_user=args.max_positives_per_user,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(seqs, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Reviewer sequences saved: {len(seqs)} reviewers -> {out_path}')
    # Class balance quick stats
    total_trans = sum(len(s.get('transitions',[])) for s in seqs)
    positives = sum(sum(1 for tr in s.get('transitions',[]) if tr['action']==1) for s in seqs)
    neg = total_trans - positives
    if total_trans:
        print(f'Class balance: positives={positives} negatives={neg} pos_rate={positives/total_trans:.3f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
