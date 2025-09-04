#!/usr/bin/env python3
"""
招待タスク (reviewer/CC) だが horizon 内に authored しなかった負例スナップショット生成。

出力: outputs/invited_negatives/negatives.json

この負例を future_window / next_authored モデル学習時に追加しクラスバランス改善。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--horizon-days', type=int, default=60)
    ap.add_argument('--output', default='outputs/invited_negatives/negatives.json')
    return ap.parse_args()

def parse_dt(ts: str):
    if not ts: return None
    ts = ts.replace('Z','+00:00').replace(' ', 'T')
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except: return None

def main():
    args = parse_args()
    ch_path = Path(args.changes)
    if not ch_path.exists():
        print('missing changes'); return 1
    changes = json.loads(ch_path.read_text())
    # collect authored times per dev
    authored: Dict[str, List[datetime]] = {}
    invites: List[Dict[str, Any]] = []
    for ch in changes:
        owner = ch.get('owner') or {}
        dev_id = owner.get('email') or owner.get('username')
        created = parse_dt(ch.get('created'))
        if dev_id and created:
            authored.setdefault(dev_id, []).append(created)
        reviewers = ch.get('reviewers') or {}
        for role_key in ['REVIEWER','CC']:
            for rv in reviewers.get(role_key, []) or []:
                rid = rv.get('email') or rv.get('username')
                if not rid: continue
                invites.append({'dev': rid, 'change_created': created})
    for dev in authored: authored[dev].sort()
    horizon = args.horizon_days
    negatives = []
    for inv in invites:
        dev = inv['dev']; t = inv['change_created']
        if not t: continue
        # did dev author within horizon after invite time?
        authored_times = authored.get(dev, [])
        engaged = False
        for at in authored_times:
            if at <= t: continue
            if (at - t).days <= horizon:
                engaged = True; break
            if (at - t).days > horizon:
                break
        if not engaged:
            negatives.append({'developer_id': dev,'invite_time': t.isoformat(),'label':0,'reason':'invited_no_author_within_horizon'})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(negatives, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'generated negatives: {len(negatives)} -> {out_path}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
