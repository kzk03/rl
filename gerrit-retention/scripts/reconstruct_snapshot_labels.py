#!/usr/bin/env python3
"""過去スナップショット再構築ユーティリティ (B)

目的:
  履歴 changes データから仮想 now 日時 (virtual_now) を指定して、その時点での
  各開発者 (特定プロジェクト限定可) の last_activity gap に基づく retained ラベルを再現。

特徴:
  - --virtual-now 未指定時はデータセット内最大 created を採用
  - プロジェクト指定 (--project) で他プロジェクト活動を無視
  - 出力: snapshots JSON (developer_id, last_activity, gap_days, label)
  - 複数 virtual_now をカンマで渡し一括 (e.g. 2024-12-31,2025-03-31)

使用例:
  uv run scripts/reconstruct_snapshot_labels.py \
    --changes data/processed/unified/all_reviews.json \
    --virtual-now 2025-01-31 --threshold 60 --project my/project
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args():
    ap = argparse.ArgumentParser(description='Reconstruct past snapshot labels (virtual now)')
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--threshold', type=int, default=60)
    ap.add_argument('--virtual-now', default=None, help='ISO8601 or comma separated multiple')
    ap.add_argument('--project', default=None)
    ap.add_argument('--output', default='outputs/retention_probability/virtual_snapshots.json')
    return ap.parse_args()


def parse_ts(ts: str):
    if not ts: return None
    ts = ts.replace('Z','+00:00').replace(' ','T')
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except: return None


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def build_last_activity(changes: List[Dict[str,Any]], project: str|None):
    last = {}
    max_dt = None
    for ch in changes:
        if project and ch.get('project') != project:
            continue
        owner = ch.get('owner') or {}
        dev = owner.get('email') or owner.get('username')
        if not dev: continue
        created = ch.get('created') or ch.get('updated') or ch.get('submitted')
        dt = parse_ts(created)
        if not dt: continue
        if dev not in last or dt > last[dev]:
            last[dev] = dt
        if max_dt is None or dt > max_dt:
            max_dt = dt
    return last, max_dt


def reconstruct(dev_last: Dict[str, datetime], virtual_now: datetime, threshold: int):
    out = []
    for dev, last_dt in dev_last.items():
        gap = (virtual_now - last_dt).days
        label = 1 if gap <= threshold else 0
        out.append({
            'developer_id': dev,
            'last_activity': last_dt.isoformat(),
            'virtual_now': virtual_now.isoformat(),
            'gap_days': gap,
            'threshold': threshold,
            'retained': label
        })
    return out


def main():
    args = parse_args()
    changes = load_json(Path(args.changes))
    dev_last, dataset_end = build_last_activity(changes, args.project)
    if not dev_last:
        print('❌ no developers found (project filter?)'); return 1
    if args.virtual_now:
        targets = []
        for part in args.virtual_now.split(','):
            dt = parse_ts(part.strip())
            if dt: targets.append(dt)
        if not targets:
            print('❌ failed to parse any virtual_now timestamp'); return 1
    else:
        targets = [dataset_end]
    result_all = []
    for vn in targets:
        rows = reconstruct(dev_last, vn, args.threshold)
        result_all.extend(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_all, indent=2), encoding='utf-8')
    print(f'✅ wrote {len(result_all)} snapshot rows -> {out_path}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
