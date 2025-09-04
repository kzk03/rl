#!/usr/bin/env python3
"""Batch run retention labeling for multiple thresholds (project-aware).

A: evaluate_workload_aware_probabilities.py を 30/45/60 など複数閾値で一括集計し
   クラスバランスと簡易指標をサマリ出力。

特長:
 - プロジェクト指定 (--project) 時は changes からその project の活動のみで last_activity を算出
 - now 基準は対象 (project フィルタ後) の最大タイムスタンプ (データ終端) を採用 (システム時刻揺らぎ排除)
 - developers JSON を補助情報として読み、存在しない開発者も changes ベースで生成可

出力: outputs/retention_probability/threshold_scan.json
  [{threshold, count, positives, pos_rate, accuracy_dummy_majority, days_span}] など

使用例:
  uv run scripts/batch_retention_threshold_scan.py \
      --developers data/processed/unified/all_developers.json \
      --changes data/processed/unified/all_reviews.json \
      --thresholds 30,45,60 --project my/project
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args():
    ap = argparse.ArgumentParser(description='Batch retention threshold scan (project-aware)')
    ap.add_argument('--developers', default='data/processed/unified/all_developers.json')
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--thresholds', default='30,45,60')
    ap.add_argument('--project', default=None, help='Filter changes to this project only')
    ap.add_argument('--min-activity', type=int, default=1)
    ap.add_argument('--output', default='outputs/retention_probability/threshold_scan.json')
    return ap.parse_args()


def parse_ts(ts: str):
    if not ts: return None
    ts = ts.replace('Z','+00:00')
    try:
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except: return None


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'missing file: {path}')
    return json.loads(path.read_text())


def build_project_last_activity(changes: List[Dict[str,Any]], project: str | None):
    last: Dict[str, datetime] = {}
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
        prev = last.get(dev)
        if prev is None or dt > prev:
            last[dev] = dt
        if max_dt is None or dt > max_dt:
            max_dt = dt
    return last, max_dt


def main():
    args = parse_args()
    thresholds = [int(x) for x in args.thresholds.split(',') if x.strip()]
    changes = load_json(Path(args.changes))
    dev_last, dataset_end = build_project_last_activity(changes, args.project)
    if not dataset_end:
        print('❌ no activity found (check project filter)'); return 1
    records = []
    for th in thresholds:
        total = 0
        pos = 0
        for dev, last_dt in dev_last.items():
            gap = (dataset_end - last_dt).days
            label = 1 if gap <= th else 0
            total += 1
            pos += label
        if total == 0: continue
        pos_rate = pos / total
        # ダミーモデル(多数派クラス予測)精度
        majority_acc = max(pos_rate, 1-pos_rate)
        records.append({
            'threshold': th,
            'count': total,
            'positives': pos,
            'pos_rate': pos_rate,
            'accuracy_dummy_majority': majority_acc,
            'dataset_end': dataset_end.isoformat(),
        })
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('✅ threshold scan summary:')
    for r in records:
        print(f"  th={r['threshold']} count={r['count']} pos_rate={r['pos_rate']:.3f}")
    print(f'💾 Saved -> {out_path}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
