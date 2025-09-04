#!/usr/bin/env python3
"""IRL 用シーケンス抽出テンプレート

出力: outputs/irl/sequences.json
構造: [{developer_id, transitions:[{t, gap_days, action, state:{...feature stub...}}], summary:{engage_rate, avg_gap}}]

action: 1=engage (authored change), 0=idle (ギャップが長い区切り点用の補完)
注: Idle 補完は簡易 (gap > idle_gap_threshold の場合に 0 行動1つ追加)
"""
from __future__ import annotations

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

def build_sequences(changes: List[Dict[str, Any]], idle_gap_threshold: int = 45):
    authored: Dict[str, List[datetime]] = {}
    for ch in changes:
        owner = ch.get('owner') or {}
        dev = owner.get('email') or owner.get('username')
        if not dev: continue
        created = parse_dt(ch.get('created'))
        if not created: continue
        authored.setdefault(dev, []).append(created)
    for v in authored.values(): v.sort()
    sequences = []
    for dev, times in authored.items():
        if len(times) < 2: continue
        # 事前に epoch 秒へ変換 (高速ウィンドウ集計用)
        ts_list = times  # already sorted
        ts_seconds = [int(t.timestamp()) for t in ts_list]
        trans = []
        gaps_so_far: List[int] = []
        for i in range(1, len(ts_list)):
            prev = ts_list[i-1]
            t = ts_list[i]
            gap = (t - prev).days
            gaps_so_far.append(gap)
            ref_time = prev  # state は prev 時点 (次 engage 前)
            # ---- window counts ----
            def count_within(days: int) -> int:
                start_ts = int((ref_time - timedelta(days=days)).timestamp())
                # ts_seconds[0 .. i-1) が ref_time 以前
                left = bisect.bisect_left(ts_seconds, start_ts, 0, i)  # i 番目は prev の index
                return i - left  # events in window (including prev)
            act_7 = count_within(7)
            act_30 = count_within(30)
            act_90 = count_within(90)
            ratio_7_30 = act_7 / act_30 if act_30 else 0.0
            ratio_30_90 = act_30 / act_90 if act_90 else 0.0
            avg_gap_recent = (sum(gaps_so_far[-5:]) / min(len(gaps_so_far),5)) if gaps_so_far else gap
            # workload proxy = 30日活動密度
            workload_level = act_30 / 30.0
            # burnout proxy: 高頻度 * 小さい平均ギャップ
            burnout_risk = 1.0 if (act_30 >= 40 and avg_gap_recent < 2) or (act_7 >= 15 and avg_gap_recent < 1.5) else 0.0
            # expertise proxy: 過去90日一意活動日数 / 90
            unique_days_90 = len({d.date() for d in ts_list[:i] if ref_time - d <= timedelta(days=90)})
            expertise_recent = unique_days_90 / 90.0
            state = {
                'prev_activity': prev.isoformat(),
                'gap_days': gap,
                'idle_gap_threshold': idle_gap_threshold,
                # dynamic numeric features (IRL モデルが自動検出)
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
                # idle action (0) before engage (1)
                trans.append({'t': prev.isoformat(), 'gap_days': gap, 'action': 0, 'state': state})
            trans.append({'t': t.isoformat(), 'gap_days': gap, 'action': 1, 'state': state})
        if trans:
            engage_rate = sum(1 for tr in trans if tr['action']==1)/len(trans)
            avg_gap = sum(tr['gap_days'] for tr in trans)/len(trans)
            sequences.append({'developer_id': dev,'transitions': trans,'summary':{'engage_rate': engage_rate,'avg_gap': avg_gap,'count': len(trans)}})
    return sequences

def main():
    changes_path = Path('data/processed/unified/all_reviews.json')
    if not changes_path.exists():
        print('missing changes'); return 1
    changes = load_changes(changes_path)
    seqs = build_sequences(changes)
    out_dir = Path('outputs/irl')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'sequences.json').write_text(json.dumps(seqs, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'IRL sequences saved: {len(seqs)} developers -> {out_dir/"sequences.json"}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
