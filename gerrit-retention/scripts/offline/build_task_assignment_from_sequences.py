#!/usr/bin/env python3
"""
reviewer_sequences.json(.gz|.jsonl) -> task-centric AssignmentTask(JSONL) 生成

各トランジションを「タスク（change_id相当）」と見なし、その時点での候補レビュア集合を作り、
MultiReviewerAssignmentEnv 用の AssignmentTask をJSONLで出力します。

簡易仕様（最小実装）:
- 各レコードの reviewer_id を候補に含める（少なくとも1名）
- 近接の他レビュア（同ファイル内の他レコード）からランダムに K-1 名を候補として追加（再現性のため seed 指定）
- positive_reviewer_ids は、元トランジションの action==1(受諾) のとき reviewer_id を含める。action==0/2 は空とする
- features は state からのヒューリスティック（activity/workload/gap など）

出力: outdir/tasks_train.jsonl, tasks_eval.jsonl, meta.json
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace('Z','+00:00')).replace(tzinfo=None)
    except Exception:
        return datetime.fromisoformat(ts)


def _open_maybe_gzip(p: Path):
    if str(p).endswith('.gz'):
        return gzip.open(p, 'rt', encoding='utf-8')
    return open(p, 'rt', encoding='utf-8')


def _iter_json_array(f) -> Iterator[Dict[str, Any]]:
    dec = json.JSONDecoder()
    buf = ''
    # skip to '['
    while True:
        chunk = f.read(65536)
        if not chunk:
            break
        buf += chunk
        i = 0
        n = len(buf)
        while i < n and buf[i].isspace():
            i += 1
        if i < n and buf[i] == '[':
            buf = buf[i+1:]
            break
    # elements
    while True:
        j = 0
        while j < len(buf) and buf[j].isspace():
            j += 1
        if j < len(buf) and buf[j] == ',':
            buf = buf[j+1:]
            j = 0
        while j < len(buf) and buf[j].isspace():
            j += 1
        if j < len(buf) and buf[j] == ']':
            return
        try:
            obj, idx = dec.raw_decode(buf[j:])
            yield obj
            buf = buf[j+idx:]
        except json.JSONDecodeError:
            more = f.read(65536)
            if not more:
                return
            buf += more


def _iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    with _open_maybe_gzip(path) as f:
        name = os.path.basename(str(path)).lower()
        if name.endswith('.jsonl') or name.endswith('.jsonl.gz'):
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception:
                    continue
        else:
            for obj in _iter_json_array(f):
                if isinstance(obj, dict):
                    yield obj


def _feat_from_state(state: Dict[str, Any]) -> Dict[str, float]:
    gap = float(state.get('gap_days', state.get('gap', 3) or 3))
    activity30 = float(state.get('activity_30d', state.get('activity30', 1.0) or 1.0))
    activity90 = float(state.get('activity_90d', state.get('activity90', max(1.0, activity30)) or max(1.0, activity30)))
    workload = float(state.get('workload_level', state.get('workload', 0.2) or 0.2))
    expertise = float(state.get('expertise_recent', state.get('expertise', 0.5) or 0.5))
    return {
        'activity30': float(activity30),
        'activity90': float(activity90),
        'workload': float(workload),
        'gap_days': float(gap),
        'expertise': float(expertise),
    }


def build_tasks_from_sequences(
    input_path: Path,
    cutoff_iso: str,
    out_dir: Path,
    max_candidates: int = 8,
    seed: int | None = 42,
    candidate_sampling: str = 'random',  # 'random' | 'time-local'
    candidate_window_days: int = 30,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = _parse_iso(cutoff_iso)

    # load minimal in-memory index for candidate sampling
    seqs: List[Dict[str, Any]] = list(_iter_records(input_path))
    reviewer_ids = [rec.get('reviewer_id') or rec.get('developer_id') for rec in seqs if (rec.get('reviewer_id') or rec.get('developer_id'))]

    # Build per-reviewer time-indexed states for time-local features
    by_reviewer: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
    for rec in seqs:
        rid = rec.get('reviewer_id') or rec.get('developer_id')
        if not rid:
            continue
        trans = rec.get('transitions') or []
        lst: List[Tuple[datetime, Dict[str, Any]]] = by_reviewer.setdefault(rid, [])
        for tr in trans:
            ts = tr.get('t') or tr.get('timestamp')
            if not ts:
                continue
            try:
                when = _parse_iso(str(ts))
            except Exception:
                continue
            state = tr.get('state', {}) or {}
            lst.append((when, state))
        lst.sort(key=lambda x: x[0])

    def _latest_state_before(rid: str, when: datetime) -> Dict[str, Any] | None:
        arr = by_reviewer.get(rid)
        if not arr:
            return None
        # binary search-like backward scan (lists are small in demo)
        latest: Dict[str, Any] | None = None
        for t, st in reversed(arr):
            if t <= when:
                latest = st
                break
        return latest
    rng = random.Random(seed)

    def _gen(phase: str, pred) -> Tuple[str, int]:
        outp = out_dir / f"tasks_{phase}.jsonl"
        n = 0
        with open(outp, 'w', encoding='utf-8') as wf:
            for rec in seqs:
                rid = rec.get('reviewer_id') or rec.get('developer_id') or 'unknown@example.com'
                trans = rec.get('transitions') or []
                for tr in trans:
                    ts = tr.get('t') or tr.get('timestamp')
                    if not ts:
                        continue
                    when = _parse_iso(str(ts))
                    if not pred(when):
                        continue
                    a = int(tr.get('action', 2))
                    state = tr.get('state', {}) or {}
                    # Build candidates
                    cands = [rid]
                    pool = [x for x in reviewer_ids if x and x != rid]
                    if candidate_sampling == 'time-local':
                        # keep only reviewers with at least one transition within ±window around 'when'
                        window = candidate_window_days
                        lo = when - timedelta(days=window)
                        hi = when + timedelta(days=window)
                        filtered = []
                        for r in pool:
                            arr = by_reviewer.get(r) or []
                            # simple scan (small lists)
                            ok = any(lo <= t <= hi for (t, _st) in arr)
                            if ok:
                                filtered.append(r)
                        pool = filtered or pool  # fallback to global if empty
                    rng.shuffle(pool)
                    for r in pool:
                        if len(cands) >= max_candidates:
                            break
                        cands.append(r)

                    # Build feature map per candidate using candidate's latest state before 'when' (fallback to source state)
                    cand_objs = []
                    for r in cands:
                        cand_state = _latest_state_before(r, when) or state
                        feats = _feat_from_state(cand_state)
                        cand_objs.append({'reviewer_id': r, 'features': feats})
                    positives = [rid] if a == 1 else []
                    task = {
                        'change_id': f"chg_{rid}_{when.strftime('%Y%m%d%H%M%S')}",
                        'timestamp': when.isoformat(),
                        'candidates': cand_objs,
                        'positive_reviewer_ids': positives,
                    }
                    wf.write(json.dumps(task, ensure_ascii=False) + "\n")
                    n += 1
        return str(outp), n

    train_path, n_train = _gen('train', lambda t: t <= cutoff)
    eval_path, n_eval = _gen('eval', lambda t: t > cutoff)
    meta = {
        'source': str(input_path),
        'cutoff': cutoff.isoformat(),
        'train_tasks': train_path,
        'train_count': n_train,
        'eval_tasks': eval_path,
        'eval_count': n_eval,
        'max_candidates': int(max_candidates),
        'seed': int(seed) if seed is not None else None,
        'candidate_sampling': str(candidate_sampling),
        'candidate_window_days': int(candidate_window_days),
    }
    (out_dir / 'tasks_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='outputs/irl/reviewer_sequences.json')
    ap.add_argument('--cutoff', type=str, default='2024-07-01T00:00:00Z')
    ap.add_argument('--outdir', type=str, default='outputs/task_assign')
    ap.add_argument('--max-candidates', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--candidate-sampling', type=str, default='random', choices=['random', 'time-local'])
    ap.add_argument('--candidate-window-days', type=int, default=30)
    args = ap.parse_args()
    meta = build_tasks_from_sequences(
        Path(args.input),
        args.cutoff,
        Path(args.outdir),
        max_candidates=int(args.max_candidates),
        seed=int(args.seed),
        candidate_sampling=str(args.candidate_sampling),
        candidate_window_days=int(args.candidate_window_days),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
