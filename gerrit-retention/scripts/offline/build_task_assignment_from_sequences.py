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
import math
import math as _math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import yaml


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
    """Base (legacy) feature extraction from raw state.

    NOTE: Extended / registry-driven features are added later in pipeline.
    """
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


def _load_feature_registry(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        feats = data.get('features') or []
        out: Dict[str, Dict[str, Any]] = {}
        for f in feats:
            name = f.get('name')
            if name:
                out[name] = f
        return out
    except Exception:
        return {}


def _compute_registry_features(
    base: Dict[str, float],
    registry: Dict[str, Dict[str, Any]],
    reviewer_id: str,
    when: datetime,
    reviewer_events_index: Dict[str, List[datetime]],
    first_seen_index: Dict[str, datetime],
    reviewer_transition_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Compute incremental Phase1 experimental features.

    Currently implemented:
      - activity7
      - workload_sq
      - reviewer_tenure_days
      - interaction_activity_gap (renamed from proposal interaction_term_activity_gap)
    """
    out: Dict[str, float] = {}
    # Pre-index lookups
    events = reviewer_events_index.get(reviewer_id) or []
    # activity7
    if 'activity7' in registry:
        lo = when - timedelta(days=7)
        cnt7 = 0
        # events sorted => backward scan
        for ts in reversed(events):
            if ts > when:
                continue
            if ts < lo:
                break
            cnt7 += 1
        out['activity7'] = float(cnt7)
    # workload_sq
    if 'workload_sq' in registry:
        w = base.get('workload', 0.0)
        out['workload_sq'] = float(w * w)
    # reviewer_tenure_days
    if 'reviewer_tenure_days' in registry:
        first = first_seen_index.get(reviewer_id)
        if first is None:
            tenure = 0.0
        else:
            tenure = max(0.0, (when - first).days)
        out['reviewer_tenure_days'] = float(min(2000.0, tenure))
    # interaction_activity_gap (activity30 / (1+gap_days))
    if 'interaction_activity_gap' in registry:
        a30 = base.get('activity30', 0.0)
        gap = base.get('gap_days', 0.0)
        out['interaction_activity_gap'] = float(a30 / (1.0 + gap))
    meta = reviewer_transition_meta.get(reviewer_id) or {}
    # backlog_open_reviews (approx). Use meta['open_reviews'] if timestamp ordering maintained, else heuristic
    if 'backlog_open_reviews' in registry:
        # naive: number of previous events - accepted count (floor at 0)
        prev_total = int(meta.get('total_before', 0))
        prev_accept = int(meta.get('accept_before', 0))
        backlog = max(0, prev_total - prev_accept)
        out['backlog_open_reviews'] = float(backlog)
    # historical_accept_ratio
    if 'historical_accept_ratio' in registry:
        prev_total = int(meta.get('total_before', 0))
        prev_accept = int(meta.get('accept_before', 0))
        if prev_total < 3:
            ratio = 0.0
        else:
            ratio = float(prev_accept) / float(max(1, prev_total))
        out['historical_accept_ratio'] = float(min(1.0, ratio))
    # active_hour_match
    if 'active_hour_match' in registry:
        # meta['hour_counts'] is Counter-like mapping hour->count for past events before 'when'
        hour_counts: Dict[int, int] = meta.get('hour_counts') or {}
        if hour_counts:
            # select top N hours (N=3) by frequency
            top_hours = sorted(hour_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            hour_set = {h for h, _ in top_hours}
            out['active_hour_match'] = 1.0 if when.hour in hour_set else 0.0
        else:
            out['active_hour_match'] = 0.0
    return out


def build_tasks_from_sequences(
    input_path: Path,
    cutoff_iso: str,
    out_dir: Path,
    max_candidates: int = 8,
    seed: int | None = 42,
    candidate_sampling: str = 'random',  # 'random' | 'time-local'
    candidate_window_days: int = 30,
    shuffle_candidates: bool = True,
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

    # Stable per-task shuffler based on a deterministic hash so that
    # shuffling is identical across runs regardless of input ordering.
    import hashlib
    def _stable_perm_indices(n: int, context_parts: List[Any]) -> List[int]:
        """Return a deterministic permutation of range(n) based on context_parts and global seed."""
        idxs = list(range(n))
        if n <= 1:
            return idxs
        joined = '\u0001'.join(str(k) for k in context_parts)
        base = f"{joined}|seed={seed if seed is not None else 'none'}".encode('utf-8')
        digest = hashlib.sha256(base).digest()
        local_seed = int.from_bytes(digest[:8], byteorder='big', signed=False)
        local_rng = random.Random(local_seed)
        local_rng.shuffle(idxs)
        return idxs

    # Helper: extract second-level directory tokens from path list
    def _dir_tokens(paths: List[str]) -> Tuple[Set[str], Set[str]]:
        dir1: Set[str] = set()
        dir2: Set[str] = set()
        for p in paths:
            if not isinstance(p, str):
                continue
            parts = p.strip('/').split('/')
            if parts and parts[0]:
                dir1.add(parts[0])
            if len(parts) >= 2:
                dir2.add('/'.join(parts[:2]))
        return dir1, dir2

    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter) / float(union) if union else 0.0

    # Pre-scan to collect owner→reviewer assignment timestamps (for pair counts)
    # and reviewer path history (dir2) with timestamps.
    owner_pair_events: Dict[Tuple[str, str], List[datetime]] = {}
    reviewer_paths: Dict[str, List[Tuple[datetime, Set[str]]]] = {}
    # For TF-IDF: maintain recent 30d file path tokens (dir1+dir2) per reviewer
    reviewer_recent_dirs: Dict[str, List[Tuple[datetime, Set[str]]]] = {}
    global_df: Counter = Counter()  # document frequency for dir tokens
    total_docs = 0
    for rec in seqs:
        rid_main = rec.get('reviewer_id') or rec.get('developer_id') or 'unknown@example.com'
        owner_id = rec.get('owner_id') or rec.get('owner') or rec.get('author')  # heuristic
        trans = rec.get('transitions') or []
        file_list = rec.get('files') or rec.get('changed_files') or []
        d1, d2 = _dir_tokens(file_list) if file_list else (set(), set())
        merged_dirs = d1 | d2
        for tr in trans:
            ts = tr.get('t') or tr.get('timestamp')
            if not ts:
                continue
            try:
                tdt = _parse_iso(str(ts))
            except Exception:
                continue
            # Pair owner→reviewer (assignment surrogate): action==1 or any transition counts as exposure
            if owner_id:
                key = (owner_id, rid_main)
                owner_pair_events.setdefault(key, []).append(tdt)
            # Path history (use transitions as proxy; attach file set if any)
            if merged_dirs:
                reviewer_paths.setdefault(rid_main, []).append((tdt, merged_dirs))
                total_docs += 1
                seen_doc_tokens = set(merged_dirs)
                for tok in seen_doc_tokens:
                    global_df[tok] += 1
                reviewer_recent_dirs.setdefault(rid_main, []).append((tdt, merged_dirs))

    # Sort path histories
    for _rid, lst in reviewer_paths.items():
        lst.sort(key=lambda x: x[0])
    for _k, lst in owner_pair_events.items():
        lst.sort()

    def _lookup_pair_count(owner: str, reviewer: str, when: datetime, days: int = 180) -> int:
        key = (owner, reviewer)
        lst = owner_pair_events.get(key)
        if not lst:
            return 0
        lo = when - timedelta(days=days)
        # backward scan
        cnt = 0
        for t in reversed(lst):
            if t > when:
                continue
            if t < lo:
                break
            cnt += 1
        return cnt

    def _lookup_dir2_history(reviewer: str, when: datetime, days: int = 180) -> Set[str]:
        lst = reviewer_paths.get(reviewer)
        if not lst:
            return set()
        lo = when - timedelta(days=days)
        acc: Set[str] = set()
        for t, dirs in reversed(lst):
            if t > when:
                continue
            if t < lo:
                break
            acc.update(dirs)
        return acc

    def _lookup_recent_dirs_tfidf(reviewer: str, when: datetime, days: int = 30) -> Counter:
        lst = reviewer_recent_dirs.get(reviewer)
        c = Counter()
        if not lst:
            return c
        lo = when - timedelta(days=days)
        for t, dirs in reversed(lst):
            if t > when:
                continue
            if t < lo:
                break
            for d in dirs:
                c[d] += 1
        return c

    def _tfidf_cosine(cur_dirs: Set[str], hist_counter: Counter) -> float:
        if not cur_dirs or not hist_counter:
            return 0.0
        # build TF-IDF vectors (sparse) for current (binary TF) and history (counts)
        # idf = log((N+1)/(df+1)) + 1
        cur_vec = {}
        hist_vec = {}
        max_tf = max(hist_counter.values()) if hist_counter else 1.0
        for tok in cur_dirs:
            df = global_df.get(tok, 0)
            idf = _math.log((total_docs + 1) / (df + 1)) + 1.0
            cur_vec[tok] = 1.0 * idf
        for tok, tf in hist_counter.items():
            df = global_df.get(tok, 0)
            idf = _math.log((total_docs + 1) / (df + 1)) + 1.0
            norm_tf = float(tf) / float(max_tf)
            hist_vec[tok] = norm_tf * idf
        # cosine
        dot = 0.0
        for tok, v in cur_vec.items():
            if tok in hist_vec:
                dot += v * hist_vec[tok]
        norm_a = _math.sqrt(sum(v*v for v in cur_vec.values()))
        norm_b = _math.sqrt(sum(v*v for v in hist_vec.values()))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _gen(phase: str, pred) -> Tuple[str, int]:
        outp = out_dir / f"tasks_{phase}.jsonl"
        n = 0
        with open(outp, 'w', encoding='utf-8') as wf:
            for rec in seqs:
                rid = rec.get('reviewer_id') or rec.get('developer_id') or 'unknown@example.com'
                trans = rec.get('transitions') or []
                # Build indexes for registry features (events & first_seen)
                # (We compute lazily only once when first transition encountered per reviewer)
                # Precompute per reviewer events list (timestamps) for activity7 etc.
                # This is outside inner transition loop but simple; acceptable as lists are small in demo.
                # Build global indexes just once
                # (moved above outside loops if scaling becomes issue)
                # Per reviewer transition meta accumulators (for historical features)
                total_before = 0
                accept_before = 0
                hour_counter: Counter = Counter()
                for tr in trans:
                    ts = tr.get('t') or tr.get('timestamp')
                    if not ts:
                        continue
                    when = _parse_iso(str(ts))
                    if not pred(when):
                        continue
                    a = int(tr.get('action', 2))
                    state = tr.get('state', {}) or {}
                    # Build candidates (ensure GT is included but remove positional bias by shuffling later)
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

                    # Deduplicate (safety)
                    seen = set()
                    cands = [x for x in cands if not (x in seen or seen.add(x))]

                    # Shuffle candidate order deterministically per task (reduce positional bias)
                    if shuffle_candidates:
                        context = [rid, when.isoformat()] + [str(x) for x in cands]
                        perm = _stable_perm_indices(len(cands), context)
                        cands = [cands[i] for i in perm]

                    # Pre-build reviewer events & first seen indices (cached)
                    if 'registry_index_built' not in locals():
                        reviewer_events_index: Dict[str, List[datetime]] = {}
                        first_seen_index: Dict[str, datetime] = {}
                        for _rec in seqs:
                            _rid = _rec.get('reviewer_id') or _rec.get('developer_id')
                            if not _rid:
                                continue
                            evs: List[datetime] = []
                            for _tr in (_rec.get('transitions') or []):
                                _ts = _tr.get('t') or _tr.get('timestamp')
                                if not _ts:
                                    continue
                                try:
                                    _when = _parse_iso(str(_ts))
                                except Exception:
                                    continue
                                evs.append(_when)
                            evs.sort()
                            if evs:
                                reviewer_events_index[_rid] = evs
                                first_seen_index[_rid] = evs[0]
                        registry_index_built = True
                    # Build feature map per candidate using candidate's latest state before 'when' (fallback to source state)
                    cand_objs = []
                    for r in cands:
                        cand_state = _latest_state_before(r, when) or state
                        feats = _feat_from_state(cand_state)
                        # Registry-driven extensions
                        if feature_registry:
                            # Build meta dict for candidate reviewer (only reliable for focal rid; others fallback minimal)
                            if r == rid:
                                meta = {
                                    'total_before': total_before,
                                    'accept_before': accept_before,
                                    'hour_counts': dict(hour_counter),
                                }
                            else:
                                meta = {'total_before': 0, 'accept_before': 0, 'hour_counts': {}}
                            ext = _compute_registry_features(
                                feats,
                                feature_registry,
                                r,
                                when,
                                reviewer_events_index,
                                first_seen_index,
                                reviewer_transition_meta={},  # deprecated path; meta passed inline
                            )
                            # Inject inline computed meta-based features (override ext where present)
                            # backlog/historical/active_hour use 'meta'
                            if 'backlog_open_reviews' in feature_registry:
                                prev_total = meta.get('total_before', 0)
                                prev_accept = meta.get('accept_before', 0)
                                feats['backlog_open_reviews'] = float(max(0, prev_total - prev_accept))
                            if 'historical_accept_ratio' in feature_registry:
                                prev_total = meta.get('total_before', 0)
                                prev_accept = meta.get('accept_before', 0)
                                if prev_total >= 3:
                                    feats['historical_accept_ratio'] = float(min(1.0, prev_accept / max(1, prev_total)))
                                else:
                                    feats['historical_accept_ratio'] = 0.0
                            if 'active_hour_match' in feature_registry:
                                hour_counts = meta.get('hour_counts') or {}
                                if hour_counts:
                                    top_hours = sorted(hour_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                                    feats['active_hour_match'] = 1.0 if when.hour in {h for h, _ in top_hours} else 0.0
                                else:
                                    feats['active_hour_match'] = 0.0
                            # Pair feature
                            if 'owner_reviewer_past_assignments_180d' in feature_registry:
                                owner_id = rec.get('owner_id') or rec.get('owner') or rec.get('author')
                                if owner_id:
                                    feats['owner_reviewer_past_assignments_180d'] = float(
                                        _lookup_pair_count(owner_id, r, when, days=180)
                                    )
                                else:
                                    feats['owner_reviewer_past_assignments_180d'] = 0.0
                            # Path Jaccard dir2 similarity
                            if 'path_jaccard_dir2_180d' in feature_registry:
                                file_list = rec.get('files') or rec.get('changed_files') or []
                                d1, d2 = _dir_tokens(file_list) if file_list else (set(), set())
                                cur_dirs = d1 | d2
                                if cur_dirs:
                                    hist_dirs = _lookup_dir2_history(r, when, days=180)
                                    feats['path_jaccard_dir2_180d'] = _jaccard(cur_dirs, hist_dirs)
                                else:
                                    feats['path_jaccard_dir2_180d'] = 0.0
                            if 'path_tfidf_cosine_recent30' in feature_registry:
                                file_list = rec.get('files') or rec.get('changed_files') or []
                                d1, d2 = _dir_tokens(file_list) if file_list else (set(), set())
                                cur_dirs = d1 | d2
                                if cur_dirs:
                                    hist_counter = _lookup_recent_dirs_tfidf(r, when, days=30)
                                    feats['path_tfidf_cosine_recent30'] = _tfidf_cosine(cur_dirs, hist_counter)
                                else:
                                    feats['path_tfidf_cosine_recent30'] = 0.0
                            # Pair normalization features
                            if 'owner_reviewer_past_assignments_180d_log1p' in feature_registry:
                                val = feats.get('owner_reviewer_past_assignments_180d', 0.0)
                                feats['owner_reviewer_past_assignments_180d_log1p'] = float(_math.log1p(val))
                            if 'owner_reviewer_past_assignments_ratio_180d' in feature_registry:
                                # denominator: reviewer total assignments (irrespective of owner) in 180d
                                reviewer_total = 0
                                for (own, rv), times in owner_pair_events.items():
                                    if rv == r:
                                        # count times within window
                                        cnt = 0; lo_t = when - timedelta(days=180)
                                        for tt in reversed(times):
                                            if tt > when: continue
                                            if tt < lo_t: break
                                            cnt += 1
                                        reviewer_total += cnt
                                pair_val = feats.get('owner_reviewer_past_assignments_180d', 0.0)
                                ratio = pair_val / reviewer_total if reviewer_total > 0 else 0.0
                                feats['owner_reviewer_past_assignments_ratio_180d'] = float(min(1.0, ratio))
                            feats.update(ext)
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
                    # Update meta after writing (current transition becomes part of history for next)
                    total_before += 1
                    if a == 1:
                        accept_before += 1
                    hour_counter[when.hour] += 1
        return str(outp), n

    # Load feature registry (optional)
    registry_path = out_dir.parent / 'configs' / 'irl_feature_registry.yaml'
    # fallback: look in repo-level configs if relative path differs
    if not registry_path.exists():
        alt = Path('configs/irl_feature_registry.yaml')
        registry_path = alt if alt.exists() else registry_path
    feature_registry = _load_feature_registry(registry_path)

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
        'shuffle_candidates': bool(shuffle_candidates),
        'registry_features_loaded': list(feature_registry.keys()),
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
    ap.add_argument('--no-shuffle-candidates', action='store_true', help='Disable deterministic candidate shuffling (reduces positional bias by default).')
    args = ap.parse_args()
    meta = build_tasks_from_sequences(
        Path(args.input),
        args.cutoff,
        Path(args.outdir),
        max_candidates=int(args.max_candidates),
        seed=int(args.seed),
        candidate_sampling=str(args.candidate_sampling),
        candidate_window_days=int(args.candidate_window_days),
        shuffle_candidates=not bool(args.no_shuffle_candidates),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
