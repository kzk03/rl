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
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

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


_SERVICE_ACCOUNT_TOKENS: Tuple[str, ...] = (
    'zuul',
    'jenkins',
    'buildbot',
    'automation',
    'ci@',
    'ci.',
    'ci-',
    'bot@',
    'bot.',
    'bot-',
    'openstack-ci',
    'openstack-infra',
    'gerrit system',
)


def _is_service_account(rid: Optional[str]) -> bool:
    if not rid:
        return False
    rid_lower = str(rid).lower()
    if rid_lower in {'zuul', 'jenkins', 'buildbot'}:
        return True
    for token in _SERVICE_ACCOUNT_TOKENS:
        if token in rid_lower:
            return True
    return False


def _dir_tokens(paths: List[str]) -> Tuple[Set[str], Set[str]]:
    dir1: Set[str] = set()
    dir2: Set[str] = set()
    for p in paths:
        if not isinstance(p, str):
            continue
        norm = p.strip('/').strip()
        if not norm:
            continue
        parts = [seg for seg in norm.split('/') if seg]
        if not parts:
            continue
        dir1.add(parts[0])
        if len(parts) >= 2:
            dir2.add('/'.join(parts[:2]))
    return dir1, dir2


def _collect_change_files(change: Dict[str, Any]) -> List[str]:
    files: Set[str] = set()
    revisions = change.get('revisions') or {}
    if isinstance(revisions, dict):
        for rev in revisions.values():
            if not isinstance(rev, dict):
                continue
            rev_files = rev.get('files') or {}
            if not isinstance(rev_files, dict):
                continue
            for path in rev_files.keys():
                if not isinstance(path, str):
                    continue
                if not path or path.startswith('/'):
                    continue
                files.add(path)
    # Fallback: some dumps store file paths in top-level 'files'
    top_files = change.get('files')
    if isinstance(top_files, list):
        for path in top_files:
            if isinstance(path, str) and path and not path.startswith('/'):
                files.add(path)
    return sorted(files)


def _analyze_change_paths(paths: List[str]) -> Dict[str, float]:
    count = len(paths)
    if count == 0:
        return {
            'doc_ratio': 0.0,
            'test_ratio': 0.0,
            'avg_depth': 0.0,
            'max_depth': 0.0,
            'primary_dir_max_frac': 0.0,
        }
    doc_count = 0
    test_count = 0
    total_depth = 0
    max_depth = 0
    dir1_counter: Counter[str] = Counter()
    for path in paths:
        norm = path.strip('/')
        parts = [seg for seg in norm.split('/') if seg]
        depth = len(parts)
        total_depth += depth
        if depth > max_depth:
            max_depth = depth
        lower_parts = [seg.lower() for seg in parts]
        if any('doc' in seg for seg in lower_parts):
            doc_count += 1
        if any('test' in seg for seg in lower_parts):
            test_count += 1
        if parts:
            dir1_counter[parts[0]] += 1
    primary_dir_max = max(dir1_counter.values()) if dir1_counter else 0
    return {
        'doc_ratio': float(doc_count) / float(count) if count else 0.0,
        'test_ratio': float(test_count) / float(count) if count else 0.0,
        'avg_depth': float(total_depth) / float(count) if count else 0.0,
        'max_depth': float(max_depth),
        'primary_dir_max_frac': float(primary_dir_max) / float(count) if count else 0.0,
    }


def _normalize_dt(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return _parse_iso(str(ts))
    except Exception:
        return None


def _load_changes_index(path: Path | None, project_filter: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

    containers: List[Dict[str, Any]]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and 'created' in raw[0]:
        containers = [{'data': {'__all__': raw}}]
    elif isinstance(raw, list):
        containers = [c for c in raw if isinstance(c, dict)]
    elif isinstance(raw, dict):
        containers = [raw]
    else:
        return None

    event_lookup: Dict[Tuple[str, datetime], List[Dict[str, Any]]] = defaultdict(list)
    project_events: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)

    for container in containers:
        data = container.get('data') if isinstance(container, dict) else None
        if not isinstance(data, dict):
            continue
        for project, changes in data.items():
            if project_filter and project not in project_filter:
                continue
            if not isinstance(changes, list):
                continue
            for ch in changes:
                created_dt = _normalize_dt(ch.get('created'))
                if created_dt is None:
                    continue
                owner = ch.get('owner') or {}
                owner_email = owner.get('email') or owner.get('username')
                reviewers = ch.get('reviewers') or {}
                participants: Set[str] = set()
                if owner_email:
                    participants.add(owner_email)
                for role in ('REVIEWER', 'CC'):
                    for rv in reviewers.get(role) or []:
                        email = rv.get('email') or rv.get('username')
                        if email:
                            participants.add(email)
                if not participants:
                    continue
                file_paths = _collect_change_files(ch)
                dir1_tokens, dir2_tokens = _dir_tokens(file_paths)
                path_stats = _analyze_change_paths(file_paths)
                insertions = int(ch.get('insertions') or 0)
                deletions = int(ch.get('deletions') or 0)
                total_churn = insertions + deletions
                subject = ch.get('subject') or ''
                change_info = {
                    'file_paths': file_paths,
                    'dir1': sorted(dir1_tokens),
                    'dir2': sorted(dir2_tokens),
                    'file_count': int(len(file_paths)),
                    'dir1_count': int(len(dir1_tokens)),
                    'dir2_count': int(len(dir2_tokens)),
                    'doc_ratio': path_stats['doc_ratio'],
                    'test_ratio': path_stats['test_ratio'],
                    'avg_path_depth': path_stats['avg_depth'],
                    'max_path_depth': path_stats['max_depth'],
                    'primary_dir_max_frac': path_stats['primary_dir_max_frac'],
                    'insertions': float(insertions),
                    'deletions': float(deletions),
                    'total_churn': float(total_churn),
                    'total_churn_log1p': float(_math.log1p(total_churn)),
                    'file_count_log1p': float(_math.log1p(max(0, len(file_paths)))),
                    'subject_length': float(len(subject)),
                    'subject_length_log1p': float(_math.log1p(len(subject))) if subject else 0.0,
                    'subject_word_count': float(len(subject.split())) if subject else 0.0,
                    'is_revert': 1.0 if subject.strip().lower().startswith('revert') else 0.0,
                    'topic_length': float(len((ch.get('topic') or '').split())) if ch.get('topic') else 0.0,
                }
                part_list = sorted(participants)
                for dev in part_list:
                    project_events[project].append((created_dt, dev))
                    key = (dev.lower(), created_dt)
                    event_lookup[key].append({
                        'project': project,
                        'owner': owner_email,
                        'participants': part_list,
                        'change_key': ch.get('id') or ch.get('change_id') or ch.get('_number'),
                        'change_info': change_info,
                        'created': created_dt.isoformat(),
                    })

    for project, events in project_events.items():
        events.sort(key=lambda x: x[0])

    return {
        'event_lookup': event_lookup,
        'project_events': project_events,
    }


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
    max_candidates: Optional[int] = None,
    seed: int | None = 42,
    candidate_sampling: str = 'random',  # 'random' | 'time-local'
    candidate_window_days: int = 30,
    shuffle_candidates: bool = True,
    train_window_days: int | None = None,
    eval_window_days: int | None = None,
    train_window_months: float | None = None,
    eval_window_months: float | None = None,
    changes_json: Path | None = None,
    candidate_activity_window_days: int | None = None,
    candidate_activity_window_months: float | None = None,
    project_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = _parse_iso(cutoff_iso)

    # load minimal in-memory index for candidate sampling
    seqs: List[Dict[str, Any]] = list(_iter_records(input_path))
    reviewer_ids = [rec.get('reviewer_id') or rec.get('developer_id') for rec in seqs if (rec.get('reviewer_id') or rec.get('developer_id'))]
    reviewer_ids = [rid for rid in reviewer_ids if rid and not _is_service_account(rid)]
    reviewer_ids = list(dict.fromkeys(reviewer_ids))
    reviewer_id_set = set(reviewer_ids)

    project_filter_set: Optional[Set[str]] = None
    if project_filter:
        project_filter_set = {p.strip() for p in project_filter if p and p.strip()}
        if not project_filter_set:
            project_filter_set = None

    changes_index = _load_changes_index(changes_json, project_filter_set)
    event_lookup: Dict[Tuple[str, datetime], List[Dict[str, Any]]] = {}
    project_events: Dict[str, List[Tuple[datetime, str]]] = {}
    if changes_index:
        event_lookup = changes_index.get('event_lookup', {})
        project_events = changes_index.get('project_events', {})

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

    def _project_active_candidates(project: str, when: datetime) -> List[str]:
        if not project or project not in project_events:
            return []
        events = project_events.get(project, [])
        if not events:
            return []
        lo = when - timedelta(days=int(candidate_activity_window_days))
        idx = bisect_left(events, (lo, ''))
        active: List[str] = []
        n = len(events)
        while idx < n:
            ts, dev = events[idx]
            if ts > when:
                break
            if dev and not _is_service_account(dev):
                active.append(dev)
            idx += 1
        return list(dict.fromkeys(active))

    # Helper: extract second-level directory tokens from path list
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
    def _select_meta(meta_list: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if not meta_list:
            return None
        if project_filter_set:
            for meta in meta_list:
                proj = meta.get('project')
                if proj in project_filter_set:
                    return meta
            return None
        return meta_list[0]

    for rec in seqs:
        rid_main = rec.get('reviewer_id') or rec.get('developer_id') or 'unknown@example.com'
        trans = rec.get('transitions') or []
        rid_lower = rid_main.lower()
        for tr in trans:
            ts = tr.get('t') or tr.get('timestamp')
            if not ts:
                continue
            try:
                tdt = _parse_iso(str(ts))
            except Exception:
                continue
            meta_list = event_lookup.get((rid_lower, tdt))
            selected_meta = _select_meta(meta_list)
            change_info = selected_meta.get('change_info') if selected_meta else {}
            owner_email = (selected_meta.get('owner') if selected_meta else None) or rec.get('owner_id') or rec.get('owner') or rec.get('author')

            file_paths = change_info.get('file_paths') if isinstance(change_info, dict) else None
            if not file_paths:
                fallback_files = rec.get('files') or rec.get('changed_files') or []
                file_paths = [p for p in fallback_files if isinstance(p, str)]
            dir1_tokens = set(change_info.get('dir1') or []) if isinstance(change_info, dict) else set()
            dir2_tokens = set(change_info.get('dir2') or []) if isinstance(change_info, dict) else set()
            merged_dirs = dir1_tokens | dir2_tokens
            if not merged_dirs and file_paths:
                d1_fallback, d2_fallback = _dir_tokens(file_paths)
                merged_dirs = d1_fallback | d2_fallback
                if not dir1_tokens:
                    dir1_tokens = d1_fallback
                if not dir2_tokens:
                    dir2_tokens = d2_fallback

            owner_norm = owner_email.lower() if isinstance(owner_email, str) else None
            if owner_norm:
                key = (owner_norm, rid_lower)
                owner_pair_events.setdefault(key, []).append(tdt)
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
        key = (str(owner).lower(), str(reviewer).lower())
        lst = owner_pair_events.get(key)
        if not lst:
            return 0
        lo = when - timedelta(days=days)
        # backward scan
        cnt = 0
        for t in reversed(lst):
            if t >= when:
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
                if _is_service_account(rid):
                    continue
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
                    rid_lower = rid.lower()
                    active_pool: List[str] = []
                    owner_email = None
                    project_name = None
                    selected_meta: Optional[Dict[str, Any]] = None
                    if event_lookup:
                        meta_list = event_lookup.get((rid_lower, when))
                        selected_meta = _select_meta(meta_list)
                        if selected_meta is None and project_filter_set:
                            continue
                        if selected_meta:
                            owner_email = selected_meta.get('owner')
                            project_name = selected_meta.get('project')
                            if project_name:
                                active_pool = _project_active_candidates(project_name, when)
                    elif project_filter_set:
                        # Metadata required to enforce project filter
                        continue
                    if project_filter_set and project_name is None:
                        # If filtering by project but metadata didn't yield project, skip task
                        continue
                    change_info = selected_meta.get('change_info') if selected_meta else {}
                    filtered_project_pool: List[str] = []
                    active_pool = list(dict.fromkeys(active_pool))
                    for candidate_id in active_pool:
                        if not candidate_id:
                            continue
                        if _is_service_account(candidate_id):
                            continue
                        cand_lower = candidate_id.lower()
                        if cand_lower == rid_lower:
                            continue
                        if owner_email and cand_lower == owner_email.lower():
                            continue
                        if candidate_id not in reviewer_id_set:
                            continue
                        filtered_project_pool.append(candidate_id)
                    filtered_project_pool = list(dict.fromkeys(filtered_project_pool))

                    change_info_dict = change_info if isinstance(change_info, dict) else {}
                    change_file_paths = list(change_info_dict.get('file_paths') or [])
                    change_dir1 = set(change_info_dict.get('dir1') or [])
                    change_dir2 = set(change_info_dict.get('dir2') or [])
                    if not change_dir1 and not change_dir2 and change_file_paths:
                        d1_tmp, d2_tmp = _dir_tokens(change_file_paths)
                        change_dir1 = d1_tmp
                        change_dir2 = d2_tmp
                    if not change_file_paths:
                        fallback_files = rec.get('files') or rec.get('changed_files') or []
                        fallback_files = [p for p in fallback_files if isinstance(p, str)]
                        if fallback_files:
                            change_file_paths = list(fallback_files)
                            if not change_dir1 and not change_dir2:
                                d1_tmp, d2_tmp = _dir_tokens(change_file_paths)
                                change_dir1 = d1_tmp
                                change_dir2 = d2_tmp
                    change_dirs_all = change_dir1 | change_dir2
                    file_count = len(change_file_paths)
                    dir1_count = len(change_dir1)
                    dir2_count = len(change_dir2)
                    total_churn_val = float(change_info_dict.get('total_churn', 0.0))
                    total_churn_log = float(change_info_dict.get('total_churn_log1p', _math.log1p(total_churn_val))) if total_churn_val >= 0 else 0.0
                    file_count_log = float(change_info_dict.get('file_count_log1p', _math.log1p(file_count))) if file_count >= 0 else 0.0
                    doc_ratio = float(change_info_dict.get('doc_ratio', 0.0))
                    test_ratio = float(change_info_dict.get('test_ratio', 0.0))
                    avg_depth = float(change_info_dict.get('avg_path_depth', 0.0))
                    max_depth = float(change_info_dict.get('max_path_depth', 0.0))
                    primary_dir_max_frac = float(change_info_dict.get('primary_dir_max_frac', 0.0))
                    subject_length = float(change_info_dict.get('subject_length', 0.0))
                    subject_length_log = float(change_info_dict.get('subject_length_log1p', _math.log1p(subject_length) if subject_length > 0 else 0.0)) if subject_length >= 0 else 0.0
                    subject_word_count = float(change_info_dict.get('subject_word_count', 0.0))
                    is_revert = float(change_info_dict.get('is_revert', 0.0))
                    topic_length = float(change_info_dict.get('topic_length', 0.0))

                    task_feature_values: Dict[str, float] = {}
                    if feature_registry:
                        def _maybe_add_task_feat(name: str, value: float) -> None:
                            if name in feature_registry:
                                task_feature_values[name] = float(value)

                        _maybe_add_task_feat('change_file_count', float(file_count))
                        _maybe_add_task_feat('change_file_count_log1p', file_count_log)
                        _maybe_add_task_feat('change_primary_dir_count', float(dir1_count))
                        _maybe_add_task_feat('change_secondary_dir_count', float(dir2_count))
                        _maybe_add_task_feat('change_primary_dir_max_frac', primary_dir_max_frac)
                        insertions_val = float(change_info_dict.get('insertions', 0.0))
                        deletions_val = float(change_info_dict.get('deletions', 0.0))
                        churn_per_file = float(total_churn_val) / float(max(1, file_count))
                        _maybe_add_task_feat('change_total_churn', total_churn_val)
                        _maybe_add_task_feat('change_total_churn_log1p', total_churn_log)
                        _maybe_add_task_feat('change_insertions', insertions_val)
                        _maybe_add_task_feat('change_insertions_log1p', float(_math.log1p(max(0.0, insertions_val))))
                        _maybe_add_task_feat('change_deletions', deletions_val)
                        _maybe_add_task_feat('change_deletions_log1p', float(_math.log1p(max(0.0, deletions_val))))
                        _maybe_add_task_feat('change_churn_per_file', churn_per_file)
                        _maybe_add_task_feat('change_doc_file_ratio', doc_ratio)
                        _maybe_add_task_feat('change_test_file_ratio', test_ratio)
                        _maybe_add_task_feat('change_avg_path_depth', avg_depth)
                        _maybe_add_task_feat('change_max_path_depth', max_depth)
                        _maybe_add_task_feat('change_subject_length', subject_length)
                        _maybe_add_task_feat('change_subject_length_log1p', subject_length_log)
                        _maybe_add_task_feat('change_subject_word_count', subject_word_count)
                        _maybe_add_task_feat('change_is_revert', is_revert)
                        _maybe_add_task_feat('change_topic_length', topic_length)
                    else:
                        task_feature_values = {}

                    # Build candidates (ensure GT is included but remove positional bias by shuffling later)
                    cands = [rid]
                    limit = max_candidates if (max_candidates is not None and max_candidates > 0) else None
                    if filtered_project_pool:
                        pool_iter = filtered_project_pool
                    else:
                        if project_filter_set:
                            pool_iter = []
                        else:
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
                            pool_iter = pool
                    for r in pool_iter:
                        if limit is not None and len(cands) >= limit:
                            break
                        cands.append(r)

                    # Deduplicate (safety)
                    seen = set()
                    cands = [x for x in cands if not (x in seen or seen.add(x))]

                    if len(cands) < 2:
                        continue

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
                        if task_feature_values:
                            feats.update(task_feature_values)
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
                                owner_id = owner_email or rec.get('owner_id') or rec.get('owner') or rec.get('author')
                                if owner_id:
                                    feats['owner_reviewer_past_assignments_180d'] = float(
                                        _lookup_pair_count(owner_id, r, when, days=180)
                                    )
                                else:
                                    feats['owner_reviewer_past_assignments_180d'] = 0.0
                            hist_dirs_for_r: Optional[Set[str]] = None
                            hist_dirs_dir2_only: Optional[Set[str]] = None
                            hist_counter_for_r: Optional[Counter] = None
                            path_dir2_tokens = change_dir2
                            path_dir_all_tokens = change_dirs_all

                            def _ensure_hist_dirs() -> Set[str]:
                                nonlocal hist_dirs_for_r, hist_dirs_dir2_only
                                if hist_dirs_for_r is None:
                                    hist_dirs_for_r = _lookup_dir2_history(r, when, days=180)
                                if hist_dirs_dir2_only is None:
                                    dir2_only = {tok for tok in hist_dirs_for_r if isinstance(tok, str) and '/' in tok}
                                    hist_dirs_dir2_only = dir2_only if dir2_only else set(hist_dirs_for_r)
                                return hist_dirs_dir2_only

                            # Path Jaccard dir2 similarity
                            if 'path_jaccard_dir2_180d' in feature_registry:
                                if path_dir2_tokens:
                                    hist_dir2 = _ensure_hist_dirs()
                                    feats['path_jaccard_dir2_180d'] = _jaccard(path_dir2_tokens, hist_dir2)
                                else:
                                    feats['path_jaccard_dir2_180d'] = 0.0
                            if 'path_overlap_count_dir2_180d' in feature_registry:
                                if path_dir2_tokens:
                                    hist_dir2 = _ensure_hist_dirs()
                                    feats['path_overlap_count_dir2_180d'] = float(len(path_dir2_tokens & hist_dir2))
                                else:
                                    feats['path_overlap_count_dir2_180d'] = 0.0
                            if 'path_overlap_ratio_dir2_180d' in feature_registry:
                                if path_dir2_tokens:
                                    hist_dir2 = _ensure_hist_dirs()
                                    overlap_cnt = len(path_dir2_tokens & hist_dir2)
                                    feats['path_overlap_ratio_dir2_180d'] = float(overlap_cnt) / float(max(1, len(path_dir2_tokens)))
                                else:
                                    feats['path_overlap_ratio_dir2_180d'] = 0.0
                            if 'path_tfidf_cosine_recent30' in feature_registry:
                                if path_dir_all_tokens:
                                    if hist_counter_for_r is None:
                                        hist_counter_for_r = _lookup_recent_dirs_tfidf(r, when, days=30)
                                    feats['path_tfidf_cosine_recent30'] = _tfidf_cosine(path_dir_all_tokens, hist_counter_for_r)
                                else:
                                    feats['path_tfidf_cosine_recent30'] = 0.0
                            if 'path_overlap_recent30' in feature_registry:
                                if path_dir_all_tokens:
                                    if hist_counter_for_r is None:
                                        hist_counter_for_r = _lookup_recent_dirs_tfidf(r, when, days=30)
                                    feats['path_overlap_recent30'] = float(sum(hist_counter_for_r.get(tok, 0) for tok in path_dir_all_tokens))
                                else:
                                    feats['path_overlap_recent30'] = 0.0
                            # Pair normalization features
                            if 'owner_reviewer_past_assignments_180d_log1p' in feature_registry:
                                val = feats.get('owner_reviewer_past_assignments_180d', 0.0)
                                feats['owner_reviewer_past_assignments_180d_log1p'] = float(_math.log1p(val))
                            if 'owner_reviewer_past_assignments_ratio_180d' in feature_registry:
                                # denominator: reviewer total assignments (irrespective of owner) in 180d
                                reviewer_total = 0
                                r_lower = r.lower()
                                for (own, rv), times in owner_pair_events.items():
                                    if rv == r_lower:
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

    if train_window_months is not None and train_window_days is not None:
        raise ValueError('train-window-days と train-window-months は同時指定できません。')
    if eval_window_months is not None and eval_window_days is not None:
        raise ValueError('eval-window-days と eval-window-months は同時指定できません。')
    if candidate_activity_window_months is not None and candidate_activity_window_days is not None:
        raise ValueError('candidate-activity-window-days と candidate-activity-window-months は同時指定できません。')

    if train_window_months is not None:
        train_window_days = int(round(float(train_window_months) * 30))
    if eval_window_months is not None:
        eval_window_days = int(round(float(eval_window_months) * 30))
    if candidate_activity_window_months is not None:
        candidate_activity_window_days = int(round(float(candidate_activity_window_months) * 30))
    if candidate_activity_window_days is None:
        candidate_activity_window_days = 180

    train_window = int(train_window_days) if train_window_days is not None else None
    eval_window = int(eval_window_days) if eval_window_days is not None else None

    if train_window is not None:
        train_start = cutoff - timedelta(days=train_window)
        train_pred = lambda t, lo=train_start, hi=cutoff: lo <= t <= hi
    else:
        train_pred = lambda t, hi=cutoff: t <= hi

    if eval_window is not None:
        eval_end = cutoff + timedelta(days=eval_window)
        eval_pred = lambda t, lo=cutoff, hi=eval_end: lo < t <= hi
    else:
        eval_pred = lambda t, lo=cutoff: t > lo

    train_path, n_train = _gen('train', train_pred)
    eval_path, n_eval = _gen('eval', eval_pred)
    meta = {
        'source': str(input_path),
        'cutoff': cutoff.isoformat(),
        'train_tasks': train_path,
        'train_count': n_train,
        'eval_tasks': eval_path,
        'eval_count': n_eval,
    'max_candidates': int(max_candidates) if isinstance(max_candidates, int) and max_candidates > 0 else None,
        'seed': int(seed) if seed is not None else None,
        'candidate_sampling': str(candidate_sampling),
        'candidate_window_days': int(candidate_window_days),
        'shuffle_candidates': bool(shuffle_candidates),
        'registry_features_loaded': list(feature_registry.keys()),
        'train_window_days': train_window,
        'eval_window_days': eval_window,
        'train_window_months': float(train_window_months) if train_window_months is not None else None,
        'eval_window_months': float(eval_window_months) if eval_window_months is not None else None,
        'changes_json': str(changes_json) if changes_json is not None else None,
        'candidate_activity_window_days': int(candidate_activity_window_days) if candidate_activity_window_days is not None else None,
        'candidate_activity_window_months': float(candidate_activity_window_months) if candidate_activity_window_months is not None else None,
        'project_filter': sorted(project_filter_set) if project_filter_set else None,
    }
    (out_dir / 'tasks_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='outputs/irl/reviewer_sequences.json')
    ap.add_argument('--cutoff', type=str, default='2024-07-01T00:00:00Z')
    ap.add_argument('--outdir', type=str, default='outputs/task_assign')
    ap.add_argument('--max-candidates', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--candidate-sampling', type=str, default='random', choices=['random', 'time-local'])
    ap.add_argument('--candidate-window-days', type=int, default=30)
    ap.add_argument('--no-shuffle-candidates', action='store_true', help='Disable deterministic candidate shuffling (reduces positional bias by default).')
    ap.add_argument('--train-window-days', type=int, default=None, help='Restrict training tasks to this many days before cutoff (inclusive).')
    ap.add_argument('--eval-window-days', type=int, default=None, help='Restrict evaluation tasks to this many days after cutoff (inclusive).')
    ap.add_argument('--train-window-months', type=float, default=None, help='train-window-days の代わりに月単位で指定します (1か月=30日換算)。')
    ap.add_argument('--eval-window-months', type=float, default=None, help='eval-window-days の代わりに月単位で指定します (1か月=30日換算)。')
    ap.add_argument('--changes-json', type=str, default='data/processed/unified/all_reviews.json', help='Project activity source (all_reviews.json)。指定しない場合はプロジェクト制約をスキップ。')
    ap.add_argument('--candidate-activity-window-days', type=int, default=None, help='候補者を抽出する過去日数 (レビュー/コミット活動)。')
    ap.add_argument('--candidate-activity-window-months', type=float, default=6.0, help='候補者抽出ウィンドウ（月単位、1か月=30日換算）。days と同時指定不可。')
    ap.add_argument('--project', action='append', default=None, help='この名前のプロジェクトのみを対象にタスクを生成します。複数指定可。')
    args = ap.parse_args()
    meta = build_tasks_from_sequences(
        Path(args.input),
        args.cutoff,
        Path(args.outdir),
    max_candidates=args.max_candidates,
        seed=int(args.seed),
        candidate_sampling=str(args.candidate_sampling),
        candidate_window_days=int(args.candidate_window_days),
        shuffle_candidates=not bool(args.no_shuffle_candidates),
        train_window_days=args.train_window_days,
        eval_window_days=args.eval_window_days,
        train_window_months=args.train_window_months,
        eval_window_months=args.eval_window_months,
        changes_json=Path(args.changes_json) if args.changes_json else None,
        candidate_activity_window_days=args.candidate_activity_window_days,
        candidate_activity_window_months=args.candidate_activity_window_months,
        project_filter=args.project,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
