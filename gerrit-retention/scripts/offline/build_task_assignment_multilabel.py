#!/usr/bin/env python3
"""Multi-label版 AssignmentTask(JSONL) 生成スクリプト。

reviewer_sequences.json(.gz|.jsonl) を入力に、change_id 単位で候補レビュア集合と
複数正解 (positive_reviewer_ids) を構築します。既存シングルラベル版
`scripts/offline/build_task_assignment_from_sequences.py` の仕様を踏襲しつつ、
最終的に action==1 を記録したレビュア全員を正解集合として保持します。

出力: outdir/tasks_train.jsonl, tasks_eval.jsonl, tasks_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm

# 既存スクリプト (シングルラベル版) を動的にロード
_THIS_DIR = Path(__file__).resolve().parent
_LEGACY_PATH = _THIS_DIR / 'build_task_assignment_from_sequences.py'
_LEGACY_SPEC = importlib.util.spec_from_file_location('_legacy_task_assignment', _LEGACY_PATH)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:
    raise ImportError(f'legacy script not found: {_LEGACY_PATH}')
legacy = importlib.util.module_from_spec(_LEGACY_SPEC)
sys.modules['_legacy_task_assignment'] = legacy
_LEGACY_SPEC.loader.exec_module(legacy)

# 既存スクリプトのユーティリティ関数を再利用
_parse_iso = legacy._parse_iso
_iter_records = legacy._iter_records
_is_service_account = legacy._is_service_account
_dir_tokens = legacy._dir_tokens
_collect_change_files = legacy._collect_change_files
_analyze_change_paths = legacy._analyze_change_paths
_load_changes_index = legacy._load_changes_index
_feat_from_state = legacy._feat_from_state
_load_feature_registry = legacy._load_feature_registry
_compute_registry_features = legacy._compute_registry_features


@dataclass
class CandidateSnapshot:
    reviewer_id: str
    features: Dict[str, float]
    first_seen: datetime


@dataclass
class AggregatedTask:
    change_key: str
    first_timestamp: datetime
    candidates: Dict[str, CandidateSnapshot]

    def add_candidate(self, reviewer_id: str, features: Dict[str, float], when: datetime) -> None:
        existing = self.candidates.get(reviewer_id)
        if existing is None or when < existing.first_seen:
            self.candidates[reviewer_id] = CandidateSnapshot(reviewer_id, features, when)
        elif when == existing.first_seen:
            # features を最新で上書きする程度の更新は許容
            self.candidates[reviewer_id] = CandidateSnapshot(reviewer_id, features, when)

    def ensure_timestamp(self, when: datetime) -> None:
        if when < self.first_timestamp:
            self.first_timestamp = when


def _stable_perm_indices(n: int, context_parts: Iterable[Any], seed: int | None) -> List[int]:
    """contextとグローバルseedに応じた決定的なシャッフル順を返す。"""
    idxs = list(range(n))
    if n <= 1:
        return idxs
    joined = "\u0001".join(str(k) for k in context_parts)
    base = f"{joined}|seed={seed if seed is not None else 'none'}".encode('utf-8')
    digest = hashlib.sha256(base).digest()
    local_seed = int.from_bytes(digest[:8], byteorder='big', signed=False)
    local_rng = random.Random(local_seed)
    local_rng.shuffle(idxs)
    return idxs


def _select_meta(meta_list: Optional[List[Dict[str, Any]]], project_filter: Optional[Set[str]]) -> Optional[Dict[str, Any]]:
    if not meta_list:
        return None
    if project_filter:
        for meta in meta_list:
            proj = meta.get('project')
            if proj in project_filter:
                return meta
        return None
    return meta_list[0]


def _resolve_change_key(selected_meta: Optional[Dict[str, Any]], transition: Dict[str, Any], fallback_owner: Optional[str], reviewer_id: str) -> Optional[str]:
    if selected_meta:
        for key in ('change_key', 'change_id', 'change', 'id'):
            val = selected_meta.get(key)
            if val:
                return str(val)
    for key in ('change_key', 'change_id', 'change', 'id'):
        val = transition.get(key)
        if val:
            return str(val)
    # owner+reviewer+timestampから擬似キー生成 (最悪のフォールバック)
    ts = transition.get('t') or transition.get('timestamp')
    if not ts:
        return None
    owner_token = fallback_owner or 'unknown-owner'
    return f"chg::{owner_token}::{reviewer_id}::{ts}"


def _project_active_candidates(
    project: str,
    when: datetime,
    project_events: Dict[str, List[Tuple[datetime, str]]],
    candidate_activity_window_days: int,
) -> List[str]:
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
    # 順序保持しつつ重複排除
    seen: Set[str] = set()
    uniq: List[str] = []
    for dev in active:
        if dev in seen:
            continue
        seen.add(dev)
        uniq.append(dev)
    return uniq


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _lookup_pair_count(
    owner_pair_events: Dict[Tuple[str, str], List[datetime]],
    owner: str,
    reviewer: str,
    when: datetime,
    days: int = 180,
) -> int:
    key = (str(owner).lower(), str(reviewer).lower())
    lst = owner_pair_events.get(key)
    if not lst:
        return 0
    lo = when - timedelta(days=days)
    cnt = 0
    for t in reversed(lst):
        if t >= when:
            continue
        if t < lo:
            break
        cnt += 1
    return cnt


def _lookup_dir2_history(
    reviewer_paths: Dict[str, List[Tuple[datetime, Set[str]]]],
    reviewer: str,
    when: datetime,
    days: int = 180,
) -> Set[str]:
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


def _lookup_recent_dirs_tfidf(
    reviewer_recent_dirs: Dict[str, List[Tuple[datetime, Set[str]]]],
    reviewer: str,
    when: datetime,
    days: int = 30,
) -> Counter:
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


def _tfidf_cosine(cur_dirs: Set[str], hist_counter: Counter, global_df: Counter, total_docs: int) -> float:
    if not cur_dirs or not hist_counter:
        return 0.0
    cur_vec: Dict[str, float] = {}
    hist_vec: Dict[str, float] = {}
    max_tf = max(hist_counter.values()) if hist_counter else 1.0
    for tok in cur_dirs:
        df = global_df.get(tok, 0)
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        cur_vec[tok] = 1.0 * idf
    for tok, tf in hist_counter.items():
        df = global_df.get(tok, 0)
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        norm_tf = float(tf) / float(max_tf)
        hist_vec[tok] = norm_tf * idf
    dot = 0.0
    for tok, v in cur_vec.items():
        if tok in hist_vec:
            dot += v * hist_vec[tok]
    norm_a = math.sqrt(sum(v * v for v in cur_vec.values()))
    norm_b = math.sqrt(sum(v * v for v in hist_vec.values()))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def build_multilabel_tasks_from_sequences(
    input_path: Path,
    cutoff_iso: str,
    out_dir: Path,
    max_candidates: Optional[int] = None,
    seed: int | None = 42,
    candidate_sampling: str = 'random',
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

    by_reviewer: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
    reviewer_events_index: Dict[str, List[datetime]] = {}
    first_seen_index: Dict[str, datetime] = {}
    for rec in tqdm(seqs, desc='index:states', unit='rec', leave=False):
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
            reviewer_events_index.setdefault(rid, []).append(when)
            if rid not in first_seen_index or when < first_seen_index[rid]:
                first_seen_index[rid] = when
        lst.sort(key=lambda x: x[0])

    for arr in reviewer_events_index.values():
        arr.sort()

    rng = random.Random(seed)

    owner_pair_events: Dict[Tuple[str, str], List[datetime]] = {}
    reviewer_paths: Dict[str, List[Tuple[datetime, Set[str]]]] = {}
    reviewer_recent_dirs: Dict[str, List[Tuple[datetime, Set[str]]]] = {}
    global_df: Counter = Counter()
    total_docs = 0
    change_positive_map: Dict[str, Set[str]] = defaultdict(set)

    def _register_dir_tokens(rid_main: str, tdt: datetime, dirs: Set[str]) -> None:
        reviewer_paths.setdefault(rid_main, []).append((tdt, dirs))
        reviewer_recent_dirs.setdefault(rid_main, []).append((tdt, dirs))
        nonlocal total_docs
        total_docs += 1
        for tok in set(dirs):
            global_df[tok] += 1

    for rec in tqdm(seqs, desc='index:meta', unit='rec', leave=False):
        rid_main = rec.get('reviewer_id') or rec.get('developer_id') or 'unknown@example.com'
        if _is_service_account(rid_main):
            continue
        rid_lower = rid_main.lower()
        trans = rec.get('transitions') or []
        for tr in trans:
            ts = tr.get('t') or tr.get('timestamp')
            if not ts:
                continue
            try:
                tdt = _parse_iso(str(ts))
            except Exception:
                continue
            meta_list = event_lookup.get((rid_lower, tdt))
            selected_meta = _select_meta(meta_list, project_filter_set)
            if project_filter_set and selected_meta is None:
                continue
            change_key = _resolve_change_key(selected_meta, tr, rec.get('owner') or rec.get('owner_id'), rid_main)
            action = int(tr.get('action', 2))
            if action == 1 and change_key:
                change_positive_map[change_key].add(rid_main)
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
                _register_dir_tokens(rid_main, tdt, merged_dirs)

    for _rid, lst in reviewer_paths.items():
        lst.sort(key=lambda x: x[0])
    for _k, lst in owner_pair_events.items():
        lst.sort()

    registry_path = out_dir.parent / 'configs' / 'irl_feature_registry.yaml'
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

    def _latest_state_before(rid: str, when: datetime) -> Optional[Dict[str, Any]]:
        arr = by_reviewer.get(rid)
        if not arr:
            return None
        latest: Optional[Dict[str, Any]] = None
        for t, st in reversed(arr):
            if t <= when:
                latest = st
                break
        return latest

    def _build_phase_tasks(phase: str, pred) -> Tuple[str, int]:
        outp = out_dir / f"tasks_{phase}.jsonl"
        aggregated: Dict[str, AggregatedTask] = {}
        for rec in tqdm(seqs, desc=f'{phase}:records', unit='rec', leave=False):
            rid = rec.get('reviewer_id') or rec.get('developer_id') or 'unknown@example.com'
            if _is_service_account(rid):
                continue
            trans = rec.get('transitions') or []
            total_before = 0
            accept_before = 0
            hour_counter: Counter = Counter()
            for tr in trans:
                ts = tr.get('t') or tr.get('timestamp')
                if not ts:
                    continue
                try:
                    when = _parse_iso(str(ts))
                except Exception:
                    continue
                if not pred(when):
                    continue
                a = int(tr.get('action', 2))
                state = tr.get('state', {}) or {}
                rid_lower = rid.lower()
                meta_list = event_lookup.get((rid_lower, when))
                selected_meta = _select_meta(meta_list, project_filter_set)
                if project_filter_set and selected_meta is None:
                    continue
                if selected_meta is None and project_filter_set:
                    continue
                owner_email = None
                project_name = None
                if selected_meta:
                    owner_email = selected_meta.get('owner')
                    project_name = selected_meta.get('project')
                else:
                    owner_email = rec.get('owner_id') or rec.get('owner') or rec.get('author')
                if project_filter_set and project_name is None:
                    continue
                change_key = _resolve_change_key(selected_meta, tr, owner_email, rid)
                if not change_key:
                    continue
                change_info = selected_meta.get('change_info') if selected_meta else {}
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
                total_churn_log = float(change_info_dict.get('total_churn_log1p', math.log1p(total_churn_val))) if total_churn_val >= 0 else 0.0
                file_count_log = float(change_info_dict.get('file_count_log1p', math.log1p(file_count))) if file_count >= 0 else 0.0
                doc_ratio = float(change_info_dict.get('doc_ratio', 0.0))
                test_ratio = float(change_info_dict.get('test_ratio', 0.0))
                avg_depth = float(change_info_dict.get('avg_path_depth', 0.0))
                max_depth = float(change_info_dict.get('max_path_depth', 0.0))
                primary_dir_max_frac = float(change_info_dict.get('primary_dir_max_frac', 0.0))
                subject_length = float(change_info_dict.get('subject_length', 0.0))
                subject_length_log = float(change_info_dict.get('subject_length_log1p', math.log1p(subject_length) if subject_length > 0 else 0.0)) if subject_length >= 0 else 0.0
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
                    _maybe_add_task_feat('change_insertions_log1p', float(math.log1p(max(0.0, insertions_val))))
                    _maybe_add_task_feat('change_deletions', deletions_val)
                    _maybe_add_task_feat('change_deletions_log1p', float(math.log1p(max(0.0, deletions_val))))
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

                # 候補生成
                cands = [rid]
                limit = max_candidates if (max_candidates is not None and max_candidates > 0) else None
                if project_name:
                    active_pool = _project_active_candidates(project_name, when, project_events, candidate_activity_window_days)
                else:
                    active_pool = []
                if active_pool:
                    pool_iter = active_pool
                else:
                    if project_filter_set:
                        pool_iter = []
                    else:
                        pool = [x for x in reviewer_ids if x and x != rid]
                        if candidate_sampling == 'time-local':
                            window = candidate_window_days
                            lo = when - timedelta(days=window)
                            hi = when + timedelta(days=window)
                            filtered = []
                            for r in pool:
                                arr = by_reviewer.get(r) or []
                                ok = any(lo <= t <= hi for (t, _st) in arr)
                                if ok:
                                    filtered.append(r)
                            pool = filtered or pool
                        rng.shuffle(pool)
                        pool_iter = pool
                for r in pool_iter:
                    if limit is not None and len(cands) >= limit:
                        break
                    if not r or _is_service_account(r):
                        continue
                    if owner_email and isinstance(owner_email, str) and r.lower() == owner_email.lower():
                        continue
                    cands.append(r)

                seen_ids: Set[str] = set()
                deduped = []
                for cid in cands:
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    deduped.append(cid)
                cands = deduped

                if len(cands) < 2:
                    continue

                cand_objs: List[Tuple[str, Dict[str, float]]] = []
                for r in cands:
                    cand_state = _latest_state_before(r, when) or state
                    feats = _feat_from_state(cand_state)
                    if task_feature_values:
                        feats.update(task_feature_values)
                    if feature_registry:
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
                            reviewer_transition_meta={},
                        )
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
                        if 'owner_reviewer_past_assignments_180d' in feature_registry:
                            owner_id = owner_email or rec.get('owner_id') or rec.get('owner') or rec.get('author')
                            if owner_id:
                                feats['owner_reviewer_past_assignments_180d'] = float(
                                    _lookup_pair_count(owner_pair_events, owner_id, r, when, days=180)
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
                                hist_dirs_for_r = _lookup_dir2_history(reviewer_paths, r, when, days=180)
                            if hist_dirs_dir2_only is None:
                                dir2_only = {tok for tok in hist_dirs_for_r if isinstance(tok, str) and '/' in tok}
                                hist_dirs_dir2_only = dir2_only if dir2_only else set(hist_dirs_for_r)
                            return hist_dirs_dir2_only

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
                                    hist_counter_for_r = _lookup_recent_dirs_tfidf(reviewer_recent_dirs, r, when, days=30)
                                feats['path_tfidf_cosine_recent30'] = _tfidf_cosine(path_dir_all_tokens, hist_counter_for_r, global_df, total_docs)
                            else:
                                feats['path_tfidf_cosine_recent30'] = 0.0
                        if 'path_overlap_recent30' in feature_registry:
                            if path_dir_all_tokens:
                                if hist_counter_for_r is None:
                                    hist_counter_for_r = _lookup_recent_dirs_tfidf(reviewer_recent_dirs, r, when, days=30)
                                feats['path_overlap_recent30'] = float(sum(hist_counter_for_r.get(tok, 0) for tok in path_dir_all_tokens))
                            else:
                                feats['path_overlap_recent30'] = 0.0
                        if 'owner_reviewer_past_assignments_180d_log1p' in feature_registry:
                            val = feats.get('owner_reviewer_past_assignments_180d', 0.0)
                            feats['owner_reviewer_past_assignments_180d_log1p'] = float(math.log1p(val))
                        if 'owner_reviewer_past_assignments_ratio_180d' in feature_registry:
                            reviewer_total = 0
                            r_lower = r.lower()
                            for (own, rv), times in owner_pair_events.items():
                                if rv == r_lower:
                                    cnt = 0
                                    lo = when - timedelta(days=180)
                                    for tt in reversed(times):
                                        if tt > when:
                                            continue
                                        if tt < lo:
                                            break
                                        cnt += 1
                                    reviewer_total += cnt
                            pair_val = feats.get('owner_reviewer_past_assignments_180d', 0.0)
                            ratio = pair_val / reviewer_total if reviewer_total > 0 else 0.0
                            feats['owner_reviewer_past_assignments_ratio_180d'] = float(min(1.0, ratio))
                        feats.update(ext)

                    cand_objs.append((r, feats))

                agg = aggregated.get(change_key)
                if agg is None:
                    agg = AggregatedTask(change_key=change_key, first_timestamp=when, candidates={})
                    aggregated[change_key] = agg
                else:
                    agg.ensure_timestamp(when)

                for reviewer_id, features in cand_objs:
                    agg.add_candidate(reviewer_id, features, when)

                total_before += 1
                if a == 1:
                    accept_before += 1
                hour_counter[when.hour] += 1

        count = 0
        with outp.open('w', encoding='utf-8') as wf:
            for change_key, agg in sorted(aggregated.items(), key=lambda kv: (kv[1].first_timestamp, kv[0])):
                candidate_items = sorted(agg.candidates.values(), key=lambda c: (c.first_seen, c.reviewer_id))
                if len(candidate_items) < 2:
                    continue
                candidate_map = {c.reviewer_id: c for c in candidate_items}
                positives_all = sorted(change_positive_map.get(change_key, set()))
                positives_in_candidates = [rid for rid in positives_all if rid in candidate_map]
                if max_candidates is not None and max_candidates > 0 and len(candidate_items) > max_candidates:
                    positive_set = set(positives_in_candidates)
                    positive_items = [candidate_map[rid] for rid in positives_in_candidates]
                    other_items = [c for c in candidate_items if c.reviewer_id not in positive_set]
                    trimmed = positive_items + other_items
                    candidate_items = trimmed[:max_candidates]
                    candidate_map = {c.reviewer_id: c for c in candidate_items}
                    positives_in_candidates = [rid for rid in positives_in_candidates if rid in candidate_map]
                # 正例が候補から抜けた場合は強制的に追加
                for rid in positives_all:
                    if rid in agg.candidates and rid not in candidate_map:
                        candidate_items.insert(0, agg.candidates[rid])
                if max_candidates is not None and max_candidates > 0 and len(candidate_items) > max_candidates:
                    seen: Set[str] = set()
                    reordered: List[CandidateSnapshot] = []
                    for item in candidate_items:
                        if item.reviewer_id in seen:
                            continue
                        seen.add(item.reviewer_id)
                        reordered.append(item)
                    positive_set = set(positives_all)
                    positive_items = [c for c in reordered if c.reviewer_id in positive_set]
                    other_items = [c for c in reordered if c.reviewer_id not in positive_set]
                    candidate_items = (positive_items + other_items)[:max_candidates]
                # 重複排除最終チェック
                seen_final: Set[str] = set()
                dedup_final: List[CandidateSnapshot] = []
                for item in candidate_items:
                    if item.reviewer_id in seen_final:
                        continue
                    seen_final.add(item.reviewer_id)
                    dedup_final.append(item)
                candidate_items = dedup_final
                if len(candidate_items) < 2:
                    continue
                context = [change_key, agg.first_timestamp.isoformat()] + [c.reviewer_id for c in candidate_items]
                perm = _stable_perm_indices(len(candidate_items), context, seed if shuffle_candidates else None)
                ordered = [candidate_items[i] for i in perm]
                positives_in_candidates = [rid for rid in positives_all if rid in {c.reviewer_id for c in candidate_items}]
                task_obj = {
                    'change_id': change_key,
                    'timestamp': agg.first_timestamp.isoformat(),
                    'candidates': [
                        {
                            'reviewer_id': c.reviewer_id,
                            'features': c.features,
                        }
                        for c in ordered
                    ],
                    'positive_reviewer_ids': positives_in_candidates,
                }
                wf.write(json.dumps(task_obj, ensure_ascii=False) + "\n")
                count += 1
        return str(outp), count

    train_path, n_train = _build_phase_tasks('train', train_pred)
    eval_path, n_eval = _build_phase_tasks('eval', eval_pred)

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
        'label_type': 'multi',
        'positive_changes': {k: sorted(v) for k, v in change_positive_map.items()},
    }
    (out_dir / 'tasks_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='outputs/irl/reviewer_sequences.json')
    ap.add_argument('--cutoff', type=str, default='2024-07-01T00:00:00Z')
    ap.add_argument('--outdir', type=str, default='outputs/task_assign_multilabel')
    ap.add_argument('--max-candidates', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--candidate-sampling', type=str, default='random', choices=['random', 'time-local'])
    ap.add_argument('--candidate-window-days', type=int, default=30)
    ap.add_argument('--no-shuffle-candidates', action='store_true')
    ap.add_argument('--train-window-days', type=int, default=None)
    ap.add_argument('--eval-window-days', type=int, default=None)
    ap.add_argument('--train-window-months', type=float, default=None)
    ap.add_argument('--eval-window-months', type=float, default=None)
    ap.add_argument('--changes-json', type=str, default='data/processed/unified/all_reviews.json')
    ap.add_argument('--candidate-activity-window-days', type=int, default=None)
    ap.add_argument('--candidate-activity-window-months', type=float, default=6.0)
    ap.add_argument('--project', action='append', default=None)
    args = ap.parse_args()

    meta = build_multilabel_tasks_from_sequences(
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
