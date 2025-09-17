from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .reviewer_invitation_ranking import _path_tokens  # type: ignore
from .reviewer_invitation_ranking import InvitationRankingBuildConfig


def _parse_dt(s: str | None):
    if not s:
        return None
    from datetime import datetime
    for fmt in (
        '%Y-%m-%d %H:%M:%S.%f000000',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d',
    ):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _tfidf_cosine(change_tokens: List[str] | set[str], reviewer_tokens: List[str] | set[str], idf: Dict[str, float]) -> float:
    if not change_tokens or not reviewer_tokens:
        return 0.0
    ch = set(change_tokens)
    rv = set(reviewer_tokens)
    common = ch & rv
    if not common:
        return 0.0
    import math
    num = sum((idf.get(t, 0.0)) ** 2 for t in common)
    denom = math.sqrt(sum((idf.get(t, 0.0)) ** 2 for t in ch)) * math.sqrt(
        sum((idf.get(t, 0.0)) ** 2 for t in rv)
    )
    return float(num / denom) if denom > 0 else 0.0


def _ranking_metrics_by_group(y_true: List[int], y_prob: List[float], group_ids: List[int], ks: List[int] = [1, 3, 5, 10]):
    from math import log2
    groups: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for yt, yp, gid in zip(y_true, y_prob, group_ids):
        groups[gid].append((yt, yp))
    recall_at = {k: [] for k in ks}
    map_at = {k: [] for k in ks}
    ndcg_at = {k: [] for k in ks}
    for g, items in groups.items():
        items_sorted = sorted(items, key=lambda t: t[1], reverse=True)
        labels = [t[0] for t in items_sorted]
        pos_count = sum(labels)
        if pos_count == 0:
            for k in ks:
                recall_at[k].append(0.0)
                map_at[k].append(0.0)
                ndcg_at[k].append(0.0)
            continue
        for k in ks:
            topk = labels[:k]
            recall_at[k].append(sum(topk) / pos_count)
            ap = 0.0
            hit = 0
            for i, lab in enumerate(topk, start=1):
                if lab == 1:
                    hit += 1
                    ap += hit / i
            map_at[k].append(ap / min(pos_count, k))
            dcg = 0.0
            for i, lab in enumerate(topk, start=1):
                if lab == 1:
                    dcg += 1.0 / log2(i + 1)
            ideal_hits = min(pos_count, k)
            idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1)) if ideal_hits > 0 else 0.0
            ndcg_at[k].append(dcg / idcg if idcg > 0 else 0.0)
    out = {}
    for k in ks:
        def avg(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0
        out[f'recall@{k}'] = avg(recall_at[k])
        out[f'map@{k}'] = avg(map_at[k])
        out[f'ndcg@{k}'] = avg(ndcg_at[k])
    return out


@dataclass
class AcceptanceBuildConfig:
    min_total_reviews: int = 20
    recent_days: int = 30
    idf_mode: str = 'global'  # kept for compatibility
    idf_recent_days: int = 90


def build_invitee_acceptance_samples(json_path: str | Path, cfg: AcceptanceBuildConfig | None = None):
    cfg = cfg or AcceptanceBuildConfig()
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    from datetime import timedelta

    # First pass: build change records and histories
    reviewer_hist: Dict[int, List[Tuple[Any, str]]] = defaultdict(list)
    reviewer_last_ts: Dict[int, Any] = {}
    reviewer_file_hist: Dict[int, List[Tuple[Any, set[str]]]] = defaultdict(list)
    raw_changes: List[Dict[str, Any]] = []

    def _extract_file_paths(ch_obj: Dict[str, Any]) -> List[str]:
        paths: set[str] = set()
        cur_rev = ch_obj.get('current_revision')
        revisions = ch_obj.get('revisions') or {}
        if isinstance(revisions, dict) and cur_rev and cur_rev in revisions:
            files_dict = revisions[cur_rev].get('files') or {}
            if isinstance(files_dict, dict):
                for p in files_dict.keys():
                    if isinstance(p, str):
                        paths.add(p)
        if not paths and isinstance(revisions, dict):
            for rev in revisions.values():
                files_dict = rev.get('files') or {}
                if isinstance(files_dict, dict):
                    for p in files_dict.keys():
                        if isinstance(p, str):
                            paths.add(p)
        cps = ch_obj.get('currentPatchSet') or {}
        for f in cps.get('files', []) or []:
            p = f.get('file') or f.get('path')
            if p:
                paths.add(p)
        for ps in ch_obj.get('patchSets', []) or []:
            for f in ps.get('files', []) or []:
                p = f.get('file') or f.get('path')
                if p:
                    paths.add(p)
        for f in ch_obj.get('files', []) or []:
            p = f.get('file') if isinstance(f, dict) else None
            if p:
                paths.add(p)
        cleaned = {p for p in paths if p not in {'COMMIT_MSG', '/COMMIT_MSG'} and not (isinstance(p, str) and p.startswith('MergeList'))}
        return list(cleaned)[:500]

    for ch in data:
        created = _parse_dt(ch.get('created') or ch.get('updated') or ch.get('submitted'))
        if not created:
            continue
        proj = ch.get('project') or 'unknown'
        reviewers_block = (ch.get('reviewers') or {}).get('REVIEWER') or []
        invited_ids = [rv.get('_account_id') for rv in reviewers_block if rv.get('_account_id') is not None]
        submitted = _parse_dt(ch.get('submitted')) if ch.get('submitted') else None
        file_paths = _extract_file_paths(ch)
        participated: set[int] = set()
        for msg in ch.get('messages', []) or []:
            tag = (msg.get('tag') or '')
            if tag.startswith('autogenerated:'):
                continue
            aid = (msg.get('author') or {}).get('_account_id')
            if aid in invited_ids:
                participated.add(aid)
        raw_changes.append({
            'created': created,
            'project': proj,
            'invited': invited_ids,
            'positives': list(participated),
            'submitted': submitted,
            'file_paths': file_paths,
        })
        # update reviewer histories only on actual participation events
        for rid in participated:
            reviewer_hist[rid].append((created, proj))
            reviewer_last_ts[rid] = created
            if file_paths:
                reviewer_file_hist[rid].append((created, set(file_paths)))

    for rid in reviewer_hist:
        reviewer_hist[rid].sort(key=lambda x: x[0])
    for rid in reviewer_file_hist:
        reviewer_file_hist[rid].sort(key=lambda x: x[0])

    # TF-IDF idf over change token docs (global)
    for ch in raw_changes:
        ch['tokens'] = set(_path_tokens(ch.get('file_paths') or []))
    token_docs: List[set[str]] = [ch['tokens'] for ch in raw_changes]
    N_docs = len(token_docs)
    df_global: Dict[str, int] = defaultdict(int)
    for toks in token_docs:
        for t in toks:
            df_global[t] += 1
    idf_global: Dict[str, float] = {}
    for t, dfi in df_global.items():
        idf_global[t] = float(np.log((N_docs + 1) / (dfi + 1)) + 1.0)

    # Eligible reviewers threshold by total historical participations
    eligible_reviewers = {rid for rid, h in reviewer_hist.items() if len(h) >= cfg.min_total_reviews}

    # Build reviewer pending intervals (created -> submitted) for leakage-free pending count
    reviewer_participations: Dict[int, List[Tuple[Any, Any]]] = defaultdict(list)
    for ch in raw_changes:
        for rid in ch['positives']:
            reviewer_participations[rid].append((ch['created'], ch['submitted']))
    for rid in reviewer_participations:
        reviewer_participations[rid].sort(key=lambda x: x[0])

    from statistics import mean, pstdev
    ordered_changes = sorted(raw_changes, key=lambda c: c['created'])
    samples: List[Dict[str, Any]] = []
    change_idx = 0
    for ch in ordered_changes:
        created = ch['created']
        proj = ch['project']
        invited = [rid for rid in ch['invited'] if rid is not None]
        if not invited:
            change_idx += 1
            continue
        # global stats at this timestamp (for z-score and overload)
        recent_30_counts = []
        for rid in eligible_reviewers:
            hist = reviewer_hist.get(rid, [])
            recent_30 = sum(1 for (ts, _) in hist if (created - ts).days <= 30 and ts < created)
            recent_30_counts.append(recent_30)
        gmean = mean(recent_30_counts) if recent_30_counts else 0.0
        gstd = pstdev(recent_30_counts) if len(recent_30_counts) > 1 else 0.0

        change_files = set(ch.get('file_paths') or [])
        change_tokens = _path_tokens(list(change_files))

        def pending_count(rid: int) -> int:
            cnt = 0
            for (c_ts, s_ts) in reviewer_participations.get(rid, []):
                if c_ts >= created:
                    break
                if s_ts and s_ts <= created:
                    continue
                cnt += 1
            return cnt

        positives = set(ch.get('positives') or [])
        for rid in invited:
            # compute features at invite time (created)
            hist = reviewer_hist.get(rid, [])
            past = [h for h in hist if h[0] < created]
            total = len(past)
            if total < cfg.min_total_reviews:
                # respect eligibility also for acceptance task to avoid extreme cold-start
                pass
            recent_7 = sum(1 for (ts, _) in past if (created - ts).days <= 7)
            recent_30 = sum(1 for (ts, _) in past if (created - ts).days <= 30)
            gap_days = (created - past[-1][0]).days if past else 999
            proj_prev_30 = sum(1 for (ts, p) in past if p == proj and (created - ts).days <= 30)
            proj_share = proj_prev_30 / recent_30 if recent_30 > 0 else 0.0
            active_flag = 1 if recent_30 > 0 else 0
            pend_cnt = pending_count(rid)
            night_events = sum(1 for (ts, _) in past if (created - ts).days <= 30 and ts.hour in (22, 23, 0, 1, 2, 3, 4, 5))
            night_share = night_events / recent_30 if recent_30 > 0 else 0.0
            overload_flag = 1 if (recent_30 >= (gmean + gstd)) and recent_30 > 0 else 0
            workload_deviation_z = (recent_30 - gmean) / (gstd + 1e-6) if gstd > 0 else 0.0
            match_off_specialty_flag = 1 if proj_prev_30 == 0 else 0
            off_specialty_recent_ratio = 1 - (proj_prev_30 / recent_30) if recent_30 > 0 else 0.0
            # TF-IDF (30d window)
            file_events = [s for (ts, s) in reviewer_file_hist.get(rid, []) if ts < created and (created - ts).days <= 30]
            past_files_30d: set[str] = set().union(*file_events) if file_events else set()
            if change_files and past_files_30d:
                file_tfidf = _tfidf_cosine(change_tokens, _path_tokens(list(past_files_30d)), idf_global)
            else:
                file_tfidf = 0.0
            state = {
                'reviewer_id': rid,
                'reviewer_recent_reviews_7d': recent_7,
                'reviewer_recent_reviews_30d': recent_30,
                'reviewer_gap_days': gap_days,
                'reviewer_total_reviews': total,
                'reviewer_proj_prev_reviews_30d': proj_prev_30,
                'reviewer_proj_share_30d': proj_share,
                'change_current_invited_cnt': len(invited),
                'reviewer_active_flag_30d': active_flag,
                'reviewer_pending_reviews': pend_cnt,
                'reviewer_night_activity_share_30d': night_share,
                'reviewer_overload_flag': overload_flag,
                'reviewer_workload_deviation_z': workload_deviation_z,
                'match_off_specialty_flag': match_off_specialty_flag,
                'off_specialty_recent_ratio': off_specialty_recent_ratio,
                'reviewer_file_tfidf_cosine_30d': file_tfidf,
            }
            label = 1 if rid in positives else 0
            samples.append({'state': state, 'label': label, 'ts': created.isoformat(), 'change_idx': change_idx})
        change_idx += 1
    return samples


@dataclass
class AcceptanceModelConfig:
    max_iter: int = 400
    l2: float = 1.0
    scale: bool = True


class AcceptanceModel:
    def __init__(self, cfg: AcceptanceModelConfig | None = None):
        self.cfg = cfg or AcceptanceModelConfig()
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        # Reuse the invitation ranking feature order for compatibility
        from .reviewer_invitation_ranking import InvitationRankingModel  # late import
        self.features = list(InvitationRankingModel().features)

    def _vec(self, s: Dict[str, Any]):
        return np.array([float(s.get(f, 0.0)) for f in self.features], dtype=float)

    def fit(self, samples: List[Dict[str, Any]]):
        X = np.vstack([self._vec(s['state']) for s in samples])
        y = np.array([s['label'] for s in samples])
        if self.cfg.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        pos_ratio = y.mean() if len(y) else 0.5
        w_pos = 0.5 / max(pos_ratio, 1e-6)
        w_neg = 0.5 / max(1 - pos_ratio, 1e-6)
        sample_weight = np.where(y == 1, w_pos, w_neg)
        self.model = LogisticRegression(max_iter=self.cfg.max_iter, C=1.0 / self.cfg.l2)
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, states: List[Dict[str, Any]]):
        if self.model is None:
            raise RuntimeError('model not fit')
        X = np.vstack([self._vec(s) for s in states])
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]


def temporal_split(samples: List[Dict[str, Any]], test_ratio: float = 0.2):
    ordered = sorted(samples, key=lambda s: s['ts'])
    n_test = max(1, int(len(ordered) * test_ratio))
    return ordered[:-n_test], ordered[-n_test:]


def evaluate_acceptance_after_invite(json_path: str | Path, base_cfg: AcceptanceBuildConfig | None = None):
    base_cfg = base_cfg or AcceptanceBuildConfig()
    samples = build_invitee_acceptance_samples(json_path, base_cfg)
    train, test = temporal_split(samples)
    model = AcceptanceModel()
    model.fit(train)
    y_true = [s['label'] for s in test]
    y_prob = model.predict_proba([s['state'] for s in test])
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    brier = brier_score_loss(y_true, y_prob) if y_true else float('nan')
    metrics = {
        'n_train': len(train),
        'n_test': len(test),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'brier': brier,
        'positive_rate_test': (sum(y_true) / len(y_true)) if y_true else 0.0,
    }
    if test and ('change_idx' in test[0]):
        group_ids = [s.get('change_idx', -1) for s in test]
        rankm = _ranking_metrics_by_group(y_true, list(y_prob), group_ids)
        metrics.update(rankm)
    return metrics, model, test, y_prob


__all__ = [
    'AcceptanceBuildConfig',
    'build_invitee_acceptance_samples',
    'AcceptanceModel',
    'AcceptanceModelConfig',
    'evaluate_acceptance_after_invite',
]
