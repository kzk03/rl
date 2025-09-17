"""Reviewer Invitation Ranking

Generates (change, reviewer) Cartesian candidates:
  Positive: reviewers who actually participated (left at least one non-auto message)
  Negative: sampled reviewers (not invited/participated) from active pool

Goal: Learn a scoring model to rank which reviewers to invite.

Negative sampling strategy:
  For each change with >=1 positive reviewer:
    - Candidate pool = reviewers with min_total_reviews & recent activity (last `recent_days`)
    - Exclude invited reviewers
    - Select up to max_neg_per_pos * num_pos combining:
        * Hard: top-K by recent_reviews_30d
        * Random: fill remainder

Features per (change, reviewer):
  reviewer_recent_reviews_7d
  reviewer_recent_reviews_30d
  reviewer_gap_days
  reviewer_total_reviews
  reviewer_proj_prev_reviews_30d
  reviewer_proj_share_30d (ratio)
  change_current_invited_cnt
  change_participated_pos_cnt (so far, leakage-free using full set? Here we use total positives -> slight lookahead; acceptable for baseline, TODO: incremental)
  reviewer_active_flag_30d (binary)

Temporal split by change creation time for evaluation.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
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


def _parse_dt(ts: str | None):
    from datetime import datetime, timezone
    if not ts:
        return None
    ts = ts.replace('T', ' ').replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _file_path_similarity_fuzzy(files1: List[str], files2: List[str]) -> float:
    """Compute a fuzzy file-path similarity between two file lists.

    Logic mirrors behavior_analysis.SimilarityCalculator._calculate_file_path_similarity:
    - If any exact file overlap exists, return min(1.0, jaccard_overlap * 2.0) to emphasize exact matches.
    - Otherwise, blend directory Jaccard (0.5), filename string similarity (0.3), and extension Jaccard (0.2).
    """
    if not files1 or not files2:
        return 0.0
    # Normalize into sets
    s1 = set(files1)
    s2 = set(files2)
    inter = len(s1 & s2)
    if inter > 0:
        union = len(s1 | s2)
        jacc = inter / union if union > 0 else 0.0
        return min(1.0, jacc * 2.0)
    # Directory-level Jaccard
    dirs1 = {str(Path(f).parent) for f in s1}
    dirs2 = {str(Path(f).parent) for f in s2}
    dir_sim = (len(dirs1 & dirs2) / len(dirs1 | dirs2)) if (dirs1 and dirs2) else 0.0
    # Max filename similarity (SequenceMatcher over stems)
    names1 = [Path(f).stem for f in s1]
    names2 = [Path(f).stem for f in s2]
    name_max = 0.0
    for n1 in names1:
        for n2 in names2:
            name_max = max(name_max, SequenceMatcher(None, n1, n2).ratio())
    # Extension Jaccard
    exts1 = {Path(f).suffix for f in s1}
    exts2 = {Path(f).suffix for f in s2}
    ext_sim = (len(exts1 & exts2) / len(exts1 | exts2)) if (exts1 and exts2) else 0.0
    # Blend
    sim = 0.5 * dir_sim + 0.3 * name_max + 0.2 * ext_sim
    return max(0.0, min(1.0, sim))


def _path_tokens(files: List[str]) -> List[str]:
    """Tokenize a list of file paths into lowercase tokens.

    Tokens include:
    - directory segments (split by '/')
    - filename stem segments (split by non-alnum and underscores/dashes)
    - extension token like ext:py
    """
    tokens: List[str] = []
    for p in files:
        try:
            s = str(p).strip()
            if not s or s in {'COMMIT_MSG', '/COMMIT_MSG'}:
                continue
            parts = s.split('/')
            # directory segments (exclude last element for filename unless there is no slash)
            if len(parts) > 1:
                for d in parts[:-1]:
                    d = d.strip().lower()
                    if d:
                        tokens.append(d)
            fname = parts[-1]
            # extension
            ext = Path(fname).suffix.lower()
            if ext:
                tokens.append(f"ext:{ext.lstrip('.')}")
            # stem tokens
            stem = Path(fname).stem.lower()
            # split by non-alphanumeric boundaries
            cur = ''
            for ch in stem:
                if ch.isalnum():
                    cur += ch
                else:
                    if cur:
                        tokens.append(cur)
                        cur = ''
            if cur:
                tokens.append(cur)
        except Exception:
            continue
    return tokens


def _tfidf_cosine(tokens_a: List[str], tokens_b: List[str], idf: Dict[str, float]) -> float:
    """Compute cosine similarity of TF-IDF vectors represented by token lists and global idf.

    Uses raw term frequency and provided IDF with smoothing already applied.
    """
    if not tokens_a or not tokens_b:
        return 0.0
    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)
    # compute dot product over intersection
    dot = 0.0
    for t in set(tf_a.keys()) & set(tf_b.keys()):
        w = idf.get(t, 1.0)
        dot += (tf_a[t] * w) * (tf_b[t] * w)
    # norms
    norm_a = 0.0
    for t, f in tf_a.items():
        w = idf.get(t, 1.0)
        norm_a += (f * w) ** 2
    norm_b = 0.0
    for t, f in tf_b.items():
        w = idf.get(t, 1.0)
        norm_b += (f * w) ** 2
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return float(dot / (norm_a ** 0.5 * norm_b ** 0.5))


@dataclass
class InvitationRankingBuildConfig:
    min_total_reviews: int = 20
    recent_days: int = 30
    max_neg_per_pos: int = 5
    hard_fraction: float = 0.5  # fraction of negatives from top recent reviewers
    leakage_free: bool = True   # if True, do NOT use final total positives (leakage)
    idf_mode: str = 'global'    # 'global' or 'recent'
    idf_recent_days: int = 90   # when idf_mode='recent', window size in days
    idf_windows: Tuple[int, ...] = ()  # optional multiple recent IDF windows, e.g., (30, 90)


def build_invitation_ranking_samples(
    json_path: str | Path,
    cfg: InvitationRankingBuildConfig | None = None,
) -> List[Dict[str, Any]]:
    cfg = cfg or InvitationRankingBuildConfig()
    data = json.loads(Path(json_path).read_text())
    from datetime import timedelta
    samples: List[Dict[str, Any]] = []
    # reviewer history: chronological list of (ts, project)
    reviewer_hist: Dict[int, List[Tuple[Any, str]]] = defaultdict(list)
    # aggregate reviewer last timestamp for recency filtering
    reviewer_last_ts: Dict[int, Any] = {}

    # First pass: collect positives, file paths, and basic change info
    raw_changes: List[Dict[str, Any]] = []
    # reviewer file history: chronological list of (ts, set(file_paths)) for participated changes
    reviewer_file_hist: Dict[int, List[Tuple[Any, set[str]]]] = defaultdict(list)

    def _extract_file_paths(ch_obj: Dict[str, Any]) -> List[str]:
        paths: set[str] = set()
        # Preferred: revisions[current_revision].files is a dict {path: {...}}
        cur_rev = ch_obj.get('current_revision')
        revisions = ch_obj.get('revisions') or {}
        if isinstance(revisions, dict) and cur_rev and cur_rev in revisions:
            files_dict = revisions[cur_rev].get('files') or {}
            if isinstance(files_dict, dict):
                for p in files_dict.keys():
                    if isinstance(p, str):
                        paths.add(p)
        # Fallback: iterate all revisions' files
        if not paths and isinstance(revisions, dict):
            for rev in revisions.values():
                files_dict = rev.get('files') or {}
                if isinstance(files_dict, dict):
                    for p in files_dict.keys():
                        if isinstance(p, str):
                            paths.add(p)
        # Legacy fallbacks (rare)
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
        for f in ch_obj.get('files', []) or []:  # rare case
            p = f.get('file') if isinstance(f, dict) else None
            if p:
                paths.add(p)
        # remove magic paths
        cleaned = {p for p in paths if p not in {'COMMIT_MSG', '/COMMIT_MSG'} and not (isinstance(p, str) and p.startswith('MergeList'))}
        return list(cleaned)[:500]  # cap
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
        # update history with participation events only (as signal of actual review)
        for rid in participated:
            reviewer_hist[rid].append((created, proj))
            reviewer_last_ts[rid] = created
            if file_paths:
                reviewer_file_hist[rid].append((created, set(file_paths)))

    # Sort histories for each reviewer
    for rid in reviewer_hist:
        reviewer_hist[rid].sort(key=lambda x: x[0])
    for rid in reviewer_file_hist:
        reviewer_file_hist[rid].sort(key=lambda x: x[0])

    # Build tokens per change and global IDF over all changes (downweight ubiquitous tokens like 'src', 'test', etc.)
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
        # smoothed idf
        idf_global[t] = float(np.log((N_docs + 1) / (dfi + 1)) + 1.0)

    # Determine global reviewer pool satisfying min_total_reviews
    eligible_reviewers = {rid for rid, h in reviewer_hist.items() if len(h) >= cfg.min_total_reviews}

    # Build participation event index for leakage-free pending calculation
    reviewer_participations: Dict[int, List[Tuple[Any, Any]]] = defaultdict(list)
    for ch in raw_changes:
        for rid in ch['positives']:
            reviewer_participations[rid].append((ch['created'], ch['submitted']))
    for rid in reviewer_participations:
        reviewer_participations[rid].sort(key=lambda x: x[0])

    from statistics import mean, pstdev

    # Prepare ordered changes for temporal processing
    ordered_changes = sorted(raw_changes, key=lambda c: c['created'])

    # If using recent IDF, maintain a sliding window of prior changes' tokens
    window = deque()  # elements: (created_ts, tokens_set)
    df_window: Dict[str, int] = defaultdict(int)
    # Multi-window structures (not used when only one TF-IDF is needed)
    windows_map: Dict[int, deque] = {}
    df_windows_map: Dict[int, Dict[str, int]] = {}

    change_counter = 0
    for ch in ordered_changes:
        positives = [rid for rid in ch['positives'] if rid in eligible_reviewers]
        if not positives:
            continue  # skip changes without positive eligible reviewers
        created = ch['created']
        proj = ch['project']
        submitted = ch['submitted']
        change_files = set(ch.get('file_paths') or [])
        change_tokens = _path_tokens(list(change_files))

    # Select IDF to use for this change
        if cfg.idf_mode == 'recent':
            # shrink window to [created - idf_recent_days, created)
            horizon = timedelta(days=cfg.idf_recent_days)
            while window and (created - window[0][0]) > horizon:
                _, toks_old = window.popleft()
                for t in toks_old:
                    df_window[t] -= 1
                    if df_window[t] <= 0:
                        df_window.pop(t, None)
            Nw = len(window)
            if Nw > 0:
                idf_current: Dict[str, float] = {}
                for t, dfi in df_window.items():
                    idf_current[t] = float(np.log((Nw + 1) / (dfi + 1)) + 1.0)
            else:
                idf_current = idf_global
        else:
            idf_current = idf_global

    # Per-window IDFs are disabled (only one TF-IDF feature is kept)

        # Global recent_30d counts for fairness/stress metrics
        # (naive computation; acceptable for baseline size)
        recent_30_counts = []
        recent_30_map: Dict[int, int] = {}
        for rid in eligible_reviewers:
            hist = reviewer_hist[rid]
            recent_30 = sum(1 for (ts, _) in hist if (created - ts).days <= 30 and ts < created)
            recent_30_counts.append(recent_30)
            recent_30_map[rid] = recent_30
        global_mean = mean(recent_30_counts) if recent_30_counts else 0.0
        global_std = pstdev(recent_30_counts) if len(recent_30_counts) > 1 else 0.0
        top5 = sorted(recent_30_counts, reverse=True)[:5]
        bus_factor_top5_share = (sum(top5) / sum(recent_30_counts)) if sum(recent_30_counts) > 0 else 0.0

        # Helper: count open (pending) participations strictly started before 'created' and not yet submitted
        def pending_count(rid: int):
            cnt = 0
            for (c_ts, s_ts) in reviewer_participations.get(rid, []):
                if c_ts >= created:
                    break  # participations are time-sorted
                if s_ts and s_ts <= created:
                    continue  # already closed
                cnt += 1
            return cnt

        # compute positive samples
        pos_states = []
        # leakage-free observed positives at (approx) invitation time (change creation)
        observed_pos_cnt = 0 if cfg.leakage_free else len(positives)
        for rid in positives:
            hist = reviewer_hist[rid]
            past = [h for h in hist if h[0] < created]
            recent_7 = sum(1 for (ts, _) in past if (created - ts).days <= 7)
            recent_30 = sum(1 for (ts, _) in past if (created - ts).days <= 30)
            gap_days = (created - past[-1][0]).days if past else 999
            total = len(past)
            proj_prev_30 = sum(1 for (ts, p) in past if p == proj and (created - ts).days <= 30)
            proj_share = proj_prev_30 / recent_30 if recent_30 > 0 else 0.0
            active_flag = 1 if recent_30 > 0 else 0
            # New features
            pending_cnt = pending_count(rid)
            night_events = sum(1 for (ts, _) in past if (created - ts).days <= 30 and ts.hour in (22,23,0,1,2,3,4,5))
            night_activity_share = night_events / recent_30 if recent_30 > 0 else 0.0
            overload_flag = 1 if (recent_30 >= (global_mean + global_std)) and recent_30 > 0 else 0
            workload_deviation_z = (recent_30 - global_mean) / (global_std + 1e-6) if global_std > 0 else 0.0
            match_off_specialty_flag = 1 if proj_prev_30 == 0 else 0
            off_specialty_recent_ratio = 1 - (proj_prev_30 / recent_30) if recent_30 > 0 else 0.0
            # File path specialization (fuzzy similarity over 30d)
            file_events = [s for (ts, s) in reviewer_file_hist.get(rid, []) if ts < created and (created - ts).days <= 30]
            past_files_30d: set[str] = set().union(*file_events) if file_events else set()
            if change_files and past_files_30d:
                file_tfidf = _tfidf_cosine(change_tokens, _path_tokens(list(past_files_30d)), idf_current)
            else:
                file_tfidf = 0.0
            # Removed placeholder similarity features (will add real ones later)
            state = {
                'reviewer_id': rid,
                'reviewer_recent_reviews_7d': recent_7,
                'reviewer_recent_reviews_30d': recent_30,
                'reviewer_gap_days': gap_days,
                'reviewer_total_reviews': total,
                'reviewer_proj_prev_reviews_30d': proj_prev_30,
                'reviewer_proj_share_30d': proj_share,
                'change_current_invited_cnt': len(ch['invited']),
                'reviewer_active_flag_30d': active_flag,
                'reviewer_pending_reviews': pending_cnt,
                'reviewer_night_activity_share_30d': night_activity_share,
                'reviewer_overload_flag': overload_flag,
                'reviewer_workload_deviation_z': workload_deviation_z,
                'match_off_specialty_flag': match_off_specialty_flag,
                'off_specialty_recent_ratio': off_specialty_recent_ratio,
                'reviewer_file_tfidf_cosine_30d': file_tfidf,
            }
            pos_states.append(state)
        # Negative sampling
        # Active pool: eligible reviewers with last activity within recent_days and not invited
        horizon = timedelta(days=cfg.recent_days)
        active_pool = []
        for rid in eligible_reviewers:
            if rid in ch['invited']:
                continue
            last_ts = reviewer_last_ts.get(rid)
            if not last_ts:
                continue
            if (created - last_ts) > horizon:
                continue
            active_pool.append(rid)
        if not active_pool:
            # fallback: broaden pool ignoring recency
            active_pool = [rid for rid in eligible_reviewers if rid not in ch['invited']]
        if not active_pool:
            # cannot form negatives
            continue
        needed = cfg.max_neg_per_pos * len(positives)
        # Hard negatives = top by recent_reviews_30d (recomputed quickly)
        hard_candidates = []
        for rid in active_pool:
            hist = reviewer_hist[rid]
            recent_30 = sum(1 for (ts, _) in hist if (created - ts).days <= 30)
            hard_candidates.append((recent_30, rid))
        hard_candidates.sort(reverse=True)
        n_hard = max(1, int(needed * cfg.hard_fraction))
        hard = [rid for _, rid in hard_candidates[:n_hard]]
        remaining_pool = [rid for rid in active_pool if rid not in hard]
        random.shuffle(remaining_pool)
        neg_ids = (hard + remaining_pool)[:needed]
        neg_states = []
        for rid in neg_ids:
            hist = reviewer_hist[rid]
            past = [h for h in hist if h[0] < created]
            recent_7 = sum(1 for (ts, _) in past if (created - ts).days <= 7)
            recent_30 = sum(1 for (ts, _) in past if (created - ts).days <= 30)
            gap_days = (created - past[-1][0]).days if past else 999
            total = len(past)
            proj_prev_30 = sum(1 for (ts, p) in past if p == proj and (created - ts).days <= 30)
            proj_share = proj_prev_30 / recent_30 if recent_30 > 0 else 0.0
            active_flag = 1 if recent_30 > 0 else 0
            pending_cnt = pending_count(rid)
            night_events = sum(1 for (ts, _) in past if (created - ts).days <= 30 and ts.hour in (22,23,0,1,2,3,4,5))
            night_activity_share = night_events / recent_30 if recent_30 > 0 else 0.0
            overload_flag = 1 if (recent_30 >= (global_mean + global_std)) and recent_30 > 0 else 0
            workload_deviation_z = (recent_30 - global_mean) / (global_std + 1e-6) if global_std > 0 else 0.0
            match_off_specialty_flag = 1 if proj_prev_30 == 0 else 0
            off_specialty_recent_ratio = 1 - (proj_prev_30 / recent_30) if recent_30 > 0 else 0.0
            file_events = [s for (ts, s) in reviewer_file_hist.get(rid, []) if ts < created and (created - ts).days <= 30]
            past_files_30d = set().union(*file_events) if file_events else set()
            if change_files and past_files_30d:
                file_tfidf = _tfidf_cosine(change_tokens, _path_tokens(list(past_files_30d)), idf_current)
            else:
                file_tfidf = 0.0
            neg_states.append({
                'reviewer_id': rid,
                'reviewer_recent_reviews_7d': recent_7,
                'reviewer_recent_reviews_30d': recent_30,
                'reviewer_gap_days': gap_days,
                'reviewer_total_reviews': total,
                'reviewer_proj_prev_reviews_30d': proj_prev_30,
                'reviewer_proj_share_30d': proj_share,
                'change_current_invited_cnt': len(ch['invited']),
                'reviewer_active_flag_30d': active_flag,
                'reviewer_pending_reviews': pending_cnt,
                'reviewer_night_activity_share_30d': night_activity_share,
                'reviewer_overload_flag': overload_flag,
                'reviewer_workload_deviation_z': workload_deviation_z,
                'match_off_specialty_flag': match_off_specialty_flag,
                'off_specialty_recent_ratio': off_specialty_recent_ratio,
                'reviewer_file_tfidf_cosine_30d': file_tfidf,
            })
        # Append labeled samples
        for st in pos_states:
            samples.append({'state': st, 'label': 1, 'ts': created.isoformat(), 'change_idx': change_counter})
        for st in neg_states:
            samples.append({'state': st, 'label': 0, 'ts': created.isoformat(), 'change_idx': change_counter})

        # Update recent IDF window AFTER using idf for current change
        if cfg.idf_mode == 'recent':
            toks_cur = ch.get('tokens') or set()
            window.append((created, toks_cur))
            for t in toks_cur:
                df_window[t] += 1
    # Multi-window updates are disabled
        
    change_counter += 1
    # No mutation required: pending counts derived from full historical index up to current time
    return samples


@dataclass
class InvitationRankingModelConfig:
    max_iter: int = 400
    l2: float = 1.0
    scale: bool = True


class InvitationRankingModel:
    def __init__(self, cfg: InvitationRankingModelConfig | None = None):
        self.cfg = cfg or InvitationRankingModelConfig()
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self.features = [
            'reviewer_recent_reviews_7d',
            'reviewer_recent_reviews_30d',
            'reviewer_gap_days',
            'reviewer_total_reviews',
            'reviewer_proj_prev_reviews_30d',
            'reviewer_proj_share_30d',
            'change_current_invited_cnt',
            'reviewer_active_flag_30d',
            # Refined leakage-free feature set
            'reviewer_pending_reviews',
            'reviewer_night_activity_share_30d',
            'reviewer_overload_flag',
            'reviewer_workload_deviation_z',
            'match_off_specialty_flag',
            'off_specialty_recent_ratio',
            'reviewer_file_tfidf_cosine_30d',
        ]

    def _vec(self, s: Dict[str, Any]):
        return np.array([float(s.get(f, 0.0)) for f in self.features], dtype=float)

    def fit(self, samples: List[Dict[str, Any]]):
        X = np.vstack([self._vec(s['state']) for s in samples])
        y = np.array([s['label'] for s in samples])
        if self.cfg.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        # class imbalance weighting
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


def _ranking_metrics_by_group(y_true: List[int], y_prob: List[float], group_ids: List[int], ks: List[int] = [1,3,5,10]):
    from math import log2

    # Group by change_idx
    groups: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for yt, yp, gid in zip(y_true, y_prob, group_ids):
        groups[gid].append((yt, yp))
    # Compute metrics per group
    recall_at = {k: [] for k in ks}
    map_at = {k: [] for k in ks}
    ndcg_at = {k: [] for k in ks}
    for g, items in groups.items():
        # sort by prob desc
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
            recall_at[k].append(sum(topk)/pos_count)
            # MAP@K
            ap = 0.0
            hit = 0
            for i, lab in enumerate(topk, start=1):
                if lab == 1:
                    hit += 1
                    ap += hit / i
            map_at[k].append(ap / min(pos_count, k))
            # NDCG@K (binary gains)
            dcg = 0.0
            for i, lab in enumerate(topk, start=1):
                if lab == 1:
                    dcg += 1.0 / log2(i + 1)
            ideal_hits = min(pos_count, k)
            idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1)) if ideal_hits > 0 else 0.0
            ndcg_at[k].append(dcg / idcg if idcg > 0 else 0.0)
    # Aggregate (macro average over groups)
    out = {}
    for k in ks:
        def avg(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0
        out[f'recall@{k}'] = avg(recall_at[k])
        out[f'map@{k}'] = avg(map_at[k])
        out[f'ndcg@{k}'] = avg(ndcg_at[k])
    return out


def evaluate_invitation_ranking(samples: List[Dict[str, Any]]):
    train, test = temporal_split(samples)
    model = InvitationRankingModel()
    model.fit(train)
    y_true = [s['label'] for s in test]
    y_prob = model.predict_proba([s['state'] for s in test])
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    brier = brier_score_loss(y_true, y_prob)
    metrics = {
        'n_train': len(train),
        'n_test': len(test),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'brier': brier,
        'positive_rate_test': sum(y_true) / len(y_true) if y_true else 0.0,
    }
    # Ranking metrics by change_idx if present
    if test and ('change_idx' in test[0]):
        group_ids = [s.get('change_idx', -1) for s in test]
        rankm = _ranking_metrics_by_group(y_true, y_prob, group_ids)
        metrics.update(rankm)
    return metrics, model, test, y_prob


class PreferenceRankingModel:
    """Pairwise preference learning: P(pos > neg) = sigmoid(w^T (phi_pos - phi_neg))."""
    def __init__(self, base_features: List[str]):
        self.features = base_features
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None

    def _vec(self, s: Dict[str, Any]):
        return np.array([float(s.get(f, 0.0)) for f in self.features], dtype=float)

    def fit_pairs(self, pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]):
        Xdiff = []
        y = []
        # Positive direction: pos - neg -> label 1
        for sp, sn in pairs:
            Xdiff.append(self._vec(sp) - self._vec(sn))
            y.append(1)
        # Reverse direction for negative class: neg - pos -> label 0
        for sp, sn in pairs:
            Xdiff.append(self._vec(sn) - self._vec(sp))
            y.append(0)
        X = np.vstack(Xdiff) if Xdiff else np.zeros((0, len(self.features)))
        y = np.array(y)
        if X.shape[0] == 0:
            raise RuntimeError('no pairs to train on')
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = LogisticRegression(max_iter=400)
        self.model.fit(Xs, y)

    def score(self, states: List[Dict[str, Any]]):
        # linear score proportional to preference: w^T x (in standardized space)
        if self.model is None or self.scaler is None:
            raise RuntimeError('model not fit')
        X = np.vstack([self._vec(s) for s in states])
        Xs = self.scaler.transform(X)
        coefs = self.model.coef_[0]
        return (Xs @ coefs)


def evaluate_invitation_pairwise(samples: List[Dict[str, Any]]):
    # temporal split
    train, test = temporal_split(samples)
    # build pairs within each training change_idx: (positive, negative)
    train_by_gid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for s in train:
        train_by_gid[s.get('change_idx', -1)].append(s)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    neg_per_pos = 5  # sample up to 5 negatives per positive to keep training tractable
    for gid, items in train_by_gid.items():
        pos_states = [it['state'] for it in items if it['label'] == 1]
        neg_states = [it['state'] for it in items if it['label'] == 0]
        if not neg_states or not pos_states:
            continue
        # sample without replacement per positive
        for sp in pos_states:
            if len(neg_states) <= neg_per_pos:
                sampled = neg_states
            else:
                # simple random sample
                idxs = np.random.choice(len(neg_states), size=neg_per_pos, replace=False)
                sampled = [neg_states[i] for i in idxs]
            for sn in sampled:
                pairs.append((sp, sn))
    # train
    base_features = InvitationRankingModel().features
    model = PreferenceRankingModel(base_features)
    if not pairs:
        raise RuntimeError('no training pairs (check data)')
    model.fit_pairs(pairs)
    # score test and compute metrics
    y_true = [s['label'] for s in test]
    group_ids = [s.get('change_idx', -1) for s in test]
    # Convert linear scores to probabilities via min-max per group (for Brier/AUC proxy)
    # But for ranking metrics we use scores directly.
    # Build ranking metrics per group
    # For AUC/Brier, approximate with logistic-prob by mapping linear scores through sigmoid centered at 0
    # Score per item
    scores = model.score([s['state'] for s in test])
    # Use NumPy sigmoid to map scores to (0,1)
    y_prob = 1.0 / (1.0 + np.exp(-scores))
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    brier = brier_score_loss(y_true, y_prob)
    metrics = {
        'n_train': len(train),
        'n_test': len(test),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'brier': brier,
        'positive_rate_test': sum(y_true) / len(y_true) if y_true else 0.0,
    }
    rankm = _ranking_metrics_by_group(y_true, list(y_prob), group_ids)
    metrics.update(rankm)
    return metrics, model, test, y_prob


__all__ = [
    'build_invitation_ranking_samples',
    'evaluate_invitation_ranking',
    'evaluate_invitation_pairwise',
    'InvitationRankingBuildConfig',
]

# =========================
# Level-1 IRL: Conditional Logit (softmax over candidates per change)
# =========================

@dataclass
class ConditionalLogitConfig:
    max_iter: int = 200
    lr: float = 0.1
    l2: float = 1.0
    scale: bool = True
    max_candidates: int = 50  # per change (including all invited + sampled others)
    hard_fraction: float = 0.7  # fraction of non-invited from top recent activity


def build_invitation_groups_irl(
    json_path: str | Path,
    base_cfg: InvitationRankingBuildConfig | None = None,
    irl_cfg: ConditionalLogitConfig | None = None,
):
    """Build per-change candidate groups for Conditional Logit IRL.

    Each group contains candidates with features and an 'invited' flag (1/0).
    Candidate pool per change:
      - Always include all invited reviewers
      - Add other active eligible reviewers (recent_days)
      - Cap size to irl_cfg.max_candidates via top recent activity (hard) + random
    """
    base_cfg = base_cfg or InvitationRankingBuildConfig()
    irl_cfg = irl_cfg or ConditionalLogitConfig()
    data = json.loads(Path(json_path).read_text())
    from datetime import timedelta

    # Histories reused from pointwise builder
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

    # First pass: build participation and file history (for features)
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
        # update reviewer participation history for recency/activity features
        for rid in participated:
            reviewer_hist[rid].append((created, proj))
            reviewer_last_ts[rid] = created
            if file_paths:
                reviewer_file_hist[rid].append((created, set(file_paths)))

    for rid in reviewer_hist:
        reviewer_hist[rid].sort(key=lambda x: x[0])
    for rid in reviewer_file_hist:
        reviewer_file_hist[rid].sort(key=lambda x: x[0])

    # Token/IDF for TF-IDF feature
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

    # Eligible pool by total reviews (using participation history as proxy)
    eligible_reviewers = {rid for rid, h in reviewer_hist.items() if len(h) >= base_cfg.min_total_reviews}

    # Build participation event index for leakage-free pending calculation (per reviewer)
    # Interval = (change_created_ts, change_submitted_ts). Sorted by start ts.
    reviewer_participations: Dict[int, List[Tuple[Any, Any]]] = defaultdict(list)
    for ch in raw_changes:
        for rid in ch['positives']:
            reviewer_participations[rid].append((ch['created'], ch['submitted']))
    for rid in reviewer_participations:
        reviewer_participations[rid].sort(key=lambda x: x[0])

    from statistics import mean, pstdev
    ordered_changes = sorted(raw_changes, key=lambda c: c['created'])
    groups = []
    group_id = 0
    for ch in ordered_changes:
        created = ch['created']
        proj = ch['project']
        change_files = set(ch.get('file_paths') or [])
        change_tokens = _path_tokens(list(change_files))

        # Helper: count open (pending) participations strictly started before 'created' and not yet submitted
        def pending_count(rid: int) -> int:
            cnt = 0
            for (c_ts, s_ts) in reviewer_participations.get(rid, []):
                if c_ts >= created:
                    break  # intervals are sorted by start
                if s_ts and s_ts <= created:
                    continue  # already closed
                cnt += 1
            return cnt

        # active pool
        horizon = timedelta(days=base_cfg.recent_days)
        active_pool = []
        for rid in eligible_reviewers:
            last_ts = reviewer_last_ts.get(rid)
            if last_ts and (created - last_ts) <= horizon:
                active_pool.append(rid)
        # always include invited
        invited = [rid for rid in ch['invited'] if rid is not None]
        for rid in invited:
            if rid not in eligible_reviewers:
                # allow invited even if below threshold
                eligible_reviewers.add(rid)
                # ensure at least a stub last_ts to pass activity filter next time
                if rid not in reviewer_last_ts:
                    reviewer_last_ts[rid] = created
        pool = list({*active_pool, *invited})
        if not pool:
            continue

        # compute global stats for features
        recent_30_counts = []
        for rid in eligible_reviewers:
            hist = reviewer_hist.get(rid, [])
            recent_30 = sum(1 for (ts, _) in hist if (created - ts).days <= 30 and ts < created)
            recent_30_counts.append(recent_30)
        gmean = mean(recent_30_counts) if recent_30_counts else 0.0
        gstd = pstdev(recent_30_counts) if len(recent_30_counts) > 1 else 0.0
        top5 = sorted(recent_30_counts, reverse=True)[:5]
        bus_factor_top5_share = (sum(top5) / sum(recent_30_counts)) if sum(recent_30_counts) > 0 else 0.0

        # rank pool by hardness (recent_30) for selection
        hard_candidates = []
        for rid in pool:
            hist = reviewer_hist.get(rid, [])
            recent_30 = sum(1 for (ts, _) in hist if (created - ts).days <= 30 and ts < created)
            hard_candidates.append((recent_30, rid))
        hard_candidates.sort(reverse=True)
        # ensure all invited are kept
        invited_set = set(invited)
        # select others up to max_candidates
        max_c = max(irl_cfg.max_candidates, len(invited))
        # pick hard fraction from top
        n_other_needed = max_c - len(invited)
        others = [rid for _, rid in hard_candidates if rid not in invited_set]
        n_hard = int(n_other_needed * irl_cfg.hard_fraction)
        hard = others[:n_hard]
        rest = others[n_hard:]
        random.shuffle(rest)
        selected = list(invited) + hard + rest[: max(0, n_other_needed - len(hard))]

        # build candidate feature rows
        candidates = []
        for rid in selected:
            hist = reviewer_hist.get(rid, [])
            past = [h for h in hist if h[0] < created]
            recent_7 = sum(1 for (ts, _) in past if (created - ts).days <= 7)
            recent_30 = sum(1 for (ts, _) in past if (created - ts).days <= 30)
            gap_days = (created - past[-1][0]).days if past else 999
            total = len(past)
            proj_prev_30 = sum(1 for (ts, p) in past if p == proj and (created - ts).days <= 30)
            proj_share = proj_prev_30 / recent_30 if recent_30 > 0 else 0.0
            active_flag = 1 if recent_30 > 0 else 0
            pending_cnt = pending_count(rid)
            night_events = sum(1 for (ts, _) in past if (created - ts).days <= 30 and ts.hour in (22,23,0,1,2,3,4,5))
            night_activity_share = night_events / recent_30 if recent_30 > 0 else 0.0
            overload_flag = 1 if (recent_30 >= (gmean + gstd)) and recent_30 > 0 else 0
            workload_deviation_z = (recent_30 - gmean) / (gstd + 1e-6) if gstd > 0 else 0.0
            match_off_specialty_flag = 1 if proj_prev_30 == 0 else 0
            off_specialty_recent_ratio = 1 - (proj_prev_30 / recent_30) if recent_30 > 0 else 0.0
            # file similarity (global idf)
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
                'reviewer_pending_reviews': pending_cnt,
                'reviewer_night_activity_share_30d': night_activity_share,
                'reviewer_overload_flag': overload_flag,
                'reviewer_workload_deviation_z': workload_deviation_z,
                'match_off_specialty_flag': match_off_specialty_flag,
                'off_specialty_recent_ratio': off_specialty_recent_ratio,
                'reviewer_file_tfidf_cosine_30d': file_tfidf,
            }
            candidates.append({'state': state, 'invited': 1 if rid in invited_set else 0})
        if candidates:
            groups.append({'ts': created.isoformat(), 'candidates': candidates, 'change_idx': group_id})
            group_id += 1
    return groups


class ConditionalLogitModel:
    def __init__(self, features: List[str], cfg: ConditionalLogitConfig | None = None):
        self.features = features
        self.cfg = cfg or ConditionalLogitConfig()
        self.scaler: StandardScaler | None = None
        self.theta: np.ndarray | None = None  # includes intercept as last dim

    def _vec(self, s: Dict[str, Any]):
        return np.array([float(s.get(f, 0.0)) for f in self.features], dtype=float)

    def fit(self, groups: List[Dict[str, Any]]):
        # Build design matrix over all training candidates
        X_list = []
        for g in groups:
            for c in g['candidates']:
                X_list.append(self._vec(c['state']))
        X = np.vstack(X_list) if X_list else np.zeros((0, len(self.features)))
        if self.cfg.scale and X.shape[0] > 0:
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
        else:
            Xs = X
        d = Xs.shape[1]
        # Prepare per-group matrices (with intercept)
        groups_X: List[np.ndarray] = []
        groups_Y: List[np.ndarray] = []
        offset = 0
        for g in groups:
            n = len(g['candidates'])
            Xg = Xs[offset: offset + n, :]
            yg = np.array([int(c['invited']) for c in g['candidates']], dtype=float)
            # append bias column
            Xg_b = np.hstack([Xg, np.ones((n, 1), dtype=float)])
            groups_X.append(Xg_b)
            groups_Y.append(yg)
            offset += n
        # init theta (d + 1)
        self.theta = np.zeros(d + 1, dtype=float)

        def logsumexp(a: np.ndarray) -> float:
            m = float(np.max(a))
            return m + float(np.log(np.sum(np.exp(a - m))))

        for it in range(self.cfg.max_iter):
            grad = np.zeros_like(self.theta)
            total_loss = 0.0
            for Xg, yg in zip(groups_X, groups_Y):
                scores = Xg @ self.theta  # (n,)
                lse = logsumexp(scores)
                # softmax
                probs = np.exp(scores - lse)
                ysum = float(np.sum(yg))
                # loss contribution
                total_loss += -float(yg @ scores) + ysum * lse
                # gradient: -X^T y + |P| * X^T softmax
                grad += -(Xg.T @ yg) + ysum * (Xg.T @ probs)
            # L2 reg
            if self.cfg.l2 > 0:
                total_loss += 0.5 * self.cfg.l2 * float(self.theta @ self.theta)
                grad += self.cfg.l2 * self.theta
            # step
            self.theta -= self.cfg.lr * grad / max(1, len(groups_X))
            # Optional: could add early stopping using loss delta

    def predict_group_probs(self, groups: List[Dict[str, Any]]):
        if self.theta is None:
            raise RuntimeError('model not fit')
        # Build X matrices per group (apply scaler)
        y_true_all: List[int] = []
        y_prob_all: List[float] = []
        group_ids: List[int] = []
        flat_items: List[Dict[str, Any]] = []
        # assemble candidates
        X_list = []
        group_sizes = []
        for g in groups:
            n = len(g['candidates'])
            group_sizes.append(n)
            for c in g['candidates']:
                X_list.append(self._vec(c['state']))
        X = np.vstack(X_list) if X_list else np.zeros((0, len(self.features)))
        if self.scaler is not None and X.shape[0] > 0:
            Xs = self.scaler.transform(X)
        else:
            Xs = X
        # add bias
        Xs_b = np.hstack([Xs, np.ones((Xs.shape[0], 1), dtype=float)]) if Xs.size else Xs
        # iterate groups
        start = 0
        gid = 0
        for g in groups:
            n = group_sizes[gid]
            Xg = Xs_b[start: start + n, :]
            scores = Xg @ self.theta
            # softmax
            m = float(np.max(scores))
            exps = np.exp(scores - m)
            probs = exps / float(np.sum(exps)) if np.sum(exps) > 0 else np.full(n, 1.0 / max(1, n))
            for i, c in enumerate(g['candidates']):
                y_true_all.append(int(c['invited']))
                y_prob_all.append(float(probs[i]))
                group_ids.append(int(g.get('change_idx', gid)))
                flat_items.append({'state': c['state'], 'label': int(c['invited']), 'ts': g['ts']})
            start += n
            gid += 1
        return y_true_all, y_prob_all, group_ids, flat_items


def evaluate_invitation_irl(json_path: str | Path, base_cfg: InvitationRankingBuildConfig | None = None, irl_cfg: ConditionalLogitConfig | None = None):
    base_cfg = base_cfg or InvitationRankingBuildConfig()
    irl_cfg = irl_cfg or ConditionalLogitConfig()
    groups = build_invitation_groups_irl(json_path, base_cfg, irl_cfg)
    # temporal split by ts
    ordered = sorted(groups, key=lambda g: g['ts'])
    n_test = max(1, int(len(ordered) * 0.2))
    train_groups = ordered[:-n_test]
    test_groups = ordered[-n_test:]
    # fit
    features = InvitationRankingModel().features
    model = ConditionalLogitModel(features, irl_cfg)
    if not train_groups:
        raise RuntimeError('no training groups')
    model.fit(train_groups)
    # predict
    y_true, y_prob, group_ids, flat_items = model.predict_group_probs(test_groups)
    # metrics
    from sklearn.metrics import brier_score_loss, roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    brier = brier_score_loss(y_true, y_prob) if y_true else float('nan')
    # NLL per invited
    # compute per group softmax already in predict; recompute NLL quickly
    pos_count = max(1, sum(y_true))
    from math import log
    nll = -sum(log(max(1e-12, p)) for y, p in zip(y_true, y_prob) if y == 1) / pos_count
    rankm = _ranking_metrics_by_group(y_true, y_prob, group_ids)
    metrics = {
        'n_groups_train': len(train_groups),
        'n_groups_test': len(test_groups),
        'n_candidates_test': len(y_true),
        'auc': auc,
        'brier': brier,
        'nll_per_invited': nll,
        **rankm,
    }
    # attach a light wrapper so the script can export weights
    class _IRLModelWrapper:
        def __init__(self, features, scaler, theta):
            self.features = features
            self.scaler = scaler
            self.theta = theta  # includes intercept at end
            self.model = None  # for script branching
    wrapped = _IRLModelWrapper(features, model.scaler, model.theta)
    return metrics, wrapped, flat_items, np.array(y_prob)


# add to exports
__all__.extend(['evaluate_invitation_irl', 'build_invitation_groups_irl', 'ConditionalLogitConfig'])

# =========================
# Level-2 IRL: Plackett–Luce Top-k (sequential softmax without replacement)
# =========================

class PlackettLuceIRLModel:
    def __init__(self, features: List[str], cfg: ConditionalLogitConfig | None = None):
        self.features = features
        self.cfg = cfg or ConditionalLogitConfig()
        self.scaler: StandardScaler | None = None
        self.theta: np.ndarray | None = None  # includes intercept as last dim

    def _vec(self, s: Dict[str, Any]):
        return np.array([float(s.get(f, 0.0)) for f in self.features], dtype=float)

    def fit(self, groups: List[Dict[str, Any]]):
        # stack all candidates to fit scaler
        X_list = []
        for g in groups:
            for c in g['candidates']:
                X_list.append(self._vec(c['state']))
        X = np.vstack(X_list) if X_list else np.zeros((0, len(self.features)))
        if self.cfg.scale and X.shape[0] > 0:
            self.scaler = StandardScaler()
            Xs_all = self.scaler.fit_transform(X)
        else:
            Xs_all = X
        # build per-group X and y
        groups_X: List[np.ndarray] = []
        invite_indices: List[List[int]] = []
        offset = 0
        for g in groups:
            n = len(g['candidates'])
            Xg = Xs_all[offset: offset + n, :]
            invited_idx = [i for i, c in enumerate(g['candidates']) if int(c['invited']) == 1]
            # append bias
            Xg_b = np.hstack([Xg, np.ones((n, 1), dtype=float)])
            groups_X.append(Xg_b)
            invite_indices.append(invited_idx)
            offset += n
        d = groups_X[0].shape[1] if groups_X else (len(self.features) + 1)
        self.theta = np.zeros(d, dtype=float)

        def logsumexp(a: np.ndarray) -> float:
            m = float(np.max(a))
            return m + float(np.log(np.sum(np.exp(a - m))))

        for it in range(self.cfg.max_iter):
            grad = np.zeros_like(self.theta)
            total_loss = 0.0
            for Xg, idxs in zip(groups_X, invite_indices):
                if not idxs:
                    continue
                n = Xg.shape[0]
                scores_all = Xg @ self.theta  # (n,)
                # deterministic pseudo-order: ascending invited index order
                remaining = list(range(n))
                for j in idxs:
                    # loss term for selecting j among remaining
                    rem_idx = remaining
                    s_rem = scores_all[rem_idx]
                    lse = logsumexp(s_rem)
                    total_loss += -(scores_all[j]) + lse
                    # gradient
                    # softmax over remaining
                    probs = np.exp(s_rem - lse)
                    # accumulate gradient: -x_j + sum_i p_i x_i
                    grad += -(Xg[j]) + (Xg[rem_idx].T @ probs)
                    # remove selected j from remaining (without replacement)
                    if j in remaining:
                        remaining.remove(j)
            # L2
            if self.cfg.l2 > 0:
                total_loss += 0.5 * self.cfg.l2 * float(self.theta @ self.theta)
                grad += self.cfg.l2 * self.theta
            # step
            self.theta -= self.cfg.lr * grad / max(1, len(groups_X))

    def predict_group_probs(self, groups: List[Dict[str, Any]]):
        if self.theta is None:
            raise RuntimeError('model not fit')
        # compute initial-step softmax probs per group for ranking
        y_true_all: List[int] = []
        y_prob_all: List[float] = []
        group_ids: List[int] = []
        flat_items: List[Dict[str, Any]] = []
        for gid, g in enumerate(groups):
            X_list = [self._vec(c['state']) for c in g['candidates']]
            X = np.vstack(X_list) if X_list else np.zeros((0, len(self.features)))
            if self.scaler is not None and X.shape[0] > 0:
                Xs = self.scaler.transform(X)
            else:
                Xs = X
            Xs_b = np.hstack([Xs, np.ones((Xs.shape[0], 1), dtype=float)]) if Xs.size else Xs
            scores = Xs_b @ self.theta
            m = float(np.max(scores)) if scores.size else 0.0
            exps = np.exp(scores - m) if scores.size else np.array([])
            probs = exps / float(np.sum(exps)) if exps.size and float(np.sum(exps)) > 0 else (np.full(Xs_b.shape[0], 1.0 / max(1, Xs_b.shape[0])) if Xs_b.size else np.array([]))
            for i, c in enumerate(g['candidates']):
                y_true_all.append(int(c['invited']))
                y_prob_all.append(float(probs[i] if probs.size else 0.0))
                group_ids.append(int(g.get('change_idx', gid)))
                flat_items.append({'state': c['state'], 'label': int(c['invited']), 'ts': g['ts']})
        return y_true_all, y_prob_all, group_ids, flat_items


def evaluate_invitation_irl_plackett(json_path: str | Path, base_cfg: InvitationRankingBuildConfig | None = None, irl_cfg: ConditionalLogitConfig | None = None):
    base_cfg = base_cfg or InvitationRankingBuildConfig()
    irl_cfg = irl_cfg or ConditionalLogitConfig()
    groups = build_invitation_groups_irl(json_path, base_cfg, irl_cfg)
    ordered = sorted(groups, key=lambda g: g['ts'])
    n_test = max(1, int(len(ordered) * 0.2))
    train_groups = ordered[:-n_test]
    test_groups = ordered[-n_test:]
    features = InvitationRankingModel().features
    model = PlackettLuceIRLModel(features, irl_cfg)
    if not train_groups:
        raise RuntimeError('no training groups')
    model.fit(train_groups)
    y_true, y_prob, group_ids, flat_items = model.predict_group_probs(test_groups)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    brier = brier_score_loss(y_true, y_prob) if y_true else float('nan')
    from math import log

    # sequential NLL: recompute using same fixed order approximation
    # For reporting, we compute NLL per invited similar to Level-1 but using initial probs as proxy
    pos_count = max(1, sum(y_true))
    nll = -sum(np.log(max(1e-12, p)) for y, p in zip(y_true, y_prob) if y == 1) / pos_count
    rankm = _ranking_metrics_by_group(y_true, y_prob, group_ids)
    metrics = {
        'n_groups_train': len(train_groups),
        'n_groups_test': len(test_groups),
        'n_candidates_test': len(y_true),
        'auc': auc,
        'brier': brier,
        'nll_per_invited': float(nll),
        **rankm,
    }
    class _IRLModelWrapper:
        def __init__(self, features, scaler, theta):
            self.features = features
            self.scaler = scaler
            self.theta = theta
            self.model = None
    wrapped = _IRLModelWrapper(features, model.scaler, model.theta)
    return metrics, wrapped, flat_items, np.array(y_prob)

__all__.extend(['evaluate_invitation_irl_plackett'])
