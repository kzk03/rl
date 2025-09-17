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
                file_jaccard = _file_path_similarity_fuzzy(list(change_files), list(past_files_30d))
                file_tfidf = _tfidf_cosine(change_tokens, _path_tokens(list(past_files_30d)), idf_current)
            else:
                file_jaccard = 0.0
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
                'change_prev_positive_cnt': observed_pos_cnt,
                'reviewer_active_flag_30d': active_flag,
                'reviewer_pending_reviews': pending_cnt,
                'reviewer_night_activity_share_30d': night_activity_share,
                'reviewer_overload_flag': overload_flag,
                'reviewer_workload_deviation_z': workload_deviation_z,
                'macro_bus_factor_top5_share': bus_factor_top5_share,
                'match_off_specialty_flag': match_off_specialty_flag,
                'off_specialty_recent_ratio': off_specialty_recent_ratio,
                'reviewer_file_jaccard_30d': file_jaccard,
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
                file_jaccard = _file_path_similarity_fuzzy(list(change_files), list(past_files_30d))
                file_tfidf = _tfidf_cosine(change_tokens, _path_tokens(list(past_files_30d)), idf_current)
            else:
                file_jaccard = 0.0
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
                'change_prev_positive_cnt': observed_pos_cnt,
                'reviewer_active_flag_30d': active_flag,
                'reviewer_pending_reviews': pending_cnt,
                'reviewer_night_activity_share_30d': night_activity_share,
                'reviewer_overload_flag': overload_flag,
                'reviewer_workload_deviation_z': workload_deviation_z,
                'macro_bus_factor_top5_share': bus_factor_top5_share,
                'match_off_specialty_flag': match_off_specialty_flag,
                'off_specialty_recent_ratio': off_specialty_recent_ratio,
                'reviewer_file_jaccard_30d': file_jaccard,
                'reviewer_file_tfidf_cosine_30d': file_tfidf,
            })
        # Append labeled samples
        for st in pos_states:
            samples.append({'state': st, 'label': 1, 'ts': created.isoformat()})
        for st in neg_states:
            samples.append({'state': st, 'label': 0, 'ts': created.isoformat()})

        # Update recent IDF window AFTER using idf for current change
        if cfg.idf_mode == 'recent':
            toks_cur = ch.get('tokens') or set()
            window.append((created, toks_cur))
            for t in toks_cur:
                df_window[t] += 1

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
            'change_prev_positive_cnt',
            'reviewer_active_flag_30d',
            # Refined leakage-free feature set
            'reviewer_pending_reviews',
            'reviewer_night_activity_share_30d',
            'reviewer_overload_flag',
            'reviewer_workload_deviation_z',
            'macro_bus_factor_top5_share',
            'match_off_specialty_flag',
            'off_specialty_recent_ratio',
            'reviewer_file_jaccard_30d',
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
    return {
        'n_train': len(train),
        'n_test': len(test),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'brier': brier,
        'positive_rate_test': sum(y_true) / len(y_true) if y_true else 0.0,
    }, model, test, y_prob


__all__ = [
    'build_invitation_ranking_samples',
    'evaluate_invitation_ranking',
    'InvitationRankingBuildConfig',
]
