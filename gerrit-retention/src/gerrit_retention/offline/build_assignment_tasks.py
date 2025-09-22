from __future__ import annotations

"""
Utilities to build MultiReviewerAssignmentEnv tasks from ranking samples.

We reuse reviewer_invitation_ranking.build_invitation_ranking_samples outputs,
which contain per-(change, reviewer) states and binary labels.

This module groups samples by change_idx into AssignmentTask items with up to K
candidate reviewers each. Ground-truth positives are reviewers with label=1.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from ..rl_environment.multi_reviewer_assignment_env import AssignmentTask, Candidate


def build_tasks_from_samples(samples: Sequence[Dict[str, Any]], feature_keys: Sequence[str], max_candidates: int = 8) -> Tuple[List[AssignmentTask], List[str]]:
    # Group by change_idx
    by_change: Dict[Any, List[Dict[str, Any]]] = {}
    for s in samples:
        idx = s.get('change_idx')
        by_change.setdefault(idx, []).append(s)
    tasks: List[AssignmentTask] = []
    # Keep deterministic feature order
    feature_order = list(feature_keys)
    for change_idx, rows in by_change.items():
        # sort rows by positive first then recent ts if available
        rows_sorted = sorted(rows, key=lambda r: (-(1 if r.get('label') == 1 else 0), r.get('ts', '')))
        cand: List[Candidate] = []
        positives: List[Any] = []
        for r in rows_sorted[:max_candidates]:
            st = r.get('state') or {}
            rid = st.get('reviewer_id') or r.get('reviewer_id')
            if r.get('label') == 1 and rid is not None:
                positives.append(rid)
            # subset features
            feats = {k: float(st.get(k, 0.0)) for k in feature_order}
            cand.append(Candidate(reviewer_id=rid, features=feats))
        tasks.append(AssignmentTask(change_id=change_idx, candidates=cand, positive_reviewer_ids=positives, timestamp=rows_sorted[0].get('ts')))
    return tasks, feature_order
