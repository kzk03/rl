#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from gerrit_retention.rl_environment.multi_reviewer_assignment_env import (
    AssignmentTask,
    Candidate,
    MultiReviewerAssignmentEnv,
)


def _read_tasks(jsonl_path: Path) -> Tuple[List[AssignmentTask], List[str]]:
    tasks: List[AssignmentTask] = []
    feat_keys: List[str] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            obj = json.loads(s)
            cands = [Candidate(reviewer_id=c['reviewer_id'], features=c['features']) for c in obj.get('candidates', [])]
            tasks.append(AssignmentTask(
                change_id=obj.get('change_id'),
                candidates=cands,
                positive_reviewer_ids=obj.get('positive_reviewer_ids') or [],
                timestamp=obj.get('timestamp'),
            ))
            if cands and not feat_keys:
                feat_keys = list(cands[0].features.keys())
    return tasks, feat_keys


def _to_dt(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace('Z','+00:00')).replace(tzinfo=None)
    except Exception:
        return datetime.fromisoformat(ts)


def _filter_by_window(tasks: List[AssignmentTask], cutoff: datetime, window: str) -> List[AssignmentTask]:
    if window == 'train':
        return [t for t in tasks if t.timestamp and _to_dt(t.timestamp) <= cutoff]
    # e.g., '1m','3m','6m','12m'
    num = int(window[:-1]); unit = window[-1]
    delta = {'m': 30, 'd': 1, 'y': 365}.get(unit, 30)
    end = cutoff + timedelta(days=num * delta)
    return [t for t in tasks if t.timestamp and cutoff < _to_dt(t.timestamp) <= end]


def evaluate_window(tasks: List[AssignmentTask], feat_order: List[str], policy_path: Path) -> Dict[str, Any]:
    total_steps = len(tasks)
    if total_steps == 0:
        return {'steps': 0, 'action_match_rate': None, 'index0_positive_rate': None, 'avg_candidates': None, 'random_top1_baseline': None}
    env = MultiReviewerAssignmentEnv(tasks, feat_order, config={'max_candidates': 8, 'use_action_mask': True, 'reward_mode': 'match_gt'})
    # policy architecture must match train script
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    import torch.nn as nn
    class SmallPolicy(nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(obs_dim,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,act_dim))
        def forward(self, x):
            return self.net(x)
    policy = SmallPolicy(obs_dim, act_dim)
    policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
    policy.eval()

    obs, _ = env.reset()
    done = False
    total = 0
    match = 0
    # auditing stats for positional bias
    idx0_pos = 0
    cand_counts = []

    with tqdm(total=total_steps, desc='eval', leave=False) as pbar:
        while not done:
            x = torch.from_numpy(obs).float().unsqueeze(0)
            logits = policy(x)
            action = int(torch.argmax(logits, dim=-1).item())
            # capture before stepping: current candidate list length and index0-positive
            cur_task = env._current_task()  # using internal accessor is acceptable here
            k = len(cur_task.candidates)
            cand_counts.append(k)
            if cur_task.positive_reviewer_ids:
                idx0 = cur_task.candidates[0].reviewer_id if k > 0 else None
                if idx0 is not None and idx0 in set(cur_task.positive_reviewer_ids):
                    idx0_pos += 1

            obs, reward, terminated, truncated, _ = env.step(action)
            total += 1
            match += int(reward >= 1.0)
            done = bool(terminated or truncated)
            pbar.update(1)
    avg_k = float(np.mean(cand_counts)) if cand_counts else None
    rand_top1 = float(np.mean([1.0 / c for c in cand_counts])) if cand_counts else None
    return {
        'steps': total,
        'action_match_rate': (match / max(1, total)),
        'index0_positive_rate': (idx0_pos / max(1, total)),
        'avg_candidates': avg_k,
        'random_top1_baseline': rand_top1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, required=True)
    ap.add_argument('--cutoff', type=str, required=True)
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--windows', type=str, default='train,1m,3m,6m,12m')
    ap.add_argument('--out', type=str, default='outputs/task_assign/replay_eval.json')
    args = ap.parse_args()

    all_tasks, feat_order = _read_tasks(Path(args.tasks))
    cutoff = _to_dt(args.cutoff)
    policy_path = Path(args.policy)

    results: Dict[str, Any] = {}
    windows_list = [w.strip() for w in args.windows.split(',') if w.strip()]
    for w in tqdm(windows_list, desc='windows'):
        subset = _filter_by_window(all_tasks, cutoff, w)
        if not subset:
            results[w] = {'steps': 0, 'action_match_rate': None}
            continue
        res = evaluate_window(subset, feat_order, policy_path)
        results[w] = res

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'cutoff': cutoff.isoformat(), 'results': results}, ensure_ascii=False, indent=2))
    print(json.dumps({'out': args.out, 'windows': results}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
