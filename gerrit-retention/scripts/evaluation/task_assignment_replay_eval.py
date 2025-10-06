#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def evaluate_window(tasks: List[AssignmentTask], feat_order: List[str], policy_path: Path, reward_mode: str = 'match_gt', irl_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    total_steps = len(tasks)
    if total_steps == 0:
        return {
            'steps': 0,
            'action_match_rate': None,
            'index0_positive_rate': None,
            'avg_candidates': None,
            'random_top1_baseline': None,
            'avg_reward': None,
            'top3_hit_rate': None,
            'top5_hit_rate': None,
            'mAP': None,
            'ECE': None,
        }
    cfg = {'max_candidates': 8, 'use_action_mask': True, 'reward_mode': reward_mode}
    # IRL related config (temperature / entropy / hybrid) will be attached by caller via irl_model metadata or CLI
    if irl_model is not None:
        # Pop out helper keys that are environment-level (temperature) vs model parameters
        temperature = irl_model.pop('_temperature', None)
        if temperature is not None:
            cfg['temperature'] = float(temperature)
        entropy_coef = irl_model.pop('_entropy_coef', None)
        if entropy_coef is not None:
            cfg['entropy_coef'] = float(entropy_coef)
        hybrid_alpha = irl_model.pop('_hybrid_alpha', None)
        if hybrid_alpha is not None:
            cfg['hybrid_alpha'] = float(hybrid_alpha)
        cfg['irl_model'] = irl_model
    env = MultiReviewerAssignmentEnv(tasks, feat_order, config=cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    import torch.nn as nn
    class SmallPolicy(nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim,128), nn.ReLU(),
                nn.Linear(128,64), nn.ReLU(),
                nn.Linear(64,act_dim)
            )
        def forward(self, x):
            return self.net(x)
    policy = SmallPolicy(obs_dim, act_dim)
    policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
    policy.eval()

    obs, _ = env.reset()
    done = False
    total = 0
    match = 0
    reward_sum = 0.0
    idx0_pos = 0
    cand_counts: List[int] = []
    # ranking & calibration accumulators
    top3_hits = 0
    top5_hits = 0
    ap_list: List[float] = []
    confs: List[float] = []
    corrects: List[int] = []

    with tqdm(total=total_steps, desc='eval', leave=False) as pbar:
        while not done:
            x = torch.from_numpy(obs).float().unsqueeze(0)
            logits = policy(x)
            # Determine current task & candidate count
            cur_task = env._current_task()
            k = min(len(cur_task.candidates), env.max_candidates)
            cand_logits = logits[0, :k]
            probs = torch.softmax(cand_logits, dim=-1)
            action = int(torch.argmax(cand_logits, dim=-1).item())

            # stats before stepping
            cand_counts.append(k)
            if cur_task.positive_reviewer_ids:
                first_id = cur_task.candidates[0].reviewer_id if k > 0 else None
                if first_id is not None and first_id in set(cur_task.positive_reviewer_ids):
                    idx0_pos += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total += 1
            reward_sum += float(reward)
            if isinstance(info, dict) and info.get('is_match') is not None:
                match += int(bool(info.get('is_match')))
            else:
                match += int(reward >= 1.0)

            # ranking metrics
            pos_set = set(cur_task.positive_reviewer_ids or [])
            cand_ids = [c.reviewer_id for c in cur_task.candidates[:k]]
            sorted_idx = torch.argsort(cand_logits, dim=-1, descending=True).tolist()
            pos_indices = [i for i, rid in enumerate(cand_ids) if rid in pos_set]
            if pos_indices:
                if any(i in sorted_idx[:min(3, k)] for i in pos_indices):
                    top3_hits += 1
                if any(i in sorted_idx[:min(5, k)] for i in pos_indices):
                    top5_hits += 1
                hits = 0
                prec_sum = 0.0
                for rank, idx in enumerate(sorted_idx, start=1):
                    if idx in pos_indices:
                        hits += 1
                        prec_sum += hits / rank
                ap_list.append(prec_sum / max(1, len(pos_indices)))
            # calibration collection
            top1_prob = float(probs[action]) if k > action else 0.0
            top1_correct = 1 if action in pos_indices else 0
            confs.append(top1_prob)
            corrects.append(top1_correct)

            done = bool(terminated or truncated)
            pbar.update(1)

    avg_k = float(np.mean(cand_counts)) if cand_counts else None
    rand_top1 = float(np.mean([1.0 / c for c in cand_counts])) if cand_counts else None
    top3_rate = top3_hits / max(1, total)
    top5_rate = top5_hits / max(1, total)
    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    # ECE
    if confs:
        bins = 10
        bin_inds = np.clip((np.array(confs) * bins).astype(int), 0, bins - 1)
        ece = 0.0
        conf_arr = np.array(confs)
        corr_arr = np.array(corrects)
        for b in range(bins):
            mask = (bin_inds == b)
            if not np.any(mask):
                continue
            conf_bin = float(np.mean(conf_arr[mask]))
            acc_bin = float(np.mean(corr_arr[mask]))
            weight = float(np.sum(mask)) / len(conf_arr)
            ece += weight * abs(acc_bin - conf_bin)
    else:
        ece = None

    return {
        'steps': total,
        'action_match_rate': (match / max(1, total)),
        'index0_positive_rate': (idx0_pos / max(1, total)),
        'avg_candidates': avg_k,
        'random_top1_baseline': rand_top1,
        'avg_reward': (reward_sum / max(1, total)),
        'top3_hit_rate': top3_rate,
        'top5_hit_rate': top5_rate,
        'mAP': mAP,
        'ECE': ece,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, required=True)
    ap.add_argument('--cutoff', type=str, required=True)
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--windows', type=str, default='train,1m,3m,6m,12m')
    ap.add_argument('--out', type=str, default='outputs/task_assign/replay_eval.json')
    ap.add_argument('--reward-mode', type=str, default='match_gt',
                    choices=['match_gt','irl_softmax','accept_prob','irl_logprob','hybrid_match_irl','irl_entropy_bonus'],
                    help='Environment reward mode for evaluation.')
    ap.add_argument('--irl-model', type=str, default=None, help='Optional IRL model JSON (theta, scaler, temperature). If provided and reward-mode is match_gt, auto-switch to irl_softmax unless another IRL mode specified.')
    ap.add_argument('--temperature', type=float, default=None, help='Override temperature for IRL softmax (if None, use model or default=1.0).')
    ap.add_argument('--entropy-coef', type=float, default=None, help='Entropy coefficient for irl_entropy_bonus reward.')
    ap.add_argument('--hybrid-alpha', type=float, default=None, help='Alpha blending factor for hybrid_match_irl (weight on match reward).')
    args = ap.parse_args()

    all_tasks, feat_order = _read_tasks(Path(args.tasks))
    cutoff = _to_dt(args.cutoff)
    policy_path = Path(args.policy)
    # Load IRL model if provided
    irl_model = None
    reward_mode = args.reward_mode
    if args.irl_model:
        try:
            irl_obj = json.loads(Path(args.irl_model).read_text(encoding='utf-8'))
            irl_model = {
                'theta': np.array(irl_obj.get('theta'), dtype=np.float64),
                'scaler': irl_obj.get('scaler')
            }
            # Extract stored temperature if present
            if 'temperature' in irl_obj and args.temperature is None:
                irl_model['_temperature'] = irl_obj.get('temperature')
            # CLI overrides (temperature / entropy / hybrid)
            if args.temperature is not None:
                irl_model['_temperature'] = args.temperature
            if args.entropy_coef is not None:
                irl_model['_entropy_coef'] = args.entropy_coef
            if args.hybrid_alpha is not None:
                irl_model['_hybrid_alpha'] = args.hybrid_alpha
            # Auto-switch only if user stayed on default and selected an IRL-compatible reward
            if reward_mode == 'match_gt':
                reward_mode = 'irl_softmax'
        except Exception:
            pass

    results: Dict[str, Any] = {}
    windows_list = [w.strip() for w in args.windows.split(',') if w.strip()]
    for w in tqdm(windows_list, desc='windows'):
        subset = _filter_by_window(all_tasks, cutoff, w)
        if not subset:
            results[w] = {'steps': 0, 'action_match_rate': None}
            continue

        res = evaluate_window(subset, feat_order, policy_path, reward_mode=reward_mode, irl_model=irl_model)
        results[w] = res

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    meta_out = {
        'cutoff': cutoff.isoformat(),
        'reward_mode': reward_mode,
        'irl_model_used': bool(irl_model is not None),
        'results': results,
    }
    Path(args.out).write_text(json.dumps(meta_out, ensure_ascii=False, indent=2))
    print(json.dumps({'out': args.out, 'windows': results, 'reward_mode': reward_mode}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
