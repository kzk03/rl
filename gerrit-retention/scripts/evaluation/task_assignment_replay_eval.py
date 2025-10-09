#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
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

CSV_COLUMNS = [
    'window',
    'change_id',
    'timestamp',
    'candidate_rank',
    'candidate_index',
    'reviewer_id',
    'score',
    'probability',
    'is_positive',
    'selected',
    'positive_reviewers',
    'candidate_count',
    'reward',
]


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


_WINDOW_RE = re.compile(r'^(?P<start>[+-]?\d+)?(?P<start_unit>[mdy])?(?:-(?P<end>[+-]?\d+)(?P<end_unit>[mdy]))?$')


def _parse_window_spec(spec: str) -> Tuple[float, str, float, str]:
    """Parse window spec.

    Formats:
      - "train" special handled elsewhere.
      - "6m" => start=0m, end=6m.
      - "3m-6m" => start=3m, end=6m.
    Units: m (30 days), d, y (365 days).
    """
    if spec == 'train':
        return (0.0, 'd', 0.0, 'd')
    m = _WINDOW_RE.match(spec)
    if not m:
        raise ValueError(f'Invalid window spec: {spec}')
    start = m.group('start')
    start_unit = m.group('start_unit') or 'm'
    end = m.group('end')
    end_unit = m.group('end_unit')
    if start is None and end is None:
        raise ValueError(f'Window spec missing range: {spec}')
    if end is None:
        # treat like cumulative end
        end = start
        end_unit = start_unit
        start = '0'
        start_unit = end_unit
    if start is None:
        start = '0'
    if start_unit is None:
        start_unit = end_unit or 'm'
    if end_unit is None:
        end_unit = start_unit
    start_val = float(start)
    end_val = float(end)
    if _window_delta(start_val, start_unit) >= _window_delta(end_val, end_unit):
        raise ValueError(f'Window spec start must be before end: {spec}')
    return (start_val, start_unit, end_val, end_unit)


def _window_delta(value: float, unit: str) -> timedelta:
    factor = {'d': 1, 'm': 30, 'y': 365}.get(unit, 30)
    return timedelta(days=value * factor)


def _filter_by_window(tasks: List[AssignmentTask], cutoff: datetime, window: str) -> List[AssignmentTask]:
    if window == 'train':
        return [t for t in tasks if t.timestamp and _to_dt(t.timestamp) <= cutoff]
    start_val, start_unit, end_val, end_unit = _parse_window_spec(window)
    start_dt = cutoff + _window_delta(start_val, start_unit)
    end_dt = cutoff + _window_delta(end_val, end_unit)
    return [t for t in tasks if t.timestamp and start_dt < _to_dt(t.timestamp) <= end_dt]


def _prepare_feature_array(feats: Dict[str, float], order: List[str]) -> np.ndarray:
    return np.array([float(feats.get(f, 0.0)) for f in order], dtype=np.float64)


def _apply_scaler(x: np.ndarray, scaler: Optional[Dict[str, Any]]) -> np.ndarray:
    if not scaler:
        return x
    mean = scaler.get('mean') if isinstance(scaler, dict) else None
    scale = scaler.get('scale') if isinstance(scaler, dict) else None
    if mean is not None and scale is not None:
        mean_arr = np.asarray(mean, dtype=np.float64)
        scale_arr = np.asarray(scale, dtype=np.float64)
        if mean_arr.shape == x.shape:
            return (x - mean_arr) / np.maximum(scale_arr, 1e-8)
    return x


def _compute_ece(confs: List[float], corrects: List[int], bins: int = 10) -> Optional[float]:
    if not confs:
        return None
    conf_arr = np.array(confs, dtype=np.float64)
    corr_arr = np.array(corrects, dtype=np.float64)
    bin_inds = np.clip((conf_arr * bins).astype(int), 0, bins - 1)
    ece = 0.0
    total = len(conf_arr)
    for b in range(bins):
        mask = (bin_inds == b)
        if not np.any(mask):
            continue
        conf_bin = float(np.mean(conf_arr[mask]))
        acc_bin = float(np.mean(corr_arr[mask]))
        weight = float(np.sum(mask)) / total
        ece += weight * abs(acc_bin - conf_bin)
    return ece


def evaluate_window_policy(
    tasks: List[AssignmentTask],
    feat_order: List[str],
    policy_path: Path,
    max_candidates: int = 8,
    reward_mode: str = 'match_gt',
    irl_model: Optional[Dict[str, Any]] = None,
    irl_temperature: Optional[float] = None,
    window: Optional[str] = None,
    csv_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
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
    cfg = {'max_candidates': int(max_candidates), 'use_action_mask': True, 'reward_mode': reward_mode}
    if irl_model is not None:
        cfg['irl_model'] = {
            'theta': irl_model.get('theta'),
            'scaler': irl_model.get('scaler'),
        }
        if reward_mode in ('irl_softmax', 'irl_logprob', 'hybrid_match_irl', 'irl_entropy_bonus'):
            cfg['temperature'] = float(irl_temperature if irl_temperature is not None else 1.0)
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
    precision_acc: Dict[int, List[float]] = {k: [] for k in (1, 3, 5)}
    recall_acc: Dict[int, List[float]] = {k: [] for k in (1, 3, 5)}
    coverage_hits = 0
    positives_tasks = 0

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
            pos_count = len(pos_indices)
            if pos_count:
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
                positives_tasks += 1
                coverage_hits += int(any(idx in pos_indices for idx in sorted_idx[:min(5, k)]))
                for topk in (1, 3, 5):
                    limit = min(topk, k)
                    if limit == 0:
                        continue
                    hits_k = sum(1 for idx in sorted_idx[:limit] if idx in pos_indices)
                    precision_acc[topk].append(hits_k / float(limit))
                    recall_acc[topk].append(hits_k / float(pos_count))
            # calibration collection
            top1_prob = float(probs[action]) if k > action else 0.0
            top1_correct = 1 if action in pos_indices else 0
            confs.append(top1_prob)
            corrects.append(top1_correct)

            if csv_rows is not None:
                positive_serialized = ';'.join(str(rid) for rid in sorted(pos_set)) if pos_set else ''
                logits_list = cand_logits.detach().cpu().tolist()
                prob_list = probs.detach().cpu().tolist()
                for rank, idx in enumerate(sorted_idx, start=1):
                    rid = cand_ids[idx]
                    csv_rows.append({
                        'window': window,
                        'change_id': cur_task.change_id,
                        'timestamp': cur_task.timestamp,
                        'candidate_rank': rank,
                        'candidate_index': idx,
                        'reviewer_id': rid,
                        'score': float(logits_list[idx]) if idx < len(logits_list) else None,
                        'probability': float(prob_list[idx]) if idx < len(prob_list) else None,
                        'is_positive': int(rid in pos_set),
                        'selected': int(idx == action),
                        'positive_reviewers': positive_serialized,
                        'candidate_count': k,
                        'reward': float(reward) if reward is not None else None,
                    })

            done = bool(terminated or truncated)
            pbar.update(1)

    avg_k = float(np.mean(cand_counts)) if cand_counts else None
    rand_top1 = float(np.mean([1.0 / c for c in cand_counts])) if cand_counts else None
    top3_rate = top3_hits / max(1, total)
    top5_rate = top5_hits / max(1, total)
    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    ece = _compute_ece(confs, corrects)

    coverage_rate = coverage_hits / max(1, positives_tasks)

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
        'precision_at_1': float(np.mean(precision_acc[1])) if precision_acc[1] else None,
        'precision_at_3': float(np.mean(precision_acc[3])) if precision_acc[3] else None,
        'precision_at_5': float(np.mean(precision_acc[5])) if precision_acc[5] else None,
        'recall_at_1': float(np.mean(recall_acc[1])) if recall_acc[1] else None,
        'recall_at_3': float(np.mean(recall_acc[3])) if recall_acc[3] else None,
        'recall_at_5': float(np.mean(recall_acc[5])) if recall_acc[5] else None,
        'positive_coverage': coverage_rate,
    }


def evaluate_window_irl(
    tasks: List[AssignmentTask],
    feat_order: List[str],
    irl_model: Dict[str, Any],
    temperature: float = 1.0,
    window: Optional[str] = None,
    csv_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    total = 0
    top_matches = 0
    idx0_pos = 0
    cand_counts: List[int] = []
    top3_hits = 0
    top5_hits = 0
    ap_list: List[float] = []
    confs: List[float] = []
    corrects: List[int] = []
    precision_acc: Dict[int, List[float]] = {k: [] for k in (1, 3, 5)}
    recall_acc: Dict[int, List[float]] = {k: [] for k in (1, 3, 5)}
    coverage_hits = 0
    positives_tasks = 0

    theta = np.asarray(irl_model.get('theta'), dtype=np.float64)
    scaler = irl_model.get('scaler')
    if theta is None or theta.size == 0:
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

    bias = theta[-1]
    weights = theta[:-1]
    temp = max(1e-6, float(temperature))

    for task in tasks:
        candidates = task.candidates or []
        k = len(candidates)
        if k == 0:
            continue
        cand_counts.append(k)
        feats_arr = []
        for cand in candidates:
            x = _prepare_feature_array(cand.features or {}, feat_order)
            x = _apply_scaler(x, scaler)
            feats_arr.append(x)
        util = np.dot(np.vstack(feats_arr), weights) + bias
        util = util / temp
        util = util - np.max(util)
        expu = np.exp(util)
        probs = expu / np.maximum(expu.sum(), 1e-12)
        top_idx = int(np.argmax(probs))
        pos_set = set(task.positive_reviewer_ids or [])
        cand_ids = [c.reviewer_id for c in candidates]

        if cand_ids and cand_ids[0] in pos_set:
            idx0_pos += 1

        top_matches += int(cand_ids[top_idx] in pos_set)
        total += 1

        sorted_idx = np.argsort(-probs)
        pos_indices = [i for i, rid in enumerate(cand_ids) if rid in pos_set]
        pos_count = len(pos_indices)
        if pos_count:
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
            positives_tasks += 1
            coverage_hits += int(any(idx in pos_indices for idx in sorted_idx[:min(5, k)]))
            for topk in (1, 3, 5):
                limit = min(topk, k)
                if limit == 0:
                    continue
                hits_k = sum(1 for idx in sorted_idx[:limit] if idx in pos_indices)
                precision_acc[topk].append(hits_k / float(limit))
                recall_acc[topk].append(hits_k / float(pos_count))
        confs.append(float(probs[top_idx]))
        corrects.append(int(cand_ids[top_idx] in pos_set))

        if csv_rows is not None:
            positive_serialized = ';'.join(str(rid) for rid in sorted(pos_set)) if pos_set else ''
            util_list = util.tolist()
            prob_list = probs.tolist()
            for rank, idx in enumerate(sorted_idx, start=1):
                rid = cand_ids[int(idx)]
                csv_rows.append({
                    'window': window,
                    'change_id': task.change_id,
                    'timestamp': task.timestamp,
                    'candidate_rank': rank,
                    'candidate_index': int(idx),
                    'reviewer_id': rid,
                    'score': float(util_list[int(idx)]) if int(idx) < len(util_list) else None,
                    'probability': float(prob_list[int(idx)]) if int(idx) < len(prob_list) else None,
                    'is_positive': int(rid in pos_set),
                    'selected': int(idx == top_idx),
                    'positive_reviewers': positive_serialized,
                    'candidate_count': k,
                    'reward': None,
                })

    if total == 0:
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

    avg_k = float(np.mean(cand_counts)) if cand_counts else None
    rand_top1 = float(np.mean([1.0 / c for c in cand_counts])) if cand_counts else None
    top3_rate = top3_hits / max(1, total)
    top5_rate = top5_hits / max(1, total)
    mAP = float(np.mean(ap_list)) if ap_list else 0.0

    ece = _compute_ece(confs, corrects)

    coverage_rate = coverage_hits / max(1, positives_tasks)

    return {
        'steps': total,
        'action_match_rate': top_matches / max(1, total),
        'index0_positive_rate': idx0_pos / max(1, total),
        'avg_candidates': avg_k,
        'random_top1_baseline': rand_top1,
        'avg_reward': None,
        'top3_hit_rate': top3_rate,
        'top5_hit_rate': top5_rate,
        'mAP': mAP,
        'ECE': ece,
        'precision_at_1': float(np.mean(precision_acc[1])) if precision_acc[1] else None,
        'precision_at_3': float(np.mean(precision_acc[3])) if precision_acc[3] else None,
        'precision_at_5': float(np.mean(precision_acc[5])) if precision_acc[5] else None,
        'recall_at_1': float(np.mean(recall_acc[1])) if recall_acc[1] else None,
        'recall_at_3': float(np.mean(recall_acc[3])) if recall_acc[3] else None,
        'recall_at_5': float(np.mean(recall_acc[5])) if recall_acc[5] else None,
        'positive_coverage': coverage_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, required=True)
    ap.add_argument('--cutoff', type=str, required=True)
    ap.add_argument('--policy', type=str, default=None, help='Policy checkpoint (.pt). Required when eval-mode=policy.')
    ap.add_argument('--max-candidates', type=int, default=8, help='Number of candidate slots exposed during evaluation.')
    ap.add_argument('--windows', type=str, default='train,1m,3m,6m,12m',
                    help='評価ウィンドウ（例: 1m, 3m-6m, -12m-0m）。負の値を使うとカットオフ以前（学習期間）を指定できます。')
    ap.add_argument('--out', type=str, default='outputs/task_assign/replay_eval.json')
    ap.add_argument('--csv-dir', type=str, default=None, help='ウィンドウごとの推薦結果をCSVとして出力するディレクトリ。指定しない場合は保存しません。')
    ap.add_argument('--reward-mode', type=str, default='match_gt',
                    choices=['match_gt','irl_softmax','accept_prob','irl_logprob','hybrid_match_irl','irl_entropy_bonus'],
                    help='Environment reward mode for evaluation.')
    ap.add_argument('--irl-model', type=str, default=None, help='Optional IRL model JSON (theta, scaler, temperature). If provided and reward-mode is match_gt, auto-switch to irl_softmax unless another IRL mode specified.')
    ap.add_argument('--temperature', type=float, default=None, help='Override temperature for IRL softmax (if None, use model or default=1.0).')
    ap.add_argument('--entropy-coef', type=float, default=None, help='Entropy coefficient for irl_entropy_bonus reward.')
    ap.add_argument('--hybrid-alpha', type=float, default=None, help='Alpha blending factor for hybrid_match_irl (weight on match reward).')
    ap.add_argument('--eval-mode', type=str, default='policy', choices=['policy', 'irl'], help='Evaluation mode: policy (default) uses a trained RL policy; irl ranks candidates directly using the IRL model.')
    args = ap.parse_args()

    all_tasks, feat_order = _read_tasks(Path(args.tasks))
    cutoff = _to_dt(args.cutoff)
    eval_mode = args.eval_mode

    policy_path: Optional[Path] = Path(args.policy) if args.policy else None

    irl_model: Optional[Dict[str, Any]] = None
    irl_temperature: Optional[float] = args.temperature
    if args.irl_model:
        try:
            irl_obj = json.loads(Path(args.irl_model).read_text(encoding='utf-8'))
            irl_model = {
                'theta': np.array(irl_obj.get('theta'), dtype=np.float64),
                'scaler': irl_obj.get('scaler'),
            }
            model_feat_order = irl_obj.get('feature_order')
            if model_feat_order:
                feat_order = [f for f in model_feat_order if f in feat_order]
            if irl_temperature is None and irl_obj.get('temperature') is not None:
                irl_temperature = float(irl_obj.get('temperature'))
        except Exception:
            irl_model = None

    reward_mode_used = None
    if eval_mode == 'policy':
        if policy_path is None:
            raise ValueError('eval-mode=policy では --policy を指定してください。')
        reward_mode = args.reward_mode
        if irl_model is not None and reward_mode == 'match_gt':
            reward_mode = 'irl_softmax'
        reward_mode_used = reward_mode
    else:
        if irl_model is None:
            raise ValueError('eval-mode=irl では --irl-model が必須です。')
        reward_mode_used = 'irl_direct'

    results: Dict[str, Any] = {}
    windows_list = [w.strip() for w in args.windows.split(',') if w.strip()]
    csv_dir: Optional[Path] = Path(args.csv_dir) if args.csv_dir else None
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)
    csv_outputs: Dict[str, str] = {}

    for w in tqdm(windows_list, desc='windows'):
        subset = _filter_by_window(all_tasks, cutoff, w)
        if not subset:
            results[w] = {'steps': 0, 'action_match_rate': None}
            continue

        csv_rows: Optional[List[Dict[str, Any]]] = [] if csv_dir is not None else None
        if eval_mode == 'policy':
            res = evaluate_window_policy(
                subset,
                feat_order,
                policy_path,
                max_candidates=int(args.max_candidates),
                reward_mode=reward_mode,
                irl_model=irl_model,
                irl_temperature=irl_temperature,
                window=w,
                csv_rows=csv_rows,
            )
        else:
            res = evaluate_window_irl(
                subset,
                feat_order,
                irl_model,
                temperature=float(irl_temperature if irl_temperature is not None else 1.0),
                window=w,
                csv_rows=csv_rows,
            )
        results[w] = res

        if csv_dir is not None and csv_rows is not None:
            csv_path = csv_dir / f'{w}.csv'
            with csv_path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for row in csv_rows:
                    writer.writerow(row)
            res['csv_path'] = str(csv_path)
            csv_outputs[w] = str(csv_path)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    meta_out = {
        'cutoff': cutoff.isoformat(),
        'reward_mode': reward_mode_used,
        'eval_mode': eval_mode,
        'irl_model_used': bool(irl_model is not None),
        'results': results,
        'csv_outputs': csv_outputs if csv_outputs else None,
    }
    Path(args.out).write_text(json.dumps(meta_out, ensure_ascii=False, indent=2))
    print(json.dumps({'out': args.out, 'windows': results, 'reward_mode': reward_mode_used}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
