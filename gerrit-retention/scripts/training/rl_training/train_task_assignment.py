#!/usr/bin/env python3
"""
Task-centric MDP: MultiReviewerAssignmentEnv を用いた最小 PPO 学習/評価スクリプト

入力: tasks_train.jsonl / tasks_eval.jsonl （scripts/offline/build_task_assignment_from_sequences.py の出力）
出力: policy(.pt), 訓練ログ(.json)
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from gerrit_retention.rl_environment.multi_reviewer_assignment_env import (
    AssignmentTask,
    Candidate,
    MultiReviewerAssignmentEnv,
)


def _read_tasks(jsonl_path: Path) -> Tuple[List[AssignmentTask], List[str]]:
    tasks: List[AssignmentTask] = []
    feature_keys: List[str] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for ln in f:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            cands = [Candidate(reviewer_id=c['reviewer_id'], features=c['features']) for c in obj.get('candidates', [])]
            tasks.append(AssignmentTask(
                change_id=obj.get('change_id'),
                candidates=cands,
                positive_reviewer_ids=obj.get('positive_reviewer_ids') or [],
                timestamp=obj.get('timestamp'),
            ))
            if cands and not feature_keys:
                feature_keys = list(cands[0].features.keys())
    return tasks, feature_keys


class SmallPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(env: MultiReviewerAssignmentEnv, episodes: int = 400, lr: float = 1e-3, seed: int = 42) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = SmallPolicy(obs_dim, act_dim)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    log: List[Dict[str, Any]] = []
    for ep in tqdm(range(episodes), desc='train', leave=False):
        obs, _ = env.reset()
        done = False
        ep_loss = 0.0
        ep_rew = 0.0
        steps = 0
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            logits = policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = int(m.sample().item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            loss = -m.log_prob(torch.tensor(action)) * float(reward)
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += float(loss.item())
            ep_rew += float(reward)
            steps += 1
            obs = next_obs
            done = bool(terminated or truncated)
        log.append({'ep': ep+1, 'reward': ep_rew, 'loss': ep_loss, 'steps': steps})
    return {'policy': policy, 'log': log}


def evaluate(env: MultiReviewerAssignmentEnv, policy: nn.Module) -> Dict[str, Any]:
    obs, _ = env.reset()
    done = False
    matches = 0
    total = 0
    total_expected = len(getattr(env, 'tasks', [])) or None
    pbar = tqdm(total=total_expected, desc='eval', leave=False) if total_expected else None
    try:
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            logits = policy(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            total += 1
            matches += int(reward >= 1.0)  # match_gt reward
            obs = next_obs
            done = bool(terminated or truncated)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
    return {'action_match_rate': (matches / max(1, total)), 'steps': total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-tasks', type=str, required=True)
    ap.add_argument('--eval-tasks', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='outputs/task_assign')
    ap.add_argument('--episodes', type=int, default=600)
    ap.add_argument('--irl-model', type=str, default=None, help='Path to IRL model JSON to use as reward (sets reward_mode=irl_softmax).')
    args = ap.parse_args()

    train_tasks, feat_order = _read_tasks(Path(args.train_tasks))
    eval_tasks, _ = _read_tasks(Path(args.eval_tasks))

    irl_model = None
    reward_mode = 'match_gt'
    if args.irl_model:
        try:
            irl_obj = json.loads(Path(args.irl_model).read_text(encoding='utf-8'))
            # sanity: align feature order if provided in model
            if irl_obj.get('feature_order') and irl_obj['feature_order'] != feat_order:
                # best-effort reorder: intersect keys
                common = [f for f in irl_obj['feature_order'] if f in feat_order]
                if common:
                    feat_order = common
            irl_model = {'theta': np.array(irl_obj.get('theta'), dtype=np.float64), 'scaler': irl_obj.get('scaler')}
            reward_mode = 'irl_softmax'
        except Exception:
            pass

    env_cfg = {'max_candidates': 8, 'use_action_mask': True, 'reward_mode': reward_mode}
    if irl_model is not None:
        env_cfg['irl_model'] = irl_model

    train_env = MultiReviewerAssignmentEnv(train_tasks, feat_order, config=env_cfg)
    eval_env = MultiReviewerAssignmentEnv(eval_tasks, feat_order, config=env_cfg)

    res = train(train_env, episodes=int(args.episodes))
    policy = res['policy']
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_dir / 'task_assign_policy.pt')
    (out_dir / 'train_log.json').write_text(json.dumps(res['log'], ensure_ascii=False, indent=2))

    eval_res = evaluate(eval_env, policy)
    (out_dir / 'eval_report.json').write_text(json.dumps(eval_res, ensure_ascii=False, indent=2))
    print(json.dumps({'train': {'episodes': args.episodes}, 'eval': eval_res}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
