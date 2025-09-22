from __future__ import annotations

"""
Train PPO to assign reviewers to changes using MultiReviewerAssignmentEnv.

Data source: reviewer_invitation_ranking.build_invitation_ranking_samples
Reward: +1 if selected reviewer is among ground-truth participants.
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from gerrit_retention.offline.build_assignment_tasks import build_tasks_from_samples
from gerrit_retention.recommendation.reviewer_invitation_ranking import (
    InvitationRankingBuildConfig,
    InvitationRankingModel,
    build_invitation_ranking_samples,
    evaluate_invitation_irl,
)
from gerrit_retention.rl_environment.multi_reviewer_assignment_env import (
    AssignmentTask,
    MultiReviewerAssignmentEnv,
)


def change_level_split(samples: List[Dict[str, Any]], split_ratio: float = 0.8):
    # Group by change_idx and sort groups by earliest ts
    by_change: Dict[Any, List[Dict[str, Any]]] = {}
    for s in samples:
        gid = s.get('change_idx')
        by_change.setdefault(gid, []).append(s)
    change_order = sorted(by_change.items(), key=lambda kv: min(x.get('ts', '') for x in kv[1]))
    n_changes = len(change_order)
    if n_changes == 0:
        return [], [], 0, 0
    n_train = int(n_changes * split_ratio)
    # Ensure at least one train and one eval when possible
    if n_train <= 0:
        n_train = 1 if n_changes >= 1 else 0
    if n_train >= n_changes:
        n_train = max(1, n_changes - 1)
    train_ids = [gid for gid, _ in change_order[:n_train]]
    eval_ids = [gid for gid, _ in change_order[n_train:]]
    # If eval becomes empty (e.g., n_changes==1), duplicate the last train change into eval for smoke evaluation
    if not eval_ids and train_ids:
        eval_ids = [train_ids[-1]]
    train_s = [s for s in samples if s.get('change_idx') in train_ids]
    eval_s = [s for s in samples if s.get('change_idx') in eval_ids]
    return train_s, eval_s, len(set(train_ids)), len(set(eval_ids))


def make_env(tasks: List[AssignmentTask], feature_order: List[str], max_candidates: int = 8, reward_mode: str = 'match_gt', irl_model: Any | None = None, continuity_weight: float = 0.0, continuity_tau: float = 2.0):
    def _make():
        return MultiReviewerAssignmentEnv(
            tasks,
            feature_order,
            config={
                'max_candidates': max_candidates,
                'use_action_mask': True,
                'invalid_action_penalty': -0.05,
                'reward_mode': reward_mode,
                'irl_model': irl_model,
                'continuity_weight': continuity_weight,
                'continuity_tau': continuity_tau,
            },
        )
    return _make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--max-candidates', type=int, default=8)
    ap.add_argument('--timesteps', type=int, default=50_000)
    ap.add_argument('--reward-mode', choices=['match_gt', 'irl_softmax'], default='match_gt')
    ap.add_argument('--continuity-weight', type=float, default=0.0)
    ap.add_argument('--continuity-tau', type=float, default=2.0)
    ap.add_argument('--output', default='outputs/assignment_rl')
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build samples using existing pipeline
    cfg = InvitationRankingBuildConfig()
    samples = build_invitation_ranking_samples(args.changes, cfg)
    if not samples:
        print('No samples built; check changes path.')
        return
    train_s, eval_s, n_train_changes, n_eval_changes = change_level_split(samples, 0.8)

    # Derive feature keys from invitation model (keeps consistency)
    feature_keys = InvitationRankingModel().features

    irl_model = None
    if args.reward_mode == 'irl_softmax':
        # Fit IRL on training split and get scaler/theta wrapper
        metrics, wrapped, _, _ = evaluate_invitation_irl(args.changes, InvitationRankingBuildConfig(), None)
        # Ensure feature order match
        if getattr(wrapped, 'features', None) and list(wrapped.features) != list(feature_keys):
            print('Warning: IRL feature list differs from env features; proceeding but scores may mismatch.')
        irl_model = {'theta': getattr(wrapped, 'theta', None), 'scaler': getattr(wrapped, 'scaler', None)}
    train_tasks, feature_order = build_tasks_from_samples(train_s, feature_keys, args.max_candidates)
    eval_tasks, _ = build_tasks_from_samples(eval_s, feature_keys, args.max_candidates)
    print(f"Split: changes train={n_train_changes}, eval={n_eval_changes} | tasks train={len(train_tasks)}, eval={len(eval_tasks)}")

    # Guard empty splits
    if len(train_tasks) == 0:
        print('No train tasks after split; aborting.')
        return
    if len(eval_tasks) == 0:
        # Fallback: evaluate on train tasks to allow pipeline to complete
        print('No eval tasks after split; evaluating on train tasks as fallback.')
        eval_tasks = train_tasks
    # Vec envs
    train_env = DummyVecEnv([make_env(train_tasks, feature_order, args.max_candidates, args.reward_mode, irl_model, args.continuity_weight, args.continuity_tau)])
    eval_env = DummyVecEnv([make_env(eval_tasks, feature_order, args.max_candidates, args.reward_mode, irl_model, args.continuity_weight, args.continuity_tau)])

    # Train PPO
    model = PPO('MlpPolicy', train_env, verbose=1, n_steps=512, batch_size=512, learning_rate=3e-4, gamma=0.99)
    model.learn(total_timesteps=args.timesteps)

    # Quick eval: average reward over eval tasks via greedy policy
    obs = eval_env.reset()
    total_r = 0.0
    steps = 0
    # run one pass over tasks
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = eval_env.step(action)
        total_r += float(reward.mean())
        steps += 1
        done = bool(dones.any())
    acc = total_r / max(1, steps)
    (out_dir / 'eval_summary.txt').write_text(f"avg_reward_per_step={acc:.4f}\nsteps={steps}\n", encoding='utf-8')
    print(f"Eval avg reward/step: {acc:.4f} over {steps} steps")


if __name__ == '__main__':
    main()
