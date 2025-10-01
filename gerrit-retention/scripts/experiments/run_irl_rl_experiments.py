#!/usr/bin/env python3
"""
IRL→RL 実験ランナー

機能:
- オフライン (state, action) から RetentionIRLSystem を学習
- IRL貪欲方策でオフライン eval をリプレイ評価（エピソード一致率）
- IRL報酬を用いたPPO学習（簡易版）を実行し、報酬/行動分布を記録

出力:
- outputs/experiments/<timestamp>/ に各種成果物（IRLモデル、評価レポート、RL統計、方策）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from gerrit_retention.rl_environment.irl_reward_wrapper import (
    IRLRewardWrapper,  # type: ignore
)
from gerrit_retention.rl_environment.ppo_agent import create_ppo_agent  # type: ignore
from gerrit_retention.rl_environment.review_env import (
    ReviewAcceptanceEnvironment,  # type: ignore
)
from gerrit_retention.rl_environment.time_split_data_wrapper import (
    TimeSplitDataWrapper,  # type: ignore
)
from gerrit_retention.rl_prediction.retention_irl_system import (
    RetentionIRLSystem,  # type: ignore
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    out = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def train_irl_offline(train_samples: List[Dict[str, Any]], epochs: int = 5, batch_size: int = 1024,
                      negatives_per_pos: int = 2, hidden: int = 128, lr: float = 1e-3,
                      state_dim: int = 20, action_dim: int = 3) -> RetentionIRLSystem:
    """簡易IRL学習（RetentionIRLSystem）。train_retention_irl_from_offline.py と同等の簡易版。"""
    cfg = {'state_dim': int(state_dim), 'action_dim': int(action_dim), 'hidden_dim': int(hidden), 'learning_rate': float(lr)}
    irl = RetentionIRLSystem(cfg)
    device = getattr(irl, 'device', torch.device('cpu'))
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCELoss()
    optim_ = torch.optim.Adam(irl.network.parameters(), lr=float(lr))

    def iter_batches():
        idxs = np.arange(len(train_samples))
        np.random.shuffle(idxs)
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start:start+batch_size]
            S_list, A_list, R_list, C_list = [], [], [], []
            for i in chunk:
                s = np.asarray(train_samples[i].get('state', []), dtype=np.float32)
                if s.size == 0:
                    continue
                a_gt = int(train_samples[i].get('action', 2))
                oh = np.zeros((action_dim,), dtype=np.float32)
                if 0 <= a_gt < action_dim:
                    oh[a_gt] = 1.0
                S_list.append(s); A_list.append(oh); R_list.append(1.0); C_list.append(1.0)
                negs = [a for a in range(action_dim) if a != a_gt]
                np.random.shuffle(negs)
                for a in negs[:negatives_per_pos]:
                    ohn = np.zeros((action_dim,), dtype=np.float32)
                    ohn[a] = 1.0
                    S_list.append(s); A_list.append(ohn); R_list.append(0.0); C_list.append(0.5)
            if not S_list:
                continue
            S = torch.from_numpy(np.stack(S_list)).float().to(device)
            A = torch.from_numpy(np.stack(A_list)).float().to(device)
            R = torch.from_numpy(np.asarray(R_list, dtype=np.float32)).view(-1, 1).to(device)
            C = torch.from_numpy(np.asarray(C_list, dtype=np.float32)).view(-1, 1).to(device)
            yield S, A, R, C

    for ep in range(int(epochs)):
        total, steps = 0.0, 0
        for S, A, R, C in iter_batches():
            optim_.zero_grad()
            pr, pc = irl.network(S, A)
            loss = mse(pr, R) + bce(pc, C)
            loss.backward()
            optim_.step()
            total += float(loss.item()); steps += 1
        print(f'[IRL] epoch {ep+1}/{epochs} loss={total/max(1,steps):.4f}')
    return irl


@torch.no_grad()
def irl_greedy_action(irl: RetentionIRLSystem, state: np.ndarray, tie_break: str = 'first') -> int:
    sdim = int(getattr(irl, 'state_dim', len(state)))
    adim = int(getattr(irl, 'action_dim', 3))
    s = np.asarray(state, dtype=np.float32)
    if s.shape[-1] < sdim:
        s = np.pad(s, (0, sdim - s.shape[-1]))
    elif s.shape[-1] > sdim:
        s = s[:sdim]
    dev = getattr(irl, 'device', torch.device('cpu'))
    s_t = torch.from_numpy(s).unsqueeze(0).to(dev)
    rewards = []
    for a in range(adim):
        one = np.zeros((adim,), dtype=np.float32); one[a] = 1.0
        a_t = torch.from_numpy(one).unsqueeze(0).to(dev)
        r, _ = irl.network(s_t, a_t)
        rewards.append(float(r.item()))
    arr = np.array(rewards, dtype=np.float32)
    idxs = np.flatnonzero(arr == arr.max())
    return int(idxs[0] if tie_break=='first' or len(idxs)==1 else np.random.choice(idxs))


def eval_irl_greedy_on_offline(irl: RetentionIRLSystem, eval_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds, gt = [], []
    for s in eval_samples:
        st = np.asarray(s.get('state', []), dtype=np.float32)
        if st.size == 0:
            preds.append(2)
        else:
            preds.append(irl_greedy_action(irl, st))
        gt.append(int(s.get('action', 2)))
    correct = [int(p==g) for p,g in zip(preds, gt)]
    overall = float(np.mean(correct)) if correct else 0.0
    # エピソード境界: done==True
    ep_rates, ep_exact = [], []
    start = 0
    for i,s in enumerate(eval_samples):
        if bool(s.get('done')):
            seg = correct[start:i+1]
            if seg:
                ep_rates.append(float(np.mean(seg)))
                ep_exact.append(1 if all(seg) else 0)
            start = i+1
    if start < len(eval_samples):
        seg = correct[start:]
        if seg:
            ep_rates.append(float(np.mean(seg)))
            ep_exact.append(1 if all(seg) else 0)
    # 混同行列
    cm = np.zeros((3,3), dtype=int)
    for g,p in zip(gt,preds):
        if 0<=g<3 and 0<=p<3: cm[g,p]+=1
    return {
        'num_steps': len(eval_samples),
        'num_episodes': len(ep_rates),
        'overall_action_match_rate': overall,
        'episode_action_match_rate_mean': float(np.mean(ep_rates) if ep_rates else 0.0),
        'episode_action_match_rate_std': float(np.std(ep_rates) if ep_rates else 0.0),
        'exact_episode_match_rate': float(np.mean(ep_exact) if ep_exact else 0.0),
        'confusion_matrix': cm.tolist(),
    }


def train_ppo_with_env(env, episodes: int = 200) -> Dict[str, Any]:
    agent = create_ppo_agent(env, config={
        'learning_rate': 1e-4,
        'buffer_size': 1024,
        'batch_size': 128,
        'mini_batch_size': 32,
        'ppo_epochs': 3,
    })
    action_names = ['reject','accept','wait']
    action_counts = {k:0 for k in action_names}
    ep_rewards, ep_lens = [], []
    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=42+ep)
        done=False; R=0.0; L=0
        while not done and L < getattr(env,'max_episode_length',100):
            a, logp, v = agent.select_action(obs, training=True)
            nxt, r, term, trunc, info = env.step(a)
            agent.store_experience(obs, a, float(r), float(v), float(logp), bool(term or trunc))
            name = action_names[a] if 0<=a<len(action_names) else 'unknown'
            if name in action_counts: action_counts[name]+=1
            R += float(r); L += 1; obs = nxt; done = bool(term or trunc)
        stats = agent.update()  # 1回更新
        ep_rewards.append(R); ep_lens.append(L)
    return {
        'episodes': int(episodes),
        'mean_reward': float(np.mean(ep_rewards) if ep_rewards else 0.0),
        'std_reward': float(np.std(ep_rewards) if ep_rewards else 0.0),
        'mean_length': float(np.mean(ep_lens) if ep_lens else 0.0),
        'action_distribution': action_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offline-train', type=str, default='outputs/offline/dataset_train.jsonl')
    ap.add_argument('--offline-eval', type=str, default='outputs/offline/dataset_eval.jsonl')
    ap.add_argument('--extended-data', type=str, default='data/extended_test_data.json')
    ap.add_argument('--cutoff', type=str, default='2023-04-01T00:00:00Z')
    ap.add_argument('--outdir', type=str, default='outputs/experiments')
    # IRL
    ap.add_argument('--irl-epochs', type=int, default=5)
    ap.add_argument('--irl-batch-size', type=int, default=1024)
    ap.add_argument('--irl-negs', type=int, default=2)
    ap.add_argument('--irl-hidden', type=int, default=128)
    ap.add_argument('--irl-lr', type=float, default=1e-3)
    # RL
    ap.add_argument('--rl-episodes', type=int, default=200)
    ap.add_argument('--irl-reward-mode', type=str, default='blend', choices=['blend','replace'])
    ap.add_argument('--irl-alpha', type=float, default=0.7)
    ap.add_argument('--engagement-bonus', type=float, default=0.0)
    args = ap.parse_args()

    out_root = Path(args.outdir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) IRL 学習
    train_samples = load_jsonl(Path(args.offline_train))
    if not train_samples:
        print(f'❌ offline train not found: {args.offline_train}')
        return 1
    irl = train_irl_offline(train_samples, epochs=args.irl_epochs, batch_size=args.irl_batch_size,
                             negatives_per_pos=args.irl_negs, hidden=args.irl_hidden, lr=args.irl_lr)
    irl_path = out_root / 'retention_irl_system.pth'
    irl.save_model(str(irl_path))
    print(f'✅ IRL saved -> {irl_path}')

    # 2) IRL 貪欲リプレイ評価
    eval_samples = load_jsonl(Path(args.offline_eval))
    if not eval_samples:
        print(f'⚠️ offline eval not found: {args.offline_eval} (skip IRL replay eval)')
        irl_report = None
    else:
        irl_report = eval_irl_greedy_on_offline(irl, eval_samples)
        (out_root/'irl_greedy_replay.json').write_text(json.dumps(irl_report, indent=2, ensure_ascii=False))
        print('📄 IRL greedy replay metrics saved')

    # 3) PPO 学習（IRL 報酬適用）
    base_env = ReviewAcceptanceEnvironment({
        'max_episode_length': 100,
        'max_queue_size': 10,
        'stress_threshold': 0.8,
        'use_random_initial_queue': False,
        'enable_random_new_reviews': False,
    })
    env = TimeSplitDataWrapper(base_env, data_path=args.extended_data, cutoff_iso=args.cutoff, phase='train')
    env = IRLRewardWrapper(env, irl_system=irl, mode=args.irl_reward_mode, alpha=float(args.irl_alpha),
                           engagement_bonus_weight=float(args.engagement_bonus), accept_action_id=1)
    rl_stats = train_ppo_with_env(env, episodes=args.rl_episodes)
    (out_root/'ppo_train_stats.json').write_text(json.dumps(rl_stats, indent=2, ensure_ascii=False))
    print('📄 PPO training stats saved')

    # 集約レポート
    report = {
        'irl_model': str(irl_path),
        'irl_replay': irl_report,
        'ppo_training': rl_stats,
        'config': {
            'offline_train': args.offline_train,
            'offline_eval': args.offline_eval,
            'extended_data': args.extended_data,
            'cutoff': args.cutoff,
            'irl': {
                'epochs': args.irl_epochs,
                'batch_size': args.irl_batch_size,
                'negatives_per_pos': args.irl_negs,
                'hidden': args.irl_hidden,
                'lr': args.irl_lr,
            },
            'rl': {
                'episodes': args.rl_episodes,
                'reward_mode': args.irl_reward_mode,
                'alpha': args.irl_alpha,
                'engagement_bonus': args.engagement_bonus,
            }
        }
    }
    (out_root/'experiment_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'✅ experiment done -> {out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
