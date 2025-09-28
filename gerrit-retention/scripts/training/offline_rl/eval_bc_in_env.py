"""
BCポリシーを環境で評価（貪欲方策）

自己完結の評価スクリプトにするため、簡易MLPポリシー定義を内蔵。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gerrit_retention.rl_environment.review_env import ReviewAcceptanceEnvironment

# 定数（環境の観測次元・行動数に合わせる）
STATE_SIZE = 20
ACTION_SIZE = 3


class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def run_eval(model_path: str, episodes: int = 20) -> dict:
    ckpt = torch.load(model_path, map_location="cpu")
    hidden = ckpt.get("config", {}).get("hidden", 128)
    model = MLPPolicy(STATE_SIZE, hidden, ACTION_SIZE)
    model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    model.eval()

    env = ReviewAcceptanceEnvironment({
        'max_episode_length': 100,
        'max_queue_size': 5,
        'stress_threshold': 0.8,
    })

    rets, lens, adist = [], [], np.zeros(ACTION_SIZE, dtype=np.int64)
    for ep in range(episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        R = 0.0
        L = 0
        while not done and L < env.max_episode_length:
            with torch.no_grad():
                logits = model(torch.from_numpy(obs))
                a = int(logits.argmax().item())
            obs, r, term, trunc, info = env.step(a)
            R += float(r)
            L += 1
            adist[a] += 1
            done = bool(term or trunc)
        rets.append(R)
        lens.append(L)

    return {
        'episodes': episodes,
        'return_mean': float(np.mean(rets) if rets else 0.0),
        'return_std': float(np.std(rets) if rets else 0.0),
        'length_mean': float(np.mean(lens) if lens else 0.0),
        'action_dist': adist.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--out', type=str, default='outputs/offline')
    args = ap.parse_args()
    rep = run_eval(args.model, args.episodes)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(Path(args.out) / 'bc_eval_report.json').write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
