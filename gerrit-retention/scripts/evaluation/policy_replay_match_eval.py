#!/usr/bin/env python3
"""
保存済みポリシーのリプレイ一致率評価

対象: stable_ppo_training.py が保存するポリシー (outputs/policies/stable_ppo_policy_*.pt)

入力:
- --offline: オフライン評価JSONL (e.g., outputs/offline/dataset_eval.jsonl)
- --policy:  保存済みポリシー .pt

出力:
- JSON (overall_action_match_rate, episode_action_match_rate_mean/std, exact_episode_match_rate, confusion_matrix)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def build_sequential_policy(obs_dim: int, action_dim: int) -> nn.Sequential:
    # stable_ppo_training.py の保存フォーマットに合わせた簡易構成
    return nn.Sequential(
        nn.Linear(obs_dim, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, action_dim),
        nn.Softmax(dim=-1)
    )


@torch.no_grad()
def greedy_policy_action(model: nn.Module, state: np.ndarray) -> int:
    if state.ndim == 1:
        x = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
    else:
        x = torch.from_numpy(state.astype(np.float32))
    probs = model(x)
    if torch.isnan(probs).any():
        probs = torch.ones_like(probs) / probs.shape[-1]
    return int(torch.argmax(probs, dim=-1).item())


def compute_metrics(samples: List[Dict[str, Any]], preds: List[int]) -> Dict[str, Any]:
    gt = [int(s.get('action', -1)) for s in samples]
    correct = [int(p == g) for p, g in zip(preds, gt)]
    overall = float(np.mean(correct)) if correct else 0.0
    # エピソード境界: done==True
    ep_rates, ep_exact = [], []
    start = 0
    for i, s in enumerate(samples):
        if bool(s.get('done')):
            seg = correct[start:i+1]
            if seg:
                ep_rates.append(float(np.mean(seg)))
                ep_exact.append(1 if all(seg) else 0)
            start = i + 1
    if start < len(samples):
        seg = correct[start:]
        if seg:
            ep_rates.append(float(np.mean(seg)))
            ep_exact.append(1 if all(seg) else 0)
    cm = np.zeros((3, 3), dtype=int)
    for g, p in zip(gt, preds):
        if 0 <= g < 3 and 0 <= p < 3:
            cm[g, p] += 1
    return {
        'num_steps': len(samples),
        'num_episodes': len(ep_rates),
        'overall_action_match_rate': overall,
        'episode_action_match_rate_mean': float(np.mean(ep_rates) if ep_rates else 0.0),
        'episode_action_match_rate_std': float(np.std(ep_rates) if ep_rates else 0.0),
        'exact_episode_match_rate': float(np.mean(ep_exact) if ep_exact else 0.0),
        'confusion_matrix': cm.tolist(),
    }


def compute_metrics_binary(samples: List[Dict[str, Any]], preds: List[int]) -> Dict[str, Any]:
    """wait(2) と reject(0) を 0（非受諾）に統合し、accept(1) を 1 にした2クラス評価。

    - action mapping: {0 -> 0, 1 -> 1, 2 -> 0}
    - confusion_matrix は 2x2（[gt][pred]）
    """
    gt_all = [int(s.get('action', -1)) for s in samples]
    gt = [0 if g in (0, 2) else (1 if g == 1 else -1) for g in gt_all]
    pr = [0 if p in (0, 2) else (1 if p == 1 else -1) for p in preds]
    mask = [i for i, (g, p) in enumerate(zip(gt, pr)) if g in (0, 1) and p in (0, 1)]
    if not mask:
        return {
            'num_steps': 0,
            'num_episodes': 0,
            'overall_action_match_rate': 0.0,
            'episode_action_match_rate_mean': 0.0,
            'episode_action_match_rate_std': 0.0,
            'exact_episode_match_rate': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]],
        }
    gt_f = [gt[i] for i in mask]
    pr_f = [pr[i] for i in mask]
    correct = [int(p == g) for p, g in zip(pr_f, gt_f)]
    overall = float(np.mean(correct)) if correct else 0.0
    # エピソード境界: done==True（元サンプルに従う）
    ep_rates, ep_exact = [], []
    start = 0
    for i, s in enumerate(samples):
        if bool(s.get('done')):
            seg_idx = [j for j in mask if start <= j <= i]
            if seg_idx:
                seg = [int(pr[idx] == gt[idx]) for idx in seg_idx]
                ep_rates.append(float(np.mean(seg)))
                ep_exact.append(1 if all(seg) else 0)
            start = i + 1
    if start < len(samples):
        seg_idx = [j for j in mask if start <= j < len(samples)]
        if seg_idx:
            seg = [int(pr[idx] == gt[idx]) for idx in seg_idx]
            ep_rates.append(float(np.mean(seg)))
            ep_exact.append(1 if all(seg) else 0)
    cm = np.zeros((2, 2), dtype=int)
    for g, p in zip(gt_f, pr_f):
        cm[g, p] += 1
    return {
        'num_steps': len(mask),
        'num_episodes': len(ep_rates),
        'overall_action_match_rate': overall,
        'episode_action_match_rate_mean': float(np.mean(ep_rates) if ep_rates else 0.0),
        'episode_action_match_rate_std': float(np.std(ep_rates) if ep_rates else 0.0),
        'exact_episode_match_rate': float(np.mean(ep_exact) if ep_exact else 0.0),
        'confusion_matrix': cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offline', type=str, required=True)
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--out', type=str, default='outputs/policy_replay_report.json')
    ap.add_argument('--collapse-wait-reject', action='store_true',
                    help='評価時に wait(2) と reject(0) を統合して 2 クラス（非受諾/受諾）として追加メトリクスを出力')
    args = ap.parse_args()

    samples = load_jsonl(Path(args.offline))
    if not samples:
        print(f'❌ no offline samples: {args.offline}')
        return 1
    # PyTorch 2.6+ defaults to weights_only=True which may fail for dict checkpoints.
    # We saved this checkpoint ourselves, so it's safe to allow full unpickling here.
    ckpt = torch.load(args.policy, map_location='cpu', weights_only=False)
    obs_dim = int(ckpt.get('obs_dim'))
    action_dim = int(ckpt.get('action_dim'))
    model = build_sequential_policy(obs_dim, action_dim)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    preds: List[int] = []
    for s in samples:
        st = np.asarray(s.get('state', []), dtype=np.float32)
        # 次元チェック（不足分は0埋め、過剰なら切り詰め）
        if st.size < obs_dim:
            st = np.pad(st, (0, obs_dim - st.size))
        elif st.size > obs_dim:
            st = st[:obs_dim]
        preds.append(greedy_policy_action(model, st))

    metrics = compute_metrics(samples, preds)
    rep_metrics: Dict[str, Any] = metrics
    if args.collapse_wait_reject:
        binary_metrics = compute_metrics_binary(samples, preds)
        rep_metrics = {
            'multiclass_metrics': metrics,
            'binary_metrics': binary_metrics,
        }
    rep = {
        'offline_path': args.offline,
        'policy_path': args.policy,
        'metrics': rep_metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
