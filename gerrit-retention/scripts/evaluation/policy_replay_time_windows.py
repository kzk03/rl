#!/usr/bin/env python3
"""
ポリシーのリプレイ一致率を、
  - IRL学習と一致する期間（train: <= cutoff）
  - cutoff 以降の累積ウィンドウ（例: +1m, +3m, +6m, +12m）
で比較評価するスクリプト。

入力:
  --outdir: オフライン出力ディレクトリ（offline_dataset_meta.json, dataset_train.jsonl, dataset_eval.jsonl を含む）
  --policy: 保存済みポリシー .pt（stable_ppo_training.py 互換）
  --windows: カンマ区切りの相対ウィンドウ（単位: d=日, w=週, m=月(=30日近似)）例: "1m,3m,6m,12m"
  --collapse-wait-reject: 評価時に wait(2) と reject(0) を非受諾(0)に統合した2クラス指標も出力

出力:
  JSON に train, 各 horizon のメトリクス（multi/binary）を保存/表示
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.fromisoformat(ts)


def _add_relative(base: datetime, token: str) -> datetime:
    token = token.strip().lower()
    if token.endswith("d"):
        n = int(token[:-1])
        return base + timedelta(days=n)
    if token.endswith("w"):
        n = int(token[:-1])
        return base + timedelta(weeks=n)
    if token.endswith("m"):
        # 月は 30 日近似（簡易）
        n = int(token[:-1])
        return base + timedelta(days=30 * n)
    if token.endswith("y"):
        n = int(token[:-1])
        return base + timedelta(days=365 * n)
    # デフォルト: 日
    return base + timedelta(days=int(token))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for ln in path.read_text().splitlines():
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
    # wait(2), reject(0) -> 0, accept(1) -> 1
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
                seg = [int(pr[j] == gt[j]) for j in seg_idx]
                ep_rates.append(float(np.mean(seg)))
                ep_exact.append(1 if all(seg) else 0)
            start = i + 1
    if start < len(samples):
        seg_idx = [j for j in mask if start <= j < len(samples)]
        if seg_idx:
            seg = [int(pr[j] == gt[j]) for j in seg_idx]
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


def _filter_by_time(samples: List[Dict[str, Any]], start: Optional[datetime], end: Optional[datetime]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in samples:
        ts = s.get('timestamp') or s.get('t')
        if not ts:
            continue
        when = _parse_iso(str(ts))
        if start and when < start:
            continue
        if end and when > end:
            continue
        out.append(s)
    return out


def _predict_all(model: nn.Module, obs_dim: int, samples: List[Dict[str, Any]]) -> List[int]:
    preds: List[int] = []
    for s in samples:
        st = np.asarray(s.get('state', []), dtype=np.float32)
        if st.size < obs_dim:
            st = np.pad(st, (0, obs_dim - st.size))
        elif st.size > obs_dim:
            st = st[:obs_dim]
        preds.append(greedy_policy_action(model, st))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=str, required=True, help='offline_dataset_meta.json を含むディレクトリ')
    ap.add_argument('--policy', type=str, required=True, help='保存済みポリシー .pt')
    ap.add_argument('--windows', type=str, default='1m,3m,6m,12m', help='相対ウィンドウ（累積, 例: 1m,3m,6m,12m）')
    ap.add_argument('--collapse-wait-reject', action='store_true')
    ap.add_argument('--out', type=str, default='outputs/irl/policy_replay_time_windows.json')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    meta_path = outdir / 'offline_dataset_meta.json'
    train_path = outdir / 'dataset_train.jsonl'
    eval_path = outdir / 'dataset_eval.jsonl'
    if not meta_path.exists():
        print(f'❌ meta not found: {meta_path}')
        return 1
    meta = json.loads(meta_path.read_text())
    cutoff = _parse_iso(meta.get('cutoff'))

    train_samples = load_jsonl(train_path)
    eval_samples = load_jsonl(eval_path)
    if not train_samples and not eval_samples:
        print('❌ no samples to evaluate')
        return 1

    # ポリシー読込
    ckpt = torch.load(args.policy, map_location='cpu', weights_only=False)
    obs_dim = int(ckpt.get('obs_dim'))
    action_dim = int(ckpt.get('action_dim'))
    model = build_sequential_policy(obs_dim, action_dim)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # 学習期間（<=cutoff）: train をそのまま使う
    train_preds = _predict_all(model, obs_dim, train_samples)
    train_metrics = compute_metrics(train_samples, train_preds)
    result: Dict[str, Any] = {
        'outdir': str(outdir),
        'policy': str(args.policy),
        'cutoff': cutoff.isoformat(),
        'train_metrics': train_metrics,
        'horizons': {},
    }
    if args.collapse_wait_reject:
        result['train_metrics_binary'] = compute_metrics_binary(train_samples, train_preds)

    # 累積ウィンドウ: (cutoff, cutoff + h] のサンプルで評価
    tokens = [t for t in (args.windows.split(',') if args.windows else []) if t.strip()]
    for tok in tokens:
        end = _add_relative(cutoff, tok)
        sub = _filter_by_time(eval_samples, start=cutoff, end=end)
        preds = _predict_all(model, obs_dim, sub)
        metrics = compute_metrics(sub, preds)
        entry: Dict[str, Any] = {'until': end.isoformat(), 'metrics': metrics}
        if args.collapse_wait_reject:
            entry['metrics_binary'] = compute_metrics_binary(sub, preds)
        result['horizons'][tok] = entry

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
