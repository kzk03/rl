#!/usr/bin/env python3
"""Grid search temperature to minimize ECE using replay evaluation quick loop.

Assumes policy was trained (or we just use theta softmax directly) – here we re-score tasks by IRL softmax
probabilities with different temperatures and compute ECE on top-1 argmax decisions.

Usage:
  uv run python scripts/analysis/temperature_tune_irl.py \
    --tasks outputs/task_assign_path_pair/tasks_eval.jsonl \
    --model outputs/task_assign_path_pair/irl_model.json \
    --temps 0.6,0.8,1.0,1.2,1.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _read_tasks(p: Path) -> List[Dict[str, Any]]:
    tasks = []
    with open(p,'r',encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            try:
                tasks.append(json.loads(s))
            except Exception:
                pass
    return tasks

def _score_task(task: Dict[str, Any], theta: np.ndarray, scaler: Dict[str, Any], feat_order: List[str], temp: float) -> Tuple[float,float]:
    cands = task.get('candidates') or []
    if not cands: return 0.0, 0.0
    utils = []
    for c in cands:
        feats = c.get('features') or {}
        x = np.array([float(feats.get(f,0.0)) for f in feat_order], dtype=np.float64)
        if scaler and scaler.get('mean') is not None:
            mean = np.array(scaler.get('mean'), dtype=np.float64)
            scale = np.array(scaler.get('scale'), dtype=np.float64)
            if mean.shape[0] == x.shape[0]:
                x = (x - mean)/np.maximum(scale,1e-8)
        xext = np.concatenate([x, np.array([1.0])])
        u = float(np.dot(xext, theta))
        utils.append(u)
    if not utils:
        return 0.0,0.0
    arr = np.array(utils, dtype=np.float64) / max(1e-6,temp)
    arr = arr - np.max(arr)
    exps = np.exp(arr)
    probs = exps / np.maximum(1e-12, exps.sum())
    top_idx = int(np.argmax(probs))
    pos_set = set(task.get('positive_reviewer_ids') or [])
    cand_ids = [c.get('reviewer_id') for c in cands]
    correct = 1 if cand_ids[top_idx] in pos_set else 0
    conf = float(probs[top_idx])
    return conf, float(correct)

def _ece(confs: List[float], corrects: List[float], bins: int = 10) -> float:
    if not confs: return 0.0
    c = np.array(confs); y = np.array(corrects)
    idx = np.clip((c * bins).astype(int), 0, bins-1)
    ece = 0.0
    for b in range(bins):
        m = (idx==b)
        if not np.any(m): continue
        conf_bin = c[m].mean(); acc_bin = y[m].mean(); w = m.mean()
        ece += w * abs(acc_bin - conf_bin)
    return float(ece)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--temps', type=str, default='0.6,0.8,1.0,1.2,1.4')
    ap.add_argument('--max-tasks', type=int, default=50000, help='Cap tasks for speed (after filtering positives presence optional)')
    args = ap.parse_args()
    obj = json.loads(Path(args.model).read_text(encoding='utf-8'))
    theta = np.array(obj.get('theta'), dtype=np.float64)
    feat_order = obj.get('feature_order') or []
    scaler = obj.get('scaler') or {}
    tasks = _read_tasks(Path(args.tasks))[:args.max_tasks]
    temps = [float(t) for t in args.temps.split(',') if t.strip()]
    results = []
    for T in temps:
        confs=[]; corr=[]
        for task in tasks:
            conf, correct = _score_task(task, theta, scaler, feat_order, T)
            confs.append(conf); corr.append(correct)
        ece = _ece(confs, corr, bins=15)
        acc = float(np.mean(corr)) if corr else 0.0
        results.append({'temperature': T, 'ECE': ece, 'top1_acc': acc})
    best = min(results, key=lambda r: r['ECE']) if results else None
    out = {'results': results, 'best': best}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
