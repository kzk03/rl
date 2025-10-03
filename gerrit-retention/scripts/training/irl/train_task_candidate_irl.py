#!/usr/bin/env python3
"""
Train a simple IRL-like utility model for Task×Candidate selection using multinomial logistic regression.

Inputs: tasks_train.jsonl (AssignmentTask JSONL)
Output: irl_model.json (theta, scaler)

We fit a linear utility u = w^T x + b per candidate and use softmax over candidates within each task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _read_tasks(jsonl_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    tasks: List[Dict[str, Any]] = []
    feat_keys: List[str] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            obj = json.loads(s)
            cands = obj.get('candidates', [])
            tasks.append(obj)
            if cands and not feat_keys:
                feat_keys = list(cands[0]['features'].keys())
    return tasks, feat_keys

def _stack_task_candidates(tasks: List[Dict[str, Any]], feat_keys: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for t in tasks:
        cands = t.get('candidates', [])
        pos = set(t.get('positive_reviewer_ids') or [])
        if not cands:
            continue
        X = np.array([[float(c['features'].get(k, 0.0)) for k in feat_keys] for c in cands], dtype=np.float32)
        y = np.array([1 if c['reviewer_id'] in pos else 0 for c in cands], dtype=np.int64)
        # Skip tasks with all-zero labels to stabilize fitting (optional)
        if y.sum() == 0:
            continue
        X_list.append(X)
        y_list.append(y)
    return X_list, y_list

def _standardize(X_list: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    # Flatten to compute mean/scale, then reshape back
    if not X_list:
        return X_list, {'mean': None, 'scale': None}
    X_all = np.concatenate(X_list, axis=0)
    mean = X_all.mean(axis=0)
    scale = X_all.std(axis=0)
    scale[scale < 1e-6] = 1.0
    out = [ (X - mean) / scale for X in X_list ]
    return out, {'mean': mean.tolist(), 'scale': scale.tolist()}

def _fit_softmax(X_list: List[np.ndarray], y_list: List[np.ndarray], iters: int = 200, lr: float = 0.1, reg: float = 1e-4) -> np.ndarray:
    # Linear utility with intercept: u = Wx + b; implement as theta = [w; b]
    D = X_list[0].shape[1]
    theta = np.zeros(D + 1, dtype=np.float64)
    for it in range(iters):
        g = np.zeros_like(theta)
        n_tasks = 0
        for X, y in zip(X_list, y_list):
            n_tasks += 1
            # add intercept
            Xext = np.concatenate([X.astype(np.float64), np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
            u = Xext @ theta
            u = u - np.max(u)  # stability
            exps = np.exp(u)
            Z = np.maximum(1e-9, exps.sum())
            p = exps / Z
            # We treat y as one-hot over candidates; if multiple positives, normalize equally
            if y.sum() > 0:
                target = y.astype(np.float64) / float(y.sum())
            else:
                target = np.zeros_like(p)
            # gradient of cross-entropy wrt theta: Xext^T (p - target)
            g += Xext.T @ (p - target)
        if n_tasks == 0:
            break
        # L2 regularization on w (exclude intercept lightly)
        reg_vec = np.concatenate([theta[:-1], np.array([0.0])])
        g = g / n_tasks + reg * reg_vec
        theta -= lr * g
    return theta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-tasks', type=str, required=True)
    ap.add_argument('--out', type=str, default='outputs/task_assign/irl_model.json')
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--reg', type=float, default=1e-4)
    args = ap.parse_args()

    tasks, feat_keys = _read_tasks(Path(args.train_tasks))
    X_list, y_list = _stack_task_candidates(tasks, feat_keys)
    if not X_list:
        print(json.dumps({'error': 'no tasks with positives'}, ensure_ascii=False))
        return 1
    Xs, scaler = _standardize(X_list)
    theta = _fit_softmax(Xs, y_list, iters=int(args.iters), lr=float(args.lr), reg=float(args.reg))
    out = {'theta': theta.tolist(), 'feature_order': feat_keys, 'scaler': scaler}
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(outp), 'dims': len(theta)-1, 'tasks_used': len(Xs)}, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
