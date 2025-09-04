#!/usr/bin/env python3
"""Reviewer IRL モデル 時系列 hold-out 評価スクリプト

手順:
 1. reviewer_sequences.json (または指定ファイル) を読み込み
 2. 各 reviewer の transitions を時系列順に 0.8 : 0.2 (デフォルト) で分割
 3. 前半を学習、後半をテストとして MaxEntBinaryIRL を訓練
 4. テスト上で各種メトリクス + キャリブレーション (信頼度ビン) を算出
 5. outputs/irl/reviewer_holdout/ 以下に metrics.json, calibration.json 保存

使い方例:
  uv run scripts/evaluate_reviewer_irl_holdout.py --sequences outputs/irl/reviewer_sequences.json --test-ratio 0.2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.gerrit_retention.irl.maxent_binary_irl import MaxEntBinaryIRL  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Reviewer IRL holdout evaluation")
    ap.add_argument('--sequences', default='outputs/irl/reviewer_sequences.json')
    ap.add_argument('--test-ratio', type=float, default=0.2)
    ap.add_argument('--min-transitions', type=int, default=4, help='学習+評価に必要な最小遷移数')
    ap.add_argument('--output-dir', default='outputs/irl/reviewer_holdout')
    return ap.parse_args()


def load_sequences(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"sequences not found: {path}")
    return json.loads(path.read_text())


def split_transitions(seqs: List[Dict[str, Any]], test_ratio: float, min_transitions: int):
    train, test = [], []
    for s in seqs:
        trs = s.get('transitions', [])
        if len(trs) < min_transitions:
            continue
        cut = max(1, int(len(trs)*(1-test_ratio)))
        cut = min(cut, len(trs)-1)  # ensure at least 1 test
        train.extend(trs[:cut])
        test.extend(trs[cut:])
    return train, test


def flatten_states_actions(trs):
    states = [t['state'] for t in trs]
    actions = [int(t['action']) for t in trs]
    return states, actions


def calibration_bins(y_true, probs, n_bins=10):
    bins = []
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    edges = np.linspace(0, 1, n_bins+1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins-1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            bins.append({'bin': i, 'lo': float(lo), 'hi': float(hi), 'count': 0, 'pred_mean': None, 'true_rate': None})
            continue
        pred_mean = float(probs[mask].mean())
        true_rate = float(y_true[mask].mean())
        bins.append({'bin': i, 'lo': float(lo), 'hi': float(hi), 'count': int(mask.sum()), 'pred_mean': pred_mean, 'true_rate': true_rate, 'gap': pred_mean-true_rate})
    return bins


def main():
    args = parse_args()
    seqs = load_sequences(Path(args.sequences))
    train_trs, test_trs = split_transitions(seqs, args.test_ratio, args.min_transitions)
    if not train_trs or not test_trs:
        print('❌ insufficient transitions after split'); return 1
    model = MaxEntBinaryIRL()
    model.fit(train_trs)
    # evaluate train (optional) & test
    _, y_train = flatten_states_actions(train_trs)
    states_test, y_test = flatten_states_actions(test_trs)
    probs_test = model.predict_proba(states_test)
    preds_test = [1 if p >= 0.5 else 0 for p in probs_test]
    metrics = {
        'train_count': len(train_trs),
        'test_count': len(test_trs),
        'positive_rate_test': float(np.mean(y_test)),
        'accuracy': accuracy_score(y_test, preds_test),
        'precision': precision_score(y_test, preds_test, zero_division=0),
        'recall': recall_score(y_test, preds_test, zero_division=0),
        'f1': f1_score(y_test, preds_test, zero_division=0),
    }
    if len(set(y_test)) > 1:
        metrics['auc'] = roc_auc_score(y_test, probs_test)
        metrics['brier'] = brier_score_loss(y_test, probs_test)
    else:
        metrics['auc'] = None
        metrics['brier'] = None
    calib = calibration_bins(y_test, probs_test, n_bins=10)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (out_dir/'calibration.json').write_text(json.dumps(calib, indent=2), encoding='utf-8')
    print('✅ Reviewer IRL holdout metrics')
    for k,v in metrics.items():
        print(f'  {k}: {v}')
    print(f'💾 Saved metrics -> {out_dir/"metrics.json"}')
    print(f'💾 Saved calibration -> {out_dir/"calibration.json"}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
