#!/usr/bin/env python3
"""Train IRL model for reviewer engagement and output continuation probabilities."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

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

from src.gerrit_retention.irl.maxent_binary_irl import MaxEntBinaryIRL

SEQ_PATH = Path('outputs/irl/reviewer_sequences.json')
OUT_DIR = Path('outputs/irl/reviewer_model')

def load_sequences():
    if not SEQ_PATH.exists():
        raise FileNotFoundError('Run extract_reviewer_sequences.py first')
    return json.loads(SEQ_PATH.read_text())

def flatten(seqs):
    trans = []
    for s in seqs:
        for tr in s.get('transitions', []):
            trans.append(tr)
    return trans

def main():
    seqs = load_sequences()
    transitions = flatten(seqs)
    model = MaxEntBinaryIRL()
    info = model.fit(transitions)
    y_true = [int(t['action']) for t in transitions]
    states = [t['state'] for t in transitions]
    probs = model.predict_proba(states)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    metrics = {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
        'positive_rate': float(np.mean(y_true)),
    }
    if len(set(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, probs)
        metrics['brier'] = brier_score_loss(y_true, probs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR/'weights.json').write_text(json.dumps(model.explain_weights(), indent=2), encoding='utf-8')
    (OUT_DIR/'training_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    (OUT_DIR/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    # Predict next engagement probability per reviewer using last transition state
    reviewer_preds = []
    for s in seqs:
        last_state = s['transitions'][-1]['state'] if s.get('transitions') else None
        if not last_state:
            continue
        p = model.predict_proba([last_state])[0]
        reviewer_preds.append({'reviewer_id': s['reviewer_id'], 'prob_next_engage': p, 'last_gap_days': last_state.get('gap_days')})
    (OUT_DIR/'reviewer_next_probabilities.json').write_text(json.dumps(reviewer_preds, indent=2), encoding='utf-8')
    print('Reviewer IRL weights:', model.explain_weights())
    print('Metrics:', metrics)
    print(f'Predictions saved: {OUT_DIR/"reviewer_next_probabilities.json"}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
