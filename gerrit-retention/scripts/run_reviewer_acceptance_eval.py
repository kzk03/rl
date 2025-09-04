#!/usr/bin/env python3
"""Run reviewer invitation acceptance evaluation.

Example:
  uv run python scripts/run_reviewer_acceptance_eval.py \
    --changes data/processed/unified/all_reviews.json \
    --min-reviews 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gerrit_retention.recommendation.reviewer_acceptance import (
    build_reviewer_invitation_samples,
    evaluate_acceptance_model,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--min-reviews', type=int, default=20)
    ap.add_argument('--output', default='outputs/reviewer_acceptance')
    ap.add_argument('--test-ratio', type=float, default=0.2)
    return ap.parse_args()


def main():
    args = parse_args()
    changes_path = Path(args.changes)
    if not changes_path.exists():
        print(f"❌ changes file not found: {changes_path}")
        return 1
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📥 building reviewer invitation samples from {changes_path}")
    samples = build_reviewer_invitation_samples(changes_path, min_reviews=args.min_reviews)
    print(f"✅ samples built (after min_reviews filter): {len(samples)}")
    if len(samples) < 30:
        print('⚠️ not enough samples (<30) for evaluation')
        return 1
    metrics, model, test_samples, probs = evaluate_acceptance_model(samples)
    print('📊 Metrics:')
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    # attach per-sample output (subset)
    detailed = []
    for s, p in zip(test_samples, probs):
        row = {
            'ts': s['ts'],
            'reviewer_id': s['state']['reviewer_id'],
            'label': s['label'],
            'prob': float(p),
        }
        detailed.append(row)
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (out_dir / 'test_predictions.json').write_text(json.dumps(detailed[:500], indent=2), encoding='utf-8')
    print(f"💾 wrote metrics + test_predictions to {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
