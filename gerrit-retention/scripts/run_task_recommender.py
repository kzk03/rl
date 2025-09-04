#!/usr/bin/env python3
"""Run initial task recommender (behavioral cloning baseline).

Example:
  uv run python scripts/run_task_recommender.py \
     --changes data/processed/unified/all_reviews.json \
     --min-actions 40 --top-k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gerrit_retention.recommendation.task_recommendation_pipeline import (
    TaskRecommender,
    build_training_samples,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--min-actions', type=int, default=30)
    ap.add_argument('--horizon-days', type=int, default=30)
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--output', default='outputs/task_recommender')
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    changes_path = Path(args.changes)
    if not changes_path.exists():
        print(f'❌ changes file not found: {changes_path}')
        return 1
    print(f'📥 building samples from {changes_path}')
    samples = build_training_samples(changes_path, min_actions=args.min_actions, horizon_days=args.horizon_days)
    print(f'✅ samples built: {len(samples)}')
    if not samples:
        print('❌ no samples'); return 1
    rec = TaskRecommender()
    try:
        rec.fit(samples)
    except ValueError as e:
        print(f'⚠️ fit aborted: {e}'); return 1
    # take last sample state as demo
    state = samples[-1]['state']
    recs = rec.recommend_tasks(state, top_k=args.top_k)
    (out_dir/'recommendations.json').write_text(json.dumps({'state': state, 'recommendations': recs}, indent=2), encoding='utf-8')
    print('🎯 Top recommendations:')
    for r in recs:
        print(f"  {r['action']} -> {r['score']:.4f}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
