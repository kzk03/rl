#!/usr/bin/env python3
"""Run reviewer invitation ranking evaluation.

Generates Cartesian (change x reviewer) candidates with negative sampling and trains a
logistic ranking baseline.

Example:
  uv run python scripts/run_reviewer_invitation_ranking.py \
    --changes data/processed/unified/all_reviews.json \
    --min-total-reviews 20 --max-neg-per-pos 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gerrit_retention.recommendation.reviewer_invitation_ranking import (
    InvitationRankingBuildConfig,
    build_invitation_ranking_samples,
    evaluate_invitation_ranking,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--min-total-reviews', type=int, default=20)
    ap.add_argument('--recent-days', type=int, default=30)
    ap.add_argument('--max-neg-per-pos', type=int, default=5)
    ap.add_argument('--hard-fraction', type=float, default=0.5)
    ap.add_argument('--output', default='outputs/reviewer_invitation_ranking')
    return ap.parse_args()


def main():
    args = parse_args()
    changes_path = Path(args.changes)
    if not changes_path.exists():
        print(f'❌ changes file not found: {changes_path}')
        return 1
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = InvitationRankingBuildConfig(
        min_total_reviews=args.min_total_reviews,
        recent_days=args.recent_days,
        max_neg_per_pos=args.max_neg_per_pos,
        hard_fraction=args.hard_fraction,
    )
    print(f'📥 building ranking samples from {changes_path}')
    samples = build_invitation_ranking_samples(changes_path, cfg)
    print(f'✅ built samples: {len(samples)} (pos rate ~ {sum(s["label"] for s in samples)/max(1,len(samples)):.3f})')
    if len(samples) < 50:
        print('⚠️ insufficient samples (<50)')
        return 1
    metrics, model, test, probs = evaluate_invitation_ranking(samples)
    print('📊 Metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')
    # Save outputs
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    detail = []
    for s, p in zip(test, probs):
        st = s['state'].copy()
        st.update({'label': s['label'], 'prob': float(p), 'ts': s['ts']})
        detail.append(st)
    (out_dir / 'test_predictions.json').write_text(json.dumps(detail[:1000], indent=2), encoding='utf-8')
    print(f'💾 wrote outputs to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
