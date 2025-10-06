#!/usr/bin/env python3
"""Dump IRL model coefficients with feature names and simple stats.

Usage:
  uv run python scripts/analysis/irl_dump_coefficients.py --model outputs/task_assign_path_pair/irl_model.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--top', type=int, default=30)
    args = ap.parse_args()
    obj = json.loads(Path(args.model).read_text(encoding='utf-8'))
    theta = obj.get('theta') or []
    feats = obj.get('feature_order') or []
    if not theta or not feats:
        print(json.dumps({'error':'missing fields'}, ensure_ascii=False))
        return 1
    w = theta[:-1]; b = theta[-1]
    rows = []
    for name, val in zip(feats, w):
        rows.append({'feature': name, 'weight': val, 'abs': abs(val)})
    rows.sort(key=lambda r: r['abs'], reverse=True)
    out = {
        'bias': b,
        'top': rows[:args.top],
        'num_features': len(feats),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
