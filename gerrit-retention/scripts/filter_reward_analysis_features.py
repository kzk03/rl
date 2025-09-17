#!/usr/bin/env python3
"""Filter reward_analysis_* JSON files to drop features not in the current model.

Usage:
  uv run python scripts/filter_reward_analysis_features.py <path/to/reward_analysis.json> [more...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from gerrit_retention.recommendation.reviewer_invitation_ranking import (
        InvitationRankingModel,
    )
except Exception:
    InvitationRankingModel = None  # type: ignore


def filter_file(p: Path) -> bool:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ skip {p}: cannot read JSON ({e})")
        return False
    feats = set()
    if InvitationRankingModel is not None:
        feats = set(InvitationRankingModel().features)
    else:
        # Fallback hardlist in case import fails
        feats = {
            'reviewer_recent_reviews_7d',
            'reviewer_recent_reviews_30d',
            'reviewer_gap_days',
            'reviewer_total_reviews',
            'reviewer_proj_prev_reviews_30d',
            'reviewer_proj_share_30d',
            'change_current_invited_cnt',
            'reviewer_active_flag_30d',
            'reviewer_pending_reviews',
            'reviewer_night_activity_share_30d',
            'reviewer_overload_flag',
            'reviewer_workload_deviation_z',
            'match_off_specialty_flag',
            'off_specialty_recent_ratio',
            'reviewer_file_tfidf_cosine_30d',
        }
    if not isinstance(obj, dict) or 'features' not in obj or not isinstance(obj['features'], list):
        print(f"⚠️ skip {p}: unexpected schema")
        return False
    before = len(obj['features'])
    obj['features'] = [row for row in obj['features'] if row.get('feature') in feats]
    after = len(obj['features'])
    if after == before:
        print(f"ℹ️ no change: {p} ({after} features)")
        return False
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ filtered {p}: {before} -> {after} features")
    return True


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/filter_reward_analysis_features.py <json> [more...]")
        return 1
    changed = 0
    for arg in argv[1:]:
        path = Path(arg)
        if path.is_file():
            if filter_file(path):
                changed += 1
        else:
            print(f"⚠️ not a file: {path}")
    return 0 if changed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
