"""
オフラインRLデータセットの作成スクリプト

使用例:
  uv run python training/offline_rl/build_offline_data.py --data data/extended_test_data.json --cutoff 2023-04-01T00:00:00Z
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gerrit_retention.offline.offline_dataset import build_offline_datasets


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", type=str, default="data/extended_test_data.json")
  p.add_argument("--cutoff", type=str, default="2023-04-01T00:00:00Z")
  p.add_argument("--out", type=str, default="outputs/offline")
  p.add_argument("--include-non-review", action="store_true", help="非レビューイベントをWAITとして含める")
  args = p.parse_args()
  meta = build_offline_datasets(
    args.data,
    args.cutoff,
    args.out,
    include_non_review_as_wait=args.include_non_review,
  )
  print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
