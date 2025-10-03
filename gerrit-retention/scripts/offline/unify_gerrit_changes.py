#!/usr/bin/env python3
"""
Raw Gerrit change JSONs -> unified all_reviews.json

Usage:
  uv run python scripts/offline/unify_gerrit_changes.py \
    --inputs data/raw/gerrit_changes/openstack_multi_5y_detail_*.json \
    --out data/processed/unified/all_reviews.json

Notes:
  - Inputs are JSON arrays (as produced by our extraction). This script streams them and writes a single JSON array.
  - Keeps records as-is. extract_reviewer_sequences.py expects fields like 'created' and 'reviewers' (with roles REVIEWER/CC).
"""
from __future__ import annotations

import argparse
import glob
import json
from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, Iterator


def _iter_json_array_file(p: Path) -> Iterator[Dict[str, Any]]:
    dec = JSONDecoder()
    with p.open('r', encoding='utf-8') as f:
        buf = ''
        # read to '['
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            buf += chunk
            i = 0
            n = len(buf)
            while i < n and buf[i].isspace():
                i += 1
            if i < n and buf[i] == '[':
                buf = buf[i+1:]
                break
        # elements
        while True:
            j = 0
            while j < len(buf) and buf[j].isspace():
                j += 1
            if j < len(buf) and buf[j] == ',':
                buf = buf[j+1:]
                j = 0
            while j < len(buf) and buf[j].isspace():
                j += 1
            if j < len(buf) and buf[j] == ']':
                return
            try:
                obj, idx = dec.raw_decode(buf[j:])
                yield obj
                buf = buf[j+idx:]
            except json.JSONDecodeError:
                more = f.read(65536)
                if not more:
                    return
                buf += more


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=str, required=True, help='Glob of input JSON arrays (extracted Gerrit changes)')
    ap.add_argument('--out', type=str, default='data/processed/unified/all_reviews.json')
    args = ap.parse_args()

    paths = [Path(p) for p in glob.glob(args.inputs)]
    if not paths:
        print(json.dumps({'error': 'no input files matched', 'pattern': args.inputs}, ensure_ascii=False))
        return 1
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with outp.open('w', encoding='utf-8') as wf:
        wf.write('[\n')
        first = True
        for p in sorted(paths):
            for obj in _iter_json_array_file(p):
                if not first:
                    wf.write(',\n')
                wf.write(json.dumps(obj, ensure_ascii=False))
                first = False
                count += 1
        wf.write('\n]\n')
    print(json.dumps({'out': str(outp), 'inputs': len(paths), 'records': count}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
