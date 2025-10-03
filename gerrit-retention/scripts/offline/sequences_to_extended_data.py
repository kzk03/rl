#!/usr/bin/env python3
"""
reviewer_sequences.json(.gz|.jsonl|.jsonl.gz) -> extended_test_data.json 変換スクリプト（本番対応）

入力:
    --input: reviewer_sequences（JSON配列 または JSONL）
                     レコード構造: {reviewer_id|developer_id, transitions:[{t|timestamp, action(0/1/2), state:{...}}]}
出力:
    --out: extended_test_data.json（配列）

マッピング方針:
    action==1 -> review(+1)
    action==0 -> review(-1)
    action==2 -> commit (wait相当, scoreなし)
    lines_added/deleted や message は state ヒントから簡易合成

本番向けの拡張:
    - JSON 配列ストリーミングパース（巨大ファイル対応、追加依存なし）
    - JSONL / .gz 圧縮対応
    - 逐次書き出し（出力全体をメモリに保持しない）
    - 可変プロジェクト名、空ヒストリのスキップなどのオプション
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
from datetime import datetime
from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, Iterator, List


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace('Z','+00:00')).replace(tzinfo=None)
    except Exception:
        return datetime.fromisoformat(ts)


def _heuristic_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    # state ヒントから規模を簡易合成
    gap_days = float(state.get('gap_days', state.get('gap', 3) or 3))
    activity30 = float(state.get('activity_30d', state.get('activity30', 1.0) or 1.0))
    activity90 = float(state.get('activity_90d', state.get('activity90', max(1.0, activity30)) or max(1.0, activity30)))
    workload = float(state.get('workload_level', state.get('workload', 0.2) or 0.2))
    total_lines = int(min(2000, max(10, 50 * activity30 + 20 * (activity90 - activity30) + 10)))
    added = int(total_lines * 0.7)
    deleted = max(0, total_lines - added)
    return {
        'lines_added': added,
        'lines_deleted': deleted,
        'message': f"auto(gen) gap={gap_days:.1f}d workload={workload:.2f}",
    }

def _open_maybe_gzip(p: Path):
    if str(p).endswith('.gz'):
        return gzip.open(p, 'rt', encoding='utf-8')
    return open(p, 'rt', encoding='utf-8')


def _iter_json_array(f) -> Iterator[Dict[str, Any]]:
    """巨大 JSON 配列をストリームで1要素ずつデコードする。
    追加依存なしで JSONDecoder.raw_decode を用いる簡易実装。
    """
    dec = JSONDecoder()
    buf = ''
    # 前方の空白と '[' をスキップ
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
        # 先頭に '[' が来るまで読み続ける
    # 要素を順次読む
    first = True
    while True:
        # カンマや空白をスキップ
        j = 0
        while j < len(buf) and buf[j].isspace():
            j += 1
        if j < len(buf) and buf[j] == ',':
            buf = buf[j+1:]
            j = 0
        while j < len(buf) and buf[j].isspace():
            j += 1
        if j < len(buf) and buf[j] == ']':
            # 配列終端
            return
        # 要素デコードを試行
        try:
            obj, idx = dec.raw_decode(buf[j:])
            yield obj
            buf = buf[j+idx:]
            continue
        except json.JSONDecodeError:
            # 追い読み
            more = f.read(65536)
            if not more:
                # 異常終了
                return
            buf += more


def _iter_records(path: Path):
    with _open_maybe_gzip(path) as f:
        name = os.path.basename(str(path)).lower()
        if name.endswith('.jsonl') or name.endswith('.jsonl.gz'):
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception:
                    continue
        else:
            # JSON 配列をストリームで読む
            for obj in _iter_json_array(f):
                if isinstance(obj, dict):
                    yield obj


def convert_sequences_to_extended(seq_path: Path, out_path: Path, projects: List[str], drop_empty: bool = True) -> int:
    # 逐次書き出し
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as wf:
        wf.write('[\n')
        count = 0
        first = True
        for rec in _iter_records(seq_path):
            rid = rec.get('reviewer_id') or rec.get('developer_id') or 'unknown@example.com'
            trans = rec.get('transitions') or []
            hist: List[Dict[str, Any]] = []
            for tr in trans:
                ts = tr.get('t') or tr.get('timestamp')
                if not ts:
                    continue
                try:
                    when = _parse_iso(str(ts))
                except Exception:
                    # ISO 以外の形式はスキップ
                    continue
                a = int(tr.get('action', 2))
                st = tr.get('state', {}) or {}
                payload = _heuristic_payload(st)
                if a == 1:
                    hist.append({
                        'timestamp': when.isoformat(),
                        'type': 'review',
                        'message': payload['message'],
                        'score': 1,
                        'lines_added': payload['lines_added'],
                        'lines_deleted': payload['lines_deleted'],
                    })
                elif a == 0:
                    hist.append({
                        'timestamp': when.isoformat(),
                        'type': 'review',
                        'message': payload['message'],
                        'score': -1,
                        'lines_added': payload['lines_added'],
                        'lines_deleted': payload['lines_deleted'],
                    })
                else:
                    hist.append({
                        'timestamp': when.isoformat(),
                        'type': 'commit',
                        'message': payload['message'],
                        'lines_added': payload['lines_added'],
                        'lines_deleted': payload['lines_deleted'],
                    })
            hist.sort(key=lambda x: x['timestamp'])
            if drop_empty and not hist:
                continue
            dev_obj = {
                'developer': {
                    'developer_id': rid,
                    'projects': projects or ['historical-project'],
                },
                'activity_history': hist,
            }
            if not first:
                wf.write(',\n')
            wf.write(json.dumps(dev_obj, ensure_ascii=False))
            first = False
            count += 1
        wf.write('\n]\n')
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='outputs/irl/reviewer_sequences.json', help='入力: JSON/JSONL（.gz対応）')
    ap.add_argument('--out', type=str, default='data/extended_test_data.json', help='出力: 拡張データ(JSON配列)')
    ap.add_argument('--projects', type=str, default='historical-project', help='カンマ区切りのプロジェクト名')
    ap.add_argument('--keep-empty', action='store_true', help='空の activity_history も出力に含める')
    args = ap.parse_args()
    projs = [p.strip() for p in (args.projects or '').split(',') if p.strip()]
    n = convert_sequences_to_extended(Path(args.input), Path(args.out), projs, drop_empty=not args.keep_empty)
    print(json.dumps({'input': args.input, 'out': args.out, 'developers': n, 'projects': projs}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
