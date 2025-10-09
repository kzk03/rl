#!/usr/bin/env python3
"""レビュアー推薦後の着手確率を推定するためのデータセットを生成します。"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.reviewer_adoption.common import ensure_numeric_state


def _open_path(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _parse_iso(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _iter_sequences(path: Path) -> Iterable[Dict[str, Any]]:
    with _open_path(path) as f:
        data = json.load(f)
    for rec in data:
        yield rec


def _build_records(seq: Dict[str, Any]) -> Iterable[Tuple[datetime, Dict[str, Any]]]:
    reviewer_id = seq.get("reviewer_id")
    for tr in seq.get("transitions", []) or []:
        ts = tr.get("t") or tr.get("timestamp")
        if not ts:
            continue
        try:
            when = _parse_iso(str(ts))
        except Exception:
            continue
        action = int(tr.get("action", 0))
        state_raw = tr.get("state", {}) or {}
        state = dict(state_raw)
        # gap_days がトップレベルにある場合も統一
        if "gap_days" not in state and tr.get("gap_days") is not None:
            state["gap_days"] = tr.get("gap_days")
        cleaned_state = ensure_numeric_state(state)
        if not cleaned_state:
            continue
        record = {
            "reviewer_id": reviewer_id,
            "timestamp": when.isoformat(),
            "action": action,
            "state": cleaned_state,
        }
        yield when, record


def build_dataset(input_path: Path, cutoff: datetime, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    feature_keys: set[str] = set()
    stats = {
        "train_count": 0,
        "eval_count": 0,
        "train_positive": 0,
        "eval_positive": 0,
    }

    with train_path.open("w", encoding="utf-8") as fw_train, eval_path.open("w", encoding="utf-8") as fw_eval:
        for seq in tqdm(_iter_sequences(input_path), desc="sequences", leave=False):
            for when, rec in _build_records(seq):
                for k in rec["state"].keys():
                    feature_keys.add(k)
                line = json.dumps(rec, ensure_ascii=False)
                if when <= cutoff:
                    fw_train.write(line + "\n")
                    stats["train_count"] += 1
                    stats["train_positive"] += int(rec["action"] == 1)
                else:
                    fw_eval.write(line + "\n")
                    stats["eval_count"] += 1
                    stats["eval_positive"] += int(rec["action"] == 1)

    meta = {
        "input": str(input_path),
        "cutoff": cutoff.isoformat(),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "feature_keys": sorted(feature_keys),
    }
    meta.update(stats)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="レビュアー adoption IRL 用の学習/評価データセットを生成します。")
    ap.add_argument("--input", type=Path, default=Path("outputs/irl/reviewer_sequences.json"))
    ap.add_argument("--cutoff", type=str, required=True, help="YYYY-mm-ddTHH:MM:SSZ 形式のISOタイムスタンプ。これ以前を学習、以降を評価に分割。")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/reviewer_adoption"))
    args = ap.parse_args()

    cutoff = _parse_iso(args.cutoff)
    meta = build_dataset(args.input, cutoff, args.outdir)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
