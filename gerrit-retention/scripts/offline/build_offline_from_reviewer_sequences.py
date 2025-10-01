#!/usr/bin/env python3
"""
reviewer_sequences.json -> オフラインRL用 JSONL(train/eval) 生成スクリプト

入力:
- --input: outputs/irl/reviewer_sequences.json のようなファイル
  構造例: [{"reviewer_id": str, "transitions": [{"t": ISO8601, "action": 0/1/2, "state": {...}}]}]

出力:
- outdir/dataset_train.jsonl, outdir/dataset_eval.jsonl, メタ情報 outdir/offline_dataset_meta.json

方針:
- 各トランジション時刻 t ごとに、ヒューリスティックな ReviewRequest を合成し、
  ReviewAcceptanceEnvironment に投入して (s,a,r,s',done) を1ステップ生成
- cutoff で時系列分割（<=cutoff を train、>cutoff を eval）
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from gerrit_retention.rl_environment.review_env import (
    ReviewAcceptanceEnvironment,
    ReviewRequest,
)


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00')).replace(tzinfo=None)
    except Exception:
        return datetime.fromisoformat(ts)


@dataclass
class _Event:
    when: datetime
    reviewer_id: str
    request: ReviewRequest
    action: int  # 0/1/2


def _heuristic_request(reviewer_id: str, when: datetime, state: Dict[str, Any]) -> ReviewRequest:
    # state のヒントからレビュー特徴を粗く合成
    gap_days = float(state.get('gap_days', state.get('gap', 3) or 3))
    activity30 = float(state.get('activity_30d', state.get('activity30', 1.0) or 1.0))
    activity90 = float(state.get('activity_90d', state.get('activity90', max(1.0, activity30)) or max(1.0, activity30)))
    workload = float(state.get('workload_level', state.get('workload', 0.2) or 0.2))
    complexity = float(np.clip(0.2 + workload, 0.1, 1.0))
    total_lines = int(np.clip(50 * activity30 + 20 * (activity90 - activity30) + 10, 10, 2000))
    lines_added = int(total_lines * 0.7)
    lines_deleted = max(0, total_lines - lines_added)
    files_changed = max(1, round(total_lines / 80))
    # 期限までの緊急度は gap が短いほど高いと仮定
    urgency = float(np.clip(0.2 + 0.8 * (1.0 / (1.0 + gap_days / 7.0)), 0.1, 1.0))
    expertise = float(np.clip(state.get('expertise_recent', 0.5), 0.0, 1.0))

    return ReviewRequest(
        change_id=f"{reviewer_id}_{when.strftime('%Y%m%d%H%M%S')}",
        author_email=reviewer_id,
        project="historical-project",
        branch="main",
        subject="historical review",
        files_changed=int(files_changed),
        lines_added=int(lines_added),
        lines_deleted=int(lines_deleted),
        complexity_score=float(complexity),
        technical_domain="backend",
        urgency_level=float(urgency),
        estimated_review_effort=float(np.clip(0.5 + total_lines / 400.0, 0.5, 4.0)),
        required_expertise=["python"],
        created_at=when,
        deadline=when,  # 環境側では締め切り差分は内部で扱う
        expertise_match=float(expertise),
        requester_relationship=0.5,
    )


def _collect_events(seq_data: List[Dict[str, Any]]) -> List[_Event]:
    events: List[_Event] = []
    for rec in seq_data:
        rid = rec.get('reviewer_id') or rec.get('developer_id') or 'unknown@example.com'
        trans = rec.get('transitions') or []
        for tr in trans:
            ts = tr.get('t') or tr.get('timestamp')
            if not ts:
                continue
            when = _parse_iso(ts)
            action = int(tr.get('action', 2))
            state = tr.get('state', {}) or {}
            req = _heuristic_request(rid, when, state)
            events.append(_Event(when=when, reviewer_id=rid, request=req, action=action))
    events.sort(key=lambda e: e.when)
    return events


def _inject_wait_actions(events: List[_Event], prob: float, seed: int | None = None, max_injected_per_event: int = 1) -> List[_Event]:
    """確率的に "wait"(action=2) サンプルを各イベントの直前に合成して3クラス化を促進。

    仕様:
    - 元のイベントの action が 0/1 のときにのみ、確率 `prob` で wait を追加
    - 追加サンプルの timestamp は元の when の 1 秒前（順序安定のため）
    - request は同一（ヒューリスティック的に同じレビュー要求のワンステップ待機とみなす）
    - 乱数は `seed` で固定可能
    """
    if prob <= 0.0:
        return events
    rng = random.Random(seed)
    out: List[_Event] = []
    for ev in events:
        if ev.action in (0, 1):
            injected = 0
            while injected < max_injected_per_event and rng.random() < prob:
                out.append(_Event(
                    when=ev.when - timedelta(seconds=1),
                    reviewer_id=ev.reviewer_id,
                    request=ev.request,
                    action=2,
                ))
                injected += 1
        out.append(ev)
    out.sort(key=lambda e: e.when)
    return out


def build_offline_from_sequences(
    input_path: str | Path,
    cutoff_iso: str,
    out_dir: str | Path,
    inject_wait_prob: float = 0.0,
    seed: int | None = 42,
    binary_actions: bool = False,
) -> Dict[str, Any]:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = _parse_iso(cutoff_iso)

    seq_data = json.loads(input_path.read_text())
    events = _collect_events(seq_data)
    # 任意: wait クラス注入で 3 クラス化を促進
    events = _inject_wait_actions(events, float(inject_wait_prob), seed=seed)

    env = ReviewAcceptanceEnvironment({
        'max_episode_length': 100,
        'max_queue_size': 1,
        'stress_threshold': 0.8,
        'use_random_initial_queue': False,
        'enable_random_new_reviews': False,
    })

    def _gen(phase: str, predicate) -> Tuple[str, int]:
        lines: List[str] = []
        samples = 0
        obs, _ = env.reset()
        for ev in filter(lambda e: predicate(e.when), events):
            setattr(env, 'review_queue', [ev.request])
            state = env._get_observation()
            a_raw = int(ev.action)
            a = 0 if binary_actions and a_raw in (0, 2) else (1 if binary_actions and a_raw == 1 else a_raw)
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            sample = {
                'timestamp': ev.when.isoformat(),
                'state': state.tolist(),
                'action': int(a),
                'reward': float(reward),
                'next_state': next_obs.tolist(),
                'done': done,
            }
            lines.append(json.dumps(sample, ensure_ascii=False))
            samples += 1
            if done:
                obs, _ = env.reset()
        outp = out_dir / f'dataset_{phase}.jsonl'
        outp.write_text("\n".join(lines) + ("\n" if lines else ""))
        return str(outp), samples

    train_path, n_train = _gen('train', lambda t: t <= cutoff)
    eval_path, n_eval = _gen('eval', lambda t: t > cutoff)

    meta = {
        'source': str(input_path),
        'cutoff': cutoff.isoformat(),
        'train_dataset': train_path,
        'train_samples': n_train,
        'eval_dataset': eval_path,
        'eval_samples': n_eval,
        'inject_wait_prob': float(inject_wait_prob),
        'seed': int(seed) if seed is not None else None,
        'binary_actions': bool(binary_actions),
    }
    (out_dir / 'offline_dataset_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='outputs/irl/reviewer_sequences.json')
    ap.add_argument('--cutoff', type=str, default='2024-10-01T00:00:00Z')
    ap.add_argument('--outdir', type=str, default='outputs/offline/expanded')
    ap.add_argument('--inject-wait-prob', type=float, default=0.0, help='各イベント前に確率的に wait(action=2) を注入')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--binary-actions', action='store_true', help='action=2(wait) を 0(非受諾) に統合して 2 クラス化する')
    args = ap.parse_args()

    meta = build_offline_from_sequences(
        args.input,
        args.cutoff,
        args.outdir,
        inject_wait_prob=float(args.inject_wait_prob),
        seed=int(args.seed),
        binary_actions=bool(args.binary_actions),
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
