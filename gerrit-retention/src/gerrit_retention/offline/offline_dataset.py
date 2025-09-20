"""
オフラインRL用データセット生成ユーティリティ

extended_test_data.json の実データから、(s, a, r, s', done) の遷移を生成して
JSONL として保存する。cutoff で時系列分割して train/eval を分ける。

前提:
- review イベントの score を行動にマップ:
  score >= +1 -> accept, score <= -1 -> reject, それ以外 -> wait
- commit 等のレビュー以外イベントは wait として扱う（レビュー待ちの状態と解釈）
- 環境は ReviewAcceptanceEnvironment を使用し、内部のランダムレビュー追加は無効化
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from gerrit_retention.rl_environment.review_env import (
    ReviewAcceptanceEnvironment,
    ReviewRequest,
)


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)


def _iter_events(data: List[Dict[str, Any]]) -> Iterable[Tuple[datetime, Dict[str, Any], Dict[str, Any]]]:
    """データから (when, dev_info, activity) を時系列順に列挙"""
    for dev in data:
        dev_info = dev.get("developer", {})
        for act in dev.get("activity_history", []):
            ts = act.get("timestamp")
            if not ts:
                continue
            yield _parse_iso(ts), dev_info, act


def _build_request_from_activity(dev: Dict[str, Any], act: Dict[str, Any]) -> ReviewRequest:
    # ヒューリスティックに ReviewRequest を構築
    dev_id = dev.get("developer_id", "unknown@example.com")
    projects = dev.get("projects", []) or ["unknown-project"]
    typ = act.get("type", "event")
    when = _parse_iso(act["timestamp"])
    lines_added = int(act.get("lines_added", 0) or 0)
    lines_deleted = int(act.get("lines_deleted", 0) or 0)
    total_lines = max(0, lines_added + lines_deleted)
    files_changed = max(1, round(total_lines / 50)) if total_lines > 0 else 1
    complexity = float(np.clip(0.2 + total_lines / 500.0, 0.1, 1.0))
    effort = float(np.clip(0.5 + total_lines / 200.0, 0.5, 4.0))
    subject = act.get("message") or f"{typ} event"
    return ReviewRequest(
        change_id=f"{dev_id}_{when.strftime('%Y%m%d%H%M%S')}",
        author_email=dev_id,
        project=str(projects[0]),
        branch="main",
        subject=subject,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        complexity_score=complexity,
        technical_domain="backend",
        urgency_level=0.5,
        estimated_review_effort=effort,
        required_expertise=["python"],
        created_at=when,
        deadline=when,
        expertise_match=0.5,
        requester_relationship=0.5,
    )


def _map_action_from_activity(act: Dict[str, Any], include_non_review_as_wait: bool) -> Optional[int]:
    """アクティビティを行動にマップ

    Returns: 0(reject)/1(accept)/2(wait) or None(スキップ)
    """
    # 0: reject, 1: accept, 2: wait
    if act.get("type") == "review":
        score = act.get("score")
        if score is not None:
            try:
                s = float(score)
            except Exception:
                s = 0.0
            if s >= 1.0:
                return 1
            if s <= -1.0:
                return 0
            return 2
        # score 未指定のレビューは保守的に待機
        return 2
    # 非レビュー系はデフォルトで除外（クラス不均衡を避ける）
    return 2 if include_non_review_as_wait else None


def build_offline_datasets(
    data_path: str | Path,
    cutoff_iso: str,
    out_dir: str | Path = "outputs/offline",
    include_non_review_as_wait: bool = False,
) -> Dict[str, Any]:
    """実データから train/eval の JSONL データセットを作成

    Returns: dict(meta)
    """
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = _parse_iso(cutoff_iso)

    data = json.loads(data_path.read_text())
    events = sorted(_iter_events(data), key=lambda x: x[0])

    # 環境（ランダム生成を完全無効化）
    env = ReviewAcceptanceEnvironment({
        'max_episode_length': 100,
        'max_queue_size': 1,
        'stress_threshold': 0.8,
        'use_random_initial_queue': False,
        'enable_random_new_reviews': False,
    })

    def _gen_dataset(phase: str, predicate) -> Tuple[str, int]:
        obs_count = 0
        lines = []
        obs, _ = env.reset()
        for when, dev, act in filter(lambda t: predicate(t[0]), events):
            # ReviewRequest を1件だけキューに置く
            req = _build_request_from_activity(dev, act)
            setattr(env, 'review_queue', [req])
            state = env._get_observation()  # 20-dim
            a = _map_action_from_activity(act, include_non_review_as_wait)
            if a is None:
                continue  # スキップ
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            sample = {
                'timestamp': when.isoformat(),
                'state': state.tolist(),
                'action': int(a),
                'reward': float(reward),
                'next_state': next_obs.tolist(),
                'done': done,
            }
            lines.append(json.dumps(sample, ensure_ascii=False))
            obs_count += 1
            if done:
                obs, _ = env.reset()
        out_path = out_dir / f"dataset_{phase}.jsonl"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        return str(out_path), obs_count

    train_path, train_n = _gen_dataset('train', lambda t: t <= cutoff)
    eval_path, eval_n = _gen_dataset('eval', lambda t: t > cutoff)

    meta = {
        'data_path': str(data_path),
        'cutoff': cutoff.isoformat(),
        'train_dataset': train_path,
        'train_samples': train_n,
        'eval_dataset': eval_path,
        'eval_samples': eval_n,
        'include_non_review_as_wait': include_non_review_as_wait,
    }
    (out_dir / 'offline_dataset_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta
