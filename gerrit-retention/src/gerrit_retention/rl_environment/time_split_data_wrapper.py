"""
時系列データ分割ラッパー

extended_test_data.json のような JSON からアクティビティを読み込み、
時系列でフィルタしたイベントを ReviewAcceptanceEnvironment の review_queue に供給する。

用途:
  - 訓練: cutoff 以下の時系列イベントを使用
  - 評価: cutoff より後の時系列イベントを使用

注意: データ構造が簡易的なため、ReviewRequest はヒューリスティックに生成する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym

from .review_env import ReviewRequest


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.fromisoformat(ts)


@dataclass
class _Event:
    when: datetime
    req: ReviewRequest


class TimeSplitDataWrapper(gym.Wrapper):
    """JSON データで review_queue を制御するラッパー。

    Args:
        env: ベース環境（ReviewAcceptanceEnvironment）
        data_path: JSONファイルパス（extended_test_data.json を想定）
        cutoff_iso: 時系列分割の ISO 8601 文字列
        phase: 'train'（<=cutoff） or 'eval'（>cutoff）
    """

    def __init__(
        self,
        env: gym.Env,
        data_path: str | Path,
        cutoff_iso: str,
        phase: str = "train",
    ) -> None:
        super().__init__(env)
        self.data_path = str(data_path)
        self.cutoff = _parse_iso(cutoff_iso)
        assert phase in ("train", "eval")
        self.phase = phase

        self._events: List[_Event] = []
        self._cursor: int = 0
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        p = Path(self.data_path)
        if not p.exists():
            # 何もできない場合は空イベント
            self._events = []
            self._loaded = True
            return
        data = json.loads(p.read_text())
        events: List[_Event] = []
        for dev in data:
            dev_info = dev.get("developer", {})
            dev_id = dev_info.get("developer_id", "unknown@example.com")
            projects = dev_info.get("projects", []) or ["unknown-project"]
            history = dev.get("activity_history", [])
            for idx, act in enumerate(history):
                ts = act.get("timestamp")
                if not ts:
                    continue
                when = _parse_iso(ts)
                # 対象タイプ: commit / review をレビュー対象イベントとして扱う
                typ = act.get("type", "")
                if typ not in ("commit", "review"):
                    continue
                lines_added = int(act.get("lines_added", 0) or 0)
                lines_deleted = int(act.get("lines_deleted", 0) or 0)
                total_lines = max(0, lines_added + lines_deleted)
                files_changed = max(1, round(total_lines / 50)) if total_lines > 0 else 1
                complexity = max(0.1, min(1.0, 0.2 + total_lines / 500.0))
                effort = max(0.5, min(4.0, 0.5 + total_lines / 200.0))
                subject = act.get("message") or f"{typ} event"
                req = ReviewRequest(
                    change_id=f"{dev_id}_{idx}_{when.strftime('%Y%m%d%H%M%S')}",
                    author_email=dev_id,
                    project=str(projects[0]),
                    branch="main",
                    subject=subject,
                    files_changed=files_changed,
                    lines_added=lines_added,
                    lines_deleted=lines_deleted,
                    complexity_score=float(complexity),
                    technical_domain="backend",
                    urgency_level=0.5,
                    estimated_review_effort=float(effort),
                    required_expertise=["python"],
                    created_at=when,
                    deadline=when + timedelta(days=3),
                    expertise_match=0.5,
                    requester_relationship=0.5,
                )
                events.append(_Event(when=when, req=req))
        # フィルタ & ソート
        if self.phase == "train":
            filtered = [e for e in events if e.when <= self.cutoff]
        else:
            filtered = [e for e in events if e.when > self.cutoff]
        filtered.sort(key=lambda e: e.when)
        self._events = filtered
        self._cursor = 0
        self._loaded = True

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self._load()
        obs, info = self.env.reset(**kwargs)
        # 初期キューを時系列イベントから供給
        self._refill_queue()
        return obs, info

    def step(self, action: int):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        # ベース環境がランダムに追加したレビューは無視し、データから供給
        self._refill_queue()
        # データが尽きたら切り詰め
        if self._cursor >= len(self._events) and len(getattr(self.env, "review_queue", [])) == 0:
            truncated = True
        return next_obs, reward, terminated, truncated, info

    def _refill_queue(self):
        # 先頭をベース env が pop 済みなので、空き分をデータから補充
        q = getattr(self.env, "review_queue", [])
        max_q = getattr(self.env, "max_queue_size", 10)
        need = max(0, max_q - len(q))
        if need <= 0:
            return
        # 追加分を events から供給
        add_reqs = []
        while need > 0 and self._cursor < len(self._events):
            add_reqs.append(self._events[self._cursor].req)
            self._cursor += 1
            need -= 1
        if add_reqs:
            q.extend(add_reqs)
            setattr(self.env, "review_queue", q)
