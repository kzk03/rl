#!/usr/bin/env python3
"""Virtual snapshot (gap-threshold) labels -> time-series training dataset builder

目的 (ユーザ要求: B の出力を時系列学習セットに昇格):
  `reconstruct_snapshot_labels.py` で得た複数 virtual_now スナップショットの
  gap ベース retained ラベルを、スナップショット時点特徴付きの学習データに変換し
  (オプションで) WorkloadAwarePredictor を学習/評価する。

特徴:
  - プロジェクトフィルタ対応 (--project) : features 生成も当該プロジェクト活動のみでスケーリング
  - developer 集計特徴を activity 割合で近似スケール (evaluate_future_window_retention と整合)
  - snapshot_date = virtual_now, label = retained (gap ラベル)
  - time-ordered split (--test-ratio) により過去→未来で評価
  - クラス単一時は定数確率ベースライン

出力 (--output-dir 基底):
  dataset.json             : 全スナップショット (sorted by snapshot_date)
  train_test_split.json    : train/test developer_id & counts
  metrics.json             : 分類指標 (accuracy / precision / recall / f1 / auc / brier / pos_rate)
  predictions_test.json    : テスト予測詳細

使用例:
  uv run scripts/build_time_series_dataset_from_virtual_snapshots.py \
      --developers data/processed/unified/all_developers.json \
      --changes data/processed/unified/all_reviews.json \
      --virtual-snapshots outputs/retention_probability/virtual_snapshots.json \
      --project openstack/nova --test-ratio 0.3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.gerrit_retention.prediction.workload_aware_predictor import (  # noqa: E402
    WorkloadAwarePredictor,
)


@dataclass
class SnapshotRow:
    developer_id: str
    snapshot_date: datetime
    label: int
    features: Dict[str, Any]


def parse_args():
    ap = argparse.ArgumentParser(description='Build time-series dataset from virtual gap-label snapshots')
    ap.add_argument('--developers', default='data/processed/unified/all_developers.json')
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--virtual-snapshots', default='outputs/retention_probability/virtual_snapshots.json')
    ap.add_argument('--project', default=None)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--min-activities', type=int, default=1, help='Minimum activities overall to include developer')
    ap.add_argument('--output-dir', default='outputs/retention_probability/time_series_from_virtual')
    return ap.parse_args()


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def parse_ts(ts: str):
    if not ts:
        return None
    ts = ts.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def build_activity_map(changes: List[Dict[str, Any]], project: str | None):
    amap: Dict[str, List[datetime]] = {}
    for ch in changes:
        if project and ch.get('project') != project:
            continue
        owner = ch.get('owner') or {}
        dev = owner.get('email') or owner.get('username')
        if not dev:
            continue
        created = ch.get('created') or ch.get('updated') or ch.get('submitted')
        dt = parse_ts(created)
        if not dt:
            continue
        amap.setdefault(dev, []).append(dt)
    for dev in list(amap.keys()):
        amap[dev].sort()
    return amap


def scale_developer_features(base_dev: Dict[str, Any], fraction: float, snapshot_date: datetime, last_activity: datetime):
    fraction = max(0.0, min(1.0, fraction))
    dev = dict(base_dev)
    for key in ['changes_authored', 'changes_reviewed', 'total_insertions', 'total_deletions']:
        if key in dev and isinstance(dev[key], (int, float)):
            dev[key] = int(math.floor(dev[key] * fraction))
    rs = dev.get('review_scores')
    if isinstance(rs, list) and rs:
        cut = max(1, int(len(rs) * fraction))
        dev['review_scores'] = rs[:cut]
    dev['last_activity'] = last_activity.isoformat()
    dev['snapshot_date'] = snapshot_date.isoformat()
    return dev


def time_order_split(rows: List[SnapshotRow], test_ratio: float) -> Tuple[List[SnapshotRow], List[SnapshotRow]]:
    ordered = sorted(rows, key=lambda r: r.snapshot_date)
    split = int(len(ordered) * (1 - test_ratio))
    return ordered[:split], ordered[split:]


def compute_metrics(y_true, probs):
    preds = [1 if p >= 0.5 else 0 for p in probs]
    out = {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
        'positive_rate': float(np.mean(y_true)),
    }
    if len(set(y_true)) > 1:
        try:
            out['auc'] = roc_auc_score(y_true, probs)
        except Exception:
            out['auc'] = None
        try:
            out['brier'] = brier_score_loss(y_true, probs)
        except Exception:
            out['brier'] = None
    else:
        out['auc'] = None
        out['brier'] = None
    return out


def main():  # noqa: C901
    args = parse_args()
    devs = load_json(Path(args.developers))
    changes = load_json(Path(args.changes))
    vs_path = Path(args.virtual_snapshots)
    if not vs_path.exists():
        print(f'❌ virtual snapshots not found: {vs_path}')
        return 1
    virtual_rows = load_json(vs_path)
    if not virtual_rows:
        print('❌ virtual snapshots empty')
        return 1
    activity_map = build_activity_map(changes, args.project)
    if not activity_map:
        print('❌ no activities for project (check --project)')
        return 1
    dev_index = {d.get('developer_id') or d.get('email'): d for d in devs}

    # Pre-compute for performance
    rows: List[SnapshotRow] = []
    skipped = 0
    for row in virtual_rows:
        dev_id = row.get('developer_id')
        if dev_id not in activity_map:
            skipped += 1
            continue
        acts = activity_map[dev_id]
        if len(acts) < args.min_activities:
            skipped += 1
            continue
        virtual_now = parse_ts(row.get('virtual_now'))
        last_act_dt = parse_ts(row.get('last_activity'))
        if not virtual_now or not last_act_dt:
            skipped += 1
            continue
        # find index of last_act_dt
        try:
            idx = acts.index(last_act_dt)
        except ValueError:
            # if last_activity not in filtered project acts (e.g., came from other project), skip
            skipped += 1
            continue
        fraction = (idx + 1) / len(acts)
        base_dev = dev_index.get(dev_id)
        if not base_dev:
            skipped += 1
            continue
        scaled = scale_developer_features(base_dev, fraction, virtual_now, last_act_dt)
        label = int(row.get('retained'))
        rows.append(SnapshotRow(dev_id, virtual_now, label, scaled))

    if not rows:
        print('❌ no usable snapshot rows constructed (all filtered)')
        return 1
    rows.sort(key=lambda r: r.snapshot_date)
    train_rows, test_rows = time_order_split(rows, args.test_ratio)
    if not train_rows or not test_rows:
        print('❌ invalid train/test split (adjust --test-ratio)')
        return 1

    X_train = [r.features for r in train_rows]
    y_train = [r.label for r in train_rows]
    X_test = [r.features for r in test_rows]
    y_test = [r.label for r in test_rows]

    unique_train = set(y_train)
    if len(unique_train) == 1:
        only = list(unique_train)[0]
        const_prob = 0.99 if only == 1 else 0.01
        probs = [const_prob] * len(X_test)
        print(f'⚠️ single-class train set => constant prob={const_prob}')
    else:
        model = WorkloadAwarePredictor()
        model.fit(X_train, y_train)
        probs = model.predict_batch(X_test)

    metrics = compute_metrics(y_test, probs)

    out_dir = Path(args.output_dir)
    if args.project:
        out_dir = out_dir / args.project.replace('/', '_')
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset.json
    dataset_json = []
    for r in rows:
        dataset_json.append({
            'developer_id': r.developer_id,
            'snapshot_date': r.snapshot_date.isoformat(),
            'label_retained': r.label,
            **r.features,
        })
    (out_dir / 'dataset.json').write_text(json.dumps(dataset_json, indent=2), encoding='utf-8')

    # predictions
    preds = []
    for r, p in zip(test_rows, probs):
        preds.append({
            'developer_id': r.developer_id,
            'snapshot_date': r.snapshot_date.isoformat(),
            'label': r.label,
            'prob_retained': float(p),
        })
    (out_dir / 'predictions_test.json').write_text(json.dumps(preds, indent=2), encoding='utf-8')

    meta = {
        'total_rows': len(rows),
        'train_rows': len(train_rows),
        'test_rows': len(test_rows),
        'skipped_rows': skipped,
        'test_ratio': args.test_ratio,
        'project': args.project,
        'metrics': metrics,
        'class_distribution_train': {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        'class_distribution_test': {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        'note': 'Features scaled by activity fraction (approx); labels are gap-threshold retained from virtual snapshots.'
    }
    (out_dir / 'metrics.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print('✅ Dataset built')
    for k, v in metrics.items():
        print(f'  {k}: {v}')
    print(f'💾 Saved -> {out_dir}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
