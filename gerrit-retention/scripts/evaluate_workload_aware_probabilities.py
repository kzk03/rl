#!/usr/bin/env python3
"""
作業負荷・専門性考慮型モデルによる個別継続確率算出 & 精度評価スクリプト

機能:
 1. all_developers.json 読み込み
 2. ラベル生成: last_activity からの経過日数 <= retention_days_threshold なら 1 (継続), それ以外 0
 3. 時系列スプリット (last_activity 昇順で train/test 分割)
 4. WorkloadAwarePredictor を訓練し test で確率出力
 5. メトリクス (accuracy / precision / recall / f1 / auc / brier / pos_rate) を算出
 6. outputs/retention_probability/ に predictions.json, metrics.json 保存

使い方 (例):
  uv run python scripts/evaluate_workload_aware_probabilities.py --data data/processed/unified/all_developers.json --threshold 90 --test-ratio 0.25
"""
import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# パス追加
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.gerrit_retention.prediction.workload_aware_predictor import (  # noqa: E402
    WorkloadAwarePredictor,
)


@dataclass
class DevSample:
    data: Dict[str, Any]
    last_activity: datetime
    label: int


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate workload-aware retention probabilities")
    p.add_argument('--data', default='data/processed/unified/all_developers.json')
    p.add_argument('--threshold', type=int, default=90, help='Retention days threshold (<= this => retained)')
    p.add_argument('--test-ratio', type=float, default=0.3, help='Test portion based on time ordering')
    p.add_argument('--min-activity', type=int, default=0, help='Minimum (changes_authored+changes_reviewed) to keep developer')
    p.add_argument('--output-dir', default='outputs/retention_probability')
    return p.parse_args()


def load_and_label(path: Path, threshold_days: int, min_activity: int) -> List[DevSample]:
    raw = json.loads(path.read_text())
    samples: List[DevSample] = []
    now = datetime.now(timezone.utc)
    for d in raw:
        dev = dict(d)  # shallow copy
        # normalize last_activity (if missing, skip)
        last_act_str = dev.get('last_activity') or dev.get('developer', {}).get('last_activity')
        if not last_act_str:
            continue
        try:
            # handle possible nanoseconds -> drop trailing zeros if length > 26
            la = last_act_str.replace('Z', '+00:00')
            if len(la) > 32 and '.' in la:
                head, tail = la.split('.', 1)
                la = head + '.' + tail[:6]  # microseconds
            last_activity = datetime.fromisoformat(la)
            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=timezone.utc)
            else:
                last_activity = last_activity.astimezone(timezone.utc)
        except Exception:
            continue
        total_changes = dev.get('changes_authored', 0) + dev.get('changes_reviewed', 0)
        if total_changes < min_activity:
            continue
        days_since_last = (now - last_activity).days
        label = 1 if days_since_last <= threshold_days else 0
        samples.append(DevSample(dev, last_activity, label))
    return samples


def time_split(samples: List[DevSample], test_ratio: float):
    samples_sorted = sorted(samples, key=lambda s: s.last_activity)
    split_index = int(len(samples_sorted) * (1 - test_ratio))
    train = samples_sorted[:split_index]
    test = samples_sorted[split_index:]
    return train, test


def to_model_format(samples: List[DevSample]):
    return [s.data for s in samples], [s.label for s in samples]


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ data not found: {data_path}")
        return 1

    print(f"📥 Loading developers: {data_path}")
    samples = load_and_label(data_path, args.threshold, args.min_activity)
    if not samples:
        print("❌ no valid samples")
        return 1
    print(f"✅ Loaded {len(samples)} developers (threshold={args.threshold}d)")

    train, test = time_split(samples, args.test_ratio)
    print(f"🧪 Split train={len(train)} test={len(test)} (time ordered)")

    train_devs, train_labels = to_model_format(train)
    test_devs, test_labels = to_model_format(test)

    predictor = WorkloadAwarePredictor()
    predictor.fit(train_devs, train_labels)

    probs = predictor.predict_batch(test_devs)
    preds = [1 if p >= 0.5 else 0 for p in probs]

    metrics = {
        'threshold_days': args.threshold,
        'test_ratio': args.test_ratio,
        'count_train': len(train),
        'count_test': len(test),
        'positive_rate_test': float(np.mean(test_labels)),
        'accuracy': accuracy_score(test_labels, preds),
        'precision': precision_score(test_labels, preds, zero_division=0),
        'recall': recall_score(test_labels, preds, zero_division=0),
        'f1': f1_score(test_labels, preds, zero_division=0),
    }
    # AUC & Brier (only if both classes present)
    if len(set(test_labels)) > 1:
        try:
            metrics['auc'] = roc_auc_score(test_labels, probs)
        except Exception:
            metrics['auc'] = None
        try:
            metrics['brier'] = brier_score_loss(test_labels, probs)
        except Exception:
            metrics['brier'] = None
    else:
        metrics['auc'] = None
        metrics['brier'] = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-dev predictions
    pred_rows = []
    for dev, label, prob in zip(test_devs, test_labels, probs):
        pred_rows.append({
            'developer_id': dev.get('developer_id') or dev.get('email'),
            'last_activity': dev.get('last_activity'),
            'changes_authored': dev.get('changes_authored'),
            'changes_reviewed': dev.get('changes_reviewed'),
            'projects': dev.get('projects', []),
            'label_retained': label,
            'prob_retained': prob
        })

    (out_dir / 'predictions.json').write_text(json.dumps(pred_rows, ensure_ascii=False, indent=2))
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n📊 Metrics")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n💾 Saved: {out_dir / 'predictions.json'}")
    print(f"💾 Saved: {out_dir / 'metrics.json'}")
    print("✅ Done")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
