#!/usr/bin/env python3
"""
開発者ごとの "次のタスク(=次の authored change) に再度取り組むか" を予測する簡易スクリプト。

目的:
 - ユーザ要望: 「最後のタスク後に次へ取り組むかを開発者単位で」
 - ここでは Gerrit change をタスク近似 (owner.email が開発者の authored change)
 - 逆強化学習/強化学習 導入前のベースライン分類器 (WorkloadAwarePredictor) を再利用

ラベリング:
 - 各開発者の authored change を作成日時で昇順ソート: t1 < t2 < ... < tN
 - 各 i (1..N-1) に対し snapshot = ti, next = t(i+1)
 - horizon_days 以内 (next - snapshot).days <= horizon なら label=1 (次に素早く取り組む)
   それ以外は 0 (次活動まで長期空白 or 無)：最終イベント tN はラベル無し

特徴量近似 (リーク低減簡易版):
 - `all_developers.json` から該当 dev の集計値を取得し、進捗率 fraction=(i/ N) でスケール
 - review_scores も先頭から fraction 切出し / last_activity を snapshot に置換

制約:
 - 正確な "招待されたが取り組まなかった" 情報が無いので negative は "次の authored change が horizon を超える" ケースに限定
 - reviewers リスト等を用いた "招待→反応" 分析は後続 (構造確認後に拡張)

今後の RL / IRL 拡張案:
 - 状態: (最近活動間隔, 累積変更数, ドメイン分散, 負荷指標)
 - 行動: engage / skip 次の招待タスク
 - 報酬 (IRL 学習): 高頻度継続開発者軌跡から MaxEnt IRL で重み推定 → 価値関数で次活動確率推定

使用例:
  uv run python scripts/predict_next_authored_change.py --horizon-days 45 --test-ratio 0.3
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

from src.gerrit_retention.prediction.historical_feature_builder import (
    build_change_index,
    build_snapshot_features,
    parse_dt,
)
from src.gerrit_retention.prediction.workload_aware_predictor import (  # noqa: E402
    WorkloadAwarePredictor,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Predict next authored change engagement within horizon")
    ap.add_argument('--developers', default='data/processed/unified/all_developers.json')
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--horizon-days', type=int, default=60)
    ap.add_argument('--min-changes', type=int, default=3, help='Min authored changes to include developer')
    ap.add_argument('--max-dev', type=int, default=0, help='Limit developers for debug (0=all)')
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--output-dir', default='outputs/next_authored_change')
    return ap.parse_args()


def parse_dt(ts: str) -> datetime | None:
    if not ts:
        return None
    ts = ts.replace('Z', '+00:00').replace(' ', 'T')
    if '.' in ts:
        head, tail = ts.split('.', 1)
        if '+' in tail:
            frac, tz = tail.split('+', 1)
            if len(frac) > 6:
                frac = frac[:6]
            ts = f"{head}.{frac}+{tz}"
        elif '-' in tail and tail.count('-') > 1:
            frac, tz = tail.split('-', 1)
            if len(frac) > 6:
                frac = frac[:6]
            ts = f"{head}.{frac}-{tz}"
        else:
            if len(tail) > 6:
                ts = f"{head}.{tail[:6]}"
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


@dataclass
class Snapshot:
    dev_id: str
    snap_time: datetime
    label: int
    features: Dict[str, Any]


def build_authored_sequences(changes: List[Dict[str, Any]]) -> Dict[str, List[datetime]]:
    seq: Dict[str, List[datetime]] = {}
    for ch in changes:
        owner = ch.get('owner') or {}
        dev_id = owner.get('email') or owner.get('username')
        created = parse_dt(ch.get('created'))
        if dev_id and created:
            seq.setdefault(dev_id, []).append(created)
    for v in seq.values():
        v.sort()
    return seq


def scale_dev(*args, **kwargs):  # backward compat placeholder (unused after hist features)
    raise RuntimeError("scale_dev should not be called after historical feature integration")


def build_snapshots(developers: List[Dict[str, Any]], sequences: Dict[str, List[datetime]],
                    horizon_days: int, min_changes: int, change_index) -> List[Snapshot]:
    dev_map = {d.get('developer_id') or d.get('email'): d for d in developers}
    snaps: List[Snapshot] = []
    for dev_id, times in sequences.items():
        if len(times) < min_changes:
            continue
        base_master = dev_map.get(dev_id)
        if not base_master:
            continue
        idx_obj = change_index.get(dev_id)
        if not idx_obj:
            continue
        total = len(times)
        for i in range(total - 1):
            t_i = times[i]
            t_next = times[i+1]
            delta_days = (t_next - t_i).days
            label = 1 if delta_days <= horizon_days else 0
            hist_feat = build_snapshot_features(dev_id, t_i, base_master, idx_obj)
            snaps.append(Snapshot(dev_id, t_i, label, hist_feat))
    return snaps


def time_split(snaps: List[Snapshot], test_ratio: float) -> Tuple[List[Snapshot], List[Snapshot]]:
    snaps_sorted = sorted(snaps, key=lambda s: s.snap_time)
    split = int(len(snaps_sorted) * (1 - test_ratio))
    return snaps_sorted[:split], snaps_sorted[split:]


def compute_metrics(y_true, probs):
    preds = [1 if p >= 0.5 else 0 for p in probs]
    out = {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
        'positive_rate': float(np.mean(y_true))
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


def main():
    args = parse_args()
    dev_path = Path(args.developers)
    ch_path = Path(args.changes)
    if not dev_path.exists() or not ch_path.exists():
        print("❌ data file missing")
        return 1
    developers = load_json(dev_path)
    changes = load_json(ch_path)
    if args.max_dev:
        developers = developers[:args.max_dev]
    sequences = build_authored_sequences(changes)
    change_index = build_change_index(changes)
    snapshots = build_snapshots(developers, sequences, args.horizon_days, args.min_changes, change_index)
    if not snapshots:
        print("❌ no snapshots")
        return 1
    train, test = time_split(snapshots, args.test_ratio)
    if not train or not test:
        print("❌ invalid split")
        return 1
    # 追加: 招待負例を学習集合へ統合 (horizon で authored 無かった招待)
    invited_neg_path = Path('outputs/invited_negatives/negatives.json')
    invited_neg = []
    if invited_neg_path.exists():
        try:
            invited_neg = json.loads(invited_neg_path.read_text())
        except Exception:
            invited_neg = []
    # build historical features for invited negatives snapshot times
    extra_X, extra_y = [], []
    for neg in invited_neg:
        dev_id = neg.get('developer_id')
        snap_time = parse_dt(neg.get('invite_time'))
        if not dev_id or not snap_time:
            continue
        base_master = next((d for d in developers if (d.get('developer_id') or d.get('email')) == dev_id), None)
        idx_obj = change_index.get(dev_id)
        if not base_master or not idx_obj:
            continue
        feat = build_snapshot_features(dev_id, snap_time, base_master, idx_obj)
        extra_X.append(feat)
        extra_y.append(0)
    X_train = [s.features for s in train] + extra_X
    y_train = [s.label for s in train] + extra_y
    X_test = [s.features for s in test]
    y_test = [s.label for s in test]

    predictor = WorkloadAwarePredictor()
    if len(set(y_train)) == 1:
        only = list(set(y_train))[0]
        const_prob = 0.99 if only == 1 else 0.01
        probs = [const_prob]*len(X_test)
    else:
        predictor.fit(X_train, y_train)
        probs = predictor.predict_batch(X_test)
    metrics = compute_metrics(y_test, probs)
    print("📊 Metrics")
    for k,v in metrics.items():
        print(f"  {k}: {v}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    import json as _json

    # predictions
    rows = []
    for s, p in zip(test, probs):
        rows.append({'developer_id': s.dev_id,'snapshot_time': s.snap_time.isoformat(),'label_next_within_horizon': s.label,'prob_next_within_horizon': p})
    (out_dir/'predictions.json').write_text(_json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir/'metrics.json').write_text(_json.dumps({'horizon_days': args.horizon_days,'count_train': len(train),'count_test': len(test),'metrics': metrics}, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out_dir/'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['developer_id','snapshot_time','label_next_within_horizon','prob_next_within_horizon'])
        for r in rows:
            w.writerow([r['developer_id'], r['snapshot_time'], r['label_next_within_horizon'], f"{r['prob_next_within_horizon']:.6f}"])
    print(f"💾 saved: {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
