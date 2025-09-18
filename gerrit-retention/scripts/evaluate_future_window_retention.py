#!/usr/bin/env python3
"""
将来ウィンドウ参照型 継続予測評価スクリプト (A + B 対応)

目的 (ユーザ要求対応):
 A) スナップショット時点 T の特徴で T+Δ 日内 (horizon) に活動があるかでラベル付け
 B) 時系列分割 (過去スナップショットで学習 / 直近期で評価) によりリークを低減

アプローチ概要:
 1. all_reviews.json (Gerrit changes) から開発者ごとの活動タイムスタンプ系列を抽出
    - 現状: change の owner.email を開発者IDとみなし 'created' (存在すれば) を活動発生日として利用
 2. 各開発者の活動系列から最大 N 個 ( --max-snapshots-per-dev ) のスナップショット日時を選択 (末尾は除く)
 3. 各スナップショット日時 snapshot_date に対し horizon_days 先までに >=1 活動があれば label=1 (継続), なければ 0 (離脱)
 4. スナップショット時点の特徴量をリーク最小化のため近似再構成:
      - all_developers.json の集計値を、その時点までの活動割合でスケール (簡易近似)
      - review_scores は先頭から割合分を切り出し
      - last_activity を snapshot_date に置換
    注意: 完全な履歴再集計が未提供なため近似 (将来改善ポイント)。
 5. 全スナップショットを snapshot_date 昇順に並べ、先頭 (1 - test_ratio) を訓練、残りをテスト
 6. WorkloadAwarePredictor で学習/推論し精度指標 (accuracy / precision / recall / f1 / auc / brier / pos_rate) を算出
 7. 出力: outputs/future_window_eval/{snapshots.csv, predictions.json, metrics.json}

制約と今後の改善余地:
 - 集計特徴を割合スケールする近似は将来データリークを完全に排除しない可能性 → 真の過去集計再計算処理へ拡張予定
 - review / insertion / deletion の正確な時系列内訳が changes データに含まれる場合は精緻化可能
 - メモリ節約のため all_reviews.json は JSON 全読み (メモリ不足時は簡易ストリーム処理に切替)

使用例:
  uv run python scripts/evaluate_future_window_retention.py \
      --developers data/processed/unified/all_developers.json \
      --changes data/processed/unified/all_reviews.json \
      --horizon-days 90 --max-snapshots-per-dev 4 --test-ratio 0.3

軽量テスト例:
  uv run python scripts/evaluate_future_window_retention.py --max-developers 20 --max-snapshots-per-dev 2 --horizon-days 30 --test-ratio 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
class Snapshot:
    developer_id: str
    snapshot_date: datetime
    label: int
    developer_features: Dict[str, Any]


def parse_args():
    p = argparse.ArgumentParser(description="Future-window retention evaluation (time-series)")
    p.add_argument('--developers', default='data/processed/unified/all_developers.json')
    p.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    p.add_argument('--project', default=None, help='If set, filter changes to this project only')
    p.add_argument('--horizon-days', type=int, default=90, help='Future window Δ (days)')
    p.add_argument('--max-snapshots-per-dev', type=int, default=5)
    p.add_argument('--min-activities', type=int, default=3, help='Minimum activity count for a developer to be considered')
    p.add_argument('--test-ratio', type=float, default=0.3, help='Proportion of latest snapshots for test (time ordered)')
    p.add_argument('--max-developers', type=int, default=0, help='Limit number of developers (0=all) for faster debug')
    p.add_argument('--random-seed', type=int, default=42)
    p.add_argument('--output-dir', default='outputs/future_window_eval')
    p.add_argument('--long-gap-days', type=int, default=0, help='If >0 and last activity gap >= this, add final snapshot forced negative')
    return p.parse_args()


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        ts = ts.replace('Z', '+00:00')
        # 長すぎるナノ秒部をマイクロ秒に丸め
        if '.' in ts:
            head, tail = ts.split('.', 1)
            if '+' in tail:
                frac, tz = tail.split('+', 1)
                if len(frac) > 6:
                    frac = frac[:6]
                ts = f"{head}.{frac}+{tz}"
            elif '-' in tail:
                frac, tz = tail.split('-', 1)
                if len(frac) > 6:
                    frac = frac[:6]
                ts = f"{head}.{frac}-{tz}"
            else:
                if len(tail) > 6:
                    ts = f"{head}.{tail[:6]}"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def build_activity_map(changes: List[Dict[str, Any]], max_developers: int = 0) -> Dict[str, List[datetime]]:
    """owner.email をキーに created 日時を活動として集約"""
    activity: Dict[str, List[datetime]] = {}
    for ch in changes:
        owner = ch.get('owner') or {}
        dev_id = owner.get('email') or owner.get('username')
        if not dev_id:
            continue
        created = ch.get('created') or ch.get('updated') or ch.get('submitted')
        dt = parse_ts(created)
        if not dt:
            continue
        activity.setdefault(dev_id, []).append(dt)
        # 早期制限 (オプション) - 適用は後段 (max_developers) で行うためここではしない
    # ソート
    for dev_id in activity:
        activity[dev_id].sort()
    if max_developers and len(activity) > max_developers:
        # 先頭 max_developers に制限 (alphabetical for determinism)
        selected_keys = sorted(activity.keys())[:max_developers]
        activity = {k: activity[k] for k in selected_keys}
    return activity


def choose_snapshot_indices(n: int, max_snapshots: int) -> List[int]:
    if n < 2:
        return []
    usable = n - 1  # 最後は未来がないので除外
    k = min(max_snapshots, usable)
    if k <= 0:
        return []
    if usable <= k:
        return list(range(usable))
    # 均等サンプリング
    positions = np.linspace(0, usable - 1, k)
    return sorted({int(round(p)) for p in positions})


def scale_developer_features(base_dev: Dict[str, Any], fraction: float, snapshot_date: datetime) -> Dict[str, Any]:
    """集計特徴を '過去割合' でスケールする近似。 fraction∈(0,1]"""
    fraction = max(0.0, min(1.0, fraction))
    dev = dict(base_dev)  # copy
    for key in ['changes_authored', 'changes_reviewed', 'total_insertions', 'total_deletions']:
        if key in dev and isinstance(dev[key], (int, float)):
            dev[key] = int(math.floor(dev[key] * fraction))
    # review_scores 切り出し
    rs = dev.get('review_scores')
    if isinstance(rs, list) and rs:
        cut = max(1, int(len(rs) * fraction))
        dev['review_scores'] = rs[:cut]
    # last_activity をスナップショットに書き換え
    dev['last_activity'] = snapshot_date.isoformat()
    return dev


def build_snapshots(
    developers: List[Dict[str, Any]],
    activity_map: Dict[str, List[datetime]],
    horizon_days: int,
    max_snapshots_per_dev: int,
    min_activities: int,
    long_gap_days: int = 0,
    dataset_end: datetime | None = None,
) -> List[Snapshot]:
    dev_index = {d.get('developer_id') or d.get('email'): d for d in developers}
    snapshots: List[Snapshot] = []
    horizon = timedelta(days=horizon_days)
    for dev_id, acts in activity_map.items():
        if len(acts) < min_activities:
            continue
        base_dev = dev_index.get(dev_id)
        if not base_dev:
            # 開発者マスターに無い場合スキップ
            continue
        idxs = choose_snapshot_indices(len(acts), max_snapshots_per_dev)
        total = len(acts)
        for idx in idxs:
            snap_date = acts[idx]
            future_end = snap_date + horizon
            # 将来活動判定
            label = 0
            for future_act in acts[idx+1:]:
                if future_act <= snap_date:
                    continue
                if future_act <= future_end:
                    label = 1
                    break
                if future_act > future_end:
                    break
            fraction = (idx + 1) / total  # その時点までの活動割合 (>=1 event)
            scaled_dev = scale_developer_features(base_dev, fraction, snap_date)
            snapshots.append(Snapshot(dev_id, snap_date, label, scaled_dev))
        # 末尾長期ギャップ強制負例
        if long_gap_days and dataset_end:
            last_act = acts[-1]
            gap = (dataset_end - last_act).days
            if gap >= long_gap_days:
                fraction = 1.0
                scaled_dev = scale_developer_features(base_dev, fraction, last_act)
                snapshots.append(Snapshot(dev_id, last_act, 0, scaled_dev))
    return snapshots


def time_order_split(snapshots: List[Snapshot], test_ratio: float) -> Tuple[List[Snapshot], List[Snapshot]]:
    snapshots_sorted = sorted(snapshots, key=lambda s: s.snapshot_date)
    split = int(len(snapshots_sorted) * (1 - test_ratio))
    train = snapshots_sorted[:split]
    test = snapshots_sorted[split:]
    return train, test


def to_model_data(snaps: List[Snapshot]):
    return [s.developer_features for s in snaps], [s.label for s in snaps]


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


def main():  # noqa: C901 (複雑度許容)
    args = parse_args()
    dev_path = Path(args.developers)
    ch_path = Path(args.changes)

    if not dev_path.exists():
        print(f"❌ developers file not found: {dev_path}")
        return 1
    if not ch_path.exists():
        print(f"❌ changes file not found: {ch_path}")
        return 1

    print(f"📥 Loading developers: {dev_path}")
    developers = load_json(dev_path)
    if args.max_developers:
        developers = developers[:args.max_developers]
    print(f"✅ Developers loaded: {len(developers)}")

    print(f"📥 Loading changes (may take time): {ch_path}")
    try:
        changes_all = load_json(ch_path)
    except MemoryError:
        print("⚠️ MemoryError: file too large. Consider preprocessing to a smaller subset.")
        return 1
    # Optional project filter
    if args.project:
        changes = [c for c in changes_all if c.get('project') == args.project]
        print(f"✅ Changes loaded: {len(changes_all)} (filtered by project='{args.project}' -> {len(changes)})")
        if not changes:
            print("❌ No changes after project filter; aborting")
            return 1
    else:
        changes = changes_all
        print(f"✅ Changes loaded: {len(changes)}")

    activity_map = build_activity_map(changes, max_developers=args.max_developers)
    print(f"🧪 Activity map built: developers with activity={len(activity_map)}")

    # dataset_end 推定 (活動最大日時)
    all_last = []
    for acts in activity_map.values():
        if acts:
            all_last.append(acts[-1])
    dataset_end = max(all_last) if all_last else None

    snapshots = build_snapshots(
        developers, activity_map, args.horizon_days, args.max_snapshots_per_dev, args.min_activities,
        long_gap_days=args.long_gap_days, dataset_end=dataset_end
    )
    if not snapshots:
        print("❌ No snapshots generated (check min-activities / data)")
        return 1
    print(f"📸 Generated snapshots: {len(snapshots)} (horizon={args.horizon_days}d)")

    train_snaps, test_snaps = time_order_split(snapshots, args.test_ratio)
    print(f"✂️ Time split: train={len(train_snaps)} test={len(test_snaps)} (ratio={args.test_ratio})")
    if len(train_snaps) < 5 or len(test_snaps) < 3:
        print("⚠️ Not enough snapshots for robust evaluation (need >=5 train & >=3 test). Try lowering --test-ratio, increasing --max-snapshots-per-dev, or removing --max-developers limit.")
    if not train_snaps or not test_snaps:
        print("❌ Train/Test split invalid")
        return 1

    X_train, y_train = to_model_data(train_snaps)
    X_test, y_test = to_model_data(test_snaps)

    predictor = WorkloadAwarePredictor()
    # 単一クラスの場合は学習せず定数確率
    unique_train = set(y_train)
    if len(unique_train) == 1:
        only_label = list(unique_train)[0]
        print(f"⚠️ Train set has single class={only_label}; using constant probability baseline")
        const_prob = 0.99 if only_label == 1 else 0.01
        probs = [const_prob] * len(X_test)
    else:
        predictor.fit(X_train, y_train)
        probs = predictor.predict_batch(X_test)
    metrics = compute_metrics(y_test, probs)
    # 追加: クラス分布
    def _class_counts(arr):
        return {int(k): int(v) for k, v in zip(*np.unique(arr, return_counts=True))}
    train_class_counts = _class_counts(y_train)
    test_class_counts = _class_counts(y_test)

    print("\n📊 Test Metrics")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存: スナップショット一覧 (CSV) と予測
    import csv
    snap_csv = out_dir / 'snapshots.csv'
    with snap_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['developer_id','snapshot_date','label'])
        for s in snapshots:
            w.writerow([s.developer_id, s.snapshot_date.isoformat(), s.label])

    preds_json = []
    for s, p in zip(test_snaps, probs):
        preds_json.append({
            'developer_id': s.developer_id,
            'snapshot_date': s.snapshot_date.isoformat(),
            'label': s.label,
            'prob_retained': p
        })

    (out_dir / 'predictions.json').write_text(json.dumps(preds_json, ensure_ascii=False, indent=2), encoding='utf-8')
    meta = {
        'horizon_days': args.horizon_days,
        'max_snapshots_per_dev': args.max_snapshots_per_dev,
        'test_ratio': args.test_ratio,
        'total_snapshots': len(snapshots),
        'train_snapshots': len(train_snaps),
        'test_snapshots': len(test_snaps),
        'train_class_counts': train_class_counts,
        'test_class_counts': test_class_counts,
        'metrics': metrics,
        'approximation_warning': 'Feature scaling is an approximation; future leakage mitigated but not fully eliminated.'
    }
    (out_dir / 'metrics.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\n💾 Saved snapshots: {snap_csv}")
    print(f"💾 Saved predictions: {out_dir / 'predictions.json'}")
    print(f"💾 Saved metrics: {out_dir / 'metrics.json'}")
    print("✅ Done (future-window evaluation)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
