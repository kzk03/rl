#!/usr/bin/env python3
"""C: プロジェクト絞り込み付き future-window 簡易評価ラッパ

既存 evaluate_future_window_retention.py を呼ぶ代わりに:
 - changes を project でフィルタしたサブセット JSON を一時生成
 - developers JSON も (developer_id が project 活動で現れた者のみ) に絞る
 - その後 core evaluator ロジックをインポートして内部関数呼び出し (コード重複最小化)

使用例:
  uv run scripts/run_future_window_project_eval.py \
     --developers data/processed/unified/all_developers.json \
     --changes data/processed/unified/all_reviews.json \
     --project my/project --horizon 60 --test-ratio 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.evaluate_future_window_retention import (
    scale_developer_features,  # re-export (silence linter)
)
from scripts.evaluate_future_window_retention import (
    Snapshot,
    build_activity_map,
    build_snapshots,
    compute_metrics,
    load_json,
)
from scripts.evaluate_future_window_retention import (  # noqa: E402
    parse_args as base_parse_args,
)
from scripts.evaluate_future_window_retention import time_order_split, to_model_data
from src.gerrit_retention.prediction.workload_aware_predictor import (  # noqa: E402
    WorkloadAwarePredictor,
)


def parse_args():
    ap = argparse.ArgumentParser(description='Project-scoped future window retention evaluation')
    ap.add_argument('--developers', default='data/processed/unified/all_developers.json')
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--project', required=True)
    ap.add_argument('--horizon-days', type=int, default=90)
    ap.add_argument('--max-snapshots-per-dev', type=int, default=5)
    ap.add_argument('--min-activities', type=int, default=3)
    ap.add_argument('--test-ratio', type=float, default=0.3)
    ap.add_argument('--output-dir', default='outputs/future_window_eval_project')
    ap.add_argument('--long-gap-days', type=int, default=0)
    # Negative augmentation options
    ap.add_argument('--add-gap-negatives', action='store_true', help='Add synthetic negative snapshots inside large gaps (> horizon)')
    ap.add_argument('--max-gap-negatives-per-dev', type=int, default=2)
    ap.add_argument('--add-tail-negative', action='store_true', help='Add a negative snapshot near tail if last gap >= horizon')
    ap.add_argument('--save-train-preds', action='store_true', help='Also save train set predictions and combined all_predictions.json')
    # IRL integration
    ap.add_argument('--use-rf', action='store_true', help='Use legacy RandomForest baseline instead of default IRL')
    ap.add_argument('--recent-windows', default='7,14,30', help='Comma separated day windows for recent activity counts')
    ap.add_argument('--decay-half-life', type=int, default=14, help='Half-life (days) for exponential decay recent activity score (within max(recent-windows))')
    return ap.parse_args()


def filter_changes(changes, project):
    return [c for c in changes if c.get('project') == project]


def main():
    args = parse_args()
    devs = load_json(Path(args.developers))
    changes = load_json(Path(args.changes))
    sub_changes = filter_changes(changes, args.project)
    if not sub_changes:
        print('❌ no changes for project'); return 1
    # build activity map scoped
    activity_map = build_activity_map(sub_changes)
    # keep only developers appearing in this project
    dev_ids = set(activity_map.keys())
    scoped_devs = [d for d in devs if (d.get('developer_id') or d.get('email')) in dev_ids]
    # dataset end time
    dataset_end = max(dt for acts in activity_map.values() for dt in acts)
    snaps = build_snapshots(
        scoped_devs,
        activity_map,
        args.horizon_days,
        args.max_snapshots_per_dev,
        args.min_activities,
        long_gap_days=args.long_gap_days,
        dataset_end=dataset_end,
    )
    # Augment negatives
    added_gap_neg = 0
    added_tail_neg = 0
    if args.add_gap_negatives or args.add_tail_negative:
        from datetime import timedelta

        # build dev index for features (consistent with build_snapshots)
        dev_index = {d.get('developer_id') or d.get('email'): d for d in scoped_devs}
        # helper: reuse scaler
        from scripts.evaluate_future_window_retention import scale_developer_features
        horizon_td = timedelta(days=args.horizon_days)
        # per-dev existing snapshot dates to avoid duplicates
        existing_keys = {(s.developer_id, s.snapshot_date) for s in snaps}
        for dev_id, acts in activity_map.items():
            base_dev = dev_index.get(dev_id)
            if not base_dev or len(acts) < args.min_activities:
                continue
            total = len(acts)
            gap_added = 0
            # gap negatives
            if args.add_gap_negatives:
                for i in range(len(acts)-1):
                    if gap_added >= args.max_gap_negatives_per_dev:
                        break
                    a0, a1 = acts[i], acts[i+1]
                    gap = a1 - a0
                    if gap <= horizon_td:
                        continue
                    # choose midpoint minus half horizon to ensure no future act within horizon
                    usable_gap = gap - horizon_td
                    if usable_gap.total_seconds() <= 0:
                        continue
                    neg_date = a0 + usable_gap / 2
                    # label will be 0 because next act > neg_date + horizon
                    fraction = (i + 1) / total
                    features = scale_developer_features(base_dev, fraction, neg_date)
                    snap = Snapshot(dev_id, neg_date, 0, features)
                    key = (snap.developer_id, snap.snapshot_date)
                    if key not in existing_keys:
                        snaps.append(snap)
                        existing_keys.add(key)
                        added_gap_neg += 1
                        gap_added += 1
            # tail negative
            if args.add_tail_negative:
                last_act = acts[-1]
                tail_gap = dataset_end - last_act
                if tail_gap >= horizon_td:
                    # place tail negative at last_act + min( (tail_gap - horizon)/2, horizon ) to stay inside gap
                    usable_tail = tail_gap - horizon_td
                    if usable_tail.total_seconds() > 0:
                        offset = min(usable_tail, horizon_td)
                        neg_date = last_act + offset / 2
                        fraction = 1.0  # up to last activity
                        features = scale_developer_features(base_dev, fraction, neg_date)
                        snap = Snapshot(dev_id, neg_date, 0, features)
                        key = (snap.developer_id, snap.snapshot_date)
                        if key not in existing_keys:
                            snaps.append(snap)
                            existing_keys.add(key)
                            added_tail_neg += 1
    if not snaps:
        print('❌ no snapshots after filtering'); return 1

    # --- enrich snapshots with gap_days feature ---
    def _annotate_gap_days(snapshots, activity_map):
        act_index = {dev_id: acts for dev_id, acts in activity_map.items()}
        for s in snapshots:
            acts = act_index.get(s.developer_id) or []
            prev = None
            for a in acts:
                if a < s.snapshot_date:
                    prev = a
                else:
                    break
            gap_days = 0 if prev is None else (s.snapshot_date - prev).days
            s.developer_features['gap_days'] = gap_days
        return snapshots
    snaps = _annotate_gap_days(snaps, activity_map)

    # --- recent activity window features (counts / ratios / decay) ---
    def _augment_recent_activity(snapshots, activity_map, window_days_list, decay_half_life):
        import math as _math
        max_window = max(window_days_list) if window_days_list else 0
        act_index = {dev_id: acts for dev_id, acts in activity_map.items()}
        ln2 = _math.log(2.0)
        for s in snapshots:
            acts = act_index.get(s.developer_id) or []
            # only consider activities strictly before snapshot_date
            prior_acts = [a for a in acts if a < s.snapshot_date and (max_window == 0 or (s.snapshot_date - a).days <= max_window)]
            # counts per window
            for w in window_days_list:
                cnt = 0
                if w > 0:
                    threshold = s.snapshot_date - timedelta(days=w)
                    for a in prior_acts:
                        if a >= threshold:
                            cnt += 1
                s.developer_features[f'recent_count_{w}d'] = cnt
            # ratios (using 30d or max window as denominator)
            denom_key = f'recent_count_{max_window}d'
            denom = s.developer_features.get(denom_key, 0) or 0
            if denom > 0:
                for w in window_days_list:
                    num = s.developer_features.get(f'recent_count_{w}d', 0) or 0
                    s.developer_features[f'recent_ratio_{w}d_{max_window}d'] = num / denom
            else:
                for w in window_days_list:
                    s.developer_features[f'recent_ratio_{w}d_{max_window}d'] = 0.0
            # velocity examples: count_7 / max(1,count_14), count_14 / max(1,count_30)
            def _safe_ratio(a,b):
                return a / b if b not in (0,None) else 0.0
            if 7 in window_days_list and 14 in window_days_list:
                c7 = s.developer_features.get('recent_count_7d',0)
                c14 = s.developer_features.get('recent_count_14d',0)
                s.developer_features['velocity_7_over_14'] = _safe_ratio(c7, c14)
            if 14 in window_days_list and max_window in window_days_list and max_window != 14:
                c14 = s.developer_features.get('recent_count_14d',0)
                cm = s.developer_features.get(f'recent_count_{max_window}d',0)
                s.developer_features[f'velocity_14_over_{max_window}'] = _safe_ratio(c14, cm)
            # exponential decay score within max_window
            if max_window > 0 and decay_half_life > 0:
                decay_score = 0.0
                for a in prior_acts:
                    age_days = (s.snapshot_date - a).days
                    if age_days <= max_window:
                        decay_score += 2 ** (-age_days / decay_half_life)
                s.developer_features[f'decay_activity_{max_window}d_h{decay_half_life}'] = decay_score
        return snapshots

    try:
        window_list = [int(x) for x in args.recent_windows.split(',') if x.strip()]
    except Exception:
        window_list = [7,14,30]
    if window_list:
        snaps = _augment_recent_activity(snaps, activity_map, sorted(set(window_list)), args.decay_half_life)

    train, test = time_order_split(snaps, args.test_ratio)
    X_train, y_train = to_model_data(train)
    X_test, y_test = to_model_data(test)

    # Default = IRL (MaxEnt). Use --use-rf to fallback.
    model_type = 'irl'
    rf_importances = None
    if not args.use_rf:
        try:
            from src.gerrit_retention.irl.maxent_binary_irl import (  # noqa: E402
                MaxEntBinaryIRL,
            )
        except Exception as e:  # pragma: no cover
            print(f'❌ Failed to import IRL module: {e}')
            return 1
        transitions = [{'state': feat, 'action': y} for feat, y in zip(X_train, y_train)]
        irl = MaxEntBinaryIRL()
        _ = irl.fit(transitions)
        probs = irl.predict_proba(X_test)
        model_type = 'irl'
    else:
        model = WorkloadAwarePredictor()
        model.fit(X_train, y_train)
        probs = model.predict_batch(X_test)
        # capture feature importances if available
        if hasattr(model.model, 'feature_importances_') and getattr(model, 'feature_names', None):
            fi = model.model.feature_importances_
            names = model.feature_names
            rf_importances = [
                {'feature': n, 'importance': float(v)} for n, v in sorted(zip(names, fi), key=lambda x: x[1], reverse=True)
            ]
        model_type = 'rf'

    metrics = compute_metrics(y_test, probs)
    metrics.update({
        'project': args.project,
        'snapshot_count': len(snaps),
        'train_count': len(train),
        'test_count': len(test),
        'horizon_days': args.horizon_days,
        'added_gap_negatives': added_gap_neg,
        'added_tail_negatives': added_tail_neg,
        'model_type': model_type,
    'recent_windows': window_list,
    })
    out_dir = Path(args.output_dir)/args.project
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s, p in zip(test, probs):
        rows.append({
            'developer_id': s.developer_id,
            'snapshot_date': s.snapshot_date.isoformat(),
            'label_future_active': s.label,
            'prob_future_active': float(p),
            'pred_label@0.5': int(p >= 0.5),
            'correct': int((p >= 0.5) == s.label)
        })
    (out_dir/'predictions.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')

    if model_type == 'irl':
        try:
            weights = irl.explain_weights()
            (out_dir/'irl_weights.json').write_text(json.dumps(weights, indent=2), encoding='utf-8')
        except Exception:
            pass
    else:
        if rf_importances is not None:
            (out_dir/'rf_feature_importances.json').write_text(json.dumps(rf_importances, indent=2), encoding='utf-8')

    if args.save_train_preds:
        if model_type == 'irl':
            train_probs = irl.predict_proba(X_train)
        else:
            if len(set(y_train)) == 1:
                only = list(set(y_train))[0]
                train_probs = [0.99 if only==1 else 0.01]*len(train)
            else:
                train_probs = model.predict_batch(X_train)
        train_rows = []
        for s, p in zip(train, train_probs):
            train_rows.append({
                'developer_id': s.developer_id,
                'snapshot_date': s.snapshot_date.isoformat(),
                'label_future_active': s.label,
                'prob_future_active': float(p),
                'pred_label@0.5': int(p >= 0.5),
                'correct': int((p >= 0.5) == s.label)
            })
        (out_dir/'train_predictions.json').write_text(json.dumps(train_rows, indent=2), encoding='utf-8')
        all_rows = train_rows + rows
        (out_dir/'all_predictions.json').write_text(json.dumps(all_rows, indent=2), encoding='utf-8')

    (out_dir/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('✅ project future-window metrics:')
    for k,v in metrics.items():
        print(f'  {k}: {v}')
    print(f'💾 Saved -> {out_dir}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
