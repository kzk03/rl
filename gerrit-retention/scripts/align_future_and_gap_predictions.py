#!/usr/bin/env python3
"""Align future-window predictions with gap(threshold)-based virtual snapshot predictions.

目的:
  future-window モデル (T → T+Δ 活動) の予測と、gap=閾値(例:60d) 基準 retained ラベル
  (virtual snapshots) を開発者 / 近接日時で突合し、確率差・ラベル差・不確実/過信ケースを
  一括 JSON 出力する。

出力:
  outputs/prediction_vs_reality_analysis/future_gap_alignment.json
  outputs/prediction_vs_reality_analysis/future_gap_summary.json

主要カテゴリ:
  - uncertain_future: |p_future - 0.5| <= future_uncert_margin
  - gap_high_conf_fp: gap_label=0 かつ p_gap >= gap_high_conf_prob
  - label_mismatch: future_label != gap_label (あれば)

マッチング: future snapshot_date 以後で最初に現れる gap snapshot (同 developer) を対応付け。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args():
    ap = argparse.ArgumentParser(description='Align future-window and gap-based predictions')
    ap.add_argument('--future-pred', default='outputs/future_window_eval_project/openstack/nova/predictions.json')
    ap.add_argument('--gap-pred', default='outputs/retention_probability/time_series_from_virtual/openstack_nova/predictions_test.json')
    ap.add_argument('--future-uncert-margin', type=float, default=0.1)
    ap.add_argument('--gap-high-conf-prob', type=float, default=0.8)
    ap.add_argument('--output-dir', default='outputs/prediction_vs_reality_analysis')
    return ap.parse_args()


def parse_ts(ts: str) -> datetime:
    ts = ts.replace('Z', '+00:00')
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def build_index_gap(gap_rows: List[Dict[str, Any]]):
    by_dev: Dict[str, List[Dict[str, Any]]] = {}
    for r in gap_rows:
        dev = r.get('developer_id')
        if not dev:
            continue
        r['_dt'] = parse_ts(r['snapshot_date'])
        by_dev.setdefault(dev, []).append(r)
    for dev in by_dev:
        by_dev[dev].sort(key=lambda x: x['_dt'])
    return by_dev


def find_gap_after(rows: List[Dict[str, Any]], dt: datetime) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    for r in rows:
        if r['_dt'] >= dt:
            return r
    return None


def main():
    args = parse_args()
    future_rows = load_json(Path(args.future_pred))
    gap_rows = load_json(Path(args.gap_pred))
    gap_index = build_index_gap(gap_rows)

    alignment = []
    metrics = {
        'future_total': len(future_rows),
        'gap_total': len(gap_rows),
        'aligned_pairs': 0,
        'uncertain_future': 0,
        'gap_high_conf_fp': 0,
        'label_mismatch': 0,
    }

    # Pre-scan gap high confidence FP (gap label=0 & prob >= threshold)
    gap_high_conf_fp_ids = set()
    for gr in gap_rows:
        if gr.get('label') == 0 and gr.get('prob_retained', 0.0) >= args.gap_high_conf_prob:
            gap_high_conf_fp_ids.add((gr.get('developer_id'), gr.get('snapshot_date')))
    metrics['gap_high_conf_fp'] = len(gap_high_conf_fp_ids)

    for fr in future_rows:
        try:
            fdt = parse_ts(fr['snapshot_date'])
        except Exception:
            continue
        dev = fr.get('developer_id')
        future_prob = fr.get('prob_future_active', 0.0)
        future_label = fr.get('label_future_active')
        gap_candidate = find_gap_after(gap_index.get(dev, []), fdt)
        record: Dict[str, Any] = {
            'developer_id': dev,
            'future_snapshot_date': fr.get('snapshot_date'),
            'future_label': future_label,
            'future_prob': future_prob,
            'future_uncertain': abs(future_prob - 0.5) <= args.future_uncert_margin,
        }
        if gap_candidate:
            record.update({
                'gap_snapshot_date': gap_candidate.get('snapshot_date'),
                'gap_label': gap_candidate.get('label'),
                'gap_prob': gap_candidate.get('prob_retained'),
                'delta_days': (gap_candidate['_dt'] - fdt).days,
                'gap_high_conf_fp': (gap_candidate.get('label') == 0 and gap_candidate.get('prob_retained', 0) >= args.gap_high_conf_prob),
            })
            metrics['aligned_pairs'] += 1
            if record['future_uncertain']:
                metrics['uncertain_future'] += 1
            if record['gap_high_conf_fp']:
                # already counted globally; keep per-pair tagging
                pass
            if gap_candidate.get('label') is not None and future_label is not None and gap_candidate.get('label') != future_label:
                metrics['label_mismatch'] += 1
        alignment.append(record)

    # Developer aggregated deltas
    by_dev: Dict[str, Dict[str, Any]] = {}
    from statistics import mean
    for rec in alignment:
        dev = rec['developer_id']
        agg = by_dev.setdefault(dev, {'developer_id': dev, 'pairs': 0, 'future_probs': [], 'gap_probs': [], 'future_labels': [], 'gap_labels': []})
        agg['pairs'] += 1
        agg['future_probs'].append(rec.get('future_prob'))
        if 'gap_prob' in rec:
            agg['gap_probs'].append(rec.get('gap_prob'))
        if rec.get('future_label') is not None:
            agg['future_labels'].append(rec.get('future_label'))
        if rec.get('gap_label') is not None:
            agg['gap_labels'].append(rec.get('gap_label'))
    dev_summary = []
    for dev, agg in by_dev.items():
        dev_summary.append({
            'developer_id': dev,
            'pairs': agg['pairs'],
            'mean_future_prob': round(mean(agg['future_probs']), 4) if agg['future_probs'] else None,
            'mean_gap_prob': round(mean(agg['gap_probs']), 4) if agg['gap_probs'] else None,
            'future_pos_rate': (sum(agg['future_labels']) / len(agg['future_labels'])) if agg['future_labels'] else None,
            'gap_pos_rate': (sum(agg['gap_labels']) / len(agg['gap_labels'])) if agg['gap_labels'] else None,
            'prob_gap_minus_future': (round(mean(agg['gap_probs']), 4) - round(mean(agg['future_probs']), 4)) if (agg['future_probs'] and agg['gap_probs']) else None,
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'future_gap_alignment.json').write_text(json.dumps(alignment, indent=2), encoding='utf-8')
    summary_payload = {
        'metrics': metrics,
        'developer_summary': dev_summary,
        'parameters': {
            'future_uncert_margin': args.future_uncert_margin,
            'gap_high_conf_prob': args.gap_high_conf_prob,
            'future_predictions_file': str(Path(args.future_pred)),
            'gap_predictions_file': str(Path(args.gap_pred)),
        }
    }
    (out_dir / 'future_gap_summary.json').write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')
    print('✅ alignment complete')
    print('  metrics:', metrics)
    print(f'💾 Saved -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
