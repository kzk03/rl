#!/usr/bin/env python3
"""各開発者の継続確率予測と実際ラベルの整合性を分析するスクリプト

前提:
  先に `scripts/evaluate_workload_aware_probabilities.py` を実行し
  `outputs/retention_probability/predictions.json` が生成されていること。

機能:
 1. predictions.json を読み込み (各 developer の label_retained / prob_retained)
 2. 分類閾値 (デフォルト0.5) で予測ラベル作成
 3. 正解/不正解、過信/過小 (確率信念に対する誤差) を計算
 4. キャリブレーション (10分割) & ECE (Expected Calibration Error) を算出
 5. 上位: 高確率外れ (False Positive), 低確率外れ (False Negative) を抽出
 6. alignment ディレクトリへ summary, calibration, misclassified JSON 保存

限界:
  - 現行ラベルは「現在時点で一定日閾値内に最後の活動があるか」のスナップショット指標であり、
    未来ウィンドウ観測による真の将来継続 (future engagement) を直接検証していません。
  - 未来ベース評価が必要な場合は過去スナップショット時点を固定し直後 Δ 日の活動有無を再構築する別パイプラインが必要です。

使用例:
  uv run scripts/analyze_retention_prediction_alignment.py --pred-file outputs/retention_probability/predictions.json --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import math
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


def parse_args():
    ap = argparse.ArgumentParser(description="Workload-aware retention prediction alignment analysis")
    ap.add_argument('--pred-file', default='outputs/retention_probability/predictions.json')
    ap.add_argument('--prob-threshold', type=float, default=0.5, help='分類用閾値')
    ap.add_argument('--calib-bins', type=int, default=10)
    ap.add_argument('--output-dir', default='outputs/retention_probability/alignment')
    return ap.parse_args()


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"predictions file not found: {path}")
    return json.loads(path.read_text())


def calibration_bins(y: List[int], probs: List[float], n_bins: int):
    y_arr = np.asarray(y)
    p_arr = np.asarray(probs)
    edges = np.linspace(0,1,n_bins+1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (p_arr >= lo) & (p_arr < hi if i < n_bins-1 else p_arr <= hi)
        if mask.sum() == 0:
            bins.append({'bin': i,'lo': float(lo),'hi': float(hi),'count':0,'pred_mean':None,'true_rate':None,'gap':None})
            continue
        pred_mean = float(p_arr[mask].mean())
        true_rate = float(y_arr[mask].mean())
        gap = pred_mean - true_rate
        w = mask.mean()  # bin weight
        ece += abs(gap)*w
        bins.append({'bin': i,'lo': float(lo),'hi': float(hi),'count': int(mask.sum()),'pred_mean': pred_mean,'true_rate': true_rate,'gap': gap})
    return bins, float(ece)


def analyze(rows: List[Dict[str, Any]], threshold: float, n_bins: int):
    y = [int(r['label_retained']) for r in rows]
    probs = [float(r['prob_retained']) for r in rows]
    preds = [1 if p >= threshold else 0 for p in probs]
    metrics = {
        'count': len(rows),
        'threshold': threshold,
        'positive_rate': float(np.mean(y)),
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds, zero_division=0),
        'recall': recall_score(y, preds, zero_division=0),
        'f1': f1_score(y, preds, zero_division=0)
    }
    if len(set(y)) > 1:
        try: metrics['auc'] = roc_auc_score(y, probs)
        except: metrics['auc'] = None
        try: metrics['brier'] = brier_score_loss(y, probs)
        except: metrics['brier'] = None
    else:
        metrics['auc'] = None
        metrics['brier'] = None

    # misclassified analysis
    detailed = []
    for dev, label, prob, pred in zip(rows, y, probs, preds):
        correct = (label == pred)
        margin = abs(prob - threshold)
        err_mag = abs(prob - (1 if label==1 else 0))  # calibration誤差
        detailed.append({
            'developer_id': dev.get('developer_id'),
            'label': label,
            'prob': prob,
            'pred': pred,
            'correct': correct,
            'margin_to_threshold': margin,
            'prob_error_vs_true': err_mag,
            'last_activity': dev.get('last_activity'),
            'changes_authored': dev.get('changes_authored'),
            'changes_reviewed': dev.get('changes_reviewed'),
        })

    false_pos = [d for d in detailed if d['pred']==1 and d['label']==0]
    false_neg = [d for d in detailed if d['pred']==0 and d['label']==1]
    # ソート: 自信の高い外れ (閾値からの距離 or 誤差大)
    false_pos.sort(key=lambda x: (-x['margin_to_threshold'], x['prob_error_vs_true']))
    false_neg.sort(key=lambda x: (-x['margin_to_threshold'], x['prob_error_vs_true']))
    top_false_pos = false_pos[:30]
    top_false_neg = false_neg[:30]

    calib, ece = calibration_bins(y, probs, n_bins)
    metrics['ece'] = ece

    return metrics, calib, top_false_pos, top_false_neg, detailed


def main():
    args = parse_args()
    rows = load_predictions(Path(args.pred_file))
    metrics, calib, fp, fn, detailed = analyze(rows, args.prob_threshold, args.calib_bins)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'metrics_alignment.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (out_dir/'calibration.json').write_text(json.dumps(calib, indent=2), encoding='utf-8')
    (out_dir/'top_false_positive.json').write_text(json.dumps(fp, indent=2), encoding='utf-8')
    (out_dir/'top_false_negative.json').write_text(json.dumps(fn, indent=2), encoding='utf-8')
    # 完全詳細はサイズが大きい場合があるので必要なら
    (out_dir/'detailed_predictions.json').write_text(json.dumps(detailed, indent=2), encoding='utf-8')

    print('✅ Alignment Summary:')
    for k,v in metrics.items():
        print(f'  {k}: {v}')
    print(f'💾 Saved -> {out_dir}')
    if fp:
        print(f'  Top false positives: {len(fp)} (showing first 3)')
        for r in fp[:3]:
            print('   FP', r['developer_id'], 'prob', round(r['prob'],3), 'label', r['label'])
    if fn:
        print(f'  Top false negatives: {len(fn)} (showing first 3)')
        for r in fn[:3]:
            print('   FN', r['developer_id'], 'prob', round(r['prob'],3), 'label', r['label'])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
