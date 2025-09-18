#!/usr/bin/env python3
"""Run reviewer invitation ranking evaluation.

Generates Cartesian (change x reviewer) candidates with negative sampling and trains a
logistic ranking baseline.

Example:
  uv run python scripts/run_reviewer_invitation_ranking.py \
    --changes data/processed/unified/all_reviews.json \
    --min-total-reviews 20 --max-neg-per-pos 5
"""
from __future__ import annotations

import argparse
import json
import sys
from math import exp
from pathlib import Path

# Ensure local src package is used even if an older version is installed
_repo_root = Path(__file__).resolve().parents[1]
_src_path = _repo_root / 'src'
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import numpy as np

from gerrit_retention.recommendation.reviewer_invitation_ranking import (
    InvitationRankingBuildConfig,
    build_invitation_ranking_samples,
    evaluate_invitation_irl,
    evaluate_invitation_irl_plackett,
    evaluate_invitation_pairwise,
    evaluate_invitation_ranking,
)
from gerrit_retention.utils.labels import jp_label


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--min-total-reviews', type=int, default=20)
    ap.add_argument('--recent-days', type=int, default=30)
    ap.add_argument('--max-neg-per-pos', type=int, default=5)
    ap.add_argument('--hard-fraction', type=float, default=0.5)
    ap.add_argument('--idf-mode', choices=['global', 'recent'], default='global', help='IDF window: global over all changes or recent sliding window')
    ap.add_argument('--idf-recent-days', type=int, default=90, help='Window size (days) when --idf-mode=recent')
    ap.add_argument('--idf-windows', type=int, nargs='*', default=[], help='(Deprecated) Additional recent IDF windows; ignored when using single TF-IDF feature')
    ap.add_argument('--output', default='outputs/reviewer_invitation_ranking')
    ap.add_argument('--mode', choices=['pointwise', 'pairwise', 'irl'], default='pointwise', help='Training/evaluation mode')
    ap.add_argument('--irl-mode', choices=['softmax', 'plackett'], default='softmax', help='IRL variant when --mode irl')
    return ap.parse_args()


def main():
    args = parse_args()
    changes_path = Path(args.changes)
    if not changes_path.exists():
        print(f'❌ changes file not found: {changes_path}')
        return 1
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = InvitationRankingBuildConfig(
        min_total_reviews=args.min_total_reviews,
        recent_days=args.recent_days,
        max_neg_per_pos=args.max_neg_per_pos,
        hard_fraction=args.hard_fraction,
    idf_mode=args.idf_mode,
    idf_recent_days=args.idf_recent_days,
    idf_windows=tuple(args.idf_windows or ()),
    )
    is_pairwise = args.mode == 'pairwise'
    is_irl = args.mode == 'irl'
    if is_irl:
        print(f'📥 building IRL groups from {changes_path}')
        if args.irl_mode == 'plackett':
            metrics, model, test, probs = evaluate_invitation_irl_plackett(changes_path, cfg)
        else:
            metrics, model, test, probs = evaluate_invitation_irl(changes_path, cfg)
    else:
        print(f'📥 building ranking samples from {changes_path}')
        samples = build_invitation_ranking_samples(changes_path, cfg)
        print(f'✅ built samples: {len(samples)} (pos rate ~ {sum(s["label"] for s in samples)/max(1,len(samples)):.3f})')
        if len(samples) < 50:
            print('⚠️ small sample size (<50): metrics may be unstable, continuing for smoke test')
        if is_pairwise:
            metrics, model, test, probs = evaluate_invitation_pairwise(samples)
        else:
            metrics, model, test, probs = evaluate_invitation_ranking(samples)
    print('📊 Metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')
    # Save outputs
    suffix = 'irl' if is_irl else ('pairwise' if is_pairwise else None)
    metrics_name = f'metrics_{suffix}.json' if suffix else 'metrics.json'
    (out_dir / metrics_name).write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    detail = []
    if is_irl:
        # test is already a flat list with 'state','label','ts'
        for s, p in zip(test, probs):
            st = s['state'].copy()
            st.update({'label': s['label'], 'prob': float(p), 'ts': s['ts']})
            detail.append(st)
        (out_dir / ('test_predictions_irl_plackett.json' if args.irl_mode == 'plackett' else 'test_predictions_irl.json')).write_text(json.dumps(detail[:1000], indent=2), encoding='utf-8')
    else:
        for s, p in zip(test, probs):
            st = s['state'].copy()
            st.update({'label': s['label'], 'prob': float(p), 'ts': s['ts']})
            detail.append(st)
        (out_dir / ('test_predictions_pairwise.json' if is_pairwise else 'test_predictions.json')).write_text(json.dumps(detail[:1000], indent=2), encoding='utf-8')

    # --- Export model weights for IRL-style preference interpretation ---
    weight_payload = {}
    try:
        if is_irl and getattr(model, 'theta', None) is not None:
            # Export IRL weights (theta with intercept at end)
            coef = model.theta[:-1].tolist()
            intercept = float(model.theta[-1])
            features = model.features
            scaler_stats = None
            if model.scaler is not None:
                scaler_stats = {
                    'mean': model.scaler.mean_.tolist(),
                    'scale': model.scaler.scale_.tolist(),
                }
            odds_ratio = [float(np.exp(c)) for c in coef]
            abs_sum = sum(abs(c) for c in coef) or 1.0
            rel_importance = [float(abs(c) / abs_sum) for c in coef]
            weight_rows = []
            for f, c, or_, ri in zip(features, coef, odds_ratio, rel_importance):
                weight_rows.append({'feature': f, 'coef_std': c, 'odds_ratio(+1SD)': or_, 'rel_importance': ri})
            weight_payload = {
                'intercept_std': intercept,
                'weights': weight_rows,
                'scaler': scaler_stats,
                'notes': 'irl: softmax方策の線形報酬θ（標準化後）。odds_ratioは+1SDでの相対重みの直感指標。'
            }
            wname = 'weights_irl_plackett.json' if args.irl_mode == 'plackett' else 'weights_irl.json'
            (out_dir / wname).write_text(json.dumps(weight_payload, indent=2, ensure_ascii=False), encoding='utf-8')
            print('🧮 exported ' + wname)
            # CSV
            try:
                csv_lines = []
                header = ['feature', 'japanese_name', 'coef_std', 'abs', 'odds_ratio_plus1SD', 'rel_importance']
                csv_lines.append(','.join(header))
                for f, c, or_, ri in zip(features, coef, odds_ratio, rel_importance):
                    jp = jp_label(f)
                    row = [f, jp, f"{c}", f"{abs(c)}", f"{or_}", f"{ri}"]
                    csv_lines.append(','.join(str(x) for x in row))
                inter_row = ['_intercept', jp_label('_intercept'), f"{intercept}", f"{abs(intercept)}", 'NA', 'NA']
                csv_lines.append(','.join(str(x) for x in inter_row))
                sname = 'weights_summary_irl_plackett.csv' if args.irl_mode == 'plackett' else 'weights_summary_irl.csv'
                (out_dir / sname).write_text('\n'.join(csv_lines), encoding='utf-8')
                print('📄 exported ' + sname)
            except Exception as e:
                print(f'⚠️ weights_summary_irl.csv export failed: {e}')
            # reward analysis style dump
            try:
                inter = intercept
                def sigmoid(z):
                    return 1.0 / (1.0 + np.exp(-z))
                base_prob = sigmoid(inter)
                analysis_rows = []
                means = weight_payload['scaler']['mean'] if scaler_stats else [0.0]*len(features)
                stds = weight_payload['scaler']['scale'] if scaler_stats else [1.0]*len(features)
                for (f, c, m, sd) in zip(features, coef, means, stds):
                    if sd == 0:
                        orig_coef = 0.0
                        delta_prob_1raw = 0.0
                    else:
                        orig_coef = c / sd
                        delta_prob_1raw = sigmoid(inter + c * (1.0 / sd)) - base_prob
                    delta_prob_1sd = sigmoid(inter + c) - base_prob
                    analysis_rows.append({'feature': f, 'coef_std': c, 'coef_per_raw_unit': orig_coef, 'odds_ratio(+1SD)': float(np.exp(c)), 'delta_prob(+1SD)': delta_prob_1sd, 'delta_prob(+1_raw_unit)': delta_prob_1raw, 'mean_raw': m, 'std_raw': sd})
                reward_analysis = {'baseline_logit': inter, 'baseline_prob': base_prob, 'features': analysis_rows, 'notes': 'irl: 条件付きロジットに基づく近似的な限界影響の可視化'}
                rname = 'reward_analysis_irl_plackett.json' if args.irl_mode == 'plackett' else 'reward_analysis_irl.json'
                (out_dir / rname).write_text(json.dumps(reward_analysis, indent=2, ensure_ascii=False), encoding='utf-8')
                print('🧾 exported ' + rname)
            except Exception as e:
                print(f'⚠️ reward analysis IRL export failed: {e}')
        elif getattr(model, 'model', None) is not None:
            coef = model.model.coef_[0].tolist()
            intercept = float(model.model.intercept_[0])
            features = model.features
            scaler_stats = None
            if model.scaler is not None:
                scaler_stats = {
                    'mean': model.scaler.mean_.tolist(),
                    'scale': model.scaler.scale_.tolist(),
                }
            # Odds ratio for +1 std (since coefficients are over standardized features)
            odds_ratio = [float(np.exp(c)) for c in coef]
            # Relative importance (normalized abs weight)
            abs_sum = sum(abs(c) for c in coef) or 1.0
            rel_importance = [float(abs(c) / abs_sum) for c in coef]
            weight_rows = []
            for f, c, or_, ri in zip(features, coef, odds_ratio, rel_importance):
                weight_rows.append({
                    'feature': f,
                    'coef_std': c,
                    'odds_ratio(+1SD)': or_,
                    'rel_importance': ri,
                })
            weight_payload = {
                'intercept_std': intercept,
                'weights': weight_rows,
                'scaler': scaler_stats,
                'notes': ('pairwise' if is_pairwise else 'pointwise') + ': coef_std は StandardScaler 適用後特徴に対する係数。odds_ratio は 1 標準偏差増加時のオッズ倍率 (他条件一定)。'
            }
            (out_dir / ('weights_pairwise.json' if is_pairwise else 'weights.json')).write_text(json.dumps(weight_payload, indent=2, ensure_ascii=False), encoding='utf-8')
            print('🧮 exported ' + ('weights_pairwise.json' if is_pairwise else 'weights.json') + ' for preference analysis')

            # --- Also export a human-friendly CSV with Japanese labels ---
            try:

                # Build rows (excluding intercept for now)
                csv_lines = []
                header = ['feature', 'japanese_name', 'coef_std', 'abs', 'odds_ratio_plus1SD', 'rel_importance']
                csv_lines.append(','.join(header))
                for f, c, or_, ri in zip(features, coef, odds_ratio, rel_importance):
                    jp = jp_label(f)
                    row = [
                        f,
                        jp,
                        f"{c}",
                        f"{abs(c)}",
                        f"{or_}",
                        f"{ri}",
                    ]
                    csv_lines.append(','.join(str(x) for x in row))

                # Append intercept row at the end
                inter_row = ['_intercept', jp_label('_intercept'), f"{intercept}", f"{abs(intercept)}", 'NA', 'NA']
                csv_lines.append(','.join(str(x) for x in inter_row))

                (out_dir / ('weights_summary_pairwise.csv' if is_pairwise else 'weights_summary.csv')).write_text('\n'.join(csv_lines), encoding='utf-8')
                print('📄 exported ' + ('weights_summary_pairwise.csv' if is_pairwise else 'weights_summary.csv'))
            except Exception as e:
                print(f'⚠️ weights_summary.csv export failed: {e}')
            # --- Reward analysis (inverse RL style decomposition) ---
            try:
                inter = intercept
                def sigmoid(z):
                    return 1.0 / (1.0 + exp(-z))
                base_prob = sigmoid(inter)
                analysis_rows = []
                means = weight_payload['scaler']['mean'] if scaler_stats else [0.0]*len(features)
                stds = weight_payload['scaler']['scale'] if scaler_stats else [1.0]*len(features)
                for i, (f, c, m, sd) in enumerate(zip(features, coef, means, stds)):
                    if sd == 0:
                        orig_coef = 0.0
                        delta_prob_1raw = 0.0
                    else:
                        orig_coef = c / sd  # coefficient in original raw units (per +1 raw unit)
                        delta_prob_1raw = sigmoid(inter + c * (1.0 / sd)) - base_prob
                    delta_prob_1sd = sigmoid(inter + c) - base_prob
                    analysis_rows.append({
                        'feature': f,
                        'coef_std': c,
                        'coef_per_raw_unit': orig_coef,
                        'odds_ratio(+1SD)': float(np.exp(c)),
                        'delta_prob(+1SD)': delta_prob_1sd,
                        'delta_prob(+1_raw_unit)': delta_prob_1raw,
                        'mean_raw': m,
                        'std_raw': sd,
                    })
                reward_analysis = {
                    'baseline_logit': inter,
                    'baseline_prob': base_prob,
                    'features': analysis_rows,
                    'notes': ('pairwise' if is_pairwise else 'pointwise') + ': delta_prob は他特徴を平均(標準化後0)に固定した条件付き近似影響。共線性により真の限界効果とは異なる可能性あり。coef_per_raw_unit は未標準化空間での log-odds 変化。'
                }
                (out_dir / ('reward_analysis_pairwise.json' if is_pairwise else 'reward_analysis.json')).write_text(json.dumps(reward_analysis, indent=2, ensure_ascii=False), encoding='utf-8')
                print('🧾 exported ' + ('reward_analysis_pairwise.json' if is_pairwise else 'reward_analysis.json'))
            except Exception as e:  # nested
                print(f'⚠️ reward analysis export failed: {e}')
    except Exception as e:
        print(f'⚠️ weight export failed: {e}')
    print(f'💾 wrote outputs to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
