#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure local src is importable
_repo_root = Path(__file__).resolve().parents[1]
_src_path = _repo_root / 'src'
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from gerrit_retention.recommendation.reviewer_acceptance_after_invite import (
    AcceptanceBuildConfig,
    evaluate_acceptance_after_invite,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--changes', default='data/processed/unified/all_reviews.json')
    ap.add_argument('--min-total-reviews', type=int, default=20)
    ap.add_argument('--recent-days', type=int, default=30)
    ap.add_argument('--output', default='outputs/reviewer_acceptance_after_invite')
    return ap.parse_args()


def main():
    args = parse_args()
    changes_path = Path(args.changes)
    if not changes_path.exists():
        print(f'❌ changes file not found: {changes_path}')
        return 1
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AcceptanceBuildConfig(
        min_total_reviews=args.min_total_reviews,
        recent_days=args.recent_days,
    )
    print(f'📥 building invited-only acceptance samples from {changes_path}')
    metrics, model, test, probs = evaluate_acceptance_after_invite(changes_path, cfg)
    print('📊 Metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    # Export predictions (subset)
    detail = []
    for s, p in zip(test, probs):
        st = s['state'].copy()
        st.update({'label': s['label'], 'prob': float(p), 'ts': s['ts']})
        detail.append(st)
    (out_dir / 'test_predictions.json').write_text(json.dumps(detail[:1000], indent=2), encoding='utf-8')
    # Export weights summary and reward-style analysis
    weight_payload = {}
    try:
        if getattr(model, 'model', None) is not None:
            coef = model.model.coef_[0].tolist()
            intercept = float(model.model.intercept_[0])
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
                'notes': 'acceptance: 招待後の実参加確率を標準化線形モデルで推定。odds_ratioは+1SDの直感指標。'
            }
            (out_dir / 'weights.json').write_text(json.dumps(weight_payload, indent=2, ensure_ascii=False), encoding='utf-8')
            # CSV
            try:
                jp_labels = {
                    'reviewer_total_reviews': '累積レビュー参加数',
                    'reviewer_recent_reviews_30d': '過去30日レビュー参加数',
                    'reviewer_gap_days': '最終参加からの経過日数',
                    'match_off_specialty_flag': '専門外フラグ(過去30日同一プロジェクト経験無)',
                    'off_specialty_recent_ratio': '専門外比率(=1-同一プロジェクト比率)',
                    'reviewer_recent_reviews_7d': '過去7日レビュー参加数',
                    'reviewer_proj_share_30d': '過去30日同一プロジェクト比率',
                    'reviewer_active_flag_30d': '過去30日活動フラグ',
                    'reviewer_proj_prev_reviews_30d': '過去30日同一プロジェクト参加数',
                    'reviewer_file_tfidf_cosine_30d': '過去30日ファイルトークンTF-IDFコサイン',
                    'reviewer_pending_reviews': '未クローズレビュー数',
                    'reviewer_workload_deviation_z': '活動量Zスコア',
                    'change_current_invited_cnt': '変更での招待人数',
                    'reviewer_night_activity_share_30d': '過去30日夜間活動比率',
                    'reviewer_overload_flag': '過負荷フラグ(平均+σ以上)',
                    '_intercept': 'ベースライン切片',
                }
                csv_lines = []
                header = ['feature', 'japanese_name', 'coef_std', 'abs', 'odds_ratio_plus1SD', 'rel_importance']
                csv_lines.append(','.join(header))
                for f, c, or_, ri in zip(features, coef, odds_ratio, rel_importance):
                    jp = jp_labels.get(f, f)
                    row = [f, jp, f"{c}", f"{abs(c)}", f"{or_}", f"{ri}"]
                    csv_lines.append(','.join(str(x) for x in row))
                inter_row = ['_intercept', jp_labels.get('_intercept', '_intercept'), f"{intercept}", f"{abs(intercept)}", 'NA', 'NA']
                csv_lines.append(','.join(str(x) for x in inter_row))
                (out_dir / 'weights_summary.csv').write_text('\n'.join(csv_lines), encoding='utf-8')
            except Exception as e:
                print(f'⚠️ weights_summary.csv export failed: {e}')
            # Reward-style analysis
            try:
                inter = intercept
                def sigmoid(z):
                    return 1.0 / (1.0 + np.exp(-z))
                base_prob = sigmoid(inter)
                analysis_rows = []
                means = weight_payload['scaler']['mean'] if scaler_stats else [0.0] * len(features)
                stds = weight_payload['scaler']['scale'] if scaler_stats else [1.0] * len(features)
                for (f, c, m, sd) in zip(features, coef, means, stds):
                    if sd == 0:
                        orig_coef = 0.0
                        delta_prob_1raw = 0.0
                    else:
                        orig_coef = c / sd
                        delta_prob_1raw = sigmoid(inter + c * (1.0 / sd)) - base_prob
                    delta_prob_1sd = sigmoid(inter + c) - base_prob
                    analysis_rows.append({'feature': f, 'coef_std': c, 'coef_per_raw_unit': orig_coef, 'odds_ratio(+1SD)': float(np.exp(c)), 'delta_prob(+1SD)': delta_prob_1sd, 'delta_prob(+1_raw_unit)': delta_prob_1raw, 'mean_raw': m, 'std_raw': sd})
                reward_analysis = {'baseline_logit': inter, 'baseline_prob': base_prob, 'features': analysis_rows, 'notes': 'acceptance: ロジスティック回帰に基づく近似的な限界影響の可視化'}
                (out_dir / 'reward_analysis.json').write_text(json.dumps(reward_analysis, indent=2, ensure_ascii=False), encoding='utf-8')
            except Exception as e:
                print(f'⚠️ reward_analysis export failed: {e}')
    except Exception as e:
        print(f'⚠️ weight export failed: {e}')
    print(f'💾 wrote outputs to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
