"""
Baseline Evaluation with Simple Train/Test Split

Train: 2021-2023 (same as Nova IRL train period)
Test: 2023-2024 (same as Nova IRL eval period)

Usage:
    uv run python scripts/experiments/run_baseline_simple_split.py \
        --reviews data/review_requests_nova_neutron.csv \
        --train-start 2021-01-01 \
        --train-end 2023-01-01 \
        --test-start 2023-01-01 \
        --test-end 2024-01-01 \
        --baselines logistic_regression random_forest \
        --output importants/baseline_simple_split/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

from gerrit_retention.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    extract_static_features,
    evaluate_predictions
)


def extract_trajectories_simple(
    df: pd.DataFrame,
    history_start: pd.Timestamp,
    history_end: pd.Timestamp,
    label_start: pd.Timestamp,
    label_end: pd.Timestamp,
    min_history: int = 3
):
    """Extract trajectories with simple time-based split"""
    history_df = df[(df['request_time'] >= history_start) & (df['request_time'] < history_end)]
    label_df = df[(df['request_time'] >= label_start) & (df['request_time'] < label_end)]

    trajectories = []
    for reviewer in history_df['reviewer_email'].unique():
        hist = history_df[history_df['reviewer_email'] == reviewer]
        if len(hist) < min_history:
            continue

        label = label_df[label_df['reviewer_email'] == reviewer]
        if len(label) == 0:
            continue

        continued = (label['label'] == 1).any()

        activity_history = [
            {'type': 'review', 'timestamp': row['request_time'],
             'project': row.get('project', 'unknown'), 'accepted': row['label'] == 1,
             'message': '', 'lines_added': 0, 'lines_deleted': 0, 'files_changed': 1}
            for _, row in hist.iterrows()
        ]

        trajectories.append({
            'developer': {'developer_id': reviewer, 'first_seen': hist['request_time'].min(),
                         'changes_authored': 0, 'changes_reviewed': len(hist),
                         'projects': hist['project'].unique().tolist()},
            'activity_history': activity_history,
            'continued': continued,
            'reviewer': reviewer
        })

    return trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviews', required=True)
    parser.add_argument('--train-start', required=True)
    parser.add_argument('--train-end', required=True)
    parser.add_argument('--test-start', required=True)
    parser.add_argument('--test-end', required=True)
    parser.add_argument('--baselines', nargs='+', default=['logistic_regression', 'random_forest'])
    parser.add_argument('--output', default='importants/baseline_simple_split/')
    parser.add_argument('--irl-results', default='importants/review_acceptance_cross_eval_nova/')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"BASELINE EVALUATION - SIMPLE TRAIN/TEST SPLIT")
    print(f"{'='*70}")
    print(f"Train: {args.train_start} to {args.train_end}")
    print(f"Test:  {args.test_start} to {args.test_end}")
    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"{'='*70}\n")

    np.random.seed(42)
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])

    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    test_start = pd.to_datetime(args.test_start)
    test_end = pd.to_datetime(args.test_end)

    print("Extracting trajectories...")
    train_traj = extract_trajectories_simple(df, train_start, train_end, train_end, test_end)
    test_traj = extract_trajectories_simple(df, test_start, test_end, test_end,
                                           test_end + pd.DateOffset(months=3))

    print(f"Train trajectories: {len(train_traj)}")
    print(f"Test trajectories: {len(test_traj)}")

    train_X, feat_names = extract_static_features(train_traj)
    train_y = np.array([t['continued'] for t in train_traj])
    test_X, _ = extract_static_features(test_traj)
    test_y = np.array([t['continued'] for t in test_traj])

    print(f"Train: {len(train_X)} samples, {train_y.mean():.1%} positive")
    print(f"Test:  {len(test_X)} samples, {test_y.mean():.1%} positive\n")

    results = {}
    for baseline_name in args.baselines:
        print(f"Running {baseline_name}...")

        model = LogisticRegressionBaseline() if baseline_name == 'logistic_regression' else RandomForestBaseline()

        start = time.time()
        model.train({'features': train_X, 'labels': train_y, 'feature_names': feat_names})
        train_time = time.time() - start

        pred = model.predict({'features': test_X})
        metrics = evaluate_predictions(test_y, pred)

        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}, AUC-PR: {metrics['auc_pr']:.3f}, "
              f"F1: {metrics['f1']:.3f} (train: {train_time:.1f}s)\n")

        results[baseline_name] = {'metrics': metrics, 'train_time': train_time}

        # Save
        out_dir = Path(args.output) / baseline_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(results[baseline_name], f, indent=2)

    # Compare with IRL
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH IRL+LSTM")
    print(f"{'='*70}")

    irl_auc_roc = pd.read_csv(Path(args.irl_results) / 'matrix_AUC_ROC.csv', index_col=0)
    irl_auc_pr = pd.read_csv(Path(args.irl_results) / 'matrix_AUC_PR.csv', index_col=0)

    print(f"\n{'Model':<25} {'AUC-ROC':>10} {'AUC-PR':>10} {'F1':>10}")
    print(f"{'-'*55}")
    print(f"{'IRL+LSTM (avg)':<25} {irl_auc_roc.values.mean():>10.3f} "
          f"{irl_auc_pr.values.mean():>10.3f} {'-':>10}")

    for name, res in results.items():
        m = res['metrics']
        print(f"{name:<25} {m['auc_roc']:>10.3f} {m['auc_pr']:>10.3f} {m['f1']:>10.3f}")

    print(f"{'='*70}\n")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
