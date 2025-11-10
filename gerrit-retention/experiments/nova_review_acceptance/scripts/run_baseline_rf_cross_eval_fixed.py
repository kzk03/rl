"""
Baseline Cross-Evaluation for Review Acceptance Prediction

This script runs baseline models (Logistic Regression, Random Forest) for
review acceptance prediction, using the same methodology as Nova IRL.

Task: Predict whether a reviewer will accept a review request
- Positive: Reviewer accepted at least one review request in eval period
- Negative: Reviewer received requests but rejected all in eval period
- Excluded: Reviewer received no requests in eval period

Usage:
    uv run python scripts/experiments/run_baseline_review_acceptance_cross_eval.py \
        --reviews data/review_requests_nova_neutron.csv \
        --train-start 2021-01-01 \
        --train-end 2023-01-01 \
        --eval-start 2023-01-01 \
        --eval-end 2024-01-01 \
        --baselines logistic_regression random_forest \
        --output importants/baseline_cross_eval_nova/
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Import baseline models
from gerrit_retention.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    evaluate_predictions,
    extract_static_features,
    format_duration,
)


def parse_date(date_str: str) -> pd.Timestamp:
    """Parse date string to pandas Timestamp"""
    return pd.to_datetime(date_str)


def load_review_requests(csv_path: str) -> pd.DataFrame:
    """Load review request data with label column"""
    print(f"\n{'='*70}")
    print(f"Loading review request data from {csv_path}")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path)
    df['request_time'] = pd.to_datetime(df['request_time'])
    df = df.sort_values('request_time').reset_index(drop=True)

    print(f"Total review requests: {len(df)}")
    print(f"Accepted: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1%})")
    print(f"Rejected: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1%})")

    return df


def split_period_into_quarters(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> List[pd.DataFrame]:
    """
    Split period into 4 quarters (3 months each).

    Returns:
        List of 4 DataFrames for each quarter
    """
    quarters = []
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    quarter_months = total_months // 4

    for i in range(4):
        q_start = start_date + pd.DateOffset(months=i * quarter_months)
        q_end = start_date + pd.DateOffset(months=(i + 1) * quarter_months)

        quarter_df = df[(df['request_time'] >= q_start) & (df['request_time'] < q_end)]
        quarters.append(quarter_df)

        quarter_name = f"{i*3}-{(i+1)*3}m"
        print(f"  {quarter_name}: {len(quarter_df)} requests ({q_start.date()} to {q_end.date()})")

    return quarters


def extract_review_acceptance_trajectories(
    history_df: pd.DataFrame,
    label_df: pd.DataFrame,
    min_history_requests: int = 3
) -> List[Dict[str, Any]]:
    """
    Extract trajectories for review acceptance prediction.

    Args:
        history_df: Training period data for feature extraction
        label_df: Evaluation period data for labeling
        min_history_requests: Minimum number of review requests in history

    Returns:
        List of trajectories with labels
    """
    trajectories = []

    # Get reviewers who received requests in training period
    active_reviewers = history_df['reviewer_email'].unique()

    skipped_min_requests = 0
    skipped_no_eval_requests = 0
    positive_count = 0
    negative_count = 0

    for reviewer in active_reviewers:
        # Training period requests
        reviewer_history = history_df[history_df['reviewer_email'] == reviewer]

        # Skip if insufficient history
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue

        # Evaluation period requests
        reviewer_label = label_df[label_df['reviewer_email'] == reviewer]

        # Skip if no requests in evaluation period
        if len(reviewer_label) == 0:
            skipped_no_eval_requests += 1
            continue

        # Label: accepted at least one request in eval period
        accepted_requests = reviewer_label[reviewer_label['label'] == 1]
        continued = len(accepted_requests) > 0

        if continued:
            positive_count += 1
        else:
            negative_count += 1

        # Build activity history
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity_history.append({
                'type': 'review',
                'timestamp': row['request_time'],
                'project': row.get('project', 'unknown'),
                'accepted': row['label'] == 1,
                'message': '',
                'lines_added': 0,
                'lines_deleted': 0,
                'files_changed': 1
            })

        # Developer info
        developer_info = {
            'developer_id': reviewer,
            'first_seen': reviewer_history['request_time'].min(),
            'changes_authored': 0,
            'changes_reviewed': len(reviewer_history),
            'projects': reviewer_history['project'].unique().tolist() if 'project' in reviewer_history.columns else []
        }

        trajectories.append({
            'developer': developer_info,
            'activity_history': activity_history,
            'continued': continued,
            'reviewer': reviewer,
            'history_count': len(reviewer_history),
            'eval_count': len(reviewer_label),
            'acceptance_count': len(accepted_requests),
            'rejection_count': len(reviewer_label) - len(accepted_requests)
        })

    print(f"\nTrajectory extraction:")
    print(f"  Total reviewers: {len(active_reviewers)}")
    print(f"  Skipped (min history): {skipped_min_requests}")
    print(f"  Skipped (no eval requests): {skipped_no_eval_requests}")
    print(f"  Final trajectories: {len(trajectories)}")
    print(f"  Positive (accepted): {positive_count} ({positive_count/len(trajectories)*100:.1%})")
    print(f"  Negative (rejected): {negative_count} ({negative_count/len(trajectories)*100:.1%})")

    return trajectories


def run_cross_evaluation(
    baseline_name: str,
    train_quarters: List[pd.DataFrame],
    eval_quarters: List[pd.DataFrame],
    output_dir: Path,
    min_history_requests: int
) -> Dict[str, Any]:
    """
    Run cross-evaluation: train on each quarter, evaluate on each quarter.

    Returns:
        results: Dictionary with 4x4 matrix of metrics
    """
    print(f"\n{'='*70}")
    print(f"Running Cross-Evaluation: {baseline_name}")
    print(f"{'='*70}")

    results = {
        'baseline': baseline_name,
        'matrix': {},
        'metrics': {
            'auc_roc': np.zeros((4, 4)),
            'auc_pr': np.zeros((4, 4)),
            'f1': np.zeros((4, 4)),
            'precision': np.zeros((4, 4)),
            'recall': np.zeros((4, 4))
        }
    }

    quarter_names = ['0-3m', '3-6m', '6-9m', '9-12m']

    # Train on each quarter
    for train_idx, train_quarter in enumerate(train_quarters):
        train_name = quarter_names[train_idx]
        print(f"\n{'='*70}")
        print(f"Training on {train_name}")
        print(f"{'='*70}")

        # Evaluate on each quarter
        for eval_idx, eval_quarter in enumerate(eval_quarters):
            eval_name = quarter_names[eval_idx]
            print(f"\n  Evaluating on {eval_name}...")

            # Extract trajectories
            trajectories = extract_review_acceptance_trajectories(
                train_quarter, eval_quarter, min_history_requests=min_history_requests
            )

            if len(trajectories) == 0:
                print(f"    Skipping: no trajectories")
                continue

            # Extract features
            features, feature_names = extract_static_features(trajectories)
            labels = np.array([t['continued'] for t in trajectories])

            # Skip if only one class
            if len(np.unique(labels)) < 2:
                print(f"    Skipping: only one class (all {labels[0]})")
                continue

            # Split train/test (80/20)
            n_samples = len(features)
            n_train = int(n_samples * 0.8)

            indices = np.random.RandomState(42).permutation(n_samples)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

            train_features = features[train_indices]
            train_labels = labels[train_indices]
            test_features = features[test_indices]
            test_labels = labels[test_indices]

            # Skip if test set has only one class
            if len(np.unique(test_labels)) < 2:
                print(f"    Skipping: test set has only one class")
                continue

            print(f"    Train: {len(train_features)} samples, {np.mean(train_labels):.1%} positive")
            print(f"    Test:  {len(test_features)} samples, {np.mean(test_labels):.1%} positive")

            # Train model
            if baseline_name == 'logistic_regression':
                model = LogisticRegressionBaseline()
            elif baseline_name == 'random_forest':
                model = RandomForestBaseline()
            else:
                raise ValueError(f"Unknown baseline: {baseline_name}")

            train_data = {
                'features': train_features,
                'labels': train_labels,
                'feature_names': feature_names
            }

            start_time = time.time()
            model.train(train_data)
            train_time = time.time() - start_time

            # Predict
            predictions = model.predict({'features': test_features})

            # Evaluate
            metrics = evaluate_predictions(test_labels, predictions)

            print(f"    AUC-ROC: {metrics['auc_roc']:.3f}, AUC-PR: {metrics['auc_pr']:.3f}, "
                  f"F1: {metrics['f1']:.3f} (train: {train_time:.1f}s)")

            # Store results
            results['matrix'][f"{train_name}_{eval_name}"] = {
                'metrics': metrics,
                'train_time': train_time,
                'n_train': len(train_features),
                'n_test': len(test_features),
                'n_trajectories': len(trajectories)
            }

            # Update metric matrices
            results['metrics']['auc_roc'][train_idx, eval_idx] = metrics['auc_roc']
            results['metrics']['auc_pr'][train_idx, eval_idx] = metrics['auc_pr']
            results['metrics']['f1'][train_idx, eval_idx] = metrics['f1']
            results['metrics']['precision'][train_idx, eval_idx] = metrics['precision']
            results['metrics']['recall'][train_idx, eval_idx] = metrics['recall']

    return results


def save_cross_eval_results(
    results: Dict[str, Any],
    output_dir: Path,
    baseline_name: str
):
    """Save cross-evaluation results in matrix format."""
    output_dir = Path(output_dir) / baseline_name
    output_dir.mkdir(parents=True, exist_ok=True)

    quarter_names = ['0-3m', '3-6m', '6-9m', '9-12m']

    # Save metric matrices as CSV
    for metric_name, matrix in results['metrics'].items():
        df = pd.DataFrame(
            matrix,
            index=quarter_names,
            columns=quarter_names
        )
        df.to_csv(output_dir / f'matrix_{metric_name.upper()}.csv')

    # Save detailed results as JSON
    json_results = {
        'baseline': results['baseline'],
        'matrix': results['matrix'],
        'metrics': {
            k: v.tolist() for k, v in results['metrics'].items()
        }
    }

    with open(output_dir / 'cross_eval_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"  - matrix_*.csv (5 files)")
    print(f"  - cross_eval_results.json")


def print_comparison_summary(
    baseline_results: Dict[str, Dict[str, Any]],
    irl_results_dir: Path
):
    """Print comparison summary with IRL results."""
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")

    # Load IRL matrices
    irl_auc_roc = pd.read_csv(irl_results_dir / 'matrix_AUC_ROC.csv', index_col=0)
    irl_auc_pr = pd.read_csv(irl_results_dir / 'matrix_AUC_PR.csv', index_col=0)

    print(f"\n{'Model':<25} {'Avg AUC-ROC':>12} {'Max AUC-ROC':>12} {'Avg AUC-PR':>12} {'Max AUC-PR':>12}")
    print(f"{'-'*70}")

    # IRL results
    print(f"{'IRL+LSTM':<25} "
          f"{irl_auc_roc.values.mean():>12.3f} "
          f"{irl_auc_roc.values.max():>12.3f} "
          f"{irl_auc_pr.values.mean():>12.3f} "
          f"{irl_auc_pr.values.max():>12.3f}")

    # Baseline results
    for baseline_name, results in baseline_results.items():
        auc_roc_matrix = results['metrics']['auc_roc']
        auc_pr_matrix = results['metrics']['auc_pr']

        # Only include non-zero values (skip empty cells)
        auc_roc_nonzero = auc_roc_matrix[auc_roc_matrix > 0]
        auc_pr_nonzero = auc_pr_matrix[auc_pr_matrix > 0]

        if len(auc_roc_nonzero) > 0:
            print(f"{baseline_name:<25} "
                  f"{auc_roc_nonzero.mean():>12.3f} "
                  f"{auc_roc_nonzero.max():>12.3f} "
                  f"{auc_pr_nonzero.mean():>12.3f} "
                  f"{auc_pr_nonzero.max():>12.3f}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run baseline cross-evaluation for review acceptance prediction'
    )
    parser.add_argument(
        '--reviews',
        type=str,
        required=True,
        help='Path to review requests CSV file (must have label column)'
    )
    parser.add_argument(
        '--train-start',
        type=str,
        required=True,
        help='Training start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--train-end',
        type=str,
        required=True,
        help='Training end date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--eval-start',
        type=str,
        required=True,
        help='Evaluation start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--eval-end',
        type=str,
        required=True,
        help='Evaluation end date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--baselines',
        type=str,
        nargs='+',
        default=['logistic_regression', 'random_forest'],
        help='Baselines to run'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='importants/baseline_review_acceptance_cross_eval/',
        help='Output directory'
    )
    parser.add_argument(
        '--irl-results',
        type=str,
        default='importants/review_acceptance_cross_eval_nova/',
        help='Path to IRL results for comparison'
    )
    parser.add_argument(
        '--min-history-events',
        type=int,
        default=3,
        help='Minimum review requests in history to include a reviewer'
    )
    parser.add_argument(
        '--irl-summary',
        type=Path,
        default=Path('analysis/irl_only_experiments/experiments/latest_run_summary.json'),
        help='Path to IRL latest_run_summary.json for metric comparison'
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"BASELINE REVIEW ACCEPTANCE CROSS-EVALUATION")
    print(f"{'='*70}")
    print(f"Task: Review Acceptance Prediction")
    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"Output: {args.output}")
    print(f"{'='*70}")

    # Set random seed
    np.random.seed(42)

    # Load review requests
    df = load_review_requests(args.reviews)

    # Split into training and evaluation periods
    train_start = parse_date(args.train_start)
    train_end = parse_date(args.train_end)
    eval_start = parse_date(args.eval_start)
    eval_end = parse_date(args.eval_end)

    train_df = df[(df['request_time'] >= train_start) & (df['request_time'] < train_end)]
    eval_df = df[(df['request_time'] >= eval_start) & (df['request_time'] < eval_end)]

    print(f"\nData split:")
    print(f"  Train: {train_start.date()} to {train_end.date()} ({len(train_df)} requests)")
    print(f"  Eval:  {eval_start.date()} to {eval_end.date()} ({len(eval_df)} requests)")

    # Split into quarters
    print(f"\nSplitting training period into quarters...")
    train_quarters = split_period_into_quarters(train_df, train_start, train_end)

    print(f"\nSplitting evaluation period into quarters...")
    eval_quarters = split_period_into_quarters(eval_df, eval_start, eval_end)

    # Run cross-evaluation for each baseline
    all_results = {}

    irl_summary_metrics: Dict[str, float] | None = None

    if args.irl_summary.exists():
        try:
            with args.irl_summary.open() as f:
                irl_entries = json.load(f)
            if isinstance(irl_entries, list) and irl_entries:
                irl_metrics = irl_entries[0].get('metrics', {})
                irl_summary_metrics = {
                    'auc_roc': irl_metrics.get('auc_roc'),
                    'auc_pr': irl_metrics.get('auc_pr'),
                    'f1_score': irl_metrics.get('f1_score'),
                }
        except Exception as exc:
            print(f"\nWarning: failed to load IRL summary at {args.irl_summary}: {exc}")

    for baseline_name in args.baselines:
        try:
            results = run_cross_evaluation(
                baseline_name,
                train_quarters,
                eval_quarters,
                Path(args.output),
                min_history_requests=args.min_history_events
            )

            save_cross_eval_results(
                results,
                Path(args.output),
                baseline_name
            )

            all_results[baseline_name] = results

        except Exception as e:
            print(f"\nError running {baseline_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison with IRL
    if Path(args.irl_results).exists():
        print_comparison_summary(all_results, Path(args.irl_results))

    if irl_summary_metrics is not None and all_results:
        # Collect baseline aggregate metrics
        comparison_rows = []
        for baseline_name, results in all_results.items():
            auc_roc_matrix = results['metrics']['auc_roc']
            auc_pr_matrix = results['metrics']['auc_pr']
            f1_matrix = results['metrics']['f1']

            auc_roc_values = auc_roc_matrix[auc_roc_matrix > 0]
            auc_pr_values = auc_pr_matrix[auc_pr_matrix > 0]
            f1_values = f1_matrix[f1_matrix > 0]

            if len(auc_roc_values) == 0:
                continue

            comparison_rows.append({
                'model': f'Baseline ({baseline_name})',
                'auc_roc': float(np.mean(auc_roc_values)),
                'auc_pr': float(np.mean(auc_pr_values)) if len(auc_pr_values) else float('nan'),
                'f1': float(np.mean(f1_values)) if len(f1_values) else float('nan'),
            })

        comparison_rows.insert(0, {
            'model': 'IRL (latest run)',
            'auc_roc': irl_summary_metrics.get('auc_roc'),
            'auc_pr': irl_summary_metrics.get('auc_pr'),
            'f1': irl_summary_metrics.get('f1_score'),
        })

        table_df = pd.DataFrame(comparison_rows)
        print("\nSimplified comparison (mean metrics across non-empty cells):")
        print(table_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "nan"))

    print(f"\n{'='*70}")
    print(f"Cross-evaluation completed!")
    print(f"Results saved to {args.output}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
