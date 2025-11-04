"""
Baseline Cross-Evaluation Script

This script runs baseline models (Logistic Regression, Random Forest) using
the same methodology as the Nova IRL cross-evaluation.

Usage:
    uv run python scripts/experiments/run_baseline_cross_evaluation.py \
        --reviews data/review_requests_nova_neutron.csv \
        --train-start 2021-01-01 \
        --train-end 2023-01-01 \
        --eval-start 2023-01-01 \
        --eval-end 2024-01-01 \
        --baselines logistic_regression random_forest \
        --output importants/baseline_cross_eval_nova/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time
from typing import Dict, List, Any, Tuple
import itertools

# Import baseline models
from gerrit_retention.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    extract_static_features,
    evaluate_predictions,
    format_duration
)


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime"""
    return pd.to_datetime(date_str)


def load_and_split_data(
    reviews_path: str,
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load review data and split into train/eval periods.

    Args:
        reviews_path: Path to reviews CSV
        train_start, train_end: Training period
        eval_start, eval_end: Evaluation period

    Returns:
        train_df, eval_df
    """
    print(f"\n{'='*70}")
    print(f"Loading data from {reviews_path}")
    print(f"{'='*70}")

    df = pd.read_csv(reviews_path)

    # Detect date column
    date_col = 'request_time' if 'request_time' in df.columns else 'created'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Parse dates
    train_start_dt = parse_date(train_start)
    train_end_dt = parse_date(train_end)
    eval_start_dt = parse_date(eval_start)
    eval_end_dt = parse_date(eval_end)

    # Split data
    train_df = df[(df[date_col] >= train_start_dt) & (df[date_col] < train_end_dt)]
    eval_df = df[(df[date_col] >= eval_start_dt) & (df[date_col] < eval_end_dt)]

    print(f"\nData split:")
    print(f"  Train: {train_start} to {train_end} ({len(train_df)} reviews)")
    print(f"  Eval:  {eval_start} to {eval_end} ({len(eval_df)} reviews)")

    return train_df, eval_df


def split_period_into_quarters(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    date_col: str = 'request_time'
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

        quarter_df = df[(df[date_col] >= q_start) & (df[date_col] < q_end)]
        quarters.append(quarter_df)

        quarter_name = f"{i*3}-{(i+1)*3}m"
        print(f"  {quarter_name}: {len(quarter_df)} reviews ({q_start.date()} to {q_end.date()})")

    return quarters


def extract_trajectories_from_period(
    period_df: pd.DataFrame,
    label_df: pd.DataFrame,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time'
) -> List[Dict[str, Any]]:
    """
    Extract trajectories from a period and label them based on future activity.

    Args:
        period_df: Historical data for feature extraction
        label_df: Future data for labeling (continuation)
        reviewer_col: Reviewer column name
        date_col: Date column name

    Returns:
        List of trajectories with labels
    """
    reviewers = set(period_df[reviewer_col].unique())
    trajectories = []

    for reviewer in reviewers:
        reviewer_history = period_df[period_df[reviewer_col] == reviewer]
        reviewer_future = label_df[label_df[reviewer_col] == reviewer]

        # Label: continued if there's activity in future period
        continued = len(reviewer_future) > 0

        # Build activity history
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity_history.append({
                'type': 'review',
                'timestamp': row[date_col],
                'project': row.get('project', 'unknown'),
                'message': '',
                'lines_added': 0,
                'lines_deleted': 0,
                'files_changed': 1
            })

        # Developer info
        developer_info = {
            'developer_id': reviewer,
            'first_seen': reviewer_history[date_col].min(),
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
            'future_count': len(reviewer_future)
        })

    return trajectories


def run_cross_evaluation(
    baseline_name: str,
    train_quarters: List[pd.DataFrame],
    eval_quarters: List[pd.DataFrame],
    output_dir: Path
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
            train_trajectories = extract_trajectories_from_period(
                train_quarter, eval_quarter
            )
            eval_trajectories = extract_trajectories_from_period(
                eval_quarter, eval_quarter
            )

            if len(train_trajectories) == 0 or len(eval_trajectories) == 0:
                print(f"    Skipping: insufficient data")
                continue

            # Extract features
            train_features, feature_names = extract_static_features(train_trajectories)
            train_labels = np.array([t['continued'] for t in train_trajectories])

            eval_features, _ = extract_static_features(eval_trajectories)
            eval_labels = np.array([t['continued'] for t in eval_trajectories])

            print(f"    Train: {len(train_features)} samples, {np.mean(train_labels):.1%} positive")
            print(f"    Eval:  {len(eval_features)} samples, {np.mean(eval_labels):.1%} positive")

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
            predictions = model.predict({'features': eval_features})

            # Evaluate
            metrics = evaluate_predictions(eval_labels, predictions)

            print(f"    AUC-ROC: {metrics['auc_roc']:.3f}, AUC-PR: {metrics['auc_pr']:.3f}, "
                  f"F1: {metrics['f1']:.3f} (train: {train_time:.1f}s)")

            # Store results
            results['matrix'][f"{train_name}_{eval_name}"] = {
                'metrics': metrics,
                'train_time': train_time,
                'n_train': len(train_features),
                'n_eval': len(eval_features)
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
    # Convert numpy arrays to lists for JSON serialization
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

        print(f"{baseline_name:<25} "
              f"{auc_roc_matrix.mean():>12.3f} "
              f"{auc_roc_matrix.max():>12.3f} "
              f"{auc_pr_matrix.mean():>12.3f} "
              f"{auc_pr_matrix.max():>12.3f}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run baseline cross-evaluation experiment'
    )
    parser.add_argument(
        '--reviews',
        type=str,
        required=True,
        help='Path to review CSV file'
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
        default='importants/baseline_cross_eval/',
        help='Output directory'
    )
    parser.add_argument(
        '--irl-results',
        type=str,
        default='importants/review_acceptance_cross_eval_nova/',
        help='Path to IRL results for comparison'
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"BASELINE CROSS-EVALUATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"Output: {args.output}")
    print(f"{'='*70}")

    # Load and split data
    train_df, eval_df = load_and_split_data(
        args.reviews,
        args.train_start,
        args.train_end,
        args.eval_start,
        args.eval_end
    )

    # Split into quarters
    print(f"\nSplitting training period into quarters...")
    train_quarters = split_period_into_quarters(
        train_df,
        parse_date(args.train_start),
        parse_date(args.train_end)
    )

    print(f"\nSplitting evaluation period into quarters...")
    eval_quarters = split_period_into_quarters(
        eval_df,
        parse_date(args.eval_start),
        parse_date(args.eval_end)
    )

    # Run cross-evaluation for each baseline
    all_results = {}

    for baseline_name in args.baselines:
        try:
            results = run_cross_evaluation(
                baseline_name,
                train_quarters,
                eval_quarters,
                Path(args.output)
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

    print(f"\n{'='*70}")
    print(f"Cross-evaluation completed!")
    print(f"Results saved to {args.output}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
