"""
Baseline Evaluation with 3-Month Windows (Correct Experimental Design)

このスクリプトは、3ヶ月幅のfuture windowsを使用した正しい実験設計版です。

Future Windows（3ヶ月幅）:
- 3ヶ月予測: future window 0-3m
- 6ヶ月予測: future window 3-6m
- 9ヶ月予測: future window 6-9m
- 12ヶ月予測: future window 9-12m

⚠️ 注意：
6ヶ月幅バージョン（IRL-aligned）は run_baseline_nova_fair_comparison.py を使用してください。

Usage:
    uv run python scripts/experiments/run_baseline_nova_3month_windows.py \
        --reviews data/review_requests_nova.csv \
        --train-start 2021-01-01 \
        --train-end 2023-01-01 \
        --eval-start 2023-01-01 \
        --eval-end 2024-01-01 \
        --baselines logistic_regression random_forest \
        --output importants/baseline_nova_3month_windows/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time
from typing import Dict, List, Any, Tuple

# Import baseline models
from gerrit_retention.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    extract_static_features,
    evaluate_predictions,
    format_duration
)


def parse_date(date_str: str) -> pd.Timestamp:
    """Parse date string to pandas Timestamp"""
    return pd.to_datetime(date_str)


def load_review_requests(csv_path: str) -> pd.DataFrame:
    """Load review request data"""
    print(f"\nLoading review request data: {csv_path}")
    df = pd.read_csv(csv_path)
    df['request_time'] = pd.to_datetime(df['request_time'])
    df = df.sort_values('request_time').reset_index(drop=True)

    print(f"  Total requests: {len(df)}")
    print(f"  Date range: {df['request_time'].min()} to {df['request_time'].max()}")
    print(f"  Accepted: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1%})")

    return df


def get_future_windows_3month() -> List[Tuple[int, int]]:
    """
    Return 3-month width future windows.

    Returns:
        List of (window_start_months, window_end_months):
        - (0, 3): 3ヶ月予測
        - (3, 6): 6ヶ月予測
        - (6, 9): 9ヶ月予測
        - (9, 12): 12ヶ月予測
    """
    return [
        (0, 3),   # 3ヶ月予測
        (3, 6),   # 6ヶ月予測
        (6, 9),   # 9ヶ月予測
        (9, 12),  # 12ヶ月予測
    ]


def split_eval_quarters(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp, int, int]]:
    """
    Split evaluation period into 4 quarters (3-month each).

    Args:
        start: Evaluation period start
        end: Evaluation period end

    Returns:
        List of (quarter_start, quarter_end, window_start, window_end)
    """
    total_months = (end.year - start.year) * 12 + (end.month - start.month)
    quarter_months = total_months // 4

    quarters = []
    for i in range(4):
        q_start = start + pd.DateOffset(months=i * quarter_months)
        q_end = start + pd.DateOffset(months=(i + 1) * quarter_months)
        window_start = i * quarter_months
        window_end = (i + 1) * quarter_months
        quarters.append((q_start, q_end, window_start, window_end))

    return quarters


def extract_trajectories_monthly_training(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    min_history: int = 3
) -> List[Dict[str, Any]]:
    """
    Extract trajectories using monthly training (same as IRL).

    For each month in training period:
    - Features: train_start ~ month_end
    - Labels: month_end + future_window

    This allows using all months including late periods (9-12m).

    Args:
        df: Full review data
        train_start: Training period start
        train_end: Training period end
        future_window_start_months: Future window start (months)
        future_window_end_months: Future window end (months)
        min_history: Minimum requests in history

    Returns:
        List of trajectories (aggregated from all months)
    """
    print(f"    [Monthly Training - Same as IRL]")
    print(f"      訓練期間: {train_start.date()} ～ {train_end.date()}")
    print(f"      Future window: {future_window_start_months}～{future_window_end_months}ヶ月")

    # Generate monthly dates
    history_months = pd.date_range(
        start=train_start,
        end=train_end,
        freq='MS'  # Month start
    )

    all_trajectories = []
    monthly_counts = []

    # For each month (except last one)
    for month_start in history_months[:-1]:
        month_end = month_start + pd.DateOffset(months=1)

        # Label period
        future_start = month_end + pd.DateOffset(months=future_window_start_months)
        future_end = month_end + pd.DateOffset(months=future_window_end_months)

        # Clip future_end to train_end (same as IRL)
        if future_end > train_end:
            future_end = train_end

        # Skip if future_start >= train_end
        if future_start >= train_end:
            continue

        # History period: train_start ~ month_end
        history_df = df[(df['request_time'] >= train_start) &
                        (df['request_time'] < month_end)]

        # Label period: future_start ~ future_end
        label_df = df[(df['request_time'] >= future_start) &
                      (df['request_time'] < future_end)]

        active_reviewers = history_df['reviewer_email'].unique()
        month_trajectories = []

        for reviewer in active_reviewers:
            reviewer_history = history_df[history_df['reviewer_email'] == reviewer]

            if len(reviewer_history) < min_history:
                continue

            reviewer_label = label_df[label_df['reviewer_email'] == reviewer]

            if len(reviewer_label) == 0:
                continue

            # Label: accepted at least one
            accepted = (reviewer_label['label'] == 1).any()

            # Activity history
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
                'projects': reviewer_history['project'].unique().tolist()
            }

            month_trajectories.append({
                'developer': developer_info,
                'activity_history': activity_history,
                'continued': accepted,
                'reviewer': reviewer,
                'month': month_start.strftime('%Y-%m')
            })

        if len(month_trajectories) > 0:
            monthly_counts.append(len(month_trajectories))
            all_trajectories.extend(month_trajectories)

    if len(all_trajectories) > 0:
        pos_count = sum(1 for t in all_trajectories if t['continued'])
        print(f"      Total: {len(all_trajectories)} trajectories from {len(monthly_counts)} months")
        print(f"      Positive: {pos_count}/{len(all_trajectories)} ({pos_count/len(all_trajectories)*100:.1%})")
        print(f"      Per month: {np.mean(monthly_counts):.1f} ± {np.std(monthly_counts):.1f}")
    else:
        print(f"      No trajectories extracted")

    return all_trajectories


def extract_trajectories_with_maxdate(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_window_end_months: int,
    label_start: pd.Timestamp,
    label_end: pd.Timestamp,
    min_history: int = 3
) -> List[Dict[str, Any]]:
    """
    Extract trajectories with max-date constraint (DEPRECATED - use monthly training instead).

    Args:
        df: Full review data
        train_start: Training period start
        train_end: Training period end
        future_window_end_months: Future window end (months) for max-date calculation
        label_start: Label period start
        label_end: Label period end
        min_history: Minimum requests in history

    Returns:
        List of trajectories
    """
    # Calculate max-date (same as IRL's constraint)
    max_date = train_end - pd.DateOffset(months=future_window_end_months)

    # History period: train_start ~ max_date (NOT train_end!)
    history_start = train_start
    history_end = max_date

    print(f"    [Fair Comparison] max-date設定")
    print(f"      特徴量期間: {history_start.date()} ～ {history_end.date()}")
    print(f"      訓練期間内ラベル期間: {max_date.date()} ～ {train_end.date()} (ラベル付けのみ)")
    print(f"      評価ラベル期間: {label_start.date()} ～ {label_end.date()}")

    # History period data (for features)
    history_df = df[(df['request_time'] >= history_start) &
                    (df['request_time'] < history_end)]

    # Label period data (for training labels)
    # For training: use max_date ~ train_end
    # For evaluation: use label_start ~ label_end
    if label_start >= train_end:
        # Evaluation mode: use eval period for labels
        label_df = df[(df['request_time'] >= label_start) &
                      (df['request_time'] < label_end)]
    else:
        # Training mode: use max_date ~ train_end for labels
        label_df = df[(df['request_time'] >= max_date) &
                      (df['request_time'] < train_end)]

    trajectories = []
    active_reviewers = history_df['reviewer_email'].unique()

    skipped_min_history = 0
    skipped_no_label = 0

    for reviewer in active_reviewers:
        # History (features)
        reviewer_history = history_df[history_df['reviewer_email'] == reviewer]

        if len(reviewer_history) < min_history:
            skipped_min_history += 1
            continue

        # Label
        reviewer_label = label_df[label_df['reviewer_email'] == reviewer]

        if len(reviewer_label) == 0:
            skipped_no_label += 1
            continue

        # Label: accepted at least one
        accepted = (reviewer_label['label'] == 1).any()

        # Activity history (from features period only)
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
            'projects': reviewer_history['project'].unique().tolist()
        }

        trajectories.append({
            'developer': developer_info,
            'activity_history': activity_history,
            'continued': accepted,
            'reviewer': reviewer,
        })

    print(f"    Extracted {len(trajectories)} trajectories "
          f"(skipped: {skipped_min_history} min_history, {skipped_no_label} no_label)")
    if len(trajectories) > 0:
        pos_count = sum(1 for t in trajectories if t['continued'])
        print(f"    Positive: {pos_count}/{len(trajectories)} ({pos_count/len(trajectories)*100:.1%})")

    return trajectories


def train_and_evaluate_baseline(
    baseline_name: str,
    train_trajectories: List[Dict[str, Any]],
    eval_trajectories: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Train baseline on train_trajectories and evaluate on eval_trajectories.
    """
    # Extract features
    train_features, feature_names = extract_static_features(train_trajectories)
    train_labels = np.array([t['continued'] for t in train_trajectories])

    eval_features, _ = extract_static_features(eval_trajectories)
    eval_labels = np.array([t['continued'] for t in eval_trajectories])

    # Check if eval has both classes
    if len(np.unique(eval_labels)) < 2:
        return None

    # Initialize model
    if baseline_name == 'logistic_regression':
        model = LogisticRegressionBaseline()
    elif baseline_name == 'random_forest':
        model = RandomForestBaseline()
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    # Train
    train_data = {
        'features': train_features,
        'labels': train_labels,
        'feature_names': feature_names
    }

    start_time = time.time()
    model.train(train_data)
    train_time = time.time() - start_time

    # Evaluate
    predictions = model.predict({'features': eval_features})
    metrics = evaluate_predictions(eval_labels, predictions)

    return {
        'metrics': metrics,
        'train_time': train_time,
        'n_train': len(train_trajectories),
        'n_eval': len(eval_trajectories),
        'predictions': predictions,
        'true_labels': eval_labels
    }


def run_cross_evaluation_fair(
    baseline_name: str,
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    future_windows: List[Tuple[int, int]],
    eval_quarters: List[Tuple[pd.Timestamp, pd.Timestamp, int, int]],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run cross-evaluation with 3-month width future windows.
    """
    print(f"\n{'='*70}")
    print(f"Running Baseline (3-Month Windows): {baseline_name}")
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
    prediction_names = ['3m prediction', '6m prediction', '9m prediction', '12m prediction']

    # For each future window (3-month width)
    for train_idx, (train_window_start, train_window_end) in enumerate(future_windows):
        train_name = quarter_names[train_idx]
        print(f"\n{'='*60}")
        print(f"Training Period: {train_name} ({prediction_names[train_idx]})")
        print(f"  Future Window: {train_window_start}～{train_window_end}ヶ月")
        print(f"{'='*60}")

        # Extract train trajectories using monthly training (same as IRL)
        train_trajectories = extract_trajectories_monthly_training(
            df,
            train_start=train_start,
            train_end=train_end,
            future_window_start_months=train_window_start,
            future_window_end_months=train_window_end,
            min_history=3
        )

        if len(train_trajectories) == 0:
            print(f"  Skipping {train_name}: no train trajectories")
            continue

        # For each eval period
        for eval_idx, (eval_q_start, eval_q_end, eval_window_start, eval_window_end) in enumerate(eval_quarters):
            eval_name = quarter_names[eval_idx]
            print(f"\n  Eval Period: {eval_name} ({eval_q_start.date()} to {eval_q_end.date()})")

            # Extract eval trajectories
            # Features: train_start ~ train_end (full training period)
            # Labels: eval period
            history_df = df[(df['request_time'] >= train_start) &
                           (df['request_time'] < train_end)]
            label_df = df[(df['request_time'] >= eval_q_start) &
                         (df['request_time'] < eval_q_end)]

            active_reviewers = history_df['reviewer_email'].unique()
            eval_trajectories = []

            for reviewer in active_reviewers:
                reviewer_history = history_df[history_df['reviewer_email'] == reviewer]

                if len(reviewer_history) < 3:
                    continue

                reviewer_label = label_df[label_df['reviewer_email'] == reviewer]

                if len(reviewer_label) == 0:
                    continue

                accepted = (reviewer_label['label'] == 1).any()

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

                developer_info = {
                    'developer_id': reviewer,
                    'first_seen': reviewer_history['request_time'].min(),
                    'changes_authored': 0,
                    'changes_reviewed': len(reviewer_history),
                    'projects': reviewer_history['project'].unique().tolist()
                }

                eval_trajectories.append({
                    'developer': developer_info,
                    'activity_history': activity_history,
                    'continued': accepted,
                    'reviewer': reviewer,
                })

            print(f"    Extracted {len(eval_trajectories)} eval trajectories")

            if len(eval_trajectories) == 0:
                print(f"    Skipping: no eval trajectories")
                continue

            # Train and evaluate
            result = train_and_evaluate_baseline(
                baseline_name,
                train_trajectories,
                eval_trajectories
            )

            if result is None:
                print(f"    Skipping: only one class in eval")
                continue

            metrics = result['metrics']
            print(f"    AUC-ROC: {metrics['auc_roc']:.3f}, "
                  f"AUC-PR: {metrics['auc_pr']:.3f}, "
                  f"F1: {metrics['f1']:.3f} "
                  f"(train: {result['train_time']:.1f}s, "
                  f"n_train: {result['n_train']}, n_eval: {result['n_eval']})")

            # Store results
            results['matrix'][f"{train_name}_{eval_name}"] = result
            results['metrics']['auc_roc'][train_idx, eval_idx] = metrics['auc_roc']
            results['metrics']['auc_pr'][train_idx, eval_idx] = metrics['auc_pr']
            results['metrics']['f1'][train_idx, eval_idx] = metrics['f1']
            results['metrics']['precision'][train_idx, eval_idx] = metrics['precision']
            results['metrics']['recall'][train_idx, eval_idx] = metrics['recall']

    return results


def save_results(results: Dict[str, Any], output_dir: Path, baseline_name: str):
    """Save results matching Nova IRL structure"""
    output_dir = Path(output_dir) / baseline_name
    output_dir.mkdir(parents=True, exist_ok=True)

    quarter_names = ['0-3m', '3-6m', '6-9m', '9-12m']

    # Save matrices
    for metric_name, matrix in results['metrics'].items():
        df = pd.DataFrame(matrix, index=quarter_names, columns=quarter_names)
        df.to_csv(output_dir / f'matrix_{metric_name.upper()}.csv')

    # Save detailed JSON
    json_results = {
        'baseline': results['baseline'],
        'metrics': {k: v.tolist() for k, v in results['metrics'].items()}
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def print_comparison(
    baseline_results: Dict[str, Dict[str, Any]],
    irl_dir: Path
):
    """Print comparison with IRL"""
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH IRL+LSTM (Fair Comparison)")
    print(f"{'='*70}")

    # Load IRL results
    irl_auc_roc = pd.read_csv(irl_dir / 'matrix_AUC_ROC.csv', index_col=0)
    irl_auc_pr = pd.read_csv(irl_dir / 'matrix_AUC_PR.csv', index_col=0)

    print(f"\n{'Model':<25} {'Avg AUC-ROC':>12} {'Max AUC-ROC':>12} "
          f"{'Avg AUC-PR':>12} {'Max AUC-PR':>12}")
    print(f"{'-'*70}")

    # IRL
    print(f"{'IRL+LSTM':<25} "
          f"{irl_auc_roc.values.mean():>12.3f} "
          f"{irl_auc_roc.values.max():>12.3f} "
          f"{irl_auc_pr.values.mean():>12.3f} "
          f"{irl_auc_pr.values.max():>12.3f}")

    # Baselines
    for name, results in baseline_results.items():
        auc_roc = results['metrics']['auc_roc']
        auc_pr = results['metrics']['auc_pr']

        # Only non-zero values
        auc_roc_nz = auc_roc[auc_roc > 0]
        auc_pr_nz = auc_pr[auc_pr > 0]

        if len(auc_roc_nz) > 0:
            print(f"{name:<25} "
                  f"{auc_roc_nz.mean():>12.3f} "
                  f"{auc_roc_nz.max():>12.3f} "
                  f"{auc_pr_nz.mean():>12.3f} "
                  f"{auc_pr_nz.max():>12.3f}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Baseline evaluation with fair comparison to Nova IRL (max-date方式)'
    )
    parser.add_argument('--reviews', required=True, help='Review requests CSV')
    parser.add_argument('--train-start', required=True, help='Train start (YYYY-MM-DD)')
    parser.add_argument('--train-end', required=True, help='Train end (YYYY-MM-DD)')
    parser.add_argument('--eval-start', required=True, help='Eval start (YYYY-MM-DD)')
    parser.add_argument('--eval-end', required=True, help='Eval end (YYYY-MM-DD)')
    parser.add_argument('--baselines', nargs='+',
                       default=['logistic_regression', 'random_forest'])
    parser.add_argument('--output', default='importants/baseline_nova_fair_comparison/')
    parser.add_argument('--irl-results',
                       default='importants/review_acceptance_cross_eval_nova/')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"BASELINE EVALUATION (3-MONTH WINDOWS)")
    print(f"{'='*70}")
    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"Output: {args.output}")
    print(f"\n✅  正しい実験設計（3ヶ月幅のfuture windows）")
    print(f"    - 3ヶ月予測: future window 0-3m")
    print(f"    - 6ヶ月予測: future window 3-6m")
    print(f"    - 9ヶ月予測: future window 6-9m")
    print(f"    - 12ヶ月予測: future window 9-12m")
    print(f"{'='*70}")

    # Set seed
    np.random.seed(42)

    # Load data
    df = load_review_requests(args.reviews)

    # Parse dates
    train_start = parse_date(args.train_start)
    train_end = parse_date(args.train_end)
    eval_start = parse_date(args.eval_start)
    eval_end = parse_date(args.eval_end)

    print(f"\nTrain period: {train_start.date()} to {train_end.date()}")
    print(f"Eval period:  {eval_start.date()} to {eval_end.date()}")

    # Get future windows (3-month width)
    future_windows = get_future_windows_3month()
    eval_quarters = split_eval_quarters(eval_start, eval_end)

    print(f"\nFuture Windows (3-month width):")
    prediction_names = ['3m prediction', '6m prediction', '9m prediction', '12m prediction']
    for i, (ws, we) in enumerate(future_windows):
        print(f"  {ws}-{we}m: {prediction_names[i]}")

    print(f"\nEval quarters:")
    for i, (s, e, ws, we) in enumerate(eval_quarters):
        print(f"  {ws}-{we}m: {s.date()} to {e.date()}")

    # Run baselines
    all_results = {}

    for baseline_name in args.baselines:
        try:
            results = run_cross_evaluation_fair(
                baseline_name, df, train_start, train_end, eval_start, eval_end,
                future_windows, eval_quarters, Path(args.output)
            )
            save_results(results, Path(args.output), baseline_name)
            all_results[baseline_name] = results

        except Exception as e:
            print(f"\nError running {baseline_name}: {e}")
            import traceback
            traceback.print_exc()

    # Compare with IRL
    if Path(args.irl_results).exists():
        print_comparison(all_results, Path(args.irl_results))

    print(f"\n{'='*70}")
    print(f"Fair Evaluation completed!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
