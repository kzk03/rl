"""
Baseline Comparison Experiment

This script runs multiple baseline models (Logistic Regression, Random Forest)
on the same data as IRL+LSTM and compares their performance.

Usage:
    # Run single baseline
    uv run python scripts/experiments/run_baseline_comparison.py \
        --reviews data/review_requests_no_bots.csv \
        --snapshot-date 2020-01-01 \
        --history-months 12 \
        --target-months 6 \
        --baseline logistic_regression \
        --output importants/baseline_experiments/logistic_regression/

    # Run multiple baselines
    uv run python scripts/experiments/run_baseline_comparison.py \
        --reviews data/review_requests_no_bots.csv \
        --snapshot-date 2020-01-01 \
        --history-months 12 \
        --target-months 6 \
        --baselines logistic_regression random_forest \
        --output importants/baseline_experiments/comparison_results/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any, Tuple

# Import baseline models
from gerrit_retention.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    extract_static_features,
    evaluate_predictions,
    save_results,
    format_duration
)

# Import IRL system for data loading
from gerrit_retention.rl_prediction.retention_irl_system import RetentionIRLSystem


def load_and_prepare_data(
    reviews_path: str,
    snapshot_date: str,
    history_months: int,
    target_months: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load review data and prepare train/test splits.

    Args:
        reviews_path: Path to reviews CSV
        snapshot_date: Snapshot date string
        history_months: Months for learning period
        target_months: Months for prediction period

    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Loading data from {reviews_path}")
    print(f"Snapshot: {snapshot_date}")
    print(f"History: {history_months} months, Target: {target_months} months")
    print(f"{'='*60}")

    # Use IRL system's data loading mechanism
    irl_system = RetentionIRLSystem({})

    # Load trajectories using the same method as IRL
    df = pd.read_csv(reviews_path)

    # Convert snapshot date
    snapshot_dt = pd.to_datetime(snapshot_date)
    history_start = snapshot_dt - pd.DateOffset(months=history_months)
    target_end = snapshot_dt + pd.DateOffset(months=target_months)

    print(f"History period: {history_start} to {snapshot_dt}")
    print(f"Target period: {snapshot_dt} to {target_end}")

    # Extract trajectories
    train_trajectories, test_trajectories = irl_system.extract_trajectories_from_reviews(
        df,
        snapshot_date=snapshot_dt,
        history_months=history_months,
        target_months=target_months,
        train_ratio=0.8
    )

    print(f"\nTrajectories extracted:")
    print(f"  Train: {len(train_trajectories)}")
    print(f"  Test: {len(test_trajectories)}")

    # Convert to static features
    print("\nExtracting static features from time-series...")
    train_features, feature_names = extract_static_features(train_trajectories)
    train_labels = np.array([t['continued'] for t in train_trajectories])

    test_features, _ = extract_static_features(test_trajectories)
    test_labels = np.array([t['continued'] for t in test_trajectories])

    print(f"Feature dimensions: {train_features.shape[1]}")
    print(f"Train positive rate: {np.mean(train_labels):.1%}")
    print(f"Test positive rate: {np.mean(test_labels):.1%}")

    train_data = {
        'features': train_features,
        'labels': train_labels,
        'feature_names': feature_names
    }

    test_data = {
        'features': test_features,
        'labels': test_labels
    }

    return train_data, test_data


def run_baseline(
    baseline_name: str,
    train_data: Dict[str, Any],
    test_data: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run a single baseline model.

    Args:
        baseline_name: Name of baseline ('logistic_regression', 'random_forest')
        train_data: Training data dictionary
        test_data: Test data dictionary
        config: Optional baseline configuration

    Returns:
        Results dictionary with predictions, metrics, feature importance
    """
    print(f"\n{'='*60}")
    print(f"Running baseline: {baseline_name}")
    print(f"{'='*60}")

    # Initialize baseline model
    if baseline_name == 'logistic_regression':
        model = LogisticRegressionBaseline(config)
    elif baseline_name == 'random_forest':
        model = RandomForestBaseline(config)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    # Train model
    print("\nTraining model...")
    train_info = model.train(train_data)

    print(f"Training completed in {format_duration(train_info['training_time'])}")
    print(f"  Samples: {train_info['n_samples']}")
    print(f"  Features: {train_info['n_features']}")
    print(f"  Positive rate: {train_info['positive_rate']:.1%}")

    if 'oob_score' in train_info:
        print(f"  OOB Score: {train_info['oob_score']:.3f}")

    # Predict on test set
    print("\nPredicting on test set...")
    start_time = time.time()
    predictions = model.predict(test_data)
    inference_time = time.time() - start_time
    avg_inference_time_ms = (inference_time / len(predictions)) * 1000

    print(f"Prediction completed in {format_duration(inference_time)}")
    print(f"  Avg per sample: {avg_inference_time_ms:.2f}ms")

    # Evaluate predictions
    print("\nEvaluating predictions...")
    metrics = evaluate_predictions(test_data['labels'], predictions)

    print(f"\nPerformance Metrics:")
    print(f"  AUC-PR:     {metrics['auc_pr']:.3f}")
    print(f"  AUC-ROC:    {metrics['auc_roc']:.3f}")
    print(f"  F1 Score:   {metrics['f1']:.3f}")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  Accuracy:   {metrics['accuracy']:.3f}")

    # Get feature importance
    print("\nExtracting feature importance...")
    feature_importance = model.get_feature_importance()

    # Show top 10 features
    print("\nTop 10 Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"  {i:2d}. {feature:30s}  {importance:.4f}")

    # Compile results
    results = {
        'baseline_name': baseline_name,
        'predictions': predictions,
        'true_labels': test_data['labels'],
        'metrics': metrics,
        'feature_importance': feature_importance,
        'training_time': train_info['training_time'],
        'inference_time': inference_time,
        'avg_inference_time_ms': avg_inference_time_ms,
        'model_info': model.get_model_info(),
        'train_info': train_info
    }

    # Add model summary if available
    if hasattr(model, 'get_model_summary'):
        results['model_summary'] = model.get_model_summary()

    return results, model


def save_baseline_results(
    results: Dict[str, Any],
    model: Any,
    output_dir: Path,
    baseline_name: str,
    history_months: int,
    target_months: int
):
    """
    Save baseline results to disk.

    Args:
        results: Results dictionary
        model: Trained model
        output_dir: Output directory
        baseline_name: Name of baseline
        history_months: History months
        target_months: Target months
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_df = pd.DataFrame({
        'prediction': results['predictions'],
        'true_label': results['true_labels']
    })
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)

    # Save metrics
    metrics_data = {
        'baseline': baseline_name,
        'history_months': history_months,
        'target_months': target_months,
        'metrics': results['metrics'],
        'training_time_seconds': results['training_time'],
        'inference_time_seconds': results['inference_time'],
        'avg_inference_time_ms': results['avg_inference_time_ms'],
        'train_info': results['train_info']
    }

    if 'model_summary' in results:
        metrics_data['model_summary'] = results['model_summary']

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)

    # Save feature importance
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in results['feature_importance'].items()
    ]).sort_values('importance', ascending=False)
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # Save model
    model_name = f"{baseline_name}_h{history_months}m_t{target_months}m.pkl"
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    model.save_model(model_dir / model_name)

    print(f"\nResults saved to {output_dir}")
    print(f"  - predictions.csv")
    print(f"  - metrics.json")
    print(f"  - feature_importance.csv")
    print(f"  - models/{model_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Run baseline comparison experiments'
    )
    parser.add_argument(
        '--reviews',
        type=str,
        required=True,
        help='Path to review CSV file'
    )
    parser.add_argument(
        '--snapshot-date',
        type=str,
        required=True,
        help='Snapshot date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--history-months',
        type=int,
        required=True,
        help='Number of months for learning period'
    )
    parser.add_argument(
        '--target-months',
        type=int,
        required=True,
        help='Number of months for prediction period'
    )
    parser.add_argument(
        '--baselines',
        type=str,
        nargs='+',
        default=['logistic_regression', 'random_forest'],
        help='Baselines to run (logistic_regression, random_forest)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='importants/baseline_experiments/',
        help='Output directory'
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"BASELINE COMPARISON EXPERIMENT")
    print(f"{'='*70}")
    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"Output: {args.output}")
    print(f"{'='*70}")

    # Load data
    train_data, test_data = load_and_prepare_data(
        args.reviews,
        args.snapshot_date,
        args.history_months,
        args.target_months
    )

    # Run each baseline
    all_results = {}

    for baseline_name in args.baselines:
        try:
            # Run baseline
            results, model = run_baseline(baseline_name, train_data, test_data)

            # Save results
            baseline_output_dir = Path(args.output) / baseline_name
            save_baseline_results(
                results,
                model,
                baseline_output_dir,
                baseline_name,
                args.history_months,
                args.target_months
            )

            all_results[baseline_name] = results

        except Exception as e:
            print(f"\nError running {baseline_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary comparison
    print(f"\n{'='*70}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Baseline':<25} {'AUC-PR':>8} {'AUC-ROC':>9} {'F1':>7} {'Time':>10}")
    print(f"{'-'*70}")

    for baseline_name, results in all_results.items():
        metrics = results['metrics']
        train_time = results['training_time']

        print(
            f"{baseline_name:<25} "
            f"{metrics['auc_pr']:>8.3f} "
            f"{metrics['auc_roc']:>9.3f} "
            f"{metrics['f1']:>7.3f} "
            f"{format_duration(train_time):>10}"
        )

    print(f"{'='*70}\n")

    # Save comparison summary
    if len(all_results) > 1:
        comparison_dir = Path(args.output) / 'comparison_results'
        comparison_dir.mkdir(parents=True, exist_ok=True)

        comparison_data = {
            'experiment_info': {
                'snapshot_date': args.snapshot_date,
                'history_months': args.history_months,
                'target_months': args.target_months,
                'n_train_samples': len(train_data['labels']),
                'n_test_samples': len(test_data['labels']),
                'baselines': list(all_results.keys())
            },
            'results': {
                name: {
                    'metrics': results['metrics'],
                    'training_time': results['training_time'],
                    'inference_time': results['avg_inference_time_ms']
                }
                for name, results in all_results.items()
            }
        }

        with open(comparison_dir / 'comparison_summary.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print(f"Comparison summary saved to {comparison_dir}/comparison_summary.json")


if __name__ == '__main__':
    main()
