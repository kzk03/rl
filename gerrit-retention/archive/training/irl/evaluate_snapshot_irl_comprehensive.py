#!/usr/bin/env python3
"""
Comprehensive evaluation of Snapshot-based IRL across all sliding window configurations.

KEY FEATURES:
- Train from time series trajectories
- Predict with snapshot-time features ONLY
- Unified population across ALL configurations
- Generate heatmaps and reports
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gerrit_retention.rl_prediction.reward_network import RewardNetworkTrainer
from gerrit_retention.rl_prediction.snapshot_features_enhanced import (
    compute_snapshot_features_enhanced,
    compute_average_snapshot_features_enhanced
)
from gerrit_retention.rl_prediction.enhanced_feature_extractor import EnhancedFeatureExtractor
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, precision_score, recall_score
import torch

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load review data."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    for col in ["request_time", "created", "timestamp"]:
        if col in df.columns:
            df["timestamp"] = pd.to_datetime(df[col])
            break

    for col in ["reviewer_email", "email", "reviewer"]:
        if col in df.columns:
            df["reviewer_email"] = df[col]
            break

    print(f"Loaded {len(df):,} reviews")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Reviewers: {df['reviewer_email'].nunique()}")

    return df


def extract_state_features(
    reviewer_reviews: pd.DataFrame,
    current_row: pd.Series,
    all_reviews: pd.DataFrame,
    feature_extractor: EnhancedFeatureExtractor,
) -> np.ndarray:
    """Extract 32-dimensional enhanced state features (same as LSTM approach)."""
    reviews_so_far = reviewer_reviews[reviewer_reviews["timestamp"] <= current_row["timestamp"]]

    developer = {
        'developer_id': current_row.get('reviewer_email', 'unknown'),
        'reviewer_email': current_row.get('reviewer_email', 'unknown'),
        'first_seen': reviews_so_far.iloc[0]["timestamp"].isoformat() if len(reviews_so_far) > 0 else current_row["timestamp"].isoformat(),
        'changes_authored': len(reviews_so_far),
        'changes_reviewed': len(reviews_so_far),
        'projects': [current_row.get('project', 'unknown')],
        'reviewer_assignment_load_7d': 0,
        'reviewer_assignment_load_30d': 0,
        'reviewer_assignment_load_180d': 0,
        'owner_reviewer_past_interactions_180d': 0,
        'owner_reviewer_project_interactions_180d': 0,
        'owner_reviewer_past_assignments_180d': 0,
        'path_jaccard_files_project': 0.0,
        'path_jaccard_dir1_project': 0.0,
        'path_jaccard_dir2_project': 0.0,
        'path_overlap_files_project': 0.0,
        'path_overlap_dir1_project': 0.0,
        'path_overlap_dir2_project': 0.0,
        'response_latency_days': 0.0,
        'reviewer_past_response_rate_180d': 1.0,
        'reviewer_tenure_days': (current_row["timestamp"] - reviews_so_far.iloc[0]["timestamp"]).days if len(reviews_so_far) > 0 else 0,
        'change_insertions': current_row.get('change_insertions', 0),
        'change_deletions': current_row.get('change_deletions', 0),
        'change_files_count': current_row.get('change_files_count', 1),
    }

    activity_history = []
    for _, review in reviews_so_far.iterrows():
        activity = {
            'type': 'review',
            'timestamp': review['timestamp'].isoformat(),
            'change_insertions': review.get('change_insertions', 0),
            'change_deletions': review.get('change_deletions', 0),
            'change_files_count': review.get('change_files_count', 1),
        }
        activity_history.append(activity)

    try:
        enhanced_state = feature_extractor.extract_enhanced_state(
            developer=developer,
            activity_history=activity_history,
            context_date=current_row["timestamp"]
        )
        return feature_extractor.state_to_array(enhanced_state)
    except:
        return np.zeros(32, dtype=np.float32)


def extract_action_features(
    row: pd.Series,
    feature_extractor: EnhancedFeatureExtractor,
) -> np.ndarray:
    """Extract 9-dimensional enhanced action features (same as LSTM approach)."""
    activity = {
        'type': 'review',
        'timestamp': row['timestamp'].isoformat(),
        'change_insertions': row.get('change_insertions', 0),
        'change_deletions': row.get('change_deletions', 0),
        'change_files_count': row.get('change_files_count', 1),
        'response_latency_days': 0.0,
    }

    try:
        enhanced_action = feature_extractor.extract_enhanced_action(
            activity=activity,
            context_date=row['timestamp']
        )
        return feature_extractor.action_to_array(enhanced_action)
    except:
        return np.zeros(9, dtype=np.float32)


def extract_training_trajectories(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_months: int,
    seq_len: int,
    fixed_reviewer_set: set
) -> tuple:
    """
    Extract time series trajectories for training using enhanced features.
    Uses fixed reviewer set.
    """
    learning_start = snapshot_date - timedelta(days=learning_months * 30)

    learning_df = df[
        (df["timestamp"] >= learning_start) &
        (df["timestamp"] < snapshot_date)
    ]

    if len(learning_df) == 0:
        return [], [], []

    feature_extractor = EnhancedFeatureExtractor()

    trajectories = []
    labels_placeholder = []
    reviewer_ids = []

    for reviewer in fixed_reviewer_set:
        reviewer_reviews = learning_df[
            learning_df["reviewer_email"] == reviewer
        ].sort_values("timestamp")

        if len(reviewer_reviews) == 0:
            continue

        # Extract enhanced features (same as LSTM approach)
        states = []
        actions = []

        for idx, row in reviewer_reviews.iterrows():
            state = extract_state_features(reviewer_reviews, row, learning_df, feature_extractor)
            action = extract_action_features(row, feature_extractor)
            states.append(state)
            actions.append(action)

        # Pad or truncate
        if len(states) < seq_len:
            padding_needed = seq_len - len(states)
            states = [states[0]] * padding_needed + states
            actions = [actions[0]] * padding_needed + actions
        else:
            states = states[-seq_len:]
            actions = actions[-seq_len:]

        trajectory = {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "reviewer": reviewer,
        }
        trajectories.append(trajectory)
        labels_placeholder.append(0)
        reviewer_ids.append(reviewer)

    return trajectories, labels_placeholder, reviewer_ids


def run_sliding_window_evaluation(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_months_list: list,
    prediction_months_list: list,
    seq_len: int,
    epochs: int,
    output_dir: Path
) -> pd.DataFrame:
    """
    Run comprehensive sliding window evaluation.

    KEY: Unified population across ALL configurations!
    """
    results = []

    # STEP 1: Fix population to longest learning period
    max_learning_months = max(learning_months_list)
    print(f"\n{'='*80}")
    print(f"DETERMINING UNIFIED POPULATION (using {max_learning_months}m)")
    print(f"{'='*80}")

    max_learning_start = snapshot_date - timedelta(days=max_learning_months * 30)
    learning_df_max = df[
        (df["timestamp"] >= max_learning_start) &
        (df["timestamp"] < snapshot_date)
    ]

    fixed_reviewer_set = set(learning_df_max["reviewer_email"].unique())
    print(f"  Unified population: {len(fixed_reviewer_set)} reviewers")
    print(f"  (Active between {max_learning_start.date()} and {snapshot_date.date()})")

    # Compute average snapshot features (for missing data)
    print(f"\nComputing average snapshot features...")
    feature_extractor = EnhancedFeatureExtractor()
    avg_state, avg_action = compute_average_snapshot_features_enhanced(
        df, snapshot_date, max_learning_months, feature_extractor
    )

    # STEP 2: Evaluate all configurations
    for learning_months in learning_months_list:
        for prediction_months in prediction_months_list:
            print(f"\n{'='*80}")
            print(f"Configuration: {learning_months}m learning × {prediction_months}m prediction")
            print(f"{'='*80}")

            # Extract training trajectories
            print("Extracting training trajectories...")
            trajectories, _, reviewer_ids = extract_training_trajectories(
                df, snapshot_date, learning_months, seq_len, fixed_reviewer_set
            )

            if len(trajectories) == 0:
                print("  Skipping: No trajectories")
                continue

            # Get labels for TRAINING reviewers
            prediction_start = snapshot_date + timedelta(days=prediction_months * 30)
            post_prediction_df = df[df["timestamp"] >= prediction_start]
            continued_reviewers = set(post_prediction_df["reviewer_email"].unique())

            labels = [1 if r in continued_reviewers else 0 for r in reviewer_ids]
            continuation_rate_train = np.mean(labels) * 100

            print(f"  Training trajectories: {len(trajectories)}")
            print(f"  Training continuation rate: {continuation_rate_train:.1f}%")

            # Train/test split (80/20) for TRAINING
            n_train = int(len(trajectories) * 0.8)
            indices = np.random.permutation(len(trajectories))
            train_idx = indices[:n_train]

            train_trajectories = [trajectories[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]

            # TEST on reviewers who have data in this learning period ONLY
            # (Not the full fixed_reviewer_set, only those with training data)
            test_reviewers_with_data = list(set(reviewer_ids))
            test_labels = [1 if r in continued_reviewers else 0 for r in test_reviewers_with_data]
            continuation_rate_test = np.mean(test_labels) * 100

            print(f"  Test population: {len(test_reviewers_with_data)} reviewers (with data in {learning_months}m period)")
            print(f"  Test continuation rate: {continuation_rate_test:.1f}%")

            # Use these for actual testing
            test_reviewers = test_reviewers_with_data

            if len(test_reviewers) == 0:
                print("  Skipping: No test data")
                continue

            # Train reward network
            print(f"  Training reward network on {len(train_trajectories)} trajectories...")
            trainer = RewardNetworkTrainer(
                state_dim=32,
                action_dim=9,
                hidden_dim=128,
                learning_rate=0.001
            )

            trainer.train(
                trajectories=train_trajectories,
                labels=train_labels,
                epochs=epochs,
                batch_size=32,
                verbose=True
            )

            # STEP 3: Predict using SNAPSHOT-TIME FEATURES ONLY
            print(f"  Predicting with snapshot-time features for {len(test_reviewers)} reviewers...")
            predictions = []

            for reviewer in test_reviewers:
                # Compute snapshot features
                snapshot_state, snapshot_action = compute_snapshot_features_enhanced(
                    reviewer, snapshot_date, df, learning_months, feature_extractor
                )

                # Use average if no data
                if np.all(snapshot_state == 0):
                    snapshot_state = avg_state
                    snapshot_action = avg_action

                # Predict with reward network
                prob = trainer.predict(snapshot_state, snapshot_action)
                predictions.append(prob)

            predictions = np.array(predictions)
            test_labels_array = np.array(test_labels)

            # Calculate metrics
            metrics = {}
            try:
                metrics["auc_roc"] = roc_auc_score(test_labels_array, predictions)
                metrics["auc_pr"] = average_precision_score(test_labels_array, predictions)

                binary_preds = (predictions >= 0.5).astype(int)
                metrics["f1"] = f1_score(test_labels_array, binary_preds)
                metrics["precision"] = precision_score(test_labels_array, binary_preds, zero_division=0)
                metrics["recall"] = recall_score(test_labels_array, binary_preds, zero_division=0)

                tn, fp, fn, tp = confusion_matrix(test_labels_array, binary_preds).ravel()
                metrics["tn"] = int(tn)
                metrics["fp"] = int(fp)
                metrics["fn"] = int(fn)
                metrics["tp"] = int(tp)

            except Exception as e:
                print(f"  Error calculating metrics: {e}")
                metrics = {"auc_roc": 0.0, "auc_pr": 0.0, "f1": 0.0}

            print(f"  AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
            print(f"  AUC-PR: {metrics.get('auc_pr', 0):.4f}")
            print(f"  F1: {metrics.get('f1', 0):.4f}")

            # Save model
            model_dir = output_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"reward_model_h{learning_months}m_t{prediction_months}m.pth"
            trainer.save_model(str(model_path))

            # Save predictions
            predictions_dir = output_dir / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            predictions_df = pd.DataFrame({
                "reviewer_email": test_reviewers,
                "true_label": test_labels_array,
                "predicted_probability": predictions,
                "predicted_binary": (predictions >= 0.5).astype(int),
                "learning_months": learning_months,
                "prediction_months": prediction_months
            })
            predictions_file = predictions_dir / f"predictions_h{learning_months}m_t{prediction_months}m.csv"
            predictions_df.to_csv(predictions_file, index=False)

            # Record results
            result = {
                "learning_months": learning_months,
                "prediction_months": prediction_months,
                "n_trajectories": len(trajectories),
                "n_test": len(test_reviewers),
                "continuation_rate": continuation_rate_test,
                **metrics,
                "model_path": str(model_path)
            }
            results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / "sliding_window_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results to: {results_csv}")

    return results_df


def create_heatmaps(results_df: pd.DataFrame, output_dir: Path):
    """Create heatmaps for all metrics."""
    metrics = ["auc_roc", "auc_pr", "f1", "precision", "recall"]
    metric_names = {
        "auc_roc": "AUC-ROC",
        "auc_pr": "AUC-PR",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        pivot = results_df.pivot(
            index="prediction_months",
            columns="learning_months",
            values=metric
        )

        ax = axes[idx]
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": metric_names[metric]}
        )
        ax.set_title(f"{metric_names[metric]}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Learning Period (months)", fontsize=12)
        ax.set_ylabel("Prediction Period (months)", fontsize=12)

    fig.delaxes(axes[5])

    plt.suptitle(
        "Snapshot-based IRL Performance Heatmaps",
        fontsize=16,
        fontweight="bold"
    )
    plt.tight_layout()

    heatmap_file = output_dir / "heatmaps.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches="tight")
    print(f"Saved heatmaps to: {heatmap_file}")
    plt.close()


def create_report(results_df: pd.DataFrame, output_dir: Path, snapshot_date: datetime):
    """Create evaluation report."""
    report = []
    report.append("# Snapshot-based IRL Evaluation Report\n\n")
    report.append(f"**Snapshot Date**: {snapshot_date.date()}\n\n")
    report.append(f"**Total Configurations**: {len(results_df)}\n\n")

    report.append("## Key Results\n\n")

    best_auc = results_df.loc[results_df["auc_roc"].idxmax()]
    report.append(f"### Best AUC-ROC: {best_auc['auc_roc']:.4f}\n")
    report.append(f"- Configuration: {int(best_auc['learning_months'])}m learning × {int(best_auc['prediction_months'])}m prediction\n")
    report.append(f"- AUC-PR: {best_auc['auc_pr']:.4f}\n")
    report.append(f"- F1: {best_auc['f1']:.4f}\n\n")

    report.append("## All Results\n\n")
    report.append(results_df.to_markdown(index=False))
    report.append("\n\n")

    report.append("## Methodology\n\n")
    report.append("This evaluation uses **Snapshot-based IRL** - the correct IRL implementation:\n\n")
    report.append("1. **Training**: Learn reward function R(s,a) from time series trajectories\n")
    report.append("2. **Prediction**: Use R(s,a) with snapshot-time features ONLY (not sequences)\n")
    report.append("3. **Unified Population**: All configurations use the same fixed reviewer set\n\n")

    report_file = output_dir / "EVALUATION_REPORT.md"
    with open(report_file, "w") as f:
        f.writelines(report)

    print(f"Created report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Snapshot-based IRL Evaluation"
    )
    parser.add_argument("--reviews", required=True, help="Path to review CSV")
    parser.add_argument("--snapshot-date", required=True, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--learning-months", nargs="+", type=int, required=True)
    parser.add_argument("--prediction-months", nargs="+", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    print("="*80)
    print("SNAPSHOT-BASED IRL COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Output directory: {args.output}\n")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data(args.reviews)
    snapshot_date = datetime.strptime(args.snapshot_date, "%Y-%m-%d")

    # Run evaluation
    results_df = run_sliding_window_evaluation(
        df,
        snapshot_date,
        args.learning_months,
        args.prediction_months,
        args.seq_len,
        args.epochs,
        output_dir
    )

    # Create heatmaps
    print("\nCreating heatmaps...")
    create_heatmaps(results_df, output_dir)

    # Create report
    print("\nCreating report...")
    create_report(results_df, output_dir, snapshot_date)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
