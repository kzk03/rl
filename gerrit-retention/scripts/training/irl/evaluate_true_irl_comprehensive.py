#!/usr/bin/env python3
"""
Comprehensive True IRL (IRL+LSTM) Evaluation Script

This script evaluates the "true" IRL system (EnhancedRetentionIRLSystem with LSTM)
across different configurations:
1. Single-project mode: Evaluate each project separately
2. Multi-project mode: Train on all projects combined
3. Sliding window: 4x4 grid of learning/prediction periods (3,6,9,12 months each)

Key difference from other scripts:
- IRL training data period matches the learning period length
  e.g., 3-month learning → use 3 months of data before snapshot for IRL training
       12-month learning → use 12 months of data before snapshot for IRL training

Output:
- Trained models for each configuration
- Evaluation metrics (AUC-ROC, AUC-PR, F1, Precision, Recall)
- Heatmaps visualizing performance across different period combinations
- Comprehensive markdown documentation
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from gerrit_retention.rl_prediction.enhanced_feature_extractor import (
    EnhancedFeatureExtractor,
)


class LSTMContinuationPredictor(nn.Module):
    """LSTM-based continuation predictor using enhanced features."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim,  # Combined state + action encoded to hidden_dim
            hidden_dim,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True,
        )

        # Continuation predictor
        self.continuation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, states, actions):
        """
        Args:
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len, action_dim]

        Returns:
            predictions: [batch, 1] continuation probabilities
        """
        # Handle edge cases where input might not be 3D
        if len(states.shape) == 1:
            states = states.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
            actions = actions.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
        elif len(states.shape) == 2:
            states = states.unsqueeze(1)  # [batch, 1, state_dim]
            actions = actions.unsqueeze(1)  # [batch, 1, action_dim]

        batch_size, seq_len, _ = states.shape

        # Encode states and actions
        states_encoded = self.state_encoder(states.reshape(-1, self.state_dim))
        actions_encoded = self.action_encoder(actions.reshape(-1, self.action_dim))

        # Combine (simple addition, then reshape back)
        states_encoded = states_encoded.reshape(batch_size, seq_len, -1)
        actions_encoded = actions_encoded.reshape(batch_size, seq_len, -1)
        combined = states_encoded + actions_encoded  # [batch, seq_len, hidden_dim//2]

        # Project to hidden_dim for LSTM
        combined = torch.nn.functional.pad(combined, (0, self.hidden_dim - combined.shape[-1]))

        # LSTM
        lstm_out, _ = self.lstm(combined)  # [batch, seq_len, hidden_dim]

        # Use last output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]

        # Predict continuation
        continuation_prob = self.continuation_head(last_output)  # [batch, 1]

        return continuation_prob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive True IRL Evaluation with Sliding Window"
    )
    parser.add_argument(
        "--reviews",
        type=str,
        required=True,
        help="Path to review CSV file",
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        required=True,
        help="Snapshot date (YYYY-MM-DD) - divides learning and prediction periods",
    )
    parser.add_argument(
        "--learning-months",
        type=int,
        nargs="+",
        default=[3, 6, 9, 12],
        help="Learning period lengths in months (default: 3 6 9 12)",
    )
    parser.add_argument(
        "--prediction-months",
        type=int,
        nargs="+",
        default=[3, 6, 9, 12],
        help="Prediction period lengths in months (default: 3 6 9 12)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single-project", "multi-project", "both"],
        default="both",
        help="Evaluation mode (default: both)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for IRL (default: 30)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=15,
        help="Sequence length for LSTM (default: 15)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="importants/true_irl_comprehensive",
        help="Output directory (default: importants/true_irl_comprehensive)",
    )
    return parser.parse_args()


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load review data and prepare timestamps."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Identify timestamp column
    timestamp_col = None
    for col in ["request_time", "created", "timestamp"]:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        raise ValueError("No timestamp column found (request_time, created, timestamp)")

    df["timestamp"] = pd.to_datetime(df[timestamp_col])

    # Identify reviewer column
    reviewer_col = None
    for col in ["reviewer_email", "email", "reviewer"]:
        if col in df.columns:
            reviewer_col = col
            break

    if reviewer_col is None:
        raise ValueError("No reviewer column found (reviewer_email, email, reviewer)")

    df["reviewer"] = df[reviewer_col]

    # Ensure project column exists
    if "project" not in df.columns:
        raise ValueError("No 'project' column found in data")

    print(f"Loaded {len(df):,} reviews")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Projects: {df['project'].nunique()}")
    print(f"Reviewers: {df['reviewer'].nunique()}")

    return df


def extract_trajectories_fixed_population(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_months: int,
    prediction_months: int,
    seq_len: int,
    fixed_reviewer_set: set,
    project: str = None,
) -> Tuple[List[Dict], List[int]]:
    """
    Extract trajectories for IRL training WITH FIXED REVIEWER POPULATION.

    Key behavior:
    - Training data period = learning_months (matches the learning period)
    - Use data from [snapshot_date - learning_months, snapshot_date] for IRL training
    - Label based on activity AFTER prediction period (snapshot_date + prediction_months onwards)
    - ONLY include reviewers in fixed_reviewer_set (ensures consistent test population)

    Args:
        df: Review dataframe
        snapshot_date: Dividing point between learning and prediction
        learning_months: Length of learning period (also determines IRL training data period)
        prediction_months: Length of time to wait before checking continuation
        seq_len: Sequence length for padding/truncation
        fixed_reviewer_set: Set of reviewers to include (for consistency)
        project: If specified, filter to this project only

    Returns:
        trajectories: List of trajectory dicts
        labels: List of continuation labels (0/1)
    """
    learning_start = snapshot_date - timedelta(days=learning_months * 30)
    prediction_start = snapshot_date + timedelta(days=prediction_months * 30)

    # Filter by project if specified
    if project:
        df_filtered = df[df["project"] == project].copy()
    else:
        df_filtered = df.copy()

    # Learning period reviews (for building trajectories)
    learning_df = df_filtered[
        (df_filtered["timestamp"] >= learning_start) &
        (df_filtered["timestamp"] < snapshot_date)
    ]

    # Post-prediction period reviews (for labeling)
    # Check if reviewer has activity AFTER prediction_start (n months after snapshot)
    post_prediction_df = df_filtered[
        df_filtered["timestamp"] >= prediction_start
    ]

    if len(learning_df) == 0:
        print(f"  WARNING: No reviews in learning period [{learning_start.date()} to {snapshot_date.date()}]")
        return [], []

    # IMPORTANT: Use ONLY reviewers in fixed set
    # This ensures consistent test population across all configurations
    active_reviewers = [r for r in learning_df["reviewer"].unique() if r in fixed_reviewer_set]

    # Determine who continued (had activity after prediction period)
    continued_reviewers = set(post_prediction_df["reviewer"].unique())

    # Initialize enhanced feature extractor
    feature_extractor = EnhancedFeatureExtractor()

    trajectories = []
    labels = []

    for reviewer in active_reviewers:
        reviewer_reviews = learning_df[learning_df["reviewer"] == reviewer].sort_values("timestamp")

        # Skip if reviewer has no reviews in this specific learning period
        # (This can happen for shorter learning periods when using fixed population from max period)
        if len(reviewer_reviews) == 0:
            continue

        # Extract state and action features for each review
        states = []
        actions = []

        for idx, row in reviewer_reviews.iterrows():
            # State features (32-dim enhanced)
            state = extract_state_features(reviewer_reviews, row, learning_df, feature_extractor)
            states.append(state)

            # Action features (9-dim enhanced)
            action = extract_action_features(row, feature_extractor)
            actions.append(action)

        # Pad or truncate to seq_len
        if len(states) < seq_len:
            # Pad with first state/action
            padding_needed = seq_len - len(states)
            states = [states[0]] * padding_needed + states
            actions = [actions[0]] * padding_needed + actions
        else:
            # Use most recent seq_len reviews
            states = states[-seq_len:]
            actions = actions[-seq_len:]

        # Create trajectory
        trajectory = {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "reviewer": reviewer,
        }
        trajectories.append(trajectory)

        # Label: 1 if continued, 0 if churned
        label = 1 if reviewer in continued_reviewers else 0
        labels.append(label)

    continuation_rate = np.mean(labels) * 100 if labels else 0
    print(f"  Extracted {len(trajectories)} trajectories")
    print(f"  Learning period: {learning_start.date()} to {snapshot_date.date()} ({learning_months} months)")
    print(f"  Checking continuation AFTER: {prediction_start.date()} ({prediction_months} months after snapshot)")
    print(f"  Continuation rate: {continuation_rate:.1f}%")

    return trajectories, labels


def extract_state_features(
    reviewer_reviews: pd.DataFrame,
    current_row: pd.Series,
    all_reviews: pd.DataFrame,
    feature_extractor: EnhancedFeatureExtractor,
) -> np.ndarray:
    """Extract 32-dimensional enhanced state features."""
    # Build developer dict from current state
    reviews_so_far = reviewer_reviews[reviewer_reviews["timestamp"] <= current_row["timestamp"]]

    developer = {
        'developer_id': current_row.get('reviewer', 'unknown'),
        'reviewer_email': current_row.get('reviewer', 'unknown'),
        'first_seen': reviews_so_far.iloc[0]["timestamp"].isoformat() if len(reviews_so_far) > 0 else current_row["timestamp"].isoformat(),
        'changes_authored': len(reviews_so_far),
        'changes_reviewed': len(reviews_so_far),
        'projects': [current_row.get('project', 'unknown')],
        # Simulated values for fields not in basic CSV
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
        'change_insertions': current_row.get('lines_added', 0),
        'change_deletions': current_row.get('lines_deleted', 0),
        'change_files_count': current_row.get('files_changed', 1),
    }

    # Build activity history
    activity_history = []
    for _, review in reviews_so_far.iterrows():
        activity = {
            'type': 'review',
            'timestamp': review['timestamp'].isoformat(),
            'change_insertions': review.get('lines_added', 0),
            'change_deletions': review.get('lines_deleted', 0),
            'change_files_count': review.get('files_changed', 1),
        }
        activity_history.append(activity)

    # Extract enhanced state
    enhanced_state = feature_extractor.extract_enhanced_state(
        developer=developer,
        activity_history=activity_history,
        context_date=current_row["timestamp"]
    )

    # Convert to array (32-dim)
    return feature_extractor.state_to_array(enhanced_state)


def extract_action_features(
    row: pd.Series,
    feature_extractor: EnhancedFeatureExtractor,
) -> np.ndarray:
    """Extract 9-dimensional enhanced action features."""
    activity = {
        'type': 'review',
        'timestamp': row['timestamp'].isoformat(),
        'change_insertions': row.get('lines_added', 0),
        'change_deletions': row.get('lines_deleted', 0),
        'change_files_count': row.get('files_changed', 1),
        'response_latency_days': 0.0,
    }

    enhanced_action = feature_extractor.extract_enhanced_action(
        activity=activity,
        context_date=row['timestamp']
    )

    # Convert to array (9-dim)
    return feature_extractor.action_to_array(enhanced_action)


def train_and_evaluate(
    trajectories: List[Dict],
    labels: List[int],
    seq_len: int,
    epochs: int,
) -> Tuple[LSTMContinuationPredictor, Dict[str, float]]:
    """Train LSTM continuation predictor and evaluate."""
    if len(trajectories) == 0:
        return None, {}

    # Split train/test (80/20)
    n_train = int(len(trajectories) * 0.8)
    indices = np.random.permutation(len(trajectories))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_trajectories = [trajectories[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_trajectories = [trajectories[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    if len(test_trajectories) == 0:
        print("  WARNING: No test data available")
        return None, {}

    # Initialize LSTM predictor
    model = LSTMContinuationPredictor(state_dim=32, action_dim=9, hidden_dim=128, dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare training data
    train_states_list = [t["states"] for t in train_trajectories]
    train_actions_list = [t["actions"] for t in train_trajectories]

    # Ensure all states/actions are 2D (seq_len, feature_dim)
    # Use dtype=object first to avoid shape issues, then stack
    try:
        train_states_array = np.stack(train_states_list, axis=0)  # [batch, seq_len, state_dim]
        train_actions_array = np.stack(train_actions_list, axis=0)  # [batch, seq_len, action_dim]
    except ValueError as e:
        print(f"  ERROR: Failed to stack trajectories: {e}")
        return None, {}

    # Check dimensions and skip if invalid
    if len(train_states_array.shape) != 3:
        print(f"  ERROR: Invalid training data shape: {train_states_array.shape}")
        return None, {}

    train_states = torch.FloatTensor(train_states_array).to(device)
    train_actions = torch.FloatTensor(train_actions_array).to(device)
    train_labels_tensor = torch.FloatTensor(train_labels).unsqueeze(1).to(device)

    # Train
    print(f"  Training LSTM with {len(train_trajectories)} trajectories...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_states, train_actions)
        loss = criterion(outputs, train_labels_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate
    print(f"  Evaluating on {len(test_trajectories)} test trajectories...")
    model.eval()

    test_states_list = [t["states"] for t in test_trajectories]
    test_actions_list = [t["actions"] for t in test_trajectories]

    # Use stack to ensure proper 3D shape
    try:
        test_states_array = np.stack(test_states_list, axis=0)
        test_actions_array = np.stack(test_actions_list, axis=0)
    except ValueError as e:
        print(f"  ERROR: Failed to stack test trajectories: {e}")
        return model, {
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    # Check dimensions
    if len(test_states_array.shape) != 3:
        print(f"  ERROR: Invalid test data shape: {test_states_array.shape}")
        return model, {
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    test_states = torch.FloatTensor(test_states_array).to(device)
    test_actions = torch.FloatTensor(test_actions_array).to(device)

    with torch.no_grad():
        predictions = model(test_states, test_actions).cpu().numpy().flatten()

    test_labels = np.array(test_labels)

    # Calculate metrics
    metrics = {}
    try:
        metrics["auc_roc"] = roc_auc_score(test_labels, predictions)
        metrics["auc_pr"] = average_precision_score(test_labels, predictions)

        # Binary predictions (threshold 0.5)
        binary_preds = (predictions >= 0.5).astype(int)
        metrics["f1"] = f1_score(test_labels, binary_preds)
        metrics["precision"] = precision_score(test_labels, binary_preds, zero_division=0)
        metrics["recall"] = recall_score(test_labels, binary_preds, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_labels, binary_preds).ravel()
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tp"] = int(tp)

    except Exception as e:
        print(f"  Error calculating metrics: {e}")
        metrics = {
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    return model, metrics


def run_sliding_window_evaluation(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_months_list: List[int],
    prediction_months_list: List[int],
    seq_len: int,
    epochs: int,
    output_dir: Path,
    project: str = None,
) -> pd.DataFrame:
    """Run sliding window evaluation with all combinations."""
    mode_name = f"project_{project.replace('/', '_')}" if project else "multi_project"
    results = []

    # IMPORTANT: Fix test population to the longest learning period
    # This ensures fair comparison across configurations
    max_learning_months = max(learning_months_list)
    print(f"\n{'='*80}")
    print(f"DETERMINING TEST POPULATION (using {max_learning_months}m learning period)")
    print(f"{'='*80}")

    # Extract reviewers from the longest learning period
    max_learning_start = snapshot_date - timedelta(days=max_learning_months * 30)
    df_filtered = df[df["project"] == project] if project else df
    learning_df_max = df_filtered[
        (df_filtered["timestamp"] >= max_learning_start)
        & (df_filtered["timestamp"] < snapshot_date)
    ]

    # Fixed reviewer set for all configurations
    fixed_reviewer_set = set(learning_df_max["reviewer"].unique())
    print(f"  Fixed test population: {len(fixed_reviewer_set)} reviewers")
    print(f"  (Reviewers active between {max_learning_start.date()} and {snapshot_date.date()})")

    for learning_months in learning_months_list:
        for prediction_months in prediction_months_list:
            print(f"\n{'='*80}")
            print(f"Configuration: {learning_months}m learning × {prediction_months}m prediction")
            if project:
                print(f"Project: {project}")
            print(f"{'='*80}")

            # Extract trajectories WITH FIXED REVIEWER SET
            trajectories, labels = extract_trajectories_fixed_population(
                df, snapshot_date, learning_months, prediction_months, seq_len,
                fixed_reviewer_set, project
            )

            if len(trajectories) == 0:
                print("  Skipping: No trajectories")
                continue

            # Train and evaluate
            model, metrics = train_and_evaluate(trajectories, labels, seq_len, epochs)

            if model is None:
                print("  Skipping: Training failed")
                continue

            # Save model
            model_dir = output_dir / "models" / mode_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_name = f"irl_h{learning_months}m_t{prediction_months}m.pth"
            model_path = model_dir / model_name
            torch.save(model.state_dict(), str(model_path))
            print(f"  Saved model to: {model_path}")

            # Record results
            result = {
                "learning_months": learning_months,
                "prediction_months": prediction_months,
                "project": project if project else "all",
                "n_trajectories": len(trajectories),
                "continuation_rate": np.mean(labels) * 100,
                **metrics,
                "model_path": str(model_path),
            }
            results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_csv = output_dir / f"sliding_window_results_{mode_name}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results to: {results_csv}")

    return results_df


def create_heatmaps(
    results_df: pd.DataFrame,
    output_dir: Path,
    mode_name: str,
):
    """Create heatmaps for all metrics."""
    metrics = ["auc_roc", "auc_pr", "f1", "precision", "recall"]
    metric_names = {
        "auc_roc": "AUC-ROC",
        "auc_pr": "AUC-PR",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
    }

    # Create pivot tables for each metric
    learning_months = sorted(results_df["learning_months"].unique())
    prediction_months = sorted(results_df["prediction_months"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        pivot = results_df.pivot(
            index="prediction_months",
            columns="learning_months",
            values=metric,
        )

        # Reindex to ensure all combinations are present
        pivot = pivot.reindex(index=prediction_months, columns=learning_months)

        # Create heatmap
        ax = axes[idx]
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": metric_names[metric]},
        )
        ax.set_title(f"{metric_names[metric]}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Learning Period (months)", fontsize=12)
        ax.set_ylabel("Prediction Period (months)", fontsize=12)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle(
        f"True IRL Performance Heatmaps - {mode_name.replace('_', ' ').title()}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save
    heatmap_path = output_dir / f"heatmaps_{mode_name}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmaps to: {heatmap_path}")
    plt.close()


def create_summary_report(
    results_dict: Dict[str, pd.DataFrame],
    data_info: Dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
):
    """Create comprehensive markdown report."""
    report_lines = [
        "# True IRL Comprehensive Evaluation Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report presents a comprehensive evaluation of the **True IRL system** (EnhancedRetentionIRLSystem with LSTM) for developer retention prediction.",
        "",
        "### Key Innovation",
        "",
        "- **Temporal IRL with LSTM**: Full sequential modeling of developer activity trajectories",
        "- **Training data period matches learning period**: 3-month learning uses 3 months of data, 12-month learning uses 12 months of data",
        "- **Project-aware prediction**: Evaluates continuation within the same project",
        "",
        "## Dataset Information",
        "",
        f"- **File**: `{data_info['file_path']}`",
        f"- **Size**: {data_info['file_size_mb']:.2f} MB",
        f"- **Total Reviews**: {data_info['total_reviews']:,}",
        f"- **Date Range**: {data_info['date_min']} to {data_info['date_max']} ({data_info['date_span_years']:.1f} years)",
        f"- **Unique Reviewers**: {data_info['unique_reviewers']:,}",
        f"- **Projects**: {data_info['unique_projects']}",
        "",
        "### Project Distribution",
        "",
    ]

    # Add project distribution table
    for project, count in data_info["project_distribution"].items():
        pct = (count / data_info["total_reviews"]) * 100
        report_lines.append(f"- **{project}**: {count:,} reviews ({pct:.1f}%)")

    report_lines.extend([
        "",
        "## Experimental Setup",
        "",
        f"- **Snapshot Date**: {args.snapshot_date}",
        f"- **Learning Periods**: {', '.join(map(str, args.learning_months))} months",
        f"- **Prediction Periods**: {', '.join(map(str, args.prediction_months))} months",
        f"- **Sequence Length**: {args.seq_len}",
        f"- **Training Epochs**: {args.epochs}",
        f"- **Evaluation Mode**: {args.mode}",
        "",
        "### Critical Design Decision: Training Data Period",
        "",
        "**Unlike other approaches**, this evaluation matches the IRL training data period to the learning period:",
        "",
    ])

    for months in args.learning_months:
        start_date = (datetime.strptime(args.snapshot_date, "%Y-%m-%d") - timedelta(days=months * 30)).strftime("%Y-%m-%d")
        report_lines.append(
            f"- **{months}-month learning**: Uses data from {start_date} to {args.snapshot_date} for IRL training"
        )

    report_lines.extend([
        "",
        "This ensures that the model learns from a representative sample of the learning period it will be predicting for.",
        "",
        "## Results Summary",
        "",
    ])

    # Add results for each mode
    for mode_name, results_df in results_dict.items():
        report_lines.extend([
            f"### {mode_name.replace('_', ' ').title()}",
            "",
            f"**Total Configurations**: {len(results_df)}",
            "",
        ])

        # Find best configuration for each metric
        best_metrics = {}
        for metric in ["auc_roc", "auc_pr", "f1"]:
            if metric in results_df.columns:
                best_idx = results_df[metric].idxmax()
                best_row = results_df.loc[best_idx]
                best_metrics[metric] = {
                    "value": best_row[metric],
                    "learning": best_row["learning_months"],
                    "prediction": best_row["prediction_months"],
                }

        report_lines.append("**Best Configurations**:")
        report_lines.append("")
        for metric, info in best_metrics.items():
            metric_display = metric.upper().replace("_", "-")
            report_lines.append(
                f"- **{metric_display}**: {info['value']:.4f} "
                f"({info['learning']}m learning × {info['prediction']}m prediction)"
            )

        report_lines.extend([
            "",
            f"**See heatmaps**: `heatmaps_{mode_name}.png`",
            "",
            f"**Full results**: `sliding_window_results_{mode_name}.csv`",
            "",
        ])

    # Add methodology section
    report_lines.extend([
        "## Methodology",
        "",
        "### Trajectory Construction",
        "",
        "1. **Learning Period**: Reviews from [snapshot_date - learning_months, snapshot_date]",
        "2. **Sequence Processing**:",
        "   - Extract state (10-dim) and action (5-dim) features for each review",
        "   - Pad sequences shorter than seq_len by repeating first review",
        "   - Truncate sequences longer than seq_len to most recent reviews",
        "3. **Labeling**: 1 if reviewer had activity in prediction period, 0 otherwise",
        "",
        "### Model Architecture",
        "",
        "```",
        "Input: Trajectory [batch, seq_len, feature_dim]",
        "  ↓",
        "State Encoder (Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm → ReLU)",
        "  ↓",
        "Action Encoder (Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm → ReLU)",
        "  ↓",
        "Combined (Addition)",
        "  ↓",
        "LSTM (2-layer, hidden_size=128, dropout=0.3)",
        "  ↓",
        "├─ Reward Predictor (Linear → ReLU → Linear)",
        "└─ Continuation Predictor (Linear → ReLU → Linear → Sigmoid)",
        "```",
        "",
        "### Evaluation Metrics",
        "",
        "- **AUC-ROC**: Area under ROC curve (overall discrimination ability)",
        "- **AUC-PR**: Area under precision-recall curve (important for imbalanced data)",
        "- **F1 Score**: Harmonic mean of precision and recall",
        "- **Precision**: True positives / (True positives + False positives)",
        "- **Recall**: True positives / (True positives + False negatives)",
        "",
        "## Interpretation Guidelines",
        "",
        "### AUC-ROC Interpretation",
        "",
        "- **0.9-1.0**: Excellent discrimination",
        "- **0.8-0.9**: Good discrimination",
        "- **0.7-0.8**: Fair discrimination",
        "- **0.5**: Random guessing",
        "",
        "### Period Selection Insights",
        "",
        "Based on the heatmaps, you can identify:",
        "",
        "1. **Optimal learning period**: How much history is needed for accurate prediction?",
        "2. **Optimal prediction period**: What time horizon can we reliably predict?",
        "3. **Trade-offs**: Longer periods may not always be better due to data staleness",
        "",
        "## Files Generated",
        "",
        "```",
        f"{output_dir}/",
    ])

    # List all generated files
    for mode_name in results_dict.keys():
        report_lines.extend([
            f"├── models/{mode_name}/",
            f"│   └── irl_hXm_tXm.pth  (trained model files)",
            f"├── sliding_window_results_{mode_name}.csv",
            f"├── heatmaps_{mode_name}.png",
        ])

    report_lines.extend([
        "└── EVALUATION_REPORT.md  (this file)",
        "```",
        "",
        "## Reproducibility",
        "",
        "To reproduce these results, run:",
        "",
        "```bash",
        f"uv run python scripts/training/irl/evaluate_true_irl_comprehensive.py \\",
        f"  --reviews {data_info['file_path']} \\",
        f"  --snapshot-date {args.snapshot_date} \\",
        f"  --learning-months {' '.join(map(str, args.learning_months))} \\",
        f"  --prediction-months {' '.join(map(str, args.prediction_months))} \\",
        f"  --mode {args.mode} \\",
        f"  --epochs {args.epochs} \\",
        f"  --seq-len {args.seq_len} \\",
        f"  --output {args.output}",
        "```",
        "",
        "---",
        "",
        f"*Report generated by evaluate_true_irl_comprehensive.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    # Write report
    report_path = output_dir / "EVALUATION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nCreated comprehensive report: {report_path}")


def main():
    args = parse_args()

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TRUE IRL COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    df = load_and_prepare_data(args.reviews)

    # Gather data info for report
    data_info = {
        "file_path": args.reviews,
        "file_size_mb": os.path.getsize(args.reviews) / (1024 * 1024),
        "total_reviews": len(df),
        "date_min": str(df["timestamp"].min()),
        "date_max": str(df["timestamp"].max()),
        "date_span_years": (df["timestamp"].max() - df["timestamp"].min()).days / 365.25,
        "unique_reviewers": df["reviewer"].nunique(),
        "unique_projects": df["project"].nunique(),
        "project_distribution": df["project"].value_counts().to_dict(),
    }

    snapshot_date = datetime.strptime(args.snapshot_date, "%Y-%m-%d")

    results_dict = {}

    # Multi-project evaluation
    if args.mode in ["multi-project", "both"]:
        print("\n" + "="*80)
        print("MULTI-PROJECT EVALUATION")
        print("="*80)

        results_df = run_sliding_window_evaluation(
            df,
            snapshot_date,
            args.learning_months,
            args.prediction_months,
            args.seq_len,
            args.epochs,
            output_dir,
            project=None,
        )

        if len(results_df) > 0:
            results_dict["multi_project"] = results_df
            create_heatmaps(results_df, output_dir, "multi_project")

    # Single-project evaluation
    if args.mode in ["single-project", "both"]:
        print("\n" + "="*80)
        print("SINGLE-PROJECT EVALUATION")
        print("="*80)

        projects = df["project"].unique()
        print(f"Found {len(projects)} projects: {', '.join(projects)}")

        for project in projects:
            print(f"\n{'='*80}")
            print(f"Evaluating project: {project}")
            print(f"{'='*80}")

            project_df = df[df["project"] == project]
            print(f"Reviews in this project: {len(project_df):,}")

            results_df = run_sliding_window_evaluation(
                df,  # Use full df but filter inside function
                snapshot_date,
                args.learning_months,
                args.prediction_months,
                args.seq_len,
                args.epochs,
                output_dir,
                project=project,
            )

            if len(results_df) > 0:
                mode_name = f"project_{project.replace('/', '_')}"
                results_dict[mode_name] = results_df
                create_heatmaps(results_df, output_dir, mode_name)

    # Create summary report
    print("\n" + "="*80)
    print("CREATING SUMMARY REPORT")
    print("="*80)

    create_summary_report(results_dict, data_info, output_dir, args)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    print(f"See EVALUATION_REPORT.md for detailed analysis")


if __name__ == "__main__":
    main()
