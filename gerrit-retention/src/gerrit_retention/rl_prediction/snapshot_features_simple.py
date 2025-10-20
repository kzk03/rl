"""
Simplified snapshot-time feature extraction.

Computes developer features at snapshot time using basic statistics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def compute_snapshot_features_simple(
    reviewer_email: str,
    snapshot_date: datetime,
    df: pd.DataFrame,
    learning_months: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute simplified snapshot features (state + action).

    Returns:
        state: [32-dim] state features
        action: [9-dim] action features
    """
    learning_start = snapshot_date - timedelta(days=learning_months * 30)

    # Get reviewer's activity in learning period
    reviewer_df = df[
        (df['reviewer_email'] == reviewer_email) &
        (df['timestamp'] >= learning_start) &
        (df['timestamp'] < snapshot_date)
    ].sort_values('timestamp')

    if len(reviewer_df) == 0:
        # No data -> return zeros
        return np.zeros(32, dtype=np.float32), np.zeros(9, dtype=np.float32)

    # === STATE FEATURES (32-dim) ===
    state_features = []

    # 1. Total reviews
    state_features.append(float(len(reviewer_df)))

    # 2. Tenure (days from first activity)
    first_activity = reviewer_df['timestamp'].min()
    tenure_days = (snapshot_date - first_activity).days
    state_features.append(float(tenure_days))

    # 3-5. Activity frequency (last 7d, 30d, 90d)
    for days_back in [7, 30, 90]:
        cutoff = snapshot_date - timedelta(days=days_back)
        recent = reviewer_df[reviewer_df['timestamp'] >= cutoff]
        freq = len(recent) / max(days_back, 1)
        state_features.append(freq)

    # 6. Days since last activity
    last_activity = reviewer_df['timestamp'].max()
    days_since = (snapshot_date - last_activity).days
    state_features.append(float(days_since))

    # 7. Average activity interval
    if len(reviewer_df) > 1:
        intervals = reviewer_df['timestamp'].diff().dt.days.dropna()
        avg_interval = intervals.mean() if len(intervals) > 0 else 0.0
    else:
        avg_interval = 0.0
    state_features.append(avg_interval)

    # 8. Number of projects
    n_projects = reviewer_df['project'].nunique() if 'project' in reviewer_df.columns else 1
    state_features.append(float(n_projects))

    # 9-32. Padding with zeros (for compatibility)
    while len(state_features) < 32:
        state_features.append(0.0)

    state = np.array(state_features[:32], dtype=np.float32)

    # === ACTION FEATURES (9-dim) ===
    action_features = []

    # Use most recent review as representative action
    last_review = reviewer_df.iloc[-1]

    # 1. Action type (always 1.0 for review)
    action_features.append(1.0)

    # 2-3. Intensity metrics
    intensity = len(reviewer_df) / max(tenure_days, 1)
    action_features.append(intensity)
    action_features.append(intensity * 10)  # scaled

    # 4-9. Padding
    while len(action_features) < 9:
        action_features.append(0.5)

    action = np.array(action_features[:9], dtype=np.float32)

    return state, action


def compute_average_snapshot_features_simple(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_months: int,
    n_samples: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute average snapshot features across random sample of developers.
    """
    learning_start = snapshot_date - timedelta(days=learning_months * 30)

    # Get reviewers in learning period
    learning_df = df[
        (df['timestamp'] >= learning_start) &
        (df['timestamp'] < snapshot_date)
    ]

    reviewers = learning_df['reviewer_email'].unique()

    # Sample reviewers
    sample_reviewers = np.random.choice(
        reviewers,
        size=min(n_samples, len(reviewers)),
        replace=False
    )

    all_states = []
    all_actions = []

    for reviewer in sample_reviewers:
        state, action = compute_snapshot_features_simple(
            reviewer, snapshot_date, df, learning_months
        )

        if not np.all(state == 0):
            all_states.append(state)
            all_actions.append(action)

    if len(all_states) == 0:
        return np.zeros(32, dtype=np.float32), np.zeros(9, dtype=np.float32)

    avg_state = np.mean(all_states, axis=0)
    avg_action = np.mean(all_actions, axis=0)

    return avg_state, avg_action
