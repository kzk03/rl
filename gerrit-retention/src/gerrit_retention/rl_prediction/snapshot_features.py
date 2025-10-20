"""
Snapshot-time feature extraction for IRL-based prediction.

This module computes developer features at a specific snapshot time,
aggregating all historical activity up to that point.

KEY DIFFERENCE from trajectory-based features:
- Trajectory: features at multiple time points [seq_len, feature_dim]
- Snapshot: features at ONE time point [feature_dim]
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from gerrit_retention.rl_prediction.enhanced_feature_extractor import EnhancedFeatureExtractor


def compute_snapshot_state_features(
    reviewer_email: str,
    snapshot_date: datetime,
    df: pd.DataFrame,
    learning_months: int,
    feature_extractor: Optional[EnhancedFeatureExtractor] = None
) -> np.ndarray:
    """
    Compute state features at snapshot time.

    This aggregates ALL activity in the learning period up to snapshot_date.

    Args:
        reviewer_email: Developer email
        snapshot_date: The snapshot date (e.g., 2022-01-01)
        df: Full review dataframe
        learning_months: Length of learning period in months
        feature_extractor: Enhanced feature extractor (optional)

    Returns:
        state_features: [32-dim] numpy array with snapshot-time state
    """
    if feature_extractor is None:
        feature_extractor = EnhancedFeatureExtractor()

    learning_start = snapshot_date - timedelta(days=learning_months * 30)

    # Get reviewer's activity in learning period
    reviewer_df = df[
        (df['reviewer_email'] == reviewer_email) &
        (df['timestamp'] >= learning_start) &
        (df['timestamp'] < snapshot_date)
    ].sort_values('timestamp')

    if len(reviewer_df) == 0:
        # No data -> return zero features
        return np.zeros(32, dtype=np.float32)

    # Build developer dict for EnhancedFeatureExtractor
    developer = {
        'developer_id': reviewer_email,
        'reviewer_email': reviewer_email,
        'first_seen': reviewer_df.iloc[0]['timestamp'].isoformat(),
        'changes_authored': len(reviewer_df),
        'changes_reviewed': len(reviewer_df),
        'projects': reviewer_df['project'].unique().tolist() if 'project' in reviewer_df.columns else [],
    }

    # Compute cumulative activity metrics
    activity_history = []
    for idx, row in reviewer_df.iterrows():
        activity = {
            'timestamp': row['timestamp'].isoformat(),
            'project': row.get('project', 'unknown'),
            'review_id': row.get('change_id', 'unknown'),
            'action_type': 'review',
        }
        activity_history.append(activity)

    # Extract state features using EnhancedFeatureExtractor
    state_dict = feature_extractor.extract_state_features(
        developer=developer,
        activity_history=activity_history,
        context_date=snapshot_date
    )

    # Convert to array
    state_array = feature_extractor.state_to_array(state_dict)

    return state_array


def compute_snapshot_action_features(
    reviewer_email: str,
    snapshot_date: datetime,
    df: pd.DataFrame,
    learning_months: int,
    feature_extractor: Optional[EnhancedFeatureExtractor] = None
) -> np.ndarray:
    """
    Compute action features at snapshot time.

    This represents the developer's most recent activity pattern.

    Args:
        reviewer_email: Developer email
        snapshot_date: The snapshot date
        df: Full review dataframe
        learning_months: Length of learning period
        feature_extractor: Enhanced feature extractor (optional)

    Returns:
        action_features: [9-dim] numpy array with snapshot-time action
    """
    if feature_extractor is None:
        feature_extractor = EnhancedFeatureExtractor()

    learning_start = snapshot_date - timedelta(days=learning_months * 30)

    # Get reviewer's activity in learning period
    reviewer_df = df[
        (df['reviewer_email'] == reviewer_email) &
        (df['timestamp'] >= learning_start) &
        (df['timestamp'] < snapshot_date)
    ].sort_values('timestamp')

    if len(reviewer_df) == 0:
        # No data -> return zero features
        return np.zeros(9, dtype=np.float32)

    # Use the most recent review as representative action
    last_review = reviewer_df.iloc[-1]

    # Extract action features
    action_dict = feature_extractor.extract_action_features(
        review_id=last_review.get('change_id', 'unknown'),
        action_type='review',
        timestamp=last_review['timestamp'],
        project=last_review.get('project', 'unknown'),
        intensity=1.0,  # Can be computed from review size
        quality_score=0.5,  # Can be computed from review outcome
        collaboration_level=1.0,  # Can be computed from co-reviewers
        response_time_hours=24.0,  # Can be computed from timestamps
        context_date=snapshot_date
    )

    # Convert to array
    action_array = feature_extractor.action_to_array(action_dict)

    return action_array


def compute_average_snapshot_features(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_months: int,
    feature_extractor: Optional[EnhancedFeatureExtractor] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute average snapshot features across all developers.

    This can be used as default features for developers with no data.

    Args:
        df: Full review dataframe
        snapshot_date: The snapshot date
        learning_months: Length of learning period
        feature_extractor: Enhanced feature extractor

    Returns:
        avg_state: Average state features [32-dim]
        avg_action: Average action features [9-dim]
    """
    if feature_extractor is None:
        feature_extractor = EnhancedFeatureExtractor()

    learning_start = snapshot_date - timedelta(days=learning_months * 30)

    # Get all reviewers in learning period
    learning_df = df[
        (df['timestamp'] >= learning_start) &
        (df['timestamp'] < snapshot_date)
    ]

    reviewers = learning_df['reviewer_email'].unique()

    # Collect features from all reviewers
    all_states = []
    all_actions = []

    for reviewer in reviewers[:100]:  # Sample first 100 for efficiency
        state = compute_snapshot_state_features(
            reviewer, snapshot_date, df, learning_months, feature_extractor
        )
        action = compute_snapshot_action_features(
            reviewer, snapshot_date, df, learning_months, feature_extractor
        )

        if not np.all(state == 0):  # Skip zero features
            all_states.append(state)
            all_actions.append(action)

    if len(all_states) == 0:
        return np.zeros(32), np.zeros(9)

    # Compute average
    avg_state = np.mean(all_states, axis=0)
    avg_action = np.mean(all_actions, axis=0)

    return avg_state, avg_action


def compute_snapshot_features_batch(
    reviewer_emails: list[str],
    snapshot_date: datetime,
    df: pd.DataFrame,
    learning_months: int,
    feature_extractor: Optional[EnhancedFeatureExtractor] = None,
    use_average_for_missing: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute snapshot features for multiple reviewers in batch.

    Args:
        reviewer_emails: List of reviewer emails
        snapshot_date: The snapshot date
        df: Full review dataframe
        learning_months: Length of learning period
        feature_extractor: Enhanced feature extractor
        use_average_for_missing: If True, use average features for reviewers with no data

    Returns:
        states: [N, 32] state features
        actions: [N, 9] action features
    """
    if feature_extractor is None:
        feature_extractor = EnhancedFeatureExtractor()

    states = []
    actions = []

    # Compute average features (for missing data)
    avg_state, avg_action = None, None
    if use_average_for_missing:
        avg_state, avg_action = compute_average_snapshot_features(
            df, snapshot_date, learning_months, feature_extractor
        )

    for reviewer in reviewer_emails:
        state = compute_snapshot_state_features(
            reviewer, snapshot_date, df, learning_months, feature_extractor
        )
        action = compute_snapshot_action_features(
            reviewer, snapshot_date, df, learning_months, feature_extractor
        )

        # Use average if no data and requested
        if use_average_for_missing and np.all(state == 0):
            state = avg_state
            action = avg_action

        states.append(state)
        actions.append(action)

    return np.array(states), np.array(actions)
