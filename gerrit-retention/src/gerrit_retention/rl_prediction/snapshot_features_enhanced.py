"""
Enhanced snapshot-time feature extraction using EnhancedFeatureExtractor.

Computes the same 32-dim state and 9-dim action features used in LSTM approach,
but aggregated at snapshot time.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from gerrit_retention.rl_prediction.enhanced_feature_extractor import EnhancedFeatureExtractor


def compute_snapshot_features_enhanced(
    reviewer_email: str,
    snapshot_date: datetime,
    df: pd.DataFrame,
    learning_months: int,
    feature_extractor: Optional[EnhancedFeatureExtractor] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute enhanced snapshot features using EnhancedFeatureExtractor.

    This uses the SAME feature extraction as the LSTM approach, but at snapshot time.

    Returns:
        state: [32-dim] enhanced state features
        action: [9-dim] enhanced action features
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
        # No data -> return zeros
        return np.zeros(32, dtype=np.float32), np.zeros(9, dtype=np.float32)

    # === BUILD DEVELOPER DICT ===
    first_review = reviewer_df.iloc[0]
    last_review = reviewer_df.iloc[-1]

    developer = {
        'developer_id': reviewer_email,
        'reviewer_email': reviewer_email,
        'first_seen': first_review["timestamp"].isoformat(),
        'changes_authored': len(reviewer_df),
        'changes_reviewed': len(reviewer_df),
        'projects': reviewer_df['project'].unique().tolist() if 'project' in reviewer_df.columns else [],
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
        'reviewer_tenure_days': (snapshot_date - first_review["timestamp"]).days,
        'change_insertions': last_review.get('change_insertions', 0),
        'change_deletions': last_review.get('change_deletions', 0),
        'change_files_count': last_review.get('change_files_count', 1),
    }

    # === BUILD ACTIVITY HISTORY ===
    activity_history = []
    for _, review in reviewer_df.iterrows():
        activity = {
            'type': 'review',
            'timestamp': review['timestamp'].isoformat(),
            'change_insertions': review.get('change_insertions', 0),
            'change_deletions': review.get('change_deletions', 0),
            'change_files_count': review.get('change_files_count', 1),
        }
        activity_history.append(activity)

    # === EXTRACT STATE FEATURES (32-dim) ===
    try:
        enhanced_state = feature_extractor.extract_enhanced_state(
            developer=developer,
            activity_history=activity_history,
            context_date=snapshot_date
        )
        state_array = feature_extractor.state_to_array(enhanced_state)
    except Exception as e:
        # Fallback to zeros if extraction fails
        state_array = np.zeros(32, dtype=np.float32)

    # === EXTRACT ACTION FEATURES (9-dim) ===
    # Use most recent review as representative action
    try:
        last_activity = {
            'type': 'review',
            'timestamp': last_review['timestamp'].isoformat(),
            'change_insertions': last_review.get('change_insertions', 0),
            'change_deletions': last_review.get('change_deletions', 0),
            'change_files_count': last_review.get('change_files_count', 1),
            'response_latency_days': 0.0,
        }

        enhanced_action = feature_extractor.extract_enhanced_action(
            activity=last_activity,
            context_date=snapshot_date
        )
        action_array = feature_extractor.action_to_array(enhanced_action)
    except Exception as e:
        # Fallback to zeros if extraction fails
        action_array = np.zeros(9, dtype=np.float32)

    return state_array, action_array


def compute_average_snapshot_features_enhanced(
    df: pd.DataFrame,
    snapshot_date: datetime,
    learning_months: int,
    feature_extractor: Optional[EnhancedFeatureExtractor] = None,
    n_samples: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute average enhanced snapshot features across sample of developers.
    """
    if feature_extractor is None:
        feature_extractor = EnhancedFeatureExtractor()

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
        state, action = compute_snapshot_features_enhanced(
            reviewer, snapshot_date, df, learning_months, feature_extractor
        )

        if not np.all(state == 0):
            all_states.append(state)
            all_actions.append(action)

    if len(all_states) == 0:
        return np.zeros(32, dtype=np.float32), np.zeros(9, dtype=np.float32)

    avg_state = np.mean(all_states, axis=0).astype(np.float32)
    avg_action = np.mean(all_actions, axis=0).astype(np.float32)

    return avg_state, avg_action
