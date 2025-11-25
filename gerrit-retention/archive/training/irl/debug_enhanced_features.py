"""
Debug script to check enhanced features for NaN/Inf values
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from gerrit_retention.rl_prediction.enhanced_feature_extractor import EnhancedFeatureExtractor

# Load data
df = pd.read_csv('data/review_requests_openstack_multi_5y_detail.csv')
df['request_time'] = pd.to_datetime(df['request_time'])

# Extract a single trajectory
snapshot_date = datetime(2023, 1, 1)
learning_start = snapshot_date - timedelta(days=12 * 30)
learning_end = snapshot_date

df_learning = df[
    (df['request_time'] >= learning_start) &
    (df['request_time'] < learning_end)
]

# Get first reviewer
reviewer_email = df_learning['reviewer_email'].iloc[0]
group = df_learning[df_learning['reviewer_email'] == reviewer_email]

# Build developer info
latest_row = group.iloc[-1]
developer = {
    'developer_id': reviewer_email,
    'reviewer_email': reviewer_email,
    'changes_authored': 0,
    'changes_reviewed': len(group),
    'projects': group['project'].unique().tolist(),
    'first_seen': group['request_time'].min().isoformat(),
    'last_activity': group['request_time'].max().isoformat(),
    'reviewer_assignment_load_7d': latest_row.get('reviewer_assignment_load_7d', 0),
    'reviewer_assignment_load_30d': latest_row.get('reviewer_assignment_load_30d', 0),
    'reviewer_assignment_load_180d': latest_row.get('reviewer_assignment_load_180d', 0),
    'reviewer_past_reviews_30d': latest_row.get('reviewer_past_reviews_30d', 0),
    'reviewer_past_reviews_90d': latest_row.get('reviewer_past_reviews_90d', 0),
    'reviewer_past_reviews_180d': latest_row.get('reviewer_past_reviews_180d', 0),
    'reviewer_past_response_rate_180d': latest_row.get('reviewer_past_response_rate_180d', 1.0),
    'reviewer_tenure_days': latest_row.get('reviewer_tenure_days', 0),
    'owner_reviewer_past_interactions_180d': latest_row.get('owner_reviewer_past_interactions_180d', 0),
    'owner_reviewer_project_interactions_180d': latest_row.get('owner_reviewer_project_interactions_180d', 0),
    'owner_reviewer_past_assignments_180d': latest_row.get('owner_reviewer_past_assignments_180d', 0),
    'path_jaccard_files_project': latest_row.get('path_jaccard_files_project', 0.0),
    'path_jaccard_dir1_project': latest_row.get('path_jaccard_dir1_project', 0.0),
    'path_jaccard_dir2_project': latest_row.get('path_jaccard_dir2_project', 0.0),
    'path_overlap_files_project': latest_row.get('path_overlap_files_project', 0.0),
    'path_overlap_dir1_project': latest_row.get('path_overlap_dir1_project', 0.0),
    'path_overlap_dir2_project': latest_row.get('path_overlap_dir2_project', 0.0),
    'response_latency_days': latest_row.get('response_latency_days', 0.0),
    'change_insertions': latest_row.get('change_insertions', 0),
    'change_deletions': latest_row.get('change_deletions', 0),
    'change_files_count': latest_row.get('change_files_count', 1)
}

# Build activity history
activity_history = []
for _, row in group.iterrows():
    activity = {
        'timestamp': row['request_time'],
        'type': 'review',
        'change_insertions': row.get('change_insertions', 0),
        'change_deletions': row.get('change_deletions', 0),
        'change_files_count': row.get('change_files_count', 1),
        'response_latency_days': row.get('response_latency_days', 0.0),
        'message': row.get('subject', ''),
        'project': row.get('project', '')
    }
    activity_history.append(activity)

# Extract features
extractor = EnhancedFeatureExtractor()
state = extractor.extract_enhanced_state(developer, activity_history, learning_end)
state_array = extractor.state_to_array(state)

print("=" * 80)
print("Enhanced State Features Debug")
print("=" * 80)
print(f"Reviewer: {reviewer_email}")
print(f"Activity count: {len(activity_history)}")
print()

# Check for NaN/Inf in state
print("State Array (32 dims):")
for i, val in enumerate(state_array):
    status = ""
    if np.isnan(val):
        status = " <-- NaN!"
    elif np.isinf(val):
        status = " <-- Inf!"
    print(f"  [{i:2d}] = {val:12.6f}{status}")

print()
print(f"Has NaN: {np.any(np.isnan(state_array))}")
print(f"Has Inf: {np.any(np.isinf(state_array))}")

# Extract action
action = extractor.extract_enhanced_action(activity_history[0], learning_end)
action_array = extractor.action_to_array(action)

print()
print("Action Array (9 dims):")
for i, val in enumerate(action_array):
    status = ""
    if np.isnan(val):
        status = " <-- NaN!"
    elif np.isinf(val):
        status = " <-- Inf!"
    print(f"  [{i:2d}] = {val:12.6f}{status}")

print()
print(f"Has NaN: {np.any(np.isnan(action_array))}")
print(f"Has Inf: {np.any(np.isinf(action_array))}")

# Test scaler
print()
print("=" * 80)
print("Testing Scaler")
print("=" * 80)

states = [state_array]
actions = [action_array]

extractor.fit_scalers(states, actions)

state_norm = extractor.normalize_state(state_array)
action_norm = extractor.normalize_action(action_array)

print("Normalized State:")
for i, val in enumerate(state_norm):
    status = ""
    if np.isnan(val):
        status = " <-- NaN!"
    elif np.isinf(val):
        status = " <-- Inf!"
    print(f"  [{i:2d}] = {val:12.6f}{status}")

print()
print(f"Has NaN: {np.any(np.isnan(state_norm))}")
print(f"Has Inf: {np.any(np.isinf(state_norm))}")

print()
print("Normalized Action:")
for i, val in enumerate(action_norm):
    status = ""
    if np.isnan(val):
        status = " <-- NaN!"
    elif np.isinf(val):
        status = " <-- Inf!"
    print(f"  [{i:2d}] = {val:12.6f}{status}")

print()
print(f"Has NaN: {np.any(np.isnan(action_norm))}")
print(f"Has Inf: {np.any(np.isinf(action_norm))}")
