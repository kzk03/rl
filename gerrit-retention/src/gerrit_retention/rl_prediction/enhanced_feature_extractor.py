"""
拡張特徴量抽出システム

高優先度の特徴量を追加:
- B1: レビュー負荷指標
- C1: 相互作用の深さ
- A1: 活動頻度の多期間比較
- D1: 専門性の一致度
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


@dataclass
class EnhancedDeveloperState:
    """拡張された開発者状態表現"""
    # 既存の基本特徴量
    developer_id: str
    experience_days: int
    total_changes: int
    total_reviews: int
    project_count: int
    recent_activity_frequency: float
    avg_activity_gap: float
    activity_trend: str
    collaboration_score: float
    code_quality_score: float
    timestamp: datetime

    # === A1: 活動頻度の多期間比較 ===
    activity_freq_7d: float = 0.0
    activity_freq_30d: float = 0.0
    activity_freq_90d: float = 0.0
    activity_acceleration: float = 0.0  # (freq_7d - freq_30d) / freq_30d
    consistency_score: float = 0.5  # 1.0 - std/mean

    # === B1: レビュー負荷指標 ===
    review_load_7d: float = 0.0  # 1日あたりレビュー数
    review_load_30d: float = 0.0
    review_load_180d: float = 0.0
    review_load_trend: float = 0.0  # (load_7d - load_30d) / load_30d
    is_overloaded: bool = False  # 1日5件以上
    is_high_load: bool = False  # 1日2件以上

    # === C1: 相互作用の深さ ===
    interaction_count_180d: float = 0.0
    interaction_intensity: float = 0.0  # interactions / months_active
    project_specific_interactions: float = 0.0
    assignment_history_180d: float = 0.0

    # === D1: 専門性の一致度 ===
    path_similarity_score: float = 0.0  # Jaccard平均
    path_overlap_score: float = 0.0  # Overlap平均

    # === その他有用な指標 ===
    avg_response_time_days: float = 0.0
    response_rate_180d: float = 1.0
    tenure_days: float = 0.0
    avg_change_size: float = 0.0  # insertions + deletions
    avg_files_changed: float = 0.0


@dataclass
class EnhancedDeveloperAction:
    """拡張された開発者行動表現"""
    # 既存の基本特徴量
    action_type: str
    intensity: float
    quality: float
    collaboration: float
    timestamp: datetime

    # === 追加特徴量 ===
    change_size: float = 0.0  # insertions + deletions
    files_count: float = 0.0
    complexity: float = 0.0  # change_size / files_count
    response_latency: float = 0.0  # days


class EnhancedFeatureExtractor:
    """拡張特徴量抽出器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 正規化用のScaler (MinMaxScaler: 0-1範囲に制限)
        self.state_scaler = MinMaxScaler()
        self.action_scaler = MinMaxScaler()
        self.scaler_fitted = False

        # 閾値設定
        self.overload_threshold = self.config.get('overload_threshold', 5.0)  # 1日5件
        self.high_load_threshold = self.config.get('high_load_threshold', 2.0)  # 1日2件

        logger.info("拡張特徴量抽出器を初期化しました")

    def extract_enhanced_state(self,
                               developer: Dict[str, Any],
                               activity_history: List[Dict[str, Any]],
                               context_date: datetime) -> EnhancedDeveloperState:
        """拡張された状態特徴量を抽出"""

        # === 基本特徴量（既存） ===
        first_seen = developer.get('first_seen', context_date.isoformat())
        if isinstance(first_seen, str):
            first_date = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
        else:
            first_date = first_seen
        experience_days = (context_date - first_date).days

        total_changes = developer.get('changes_authored', 0)
        total_reviews = developer.get('changes_reviewed', 0)
        projects = developer.get('projects', [])
        project_count = len(projects) if isinstance(projects, list) else 0

        # === A1: 活動頻度の多期間比較 ===
        activities_7d = self._get_recent_activities(activity_history, context_date, days=7)
        activities_30d = self._get_recent_activities(activity_history, context_date, days=30)
        activities_90d = self._get_recent_activities(activity_history, context_date, days=90)

        activity_freq_7d = len(activities_7d) / 7.0
        activity_freq_30d = len(activities_30d) / 30.0
        activity_freq_90d = len(activities_90d) / 90.0

        # 活動の加速度
        if activity_freq_30d > 0:
            activity_acceleration = (activity_freq_7d - activity_freq_30d) / activity_freq_30d
        else:
            activity_acceleration = 0.0

        # 一貫性スコア（週次活動のばらつき）
        consistency_score = self._calculate_consistency(activity_history, context_date)

        # === B1: レビュー負荷指標 ===
        review_load_7d = developer.get('reviewer_assignment_load_7d', 0) / 7.0
        review_load_30d = developer.get('reviewer_assignment_load_30d', 0) / 30.0
        review_load_180d = developer.get('reviewer_assignment_load_180d', 0) / 180.0

        # レビュー負荷のトレンド
        if review_load_30d > 0:
            review_load_trend = (review_load_7d - review_load_30d) / review_load_30d
        else:
            review_load_trend = 0.0

        is_overloaded = review_load_7d > self.overload_threshold
        is_high_load = review_load_7d > self.high_load_threshold

        # === C1: 相互作用の深さ ===
        interaction_count_180d = developer.get('owner_reviewer_past_interactions_180d', 0)
        project_specific_interactions = developer.get('owner_reviewer_project_interactions_180d', 0)
        assignment_history_180d = developer.get('owner_reviewer_past_assignments_180d', 0)

        # 相互作用の強度（月あたりの相互作用数）
        months_active = max(experience_days / 30.0, 1.0)
        interaction_intensity = interaction_count_180d / months_active

        # === D1: 専門性の一致度 ===
        path_jaccard_features = [
            developer.get('path_jaccard_files_project', 0.0),
            developer.get('path_jaccard_dir1_project', 0.0),
            developer.get('path_jaccard_dir2_project', 0.0)
        ]
        valid_jaccard = [f for f in path_jaccard_features if f > 0]
        path_similarity_score = np.mean(valid_jaccard) if len(valid_jaccard) > 0 else 0.0

        path_overlap_features = [
            developer.get('path_overlap_files_project', 0.0),
            developer.get('path_overlap_dir1_project', 0.0),
            developer.get('path_overlap_dir2_project', 0.0)
        ]
        valid_overlap = [f for f in path_overlap_features if f > 0]
        path_overlap_score = np.mean(valid_overlap) if len(valid_overlap) > 0 else 0.0

        # === その他有用な指標 ===
        avg_response_time_days = developer.get('response_latency_days', 0.0)
        if pd.isna(avg_response_time_days):
            avg_response_time_days = 0.0
        response_rate_180d = developer.get('reviewer_past_response_rate_180d', 1.0)
        if pd.isna(response_rate_180d):
            response_rate_180d = 1.0
        tenure_days = developer.get('reviewer_tenure_days', experience_days)
        if pd.isna(tenure_days):
            tenure_days = experience_days

        avg_change_size = developer.get('change_insertions', 0) + developer.get('change_deletions', 0)
        avg_files_changed = developer.get('change_files_count', 1)

        # 既存の計算
        recent_activity_frequency = activity_freq_30d
        activity_gaps = self._calculate_activity_gaps(activity_history)
        avg_activity_gap = np.mean(activity_gaps) if activity_gaps else 30.0
        activity_trend = self._analyze_activity_trend(activity_history, context_date)
        collaboration_score = self._calculate_collaboration_score(activity_history)
        code_quality_score = self._calculate_code_quality_score(activity_history)

        return EnhancedDeveloperState(
            developer_id=developer.get('developer_id', developer.get('reviewer_email', 'unknown')),
            experience_days=experience_days,
            total_changes=total_changes,
            total_reviews=total_reviews,
            project_count=project_count,
            recent_activity_frequency=recent_activity_frequency,
            avg_activity_gap=avg_activity_gap,
            activity_trend=activity_trend,
            collaboration_score=collaboration_score,
            code_quality_score=code_quality_score,
            timestamp=context_date,
            # A1
            activity_freq_7d=activity_freq_7d,
            activity_freq_30d=activity_freq_30d,
            activity_freq_90d=activity_freq_90d,
            activity_acceleration=activity_acceleration,
            consistency_score=consistency_score,
            # B1
            review_load_7d=review_load_7d,
            review_load_30d=review_load_30d,
            review_load_180d=review_load_180d,
            review_load_trend=review_load_trend,
            is_overloaded=is_overloaded,
            is_high_load=is_high_load,
            # C1
            interaction_count_180d=interaction_count_180d,
            interaction_intensity=interaction_intensity,
            project_specific_interactions=project_specific_interactions,
            assignment_history_180d=assignment_history_180d,
            # D1
            path_similarity_score=path_similarity_score,
            path_overlap_score=path_overlap_score,
            # その他
            avg_response_time_days=avg_response_time_days,
            response_rate_180d=response_rate_180d,
            tenure_days=tenure_days,
            avg_change_size=avg_change_size,
            avg_files_changed=avg_files_changed
        )

    def extract_enhanced_action(self,
                                activity: Dict[str, Any],
                                context_date: datetime) -> EnhancedDeveloperAction:
        """拡張された行動特徴量を抽出"""

        # 基本特徴量
        action_type = activity.get('type', 'unknown')
        intensity = self._calculate_action_intensity(activity)
        quality = self._calculate_action_quality(activity)
        collaboration = self._calculate_action_collaboration(activity)

        timestamp_str = activity.get('timestamp', context_date.isoformat())
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str

        # 拡張特徴量
        change_size = activity.get('change_insertions', 0) + activity.get('change_deletions', 0)
        files_count = max(activity.get('change_files_count', 1), 1)
        complexity = change_size / files_count
        response_latency = activity.get('response_latency_days', 0.0)
        if pd.isna(response_latency):
            response_latency = 0.0

        return EnhancedDeveloperAction(
            action_type=action_type,
            intensity=intensity,
            quality=quality,
            collaboration=collaboration,
            timestamp=timestamp,
            change_size=change_size,
            files_count=files_count,
            complexity=complexity,
            response_latency=response_latency
        )

    def state_to_array(self, state: EnhancedDeveloperState) -> np.ndarray:
        """拡張状態をNumPy配列に変換（正規化前）"""

        trend_encoding = {
            'increasing': 1.0,
            'stable': 0.5,
            'decreasing': 0.0,
            'unknown': 0.25
        }

        features = [
            # 基本特徴量 (10)
            state.experience_days,
            state.total_changes,
            state.total_reviews,
            state.project_count,
            state.recent_activity_frequency,
            state.avg_activity_gap,
            trend_encoding.get(state.activity_trend, 0.25),
            state.collaboration_score,
            state.code_quality_score,
            (datetime.now() - state.timestamp).days,

            # A1: 活動頻度 (5)
            state.activity_freq_7d,
            state.activity_freq_30d,
            state.activity_freq_90d,
            state.activity_acceleration,
            state.consistency_score,

            # B1: レビュー負荷 (6)
            state.review_load_7d,
            state.review_load_30d,
            state.review_load_180d,
            state.review_load_trend,
            float(state.is_overloaded),
            float(state.is_high_load),

            # C1: 相互作用 (4)
            state.interaction_count_180d,
            state.interaction_intensity,
            state.project_specific_interactions,
            state.assignment_history_180d,

            # D1: 専門性 (2)
            state.path_similarity_score,
            state.path_overlap_score,

            # その他 (5)
            state.avg_response_time_days,
            state.response_rate_180d,
            state.tenure_days,
            state.avg_change_size,
            state.avg_files_changed
        ]

        return np.array(features, dtype=np.float32)

    def action_to_array(self, action: EnhancedDeveloperAction) -> np.ndarray:
        """拡張行動をNumPy配列に変換（正規化前）"""

        type_encoding = {
            'commit': 1.0,
            'review': 0.8,
            'merge': 0.9,
            'documentation': 0.6,
            'issue': 0.4,
            'collaboration': 0.7,
            'unknown': 0.1
        }

        features = [
            # 基本特徴量 (5)
            type_encoding.get(action.action_type, 0.1),
            action.intensity,
            action.quality,
            action.collaboration,
            (datetime.now() - action.timestamp).days,

            # 拡張特徴量 (4)
            action.change_size,
            action.files_count,
            action.complexity,
            action.response_latency
        ]

        return np.array(features, dtype=np.float32)

    def fit_scalers(self, states: List[np.ndarray], actions: List[np.ndarray]):
        """Scalerを学習データでフィット"""
        if states:
            # NaN/Inf を置換してからフィット
            states_array = np.array(states)
            states_array = np.nan_to_num(states_array, nan=0.0, posinf=1e6, neginf=-1e6)
            self.state_scaler.fit(states_array)
        if actions:
            # NaN/Inf を置換してからフィット
            actions_array = np.array(actions)
            actions_array = np.nan_to_num(actions_array, nan=0.0, posinf=1e6, neginf=-1e6)
            self.action_scaler.fit(actions_array)
        self.scaler_fitted = True
        logger.info("Scalerのフィットが完了しました")

    def normalize_state(self, state_array: np.ndarray) -> np.ndarray:
        """状態特徴量を正規化"""
        if not self.scaler_fitted:
            logger.warning("Scalerが未フィット状態です。正規化をスキップします")
            return state_array

        # NaN/Inf チェックと置換
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=1e6, neginf=-1e6)
        normalized = self.state_scaler.transform(state_array.reshape(1, -1)).flatten()
        # 正規化後も念のためチェック
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        return normalized

    def normalize_action(self, action_array: np.ndarray) -> np.ndarray:
        """行動特徴量を正規化"""
        if not self.scaler_fitted:
            logger.warning("Scalerが未フィット状態です。正規化をスキップします")
            return action_array

        # NaN/Inf チェックと置換
        action_array = np.nan_to_num(action_array, nan=0.0, posinf=1e6, neginf=-1e6)
        normalized = self.action_scaler.transform(action_array.reshape(1, -1)).flatten()
        # 正規化後も念のためチェック
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        return normalized

    # === ヘルパーメソッド ===

    def _get_recent_activities(self, activity_history: List[Dict[str, Any]],
                               context_date: datetime, days: int) -> List[Dict[str, Any]]:
        """指定期間の最近の活動を取得"""
        cutoff_date = context_date - timedelta(days=days)
        recent_activities = []

        for activity in activity_history:
            try:
                timestamp_str = activity.get('timestamp', context_date.isoformat())
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str

                if timestamp >= cutoff_date:
                    recent_activities.append(activity)
            except:
                continue

        return recent_activities

    def _calculate_consistency(self, activity_history: List[Dict[str, Any]],
                               context_date: datetime) -> float:
        """活動の一貫性スコアを計算（週次のばらつき）"""
        # 過去12週の週次活動数を計算
        weekly_counts = []
        for week in range(12):
            start_date = context_date - timedelta(days=(week + 1) * 7)
            end_date = context_date - timedelta(days=week * 7)

            count = sum(1 for act in activity_history
                       if start_date <= self._get_timestamp(act) < end_date)
            weekly_counts.append(count)

        if not weekly_counts or np.mean(weekly_counts) == 0:
            return 0.5

        # 変動係数の逆数（1.0に近いほど一貫性が高い）
        cv = np.std(weekly_counts) / np.mean(weekly_counts)
        consistency = 1.0 / (1.0 + cv)
        return consistency

    def _get_timestamp(self, activity: Dict[str, Any]) -> datetime:
        """activityからタイムスタンプを取得"""
        timestamp_str = activity.get('timestamp', activity.get('request_time', '2020-01-01'))
        if isinstance(timestamp_str, str):
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return timestamp_str

    def _calculate_activity_gaps(self, activity_history: List[Dict[str, Any]]) -> List[float]:
        """活動間隔を計算"""
        timestamps = []
        for activity in activity_history:
            try:
                timestamps.append(self._get_timestamp(activity))
            except:
                continue

        if len(timestamps) < 2:
            return []

        timestamps.sort()
        gaps = [(timestamps[i] - timestamps[i-1]).days for i in range(1, len(timestamps))]
        return gaps

    def _analyze_activity_trend(self, activity_history: List[Dict[str, Any]],
                                context_date: datetime) -> str:
        """活動トレンドを分析"""
        recent_activities = self._get_recent_activities(activity_history, context_date, 30)
        past_activities = self._get_recent_activities(activity_history, context_date - timedelta(days=30), 30)

        recent_count = len(recent_activities)
        past_count = len(past_activities)

        if past_count == 0:
            return 'unknown'

        ratio = recent_count / past_count

        if ratio > 1.2:
            return 'increasing'
        elif ratio < 0.8:
            return 'decreasing'
        else:
            return 'stable'

    def _calculate_collaboration_score(self, activity_history: List[Dict[str, Any]]) -> float:
        """協力スコアを計算"""
        collaboration_activities = ['review', 'merge', 'collaboration', 'mentoring']
        total = len(activity_history)

        if total == 0:
            return 0.0

        collab_count = sum(1 for act in activity_history
                          if act.get('type', '').lower() in collaboration_activities)

        return collab_count / total

    def _calculate_code_quality_score(self, activity_history: List[Dict[str, Any]]) -> float:
        """コード品質スコアを計算"""
        quality_indicators = ['test', 'documentation', 'refactor', 'fix']
        total = len(activity_history)

        if total == 0:
            return 0.5

        quality_count = 0
        for activity in activity_history:
            message = activity.get('message', '').lower()
            if any(indicator in message for indicator in quality_indicators):
                quality_count += 1

        return min(quality_count / total + 0.3, 1.0)

    def _calculate_action_intensity(self, activity: Dict[str, Any]) -> float:
        """行動の強度を計算"""
        lines_added = activity.get('lines_added', activity.get('change_insertions', 0))
        lines_deleted = activity.get('lines_deleted', activity.get('change_deletions', 0))
        files_changed = max(activity.get('files_changed', activity.get('change_files_count', 1)), 1)

        intensity = min((lines_added + lines_deleted) / (files_changed * 50), 1.0)
        return max(intensity, 0.1)

    def _calculate_action_quality(self, activity: Dict[str, Any]) -> float:
        """行動の質を計算"""
        message = activity.get('message', activity.get('subject', '')).lower()
        quality_keywords = ['fix', 'improve', 'optimize', 'test', 'document', 'refactor']

        quality_score = 0.5
        for keyword in quality_keywords:
            if keyword in message:
                quality_score += 0.1

        return min(quality_score, 1.0)

    def _calculate_action_collaboration(self, activity: Dict[str, Any]) -> float:
        """行動の協力度を計算"""
        action_type = activity.get('type', '').lower()
        collaboration_types = {
            'review': 0.8,
            'merge': 0.7,
            'collaboration': 1.0,
            'mentoring': 0.9,
            'documentation': 0.6
        }

        return collaboration_types.get(action_type, 0.3)

    def get_feature_names(self) -> List[str]:
        """特徴量名のリストを返す"""
        return [
            # 基本特徴量 (10)
            'experience_days', 'total_changes', 'total_reviews', 'project_count',
            'recent_activity_frequency', 'avg_activity_gap', 'activity_trend',
            'collaboration_score', 'code_quality_score', 'timestamp_age',

            # A1: 活動頻度 (5)
            'activity_freq_7d', 'activity_freq_30d', 'activity_freq_90d',
            'activity_acceleration', 'consistency_score',

            # B1: レビュー負荷 (6)
            'review_load_7d', 'review_load_30d', 'review_load_180d',
            'review_load_trend', 'is_overloaded', 'is_high_load',

            # C1: 相互作用 (4)
            'interaction_count_180d', 'interaction_intensity',
            'project_specific_interactions', 'assignment_history_180d',

            # D1: 専門性 (2)
            'path_similarity_score', 'path_overlap_score',

            # その他 (5)
            'avg_response_time_days', 'response_rate_180d', 'tenure_days',
            'avg_change_size', 'avg_files_changed'
        ]

    def get_state_dim(self) -> int:
        """状態特徴量の次元数を返す"""
        return 32  # 10 + 5 + 6 + 4 + 2 + 5

    def get_action_dim(self) -> int:
        """行動特徴量の次元数を返す"""
        return 9  # 5 + 4
