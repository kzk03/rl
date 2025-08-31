"""
時系列特徴量エンジニアリング

時間的パターン、トレンド、周期性の特徴量化と時系列整合性の検証機能を提供する。
開発者定着予測システムにおける時系列データの適切な処理を保証する。
"""

import logging
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft

logger = logging.getLogger(__name__)


@dataclass
class TemporalFeatures:
    """時系列特徴量データクラス"""
    developer_email: str
    context_date: datetime
    
    # 活動パターン特徴量
    activity_trend: float
    activity_seasonality: Dict[str, float]
    activity_volatility: float
    activity_momentum: float
    
    # 周期性特徴量
    weekly_pattern: List[float]  # 曜日別活動パターン
    daily_pattern: List[float]   # 時間別活動パターン
    monthly_pattern: List[float] # 月別活動パターン
    
    # トレンド特徴量
    short_term_trend: float      # 短期トレンド（1週間）
    medium_term_trend: float     # 中期トレンド（1ヶ月）
    long_term_trend: float       # 長期トレンド（3ヶ月）
    
    # 変化点検出
    change_points: List[datetime]
    trend_stability: float
    pattern_consistency: float
    
    # 予測特徴量
    activity_forecast: float
    engagement_forecast: float
    retention_risk_trend: float
    
    # 時系列整合性検証
    temporal_consistency_score: float
    data_leakage_risk: float


class TemporalFeatureExtractor:
    """時系列特徴量抽出器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.short_term_window = config.get('short_term_window_days', 7)
        self.medium_term_window = config.get('medium_term_window_days', 30)
        self.long_term_window = config.get('long_term_window_days', 90)
        self.min_data_points = config.get('min_data_points', 5)
        self.enable_strict_validation = config.get('enable_strict_validation', True)
        
    def extract_features(self, 
                        developer_email: str,
                        changes_data: List[Dict[str, Any]],
                        reviews_data: List[Dict[str, Any]],
                        context_date: datetime) -> TemporalFeatures:
        """
        時系列特徴量を抽出
        
        Args:
            developer_email: 開発者のメールアドレス
            changes_data: Change データのリスト
            reviews_data: Review データのリスト
            context_date: 特徴量計算の基準日時
            
        Returns:
            TemporalFeatures: 抽出された時系列特徴量
        """
        logger.info(f"時系列特徴量を抽出中: {developer_email}")
        
        # 時系列整合性を検証
        if self.enable_strict_validation:
            self._validate_temporal_consistency(
                changes_data, reviews_data, context_date
            )
        
        # 開発者の活動データを抽出
        developer_activities = self._extract_developer_activities(
            developer_email, changes_data, reviews_data, context_date
        )
        
        # 各カテゴリの特徴量を計算
        activity_features = self._extract_activity_pattern_features(
            developer_activities, context_date
        )
        
        periodicity_features = self._extract_periodicity_features(
            developer_activities, context_date
        )
        
        trend_features = self._extract_trend_features(
            developer_activities, context_date
        )
        
        change_point_features = self._extract_change_point_features(
            developer_activities, context_date
        )
        
        forecast_features = self._extract_forecast_features(
            developer_activities, context_date
        )
        
        consistency_features = self._extract_consistency_features(
            changes_data, reviews_data, context_date
        )
        
        return TemporalFeatures(
            developer_email=developer_email,
            context_date=context_date,
            **activity_features,
            **periodicity_features,
            **trend_features,
            **change_point_features,
            **forecast_features,
            **consistency_features
        )
    
    def _validate_temporal_consistency(self, 
                                     changes_data: List[Dict[str, Any]],
                                     reviews_data: List[Dict[str, Any]],
                                     context_date: datetime) -> None:
        """時系列整合性を検証"""
        
        # 未来データの使用をチェック
        future_changes = [
            c for c in changes_data 
            if datetime.fromisoformat(c.get('created', '')) > context_date
        ]
        
        future_reviews = [
            r for r in reviews_data 
            if datetime.fromisoformat(r.get('timestamp', '')) > context_date
        ]
        
        if future_changes or future_reviews:
            error_msg = (
                f"時系列整合性エラー: 基準日時 {context_date} より未来のデータが検出されました。"
                f"未来のChange: {len(future_changes)}, 未来のReview: {len(future_reviews)}"
            )
            if self.enable_strict_validation:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)
    
    def _extract_developer_activities(self, 
                                    developer_email: str,
                                    changes_data: List[Dict[str, Any]],
                                    reviews_data: List[Dict[str, Any]],
                                    context_date: datetime) -> List[Dict[str, Any]]:
        """開発者の活動データを時系列で抽出"""
        
        activities = []
        
        # Changeの作成活動
        for change in changes_data:
            if (change.get('author') == developer_email and 
                datetime.fromisoformat(change.get('created', '')) <= context_date):
                
                activities.append({
                    'timestamp': datetime.fromisoformat(change.get('created', '')),
                    'type': 'change_created',
                    'data': change
                })
        
        # レビュー活動
        for review in reviews_data:
            if (review.get('reviewer_email') == developer_email and 
                datetime.fromisoformat(review.get('timestamp', '')) <= context_date):
                
                activities.append({
                    'timestamp': datetime.fromisoformat(review.get('timestamp', '')),
                    'type': 'review_given',
                    'data': review
                })
        
        # 時系列順にソート
        activities.sort(key=lambda x: x['timestamp'])
        
        return activities
    
    def _extract_activity_pattern_features(self, 
                                         activities: List[Dict[str, Any]],
                                         context_date: datetime) -> Dict[str, Any]:
        """活動パターン特徴量を抽出"""
        
        if len(activities) < self.min_data_points:
            return {
                'activity_trend': 0.0,
                'activity_seasonality': {},
                'activity_volatility': 0.0,
                'activity_momentum': 0.0
            }
        
        # 日別活動数の時系列を作成
        daily_activities = self._create_daily_activity_series(
            activities, context_date
        )
        
        # 活動トレンドを計算
        activity_trend = self._calculate_activity_trend(daily_activities)
        
        # 季節性を計算
        activity_seasonality = self._calculate_activity_seasonality(activities)
        
        # 活動の変動性を計算
        activity_volatility = self._calculate_activity_volatility(daily_activities)
        
        # 活動の勢いを計算
        activity_momentum = self._calculate_activity_momentum(daily_activities)
        
        return {
            'activity_trend': activity_trend,
            'activity_seasonality': activity_seasonality,
            'activity_volatility': activity_volatility,
            'activity_momentum': activity_momentum
        }
    
    def _create_daily_activity_series(self, 
                                    activities: List[Dict[str, Any]],
                                    context_date: datetime) -> pd.Series:
        """日別活動数の時系列を作成"""
        
        if not activities:
            return pd.Series(dtype=float)
        
        # 活動を日別に集計
        daily_counts = defaultdict(int)
        
        for activity in activities:
            date_key = activity['timestamp'].date()
            daily_counts[date_key] += 1
        
        # 連続した日付範囲を作成
        start_date = min(daily_counts.keys())
        end_date = context_date.date()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 時系列データを作成（活動がない日は0）
        series_data = []
        for date in date_range:
            count = daily_counts.get(date.date(), 0)
            series_data.append(count)
        
        return pd.Series(series_data, index=date_range)
    
    def _calculate_activity_trend(self, 
                                daily_activities: pd.Series) -> float:
        """活動トレンドを計算"""
        
        if len(daily_activities) < 2:
            return 0.0
        
        # 線形回帰による傾きを計算
        x = np.arange(len(daily_activities))
        y = daily_activities.values
        
        if np.std(y) == 0:  # 分散がゼロの場合
            return 0.0
        
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        # R²値で重み付けした傾き
        weighted_slope = slope * (r_value ** 2)
        
        # 正規化（-1から1の範囲）
        max_possible_slope = np.max(y) / len(y)
        if max_possible_slope > 0:
            normalized_slope = weighted_slope / max_possible_slope
            return max(-1.0, min(1.0, normalized_slope))
        
        return 0.0
    
    def _calculate_activity_seasonality(self, 
                                      activities: List[Dict[str, Any]]) -> Dict[str, float]:
        """活動の季節性を計算"""
        
        seasonality = {}
        
        # 曜日別の季節性
        weekday_counts = defaultdict(int)
        for activity in activities:
            weekday = activity['timestamp'].weekday()
            weekday_counts[weekday] += 1
        
        total_activities = len(activities)
        if total_activities > 0:
            weekday_seasonality = {}
            for weekday in range(7):
                ratio = weekday_counts[weekday] / total_activities
                weekday_seasonality[f'weekday_{weekday}'] = ratio
            seasonality.update(weekday_seasonality)
        
        # 時間帯別の季節性
        hour_counts = defaultdict(int)
        for activity in activities:
            hour = activity['timestamp'].hour
            hour_counts[hour] += 1
        
        if total_activities > 0:
            hour_seasonality = {}
            for hour in range(24):
                ratio = hour_counts[hour] / total_activities
                hour_seasonality[f'hour_{hour}'] = ratio
            seasonality.update(hour_seasonality)
        
        return seasonality
    
    def _calculate_activity_volatility(self, 
                                     daily_activities: pd.Series) -> float:
        """活動の変動性を計算"""
        
        if len(daily_activities) < 2:
            return 0.0
        
        # 変動係数（標準偏差/平均）
        mean_activity = daily_activities.mean()
        if mean_activity == 0:
            return 0.0
        
        std_activity = daily_activities.std()
        volatility = std_activity / mean_activity
        
        # 正規化（0から1の範囲）
        return min(volatility / 2.0, 1.0)
    
    def _calculate_activity_momentum(self, 
                                   daily_activities: pd.Series) -> float:
        """活動の勢いを計算"""
        
        if len(daily_activities) < 4:
            return 0.0
        
        # 最近の活動と過去の活動を比較
        recent_period = len(daily_activities) // 4  # 最新25%
        recent_activities = daily_activities[-recent_period:].mean()
        past_activities = daily_activities[:-recent_period].mean()
        
        if past_activities == 0:
            return 1.0 if recent_activities > 0 else 0.0
        
        momentum = (recent_activities - past_activities) / past_activities
        
        # 正規化（-1から1の範囲）
        return max(-1.0, min(1.0, momentum))
    
    def _extract_periodicity_features(self, 
                                    activities: List[Dict[str, Any]],
                                    context_date: datetime) -> Dict[str, Any]:
        """周期性特徴量を抽出"""
        
        if len(activities) < self.min_data_points:
            return {
                'weekly_pattern': [0.0] * 7,
                'daily_pattern': [0.0] * 24,
                'monthly_pattern': [0.0] * 12
            }
        
        # 曜日別パターン
        weekly_pattern = self._calculate_weekly_pattern(activities)
        
        # 時間別パターン
        daily_pattern = self._calculate_daily_pattern(activities)
        
        # 月別パターン
        monthly_pattern = self._calculate_monthly_pattern(activities)
        
        return {
            'weekly_pattern': weekly_pattern,
            'daily_pattern': daily_pattern,
            'monthly_pattern': monthly_pattern
        }
    
    def _calculate_weekly_pattern(self, 
                                activities: List[Dict[str, Any]]) -> List[float]:
        """曜日別活動パターンを計算"""
        
        weekday_counts = [0] * 7
        
        for activity in activities:
            weekday = activity['timestamp'].weekday()
            weekday_counts[weekday] += 1
        
        total = sum(weekday_counts)
        if total == 0:
            return [0.0] * 7
        
        return [count / total for count in weekday_counts]
    
    def _calculate_daily_pattern(self, 
                               activities: List[Dict[str, Any]]) -> List[float]:
        """時間別活動パターンを計算"""
        
        hour_counts = [0] * 24
        
        for activity in activities:
            hour = activity['timestamp'].hour
            hour_counts[hour] += 1
        
        total = sum(hour_counts)
        if total == 0:
            return [0.0] * 24
        
        return [count / total for count in hour_counts]
    
    def _calculate_monthly_pattern(self, 
                                 activities: List[Dict[str, Any]]) -> List[float]:
        """月別活動パターンを計算"""
        
        month_counts = [0] * 12
        
        for activity in activities:
            month = activity['timestamp'].month - 1  # 0-based index
            month_counts[month] += 1
        
        total = sum(month_counts)
        if total == 0:
            return [0.0] * 12
        
        return [count / total for count in month_counts]
    
    def _extract_trend_features(self, 
                              activities: List[Dict[str, Any]],
                              context_date: datetime) -> Dict[str, Any]:
        """トレンド特徴量を抽出"""
        
        # 短期トレンド（1週間）
        short_term_trend = self._calculate_period_trend(
            activities, context_date, self.short_term_window
        )
        
        # 中期トレンド（1ヶ月）
        medium_term_trend = self._calculate_period_trend(
            activities, context_date, self.medium_term_window
        )
        
        # 長期トレンド（3ヶ月）
        long_term_trend = self._calculate_period_trend(
            activities, context_date, self.long_term_window
        )
        
        return {
            'short_term_trend': short_term_trend,
            'medium_term_trend': medium_term_trend,
            'long_term_trend': long_term_trend
        }
    
    def _calculate_period_trend(self, 
                              activities: List[Dict[str, Any]],
                              context_date: datetime,
                              window_days: int) -> float:
        """指定期間のトレンドを計算"""
        
        start_date = context_date - timedelta(days=window_days)
        
        # 期間内の活動をフィルタリング
        period_activities = [
            a for a in activities 
            if start_date <= a['timestamp'] <= context_date
        ]
        
        if len(period_activities) < 2:
            return 0.0
        
        # 日別活動数を計算
        daily_counts = defaultdict(int)
        for activity in period_activities:
            date_key = activity['timestamp'].date()
            daily_counts[date_key] += 1
        
        # 時系列データを作成
        dates = []
        counts = []
        
        current_date = start_date.date()
        end_date = context_date.date()
        
        while current_date <= end_date:
            dates.append(current_date)
            counts.append(daily_counts.get(current_date, 0))
            current_date += timedelta(days=1)
        
        if len(counts) < 2:
            return 0.0
        
        # 線形回帰による傾きを計算
        x = np.arange(len(counts))
        y = np.array(counts)
        
        if np.std(y) == 0:
            return 0.0
        
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        # R²値で重み付けした傾き
        weighted_slope = slope * (r_value ** 2)
        
        # 正規化
        max_possible_slope = np.max(y) / len(y) if len(y) > 0 else 1.0
        if max_possible_slope > 0:
            normalized_slope = weighted_slope / max_possible_slope
            return max(-1.0, min(1.0, normalized_slope))
        
        return 0.0
    
    def _extract_change_point_features(self, 
                                     activities: List[Dict[str, Any]],
                                     context_date: datetime) -> Dict[str, Any]:
        """変化点検出特徴量を抽出"""
        
        # 変化点を検出
        change_points = self._detect_change_points(activities, context_date)
        
        # トレンドの安定性を計算
        trend_stability = self._calculate_trend_stability(activities, context_date)
        
        # パターンの一貫性を計算
        pattern_consistency = self._calculate_pattern_consistency(activities)
        
        return {
            'change_points': change_points,
            'trend_stability': trend_stability,
            'pattern_consistency': pattern_consistency
        }
    
    def _detect_change_points(self, 
                            activities: List[Dict[str, Any]],
                            context_date: datetime) -> List[datetime]:
        """活動パターンの変化点を検出"""
        
        if len(activities) < 10:  # 最小データ点数
            return []
        
        # 週別活動数を計算
        weekly_activities = defaultdict(int)
        
        for activity in activities:
            # 週の開始日（月曜日）を計算
            week_start = activity['timestamp'] - timedelta(
                days=activity['timestamp'].weekday()
            )
            week_key = week_start.date()
            weekly_activities[week_key] += 1
        
        # 時系列データを作成
        weeks = sorted(weekly_activities.keys())
        counts = [weekly_activities[week] for week in weeks]
        
        if len(counts) < 4:
            return []
        
        # 簡易的な変化点検出（移動平均の大きな変化）
        change_points = []
        window_size = 3
        
        for i in range(window_size, len(counts) - window_size):
            before_avg = np.mean(counts[i-window_size:i])
            after_avg = np.mean(counts[i:i+window_size])
            
            # 大きな変化を検出
            if before_avg > 0:
                change_ratio = abs(after_avg - before_avg) / before_avg
                if change_ratio > 0.5:  # 50%以上の変化
                    change_date = datetime.combine(weeks[i], datetime.min.time())
                    change_points.append(change_date)
        
        return change_points
    
    def _calculate_trend_stability(self, 
                                 activities: List[Dict[str, Any]],
                                 context_date: datetime) -> float:
        """トレンドの安定性を計算"""
        
        if len(activities) < 4:
            return 0.0
        
        # 複数の期間でトレンドを計算
        periods = [7, 14, 30]  # 1週間、2週間、1ヶ月
        trends = []
        
        for period in periods:
            trend = self._calculate_period_trend(activities, context_date, period)
            trends.append(trend)
        
        # トレンドの一貫性（分散の逆数）
        if len(trends) > 1:
            trend_variance = np.var(trends)
            stability = 1.0 / (1.0 + trend_variance)
        else:
            stability = 0.0
        
        return stability
    
    def _calculate_pattern_consistency(self, 
                                     activities: List[Dict[str, Any]]) -> float:
        """パターンの一貫性を計算"""
        
        if len(activities) < 7:  # 最低1週間分のデータ
            return 0.0
        
        # 曜日別活動パターンの一貫性を評価
        weekly_patterns = []
        
        # 週ごとにパターンを計算
        current_week_activities = []
        current_week_start = None
        
        for activity in activities:
            week_start = activity['timestamp'] - timedelta(
                days=activity['timestamp'].weekday()
            )
            
            if current_week_start is None:
                current_week_start = week_start
            
            if week_start == current_week_start:
                current_week_activities.append(activity)
            else:
                # 前の週のパターンを計算
                if len(current_week_activities) > 0:
                    pattern = self._calculate_weekly_pattern(current_week_activities)
                    weekly_patterns.append(pattern)
                
                # 新しい週を開始
                current_week_start = week_start
                current_week_activities = [activity]
        
        # 最後の週を追加
        if len(current_week_activities) > 0:
            pattern = self._calculate_weekly_pattern(current_week_activities)
            weekly_patterns.append(pattern)
        
        if len(weekly_patterns) < 2:
            return 0.0
        
        # パターン間の類似度を計算
        similarities = []
        for i in range(len(weekly_patterns) - 1):
            pattern1 = np.array(weekly_patterns[i])
            pattern2 = np.array(weekly_patterns[i + 1])
            
            # コサイン類似度
            if np.linalg.norm(pattern1) > 0 and np.linalg.norm(pattern2) > 0:
                similarity = np.dot(pattern1, pattern2) / (
                    np.linalg.norm(pattern1) * np.linalg.norm(pattern2)
                )
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _extract_forecast_features(self, 
                                 activities: List[Dict[str, Any]],
                                 context_date: datetime) -> Dict[str, Any]:
        """予測特徴量を抽出"""
        
        # 活動予測
        activity_forecast = self._forecast_activity(activities, context_date)
        
        # エンゲージメント予測
        engagement_forecast = self._forecast_engagement(activities, context_date)
        
        # 定着リスクトレンド
        retention_risk_trend = self._calculate_retention_risk_trend(
            activities, context_date
        )
        
        return {
            'activity_forecast': activity_forecast,
            'engagement_forecast': engagement_forecast,
            'retention_risk_trend': retention_risk_trend
        }
    
    def _forecast_activity(self, 
                         activities: List[Dict[str, Any]],
                         context_date: datetime) -> float:
        """活動レベルを予測"""
        
        if len(activities) < 4:
            return 0.0
        
        # 最近の活動トレンドから予測
        recent_trend = self._calculate_period_trend(
            activities, context_date, self.short_term_window
        )
        
        # 現在の活動レベル
        recent_start = context_date - timedelta(days=self.short_term_window)
        recent_activities = [
            a for a in activities 
            if recent_start <= a['timestamp'] <= context_date
        ]
        
        current_activity_level = len(recent_activities) / self.short_term_window
        
        # 予測（現在レベル + トレンド効果）
        forecast = current_activity_level * (1 + recent_trend)
        
        return max(0.0, forecast)
    
    def _forecast_engagement(self, 
                           activities: List[Dict[str, Any]],
                           context_date: datetime) -> float:
        """エンゲージメントレベルを予測"""
        
        if len(activities) < 2:
            return 0.0
        
        # エンゲージメント指標（レビューの質、応答時間など）
        engagement_scores = []
        
        for activity in activities:
            if activity['type'] == 'review_given':
                review_data = activity['data']
                
                # レビューの質を評価
                score = abs(review_data.get('score', 0))
                message_length = len(review_data.get('message', ''))
                response_time = review_data.get('response_time_hours', 24)
                
                # エンゲージメントスコア
                engagement = (
                    (score / 2.0) * 0.4 +  # スコアの絶対値
                    min(message_length / 100.0, 1.0) * 0.3 +  # コメントの詳細度
                    max(0, 1.0 - response_time / 48.0) * 0.3  # 応答速度
                )
                
                engagement_scores.append(engagement)
        
        if not engagement_scores:
            return 0.0
        
        # 最近のエンゲージメントトレンド
        if len(engagement_scores) >= 4:
            recent_half = engagement_scores[-len(engagement_scores)//2:]
            past_half = engagement_scores[:len(engagement_scores)//2]
            
            recent_avg = np.mean(recent_half)
            past_avg = np.mean(past_half)
            
            if past_avg > 0:
                trend = (recent_avg - past_avg) / past_avg
                forecast = recent_avg * (1 + trend)
            else:
                forecast = recent_avg
        else:
            forecast = np.mean(engagement_scores)
        
        return max(0.0, min(1.0, forecast))
    
    def _calculate_retention_risk_trend(self, 
                                      activities: List[Dict[str, Any]],
                                      context_date: datetime) -> float:
        """定着リスクトレンドを計算"""
        
        if len(activities) < 4:
            return 0.5  # 中立的なリスク
        
        # リスク指標
        risk_indicators = []
        
        # 1. 活動減少トレンド
        activity_trend = self._calculate_period_trend(
            activities, context_date, self.medium_term_window
        )
        if activity_trend < -0.2:  # 20%以上の減少
            risk_indicators.append(0.8)
        elif activity_trend < 0:
            risk_indicators.append(0.6)
        else:
            risk_indicators.append(0.2)
        
        # 2. エンゲージメント低下
        engagement_forecast = self._forecast_engagement(activities, context_date)
        if engagement_forecast < 0.3:
            risk_indicators.append(0.8)
        elif engagement_forecast < 0.5:
            risk_indicators.append(0.6)
        else:
            risk_indicators.append(0.2)
        
        # 3. 活動の不規則性
        pattern_consistency = self._calculate_pattern_consistency(activities)
        if pattern_consistency < 0.3:
            risk_indicators.append(0.7)
        elif pattern_consistency < 0.5:
            risk_indicators.append(0.5)
        else:
            risk_indicators.append(0.3)
        
        return np.mean(risk_indicators)
    
    def _extract_consistency_features(self, 
                                    changes_data: List[Dict[str, Any]],
                                    reviews_data: List[Dict[str, Any]],
                                    context_date: datetime) -> Dict[str, Any]:
        """時系列整合性特徴量を抽出"""
        
        # 時系列整合性スコア
        temporal_consistency_score = self._calculate_temporal_consistency_score(
            changes_data, reviews_data, context_date
        )
        
        # データリークリスク
        data_leakage_risk = self._calculate_data_leakage_risk(
            changes_data, reviews_data, context_date
        )
        
        return {
            'temporal_consistency_score': temporal_consistency_score,
            'data_leakage_risk': data_leakage_risk
        }
    
    def _calculate_temporal_consistency_score(self, 
                                            changes_data: List[Dict[str, Any]],
                                            reviews_data: List[Dict[str, Any]],
                                            context_date: datetime) -> float:
        """時系列整合性スコアを計算"""
        
        consistency_checks = []
        
        # 1. 未来データの不存在
        future_changes = sum(
            1 for c in changes_data 
            if datetime.fromisoformat(c.get('created', '')) > context_date
        )
        future_reviews = sum(
            1 for r in reviews_data 
            if datetime.fromisoformat(r.get('timestamp', '')) > context_date
        )
        
        total_data = len(changes_data) + len(reviews_data)
        if total_data > 0:
            future_ratio = (future_changes + future_reviews) / total_data
            consistency_checks.append(1.0 - future_ratio)
        
        # 2. データの時系列順序
        change_dates = [
            datetime.fromisoformat(c.get('created', '')) 
            for c in changes_data if c.get('created')
        ]
        review_dates = [
            datetime.fromisoformat(r.get('timestamp', '')) 
            for r in reviews_data if r.get('timestamp')
        ]
        
        all_dates = change_dates + review_dates
        if len(all_dates) > 1:
            sorted_dates = sorted(all_dates)
            order_consistency = 1.0 if all_dates == sorted_dates else 0.8
            consistency_checks.append(order_consistency)
        
        # 3. レビューとChangeの時系列関係
        review_consistency = self._check_review_change_consistency(
            changes_data, reviews_data
        )
        consistency_checks.append(review_consistency)
        
        return np.mean(consistency_checks) if consistency_checks else 1.0
    
    def _check_review_change_consistency(self, 
                                       changes_data: List[Dict[str, Any]],
                                       reviews_data: List[Dict[str, Any]]) -> float:
        """レビューとChangeの時系列整合性をチェック"""
        
        consistency_violations = 0
        total_reviews = 0
        
        for review in reviews_data:
            change_id = review.get('change_id')
            review_time = datetime.fromisoformat(review.get('timestamp', ''))
            
            # 対応するChangeを検索
            matching_change = None
            for change in changes_data:
                if change.get('change_id') == change_id:
                    matching_change = change
                    break
            
            if matching_change:
                change_time = datetime.fromisoformat(matching_change.get('created', ''))
                
                # レビューはChangeの後に行われるべき
                if review_time < change_time:
                    consistency_violations += 1
                
                total_reviews += 1
        
        if total_reviews == 0:
            return 1.0
        
        consistency_ratio = 1.0 - (consistency_violations / total_reviews)
        return consistency_ratio
    
    def _calculate_data_leakage_risk(self, 
                                   changes_data: List[Dict[str, Any]],
                                   reviews_data: List[Dict[str, Any]],
                                   context_date: datetime) -> float:
        """データリークリスクを計算"""
        
        risk_factors = []
        
        # 1. 未来データの存在
        future_data_count = 0
        total_data_count = len(changes_data) + len(reviews_data)
        
        for change in changes_data:
            if datetime.fromisoformat(change.get('created', '')) > context_date:
                future_data_count += 1
        
        for review in reviews_data:
            if datetime.fromisoformat(review.get('timestamp', '')) > context_date:
                future_data_count += 1
        
        if total_data_count > 0:
            future_data_ratio = future_data_count / total_data_count
            risk_factors.append(future_data_ratio)
        
        # 2. 時系列順序の違反
        order_violations = self._count_temporal_order_violations(
            changes_data, reviews_data
        )
        
        if total_data_count > 0:
            violation_ratio = order_violations / total_data_count
            risk_factors.append(violation_ratio)
        
        return np.mean(risk_factors) if risk_factors else 0.0
    
    def _count_temporal_order_violations(self, 
                                       changes_data: List[Dict[str, Any]],
                                       reviews_data: List[Dict[str, Any]]) -> int:
        """時系列順序違反の数をカウント"""
        
        violations = 0
        
        # レビューがChangeより前に行われている場合
        for review in reviews_data:
            change_id = review.get('change_id')
            review_time = datetime.fromisoformat(review.get('timestamp', ''))
            
            for change in changes_data:
                if (change.get('change_id') == change_id and 
                    datetime.fromisoformat(change.get('created', '')) > review_time):
                    violations += 1
                    break
        
        return violations


def create_temporal_feature_extractor(config_path: str) -> TemporalFeatureExtractor:
    """
    設定ファイルから時系列特徴量抽出器を作成
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        TemporalFeatureExtractor: 設定済みの抽出器
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    feature_config = config.get('temporal_features', {})
    return TemporalFeatureExtractor(feature_config)


def validate_temporal_data_integrity(changes_data: List[Dict[str, Any]],
                                    reviews_data: List[Dict[str, Any]],
                                    context_date: datetime) -> Dict[str, Any]:
    """
    時系列データの整合性を検証
    
    Args:
        changes_data: Change データのリスト
        reviews_data: Review データのリスト
        context_date: 基準日時
        
    Returns:
        Dict[str, Any]: 検証結果
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    # 未来データのチェック
    future_changes = [
        c for c in changes_data 
        if datetime.fromisoformat(c.get('created', '')) > context_date
    ]
    future_reviews = [
        r for r in reviews_data 
        if datetime.fromisoformat(r.get('timestamp', '')) > context_date
    ]
    
    if future_changes or future_reviews:
        error_msg = f"未来データが検出されました: Changes={len(future_changes)}, Reviews={len(future_reviews)}"
        validation_results['errors'].append(error_msg)
        validation_results['is_valid'] = False
    
    # 時系列順序のチェック
    all_timestamps = []
    for change in changes_data:
        if change.get('created'):
            all_timestamps.append(datetime.fromisoformat(change.get('created')))
    
    for review in reviews_data:
        if review.get('timestamp'):
            all_timestamps.append(datetime.fromisoformat(review.get('timestamp')))
    
    if all_timestamps:
        sorted_timestamps = sorted(all_timestamps)
        if all_timestamps != sorted_timestamps:
            validation_results['warnings'].append("データが時系列順序でソートされていません")
    
    # 統計情報
    validation_results['statistics'] = {
        'total_changes': len(changes_data),
        'total_reviews': len(reviews_data),
        'future_changes': len(future_changes),
        'future_reviews': len(future_reviews),
        'date_range': {
            'earliest': min(all_timestamps).isoformat() if all_timestamps else None,
            'latest': max(all_timestamps).isoformat() if all_timestamps else None,
            'context_date': context_date.isoformat()
        }
    }
    
    return validation_results


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_changes = [
        {
            'change_id': 'change1',
            'author': 'dev1@example.com',
            'created': '2023-01-10T10:00:00',
            'project': 'project-a'
        },
        {
            'change_id': 'change2',
            'author': 'dev1@example.com',
            'created': '2023-01-15T14:00:00',
            'project': 'project-a'
        }
    ]
    
    sample_reviews = [
        {
            'change_id': 'change1',
            'reviewer_email': 'dev1@example.com',
            'timestamp': '2023-01-12T16:00:00',
            'score': 2,
            'message': 'Good implementation',
            'response_time_hours': 6.0
        },
        {
            'change_id': 'change2',
            'reviewer_email': 'dev1@example.com',
            'timestamp': '2023-01-20T10:00:00',
            'score': 1,
            'message': 'LGTM',
            'response_time_hours': 2.0
        }
    ]
    
    # 時系列特徴量抽出器のテスト
    config = {
        'short_term_window_days': 7,
        'medium_term_window_days': 30,
        'long_term_window_days': 90,
        'min_data_points': 2,
        'enable_strict_validation': True
    }
    
    extractor = TemporalFeatureExtractor(config)
    
    # 時系列整合性の検証
    validation_result = validate_temporal_data_integrity(
        sample_changes, sample_reviews, datetime(2023, 2, 1)
    )
    
    print("時系列データ整合性検証結果:")
    print(f"有効: {validation_result['is_valid']}")
    print(f"警告: {validation_result['warnings']}")
    print(f"エラー: {validation_result['errors']}")
    
    # 特徴量抽出のテスト
    features = extractor.extract_features(
        'dev1@example.com',
        sample_changes,
        sample_reviews,
        datetime(2023, 2, 1)
    )
    
    print(f"\n時系列特徴量抽出完了: {features.developer_email}")
    print(f"活動トレンド: {features.activity_trend:.3f}")
    print(f"短期トレンド: {features.short_term_trend:.3f}")
    print(f"中期トレンド: {features.medium_term_trend:.3f}")
    print(f"長期トレンド: {features.long_term_trend:.3f}")
    print(f"時系列整合性スコア: {features.temporal_consistency_score:.3f}")
    print(f"データリークリスク: {features.data_leakage_risk:.3f}")