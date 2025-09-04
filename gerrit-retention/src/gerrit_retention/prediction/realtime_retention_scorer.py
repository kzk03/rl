"""
リアルタイム継続スコアシステム

開発者の活動に基づいてリアルタイムで継続スコアを更新し、
継続リスクの変化を即座に検出・追跡する。
"""

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dynamic_threshold_calculator import DynamicThresholdCalculator
from .staged_retention_predictor import StagedRetentionPredictor

logger = logging.getLogger(__name__)


class RealtimeRetentionScorer:
    """リアルタイム継続スコア計算器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # 段階的予測器
        self.staged_predictor = StagedRetentionPredictor(
            config.get('staged_prediction', {})
        )
        
        # 動的閾値計算器
        self.threshold_calculator = DynamicThresholdCalculator(
            config.get('dynamic_threshold', {})
        )
        
        # リアルタイムスコア設定
        self.score_update_frequency = config.get('score_update_frequency', 'daily')  # daily, hourly, immediate
        self.activity_window_days = config.get('activity_window_days', 30)  # 活動監視期間
        self.score_decay_rate = config.get('score_decay_rate', 0.95)  # 日次減衰率
        
        # アラート設定
        self.alert_thresholds = config.get('alert_thresholds', {
            'critical': 0.3,    # 30%以下で危険アラート
            'warning': 0.5,     # 50%以下で警告アラート
            'improvement': 0.8  # 80%以上で改善通知
        })
        
        # 活動ブースト設定
        self.activity_boosts = config.get('activity_boosts', {
            'commit': 0.05,           # コミット1回で5%ブースト
            'review': 0.03,           # レビュー1回で3%ブースト
            'comment': 0.01,          # コメント1回で1%ブースト
            'merge': 0.08,            # マージ1回で8%ブースト
            'issue_creation': 0.02,   # Issue作成で2%ブースト
            'issue_resolution': 0.04, # Issue解決で4%ブースト
            'documentation': 0.03,    # ドキュメント更新で3%ブースト
            'collaboration': 0.02     # 他者との協力で2%ブースト
        })
        
        # スコア履歴の保存
        self.score_history = defaultdict(deque)  # 開発者ID -> スコア履歴
        self.last_update_times = {}  # 開発者ID -> 最終更新時刻
        self.activity_buffers = defaultdict(deque)  # 開発者ID -> 最近の活動バッファ
        
        # アラート履歴
        self.alert_history = defaultdict(list)  # 開発者ID -> アラート履歴
        
        logger.info("リアルタイム継続スコア計算器を初期化しました")
    
    def initialize_developer_score(self, 
                                 developer: Dict[str, Any], 
                                 activity_history: List[Dict[str, Any]],
                                 context_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        開発者の初期継続スコアを計算
        
        Args:
            developer: 開発者データ
            activity_history: 活動履歴
            context_date: 基準日
            
        Returns:
            Dict[str, Any]: 初期スコア情報
        """
        if context_date is None:
            context_date = datetime.now()
        
        developer_id = developer.get('developer_id', developer.get('email', 'unknown'))
        
        logger.debug(f"初期継続スコア計算開始: {developer_id}")
        
        # 段階的予測を実行
        staged_prediction = self.staged_predictor.predict_staged_retention(
            developer, activity_history, context_date
        )
        
        # 初期スコアを計算（短期・中期予測の重み付き平均）
        stage_predictions = staged_prediction.get('stage_predictions', {})
        
        initial_score = self._calculate_weighted_score(stage_predictions)
        
        # 動的閾値情報
        dynamic_threshold = staged_prediction.get('dynamic_threshold', {})
        
        # スコア履歴を初期化
        self.score_history[developer_id] = deque(maxlen=100)  # 最大100件の履歴
        self.score_history[developer_id].append({
            'timestamp': context_date,
            'score': initial_score,
            'type': 'initial',
            'details': staged_prediction
        })
        
        # 最終更新時刻を記録
        self.last_update_times[developer_id] = context_date
        
        # 活動バッファを初期化
        self.activity_buffers[developer_id] = deque(maxlen=1000)  # 最大1000件の活動
        
        # 最近の活動をバッファに追加
        recent_cutoff = context_date - timedelta(days=self.activity_window_days)
        for activity in activity_history:
            activity_date = self._parse_activity_date(activity)
            if activity_date and activity_date >= recent_cutoff:
                self.activity_buffers[developer_id].append({
                    'timestamp': activity_date,
                    'activity': activity,
                    'processed': True  # 初期化時は処理済みとする
                })
        
        result = {
            'developer_id': developer_id,
            'initial_score': initial_score,
            'initialization_date': context_date.isoformat(),
            'staged_prediction': staged_prediction,
            'dynamic_threshold': dynamic_threshold,
            'risk_level': self._classify_risk_level(initial_score),
            'next_update_due': self._calculate_next_update_time(context_date)
        }
        
        logger.debug(f"初期継続スコア計算完了: {developer_id} -> {initial_score:.3f}")
        
        return result
    
    def update_score_with_activity(self, 
                                 developer_id: str, 
                                 new_activity: Dict[str, Any],
                                 context_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        新しい活動に基づいてスコアを更新
        
        Args:
            developer_id: 開発者ID
            new_activity: 新しい活動データ
            context_date: 基準日
            
        Returns:
            Dict[str, Any]: 更新結果
        """
        if context_date is None:
            context_date = datetime.now()
        
        if developer_id not in self.score_history:
            logger.warning(f"開発者 {developer_id} の初期スコアが設定されていません")
            return {'error': 'developer_not_initialized'}
        
        logger.debug(f"活動ベーススコア更新: {developer_id}")
        
        # 現在のスコアを取得
        current_score_entry = self.score_history[developer_id][-1]
        current_score = current_score_entry['score']
        
        # 時間減衰を適用
        time_since_last_update = context_date - self.last_update_times[developer_id]
        decay_factor = self.score_decay_rate ** time_since_last_update.days
        decayed_score = current_score * decay_factor
        
        # 活動ブーストを計算
        activity_boost = self._calculate_activity_boost(new_activity)
        
        # 活動の質的評価
        quality_multiplier = self._evaluate_activity_quality(new_activity, developer_id)
        
        # 最終的なブースト
        final_boost = activity_boost * quality_multiplier
        
        # 新しいスコアを計算
        new_score = min(1.0, decayed_score + final_boost)
        
        # 活動をバッファに追加
        self.activity_buffers[developer_id].append({
            'timestamp': context_date,
            'activity': new_activity,
            'processed': False
        })
        
        # スコア履歴を更新
        self.score_history[developer_id].append({
            'timestamp': context_date,
            'score': new_score,
            'type': 'activity_update',
            'previous_score': current_score,
            'decayed_score': decayed_score,
            'activity_boost': final_boost,
            'activity_type': new_activity.get('type', 'unknown'),
            'details': {
                'activity': new_activity,
                'quality_multiplier': quality_multiplier,
                'time_decay_days': time_since_last_update.days
            }
        })
        
        # 最終更新時刻を更新
        self.last_update_times[developer_id] = context_date
        
        # アラートチェック
        alerts = self._check_alerts(developer_id, current_score, new_score)
        
        result = {
            'developer_id': developer_id,
            'previous_score': current_score,
            'new_score': new_score,
            'score_change': new_score - current_score,
            'activity_boost': final_boost,
            'time_decay_applied': time_since_last_update.days > 0,
            'risk_level': self._classify_risk_level(new_score),
            'alerts': alerts,
            'update_timestamp': context_date.isoformat()
        }
        
        logger.debug(f"活動ベーススコア更新完了: {developer_id} -> {new_score:.3f} (変化: {new_score - current_score:+.3f})")
        
        return result
    
    def periodic_score_update(self, 
                            developer_id: str, 
                            developer: Dict[str, Any],
                            context_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        定期的なスコア更新（段階的予測の再実行）
        
        Args:
            developer_id: 開発者ID
            developer: 開発者データ
            context_date: 基準日
            
        Returns:
            Dict[str, Any]: 更新結果
        """
        if context_date is None:
            context_date = datetime.now()
        
        if developer_id not in self.score_history:
            logger.warning(f"開発者 {developer_id} の初期スコアが設定されていません")
            return {'error': 'developer_not_initialized'}
        
        logger.debug(f"定期スコア更新: {developer_id}")
        
        # 最近の活動履歴を構築
        recent_activities = []
        if developer_id in self.activity_buffers:
            for activity_entry in self.activity_buffers[developer_id]:
                recent_activities.append(activity_entry['activity'])
        
        # 段階的予測を再実行
        try:
            staged_prediction = self.staged_predictor.predict_staged_retention(
                developer, recent_activities, context_date
            )
            
            # 新しいスコアを計算
            stage_predictions = staged_prediction.get('stage_predictions', {})
            new_score = self._calculate_weighted_score(stage_predictions)
            
            # 現在のスコアと比較
            current_score = self.score_history[developer_id][-1]['score']
            
            # スコア履歴を更新
            self.score_history[developer_id].append({
                'timestamp': context_date,
                'score': new_score,
                'type': 'periodic_update',
                'previous_score': current_score,
                'details': staged_prediction
            })
            
            # 最終更新時刻を更新
            self.last_update_times[developer_id] = context_date
            
            # アラートチェック
            alerts = self._check_alerts(developer_id, current_score, new_score)
            
            result = {
                'developer_id': developer_id,
                'previous_score': current_score,
                'new_score': new_score,
                'score_change': new_score - current_score,
                'update_type': 'periodic',
                'staged_prediction': staged_prediction,
                'risk_level': self._classify_risk_level(new_score),
                'alerts': alerts,
                'update_timestamp': context_date.isoformat(),
                'next_update_due': self._calculate_next_update_time(context_date)
            }
            
            logger.debug(f"定期スコア更新完了: {developer_id} -> {new_score:.3f} (変化: {new_score - current_score:+.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"定期スコア更新でエラー ({developer_id}): {e}")
            return {'error': str(e)}
    
    def get_current_score(self, developer_id: str) -> Optional[Dict[str, Any]]:
        """現在のスコア情報を取得"""
        
        if developer_id not in self.score_history:
            return None
        
        latest_entry = self.score_history[developer_id][-1]
        
        return {
            'developer_id': developer_id,
            'current_score': latest_entry['score'],
            'last_update': latest_entry['timestamp'].isoformat(),
            'risk_level': self._classify_risk_level(latest_entry['score']),
            'update_type': latest_entry['type']
        }
    
    def get_score_trend(self, 
                       developer_id: str, 
                       days: int = 30) -> Optional[Dict[str, Any]]:
        """スコアのトレンド分析"""
        
        if developer_id not in self.score_history:
            return None
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 指定期間内のスコア履歴を取得
        recent_scores = []
        for entry in self.score_history[developer_id]:
            if entry['timestamp'] >= cutoff_date:
                recent_scores.append({
                    'timestamp': entry['timestamp'],
                    'score': entry['score'],
                    'type': entry['type']
                })
        
        if len(recent_scores) < 2:
            return {
                'developer_id': developer_id,
                'trend': 'insufficient_data',
                'score_count': len(recent_scores)
            }
        
        # トレンド分析
        scores = [entry['score'] for entry in recent_scores]
        timestamps = [(entry['timestamp'] - recent_scores[0]['timestamp']).days 
                     for entry in recent_scores]
        
        # 線形回帰でトレンドを計算
        if len(scores) > 1:
            slope = np.polyfit(timestamps, scores, 1)[0]
            
            if slope > 0.01:
                trend = 'improving'
            elif slope < -0.01:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            slope = 0.0
            trend = 'stable'
        
        # 変動性の計算
        score_std = np.std(scores)
        
        return {
            'developer_id': developer_id,
            'trend': trend,
            'slope': float(slope),
            'volatility': float(score_std),
            'score_count': len(recent_scores),
            'period_days': days,
            'min_score': min(scores),
            'max_score': max(scores),
            'current_score': scores[-1],
            'score_range': max(scores) - min(scores)
        }
    
    def get_risk_dashboard(self, 
                          developer_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """リスクダッシュボード情報を取得"""
        
        if developer_ids is None:
            developer_ids = list(self.score_history.keys())
        
        dashboard_data = {
            'total_developers': len(developer_ids),
            'risk_distribution': {'critical': 0, 'high': 0, 'moderate': 0, 'low': 0},
            'recent_alerts': [],
            'trending_down': [],
            'trending_up': [],
            'inactive_developers': [],
            'summary_stats': {}
        }
        
        current_scores = []
        
        for developer_id in developer_ids:
            if developer_id not in self.score_history:
                continue
            
            # 現在のスコア
            current_score_info = self.get_current_score(developer_id)
            if current_score_info:
                current_score = current_score_info['current_score']
                current_scores.append(current_score)
                
                # リスク分布
                risk_level = current_score_info['risk_level']
                dashboard_data['risk_distribution'][risk_level] += 1
                
                # トレンド分析
                trend_info = self.get_score_trend(developer_id, days=14)  # 2週間のトレンド
                if trend_info:
                    if trend_info['trend'] == 'declining' and trend_info['slope'] < -0.02:
                        dashboard_data['trending_down'].append({
                            'developer_id': developer_id,
                            'current_score': current_score,
                            'trend_slope': trend_info['slope']
                        })
                    elif trend_info['trend'] == 'improving' and trend_info['slope'] > 0.02:
                        dashboard_data['trending_up'].append({
                            'developer_id': developer_id,
                            'current_score': current_score,
                            'trend_slope': trend_info['slope']
                        })
                
                # 非活動開発者
                last_update = datetime.fromisoformat(current_score_info['last_update'])
                days_since_update = (datetime.now() - last_update).days
                if days_since_update > 7:  # 1週間以上更新なし
                    dashboard_data['inactive_developers'].append({
                        'developer_id': developer_id,
                        'current_score': current_score,
                        'days_since_update': days_since_update
                    })
            
            # 最近のアラート
            if developer_id in self.alert_history:
                recent_alerts = [
                    alert for alert in self.alert_history[developer_id]
                    if (datetime.now() - alert['timestamp']).days <= 7
                ]
                dashboard_data['recent_alerts'].extend(recent_alerts)
        
        # 統計情報
        if current_scores:
            dashboard_data['summary_stats'] = {
                'mean_score': float(np.mean(current_scores)),
                'median_score': float(np.median(current_scores)),
                'std_score': float(np.std(current_scores)),
                'min_score': float(min(current_scores)),
                'max_score': float(max(current_scores))
            }
        
        # ソート
        dashboard_data['trending_down'].sort(key=lambda x: x['trend_slope'])
        dashboard_data['trending_up'].sort(key=lambda x: x['trend_slope'], reverse=True)
        dashboard_data['recent_alerts'].sort(key=lambda x: x['timestamp'], reverse=True)
        
        return dashboard_data
    
    def _calculate_weighted_score(self, stage_predictions: Dict[str, Any]) -> float:
        """段階別予測から重み付きスコアを計算"""
        
        # 段階別の重み
        stage_weights = {
            'immediate': 0.4,    # 短期予測を重視
            'short_term': 0.3,
            'medium_term': 0.2,
            'long_term': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for stage_name, prediction in stage_predictions.items():
            if 'probability' in prediction:
                weight = stage_weights.get(stage_name, 0.1)
                weighted_score += prediction['probability'] * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.5  # デフォルト値
    
    def _calculate_activity_boost(self, activity: Dict[str, Any]) -> float:
        """活動に基づくスコアブーストを計算"""
        
        activity_type = activity.get('type', 'unknown').lower()
        
        # 基本ブースト
        base_boost = self.activity_boosts.get(activity_type, 0.01)
        
        # 活動の規模による調整
        size_multiplier = 1.0
        
        if 'lines_added' in activity or 'insertions' in activity:
            lines = activity.get('lines_added', activity.get('insertions', 0))
            if lines > 100:
                size_multiplier = 1.5
            elif lines > 50:
                size_multiplier = 1.2
        
        if 'files_changed' in activity:
            files = activity.get('files_changed', 0)
            if files > 10:
                size_multiplier *= 1.3
            elif files > 5:
                size_multiplier *= 1.1
        
        return base_boost * size_multiplier
    
    def _evaluate_activity_quality(self, 
                                 activity: Dict[str, Any], 
                                 developer_id: str) -> float:
        """活動の質的評価"""
        
        quality_multiplier = 1.0
        
        # コミットメッセージの質
        if 'message' in activity:
            message = activity['message'].lower()
            if len(message) > 50:  # 詳細なメッセージ
                quality_multiplier *= 1.2
            if any(keyword in message for keyword in ['fix', 'bug', 'issue']):
                quality_multiplier *= 1.1  # バグ修正
            if any(keyword in message for keyword in ['test', 'spec']):
                quality_multiplier *= 1.15  # テスト関連
            if any(keyword in message for keyword in ['doc', 'readme']):
                quality_multiplier *= 1.1  # ドキュメント
        
        # レビューの質
        if activity.get('type') == 'review':
            if 'score' in activity:
                score = activity['score']
                if score > 0:  # 承認
                    quality_multiplier *= 1.2
                elif score < 0:  # 拒否（建設的フィードバック）
                    quality_multiplier *= 1.1
            
            if 'comments' in activity:
                comment_count = len(activity['comments'])
                if comment_count > 3:  # 詳細なレビュー
                    quality_multiplier *= 1.3
        
        # 時間帯による調整（勤務時間外の活動は若干減点）
        if 'timestamp' in activity:
            activity_time = self._parse_activity_date(activity)
            if activity_time:
                hour = activity_time.hour
                if hour < 9 or hour > 18:  # 勤務時間外
                    quality_multiplier *= 0.95
                if activity_time.weekday() >= 5:  # 週末
                    quality_multiplier *= 0.9
        
        return min(quality_multiplier, 2.0)  # 最大2倍まで
    
    def _parse_activity_date(self, activity: Dict[str, Any]) -> Optional[datetime]:
        """活動の日時を解析"""
        
        try:
            if 'timestamp' in activity:
                date_str = activity['timestamp']
            elif 'date' in activity:
                date_str = activity['date']
            elif 'created' in activity:
                date_str = activity['created']
            else:
                return None
            
            if isinstance(date_str, str):
                if 'T' in date_str:
                    date_str = date_str.replace('Z', '+00:00')
                    return datetime.fromisoformat(date_str)
                else:
                    return datetime.strptime(date_str, '%Y-%m-%d')
            else:
                return date_str
                
        except (ValueError, TypeError):
            return None
    
    def _classify_risk_level(self, score: float) -> str:
        """スコアからリスクレベルを分類"""
        
        if score >= 0.8:
            return 'low'
        elif score >= 0.6:
            return 'moderate'
        elif score >= 0.4:
            return 'high'
        else:
            return 'critical'
    
    def _check_alerts(self, 
                     developer_id: str, 
                     old_score: float, 
                     new_score: float) -> List[Dict[str, Any]]:
        """アラート条件をチェック"""
        
        alerts = []
        current_time = datetime.now()
        
        # 閾値を下回った場合
        for alert_type, threshold in self.alert_thresholds.items():
            if alert_type in ['critical', 'warning']:
                if old_score > threshold and new_score <= threshold:
                    alert = {
                        'type': alert_type,
                        'message': f'継続スコアが{alert_type}レベル（{threshold}）を下回りました',
                        'old_score': old_score,
                        'new_score': new_score,
                        'threshold': threshold,
                        'timestamp': current_time,
                        'developer_id': developer_id
                    }
                    alerts.append(alert)
                    
                    # アラート履歴に追加
                    self.alert_history[developer_id].append(alert)
        
        # 改善した場合
        if 'improvement' in self.alert_thresholds:
            improvement_threshold = self.alert_thresholds['improvement']
            if old_score < improvement_threshold and new_score >= improvement_threshold:
                alert = {
                    'type': 'improvement',
                    'message': f'継続スコアが改善レベル（{improvement_threshold}）を上回りました',
                    'old_score': old_score,
                    'new_score': new_score,
                    'threshold': improvement_threshold,
                    'timestamp': current_time,
                    'developer_id': developer_id
                }
                alerts.append(alert)
                self.alert_history[developer_id].append(alert)
        
        # 急激な変化
        score_change = abs(new_score - old_score)
        if score_change > 0.2:  # 20%以上の変化
            alert = {
                'type': 'sudden_change',
                'message': f'継続スコアが急激に変化しました（{score_change:.1%}）',
                'old_score': old_score,
                'new_score': new_score,
                'change_magnitude': score_change,
                'timestamp': current_time,
                'developer_id': developer_id
            }
            alerts.append(alert)
            self.alert_history[developer_id].append(alert)
        
        return alerts
    
    def _calculate_next_update_time(self, current_time: datetime) -> datetime:
        """次回更新予定時刻を計算"""
        
        if self.score_update_frequency == 'hourly':
            return current_time + timedelta(hours=1)
        elif self.score_update_frequency == 'daily':
            return current_time + timedelta(days=1)
        else:  # immediate
            return current_time
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """古いデータをクリーンアップ"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleanup_stats = {'developers_cleaned': 0, 'entries_removed': 0, 'alerts_removed': 0}
        
        # スコア履歴のクリーンアップ
        for developer_id in list(self.score_history.keys()):
            original_length = len(self.score_history[developer_id])
            
            # 古いエントリを削除
            self.score_history[developer_id] = deque(
                [entry for entry in self.score_history[developer_id] 
                 if entry['timestamp'] >= cutoff_date],
                maxlen=100
            )
            
            new_length = len(self.score_history[developer_id])
            cleanup_stats['entries_removed'] += original_length - new_length
            
            if new_length == 0:
                del self.score_history[developer_id]
                if developer_id in self.last_update_times:
                    del self.last_update_times[developer_id]
                if developer_id in self.activity_buffers:
                    del self.activity_buffers[developer_id]
                cleanup_stats['developers_cleaned'] += 1
        
        # アラート履歴のクリーンアップ
        for developer_id in list(self.alert_history.keys()):
            original_length = len(self.alert_history[developer_id])
            
            self.alert_history[developer_id] = [
                alert for alert in self.alert_history[developer_id]
                if alert['timestamp'] >= cutoff_date
            ]
            
            new_length = len(self.alert_history[developer_id])
            cleanup_stats['alerts_removed'] += original_length - new_length
            
            if new_length == 0:
                del self.alert_history[developer_id]
        
        logger.info(f"データクリーンアップ完了: {cleanup_stats}")
        
        return cleanup_stats


if __name__ == "__main__":
    # テスト用のサンプル設定
    sample_config = {
        'staged_prediction': {
            'prediction_stages': {
                'immediate': 7,
                'short_term': 30,
                'medium_term': 90
            }
        },
        'dynamic_threshold': {
            'min_threshold_days': 14,
            'max_threshold_days': 365
        },
        'score_update_frequency': 'daily',
        'activity_window_days': 30,
        'score_decay_rate': 0.95,
        'alert_thresholds': {
            'critical': 0.3,
            'warning': 0.5,
            'improvement': 0.8
        }
    }
    
    scorer = RealtimeRetentionScorer(sample_config)
    
    print("リアルタイム継続スコア計算器のテスト完了")
    print(f"アラート閾値: {scorer.alert_thresholds}")
    print(f"活動ブースト設定: {list(scorer.activity_boosts.keys())}")