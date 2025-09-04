"""
動的閾値計算モジュール

開発者の過去の活動パターンに基づいて個別の継続判定閾値を計算する。
固定的な90日閾値ではなく、開発者の特性に応じた適応的な閾値を提供する。
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DynamicThresholdCalculator:
    """動的閾値計算器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # 閾値計算パラメータ
        self.min_threshold_days = config.get('min_threshold_days', 14)  # 最小2週間
        self.max_threshold_days = config.get('max_threshold_days', 365)  # 最大1年
        self.default_threshold_days = config.get('default_threshold_days', 90)  # デフォルト3ヶ月
        
        # 活動パターン分析パラメータ
        self.min_history_days = config.get('min_history_days', 30)  # 最低履歴期間
        self.activity_gap_multiplier = config.get('activity_gap_multiplier', 1.5)  # 活動間隔の倍率
        self.seasonal_adjustment = config.get('seasonal_adjustment', True)  # 季節調整
        
        # 開発者タイプ別調整
        self.developer_type_adjustments = config.get('developer_type_adjustments', {
            'newcomer': 0.7,      # 新人は短めの閾値
            'regular': 1.0,       # 通常開発者
            'veteran': 1.3,       # ベテランは長めの閾値
            'maintainer': 1.5,    # メンテナーは最も長い閾値
            'occasional': 2.0     # 時々参加する開発者は非常に長い閾値
        })
        
        logger.info("動的閾値計算器を初期化しました")
    
    def calculate_dynamic_threshold(self, 
                                  developer: Dict[str, Any], 
                                  activity_history: List[Dict[str, Any]],
                                  context_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        開発者の動的閾値を計算
        
        Args:
            developer: 開発者データ
            activity_history: 活動履歴
            context_date: 基準日（Noneの場合は現在日時）
            
        Returns:
            Dict[str, Any]: 閾値情報
        """
        if context_date is None:
            context_date = datetime.now()
        
        # タイムゾーンを統一（naive datetimeの場合はローカルタイムゾーンを追加）
        if context_date.tzinfo is None:
            context_date = context_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
        
        developer_id = developer.get('developer_id', developer.get('email', 'unknown'))
        
        logger.debug(f"動的閾値計算開始: {developer_id}")
        
        # 基本的な活動パターン分析
        activity_patterns = self._analyze_activity_patterns(activity_history, context_date)
        
        # 開発者タイプの判定
        developer_type = self._classify_developer_type(developer, activity_history)
        
        # 季節性・時期的要因の分析
        seasonal_factors = self._analyze_seasonal_factors(activity_history, context_date)
        
        # 基本閾値の計算
        base_threshold = self._calculate_base_threshold(activity_patterns)
        
        # 開発者タイプによる調整
        type_adjusted_threshold = base_threshold * self.developer_type_adjustments.get(
            developer_type, 1.0
        )
        
        # 季節調整
        seasonally_adjusted_threshold = self._apply_seasonal_adjustment(
            type_adjusted_threshold, seasonal_factors, context_date
        )
        
        # 最終的な閾値（範囲制限）
        final_threshold = max(
            self.min_threshold_days,
            min(seasonally_adjusted_threshold, self.max_threshold_days)
        )
        
        # 信頼度の計算
        confidence = self._calculate_confidence(activity_patterns, len(activity_history))
        
        result = {
            'threshold_days': int(final_threshold),
            'base_threshold': int(base_threshold),
            'developer_type': developer_type,
            'type_adjustment': self.developer_type_adjustments.get(developer_type, 1.0),
            'seasonal_adjustment': seasonal_factors.get('adjustment_factor', 1.0),
            'confidence': confidence,
            'activity_patterns': activity_patterns,
            'calculation_date': context_date.isoformat(),
            'reasoning': self._generate_reasoning(
                base_threshold, developer_type, seasonal_factors, final_threshold
            )
        }
        
        logger.debug(f"動的閾値計算完了: {developer_id} -> {final_threshold}日 (信頼度: {confidence:.2f})")
        
        return result
    
    def _analyze_activity_patterns(self, 
                                 activity_history: List[Dict[str, Any]], 
                                 context_date: datetime) -> Dict[str, Any]:
        """活動パターンを分析"""
        
        if not activity_history:
            return {
                'avg_gap_days': self.default_threshold_days,
                'median_gap_days': self.default_threshold_days,
                'gap_std': 0,
                'activity_frequency': 0,
                'longest_gap': self.default_threshold_days,
                'recent_trend': 'unknown'
            }
        
        # 活動日時を抽出・ソート
        activity_dates = []
        for activity in activity_history:
            try:
                if 'timestamp' in activity:
                    date_str = activity['timestamp']
                elif 'date' in activity:
                    date_str = activity['date']
                elif 'created' in activity:
                    date_str = activity['created']
                else:
                    continue
                
                # 日時解析
                if isinstance(date_str, str):
                    # ISO形式の場合
                    if 'T' in date_str:
                        date_str = date_str.replace('Z', '+00:00')
                        activity_date = datetime.fromisoformat(date_str)
                    else:
                        activity_date = datetime.strptime(date_str, '%Y-%m-%d')
                        # タイムゾーンを追加
                        activity_date = activity_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
                else:
                    activity_date = date_str
                
                # タイムゾーンを統一
                if activity_date.tzinfo is None:
                    activity_date = activity_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
                
                activity_dates.append(activity_date)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"活動日時の解析エラー: {e}")
                continue
        
        if len(activity_dates) < 2:
            return {
                'avg_gap_days': self.default_threshold_days,
                'median_gap_days': self.default_threshold_days,
                'gap_std': 0,
                'activity_frequency': len(activity_dates),
                'longest_gap': self.default_threshold_days,
                'recent_trend': 'insufficient_data'
            }
        
        # 活動日時をソート
        activity_dates.sort()
        
        # 活動間隔を計算
        gaps = []
        for i in range(1, len(activity_dates)):
            gap = (activity_dates[i] - activity_dates[i-1]).days
            gaps.append(gap)
        
        # 統計値を計算
        avg_gap = np.mean(gaps)
        median_gap = np.median(gaps)
        gap_std = np.std(gaps)
        longest_gap = max(gaps)
        
        # 活動頻度（日あたりの活動数）
        total_period = (activity_dates[-1] - activity_dates[0]).days
        activity_frequency = len(activity_dates) / max(total_period, 1)
        
        # 最近の傾向分析
        recent_trend = self._analyze_recent_trend(activity_dates, context_date)
        
        return {
            'avg_gap_days': avg_gap,
            'median_gap_days': median_gap,
            'gap_std': gap_std,
            'activity_frequency': activity_frequency,
            'longest_gap': longest_gap,
            'recent_trend': recent_trend,
            'total_activities': len(activity_dates),
            'analysis_period_days': total_period
        }
    
    def _analyze_recent_trend(self, 
                            activity_dates: List[datetime], 
                            context_date: datetime) -> str:
        """最近の活動傾向を分析"""
        
        if len(activity_dates) < 3:
            return 'insufficient_data'
        
        # 最近30日間の活動
        recent_cutoff = context_date - timedelta(days=30)
        recent_activities = []
        for d in activity_dates:
            if d.tzinfo is None:
                d = d.replace(tzinfo=context_date.tzinfo)
            if d >= recent_cutoff:
                recent_activities.append(d)
        
        # 過去30-60日間の活動
        past_cutoff = context_date - timedelta(days=60)
        past_activities = []
        for d in activity_dates:
            if d.tzinfo is None:
                d = d.replace(tzinfo=context_date.tzinfo)
            if past_cutoff <= d < recent_cutoff:
                past_activities.append(d)
        
        recent_count = len(recent_activities)
        past_count = len(past_activities)
        
        if past_count == 0:
            return 'new_developer'
        
        # 活動変化率
        change_ratio = recent_count / past_count
        
        if change_ratio > 1.5:
            return 'increasing'
        elif change_ratio < 0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _classify_developer_type(self, 
                               developer: Dict[str, Any], 
                               activity_history: List[Dict[str, Any]]) -> str:
        """開発者タイプを分類"""
        
        # 活動期間
        first_seen = developer.get('first_seen')
        if first_seen:
            try:
                if isinstance(first_seen, str):
                    first_date = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                else:
                    first_date = first_seen
                
                # タイムゾーンを統一
                if first_date.tzinfo is None:
                    first_date = first_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
                
                current_time = datetime.now().astimezone()
                experience_days = (current_time - first_date).days
            except:
                experience_days = 0
        else:
            experience_days = 0
        
        # 活動量
        total_changes = developer.get('changes_authored', 0)
        total_reviews = developer.get('changes_reviewed', 0)
        total_activity = total_changes + total_reviews
        
        # プロジェクト数
        projects = developer.get('projects', [])
        project_count = len(projects) if isinstance(projects, list) else 0
        
        # 分類ロジック
        if experience_days < 30:
            return 'newcomer'
        elif experience_days < 180:
            if total_activity > 50:
                return 'regular'
            else:
                return 'newcomer'
        elif experience_days < 365:
            if total_activity > 100 and project_count > 2:
                return 'veteran'
            else:
                return 'regular'
        else:
            if total_activity > 500 or project_count > 5:
                return 'maintainer'
            elif total_activity < 50:
                return 'occasional'
            else:
                return 'veteran'
    
    def _analyze_seasonal_factors(self, 
                                activity_history: List[Dict[str, Any]], 
                                context_date: datetime) -> Dict[str, Any]:
        """季節性・時期的要因を分析"""
        
        if not self.seasonal_adjustment:
            return {'adjustment_factor': 1.0, 'seasonal_pattern': 'none'}
        
        # 現在の月
        current_month = context_date.month
        
        # 一般的な開発活動の季節パターン
        # 12月-1月: 休暇期間で活動低下
        # 7-8月: 夏休み期間で活動低下
        # 3-5月, 9-11月: 活発な期間
        seasonal_multipliers = {
            1: 1.3,   # 1月: 休暇明けで閾値を長めに
            2: 1.1,   # 2月: やや長め
            3: 0.9,   # 3月: 活発期間で短め
            4: 0.9,   # 4月: 活発期間で短め
            5: 0.9,   # 5月: 活発期間で短め
            6: 1.0,   # 6月: 標準
            7: 1.2,   # 7月: 夏休み前で長め
            8: 1.3,   # 8月: 夏休み期間で長め
            9: 0.9,   # 9月: 活発期間で短め
            10: 0.9,  # 10月: 活発期間で短め
            11: 0.9,  # 11月: 活発期間で短め
            12: 1.4   # 12月: 年末で最も長め
        }
        
        adjustment_factor = seasonal_multipliers.get(current_month, 1.0)
        
        # 過去の同月の活動パターンがあれば考慮
        historical_pattern = self._analyze_historical_seasonal_pattern(
            activity_history, current_month
        )
        
        if historical_pattern:
            # 履歴がある場合は履歴ベースの調整を優先
            adjustment_factor = (adjustment_factor + historical_pattern) / 2
        
        return {
            'adjustment_factor': adjustment_factor,
            'seasonal_pattern': 'detected' if historical_pattern else 'general',
            'current_month': current_month,
            'historical_factor': historical_pattern
        }
    
    def _analyze_historical_seasonal_pattern(self, 
                                           activity_history: List[Dict[str, Any]], 
                                           target_month: int) -> Optional[float]:
        """過去の同月の活動パターンを分析"""
        
        if len(activity_history) < 12:  # 1年分のデータがない場合
            return None
        
        # 月別活動数を集計
        monthly_activities = {}
        
        for activity in activity_history:
            try:
                if 'timestamp' in activity:
                    date_str = activity['timestamp']
                elif 'date' in activity:
                    date_str = activity['date']
                else:
                    continue
                
                if isinstance(date_str, str):
                    activity_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    activity_date = date_str
                
                month = activity_date.month
                monthly_activities[month] = monthly_activities.get(month, 0) + 1
                
            except:
                continue
        
        if len(monthly_activities) < 6:  # 半年分のデータがない場合
            return None
        
        # 対象月の活動レベル
        target_month_activity = monthly_activities.get(target_month, 0)
        avg_monthly_activity = np.mean(list(monthly_activities.values()))
        
        if avg_monthly_activity == 0:
            return None
        
        # 活動比率に基づく調整係数
        activity_ratio = target_month_activity / avg_monthly_activity
        
        if activity_ratio > 1.2:
            return 0.8  # 活発な月は閾値を短く
        elif activity_ratio < 0.8:
            return 1.3  # 不活発な月は閾値を長く
        else:
            return 1.0  # 標準
    
    def _calculate_base_threshold(self, activity_patterns: Dict[str, Any]) -> float:
        """基本閾値を計算"""
        
        avg_gap = activity_patterns.get('avg_gap_days', self.default_threshold_days)
        median_gap = activity_patterns.get('median_gap_days', self.default_threshold_days)
        longest_gap = activity_patterns.get('longest_gap', self.default_threshold_days)
        
        # 複数の指標を組み合わせて基本閾値を計算
        # 平均間隔 * 倍率を基本とし、中央値と最長間隔も考慮
        base_from_avg = avg_gap * self.activity_gap_multiplier
        base_from_median = median_gap * self.activity_gap_multiplier
        base_from_longest = longest_gap * 0.8  # 最長間隔の80%
        
        # 重み付き平均
        base_threshold = (
            base_from_avg * 0.4 +
            base_from_median * 0.4 +
            base_from_longest * 0.2
        )
        
        # 異常値の処理
        if base_threshold > self.max_threshold_days:
            base_threshold = self.max_threshold_days
        elif base_threshold < self.min_threshold_days:
            base_threshold = self.default_threshold_days
        
        return base_threshold
    
    def _apply_seasonal_adjustment(self, 
                                 threshold: float, 
                                 seasonal_factors: Dict[str, Any], 
                                 context_date: datetime) -> float:
        """季節調整を適用"""
        
        adjustment_factor = seasonal_factors.get('adjustment_factor', 1.0)
        adjusted_threshold = threshold * adjustment_factor
        
        return adjusted_threshold
    
    def _calculate_confidence(self, 
                            activity_patterns: Dict[str, Any], 
                            history_length: int) -> float:
        """閾値計算の信頼度を計算"""
        
        # 履歴の長さによる信頼度
        history_confidence = min(history_length / 50.0, 1.0)  # 50件で最大信頼度
        
        # 活動パターンの安定性による信頼度
        gap_std = activity_patterns.get('gap_std', 0)
        avg_gap = activity_patterns.get('avg_gap_days', 1)
        
        if avg_gap > 0:
            stability_confidence = max(0.1, 1.0 - (gap_std / avg_gap))
        else:
            stability_confidence = 0.1
        
        # 総合信頼度
        overall_confidence = (history_confidence + stability_confidence) / 2.0
        
        return min(max(overall_confidence, 0.1), 1.0)
    
    def _generate_reasoning(self, 
                          base_threshold: float, 
                          developer_type: str, 
                          seasonal_factors: Dict[str, Any], 
                          final_threshold: float) -> str:
        """閾値計算の理由を生成"""
        
        reasoning_parts = []
        
        # 基本閾値の説明
        reasoning_parts.append(f"活動パターン分析により基本閾値を{base_threshold:.0f}日に設定")
        
        # 開発者タイプによる調整
        type_adjustment = self.developer_type_adjustments.get(developer_type, 1.0)
        if type_adjustment != 1.0:
            reasoning_parts.append(
                f"開発者タイプ（{developer_type}）により{type_adjustment:.1f}倍に調整"
            )
        
        # 季節調整
        seasonal_adjustment = seasonal_factors.get('adjustment_factor', 1.0)
        if seasonal_adjustment != 1.0:
            reasoning_parts.append(
                f"季節要因により{seasonal_adjustment:.1f}倍に調整"
            )
        
        # 最終結果
        reasoning_parts.append(f"最終的な継続判定閾値: {final_threshold:.0f}日")
        
        return "。".join(reasoning_parts)
    
    def batch_calculate_thresholds(self, 
                                 developers_with_history: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
                                 context_date: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        """複数開発者の動的閾値を一括計算"""
        
        if context_date is None:
            context_date = datetime.now()
        
        logger.info(f"動的閾値一括計算開始: {len(developers_with_history)}人")
        
        results = {}
        
        for developer, activity_history in developers_with_history:
            developer_id = developer.get('developer_id', developer.get('email', 'unknown'))
            
            try:
                threshold_info = self.calculate_dynamic_threshold(
                    developer, activity_history, context_date
                )
                results[developer_id] = threshold_info
                
            except Exception as e:
                logger.error(f"動的閾値計算エラー ({developer_id}): {e}")
                # デフォルト値を設定
                results[developer_id] = {
                    'threshold_days': self.default_threshold_days,
                    'base_threshold': self.default_threshold_days,
                    'developer_type': 'unknown',
                    'confidence': 0.1,
                    'error': str(e)
                }
        
        logger.info(f"動的閾値一括計算完了: {len(results)}人")
        
        return results
    
    def get_threshold_statistics(self, threshold_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """閾値計算結果の統計情報を取得"""
        
        thresholds = [result['threshold_days'] for result in threshold_results.values()]
        confidences = [result['confidence'] for result in threshold_results.values()]
        
        # 開発者タイプ別統計
        type_stats = {}
        for result in threshold_results.values():
            dev_type = result.get('developer_type', 'unknown')
            if dev_type not in type_stats:
                type_stats[dev_type] = []
            type_stats[dev_type].append(result['threshold_days'])
        
        type_averages = {
            dev_type: np.mean(thresholds_list)
            for dev_type, thresholds_list in type_stats.items()
        }
        
        return {
            'total_developers': len(threshold_results),
            'threshold_stats': {
                'mean': np.mean(thresholds),
                'median': np.median(thresholds),
                'std': np.std(thresholds),
                'min': min(thresholds),
                'max': max(thresholds)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'min': min(confidences),
                'max': max(confidences)
            },
            'developer_type_distribution': {
                dev_type: len(thresholds_list)
                for dev_type, thresholds_list in type_stats.items()
            },
            'developer_type_averages': type_averages
        }


def create_dynamic_threshold_calculator(config_path: str) -> DynamicThresholdCalculator:
    """
    設定ファイルから動的閾値計算器を作成
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        DynamicThresholdCalculator: 設定済みの計算器
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return DynamicThresholdCalculator(config.get('dynamic_threshold', {}))


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_config = {
        'min_threshold_days': 14,
        'max_threshold_days': 365,
        'default_threshold_days': 90,
        'activity_gap_multiplier': 1.5,
        'seasonal_adjustment': True
    }
    
    calculator = DynamicThresholdCalculator(sample_config)
    
    # サンプル開発者データ
    sample_developer = {
        'developer_id': 'test_dev@example.com',
        'first_seen': '2023-01-01 00:00:00.000000',
        'changes_authored': 150,
        'changes_reviewed': 200,
        'projects': ['project1', 'project2', 'project3']
    }
    
    # サンプル活動履歴
    sample_activities = [
        {'timestamp': '2023-01-15T10:00:00Z', 'type': 'commit'},
        {'timestamp': '2023-01-20T14:30:00Z', 'type': 'review'},
        {'timestamp': '2023-02-01T09:15:00Z', 'type': 'commit'},
        {'timestamp': '2023-02-15T16:45:00Z', 'type': 'review'},
        {'timestamp': '2023-03-01T11:20:00Z', 'type': 'commit'},
        {'timestamp': '2023-03-20T13:10:00Z', 'type': 'review'},
    ]
    
    # 動的閾値計算のテスト
    result = calculator.calculate_dynamic_threshold(
        sample_developer, 
        sample_activities, 
        datetime(2023, 4, 1)
    )
    
    print("動的閾値計算結果:")
    for key, value in result.items():
        print(f"  {key}: {value}")