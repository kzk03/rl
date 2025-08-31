"""
好み・許容限界分析システム

過去の受諾・拒否パターン学習、好みプロファイル生成、
許容限界動的推定を行う。
"""

import logging
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ReviewDecision(Enum):
    """レビュー決定の種類"""
    ACCEPT = "accept"
    DECLINE = "decline"
    DEFER = "defer"


class PreferenceCategory(Enum):
    """好みカテゴリ"""
    TECHNICAL_DOMAIN = "technical_domain"
    FUNCTIONAL_AREA = "functional_area"
    COMPLEXITY_LEVEL = "complexity_level"
    CHANGE_SIZE = "change_size"
    AUTHOR_RELATIONSHIP = "author_relationship"
    TIME_PATTERN = "time_pattern"


@dataclass
class ReviewHistoryEntry:
    """レビュー履歴エントリ"""
    change_id: str
    author_email: str
    decision: ReviewDecision
    timestamp: datetime
    technical_domains: List[str]
    functional_areas: List[str]
    complexity_score: float
    change_size: int  # lines added + deleted
    response_time_hours: float
    relationship_score: float
    context_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferenceProfile:
    """好みプロファイル"""
    developer_email: str
    technical_domain_preferences: Dict[str, float]
    functional_area_preferences: Dict[str, float]
    complexity_preference_range: Tuple[float, float]
    size_preference_range: Tuple[int, int]
    author_preferences: Dict[str, float]
    time_preferences: Dict[str, float]  # hour -> preference
    overall_acceptance_rate: float
    confidence_score: float
    last_updated: datetime


@dataclass
class ToleranceLimit:
    """許容限界"""
    consecutive_non_preferred_limit: int
    stress_threshold: float
    workload_threshold: float
    complexity_upper_limit: float
    size_upper_limit: int
    relationship_lower_limit: float
    time_pressure_limit: float
    burnout_risk_score: float


@dataclass
class PreferenceAnalysisResult:
    """好み分析結果"""
    preference_profile: PreferenceProfile
    tolerance_limit: ToleranceLimit
    current_tolerance_status: Dict[str, Any]
    risk_assessment: Dict[str, float]
    recommendations: List[str]


class PreferenceAnalyzer:
    """好み・許容限界分析システム
    
    過去の受諾・拒否パターン学習、好みプロファイル生成、
    許容限界動的推定を行う。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定辞書
                - min_history_size: 最小履歴サイズ (default: 10)
                - preference_decay_days: 好み減衰日数 (default: 90)
                - tolerance_window_days: 許容限界計算窓 (default: 30)
                - confidence_threshold: 信頼度閾値 (default: 0.7)
        """
        self.config = config
        self.min_history_size = config.get('min_history_size', 10)
        self.preference_decay_days = config.get('preference_decay_days', 90)
        self.tolerance_window_days = config.get('tolerance_window_days', 30)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # 機械学習モデル
        self.preference_model = LogisticRegression(random_state=42)
        self.tolerance_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        
        # モデル訓練済みフラグ
        self.models_trained = False
        
        logger.info(f"PreferenceAnalyzer initialized with config: {config}")
    
    def analyze_preferences(
        self,
        developer_email: str,
        review_history: List[ReviewHistoryEntry],
        current_context: Optional[Dict[str, Any]] = None
    ) -> PreferenceAnalysisResult:
        """開発者の好みと許容限界を分析
        
        Args:
            developer_email: 開発者メールアドレス
            review_history: レビュー履歴
            current_context: 現在のコンテキスト情報
            
        Returns:
            PreferenceAnalysisResult: 分析結果
        """
        try:
            if len(review_history) < self.min_history_size:
                logger.warning(f"履歴サイズが不足: {len(review_history)} < {self.min_history_size}")
                return self._create_default_analysis_result(developer_email)
            
            # 好みプロファイルを生成
            preference_profile = self._generate_preference_profile(
                developer_email, review_history
            )
            
            # 許容限界を推定
            tolerance_limit = self._estimate_tolerance_limit(
                review_history, current_context
            )
            
            # 現在の許容状況を評価
            current_tolerance_status = self._evaluate_current_tolerance_status(
                review_history, tolerance_limit, current_context
            )
            
            # リスク評価を実行
            risk_assessment = self._assess_risks(
                preference_profile, tolerance_limit, current_tolerance_status
            )
            
            # 推奨事項を生成
            recommendations = self._generate_recommendations(
                preference_profile, tolerance_limit, risk_assessment
            )
            
            return PreferenceAnalysisResult(
                preference_profile=preference_profile,
                tolerance_limit=tolerance_limit,
                current_tolerance_status=current_tolerance_status,
                risk_assessment=risk_assessment,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"好み分析中にエラーが発生: {e}")
            return self._create_default_analysis_result(developer_email)
    
    def predict_acceptance_probability(
        self,
        developer_email: str,
        review_history: List[ReviewHistoryEntry],
        candidate_review: Dict[str, Any]
    ) -> float:
        """特定のレビューの受諾確率を予測
        
        Args:
            developer_email: 開発者メールアドレス
            review_history: レビュー履歴
            candidate_review: 候補レビュー情報
            
        Returns:
            float: 受諾確率 (0.0-1.0)
        """
        try:
            # 好みプロファイルを取得
            preference_profile = self._generate_preference_profile(
                developer_email, review_history
            )
            
            # 候補レビューの特徴量を抽出
            features = self._extract_review_features(candidate_review, preference_profile)
            
            # モデルが訓練されていない場合は訓練
            if not self.models_trained:
                self._train_models(review_history)
            
            # 予測実行
            if hasattr(self.preference_model, 'predict_proba'):
                features_scaled = self.scaler.transform([features])
                probability = self.preference_model.predict_proba(features_scaled)[0][1]
            else:
                # フォールバック: ルールベース予測
                probability = self._rule_based_prediction(candidate_review, preference_profile)
            
            return max(0.0, min(1.0, probability))
            
        except Exception as e:
            logger.error(f"受諾確率予測中にエラーが発生: {e}")
            return 0.5  # デフォルト値
    
    def update_tolerance_limit(
        self,
        developer_email: str,
        review_history: List[ReviewHistoryEntry],
        recent_stress_indicators: Dict[str, float]
    ) -> ToleranceLimit:
        """許容限界を動的に更新
        
        Args:
            developer_email: 開発者メールアドレス
            review_history: レビュー履歴
            recent_stress_indicators: 最近のストレス指標
            
        Returns:
            ToleranceLimit: 更新された許容限界
        """
        try:
            # 基本許容限界を計算
            base_tolerance = self._estimate_tolerance_limit(review_history)
            
            # ストレス指標による調整
            stress_adjustment = self._calculate_stress_adjustment(recent_stress_indicators)
            
            # 最近のパフォーマンスによる調整
            performance_adjustment = self._calculate_performance_adjustment(review_history)
            
            # 調整された許容限界を計算
            adjusted_tolerance = ToleranceLimit(
                consecutive_non_preferred_limit=max(1, int(
                    base_tolerance.consecutive_non_preferred_limit * stress_adjustment
                )),
                stress_threshold=min(1.0, 
                    base_tolerance.stress_threshold * (1.0 + performance_adjustment)
                ),
                workload_threshold=min(1.0,
                    base_tolerance.workload_threshold * (1.0 + performance_adjustment)
                ),
                complexity_upper_limit=min(1.0,
                    base_tolerance.complexity_upper_limit * (1.0 + stress_adjustment)
                ),
                size_upper_limit=max(10, int(
                    base_tolerance.size_upper_limit * (1.0 + stress_adjustment)
                )),
                relationship_lower_limit=max(0.0,
                    base_tolerance.relationship_lower_limit * (1.0 - stress_adjustment * 0.5)
                ),
                time_pressure_limit=min(48.0,
                    base_tolerance.time_pressure_limit * (1.0 + stress_adjustment)
                ),
                burnout_risk_score=min(1.0,
                    base_tolerance.burnout_risk_score + recent_stress_indicators.get('burnout_risk', 0.0)
                )
            )
            
            return adjusted_tolerance
            
        except Exception as e:
            logger.error(f"許容限界更新中にエラーが発生: {e}")
            return self._create_default_tolerance_limit()
    
    def _generate_preference_profile(
        self,
        developer_email: str,
        review_history: List[ReviewHistoryEntry]
    ) -> PreferenceProfile:
        """好みプロファイルを生成"""
        # 時間重み付け（最近の履歴により重みを付ける）
        now = datetime.now()
        weighted_history = []
        
        for entry in review_history:
            days_ago = (now - entry.timestamp).days
            weight = np.exp(-days_ago / self.preference_decay_days)
            weighted_history.append((entry, weight))
        
        # 技術ドメイン好みを計算
        tech_domain_prefs = self._calculate_technical_domain_preferences(weighted_history)
        
        # 機能領域好みを計算
        functional_area_prefs = self._calculate_functional_area_preferences(weighted_history)
        
        # 複雑度好み範囲を計算
        complexity_range = self._calculate_complexity_preference_range(weighted_history)
        
        # サイズ好み範囲を計算
        size_range = self._calculate_size_preference_range(weighted_history)
        
        # 作者好みを計算
        author_prefs = self._calculate_author_preferences(weighted_history)
        
        # 時間好みを計算
        time_prefs = self._calculate_time_preferences(weighted_history)
        
        # 全体受諾率を計算
        acceptance_rate = self._calculate_overall_acceptance_rate(review_history)
        
        # 信頼度スコアを計算
        confidence_score = self._calculate_confidence_score(review_history)
        
        return PreferenceProfile(
            developer_email=developer_email,
            technical_domain_preferences=tech_domain_prefs,
            functional_area_preferences=functional_area_prefs,
            complexity_preference_range=complexity_range,
            size_preference_range=size_range,
            author_preferences=author_prefs,
            time_preferences=time_prefs,
            overall_acceptance_rate=acceptance_rate,
            confidence_score=confidence_score,
            last_updated=now
        )
    
    def _estimate_tolerance_limit(
        self,
        review_history: List[ReviewHistoryEntry],
        current_context: Optional[Dict[str, Any]] = None
    ) -> ToleranceLimit:
        """許容限界を推定"""
        # 連続非好みタスクの限界を計算
        consecutive_limit = self._calculate_consecutive_non_preferred_limit(review_history)
        
        # ストレス閾値を計算
        stress_threshold = self._calculate_stress_threshold(review_history)
        
        # ワークロード閾値を計算
        workload_threshold = self._calculate_workload_threshold(review_history)
        
        # 複雑度上限を計算
        complexity_upper = self._calculate_complexity_upper_limit(review_history)
        
        # サイズ上限を計算
        size_upper = self._calculate_size_upper_limit(review_history)
        
        # 関係性下限を計算
        relationship_lower = self._calculate_relationship_lower_limit(review_history)
        
        # 時間プレッシャー限界を計算
        time_pressure_limit = self._calculate_time_pressure_limit(review_history)
        
        # バーンアウトリスクスコアを計算
        burnout_risk = self._calculate_burnout_risk_score(review_history, current_context)
        
        return ToleranceLimit(
            consecutive_non_preferred_limit=consecutive_limit,
            stress_threshold=stress_threshold,
            workload_threshold=workload_threshold,
            complexity_upper_limit=complexity_upper,
            size_upper_limit=size_upper,
            relationship_lower_limit=relationship_lower,
            time_pressure_limit=time_pressure_limit,
            burnout_risk_score=burnout_risk
        )
    
    def _calculate_technical_domain_preferences(
        self,
        weighted_history: List[Tuple[ReviewHistoryEntry, float]]
    ) -> Dict[str, float]:
        """技術ドメイン好みを計算"""
        domain_stats = defaultdict(lambda: {'accept': 0.0, 'total': 0.0})
        
        for entry, weight in weighted_history:
            for domain in entry.technical_domains:
                domain_stats[domain]['total'] += weight
                if entry.decision == ReviewDecision.ACCEPT:
                    domain_stats[domain]['accept'] += weight
        
        preferences = {}
        for domain, stats in domain_stats.items():
            if stats['total'] > 0:
                preferences[domain] = stats['accept'] / stats['total']
            else:
                preferences[domain] = 0.5  # デフォルト値
        
        return preferences
    
    def _calculate_functional_area_preferences(
        self,
        weighted_history: List[Tuple[ReviewHistoryEntry, float]]
    ) -> Dict[str, float]:
        """機能領域好みを計算"""
        area_stats = defaultdict(lambda: {'accept': 0.0, 'total': 0.0})
        
        for entry, weight in weighted_history:
            for area in entry.functional_areas:
                area_stats[area]['total'] += weight
                if entry.decision == ReviewDecision.ACCEPT:
                    area_stats[area]['accept'] += weight
        
        preferences = {}
        for area, stats in area_stats.items():
            if stats['total'] > 0:
                preferences[area] = stats['accept'] / stats['total']
            else:
                preferences[area] = 0.5  # デフォルト値
        
        return preferences
    
    def _calculate_complexity_preference_range(
        self,
        weighted_history: List[Tuple[ReviewHistoryEntry, float]]
    ) -> Tuple[float, float]:
        """複雑度好み範囲を計算"""
        accepted_complexities = []
        
        for entry, weight in weighted_history:
            if entry.decision == ReviewDecision.ACCEPT:
                # 重みに応じて複数回追加（重み付きサンプリング）
                count = max(1, int(weight * 10))
                accepted_complexities.extend([entry.complexity_score] * count)
        
        if not accepted_complexities:
            return (0.0, 1.0)  # デフォルト範囲
        
        complexities = np.array(accepted_complexities)
        lower_bound = np.percentile(complexities, 10)
        upper_bound = np.percentile(complexities, 90)
        
        return (float(lower_bound), float(upper_bound))
    
    def _calculate_size_preference_range(
        self,
        weighted_history: List[Tuple[ReviewHistoryEntry, float]]
    ) -> Tuple[int, int]:
        """サイズ好み範囲を計算"""
        accepted_sizes = []
        
        for entry, weight in weighted_history:
            if entry.decision == ReviewDecision.ACCEPT:
                count = max(1, int(weight * 10))
                accepted_sizes.extend([entry.change_size] * count)
        
        if not accepted_sizes:
            return (0, 1000)  # デフォルト範囲
        
        sizes = np.array(accepted_sizes)
        lower_bound = int(np.percentile(sizes, 10))
        upper_bound = int(np.percentile(sizes, 90))
        
        return (lower_bound, upper_bound)
    
    def _calculate_author_preferences(
        self,
        weighted_history: List[Tuple[ReviewHistoryEntry, float]]
    ) -> Dict[str, float]:
        """作者好みを計算"""
        author_stats = defaultdict(lambda: {'accept': 0.0, 'total': 0.0})
        
        for entry, weight in weighted_history:
            author_stats[entry.author_email]['total'] += weight
            if entry.decision == ReviewDecision.ACCEPT:
                author_stats[entry.author_email]['accept'] += weight
        
        preferences = {}
        for author, stats in author_stats.items():
            if stats['total'] > 0:
                preferences[author] = stats['accept'] / stats['total']
        
        return preferences
    
    def _calculate_time_preferences(
        self,
        weighted_history: List[Tuple[ReviewHistoryEntry, float]]
    ) -> Dict[str, float]:
        """時間好みを計算"""
        hour_stats = defaultdict(lambda: {'accept': 0.0, 'total': 0.0})
        
        for entry, weight in weighted_history:
            hour = str(entry.timestamp.hour)
            hour_stats[hour]['total'] += weight
            if entry.decision == ReviewDecision.ACCEPT:
                hour_stats[hour]['accept'] += weight
        
        preferences = {}
        for hour, stats in hour_stats.items():
            if stats['total'] > 0:
                preferences[hour] = stats['accept'] / stats['total']
        
        return preferences
    
    def _calculate_overall_acceptance_rate(
        self,
        review_history: List[ReviewHistoryEntry]
    ) -> float:
        """全体受諾率を計算"""
        if not review_history:
            return 0.5
        
        accepted_count = sum(
            1 for entry in review_history
            if entry.decision == ReviewDecision.ACCEPT
        )
        
        return accepted_count / len(review_history)
    
    def _calculate_confidence_score(
        self,
        review_history: List[ReviewHistoryEntry]
    ) -> float:
        """信頼度スコアを計算"""
        # 履歴の豊富さ
        history_richness = min(1.0, len(review_history) / 100.0)
        
        # 時間的分散
        if len(review_history) > 1:
            timestamps = [entry.timestamp for entry in review_history]
            time_span = (max(timestamps) - min(timestamps)).days
            time_diversity = min(1.0, time_span / 365.0)  # 1年で最大
        else:
            time_diversity = 0.0
        
        # 決定の一貫性
        decisions = [entry.decision for entry in review_history]
        decision_counts = Counter(decisions)
        decision_entropy = -sum(
            (count / len(decisions)) * np.log2(count / len(decisions))
            for count in decision_counts.values()
        )
        consistency = 1.0 - (decision_entropy / np.log2(len(ReviewDecision)))
        
        confidence = (
            0.4 * history_richness +
            0.3 * time_diversity +
            0.3 * consistency
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_consecutive_non_preferred_limit(
        self,
        review_history: List[ReviewHistoryEntry]
    ) -> int:
        """連続非好みタスクの限界を計算"""
        # 過去の連続拒否パターンを分析
        consecutive_declines = []
        current_decline_streak = 0
        
        for entry in review_history:
            if entry.decision == ReviewDecision.DECLINE:
                current_decline_streak += 1
            else:
                if current_decline_streak > 0:
                    consecutive_declines.append(current_decline_streak)
                current_decline_streak = 0
        
        if current_decline_streak > 0:
            consecutive_declines.append(current_decline_streak)
        
        if not consecutive_declines:
            return 3  # デフォルト値
        
        # 90パーセンタイルを限界とする
        limit = int(np.percentile(consecutive_declines, 90))
        return max(1, min(10, limit))  # 1-10の範囲に制限
    
    def _calculate_stress_threshold(
        self,
        review_history: List[ReviewHistoryEntry]
    ) -> float:
        """ストレス閾値を計算"""
        # 拒否時のコンテキストからストレス要因を分析
        decline_stress_levels = []
        
        for entry in review_history:
            if entry.decision == ReviewDecision.DECLINE:
                # コンテキストからストレスレベルを推定
                stress_level = entry.context_features.get('stress_level', 0.5)
                decline_stress_levels.append(stress_level)
        
        if not decline_stress_levels:
            return 0.7  # デフォルト値
        
        # 拒否が始まるストレスレベルの中央値
        threshold = np.median(decline_stress_levels)
        return max(0.3, min(0.9, threshold))
    
    def _calculate_workload_threshold(
        self,
        review_history: List[ReviewHistoryEntry]
    ) -> float:
        """ワークロード閾値を計算"""
        decline_workloads = []
        
        for entry in review_history:
            if entry.decision == ReviewDecision.DECLINE:
                workload = entry.context_features.get('workload', 0.5)
                decline_workloads.append(workload)
        
        if not decline_workloads:
            return 0.8  # デフォルト値
        
        threshold = np.median(decline_workloads)
        return max(0.5, min(1.0, threshold))
    
    def _calculate_complexity_upper_limit(
        self,
        review_history: List[ReviewHistoryEntry]
    ) -> float:
        """複雑度上限を計算"""
        decline_complexities = [
            entry.complexity_score for entry in review_history
            if entry.decision == ReviewDecision.DECLINE
        ]
        
        if not decline_complexities:
            return 0.9  # デフォルト値
        
        # 拒否される複雑度の下限（90パーセンタイル）
        limit = np.percentile(decline_complexities, 10)
        return max(0.5, min(1.0, limit))
    
    def _calculate_size_upper_limit(
        self,
        review_history: List[ReviewHistoryEntry]
    ) -> int:
        """サイズ上限を計算"""
        decline_sizes = [
            entry.change_size for entry in review_history
            if entry.decision == ReviewDecision.DECLINE
        ]
        
        if not decline_sizes:
            return 500  # デフォルト値
        
        limit = int(np.percentile(decline_sizes, 10))
        return max(50, min(2000, limit))
    
    def _calculate_relationship_lower_limit(
        self,
        review_history: List[ReviewHistoryEntry]
    ) -> float:
        """関係性下限を計算"""
        decline_relationships = [
            entry.relationship_score for entry in review_history
            if entry.decision == ReviewDecision.DECLINE
        ]
        
        if not decline_relationships:
            return 0.3  # デフォルト値
        
        # 拒否される関係性スコアの上限（90パーセンタイル）
        limit = np.percentile(decline_relationships, 90)
        return max(0.0, min(0.8, limit))
    
    def _calculate_time_pressure_limit(
        self,
        review_history: List[ReviewHistoryEntry]
    ) -> float:
        """時間プレッシャー限界を計算"""
        decline_response_times = [
            entry.response_time_hours for entry in review_history
            if entry.decision == ReviewDecision.DECLINE
        ]
        
        if not decline_response_times:
            return 24.0  # デフォルト値（24時間）
        
        # 拒否される応答時間の上限
        limit = np.percentile(decline_response_times, 90)
        return max(1.0, min(72.0, limit))
    
    def _calculate_burnout_risk_score(
        self,
        review_history: List[ReviewHistoryEntry],
        current_context: Optional[Dict[str, Any]]
    ) -> float:
        """バーンアウトリスクスコアを計算"""
        # 最近の拒否率の増加
        recent_history = [
            entry for entry in review_history
            if (datetime.now() - entry.timestamp).days <= 30
        ]
        
        if len(recent_history) < 5:
            return 0.0
        
        recent_decline_rate = sum(
            1 for entry in recent_history
            if entry.decision == ReviewDecision.DECLINE
        ) / len(recent_history)
        
        # 全体の拒否率と比較
        overall_decline_rate = sum(
            1 for entry in review_history
            if entry.decision == ReviewDecision.DECLINE
        ) / len(review_history)
        
        decline_rate_increase = recent_decline_rate - overall_decline_rate
        
        # 現在のコンテキストからリスク要因を追加
        context_risk = 0.0
        if current_context:
            context_risk = current_context.get('stress_level', 0.0) * 0.5
        
        burnout_risk = max(0.0, decline_rate_increase * 2.0 + context_risk)
        return min(1.0, burnout_risk)
    
    def _evaluate_current_tolerance_status(
        self,
        review_history: List[ReviewHistoryEntry],
        tolerance_limit: ToleranceLimit,
        current_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """現在の許容状況を評価"""
        # 最近の連続拒否数
        recent_consecutive_declines = 0
        for entry in reversed(review_history[-10:]):  # 最新10件
            if entry.decision == ReviewDecision.DECLINE:
                recent_consecutive_declines += 1
            else:
                break
        
        # 現在のストレス・ワークロード状況
        current_stress = current_context.get('stress_level', 0.5) if current_context else 0.5
        current_workload = current_context.get('workload', 0.5) if current_context else 0.5
        
        return {
            'consecutive_declines': recent_consecutive_declines,
            'consecutive_limit_ratio': recent_consecutive_declines / tolerance_limit.consecutive_non_preferred_limit,
            'current_stress': current_stress,
            'stress_limit_ratio': current_stress / tolerance_limit.stress_threshold,
            'current_workload': current_workload,
            'workload_limit_ratio': current_workload / tolerance_limit.workload_threshold,
            'burnout_risk': tolerance_limit.burnout_risk_score
        }
    
    def _assess_risks(
        self,
        preference_profile: PreferenceProfile,
        tolerance_limit: ToleranceLimit,
        current_status: Dict[str, Any]
    ) -> Dict[str, float]:
        """リスク評価を実行"""
        risks = {}
        
        # 連続拒否リスク
        risks['consecutive_decline_risk'] = min(1.0, current_status['consecutive_limit_ratio'])
        
        # ストレス超過リスク
        risks['stress_overload_risk'] = min(1.0, current_status['stress_limit_ratio'])
        
        # ワークロード超過リスク
        risks['workload_overload_risk'] = min(1.0, current_status['workload_limit_ratio'])
        
        # バーンアウトリスク
        risks['burnout_risk'] = tolerance_limit.burnout_risk_score
        
        # 低受諾率リスク
        risks['low_acceptance_risk'] = max(0.0, 0.5 - preference_profile.overall_acceptance_rate) * 2.0
        
        # 総合リスク
        risks['overall_risk'] = np.mean(list(risks.values()))
        
        return risks
    
    def _generate_recommendations(
        self,
        preference_profile: PreferenceProfile,
        tolerance_limit: ToleranceLimit,
        risk_assessment: Dict[str, float]
    ) -> List[str]:
        """推奨事項を生成"""
        recommendations = []
        
        if risk_assessment['consecutive_decline_risk'] > 0.7:
            recommendations.append("連続拒否が限界に近づいています。好みに合うタスクを優先してください。")
        
        if risk_assessment['stress_overload_risk'] > 0.8:
            recommendations.append("ストレスレベルが高すぎます。負荷軽減を検討してください。")
        
        if risk_assessment['workload_overload_risk'] > 0.8:
            recommendations.append("ワークロードが限界を超えています。タスクの再配分を検討してください。")
        
        if risk_assessment['burnout_risk'] > 0.6:
            recommendations.append("バーンアウトリスクが高いです。休息を取ることを強く推奨します。")
        
        if preference_profile.overall_acceptance_rate < 0.3:
            recommendations.append("受諾率が低すぎます。好みの分析を見直し、適合するタスクを増やしてください。")
        
        if preference_profile.confidence_score < self.confidence_threshold:
            recommendations.append("好み分析の信頼度が低いです。より多くの履歴データが必要です。")
        
        return recommendations
    
    def _train_models(self, review_history: List[ReviewHistoryEntry]):
        """機械学習モデルを訓練"""
        try:
            if len(review_history) < self.min_history_size:
                return
            
            # 特徴量とラベルを準備
            features = []
            labels = []
            
            for entry in review_history:
                feature_vector = [
                    entry.complexity_score,
                    entry.change_size / 1000.0,  # 正規化
                    entry.relationship_score,
                    entry.response_time_hours / 24.0,  # 正規化
                    len(entry.technical_domains),
                    len(entry.functional_areas)
                ]
                features.append(feature_vector)
                labels.append(1 if entry.decision == ReviewDecision.ACCEPT else 0)
            
            if len(set(labels)) > 1:  # 複数のクラスが存在する場合のみ訓練
                features_array = np.array(features)
                labels_array = np.array(labels)
                
                # 特徴量を正規化
                features_scaled = self.scaler.fit_transform(features_array)
                
                # モデルを訓練
                self.preference_model.fit(features_scaled, labels_array)
                self.models_trained = True
                
                logger.info("好み予測モデルの訓練が完了しました")
            
        except Exception as e:
            logger.error(f"モデル訓練中にエラーが発生: {e}")
    
    def _extract_review_features(
        self,
        candidate_review: Dict[str, Any],
        preference_profile: PreferenceProfile
    ) -> List[float]:
        """候補レビューから特徴量を抽出"""
        return [
            candidate_review.get('complexity_score', 0.5),
            candidate_review.get('change_size', 100) / 1000.0,
            candidate_review.get('relationship_score', 0.5),
            candidate_review.get('estimated_hours', 2.0) / 24.0,
            len(candidate_review.get('technical_domains', [])),
            len(candidate_review.get('functional_areas', []))
        ]
    
    def _rule_based_prediction(
        self,
        candidate_review: Dict[str, Any],
        preference_profile: PreferenceProfile
    ) -> float:
        """ルールベースの受諾確率予測"""
        score = 0.5  # ベーススコア
        
        # 技術ドメインの好み
        tech_domains = candidate_review.get('technical_domains', [])
        if tech_domains:
            domain_scores = [
                preference_profile.technical_domain_preferences.get(domain, 0.5)
                for domain in tech_domains
            ]
            score += 0.2 * (np.mean(domain_scores) - 0.5)
        
        # 複雑度の好み
        complexity = candidate_review.get('complexity_score', 0.5)
        complexity_range = preference_profile.complexity_preference_range
        if complexity_range[0] <= complexity <= complexity_range[1]:
            score += 0.1
        else:
            score -= 0.1
        
        # 関係性スコア
        relationship = candidate_review.get('relationship_score', 0.5)
        score += 0.2 * (relationship - 0.5)
        
        return max(0.0, min(1.0, score))
    
    def _create_default_analysis_result(self, developer_email: str) -> PreferenceAnalysisResult:
        """デフォルトの分析結果を作成"""
        return PreferenceAnalysisResult(
            preference_profile=PreferenceProfile(
                developer_email=developer_email,
                technical_domain_preferences={},
                functional_area_preferences={},
                complexity_preference_range=(0.0, 1.0),
                size_preference_range=(0, 1000),
                author_preferences={},
                time_preferences={},
                overall_acceptance_rate=0.5,
                confidence_score=0.0,
                last_updated=datetime.now()
            ),
            tolerance_limit=self._create_default_tolerance_limit(),
            current_tolerance_status={
                'consecutive_declines': 0,
                'consecutive_limit_ratio': 0.0,
                'current_stress': 0.5,
                'stress_limit_ratio': 0.5,
                'current_workload': 0.5,
                'workload_limit_ratio': 0.5,
                'burnout_risk': 0.0
            },
            risk_assessment={
                'consecutive_decline_risk': 0.0,
                'stress_overload_risk': 0.0,
                'workload_overload_risk': 0.0,
                'burnout_risk': 0.0,
                'low_acceptance_risk': 0.0,
                'overall_risk': 0.0
            },
            recommendations=["履歴データが不足しています。より多くのレビュー履歴が必要です。"]
        )
    
    def _create_default_tolerance_limit(self) -> ToleranceLimit:
        """デフォルトの許容限界を作成"""
        return ToleranceLimit(
            consecutive_non_preferred_limit=3,
            stress_threshold=0.7,
            workload_threshold=0.8,
            complexity_upper_limit=0.9,
            size_upper_limit=500,
            relationship_lower_limit=0.3,
            time_pressure_limit=24.0,
            burnout_risk_score=0.0
        )
    
    def _calculate_stress_adjustment(self, stress_indicators: Dict[str, float]) -> float:
        """ストレス指標による調整係数を計算"""
        stress_level = stress_indicators.get('stress_level', 0.5)
        workload = stress_indicators.get('workload', 0.5)
        
        # ストレスが高いほど許容限界を下げる
        adjustment = 1.0 - 0.3 * stress_level - 0.2 * workload
        return max(0.5, min(1.5, adjustment))
    
    def _calculate_performance_adjustment(self, review_history: List[ReviewHistoryEntry]) -> float:
        """パフォーマンスによる調整係数を計算"""
        if len(review_history) < 10:
            return 0.0
        
        recent_history = review_history[-10:]
        recent_acceptance_rate = sum(
            1 for entry in recent_history
            if entry.decision == ReviewDecision.ACCEPT
        ) / len(recent_history)
        
        # 受諾率が高いほど許容限界を上げる
        adjustment = (recent_acceptance_rate - 0.5) * 0.2
        return max(-0.2, min(0.2, adjustment))


def create_sample_review_history() -> List[ReviewHistoryEntry]:
    """サンプルのレビュー履歴を作成（テスト用）"""
    history = []
    base_time = datetime.now() - timedelta(days=90)
    
    for i in range(20):
        entry = ReviewHistoryEntry(
            change_id=f"I{i:010d}",
            author_email=f"author{i % 3}@example.com",
            decision=ReviewDecision.ACCEPT if i % 3 != 0 else ReviewDecision.DECLINE,
            timestamp=base_time + timedelta(days=i * 4),
            technical_domains=["python", "javascript"][i % 2:i % 2 + 1],
            functional_areas=["api", "testing", "frontend"][i % 3:i % 3 + 1],
            complexity_score=0.3 + (i % 7) * 0.1,
            change_size=50 + i * 20,
            response_time_hours=2.0 + (i % 5),
            relationship_score=0.4 + (i % 6) * 0.1,
            context_features={
                'stress_level': 0.3 + (i % 4) * 0.1,
                'workload': 0.4 + (i % 5) * 0.1
            }
        )
        history.append(entry)
    
    return history


if __name__ == "__main__":
    # テスト実行
    config = {
        'min_history_size': 10,
        'preference_decay_days': 90,
        'tolerance_window_days': 30,
        'confidence_threshold': 0.7
    }
    
    analyzer = PreferenceAnalyzer(config)
    
    # サンプルデータでテスト
    developer_email = "test@example.com"
    review_history = create_sample_review_history()
    
    result = analyzer.analyze_preferences(developer_email, review_history)
    
    print(f"開発者: {result.preference_profile.developer_email}")
    print(f"全体受諾率: {result.preference_profile.overall_acceptance_rate:.3f}")
    print(f"信頼度: {result.preference_profile.confidence_score:.3f}")
    print(f"技術ドメイン好み: {result.preference_profile.technical_domain_preferences}")
    print(f"許容限界 - 連続拒否: {result.tolerance_limit.consecutive_non_preferred_limit}")
    print(f"許容限界 - ストレス閾値: {result.tolerance_limit.stress_threshold:.3f}")
    print(f"リスク評価: {result.risk_assessment}")
    print(f"推奨事項: {result.recommendations}")