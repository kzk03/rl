"""
開発者特徴量エンジニアリング

Gerrit特化の開発者特徴量を抽出・計算するモジュール。
専門性、活動パターン、協力関係、Gerrit特有指標を含む。
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DeveloperFeatures:
    """開発者特徴量データクラス"""
    developer_email: str
    
    # 専門性特徴量
    expertise_level: float
    technical_domains: Dict[str, float]
    file_path_expertise: Dict[str, float]
    language_expertise: Dict[str, float]
    
    # 活動パターン特徴量
    activity_frequency: float
    activity_consistency: float
    peak_activity_hours: List[int]
    preferred_days: List[int]
    
    # 協力関係特徴量
    collaboration_network_size: int
    collaboration_quality: float
    mentoring_activity: float
    cross_team_collaboration: float
    
    # Gerrit特有指標
    avg_review_score_given: float
    avg_review_score_received: float
    review_response_time_avg: float
    code_review_thoroughness: float
    change_approval_rate: float
    review_acceptance_rate: float
    
    # 時系列特徴量
    recent_activity_trend: float
    expertise_growth_rate: float
    stress_level_estimate: float


class DeveloperFeatureExtractor:
    """開発者特徴量抽出器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.time_window_days = config.get('time_window_days', 90)
        self.expertise_threshold = config.get('expertise_threshold', 0.1)
        self.collaboration_threshold = config.get('collaboration_threshold', 3)
        
    def extract_features(self, 
                        developer_email: str,
                        changes_data: List[Dict[str, Any]],
                        reviews_data: List[Dict[str, Any]],
                        context_date: datetime) -> DeveloperFeatures:
        """
        開発者の特徴量を抽出
        
        Args:
            developer_email: 開発者のメールアドレス
            changes_data: Change データのリスト
            reviews_data: Review データのリスト
            context_date: 特徴量計算の基準日時
            
        Returns:
            DeveloperFeatures: 抽出された特徴量
        """
        logger.info(f"開発者特徴量を抽出中: {developer_email}")
        
        # 時間窓でデータをフィルタリング
        filtered_changes = self._filter_by_time_window(changes_data, context_date)
        filtered_reviews = self._filter_by_time_window(reviews_data, context_date)
        
        # 各カテゴリの特徴量を計算
        expertise_features = self._extract_expertise_features(
            developer_email, filtered_changes, filtered_reviews
        )
        
        activity_features = self._extract_activity_features(
            developer_email, filtered_changes, filtered_reviews, context_date
        )
        
        collaboration_features = self._extract_collaboration_features(
            developer_email, filtered_changes, filtered_reviews
        )
        
        gerrit_features = self._extract_gerrit_specific_features(
            developer_email, filtered_changes, filtered_reviews
        )
        
        temporal_features = self._extract_temporal_features(
            developer_email, changes_data, reviews_data, context_date
        )
        
        return DeveloperFeatures(
            developer_email=developer_email,
            **expertise_features,
            **activity_features,
            **collaboration_features,
            **gerrit_features,
            **temporal_features
        )
    
    def _filter_by_time_window(self, 
                              data: List[Dict[str, Any]], 
                              context_date: datetime) -> List[Dict[str, Any]]:
        """時間窓でデータをフィルタリング"""
        start_date = context_date - timedelta(days=self.time_window_days)
        
        filtered_data = []
        for item in data:
            item_date = datetime.fromisoformat(item.get('created', item.get('timestamp', '')))
            if start_date <= item_date <= context_date:
                filtered_data.append(item)
        
        return filtered_data
    
    def _extract_expertise_features(self, 
                                   developer_email: str,
                                   changes_data: List[Dict[str, Any]],
                                   reviews_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """専門性特徴量を抽出"""
        
        # 開発者が作成したChangeを取得
        authored_changes = [c for c in changes_data if c.get('author') == developer_email]
        
        # 開発者が行ったレビューを取得
        given_reviews = [r for r in reviews_data if r.get('reviewer_email') == developer_email]
        
        # 技術領域の専門性を計算
        technical_domains = self._calculate_technical_domain_expertise(
            authored_changes, given_reviews
        )
        
        # ファイルパスの専門性を計算
        file_path_expertise = self._calculate_file_path_expertise(
            authored_changes, given_reviews
        )
        
        # プログラミング言語の専門性を計算
        language_expertise = self._calculate_language_expertise(
            authored_changes, given_reviews
        )
        
        # 総合専門性レベルを計算
        expertise_level = self._calculate_overall_expertise_level(
            technical_domains, file_path_expertise, language_expertise
        )
        
        return {
            'expertise_level': expertise_level,
            'technical_domains': technical_domains,
            'file_path_expertise': file_path_expertise,
            'language_expertise': language_expertise
        }
    
    def _calculate_technical_domain_expertise(self, 
                                            authored_changes: List[Dict[str, Any]],
                                            given_reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """技術領域の専門性を計算"""
        domain_counts = defaultdict(int)
        
        # 作成したChangeから技術領域を抽出
        for change in authored_changes:
            domain = change.get('technical_domain', 'unknown')
            domain_counts[domain] += 1
        
        # レビューした内容から技術領域を抽出
        for review in given_reviews:
            domain = review.get('technical_domain', 'unknown')
            domain_counts[domain] += 0.5  # レビューは作成の半分の重み
        
        # 正規化
        total_count = sum(domain_counts.values())
        if total_count == 0:
            return {}
        
        return {domain: count / total_count for domain, count in domain_counts.items()}
    
    def _calculate_file_path_expertise(self, 
                                     authored_changes: List[Dict[str, Any]],
                                     given_reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """ファイルパスの専門性を計算"""
        path_counts = defaultdict(int)
        
        # 作成したChangeのファイルパスを分析
        for change in authored_changes:
            files = change.get('files_changed', [])
            for file_path in files:
                # ディレクトリレベルでの専門性を計算
                path_parts = file_path.split('/')
                for i in range(len(path_parts)):
                    partial_path = '/'.join(path_parts[:i+1])
                    path_counts[partial_path] += 1
        
        # レビューしたファイルパスを分析
        for review in given_reviews:
            files = review.get('files_changed', [])
            for file_path in files:
                path_parts = file_path.split('/')
                for i in range(len(path_parts)):
                    partial_path = '/'.join(path_parts[:i+1])
                    path_counts[partial_path] += 0.5
        
        # 正規化
        total_count = sum(path_counts.values())
        if total_count == 0:
            return {}
        
        return {path: count / total_count for path, count in path_counts.items()
                if count / total_count >= self.expertise_threshold}
    
    def _calculate_language_expertise(self, 
                                    authored_changes: List[Dict[str, Any]],
                                    given_reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """プログラミング言語の専門性を計算"""
        language_counts = defaultdict(int)
        
        # ファイル拡張子からプログラミング言語を推定
        extension_to_language = {
            '.py': 'Python',
            '.java': 'Java',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.sh': 'Shell',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.json': 'JSON',
            '.xml': 'XML',
            '.html': 'HTML',
            '.css': 'CSS'
        }
        
        # 作成したChangeから言語を抽出
        for change in authored_changes:
            files = change.get('files_changed', [])
            for file_path in files:
                for ext, lang in extension_to_language.items():
                    if file_path.endswith(ext):
                        language_counts[lang] += 1
                        break
        
        # レビューから言語を抽出
        for review in given_reviews:
            files = review.get('files_changed', [])
            for file_path in files:
                for ext, lang in extension_to_language.items():
                    if file_path.endswith(ext):
                        language_counts[lang] += 0.5
                        break
        
        # 正規化
        total_count = sum(language_counts.values())
        if total_count == 0:
            return {}
        
        return {lang: count / total_count for lang, count in language_counts.items()}
    
    def _calculate_overall_expertise_level(self, 
                                         technical_domains: Dict[str, float],
                                         file_path_expertise: Dict[str, float],
                                         language_expertise: Dict[str, float]) -> float:
        """総合専門性レベルを計算"""
        # エントロピーベースの専門性計算（低いエントロピー = 高い専門性）
        def calculate_entropy(distribution: Dict[str, float]) -> float:
            if not distribution:
                return 0.0
            entropy = 0.0
            for prob in distribution.values():
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            return entropy
        
        domain_entropy = calculate_entropy(technical_domains)
        path_entropy = calculate_entropy(file_path_expertise)
        language_entropy = calculate_entropy(language_expertise)
        
        # エントロピーを専門性スコアに変換（0-1の範囲）
        max_entropy = np.log2(max(len(technical_domains), 1))
        if max_entropy == 0:
            return 0.0
        
        expertise_score = 1.0 - (domain_entropy / max_entropy)
        return max(0.0, min(1.0, expertise_score))    

    def _extract_activity_features(self, 
                                  developer_email: str,
                                  changes_data: List[Dict[str, Any]],
                                  reviews_data: List[Dict[str, Any]],
                                  context_date: datetime) -> Dict[str, Any]:
        """活動パターン特徴量を抽出"""
        
        # 開発者の活動データを取得
        authored_changes = [c for c in changes_data if c.get('author') == developer_email]
        given_reviews = [r for r in reviews_data if r.get('reviewer_email') == developer_email]
        
        # 活動頻度を計算
        activity_frequency = self._calculate_activity_frequency(
            authored_changes, given_reviews
        )
        
        # 活動の一貫性を計算
        activity_consistency = self._calculate_activity_consistency(
            authored_changes, given_reviews, context_date
        )
        
        # ピーク活動時間を計算
        peak_activity_hours = self._calculate_peak_activity_hours(
            authored_changes, given_reviews
        )
        
        # 好みの曜日を計算
        preferred_days = self._calculate_preferred_days(
            authored_changes, given_reviews
        )
        
        return {
            'activity_frequency': activity_frequency,
            'activity_consistency': activity_consistency,
            'peak_activity_hours': peak_activity_hours,
            'preferred_days': preferred_days
        }
    
    def _calculate_activity_frequency(self, 
                                    authored_changes: List[Dict[str, Any]],
                                    given_reviews: List[Dict[str, Any]]) -> float:
        """活動頻度を計算（1日あたりの活動数）"""
        total_activities = len(authored_changes) + len(given_reviews)
        return total_activities / max(self.time_window_days, 1)
    
    def _calculate_activity_consistency(self, 
                                      authored_changes: List[Dict[str, Any]],
                                      given_reviews: List[Dict[str, Any]],
                                      context_date: datetime) -> float:
        """活動の一貫性を計算（活動の分散の逆数）"""
        all_activities = []
        
        # 全活動の日付を収集
        for change in authored_changes:
            date = datetime.fromisoformat(change.get('created', ''))
            all_activities.append(date)
        
        for review in given_reviews:
            date = datetime.fromisoformat(review.get('timestamp', ''))
            all_activities.append(date)
        
        if len(all_activities) < 2:
            return 0.0
        
        # 日別の活動数を計算
        daily_activities = defaultdict(int)
        for activity_date in all_activities:
            day_key = activity_date.date()
            daily_activities[day_key] += 1
        
        # 分散を計算
        activity_counts = list(daily_activities.values())
        if len(activity_counts) < 2:
            return 0.0
        
        variance = np.var(activity_counts)
        return 1.0 / (1.0 + variance)  # 分散の逆数（一貫性が高いほど大きい値）
    
    def _calculate_peak_activity_hours(self, 
                                     authored_changes: List[Dict[str, Any]],
                                     given_reviews: List[Dict[str, Any]]) -> List[int]:
        """ピーク活動時間を計算"""
        hour_counts = defaultdict(int)
        
        # 時間別の活動数を集計
        for change in authored_changes:
            date = datetime.fromisoformat(change.get('created', ''))
            hour_counts[date.hour] += 1
        
        for review in given_reviews:
            date = datetime.fromisoformat(review.get('timestamp', ''))
            hour_counts[date.hour] += 1
        
        if not hour_counts:
            return []
        
        # 上位3つの時間帯を返す
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, _ in sorted_hours[:3]]
    
    def _calculate_preferred_days(self, 
                                authored_changes: List[Dict[str, Any]],
                                given_reviews: List[Dict[str, Any]]) -> List[int]:
        """好みの曜日を計算（0=月曜日, 6=日曜日）"""
        day_counts = defaultdict(int)
        
        # 曜日別の活動数を集計
        for change in authored_changes:
            date = datetime.fromisoformat(change.get('created', ''))
            day_counts[date.weekday()] += 1
        
        for review in given_reviews:
            date = datetime.fromisoformat(review.get('timestamp', ''))
            day_counts[date.weekday()] += 1
        
        if not day_counts:
            return []
        
        # 上位3つの曜日を返す
        sorted_days = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)
        return [day for day, _ in sorted_days[:3]]
    
    def _extract_collaboration_features(self, 
                                      developer_email: str,
                                      changes_data: List[Dict[str, Any]],
                                      reviews_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """協力関係特徴量を抽出"""
        
        # 協力ネットワークのサイズを計算
        collaboration_network_size = self._calculate_collaboration_network_size(
            developer_email, changes_data, reviews_data
        )
        
        # 協力の質を計算
        collaboration_quality = self._calculate_collaboration_quality(
            developer_email, changes_data, reviews_data
        )
        
        # メンタリング活動を計算
        mentoring_activity = self._calculate_mentoring_activity(
            developer_email, changes_data, reviews_data
        )
        
        # チーム横断協力を計算
        cross_team_collaboration = self._calculate_cross_team_collaboration(
            developer_email, changes_data, reviews_data
        )
        
        return {
            'collaboration_network_size': collaboration_network_size,
            'collaboration_quality': collaboration_quality,
            'mentoring_activity': mentoring_activity,
            'cross_team_collaboration': cross_team_collaboration
        }
    
    def _calculate_collaboration_network_size(self, 
                                            developer_email: str,
                                            changes_data: List[Dict[str, Any]],
                                            reviews_data: List[Dict[str, Any]]) -> int:
        """協力ネットワークのサイズを計算"""
        collaborators = set()
        
        # 自分のChangeをレビューした人を追加
        for change in changes_data:
            if change.get('author') == developer_email:
                for review in reviews_data:
                    if (review.get('change_id') == change.get('change_id') and 
                        review.get('reviewer_email') != developer_email):
                        collaborators.add(review.get('reviewer_email'))
        
        # 自分がレビューしたChangeの作者を追加
        for review in reviews_data:
            if review.get('reviewer_email') == developer_email:
                for change in changes_data:
                    if (change.get('change_id') == review.get('change_id') and 
                        change.get('author') != developer_email):
                        collaborators.add(change.get('author'))
        
        return len(collaborators)
    
    def _calculate_collaboration_quality(self, 
                                       developer_email: str,
                                       changes_data: List[Dict[str, Any]],
                                       reviews_data: List[Dict[str, Any]]) -> float:
        """協力の質を計算"""
        collaboration_scores = []
        
        # 受けたレビューの質を評価
        for change in changes_data:
            if change.get('author') == developer_email:
                change_reviews = [r for r in reviews_data 
                                if r.get('change_id') == change.get('change_id')]
                
                if change_reviews:
                    # レビューの詳細度と建設的さを評価
                    avg_score = np.mean([abs(r.get('score', 0)) for r in change_reviews])
                    avg_response_time = np.mean([r.get('response_time_hours', 24) 
                                               for r in change_reviews])
                    
                    # 質のスコア（高いスコア、速い応答時間が良い）
                    quality_score = avg_score * (24 / max(avg_response_time, 1))
                    collaboration_scores.append(min(quality_score, 5.0))
        
        return np.mean(collaboration_scores) if collaboration_scores else 0.0
    
    def _calculate_mentoring_activity(self, 
                                    developer_email: str,
                                    changes_data: List[Dict[str, Any]],
                                    reviews_data: List[Dict[str, Any]]) -> float:
        """メンタリング活動を計算"""
        mentoring_indicators = []
        
        # 詳細なレビューコメントの提供
        detailed_reviews = [r for r in reviews_data 
                          if (r.get('reviewer_email') == developer_email and 
                              len(r.get('message', '')) > 100)]  # 長いコメント
        
        # 建設的なフィードバック（正のスコア）
        constructive_reviews = [r for r in reviews_data 
                              if (r.get('reviewer_email') == developer_email and 
                                  r.get('score', 0) > 0)]
        
        total_reviews = len([r for r in reviews_data 
                           if r.get('reviewer_email') == developer_email])
        
        if total_reviews == 0:
            return 0.0
        
        # メンタリング指標の計算
        detailed_ratio = len(detailed_reviews) / total_reviews
        constructive_ratio = len(constructive_reviews) / total_reviews
        
        return (detailed_ratio + constructive_ratio) / 2.0
    
    def _calculate_cross_team_collaboration(self, 
                                          developer_email: str,
                                          changes_data: List[Dict[str, Any]],
                                          reviews_data: List[Dict[str, Any]]) -> float:
        """チーム横断協力を計算"""
        # プロジェクト別の活動を分析
        projects = set()
        
        # 自分が関与したプロジェクトを収集
        for change in changes_data:
            if change.get('author') == developer_email:
                projects.add(change.get('project', 'unknown'))
        
        for review in reviews_data:
            if review.get('reviewer_email') == developer_email:
                # レビューしたChangeのプロジェクトを取得
                for change in changes_data:
                    if change.get('change_id') == review.get('change_id'):
                        projects.add(change.get('project', 'unknown'))
        
        # プロジェクト数が多いほど横断協力が高い
        return min(len(projects) / 5.0, 1.0)  # 5プロジェクト以上で最大値
    
    def _extract_gerrit_specific_features(self, 
                                        developer_email: str,
                                        changes_data: List[Dict[str, Any]],
                                        reviews_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gerrit特有の特徴量を抽出"""
        
        # 与えたレビューの平均スコア
        avg_review_score_given = self._calculate_avg_review_score_given(
            developer_email, reviews_data
        )
        
        # 受けたレビューの平均スコア
        avg_review_score_received = self._calculate_avg_review_score_received(
            developer_email, changes_data, reviews_data
        )
        
        # レビュー応答時間の平均
        review_response_time_avg = self._calculate_review_response_time_avg(
            developer_email, reviews_data
        )
        
        # コードレビューの詳細度
        code_review_thoroughness = self._calculate_code_review_thoroughness(
            developer_email, reviews_data
        )
        
        # Change承認率
        change_approval_rate = self._calculate_change_approval_rate(
            developer_email, changes_data, reviews_data
        )
        
        # レビュー受諾率
        review_acceptance_rate = self._calculate_review_acceptance_rate(
            developer_email, changes_data, reviews_data
        )
        
        return {
            'avg_review_score_given': avg_review_score_given,
            'avg_review_score_received': avg_review_score_received,
            'review_response_time_avg': review_response_time_avg,
            'code_review_thoroughness': code_review_thoroughness,
            'change_approval_rate': change_approval_rate,
            'review_acceptance_rate': review_acceptance_rate
        }
    
    def _calculate_avg_review_score_given(self, 
                                        developer_email: str,
                                        reviews_data: List[Dict[str, Any]]) -> float:
        """与えたレビューの平均スコア"""
        given_reviews = [r for r in reviews_data 
                        if r.get('reviewer_email') == developer_email]
        
        if not given_reviews:
            return 0.0
        
        scores = [r.get('score', 0) for r in given_reviews]
        return np.mean(scores)
    
    def _calculate_avg_review_score_received(self, 
                                           developer_email: str,
                                           changes_data: List[Dict[str, Any]],
                                           reviews_data: List[Dict[str, Any]]) -> float:
        """受けたレビューの平均スコア"""
        authored_changes = [c for c in changes_data if c.get('author') == developer_email]
        
        received_scores = []
        for change in authored_changes:
            change_reviews = [r for r in reviews_data 
                            if r.get('change_id') == change.get('change_id')]
            for review in change_reviews:
                received_scores.append(review.get('score', 0))
        
        return np.mean(received_scores) if received_scores else 0.0
    
    def _calculate_review_response_time_avg(self, 
                                          developer_email: str,
                                          reviews_data: List[Dict[str, Any]]) -> float:
        """レビュー応答時間の平均"""
        given_reviews = [r for r in reviews_data 
                        if r.get('reviewer_email') == developer_email]
        
        if not given_reviews:
            return 0.0
        
        response_times = [r.get('response_time_hours', 24) for r in given_reviews]
        return np.mean(response_times)
    
    def _calculate_code_review_thoroughness(self, 
                                          developer_email: str,
                                          reviews_data: List[Dict[str, Any]]) -> float:
        """コードレビューの詳細度"""
        given_reviews = [r for r in reviews_data 
                        if r.get('reviewer_email') == developer_email]
        
        if not given_reviews:
            return 0.0
        
        # コメントの長さと詳細度を評価
        thoroughness_scores = []
        for review in given_reviews:
            message_length = len(review.get('message', ''))
            effort_estimate = review.get('review_effort_estimated', 1.0)
            
            # 詳細度スコア（コメント長 × 推定労力）
            thoroughness = min(message_length / 100.0, 1.0) * effort_estimate
            thoroughness_scores.append(thoroughness)
        
        return np.mean(thoroughness_scores)
    
    def _calculate_change_approval_rate(self, 
                                      developer_email: str,
                                      changes_data: List[Dict[str, Any]],
                                      reviews_data: List[Dict[str, Any]]) -> float:
        """Change承認率"""
        authored_changes = [c for c in changes_data if c.get('author') == developer_email]
        
        if not authored_changes:
            return 0.0
        
        approved_changes = 0
        for change in authored_changes:
            change_reviews = [r for r in reviews_data 
                            if r.get('change_id') == change.get('change_id')]
            
            # +2スコアがあれば承認とみなす
            if any(r.get('score', 0) >= 2 for r in change_reviews):
                approved_changes += 1
        
        return approved_changes / len(authored_changes)
    
    def _calculate_review_acceptance_rate(self, 
                                        developer_email: str,
                                        changes_data: List[Dict[str, Any]],
                                        reviews_data: List[Dict[str, Any]]) -> float:
        """レビュー受諾率（推定）"""
        # レビュー依頼に対する実際のレビュー実施率を推定
        # 実際のGerritデータでは依頼情報が限定的なため、
        # 活動パターンから推定
        
        given_reviews = [r for r in reviews_data 
                        if r.get('reviewer_email') == developer_email]
        
        # 簡易的な推定：レビュー頻度から受諾率を推定
        review_frequency = len(given_reviews) / max(self.time_window_days, 1)
        
        # 頻度を0-1の範囲にマッピング
        return min(review_frequency / 2.0, 1.0)  # 1日2件以上で最大値
    
    def _extract_temporal_features(self, 
                                 developer_email: str,
                                 changes_data: List[Dict[str, Any]],
                                 reviews_data: List[Dict[str, Any]],
                                 context_date: datetime) -> Dict[str, Any]:
        """時系列特徴量を抽出"""
        
        # 最近の活動トレンド
        recent_activity_trend = self._calculate_recent_activity_trend(
            developer_email, changes_data, reviews_data, context_date
        )
        
        # 専門性成長率
        expertise_growth_rate = self._calculate_expertise_growth_rate(
            developer_email, changes_data, reviews_data, context_date
        )
        
        # ストレスレベル推定
        stress_level_estimate = self._calculate_stress_level_estimate(
            developer_email, changes_data, reviews_data, context_date
        )
        
        return {
            'recent_activity_trend': recent_activity_trend,
            'expertise_growth_rate': expertise_growth_rate,
            'stress_level_estimate': stress_level_estimate
        }
    
    def _calculate_recent_activity_trend(self, 
                                       developer_email: str,
                                       changes_data: List[Dict[str, Any]],
                                       reviews_data: List[Dict[str, Any]],
                                       context_date: datetime) -> float:
        """最近の活動トレンドを計算"""
        # 過去30日と前30日の活動を比較
        recent_start = context_date - timedelta(days=30)
        previous_start = context_date - timedelta(days=60)
        
        # 最近30日の活動
        recent_activities = 0
        for change in changes_data:
            change_date = datetime.fromisoformat(change.get('created', ''))
            if (change.get('author') == developer_email and 
                recent_start <= change_date <= context_date):
                recent_activities += 1
        
        for review in reviews_data:
            review_date = datetime.fromisoformat(review.get('timestamp', ''))
            if (review.get('reviewer_email') == developer_email and 
                recent_start <= review_date <= context_date):
                recent_activities += 1
        
        # 前30日の活動
        previous_activities = 0
        for change in changes_data:
            change_date = datetime.fromisoformat(change.get('created', ''))
            if (change.get('author') == developer_email and 
                previous_start <= change_date < recent_start):
                previous_activities += 1
        
        for review in reviews_data:
            review_date = datetime.fromisoformat(review.get('timestamp', ''))
            if (review.get('reviewer_email') == developer_email and 
                previous_start <= review_date < recent_start):
                previous_activities += 1
        
        # トレンド計算（-1から1の範囲）
        if previous_activities == 0:
            return 1.0 if recent_activities > 0 else 0.0
        
        trend = (recent_activities - previous_activities) / previous_activities
        return max(-1.0, min(1.0, trend))
    
    def _calculate_expertise_growth_rate(self, 
                                       developer_email: str,
                                       changes_data: List[Dict[str, Any]],
                                       reviews_data: List[Dict[str, Any]],
                                       context_date: datetime) -> float:
        """専門性成長率を計算"""
        # 新しい技術領域への参入を成長の指標とする
        recent_start = context_date - timedelta(days=30)
        previous_start = context_date - timedelta(days=90)
        
        # 最近30日の技術領域
        recent_domains = set()
        for change in changes_data:
            change_date = datetime.fromisoformat(change.get('created', ''))
            if (change.get('author') == developer_email and 
                recent_start <= change_date <= context_date):
                recent_domains.add(change.get('technical_domain', 'unknown'))
        
        # 過去90日の技術領域
        historical_domains = set()
        for change in changes_data:
            change_date = datetime.fromisoformat(change.get('created', ''))
            if (change.get('author') == developer_email and 
                previous_start <= change_date < recent_start):
                historical_domains.add(change.get('technical_domain', 'unknown'))
        
        # 新しい領域の割合
        if not recent_domains:
            return 0.0
        
        new_domains = recent_domains - historical_domains
        growth_rate = len(new_domains) / len(recent_domains)
        
        return growth_rate
    
    def _calculate_stress_level_estimate(self, 
                                       developer_email: str,
                                       changes_data: List[Dict[str, Any]],
                                       reviews_data: List[Dict[str, Any]],
                                       context_date: datetime) -> float:
        """ストレスレベル推定"""
        stress_indicators = []
        
        # 高負荷指標
        recent_start = context_date - timedelta(days=7)  # 過去1週間
        
        # 1. 高頻度活動
        recent_activities = 0
        for change in changes_data:
            change_date = datetime.fromisoformat(change.get('created', ''))
            if (change.get('author') == developer_email and 
                recent_start <= change_date <= context_date):
                recent_activities += 1
        
        for review in reviews_data:
            review_date = datetime.fromisoformat(review.get('timestamp', ''))
            if (review.get('reviewer_email') == developer_email and 
                recent_start <= review_date <= context_date):
                recent_activities += 1
        
        activity_stress = min(recent_activities / 10.0, 1.0)  # 週10件以上で最大
        stress_indicators.append(activity_stress)
        
        # 2. 低評価レビューの受領
        authored_changes = [c for c in changes_data if c.get('author') == developer_email]
        negative_reviews = 0
        total_reviews = 0
        
        for change in authored_changes:
            change_reviews = [r for r in reviews_data 
                            if r.get('change_id') == change.get('change_id')]
            for review in change_reviews:
                total_reviews += 1
                if review.get('score', 0) < 0:
                    negative_reviews += 1
        
        if total_reviews > 0:
            negative_ratio = negative_reviews / total_reviews
            stress_indicators.append(negative_ratio)
        
        # 3. 長時間のレビュー応答時間
        given_reviews = [r for r in reviews_data 
                        if r.get('reviewer_email') == developer_email]
        
        if given_reviews:
            avg_response_time = np.mean([r.get('response_time_hours', 24) 
                                       for r in given_reviews])
            time_stress = min(avg_response_time / 48.0, 1.0)  # 48時間以上で最大
            stress_indicators.append(time_stress)
        
        return np.mean(stress_indicators) if stress_indicators else 0.0


def create_developer_feature_extractor(config_path: str) -> DeveloperFeatureExtractor:
    """
    設定ファイルから開発者特徴量抽出器を作成
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        DeveloperFeatureExtractor: 設定済みの抽出器
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    feature_config = config.get('developer_features', {})
    return DeveloperFeatureExtractor(feature_config)


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_changes = [
        {
            'change_id': 'change1',
            'author': 'dev1@example.com',
            'created': '2023-01-15T10:00:00',
            'project': 'project-a',
            'technical_domain': 'backend',
            'files_changed': ['src/main.py', 'src/utils.py']
        }
    ]
    
    sample_reviews = [
        {
            'change_id': 'change1',
            'reviewer_email': 'dev2@example.com',
            'timestamp': '2023-01-15T14:00:00',
            'score': 2,
            'message': 'LGTM! Good implementation.',
            'response_time_hours': 4.0,
            'review_effort_estimated': 1.5
        }
    ]
    
    # 特徴量抽出器のテスト
    config = {
        'time_window_days': 90,
        'expertise_threshold': 0.1,
        'collaboration_threshold': 3
    }
    
    extractor = DeveloperFeatureExtractor(config)
    features = extractor.extract_features(
        'dev1@example.com',
        sample_changes,
        sample_reviews,
        datetime(2023, 2, 1)
    )
    
    print(f"開発者特徴量抽出完了: {features.developer_email}")
    print(f"専門性レベル: {features.expertise_level:.3f}")
    print(f"活動頻度: {features.activity_frequency:.3f}")
    print(f"協力ネットワークサイズ: {features.collaboration_network_size}")