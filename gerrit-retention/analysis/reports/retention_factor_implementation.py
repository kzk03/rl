#!/usr/bin/env python3
"""
継続要因分析の実装部分

mockの部分を実際の実装に置き換えるためのヘルパー関数群
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class RetentionFactorImplementation:
    """継続要因分析の実装クラス"""
    
    def __init__(self, developers_data: List[Dict], reviews_data: List[Dict], features_data: List[Dict]):
        self.developers_data = developers_data
        self.reviews_data = reviews_data
        self.features_data = features_data
    
    def calculate_activity_frequency(self, dev_id: str) -> float:
        """活動頻度を計算"""
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.5
        
        # 活動量から頻度を推定
        total_activities = dev_data.get('changes_authored', 0) + dev_data.get('changes_reviewed', 0)
        
        # 初回活動からの日数
        first_seen = dev_data.get('first_seen', '')
        if first_seen:
            try:
                first_date = self._parse_date(first_seen)
                days_active = (datetime.now() - first_date).days
                if days_active > 0:
                    frequency = min(1.0, total_activities / days_active)
                    return frequency
            except:
                pass
        
        # デフォルト計算
        return min(1.0, total_activities / 100.0)
    
    def calculate_activity_consistency(self, dev_id: str) -> float:
        """活動一貫性を計算"""
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.5
        
        authored = dev_data.get('changes_authored', 0)
        reviewed = dev_data.get('changes_reviewed', 0)
        
        # 作成とレビューのバランス
        total = authored + reviewed
        if total == 0:
            return 0.0
        
        # バランスが良いほど一貫性が高い
        balance = 1.0 - abs(authored - reviewed) / total
        return balance
    
    def calculate_collaboration_diversity(self, dev_id: str) -> float:
        """協力多様性を計算"""
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.5
        
        projects = dev_data.get('projects', [])
        project_count = len(projects)
        
        # プロジェクト数から多様性を推定
        diversity = min(1.0, project_count / 5.0)  # 5プロジェクト以上で最大
        return diversity
    
    def calculate_network_centrality(self, dev_id: str) -> float:
        """ネットワーク中心性を計算"""
        # レビューデータから協力関係を分析
        collaborations = set()
        
        for review in self.reviews_data:
            # レビューデータの構造に応じて協力者を特定
            # 簡易実装: レビュー数から推定
            pass
        
        # 簡易実装: レビュー数から中心性を推定
        dev_data = self._get_developer_data(dev_id)
        if dev_data:
            reviewed = dev_data.get('changes_reviewed', 0)
            centrality = min(1.0, reviewed / 50.0)  # 50レビュー以上で最大
            return centrality
        
        return 0.5
    
    def calculate_mentoring_activity(self, dev_id: str) -> float:
        """メンタリング活動を計算"""
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.0
        
        # レビュー数が多い場合はメンタリング活動が高いと推定
        reviewed = dev_data.get('changes_reviewed', 0)
        authored = dev_data.get('changes_authored', 0)
        
        if authored == 0:
            return 0.0
        
        # レビュー/作成比率が高いほどメンタリング活動が高い
        ratio = reviewed / (authored + 1)  # +1 to avoid division by zero
        mentoring = min(1.0, ratio / 3.0)  # 3倍以上で最大
        return mentoring
    
    def calculate_avg_review_quality(self, dev_id: str) -> float:
        """平均レビュー品質を計算"""
        # 簡易実装: レビュー数と活動期間から品質を推定
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.7
        
        reviewed = dev_data.get('changes_reviewed', 0)
        authored = dev_data.get('changes_authored', 0)
        
        # 経験値から品質を推定
        experience = authored + reviewed
        quality = 0.5 + min(0.4, experience / 100.0)  # 経験に基づく品質向上
        return quality
    
    def calculate_review_thoroughness(self, dev_id: str) -> float:
        """レビュー徹底度を計算"""
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.6
        
        # コード変更量とレビュー数から徹底度を推定
        insertions = dev_data.get('total_insertions', 0)
        deletions = dev_data.get('total_deletions', 0)
        reviewed = dev_data.get('changes_reviewed', 0)
        
        if reviewed == 0:
            return 0.3
        
        # 変更量に対するレビュー数の比率
        code_volume = insertions + deletions
        if code_volume > 0:
            thoroughness = min(1.0, reviewed / (code_volume / 100.0))
        else:
            thoroughness = 0.6
        
        return thoroughness
    
    def calculate_response_speed(self, dev_id: str) -> float:
        """応答速度を計算"""
        # 簡易実装: 活動頻度から応答速度を推定
        frequency = self.calculate_activity_frequency(dev_id)
        
        # 頻度が高いほど応答が早いと推定
        speed = frequency * 0.8 + 0.2  # 0.2-1.0の範囲
        return speed
    
    def calculate_workload_variability(self, dev_id: str) -> float:
        """ワークロード変動性を計算"""
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.5
        
        authored = dev_data.get('changes_authored', 0)
        reviewed = dev_data.get('changes_reviewed', 0)
        projects = len(dev_data.get('projects', []))
        
        # プロジェクト数と活動量から変動性を推定
        if projects <= 1:
            variability = 0.2  # 単一プロジェクトは変動が少ない
        else:
            # 複数プロジェクトは変動が大きい
            activity_per_project = (authored + reviewed) / projects
            variability = min(1.0, 0.3 + projects * 0.1)
        
        return variability
    
    def calculate_acceptance_rate(self, dev_id: str) -> float:
        """受諾率を計算"""
        # 簡易実装: レビュー活動から受諾率を推定
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.7
        
        reviewed = dev_data.get('changes_reviewed', 0)
        
        # レビュー数が多いほど受諾率が高いと推定
        acceptance_rate = 0.5 + min(0.4, reviewed / 50.0)
        return acceptance_rate
    
    def calculate_positive_feedback_ratio(self, dev_id: str) -> float:
        """ポジティブフィードバック比率を計算"""
        # 簡易実装: レビュー品質から推定
        quality = self.calculate_avg_review_quality(dev_id)
        
        # 品質が高いほどポジティブフィードバックが多いと推定
        positive_ratio = quality * 0.8 + 0.2
        return positive_ratio
    
    def calculate_skill_diversity(self, dev_id: str) -> float:
        """スキル多様性を計算"""
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.4
        
        projects = dev_data.get('projects', [])
        insertions = dev_data.get('total_insertions', 0)
        deletions = dev_data.get('total_deletions', 0)
        
        # プロジェクト数とコード変更量からスキル多様性を推定
        project_diversity = min(1.0, len(projects) / 3.0)
        code_diversity = min(1.0, (insertions + deletions) / 1000.0)
        
        skill_diversity = (project_diversity + code_diversity) / 2.0
        return skill_diversity
    
    def calculate_learning_trajectory(self, dev_id: str) -> float:
        """学習軌跡を計算"""
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.5
        
        # 活動期間と成長を推定
        first_seen = dev_data.get('first_seen', '')
        if first_seen:
            try:
                first_date = self._parse_date(first_seen)
                days_active = (datetime.now() - first_date).days
                
                # 時間経過と活動量から学習軌跡を推定
                total_activity = dev_data.get('changes_authored', 0) + dev_data.get('changes_reviewed', 0)
                if days_active > 0:
                    learning_rate = min(1.0, total_activity / days_active * 10)
                    return learning_rate
            except:
                pass
        
        return 0.5
    
    def calculate_expertise_recognition(self, dev_id: str) -> float:
        """専門性認知を計算"""
        dev_data = self._get_developer_data(dev_id)
        if not dev_data:
            return 0.4
        
        reviewed = dev_data.get('changes_reviewed', 0)
        authored = dev_data.get('changes_authored', 0)
        
        # レビュー数が多いほど専門性が認知されていると推定
        recognition = min(1.0, reviewed / 30.0)  # 30レビュー以上で最大
        
        # 作成数も考慮
        if authored > 0:
            recognition += min(0.3, authored / 20.0)
        
        return min(1.0, recognition)
    
    def calculate_community_integration(self, dev_id: str) -> float:
        """コミュニティ統合度を計算"""
        diversity = self.calculate_collaboration_diversity(dev_id)
        mentoring = self.calculate_mentoring_activity(dev_id)
        
        # 協力多様性とメンタリング活動からコミュニティ統合度を推定
        integration = (diversity + mentoring) / 2.0
        return integration
    
    def calculate_leadership_indicators(self, dev_id: str) -> float:
        """リーダーシップ指標を計算"""
        mentoring = self.calculate_mentoring_activity(dev_id)
        recognition = self.calculate_expertise_recognition(dev_id)
        
        # メンタリング活動と専門性認知からリーダーシップを推定
        leadership = (mentoring * 0.6 + recognition * 0.4)
        return leadership
    
    def calculate_social_support(self, dev_id: str) -> float:
        """社会的支援レベルを計算"""
        integration = self.calculate_community_integration(dev_id)
        positive_feedback = self.calculate_positive_feedback_ratio(dev_id)
        
        # コミュニティ統合度とポジティブフィードバックから社会的支援を推定
        support = (integration + positive_feedback) / 2.0
        return support
    
    def statistical_significance_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """統計的有意性テスト"""
        results = {}
        
        # 継続者と非継続者の特徴比較
        retained = df[df['retention_label'] == True]
        churned = df[df['retention_label'] == False]
        
        if len(retained) == 0 or len(churned) == 0:
            return {"error": "insufficient_data"}
        
        feature_cols = [col for col in df.columns if col not in ['developer_id', 'retention_label']]
        
        significant_features = []
        
        for feature in feature_cols:
            try:
                retained_values = retained[feature].dropna()
                churned_values = churned[feature].dropna()
                
                if len(retained_values) > 0 and len(churned_values) > 0:
                    # t検定
                    t_stat, p_value = stats.ttest_ind(retained_values, churned_values)
                    
                    if p_value < 0.05:  # 有意水準5%
                        significant_features.append({
                            'feature': feature,
                            'p_value': p_value,
                            't_statistic': t_stat,
                            'retained_mean': retained_values.mean(),
                            'churned_mean': churned_values.mean(),
                            'effect_size': abs(retained_values.mean() - churned_values.mean()) / 
                                         np.sqrt((retained_values.var() + churned_values.var()) / 2)
                        })
            except Exception as e:
                continue
        
        results['significant_features'] = significant_features
        results['total_features_tested'] = len(feature_cols)
        results['significant_count'] = len(significant_features)
        
        return results
    
    def identify_top_factors(self, importance_results: Dict[str, Any]) -> Dict[str, Any]:
        """最重要要因を特定"""
        top_factors = {}
        
        # 各モデルからの重要度を統合
        all_importances = []
        
        for model_name, model_results in importance_results.items():
            if model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                importance_df = model_results.get('importance')
                if importance_df is not None:
                    for _, row in importance_df.head(10).iterrows():
                        all_importances.append({
                            'feature': row['feature'],
                            'importance': row['importance'],
                            'model': model_name
                        })
        
        # 特徴量ごとに重要度を集計
        feature_scores = defaultdict(list)
        for item in all_importances:
            feature_scores[item['feature']].append(item['importance'])
        
        # 平均重要度でランキング
        ranked_features = []
        for feature, scores in feature_scores.items():
            avg_importance = np.mean(scores)
            std_importance = np.std(scores) if len(scores) > 1 else 0
            ranked_features.append({
                'feature': feature,
                'avg_importance': avg_importance,
                'std_importance': std_importance,
                'model_count': len(scores)
            })
        
        ranked_features.sort(key=lambda x: x['avg_importance'], reverse=True)
        
        top_factors['ranking'] = ranked_features[:15]  # トップ15
        top_factors['top_3'] = ranked_features[:3]
        top_factors['summary'] = {
            'most_important': ranked_features[0]['feature'] if ranked_features else None,
            'avg_top_3_importance': np.mean([f['avg_importance'] for f in ranked_features[:3]]) if len(ranked_features) >= 3 else 0
        }
        
        return top_factors
    
    def compare_retained_vs_churned(self, df: pd.DataFrame) -> Dict[str, Any]:
        """継続者と離脱者を比較"""
        retained = df[df['retention_label'] == True]
        churned = df[df['retention_label'] == False]
        
        if len(retained) == 0 or len(churned) == 0:
            return {"error": "insufficient_data"}
        
        comparison = {
            'sample_sizes': {
                'retained': len(retained),
                'churned': len(churned),
                'retention_rate': len(retained) / len(df)
            }
        }
        
        feature_cols = [col for col in df.columns if col not in ['developer_id', 'retention_label']]
        
        feature_comparison = []
        for feature in feature_cols:
            retained_values = retained[feature].dropna()
            churned_values = churned[feature].dropna()
            
            if len(retained_values) > 0 and len(churned_values) > 0:
                comparison_item = {
                    'feature': feature,
                    'retained_mean': retained_values.mean(),
                    'churned_mean': churned_values.mean(),
                    'retained_std': retained_values.std(),
                    'churned_std': churned_values.std(),
                    'difference': retained_values.mean() - churned_values.mean(),
                    'relative_difference': (retained_values.mean() - churned_values.mean()) / churned_values.mean() if churned_values.mean() != 0 else 0
                }
                feature_comparison.append(comparison_item)
        
        # 差が大きい特徴量でソート
        feature_comparison.sort(key=lambda x: abs(x['relative_difference']), reverse=True)
        
        comparison['feature_comparison'] = feature_comparison
        comparison['top_differentiators'] = feature_comparison[:10]
        
        return comparison
    
    def find_optimal_clusters(self, X: np.ndarray) -> int:
        """最適クラスター数を発見"""
        if len(X) < 10:
            return min(3, len(X))
        
        silhouette_scores = []
        k_range = range(2, min(8, len(X)))
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                score = silhouette_score(X, cluster_labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(-1)
        
        if not silhouette_scores:
            return 3
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        return optimal_k
    
    def analyze_cluster_characteristics(self, cluster_data: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        """クラスター特徴を分析"""
        characteristics = {}
        
        for feature in feature_cols:
            if feature in cluster_data.columns:
                values = cluster_data[feature].dropna()
                if len(values) > 0:
                    characteristics[feature] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median(),
                        'percentile_75': values.quantile(0.75),
                        'percentile_25': values.quantile(0.25)
                    }
        
        return characteristics
    
    def _get_developer_data(self, dev_id: str) -> Optional[Dict]:
        """開発者データを取得"""
        for dev in self.developers_data:
            if dev.get('developer_id') == dev_id:
                return dev
        return None
    
    def _parse_date(self, date_str: str) -> datetime:
        """日付文字列を解析"""
        # ナノ秒の場合は切り詰める
        if date_str.count('.') == 1 and len(date_str.split('.')[-1]) > 6:
            date_str = date_str[:26]  # マイクロ秒まで
        
        formats = [
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse date: {date_str}")