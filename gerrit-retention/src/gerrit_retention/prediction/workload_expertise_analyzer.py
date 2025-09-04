"""
作業負荷・専門性一致率分析システム

開発者の作業負荷、専門性の一致率、ストレス指標を考慮した
高度な継続率予測システム
"""

import json
import logging
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

class WorkloadExpertiseAnalyzer:
    """作業負荷・専門性一致率分析システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        self.project_expertise_profiles = {}
        self.developer_expertise_profiles = {}
        self.workload_metrics = {}
        self.expertise_similarity_cache = {}
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('WorkloadExpertiseAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_workload_features(self, developer_data: Dict[str, Any]) -> Dict[str, float]:
        """作業負荷特徴量の分析"""
        features = {}
        
        # 基本活動量
        changes_authored = developer_data.get('changes_authored', 0)
        changes_reviewed = developer_data.get('changes_reviewed', 0)
        total_insertions = developer_data.get('total_insertions', 0)
        total_deletions = developer_data.get('total_deletions', 0)
        
        # 時間ベースの負荷計算
        try:
            first_seen = datetime.fromisoformat(
                developer_data.get('first_seen', '').replace(' ', 'T')
            )
            last_activity = datetime.fromisoformat(
                developer_data.get('last_activity', '').replace(' ', 'T')
            )
            
            activity_duration = (last_activity - first_seen).days
            if activity_duration > 0:
                # 日次作業負荷
                features['daily_changes_load'] = (changes_authored + changes_reviewed) / activity_duration
                features['daily_code_load'] = (total_insertions + total_deletions) / activity_duration
                
                # 作業強度（コード行数/チェンジ数）
                total_changes = changes_authored + changes_reviewed
                if total_changes > 0:
                    features['code_intensity'] = (total_insertions + total_deletions) / total_changes
                else:
                    features['code_intensity'] = 0.0
            else:
                features['daily_changes_load'] = 0.0
                features['daily_code_load'] = 0.0
                features['code_intensity'] = 0.0
        except:
            features['daily_changes_load'] = 0.0
            features['daily_code_load'] = 0.0
            features['code_intensity'] = 0.0
        
        # 作業バランス指標
        if changes_authored + changes_reviewed > 0:
            features['authoring_ratio'] = changes_authored / (changes_authored + changes_reviewed)
            features['reviewing_ratio'] = changes_reviewed / (changes_authored + changes_reviewed)
        else:
            features['authoring_ratio'] = 0.0
            features['reviewing_ratio'] = 0.0
        
        # 作業負荷レベルの分類
        total_workload = features['daily_changes_load'] + features['daily_code_load'] / 1000
        
        if total_workload > 5.0:
            features['workload_level'] = 4.0  # 過負荷
            features['workload_stress'] = 1.0
        elif total_workload > 2.0:
            features['workload_level'] = 3.0  # 高負荷
            features['workload_stress'] = 0.7
        elif total_workload > 0.5:
            features['workload_level'] = 2.0  # 中負荷
            features['workload_stress'] = 0.3
        elif total_workload > 0.1:
            features['workload_level'] = 1.0  # 低負荷
            features['workload_stress'] = 0.1
        else:
            features['workload_level'] = 0.0  # 無負荷
            features['workload_stress'] = 0.0
        
        # レビュー負荷ストレス
        review_scores = developer_data.get('review_scores', [])
        if review_scores:
            negative_reviews = sum(1 for score in review_scores if score < 0)
            review_stress_ratio = negative_reviews / len(review_scores)
            features['review_stress'] = review_stress_ratio
            
            # レビュー負荷の変動性（ストレス指標）
            features['review_volatility'] = np.std(review_scores) if len(review_scores) > 1 else 0.0
        else:
            features['review_stress'] = 0.0
            features['review_volatility'] = 0.0
        
        # プロジェクト分散負荷
        projects = developer_data.get('projects', [])
        if projects:
            features['project_fragmentation'] = len(projects)
            # プロジェクト間の負荷分散度（多すぎると分散負荷）
            if len(projects) > 5:
                features['fragmentation_stress'] = min(1.0, (len(projects) - 5) * 0.2)
            else:
                features['fragmentation_stress'] = 0.0
        else:
            features['project_fragmentation'] = 0.0
            features['fragmentation_stress'] = 0.0
        
        return features
    
    def analyze_expertise_match(self, developer_data: Dict[str, Any]) -> Dict[str, float]:
        """専門性一致率の分析"""
        features = {}
        
        # プロジェクト専門性の分析
        projects = developer_data.get('projects', [])
        developer_id = developer_data.get('developer_id', '')
        
        if not projects:
            return {
                'expertise_match_score': 0.0,
                'domain_consistency': 0.0,
                'skill_alignment': 0.0,
                'expertise_confidence': 0.0,
                'cross_domain_penalty': 0.0
            }
        
        # プロジェクト名からドメイン推定
        domain_keywords = self._extract_domain_keywords(projects)
        
        # ドメインの一貫性
        if domain_keywords:
            domain_counts = Counter(domain_keywords)
            total_domains = len(domain_keywords)
            max_domain_count = max(domain_counts.values())
            features['domain_consistency'] = max_domain_count / total_domains
            
            # 主要ドメインの特定
            primary_domain = domain_counts.most_common(1)[0][0]
            features['primary_domain_ratio'] = max_domain_count / total_domains
        else:
            features['domain_consistency'] = 0.0
            features['primary_domain_ratio'] = 0.0
        
        # 活動量と専門性の関係
        changes_authored = developer_data.get('changes_authored', 0)
        changes_reviewed = developer_data.get('changes_reviewed', 0)
        
        # 専門性スコアの計算
        expertise_indicators = []
        
        # 1. プロジェクト集中度（専門性の指標）
        if len(projects) <= 3:
            expertise_indicators.append(0.8)  # 高い専門性
        elif len(projects) <= 6:
            expertise_indicators.append(0.6)  # 中程度の専門性
        else:
            expertise_indicators.append(0.3)  # 分散した活動
        
        # 2. 活動の深さ（レビュー比率が高い = 専門性）
        total_activity = changes_authored + changes_reviewed
        if total_activity > 0:
            review_ratio = changes_reviewed / total_activity
            if review_ratio > 0.7:
                expertise_indicators.append(0.9)  # 高い専門性（レビュー中心）
            elif review_ratio > 0.3:
                expertise_indicators.append(0.7)  # バランス型
            else:
                expertise_indicators.append(0.5)  # 作成中心
        else:
            expertise_indicators.append(0.0)
        
        # 3. コード品質指標（専門性の間接指標）
        review_scores = developer_data.get('review_scores', [])
        if review_scores:
            avg_score = np.mean(review_scores)
            positive_ratio = sum(1 for s in review_scores if s > 0) / len(review_scores)
            quality_score = (avg_score + 2) / 4 * positive_ratio  # -2~2を0~1に正規化
            expertise_indicators.append(min(1.0, max(0.0, quality_score)))
        else:
            expertise_indicators.append(0.5)
        
        # 専門性マッチスコアの統合
        if expertise_indicators:
            features['expertise_match_score'] = np.mean(expertise_indicators)
            features['expertise_confidence'] = 1.0 - np.std(expertise_indicators)
        else:
            features['expertise_match_score'] = 0.0
            features['expertise_confidence'] = 0.0
        
        # スキルアライメント（活動パターンの一貫性）
        features['skill_alignment'] = self._calculate_skill_alignment(developer_data)
        
        # クロスドメインペナルティ
        unique_domains = len(set(domain_keywords)) if domain_keywords else 0
        if unique_domains > 3:
            features['cross_domain_penalty'] = min(1.0, (unique_domains - 3) * 0.2)
        else:
            features['cross_domain_penalty'] = 0.0
        
        return features
    
    def _extract_domain_keywords(self, projects: List[str]) -> List[str]:
        """プロジェクト名からドメインキーワードを抽出"""
        domain_keywords = []
        
        # ドメイン分類辞書
        domain_mapping = {
            'web': ['web', 'http', 'html', 'css', 'js', 'frontend', 'backend'],
            'database': ['db', 'sql', 'mysql', 'postgres', 'mongo', 'redis'],
            'cloud': ['aws', 'gcp', 'azure', 'cloud', 'docker', 'k8s', 'kubernetes'],
            'mobile': ['android', 'ios', 'mobile', 'app'],
            'ai_ml': ['ml', 'ai', 'tensorflow', 'pytorch', 'data', 'analytics'],
            'security': ['security', 'auth', 'crypto', 'ssl', 'tls'],
            'devops': ['ci', 'cd', 'deploy', 'build', 'test', 'jenkins'],
            'ui_ux': ['ui', 'ux', 'design', 'theme', 'style'],
            'api': ['api', 'rest', 'graphql', 'grpc'],
            'system': ['system', 'kernel', 'driver', 'os'],
            'network': ['network', 'tcp', 'udp', 'socket', 'protocol'],
            'office': ['office', 'document', 'word', 'excel', 'calc', 'writer'],
            'translation': ['l10n', 'i18n', 'translation', 'locale', 'lang'],
            'openstack': ['openstack', 'nova', 'neutron', 'cinder', 'keystone', 'glance', 'heat'],
            'libreoffice': ['libreoffice', 'core', 'writer', 'calc', 'impress', 'draw'],
            'chromium': ['chromium', 'chrome', 'blink', 'v8'],
            'wikimedia': ['wikimedia', 'wikipedia', 'mediawiki']
        }
        
        for project in projects:
            project_lower = project.lower()
            
            # 直接マッチング
            for domain, keywords in domain_mapping.items():
                if any(keyword in project_lower for keyword in keywords):
                    domain_keywords.append(domain)
                    break
            else:
                # パターンマッチング
                if 'openstack/' in project_lower:
                    domain_keywords.append('openstack')
                elif any(x in project_lower for x in ['core', 'main', 'base']):
                    domain_keywords.append('core')
                else:
                    domain_keywords.append('other')
        
        return domain_keywords
    
    def _calculate_skill_alignment(self, developer_data: Dict[str, Any]) -> float:
        """スキルアライメントの計算"""
        changes_authored = developer_data.get('changes_authored', 0)
        changes_reviewed = developer_data.get('changes_reviewed', 0)
        total_insertions = developer_data.get('total_insertions', 0)
        total_deletions = developer_data.get('total_deletions', 0)
        
        alignment_factors = []
        
        # 1. 作成/レビューバランス
        total_changes = changes_authored + changes_reviewed
        if total_changes > 0:
            balance_score = 1.0 - abs(0.5 - (changes_authored / total_changes))
            alignment_factors.append(balance_score)
        
        # 2. コード変更の一貫性
        if changes_authored > 0:
            avg_lines_per_change = (total_insertions + total_deletions) / changes_authored
            # 適度なサイズの変更が一貫性を示す
            if 10 <= avg_lines_per_change <= 500:
                consistency_score = 1.0
            elif avg_lines_per_change < 10:
                consistency_score = avg_lines_per_change / 10
            else:
                consistency_score = max(0.0, 1.0 - (avg_lines_per_change - 500) / 1000)
            alignment_factors.append(consistency_score)
        
        # 3. 挿入/削除バランス
        total_lines = total_insertions + total_deletions
        if total_lines > 0:
            insertion_ratio = total_insertions / total_lines
            # 適度な挿入/削除バランス
            balance_score = 1.0 - abs(0.6 - insertion_ratio)  # 60%挿入が理想的
            alignment_factors.append(balance_score)
        
        return np.mean(alignment_factors) if alignment_factors else 0.0
    
    def calculate_burnout_risk(self, developer_data: Dict[str, Any], 
                              workload_features: Dict[str, float],
                              expertise_features: Dict[str, float]) -> Dict[str, float]:
        """バーンアウトリスクの計算"""
        risk_factors = {}
        
        # 1. 作業負荷リスク
        workload_stress = workload_features.get('workload_stress', 0.0)
        review_stress = workload_features.get('review_stress', 0.0)
        fragmentation_stress = workload_features.get('fragmentation_stress', 0.0)
        
        workload_risk = (workload_stress * 0.4 + review_stress * 0.3 + 
                        fragmentation_stress * 0.3)
        risk_factors['workload_burnout_risk'] = workload_risk
        
        # 2. 専門性ミスマッチリスク
        expertise_match = expertise_features.get('expertise_match_score', 0.0)
        cross_domain_penalty = expertise_features.get('cross_domain_penalty', 0.0)
        
        mismatch_risk = (1.0 - expertise_match) * 0.7 + cross_domain_penalty * 0.3
        risk_factors['expertise_mismatch_risk'] = mismatch_risk
        
        # 3. 継続性リスク
        try:
            last_activity = datetime.fromisoformat(
                developer_data.get('last_activity', '').replace(' ', 'T')
            )
            days_since_last = (datetime.now() - last_activity).days
            
            if days_since_last > 90:
                continuity_risk = min(1.0, days_since_last / 365)
            else:
                continuity_risk = 0.0
        except:
            continuity_risk = 1.0
        
        risk_factors['continuity_risk'] = continuity_risk
        
        # 4. 総合バーンアウトリスク
        total_risk = (workload_risk * 0.35 + mismatch_risk * 0.35 + 
                     continuity_risk * 0.30)
        risk_factors['total_burnout_risk'] = total_risk
        
        # 5. リスクレベルの分類
        if total_risk > 0.7:
            risk_factors['burnout_level'] = 'critical'
            risk_factors['burnout_score'] = 4.0
        elif total_risk > 0.5:
            risk_factors['burnout_level'] = 'high'
            risk_factors['burnout_score'] = 3.0
        elif total_risk > 0.3:
            risk_factors['burnout_level'] = 'medium'
            risk_factors['burnout_score'] = 2.0
        elif total_risk > 0.1:
            risk_factors['burnout_level'] = 'low'
            risk_factors['burnout_score'] = 1.0
        else:
            risk_factors['burnout_level'] = 'minimal'
            risk_factors['burnout_score'] = 0.0
        
        return risk_factors
    
    def generate_workload_recommendations(self, developer_data: Dict[str, Any],
                                        workload_features: Dict[str, float],
                                        expertise_features: Dict[str, float],
                                        burnout_risk: Dict[str, float]) -> List[str]:
        """作業負荷ベースの推奨アクション"""
        recommendations = []
        
        # バーンアウトリスクベースの推奨
        burnout_level = burnout_risk.get('burnout_level', 'minimal')
        
        if burnout_level == 'critical':
            recommendations.append("🚨 クリティカルなバーンアウトリスクです。即座に作業負荷を軽減してください")
            recommendations.append("🛑 一時的な作業停止と休息を強く推奨します")
        elif burnout_level == 'high':
            recommendations.append("⚠️ 高いバーンアウトリスクです。作業負荷の見直しが必要です")
            recommendations.append("📉 作業量を30-50%削減することを検討してください")
        elif burnout_level == 'medium':
            recommendations.append("⚖️ 中程度のバーンアウトリスクです。作業バランスの調整を推奨します")
        
        # 作業負荷固有の推奨
        workload_stress = workload_features.get('workload_stress', 0.0)
        if workload_stress > 0.7:
            recommendations.append("📊 作業負荷が過大です。タスクの優先順位付けと分散を検討してください")
        
        review_stress = workload_features.get('review_stress', 0.0)
        if review_stress > 0.5:
            recommendations.append("📝 レビューでのネガティブフィードバックが多いです。メンタリングサポートを提供してください")
        
        fragmentation_stress = workload_features.get('fragmentation_stress', 0.0)
        if fragmentation_stress > 0.3:
            recommendations.append("🔄 プロジェクトが分散しすぎています。フォーカスするプロジェクトを絞ってください")
        
        # 専門性ミスマッチの推奨
        expertise_match = expertise_features.get('expertise_match_score', 0.0)
        if expertise_match < 0.4:
            recommendations.append("🎯 専門性とタスクのミスマッチがあります。スキルに合った作業の割り当てを検討してください")
            recommendations.append("📚 必要なスキル習得のための研修機会を提供してください")
        
        cross_domain_penalty = expertise_features.get('cross_domain_penalty', 0.0)
        if cross_domain_penalty > 0.3:
            recommendations.append("🌐 複数ドメインにまたがる作業が負担になっています。専門領域への集中を推奨します")
        
        # ポジティブな推奨
        if burnout_level in ['minimal', 'low']:
            if expertise_match > 0.7:
                recommendations.append("✨ 専門性と作業が良くマッチしています。現在の方向性を継続してください")
            
            workload_level = workload_features.get('workload_level', 0.0)
            if workload_level == 2.0:  # 適度な負荷
                recommendations.append("⚖️ 適度な作業負荷を維持しています。このバランスを保ってください")
        
        # 成長機会の推奨
        skill_alignment = expertise_features.get('skill_alignment', 0.0)
        if skill_alignment > 0.6:
            recommendations.append("🚀 スキルアライメントが良好です。より挑戦的なタスクへのステップアップを検討してください")
        
        return recommendations
    
    def analyze_comprehensive_features(self, developer_data: Dict[str, Any]) -> Dict[str, float]:
        """包括的な特徴量分析"""
        # 既存の特徴量
        from .advanced_accuracy_improver import AdvancedAccuracyImprover
        config = {'output_path': 'temp'}
        improver = AdvancedAccuracyImprover(config)
        base_features = improver.extract_advanced_features(developer_data)
        
        # 作業負荷特徴量
        workload_features = self.analyze_workload_features(developer_data)
        
        # 専門性特徴量
        expertise_features = self.analyze_expertise_match(developer_data)
        
        # バーンアウトリスク
        burnout_risk = self.calculate_burnout_risk(developer_data, workload_features, expertise_features)
        
        # 全特徴量の統合
        comprehensive_features = {}
        comprehensive_features.update(base_features)
        comprehensive_features.update(workload_features)
        comprehensive_features.update(expertise_features)
        comprehensive_features.update(burnout_risk)
        
        return comprehensive_features
    
    def generate_comprehensive_analysis(self, developer_data: Dict[str, Any]) -> Dict[str, Any]:
        """包括的な開発者分析"""
        # 特徴量の抽出
        workload_features = self.analyze_workload_features(developer_data)
        expertise_features = self.analyze_expertise_match(developer_data)
        burnout_risk = self.calculate_burnout_risk(developer_data, workload_features, expertise_features)
        
        # 推奨アクションの生成
        recommendations = self.generate_workload_recommendations(
            developer_data, workload_features, expertise_features, burnout_risk
        )
        
        # 総合スコアの計算
        workload_score = 1.0 - workload_features.get('workload_stress', 0.0)
        expertise_score = expertise_features.get('expertise_match_score', 0.0)
        burnout_score = 1.0 - burnout_risk.get('total_burnout_risk', 0.0)
        
        overall_wellness_score = (workload_score * 0.4 + expertise_score * 0.3 + 
                                burnout_score * 0.3)
        
        return {
            'workload_analysis': workload_features,
            'expertise_analysis': expertise_features,
            'burnout_risk': burnout_risk,
            'recommendations': recommendations,
            'wellness_scores': {
                'workload_score': workload_score,
                'expertise_score': expertise_score,
                'burnout_score': burnout_score,
                'overall_wellness': overall_wellness_score
            }
        }

def main():
    """メイン実行関数"""
    config = {
        'output_path': 'outputs/workload_expertise_analysis'
    }
    
    analyzer = WorkloadExpertiseAnalyzer(config)
    
    print("🔧 作業負荷・専門性分析システムを初期化しました")
    return analyzer

if __name__ == "__main__":
    main()