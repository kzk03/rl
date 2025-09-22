"""
特徴量統合・正規化・選択モジュール

開発者、レビュー、時系列特徴量を統合し、正規化・選択機能を提供する。
機械学習モデル用の特徴量ベクトルを生成する。
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .developer_features import DeveloperFeatureExtractor, DeveloperFeatures
from .review_features import ReviewFeatureExtractor, ReviewFeatures
from .temporal_features import TemporalFeatureExtractor, TemporalFeatures

logger = logging.getLogger(__name__)


@dataclass
class IntegratedFeatures:
    """統合特徴量データクラス"""
    developer_email: str
    change_id: Optional[str]
    context_date: datetime
    
    # 特徴量ベクトル
    feature_vector: np.ndarray
    feature_names: List[str]
    
    # 元の特徴量オブジェクト
    developer_features: Optional[DeveloperFeatures]
    review_features: Optional[ReviewFeatures]
    temporal_features: Optional[TemporalFeatures]
    
    # メタデータ
    feature_categories: Dict[str, List[int]]  # カテゴリ別特徴量インデックス
    normalization_info: Dict[str, Any]


class FeatureIntegrator:
    """特徴量統合器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # 特徴量抽出器の初期化
        self.developer_extractor = DeveloperFeatureExtractor(
            config.get('developer_features', {})
        )
        self.review_extractor = ReviewFeatureExtractor(
            config.get('review_features', {})
        )
        self.temporal_extractor = TemporalFeatureExtractor(
            config.get('temporal_features', {})
        )
        
        # 正規化・選択の設定
        self.normalization_method = config.get('normalization_method', 'standard')
        self.feature_selection_method = config.get('feature_selection_method', 'k_best')
        self.max_features = config.get('max_features', 100)
        self.enable_pca = config.get('enable_pca', False)
        self.pca_components = config.get('pca_components', 50)
        
        # 学習済みの変換器
        self.scaler = None
        self.feature_selector = None
        self.pca_transformer = None
        self.feature_names = None
        self.feature_categories = None
        
    def extract_integrated_features(self, 
                                   developer_email: str,
                                   changes_data: List[Dict[str, Any]],
                                   reviews_data: List[Dict[str, Any]],
                                   context_date: datetime,
                                   change_id: Optional[str] = None) -> IntegratedFeatures:
        """
        統合特徴量を抽出
        
        Args:
            developer_email: 開発者のメールアドレス
            changes_data: Change データのリスト
            reviews_data: Review データのリスト
            context_date: 基準日時
            change_id: 特定のChange ID（レビュー特徴量用）
            
        Returns:
            IntegratedFeatures: 統合された特徴量
        """
        logger.info(f"統合特徴量を抽出中: {developer_email}")
        
        # 各タイプの特徴量を抽出
        developer_features = self.developer_extractor.extract_features(
            developer_email, changes_data, reviews_data, context_date
        )
        
        review_features = None
        if change_id:
            # 特定のChangeの特徴量を抽出
            change_data = next(
                (c for c in changes_data if c.get('change_id') == change_id), 
                None
            )
            if change_data:
                review_features = self.review_extractor.extract_features(
                    change_data, reviews_data, context_date
                )
        
        temporal_features = self.temporal_extractor.extract_features(
            developer_email, changes_data, reviews_data, context_date
        )
        
        # 特徴量ベクトルを構築
        feature_vector, feature_names, feature_categories = self._build_feature_vector(
            developer_features, review_features, temporal_features
        )
        
        return IntegratedFeatures(
            developer_email=developer_email,
            change_id=change_id,
            context_date=context_date,
            feature_vector=feature_vector,
            feature_names=feature_names,
            developer_features=developer_features,
            review_features=review_features,
            temporal_features=temporal_features,
            feature_categories=feature_categories,
            normalization_info={}
        )
    
    def _build_feature_vector(self, 
                            developer_features: DeveloperFeatures,
                            review_features: Optional[ReviewFeatures],
                            temporal_features: TemporalFeatures) -> Tuple[np.ndarray, List[str], Dict[str, List[int]]]:
        """特徴量ベクトルを構築"""
        
        features = []
        feature_names = []
        feature_categories = {
            'developer': [],
            'review': [],
            'temporal': []
        }
        
        current_index = 0
        
        # 開発者特徴量
        dev_features, dev_names = self._extract_developer_feature_vector(developer_features)
        features.extend(dev_features)
        feature_names.extend([f"dev_{name}" for name in dev_names])
        feature_categories['developer'] = list(range(current_index, current_index + len(dev_features)))
        current_index += len(dev_features)
        
        # レビュー特徴量（存在する場合）
        if review_features:
            rev_features, rev_names = self._extract_review_feature_vector(review_features)
            features.extend(rev_features)
            feature_names.extend([f"rev_{name}" for name in rev_names])
            feature_categories['review'] = list(range(current_index, current_index + len(rev_features)))
            current_index += len(rev_features)
        
        # 時系列特徴量
        temp_features, temp_names = self._extract_temporal_feature_vector(temporal_features)
        features.extend(temp_features)
        feature_names.extend([f"temp_{name}" for name in temp_names])
        feature_categories['temporal'] = list(range(current_index, current_index + len(temp_features)))
        
        return np.array(features), feature_names, feature_categories
    
    def _extract_developer_feature_vector(self, 
                                        features: DeveloperFeatures) -> Tuple[List[float], List[str]]:
        """開発者特徴量からベクトルを抽出"""
        
        vector = []
        names = []
        
        # 基本特徴量
        vector.extend([
            features.expertise_level,
            features.activity_frequency,
            features.activity_consistency,
            features.collaboration_network_size,
            features.collaboration_quality,
            features.mentoring_activity,
            features.cross_team_collaboration
        ])
        names.extend([
            'expertise_level', 'activity_frequency', 'activity_consistency',
            'collaboration_network_size', 'collaboration_quality',
            'mentoring_activity', 'cross_team_collaboration'
        ])
        
        # Gerrit特有指標
        vector.extend([
            features.avg_review_score_given,
            features.avg_review_score_received,
            features.review_response_time_avg,
            features.code_review_thoroughness,
            features.change_approval_rate,
            features.review_acceptance_rate
        ])
        names.extend([
            'avg_review_score_given', 'avg_review_score_received',
            'review_response_time_avg', 'code_review_thoroughness',
            'change_approval_rate', 'review_acceptance_rate'
        ])
        
        # 時系列特徴量
        vector.extend([
            features.recent_activity_trend,
            features.expertise_growth_rate,
            features.stress_level_estimate
        ])
        names.extend([
            'recent_activity_trend', 'expertise_growth_rate', 'stress_level_estimate'
        ])
        
        # ピーク活動時間（上位3つ）
        peak_hours = features.peak_activity_hours + [0] * (3 - len(features.peak_activity_hours))
        vector.extend(peak_hours[:3])
        names.extend(['peak_hour_1', 'peak_hour_2', 'peak_hour_3'])
        
        # 好みの曜日（上位3つ）
        preferred_days = features.preferred_days + [0] * (3 - len(features.preferred_days))
        vector.extend(preferred_days[:3])
        names.extend(['preferred_day_1', 'preferred_day_2', 'preferred_day_3'])
        
        # 技術領域の多様性（上位5つ）
        domain_values = list(features.technical_domains.values())
        domain_values.sort(reverse=True)
        domain_features = domain_values[:5] + [0.0] * (5 - len(domain_values[:5]))
        vector.extend(domain_features)
        names.extend([f'tech_domain_{i+1}' for i in range(5)])
        
        return vector, names
    
    def _extract_review_feature_vector(self, 
                                     features: ReviewFeatures) -> Tuple[List[float], List[str]]:
        """レビュー特徴量からベクトルを抽出"""
        
        vector = []
        names = []
        
        # 基本Change特徴量
        vector.extend([
            features.change_size,
            features.files_changed_count,
            features.lines_added,
            features.lines_deleted,
            features.change_complexity,
            features.directory_depth
        ])
        names.extend([
            'change_size', 'files_changed_count', 'lines_added',
            'lines_deleted', 'change_complexity', 'directory_depth'
        ])
        
        # レビュー特徴量
        vector.extend([
            features.avg_review_score,
            features.review_consensus,
            features.review_count,
            features.review_thoroughness,
            features.review_response_time_avg,
            features.review_effort_total,
            features.constructive_feedback_ratio
        ])
        names.extend([
            'avg_review_score', 'review_consensus', 'review_count',
            'review_thoroughness', 'review_response_time_avg',
            'review_effort_total', 'constructive_feedback_ratio'
        ])
        
        # Change特性
        vector.extend([
            features.change_urgency,
            features.change_risk_level,
            features.reviewer_diversity,
            1.0 if features.cross_team_review else 0.0,
            features.expert_involvement
        ])
        names.extend([
            'change_urgency', 'change_risk_level', 'reviewer_diversity',
            'cross_team_review', 'expert_involvement'
        ])
        
        # Changeタイプ（ワンホットエンコーディング）
        change_types = ['feature', 'bugfix', 'refactor', 'docs', 'test', 'other']
        for change_type in change_types:
            vector.append(1.0 if features.change_type == change_type else 0.0)
            names.append(f'change_type_{change_type}')
        
        # Changeスコープ（ワンホットエンコーディング）
        change_scopes = ['local', 'module', 'system', 'unknown']
        for scope in change_scopes:
            vector.append(1.0 if features.change_scope == scope else 0.0)
            names.append(f'change_scope_{scope}')
        
        # プログラミング言語（上位5つ）
        languages = ['Python', 'Java', 'JavaScript', 'C++', 'Go']
        for lang in languages:
            vector.append(1.0 if lang in features.programming_languages else 0.0)
            names.append(f'lang_{lang.lower()}')
        
        return vector, names
    
    def _extract_temporal_feature_vector(self, 
                                       features: TemporalFeatures) -> Tuple[List[float], List[str]]:
        """時系列特徴量からベクトルを抽出"""
        
        vector = []
        names = []
        
        # 基本時系列特徴量
        vector.extend([
            features.activity_trend,
            features.activity_volatility,
            features.activity_momentum,
            features.short_term_trend,
            features.medium_term_trend,
            features.long_term_trend,
            features.trend_stability,
            features.pattern_consistency
        ])
        names.extend([
            'activity_trend', 'activity_volatility', 'activity_momentum',
            'short_term_trend', 'medium_term_trend', 'long_term_trend',
            'trend_stability', 'pattern_consistency'
        ])
        
        # 予測特徴量
        vector.extend([
            features.activity_forecast,
            features.engagement_forecast,
            features.retention_risk_trend
        ])
        names.extend([
            'activity_forecast', 'engagement_forecast', 'retention_risk_trend'
        ])
        
        # 整合性特徴量
        vector.extend([
            features.temporal_consistency_score,
            features.data_leakage_risk
        ])
        names.extend([
            'temporal_consistency_score', 'data_leakage_risk'
        ])
        
        # 週次パターン（7次元）
        weekly_pattern = features.weekly_pattern + [0.0] * (7 - len(features.weekly_pattern))
        vector.extend(weekly_pattern[:7])
        names.extend([f'weekly_pattern_{i}' for i in range(7)])
        
        # 日次パターン（24時間を6つの時間帯に集約）
        daily_pattern = features.daily_pattern + [0.0] * (24 - len(features.daily_pattern))
        time_slots = []
        for i in range(0, 24, 4):  # 4時間ごとの時間帯
            slot_avg = np.mean(daily_pattern[i:i+4])
            time_slots.append(slot_avg)
        vector.extend(time_slots)
        names.extend([f'time_slot_{i}' for i in range(6)])
        
        # 季節性特徴量（主要な指標のみ）
        seasonality = features.activity_seasonality
        vector.extend([
            seasonality.get('weekday_0', 0.0),  # 月曜日
            seasonality.get('weekday_4', 0.0),  # 金曜日
            seasonality.get('weekday_5', 0.0),  # 土曜日
            seasonality.get('weekday_6', 0.0),  # 日曜日
            seasonality.get('hour_9', 0.0),     # 午前9時
            seasonality.get('hour_14', 0.0),    # 午後2時
            seasonality.get('hour_21', 0.0)     # 午後9時
        ])
        names.extend([
            'monday_activity', 'friday_activity', 'saturday_activity', 'sunday_activity',
            'morning_activity', 'afternoon_activity', 'evening_activity'
        ])
        
        return vector, names
    
    def fit_transformers(self, 
                        integrated_features_list: List[IntegratedFeatures],
                        labels: Optional[List[float]] = None) -> None:
        """変換器を学習"""
        
        logger.info("特徴量変換器を学習中...")
        
        # 特徴量行列を構築
        feature_matrix = np.array([f.feature_vector for f in integrated_features_list])
        
        if len(integrated_features_list) > 0:
            self.feature_names = integrated_features_list[0].feature_names
            self.feature_categories = integrated_features_list[0].feature_categories
        
        # 正規化器の学習
        if self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.normalization_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        if self.scaler:
            self.scaler.fit(feature_matrix)
            feature_matrix = self.scaler.transform(feature_matrix)
        
        # 特徴量選択器の学習
        if labels is not None and self.feature_selection_method != 'none':
            if self.feature_selection_method == 'k_best':
                self.feature_selector = SelectKBest(
                    score_func=f_classif, 
                    k=min(self.max_features, feature_matrix.shape[1])
                )
            elif self.feature_selection_method == 'mutual_info':
                self.feature_selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(self.max_features, feature_matrix.shape[1])
                )
            
            if self.feature_selector:
                self.feature_selector.fit(feature_matrix, labels)
                feature_matrix = self.feature_selector.transform(feature_matrix)
                
                # 選択された特徴量名を更新
                selected_indices = self.feature_selector.get_support(indices=True)
                self.feature_names = [self.feature_names[i] for i in selected_indices]
        
        # PCA変換器の学習
        if self.enable_pca and feature_matrix.shape[1] > self.pca_components:
            self.pca_transformer = PCA(n_components=self.pca_components)
            self.pca_transformer.fit(feature_matrix)
            
            # PCA後の特徴量名を生成
            self.feature_names = [f'pca_{i}' for i in range(self.pca_components)]
        
        logger.info(f"変換器学習完了: 最終特徴量数={len(self.feature_names)}")
    
    def transform_features(self, 
                          integrated_features: IntegratedFeatures) -> np.ndarray:
        """特徴量を変換"""
        
        feature_vector = integrated_features.feature_vector.reshape(1, -1)
        
        # 正規化
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)
        
        # 特徴量選択
        if self.feature_selector:
            feature_vector = self.feature_selector.transform(feature_vector)
        
        # PCA変換
        if self.pca_transformer:
            feature_vector = self.pca_transformer.transform(feature_vector)
        
        return feature_vector.flatten()
    
    def save_transformers(self, filepath: str) -> None:
        """変換器を保存"""
        
        transformers = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'pca_transformer': self.pca_transformer,
            'feature_names': self.feature_names,
            'feature_categories': self.feature_categories,
            'config': self.config
        }
        
        joblib.dump(transformers, filepath)
        logger.info(f"変換器を保存しました: {filepath}")
    
    def load_transformers(self, filepath: str) -> None:
        """変換器を読み込み"""
        
        transformers = joblib.load(filepath)
        
        self.scaler = transformers['scaler']
        self.feature_selector = transformers['feature_selector']
        self.pca_transformer = transformers['pca_transformer']
        self.feature_names = transformers['feature_names']
        self.feature_categories = transformers['feature_categories']
        
        logger.info(f"変換器を読み込みました: {filepath}")


def create_feature_integrator(config_path: str) -> FeatureIntegrator:
    """
    設定ファイルから特徴量統合器を作成
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        FeatureIntegrator: 設定済みの統合器
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return FeatureIntegrator(config)


def batch_extract_integrated_features(developers: List[str],
                                     changes_data: List[Dict[str, Any]],
                                     reviews_data: List[Dict[str, Any]],
                                     integrator: FeatureIntegrator,
                                     context_date: datetime) -> List[IntegratedFeatures]:
    """
    複数の開発者に対してバッチで統合特徴量を抽出
    
    Args:
        developers: 開発者のメールアドレスリスト
        changes_data: Change データのリスト
        reviews_data: Review データのリスト
        integrator: 特徴量統合器
        context_date: 基準日時
        
    Returns:
        List[IntegratedFeatures]: 抽出された統合特徴量のリスト
    """
    features_list = []
    
    for developer_email in developers:
        try:
            features = integrator.extract_integrated_features(
                developer_email, changes_data, reviews_data, context_date
            )
            features_list.append(features)
        except Exception as e:
            logger.error(f"統合特徴量抽出エラー (開発者: {developer_email}): {e}")
            continue
    
    return features_list


if __name__ == "__main__":
    # テスト用のサンプルデータ
    sample_changes = [
        {
            'change_id': 'change1',
            'author': 'dev1@example.com',
            'created': '2023-01-15T10:00:00',
            'subject': 'Fix authentication bug',
            'project': 'project-a',
            'technical_domain': 'security',
            'files_changed': ['src/auth.py', 'tests/test_auth.py'],
            'lines_added': 50,
            'lines_deleted': 20
        }
    ]
    
    sample_reviews = [
        {
            'change_id': 'change1',
            'reviewer_email': 'dev2@example.com',
            'timestamp': '2023-01-15T14:00:00',
            'score': 2,
            'message': 'Excellent fix!',
            'response_time_hours': 4.0,
            'review_effort_estimated': 1.5
        }
    ]
    
    # 統合器のテスト
    config = {
        'developer_features': {
            'time_window_days': 90,
            'expertise_threshold': 0.1
        },
        'review_features': {
            'complexity_weights': {
                'file_count': 0.3,
                'line_changes': 0.4,
                'directory_spread': 0.2,
                'language_diversity': 0.1
            }
        },
        'temporal_features': {
            'short_term_window_days': 7,
            'medium_term_window_days': 30,
            'long_term_window_days': 90
        },
        'normalization_method': 'standard',
        'feature_selection_method': 'k_best',
        'max_features': 50
    }
    
    integrator = FeatureIntegrator(config)
    
    # 統合特徴量の抽出
    integrated_features = integrator.extract_integrated_features(
        'dev1@example.com',
        sample_changes,
        sample_reviews,
        datetime(2023, 2, 1),
        'change1'
    )
    
    print(f"統合特徴量抽出完了: {integrated_features.developer_email}")
    print(f"特徴量数: {len(integrated_features.feature_vector)}")
    print(f"特徴量名（最初の10個）: {integrated_features.feature_names[:10]}")
    print(f"開発者特徴量インデックス: {integrated_features.feature_categories['developer'][:5]}")
    print(f"レビュー特徴量インデックス: {integrated_features.feature_categories['review'][:5]}")
    print(f"時系列特徴量インデックス: {integrated_features.feature_categories['temporal'][:5]}")