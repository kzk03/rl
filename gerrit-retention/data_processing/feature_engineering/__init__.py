"""
特徴量エンジニアリングモジュール

Gerrit特化の開発者定着予測システム用特徴量抽出機能を提供する。
開発者特徴量、レビュー特徴量、時系列特徴量の抽出と正規化・選択機能を含む。
"""

from .developer_features import (
    DeveloperFeatureExtractor,
    DeveloperFeatures,
    create_developer_feature_extractor,
)
from .feature_integration import (
    FeatureIntegrator,
    IntegratedFeatures,
    batch_extract_integrated_features,
    create_feature_integrator,
)
from .review_features import (
    ReviewFeatureExtractor,
    ReviewFeatures,
    batch_extract_review_features,
    create_review_feature_extractor,
)
from .temporal_features import (
    TemporalFeatureExtractor,
    TemporalFeatures,
    create_temporal_feature_extractor,
    validate_temporal_data_integrity,
)

__all__ = [
    # 開発者特徴量
    'DeveloperFeatures',
    'DeveloperFeatureExtractor',
    'create_developer_feature_extractor',
    
    # レビュー特徴量
    'ReviewFeatures',
    'ReviewFeatureExtractor',
    'create_review_feature_extractor',
    'batch_extract_review_features',
    
    # 時系列特徴量
    'TemporalFeatures',
    'TemporalFeatureExtractor',
    'create_temporal_feature_extractor',
    'validate_temporal_data_integrity',
    
    # 統合特徴量
    'IntegratedFeatures',
    'FeatureIntegrator',
    'create_feature_integrator',
    'batch_extract_integrated_features'
]

__version__ = '1.0.0'