"""
予測モデルモジュール

開発者定着予測、ストレス分析、沸点予測を担当するモジュールです。

主要コンポーネント:
- retention_predictor: 定着予測器
- retention_factor_analyzer: 定着要因分析器
- stress_analyzer: ストレス分析器
- boiling_point_predictor: 沸点予測器
"""

try:
    from .boiling_point_predictor import (
        BoilingPointPrediction,
        BoilingPointPredictor,
        DeveloperExitPattern,
    )
    from .retention_factor_analyzer import RetentionFactorAnalyzer
    from .retention_predictor import RetentionFeatureExtractor, RetentionPredictor
    from .stress_analyzer import StressAnalyzer, StressIndicators
    from .stress_mitigation_advisor import (
        DeveloperMatchingProposal,
        ImplementationDifficulty,
        MitigationCategory,
        MitigationProposal,
        StressMitigationAdvisor,
    )
    
    __all__ = [
        'RetentionPredictor',
        'RetentionFeatureExtractor', 
        'RetentionFactorAnalyzer',
        'StressAnalyzer',
        'StressIndicators',
        'BoilingPointPredictor',
        'BoilingPointPrediction',
        'DeveloperExitPattern',
        'StressMitigationAdvisor',
        'MitigationProposal',
        'DeveloperMatchingProposal',
        'MitigationCategory',
        'ImplementationDifficulty'
    ]
    
except ImportError as e:
    # 依存関係が不足している場合のフォールバック
    __all__ = []
    import logging
    logging.warning(f"予測モジュールの一部をインポートできませんでした: {e}")

# モジュールレベルでの設定
try:
    from gerrit_retention.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.debug("予測モデルモジュールが読み込まれました")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("予測モデルモジュールが読み込まれました（基本ロガー使用）")