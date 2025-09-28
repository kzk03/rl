"""
A/Bテストモジュール

このモジュールは、異なる推薦戦略の比較実験を設計・実行し、
統計的分析を行うためのツールを提供する。
"""

from .experiment_design import (
    AllocationMethod,
    ExperimentConfig,
    ExperimentDesigner,
    ExperimentMetric,
    ExperimentStatus,
    ExperimentVariant,
    ParticipantAllocator,
)
from .statistical_analysis import (
    ExperimentAnalysisResult,
    StatisticalAnalyzer,
    StatisticalTestResult,
    TestResult,
    VariantPerformance,
)

__all__ = [
    # 実験設計
    'ExperimentDesigner',
    'ParticipantAllocator',
    'ExperimentConfig',
    'ExperimentVariant',
    'ExperimentMetric',
    'ExperimentStatus',
    'AllocationMethod',
    
    # 統計分析
    'StatisticalAnalyzer',
    'StatisticalTestResult',
    'VariantPerformance',
    'ExperimentAnalysisResult',
    'TestResult'
]