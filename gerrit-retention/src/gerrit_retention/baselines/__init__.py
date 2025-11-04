"""
Baseline models for developer retention prediction.

This package provides traditional machine learning baselines for comparison
with the IRL+LSTM model.

Available baselines:
- LogisticRegressionBaseline: Linear model with interpretable coefficients
- RandomForestBaseline: Ensemble learning with decision trees

Usage:
    from gerrit_retention.baselines import LogisticRegressionBaseline, RandomForestBaseline

    # Initialize baseline
    lr = LogisticRegressionBaseline()

    # Train
    lr.train({'features': X_train, 'labels': y_train})

    # Predict
    predictions = lr.predict({'features': X_test})

    # Get feature importance
    importance = lr.get_feature_importance()
"""

from .base_baseline import BaseBaseline
from .logistic_regression import LogisticRegressionBaseline
from .random_forest import RandomForestBaseline
from .utils import (
    extract_static_features,
    evaluate_predictions,
    save_results,
    format_duration
)

__all__ = [
    'BaseBaseline',
    'LogisticRegressionBaseline',
    'RandomForestBaseline',
    'extract_static_features',
    'evaluate_predictions',
    'save_results',
    'format_duration'
]

__version__ = '0.1.0'
