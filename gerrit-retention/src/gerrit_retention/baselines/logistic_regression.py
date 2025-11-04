"""
Logistic Regression baseline for developer retention prediction.

This module implements a logistic regression baseline that aggregates time-series
features into static features and uses traditional logistic regression for prediction.
"""

import numpy as np
import time
import pickle
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import StandardScaler

from .base_baseline import BaseBaseline


class LogisticRegressionBaseline(BaseBaseline):
    """
    Logistic Regression baseline model.

    This model aggregates time-series activity history into static features
    and uses logistic regression for binary classification.

    Key features:
    - Handles class imbalance with class_weight='balanced'
    - Standardizes features with StandardScaler
    - Provides interpretable coefficients as feature importance
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Logistic Regression baseline.

        Args:
            config: Configuration dictionary with optional keys:
                - 'max_iter': Maximum iterations (default: 1000)
                - 'C': Regularization parameter (default: 1.0)
                - 'solver': Solver to use (default: 'lbfgs')
                - 'class_weight': Class weight strategy (default: 'balanced')
                - 'random_state': Random seed (default: 42)
        """
        super().__init__(config)

        # Extract hyperparameters
        max_iter = self.config.get('max_iter', 1000)
        C = self.config.get('C', 1.0)
        solver = self.config.get('solver', 'lbfgs')
        class_weight = self.config.get('class_weight', 'balanced')
        random_state = self.config.get('random_state', 42)

        # Initialize model
        self.model = SklearnLR(
            max_iter=max_iter,
            C=C,
            solver=solver,
            class_weight=class_weight,
            random_state=random_state
        )

        # Feature scaler
        self.scaler = StandardScaler()

    def train(self, train_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train logistic regression model.

        Args:
            train_data: Dictionary containing:
                - 'features': numpy array of shape [n_samples, n_features]
                - 'labels': numpy array of shape [n_samples]
                - 'feature_names': list of feature names (optional)

        Returns:
            Dictionary with training metrics
        """
        # Validate input data
        self.validate_data(train_data, require_labels=True)

        X_train = train_data['features']
        y_train = train_data['labels']

        # Store feature names
        if 'feature_names' in train_data:
            self.feature_names = train_data['feature_names']
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Train model
        start_time = time.time()

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit logistic regression
        self.model.fit(X_train_scaled, y_train)

        training_time = time.time() - start_time

        self.is_trained = True

        # Return training info
        return {
            'training_time': training_time,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'positive_rate': float(np.mean(y_train)),
            'model_type': 'logistic_regression'
        }

    def predict(self, test_data: Dict[str, Any]) -> np.ndarray:
        """
        Predict continuation probabilities.

        Args:
            test_data: Dictionary containing:
                - 'features': numpy array of shape [n_samples, n_features]

        Returns:
            Numpy array of continuation probabilities, shape [n_samples]
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Validate input
        self.validate_data(test_data, require_labels=False)

        X_test = test_data['features']

        # Standardize features
        X_test_scaled = self.scaler.transform(X_test)

        # Predict probabilities (probability of class 1)
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]

        return probabilities

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from logistic regression coefficients.

        Returns:
            Dictionary mapping feature names to absolute coefficient values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")

        # Get coefficients (absolute values as importance)
        coefficients = np.abs(self.model.coef_[0])

        # Map to feature names
        importance_dict = {
            name: float(coef)
            for name, coef in zip(self.feature_names, coefficients)
        }

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict

    def get_raw_coefficients(self) -> Dict[str, float]:
        """
        Get raw (signed) coefficients from logistic regression.

        Positive coefficient = increases continuation probability
        Negative coefficient = decreases continuation probability

        Returns:
            Dictionary mapping feature names to raw coefficient values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting coefficients")

        coefficients = self.model.coef_[0]

        coef_dict = {
            name: float(coef)
            for name, coef in zip(self.feature_names, coefficients)
        }

        # Sort by absolute value
        coef_dict = dict(
            sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return coef_dict

    def save_model(self, path: str) -> None:
        """
        Save model to disk using pickle.

        Args:
            path: File path to save the model (e.g., 'model.pkl')
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_trained': self.is_trained
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: File path to load the model from
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']

        print(f"Model loaded from {path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get detailed model summary.

        Returns:
            Dictionary with model details including intercept, top coefficients, etc.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting summary")

        # Get top positive and negative coefficients
        coef_dict = self.get_raw_coefficients()
        coef_items = list(coef_dict.items())

        top_positive = [(name, coef) for name, coef in coef_items if coef > 0][:5]
        top_negative = [(name, coef) for name, coef in coef_items if coef < 0][:5]

        return {
            'model_type': 'LogisticRegression',
            'n_features': len(self.feature_names),
            'intercept': float(self.model.intercept_[0]),
            'top_positive_features': top_positive,
            'top_negative_features': top_negative,
            'hyperparameters': {
                'C': self.model.C,
                'max_iter': self.model.max_iter,
                'solver': self.model.solver,
                'class_weight': self.config.get('class_weight')
            }
        }
