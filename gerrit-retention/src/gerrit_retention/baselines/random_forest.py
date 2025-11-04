"""
Random Forest baseline for developer retention prediction.

This module implements a Random Forest baseline that aggregates time-series
features and uses ensemble learning for prediction.
"""

import numpy as np
import time
import pickle
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier as SklearnRF

from .base_baseline import BaseBaseline


class RandomForestBaseline(BaseBaseline):
    """
    Random Forest baseline model.

    This model uses ensemble learning with decision trees to predict
    developer continuation. It provides robust performance and natural
    feature importance ranking.

    Key features:
    - Ensemble of decision trees (default: 100 trees)
    - Handles class imbalance with class_weight='balanced'
    - Out-of-Bag (OOB) error estimation
    - Natural feature importance from tree splits
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Random Forest baseline.

        Args:
            config: Configuration dictionary with optional keys:
                - 'n_estimators': Number of trees (default: 100)
                - 'max_depth': Maximum depth of trees (default: None)
                - 'min_samples_split': Min samples to split (default: 2)
                - 'min_samples_leaf': Min samples in leaf (default: 1)
                - 'max_features': Features to consider for split (default: 'sqrt')
                - 'class_weight': Class weight strategy (default: 'balanced')
                - 'oob_score': Compute OOB score (default: True)
                - 'n_jobs': Number of parallel jobs (default: -1)
                - 'random_state': Random seed (default: 42)
        """
        super().__init__(config)

        # Extract hyperparameters
        n_estimators = self.config.get('n_estimators', 100)
        max_depth = self.config.get('max_depth', None)
        min_samples_split = self.config.get('min_samples_split', 2)
        min_samples_leaf = self.config.get('min_samples_leaf', 1)
        max_features = self.config.get('max_features', 'sqrt')
        class_weight = self.config.get('class_weight', 'balanced')
        oob_score = self.config.get('oob_score', True)
        n_jobs = self.config.get('n_jobs', -1)
        random_state = self.config.get('random_state', 42)

        # Initialize model
        self.model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state
        )

        self.oob_score_value = None

    def train(self, train_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train Random Forest model.

        Args:
            train_data: Dictionary containing:
                - 'features': numpy array of shape [n_samples, n_features]
                - 'labels': numpy array of shape [n_samples]
                - 'feature_names': list of feature names (optional)

        Returns:
            Dictionary with training metrics including OOB score
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

        self.model.fit(X_train, y_train)

        training_time = time.time() - start_time

        self.is_trained = True

        # Get OOB score if enabled
        if self.config.get('oob_score', True):
            self.oob_score_value = self.model.oob_score_

        # Return training info
        training_info = {
            'training_time': training_time,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'positive_rate': float(np.mean(y_train)),
            'model_type': 'random_forest',
            'n_trees': self.model.n_estimators
        }

        if self.oob_score_value is not None:
            training_info['oob_score'] = float(self.oob_score_value)

        return training_info

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

        # Predict probabilities (probability of class 1)
        probabilities = self.model.predict_proba(X_test)[:, 1]

        return probabilities

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from Random Forest.

        Feature importance is calculated based on mean decrease in impurity
        (Gini importance) across all trees.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")

        # Get feature importances from the model
        importances = self.model.feature_importances_

        # Map to feature names
        importance_dict = {
            name: float(imp)
            for name, imp in zip(self.feature_names, importances)
        }

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict

    def get_feature_importance_with_std(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance with standard deviation across trees.

        Returns:
            Dictionary mapping feature names to dicts with 'mean' and 'std'
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")

        # Get importances from all trees
        tree_importances = np.array([
            tree.feature_importances_
            for tree in self.model.estimators_
        ])

        # Calculate mean and std
        mean_importances = np.mean(tree_importances, axis=0)
        std_importances = np.std(tree_importances, axis=0)

        # Map to feature names
        importance_dict = {
            name: {
                'mean': float(mean_imp),
                'std': float(std_imp)
            }
            for name, mean_imp, std_imp in zip(
                self.feature_names, mean_importances, std_importances
            )
        }

        # Sort by mean importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1]['mean'], reverse=True)
        )

        return importance_dict

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
            'feature_names': self.feature_names,
            'config': self.config,
            'is_trained': self.is_trained,
            'oob_score': self.oob_score_value
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
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        self.oob_score_value = model_data.get('oob_score')

        print(f"Model loaded from {path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get detailed model summary.

        Returns:
            Dictionary with model details including tree statistics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting summary")

        # Get top features
        importance_dict = self.get_feature_importance()
        top_features = list(importance_dict.items())[:10]

        # Calculate tree statistics
        tree_depths = [tree.get_depth() for tree in self.model.estimators_]
        tree_nodes = [tree.get_n_leaves() for tree in self.model.estimators_]

        summary = {
            'model_type': 'RandomForest',
            'n_features': len(self.feature_names),
            'n_estimators': self.model.n_estimators,
            'top_10_features': top_features,
            'tree_statistics': {
                'mean_depth': float(np.mean(tree_depths)),
                'max_depth': int(np.max(tree_depths)),
                'mean_leaves': float(np.mean(tree_nodes)),
                'max_leaves': int(np.max(tree_nodes))
            },
            'hyperparameters': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'min_samples_split': self.model.min_samples_split,
                'min_samples_leaf': self.model.min_samples_leaf,
                'max_features': self.model.max_features,
                'class_weight': self.config.get('class_weight')
            }
        }

        if self.oob_score_value is not None:
            summary['oob_score'] = float(self.oob_score_value)

        return summary

    def predict_with_uncertainty(self, test_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty estimates.

        Returns predictions along with standard deviation across trees.

        Args:
            test_data: Dictionary containing test features

        Returns:
            Dictionary with:
                - 'predictions': mean predictions across trees
                - 'std': standard deviation across trees
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.validate_data(test_data, require_labels=False)
        X_test = test_data['features']

        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict_proba(X_test)[:, 1]
            for tree in self.model.estimators_
        ])

        # Calculate mean and std
        mean_pred = np.mean(tree_predictions, axis=0)
        std_pred = np.std(tree_predictions, axis=0)

        return {
            'predictions': mean_pred,
            'std': std_pred
        }
