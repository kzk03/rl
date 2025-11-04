"""
Base class for all baseline models in the retention prediction task.

This module defines the abstract interface that all baseline models must implement,
ensuring consistency across different baseline implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseBaseline(ABC):
    """
    Abstract base class for all baseline models.

    All baseline models (Logistic Regression, Random Forest, XGBoost, etc.)
    must inherit from this class and implement its abstract methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the baseline model.

        Args:
            config: Configuration dictionary for the model
        """
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_names = []

    @abstractmethod
    def train(self, train_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the baseline model on the given data.

        Args:
            train_data: Dictionary containing:
                - 'features': numpy array of shape [n_samples, n_features]
                - 'labels': numpy array of shape [n_samples]
                - 'feature_names': list of feature names (optional)

        Returns:
            Dictionary containing training metrics:
                - 'training_time': time taken to train in seconds
                - 'n_samples': number of training samples
                - 'n_features': number of features
        """
        pass

    @abstractmethod
    def predict(self, test_data: Dict[str, Any]) -> np.ndarray:
        """
        Predict continuation probabilities for test data.

        Args:
            test_data: Dictionary containing:
                - 'features': numpy array of shape [n_samples, n_features]

        Returns:
            Numpy array of continuation probabilities, shape [n_samples]
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: File path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path: File path to load the model from
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model information:
                - 'model_type': name of the model
                - 'is_trained': whether the model has been trained
                - 'config': model configuration
        """
        return {
            'model_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'config': self.config
        }

    def validate_data(self, data: Dict[str, Any], require_labels: bool = True) -> None:
        """
        Validate that the input data has the correct format.

        Args:
            data: Data dictionary to validate
            require_labels: Whether labels are required in the data

        Raises:
            ValueError: If data format is invalid
        """
        if 'features' not in data:
            raise ValueError("Data must contain 'features' key")

        features = data['features']
        if not isinstance(features, np.ndarray):
            raise ValueError("Features must be a numpy array")

        if len(features.shape) != 2:
            raise ValueError(f"Features must be 2D array, got shape {features.shape}")

        if require_labels:
            if 'labels' not in data:
                raise ValueError("Data must contain 'labels' key for training")

            labels = data['labels']
            if not isinstance(labels, np.ndarray):
                raise ValueError("Labels must be a numpy array")

            if len(labels) != len(features):
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match "
                    f"number of features ({len(features)})"
                )
