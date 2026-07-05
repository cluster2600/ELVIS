"""
Ensemble models for trading predictions.
Implements various ensemble methods including stacking, bagging, and boosting.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


class EnsembleModel:
    """Base class for ensemble models."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = []
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the ensemble model."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        raise NotImplementedError


class StackingEnsemble(EnsembleModel):
    """Stacking ensemble using multiple base models and a meta-learner."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.meta_learner = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize base models and meta-learner."""
        # Base models
        self.models = [
            RandomForestRegressor(**self.config.get("rf_params", {})),
            GradientBoostingRegressor(**self.config.get("gb_params", {})),
            xgb.XGBRegressor(**self.config.get("xgb_params", {})),
            lgb.LGBMRegressor(**self.config.get("lgb_params", {})),
        ]

        # Meta-learner
        self.meta_learner = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the stacking ensemble."""
        # Fit base models
        base_predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            model.fit(X, y)
            base_predictions[:, i] = model.predict(X)

        # Fit meta-learner
        self.meta_learner.fit(base_predictions, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the stacking ensemble."""
        base_predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            base_predictions[:, i] = model.predict(X)
        return self.meta_learner.predict(base_predictions)


class WeightedEnsemble(EnsembleModel):
    """Weighted ensemble using multiple models with learned weights."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self._initialize_models()

    def _initialize_models(self):
        """Initialize base models."""
        self.models = [
            RandomForestRegressor(**self.config.get("rf_params", {})),
            GradientBoostingRegressor(**self.config.get("gb_params", {})),
            xgb.XGBRegressor(**self.config.get("xgb_params", {})),
            lgb.LGBMRegressor(**self.config.get("lgb_params", {})),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the weighted ensemble."""
        # Fit base models
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            model.fit(X, y)
            predictions[:, i] = model.predict(X)

        # Learn optimal weights using validation performance
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        weights = np.zeros(len(self.models))

        for train_idx, val_idx in kf.split(X):
            val_predictions = predictions[val_idx]
            val_true = y[val_idx]

            # Calculate MSE for each model
            mse = np.mean((val_predictions - val_true.reshape(-1, 1)) ** 2, axis=0)
            weights += 1 / (mse + 1e-10)  # Add small constant to avoid division by zero

        self.weights = weights / np.sum(weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the weighted ensemble."""
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return np.sum(predictions * self.weights, axis=1)


class NeuralEnsemble(EnsembleModel):
    """Neural network ensemble using multiple architectures."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()

    def _initialize_models(self):
        """Initialize neural network models."""
        input_dim = self.config.get("input_dim", 10)
        hidden_dims = self.config.get("hidden_dims", [64, 32])

        # Create different architectures
        self.models = [
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dims[0], 1),
            ),
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], 1),
            ),
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.Linear(hidden_dims[0], 1),
            ),
        ]

        # Move models to device
        for model in self.models:
            model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the neural ensemble."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        for model in self.models:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the neural ensemble."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
                predictions.append(pred)

        return np.mean(predictions, axis=0)
