import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna
import joblib
import logging
import os

# Import other modules (no circular imports here)
from .transformer_models import FinancialTransformer
from .rl_agents import MultiAgentTradingSystem
from .explainable_ai import SHAPExplainer, LIMEExplainer
from .ensemble_models import StackingEnsemble, WeightedEnsemble, NeuralEnsemble

class ModelTrainer:
    """
    Master class to orchestrate training, evaluation, explanation, and optimization of trading models.
    """

    def __init__(self, config: Dict):
        """
        Initialize the ModelTrainer with configurations for models and directories.

        Args:
            config (Dict): Configuration parameters for models, features, logging, etc.
        """
        self.config = config

        # Detect appropriate device for PyTorch computations
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.logger = logging.getLogger(__name__)

        # Setup directories for models and logs
        self.model_dir = os.path.join(config.get('model_dir', 'models'))
        self.log_dir = os.path.join(config.get('log_dir', 'logs'))
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize ensemble model instances
        self.ensemble_models = {
            'stacking': StackingEnsemble(config.get('stacking_config', {})),
            'weighted': WeightedEnsemble(config.get('weighted_config', {})),
            'neural': NeuralEnsemble(config.get('neural_config', {}))
        }

        # Placeholder to store model state
        self.model_state = None

        # Add model attribute to hold main model instance
        self.model = None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and target arrays from DataFrame for training.

        Args:
            data (pd.DataFrame): Raw input DataFrame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (features array, target array)
        """
        # Extract feature names from feature_config
        feature_config = self.config.get('feature_config', {})
        features = [f['name'] for f in feature_config.get('features', [])]
        target = self.config.get('target', 'price')
        # Validate that features and target exist in data columns
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        return data[features].values, data[target].values

    def train_epoch(self, train_loader, epoch: int) -> dict:
        """
        Basic example of training a simple PyTorch model for one epoch.

        Args:
            train_loader: PyTorch DataLoader for training data.
            epoch (int): Epoch number.

        Returns:
            dict: Training metrics including loss.
        """
        self.logger.info(f"[EPOCH {epoch + 1}] Training started")
        # Define a simple model if not already defined
        if self.model is None:
            input_dim = train_loader.dataset.tensors[0].shape[1]
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = torch.nn.MSELoss()

        self.model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch_X.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        self.model_state = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
        self.logger.info(f"[EPOCH {epoch + 1}] Training completed with loss: {avg_loss:.4f}")
        return {'loss': avg_loss}

    def validate(self, val_loader) -> dict:
        """
        Placeholder method for validating—extend with validation logic.

        Args:
            val_loader: PyTorch DataLoader for validation data.

        Returns:
            dict: Example validation metrics.
        """
        self.logger.info("[VALIDATION] Validation placeholder (no-op)")
        return {'val_loss': 0.01}

    def state_dict(self) -> dict:
        """
        Get current model state dictionary (useful for checkpointing).

        Returns:
            dict: Model state.
        """
        return self.model_state

    def create_data_loaders(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train/validation splits.

        Uses time-series split for validation.

        Args:
            X (np.ndarray): Features array.
            y (np.ndarray): Target array.
            batch_size (int): Batch size for DataLoaders.

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, val_loader
        """
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, val_idx = list(tscv.split(X))[-1]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Union[StackingEnsemble, WeightedEnsemble, NeuralEnsemble]]:
        """
        Train all configured ensemble models.

        Args:
            X (np.ndarray): Features array.
            y (np.ndarray): Target array.

        Returns:
            dict: Trained ensemble models keyed by model name.
        """
        trained_models = {}
        for name, model in self.ensemble_models.items():
            self.logger.info(f"[ENSEMBLE] Training {name}")
            model.fit(X, y)
            trained_models[name] = model
        return trained_models

    def evaluate_ensemble(self, models: Dict[str, Union[StackingEnsemble, WeightedEnsemble, NeuralEnsemble]], X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ensemble models with regression metrics.

        Args:
            models (dict): Trained models.
            X (np.ndarray): Validation features.
            y (np.ndarray): Validation target.

        Returns:
            dict: Metrics per model.
        """
        results = {}
        for name, model in models.items():
            predictions = model.predict(X)
            results[name] = {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'r2': r2_score(y, predictions)
            }
        return results

    def save_ensemble(self, models: Dict[str, Union[StackingEnsemble, WeightedEnsemble, NeuralEnsemble]], path: Optional[str] = None):
        """
        Persist ensemble models to disk.

        Args:
            models (dict): Trained models.
            path (str): Directory to save models.
        """
        path = path or os.path.join(self.model_dir, 'ensemble_models')
        os.makedirs(path, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, os.path.join(path, f'{name}_model.joblib'))

    def load_ensemble(self, path: Optional[str] = None) -> Dict[str, Union[StackingEnsemble, WeightedEnsemble, NeuralEnsemble]]:
        """
        Load saved ensemble models from disk.

        Args:
            path (str): Directory to load models from.

        Returns:
            dict: Loaded models.
        """
        path = path or os.path.join(self.model_dir, 'ensemble_models')
        loaded_models = {}
        for name in self.ensemble_models.keys():
            model_path = os.path.join(path, f'{name}_model.joblib')
            if os.path.exists(model_path):
                loaded_models[name] = joblib.load(model_path)
        return loaded_models

    def explain_model(self, model, X, feature_names):
        """
        Generate explanations for the given model using SHAP or LIME.

        Args:
            model: The model to explain.
            X (np.ndarray): Feature matrix.
            feature_names (list): List of feature names.

        Returns:
            dict: Explanation results.
        """
        try:
            # Import explainers
            from .explainable_ai import SHAPExplainer, LIMEExplainer

            # Convert X to numpy array if it's a tensor
            if isinstance(X, torch.Tensor):
                X_numpy = X.cpu().detach().numpy()
            else:
                X_numpy = np.array(X)

            # Choose explainer based on model type
            if hasattr(model, 'predict_proba') or hasattr(model, 'predict'):
                explainer = LIMEExplainer(model, feature_names, X_numpy)
            else:
                explainer = SHAPExplainer(model, feature_names, X_numpy[:100])

            explanation = explainer.explain(X_numpy)
            return explanation
        except Exception as e:
            self.logger.error(f"Error in explain_model: {e}")
            return {}
