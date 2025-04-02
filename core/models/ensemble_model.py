"""
Ensemble model for the ELVIS project.
This module provides a concrete implementation of the BaseModel using ensemble methods.
"""

import os
import pandas as pd
import numpy as np
import logging
import joblib
import json # For saving/loading config
from typing import Dict, Any, List, Optional, Tuple, Type
# Removed VotingClassifier import
# from sklearn.ensemble import VotingClassifier

from core.models.base_model import BaseModel
# Import necessary model classes for loading
from core.models.random_forest_model import RandomForestModel
from core.models.neural_network_model import NeuralNetworkModel
from config import FILE_PATHS

class EnsembleModel(BaseModel):
    """
    Ensemble model for trading.
    Combines multiple models for improved predictions.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the Ensemble model.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        self.logger = logger

        # Ensemble configuration
        # models: List of tuples (name: str, model_instance: BaseModel)
        self.models: List[Tuple[str, BaseModel]] = kwargs.get('models', [])
        # model_configs: List of dicts {'name': str, 'class': str, 'path': str} for saving/loading
        self.model_configs: List[Dict[str, str]] = kwargs.get('model_configs', [])
        self.weights: Optional[List[float]] = kwargs.get('weights', None)
        self.voting: str = kwargs.get('voting', 'soft')  # 'hard' or 'soft'
        self.threshold: float = kwargs.get('threshold', 0.5) # Used for 'hard' voting or final decision

        # Path to save/load the ensemble configuration (not the models themselves)
        self.config_path: str = kwargs.get('config_path', os.path.join(FILE_PATHS['TRAIN_RESULTS_DIR'], 'ensemble_config.json'))

        # Ensure weights match number of models if provided
        if self.weights and len(self.weights) != len(self.models):
            self.logger.warning(f"Number of weights ({len(self.weights)}) does not match number of models ({len(self.models)}). Ignoring weights.")
            self.weights = None
        if not self.weights:
             self.weights = [1.0] * len(self.models) # Default to equal weights

        # Initialize models from configs if models list is empty
        if not self.models and self.model_configs:
            self._load_models_from_configs()
        elif not self.model_configs and self.models:
             # Populate configs from models if possible (requires models to have a path attribute)
             self._populate_configs_from_models()


    def _get_model_class(self, class_name: str) -> Type[BaseModel]:
        """Helper to get model class type from string name."""
        # Add other model types here as needed
        if class_name == 'RandomForestModel':
            return RandomForestModel
        elif class_name == 'NeuralNetworkModel':
            return NeuralNetworkModel
        # Add other models like TransformerModel, RLModel etc.
        # from .transformer_model import TransformerModel # Example
        # if class_name == 'TransformerModel': return TransformerModel
        else:
            raise ValueError(f"Unknown model class name: {class_name}")

    def _load_models_from_configs(self):
        """Loads sub-model instances based on self.model_configs."""
        self.models = []
        self.logger.info(f"Loading {len(self.model_configs)} sub-models from configuration...")
        for config in self.model_configs:
            name = config.get('name')
            class_name = config.get('class')
            path = config.get('path')
            if not all([name, class_name, path]):
                self.logger.error(f"Invalid model config: {config}. Skipping.")
                continue
            try:
                ModelClass = self._get_model_class(class_name)
                # Use the class's load method
                model_instance = ModelClass.load(path)
                self.models.append((name, model_instance))
                self.logger.info(f"Successfully loaded sub-model '{name}' from {path}")
            except Exception as e:
                self.logger.error(f"Failed to load sub-model '{name}' from config {config}: {e}", exc_info=True)
        self.logger.info(f"Finished loading sub-models. {len(self.models)} loaded.")
        # Reset weights if model count changed
        if len(self.weights) != len(self.models):
             self.weights = [1.0] * len(self.models)


    def _populate_configs_from_models(self):
        """Populates self.model_configs based on self.models list."""
        self.model_configs = []
        for name, model_instance in self.models:
             # Attempt to get path - assumes models store their load path
             # NeuralNetworkModel stores it as self.model_path
             # RandomForestModel needs adjustment to store its path
             model_path = getattr(model_instance, 'model_path', None) # Adapt as needed per model
             if model_path:
                 self.model_configs.append({
                     'name': name,
                     'class': model_instance.__class__.__name__,
                     'path': model_path
                 })
             else:
                 self.logger.warning(f"Could not determine path for model '{name}' ({model_instance.__class__.__name__}). It won't be saved in ensemble config.")


    # BaseModel interface methods - Refactored

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the ensemble model.
        NOTE: This method assumes the individual sub-models are already trained.
        It primarily ensures the ensemble configuration is ready.
        If sub-models need training, train them independently first.

        Args:
            X_train: Not used directly for ensemble training itself.
            y_train: Not used directly for ensemble training itself.
        """
        self.logger.info("EnsembleModel 'train' called. Ensuring sub-models are loaded.")
        # Ensure models are loaded if initialized via config
        if not self.models and self.model_configs:
            self._load_models_from_configs()
        elif not self.model_configs and self.models:
             self._populate_configs_from_models() # Ensure configs are ready for saving

        if not self.models:
             self.logger.error("No sub-models configured or loaded for the ensemble. Cannot 'train'.")
             return

        # Optional: Validate sub-models can predict (basic check)
        try:
            if not X_train.empty:
                 _ = self.predict(X_train.head(1)) # Try predicting on a small sample
                 self.logger.info("Sub-models seem ready for prediction.")
            else:
                 self.logger.warning("No training data provided to validate sub-model prediction readiness.")
        except Exception as e:
            self.logger.error(f"Error validating sub-model prediction during ensemble 'train': {e}", exc_info=True)

        # Ensemble doesn't require fitting like VotingClassifier anymore
        self.logger.info("Ensemble model configuration verified.")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions by aggregating predictions from sub-models.

        Args:
            X (pd.DataFrame): The features to predict on.

        Returns:
            np.ndarray: The aggregated predictions (probabilities if soft voting, classes if hard voting).
                        Returns array of zeros if prediction fails.
        """
        if not self.models:
            self.logger.error("No sub-models loaded or configured for prediction.")
            return np.zeros(len(X))

        all_predictions = []
        valid_models_count = 0
        effective_weights = []

        # Get predictions from each sub-model
        for i, (name, model) in enumerate(self.models):
            try:
                # Assume predict returns probabilities or class labels
                model_pred = model.predict(X)
                all_predictions.append(model_pred)
                effective_weights.append(self.weights[i])
                valid_models_count += 1
            except Exception as e:
                self.logger.error(f"Error predicting with sub-model '{name}': {e}", exc_info=True)
                # Optionally skip this model or handle error

        if valid_models_count == 0:
            self.logger.error("Prediction failed for all sub-models.")
            return np.zeros(len(X))

        # Aggregate predictions
        try:
            # Normalize effective weights
            total_weight = sum(effective_weights)
            normalized_weights = [w / total_weight for w in effective_weights] if total_weight > 0 else [1.0 / valid_models_count] * valid_models_count

            if self.voting == 'soft':
                # Average probabilities (assuming predict returns probabilities)
                # Ensure all predictions are numpy arrays for broadcasting
                np_preds = [np.array(p) for p in all_predictions]
                weighted_preds = [pred * weight for pred, weight in zip(np_preds, normalized_weights)]
                final_predictions = np.sum(weighted_preds, axis=0)
                # Note: If sub-models return binary (0/1), soft voting averages these.
                # If they return probabilities, it averages probabilities.
                # The interpretation depends on what sub-models return.
                # For consistency, sub-models should ideally return probabilities for soft voting.
                self.logger.info(f"Aggregated {valid_models_count} predictions using soft voting.")

            elif self.voting == 'hard':
                # Majority vote (assuming predict returns class labels 0 or 1)
                np_preds = np.array([np.round(p) for p in all_predictions]).astype(int) # Ensure binary predictions
                # Weighted majority vote
                weighted_votes = np.tensordot(normalized_weights, np_preds, axes=([0],[0]))
                final_predictions = (weighted_votes >= self.threshold).astype(int) # Use threshold for final decision
                self.logger.info(f"Aggregated {valid_models_count} predictions using hard voting.")

            else:
                self.logger.error(f"Unsupported voting type: {self.voting}")
                return np.zeros(len(X))

            return final_predictions

        except Exception as e:
            self.logger.error(f"Error aggregating predictions: {e}", exc_info=True)
            return np.zeros(len(X))
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X_test (pd.DataFrame): The test features.
            y_test (pd.Series): The test labels.
            
        Returns:
            Dict[str, float]: The evaluation metrics.
        """
        try:
            self.logger.info("Evaluating Ensemble model")
            
            # Evaluate individual models (optional, can be intensive)
            # model_metrics = {}
            # for name, model in self.models:
            #     if hasattr(model, 'evaluate'):
            #         try:
            #             model_metrics[name] = model.evaluate(X_test, y_test)
            #         except Exception as e:
            #             self.logger.error(f"Error evaluating sub-model {name}: {e}")
            #             model_metrics[name] = {"error": str(e)}

            # Make predictions with the ensemble
            predictions = self.predict(X_test) # Returns probabilities or classes based on voting

            # Convert to binary predictions using the threshold for metrics calculation
            # If soft voting, predictions are probabilities; if hard voting, they are already 0/1
            if self.voting == 'soft':
                 binary_predictions = (predictions >= self.threshold).astype(int)
            else: # Hard voting already produced binary predictions
                 binary_predictions = predictions.astype(int)

            # Calculate standard classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, binary_predictions),
                'precision': precision_score(y_test, binary_predictions, zero_division=0),
                'recall': recall_score(y_test, binary_predictions, zero_division=0),
                'f1': f1_score(y_test, binary_predictions, zero_division=0)
                # 'sub_model_metrics': model_metrics # Optionally include sub-model metrics
            }

            self.logger.info(f"Ensemble evaluation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating Ensemble model: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get the feature importance.
        For ensemble models, we aggregate feature importance from models that support it.
        
        Returns:
            pd.DataFrame: The feature importance.
        """
        try:
            self.logger.info("Getting feature importance for Ensemble model")
            
            # Aggregate feature importance from sub-models that support it
            all_importances = []
            for name, model in self.models:
                if hasattr(model, 'get_feature_importance'):
                    try:
                        importance = model.get_feature_importance()
                        if isinstance(importance, pd.DataFrame) and not importance.empty:
                             # Assume DataFrame has 'feature' and 'importance' columns
                             importance['model_name'] = name
                             all_importances.append(importance)
                        elif isinstance(importance, dict): # Handle dict format if needed
                             df_importance = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
                             df_importance['model_name'] = name
                             all_importances.append(df_importance)
                    except Exception as e:
                        self.logger.error(f"Error getting feature importance from {name}: {e}")

            if not all_importances:
                self.logger.warning("No feature importance available from any sub-model.")
                return pd.DataFrame()

            # Combine and average importances
            combined = pd.concat(all_importances, ignore_index=True)
            # Average importance per feature across models
            # Ensure importance is numeric before aggregation
            combined['importance'] = pd.to_numeric(combined['importance'], errors='coerce')
            combined = combined.dropna(subset=['importance'])

            avg_importance = combined.groupby('feature')['importance'].mean().reset_index()
            avg_importance = avg_importance.sort_values(by='importance', ascending=False)

            self.logger.info("Aggregated feature importance calculated.")
            return avg_importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Save the ensemble configuration (not the models themselves).
        Assumes sub-models are saved independently.

        Args:
            path (str): Path to save the ensemble configuration JSON file.
                        If None, uses self.config_path.
        """
        save_path = path if path else self.config_path
        try:
            self.logger.info(f"Saving Ensemble configuration to {save_path}")

            # Ensure model configs are up-to-date if models were added directly
            if not self.model_configs and self.models:
                 self._populate_configs_from_models()

            config_data = {
                'model_configs': self.model_configs,
                'weights': self.weights,
                'voting': self.voting,
                'threshold': self.threshold
            }

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save config as JSON
            with open(save_path, 'w') as f:
                json.dump(config_data, f, indent=4)

            self.logger.info("Ensemble configuration saved successfully.")

        except Exception as e:
            self.logger.error(f"Error saving Ensemble configuration to {save_path}: {e}", exc_info=True)
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """
        Load model from disk.
        
        Load the ensemble configuration and its constituent models.

        Args:
            path (str): Path to the ensemble configuration JSON file.

        Returns:
            EnsembleModel: The loaded ensemble model instance.
        """
        logger = logging.getLogger('EnsembleModel') # Use a logger
        logger.info(f"Loading Ensemble configuration from {path}")
        try:
            if not os.path.exists(path):
                 logger.error(f"Ensemble configuration file not found at {path}")
                 # Return a default/empty instance? Or raise error?
                 return cls(logger=logger) # Return empty instance

            with open(path, 'r') as f:
                config_data = json.load(f)

            # Create instance with loaded config, models will be loaded by _load_models_from_configs
            instance = cls(
                logger=logger,
                model_configs=config_data.get('model_configs', []),
                weights=config_data.get('weights'),
                voting=config_data.get('voting', 'soft'),
                threshold=config_data.get('threshold', 0.5),
                config_path=path # Store the path it was loaded from
            )
            # _load_models_from_configs is called in __init__ if model_configs is present

            logger.info(f"Ensemble configuration loaded. {len(instance.models)} sub-models loaded.")
            return instance

        except Exception as e:
            logger.error(f"Error loading Ensemble configuration from {path}: {e}", exc_info=True)
            # Return a default/empty instance in case of error
            return cls(logger=logger)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Get the ensemble model parameters (configuration).

        Returns:
            Dict[str, Any]: The ensemble parameters.
        """
        # Return config rather than model instances
        return {
            'model_configs': self.model_configs,
            'weights': self.weights,
            'voting': self.voting,
            'threshold': self.threshold,
            'config_path': self.config_path
        }

    def set_params(self, **params) -> None:
        """
        Set the model parameters.
        
        Set the ensemble model parameters.

        Args:
            **params: The parameters to set.
        """
        if 'model_configs' in params:
            self.model_configs = params['model_configs']
            # Reload models based on new configs
            self._load_models_from_configs()

        if 'weights' in params:
            new_weights = params['weights']
            if new_weights and len(new_weights) == len(self.models):
                 self.weights = new_weights
            elif not new_weights:
                 self.weights = [1.0] * len(self.models) # Reset to equal weights
            else:
                 self.logger.warning(f"Attempted to set weights with incorrect length ({len(new_weights)}). Expected {len(self.models)}. Keeping existing weights.")

        if 'voting' in params:
            self.voting = params['voting']

        if 'threshold' in params:
            self.threshold = params['threshold']

        if 'config_path' in params:
             self.config_path = params['config_path']

        self.logger.info("Ensemble parameters updated.")
