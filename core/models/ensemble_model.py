"""
Ensemble model for the ELVIS project.
This module provides a concrete implementation of the BaseModel using ensemble methods.
"""

import os
import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import VotingClassifier

from core.models.base_model import BaseModel
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
        
        # Model parameters
        self.models = kwargs.get('models', [])
        self.weights = kwargs.get('weights', None)
        self.voting = kwargs.get('voting', 'soft')  # 'hard' or 'soft'
        self.threshold = kwargs.get('threshold', 0.5)
        
        # Model path
        self.model_path = kwargs.get('model_path', os.path.join(FILE_PATHS['TRAIN_RESULTS_DIR'], 'ensemble_model.pkl'))
        
        # Initialize model
        self.model = None
        
        # Initialize sub-models if not provided
        if not self.models:
            self._initialize_default_models()
    
    def _initialize_default_models(self) -> None:
        """
        Initialize default models for the ensemble.
        """
        try:
            self.logger.info("Initializing default models for ensemble")
            
            # Random Forest model
            rf_model = RandomForestModel(
                logger=self.logger,
                model_path=os.path.join(FILE_PATHS['TRAIN_RESULTS_DIR'], 'model_rf.ydf')
            )
            
            # Neural Network model
            nn_model = NeuralNetworkModel(
                logger=self.logger,
                model_path=os.path.join(FILE_PATHS['TRAIN_RESULTS_DIR'], 'nn_model.h5')
            )
            
            # Add models to ensemble
            self.models = [
                ('random_forest', rf_model),
                ('neural_network', nn_model)
            ]
            
            self.logger.info(f"Initialized {len(self.models)} default models for ensemble")
            
        except Exception as e:
            self.logger.error(f"Error initializing default models: {e}")
    
    def load_model(self) -> None:
        """
        Load the model from disk.
        """
        try:
            self.logger.info(f"Loading Ensemble model from {self.model_path}")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}")
                
                # Try to load individual models
                for name, model in self.models:
                    model.load_model()
                
                return
            
            # Load model
            self.model = joblib.load(self.model_path)
            
            self.logger.info("Ensemble model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading Ensemble model: {e}")
            
            # Try to load individual models
            for name, model in self.models:
                model.load_model()
    
    def save_model(self) -> None:
        """
        Save the model to disk.
        """
        try:
            self.logger.info(f"Saving Ensemble model to {self.model_path}")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model to save")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            self.logger.info("Ensemble model saved successfully")
            
            # Save individual models
            for name, model in self.models:
                model.save_model()
            
        except Exception as e:
            self.logger.error(f"Error saving Ensemble model: {e}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training labels.
        """
        try:
            self.logger.info("Training Ensemble model")
            
            # Train individual models
            for name, model in self.models:
                self.logger.info(f"Training {name} model")
                model.train(X_train, y_train)
            
            # Create ensemble model
            estimators = []
            for name, model in self.models:
                if hasattr(model, 'model') and model.model is not None:
                    estimators.append((name, model.model))
            
            if estimators:
                self.model = VotingClassifier(
                    estimators=estimators,
                    voting=self.voting,
                    weights=self.weights
                )
                
                # Train ensemble model
                self.model.fit(X_train, y_train)
                
                self.logger.info("Ensemble model trained successfully")
                
                # Save model
                self.save_model()
            else:
                self.logger.warning("No models available for ensemble")
            
        except Exception as e:
            self.logger.error(f"Error training Ensemble model: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X (pd.DataFrame): The features to predict on.
            
        Returns:
            np.ndarray: The predictions.
        """
        try:
            self.logger.info("Making predictions with Ensemble model")
            
            # Check if model exists
            if self.model is not None:
                # Use ensemble model for prediction
                if self.voting == 'soft':
                    # Get probability predictions
                    predictions = self.model.predict_proba(X)[:, 1]
                else:
                    # Get binary predictions
                    predictions = self.model.predict(X)
                
                self.logger.info(f"Made {len(predictions)} predictions with ensemble model")
                
                return predictions
            else:
                # Use individual models for prediction
                self.logger.info("Using individual models for prediction")
                
                predictions = np.zeros(len(X))
                weights = self.weights if self.weights else [1.0] * len(self.models)
                total_weight = sum(weights)
                
                for i, (name, model) in enumerate(self.models):
                    model_predictions = model.predict(X)
                    predictions += (model_predictions * weights[i] / total_weight)
                
                self.logger.info(f"Made {len(predictions)} predictions with individual models")
                
                # Apply threshold for binary classification
                if self.voting == 'hard':
                    predictions = (predictions >= self.threshold).astype(int)
                
                return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with Ensemble model: {e}")
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
            
            # Evaluate individual models
            model_metrics = {}
            for name, model in self.models:
                model_metrics[name] = model.evaluate(X_test, y_test)
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Apply threshold for binary classification
            binary_predictions = (predictions >= self.threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, binary_predictions),
                'precision': precision_score(y_test, binary_predictions, zero_division=0),
                'recall': recall_score(y_test, binary_predictions, zero_division=0),
                'f1': f1_score(y_test, binary_predictions, zero_division=0),
                'model_metrics': model_metrics
            }
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            
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
            
            # Get feature importance from models that support it
            importance_dfs = []
            
            for name, model in self.models:
                importance = model.get_feature_importance()
                if not importance.empty:
                    importance['model'] = name
                    importance_dfs.append(importance)
            
            if importance_dfs:
                # Combine feature importance from all models
                combined_importance = pd.concat(importance_dfs, ignore_index=True)
                
                # Aggregate by feature
                aggregated_importance = combined_importance.groupby('feature').agg({
                    'importance': 'mean',
                    'model': lambda x: ', '.join(x)
                }).reset_index()
                
                # Sort by importance
                aggregated_importance = aggregated_importance.sort_values('importance', ascending=False)
                
                self.logger.info(f"Feature importance: {aggregated_importance}")
                
                return aggregated_importance
            else:
                self.logger.warning("No feature importance available")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: The path to save the model to.
        """
        try:
            self.logger.info(f"Saving Ensemble model to {path}")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model to save")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, path)
            
            self.logger.info("Ensemble model saved successfully")
            
            # Save individual models
            for i, (name, model) in enumerate(self.models):
                model_path = os.path.join(os.path.dirname(path), f"{name}_model.pkl")
                model.save(model_path)
            
        except Exception as e:
            self.logger.error(f"Error saving Ensemble model: {e}")
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """
        Load model from disk.
        
        Args:
            path: The path to load the model from.
            
        Returns:
            EnsembleModel: The loaded model.
        """
        try:
            # Create logger
            logger = logging.getLogger('EnsembleModel')
            
            # Create model instance
            model_instance = cls(logger)
            
            # Set model path
            model_instance.model_path = path
            
            # Load model
            if os.path.exists(path):
                model_instance.model = joblib.load(path)
                logger.info("Ensemble model loaded successfully")
            else:
                logger.warning(f"Model not found at {path}")
            
            # Load individual models
            for i, (name, model) in enumerate(model_instance.models):
                model_path = os.path.join(os.path.dirname(path), f"{name}_model.pkl")
                if os.path.exists(model_path):
                    model_instance.models[i] = (name, model.__class__.load(model_path))
            
            return model_instance
            
        except Exception as e:
            logger = logging.getLogger('EnsembleModel')
            logger.error(f"Error loading Ensemble model: {e}")
            return cls(logger)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: The model parameters.
        """
        return {
            'models': self.models,
            'weights': self.weights,
            'voting': self.voting,
            'threshold': self.threshold
        }
    
    def set_params(self, **params) -> None:
        """
        Set the model parameters.
        
        Args:
            **params: The model parameters.
        """
        if 'models' in params:
            self.models = params['models']
        
        if 'weights' in params:
            self.weights = params['weights']
        
        if 'voting' in params:
            self.voting = params['voting']
        
        if 'threshold' in params:
            self.threshold = params['threshold']
