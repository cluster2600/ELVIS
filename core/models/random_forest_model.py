"""
Random Forest model for the ELVIS project.
This module provides a concrete implementation of the BaseModel using Random Forest.
"""

import os
import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Any, List, Optional, Tuple
import tensorflow_decision_forests as tfdf
import tensorflow as tf

from core.models.base_model import BaseModel
from config import FILE_PATHS

class RandomForestModel(BaseModel):
    """
    Random Forest model for trading.
    Uses TensorFlow Decision Forests for implementation.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the Random Forest model.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__('random_forest', logger, **kwargs)
        
        # Model parameters
        self.num_trees = kwargs.get('num_trees', 100)
        self.max_depth = kwargs.get('max_depth', 20)
        self.min_examples = kwargs.get('min_examples', 5)
        
        # Model path
        self.model_path = kwargs.get('model_path', os.path.join(FILE_PATHS['TRAIN_RESULTS_DIR'], 'model_rf.ydf'))
        
        # Initialize model
        self.model = None
    
    def load_model(self) -> None:
        """
        Load the model from disk.
        """
        try:
            self.logger.info(f"Loading Random Forest model from {self.model_path}")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}")
                return
            
            # Load model
            self.model = tfdf.keras.RandomForestModel.load(self.model_path)
            
            self.logger.info("Random Forest model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading Random Forest model: {e}")
    
    def save_model(self) -> None:
        """
        Save the model to disk.
        """
        try:
            self.logger.info(f"Saving Random Forest model to {self.model_path}")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model to save")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            self.model.save(self.model_path)
            
            self.logger.info("Random Forest model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving Random Forest model: {e}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training labels.
        """
        try:
            self.logger.info("Training Random Forest model")
            
            # Convert to TensorFlow dataset
            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_train, label=y_train)
            
            # Create model
            self.model = tfdf.keras.RandomForestModel(
                num_trees=self.num_trees,
                max_depth=self.max_depth,
                min_examples=self.min_examples,
                verbose=2
            )
            
            # Train model
            self.model.fit(train_ds)
            
            self.logger.info("Random Forest model trained successfully")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X (pd.DataFrame): The features to predict on.
            
        Returns:
            np.ndarray: The predictions.
        """
        try:
            self.logger.info("Making predictions with Random Forest model")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return np.zeros(len(X))
            
            # Convert to TensorFlow dataset
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X, label=None)
            
            # Make predictions
            predictions = self.model.predict(test_ds)
            
            # Convert to numpy array
            predictions = predictions.numpy().flatten()
            
            self.logger.info(f"Made {len(predictions)} predictions")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with Random Forest model: {e}")
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
            self.logger.info("Evaluating Random Forest model")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            # Convert to TensorFlow dataset
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test, label=y_test)
            
            # Evaluate model
            evaluation = self.model.evaluate(test_ds, return_dict=True)
            
            # Extract metrics
            metrics = {
                'accuracy': evaluation['accuracy'],
                'loss': evaluation['loss']
            }
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating Random Forest model: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get the feature importance.
        
        Returns:
            pd.DataFrame: The feature importance.
        """
        try:
            self.logger.info("Getting feature importance")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return pd.DataFrame()
            
            # Get feature importance
            importance = self.model.make_inspector().variable_importances()
            
            # Convert to DataFrame
            if importance and len(importance) > 0 and 'MEAN_DECREASE_IN_ACCURACY' in importance:
                importance_df = pd.DataFrame(
                    importance['MEAN_DECREASE_IN_ACCURACY'],
                    columns=['feature', 'importance']
                )
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                self.logger.info(f"Feature importance: {importance_df}")
                
                return importance_df
            else:
                self.logger.warning("No feature importance available")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
