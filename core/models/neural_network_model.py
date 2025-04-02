"""
Neural Network model for the ELVIS project.
This module provides a concrete implementation of the BaseModel using neural networks.
"""

import os
import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Any, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from core.models.base_model import BaseModel
from config import FILE_PATHS

class NeuralNetworkModel(BaseModel):
    """
    Neural Network model for trading.
    Uses TensorFlow/Keras for implementation.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the Neural Network model.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__('neural_network', logger, **kwargs)
        
        # Model parameters
        self.input_shape = kwargs.get('input_shape', (60, 10))  # (sequence_length, features)
        self.lstm_units = kwargs.get('lstm_units', [128, 64])
        self.dense_units = kwargs.get('dense_units', [32, 16])
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs = kwargs.get('epochs', 100)
        
        # Model path
        self.model_path = kwargs.get('model_path', os.path.join(FILE_PATHS['TRAIN_RESULTS_DIR'], 'nn_model.h5'))
        
        # Initialize model
        self.model = None
        self.scaler = None
    
    def load_model(self) -> None:
        """
        Load the model from disk.
        """
        try:
            self.logger.info(f"Loading Neural Network model from {self.model_path}")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}")
                return
            
            # Load model
            self.model = load_model(self.model_path)
            
            # Load scaler if available
            scaler_path = os.path.join(os.path.dirname(self.model_path), 'nn_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.logger.info("Neural Network model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading Neural Network model: {e}")
    
    def save_model(self) -> None:
        """
        Save the model to disk.
        """
        try:
            self.logger.info(f"Saving Neural Network model to {self.model_path}")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model to save")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            self.model.save(self.model_path)
            
            # Save scaler if available
            if self.scaler is not None:
                scaler_path = os.path.join(os.path.dirname(self.model_path), 'nn_scaler.pkl')
                joblib.dump(self.scaler, scaler_path)
            
            self.logger.info("Neural Network model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving Neural Network model: {e}")
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            X (pd.DataFrame): The features.
            y (pd.Series): The labels.
            sequence_length (int): The sequence length.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: The sequences and labels.
        """
        X_array = X.values
        y_array = y.values
        
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_array) - sequence_length):
            X_sequences.append(X_array[i:i+sequence_length])
            y_sequences.append(y_array[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build the neural network model.
        
        Returns:
            tf.keras.Model: The built model.
        """
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(self.lstm_units[0], return_sequences=True, input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        for units in self.lstm_units[1:]:
            model.add(LSTM(units, return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training labels.
        """
        try:
            self.logger.info("Training Neural Network model")
            
            # Create sequences
            sequence_length = self.input_shape[0]
            X_seq, y_seq = self._create_sequences(X_train, y_train, sequence_length)
            
            # Build model
            self.model = self._build_model()
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss')
            ]
            
            # Train model
            self.model.fit(
                X_seq, y_seq,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Neural Network model trained successfully")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            self.logger.error(f"Error training Neural Network model: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X (pd.DataFrame): The features to predict on.
            
        Returns:
            np.ndarray: The predictions.
        """
        try:
            self.logger.info("Making predictions with Neural Network model")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return np.zeros(len(X))
            
            # Create sequences
            sequence_length = self.input_shape[0]
            if len(X) < sequence_length:
                self.logger.warning(f"Not enough data for prediction. Need at least {sequence_length} samples.")
                return np.zeros(len(X))
            
            # Create sequences for prediction
            X_array = X.values
            X_sequences = []
            
            for i in range(len(X_array) - sequence_length + 1):
                X_sequences.append(X_array[i:i+sequence_length])
            
            X_sequences = np.array(X_sequences)
            
            # Make predictions
            predictions = self.model.predict(X_sequences)
            
            # Pad with zeros for the first sequence_length-1 samples
            padded_predictions = np.zeros(len(X))
            padded_predictions[sequence_length-1:] = predictions.flatten()
            
            self.logger.info(f"Made {len(predictions)} predictions")
            
            return padded_predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with Neural Network model: {e}")
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
            self.logger.info("Evaluating Neural Network model")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return {'accuracy': 0.0, 'loss': 0.0}
            
            # Create sequences
            sequence_length = self.input_shape[0]
            X_seq, y_seq = self._create_sequences(X_test, y_test, sequence_length)
            
            # Evaluate model
            evaluation = self.model.evaluate(X_seq, y_seq, verbose=0)
            
            # Extract metrics
            metrics = {
                'loss': evaluation[0],
                'accuracy': evaluation[1]
            }
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating Neural Network model: {e}")
            return {'accuracy': 0.0, 'loss': 0.0}
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get the feature importance.
        Neural networks don't have direct feature importance like tree-based models,
        so we'll use a simple sensitivity analysis.
        
        Returns:
            pd.DataFrame: The feature importance.
        """
        self.logger.warning("Feature importance not directly available for Neural Network models")
        return pd.DataFrame()
