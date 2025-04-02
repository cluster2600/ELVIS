"""
Transformer model for the ELVIS project.
This module provides a concrete implementation of the BaseModel using transformer architecture.
"""

import os
import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core.models.base_model import BaseModel
from config import FILE_PATHS

class TransformerBlock(nn.Module):
    """
    Transformer block for time series forecasting.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the transformer block.
        
        Args:
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): The input tensor of shape (seq_len, batch_size, input_dim).
            mask (torch.Tensor, optional): The attention mask.
            
        Returns:
            torch.Tensor: The output tensor of shape (seq_len, batch_size, input_dim).
        """
        # Self-attention with residual connection and layer normalization
        attended, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual connection and layer normalization
        x = self.norm2(x + self.feed_forward(x))
        
        return x

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series forecasting.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        seq_len: int,
        hidden_dim: int = 128, 
        num_layers: int = 3, 
        num_heads: int = 4, 
        dropout: float = 0.1
    ):
        """
        Initialize the transformer model.
        
        Args:
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
            seq_len (int): The sequence length.
            hidden_dim (int): The hidden dimension.
            num_layers (int): The number of transformer layers.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Positional encoding
        self.register_buffer(
            "positional_encoding",
            self._generate_positional_encoding(seq_len, input_dim)
        )
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, input_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(input_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(input_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _generate_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """
        Generate positional encoding.
        
        Args:
            seq_len (int): The sequence length.
            d_model (int): The model dimension.
            
        Returns:
            torch.Tensor: The positional encoding of shape (seq_len, 1, d_model).
        """
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(seq_len, 1, d_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_dim).
            
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """
        # Transpose to (seq_len, batch_size, input_dim)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Take the last sequence element
        x = x[-1]
        
        # Output projection
        x = self.output_projection(x)
        
        return x

class TransformerModel(BaseModel):
    """
    Transformer model for trading.
    Uses transformer architecture for time series forecasting.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the transformer model.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__('transformer', logger, **kwargs)
        
        # Model parameters
        self.input_dim = kwargs.get('input_dim', 10)
        self.output_dim = kwargs.get('output_dim', 1)
        self.seq_len = kwargs.get('seq_len', 60)
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.num_layers = kwargs.get('num_layers', 3)
        self.num_heads = kwargs.get('num_heads', 4)
        self.dropout = kwargs.get('dropout', 0.1)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs = kwargs.get('epochs', 100)
        
        # Model path
        self.model_path = kwargs.get('model_path', os.path.join(FILE_PATHS['TRAIN_RESULTS_DIR'], 'transformer_model.pt'))
        
        # Initialize model
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self) -> TimeSeriesTransformer:
        """
        Build the transformer model.
        
        Returns:
            TimeSeriesTransformer: The built model.
        """
        model = TimeSeriesTransformer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        return model.to(self.device)
    
    def _create_sequences(self, X: pd.DataFrame) -> torch.Tensor:
        """
        Create sequences for transformer input.
        
        Args:
            X (pd.DataFrame): The features.
            
        Returns:
            torch.Tensor: The sequences.
        """
        X_array = X.values
        
        # Create sequences
        sequences = []
        for i in range(len(X_array) - self.seq_len + 1):
            sequences.append(X_array[i:i+self.seq_len])
        
        return torch.tensor(sequences, dtype=torch.float32).to(self.device)
    
    def load_model(self) -> None:
        """
        Load the model from disk.
        """
        try:
            self.logger.info(f"Loading Transformer model from {self.model_path}")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}")
                return
            
            # Load model
            self.model = self._build_model()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            # Load scaler if available
            scaler_path = os.path.join(os.path.dirname(self.model_path), 'transformer_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.logger.info("Transformer model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading Transformer model: {e}")
    
    def save_model(self) -> None:
        """
        Save the model to disk.
        """
        try:
            self.logger.info(f"Saving Transformer model to {self.model_path}")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model to save")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            torch.save(self.model.state_dict(), self.model_path)
            
            # Save scaler if available
            if self.scaler is not None:
                scaler_path = os.path.join(os.path.dirname(self.model_path), 'transformer_scaler.pkl')
                joblib.dump(self.scaler, scaler_path)
            
            self.logger.info("Transformer model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving Transformer model: {e}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training labels.
        """
        try:
            self.logger.info("Training Transformer model")
            
            # Create sequences
            X_seq = self._create_sequences(X_train)
            
            # Create target tensor
            y_tensor = torch.tensor(y_train.values[self.seq_len-1:], dtype=torch.float32).to(self.device)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_seq, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Build model
            self.model = self._build_model()
            
            # Define optimizer and loss function
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
            
            # Train model
            self.model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    # Forward pass
                    outputs = self.model(batch_X)
                    
                    # Reshape outputs if needed
                    if self.output_dim == 1:
                        outputs = outputs.squeeze()
                    
                    # Calculate loss
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
            
            self.logger.info("Transformer model trained successfully")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            self.logger.error(f"Error training Transformer model: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X (pd.DataFrame): The features to predict on.
            
        Returns:
            np.ndarray: The predictions.
        """
        try:
            self.logger.info("Making predictions with Transformer model")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return np.zeros(len(X))
            
            # Create sequences
            X_seq = self._create_sequences(X)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_seq)
                
                # Apply sigmoid if binary classification
                if self.output_dim == 1:
                    predictions = torch.sigmoid(predictions.squeeze())
            
            # Convert to numpy array
            predictions_np = predictions.cpu().numpy()
            
            # Pad with zeros for the first seq_len-1 samples
            padded_predictions = np.zeros(len(X))
            padded_predictions[self.seq_len-1:] = predictions_np
            
            self.logger.info(f"Made {len(predictions_np)} predictions")
            
            return padded_predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with Transformer model: {e}")
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
            self.logger.info("Evaluating Transformer model")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return {'accuracy': 0.0, 'loss': 0.0}
            
            # Create sequences
            X_seq = self._create_sequences(X_test)
            
            # Create target tensor
            y_tensor = torch.tensor(y_test.values[self.seq_len-1:], dtype=torch.float32).to(self.device)
            
            # Define loss function
            criterion = nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
            
            # Evaluate model
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_seq)
                
                # Reshape outputs if needed
                if self.output_dim == 1:
                    outputs = outputs.squeeze()
                
                # Calculate loss
                loss = criterion(outputs, y_tensor).item()
                
                # Calculate accuracy
                if self.output_dim == 1:
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    accuracy = (predictions == y_tensor).float().mean().item()
                else:
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = (predictions == y_tensor).float().mean().item()
            
            # Calculate metrics
            metrics = {
                'loss': loss,
                'accuracy': accuracy
            }
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating Transformer model: {e}")
            return {'accuracy': 0.0, 'loss': 0.0}
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get the feature importance.
        For transformer models, we use attention weights as a proxy for feature importance.
        
        Returns:
            pd.DataFrame: The feature importance.
        """
        self.logger.warning("Feature importance calculation for Transformer models is not directly available")
        return pd.DataFrame()
    
    def get_attention_weights(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get attention weights for interpretability.
        
        Args:
            X (pd.DataFrame): The features.
            
        Returns:
            Dict[str, np.ndarray]: The attention weights for each layer.
        """
        try:
            self.logger.info("Getting attention weights from Transformer model")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return {}
            
            # Create sequences
            X_seq = self._create_sequences(X)
            
            # Get attention weights
            attention_weights = {}
            
            # This is a simplified version - in a real implementation,
            # you would need to modify the transformer to return attention weights
            self.logger.warning("Attention weight extraction not implemented in this version")
            
            return attention_weights
            
        except Exception as e:
            self.logger.error(f"Error getting attention weights: {e}")
            return {}
