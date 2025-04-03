"""
Transformer-based models for time series forecasting in trading.
Implements attention mechanisms and pre-trained transformer models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series forecasting."""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 max_seq_length: int = 100):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_layer = nn.Linear(d_model, 1)  # Predict next price
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer model.
        Returns predictions and attention weights.
        """
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Store attention weights from the last layer
        self.attention_weights = self.transformer_encoder.layers[-1].self_attn.attn
        
        # Output prediction
        output = self.output_layer(x[:, -1, :])  # Use last time step for prediction
        
        return output, self.attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FinancialTransformer(TimeSeriesTransformer):
    """Specialized transformer for financial time series."""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 max_seq_length: int = 100):
        super().__init__(input_dim, d_model, nhead, num_layers, dropout, max_seq_length)
        
        # Additional layers for financial data
        self.technical_embedding = nn.Linear(10, d_model)  # Technical indicators
        self.fundamental_embedding = nn.Linear(5, d_model)  # Fundamental data
        
        # Self-attention for feature importance
        self.feature_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
    def forward(self, 
                price_data: torch.Tensor,
                technical_data: torch.Tensor,
                fundamental_data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multiple input types.
        Returns predictions and attention weights for interpretability.
        """
        # Embed different data types
        price_emb = self.input_embedding(price_data)
        tech_emb = self.technical_embedding(technical_data)
        fund_emb = self.fundamental_embedding(fundamental_data)
        
        # Combine embeddings
        x = price_emb + tech_emb + fund_emb
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Feature importance through self-attention
        attn_output, feature_weights = self.feature_attention(x, x, x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Store attention weights
        self.attention_weights = self.transformer_encoder.layers[-1].self_attn.attn
        
        # Output prediction
        output = self.output_layer(x[:, -1, :])
        
        # Return predictions and attention weights for interpretability
        attention_dict = {
            'feature_importance': feature_weights,
            'temporal_attention': self.attention_weights
        }
        
        return output, attention_dict

def load_pretrained_model(model_path: str) -> FinancialTransformer:
    """Load a pre-trained financial transformer model."""
    try:
        model = torch.load(model_path)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading pre-trained model: {e}")
        raise 