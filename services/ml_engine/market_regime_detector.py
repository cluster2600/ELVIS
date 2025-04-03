"""
Market Regime Detection using Transformer-based model.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class MarketRegimeDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MarketRegimeDetector(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model architecture
        self.input_size = config.get('input_size', 64)
        self.hidden_size = config.get('hidden_size', 256)
        self.num_layers = config.get('num_layers', 4)
        self.num_heads = config.get('num_heads', 8)
        self.num_regimes = config.get('num_regimes', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=self.num_layers
        )
        
        # Input projection
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        
        # Output layers
        self.regime_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.num_regimes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        batch_size, seq_len, _ = x.size()
        pos_encoding = self._get_positional_encoding(seq_len, self.hidden_size)
        x = x + pos_encoding.unsqueeze(0)
        
        # Transformer encoder
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch_size, hidden_size)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, hidden_size)
        
        # Use last hidden state for classification
        x = x[:, -1, :]
        
        # Regime classification
        logits = self.regime_classifier(x)
        return logits
        
    def _get_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Generate positional encoding."""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

class MarketRegimeManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = MarketRegimeDetector(config)
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime detection."""
        # Calculate technical indicators
        features = []
        
        # Price features
        features.append(data['close'].pct_change())
        features.append(data['close'].pct_change().rolling(5).mean())
        features.append(data['close'].pct_change().rolling(20).mean())
        
        # Volatility features
        features.append(data['close'].pct_change().rolling(20).std())
        features.append(data['high'] / data['low'] - 1)
        
        # Volume features
        features.append(data['volume'].pct_change())
        features.append(data['volume'].rolling(20).mean())
        
        # Combine features
        features = pd.concat(features, axis=1)
        features = features.dropna()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled
        
    def create_sequences(self, features: np.ndarray, sequence_length: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        sequences = []
        labels = []
        
        for i in range(len(features) - sequence_length):
            sequence = features[i:i + sequence_length]
            label = self._get_regime_label(features[i + sequence_length - 1])
            sequences.append(sequence)
            labels.append(label)
            
        return np.array(sequences), np.array(labels)
        
    def _get_regime_label(self, features: np.ndarray) -> int:
        """Get regime label based on feature values."""
        # Simple regime classification based on volatility and trend
        volatility = np.abs(features[3])  # Volatility feature
        trend = features[1]  # Short-term trend
        
        if volatility > 0.02 and trend > 0:
            return 0  # High volatility uptrend
        elif volatility > 0.02 and trend < 0:
            return 1  # High volatility downtrend
        elif volatility <= 0.02 and trend > 0:
            return 2  # Low volatility uptrend
        else:
            return 3  # Low volatility downtrend
            
    def train(self, data: pd.DataFrame, epochs: int = 10, batch_size: int = 32):
        """Train the regime detection model."""
        # Prepare features
        features = self.prepare_features(data)
        sequences, labels = self.create_sequences(features)
        
        # Create dataset and dataloader
        dataset = MarketRegimeDataset(sequences, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
    def detect_regime(self, data: pd.DataFrame) -> Dict:
        """Detect current market regime."""
        self.model.eval()
        
        # Prepare features
        features = self.prepare_features(data)
        sequence = features[-64:]  # Use last 64 time steps
        sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Get regime prediction
        with torch.no_grad():
            logits = self.model(sequence)
            probabilities = torch.softmax(logits, dim=1)
            regime = torch.argmax(probabilities, dim=1).item()
            
        # Get regime description
        regime_descriptions = {
            0: "High Volatility Uptrend",
            1: "High Volatility Downtrend",
            2: "Low Volatility Uptrend",
            3: "Low Volatility Downtrend"
        }
        
        return {
            'regime': regime,
            'description': regime_descriptions[regime],
            'confidence': probabilities[0, regime].item(),
            'probabilities': probabilities[0].tolist()
        }
        
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler
        }, path)
        
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler'] 