"""
Ensemble Learning Model for combining multiple model predictions with confidence scoring.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class EnsembleDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MetaLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class EnsembleModel:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.base_models = []
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize base models
        self._initialize_base_models()
        
    def _initialize_base_models(self):
        """Initialize base models based on configuration."""
        model_configs = self.config.get('base_models', [])
        
        for model_config in model_configs:
            model_type = model_config.get('type')
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    max_depth=model_config.get('max_depth', None),
                    random_state=model_config.get('random_state', 42)
                )
            elif model_type == 'neural_network':
                model = self._create_neural_network(model_config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            self.base_models.append({
                'model': model,
                'type': model_type,
                'config': model_config
            })
            
    def _create_neural_network(self, config: Dict) -> nn.Module:
        """Create a neural network model."""
        input_dim = config.get('input_dim', 64)
        hidden_dim = config.get('hidden_dim', 128)
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ensemble prediction."""
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
        
    def train_base_models(self, data: pd.DataFrame, targets: np.ndarray):
        """Train base models on the data."""
        features = self.prepare_features(data)
        
        for model_info in self.base_models:
            model = model_info['model']
            model_type = model_info['type']
            
            if model_type == 'random_forest':
                model.fit(features, targets)
            elif model_type == 'neural_network':
                self._train_neural_network(model, features, targets)
                
    def _train_neural_network(self, model: nn.Module, features: np.ndarray, targets: np.ndarray):
        """Train a neural network model."""
        dataset = EnsembleDataset(features, targets)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        model.train()
        for epoch in range(10):
            total_loss = 0
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch + 1}/10, Loss: {avg_loss:.4f}")
            
    def train_meta_learner(self, data: pd.DataFrame, targets: np.ndarray):
        """Train the meta-learner on base model predictions."""
        features = self.prepare_features(data)
        
        # Get base model predictions
        base_predictions = []
        for model_info in self.base_models:
            model = model_info['model']
            model_type = model_info['type']
            
            if model_type == 'random_forest':
                preds = model.predict_proba(features)[:, 1]
            elif model_type == 'neural_network':
                model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).to(self.device)
                    preds = model(features_tensor).cpu().numpy().flatten()
                    
            base_predictions.append(preds)
            
        # Combine predictions
        meta_features = np.column_stack(base_predictions)
        
        # Create and train meta-learner
        self.meta_learner = MetaLearner(len(self.base_models))
        self.meta_learner.to(self.device)
        
        dataset = EnsembleDataset(meta_features, targets)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=1e-4)
        
        self.meta_learner.train()
        for epoch in range(10):
            total_loss = 0
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.meta_learner(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            self.logger.info(f"Meta-learner Epoch {epoch + 1}/10, Loss: {avg_loss:.4f}")
            
    def predict(self, data: pd.DataFrame) -> Dict:
        """Make predictions using the ensemble model."""
        features = self.prepare_features(data)
        
        # Get base model predictions
        base_predictions = []
        base_confidences = []
        
        for model_info in self.base_models:
            model = model_info['model']
            model_type = model_info['type']
            
            if model_type == 'random_forest':
                preds = model.predict_proba(features)[:, 1]
                confidences = np.max(model.predict_proba(features), axis=1)
            elif model_type == 'neural_network':
                model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).to(self.device)
                    preds = model(features_tensor).cpu().numpy().flatten()
                    confidences = np.abs(preds - 0.5) * 2  # Convert to confidence score
                    
            base_predictions.append(preds)
            base_confidences.append(confidences)
            
        # Combine predictions
        meta_features = np.column_stack(base_predictions)
        
        # Get meta-learner prediction
        self.meta_learner.eval()
        with torch.no_grad():
            meta_features_tensor = torch.FloatTensor(meta_features).to(self.device)
            ensemble_prediction = self.meta_learner(meta_features_tensor).cpu().numpy().flatten()
            
        # Calculate ensemble confidence
        ensemble_confidence = np.mean(base_confidences, axis=0)
        
        return {
            'prediction': ensemble_prediction,
            'confidence': ensemble_confidence,
            'base_predictions': dict(zip([m['type'] for m in self.base_models], base_predictions)),
            'base_confidences': dict(zip([m['type'] for m in self.base_models], base_confidences))
        }
        
    def save_model(self, path: str):
        """Save the ensemble model."""
        model_state = {
            'base_models': [
                {
                    'type': m['type'],
                    'config': m['config'],
                    'state': m['model'].state_dict() if isinstance(m['model'], nn.Module) else m['model']
                }
                for m in self.base_models
            ],
            'meta_learner': self.meta_learner.state_dict() if self.meta_learner else None,
            'scaler': self.scaler
        }
        
        torch.save(model_state, path)
        
    def load_model(self, path: str):
        """Load a saved ensemble model."""
        model_state = torch.load(path)
        
        # Load base models
        self.base_models = []
        for model_info in model_state['base_models']:
            model_type = model_info['type']
            config = model_info['config']
            
            if model_type == 'random_forest':
                model = model_info['state']
            elif model_type == 'neural_network':
                model = self._create_neural_network(config)
                model.load_state_dict(model_info['state'])
                model.to(self.device)
                
            self.base_models.append({
                'model': model,
                'type': model_type,
                'config': config
            })
            
        # Load meta-learner
        if model_state['meta_learner']:
            self.meta_learner = MetaLearner(len(self.base_models))
            self.meta_learner.load_state_dict(model_state['meta_learner'])
            self.meta_learner.to(self.device)
            
        # Load scaler
        self.scaler = model_state['scaler'] 