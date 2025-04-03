"""
Model training and evaluation module for the ELVIS trading system.
Implements training pipelines, cross-validation, and model evaluation metrics.
"""

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
from optuna.integration import TFKerasPruningCallback
import joblib
import logging
from datetime import datetime
import os

from .transformer_models import FinancialTransformer
from .rl_agents import MultiAgentTradingSystem
from .explainable_ai import ModelExplainer, SHAPExplainer, LIMEExplainer

class ModelTrainer:
    """Handles model training, validation, and evaluation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary containing training parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.model_dir = os.path.join(config.get('model_dir', 'models'))
        self.log_dir = os.path.join(config.get('log_dir', 'logs'))
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            data: DataFrame containing features and target
            
        Returns:
            Tuple of (X, y) numpy arrays
        """
        try:
            # Split features and target
            X = data.drop(columns=['target']).values
            y = data['target'].values
            
            # Normalize features
            self.feature_scaler = joblib.load(os.path.join(self.model_dir, 'feature_scaler.joblib'))
            X = self.feature_scaler.transform(X)
            
            return X, y
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def create_data_loaders(self, X: np.ndarray, y: np.ndarray, 
                          batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders for training and validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            batch_size: Batch size for data loading
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader
        
    def train_transformer(self, train_loader: DataLoader, val_loader: DataLoader,
                         model_config: Dict) -> FinancialTransformer:
        """
        Train a transformer model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_config: Model configuration dictionary
            
        Returns:
            Trained FinancialTransformer model
        """
        try:
            # Initialize model
            model = FinancialTransformer(
                input_dim=model_config['input_dim'],
                d_model=model_config['d_model'],
                nhead=model_config['nhead'],
                num_layers=model_config['num_layers'],
                dim_feedforward=model_config['dim_feedforward'],
                dropout=model_config['dropout']
            ).to(self.device)
            
            # Define optimizer and loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
            criterion = torch.nn.MSELoss()
            
            # Training loop
            best_val_loss = float('inf')
            for epoch in range(model_config['epochs']):
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        output = model(batch_X)
                        val_loss += criterion(output, batch_y).item()
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 
                             os.path.join(self.model_dir, 'best_transformer.pth'))
                
                self.logger.info(f"Epoch {epoch+1}/{model_config['epochs']}: "
                               f"Train Loss: {train_loss/len(train_loader):.4f}, "
                               f"Val Loss: {val_loss/len(val_loader):.4f}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training transformer: {str(e)}")
            raise
            
    def train_rl_agents(self, env_config: Dict, agent_config: Dict) -> MultiAgentTradingSystem:
        """
        Train reinforcement learning agents.
        
        Args:
            env_config: Environment configuration dictionary
            agent_config: Agent configuration dictionary
            
        Returns:
            Trained MultiAgentTradingSystem
        """
        try:
            # Initialize multi-agent system
            multi_agent = MultiAgentTradingSystem(env_config, agent_config)
            
            # Train agents
            multi_agent.train(
                total_timesteps=agent_config['total_timesteps'],
                eval_freq=agent_config['eval_freq'],
                n_eval_episodes=agent_config['n_eval_episodes']
            )
            
            # Save trained agents
            multi_agent.save(os.path.join(self.model_dir, 'rl_agents'))
            
            return multi_agent
            
        except Exception as e:
            self.logger.error(f"Error training RL agents: {str(e)}")
            raise
            
    def evaluate_model(self, model: Union[FinancialTransformer, MultiAgentTradingSystem],
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if isinstance(model, FinancialTransformer):
                # Convert to PyTorch tensor
                X_test = torch.FloatTensor(X_test).to(self.device)
                
                # Get predictions
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test).cpu().numpy()
            else:
                # RL agent evaluation
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def explain_model(self, model: Union[FinancialTransformer, MultiAgentTradingSystem],
                     X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Generate model explanations using SHAP and LIME.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary containing explanations
        """
        try:
            explanations = {}
            
            # SHAP explanation
            shap_explainer = SHAPExplainer(model)
            shap_values = shap_explainer.explain(X)
            explanations['shap'] = {
                'values': shap_values,
                'plot': shap_explainer.visualize(shap_values, feature_names)
            }
            
            # LIME explanation
            lime_explainer = LIMEExplainer(model)
            lime_exp = lime_explainer.explain(X[0])  # Explain first sample
            explanations['lime'] = {
                'explanation': lime_exp,
                'plot': lime_explainer.visualize(lime_exp, feature_names)
            }
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Error generating model explanations: {str(e)}")
            raise
            
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                               model_type: str, n_trials: int = 100) -> Dict:
            """
            Optimize model hyperparameters using Optuna.
            
            Args:
                X: Feature matrix
                y: Target vector
                model_type: Type of model to optimize ('transformer' or 'rl')
                n_trials: Number of optimization trials
                
            Returns:
                Dictionary of optimized hyperparameters
            """
            try:
                def objective(trial):
                    if model_type == 'transformer':
                        # Define hyperparameter search space
                        params = {
                            'd_model': trial.suggest_int('d_model', 32, 256),
                            'nhead': trial.suggest_int('nhead', 2, 8),
                            'num_layers': trial.suggest_int('num_layers', 2, 6),
                            'dim_feedforward': trial.suggest_int('dim_feedforward', 128, 1024),
                            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
                        }
                        
                        # Train and evaluate model
                        train_loader, val_loader = self.create_data_loaders(X, y)
                        model = self.train_transformer(train_loader, val_loader, params)
                        metrics = self.evaluate_model(model, X, y)
                        
                        return metrics['mse']
                        
                    else:  # RL optimization
                        params = {
                            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
                            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
                            'batch_size': trial.suggest_int('batch_size', 32, 256)
                        }
                        
                        # Train and evaluate RL agents
                        env_config = self.config['env_config']
                        agent_config = {**self.config['agent_config'], **params}
                        multi_agent = self.train_rl_agents(env_config, agent_config)
                        metrics = self.evaluate_model(multi_agent, X, y)
                        
                        return metrics['mse']
                
                # Run optimization
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=n_trials)
                
                return study.best_params
                
            except Exception as e:
                self.logger.error(f"Error optimizing hyperparameters: {str(e)}")
                raise 