"""
Feature Importance Analyzer for understanding model predictions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for analysis."""
        features = []
        feature_names = []
        
        # Price features
        features.append(data['close'].pct_change())
        feature_names.append('price_return')
        
        features.append(data['close'].pct_change().rolling(5).mean())
        feature_names.append('price_ma_5')
        
        features.append(data['close'].pct_change().rolling(20).mean())
        feature_names.append('price_ma_20')
        
        # Volatility features
        features.append(data['close'].pct_change().rolling(20).std())
        feature_names.append('volatility_20')
        
        features.append(data['high'] / data['low'] - 1)
        feature_names.append('price_range')
        
        # Volume features
        features.append(data['volume'].pct_change())
        feature_names.append('volume_return')
        
        features.append(data['volume'].rolling(20).mean())
        feature_names.append('volume_ma_20')
        
        # Combine features
        features = pd.concat(features, axis=1)
        features = features.dropna()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled, feature_names
        
    def analyze_permutation_importance(self, model: Union[nn.Module, object], 
                                     data: pd.DataFrame, targets: np.ndarray) -> Dict:
        """Analyze feature importance using permutation importance."""
        features, feature_names = self.prepare_features(data)
        self.feature_names = feature_names
        
        if isinstance(model, nn.Module):
            # Convert PyTorch model to scikit-learn compatible function
            def predict_fn(X):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(next(model.parameters()).device)
                    return model(X_tensor).cpu().numpy()
        else:
            predict_fn = model.predict_proba
            
        # Calculate permutation importance
        result = permutation_importance(
            model, features, targets,
            n_repeats=10,
            random_state=42,
            scoring='neg_log_loss'
        )
        
        # Organize results
        importance_scores = result.importances_mean
        importance_std = result.importances_std
        
        return {
            'feature_names': feature_names,
            'importance_scores': importance_scores,
            'importance_std': importance_std
        }
        
    def analyze_shap_values(self, model: Union[nn.Module, object], 
                          data: pd.DataFrame) -> Dict:
        """Analyze feature importance using SHAP values."""
        features, feature_names = self.prepare_features(data)
        self.feature_names = feature_names
        
        if isinstance(model, nn.Module):
            # Convert PyTorch model to SHAP compatible function
            def predict_fn(X):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(next(model.parameters()).device)
                    return model(X_tensor).cpu().numpy()
        else:
            predict_fn = model.predict_proba
            
        # Calculate SHAP values
        explainer = shap.KernelExplainer(predict_fn, features[:100])
        shap_values = explainer.shap_values(features)
        
        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            # For multi-class models
            mean_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            # For binary models
            mean_shap_values = np.abs(shap_values)
            
        mean_shap_values = np.mean(mean_shap_values, axis=0)
        
        return {
            'feature_names': feature_names,
            'shap_values': shap_values,
            'mean_shap_values': mean_shap_values
        }
        
    def analyze_feature_correlations(self, data: pd.DataFrame) -> Dict:
        """Analyze correlations between features."""
        features, feature_names = self.prepare_features(data)
        self.feature_names = feature_names
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(features.T)
        
        return {
            'feature_names': feature_names,
            'correlation_matrix': correlation_matrix
        }
        
    def plot_feature_importance(self, importance_scores: np.ndarray, 
                              importance_std: Optional[np.ndarray] = None,
                              title: str = "Feature Importance") -> None:
        """Plot feature importance scores."""
        plt.figure(figsize=(10, 6))
        
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        
        plt.barh(pos, importance_scores[sorted_idx], xerr=importance_std[sorted_idx] if importance_std is not None else None)
        plt.yticks(pos, [self.feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    def plot_shap_summary(self, shap_values: np.ndarray, 
                         features: np.ndarray) -> None:
        """Plot SHAP summary plot."""
        shap.summary_plot(shap_values, features, 
                         feature_names=self.feature_names,
                         show=False)
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_heatmap(self, correlation_matrix: np.ndarray) -> None:
        """Plot correlation heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   xticklabels=self.feature_names,
                   yticklabels=self.feature_names,
                   cmap='coolwarm',
                   center=0,
                   annot=True,
                   fmt='.2f')
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()
        
    def generate_report(self, data: pd.DataFrame, model: Union[nn.Module, object], 
                       targets: np.ndarray) -> Dict:
        """Generate comprehensive feature analysis report."""
        # Permutation importance
        perm_importance = self.analyze_permutation_importance(model, data, targets)
        
        # SHAP values
        shap_analysis = self.analyze_shap_values(model, data)
        
        # Feature correlations
        correlations = self.analyze_feature_correlations(data)
        
        # Generate plots
        self.plot_feature_importance(
            perm_importance['importance_scores'],
            perm_importance['importance_std'],
            "Permutation Importance"
        )
        
        self.plot_shap_summary(
            shap_analysis['shap_values'],
            self.prepare_features(data)[0]
        )
        
        self.plot_correlation_heatmap(
            correlations['correlation_matrix']
        )
        
        return {
            'permutation_importance': perm_importance,
            'shap_analysis': shap_analysis,
            'correlations': correlations,
            'timestamp': datetime.now().isoformat()
        } 