"""
Explainable AI tools for trading models.
Implements SHAP values, LIME, and visualization tools for model interpretability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
import shap
from lime import lime_tabular
import torch
from torch import nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelExplainer:
    """Base class for model explanation tools."""
    
    def __init__(self, model: nn.Module, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.logger = logging.getLogger(__name__)
        
    def explain(self, data: np.ndarray) -> Dict:
        """Explain model predictions for given data."""
        raise NotImplementedError
        
    def visualize(self, explanation: Dict, save_path: Optional[str] = None):
        """Visualize model explanations."""
        raise NotImplementedError

class SHAPExplainer(ModelExplainer):
    """SHAP values explainer for trading models."""
    
    def __init__(self, model: nn.Module, feature_names: List[str], background_data: np.ndarray):
        super().__init__(model, feature_names)
        self.background_data = background_data
        self.explainer = shap.DeepExplainer(model, torch.tensor(background_data).float())
        
    def explain(self, data: np.ndarray) -> Dict:
        """Calculate SHAP values for the given data."""
        try:
            # Convert data to tensor
            data_tensor = torch.tensor(data).float()
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(data_tensor)
            
            # Convert to numpy array if it's a list
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
                
            return {
                'shap_values': shap_values,
                'expected_value': self.explainer.expected_value,
                'feature_names': self.feature_names
            }
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {e}")
            raise
            
    def visualize(self, explanation: Dict, save_path: Optional[str] = None):
        """Visualize SHAP values."""
        try:
            # Create summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                explanation['shap_values'],
                features=explanation['feature_names'],
                show=False
            )
            plt.title("Feature Importance (SHAP Values)")
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error visualizing SHAP values: {e}")
            raise

class LIMEExplainer(ModelExplainer):
    """LIME explainer for trading models."""
    
    def __init__(self, 
                 model: nn.Module, 
                 feature_names: List[str],
                 training_data: np.ndarray,
                 mode: str = 'regression'):
        super().__init__(model, feature_names)
        self.training_data = training_data
        self.mode = mode
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True
        )
        
    def explain(self, data: np.ndarray) -> Dict:
        """Generate LIME explanations for the given data."""
        try:
            explanations = []
            for i in range(len(data)):
                exp = self.explainer.explain_instance(
                    data[i],
                    lambda x: self.model(torch.tensor(x).float()).detach().numpy()
                )
                explanations.append(exp)
                
            return {
                'explanations': explanations,
                'feature_names': self.feature_names
            }
        except Exception as e:
            self.logger.error(f"Error generating LIME explanations: {e}")
            raise
            
    def visualize(self, explanation: Dict, save_path: Optional[str] = None):
        """Visualize LIME explanations."""
        try:
            # Create interactive plot using plotly
            fig = make_subplots(rows=len(explanation['explanations']), cols=1,
                              subplot_titles=[f"Explanation {i+1}" for i in range(len(explanation['explanations']))])
            
            for i, exp in enumerate(explanation['explanations']):
                # Get feature importance values
                feature_importance = exp.as_list()
                features = [x[0] for x in feature_importance]
                values = [x[1] for x in feature_importance]
                
                # Add bar plot
                fig.add_trace(
                    go.Bar(
                        x=features,
                        y=values,
                        name=f"Explanation {i+1}"
                    ),
                    row=i+1, col=1
                )
                
            fig.update_layout(
                height=300 * len(explanation['explanations']),
                title_text="LIME Explanations",
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
                
        except Exception as e:
            self.logger.error(f"Error visualizing LIME explanations: {e}")
            raise

class AttentionVisualizer:
    """Visualizer for attention mechanisms in transformer models."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def visualize_attention(self, 
                          input_data: torch.Tensor,
                          layer_idx: int = -1,
                          head_idx: Optional[int] = None,
                          save_path: Optional[str] = None):
        """Visualize attention weights from transformer model."""
        try:
            # Get attention weights
            with torch.no_grad():
                _, attention_weights = self.model(input_data)
                
            # Select specific layer and head if specified
            if layer_idx != -1:
                attention_weights = attention_weights[layer_idx]
            if head_idx is not None:
                attention_weights = attention_weights[head_idx]
                
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                attention_weights.cpu().numpy(),
                cmap='viridis',
                xticklabels=self.model.feature_names,
                yticklabels=self.model.feature_names
            )
            plt.title("Attention Weights Heatmap")
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logging.error(f"Error visualizing attention weights: {e}")
            raise

class DecisionBoundaryVisualizer:
    """Visualizer for model decision boundaries."""
    
    def __init__(self, model: nn.Module, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
    def visualize_decision_boundary(self,
                                  data: np.ndarray,
                                  feature1: str,
                                  feature2: str,
                                  save_path: Optional[str] = None):
        """Visualize decision boundary for two features."""
        try:
            # Get feature indices
            idx1 = self.feature_names.index(feature1)
            idx2 = self.feature_names.index(feature2)
            
            # Create mesh grid
            x_min, x_max = data[:, idx1].min() - 1, data[:, idx1].max() + 1
            y_min, y_max = data[:, idx2].min() - 1, data[:, idx2].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                np.arange(y_min, y_max, 0.1))
            
            # Make predictions
            grid_data = np.zeros((xx.ravel().shape[0], data.shape[1]))
            grid_data[:, idx1] = xx.ravel()
            grid_data[:, idx2] = yy.ravel()
            
            with torch.no_grad():
                predictions = self.model(torch.tensor(grid_data).float())
                predictions = predictions.numpy().reshape(xx.shape)
            
            # Create contour plot
            plt.figure(figsize=(10, 8))
            plt.contourf(xx, yy, predictions, alpha=0.8)
            plt.scatter(data[:, idx1], data[:, idx2], c='red', edgecolors='k')
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.title("Decision Boundary")
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logging.error(f"Error visualizing decision boundary: {e}")
            raise 