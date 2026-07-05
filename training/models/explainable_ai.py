"""
Explainable AI tools for trading models.
Implements SHAP values, LIME, and visualization tools for model interpretability.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn

logger = logging.getLogger(__name__)

# Optional dependencies. These have no Python 3.14 wheels (or are heavy extras)
# and are absent in CI / minimal environments. Guard them so this module always
# imports; the explainer classes and generate_explanations() degrade gracefully
# when the corresponding library is missing.
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in deps-absent CI
    shap = None
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular

    LIME_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in deps-absent CI
    lime_tabular = None
    LIME_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in deps-absent CI
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False


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

    def __init__(
        self, model: nn.Module, feature_names: List[str], background_data: np.ndarray
    ):
        super().__init__(model, feature_names)
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAPExplainer requires the 'shap' package, which is not "
                "installed. Install it with `pip install shap` to enable "
                "SHAP-based explanations."
            )
        self.background_data = background_data

        # Get model device
        self.device = next(model.parameters()).device

        # Convert background data to tensor on the same device as model
        background_tensor = torch.from_numpy(background_data).float().to(self.device)
        self.explainer = shap.DeepExplainer(model, background_tensor)

    def explain(self, data: np.ndarray) -> Dict:
        """Calculate SHAP values for the given data."""
        try:
            # Convert data to tensor on the same device as model
            data_tensor = torch.from_numpy(data).float().to(self.device)

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(data_tensor)

            # Convert tensor/list to numpy array
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.cpu().detach().numpy()
            elif isinstance(shap_values, list):
                # Handle list of tensors
                shap_values = [
                    sv.cpu().detach().numpy() if isinstance(sv, torch.Tensor) else sv
                    for sv in shap_values
                ]
                shap_values = np.array(shap_values)

            return {
                "shap_values": shap_values,
                "expected_value": self.explainer.expected_value,
                "feature_names": self.feature_names,
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
                explanation["shap_values"],
                features=explanation["feature_names"],
                show=False,
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

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        training_data: np.ndarray,
        mode: str = "regression",
    ):
        super().__init__(model, feature_names)
        if not LIME_AVAILABLE:
            raise ImportError(
                "LIMEExplainer requires the 'lime' package, which is not "
                "installed. Install it with `pip install lime` to enable "
                "LIME-based explanations."
            )
        self.training_data = training_data
        self.mode = mode
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True,
        )

    def explain(self, data: np.ndarray) -> Dict:
        """Generate LIME explanations for the given data."""
        try:
            explanations = []

            def model_predict_fn(x):
                """Prediction function for LIME that handles tensor conversion."""
                try:
                    if hasattr(self.model, "predict"):
                        # For sklearn-like models
                        return self.model.predict(x)
                    elif hasattr(self.model, "predict_proba"):
                        # For sklearn-like models with probabilities
                        return self.model.predict_proba(x)
                    else:
                        # For PyTorch models - handle device placement
                        device = next(self.model.parameters()).device
                        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            predictions = self.model(x_tensor)
                            if isinstance(predictions, torch.Tensor):
                                return predictions.cpu().numpy()
                            return predictions
                except Exception as e:
                    self.logger.warning(f"Error in model prediction: {e}")
                    # Return zeros as fallback
                    return np.zeros((x.shape[0], 1))

            for i in range(len(data)):
                exp = self.explainer.explain_instance(data[i], model_predict_fn)
                explanations.append(exp)

            return {"explanations": explanations, "feature_names": self.feature_names}
        except Exception as e:
            self.logger.error(f"Error generating LIME explanations: {e}")
            raise

    def visualize(self, explanation: Dict, save_path: Optional[str] = None):
        """Visualize LIME explanations."""
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Visualizing LIME explanations requires the 'plotly' package, "
                "which is not installed. Install it with `pip install plotly`."
            )
        try:
            # Create interactive plot using plotly
            fig = make_subplots(
                rows=len(explanation["explanations"]),
                cols=1,
                subplot_titles=[
                    f"Explanation {i+1}"
                    for i in range(len(explanation["explanations"]))
                ],
            )

            for i, exp in enumerate(explanation["explanations"]):
                # Get feature importance values
                feature_importance = exp.as_list()
                features = [x[0] for x in feature_importance]
                values = [x[1] for x in feature_importance]

                # Add bar plot
                fig.add_trace(
                    go.Bar(x=features, y=values, name=f"Explanation {i+1}"),
                    row=i + 1,
                    col=1,
                )

            fig.update_layout(
                height=300 * len(explanation["explanations"]),
                title_text="LIME Explanations",
                showlegend=False,
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

    def visualize_attention(
        self,
        input_data: torch.Tensor,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
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
                cmap="viridis",
                xticklabels=self.model.feature_names,
                yticklabels=self.model.feature_names,
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

    def visualize_decision_boundary(
        self,
        data: np.ndarray,
        feature1: str,
        feature2: str,
        save_path: Optional[str] = None,
    ):
        """Visualize decision boundary for two features."""
        try:
            # Get feature indices
            idx1 = self.feature_names.index(feature1)
            idx2 = self.feature_names.index(feature2)

            # Create mesh grid
            x_min, x_max = data[:, idx1].min() - 1, data[:, idx1].max() + 1
            y_min, y_max = data[:, idx2].min() - 1, data[:, idx2].max() + 1
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
            )

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
            plt.scatter(data[:, idx1], data[:, idx2], c="red", edgecolors="k")
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


def generate_explanations(
    model,
    data: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """Generate model explanations, degrading to a no-op when deps are absent.

    Chooses SHAP for tensor/PyTorch-style models and LIME for sklearn-style
    models (those exposing ``predict``/``predict_proba``). If neither the
    required backend (``shap``/``lime``) is installed, this logs a warning and
    returns an empty dict instead of raising, so training pipelines keep
    running in minimal / CI environments where those extras are unavailable.

    Args:
        model: The model to explain.
        data (np.ndarray): Feature matrix to explain.
        feature_names (list): Feature names matching ``data`` columns.

    Returns:
        dict: Explanation results, or ``{}`` when explanations are skipped.
    """
    is_sklearn_like = hasattr(model, "predict") or hasattr(model, "predict_proba")

    if is_sklearn_like and not LIME_AVAILABLE:
        logger.warning(
            "Skipping model explanations: the 'lime' package is not installed. "
            "Install it with `pip install lime` to enable LIME explanations."
        )
        return {}
    if not is_sklearn_like and not SHAP_AVAILABLE:
        logger.warning(
            "Skipping model explanations: the 'shap' package is not installed. "
            "Install it with `pip install shap` to enable SHAP explanations."
        )
        return {}

    data = np.asarray(data)
    if is_sklearn_like:
        explainer = LIMEExplainer(model, feature_names, data)
    else:
        explainer = SHAPExplainer(model, feature_names, data[:100])
    return explainer.explain(data)
