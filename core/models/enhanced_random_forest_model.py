"""
Enhanced Random Forest Model for ELVIS Trading System
Production-grade implementation with MLOps integration, explainability, and monitoring.
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate

warnings.filterwarnings("ignore")

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from core.models.base_model import BaseModel


class EnhancedRandomForestModel(BaseModel):
    """
    Production-grade Random Forest model with full MLOps integration.

    Features:
    - Hyperparameter optimization with Optuna
    - SHAP explainability
    - Drift detection and monitoring
    - Prometheus metrics integration
    - Incremental learning simulation
    - Performance tracking and alerting
    """

    def __init__(
        self,
        logger: logging.Logger = None,
        model_path: str = "models/rf_enhanced",
        enable_optuna: bool = True,
        enable_shap: bool = True,
        enable_monitoring: bool = True,
        prometheus_gateway: str = "localhost:9091",
    ):

        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = model_path
        self.enable_optuna = enable_optuna and OPTUNA_AVAILABLE
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        self.enable_monitoring = enable_monitoring
        self.prometheus_gateway = prometheus_gateway

        # Model components
        self.model = None
        self.feature_names = None
        self.training_data_stats = {}
        self.shap_explainer = None

        # Performance tracking
        self.performance_history = []
        self.prediction_count = 0
        self.last_retrain_time = None
        self.training_samples_count = 0

        # Incremental learning buffer
        self.incremental_buffer = {"X": [], "y": []}
        self.buffer_size = 1000
        self.retrain_threshold = 0.05

        # Monitoring metrics
        if self.enable_monitoring:
            self._setup_prometheus_metrics()

        # Create model directory
        os.makedirs(model_path, exist_ok=True)

        # Initialize with warning about missing packages
        if not OPTUNA_AVAILABLE and enable_optuna:
            self.logger.warning(
                "Optuna not available - hyperparameter optimization disabled"
            )
        if not SHAP_AVAILABLE and enable_shap:
            self.logger.warning("SHAP not available - explainability features disabled")

        self.logger.info("✅ Enhanced Random Forest model initialized")

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring."""
        try:
            self.registry = CollectorRegistry()

            self.prediction_counter = Counter(
                "rf_predictions_total",
                "Total number of RF predictions made",
                registry=self.registry,
            )

            self.accuracy_gauge = Gauge(
                "rf_current_accuracy",
                "Current RF model accuracy",
                registry=self.registry,
            )

            self.feature_importance_gauge = Gauge(
                "rf_feature_importance",
                "Random Forest feature importance scores",
                ["feature_name"],
                registry=self.registry,
            )

            self.training_time_gauge = Gauge(
                "rf_last_training_duration_seconds",
                "Duration of last RF training in seconds",
                registry=self.registry,
            )

        except Exception as e:
            self.logger.warning(f"Failed to setup Prometheus metrics: {e}")
            self.enable_monitoring = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        trial: "optuna.Trial" = None,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ):
        """
        Enhanced training with hyperparameter optimization and validation.

        Args:
            X_train: Training features
            y_train: Training labels
            trial: Optuna trial for hyperparameter optimization
            validation_data: Optional validation data tuple (X_val, y_val)
        """

        start_time = datetime.now()
        self.logger.info(
            f"🚀 Starting enhanced Random Forest training with {len(X_train)} samples"
        )

        self.feature_names = list(X_train.columns)

        # Hyperparameter optimization
        if self.enable_optuna and trial is not None:
            params = self._suggest_hyperparameters(trial)
            self.logger.info("Using Optuna-optimized hyperparameters")
        else:
            params = self._get_default_hyperparameters()
            self.logger.info("Using default hyperparameters")

        # Initialize model with optimized parameters
        self.model = RandomForestClassifier(**params, random_state=42)

        # Train model
        self.model.fit(X_train, y_train)

        # Store training statistics for drift detection
        self._store_training_statistics(X_train, y_train)

        # Initialize SHAP explainer
        if self.enable_shap:
            self._initialize_shap_explainer(X_train.sample(min(100, len(X_train))))

        # Update tracking variables
        self.training_samples_count = len(X_train)
        self.last_retrain_time = datetime.now()
        training_duration = (datetime.now() - start_time).total_seconds()

        # Update Prometheus metrics
        if self.enable_monitoring:
            try:
                self.training_time_gauge.set(training_duration)

                # Update feature importance metrics
                feature_importance = self.get_feature_importance()
                for _, row in feature_importance.iterrows():
                    self.feature_importance_gauge.labels(
                        feature_name=row["feature"]
                    ).set(row["importance"])

            except Exception as e:
                self.logger.warning(f"Failed to update training metrics: {e}")

        self.logger.info(f"✅ Training completed in {training_duration:.2f}s")

        # Validation if provided
        validation_metrics = {}
        if validation_data:
            X_val, y_val = validation_data
            validation_metrics = self.evaluate(X_val, y_val)
            self.logger.info(f"📊 Validation metrics: {validation_metrics}")

        return validation_metrics

    def _suggest_hyperparameters(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Suggest hyperparameters for Optuna optimization."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
        }

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Optimized default hyperparameters for trading."""
        return {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": "balanced",
            "n_jobs": -1,
        }

    def _store_training_statistics(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Store training statistics for drift detection."""
        self.training_data_stats = {
            "feature_means": X_train.mean().to_dict(),
            "feature_stds": X_train.std().to_dict(),
            "feature_mins": X_train.min().to_dict(),
            "feature_maxs": X_train.max().to_dict(),
            "class_distribution": y_train.value_counts().to_dict(),
            "training_time": datetime.now().isoformat(),
            "n_samples": len(X_train),
            "n_features": len(X_train.columns),
        }

    def _initialize_shap_explainer(self, background_data: pd.DataFrame):
        """Initialize SHAP explainer for model interpretability."""
        if not self.enable_shap:
            return

        try:
            self.shap_explainer = shap.TreeExplainer(self.model, background_data)
            self.logger.info("✅ SHAP explainer initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.enable_shap = False

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions with monitoring and caching."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        predictions = self.model.predict(X_test)

        # Update monitoring
        if self.enable_monitoring:
            self.prediction_count += len(X_test)
            try:
                self.prediction_counter.inc(len(X_test))
            except Exception:
                pass

        return predictions

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict_proba(X_test)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
            "recall": float(
                recall_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
        }

        # Add ROC AUC if binary classification
        if len(np.unique(y_test)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))

        # Update performance history
        self.performance_history.append(metrics["accuracy"])

        # Update Prometheus metrics
        if self.enable_monitoring:
            try:
                self.accuracy_gauge.set(metrics["accuracy"])
                self._push_metrics_to_prometheus()
            except Exception as e:
                self.logger.warning(f"Failed to push metrics to Prometheus: {e}")

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: List[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Enhanced cross-validation with time-series awareness."""

        if scoring is None:
            scoring = [
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            ]

        # Use TimeSeriesSplit for time-aware validation
        cv = TimeSeriesSplit(n_splits=cv_folds)

        try:
            cv_results = cross_validate(
                self.model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
            )

            # Log results
            for metric in scoring:
                test_scores = cv_results[f"test_{metric}"]
                self.logger.info(
                    f"CV {metric}: {test_scores.mean():.4f} (+/- {test_scores.std() * 2:.4f})"
                )

            return cv_results

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {}

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance with enhanced analysis."""
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()

        importance_df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                    "rank": range(1, len(self.feature_names) + 1),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Add rank
        importance_df["rank"] = range(1, len(importance_df) + 1)

        return importance_df

    def explain_prediction(
        self, X: pd.DataFrame, max_samples: int = 10
    ) -> Optional[np.ndarray]:
        """Generate SHAP explanations for predictions."""
        if not self.enable_shap or self.shap_explainer is None:
            self.logger.warning("SHAP explainer not available")
            return None

        try:
            # Limit samples for performance
            sample_data = X.head(max_samples)
            shap_values = self.shap_explainer.shap_values(sample_data)

            # Return SHAP values for the positive class if binary classification
            if isinstance(shap_values, list) and len(shap_values) == 2:
                return shap_values[1]  # Positive class

            return shap_values

        except Exception as e:
            self.logger.error(f"SHAP explanation failed: {e}")
            return None

    def partial_fit(self, X_new: pd.DataFrame, y_new: pd.Series):
        """Simulate incremental learning by buffering data."""

        # Add to buffer
        self.incremental_buffer["X"].extend(X_new.to_dict("records"))
        self.incremental_buffer["y"].extend(y_new.tolist())

        self.logger.info(
            f"Added {len(X_new)} samples to incremental buffer (total: {len(self.incremental_buffer['X'])})"
        )

        # Retrain if buffer is full or performance degraded
        if (
            len(self.incremental_buffer["X"]) >= self.buffer_size
            or self._should_retrain()
        ):

            self._retrain_with_buffer()

    def _should_retrain(self) -> bool:
        """Determine if model should be retrained based on performance."""

        # Check if we have enough performance history
        if len(self.performance_history) < 10:
            return False

        # Check for performance degradation
        recent_performance = np.mean(self.performance_history[-5:])
        baseline_performance = np.mean(self.performance_history[:5])

        performance_drop = baseline_performance - recent_performance

        if performance_drop > self.retrain_threshold:
            self.logger.warning(
                f"Performance degradation detected: {performance_drop:.4f}"
            )
            return True

        return False

    def _retrain_with_buffer(self):
        """Retrain model with buffered incremental data."""

        if not self.incremental_buffer["X"]:
            return

        try:
            # Convert buffer to DataFrame
            buffer_df = pd.DataFrame(self.incremental_buffer["X"])
            buffer_labels = pd.Series(self.incremental_buffer["y"])

            # Retrain model
            self.train(buffer_df, buffer_labels)

            # Clear buffer
            self.incremental_buffer = {"X": [], "y": []}

            self.logger.info("✅ Model retrained with incremental data")

        except Exception as e:
            self.logger.error(f"Incremental retraining failed: {e}")

    def _push_metrics_to_prometheus(self):
        """Push metrics to Prometheus pushgateway."""
        try:
            push_to_gateway(
                self.prometheus_gateway, job="rf_model_metrics", registry=self.registry
            )
        except Exception as e:
            self.logger.debug(f"Failed to push to Prometheus: {e}")

    def save(self, path: str = None):
        """Save model with metadata."""
        save_path = path or self.model_path

        # Save model
        model_file = os.path.join(save_path, "rf_model.pkl")
        joblib.dump(self.model, model_file)

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "training_data_stats": self.training_data_stats,
            "performance_history": self.performance_history,
            "training_samples_count": self.training_samples_count,
            "last_retrain_time": (
                self.last_retrain_time.isoformat() if self.last_retrain_time else None
            ),
            "model_params": self.model.get_params() if self.model else {},
            "save_time": datetime.now().isoformat(),
        }

        metadata_file = os.path.join(save_path, "rf_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"✅ Enhanced RF model saved to {save_path}")

    @classmethod
    def load(cls, path: str, logger: logging.Logger = None):
        """Load model with metadata."""
        instance = cls(logger=logger, model_path=path)

        try:
            # Load model
            model_file = os.path.join(path, "rf_model.pkl")
            if os.path.exists(model_file):
                instance.model = joblib.load(model_file)

            # Load metadata
            metadata_file = os.path.join(path, "rf_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                instance.feature_names = metadata.get("feature_names", [])
                instance.training_data_stats = metadata.get("training_data_stats", {})
                instance.performance_history = metadata.get("performance_history", [])
                instance.training_samples_count = metadata.get(
                    "training_samples_count", 0
                )

                if metadata.get("last_retrain_time"):
                    instance.last_retrain_time = datetime.fromisoformat(
                        metadata["last_retrain_time"]
                    )

            if instance.model is not None:
                instance.logger.info(f"✅ Enhanced RF model loaded from {path}")
            else:
                instance.logger.warning(f"No model found at {path}")

        except Exception as e:
            instance.logger.error(f"Failed to load model from {path}: {e}")

        return instance

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters (required by BaseModel)."""
        if self.model is not None:
            return self.model.get_params()
        return {}

    def set_params(self, **params):
        """Set model parameters (required by BaseModel)."""
        if self.model is not None:
            self.model.set_params(**params)
        return self

    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""

        stats = {
            "model_trained": self.model is not None,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "n_estimators": self.model.n_estimators if self.model else 0,
            "training_samples": self.training_samples_count,
            "prediction_count": self.prediction_count,
            "buffer_size": len(self.incremental_buffer["X"]),
            "performance_trend": (
                "improving"
                if len(self.performance_history) > 5
                and np.mean(self.performance_history[-3:])
                > np.mean(self.performance_history[-6:-3])
                else (
                    "stable"
                    if len(self.performance_history) > 5
                    else "insufficient_data"
                )
            ),
            "last_retrain": (
                self.last_retrain_time.isoformat() if self.last_retrain_time else None
            ),
            "current_accuracy": (
                self.performance_history[-1] if self.performance_history else None
            ),
            "feature_importance_available": self.model is not None,
            "shap_available": self.enable_shap and self.shap_explainer is not None,
            "monitoring_enabled": self.enable_monitoring,
        }

        return stats
