import logging
from typing import Dict, List
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from sklearn.ensemble import RandomForestClassifier
from core.models.base_model import BaseModel
import pandas as pd
import joblib

class RandomForestModel(BaseModel):
    """
    A Random Forest model for classification.
    """

    def __init__(self, logger: logging.Logger = None, n_estimators: int = 100, max_depth: int = None):
        """
        Initialize the RandomForestModel.

        Args:
            logger (logging.Logger): The logger to use.
            n_estimators (int): The number of trees in the forest.
            max_depth (int): The maximum depth of the tree.
        """
        super().__init__(logger)
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        """
        Make predictions with the model.
        """
        return self.model.predict(X_test)

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        model = joblib.load(path)
        # We need to return an instance of the class, not the model itself
        instance = cls()
        instance.model = model
        return instance

def push_cv_metrics_to_prometheus(
    metrics: Dict[str, List[float]],
    job_name: str = "cv_metrics",
    gateway: str = "localhost:9091"
) -> None:
    """
    Push mean cross-validation metrics to Prometheus Pushgateway.

    Args:
        metrics (Dict[str, List[float]]): Dictionary of metric names to list of float values.
        job_name (str): Job name for Prometheus Pushgateway.
        gateway (str): Address of the Prometheus Pushgateway.

    This function calculates the mean of each metric's values, registers a Gauge metric
    with the prefix 'rf_' for each metric, and pushes them to the Prometheus Pushgateway.
    Includes error handling with logging warnings and errors.
    """
    registry = CollectorRegistry()
    for metric_name, values in metrics.items():
        try:
            mean_value = sum(values) / len(values) if values else 0.0
        except Exception as e:
            logging.warning(f"Failed to calculate mean for metric '{metric_name}': {e}")
            continue

        gauge_name = f"rf_{metric_name}"
        try:
            gauge = Gauge(gauge_name, f"RandomForest cross-validation metric {metric_name}", registry=registry)
            gauge.set(mean_value)
        except ValueError as ve:
            logging.warning(f"Gauge creation failed for metric '{metric_name}': {ve}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error creating gauge for metric '{metric_name}': {e}")
            continue

    try:
        push_to_gateway(gateway, job=job_name, registry=registry)
    except Exception as e:
        logging.error(f"Failed to push metrics to Prometheus Pushgateway at {gateway}: {e}")
        raise
