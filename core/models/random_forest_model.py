import logging
from typing import Dict, List
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

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
