"""Export helpers documented in docs/random_forest.md: CSV, Prometheus, SHAP.

CSV and Prometheus are always available; SHAP is optional (no wheels on some
Python versions) and degrades to feature-importance export when absent.
"""
import csv
import logging
from typing import Any, Dict, Sequence

logger = logging.getLogger(__name__)


def export_to_csv(rows: Sequence[Dict[str, Any]], path: str) -> str:
    """Write a list of dict rows to CSV. Returns the path."""
    if not rows:
        open(path, "w").close()
        return path
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def push_metrics_to_prometheus(metrics: Dict[str, float], job_name: str,
                               gateway: str = "localhost:9091", prefix: str = "rf_") -> bool:
    """Push a dict of metrics to a Prometheus Pushgateway. Returns success."""
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        reg = CollectorRegistry()
        for k, v in metrics.items():
            Gauge(f"{prefix}{k}", f"{k} metric", registry=reg).set(float(v))
        push_to_gateway(gateway, job=job_name, registry=reg)
        return True
    except Exception as e:
        logger.warning(f"Prometheus push failed ({gateway}): {e}")
        return False


def export_feature_importance(model, feature_names: Sequence[str], path: str) -> str:
    """Export a fitted model's feature_importances_ to CSV (SHAP fallback)."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        raise ValueError("model has no feature_importances_")
    rows = [{"feature": n, "importance": float(i)} for n, i in zip(feature_names, imp)]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return export_to_csv(rows, path)


def export_shap_summary(model, X, path: str, feature_names=None) -> str:
    """Export SHAP values if `shap` is installed, else fall back to importances."""
    try:
        import shap  # optional dependency
    except ImportError:
        logger.info("shap not installed - exporting feature importances instead")
        names = feature_names if feature_names is not None else list(getattr(X, "columns", range(X.shape[1])))
        return export_feature_importance(model, names, path)
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)
    import numpy as np
    mean_abs = np.abs(np.asarray(vals)).mean(axis=tuple(range(np.asarray(vals).ndim - 1)))
    names = feature_names if feature_names is not None else list(getattr(X, "columns", range(len(mean_abs))))
    rows = [{"feature": n, "mean_abs_shap": float(v)} for n, v in zip(names, mean_abs)]
    rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
    return export_to_csv(rows, path)
