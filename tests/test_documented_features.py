"""Tests for capabilities the docs promise (built to match the documentation).

Covers: documented import paths, the export helpers, the unified trading config
loader, and the RandomForest explainability/tuning methods.
"""
import os
import numpy as np
import pandas as pd
import pytest


def test_documented_import_paths():
    # docs reference these paths; they must resolve to the real classes
    from trading.risk.advanced_risk_manager import AdvancedRiskManager
    from trading.risk.risk_manager import AdvancedRiskManager as Real
    assert AdvancedRiskManager is Real

    from core.features.feature_pipeline import FeaturePipeline
    from core.models.features.feature_pipeline import FeaturePipeline as RealFP
    assert FeaturePipeline is RealFP


def test_unified_trading_config_has_documented_sections():
    from config.trading_config import load_trading_config
    cfg = load_trading_config()
    for section in ("trading", "risk_management", "execution", "monitoring"):
        assert section in cfg, f"missing documented section {section}"
    assert cfg["risk_management"]["max_position_size"] == 0.1
    assert cfg["monitoring"]["grafana_port"] == 3001


def test_default_leverage_env_override():
    from config.trading_config import load_trading_config
    os.environ["DEFAULT_LEVERAGE"] = "7"
    try:
        assert load_trading_config()["trading"]["default_leverage"] == 7
    finally:
        del os.environ["DEFAULT_LEVERAGE"]


def test_export_utils_csv_and_prometheus():
    from core.viz.export_utils import export_to_csv, push_metrics_to_prometheus
    p = export_to_csv([{"a": 1, "b": 2}, {"a": 3, "b": 4}], "/tmp/_elvis_test_export.csv")
    assert open(p).read().splitlines()[0] == "a,b"
    # no gateway running -> returns False, does not raise
    assert push_metrics_to_prometheus({"x": 1.0}, "test", gateway="127.0.0.1:1") is False


def _fitted_rf():
    from core.models.random_forest_model import RandomForestModel
    X = pd.DataFrame(np.random.RandomState(0).randn(120, 4), columns=list("abcd"))
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))
    m = RandomForestModel(n_estimators=25)
    m.train(X, y)
    return m, X, y


def test_rf_explain_predictions():
    m, X, _ = _fitted_rf()
    df = m.explain_predictions(X)
    assert "feature" in df.columns and len(df) == 4
    # falls back to importances when shap is absent; column is one of the two
    assert set(df.columns) & {"importance", "mean_abs_shap"}


def test_rf_tune_hyperparameters_applies_params():
    m, X, y = _fitted_rf()
    best = m.tune_hyperparameters(X, y, n_trials=3)
    assert "n_estimators" in best
    assert m.model.get_params()["n_estimators"] == best["n_estimators"]


def test_export_feature_importance_sorted():
    from core.viz.export_utils import export_feature_importance
    m, X, _ = _fitted_rf()
    p = export_feature_importance(m.model, list(X.columns), "/tmp/_elvis_fi.csv")
    rows = open(p).read().splitlines()
    assert rows[0] == "feature,importance"
    vals = [float(r.split(",")[1]) for r in rows[1:]]
    assert vals == sorted(vals, reverse=True)
