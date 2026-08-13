"""Tests that training modules import when optional ML deps are absent.

The training stack references several optional libraries that are not included
in the canonical ELVIS dependency sets:

    - optuna       (AutoML hyperparameter search)
    - shap         (SHAP explanations)
    - lime         (LIME explanations)

These imports used to be unconditional at module top-level, which broke plain
``import`` / pytest collection in a minimal environment. They are now guarded
behind ``*_AVAILABLE`` flags. These tests assert that the affected modules
import cleanly with those libraries absent and that the explanation entry point
degrades to a warning + no-op instead of raising.

The tests only exercise the guarded (deps-absent) path; if a given library
happens to be installed they skip the assertion that depends on its absence,
so they stay meaningful in a full environment too.
"""

import importlib
import sys
import types

import pytest

# Torch belongs to the ``ml`` extra and is absent from the base CI environment.
# Skip this module there rather than failing collection; the Python 3.14
# contract and CI separately prove that the complete extra resolves on 3.14.
pytest.importorskip(
    "torch", reason="training modules require the ML extra, absent in base CI"
)


def _absent(module_name: str) -> bool:
    """Return True if ``module_name`` cannot be imported in this environment."""
    try:
        importlib.import_module(module_name)
        return False
    except ImportError:
        return True


def test_explainable_ai_imports_without_shap_lime():
    """training.models.explainable_ai imports even when shap/lime are absent."""
    mod = importlib.import_module("training.models.explainable_ai")

    # Guarded flags exist and reflect the real environment.
    assert hasattr(mod, "SHAP_AVAILABLE")
    assert hasattr(mod, "LIME_AVAILABLE")
    assert hasattr(mod, "PLOTLY_AVAILABLE")

    if _absent("shap"):
        assert mod.SHAP_AVAILABLE is False
        assert mod.shap is None
    if _absent("lime"):
        assert mod.LIME_AVAILABLE is False
        assert mod.lime_tabular is None


def test_generate_explanations_is_noop_when_deps_absent(caplog):
    """generate_explanations() warns and returns {} when shap/lime missing."""
    mod = importlib.import_module("training.models.explainable_ai")

    if not _absent("shap") and not _absent("lime"):
        pytest.skip("shap and lime both installed; no-op path not exercised")

    # A model exposing predict() routes to LIME; a bare object routes to SHAP.
    class _SklearnLike:
        def predict(self, x):  # pragma: no cover - never called in no-op path
            return x

    data = [[1.0, 2.0], [3.0, 4.0]]
    names = ["a", "b"]

    with caplog.at_level("WARNING"):
        if _absent("lime"):
            assert mod.generate_explanations(_SklearnLike(), data, names) == {}
        if _absent("shap"):
            assert mod.generate_explanations(object(), data, names) == {}

    assert any("not installed" in rec.message for rec in caplog.records)


def test_shap_explainer_raises_clear_error_when_shap_absent():
    """Instantiating SHAPExplainer without shap raises a clear ImportError."""
    if not _absent("shap"):
        pytest.skip("shap is installed")
    mod = importlib.import_module("training.models.explainable_ai")
    with pytest.raises(ImportError, match="shap"):
        # torch is present locally; the guard fires before any model access.
        mod.SHAPExplainer.__init__(object.__new__(mod.SHAPExplainer), None, ["a"], None)


def test_hyperparameter_optimizer_imports_without_optuna():
    """training.automl.hyperparameter_optimizer imports with optuna absent."""
    mod = importlib.import_module("training.automl.hyperparameter_optimizer")

    assert hasattr(mod, "OPTUNA_AVAILABLE")
    if _absent("optuna"):
        assert mod.OPTUNA_AVAILABLE is False
        assert mod.optuna is None
    assert not hasattr(mod, "TF_AVAILABLE")


def test_model_trainer_optional_import_guards():
    """training.models.model_trainer guards its optional Optuna dependency.

    ``model_trainer`` also pulls in sibling modules that depend on lightgbm /
    xgboost (out of scope for this task and not part of the optional-deps
    contract under test). We stub those siblings so the import exercises only
    the Optuna guard under test.
    """
    # Only stub siblings if their real heavy deps are unavailable; otherwise
    # let the real modules load.
    stubbed = []
    if _absent("lightgbm") or _absent("xgboost"):
        specs = {
            "training.models.ensemble_models": [
                "NeuralEnsemble",
                "StackingEnsemble",
                "WeightedEnsemble",
            ],
            "training.models.rl_agents": ["MultiAgentTradingSystem"],
            "training.models.transformer_models": ["FinancialTransformer"],
        }
        for name, attrs in specs.items():
            if name in sys.modules:
                continue
            stub = types.ModuleType(name)
            for attr in attrs:
                setattr(stub, attr, type(attr, (), {}))
            sys.modules[name] = stub
            stubbed.append(name)

    try:
        mod = importlib.import_module("training.models.model_trainer")
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)

    assert hasattr(mod, "OPTUNA_AVAILABLE")
    assert not hasattr(mod, "TENSORFLOW_AVAILABLE")
    if _absent("optuna"):
        assert mod.OPTUNA_AVAILABLE is False
        assert mod.optuna is None
