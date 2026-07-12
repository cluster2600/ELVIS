"""Optional-dependency guarding for the ensemble base models.

xgboost and lightgbm used to be unguarded top-level imports, so importing
``training.models.ensemble_models`` (and therefore the whole
``training.train_models`` pipeline) crashed in any environment without them.
These tests lock in the graceful-degradation behavior.
"""

import pytest

pytest.importorskip("torch", reason="ensemble_models imports torch")

import training.models.ensemble_models as em


def test_module_imports_and_exposes_guarded_boosters():
    # Guarded imports mean lgb/xgb are either the module or None — but the
    # module always imports, which is what unblocks the training pipeline.
    assert hasattr(em, "lgb")
    assert hasattr(em, "xgb")


def test_base_regressors_degrades_when_boosters_absent(monkeypatch):
    monkeypatch.setattr(em, "xgb", None)
    monkeypatch.setattr(em, "lgb", None)
    models = em._base_regressors({})
    # Only the two always-present sklearn regressors remain.
    assert len(models) == 2
    assert {type(m).__name__ for m in models} == {
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }


def test_base_regressors_includes_boosters_when_present():
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    names = {type(m).__name__ for m in em._base_regressors({})}
    assert "XGBRegressor" in names
    assert "LGBMRegressor" in names


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
