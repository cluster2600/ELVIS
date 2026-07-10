"""Tests for :mod:`utils.experiment_tracker`.

These tests exercise the JSON fallback path and therefore pass whether or not
MLflow is installed. ``use_mlflow=False`` forces the local JSONL backend so the
behaviour is deterministic in a minimal environment (mlflow has no py3.14 wheel
in CI). A separate test asserts the module imports cleanly with mlflow absent.
"""

import json
import sys

import pytest

from utils.experiment_tracker import ExperimentTracker


def _read_runs(path):
    """Parse a JSONL experiments file into a list of run dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_module_imports_without_mlflow(monkeypatch):
    """The module (and HAS_MLFLOW flag) works when mlflow is not importable."""
    # Simulate a minimal environment where mlflow cannot be imported.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    import importlib

    import utils.experiment_tracker as et

    reloaded = importlib.reload(et)
    try:
        assert reloaded.HAS_MLFLOW is False
        tracker = reloaded.ExperimentTracker()
        assert tracker.use_mlflow is False
    finally:
        # Restore the module so other tests see the real import state.
        monkeypatch.undo()
        importlib.reload(reloaded)


def test_fallback_writes_full_run(tmp_path):
    """A complete run lands as one JSON line with params/metrics/artifacts."""
    out = tmp_path / "experiments.jsonl"
    tracker = ExperimentTracker(fallback_path=out, use_mlflow=False)

    tracker.start_run("rf_baseline")
    tracker.log_params({"n_estimators": 200, "max_depth": 8})
    tracker.log_metrics({"accuracy": 0.71, "f1": 0.68})
    tracker.log_artifact("models/model_rf.joblib")
    tracker.end_run()

    runs = _read_runs(out)
    assert len(runs) == 1
    run = runs[0]
    assert run["run_name"] == "rf_baseline"
    assert run["backend"] == "jsonl"
    assert run["params"] == {"n_estimators": 200, "max_depth": 8}
    assert run["metrics"] == {"accuracy": 0.71, "f1": 0.68}
    assert run["artifacts"] == ["models/model_rf.joblib"]
    assert "start_time" in run and "end_time" in run
    assert isinstance(run["duration_seconds"], (int, float))


def test_multiple_runs_append(tmp_path):
    """Each run appends a new line rather than overwriting the file."""
    out = tmp_path / "experiments.jsonl"
    tracker = ExperimentTracker(fallback_path=out, use_mlflow=False)

    for i in range(3):
        tracker.start_run(f"run_{i}")
        tracker.log_metrics({"step": i})
        tracker.end_run()

    runs = _read_runs(out)
    assert [r["run_name"] for r in runs] == ["run_0", "run_1", "run_2"]
    assert [r["metrics"]["step"] for r in runs] == [0, 1, 2]


def test_context_manager(tmp_path):
    """The tracker works as a context manager and flushes on exit."""
    out = tmp_path / "experiments.jsonl"
    with ExperimentTracker(fallback_path=out, use_mlflow=False) as tracker:
        tracker.log_params({"lr": 0.01})

    runs = _read_runs(out)
    assert len(runs) == 1
    assert runs[0]["params"] == {"lr": 0.01}


def test_log_before_start_raises(tmp_path):
    """Logging before start_run is a programming error, not silent no-op."""
    tracker = ExperimentTracker(fallback_path=tmp_path / "e.jsonl", use_mlflow=False)
    with pytest.raises(RuntimeError):
        tracker.log_params({"a": 1})
    with pytest.raises(RuntimeError):
        tracker.log_metrics({"a": 1})
    with pytest.raises(RuntimeError):
        tracker.log_artifact("x")


def test_end_run_without_start_is_noop(tmp_path):
    """Calling end_run with no active run neither raises nor writes a file."""
    out = tmp_path / "experiments.jsonl"
    tracker = ExperimentTracker(fallback_path=out, use_mlflow=False)
    tracker.end_run()  # should be a no-op
    assert not out.exists()


def test_start_run_closes_previous(tmp_path):
    """Starting a new run auto-closes a still-open previous run."""
    out = tmp_path / "experiments.jsonl"
    tracker = ExperimentTracker(fallback_path=out, use_mlflow=False)
    tracker.start_run("first")
    tracker.log_metrics({"x": 1})
    tracker.start_run("second")  # implicitly ends "first"
    tracker.log_metrics({"x": 2})
    tracker.end_run()

    runs = _read_runs(out)
    assert [r["run_name"] for r in runs] == ["first", "second"]
