"""Experiment tracking for the ELVIS Trading Bot.

Provides a small, dependency-optional :class:`ExperimentTracker`.

If `mlflow <https://mlflow.org>`_ is importable, params/metrics/artifacts are
forwarded to MLflow (respecting ``MLFLOW_TRACKING_URI`` and any active
experiment). Otherwise the tracker degrades gracefully to a **no-op local
fallback** that appends one JSON object per run to ``models/experiments.jsonl``
so experiment tracking always works, even in a minimal environment without
MLflow installed.

The public API is intentionally tiny and mirrors the subset of MLflow most
training loops need::

    tracker = ExperimentTracker()
    tracker.start_run("rf_baseline")
    tracker.log_params({"n_estimators": 200, "max_depth": 8})
    tracker.log_metrics({"accuracy": 0.71, "f1": 0.68})
    tracker.log_artifact("models/model_rf.joblib")
    tracker.end_run()

This module is standalone and importable without any heavy dependency; it is
deliberately **not** wired into the training pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:  # mlflow has no guaranteed wheel in every environment (e.g. CI / py3.14).
    import mlflow  # type: ignore

    HAS_MLFLOW = True
except ImportError:  # pragma: no cover - exercised via the JSON fallback path.
    mlflow = None  # type: ignore
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_PATH = Path("models") / "experiments.jsonl"


class ExperimentTracker:
    """Log experiment params/metrics/artifacts to MLflow or a JSONL fallback.

    Args:
        fallback_path: Where to append runs when MLflow is unavailable.
            Defaults to ``models/experiments.jsonl``.
        use_mlflow: Force-disable MLflow even when it is importable by passing
            ``False``. Defaults to ``True`` (use MLflow if available).
    """

    def __init__(
        self,
        fallback_path: os.PathLike | str | None = None,
        use_mlflow: bool = True,
    ) -> None:
        self.fallback_path = Path(fallback_path or DEFAULT_FALLBACK_PATH)
        self.use_mlflow = bool(use_mlflow and HAS_MLFLOW)
        self._active = False
        # In-memory record for the JSON fallback (rebuilt on every start_run).
        self._record: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------

    def start_run(self, name: Optional[str] = None) -> "ExperimentTracker":
        """Begin a new run. Idempotent-safe: ends any run already active."""
        if self._active:
            # Never leak a dangling run; close the previous one first.
            self.end_run()

        self._active = True
        self._record = {
            "run_name": name,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "params": {},
            "metrics": {},
            "artifacts": [],
            "backend": "mlflow" if self.use_mlflow else "jsonl",
        }
        self._start_perf = time.perf_counter()

        if self.use_mlflow:
            try:
                mlflow.start_run(run_name=name)
                return self
            except Exception as exc:  # pragma: no cover - needs live mlflow.
                logger.warning("mlflow.start_run failed (%s); using JSON fallback", exc)
                self.use_mlflow = False
                self._record["backend"] = "jsonl"

        logger.debug("ExperimentTracker started run %r (JSON fallback)", name)
        return self

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a mapping of hyperparameters for the active run."""
        if not self._active:
            raise RuntimeError("log_params called before start_run")
        self._record["params"].update(params)
        if self.use_mlflow:
            try:
                mlflow.log_params(params)
            except Exception as exc:  # pragma: no cover - needs live mlflow.
                logger.warning("mlflow.log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log a mapping of numeric metrics for the active run."""
        if not self._active:
            raise RuntimeError("log_metrics called before start_run")
        self._record["metrics"].update(metrics)
        if self.use_mlflow:
            try:
                mlflow.log_metrics(metrics)
            except Exception as exc:  # pragma: no cover - needs live mlflow.
                logger.warning("mlflow.log_metrics failed: %s", exc)

    def log_artifact(self, path: os.PathLike | str) -> None:
        """Record an artifact path for the active run."""
        if not self._active:
            raise RuntimeError("log_artifact called before start_run")
        self._record["artifacts"].append(str(path))
        if self.use_mlflow:
            try:
                mlflow.log_artifact(str(path))
            except Exception as exc:  # pragma: no cover - needs live mlflow.
                logger.warning("mlflow.log_artifact failed: %s", exc)

    def end_run(self) -> None:
        """End the active run, flushing the JSON fallback record if needed.

        Safe to call when no run is active (no-op).
        """
        if not self._active:
            return
        self._active = False

        if self.use_mlflow:
            try:
                mlflow.end_run()
                return
            except Exception as exc:  # pragma: no cover - needs live mlflow.
                logger.warning("mlflow.end_run failed (%s); writing JSON fallback", exc)
                self.use_mlflow = False
                self._record["backend"] = "jsonl"

        self._record["end_time"] = datetime.now(timezone.utc).isoformat()
        self._record["duration_seconds"] = round(
            time.perf_counter() - getattr(self, "_start_perf", time.perf_counter()),
            6,
        )
        self._append_fallback(self._record)

    # -- context manager sugar --------------------------------------------

    def __enter__(self) -> "ExperimentTracker":
        if not self._active:
            self.start_run()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.end_run()

    # -- internals ---------------------------------------------------------

    def _append_fallback(self, record: dict[str, Any]) -> None:
        """Append one JSON object (one line) to the fallback file."""
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.fallback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        logger.debug("ExperimentTracker wrote run to %s", self.fallback_path)
