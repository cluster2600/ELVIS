"""Accuracy-weighted adaptive ensemble aggregation (roadmap item #11).

How it works
------------
Each model in the ensemble carries a rolling accuracy score in ``[0, 1]``.
Scores are updated with an exponential moving average (EMA) so that recent
performance dominates while older performance decays::

    new_accuracy = ema_alpha * observed_accuracy + (1 - ema_alpha) * old_accuracy

The per-model *weights* are simply the accuracies normalized to sum to 1.
When aggregating a set of predictions, only the models present in BOTH the
weight table and the prediction dict contribute, and their weights are
re-normalized over that subset so the output stays on the same scale as the
inputs regardless of which models reported.

State can optionally be persisted to a JSON file (atomic write via a
temporary file + ``os.replace``) so weights survive process restarts.
Loading tolerates a missing or corrupt file by falling back to the initial
accuracies — the ensemble is always usable.

How to use it
-------------
>>> weights = AdaptiveEnsembleWeights(
...     {"lstm": 0.55, "rf": 0.60, "xgb": 0.50},
...     ema_alpha=0.7,
...     state_path="state/ensemble_weights.json",  # optional persistence
... )
>>> signal = weights.weighted_signal({"lstm": 1.0, "rf": -1.0, "xgb": 0.0})
>>> weights.update("lstm", accuracy=0.72)  # after evaluating a prediction
>>> weights.save()  # no-op when state_path is None

All entry points degrade gracefully: NaN/inf accuracies or predictions are
ignored (with a warning), empty inputs yield a neutral ``0.0`` signal, and
no method raises on bad runtime data.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile

logger = logging.getLogger(__name__)

__all__ = ["AdaptiveEnsembleWeights"]


class AdaptiveEnsembleWeights:
    """Maintain EMA-updated per-model accuracies and aggregate predictions.

    Args:
        initial_accuracies: Mapping of model name to starting accuracy.
            Values are clamped to ``[0, 1]``; non-finite values become 0.0.
        ema_alpha: EMA smoothing factor in ``[0, 1]``. Higher values react
            faster to new accuracy observations. Out-of-range values are
            clamped (with a warning).
        state_path: Optional JSON file used by :meth:`save`/:meth:`load`.
            When provided, existing state is loaded immediately; a missing
            or corrupt file silently falls back to ``initial_accuracies``.
    """

    def __init__(
        self,
        initial_accuracies: dict[str, float],
        ema_alpha: float = 0.7,
        state_path: str | None = None,
    ) -> None:
        self._initial: dict[str, float] = {
            str(name): self._sanitize_accuracy(value, model=str(name))
            for name, value in dict(initial_accuracies or {}).items()
        }
        if not 0.0 <= ema_alpha <= 1.0 or not math.isfinite(ema_alpha):
            clamped = min(max(ema_alpha, 0.0), 1.0)
            clamped = clamped if math.isfinite(clamped) else 0.7
            logger.warning(
                "ema_alpha=%r outside [0, 1]; clamped to %.3f", ema_alpha, clamped
            )
            ema_alpha = clamped
        self._ema_alpha = float(ema_alpha)
        self._state_path = state_path
        self._accuracies: dict[str, float] = dict(self._initial)
        if state_path is not None:
            self.load()

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    @property
    def accuracies(self) -> dict[str, float]:
        """Current (unnormalized) per-model accuracy scores."""
        return dict(self._accuracies)

    @property
    def weights(self) -> dict[str, float]:
        """Accuracies normalized to sum to 1.

        Returns a uniform distribution when every accuracy is zero, and an
        empty dict when no models are tracked.
        """
        if not self._accuracies:
            return {}
        total = sum(self._accuracies.values())
        if total <= 0.0:
            uniform = 1.0 / len(self._accuracies)
            return {model: uniform for model in self._accuracies}
        return {model: acc / total for model, acc in self._accuracies.items()}

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update(self, model: str, accuracy: float) -> None:
        """EMA-update a model's accuracy; unknown models are added.

        ``new = ema_alpha * accuracy + (1 - ema_alpha) * old``. A model not
        yet tracked is inserted with the observed accuracy directly.
        Non-finite accuracies are ignored with a warning (no-op).
        """
        try:
            observed = float(accuracy)
        except TypeError, ValueError:
            logger.warning("Ignoring non-numeric accuracy %r for %s", accuracy, model)
            return
        if not math.isfinite(observed):
            logger.warning("Ignoring non-finite accuracy %r for %s", accuracy, model)
            return
        observed = self._sanitize_accuracy(observed, model=model)
        old = self._accuracies.get(model)
        if old is None:
            self._accuracies[model] = observed
            logger.info("Added ensemble model %s with accuracy %.4f", model, observed)
            return
        self._accuracies[model] = (
            self._ema_alpha * observed + (1.0 - self._ema_alpha) * old
        )
        logger.debug(
            "EMA update %s: observed=%.4f old=%.4f new=%.4f",
            model,
            observed,
            old,
            self._accuracies[model],
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def weighted_signal(self, predictions: dict[str, float]) -> float:
        """Weight-average predictions over models known to the ensemble.

        Only models present in BOTH :attr:`weights` and ``predictions``
        contribute; weights are re-normalized over that subset. Non-finite
        predictions are skipped. Returns ``0.0`` (neutral) when the
        intersection is empty.
        """
        if not predictions:
            return 0.0
        weights = self.weights
        usable: dict[str, float] = {}
        for model, value in predictions.items():
            if model not in weights:
                continue
            try:
                pred = float(value)
            except TypeError, ValueError:
                logger.warning(
                    "Skipping non-numeric prediction %r for %s", value, model
                )
                continue
            if not math.isfinite(pred):
                logger.warning("Skipping non-finite prediction %r for %s", value, model)
                continue
            usable[model] = pred
        if not usable:
            return 0.0
        subset_total = sum(weights[model] for model in usable)
        if subset_total <= 0.0:
            # Every overlapping weight is zero: fall back to a plain mean.
            return float(sum(usable.values()) / len(usable))
        return float(
            sum(weights[model] * pred for model, pred in usable.items()) / subset_total
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Atomically persist accuracies to ``state_path`` as JSON.

        No-op (with a debug log) when ``state_path`` is None. The write goes
        to a temporary file in the destination directory followed by
        ``os.replace`` so readers never observe a partial file.
        """
        if self._state_path is None:
            logger.debug("save() skipped: no state_path configured")
            return
        payload = {"ema_alpha": self._ema_alpha, "accuracies": self._accuracies}
        directory = os.path.dirname(os.path.abspath(self._state_path))
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".ensemble_weights_", suffix=".json.tmp", dir=directory
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                os.replace(tmp_path, self._state_path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.error(
                "Failed to save ensemble state to %s: %s", self._state_path, exc
            )
            return
        logger.debug("Saved ensemble state to %s", self._state_path)

    def load(self) -> bool:
        """Load accuracies from ``state_path``, merging over the initials.

        Models present in the file override their initial accuracy; models
        only present in the initials are kept (so newly added models are not
        lost when older state is loaded). A missing, unreadable, or corrupt
        file leaves the initial accuracies in place.

        Returns:
            True when state was successfully loaded, False otherwise.
        """
        if self._state_path is None:
            logger.debug("load() skipped: no state_path configured")
            return False
        try:
            with open(self._state_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            loaded = payload["accuracies"]
            if not isinstance(loaded, dict):
                raise TypeError("'accuracies' must be a JSON object")
            merged = dict(self._initial)
            for name, value in loaded.items():
                merged[str(name)] = self._sanitize_accuracy(
                    float(value), model=str(name)
                )
        except FileNotFoundError:
            logger.info(
                "No ensemble state at %s; using initial accuracies", self._state_path
            )
            self._accuracies = dict(self._initial)
            return False
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "Corrupt or unreadable ensemble state at %s (%s); "
                "falling back to initial accuracies",
                self._state_path,
                exc,
            )
            self._accuracies = dict(self._initial)
            return False
        self._accuracies = merged
        logger.debug("Loaded ensemble state from %s", self._state_path)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_accuracy(value: float, model: str = "?") -> float:
        """Clamp an accuracy to ``[0, 1]``; non-finite values become 0.0."""
        try:
            value = float(value)
        except TypeError, ValueError:
            logger.warning("Non-numeric accuracy %r for %s; using 0.0", value, model)
            return 0.0
        if not math.isfinite(value):
            logger.warning("Non-finite accuracy %r for %s; using 0.0", value, model)
            return 0.0
        if value < 0.0 or value > 1.0:
            clamped = min(max(value, 0.0), 1.0)
            logger.warning(
                "Accuracy %.4f for %s outside [0, 1]; clamped to %.1f",
                value,
                model,
                clamped,
            )
            return clamped
        return value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(models={sorted(self._accuracies)}, "
            f"ema_alpha={self._ema_alpha}, state_path={self._state_path!r})"
        )


def combine_weights(
    static: dict[str, float],
    adaptive: dict[str, float],
    min_multiplier: float = 0.25,
    max_multiplier: float = 4.0,
) -> dict[str, float]:
    """Modulate hand-tuned static source weights by learned adaptive weights.

    How it works: with N tracked models the neutral adaptive weight is 1/N,
    so each source's multiplier is ``adaptive[k] / neutral`` — exactly 1.0
    when weights are uniform (zero behavior change until real feedback
    accumulates). Multipliers are clamped to [min_multiplier, max_multiplier]
    so no model can be silenced or dominate off a handful of scored trades.
    Sources unknown to the adaptive tracker get a neutral 1.0. The result is
    renormalized over the sources present in ``static``.

    How to use::

        combined = combine_weights({"Technical": 0.3, "Research": 0.35},
                                   tracker.weights)
    """
    if not static:
        return {}
    n_known = len(adaptive)
    neutral = (1.0 / n_known) if n_known else 1.0
    combined: dict[str, float] = {}
    for source, static_w in static.items():
        adaptive_w = adaptive.get(source)
        if adaptive_w is None or neutral <= 0:
            multiplier = 1.0
        else:
            multiplier = adaptive_w / neutral
        multiplier = max(min_multiplier, min(max_multiplier, multiplier))
        combined[source] = max(0.0, float(static_w)) * multiplier
    total = sum(combined.values())
    if total <= 0:
        # degenerate statics: fall back to uniform over present sources
        uniform = 1.0 / len(combined)
        return {source: uniform for source in combined}
    return {source: weight / total for source, weight in combined.items()}
