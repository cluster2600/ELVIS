"""Walk-forward parameter optimization (roadmap item #15).

How it works
------------
Classic in-sample grid search overfits: parameters tuned on the whole
history look great on that same history and fall apart live.  Walk-forward
optimization guards against this by rolling a train/test window pair
through the data:

    |----- train (fit params) -----|-- test (out-of-sample) --|
             |----- train -----------------|-- test --|
                      ... step forward ...

For every fold the optimizer grid-searches all parameter combinations
(``itertools.product`` over ``param_grid``) on the *train* slice, picks the
combination with the best ``metric`` as reported by ``backtest_fn``, then
re-runs ``backtest_fn`` with that single pick on the *following, unseen
test slice*.  The distribution of out-of-sample test metrics — not the
in-sample train metrics — tells you whether the strategy parameters are
robust.

How to use
----------
>>> import pandas as pd
>>> from trading.optimization.walk_forward import (
...     WalkForwardOptimizer, sma_crossover_backtest,
... )
>>> data = pd.DataFrame({"close": [...]})  # OHLCV or at least a close column
>>> optimizer = WalkForwardOptimizer(
...     backtest_fn=sma_crossover_backtest,
...     param_grid={"sma_short": [5, 10], "sma_long": [20, 50]},
...     metric="sharpe",
... )
>>> report = optimizer.optimize(data, train_window=500, test_window=100)
>>> report["best_params"]       # pick from the most recent fold
>>> report["mean_test_metric"]  # average out-of-sample performance

``backtest_fn`` can be any callable ``(data, params) -> dict`` that returns
a dict of metrics (e.g. ``{"sharpe": 1.2, "total_return": 0.08}``).  A
combination whose result is missing ``metric`` (or errors out) scores
``-inf`` and can never be selected over a valid one.

Only stdlib + numpy + pandas are used, so this module is importable in CI
environments without heavy ML dependencies.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Result type of a backtest callable: metric name -> value.
BacktestFn = Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]]

_NEUTRAL_RESULT: Dict[str, float] = {"sharpe": 0.0, "total_return": 0.0}


def sma_crossover_backtest(
    data: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, float]:
    """Vectorized SMA-crossover backtest usable as a ``backtest_fn``.

    Strategy: be long (position = 1) whenever
    ``SMA(close, params["sma_short"]) > SMA(close, params["sma_long"])``,
    flat otherwise.  The position is shifted by one bar so the signal
    computed on bar *t* earns the close-to-close return of bar *t+1*
    (no look-ahead).

    Metrics returned:

    - ``sharpe``: ``mean(strategy_returns) / std(strategy_returns) *
      sqrt(len(strategy_returns))``; ``0.0`` when the standard deviation
      is zero (e.g. never in the market).
    - ``total_return``: compounded close-to-close strategy return.

    Args:
        data: DataFrame with a ``close`` column (extra columns ignored).
        params: dict with integer ``sma_short`` and ``sma_long`` windows.

    Returns:
        ``{"sharpe": float, "total_return": float}``.  Degenerate input
        (empty/short data, missing ``close`` column, missing or invalid
        window params, all-NaN prices) yields the neutral result
        ``{"sharpe": 0.0, "total_return": 0.0}`` instead of raising.
    """
    short_window = params.get("sma_short")
    long_window = params.get("sma_long")
    if short_window is None or long_window is None:
        logger.warning(
            "sma_crossover_backtest: params must contain 'sma_short' and "
            "'sma_long' (got %s); returning neutral result",
            sorted(params),
        )
        return dict(_NEUTRAL_RESULT)

    try:
        short_window = int(short_window)
        long_window = int(long_window)
    except (TypeError, ValueError):
        logger.warning(
            "sma_crossover_backtest: non-integer SMA windows (%r, %r); "
            "returning neutral result",
            params.get("sma_short"),
            params.get("sma_long"),
        )
        return dict(_NEUTRAL_RESULT)

    if short_window < 1 or long_window < 1:
        logger.warning(
            "sma_crossover_backtest: SMA windows must be >= 1 (got %d, %d); "
            "returning neutral result",
            short_window,
            long_window,
        )
        return dict(_NEUTRAL_RESULT)

    if (
        not isinstance(data, pd.DataFrame)
        or "close" not in data.columns
        or len(data) < 2
    ):
        logger.debug(
            "sma_crossover_backtest: missing 'close' column or fewer than 2 "
            "rows; returning neutral result"
        )
        return dict(_NEUTRAL_RESULT)

    close = pd.to_numeric(data["close"], errors="coerce")
    if close.notna().sum() < 2:
        logger.debug(
            "sma_crossover_backtest: fewer than 2 valid close prices; "
            "returning neutral result"
        )
        return dict(_NEUTRAL_RESULT)

    sma_short = close.rolling(short_window).mean()
    sma_long = close.rolling(long_window).mean()
    # Signal on bar t earns the return of bar t+1 (shift avoids look-ahead).
    position = (sma_short > sma_long).astype(float).shift(1).fillna(0.0)

    # Close-to-close returns, version-proof (no pct_change fill_method).
    returns = close / close.shift(1) - 1.0
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    strategy_returns = position * returns
    std = float(strategy_returns.std())
    if not math.isfinite(std) or std == 0.0:
        sharpe = 0.0
    else:
        sharpe = float(strategy_returns.mean() / std * math.sqrt(len(strategy_returns)))

    total_return = float((1.0 + strategy_returns).prod() - 1.0)
    if not math.isfinite(total_return):
        total_return = 0.0
    if not math.isfinite(sharpe):
        sharpe = 0.0

    return {"sharpe": sharpe, "total_return": total_return}


class WalkForwardOptimizer:
    """Rolling train/test grid search for strategy parameters.

    Args:
        backtest_fn: callable ``(data, params) -> dict`` returning at least
            the metric named by ``metric``.  A result missing that metric
            (or a call that raises) is scored ``-inf`` for that combination.
        param_grid: mapping of parameter name to the list of candidate
            values, e.g. ``{"sma_short": [5, 10], "sma_long": [20, 50]}``.
            The full cartesian product is evaluated on every fold.
        metric: key of ``backtest_fn``'s result dict to maximize
            (default ``"sharpe"``).

    Determinism: combinations are generated with ``itertools.product`` in
    the insertion order of ``param_grid``, and ties keep the earliest
    combination, so repeated runs on the same data give identical output.

    Example:
        >>> optimizer = WalkForwardOptimizer(
        ...     sma_crossover_backtest,
        ...     {"sma_short": [5, 10], "sma_long": [20, 50]},
        ... )
        >>> report = optimizer.optimize(data, train_window=500, test_window=100)
    """

    def __init__(
        self,
        backtest_fn: BacktestFn,
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe",
    ) -> None:
        if not callable(backtest_fn):
            raise TypeError("backtest_fn must be callable")
        if not param_grid or not isinstance(param_grid, dict):
            raise ValueError(
                "param_grid must be a non-empty dict of {name: [values, ...]}"
            )
        normalized: Dict[str, List[Any]] = {}
        for name, values in param_grid.items():
            values = list(values)
            if not values:
                raise ValueError(
                    f"param_grid[{name!r}] must contain at least one value"
                )
            normalized[name] = values
        self.backtest_fn = backtest_fn
        self.param_grid = normalized
        self.metric = metric

    def _param_combinations(self) -> List[Dict[str, Any]]:
        """All grid combinations, in deterministic itertools.product order."""
        names = list(self.param_grid)
        return [
            dict(zip(names, values))
            for values in itertools.product(*(self.param_grid[name] for name in names))
        ]

    def _score(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Run ``backtest_fn`` and extract ``self.metric``; ``-inf`` on failure."""
        try:
            result = self.backtest_fn(data, params)
        except Exception:
            logger.warning(
                "backtest_fn raised for params %s; scoring -inf", params, exc_info=True
            )
            return float("-inf")
        if not isinstance(result, dict) or self.metric not in result:
            logger.warning(
                "backtest_fn result missing metric %r for params %s; scoring -inf",
                self.metric,
                params,
            )
            return float("-inf")
        try:
            value = float(result[self.metric])
        except (TypeError, ValueError):
            logger.warning(
                "backtest_fn metric %r is not numeric (%r) for params %s; scoring -inf",
                self.metric,
                result[self.metric],
                params,
            )
            return float("-inf")
        if math.isnan(value):
            return float("-inf")
        return value

    def optimize(
        self,
        data: pd.DataFrame,
        train_window: int,
        test_window: int,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the walk-forward optimization over ``data``.

        Args:
            data: full history, oldest row first.  Slicing is positional
                (``iloc``), so any index type works.
            train_window: number of rows in each in-sample training slice.
            test_window: number of rows in each out-of-sample test slice.
            step: how many rows to roll forward between folds; defaults to
                ``test_window`` (adjacent, non-overlapping test slices).

        Returns:
            dict with:

            - ``"folds"``: list of per-fold dicts with keys
              ``"train_range"`` (tuple of positional ``(start, end)``,
              end-exclusive), ``"best_params"``, ``"train_metric"`` and
              ``"test_metric"``.
            - ``"best_params"``: best parameters from the **last** fold
              (the most recent data, i.e. what you would trade next).
            - ``"mean_test_metric"``: mean of the out-of-sample metrics.

        Raises:
            ValueError: if the windows are not positive or ``data`` is too
                short for even one train+test fold.
        """
        if step is None:
            step = test_window
        if train_window < 1 or test_window < 1 or step < 1:
            raise ValueError(
                f"train_window ({train_window}), test_window ({test_window}) and "
                f"step ({step}) must all be >= 1"
            )
        n_rows = 0 if data is None else len(data)
        min_rows = train_window + test_window
        if n_rows < min_rows:
            raise ValueError(
                f"Not enough data for one walk-forward fold: need at least "
                f"{min_rows} rows (train_window={train_window} + "
                f"test_window={test_window}), got {n_rows}"
            )

        combinations = self._param_combinations()
        logger.info(
            "Walk-forward optimization: %d rows, train=%d, test=%d, step=%d, "
            "%d parameter combinations",
            n_rows,
            train_window,
            test_window,
            step,
            len(combinations),
        )

        folds: List[Dict[str, Any]] = []
        start = 0
        while start + train_window + test_window <= n_rows:
            train_end = start + train_window
            test_end = train_end + test_window
            train_slice = data.iloc[start:train_end]
            test_slice = data.iloc[train_end:test_end]

            # Grid search on the train slice; ties keep the earliest combo
            # (strict '>') so the search is deterministic.
            best_params = combinations[0]
            best_score = self._score(train_slice, best_params)
            for candidate in combinations[1:]:
                score = self._score(train_slice, candidate)
                if score > best_score:
                    best_score = score
                    best_params = candidate

            test_score = self._score(test_slice, best_params)
            folds.append(
                {
                    "train_range": (start, train_end),
                    "best_params": dict(best_params),
                    "train_metric": best_score,
                    "test_metric": test_score,
                }
            )
            logger.debug(
                "Fold %d: train=[%d:%d) best=%s train_%s=%.6g test_%s=%.6g",
                len(folds),
                start,
                train_end,
                best_params,
                self.metric,
                best_score,
                self.metric,
                test_score,
            )
            start += step

        test_metrics = [fold["test_metric"] for fold in folds]
        return {
            "folds": folds,
            "best_params": dict(folds[-1]["best_params"]),
            "mean_test_metric": float(np.mean(test_metrics)),
        }
