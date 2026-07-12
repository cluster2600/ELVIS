"""Tests for trading.optimization.walk_forward (roadmap item #15).

Covers the WalkForwardOptimizer fold mechanics (train/test slicing,
out-of-sample evaluation, determinism, guards) and the built-in
sma_crossover_backtest, including short/NaN/degenerate inputs.
"""

import math

import numpy as np
import pandas as pd
import pytest

from trading.optimization.walk_forward import (
    WalkForwardOptimizer,
    sma_crossover_backtest,
)


def make_trend_data(
    n: int = 200, slope: float = 1.0, start: float = 100.0
) -> pd.DataFrame:
    """Deterministic monotonic uptrend close series."""
    close = start + slope * np.arange(n, dtype=float)
    return pd.DataFrame({"close": close})


def make_noisy_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Seeded random-walk close series (deterministic across runs)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame({"close": close})


# ---------------------------------------------------------------------------
# sma_crossover_backtest
# ---------------------------------------------------------------------------


class TestSmaCrossoverBacktest:
    def test_uptrend_is_profitable(self):
        data = make_trend_data(n=200)
        result = sma_crossover_backtest(data, {"sma_short": 5, "sma_long": 20})
        assert result["total_return"] > 0
        assert result["sharpe"] > 0

    def test_returns_expected_keys_and_floats(self):
        data = make_noisy_data(n=100)
        result = sma_crossover_backtest(data, {"sma_short": 5, "sma_long": 20})
        assert set(result) == {"sharpe", "total_return"}
        assert all(isinstance(value, float) for value in result.values())
        assert all(math.isfinite(value) for value in result.values())

    def test_empty_data_is_neutral(self):
        result = sma_crossover_backtest(
            pd.DataFrame(), {"sma_short": 5, "sma_long": 20}
        )
        assert result == {"sharpe": 0.0, "total_return": 0.0}

    def test_short_data_is_neutral(self):
        data = make_trend_data(n=3)
        result = sma_crossover_backtest(data, {"sma_short": 5, "sma_long": 20})
        assert result == {"sharpe": 0.0, "total_return": 0.0}

    def test_single_row_is_neutral(self):
        data = pd.DataFrame({"close": [100.0]})
        result = sma_crossover_backtest(data, {"sma_short": 2, "sma_long": 3})
        assert result == {"sharpe": 0.0, "total_return": 0.0}

    def test_missing_close_column_is_neutral(self):
        data = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        result = sma_crossover_backtest(data, {"sma_short": 2, "sma_long": 3})
        assert result == {"sharpe": 0.0, "total_return": 0.0}

    def test_nan_close_values_do_not_raise(self):
        data = make_trend_data(n=60)
        data.loc[10:20, "close"] = np.nan
        result = sma_crossover_backtest(data, {"sma_short": 3, "sma_long": 10})
        assert math.isfinite(result["sharpe"])
        assert math.isfinite(result["total_return"])

    def test_all_nan_close_is_neutral(self):
        data = pd.DataFrame({"close": [np.nan] * 30})
        result = sma_crossover_backtest(data, {"sma_short": 2, "sma_long": 5})
        assert result == {"sharpe": 0.0, "total_return": 0.0}

    def test_missing_params_are_neutral(self):
        data = make_trend_data(n=50)
        assert sma_crossover_backtest(data, {}) == {"sharpe": 0.0, "total_return": 0.0}
        assert sma_crossover_backtest(data, {"sma_short": 5}) == {
            "sharpe": 0.0,
            "total_return": 0.0,
        }

    def test_invalid_window_values_are_neutral(self):
        data = make_trend_data(n=50)
        neutral = {"sharpe": 0.0, "total_return": 0.0}
        assert sma_crossover_backtest(data, {"sma_short": 0, "sma_long": 10}) == neutral
        assert (
            sma_crossover_backtest(data, {"sma_short": -3, "sma_long": 10}) == neutral
        )
        assert (
            sma_crossover_backtest(data, {"sma_short": "abc", "sma_long": 10})
            == neutral
        )

    def test_constant_price_has_zero_sharpe(self):
        data = pd.DataFrame({"close": [100.0] * 50})
        result = sma_crossover_backtest(data, {"sma_short": 3, "sma_long": 10})
        assert result == {"sharpe": 0.0, "total_return": 0.0}


# ---------------------------------------------------------------------------
# WalkForwardOptimizer
# ---------------------------------------------------------------------------


class TestWalkForwardOptimizer:
    def test_known_best_combo_wins_on_trend(self):
        # On a monotonic uptrend the crossover goes long once SMA(long) has
        # warmed up, so the fastest-warming combo (2, 3) is in the market
        # longest and must win on total_return over the slow (10, 50) combo.
        data = make_trend_data(n=240)
        optimizer = WalkForwardOptimizer(
            sma_crossover_backtest,
            {"sma_short": [2, 10], "sma_long": [3, 50]},
            metric="total_return",
        )
        report = optimizer.optimize(data, train_window=120, test_window=60)
        assert report["best_params"] == {"sma_short": 2, "sma_long": 3}
        for fold in report["folds"]:
            assert fold["best_params"] == {"sma_short": 2, "sma_long": 3}
            assert fold["train_metric"] > 0
        assert report["mean_test_metric"] > 0

    def test_out_of_sample_uses_test_slice(self):
        # backtest_fn reports the first positional index of the slice it was
        # given, so train_metric/test_metric prove exactly which rows each
        # evaluation saw.
        data = pd.DataFrame({"close": np.arange(100, dtype=float)})

        def first_row_backtest(df: pd.DataFrame, params: dict) -> dict:
            return {"sharpe": float(df["close"].iloc[0])}

        optimizer = WalkForwardOptimizer(first_row_backtest, {"x": [1]})
        report = optimizer.optimize(data, train_window=60, test_window=20, step=20)

        assert len(report["folds"]) == 2
        fold_one, fold_two = report["folds"]
        assert fold_one["train_range"] == (0, 60)
        assert fold_one["train_metric"] == 0.0  # train slice starts at row 0
        assert fold_one["test_metric"] == 60.0  # test slice starts at row 60
        assert fold_two["train_range"] == (20, 80)
        assert fold_two["train_metric"] == 20.0
        assert fold_two["test_metric"] == 80.0
        assert report["mean_test_metric"] == pytest.approx(70.0)

    def test_test_slice_has_expected_length(self):
        seen = []

        def spy_backtest(df: pd.DataFrame, params: dict) -> dict:
            seen.append(len(df))
            return {"sharpe": 1.0}

        data = make_trend_data(n=100)
        optimizer = WalkForwardOptimizer(spy_backtest, {"x": [1, 2]})
        optimizer.optimize(data, train_window=60, test_window=20, step=20)
        # Per fold: 2 train evaluations (grid) + 1 test evaluation.
        assert seen == [60, 60, 20, 60, 60, 20]

    def test_step_defaults_to_test_window(self):
        data = make_trend_data(n=100)
        optimizer = WalkForwardOptimizer(lambda df, p: {"sharpe": 1.0}, {"x": [1]})
        default_step = optimizer.optimize(data, train_window=40, test_window=20)
        explicit_step = optimizer.optimize(
            data, train_window=40, test_window=20, step=20
        )
        assert len(default_step["folds"]) == len(explicit_step["folds"]) == 3
        assert [f["train_range"] for f in default_step["folds"]] == [
            f["train_range"] for f in explicit_step["folds"]
        ]

    def test_tiny_data_raises_value_error(self):
        data = make_trend_data(n=10)
        optimizer = WalkForwardOptimizer(
            sma_crossover_backtest, {"sma_short": [2], "sma_long": [5]}
        )
        with pytest.raises(ValueError, match="Not enough data"):
            optimizer.optimize(data, train_window=60, test_window=20)

    def test_empty_data_raises_value_error(self):
        optimizer = WalkForwardOptimizer(
            sma_crossover_backtest, {"sma_short": [2], "sma_long": [5]}
        )
        with pytest.raises(ValueError, match="Not enough data"):
            optimizer.optimize(pd.DataFrame(), train_window=10, test_window=5)

    def test_exactly_one_fold_boundary(self):
        data = make_trend_data(n=80)
        optimizer = WalkForwardOptimizer(lambda df, p: {"sharpe": 1.0}, {"x": [1]})
        report = optimizer.optimize(data, train_window=60, test_window=20)
        assert len(report["folds"]) == 1
        assert report["folds"][0]["train_range"] == (0, 60)

    def test_invalid_windows_raise_value_error(self):
        data = make_trend_data(n=100)
        optimizer = WalkForwardOptimizer(lambda df, p: {"sharpe": 1.0}, {"x": [1]})
        with pytest.raises(ValueError):
            optimizer.optimize(data, train_window=0, test_window=20)
        with pytest.raises(ValueError):
            optimizer.optimize(data, train_window=20, test_window=0)
        with pytest.raises(ValueError):
            optimizer.optimize(data, train_window=20, test_window=10, step=0)

    def test_empty_param_grid_raises(self):
        with pytest.raises(ValueError):
            WalkForwardOptimizer(sma_crossover_backtest, {})
        with pytest.raises(ValueError):
            WalkForwardOptimizer(sma_crossover_backtest, {"sma_short": []})

    def test_missing_metric_scores_minus_inf(self):
        # Combo x=1 never reports the metric -> -inf, so x=2 must win even
        # though its metric value is tiny.
        def partial_backtest(df: pd.DataFrame, params: dict) -> dict:
            if params["x"] == 1:
                return {"other_metric": 999.0}
            return {"sharpe": 0.001}

        data = make_trend_data(n=80)
        optimizer = WalkForwardOptimizer(partial_backtest, {"x": [1, 2]})
        report = optimizer.optimize(data, train_window=60, test_window=20)
        assert report["best_params"] == {"x": 2}
        assert report["folds"][0]["train_metric"] == pytest.approx(0.001)

    def test_all_missing_metric_does_not_raise(self):
        data = make_trend_data(n=80)
        optimizer = WalkForwardOptimizer(lambda df, p: {}, {"x": [1, 2]})
        report = optimizer.optimize(data, train_window=60, test_window=20)
        assert report["folds"][0]["train_metric"] == float("-inf")
        assert report["best_params"] == {"x": 1}  # deterministic first combo

    def test_backtest_exception_scores_minus_inf(self):
        def flaky_backtest(df: pd.DataFrame, params: dict) -> dict:
            if params["x"] == 1:
                raise RuntimeError("boom")
            return {"sharpe": 0.5}

        data = make_trend_data(n=80)
        optimizer = WalkForwardOptimizer(flaky_backtest, {"x": [1, 2]})
        report = optimizer.optimize(data, train_window=60, test_window=20)
        assert report["best_params"] == {"x": 2}

    def test_deterministic_across_runs(self):
        data = make_noisy_data(n=300, seed=7)
        grid = {"sma_short": [3, 5, 8], "sma_long": [13, 21, 34]}
        first = WalkForwardOptimizer(sma_crossover_backtest, grid).optimize(
            data, train_window=150, test_window=50
        )
        second = WalkForwardOptimizer(sma_crossover_backtest, grid).optimize(
            data, train_window=150, test_window=50
        )
        assert first == second

    def test_deterministic_tie_break_keeps_first_combo(self):
        # Constant metric: every combo ties, the first itertools.product
        # combination must win on every fold, every run.
        data = make_trend_data(n=100)
        grid = {"a": [1, 2], "b": [10, 20]}
        optimizer = WalkForwardOptimizer(lambda df, p: {"sharpe": 1.0}, grid)
        for _ in range(2):
            report = optimizer.optimize(data, train_window=40, test_window=20)
            for fold in report["folds"]:
                assert fold["best_params"] == {"a": 1, "b": 10}

    def test_report_structure(self):
        data = make_noisy_data(n=200)
        optimizer = WalkForwardOptimizer(
            sma_crossover_backtest, {"sma_short": [3, 5], "sma_long": [10, 20]}
        )
        report = optimizer.optimize(data, train_window=100, test_window=40)
        assert set(report) == {"folds", "best_params", "mean_test_metric"}
        assert isinstance(report["mean_test_metric"], float)
        for fold in report["folds"]:
            assert set(fold) == {
                "train_range",
                "best_params",
                "train_metric",
                "test_metric",
            }
        assert report["best_params"] == report["folds"][-1]["best_params"]
        expected_mean = np.mean([f["test_metric"] for f in report["folds"]])
        assert report["mean_test_metric"] == pytest.approx(expected_mean)
