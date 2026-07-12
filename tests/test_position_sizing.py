"""
Tests for trading.risk.position_sizing (roadmap items #3 and #13).

Covers the happy paths, the exact Kelly formula on a pinned case, clamping at
both bounds, and defensive behavior on empty/short/NaN/degenerate inputs.
"""

import math

import numpy as np
import pandas as pd
import pytest

from trading.risk.position_sizing import (
    kelly_fraction,
    kelly_from_trades,
    volume_multiplier,
)


def make_volume_df(volumes) -> pd.DataFrame:
    """Build a minimal OHLCV-ish DataFrame with the given volume column."""
    n = len(volumes)
    return pd.DataFrame(
        {
            "close": [100.0] * n,
            "volume": volumes,
        }
    )


class TestVolumeMultiplier:
    def test_happy_path_ratio_within_bounds(self):
        # 19 bars at 100, current bar at 150: mean = (19*100 + 150) / 20
        df = make_volume_df([100.0] * 19 + [150.0])
        expected = 150.0 / ((19 * 100.0 + 150.0) / 20)
        assert volume_multiplier(df, window=20) == pytest.approx(expected)

    def test_flat_volume_is_exactly_one(self):
        df = make_volume_df([250.0] * 30)
        assert volume_multiplier(df, window=20) == pytest.approx(1.0)

    def test_spike_clamped_to_cap(self):
        df = make_volume_df([100.0] * 19 + [10_000.0])
        assert volume_multiplier(df, window=20, cap=2.0) == 2.0

    def test_dry_volume_clamped_to_floor(self):
        df = make_volume_df([100.0] * 19 + [1.0])
        assert volume_multiplier(df, window=20, floor=0.5) == 0.5

    def test_custom_bounds_respected(self):
        df = make_volume_df([100.0] * 19 + [10_000.0])
        assert volume_multiplier(df, window=20, floor=0.8, cap=1.5) == 1.5

    def test_uses_only_last_window_bars(self):
        # Huge old volumes outside the window must not affect the ratio.
        df = make_volume_df([1e9] * 50 + [100.0] * 20)
        assert volume_multiplier(df, window=20) == pytest.approx(1.0)

    def test_none_input_is_neutral(self):
        assert volume_multiplier(None) == 1.0

    def test_empty_dataframe_is_neutral(self):
        assert volume_multiplier(pd.DataFrame()) == 1.0

    def test_missing_volume_column_is_neutral(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        assert volume_multiplier(df) == 1.0

    def test_short_data_is_neutral(self):
        df = make_volume_df([100.0] * 5)
        assert volume_multiplier(df, window=20) == 1.0

    def test_exactly_window_rows_is_computed(self):
        df = make_volume_df([100.0] * 20)
        assert volume_multiplier(df, window=20) == pytest.approx(1.0)

    def test_nan_rows_reduce_valid_sample_to_neutral(self):
        # 20 rows but only 15 valid -> not enough evidence -> neutral.
        df = make_volume_df([100.0] * 15 + [np.nan] * 5)
        assert volume_multiplier(df, window=20) == 1.0

    def test_nan_rows_skipped_when_enough_valid_data(self):
        # 25 valid rows among NaNs: last 20 valid are used.
        volumes = [100.0, np.nan] * 24 + [150.0, np.nan]
        df = make_volume_df(volumes)
        expected = 150.0 / ((19 * 100.0 + 150.0) / 20)
        assert volume_multiplier(df, window=20) == pytest.approx(expected)

    def test_all_zero_volume_is_neutral(self):
        df = make_volume_df([0.0] * 30)
        assert volume_multiplier(df, window=20) == 1.0

    def test_negative_current_volume_is_neutral(self):
        df = make_volume_df([100.0] * 19 + [-5.0])
        assert volume_multiplier(df, window=20) == 1.0

    def test_non_numeric_volume_is_neutral(self):
        df = make_volume_df(["oops"] * 30)
        assert volume_multiplier(df, window=20) == 1.0

    def test_invalid_window_is_neutral(self):
        df = make_volume_df([100.0] * 30)
        assert volume_multiplier(df, window=0) == 1.0

    def test_case_insensitive_volume_column(self):
        df = pd.DataFrame({"Volume": [100.0] * 19 + [150.0]})
        expected = 150.0 / ((19 * 100.0 + 150.0) / 20)
        assert volume_multiplier(df, window=20) == pytest.approx(expected)

    def test_result_is_float(self):
        df = make_volume_df([100] * 30)  # integer volumes
        result = volume_multiplier(df, window=20)
        assert isinstance(result, float)


class TestKellyFraction:
    def test_pinned_formula_case_clamped_to_cap(self):
        # p=0.45, avg_win=50, avg_loss=-20 -> b=2.5
        # f* = (2.5*0.45 - 0.55) / 2.5 = 0.23 -> clamped to cap 0.05
        assert kelly_fraction(0.45, 50.0, -20.0) == pytest.approx(0.05)

    def test_pinned_formula_case_unclamped(self):
        # p=0.52, b=1 -> f* = (0.52 - 0.48) / 1 = 0.04, inside [0.01, 0.05]
        assert kelly_fraction(0.52, 25.0, -25.0) == pytest.approx(0.04)

    def test_exact_formula_with_wide_bounds(self):
        # Widen the clamp so the raw formula value is observable.
        p, avg_win, avg_loss = 0.45, 50.0, -20.0
        b = abs(avg_win / avg_loss)
        expected = (b * p - (1 - p)) / b  # 0.23
        result = kelly_fraction(p, avg_win, avg_loss, floor=0.0, cap=1.0)
        assert result == pytest.approx(expected)
        assert result == pytest.approx(0.23)

    def test_negative_edge_clamped_to_floor(self):
        # Losing system: f* < 0 -> floor.
        assert kelly_fraction(0.30, 20.0, -40.0) == 0.01

    def test_positive_avg_loss_uses_magnitude(self):
        # Sign convention of avg_loss must not matter.
        assert kelly_fraction(0.45, 50.0, 20.0) == kelly_fraction(0.45, 50.0, -20.0)

    def test_zero_avg_loss_returns_floor(self):
        assert kelly_fraction(0.60, 50.0, 0.0) == 0.01

    def test_zero_avg_win_returns_floor(self):
        # b == 0 -> degenerate.
        assert kelly_fraction(0.60, 0.0, -20.0) == 0.01

    @pytest.mark.parametrize("win_rate", [0.0, 1.0, -0.2, 1.5])
    def test_win_rate_outside_open_interval_returns_floor(self, win_rate):
        assert kelly_fraction(win_rate, 50.0, -20.0) == 0.01

    @pytest.mark.parametrize(
        "win_rate,avg_win,avg_loss",
        [
            (math.nan, 50.0, -20.0),
            (0.5, math.nan, -20.0),
            (0.5, 50.0, math.nan),
            (math.inf, 50.0, -20.0),
            (0.5, math.inf, -20.0),
        ],
    )
    def test_non_finite_inputs_return_floor(self, win_rate, avg_win, avg_loss):
        assert kelly_fraction(win_rate, avg_win, avg_loss) == 0.01

    def test_non_numeric_inputs_return_floor(self):
        assert kelly_fraction(None, 50.0, -20.0) == 0.01
        assert kelly_fraction("high", 50.0, -20.0) == 0.01

    def test_custom_floor_and_cap(self):
        assert kelly_fraction(0.45, 50.0, -20.0, floor=0.02, cap=0.10) == 0.10
        assert kelly_fraction(0.30, 20.0, -40.0, floor=0.02, cap=0.10) == 0.02

    def test_result_always_within_bounds(self):
        for p in np.linspace(0.05, 0.95, 19):
            result = kelly_fraction(float(p), 30.0, -25.0)
            assert 0.01 <= result <= 0.05


class TestKellyFromTrades:
    @staticmethod
    def make_trades(n_wins: int, n_losses: int, win: float, loss: float):
        return [{"pnl": win}] * n_wins + [{"pnl": loss}] * n_losses

    def test_happy_path_matches_kelly_fraction(self):
        # 9 wins of +50, 11 losses of -20 -> p=0.45, b=2.5 -> cap 0.05.
        trades = self.make_trades(9, 11, 50.0, -20.0)
        assert len(trades) == 20
        assert kelly_from_trades(trades) == pytest.approx(
            kelly_fraction(0.45, 50.0, -20.0)
        )
        assert kelly_from_trades(trades) == pytest.approx(0.05)

    def test_stats_derived_correctly_with_wide_bounds(self):
        # Mixed magnitudes: p=0.5, avg_win=40, avg_loss=-20 -> b=2
        # f* = (2*0.5 - 0.5) / 2 = 0.25.
        trades = self.make_trades(5, 10, 30.0, -20.0) + self.make_trades(
            5, 0, 50.0, 0.0
        )
        assert kelly_from_trades(trades, floor=0.0, cap=1.0) == pytest.approx(0.25)

    def test_fewer_than_min_trades_returns_floor(self):
        trades = self.make_trades(10, 9, 50.0, -20.0)  # 19 < 20
        assert kelly_from_trades(trades) == 0.01

    def test_min_trades_boundary(self):
        trades = self.make_trades(9, 11, 50.0, -20.0)  # exactly 20
        assert kelly_from_trades(trades, min_trades=20) == pytest.approx(0.05)
        assert kelly_from_trades(trades, min_trades=21) == 0.01

    def test_empty_and_none_return_floor(self):
        assert kelly_from_trades([]) == 0.01
        assert kelly_from_trades(None) == 0.01

    def test_custom_floor_propagates_to_short_history(self):
        assert kelly_from_trades([], floor=0.02) == 0.02
        assert kelly_from_trades([{"pnl": 5.0}], floor=0.02) == 0.02

    def test_all_wins_returns_floor(self):
        trades = self.make_trades(25, 0, 50.0, 0.0)
        assert kelly_from_trades(trades) == 0.01

    def test_all_losses_returns_floor(self):
        trades = self.make_trades(0, 25, 0.0, -20.0)
        assert kelly_from_trades(trades) == 0.01

    def test_invalid_records_are_skipped(self):
        trades = self.make_trades(9, 11, 50.0, -20.0)
        junk = [
            {"pnl": None},
            {"pnl": "oops"},
            {"pnl": math.nan},
            {"amount": 3.0},  # missing 'pnl'
            "not-a-dict",
            None,
        ]
        assert kelly_from_trades(trades + junk) == pytest.approx(0.05)

    def test_junk_only_history_returns_floor(self):
        junk = [{"pnl": math.nan}] * 30
        assert kelly_from_trades(junk) == 0.01

    def test_zero_pnl_trades_count_toward_sample_not_sides(self):
        # 9 wins, 11 losses, 5 break-even -> p = 9/25 = 0.36, b = 2.5
        # f* = (2.5*0.36 - 0.64) / 2.5 = 0.104
        trades = self.make_trades(9, 11, 50.0, -20.0) + [{"pnl": 0.0}] * 5
        assert kelly_from_trades(trades, floor=0.0, cap=1.0) == pytest.approx(0.104)

    def test_kwargs_forwarded_to_kelly_fraction(self):
        trades = self.make_trades(9, 11, 50.0, -20.0)
        assert kelly_from_trades(trades, floor=0.02, cap=0.10) == pytest.approx(0.10)
