"""Tests for trading/signals/filters.py (roadmap items #2, #6-#9).

Plain pytest + numpy + pandas only (no torch/talib/network) so the suite
runs in CI where heavy ML dependencies are absent.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trading.signals.filters import (
    DEFAULT_FILTER_CONFIG,
    apply_signal_filters,
    detect_bb_squeeze,
    detect_macd_divergence,
    has_momentum,
    is_optimal_trading_hour,
    rsi_gate,
)

OPTIMAL_NOW = datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)
OFF_HOURS_NOW = datetime(2026, 7, 10, 3, 0, tzinfo=timezone.utc)


def make_df(closes, macd_histogram=None):
    """Build a candle DataFrame with close/volume (+ optional histogram)."""
    frame = {
        "close": list(closes),
        "volume": [1000.0] * len(closes),
    }
    if macd_histogram is not None:
        frame["macd_histogram"] = list(macd_histogram)
    return pd.DataFrame(frame)


def high_then_low_vol_closes():
    """60 closes: 30 high-volatility bars then 30 near-flat bars (squeeze)."""
    high = [100.0 + (5.0 if i % 2 == 0 else -5.0) for i in range(30)]
    low = [100.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(30)]
    return high + low


# ---------------------------------------------------------------------------
# rsi_gate (item #2)
# ---------------------------------------------------------------------------


class TestRsiGate:
    def test_buy_blocked_when_overbought(self):
        signal, reason = rsi_gate("BUY", 75.0)
        assert signal == "HOLD"
        assert reason is not None and "rsi_gate" in reason

    def test_sell_blocked_when_oversold(self):
        signal, reason = rsi_gate("SELL", 25.0)
        assert signal == "HOLD"
        assert reason is not None and "rsi_gate" in reason

    def test_buy_passes_at_boundary(self):
        # Strict comparison: exactly overbought is still allowed.
        assert rsi_gate("BUY", 70.0) == ("BUY", None)

    def test_sell_passes_at_boundary(self):
        assert rsi_gate("SELL", 30.0) == ("SELL", None)

    def test_buy_passes_in_neutral_zone(self):
        assert rsi_gate("BUY", 50.0) == ("BUY", None)

    def test_sell_passes_when_overbought(self):
        # Selling into overbought is fine (that is the point of the trade).
        assert rsi_gate("SELL", 80.0) == ("SELL", None)

    def test_hold_unaffected(self):
        assert rsi_gate("HOLD", 99.0) == ("HOLD", None)

    def test_none_rsi_passes(self):
        assert rsi_gate("BUY", None) == ("BUY", None)

    def test_nan_rsi_passes(self):
        assert rsi_gate("BUY", float("nan")) == ("BUY", None)

    def test_custom_thresholds(self):
        assert rsi_gate("BUY", 65.0, overbought=60.0)[0] == "HOLD"
        assert rsi_gate("SELL", 38.0, oversold=40.0)[0] == "HOLD"

    def test_lowercase_signal_normalized(self):
        assert rsi_gate("buy", 50.0) == ("BUY", None)

    def test_unknown_signal_treated_as_hold(self):
        assert rsi_gate("FOO", 50.0) == ("HOLD", None)


# ---------------------------------------------------------------------------
# has_momentum (item #6)
# ---------------------------------------------------------------------------


class TestHasMomentum:
    def test_rising_closes_confirm_buy(self):
        assert has_momentum(make_df([100.0, 101.0, 102.0]), "BUY") is True

    def test_falling_closes_confirm_sell(self):
        assert has_momentum(make_df([102.0, 101.0, 100.0]), "SELL") is True

    def test_flat_last_bar_fails_buy(self):
        assert has_momentum(make_df([100.0, 101.0, 101.0]), "BUY") is False

    def test_falling_closes_fail_buy(self):
        assert has_momentum(make_df([102.0, 101.0, 100.0]), "BUY") is False

    def test_short_data_returns_false(self):
        # Needs bars + 1 = 3 closes; only 2 available.
        assert has_momentum(make_df([100.0, 101.0]), "BUY") is False

    def test_empty_dataframe_returns_false(self):
        assert has_momentum(pd.DataFrame(), "BUY") is False

    def test_missing_close_column_returns_false(self):
        df = pd.DataFrame({"volume": [1.0, 2.0, 3.0]})
        assert has_momentum(df, "BUY") is False

    def test_nan_close_returns_false(self):
        assert has_momentum(make_df([100.0, np.nan, 102.0]), "BUY") is False

    def test_custom_bars(self):
        df = make_df([100.0, 101.0, 102.0, 103.0])
        assert has_momentum(df, "BUY", bars=3) is True
        # One dip inside the window breaks the streak.
        df2 = make_df([100.0, 99.0, 102.0, 103.0])
        assert has_momentum(df2, "BUY", bars=3) is False

    def test_only_last_bars_considered(self):
        # Earlier history falls outside the bars window and is ignored.
        assert has_momentum(make_df([110.0, 90.0, 100.0, 101.0, 102.0]), "BUY") is True

    def test_zero_bars_returns_false(self):
        assert has_momentum(make_df([100.0, 101.0]), "BUY", bars=0) is False

    def test_unknown_direction_returns_false(self):
        assert has_momentum(make_df([100.0, 101.0, 102.0]), "SIDEWAYS") is False


# ---------------------------------------------------------------------------
# detect_bb_squeeze (item #7)
# ---------------------------------------------------------------------------


class TestDetectBbSqueeze:
    def test_squeeze_detected_after_volatility_contraction(self):
        df = make_df(high_then_low_vol_closes())
        assert detect_bb_squeeze(df) is True

    def test_no_squeeze_when_volatility_expands(self):
        df = make_df(list(reversed(high_then_low_vol_closes())))
        assert detect_bb_squeeze(df) is False

    def test_insufficient_data_returns_false(self):
        # Needs 2 * window - 1 = 39 rows; provide fewer.
        df = make_df(high_then_low_vol_closes()[:30])
        assert detect_bb_squeeze(df) is False

    def test_empty_dataframe_returns_false(self):
        assert detect_bb_squeeze(pd.DataFrame()) is False

    def test_missing_close_column_returns_false(self):
        df = pd.DataFrame({"volume": np.ones(60)})
        assert detect_bb_squeeze(df) is False

    def test_constant_closes_return_false(self):
        # Zero-width baseline must not divide-by-zero or flag a squeeze.
        df = make_df([100.0] * 60)
        assert detect_bb_squeeze(df) is False

    def test_nan_tail_returns_false(self):
        closes = high_then_low_vol_closes()
        closes[-1] = np.nan
        assert detect_bb_squeeze(make_df(closes)) is False

    def test_custom_window(self):
        # 14 high-vol + 6 low-vol bars with window=5: the width baseline
        # still remembers the loud regime while the current width is tiny.
        closes = [100.0 + (4.0 if i % 2 == 0 else -4.0) for i in range(14)]
        closes += [100.0 + (0.02 if i % 2 == 0 else -0.02) for i in range(6)]
        assert detect_bb_squeeze(make_df(closes), window=5) is True


# ---------------------------------------------------------------------------
# is_optimal_trading_hour (item #8)
# ---------------------------------------------------------------------------


class TestIsOptimalTradingHour:
    @pytest.mark.parametrize("hour", [14, 18, 22])
    def test_optimal_hours_true(self, hour):
        now = datetime(2026, 7, 10, hour, 30, tzinfo=timezone.utc)
        assert is_optimal_trading_hour(now) is True

    @pytest.mark.parametrize("hour", [0, 3, 13, 23])
    def test_off_hours_false(self, hour):
        now = datetime(2026, 7, 10, hour, 30, tzinfo=timezone.utc)
        assert is_optimal_trading_hour(now) is False

    def test_naive_datetime_treated_as_utc(self):
        assert is_optimal_trading_hour(datetime(2026, 7, 10, 15, 0)) is True
        assert is_optimal_trading_hour(datetime(2026, 7, 10, 3, 0)) is False

    def test_aware_non_utc_converted(self):
        # 16:00 at UTC+3 is 13:00 UTC -> outside the window.
        tz = timezone(timedelta(hours=3))
        assert is_optimal_trading_hour(datetime(2026, 7, 10, 16, 0, tzinfo=tz)) is False
        # 17:00 at UTC+3 is 14:00 UTC -> inside.
        assert is_optimal_trading_hour(datetime(2026, 7, 10, 17, 0, tzinfo=tz)) is True

    def test_custom_hours(self):
        now = datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc)
        assert is_optimal_trading_hour(now, optimal_hours={1, 2, 3}) is True

    def test_default_now_returns_bool(self):
        assert isinstance(is_optimal_trading_hour(), bool)


# ---------------------------------------------------------------------------
# detect_macd_divergence (item #9)
# ---------------------------------------------------------------------------


class TestDetectMacdDivergence:
    def test_bullish_divergence(self):
        df = make_df([101.0, 100.0], macd_histogram=[-0.5, -0.2])
        assert detect_macd_divergence(df) == "BULLISH_DIVERGENCE"

    def test_bearish_divergence(self):
        df = make_df([100.0, 101.0], macd_histogram=[0.5, 0.2])
        assert detect_macd_divergence(df) == "BEARISH_DIVERGENCE"

    def test_no_divergence_when_aligned(self):
        df = make_df([100.0, 101.0], macd_histogram=[0.2, 0.5])
        assert detect_macd_divergence(df) is None
        df = make_df([101.0, 100.0], macd_histogram=[0.5, 0.2])
        assert detect_macd_divergence(df) is None

    def test_flat_price_returns_none(self):
        df = make_df([100.0, 100.0], macd_histogram=[0.5, 0.2])
        assert detect_macd_divergence(df) is None

    def test_flat_histogram_returns_none(self):
        df = make_df([100.0, 101.0], macd_histogram=[0.3, 0.3])
        assert detect_macd_divergence(df) is None

    def test_single_row_returns_none(self):
        df = make_df([100.0], macd_histogram=[0.5])
        assert detect_macd_divergence(df) is None

    def test_empty_dataframe_returns_none(self):
        assert detect_macd_divergence(pd.DataFrame()) is None

    def test_missing_histogram_column_returns_none(self):
        assert detect_macd_divergence(make_df([101.0, 100.0])) is None

    def test_nan_histogram_returns_none(self):
        df = make_df([101.0, 100.0], macd_histogram=[np.nan, 0.2])
        assert detect_macd_divergence(df) is None

    def test_only_last_two_rows_considered(self):
        closes = [90.0, 95.0, 101.0, 100.0]
        hist = [1.0, 2.0, -0.5, -0.2]
        assert detect_macd_divergence(make_df(closes, hist)) == "BULLISH_DIVERGENCE"


# ---------------------------------------------------------------------------
# apply_signal_filters (composite)
# ---------------------------------------------------------------------------


def clean_buy_df():
    """Data that passes every filter for a BUY: no squeeze, rising closes."""
    closes = high_then_low_vol_closes()[:40]  # expanding-vol tail -> no squeeze
    closes = list(reversed(closes))
    closes += [200.0, 201.0, 202.0]  # rising tail confirms BUY momentum
    hist = list(np.linspace(-1.0, 1.0, len(closes)))  # rising hist, no bear div
    return make_df(closes, macd_histogram=hist)


class TestApplySignalFilters:
    def test_clean_buy_passes_unchanged(self):
        signal, confidence, reasons = apply_signal_filters(
            "BUY", 0.82, clean_buy_df(), rsi=50.0, now=OPTIMAL_NOW
        )
        assert (signal, confidence, reasons) == ("BUY", 0.82, [])

    def test_rsi_blocks_buy(self):
        signal, confidence, reasons = apply_signal_filters(
            "BUY", 0.82, clean_buy_df(), rsi=85.0, now=OPTIMAL_NOW
        )
        assert (signal, confidence) == ("HOLD", 0.0)
        assert len(reasons) == 1 and "rsi_gate" in reasons[0]

    def test_momentum_blocks_buy_on_falling_tape(self):
        df = clean_buy_df()
        df.loc[df.index[-1], "close"] = 100.0  # break the rising streak
        df.loc[df.index[-1], "macd_histogram"] = 2.0  # keep hist rising (no div)
        signal, confidence, reasons = apply_signal_filters(
            "BUY", 0.82, df, rsi=50.0, now=OPTIMAL_NOW
        )
        assert (signal, confidence) == ("HOLD", 0.0)
        assert any("momentum" in r for r in reasons)

    def test_squeeze_blocks_signal(self):
        closes = high_then_low_vol_closes()
        closes += [100.0, 100.2, 100.4]  # rising tail, squeeze still active
        df = make_df(closes, macd_histogram=list(np.linspace(-1, 1, len(closes))))
        signal, confidence, reasons = apply_signal_filters(
            "BUY", 0.82, df, rsi=50.0, now=OPTIMAL_NOW
        )
        assert (signal, confidence) == ("HOLD", 0.0)
        assert any("bb_squeeze" in r for r in reasons)

    def test_off_hours_blocks_signal(self):
        signal, confidence, reasons = apply_signal_filters(
            "BUY", 0.82, clean_buy_df(), rsi=50.0, now=OFF_HOURS_NOW
        )
        assert (signal, confidence) == ("HOLD", 0.0)
        assert any("trading_hours" in r for r in reasons)

    def test_bearish_divergence_blocks_buy(self):
        df = clean_buy_df()
        # Price still rising but histogram now falling -> bearish divergence.
        df.loc[df.index[-1], "macd_histogram"] = -5.0
        signal, confidence, reasons = apply_signal_filters(
            "BUY", 0.82, df, rsi=50.0, now=OPTIMAL_NOW
        )
        assert (signal, confidence) == ("HOLD", 0.0)
        assert any("BEARISH_DIVERGENCE" in r for r in reasons)

    def test_multiple_reasons_collected(self):
        signal, confidence, reasons = apply_signal_filters(
            "BUY", 0.82, clean_buy_df(), rsi=85.0, now=OFF_HOURS_NOW
        )
        assert (signal, confidence) == ("HOLD", 0.0)
        assert len(reasons) == 2

    def test_disabled_filter_lets_signal_through(self):
        signal, confidence, reasons = apply_signal_filters(
            "BUY",
            0.82,
            clean_buy_df(),
            rsi=50.0,
            now=OFF_HOURS_NOW,
            config={"trading_hours": False},
        )
        assert (signal, confidence, reasons) == ("BUY", 0.82, [])

    def test_all_filters_disabled_passes_anything(self):
        config = {key: False for key in DEFAULT_FILTER_CONFIG}
        signal, confidence, reasons = apply_signal_filters(
            "SELL", 0.6, pd.DataFrame(), rsi=1.0, now=OFF_HOURS_NOW, config=config
        )
        assert (signal, confidence, reasons) == ("SELL", 0.6, [])

    def test_none_rsi_skips_gate(self):
        signal, _, reasons = apply_signal_filters(
            "BUY", 0.82, clean_buy_df(), rsi=None, now=OPTIMAL_NOW
        )
        assert signal == "BUY" and reasons == []

    def test_empty_data_blocks_conservatively_without_raising(self):
        signal, confidence, reasons = apply_signal_filters(
            "BUY", 0.82, pd.DataFrame(), rsi=50.0, now=OPTIMAL_NOW
        )
        assert (signal, confidence) == ("HOLD", 0.0)
        assert any("momentum" in r for r in reasons)

    def test_hold_stays_hold_without_override(self):
        df = make_df([101.0, 100.0], macd_histogram=[-0.5, -0.2])  # bullish div
        signal, confidence, reasons = apply_signal_filters(
            "HOLD", 0.0, df, now=OPTIMAL_NOW
        )
        assert (signal, confidence, reasons) == ("HOLD", 0.0, [])

    def test_override_promotes_hold_to_buy(self):
        df = make_df([101.0, 100.0], macd_histogram=[-0.5, -0.2])  # bullish div
        signal, confidence, reasons = apply_signal_filters(
            "HOLD", 0.0, df, now=OPTIMAL_NOW, config={"macd_divergence_override": True}
        )
        assert (signal, confidence) == ("BUY", 0.75)
        assert any("BULLISH_DIVERGENCE" in r for r in reasons)

    def test_override_promotes_hold_to_sell(self):
        df = make_df([100.0, 101.0], macd_histogram=[0.5, 0.2])  # bearish div
        signal, confidence, reasons = apply_signal_filters(
            "HOLD", 0.0, df, now=OPTIMAL_NOW, config={"macd_divergence_override": True}
        )
        assert (signal, confidence) == ("SELL", 0.75)
        assert any("BEARISH_DIVERGENCE" in r for r in reasons)

    def test_override_noop_without_divergence(self):
        df = make_df([100.0, 101.0], macd_histogram=[0.2, 0.5])  # aligned
        signal, confidence, reasons = apply_signal_filters(
            "HOLD", 0.0, df, now=OPTIMAL_NOW, config={"macd_divergence_override": True}
        )
        assert (signal, confidence, reasons) == ("HOLD", 0.0, [])

    def test_override_does_not_resurrect_blocked_signal(self):
        # BUY blocked by RSI; bullish divergence present; override enabled.
        # The block must stand: no divergence-generated re-entry.
        df = make_df([101.0, 100.0], macd_histogram=[-0.5, -0.2])  # bullish divergence
        signal, confidence, reasons = apply_signal_filters(
            "BUY",
            0.82,
            df,
            rsi=85.0,
            now=OPTIMAL_NOW,
            config={"macd_divergence_override": True},
        )
        assert (signal, confidence) == ("HOLD", 0.0)
        assert reasons  # veto reasons recorded, no override applied

    def test_unknown_signal_treated_as_hold(self):
        signal, confidence, reasons = apply_signal_filters(
            "LONG_SHOT", 0.9, clean_buy_df(), now=OPTIMAL_NOW
        )
        assert signal == "HOLD" and reasons == []

    def test_non_numeric_confidence_coerced(self):
        signal, confidence, _ = apply_signal_filters(
            "BUY", None, clean_buy_df(), rsi=50.0, now=OPTIMAL_NOW
        )
        assert signal == "BUY" and confidence == 0.0
