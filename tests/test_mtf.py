"""Tests for trading.signals.mtf (multi-timeframe alignment, roadmap #14)."""

import numpy as np
import pandas as pd
import pytest

from trading.signals.mtf import BUY, HOLD, SELL, MTFAnalyzer, sma_trend_signal


def make_df(closes) -> pd.DataFrame:
    """Build a minimal kline-like DataFrame with a close column."""
    return pd.DataFrame({"close": list(closes)})


def rising_df(n: int = 60) -> pd.DataFrame:
    return make_df(np.linspace(100.0, 130.0, n))


def falling_df(n: int = 60) -> pd.DataFrame:
    return make_df(np.linspace(130.0, 100.0, n))


def flat_df(n: int = 60) -> pd.DataFrame:
    return make_df([100.0] * n)


class TestSmaTrendSignal:
    def test_uptrend_returns_buy(self):
        assert sma_trend_signal(rising_df()) == BUY

    def test_downtrend_returns_sell(self):
        assert sma_trend_signal(falling_df()) == SELL

    def test_flat_returns_hold(self):
        assert sma_trend_signal(flat_df()) == HOLD

    def test_short_data_returns_hold(self):
        assert sma_trend_signal(rising_df(49)) == HOLD

    def test_exactly_slow_rows_is_enough(self):
        assert sma_trend_signal(rising_df(50)) == BUY

    def test_empty_dataframe_returns_hold(self):
        assert sma_trend_signal(pd.DataFrame()) == HOLD

    def test_none_returns_hold(self):
        assert sma_trend_signal(None) == HOLD

    def test_missing_close_column_returns_hold(self):
        df = pd.DataFrame({"open": np.linspace(100, 130, 60)})
        assert sma_trend_signal(df) == HOLD

    def test_spread_below_threshold_returns_hold(self):
        # 30 closes at 100 then 20 at 100.05:
        # SMA(20)=100.05, SMA(50)=100.02 -> spread ~0.03% < 0.1%
        assert sma_trend_signal(make_df([100.0] * 30 + [100.05] * 20)) == HOLD

    def test_spread_above_threshold_returns_buy(self):
        # SMA(20)=100.3, SMA(50)=100.12 -> spread ~0.18% > 0.1%
        assert sma_trend_signal(make_df([100.0] * 30 + [100.3] * 20)) == BUY

    def test_spread_below_negative_threshold_returns_sell(self):
        # SMA(20)=99.7, SMA(50)=99.88 -> spread ~-0.18% < -0.1%
        assert sma_trend_signal(make_df([100.0] * 30 + [99.7] * 20)) == SELL

    def test_nan_closes_are_dropped(self):
        closes = list(np.linspace(100.0, 130.0, 60))
        closes[3] = np.nan
        closes[10] = np.nan
        assert sma_trend_signal(make_df(closes)) == BUY

    def test_nans_reducing_below_slow_returns_hold(self):
        closes = list(np.linspace(100.0, 130.0, 52))
        for i in range(5):
            closes[i] = np.nan  # 47 valid closes < slow=50
        assert sma_trend_signal(make_df(closes)) == HOLD

    def test_non_numeric_closes_returns_hold(self):
        assert sma_trend_signal(make_df(["oops"] * 60)) == HOLD

    def test_string_numbers_are_coerced(self):
        closes = [f"{v:.4f}" for v in np.linspace(100.0, 130.0, 60)]
        assert sma_trend_signal(make_df(closes)) == BUY

    def test_invalid_windows_return_hold(self):
        assert sma_trend_signal(rising_df(), fast=0) == HOLD
        assert sma_trend_signal(rising_df(), fast=50, slow=20) == HOLD
        assert sma_trend_signal(rising_df(), slow=-1) == HOLD

    def test_custom_windows(self):
        assert sma_trend_signal(rising_df(12), fast=3, slow=10) == BUY


def make_klines(closes):
    """Build Binance-style raw kline lists (index 4 = close, as strings)."""
    return [
        [
            1700000000000 + i,
            "1",
            "2",
            "0.5",
            f"{c:.8f}",
            "10.0",
            0,
            "0",
            0,
            "0",
            "0",
            "0",
        ]
        for i, c in enumerate(closes)
    ]


class RecordingFetcher:
    """Fake fetch_klines returning canned data per interval."""

    def __init__(self, data_by_interval, default=None):
        self.data_by_interval = data_by_interval
        self.default = default
        self.calls = []

    def __call__(self, symbol, interval, limit):
        self.calls.append((symbol, interval, limit))
        return self.data_by_interval.get(interval, self.default)


class TestMTFAnalyzer:
    def test_full_buy_alignment(self):
        fetcher = RecordingFetcher({}, default=rising_df())
        result = MTFAnalyzer(fetcher).get_signal("BTCUSDT")
        assert result["aligned"] == BUY
        assert result["signals"] == {"15m": BUY, "1h": BUY, "4h": BUY}
        assert result["symbol"] == "BTCUSDT"

    def test_full_sell_alignment(self):
        fetcher = RecordingFetcher({}, default=falling_df())
        result = MTFAnalyzer(fetcher).get_signal("BTCUSDT")
        assert result["aligned"] == SELL
        assert set(result["signals"].values()) == {SELL}

    def test_partial_alignment_returns_hold(self):
        fetcher = RecordingFetcher(
            {"15m": rising_df(), "1h": rising_df(), "4h": flat_df()}
        )
        result = MTFAnalyzer(fetcher).get_signal("BTCUSDT")
        assert result["signals"] == {"15m": BUY, "1h": BUY, "4h": HOLD}
        assert result["aligned"] == HOLD

    def test_mixed_buy_sell_returns_hold(self):
        fetcher = RecordingFetcher(
            {"15m": rising_df(), "1h": falling_df(), "4h": rising_df()}
        )
        assert MTFAnalyzer(fetcher).get_signal("BTCUSDT")["aligned"] == HOLD

    def test_fetch_exception_maps_to_hold_and_never_raises(self):
        def fetcher(symbol, interval, limit):
            if interval == "1h":
                raise ConnectionError("exchange down")
            return rising_df()

        result = MTFAnalyzer(fetcher).get_signal("BTCUSDT")
        assert result["signals"]["1h"] == HOLD
        assert result["signals"]["15m"] == BUY
        assert result["aligned"] == HOLD

    def test_fetch_none_maps_to_hold(self):
        fetcher = RecordingFetcher(
            {"15m": rising_df(), "1h": rising_df()}
        )  # 4h -> None
        result = MTFAnalyzer(fetcher).get_signal("BTCUSDT")
        assert result["signals"]["4h"] == HOLD
        assert result["aligned"] == HOLD

    def test_fetch_empty_dataframe_maps_to_hold(self):
        fetcher = RecordingFetcher({}, default=pd.DataFrame())
        result = MTFAnalyzer(fetcher).get_signal("BTCUSDT")
        assert result["aligned"] == HOLD
        assert set(result["signals"].values()) == {HOLD}

    def test_list_shaped_klines_are_coerced(self):
        fetcher = RecordingFetcher({}, default=make_klines(np.linspace(100, 130, 60)))
        result = MTFAnalyzer(fetcher).get_signal("BTCUSDT")
        assert result["aligned"] == BUY

    def test_short_list_klines_map_to_hold(self):
        fetcher = RecordingFetcher({}, default=make_klines([100.0] * 10))
        assert MTFAnalyzer(fetcher).get_signal("BTCUSDT")["aligned"] == HOLD

    def test_malformed_rows_map_to_hold(self):
        fetcher = RecordingFetcher({}, default=[[1, 2], "junk", None])
        result = MTFAnalyzer(fetcher).get_signal("BTCUSDT")
        assert result["aligned"] == HOLD

    def test_fetcher_called_once_per_timeframe_with_args(self):
        fetcher = RecordingFetcher({}, default=rising_df())
        MTFAnalyzer(fetcher, timeframes=("5m", "1d"), limit=75).get_signal("ETHUSDT")
        assert fetcher.calls == [("ETHUSDT", "5m", 75), ("ETHUSDT", "1d", 75)]

    def test_custom_timeframes_in_result(self):
        fetcher = RecordingFetcher({}, default=rising_df())
        result = MTFAnalyzer(fetcher, timeframes=("1m",)).get_signal("BTCUSDT")
        assert list(result["signals"]) == ["1m"]
        assert result["aligned"] == BUY

    def test_empty_timeframes_returns_hold_not_buy(self):
        # all() over an empty dict is vacuously True; must not report BUY.
        fetcher = RecordingFetcher({}, default=rising_df())
        result = MTFAnalyzer(fetcher, timeframes=()).get_signal("BTCUSDT")
        assert result["signals"] == {}
        assert result["aligned"] == HOLD

    def test_limit_below_slow_yields_hold(self):
        fetcher = RecordingFetcher({}, default=rising_df(30))
        result = MTFAnalyzer(fetcher, limit=30).get_signal("BTCUSDT")
        assert result["aligned"] == HOLD

    def test_non_callable_fetcher_raises_typeerror(self):
        with pytest.raises(TypeError):
            MTFAnalyzer(fetch_klines="not-callable")
