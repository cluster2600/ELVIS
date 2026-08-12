"""Regression for the winrate filter's regime unit bug (trade-flow blocker).

trend_strength was |slope per bar| / price * 100 with a >1 "favorable"
threshold — on 1m data that demands >1% PER MINUTE sustained for 20 minutes,
so the filter vetoed 100% of signals ("Unfavorable market regime", 7750/7750
in a 2h live run). The strength is now percent over the whole window.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from trading.analysis.high_winrate_filter import (
    HighWinRateFilter,
    classify_take_profit_regime,
)


def _df(closes):
    n = len(closes)
    return pd.DataFrame(
        {
            "close": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "volume": [100.0] * n,
        }
    )


@pytest.fixture
def filt():
    return HighWinRateFilter(logging.getLogger("t"))


def test_three_pct_window_trend_is_favorable(filt):
    # +3% over 20 bars — a strong, real crypto move
    closes = list(np.linspace(64000, 64000 * 1.03, 25))
    regime = filt._detect_market_regime(_df(closes))
    assert bool(regime["favorable"]) is True
    assert regime["trend_direction"] == "bullish"


def test_flat_market_stays_unfavorable(filt):
    closes = [64000 + ((-1) ** i) * 5 for i in range(25)]  # ±5$ noise
    regime = filt._detect_market_regime(_df(closes))
    assert bool(regime["favorable"]) is False


def test_typical_1m_drift_no_longer_impossible(filt):
    # ~0.8% over the window: not favorable (below the 1% moderate bar) but
    # the STRENGTH value must now be in a reachable range, not ~0.04
    closes = list(np.linspace(64000, 64000 * 1.008, 25))
    regime = filt._detect_market_regime(_df(closes))
    assert 0.5 < regime["strength"] < 1.1


def test_min_confidence_env_override(monkeypatch):
    monkeypatch.setenv("ELVIS_WINRATE_MIN_CONF", "0.6")
    f = HighWinRateFilter(logging.getLogger("t"))
    assert f.min_confidence_threshold == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("regime", "volatility", "expected"),
    [
        ("trending_strong", 0.0, "TRENDING"),
        ("trending_moderate", 0.049999, "TRENDING"),
        ("trending_weak", 0.01, "RANGING"),
        ("ranging", 0.01, "RANGING"),
        ("trending_strong", 0.05, "CHOPPY"),
        ("ranging", 0.50, "CHOPPY"),
    ],
)
def test_take_profit_regime_uses_topology_and_volatility(
    regime: str, volatility: float, expected: str
) -> None:
    assert classify_take_profit_regime(regime, volatility) == expected


@pytest.mark.parametrize(
    ("regime", "volatility"),
    [
        ("unknown", 0.01),
        ("optimal", 0.01),
        (None, 0.01),
        ("ranging", None),
        ("ranging", True),
        ("ranging", np.bool_(True)),
        ("ranging", "0.01"),
        ("ranging", float("nan")),
        ("ranging", float("inf")),
        ("ranging", -0.01),
    ],
)
def test_take_profit_regime_rejects_unknown_or_invalid_inputs(
    regime: object, volatility: object
) -> None:
    assert classify_take_profit_regime(regime, volatility) is None


def test_detected_market_regime_exposes_a_separate_take_profit_axis(filt) -> None:
    trending = filt._detect_market_regime(_df(np.linspace(100.0, 103.0, 50)))
    ranging = filt._detect_market_regime(_df(np.full(50, 100.0)))
    public_result = filt.analyze_signal_quality(
        "BTCUSDT", {}, _df(np.linspace(100.0, 103.0, 50))
    )

    assert trending["take_profit_regime"] == "TRENDING"
    assert ranging["take_profit_regime"] == "RANGING"
    assert public_result["market_regime"]["take_profit_regime"] == "TRENDING"
    assert {
        trending["take_profit_regime"],
        ranging["take_profit_regime"],
    }.isdisjoint({"REVERSAL", "optimal", "favorable", "neutral", "unfavorable"})


def test_short_history_has_no_take_profit_regime(filt) -> None:
    regime = filt._detect_market_regime(_df(np.linspace(100.0, 101.0, 19)))

    assert regime["regime"] == "unknown"
    assert regime["take_profit_regime"] is None
