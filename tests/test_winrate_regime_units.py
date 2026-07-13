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

from trading.analysis.high_winrate_filter import HighWinRateFilter


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
