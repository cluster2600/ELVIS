"""Directional performance guard (loss forensics): a side that's been
losing must clear a higher confidence bar, self-relaxing when it recovers."""

import pytest

from trading.signals import directional_guard as dg


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "ELVIS_DG_TARGET_WR",
        "ELVIS_DG_MIN_TRADES",
        "ELVIS_DG_MAX_FLOOR",
        "ELVIS_DG_LOOKBACK",
    ):
        monkeypatch.delenv(k, raising=False)


def test_healthy_side_uses_base(_clear_env):
    # win rate above the 0.35 target -> base bar, no throttle
    assert dg.required_confidence("BUY", winrate=0.45, n=40, base=0.60) == 0.60


def test_losing_side_raises_bar_toward_max(_clear_env):
    # the actual SELL case: 3% win rate -> near the 0.95 ceiling
    floor = dg.required_confidence("SELL", winrate=0.03, n=95, base=0.60)
    assert floor > 0.90
    assert floor <= 0.95


def test_ramp_is_monotonic(_clear_env):
    worse = dg.required_confidence("SELL", winrate=0.05, n=50, base=0.60)
    better = dg.required_confidence("SELL", winrate=0.25, n=50, base=0.60)
    assert worse > better >= 0.60  # lower win rate -> higher bar


def test_insufficient_history_abstains(_clear_env):
    # below min_trades the guard must not judge the direction
    assert dg.required_confidence("SELL", winrate=0.0, n=5, base=0.60) == 0.60


def test_no_history_abstains(_clear_env):
    assert dg.required_confidence("BUY", winrate=None, n=0, base=0.60) == 0.60


def test_target_boundary_uses_base(_clear_env):
    assert dg.required_confidence("SELL", winrate=0.35, n=50, base=0.60) == 0.60


def test_env_target_override(monkeypatch):
    monkeypatch.setenv("ELVIS_DG_TARGET_WR", "0.50")
    # 0.35 is now below target -> bar rises above base
    assert dg.required_confidence("SELL", winrate=0.35, n=50, base=0.60) > 0.60
