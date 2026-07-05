"""Focused tests for the Kelly-criterion sizing helper.

These exercise the pure-math ``kelly_fraction`` function and the opt-in
``RiskManager.calculate_kelly_position_size`` wiring. No live service (DB,
Redis, exchange) is required; the RiskManager is built against a mocked
executor and the module's third-party deps are skipped if unavailable.
"""

import math
from unittest.mock import MagicMock

import pytest

# risk_management imports numpy/pandas/ta at module load; skip cleanly if a
# minimal env lacks them rather than erroring the whole test session.
pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("ta")

from trading.risk_management import RiskManager, kelly_fraction  # noqa: E402


def test_kelly_known_value_positive_edge():
    # W=0.6, R=2 -> 0.6 - 0.4/2 = 0.4
    assert math.isclose(kelly_fraction(0.6, 2.0, cap=1.0), 0.4, abs_tol=1e-12)


def test_kelly_floored_at_zero_no_edge():
    # W=0.4, R=1 -> 0.4 - 0.6/1 = -0.2, floored to 0
    assert kelly_fraction(0.4, 1.0) == 0.0


def test_kelly_capped():
    # W=0.9, R=5 -> 0.9 - 0.1/5 = 0.88, but cap limits it
    assert kelly_fraction(0.9, 5.0, cap=0.2) == 0.2


def test_kelly_default_cap_is_twenty_percent():
    # Raw f* = 0.88 exceeds the default 0.2 cap
    assert kelly_fraction(0.9, 5.0) == 0.2


def test_kelly_break_even_edge_is_zero():
    # W=0.5, R=1 -> 0.5 - 0.5/1 = 0.0
    assert kelly_fraction(0.5, 1.0) == 0.0


@pytest.mark.parametrize(
    "win_rate,payoff_ratio",
    [(-0.1, 2.0), (1.1, 2.0)],
)
def test_kelly_rejects_out_of_range_win_rate(win_rate, payoff_ratio):
    with pytest.raises(ValueError):
        kelly_fraction(win_rate, payoff_ratio)


@pytest.mark.parametrize("payoff_ratio", [0.0, -1.0])
def test_kelly_rejects_non_positive_payoff(payoff_ratio):
    with pytest.raises(ValueError):
        kelly_fraction(0.6, payoff_ratio)


def test_kelly_rejects_negative_cap():
    with pytest.raises(ValueError):
        kelly_fraction(0.6, 2.0, cap=-0.1)


def _make_manager():
    return RiskManager(executor=MagicMock(), logger=MagicMock())


def test_calculate_kelly_position_size_scales_by_leverage_and_price():
    rm = _make_manager()
    # f* for W=0.6, R=2 (cap 1.0) = 0.4; capital 1000, lev 100, price 100.
    # (leverage 100 is at leverage_target, so enforce_minimum_leverage keeps
    # it as-is.) notional = 1000 * 0.4 * 100 = 40000 ; size = 40000/100 = 400
    size = rm.calculate_kelly_position_size(
        available_capital=1000.0,
        current_price=100.0,
        payoff_ratio=2.0,
        win_rate=0.6,
        leverage=100.0,
        cap=1.0,
    )
    assert math.isclose(size, 400.0, rel_tol=1e-9)


def test_calculate_kelly_position_size_defaults_to_tracked_win_rate():
    rm = _make_manager()
    rm.win_rate = 0.6
    size = rm.calculate_kelly_position_size(
        available_capital=1000.0,
        current_price=100.0,
        payoff_ratio=2.0,
        leverage=100.0,
        cap=1.0,
    )
    assert math.isclose(size, 400.0, rel_tol=1e-9)


def test_calculate_kelly_position_size_no_edge_is_zero():
    rm = _make_manager()
    size = rm.calculate_kelly_position_size(
        available_capital=1000.0,
        current_price=100.0,
        payoff_ratio=1.0,
        win_rate=0.4,
        leverage=50.0,
    )
    assert size == 0.0


def test_calculate_kelly_position_size_rejects_bad_price():
    rm = _make_manager()
    with pytest.raises(ValueError):
        rm.calculate_kelly_position_size(
            available_capital=1000.0,
            current_price=0.0,
            payoff_ratio=2.0,
            win_rate=0.6,
        )
