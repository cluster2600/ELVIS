"""Tests for portfolio allocation / risk parity / rebalancing helpers."""

import numpy as np

from trading.risk.portfolio_allocation import (
    inverse_volatility_weights,
    rebalance_orders,
    risk_parity_weights,
)


def test_inverse_vol_weights_favor_low_vol_and_sum_to_one():
    rng = np.random.RandomState(0)
    low = rng.normal(0, 0.01, 500)
    high = rng.normal(0, 0.05, 500)
    w = inverse_volatility_weights(np.column_stack([low, high]))
    assert abs(w.sum() - 1.0) < 1e-9
    assert w[0] > w[1]  # lower-vol asset gets more weight


def test_risk_parity_equalizes_risk_contributions():
    cov = np.array([[0.04, 0.0], [0.0, 0.01]])  # asset0 4x variance of asset1
    w = risk_parity_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-6
    m = cov @ w
    rc = w * m / (w @ m)
    assert abs(rc[0] - rc[1]) < 1e-3  # equal risk contributions
    assert w[1] > w[0]  # lower-variance asset carries more weight


def test_rebalance_orders_and_min_trade_dust_filter():
    orders = rebalance_orders(
        current_positions={"BTC": 600.0, "ETH": 400.0},
        target_weights={"BTC": 0.5, "ETH": 0.5},
        equity=1000.0,
        min_trade_usd=10.0,
    )
    by = {o["symbol"]: o for o in orders}
    assert by["BTC"]["side"] == "SELL" and by["BTC"]["usd_amount"] == 100.0
    assert by["ETH"]["side"] == "BUY" and by["ETH"]["usd_amount"] == 100.0
    # already balanced -> tiny drift below min_trade produces no orders
    assert rebalance_orders({"BTC": 501.0}, {"BTC": 0.5}, 1000.0, 10.0) == []
