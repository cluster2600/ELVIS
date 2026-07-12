"""Tests for trading/fees/fee_gate.py (fee-aware trade viability gate)."""

import math

import pytest

from trading.fees.fee_gate import (
    DEFAULT_FUNDING_RATE,
    DEFAULT_TAKER_FEE,
    all_in_cost,
    is_trade_viable,
)

ENTRY = 50_000.0
QTY = 0.01  # notional = 500 USDT at 1x


class TestAllInCost:
    def test_default_breakdown_exact_math(self):
        costs = all_in_cost(ENTRY, QTY)
        # notional = 500; taker 0.0004 -> 0.2 each side; funding 0.0001 -> 0.05
        assert costs["entry_fee"] == pytest.approx(0.2)
        assert costs["exit_fee"] == pytest.approx(0.2)
        assert costs["funding_fee"] == pytest.approx(0.05)
        assert costs["total"] == pytest.approx(0.45)
        assert set(costs) == {"entry_fee", "exit_fee", "funding_fee", "total"}

    def test_defaults_mirror_binance_constants(self):
        assert DEFAULT_TAKER_FEE == 0.0004  # 0.04% futures taker
        assert DEFAULT_FUNDING_RATE == 0.0001

    def test_leverage_scales_notional(self):
        costs_1x = all_in_cost(ENTRY, QTY, leverage=1.0)
        costs_10x = all_in_cost(ENTRY, QTY, leverage=10.0)
        for key in ("entry_fee", "exit_fee", "funding_fee", "total"):
            assert costs_10x[key] == pytest.approx(10.0 * costs_1x[key])

    def test_funding_periods_multiply_funding_only(self):
        costs = all_in_cost(ENTRY, QTY, funding_periods=3)
        assert costs["funding_fee"] == pytest.approx(0.15)
        assert costs["entry_fee"] == pytest.approx(0.2)
        assert costs["total"] == pytest.approx(0.55)

    def test_zero_funding_periods(self):
        costs = all_in_cost(ENTRY, QTY, funding_periods=0)
        assert costs["funding_fee"] == 0.0
        assert costs["total"] == pytest.approx(0.4)

    @pytest.mark.parametrize(
        "entry_price,quantity,leverage",
        [
            (float("nan"), QTY, 1.0),
            (ENTRY, float("nan"), 1.0),
            (ENTRY, QTY, float("nan")),
            (float("inf"), QTY, 1.0),
            (0.0, QTY, 1.0),
            (ENTRY, 0.0, 1.0),
            (-1.0, QTY, 1.0),
            (ENTRY, -0.5, 1.0),
            (ENTRY, QTY, 0.0),
            (None, QTY, 1.0),
        ],
    )
    def test_invalid_inputs_return_zero_costs(self, entry_price, quantity, leverage):
        costs = all_in_cost(entry_price, quantity, leverage=leverage)
        assert costs == {
            "entry_fee": 0.0,
            "exit_fee": 0.0,
            "funding_fee": 0.0,
            "total": 0.0,
        }

    def test_non_finite_fee_params_return_zero_costs(self):
        assert all_in_cost(ENTRY, QTY, taker_fee=float("nan"))["total"] == 0.0
        assert all_in_cost(ENTRY, QTY, funding_rate=float("inf"))["total"] == 0.0


class TestIsTradeViable:
    def test_tight_move_eaten_by_fees(self):
        # gross = 10 * 0.01 = 0.10 vs total fees 0.45 -> net -0.35
        viable, net, costs = is_trade_viable(ENTRY, ENTRY + 10.0, QTY, side="BUY")
        assert viable is False
        assert net == pytest.approx(-0.35)
        assert costs["total"] == pytest.approx(0.45)

    def test_wide_move_viable_exact_net(self):
        # gross = 100 * 0.01 = 1.0; net = 1.0 - 0.45 = 0.55
        viable, net, costs = is_trade_viable(ENTRY, ENTRY + 100.0, QTY, side="BUY")
        assert viable is True
        assert net == pytest.approx(0.55)
        assert costs["entry_fee"] == pytest.approx(0.2)

    def test_sell_direction(self):
        # SELL profits on a drop: gross = (50000 - 49900) * 0.01 = 1.0
        viable, net, _ = is_trade_viable(ENTRY, ENTRY - 100.0, QTY, side="SELL")
        assert viable is True
        assert net == pytest.approx(0.55)
        # A SELL into a rising price loses gross AND pays fees.
        viable_up, net_up, _ = is_trade_viable(ENTRY, ENTRY + 100.0, QTY, side="SELL")
        assert viable_up is False
        assert net_up == pytest.approx(-1.45)

    def test_leverage_scales_gross_and_fees(self):
        viable, net, costs = is_trade_viable(
            ENTRY, ENTRY + 100.0, QTY, side="BUY", leverage=10.0
        )
        # gross = 1.0 * 10 = 10.0; fees = 0.45 * 10 = 4.5; net = 5.5
        assert viable is True
        assert net == pytest.approx(5.5)
        assert costs["total"] == pytest.approx(4.5)

    def test_min_net_profit_boundary_is_strict(self):
        # net is exactly 0.55: threshold 0.55 must reject, just below accepts.
        viable_at, net, _ = is_trade_viable(
            ENTRY, ENTRY + 100.0, QTY, min_net_profit=0.55
        )
        assert net == pytest.approx(0.55)
        assert viable_at is False
        viable_below, _, _ = is_trade_viable(
            ENTRY, ENTRY + 100.0, QTY, min_net_profit=0.5
        )
        assert viable_below is True

    def test_side_aliases_and_case_insensitive(self):
        viable_buy, net_buy, _ = is_trade_viable(ENTRY, ENTRY + 100.0, QTY, side="buy")
        viable_long, net_long, _ = is_trade_viable(
            ENTRY, ENTRY + 100.0, QTY, side="LONG"
        )
        assert viable_buy is viable_long is True
        assert net_buy == pytest.approx(net_long)
        viable_short, net_short, _ = is_trade_viable(
            ENTRY, ENTRY - 100.0, QTY, side="short"
        )
        assert viable_short is True
        assert net_short == pytest.approx(0.55)

    def test_unknown_side_is_neutral(self):
        viable, net, costs = is_trade_viable(ENTRY, ENTRY + 100.0, QTY, side="HODL")
        assert viable is False
        assert net == 0.0
        assert costs["total"] == 0.0

    @pytest.mark.parametrize(
        "entry_price,exit_price,quantity",
        [
            (float("nan"), ENTRY + 100.0, QTY),
            (ENTRY, float("nan"), QTY),
            (ENTRY, ENTRY + 100.0, float("nan")),
            (ENTRY, ENTRY + 100.0, 0.0),
            (ENTRY, ENTRY + 100.0, -1.0),
            (ENTRY, 0.0, QTY),
            (ENTRY, float("inf"), QTY),
        ],
    )
    def test_invalid_inputs_neutral_no_raise(self, entry_price, exit_price, quantity):
        viable, net, costs = is_trade_viable(entry_price, exit_price, quantity)
        assert viable is False
        assert net == 0.0
        assert costs["total"] == 0.0

    def test_nan_min_net_profit_neutral(self):
        viable, net, _ = is_trade_viable(
            ENTRY, ENTRY + 100.0, QTY, min_net_profit=float("nan")
        )
        assert viable is False
        assert net == 0.0

    def test_fee_kw_passthrough(self):
        # Zero out all fees: net == gross and any positive move is viable.
        viable, net, costs = is_trade_viable(
            ENTRY,
            ENTRY + 10.0,
            QTY,
            taker_fee=0.0,
            funding_rate=0.0,
        )
        assert viable is True
        assert net == pytest.approx(0.10)
        assert costs["total"] == 0.0

    def test_breakeven_move_not_viable(self):
        # No price move: gross 0, net = -fees.
        viable, net, _ = is_trade_viable(ENTRY, ENTRY, QTY)
        assert viable is False
        assert net == pytest.approx(-0.45)
        assert math.isfinite(net)
