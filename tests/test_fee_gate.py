"""Tests for trading/fees/fee_gate.py (fee-aware trade viability gate)."""

import logging
import math

import numpy as np
import pytest

from trading.fees.binance_fee_calculator import BinanceFeeCalculator
from trading.fees.fee_gate import (
    DEFAULT_FUNDING_RATE,
    DEFAULT_TAKER_FEE,
    all_in_cost,
    is_trade_viable,
)

ENTRY = 50_000.0
EXIT = 50_100.0
QTY = 0.01  # contract quantity; entry notional = 500 USDT at every leverage


class TestAllInCost:
    def test_default_breakdown_exact_math(self):
        costs = all_in_cost(ENTRY, QTY, expected_exit_price=EXIT)
        # Entry notional=500, exit notional=501, and one funding period.
        assert costs["entry_fee"] == pytest.approx(0.2)
        assert costs["exit_fee"] == pytest.approx(0.2004)
        assert costs["funding_fee"] == pytest.approx(0.05)
        assert costs["total"] == pytest.approx(0.4504)
        assert set(costs) == {"entry_fee", "exit_fee", "funding_fee", "total"}

    def test_defaults_mirror_binance_constants(self):
        assert DEFAULT_TAKER_FEE == 0.0004  # 0.04% futures taker
        assert DEFAULT_FUNDING_RATE == 0.0001

    def test_leverage_is_not_part_of_the_cost_api(self):
        with pytest.raises(TypeError, match="leverage"):
            all_in_cost(ENTRY, QTY, expected_exit_price=EXIT, leverage=10.0)
        with pytest.raises(TypeError):
            all_in_cost(ENTRY, QTY, 10.0)

    def test_funding_periods_multiply_funding_only(self):
        costs = all_in_cost(
            ENTRY,
            QTY,
            expected_exit_price=EXIT,
            funding_periods=3,
        )
        assert costs["funding_fee"] == pytest.approx(0.15)
        assert costs["entry_fee"] == pytest.approx(0.2)
        assert costs["total"] == pytest.approx(0.5504)

    def test_zero_funding_periods(self):
        costs = all_in_cost(
            ENTRY,
            QTY,
            expected_exit_price=EXIT,
            funding_periods=0,
        )
        assert costs["funding_fee"] == 0.0
        assert costs["total"] == pytest.approx(0.4004)

    @pytest.mark.parametrize(
        "entry_price,exit_price,quantity",
        [
            (float("nan"), EXIT, QTY),
            (ENTRY, float("nan"), QTY),
            (ENTRY, EXIT, float("nan")),
            (float("inf"), EXIT, QTY),
            (ENTRY, float("inf"), QTY),
            (0.0, EXIT, QTY),
            (ENTRY, 0.0, QTY),
            (ENTRY, EXIT, 0.0),
            (-1.0, EXIT, QTY),
            (ENTRY, EXIT, -0.5),
            (None, EXIT, QTY),
        ],
    )
    def test_invalid_inputs_return_zero_costs(self, entry_price, exit_price, quantity):
        costs = all_in_cost(
            entry_price,
            quantity,
            expected_exit_price=exit_price,
        )
        assert costs == {
            "entry_fee": 0.0,
            "exit_fee": 0.0,
            "funding_fee": 0.0,
            "total": 0.0,
        }

    def test_non_finite_fee_params_return_zero_costs(self):
        assert (
            all_in_cost(
                ENTRY,
                QTY,
                expected_exit_price=EXIT,
                taker_fee=float("nan"),
            )["total"]
            == 0.0
        )

    def test_numpy_boolean_scalars_are_rejected_as_numbers(self):
        assert (
            all_in_cost(
                np.bool_(True),
                QTY,
                expected_exit_price=EXIT,
            )["total"]
            == 0.0
        )
        viable, net, costs = is_trade_viable(
            ENTRY,
            EXIT,
            QTY,
            taker_fee=np.bool_(True),
        )
        assert viable is False
        assert net == 0.0
        assert costs["total"] == 0.0
        assert (
            all_in_cost(
                ENTRY,
                QTY,
                expected_exit_price=EXIT,
                funding_rate=float("inf"),
            )["total"]
            == 0.0
        )

    @pytest.mark.parametrize("leverage", [1, 3, 10])
    def test_costs_match_binance_contract_math_at_every_leverage(
        self, leverage: int
    ) -> None:
        costs = all_in_cost(
            ENTRY,
            QTY,
            expected_exit_price=EXIT,
            funding_periods=0,
        )
        canonical = BinanceFeeCalculator().calculate_total_position_cost(
            entry_price=ENTRY,
            exit_price=EXIT,
            quantity=QTY,
            leverage=leverage,
            hours_held=0,
        )

        assert costs["entry_fee"] == pytest.approx(canonical["entry_fee"])
        assert costs["exit_fee"] == pytest.approx(canonical["exit_fee"])
        assert costs["total"] == pytest.approx(canonical["total_fees"])
        assert canonical["position_value"] == pytest.approx(ENTRY * QTY)
        assert canonical["margin_used"] == pytest.approx(ENTRY * QTY / leverage)

    @pytest.mark.parametrize(
        ("parameter", "value"),
        [
            ("taker_fee", float("nan")),
            ("taker_fee", float("inf")),
            ("taker_fee", "bad"),
            ("taker_fee", -0.1),
            ("taker_fee", True),
            ("funding_rate", -0.1),
            ("funding_periods", float("inf")),
            ("funding_periods", "bad"),
            ("funding_periods", -1),
            ("funding_periods", True),
        ],
    )
    def test_invalid_cost_parameters_return_zero_costs(
        self, parameter: str, value: object
    ) -> None:
        costs = all_in_cost(
            ENTRY,
            QTY,
            expected_exit_price=EXIT,
            **{parameter: value},
        )

        assert costs == {
            "entry_fee": 0.0,
            "exit_fee": 0.0,
            "funding_fee": 0.0,
            "total": 0.0,
        }


class TestIsTradeViable:
    def test_tight_move_eaten_by_fees(self):
        # gross = 0.10 vs entry/exit/funding costs of 0.45004.
        viable, net, costs = is_trade_viable(ENTRY, ENTRY + 10.0, QTY, side="BUY")
        assert viable is False
        assert net == pytest.approx(-0.35004)
        assert costs["total"] == pytest.approx(0.45004)

    def test_wide_move_viable_exact_net(self):
        # gross = 1.0; exit-price-aware total fees = 0.4504.
        viable, net, costs = is_trade_viable(ENTRY, ENTRY + 100.0, QTY, side="BUY")
        assert viable is True
        assert net == pytest.approx(0.5496)
        assert costs["entry_fee"] == pytest.approx(0.2)

    def test_sell_direction(self):
        # SELL profits on a drop: gross = (50000 - 49900) * 0.01 = 1.0
        viable, net, _ = is_trade_viable(ENTRY, ENTRY - 100.0, QTY, side="SELL")
        assert viable is True
        assert net == pytest.approx(0.5504)
        # A SELL into a rising price loses gross AND pays fees.
        viable_up, net_up, _ = is_trade_viable(ENTRY, ENTRY + 100.0, QTY, side="SELL")
        assert viable_up is False
        assert net_up == pytest.approx(-1.4504)

    def test_leverage_is_not_part_of_the_viability_api(self):
        with pytest.raises(TypeError, match="leverage"):
            is_trade_viable(ENTRY, EXIT, QTY, side="BUY", leverage=10.0)

    def test_min_net_profit_boundary_is_strict(self):
        # The threshold is strict: an exact net match must reject.
        _, exact_net, _ = is_trade_viable(ENTRY, EXIT, QTY)
        viable_at, net, _ = is_trade_viable(ENTRY, EXIT, QTY, min_net_profit=exact_net)
        assert net == pytest.approx(0.5496)
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
        assert net_short == pytest.approx(0.5504)

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

    @pytest.mark.parametrize(
        ("parameter", "value"),
        [
            ("taker_fee", float("nan")),
            ("taker_fee", float("inf")),
            ("taker_fee", "bad"),
            ("taker_fee", -0.1),
            ("taker_fee", True),
            ("funding_rate", -0.1),
            ("funding_periods", float("inf")),
            ("funding_periods", "bad"),
            ("funding_periods", -1),
            ("funding_periods", True),
        ],
    )
    def test_invalid_cost_parameters_fail_closed(
        self, parameter: str, value: object
    ) -> None:
        viable, net, costs = is_trade_viable(
            ENTRY,
            EXIT,
            QTY,
            **{parameter: value},
        )

        assert viable is False
        assert net == 0.0
        assert costs == {
            "entry_fee": 0.0,
            "exit_fee": 0.0,
            "funding_fee": 0.0,
            "total": 0.0,
        }

    @pytest.mark.parametrize(
        "min_net_profit",
        [float("inf"), -1.0, "bad", True, 10**400],
    )
    def test_invalid_minimum_profit_fails_closed(self, min_net_profit: object) -> None:
        viable, net, costs = is_trade_viable(
            ENTRY,
            EXIT,
            QTY,
            min_net_profit=min_net_profit,
        )

        assert viable is False
        assert net == 0.0
        assert costs["total"] == 0.0

    @pytest.mark.parametrize(
        ("entry_price", "exit_price", "quantity"),
        [
            (10**400, EXIT, QTY),
            (ENTRY, 10**400, QTY),
            (ENTRY, EXIT, 10**400),
            (1e308, 1.1e308, 2.0),
        ],
    )
    def test_conversion_and_calculation_overflow_fail_closed(
        self, entry_price: object, exit_price: object, quantity: object
    ) -> None:
        viable, net, costs = is_trade_viable(
            entry_price,
            exit_price,
            quantity,
        )

        assert viable is False
        assert math.isfinite(net)
        assert all(math.isfinite(value) for value in costs.values())

    def test_conversion_overflow_logging_never_formats_raw_value(self, caplog) -> None:
        huge_integer = 10**10_000

        with caplog.at_level(logging.WARNING, logger="trading.fees.fee_gate"):
            costs = all_in_cost(
                huge_integer,
                QTY,
                expected_exit_price=EXIT,
            )
            viable, net, viability_costs = is_trade_viable(
                huge_integer,
                EXIT,
                QTY,
            )

        messages = [record.getMessage() for record in caplog.records]
        assert costs["total"] == 0.0
        assert viable is False
        assert net == 0.0
        assert viability_costs["total"] == 0.0
        assert len(messages) == 2
        assert all("entry_price_type=int" in message for message in messages)

    def test_non_string_side_fails_closed_without_string_conversion(
        self, caplog
    ) -> None:
        huge_integer = 10**10_000

        with caplog.at_level(logging.WARNING, logger="trading.fees.fee_gate"):
            viable, net, costs = is_trade_viable(
                ENTRY,
                EXIT,
                QTY,
                side=huge_integer,
            )

        assert viable is False
        assert net == 0.0
        assert costs["total"] == 0.0
        assert (
            caplog.records[-1]
            .getMessage()
            .endswith("unknown side type int; treating trade as not viable")
        )

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
