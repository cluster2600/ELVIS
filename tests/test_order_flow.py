"""Tests for trading.signals.order_flow (roadmap item #12).

Covers order-book imbalance computation (balanced/one-sided/empty books,
string quantities, depth truncation, malformed levels), threshold mapping to
BUY/SELL/NEUTRAL, and signal confirmation/contradiction blocking.
"""

import math

import pytest

from trading.signals.order_flow import (
    confirm_signal_with_flow,
    order_book_imbalance,
    order_flow_signal,
)


def make_book(bid_qtys, ask_qtys, price=100.0):
    """Build a Binance-shaped order book from per-level quantities."""
    bids = [[price - i, qty] for i, qty in enumerate(bid_qtys)]
    asks = [[price + 1 + i, qty] for i, qty in enumerate(ask_qtys)]
    return {"bids": bids, "asks": asks}


class TestOrderBookImbalance:
    def test_balanced_book_is_zero(self):
        book = make_book([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert order_book_imbalance(book) == pytest.approx(0.0)

    def test_bid_heavy_book_is_positive(self):
        book = make_book([3.0], [1.0])
        assert order_book_imbalance(book) == pytest.approx(0.5)

    def test_ask_heavy_book_is_negative(self):
        book = make_book([1.0], [3.0])
        assert order_book_imbalance(book) == pytest.approx(-0.5)

    def test_only_bids_hits_upper_bound(self):
        book = {"bids": [[100.0, 5.0]], "asks": []}
        assert order_book_imbalance(book) == pytest.approx(1.0)

    def test_only_asks_hits_lower_bound(self):
        book = {"bids": [], "asks": [[101.0, 5.0]]}
        assert order_book_imbalance(book) == pytest.approx(-1.0)

    def test_empty_book_paper_mode(self):
        assert order_book_imbalance({"bids": [], "asks": []}) == 0.0

    def test_missing_keys(self):
        assert order_book_imbalance({}) == 0.0

    def test_none_book(self):
        assert order_book_imbalance(None) == 0.0

    def test_non_mapping_book(self):
        assert order_book_imbalance([["100", "1"]]) == 0.0  # type: ignore[arg-type]

    def test_string_quantities_binance_rest(self):
        book = {
            "bids": [["100.0", "3.0"], ["99.5", "1.0"]],
            "asks": [["100.5", "1.0"], ["101.0", "1.0"]],
        }
        # bid_liq=4, ask_liq=2 -> (4-2)/6
        assert order_book_imbalance(book) == pytest.approx(2.0 / 6.0)

    def test_depth_truncation(self):
        # Deep bid liquidity beyond the requested depth must be ignored.
        book = make_book([1.0, 1.0, 100.0], [1.0, 1.0])
        assert order_book_imbalance(book, depth=2) == pytest.approx(0.0)
        # Full depth sees the large third bid level.
        assert order_book_imbalance(book, depth=10) > 0.9

    def test_zero_or_negative_depth_is_neutral(self):
        book = make_book([5.0], [1.0])
        assert order_book_imbalance(book, depth=0) == 0.0
        assert order_book_imbalance(book, depth=-3) == 0.0

    def test_malformed_levels_are_skipped(self):
        book = {
            "bids": [["100.0", "2.0"], ["bad"], None, ["99.0", "not-a-number"]],
            "asks": [["101.0", "1.0"], 42],
        }
        # Only the two valid levels count: (2-1)/3
        assert order_book_imbalance(book) == pytest.approx(1.0 / 3.0)

    def test_nan_and_negative_quantities_are_skipped(self):
        book = {
            "bids": [[100.0, float("nan")], [99.0, -5.0], [98.0, 2.0]],
            "asks": [[101.0, 2.0]],
        }
        assert order_book_imbalance(book) == pytest.approx(0.0)

    def test_zero_quantities_are_neutral(self):
        book = {"bids": [[100.0, 0.0]], "asks": [[101.0, 0.0]]}
        assert order_book_imbalance(book) == 0.0

    def test_result_within_bounds(self):
        book = make_book([10.0, 20.0, 0.5], [0.1, 7.0])
        assert -1.0 <= order_book_imbalance(book) <= 1.0


class TestOrderFlowSignal:
    def test_buy_above_threshold(self):
        assert order_flow_signal(0.5) == "BUY"

    def test_sell_below_negative_threshold(self):
        assert order_flow_signal(-0.5) == "SELL"

    def test_neutral_inside_band(self):
        assert order_flow_signal(0.0) == "NEUTRAL"
        assert order_flow_signal(0.1) == "NEUTRAL"
        assert order_flow_signal(-0.1) == "NEUTRAL"

    def test_exact_threshold_is_neutral(self):
        assert order_flow_signal(0.15) == "NEUTRAL"
        assert order_flow_signal(-0.15) == "NEUTRAL"

    def test_just_past_threshold_is_directional(self):
        assert order_flow_signal(0.150001) == "BUY"
        assert order_flow_signal(-0.150001) == "SELL"

    def test_custom_threshold(self):
        assert order_flow_signal(0.2, threshold=0.3) == "NEUTRAL"
        assert order_flow_signal(0.2, threshold=0.1) == "BUY"

    def test_nan_is_neutral(self):
        assert order_flow_signal(float("nan")) == "NEUTRAL"

    def test_non_numeric_is_neutral(self):
        assert order_flow_signal(None) == "NEUTRAL"  # type: ignore[arg-type]
        assert order_flow_signal("oops") == "NEUTRAL"  # type: ignore[arg-type]

    def test_extreme_values(self):
        assert order_flow_signal(1.0) == "BUY"
        assert order_flow_signal(-1.0) == "SELL"


class TestConfirmSignalWithFlow:
    def test_buy_confirmed_by_bid_heavy_book(self):
        book = make_book([3.0], [1.0])
        confirmed, imbalance = confirm_signal_with_flow("BUY", book)
        assert confirmed is True
        assert imbalance == pytest.approx(0.5)

    def test_buy_blocked_by_ask_heavy_book(self):
        book = make_book([1.0], [3.0])
        confirmed, imbalance = confirm_signal_with_flow("BUY", book)
        assert confirmed is False
        assert imbalance == pytest.approx(-0.5)

    def test_sell_confirmed_by_ask_heavy_book(self):
        book = make_book([1.0], [3.0])
        confirmed, imbalance = confirm_signal_with_flow("SELL", book)
        assert confirmed is True
        assert imbalance == pytest.approx(-0.5)

    def test_sell_blocked_by_bid_heavy_book(self):
        book = make_book([3.0], [1.0])
        confirmed, imbalance = confirm_signal_with_flow("SELL", book)
        assert confirmed is False
        assert imbalance == pytest.approx(0.5)

    def test_neutral_flow_never_blocks(self):
        balanced = make_book([1.0], [1.0])
        for signal in ("BUY", "SELL"):
            confirmed, imbalance = confirm_signal_with_flow(signal, balanced)
            assert confirmed is True
            assert imbalance == pytest.approx(0.0)

    def test_empty_book_paper_mode_never_blocks(self):
        empty = {"bids": [], "asks": []}
        for signal in ("BUY", "SELL"):
            confirmed, imbalance = confirm_signal_with_flow(signal, empty)
            assert confirmed is True
            assert imbalance == 0.0

    def test_none_book_never_blocks(self):
        confirmed, imbalance = confirm_signal_with_flow("BUY", None)
        assert confirmed is True
        assert imbalance == 0.0

    def test_non_directional_signal_always_confirmed(self):
        bid_heavy = make_book([3.0], [1.0])
        confirmed, imbalance = confirm_signal_with_flow("HOLD", bid_heavy)
        assert confirmed is True
        assert imbalance == pytest.approx(0.5)

    def test_signal_case_insensitive(self):
        ask_heavy = make_book([1.0], [3.0])
        confirmed, _ = confirm_signal_with_flow("buy", ask_heavy)
        assert confirmed is False
        confirmed, _ = confirm_signal_with_flow(" sell ", ask_heavy)
        assert confirmed is True

    def test_custom_threshold_widens_neutral_band(self):
        # imbalance = -0.5: blocks BUY at default threshold, passes at 0.6.
        ask_heavy = make_book([1.0], [3.0])
        confirmed, _ = confirm_signal_with_flow("BUY", ask_heavy, threshold=0.6)
        assert confirmed is True

    def test_returns_tuple_of_bool_and_float(self):
        result = confirm_signal_with_flow("BUY", make_book([2.0], [1.0]))
        assert isinstance(result, tuple)
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        assert not math.isnan(result[1])
