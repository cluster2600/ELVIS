"""
Tests for trading.execution.exits: TrailingStop and dynamic_take_profit.

Plain pytest, stdlib-only assertions -- no torch, no network.
"""

import math

import pytest

from trading.execution.exits import (
    DEFAULT_REGIME_TP_PCT,
    TrailingStop,
    dynamic_take_profit,
)


class TestTrailingStopBuy:
    """BUY (long) side: stop trails below the high-water mark."""

    def test_first_update_seeds_with_max_of_entry_and_current(self):
        ts = TrailingStop(trail_pct=0.02)
        # current above entry -> extreme seeded at current
        should_close, stop = ts.update("p1", "BUY", 100.0, 105.0)
        assert should_close is False
        assert stop == pytest.approx(105.0 * 0.98)

    def test_first_update_seeds_with_entry_when_current_lower(self):
        ts = TrailingStop(trail_pct=0.02)
        should_close, stop = ts.update("p1", "BUY", 100.0, 99.5)
        # extreme = max(100, 99.5) = 100 -> stop = 98
        assert stop == pytest.approx(98.0)
        assert should_close is False

    def test_stop_rises_with_new_highs(self):
        ts = TrailingStop(trail_pct=0.02)
        _, stop1 = ts.update("p1", "BUY", 100.0, 100.0)
        _, stop2 = ts.update("p1", "BUY", 100.0, 110.0)
        _, stop3 = ts.update("p1", "BUY", 100.0, 120.0)
        assert stop1 < stop2 < stop3
        assert stop3 == pytest.approx(120.0 * 0.98)

    def test_stop_never_falls_on_pullback(self):
        ts = TrailingStop(trail_pct=0.02)
        ts.update("p1", "BUY", 100.0, 120.0)
        # pull back but stay above the stop (120 * 0.98 = 117.6)
        should_close, stop = ts.update("p1", "BUY", 100.0, 118.0)
        assert should_close is False
        assert stop == pytest.approx(120.0 * 0.98)  # unchanged high-water mark

    def test_triggers_when_price_crosses_stop(self):
        ts = TrailingStop(trail_pct=0.02)
        ts.update("p1", "BUY", 100.0, 120.0)
        should_close, stop = ts.update("p1", "BUY", 100.0, 117.0)
        assert should_close is True
        assert stop == pytest.approx(117.6)

    def test_triggers_exactly_at_stop_price(self):
        ts = TrailingStop(trail_pct=0.02)
        ts.update("p1", "BUY", 100.0, 100.0)
        should_close, stop = ts.update("p1", "BUY", 100.0, 98.0)
        assert stop == pytest.approx(98.0)
        assert should_close is True  # current <= stop triggers


class TestTrailingStopSell:
    """SELL (short) side: mirrored -- stop trails above the low-water mark."""

    def test_first_update_seeds_with_min_of_entry_and_current(self):
        ts = TrailingStop(trail_pct=0.02)
        should_close, stop = ts.update("s1", "SELL", 100.0, 95.0)
        assert should_close is False
        assert stop == pytest.approx(95.0 * 1.02)

    def test_first_update_seeds_with_entry_when_current_higher(self):
        ts = TrailingStop(trail_pct=0.02)
        _, stop = ts.update("s1", "SELL", 100.0, 100.5)
        # extreme = min(100, 100.5) = 100 -> stop = 102
        assert stop == pytest.approx(102.0)

    def test_stop_falls_with_new_lows(self):
        ts = TrailingStop(trail_pct=0.02)
        _, stop1 = ts.update("s1", "SELL", 100.0, 100.0)
        _, stop2 = ts.update("s1", "SELL", 100.0, 90.0)
        _, stop3 = ts.update("s1", "SELL", 100.0, 80.0)
        assert stop1 > stop2 > stop3
        assert stop3 == pytest.approx(80.0 * 1.02)

    def test_stop_never_rises_on_bounce(self):
        ts = TrailingStop(trail_pct=0.02)
        ts.update("s1", "SELL", 100.0, 80.0)
        # bounce but stay below the stop (80 * 1.02 = 81.6)
        should_close, stop = ts.update("s1", "SELL", 100.0, 81.0)
        assert should_close is False
        assert stop == pytest.approx(81.6)

    def test_triggers_when_price_crosses_stop(self):
        ts = TrailingStop(trail_pct=0.02)
        ts.update("s1", "SELL", 100.0, 80.0)
        should_close, stop = ts.update("s1", "SELL", 100.0, 82.0)
        assert should_close is True
        assert stop == pytest.approx(81.6)


class TestTrailingStopState:
    """Per-position state, clear(), and input hardening."""

    def test_positions_are_independent(self):
        ts = TrailingStop(trail_pct=0.02)
        ts.update("a", "BUY", 100.0, 120.0)
        _, stop_b = ts.update("b", "BUY", 100.0, 100.0)
        # position b's stop unaffected by a's high-water mark
        assert stop_b == pytest.approx(98.0)

    def test_clear_forgets_state(self):
        ts = TrailingStop(trail_pct=0.02)
        ts.update("p1", "BUY", 100.0, 120.0)
        ts.clear("p1")
        # re-seeded from scratch: extreme = max(100, 100) = 100
        _, stop = ts.update("p1", "BUY", 100.0, 100.0)
        assert stop == pytest.approx(98.0)

    def test_clear_unknown_id_is_noop(self):
        ts = TrailingStop()
        ts.clear("never-seen")  # must not raise

    def test_invalid_current_price_is_neutral(self):
        ts = TrailingStop()
        assert ts.update("p1", "BUY", 100.0, float("nan")) == (False, 0.0)
        assert ts.update("p1", "BUY", 100.0, -5.0) == (False, 0.0)
        assert ts.update("p1", "BUY", 100.0, 0.0) == (False, 0.0)
        assert ts.update("p1", "BUY", 100.0, float("inf")) == (False, 0.0)

    def test_invalid_entry_price_falls_back_to_current(self):
        ts = TrailingStop(trail_pct=0.02)
        should_close, stop = ts.update("p1", "BUY", float("nan"), 100.0)
        assert should_close is False
        assert stop == pytest.approx(98.0)

    def test_invalid_trail_pct_uses_default(self):
        assert TrailingStop(trail_pct=float("nan")).trail_pct == pytest.approx(0.02)
        assert TrailingStop(trail_pct=-0.1).trail_pct == pytest.approx(0.02)

    def test_side_is_case_insensitive(self):
        ts = TrailingStop(trail_pct=0.02)
        _, stop = ts.update("p1", "buy", 100.0, 110.0)
        assert stop == pytest.approx(110.0 * 0.98)

    def test_unknown_side_treated_as_buy(self):
        ts = TrailingStop(trail_pct=0.02)
        _, stop = ts.update("p1", "HOLD", 100.0, 110.0)
        assert stop == pytest.approx(110.0 * 0.98)

    def test_side_flip_reseeds_state(self):
        ts = TrailingStop(trail_pct=0.02)
        ts.update("p1", "BUY", 100.0, 120.0)
        # same id re-used as SELL: old high-water mark must not leak
        _, stop = ts.update("p1", "SELL", 100.0, 100.0)
        assert stop == pytest.approx(102.0)


class TestDynamicTakeProfitBuy:
    def test_trending(self):
        assert dynamic_take_profit("TRENDING", 100.0, "BUY") == pytest.approx(105.0)

    def test_reversal(self):
        assert dynamic_take_profit("REVERSAL", 100.0, "BUY") == pytest.approx(101.0)

    def test_ranging(self):
        assert dynamic_take_profit("RANGING", 100.0, "BUY") == pytest.approx(100.25)

    def test_choppy(self):
        assert dynamic_take_profit("CHOPPY", 100.0, "BUY") == pytest.approx(100.1)

    def test_unknown_regime_uses_ranging_fallback(self):
        assert dynamic_take_profit("LUNAR", 100.0, "BUY") == pytest.approx(100.25)

    def test_default_side_is_buy(self):
        assert dynamic_take_profit("TRENDING", 100.0) == pytest.approx(105.0)


class TestDynamicTakeProfitSell:
    def test_trending(self):
        assert dynamic_take_profit("TRENDING", 100.0, "SELL") == pytest.approx(95.0)

    def test_reversal(self):
        assert dynamic_take_profit("REVERSAL", 100.0, "SELL") == pytest.approx(99.0)

    def test_ranging(self):
        assert dynamic_take_profit("RANGING", 100.0, "SELL") == pytest.approx(99.75)

    def test_choppy(self):
        assert dynamic_take_profit("CHOPPY", 100.0, "SELL") == pytest.approx(99.9)

    def test_unknown_regime_uses_ranging_fallback(self):
        assert dynamic_take_profit("???", 100.0, "SELL") == pytest.approx(99.75)


class TestDynamicTakeProfitEdges:
    def test_regime_case_insensitive(self):
        assert dynamic_take_profit("trending", 100.0, "BUY") == pytest.approx(105.0)

    def test_side_case_insensitive(self):
        assert dynamic_take_profit("TRENDING", 100.0, "sell") == pytest.approx(95.0)

    def test_unknown_side_treated_as_buy(self):
        assert dynamic_take_profit("TRENDING", 100.0, "WAT") == pytest.approx(105.0)

    def test_custom_map_merges_over_defaults(self):
        custom = {"TRENDING": 0.10}
        assert dynamic_take_profit(
            "TRENDING", 100.0, "BUY", regime_pct=custom
        ) == pytest.approx(110.0)
        # keys not overridden keep their defaults
        assert dynamic_take_profit(
            "CHOPPY", 100.0, "BUY", regime_pct=custom
        ) == pytest.approx(100.1)

    def test_custom_map_can_add_new_regime(self):
        custom = {"PARABOLIC": 0.2}
        assert dynamic_take_profit(
            "PARABOLIC", 100.0, "BUY", regime_pct=custom
        ) == pytest.approx(120.0)

    def test_invalid_custom_value_ignored(self):
        custom = {"TRENDING": float("nan")}
        assert dynamic_take_profit(
            "TRENDING", 100.0, "BUY", regime_pct=custom
        ) == pytest.approx(105.0)

    def test_custom_map_does_not_mutate_defaults(self):
        before = dict(DEFAULT_REGIME_TP_PCT)
        dynamic_take_profit("TRENDING", 100.0, "BUY", regime_pct={"TRENDING": 0.5})
        assert DEFAULT_REGIME_TP_PCT == before

    def test_invalid_entry_price_is_neutral(self):
        assert math.isnan(dynamic_take_profit("TRENDING", float("nan"), "BUY"))
        assert dynamic_take_profit("TRENDING", 0.0, "BUY") == pytest.approx(0.0)
        assert dynamic_take_profit("TRENDING", -10.0, "BUY") == pytest.approx(-10.0)

    def test_none_regime_uses_fallback(self):
        assert dynamic_take_profit(None, 100.0, "BUY") == pytest.approx(100.25)
