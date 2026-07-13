"""Paper-fill integrity (review on #48): regression tests for the
mock-price refusal (bug #3) and percent-of-notional netting (bug #4)."""

import logging
from unittest.mock import patch

import pytest

import trading.execution.binance_executor as bx
from trading.execution.binance_executor import BinanceExecutor


@pytest.fixture
def ex():
    e = BinanceExecutor(logger=logging.getLogger("t"))
    e.db_available = True
    return e


class TestMockPriceRefusal:
    def test_none_price_refused(self, ex):
        with patch.object(bx, "record_trade") as rt:
            assert ex.execute_sell("BTCUSDT", 1.0, None) == {}
        rt.assert_not_called()  # nothing reaches the books

    def test_zero_price_refused(self, ex):
        with patch.object(bx, "record_trade") as rt:
            assert ex.execute_buy("BTCUSDT", 1.0, 0.0) == {}
        rt.assert_not_called()

    def test_negative_price_refused(self, ex):
        assert ex.execute_buy("BTCUSDT", 1.0, -5.0) == {}


class TestNettingThresholds:
    """Netting closes the OPPOSITE position only past pct-of-notional
    thresholds (were hardcoded -$50/+$25 — unreachable at micro notional)."""

    def _run(self, ex, current_price, monkeypatch, opp_entry=100.0, qty=1.0):
        monkeypatch.delenv("ELVIS_NET_SL_PCT", raising=False)
        monkeypatch.delenv("ELVIS_NET_TP_PCT", raising=False)
        opp = (1, "BTCUSDT", "SELL", opp_entry, qty, 1.0, None)
        calls = {}
        with patch.object(bx, "get_open_positions", return_value=[opp]):
            with patch.object(bx, "record_trade") as rt:
                with patch.object(bx, "close_open_position") as cp:
                    with patch.object(bx, "add_open_position"):
                        with patch.object(
                            ex,
                            "_calculate_position_pnl",
                            return_value=(opp_entry - current_price) * qty,
                        ):
                            ex.execute_buy("BTCUSDT", qty, current_price)
        calls["recorded"] = rt.call_args_list
        calls["closed"] = cp.called
        return calls

    def test_small_move_does_not_force_close(self, ex, monkeypatch):
        # SELL from 100, price 100.2 -> pnl -0.2 = -0.2% of notional: within
        # the 0.5% netting stop, so the opposite position must survive
        calls = self._run(ex, current_price=100.2, monkeypatch=monkeypatch)
        assert calls["closed"] is False

    def test_loss_past_pct_forces_close(self, ex, monkeypatch):
        # price 101 -> pnl -1.0 = -1% of $100 notional: past the 0.5% stop
        calls = self._run(ex, current_price=101.0, monkeypatch=monkeypatch)
        assert calls["closed"] is True
        # and the close is recorded with the REAL caller price, never a mock
        assert calls["recorded"][0].args[2] == 101.0

    def test_profit_past_pct_takes_profit(self, ex, monkeypatch):
        # price 99 -> pnl +1.0 = +1% > 0.25% target
        calls = self._run(ex, current_price=99.0, monkeypatch=monkeypatch)
        assert calls["closed"] is True
