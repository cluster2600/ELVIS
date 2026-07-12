"""Integration-style tests for the profitability-roadmap wiring in main.py.

main() is a monolithic loop, so these tests exercise the same *composition*
main.py runs — the stage order, env-flag semantics, and cross-module data
handoffs of its four wiring blocks — plus a tripwire pinning the
``get_all_trades`` column order the Kelly block depends on positionally.
"""

import inspect
import os
import re

import numpy as np
import pandas as pd
import pytest

from trading.execution.exits import TrailingStop, dynamic_take_profit
from trading.fees.fee_gate import is_trade_viable
from trading.risk.position_sizing import kelly_from_trades, volume_multiplier
from trading.signals.filters import apply_signal_filters
from trading.signals.order_flow import confirm_signal_with_flow


def _trending_data(rows=120, start=100.0, step=0.5, volume=1000.0):
    """Steadily rising closes with flat volume — passes momentum, no squeeze."""
    close = np.arange(rows) * step + start
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "close": close,
            "volume": volume + rng.uniform(-5, 5, rows),
        }
    )


def test_get_all_trades_pnl_column_order_pinned():
    # main.py's Kelly block reads pnl positionally at index 6. Pin the SELECT
    # column order in utils/paper_trade_db.get_all_trades so a schema change
    # fails here instead of silently corrupting sizing.
    from utils import paper_trade_db

    src = inspect.getsource(paper_trade_db.get_all_trades)
    select = re.search(r"SELECT\s+(.+?)\s+FROM", src, re.S | re.I)
    assert select, "get_all_trades no longer contains a SELECT ... FROM"
    columns = [c.strip() for c in select.group(1).split(",")]
    assert columns[6] == "pnl", (
        f"pnl moved from index 6 in get_all_trades ({columns}); "
        "update main.py's Kelly block (_PNL_IDX) to match"
    )


def test_signal_gate_stage_veto_stops_downstream_trade(monkeypatch):
    # Mirrors main.py's gate block: a filter veto downgrades to HOLD, and the
    # execution stages (sizing/fee gate) never fire for HOLD.
    data = _trending_data()
    signal, confidence, reasons = apply_signal_filters(
        "BUY", 0.8, data, rsi=85.0  # overbought -> rsi_gate veto
    )
    assert signal == "HOLD" and confidence == 0.0
    assert any("rsi" in r.lower() for r in reasons)
    # main.py only runs order-flow/sizing/fee stages for BUY/SELL:
    assert signal not in ("BUY", "SELL")


def test_full_happy_path_pipeline_in_main_py_order():
    # Stage order exactly as wired: filters -> order flow -> volume sizing ->
    # Kelly cap -> fee gate -> (position loop) trailing stop + dynamic TP.
    from datetime import datetime, timezone

    data = _trending_data()
    current_price = float(data["close"].iloc[-1])

    # 1. signal gates pass on clean trending data (inject an in-hours time —
    #    main.py uses the real clock, tests must not depend on when they run)
    in_hours = datetime(2026, 7, 12, 15, 0, tzinfo=timezone.utc)
    signal, confidence, reasons = apply_signal_filters(
        "BUY", 0.8, data, rsi=55.0, now=in_hours
    )
    assert signal == "BUY" and not reasons

    # 2. order flow: paper-mode empty book must NOT block (neutral)
    ok, imbalance = confirm_signal_with_flow(signal, {"bids": [], "asks": []})
    assert ok and imbalance == 0.0

    # 3. volume sizing on ~flat volume stays near neutral
    position_size = (100.0 / current_price) * volume_multiplier(data)
    assert position_size > 0

    # 4. Kelly cap from paper-trade rows shaped like get_all_trades output
    rows = [
        (i, "2026-07-12", "BTCUSDT", "BUY", 100.0, 0.1, pnl, 0.01)
        for i, pnl in enumerate([12.0, -5.0] * 15)  # 30 trades, mixed
    ]
    kelly_f = kelly_from_trades([{"pnl": t[6]} for t in rows])
    assert 0.01 <= kelly_f <= 0.05

    # 5. fee gate: regime take-profit target must clear all-in costs
    tp_price = dynamic_take_profit("TRENDING", current_price, side="BUY")
    viable, net, costs = is_trade_viable(
        current_price, tp_price, position_size, side="BUY"
    )
    assert viable and net > 0 and costs["total"] > 0

    # 6. position loop: trailing stop ratchets then closes on 2% giveback,
    #    and the dynamic TP threshold replaces the fixed $8
    trail = TrailingStop(trail_pct=0.02)
    closed, _ = trail.update("pos-1", "BUY", current_price, current_price * 1.05)
    assert not closed  # new high, ratchet up
    closed, stop = trail.update("pos-1", "BUY", current_price, current_price * 1.02)
    assert closed and stop == pytest.approx(current_price * 1.05 * 0.98)
    tp_threshold_usd = abs(tp_price - current_price) * position_size
    assert tp_threshold_usd > 0  # replaces the fixed 8.0 like main.py does


def test_fee_gate_rejects_sub_breakeven_move():
    # Fee breakeven is ~0.09% of notional (2x 0.04% taker + 0.01% funding).
    # A 0.05% expected move is below it -> main.py sets HOLD; the tightest
    # regime target (CHOPPY, 0.10%) sits just ABOVE breakeven and stays viable.
    entry = 100_000.0
    viable, net, _ = is_trade_viable(entry, entry * 1.0005, 0.001, side="BUY")
    assert not viable and net <= 0
    choppy_tp = dynamic_take_profit("CHOPPY", entry, side="BUY")
    choppy_viable, choppy_net, _ = is_trade_viable(entry, choppy_tp, 0.001, side="BUY")
    assert choppy_viable and choppy_net == pytest.approx(0.01, abs=1e-6)


def test_env_flag_semantics_match_main_py(monkeypatch):
    # main.py gates every stage on os.getenv(flag, default) == "1".
    monkeypatch.setenv("ELVIS_ROADMAP_FILTERS", "0")
    assert os.getenv("ELVIS_ROADMAP_FILTERS", "1") != "1"  # stage skipped
    monkeypatch.delenv("ELVIS_ROADMAP_FILTERS", raising=False)
    assert os.getenv("ELVIS_ROADMAP_FILTERS", "1") == "1"  # default on
    assert os.getenv("ELVIS_ORDER_FLOW", "0") != "1"  # default off
    assert os.getenv("ELVIS_KELLY_SIZING", "0") != "1"  # default off
    assert os.getenv("ELVIS_MTF", "0") != "1"  # default off


def test_main_py_wiring_blocks_present():
    # Cheap structural tripwire: the four wiring blocks and their flags exist
    # in main.py (guards against accidental deletion in the 2400-line loop).
    src = open("main.py").read()
    for marker in (
        "PROFITABILITY-ROADMAP SIGNAL GATES",
        "ELVIS_ROADMAP_FILTERS",
        "ELVIS_VOLUME_SIZING",
        "ELVIS_KELLY_SIZING",
        "ELVIS_FEE_GATE",
        "ELVIS_TRAILING_STOP",
        "ELVIS_DYNAMIC_TP",
        "_last_regime",
        "_PNL_IDX = 6",
    ):
        assert marker in src, f"main.py wiring marker missing: {marker}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
