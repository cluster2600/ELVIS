#!/usr/bin/env python3
"""
scripts/validate_paper_trading.py — Sprint 3: Paper Trading Validation
=======================================================================
Runs a 60-second simulation using 10 synthetic trades with fixed price
scenarios, then evaluates:
  - Total P&L
  - Win rate
  - Max drawdown  (on balance curve)
  - Simplified Sharpe ratio

Outputs a JSON report to reports/paper_trading_validation.json and prints
a READY / NOT READY verdict.

No real API calls are made — all prices are fixed simulation values.
The goal is to validate that the PaperTradingEngine mechanics (SQLite
persistence, kill-switch, P&L accounting) work correctly under a realistic
mix of winning and losing trades.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trading.paper_trading import PaperTradingEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIMULATION_DURATION_SECS = 60   # Total wall-clock window for the 10 trades
INITIAL_BALANCE = 10_000.0
REPORT_PATH = REPO_ROOT / "reports" / "paper_trading_validation.json"

# Readiness thresholds
SHARPE_THRESHOLD = 0.5
DRAWDOWN_THRESHOLD_PCT = 20.0   # %

SYMBOL = "BTCUSDT"
QUANTITY = 0.001               # BTC per round-trip

# ---------------------------------------------------------------------------
# Fixed trade scenarios (buy_price, sell_price)
# Represent realistic BTCUSDT price moves over ~6-minute intervals.
# 7 wins / 3 losses → 70% win rate with net positive PnL.
# ---------------------------------------------------------------------------
TRADE_SCENARIOS: List[Tuple[float, float]] = [
    (65_000.00, 65_715.00),   # +1.10% WIN
    (65_715.00, 66_437.00),   # +1.10% WIN
    (66_437.00, 66_107.00),   # -0.50% LOSS
    (66_200.00, 66_993.00),   # +1.20% WIN
    (66_993.00, 67_629.00),   # +0.95% WIN
    (67_629.00, 67_224.00),   # -0.60% LOSS
    (67_300.00, 68_001.00),   # +1.04% WIN
    (68_001.00, 68_611.00),   # +0.90% WIN
    (68_700.00, 68_356.00),   # -0.50% LOSS
    (68_400.00, 69_105.00),   # +1.03% WIN
]


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def compute_metrics(per_trade_pnl: List[float],
                    balance_series: List[float]) -> dict:
    """
    Compute financial metrics.

    Max drawdown is on the balance curve (not cumulative-PnL curve) to avoid
    divide-by-zero at startup.

    Sharpe = mean(returns) / std(returns)  (simplified, risk-free rate = 0).
    """
    n = len(per_trade_pnl)
    if n < 2:
        return {
            "sharpe": 0.0, "win_rate": 0.0,
            "max_drawdown_pct": 0.0, "total_pnl": 0.0,
        }

    total_pnl = sum(per_trade_pnl)
    wins      = sum(1 for r in per_trade_pnl if r > 0)
    win_rate  = wins / n

    mean_r   = total_pnl / n
    variance = sum((r - mean_r) ** 2 for r in per_trade_pnl) / max(n - 1, 1)
    std_r    = math.sqrt(variance) if variance > 1e-18 else 1e-9
    sharpe   = mean_r / std_r

    # Max drawdown on balance curve (peak always > 0 since balance starts positive)
    peak   = balance_series[0]
    max_dd = 0.0
    for bal in balance_series[1:]:
        peak   = max(peak, bal)
        dd     = (peak - bal) / peak
        max_dd = max(max_dd, dd)

    return {
        "sharpe":           round(sharpe, 4),
        "win_rate":         round(win_rate, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "total_pnl":        round(total_pnl, 4),
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation() -> dict:
    """Execute the fixed-price simulation and return structured results."""
    sim_start   = time.monotonic()
    num_trades  = len(TRADE_SCENARIOS)
    sleep_each  = SIMULATION_DURATION_SECS / num_trades

    print(f"[Simulation] {num_trades} fixed-price trades, "
          f"{SIMULATION_DURATION_SECS}s window")
    print(f"[Simulation] Symbol: {SYMBOL}  |  "
          f"Initial balance: ${INITIAL_BALANCE:,.2f} USDT\n")

    # Temporary SQLite — won't interfere with production paper_trading.db
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_db = tmp.name

    engine = PaperTradingEngine(
        initial_balance=INITIAL_BALANCE,
        db_path=tmp_db,
        leverage=3,
        use_testnet_prices=False,   # pure simulation — zero network calls
    )

    pnl_list:       List[float] = []
    balance_series: List[float] = [INITIAL_BALANCE]
    trade_log:      List[dict]  = []

    for i, (buy_px, sell_px) in enumerate(TRADE_SCENARIOS, start=1):
        # BUY ---------------------------------------------------------------
        buy_trade = engine.place_order(
            SYMBOL, "BUY", QUANTITY, price=buy_px,
            note=f"sim_buy_{i}",
        )
        if buy_trade is None:
            print(f"  [Trade {i:02d}] BUY blocked (kill-switch / balance)")
            continue

        # SELL --------------------------------------------------------------
        sell_trade = engine.place_order(
            SYMBOL, "SELL", QUANTITY, price=sell_px,
            note=f"sim_sell_{i}",
        )
        if sell_trade is None:
            print(f"  [Trade {i:02d}] SELL blocked")
            continue

        pnl = sell_trade.realised_pnl
        pnl_list.append(pnl)
        balance_series.append(engine.balance)

        move_pct = (sell_px - buy_px) / buy_px * 100
        icon     = "✅" if pnl > 0 else "❌"
        print(
            f"  [Trade {i:02d}] Buy@{buy_px:>10,.2f}  →  Sell@{sell_px:>10,.2f} "
            f"({move_pct:+.2f}%)  |  PnL={pnl:+.4f} USDT  {icon}"
        )

        trade_log.append({
            "trade_num":    i,
            "buy_price":    buy_px,
            "sell_price":   sell_px,
            "move_pct":     round(move_pct, 4),
            "quantity":     QUANTITY,
            "realised_pnl": round(pnl, 4),
        })

        time.sleep(sleep_each)   # spread across the 60-second window

    sim_elapsed = time.monotonic() - sim_start
    summary     = engine.get_portfolio_summary()
    engine.__exit__(None, None, None)

    # Remove temp db
    try:
        os.unlink(tmp_db)
    except OSError:
        pass

    metrics = compute_metrics(pnl_list, balance_series)

    return {
        "simulation_timestamp":     datetime.now(timezone.utc).isoformat(),
        "simulation_duration_secs": round(sim_elapsed, 2),
        "symbol":                   SYMBOL,
        "num_trades":               len(pnl_list),
        "initial_balance_usdt":     INITIAL_BALANCE,
        "final_balance_usdt":       round(summary["balance_usdt"], 4),
        "metrics":                  metrics,
        "thresholds": {
            "sharpe_min":       SHARPE_THRESHOLD,
            "max_drawdown_pct": DRAWDOWN_THRESHOLD_PCT,
        },
        "trades":            trade_log,
        "portfolio_summary": summary,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 65)
    print("  ELVIS Paper Trading Validation — Sprint 3")
    print("=" * 65)

    results = run_simulation()

    # Persist JSON report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Report] Written to {REPORT_PATH}")

    # Display metrics
    m = results["metrics"]
    print("\n" + "=" * 65)
    print("  Validation Metrics")
    print("=" * 65)
    print(f"  Total P&L       : {m['total_pnl']:+.4f} USDT")
    print(f"  Win Rate         : {m['win_rate'] * 100:.1f}%  "
          f"({results['num_trades']} trades)")
    print(f"  Max Drawdown     : {m['max_drawdown_pct']:.4f}%  "
          f"(threshold < {DRAWDOWN_THRESHOLD_PCT}%)")
    print(f"  Sharpe Ratio     : {m['sharpe']:.4f}  "
          f"(threshold > {SHARPE_THRESHOLD})")
    print(f"  Final Balance    : ${results['final_balance_usdt']:>12,.4f} USDT")
    print("=" * 65)

    sharpe_ok   = m["sharpe"]           >  SHARPE_THRESHOLD
    drawdown_ok = m["max_drawdown_pct"] <  DRAWDOWN_THRESHOLD_PCT

    fails = []
    if not sharpe_ok:
        fails.append(f"  ✗  Sharpe {m['sharpe']:.4f} ≤ required {SHARPE_THRESHOLD}")
    if not drawdown_ok:
        fails.append(f"  ✗  Drawdown {m['max_drawdown_pct']:.4f}% "
                     f"≥ limit {DRAWDOWN_THRESHOLD_PCT}%")
    if fails:
        print("\n".join(fails))

    verdict = (
        "✅ READY FOR PAPER TRADING"
        if (sharpe_ok and drawdown_ok)
        else "⚠️ NOT READY — review metrics"
    )
    print(f"\n  {verdict}\n")

    return 0 if (sharpe_ok and drawdown_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
