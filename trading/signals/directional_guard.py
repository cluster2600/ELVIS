"""Directional performance guard — learn from realized P&L per side.

Forensics of the honest book (post-PR-#48) found the account bled from a
systematic short bias: BTCUSDT SELL won **3.2% of its last 95 trades** while
nothing fed that back into the decision. This guard reads the realized
win rate per (symbol, side) straight from ``np.trades`` and raises the
confidence bar for a direction that has been losing — so the ensemble must
be near-certain to keep shorting a market that keeps not falling. It is
**self-relaxing**: the moment a side's realized win rate recovers, its bar
drops back to the base. Nothing about a direction is hardcoded; the data
decides.

How to use
----------
    floor = required_confidence("SELL", symbol="BTCUSDT", base=0.60)
    if confidence < floor:
        signal = "HOLD"   # this direction hasn't earned the trade

Wired in main.py behind ``ELVIS_DIRECTIONAL_GUARD`` (default on). Tunables:
``ELVIS_DG_TARGET_WR`` (win rate at/above which the base bar applies,
default 0.35), ``ELVIS_DG_LOOKBACK`` (closes examined, default 50),
``ELVIS_DG_MIN_TRADES`` (below this the guard abstains, default 15),
``ELVIS_DG_MAX_FLOOR`` (hardest bar a losing side faces, default 0.95).
"""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def recent_side_winrate(
    side: str, symbol: Optional[str] = None, lookback: int = 50
) -> Tuple[Optional[float], int]:
    """Realized win rate over the most recent ``lookback`` closes of ``side``.

    Returns ``(win_rate, n)``; ``win_rate`` is None when there are no closed
    trades. Only rows with realized PnL (``pnl <> 0``) count as closes.
    """
    from utils.paper_trade_db import get_conn

    conn = get_conn()
    if conn is None:
        return None, 0
    try:
        c = conn.cursor()
        params = [side.upper()]
        sym_clause = ""
        if symbol:
            sym_clause = " AND symbol = %s"
            params.append(symbol)
        params.append(int(lookback))
        c.execute(
            f"""
            SELECT COUNT(*) FILTER (WHERE pnl > 0), COUNT(*)
            FROM (
                SELECT pnl FROM np.trades
                WHERE side = %s AND pnl <> 0{sym_clause}
                ORDER BY timestamp DESC
                LIMIT %s
            ) recent
            """,
            tuple(params),
        )
        wins, n = c.fetchone()
        if not n:
            return None, 0
        return wins / n, n
    except Exception as exc:
        logger.warning(f"directional guard: winrate query failed: {exc}")
        return None, 0
    finally:
        conn.close()


def required_confidence(
    side: str,
    symbol: Optional[str] = None,
    base: float = 0.60,
    winrate: Optional[float] = None,
    n: Optional[int] = None,
) -> float:
    """Confidence a ``side`` must clear, raised when it has been losing.

    Base bar when the side's recent win rate is at/above ``ELVIS_DG_TARGET_WR``
    or there isn't enough history (< ``ELVIS_DG_MIN_TRADES``). Below target the
    bar scales linearly toward ``ELVIS_DG_MAX_FLOOR`` as the win rate falls to
    zero, so a 3%-win-rate side faces a near-veto. Pass ``winrate``/``n`` to
    skip the DB read (used by callers that already have the stats/tests).
    """
    target = float(os.getenv("ELVIS_DG_TARGET_WR", "0.35"))
    min_trades = int(os.getenv("ELVIS_DG_MIN_TRADES", "15"))
    max_floor = float(os.getenv("ELVIS_DG_MAX_FLOOR", "0.95"))
    lookback = int(os.getenv("ELVIS_DG_LOOKBACK", "50"))

    if winrate is None:
        winrate, n = recent_side_winrate(side, symbol, lookback)
    if winrate is None or (n or 0) < min_trades or target <= 0:
        return base  # not enough evidence to judge this direction
    if winrate >= target:
        return base
    # linear ramp: base at target win rate, max_floor at zero win rate
    severity = 1.0 - (winrate / target)  # 0..1
    floor = base + (max_floor - base) * severity
    return min(max_floor, max(base, floor))
