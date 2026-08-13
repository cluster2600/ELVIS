"""
Position sizing helpers: volume-based size scaling and Kelly-criterion capital
allocation (profitability roadmap items #3 and #13).

How it works
------------
``volume_multiplier`` measures how active the market currently is by dividing
the latest volume observation by its rolling mean over ``window`` bars.  A
value above 1.0 means volume is running hotter than average (conviction behind
the move), below 1.0 means the market is quiet.  The ratio is clamped to
``[floor, cap]`` so a single anomalous print can never blow up position size.

``kelly_fraction`` computes the classic Kelly criterion

    f* = (b * p - q) / b

where ``p`` is the win rate, ``q = 1 - p`` and ``b`` is the payoff ratio
``|avg_win / avg_loss|``.  The raw Kelly output is aggressive, so the result
is clamped to a conservative ``[floor, cap]`` band (fractional Kelly by
construction).  ``kelly_from_trades`` derives ``p``, ``avg_win`` and
``avg_loss`` from a realized trade history and delegates to
``kelly_fraction``; with fewer than ``min_trades`` samples there is not
enough statistical evidence, so it stays at ``floor``.

How to use it
-------------
::

    from trading.risk.position_sizing import (
        kelly_fraction,
        kelly_from_trades,
        volume_multiplier,
    )

    # Scale a base position size by current volume activity.
    size = base_size * volume_multiplier(ohlcv_df)  # ohlcv_df has 'volume'

    # Fraction of equity to risk per trade, from summary stats ...
    f = kelly_fraction(win_rate=0.55, avg_win=42.0, avg_loss=-30.0)

    # ... or straight from a trade log of {'pnl': float} dicts.
    f = kelly_from_trades(closed_trades)
    risk_dollars = equity * f

All functions are defensive: empty/short/NaN inputs return a neutral or floor
value instead of raising, so callers never need try/except around sizing.
Only stdlib + numpy + pandas are required.
"""

import logging
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["volume_multiplier", "kelly_fraction", "kelly_from_trades"]


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]`` (``lo`` wins if ``lo > hi``)."""
    return max(lo, min(hi, value))


def volume_multiplier(
    data: Optional[pd.DataFrame],
    window: int = 20,
    floor: float = 0.5,
    cap: float = 2.0,
) -> float:
    """Position-size multiplier based on current vs. average volume.

    Computes ``current_volume / rolling_mean(volume, window)`` where the
    rolling mean covers the last ``window`` valid observations (including the
    current bar) and clamps the result to ``[floor, cap]``.

    Args:
        data: OHLCV DataFrame containing a ``volume`` column (matched
            case-insensitively, e.g. ``Volume`` also works).
        window: Number of bars in the rolling mean. Must be >= 1.
        floor: Lower clamp bound (quiet market -> shrink size).
        cap: Upper clamp bound (volume spike -> cap the boost).

    Returns:
        The clamped volume ratio, or ``1.0`` (neutral, no scaling) when the
        input is missing, has no usable volume column, or has fewer than
        ``window`` valid volume observations.
    """
    neutral = 1.0

    if not isinstance(data, pd.DataFrame) or data.empty:
        logger.debug("volume_multiplier: missing/empty data -> neutral 1.0")
        return neutral
    if window < 1:
        logger.debug("volume_multiplier: invalid window=%s -> neutral 1.0", window)
        return neutral

    volume_col = next(
        (col for col in data.columns if str(col).lower() == "volume"), None
    )
    if volume_col is None:
        logger.debug("volume_multiplier: no volume column -> neutral 1.0")
        return neutral

    volumes = pd.to_numeric(data[volume_col], errors="coerce")
    volumes = volumes.replace([np.inf, -np.inf], np.nan).dropna()
    if len(volumes) < window:
        logger.debug(
            "volume_multiplier: %d valid volume rows < window=%d -> neutral 1.0",
            len(volumes),
            window,
        )
        return neutral

    recent = volumes.iloc[-window:]
    mean_volume = float(recent.mean())
    current_volume = float(recent.iloc[-1])
    if mean_volume <= 0.0 or current_volume < 0.0:
        logger.debug(
            "volume_multiplier: degenerate volumes (current=%s, mean=%s) "
            "-> neutral 1.0",
            current_volume,
            mean_volume,
        )
        return neutral

    multiplier = _clamp(current_volume / mean_volume, floor, cap)
    logger.debug(
        "volume_multiplier: current=%.6g mean=%.6g -> %.4f",
        current_volume,
        mean_volume,
        multiplier,
    )
    return multiplier


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    floor: float = 0.01,
    cap: float = 0.05,
) -> float:
    """Clamped Kelly criterion fraction of capital to risk per trade.

    Uses ``f* = (b * p - q) / b`` with ``p = win_rate``, ``q = 1 - p`` and
    payoff ratio ``b = |avg_win / avg_loss|``, then clamps to
    ``[floor, cap]``.

    Args:
        win_rate: Historical probability of a winning trade, in ``(0, 1)``.
        avg_win: Average PnL of winning trades (positive).
        avg_loss: Average PnL of losing trades (typically negative; only its
            magnitude is used).
        floor: Minimum fraction returned (also the fallback for degenerate
            inputs).
        cap: Maximum fraction returned (fractional-Kelly safety cap).

    Returns:
        The clamped Kelly fraction. Degenerate inputs -- non-finite values,
        ``win_rate`` outside ``(0, 1)``, ``avg_loss == 0`` or a non-positive
        payoff ratio -- return ``floor``.
    """
    try:
        p = float(win_rate)
        win = float(avg_win)
        loss = float(avg_loss)
    except TypeError, ValueError:
        logger.debug("kelly_fraction: non-numeric inputs -> floor %.4f", floor)
        return floor

    if not (math.isfinite(p) and math.isfinite(win) and math.isfinite(loss)):
        logger.debug("kelly_fraction: non-finite inputs -> floor %.4f", floor)
        return floor
    if not 0.0 < p < 1.0:
        logger.debug("kelly_fraction: win_rate=%s not in (0, 1) -> floor", p)
        return floor
    if loss == 0.0:
        logger.debug("kelly_fraction: avg_loss == 0 -> floor %.4f", floor)
        return floor

    b = abs(win / loss)
    if b <= 0.0 or not math.isfinite(b):
        logger.debug("kelly_fraction: payoff ratio b=%s <= 0 -> floor", b)
        return floor

    q = 1.0 - p
    f_star = (b * p - q) / b
    fraction = _clamp(f_star, floor, cap)
    logger.debug(
        "kelly_fraction: p=%.4f b=%.4f f*=%.4f -> %.4f", p, b, f_star, fraction
    )
    return fraction


def kelly_from_trades(
    trades: Optional[Sequence[Mapping[str, Any]]],
    min_trades: int = 20,
    **kwargs: float,
) -> float:
    """Derive a Kelly fraction from a realized trade history.

    Extracts ``pnl`` from each trade dict, computes the win rate and the
    average winning/losing PnL, and delegates to :func:`kelly_fraction`.

    Args:
        trades: Sequence of trade records, each a mapping with a numeric
            ``'pnl'`` key (e.g. ``{'pnl': 12.5}``). Records with a missing or
            non-numeric ``pnl`` are ignored. Zero-PnL trades count toward the
            sample size but are neither wins nor losses.
        min_trades: Minimum number of valid trades required before the
            statistics are trusted.
        **kwargs: Forwarded to :func:`kelly_fraction` (``floor``, ``cap``).

    Returns:
        The clamped Kelly fraction, or ``floor`` when there are fewer than
        ``min_trades`` valid trades or the history has no wins or no losses
        (not enough evidence to size above the minimum).
    """
    floor = float(kwargs.get("floor", 0.01))

    pnls: list[float] = []
    for trade in trades or ():
        try:
            pnl = float(trade["pnl"])
        except TypeError, KeyError, ValueError:
            continue
        if math.isfinite(pnl):
            pnls.append(pnl)

    if len(pnls) < min_trades:
        logger.debug(
            "kelly_from_trades: %d valid trades < min_trades=%d -> floor %.4f",
            len(pnls),
            min_trades,
            floor,
        )
        return floor

    wins = [pnl for pnl in pnls if pnl > 0.0]
    losses = [pnl for pnl in pnls if pnl < 0.0]
    if not wins or not losses:
        logger.debug(
            "kelly_from_trades: one-sided history (%d wins, %d losses) "
            "-> floor %.4f",
            len(wins),
            len(losses),
            floor,
        )
        return floor

    win_rate = len(wins) / len(pnls)
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    logger.debug(
        "kelly_from_trades: n=%d win_rate=%.4f avg_win=%.4f avg_loss=%.4f",
        len(pnls),
        win_rate,
        avg_win,
        avg_loss,
    )
    return kelly_fraction(win_rate, avg_win, avg_loss, **kwargs)
