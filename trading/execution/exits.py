"""
Exit management utilities: trailing stops and regime-aware take-profit levels.

This module implements two independent exit mechanisms used by the
profitability roadmap (items #4 and #10):

1. ``TrailingStop`` -- a stateful, per-position trailing stop. For a BUY
   (long) position it tracks the highest price seen since entry (the
   "high-water mark") and places the stop a fixed percentage below it, so
   the stop only ever ratchets upward and locks in profit as the price
   rises. For a SELL (short) position the logic is mirrored: it tracks the
   lowest price seen (the "low-water mark") and places the stop a fixed
   percentage above it.

2. ``dynamic_take_profit`` -- a regime-keyed take-profit price. Trending
   markets justify wide targets, while ranging/choppy markets call for
   quick scalps. The take-profit distance is expressed as a *percentage*
   of the entry price (not an absolute dollar offset, as the original
   roadmap sketch suggested) so the behaviour stays consistent whether BTC
   trades at $30k or $120k.

Both utilities are pure stdlib (plus ``math`` for NaN checks): no numpy,
pandas, torch, or network access is required, so they are safe to import
in constrained CI environments.

Usage
-----
Trailing stop::

    from trading.execution.exits import TrailingStop

    ts = TrailingStop(trail_pct=0.02)  # trail 2% behind the extreme
    should_close, stop_price = ts.update("pos-1", "BUY", 100.0, 105.0)
    if should_close:
        # close the position at market, then forget its state
        ts.clear("pos-1")

Dynamic take-profit::

    from trading.execution.exits import dynamic_take_profit

    tp_price = dynamic_take_profit("TRENDING", entry_price=50_000.0, side="BUY")
    # -> 52_500.0  (entry * 1.05 in a trending regime)

Both are defensive: invalid, non-finite, or non-positive prices produce a
neutral "do nothing" result (no close signal / entry price returned)
rather than raising, so a bad tick can never crash the trading loop.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Default take-profit distance (as a fraction of entry price) per market
# regime. Unknown regimes fall back to the conservative RANGING value.
DEFAULT_REGIME_TP_PCT: Dict[str, float] = {
    "TRENDING": 0.05,  # ride trends: 5% target
    "REVERSAL": 0.01,  # reversals resolve quickly: 1% target
    "RANGING": 0.0025,  # scalp the range: 0.25% target
    "CHOPPY": 0.001,  # get out fast: 0.1% target
}

_FALLBACK_REGIME = "RANGING"


def _is_valid_price(value: float) -> bool:
    """Return True if ``value`` is a finite, strictly positive number."""
    try:
        return math.isfinite(value) and value > 0
    except TypeError:
        return False


class TrailingStop:
    """Stateful per-position trailing stop manager.

    How it works
    ------------
    Each position (keyed by ``position_id``) gets its own tracked
    "extreme" price:

    * BUY (long): the extreme is the *highest* price observed since the
      position was first seen (seeded with ``max(entry_price,
      current_price)`` on the first update). The stop sits at
      ``extreme * (1 - trail_pct)``. Because the extreme can only rise,
      the stop can only rise -- it never gives back locked-in profit.
    * SELL (short): mirrored. The extreme is the *lowest* price observed
      (seeded with ``min(entry_price, current_price)``) and the stop sits
      at ``extreme * (1 + trail_pct)``, only ever moving down.

    ``update`` returns ``(should_close, stop_price)`` where
    ``should_close`` is True once the current price crosses the stop
    (``current <= stop`` for BUY, ``current >= stop`` for SELL).

    How to use it
    -------------
    Create one instance per strategy (or per bot) and call
    :meth:`update` on every tick/candle for each open position. When a
    position closes -- whether because the stop fired or for any other
    reason -- call :meth:`clear` so the stale extreme is not reused if
    the same id is recycled.

    Args:
        trail_pct: Trailing distance as a fraction of the extreme price
            (e.g. ``0.02`` trails 2% behind the high/low-water mark).
            Non-finite or negative values are coerced to the 0.02
            default with a warning.
    """

    DEFAULT_TRAIL_PCT = 0.02

    def __init__(self, trail_pct: float = DEFAULT_TRAIL_PCT) -> None:
        if (
            not isinstance(trail_pct, (int, float))
            or not math.isfinite(trail_pct)
            or trail_pct < 0
        ):
            logger.warning(
                "Invalid trail_pct %r; falling back to default %.4f",
                trail_pct,
                self.DEFAULT_TRAIL_PCT,
            )
            trail_pct = self.DEFAULT_TRAIL_PCT
        self.trail_pct: float = float(trail_pct)
        # position_id -> extreme price (high-water mark for BUY,
        # low-water mark for SELL).
        self._extremes: Dict[str, float] = {}
        # position_id -> normalized side, so a side flip on an existing
        # id can be detected and handled by re-seeding.
        self._sides: Dict[str, str] = {}

    def update(
        self,
        position_id: str,
        side: str,
        entry_price: float,
        current_price: float,
    ) -> Tuple[bool, float]:
        """Advance the trailing stop for one position and check it.

        Args:
            position_id: Stable identifier of the open position.
            side: ``"BUY"`` for long positions, ``"SELL"`` for short.
                Case-insensitive; unknown sides are treated as BUY with
                a warning.
            entry_price: The position's entry price. Only used to seed
                the extreme on the first update for this id — and again
                whenever a recycled ``position_id`` flips side (the
                extreme re-seeds from this call's ``entry_price``), so on
                a side flip pass the NEW position's entry price, not the
                old side's.
            current_price: Latest observed market price.

        Returns:
            ``(should_close, stop_price)``. ``should_close`` is True when
            ``current_price`` has crossed the stop. ``stop_price`` is the
            current stop level. On invalid input the method is a no-op
            returning ``(False, 0.0)``.
        """
        normalized_side = str(side).upper()
        if normalized_side not in ("BUY", "SELL"):
            logger.warning(
                "Unknown side %r for position %s; treating as BUY", side, position_id
            )
            normalized_side = "BUY"

        if not _is_valid_price(current_price):
            logger.warning(
                "Invalid current_price %r for position %s; skipping update",
                current_price,
                position_id,
            )
            return (False, 0.0)

        if not _is_valid_price(entry_price):
            logger.warning(
                "Invalid entry_price %r for position %s; using current_price as seed",
                entry_price,
                position_id,
            )
            entry_price = current_price

        key = str(position_id)

        # Re-seed if this is a new position id or the side changed
        # (a recycled id whose old state would be meaningless).
        if key not in self._extremes or self._sides.get(key) != normalized_side:
            if self._sides.get(key) not in (None, normalized_side):
                logger.info(
                    "Side changed for position %s (%s -> %s); re-seeding extreme",
                    key,
                    self._sides.get(key),
                    normalized_side,
                )
            if normalized_side == "BUY":
                self._extremes[key] = max(float(entry_price), float(current_price))
            else:
                self._extremes[key] = min(float(entry_price), float(current_price))
            self._sides[key] = normalized_side

        if normalized_side == "BUY":
            self._extremes[key] = max(self._extremes[key], float(current_price))
            stop_price = self._extremes[key] * (1.0 - self.trail_pct)
            should_close = float(current_price) <= stop_price
        else:
            self._extremes[key] = min(self._extremes[key], float(current_price))
            stop_price = self._extremes[key] * (1.0 + self.trail_pct)
            should_close = float(current_price) >= stop_price

        if should_close:
            logger.info(
                "Trailing stop hit for %s position %s: price %.8f crossed stop %.8f",
                normalized_side,
                key,
                float(current_price),
                stop_price,
            )
        return (should_close, stop_price)

    def clear(self, position_id: str) -> None:
        """Forget the tracked state for a closed position.

        Safe to call with an unknown or already-cleared id (no-op).

        Args:
            position_id: Identifier previously passed to :meth:`update`.
        """
        key = str(position_id)
        self._extremes.pop(key, None)
        self._sides.pop(key, None)


def dynamic_take_profit(
    regime: str,
    entry_price: float,
    side: str = "BUY",
    regime_pct: Optional[Dict[str, float]] = None,
) -> float:
    """Compute a regime-aware take-profit *price* for a position.

    How it works
    ------------
    Each market regime maps to a take-profit distance expressed as a
    fraction of the entry price (see :data:`DEFAULT_REGIME_TP_PCT`):

    ========== ======= =========================================
    Regime     Pct     Rationale
    ========== ======= =========================================
    TRENDING   0.05    let winners run in a directional market
    REVERSAL   0.01    capture the snap-back, then get out
    RANGING    0.0025  scalp inside the range
    CHOPPY     0.001   minimal target, minimal exposure
    unknown    0.0025  conservative RANGING fallback
    ========== ======= =========================================

    The take-profit is then ``entry * (1 + pct)`` for BUY and
    ``entry * (1 - pct)`` for SELL.

    Note: the original roadmap sketched these targets as absolute dollar
    offsets. Percentages are used here instead because a fixed dollar
    offset means completely different risk/reward depending on the asset
    price level; a percentage keeps the target economically consistent
    across price regimes and across assets.

    How to use it
    -------------
    Call once when a position is opened (or when the detected regime
    changes) and register the returned price as the take-profit level::

        tp = dynamic_take_profit("CHOPPY", entry_price=100.0, side="SELL")
        # -> 99.9  (entry * (1 - 0.001))

    Args:
        regime: Market regime label (e.g. ``"TRENDING"``). Case-
            insensitive; unknown labels use the RANGING fallback.
        entry_price: Position entry price. Non-finite or non-positive
            values are returned unchanged (neutral no-op) with a warning.
        side: ``"BUY"`` (default) or ``"SELL"``. Case-insensitive;
            unknown sides are treated as BUY with a warning.
        regime_pct: Optional override map of ``regime -> pct`` that is
            merged over the defaults (defaults remain for keys not
            supplied). Invalid override values are ignored.

    Returns:
        The take-profit price as a float.
    """
    if not _is_valid_price(entry_price):
        logger.warning(
            "Invalid entry_price %r for dynamic_take_profit; returning it unchanged",
            entry_price,
        )
        try:
            return float(entry_price)
        except (TypeError, ValueError):
            return 0.0

    pct_map = dict(DEFAULT_REGIME_TP_PCT)
    if regime_pct:
        for key, value in regime_pct.items():
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                pct_map[str(key).upper()] = float(value)
            else:
                logger.warning(
                    "Ignoring invalid take-profit pct %r for regime %r", value, key
                )

    normalized_regime = str(regime).upper() if regime is not None else ""
    pct = pct_map.get(normalized_regime)
    if pct is None:
        logger.debug(
            "Unknown regime %r; using %s fallback take-profit pct",
            regime,
            _FALLBACK_REGIME,
        )
        pct = pct_map.get(_FALLBACK_REGIME, DEFAULT_REGIME_TP_PCT[_FALLBACK_REGIME])

    normalized_side = str(side).upper()
    if normalized_side not in ("BUY", "SELL"):
        logger.warning("Unknown side %r for dynamic_take_profit; treating as BUY", side)
        normalized_side = "BUY"

    if normalized_side == "BUY":
        return float(entry_price) * (1.0 + pct)
    return float(entry_price) * (1.0 - pct)
