"""Order-book imbalance signals (profitability roadmap item #12).

How it works
------------
The order book snapshot returned by the executor (Binance ``get_order_book``
shape) contains price levels as ``{'bids': [[price, qty], ...],
'asks': [[price, qty], ...]}``.  Summing the quantities on each side over the
top ``depth`` levels gives the resting liquidity near the touch.  The
normalized imbalance

    imbalance = (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity)

lies in ``[-1, 1]``: positive values mean buyers dominate the near book
(bullish pressure), negative values mean sellers dominate (bearish pressure),
and values near zero mean the book is balanced (no information).

How to use it
-------------
>>> book = executor.get_order_book(symbol)  # or {'bids': [], 'asks': []} in paper mode
>>> imbalance = order_book_imbalance(book, depth=10)
>>> order_flow_signal(imbalance)
'BUY'
>>> confirmed, imb = confirm_signal_with_flow("BUY", book)
>>> if not confirmed:
...     pass  # skip the trade: the order book contradicts the strategy signal

All functions degrade gracefully: empty/missing/malformed books (paper mode
returns empty lists) yield a neutral ``0.0`` imbalance and never raise, so the
confirmation step becomes a no-op rather than blocking trading.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Signal labels used across the bot.
BUY = "BUY"
SELL = "SELL"
NEUTRAL = "NEUTRAL"

#: Default number of book levels per side used to measure liquidity.
DEFAULT_DEPTH = 10

#: Default minimum |imbalance| required for a directional flow signal.
DEFAULT_THRESHOLD = 0.15


def _side_liquidity(levels: Any, depth: int) -> float:
    """Sum the quantities of the top ``depth`` levels of one book side.

    Malformed levels (wrong shape, non-numeric, negative or non-finite
    quantities) are skipped instead of raising, so a partially corrupt
    snapshot still produces a usable number.
    """
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
        return 0.0

    total = 0.0
    for level in list(levels)[:depth]:
        try:
            qty = float(level[1])
        except TypeError, ValueError, IndexError, KeyError:
            logger.debug("Skipping malformed order-book level: %r", level)
            continue
        if not math.isfinite(qty) or qty < 0.0:
            logger.debug("Skipping non-finite/negative quantity: %r", level)
            continue
        total += qty
    return total


def order_book_imbalance(
    order_book: Mapping[str, Any] | None, depth: int = DEFAULT_DEPTH
) -> float:
    """Compute the normalized bid/ask liquidity imbalance of an order book.

    Args:
        order_book: Snapshot shaped like ``{'bids': [[price, qty], ...],
            'asks': [[price, qty], ...]}``.  Prices/quantities may be strings
            (Binance REST returns strings); they are coerced to ``float``.
        depth: Number of levels per side to include (top of book first).

    Returns:
        ``(bid_liq - ask_liq) / (bid_liq + ask_liq)`` clamped to ``[-1, 1]``.
        Returns ``0.0`` (neutral) when the book is empty, missing, malformed,
        or has no measurable liquidity — e.g. paper mode returning
        ``{'bids': [], 'asks': []}``.
    """
    if not isinstance(order_book, Mapping):
        if order_book is not None:
            logger.debug("order_book is not a mapping: %r", type(order_book))
        return 0.0
    if depth <= 0:
        logger.debug("Non-positive depth=%s; returning neutral imbalance", depth)
        return 0.0

    bid_liq = _side_liquidity(order_book.get("bids"), depth)
    ask_liq = _side_liquidity(order_book.get("asks"), depth)
    total = bid_liq + ask_liq
    if total <= 0.0 or not math.isfinite(total):
        return 0.0

    imbalance = (bid_liq - ask_liq) / total
    # Mathematically already in [-1, 1] for non-negative liquidity, but clamp
    # defensively against float artifacts.
    return max(-1.0, min(1.0, imbalance))


def order_flow_signal(imbalance: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Map an imbalance value to a directional flow signal.

    Args:
        imbalance: Value from :func:`order_book_imbalance`, ideally in
            ``[-1, 1]``.  ``NaN`` is treated as no information.
        threshold: Minimum magnitude (exclusive) required for a directional
            call; values in ``[-threshold, +threshold]`` are ``'NEUTRAL'``.

    Returns:
        ``'BUY'`` when ``imbalance > threshold``, ``'SELL'`` when
        ``imbalance < -threshold``, otherwise ``'NEUTRAL'``.
    """
    try:
        value = float(imbalance)
    except TypeError, ValueError:
        logger.debug("Non-numeric imbalance %r; returning NEUTRAL", imbalance)
        return NEUTRAL
    if math.isnan(value):
        return NEUTRAL
    if value > threshold:
        return BUY
    if value < -threshold:
        return SELL
    return NEUTRAL


def confirm_signal_with_flow(
    signal: str,
    order_book: Mapping[str, Any] | None,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, float]:
    """Check whether order flow confirms (or at least does not contradict) a
    strategy signal.

    The confirmation is intentionally permissive: a ``'NEUTRAL'`` flow signal
    carries no information, so it never blocks a trade.  Only an outright
    contradiction (e.g. strategy says ``'BUY'`` while the book is ask-heavy
    beyond ``threshold``) vetoes the signal.  Non-directional strategy
    signals (anything other than ``'BUY'``/``'SELL'``, e.g. ``'HOLD'``)
    cannot be contradicted and are always confirmed.

    Args:
        signal: Strategy signal, typically ``'BUY'`` or ``'SELL'``
            (case-insensitive).
        order_book: Order book snapshot; see :func:`order_book_imbalance`.
        threshold: Imbalance magnitude needed for the flow to take a side.

    Returns:
        ``(confirmed, imbalance)`` where ``confirmed`` is ``True`` when the
        flow agrees with ``signal`` or is neutral/uninformative, and
        ``imbalance`` is the computed order-book imbalance (``0.0`` for
        empty/paper-mode books, which therefore never block).
    """
    imbalance = order_book_imbalance(order_book)
    flow = order_flow_signal(imbalance, threshold)

    normalized = signal.strip().upper() if isinstance(signal, str) else ""
    if normalized not in (BUY, SELL):
        return True, imbalance

    confirmed = flow == NEUTRAL or flow == normalized
    if not confirmed:
        logger.info(
            "Order flow contradicts signal %s (flow=%s, imbalance=%.4f)",
            normalized,
            flow,
            imbalance,
        )
    return confirmed, imbalance
