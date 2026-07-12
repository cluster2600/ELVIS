"""Fee-aware trade viability gate (roadmap item #5).

How it works
------------
Every futures round-trip pays three costs on the *leveraged notional*
(``entry_price * quantity * leverage``):

1. an entry taker fee  (``notional * taker_fee``),
2. an exit taker fee   (``notional * taker_fee``),
3. funding             (``notional * funding_rate * funding_periods``).

:func:`all_in_cost` itemises those costs, and :func:`is_trade_viable`
compares them against the gross PnL of an expected move so the bot can
skip trades whose edge is smaller than the fees they would incur.

The default fee constants here deliberately mirror the authoritative
Binance fee schedule kept in
``trading/fees/binance_fee_calculator.py`` (``BinanceFeeCalculator``):
0.04% (0.0004) futures taker fee for a VIP 0 account, and a typical
0.01% (0.0001) funding rate per 8-hour period. If the schedule changes,
update ``BinanceFeeCalculator`` first and keep these defaults in sync.

How to use
----------
>>> from trading.fees.fee_gate import all_in_cost, is_trade_viable
>>> all_in_cost(entry_price=50_000.0, quantity=0.01)
{'entry_fee': 0.2, 'exit_fee': 0.2, 'funding_fee': 0.05, 'total': 0.45}
>>> viable, net, costs = is_trade_viable(
...     entry_price=50_000.0, expected_exit_price=50_100.0, quantity=0.01
... )
>>> viable, round(net, 8)
(True, 0.55)

Both functions are defensive: any non-finite (NaN/inf) or non-positive
price/quantity input yields a neutral result (zero costs, trade flagged
as not viable) instead of raising, so callers in the live loop never
crash on bad ticker data.

Only the standard library is required; numpy scalars are accepted
transparently because they coerce cleanly through ``float()``.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

#: Binance USD-M futures taker fee (VIP 0). Mirrors
#: ``BinanceFeeCalculator.futures_taker_fee`` in binance_fee_calculator.py.
DEFAULT_TAKER_FEE: float = 0.0004

#: Typical funding rate per 8-hour funding period. Mirrors the default used
#: by ``BinanceFeeCalculator.calculate_funding_fee``.
DEFAULT_FUNDING_RATE: float = 0.0001

_BUY_SIDES = frozenset({"BUY", "LONG"})
_SELL_SIDES = frozenset({"SELL", "SHORT"})


def _zero_costs() -> dict[str, float]:
    """Return an all-zero cost breakdown (neutral no-op result)."""
    return {"entry_fee": 0.0, "exit_fee": 0.0, "funding_fee": 0.0, "total": 0.0}


def _is_valid_positive(value: float) -> bool:
    """True if ``value`` is a finite number strictly greater than zero."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def all_in_cost(
    entry_price: float,
    quantity: float,
    leverage: float = 1.0,
    taker_fee: float = DEFAULT_TAKER_FEE,
    funding_rate: float = DEFAULT_FUNDING_RATE,
    funding_periods: float = 1,
) -> dict[str, float]:
    """Itemise the all-in cost of a leveraged futures round-trip.

    The position notional is ``entry_price * quantity * leverage``.
    Entry and exit are both assumed to be taker orders (worst case), and
    funding accrues linearly per funding period held.

    Args:
        entry_price: Expected entry price (quote currency, e.g. USDT).
        quantity: Position size in base asset (e.g. BTC).
        leverage: Leverage multiplier applied to the notional.
        taker_fee: Taker fee rate per fill (default 0.0004 = 0.04%,
            the Binance futures VIP 0 rate from
            ``trading/fees/binance_fee_calculator.py``).
        funding_rate: Funding rate per funding period (default 0.0001).
        funding_periods: Number of 8-hour funding periods the position
            is expected to be held (may be fractional).

    Returns:
        Dict with ``'entry_fee'``, ``'exit_fee'``, ``'funding_fee'`` and
        ``'total'`` in quote currency. All zeros if any input is
        non-finite or if price/quantity/leverage is not strictly
        positive (neutral no-op — never raises on bad data).
    """
    if not (
        _is_valid_positive(entry_price)
        and _is_valid_positive(quantity)
        and _is_valid_positive(leverage)
    ):
        logger.warning(
            "all_in_cost: invalid inputs (entry_price=%r, quantity=%r, "
            "leverage=%r); returning zero costs",
            entry_price,
            quantity,
            leverage,
        )
        return _zero_costs()

    rates = (float(taker_fee), float(funding_rate), float(funding_periods))
    if not all(math.isfinite(rate) for rate in rates):
        logger.warning(
            "all_in_cost: non-finite fee parameters (taker_fee=%r, "
            "funding_rate=%r, funding_periods=%r); returning zero costs",
            taker_fee,
            funding_rate,
            funding_periods,
        )
        return _zero_costs()
    taker_fee_f, funding_rate_f, funding_periods_f = rates

    notional = float(entry_price) * float(quantity) * float(leverage)
    entry_fee = notional * taker_fee_f
    exit_fee = notional * taker_fee_f
    funding_fee = notional * funding_rate_f * funding_periods_f
    total = entry_fee + exit_fee + funding_fee

    logger.debug(
        "all_in_cost: notional=%.8f entry_fee=%.8f exit_fee=%.8f "
        "funding_fee=%.8f total=%.8f",
        notional,
        entry_fee,
        exit_fee,
        funding_fee,
        total,
    )
    return {
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "funding_fee": funding_fee,
        "total": total,
    }


def is_trade_viable(
    entry_price: float,
    expected_exit_price: float,
    quantity: float,
    side: str = "BUY",
    leverage: float = 1.0,
    min_net_profit: float = 0.0,
    **fee_kw: float,
) -> tuple[bool, float, dict[str, float]]:
    """Decide whether an expected move survives fees.

    Gross PnL is ``(exit - entry) * quantity * leverage`` for BUY/LONG
    and ``(entry - exit) * quantity * leverage`` for SELL/SHORT. The net
    PnL subtracts the :func:`all_in_cost` total, and the trade is viable
    only if ``net > min_net_profit`` (strict inequality: a trade that
    exactly hits the threshold is rejected).

    Args:
        entry_price: Expected entry price.
        expected_exit_price: Target/expected exit price.
        quantity: Position size in base asset.
        side: ``'BUY'``/``'LONG'`` or ``'SELL'``/``'SHORT'``
            (case-insensitive).
        leverage: Leverage multiplier (scales both gross PnL and fees).
        min_net_profit: Minimum net profit (quote currency) required for
            the trade to be worth taking.
        **fee_kw: Passed through to :func:`all_in_cost`
            (``taker_fee``, ``funding_rate``, ``funding_periods``).

    Returns:
        ``(viable, net_profit, cost_breakdown)``. On any invalid input
        (NaN/inf prices, non-positive quantity, unknown side) returns
        ``(False, 0.0, zero_costs)`` instead of raising.

    Example:
        >>> viable, net, costs = is_trade_viable(
        ...     50_000.0, 50_100.0, 0.01, side="BUY", leverage=1.0
        ... )
        >>> viable
        True
    """
    side_norm = str(side).upper()
    if side_norm not in _BUY_SIDES | _SELL_SIDES:
        logger.warning(
            "is_trade_viable: unknown side %r; treating trade as not viable",
            side,
        )
        return False, 0.0, _zero_costs()

    exit_valid = _is_valid_positive(expected_exit_price)
    min_net_valid = math.isfinite(float(min_net_profit))
    if not (exit_valid and min_net_valid):
        logger.warning(
            "is_trade_viable: invalid inputs (expected_exit_price=%r, "
            "min_net_profit=%r); treating trade as not viable",
            expected_exit_price,
            min_net_profit,
        )
        return False, 0.0, _zero_costs()

    costs = all_in_cost(entry_price, quantity, leverage=leverage, **fee_kw)
    if costs["total"] == 0.0 and not (
        _is_valid_positive(entry_price)
        and _is_valid_positive(quantity)
        and _is_valid_positive(leverage)
    ):
        # all_in_cost already logged the reason; stay neutral.
        return False, 0.0, costs

    direction = 1.0 if side_norm in _BUY_SIDES else -1.0
    gross = (
        (float(expected_exit_price) - float(entry_price))
        * float(quantity)
        * float(leverage)
        * direction
    )
    net = gross - costs["total"]
    viable = net > float(min_net_profit)

    logger.debug(
        "is_trade_viable: side=%s gross=%.8f fees=%.8f net=%.8f "
        "min_net_profit=%.8f -> viable=%s",
        side_norm,
        gross,
        costs["total"],
        net,
        min_net_profit,
        viable,
    )
    return viable, net, costs
