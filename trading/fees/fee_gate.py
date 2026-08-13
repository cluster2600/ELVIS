"""Fee-aware trade viability gate (roadmap item #5).

How it works
------------
Every futures round-trip pays three costs on the contract notional
(``fill_price * quantity``):

1. an entry taker fee  (``notional * taker_fee``),
2. an exit taker fee   (``notional * taker_fee``),
3. funding             (``entry_notional * funding_rate * funding_periods``).

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
>>> costs = all_in_cost(
...     entry_price=50_000.0, quantity=0.01, expected_exit_price=50_100.0
... )
>>> tuple(round(costs[name], 4) for name in costs)
(0.2, 0.2004, 0.05, 0.4504)
>>> viable, net, costs = is_trade_viable(
...     entry_price=50_000.0, expected_exit_price=50_100.0, quantity=0.01
... )
>>> viable, round(net, 8)
(True, 0.5496)

Both functions are defensive: any non-finite (NaN/inf) or non-positive
price/quantity input, invalid cost parameter, or arithmetic overflow
yields a neutral result (zero costs, trade flagged as not viable) instead
of raising, so callers in the live loop never crash on bad ticker data.

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


def _as_finite_float(value: object) -> float | None:
    """Return a finite float while rejecting booleans and conversion overflow."""
    value_type = type(value)
    if isinstance(value, bool) or (
        value_type.__module__ == "numpy" and value_type.__name__ in {"bool", "bool_"}
    ):
        return None
    try:
        converted = float(value)
    except TypeError, ValueError, OverflowError:
        return None
    return converted if math.isfinite(converted) else None


def _as_positive_float(value: object) -> float | None:
    """Return a finite, strictly positive float or ``None``."""
    converted = _as_finite_float(value)
    if converted is None or converted <= 0.0:
        return None
    return converted


def _cost_parameters(
    taker_fee: object,
    funding_rate: object,
    funding_periods: object,
) -> tuple[float, float, float] | None:
    """Return conservative, non-negative cost parameters or ``None``."""
    taker_fee_f = _as_finite_float(taker_fee)
    funding_rate_f = _as_finite_float(funding_rate)
    funding_periods_f = _as_finite_float(funding_periods)
    if (
        taker_fee_f is None
        or funding_rate_f is None
        or funding_periods_f is None
        or taker_fee_f < 0.0
        or funding_rate_f < 0.0
        or funding_periods_f < 0.0
    ):
        return None
    return taker_fee_f, funding_rate_f, funding_periods_f


def _calculate_costs(
    entry_price: object,
    expected_exit_price: object,
    quantity: object,
    *,
    taker_fee: object,
    funding_rate: object,
    funding_periods: object,
) -> dict[str, float] | None:
    """Calculate finite costs, keeping invalid and zero-cost cases distinct."""
    entry = _as_positive_float(entry_price)
    exit_price = _as_positive_float(expected_exit_price)
    contract_quantity = _as_positive_float(quantity)
    parameters = _cost_parameters(taker_fee, funding_rate, funding_periods)
    if (
        entry is None
        or exit_price is None
        or contract_quantity is None
        or parameters is None
    ):
        return None

    taker_fee_f, funding_rate_f, funding_periods_f = parameters
    entry_notional = entry * contract_quantity
    exit_notional = exit_price * contract_quantity
    entry_fee = entry_notional * taker_fee_f
    exit_fee = exit_notional * taker_fee_f
    funding_fee = entry_notional * funding_rate_f * funding_periods_f
    total = entry_fee + exit_fee + funding_fee
    calculated = (
        entry_notional,
        exit_notional,
        entry_fee,
        exit_fee,
        funding_fee,
        total,
    )
    if not all(math.isfinite(value) for value in calculated):
        return None

    return {
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "funding_fee": funding_fee,
        "total": total,
    }


def all_in_cost(
    entry_price: float,
    quantity: float,
    *,
    expected_exit_price: float,
    taker_fee: float = DEFAULT_TAKER_FEE,
    funding_rate: float = DEFAULT_FUNDING_RATE,
    funding_periods: float = 1,
) -> dict[str, float]:
    """Itemise the all-in cost of a leveraged futures round-trip.

    Entry and exit are both assumed to be taker orders (worst case), using
    their respective fill notionals. Funding accrues on entry notional.

    Args:
        entry_price: Expected entry price (quote currency, e.g. USDT).
        expected_exit_price: Expected exit price in the same quote currency.
        quantity: Position size in base asset (e.g. BTC).
        taker_fee: Taker fee rate per fill (default 0.0004 = 0.04%,
            the Binance futures VIP 0 rate from
            ``trading/fees/binance_fee_calculator.py``).
        funding_rate: Funding rate per funding period (default 0.0001).
        funding_periods: Number of 8-hour funding periods the position
            is expected to be held (may be fractional).

    Returns:
        Dict with ``'entry_fee'``, ``'exit_fee'``, ``'funding_fee'`` and
        ``'total'`` in quote currency. All zeros if any input is
        non-finite or if either price or quantity is not strictly
        positive (neutral no-op — never raises on bad data).
    """
    costs = _calculate_costs(
        entry_price,
        expected_exit_price,
        quantity,
        taker_fee=taker_fee,
        funding_rate=funding_rate,
        funding_periods=funding_periods,
    )
    if costs is None:
        logger.warning(
            "all_in_cost: invalid or non-finite inputs "
            "(entry_price_type=%s, expected_exit_price_type=%s, "
            "quantity_type=%s, taker_fee_type=%s, funding_rate_type=%s, "
            "funding_periods_type=%s); "
            "returning zero costs",
            type(entry_price).__name__,
            type(expected_exit_price).__name__,
            type(quantity).__name__,
            type(taker_fee).__name__,
            type(funding_rate).__name__,
            type(funding_periods).__name__,
        )
        return _zero_costs()

    logger.debug(
        "all_in_cost: entry_fee=%.8f exit_fee=%.8f funding_fee=%.8f total=%.8f",
        costs["entry_fee"],
        costs["exit_fee"],
        costs["funding_fee"],
        costs["total"],
    )
    return costs


def is_trade_viable(
    entry_price: float,
    expected_exit_price: float,
    quantity: float,
    side: str = "BUY",
    *,
    min_net_profit: float = 0.0,
    taker_fee: float = DEFAULT_TAKER_FEE,
    funding_rate: float = DEFAULT_FUNDING_RATE,
    funding_periods: float = 1,
) -> tuple[bool, float, dict[str, float]]:
    """Decide whether an expected move survives fees.

    Gross PnL is ``(exit - entry) * quantity`` for BUY/LONG
    and ``(entry - exit) * quantity`` for SELL/SHORT. The net
    PnL subtracts the :func:`all_in_cost` total, and the trade is viable
    only if ``net > min_net_profit`` (strict inequality: a trade that
    exactly hits the threshold is rejected).

    Args:
        entry_price: Expected entry price.
        expected_exit_price: Target/expected exit price.
        quantity: Position size in base asset.
        side: ``'BUY'``/``'LONG'`` or ``'SELL'``/``'SHORT'``
            (case-insensitive).
        min_net_profit: Minimum net profit (quote currency) required for
            the trade to be worth taking.
        taker_fee: Taker fee rate applied to each fill notional.
        funding_rate: Funding rate applied to entry notional per period.
        funding_periods: Expected number of funding periods held.

    Returns:
        ``(viable, net_profit, cost_breakdown)``. On any invalid input
        (NaN/inf prices, non-positive quantity, unknown side) returns
        ``(False, 0.0, zero_costs)`` instead of raising.

    Example:
        >>> viable, net, costs = is_trade_viable(
        ...     50_000.0, 50_100.0, 0.01, side="BUY"
        ... )
        >>> viable
        True
    """
    if not isinstance(side, str):
        logger.warning(
            "is_trade_viable: unknown side type %s; treating trade as not viable",
            type(side).__name__,
        )
        return False, 0.0, _zero_costs()

    side_norm = side.upper()
    if side_norm not in _BUY_SIDES | _SELL_SIDES:
        logger.warning(
            "is_trade_viable: unknown side string; treating trade as not viable"
        )
        return False, 0.0, _zero_costs()

    entry = _as_positive_float(entry_price)
    exit_price = _as_positive_float(expected_exit_price)
    contract_quantity = _as_positive_float(quantity)
    min_net_profit_f = _as_finite_float(min_net_profit)
    costs = _calculate_costs(
        entry_price,
        expected_exit_price,
        quantity,
        taker_fee=taker_fee,
        funding_rate=funding_rate,
        funding_periods=funding_periods,
    )
    if (
        entry is None
        or exit_price is None
        or contract_quantity is None
        or min_net_profit_f is None
        or min_net_profit_f < 0.0
        or costs is None
    ):
        logger.warning(
            "is_trade_viable: invalid or non-finite trade/cost inputs "
            "(entry_price_type=%s, expected_exit_price_type=%s, "
            "quantity_type=%s, min_net_profit_type=%s); "
            "treating trade as not viable",
            type(entry_price).__name__,
            type(expected_exit_price).__name__,
            type(quantity).__name__,
            type(min_net_profit).__name__,
        )
        return False, 0.0, _zero_costs()

    direction = 1.0 if side_norm in _BUY_SIDES else -1.0
    gross = (exit_price - entry) * contract_quantity * direction
    net = gross - costs["total"]
    if not (math.isfinite(gross) and math.isfinite(net)):
        logger.warning(
            "is_trade_viable: calculated gross/net is non-finite; "
            "treating trade as not viable"
        )
        return False, 0.0, _zero_costs()
    viable = net > min_net_profit_f

    logger.debug(
        "is_trade_viable: side=%s gross=%.8f fees=%.8f net=%.8f "
        "min_net_profit=%.8f -> viable=%s",
        side_norm,
        gross,
        costs["total"],
        net,
        min_net_profit_f,
        viable,
    )
    return viable, net, costs
