"""Signal-quality filters for the profitability roadmap (items #2, #6-#9).

This module provides pure, dependency-light (stdlib + numpy + pandas) gate
functions that sit between a raw model/strategy signal and order execution.
Each filter answers one question about signal quality:

- :func:`rsi_gate` -- item #2: veto BUYs into overbought / SELLs into
  oversold conditions.
- :func:`has_momentum` -- item #6: require the last N closes to already be
  moving in the trade direction before entering.
- :func:`detect_bb_squeeze` -- item #7: detect Bollinger-band volatility
  squeezes (compressed bands = chop risk, avoid entering).
- :func:`is_optimal_trading_hour` -- item #8: only trade during the
  historically liquid UTC hours (default 14:00-22:59, the US session).
- :func:`detect_macd_divergence` -- item #9: spot short-term price/MACD
  histogram divergences that either veto a trade or (optionally) generate
  one.

:func:`apply_signal_filters` composes all five with per-filter enable flags.

How to use::

    from trading.signals.filters import apply_signal_filters

    signal, confidence, reasons = apply_signal_filters(
        "BUY",
        0.82,
        candles_df,          # DataFrame with 'close' (+ 'macd_histogram')
        rsi=latest_rsi,      # optional; RSI gate is skipped when None
        config={"trading_hours": False},  # per-filter enable flags
    )
    if signal == "HOLD":
        logger.info("trade vetoed: %s", reasons)

Design notes:

- All functions degrade gracefully: short, empty, missing-column, or NaN
  data returns a neutral answer (``False`` / ``None`` / unchanged signal)
  instead of raising, so the bot keeps running in CI and in production
  when a data feed hiccups.
- The composite filter is conservative: it only ever *downgrades* an
  actionable signal to HOLD, except for the opt-in MACD divergence
  override (``config["macd_divergence_override"]``) which may promote an
  *incoming* HOLD to BUY/SELL at confidence 0.75. The override never
  resurrects a signal that another filter just blocked in the same call.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

#: Default enable flags consumed by :func:`apply_signal_filters`. Copy and
#: tweak (or pass a partial dict) to switch individual filters off.
DEFAULT_FILTER_CONFIG: dict[str, bool] = {
    "rsi_gate": True,
    "momentum": True,
    "bb_squeeze": True,
    "trading_hours": True,
    "macd_divergence": True,
    # Opt-in: allow a detected divergence to turn an incoming HOLD into a
    # BUY/SELL at confidence 0.75.
    "macd_divergence_override": False,
}


def _normalize_signal(signal: object) -> str:
    """Map arbitrary input to one of BUY/SELL/HOLD (unknown -> HOLD)."""
    if isinstance(signal, str):
        normalized = signal.strip().upper()
        if normalized in (BUY, SELL, HOLD):
            return normalized
    logger.warning("Unknown signal %r treated as HOLD", signal)
    return HOLD


def _last_closes(data: pd.DataFrame | None, count: int) -> np.ndarray | None:
    """Return the last ``count`` closes as float array, or None if unavailable."""
    if data is None or not isinstance(data, pd.DataFrame) or "close" not in data:
        return None
    if len(data) < count:
        return None
    return data["close"].to_numpy(dtype=float)[-count:]


def rsi_gate(
    signal: str,
    rsi: float | None,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> tuple[str, str | None]:
    """Veto trades that chase an already-stretched RSI (roadmap item #2).

    Blocks a BUY when ``rsi > overbought`` (buying into an overbought
    market) and a SELL when ``rsi < oversold`` (selling into an oversold
    market). Boundary values are allowed through (strict comparison).

    Args:
        signal: 'BUY', 'SELL' or 'HOLD' (case-insensitive).
        rsi: Latest RSI reading (0-100). ``None`` or NaN skips the gate.
        overbought: RSI above which BUYs are blocked.
        oversold: RSI below which SELLs are blocked.

    Returns:
        ``(signal, None)`` when the trade passes, or ``('HOLD', reason)``
        when it is blocked.
    """
    normalized = _normalize_signal(signal)
    if rsi is None or normalized == HOLD:
        return normalized, None
    # NaN comparisons are False, so a NaN RSI naturally passes through.
    if normalized == BUY and rsi > overbought:
        reason = f"rsi_gate: RSI {rsi:.1f} > overbought {overbought:.1f}, BUY blocked"
        logger.debug(reason)
        return HOLD, reason
    if normalized == SELL and rsi < oversold:
        reason = f"rsi_gate: RSI {rsi:.1f} < oversold {oversold:.1f}, SELL blocked"
        logger.debug(reason)
        return HOLD, reason
    return normalized, None


def has_momentum(data: pd.DataFrame, direction: str, bars: int = 2) -> bool:
    """Check that price is already moving in the trade direction (item #6).

    Looks at the last ``bars`` closes and requires each one to have moved
    versus its predecessor: strictly rising for 'BUY', strictly falling
    for 'SELL'. This filters out counter-trend entries into flat or
    opposing tape.

    Args:
        data: DataFrame with a ``close`` column.
        direction: 'BUY' (rising) or 'SELL' (falling), case-insensitive.
        bars: Number of consecutive closes that must confirm the move.

    Returns:
        True only when momentum is confirmed. Returns False on
        insufficient data (``len(data) < bars + 1``), missing column,
        NaN closes, non-positive ``bars``, or an unknown direction --
        i.e. "cannot confirm" is treated as "no momentum".
    """
    if bars < 1:
        logger.debug("has_momentum: non-positive bars=%d -> False", bars)
        return False
    normalized = _normalize_signal(direction)
    if normalized not in (BUY, SELL):
        return False
    closes = _last_closes(data, bars + 1)
    if closes is None:
        return False
    moves = np.diff(closes)
    if normalized == BUY:
        return bool(np.all(moves > 0))
    return bool(np.all(moves < 0))


def detect_bb_squeeze(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    squeeze_ratio: float = 0.75,
) -> bool:
    """Detect a Bollinger-band volatility squeeze (roadmap item #7).

    Computes the normalized band width ``(upper - lower) / close`` where
    the bands are the rolling ``window`` mean +/- ``num_std`` standard
    deviations of the close. A squeeze is flagged when the *current*
    width has compressed below ``squeeze_ratio`` times its own rolling
    ``window``-bar average -- i.e. volatility is unusually low versus its
    recent norm, which typically precedes a breakout and makes
    mean-reversion entries risky.

    Args:
        data: DataFrame with a ``close`` column. Needs at least
            ``2 * window - 1`` rows for a fully-formed comparison.
        window: Rolling window for both the bands and the width baseline.
        num_std: Band width in standard deviations.
        squeeze_ratio: Current width must be below this fraction of the
            average width to count as a squeeze.

    Returns:
        True when a squeeze is active; False otherwise, including on
        insufficient/NaN data or a degenerate (zero-width) baseline.
    """
    if (
        data is None
        or not isinstance(data, pd.DataFrame)
        or "close" not in data
        or len(data) < 2 * window - 1
    ):
        return False
    close = data["close"].astype(float)
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mean + num_std * std
    lower = mean - num_std * std
    with np.errstate(divide="ignore", invalid="ignore"):
        width = (upper - lower) / close
    baseline = width.rolling(window).mean()
    current_width = width.iloc[-1]
    current_baseline = baseline.iloc[-1]
    if (
        not np.isfinite(current_width)
        or not np.isfinite(current_baseline)
        or current_baseline <= 0
    ):
        return False
    squeezed = bool(current_width < squeeze_ratio * current_baseline)
    if squeezed:
        logger.debug(
            "bb_squeeze: width %.5f < %.2f * baseline %.5f",
            current_width,
            squeeze_ratio,
            current_baseline,
        )
    return squeezed


def is_optimal_trading_hour(
    now: datetime | None = None,
    optimal_hours: Iterable[int] = range(14, 23),
) -> bool:
    """Gate trading to the most liquid UTC hours (roadmap item #8).

    The default window 14-22 UTC covers the US session overlap where BTC
    volume and spreads are historically most favorable.

    Args:
        now: Time to evaluate; injectable for tests. Defaults to the
            current UTC time. Naive datetimes are assumed to already be
            UTC; aware datetimes are converted.
        optimal_hours: Container of allowed UTC hours (0-23).

    Returns:
        True when the UTC hour of ``now`` is in ``optimal_hours``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is not None:
        now = now.astimezone(timezone.utc)
    return now.hour in optimal_hours


def detect_macd_divergence(data: pd.DataFrame) -> str | None:
    """Spot a short-term price/MACD-histogram divergence (roadmap item #9).

    Compares the direction of the last two closes against the direction
    of the last two ``macd_histogram`` values:

    - price down while histogram up -> ``'BULLISH_DIVERGENCE'``
      (selling pressure fading; upside reversal risk).
    - price up while histogram down -> ``'BEARISH_DIVERGENCE'``
      (buying pressure fading; downside reversal risk).

    Args:
        data: DataFrame with ``close`` and ``macd_histogram`` columns and
            at least 2 rows.

    Returns:
        The divergence label, or ``None`` when there is no divergence or
        the data is insufficient (missing columns, < 2 rows, NaNs, flat
        values).
    """
    if (
        data is None
        or not isinstance(data, pd.DataFrame)
        or "close" not in data
        or "macd_histogram" not in data
        or len(data) < 2
    ):
        return None
    closes = data["close"].to_numpy(dtype=float)[-2:]
    hist = data["macd_histogram"].to_numpy(dtype=float)[-2:]
    price_up = closes[1] > closes[0]
    price_down = closes[1] < closes[0]
    hist_up = hist[1] > hist[0]
    hist_down = hist[1] < hist[0]
    if price_down and hist_up:
        return "BULLISH_DIVERGENCE"
    if price_up and hist_down:
        return "BEARISH_DIVERGENCE"
    return None


def apply_signal_filters(
    signal: str,
    confidence: float,
    data: pd.DataFrame,
    *,
    rsi: float | None = None,
    now: datetime | None = None,
    config: dict[str, bool] | None = None,
) -> tuple[str, float, list[str]]:
    """Run a signal through every quality filter and veto weak setups.

    Evaluates all enabled filters against the *incoming* signal so the
    returned reasons list every objection at once (useful for logging and
    tuning). If any filter objects, the signal is downgraded to
    ``('HOLD', 0.0)``.

    Filters and their config flags (all enabled by default, see
    :data:`DEFAULT_FILTER_CONFIG`):

    - ``rsi_gate`` -- skipped when ``rsi`` is None.
    - ``momentum`` -- requires the last 2 closes to confirm direction;
      blocks when data is too short to confirm (conservative).
    - ``bb_squeeze`` -- blocks entries while a volatility squeeze is on.
    - ``trading_hours`` -- blocks outside 14-22 UTC.
    - ``macd_divergence`` -- blocks a BUY on bearish divergence and a
      SELL on bullish divergence.
    - ``macd_divergence_override`` (default **False**) -- when True and
      the *incoming* signal is HOLD, a detected divergence promotes it to
      BUY (bullish) or SELL (bearish) at confidence 0.75. It never
      overrides a HOLD produced by another filter in the same call.

    Args:
        signal: 'BUY', 'SELL' or 'HOLD' (case-insensitive; unknown values
            are treated as HOLD).
        confidence: Confidence of the incoming signal (passed through on
            success, zeroed on veto).
        data: Candle DataFrame with ``close`` (and ``macd_histogram`` for
            the divergence checks).
        rsi: Latest RSI value for the RSI gate; None skips that gate.
        now: Injectable timestamp for the trading-hours gate.
        config: Partial override of :data:`DEFAULT_FILTER_CONFIG`.

    Returns:
        ``(signal, confidence, reasons)`` -- the possibly-downgraded (or
        divergence-promoted) signal, its confidence, and the list of
        human-readable veto/override reasons (empty when nothing fired).
    """
    cfg = dict(DEFAULT_FILTER_CONFIG)
    if config:
        cfg.update(config)

    try:
        confidence = float(confidence)
    except TypeError, ValueError:
        confidence = 0.0

    normalized = _normalize_signal(signal)
    reasons: list[str] = []

    if normalized in (BUY, SELL):
        if cfg.get("rsi_gate", True) and rsi is not None:
            _, reason = rsi_gate(normalized, rsi)
            if reason is not None:
                reasons.append(reason)

        if cfg.get("momentum", True) and not has_momentum(data, normalized):
            direction = "rising" if normalized == BUY else "falling"
            reasons.append(
                f"momentum: last closes not consistently {direction},"
                f" {normalized} blocked"
            )

        if cfg.get("bb_squeeze", True) and detect_bb_squeeze(data):
            reasons.append(
                f"bb_squeeze: volatility squeeze active, {normalized} blocked"
            )

        if cfg.get("trading_hours", True) and not is_optimal_trading_hour(now):
            reasons.append(
                f"trading_hours: outside optimal UTC hours, {normalized} blocked"
            )

        if cfg.get("macd_divergence", True):
            divergence = detect_macd_divergence(data)
            if (normalized == BUY and divergence == "BEARISH_DIVERGENCE") or (
                normalized == SELL and divergence == "BULLISH_DIVERGENCE"
            ):
                reasons.append(
                    f"macd_divergence: {divergence} contradicts {normalized}"
                )

        if reasons:
            logger.info(
                "Signal %s downgraded to HOLD by %d filter(s): %s",
                normalized,
                len(reasons),
                "; ".join(reasons),
            )
            return HOLD, 0.0, reasons
        return normalized, confidence, []

    # Incoming HOLD: optionally let a divergence generate a signal.
    if cfg.get("macd_divergence_override", False):
        divergence = detect_macd_divergence(data)
        if divergence == "BULLISH_DIVERGENCE":
            reasons.append("macd_divergence_override: BULLISH_DIVERGENCE -> BUY")
            logger.info(reasons[-1])
            return BUY, 0.75, reasons
        if divergence == "BEARISH_DIVERGENCE":
            reasons.append("macd_divergence_override: BEARISH_DIVERGENCE -> SELL")
            logger.info(reasons[-1])
            return SELL, 0.75, reasons

    return HOLD, confidence, []
