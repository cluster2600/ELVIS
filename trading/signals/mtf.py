"""Multi-timeframe (MTF) trend alignment signals (roadmap item #14).

How it works
------------
A simple-moving-average (SMA) crossover is computed independently on several
timeframes (e.g. 15m / 1h / 4h). Each timeframe yields ``'BUY'``, ``'SELL'``
or ``'HOLD'`` depending on whether the fast SMA is above/below the slow SMA
by more than a small threshold (0.1% by default, to filter out noise around
the crossover point). The overall ("aligned") signal is only ``'BUY'`` when
*every* timeframe agrees on ``'BUY'`` (and ``'SELL'`` when all agree on
``'SELL'``); any disagreement or missing data degrades to ``'HOLD'``.
Requiring agreement across timeframes filters counter-trend entries: a
15-minute bounce inside a 4-hour downtrend will not produce a ``'BUY'``.

How to use it
-------------
The kline source is dependency-injected so the module stays free of network
and exchange-client dependencies (stdlib + numpy + pandas only)::

    from trading.signals.mtf import MTFAnalyzer
    from utils.price_fetcher import PriceFetcher

    fetcher = PriceFetcher(...)
    mtf = MTFAnalyzer(fetch_klines=fetcher.get_historical_klines)
    result = mtf.get_signal("BTCUSDT")
    # result == {
    #     "symbol": "BTCUSDT",
    #     "signals": {"15m": "BUY", "1h": "BUY", "4h": "HOLD"},
    #     "aligned": "HOLD",
    # }

``fetch_klines(symbol, interval, limit)`` may return either a
``pd.DataFrame`` with a ``close`` column (what
``PriceFetcher.get_historical_klines`` returns) or a raw list of Binance
kline lists where index 4 is the close price. Any fetch failure (exception,
``None``, empty or malformed data) maps that timeframe to ``'HOLD'``;
``get_signal`` never raises.

``sma_trend_signal`` can also be used standalone on any DataFrame that has a
``close`` column.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Signal constants (shared vocabulary with the rest of the bot).
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

#: Minimum relative SMA separation for a directional signal (0.1%).
DEFAULT_THRESHOLD = 0.001

#: Position of the close price in a raw Binance kline list.
_KLINE_CLOSE_INDEX = 4

#: What an injected kline fetcher may return.
KlinesData = Union[pd.DataFrame, Sequence[Sequence[object]]]
FetchKlines = Callable[[str, str, int], Optional[KlinesData]]


def _coerce_close(data: Optional[KlinesData]) -> pd.Series:
    """Extract a numeric close-price Series from heterogeneous kline data.

    Accepts a DataFrame with a ``close`` column, a DataFrame whose 5th
    column is the close (raw klines loaded without headers), or a
    list/tuple/ndarray of kline rows where index 4 is the close. Values are
    coerced to float and NaNs dropped. Returns an empty Series when the
    data is missing, empty, or malformed (never raises).
    """
    if data is None:
        return pd.Series(dtype=float)

    try:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return pd.Series(dtype=float)
            if "close" in data.columns:
                raw = data["close"]
            elif data.shape[1] > _KLINE_CLOSE_INDEX:
                raw = data.iloc[:, _KLINE_CLOSE_INDEX]
            else:
                logger.warning(
                    "Kline DataFrame has no 'close' column and too few "
                    "columns (%d) to infer one",
                    data.shape[1],
                )
                return pd.Series(dtype=float)
        else:
            rows = list(data)
            closes = [
                row[_KLINE_CLOSE_INDEX]
                for row in rows
                if isinstance(row, (list, tuple, np.ndarray))
                and len(row) > _KLINE_CLOSE_INDEX
            ]
            raw = pd.Series(closes, dtype=object)

        close = pd.to_numeric(raw, errors="coerce").dropna()
        return close.reset_index(drop=True).astype(float)
    except Exception:
        logger.warning("Failed to coerce kline data to close series", exc_info=True)
        return pd.Series(dtype=float)


def sma_trend_signal(
    data: Optional[pd.DataFrame],
    fast: int = 20,
    slow: int = 50,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Classify the trend of one timeframe via an SMA(fast)/SMA(slow) spread.

    Returns ``'BUY'`` when the fast SMA exceeds the slow SMA by more than
    ``threshold`` (relative, default 0.1%), ``'SELL'`` when it is below by
    more than ``threshold``, and ``'HOLD'`` otherwise. Insufficient data
    (fewer than ``slow`` valid closes after dropping NaNs), missing/empty
    input, or invalid parameters all return ``'HOLD'`` — this function
    never raises on bad market data.

    Args:
        data: DataFrame with a ``close`` column (raw kline rows also
            tolerated, see :func:`_coerce_close`).
        fast: Fast SMA window (must be ``0 < fast <= slow``).
        slow: Slow SMA window; also the minimum number of closes required.
        threshold: Relative separation required for a directional signal.

    Returns:
        ``'BUY'``, ``'SELL'`` or ``'HOLD'``.
    """
    if fast <= 0 or slow <= 0 or fast > slow:
        logger.warning(
            "Invalid SMA windows fast=%s slow=%s; returning HOLD", fast, slow
        )
        return HOLD

    close = _coerce_close(data)
    if len(close) < slow:
        logger.debug(
            "Not enough closes for SMA trend (%d < %d); returning HOLD",
            len(close),
            slow,
        )
        return HOLD

    sma_fast = float(close.iloc[-fast:].mean())
    sma_slow = float(close.iloc[-slow:].mean())
    if not np.isfinite(sma_fast) or not np.isfinite(sma_slow) or sma_slow <= 0:
        logger.warning(
            "Non-finite or non-positive SMAs (fast=%s slow=%s); returning HOLD",
            sma_fast,
            sma_slow,
        )
        return HOLD

    spread = sma_fast / sma_slow - 1.0
    if spread > threshold:
        return BUY
    if spread < -threshold:
        return SELL
    return HOLD


class MTFAnalyzer:
    """Multi-timeframe SMA trend alignment.

    Fetches klines for each configured timeframe through an injected
    ``fetch_klines(symbol, interval, limit)`` callable, computes a per-
    timeframe :func:`sma_trend_signal`, and reports the aligned consensus:
    ``'BUY'`` only if every timeframe says ``'BUY'``, ``'SELL'`` only if
    every timeframe says ``'SELL'``, otherwise ``'HOLD'``.

    Live wiring::

        mtf = MTFAnalyzer(fetch_klines=price_fetcher.get_historical_klines)
        if mtf.get_signal("BTCUSDT")["aligned"] == "BUY":
            ...

    Any per-timeframe fetch failure (raised exception, ``None``, empty or
    malformed data) maps that timeframe to ``'HOLD'`` — which forces the
    aligned signal to ``'HOLD'`` — and :meth:`get_signal` never raises.

    Args:
        fetch_klines: Callable returning a DataFrame with a ``close``
            column or a list of raw kline lists (index 4 = close).
        timeframes: Interval strings to analyze (order preserved in the
            result). Must be non-empty.
        limit: Number of klines requested per timeframe; must be at least
            ``slow`` for directional signals to be possible.
        fast: Fast SMA window passed to :func:`sma_trend_signal`.
        slow: Slow SMA window passed to :func:`sma_trend_signal`.
    """

    def __init__(
        self,
        fetch_klines: FetchKlines,
        timeframes: Tuple[str, ...] = ("15m", "1h", "4h"),
        limit: int = 60,
        fast: int = 20,
        slow: int = 50,
    ) -> None:
        if not callable(fetch_klines):
            raise TypeError("fetch_klines must be callable")
        self.fetch_klines = fetch_klines
        self.timeframes: Tuple[str, ...] = tuple(timeframes)
        self.limit = int(limit)
        self.fast = int(fast)
        self.slow = int(slow)
        if self.limit < self.slow:
            logger.warning(
                "limit=%d is below slow=%d; every timeframe will return HOLD",
                self.limit,
                self.slow,
            )

    def _timeframe_signal(self, symbol: str, interval: str) -> str:
        """Fetch one timeframe and classify it; failures map to HOLD."""
        try:
            raw = self.fetch_klines(symbol, interval, self.limit)
        except Exception:
            logger.warning(
                "fetch_klines failed for %s %s; treating as HOLD",
                symbol,
                interval,
                exc_info=True,
            )
            return HOLD

        close = _coerce_close(raw)
        if close.empty:
            logger.debug("No usable closes for %s %s; HOLD", symbol, interval)
            return HOLD
        return sma_trend_signal(
            pd.DataFrame({"close": close}), fast=self.fast, slow=self.slow
        )

    def get_signal(self, symbol: str) -> Dict[str, object]:
        """Compute per-timeframe signals and their aligned consensus.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTCUSDT"``.

        Returns:
            ``{"symbol": symbol, "signals": {interval: signal, ...},
            "aligned": "BUY"|"SELL"|"HOLD"}``. Never raises: any failure
            degrades the affected timeframe (and hence the aligned
            signal) to ``'HOLD'``.
        """
        signals: Dict[str, str] = {}
        for interval in self.timeframes:
            try:
                signals[interval] = self._timeframe_signal(symbol, interval)
            except Exception:  # pragma: no cover - defensive belt-and-braces
                logger.error(
                    "Unexpected error computing %s %s signal; HOLD",
                    symbol,
                    interval,
                    exc_info=True,
                )
                signals[interval] = HOLD

        if signals and all(s == BUY for s in signals.values()):
            aligned = BUY
        elif signals and all(s == SELL for s in signals.values()):
            aligned = SELL
        else:
            aligned = HOLD

        logger.debug("MTF %s: signals=%s aligned=%s", symbol, signals, aligned)
        return {"symbol": symbol, "signals": signals, "aligned": aligned}
