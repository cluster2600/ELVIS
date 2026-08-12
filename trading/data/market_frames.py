"""Helpers for building isolated, indicator-enriched market frames."""

import logging
import math
from typing import Mapping

import pandas as pd

from trading.analysis.technical_indicators import add_technical_indicators

_REQUIRED_INDICATOR_COLUMNS = (
    "sma_20",
    "sma_50",
    "adx",
    "rsi",
    "macd",
    "signal_line",
    "macd_histogram",
    "lower_bb",
    "sma_bb",
    "upper_bb",
    "atr",
)
_REQUIRED_MARKET_COLUMNS = ("close", "high", "low", "volume")
_REQUIRED_LATEST_COLUMNS = _REQUIRED_MARKET_COLUMNS + _REQUIRED_INDICATOR_COLUMNS
_REQUIRED_DIVERGENCE_COLUMNS = ("close", "macd_histogram")


def _has_finite_tail_values(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
    *,
    rows: int = 1,
) -> bool:
    if (
        rows < 1
        or len(frame) < rows
        or not set(required_columns).issubset(frame.columns)
    ):
        return False
    try:
        return all(
            math.isfinite(float(value))
            for column in required_columns
            for value in frame[column].iloc[-rows:]
        )
    except (TypeError, ValueError, OverflowError):
        return False


def enrich_symbol_frames(
    frames: Mapping[str, pd.DataFrame],
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Copy and fully enrich valid symbols, omitting incomplete results."""
    active_logger = logger or logging.getLogger(__name__)
    enriched: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            active_logger.warning(
                "Skipping %s: market frame is missing or empty", symbol
            )
            continue
        if not _has_finite_tail_values(frame, _REQUIRED_MARKET_COLUMNS):
            active_logger.warning(
                "Skipping %s: latest market observation is incomplete", symbol
            )
            continue
        try:
            candidate = add_technical_indicators(frame.copy(deep=True), logger)
        except Exception as exc:
            active_logger.warning(
                "Skipping %s: indicator enrichment raised %s",
                symbol,
                type(exc).__name__,
            )
            continue
        if (
            not isinstance(candidate, pd.DataFrame)
            or not _has_finite_tail_values(candidate, _REQUIRED_LATEST_COLUMNS)
            or not _has_finite_tail_values(
                candidate, _REQUIRED_DIVERGENCE_COLUMNS, rows=2
            )
        ):
            active_logger.warning(
                "Skipping %s: indicator enrichment is incomplete", symbol
            )
            continue
        enriched[symbol] = candidate
    return enriched
