"""
utils/binance_rate_limiter.py — Issue #12: Binance API Rate Limiting

Provides a retry decorator (using tenacity) with exponential backoff for all
Binance API calls, respects Binance rate limits, logs rate-limit response
headers, and auto-pauses the caller when the limit window is nearly exhausted.

Binance documented limits (USDⓈ-M Futures):
    - 1200 request-weight per minute (HTTP header: X-MBX-USED-WEIGHT-1M)
    - 10 orders per second (HTTP header: X-MBX-ORDER-COUNT-1S)

Usage:
    from utils.binance_rate_limiter import binance_retry, check_rate_limit_headers

    @binance_retry
    def my_binance_call():
        return client.some_endpoint()
"""

import functools
import logging
import time
from typing import Any, Callable, Optional

from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit thresholds (env-overridable via os.getenv, but sane defaults)
# ---------------------------------------------------------------------------
import os

# Warn / pause when used-weight exceeds this fraction of the per-minute limit.
_WEIGHT_WARN_FRACTION = float(os.getenv("BINANCE_RATE_LIMIT_WARN_FRACTION", "0.80"))
_WEIGHT_LIMIT_PER_MIN = int(os.getenv("BINANCE_WEIGHT_LIMIT_PER_MIN", "1200"))
_ORDER_LIMIT_PER_SEC = int(os.getenv("BINANCE_ORDER_LIMIT_PER_SEC", "10"))

# Seconds to sleep when we detect the window is nearly full (>80 % weight used).
_RATE_LIMIT_PAUSE_SECONDS = float(os.getenv("BINANCE_RATE_LIMIT_PAUSE_SECONDS", "5.0"))


# ---------------------------------------------------------------------------
# Exceptions that should trigger a retry
# ---------------------------------------------------------------------------
try:
    from binance.error import ClientError as BinanceFuturesClientError
except ImportError:
    BinanceFuturesClientError = None  # futures connector not installed

try:
    from binance.exceptions import BinanceAPIException, BinanceRequestException
except ImportError:
    BinanceAPIException = Exception
    BinanceRequestException = Exception

# Build the tuple of retriable exception types dynamically.
_RETRIABLE_EXCEPTIONS = [
    BinanceAPIException,
    BinanceRequestException,
    ConnectionError,
    TimeoutError,
]
if BinanceFuturesClientError is not None:
    _RETRIABLE_EXCEPTIONS.append(BinanceFuturesClientError)

_RETRIABLE_EXCEPTIONS = tuple(set(_RETRIABLE_EXCEPTIONS))


def _is_retriable(exc: BaseException) -> bool:
    """
    Decide whether an exception should trigger a retry.

    We skip retry for permanent client errors (4xx) that are not rate-limit
    responses (HTTP 429 / 418) because retrying them is pointless.
    """
    # Always retry network-level errors.
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    # Binance futures ClientError carries an HTTP status code.
    if BinanceFuturesClientError is not None and isinstance(
        exc, BinanceFuturesClientError
    ):
        status = getattr(exc, "status_code", None)
        if status in (429, 418):
            return True  # Rate limited — definitely retry.
        if status and 400 <= status < 500:
            return False  # Other 4xx (bad request etc.) — don't retry.
        return True  # 5xx or unknown — retry.

    # python-binance BinanceAPIException.
    if isinstance(exc, BinanceAPIException):
        code = getattr(exc, "status_code", None)
        if code in (429, 418):
            return True
        if code and 400 <= code < 500:
            return False
        return True

    # Any other exception in the retriable set: retry.
    return isinstance(exc, _RETRIABLE_EXCEPTIONS)


def _log_retry(retry_state: RetryCallState) -> None:
    """Called by tenacity before each sleep between retries."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt = retry_state.attempt_number
    sleep_time = getattr(retry_state.next_action, "sleep", "?")
    logger.warning(
        "Binance API call failed (attempt %d). Retrying in %.1fs. Error: %s",
        attempt,
        sleep_time,
        exc,
    )


# ---------------------------------------------------------------------------
# The core retry decorator
# ---------------------------------------------------------------------------
binance_retry = retry(
    reraise=True,
    retry=retry_if_exception_type(_RETRIABLE_EXCEPTIONS),
    stop=stop_after_attempt(int(os.getenv("BINANCE_MAX_RETRIES", "5"))),
    wait=wait_exponential(
        multiplier=float(os.getenv("BINANCE_RETRY_MULTIPLIER", "1")),
        min=float(os.getenv("BINANCE_RETRY_MIN_WAIT", "1")),
        max=float(os.getenv("BINANCE_RETRY_MAX_WAIT", "60")),
    ),
    before_sleep=_log_retry,
)
"""
Decorator: wrap any Binance API call with automatic exponential-backoff retry.

    Attempts : 5  (BINANCE_MAX_RETRIES env var)
    Back-off  : 1 s → 2 s → 4 s → … up to 60 s (BINANCE_RETRY_MIN/MAX_WAIT)
    Multiplier: 1  (BINANCE_RETRY_MULTIPLIER)

Example::

    @binance_retry
    def fetch_klines(client, symbol):
        return client.klines(symbol=symbol, interval="1m")
"""


# ---------------------------------------------------------------------------
# Response-header inspection helpers
# ---------------------------------------------------------------------------


def check_rate_limit_headers(response_headers: dict) -> None:
    """
    Inspect Binance HTTP response headers and log rate-limit usage.

    Pauses execution (_RATE_LIMIT_PAUSE_SECONDS) when used weight exceeds
    _WEIGHT_WARN_FRACTION of _WEIGHT_LIMIT_PER_MIN, giving the window time
    to roll over.

    Args:
        response_headers: The raw headers dict from a requests.Response or
                          equivalent mapping (keys are case-insensitive).
    """
    # Normalise header keys to lower-case for consistent lookup.
    headers = {k.lower(): v for k, v in (response_headers or {}).items()}

    used_weight = _parse_int(headers.get("x-mbx-used-weight-1m"))
    order_count_1s = _parse_int(headers.get("x-mbx-order-count-1s"))
    order_count_1d = _parse_int(headers.get("x-mbx-order-count-1d"))
    retry_after = _parse_int(headers.get("retry-after"))

    # Always log current utilisation at DEBUG level.
    if used_weight is not None:
        utilisation_pct = (used_weight / _WEIGHT_LIMIT_PER_MIN) * 100
        logger.debug(
            "Binance rate-limit — weight used: %d / %d (%.1f%%)",
            used_weight,
            _WEIGHT_LIMIT_PER_MIN,
            utilisation_pct,
        )

    if order_count_1s is not None:
        logger.debug(
            "Binance order count (1 s): %d / %d", order_count_1s, _ORDER_LIMIT_PER_SEC
        )

    if order_count_1d is not None:
        logger.debug("Binance order count (1 day): %d", order_count_1d)

    # Warn and pause when approaching the weight limit.
    warn_threshold = _WEIGHT_WARN_FRACTION * _WEIGHT_LIMIT_PER_MIN
    if used_weight is not None and used_weight >= warn_threshold:
        logger.warning(
            "⚠️  Binance rate-limit approaching: %d / %d weight used (%.0f%%). "
            "Pausing %.1f s to avoid HTTP 429.",
            used_weight,
            _WEIGHT_LIMIT_PER_MIN,
            (used_weight / _WEIGHT_LIMIT_PER_MIN) * 100,
            _RATE_LIMIT_PAUSE_SECONDS,
        )
        time.sleep(_RATE_LIMIT_PAUSE_SECONDS)

    # If Binance told us explicitly to back off, honour it.
    if retry_after is not None and retry_after > 0:
        logger.warning(
            "🛑 Binance returned Retry-After: %d s. Pausing to comply.", retry_after
        )
        time.sleep(retry_after)


def rate_limited_call(
    api_func: Callable, *args, response_attr: str = "headers", **kwargs
) -> Any:
    """
    Call *api_func* with retry logic and automatic rate-limit header checking.

    This is a convenience wrapper for callers that have access to a raw
    requests.Response object.  If the library returns something other than a
    Response (e.g. a dict), header checking is skipped gracefully.

    Args:
        api_func    : The Binance SDK method to call.
        *args       : Positional arguments forwarded to api_func.
        response_attr: Attribute name on the result that holds headers
                       (default "headers"; only used if it exists).
        **kwargs    : Keyword arguments forwarded to api_func.

    Returns:
        The return value of api_func.
    """

    @binance_retry
    def _call():
        result = api_func(*args, **kwargs)
        # Check headers if accessible (raw requests.Response objects).
        raw_headers = getattr(result, response_attr, None)
        if isinstance(raw_headers, dict):
            check_rate_limit_headers(raw_headers)
        return result

    return _call()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_int(value: Optional[str]) -> Optional[int]:
    """Safely parse a string header value to int."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError, TypeError:
        return None
