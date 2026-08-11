"""Validation helpers shared by immutable domain values."""

import math
from datetime import datetime
from decimal import Decimal
from numbers import Real


def require_clean_text(name: str, value: object) -> None:
    """Require a non-empty string without ambiguous surrounding whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")


def require_optional_clean_text(name: str, value: object | None) -> None:
    """Validate an optional identifier or status string."""
    if value is not None:
        require_clean_text(name, value)


def require_aware_datetime(name: str, value: object) -> None:
    """Require a timestamp whose UTC offset is defined."""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def require_finite_real(name: str, value: object) -> float:
    """Return a finite real value while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def require_positive_decimal(name: str, value: object) -> None:
    """Require exact, finite, positive order-boundary arithmetic."""
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def require_non_negative_decimal(name: str, value: object) -> None:
    """Require an exact, finite Decimal which may be zero."""
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
