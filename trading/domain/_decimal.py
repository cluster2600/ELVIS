"""Exact Decimal arithmetic shared by pure domain reducers."""

from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    Inexact,
    InvalidOperation,
    Overflow,
)

_MAX_EXACT_ARITHMETIC_DIGITS = 10_000
_MAX_EXACT_ARITHMETIC_EXPONENT = 999_999_999


def _exact_context(required_digits: int) -> Context:
    return Context(
        prec=max(1, required_digits),
        rounding=ROUND_HALF_EVEN,
        Emax=_MAX_EXACT_ARITHMETIC_EXPONENT,
        Emin=-_MAX_EXACT_ARITHMETIC_EXPONENT,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[Inexact, InvalidOperation, Overflow],
    )


def _require_exact_values(values: object) -> tuple[Decimal, ...]:
    if not isinstance(values, tuple):
        raise TypeError("exact Decimal values must be a tuple")
    if any(not isinstance(value, Decimal) for value in values):
        raise TypeError("exact Decimal values must contain only Decimals")
    if any(not value.is_finite() for value in values):
        raise ValueError("exact Decimal values must be finite")
    return values


def exact_decimal_sum(values: tuple[Decimal, ...]) -> Decimal:
    """Sum finite Decimals exactly without consulting the ambient context."""
    values = _require_exact_values(values)
    values = tuple(value for value in values if value)
    if not values:
        return Decimal("0")

    components = tuple(value.as_tuple() for value in values)
    exponents = tuple(int(component.exponent) for component in components)
    min_exponent = min(exponents)
    required_digits = (
        max(
            len(component.digits) + exponent - min_exponent
            for component, exponent in zip(components, exponents)
        )
        + len(str(len(values)))
        + 1
    )
    if required_digits > _MAX_EXACT_ARITHMETIC_DIGITS:
        raise ValueError("exact Decimal arithmetic exceeds the supported precision")
    if any(
        abs(value.adjusted()) > _MAX_EXACT_ARITHMETIC_EXPONENT
        or abs(int(value.as_tuple().exponent)) > _MAX_EXACT_ARITHMETIC_EXPONENT
        for value in values
        if value
    ):
        raise ValueError("exact Decimal arithmetic exceeds the supported exponent")

    try:
        context = _exact_context(required_digits)
        total = Decimal("0")
        for value in values:
            total = context.add(total, value)
    except (DecimalException, OverflowError, ValueError) as exc:
        raise ValueError("Decimal value cannot be represented exactly") from exc
    if not total.is_finite():
        raise ValueError("Decimal result must remain finite")
    return total


def exact_decimal_product(values: tuple[Decimal, ...]) -> Decimal:
    """Multiply finite Decimals exactly without using the ambient context."""
    values = _require_exact_values(values)
    if not values:
        return Decimal("1")
    if any(not value for value in values):
        return Decimal("0")

    components = tuple(value.as_tuple() for value in values)
    required_digits = sum(len(component.digits) for component in components)
    if required_digits > _MAX_EXACT_ARITHMETIC_DIGITS:
        raise ValueError("exact Decimal arithmetic exceeds the supported precision")
    if any(
        abs(value.adjusted()) > _MAX_EXACT_ARITHMETIC_EXPONENT
        or abs(int(value.as_tuple().exponent)) > _MAX_EXACT_ARITHMETIC_EXPONENT
        for value in values
    ):
        raise ValueError("exact Decimal arithmetic exceeds the supported exponent")

    try:
        context = _exact_context(required_digits)
        product = Decimal("1")
        for value in values:
            product = context.multiply(product, value)
    except (DecimalException, OverflowError, ValueError) as exc:
        raise ValueError("Decimal value cannot be represented exactly") from exc
    if not product.is_finite():
        raise ValueError("Decimal result must remain finite")
    if product and (
        abs(product.adjusted()) > _MAX_EXACT_ARITHMETIC_EXPONENT
        or abs(int(product.as_tuple().exponent)) > _MAX_EXACT_ARITHMETIC_EXPONENT
    ):
        raise ValueError("exact Decimal arithmetic exceeds the supported exponent")
    return product
