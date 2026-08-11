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


def exact_decimal_sum(values: tuple[Decimal, ...]) -> Decimal:
    """Sum finite Decimals exactly without consulting the ambient context."""
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
        context = Context(
            prec=max(1, required_digits),
            rounding=ROUND_HALF_EVEN,
            Emax=_MAX_EXACT_ARITHMETIC_EXPONENT,
            Emin=-_MAX_EXACT_ARITHMETIC_EXPONENT,
            capitals=1,
            clamp=0,
            flags=[],
            traps=[Inexact, InvalidOperation, Overflow],
        )
        total = Decimal("0")
        for value in values:
            total = context.add(total, value)
    except (DecimalException, OverflowError, ValueError) as exc:
        raise ValueError("Decimal value cannot be represented exactly") from exc
    if not total.is_finite():
        raise ValueError("Decimal result must remain finite")
    return total
