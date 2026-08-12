import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)
from trading.domain.signals import Signal, SignalAction
from trading.orders.base_order import OrderSide as LegacyOrderSide

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def make_signal(**overrides: object) -> Signal:
    values = {
        "decision_id": "decision-1",
        "symbol": "BTCUSDT",
        "action": SignalAction.BUY,
        "confidence": 0.8,
        "reference_price": 123_456.75,
        "observed_at": NOW,
        "strategy_id": "ensemble-v1",
        "reasons": ("ensemble quorum",),
    }
    values.update(overrides)
    return Signal(**values)


def make_intent(**overrides: object) -> OrderIntent:
    values = {
        "client_order_id": "elvis-order-1",
        "decision_id": "decision-1",
        "symbol": "BTCUSDT",
        "side": OrderSide.BUY,
        "quantity": Decimal("0.001"),
        "order_type": OrderType.MARKET,
        "reference_price": Decimal("123456.75"),
        "leverage": 3,
        "created_at": NOW,
    }
    values.update(overrides)
    return OrderIntent(**values)


def make_report(**overrides: object) -> SubmissionReport:
    values = {
        "client_order_id": "elvis-order-1",
        "status": SubmissionStatus.SUBMITTED,
        "retry_safety": RetrySafety.UNSAFE,
        "venue_order_id": "paper-order-1",
        "venue_status": "FILLED",
    }
    values.update(overrides)
    return SubmissionReport(**values)


def test_signal_is_immutable_and_accepts_hold() -> None:
    signal = make_signal(action=SignalAction.HOLD)

    assert signal.action is SignalAction.HOLD
    with pytest.raises(FrozenInstanceError):
        signal.confidence = 0.9  # type: ignore[misc]


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_signal_accepts_confidence_boundaries(confidence: float) -> None:
    assert make_signal(confidence=confidence).confidence == confidence


def test_signal_normalises_real_values_to_float() -> None:
    signal = make_signal(confidence=1, reference_price=123_456)

    assert type(signal.confidence) is float
    assert type(signal.reference_price) is float


@pytest.mark.parametrize("field", ["decision_id", "symbol", "strategy_id"])
@pytest.mark.parametrize("value", ["", "   ", " padded "])
def test_signal_rejects_invalid_identifiers(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        make_signal(**{field: value})


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_signal_rejects_out_of_range_or_non_finite_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValueError):
        make_signal(confidence=confidence)


def test_signal_rejects_boolean_confidence() -> None:
    with pytest.raises(TypeError):
        make_signal(confidence=True)


@pytest.mark.parametrize(
    "reference_price",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_signal_rejects_invalid_reference_price(reference_price: float) -> None:
    with pytest.raises(ValueError):
        make_signal(reference_price=reference_price)


@pytest.mark.parametrize("reference_price", [True, "1"])
def test_signal_rejects_non_numeric_reference_price(reference_price: object) -> None:
    with pytest.raises(TypeError):
        make_signal(reference_price=reference_price)


def test_signal_rejects_invalid_action_and_naive_timestamp() -> None:
    with pytest.raises(TypeError):
        make_signal(action="BUY")

    with pytest.raises(ValueError):
        make_signal(observed_at=datetime(2026, 8, 11, 12, 0))


@pytest.mark.parametrize("reasons", [["mutable"], ("",), (" padded ",)])
def test_signal_rejects_invalid_reasons(reasons: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_signal(reasons=reasons)


def test_order_intent_is_immutable_and_accepts_sell() -> None:
    intent = make_intent(side=OrderSide.SELL)

    assert intent.side is OrderSide.SELL
    with pytest.raises(FrozenInstanceError):
        intent.quantity = Decimal("1")  # type: ignore[misc]


@pytest.mark.parametrize("field", ["client_order_id", "decision_id", "symbol"])
@pytest.mark.parametrize("value", ["", "   ", " padded "])
def test_order_intent_rejects_invalid_identifiers(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        make_intent(**{field: value})


@pytest.mark.parametrize("field", ["quantity", "reference_price"])
@pytest.mark.parametrize(
    "value",
    [
        Decimal("0"),
        Decimal("-1"),
        Decimal("NaN"),
        Decimal("Infinity"),
    ],
)
def test_order_intent_rejects_invalid_decimal_values(
    field: str, value: Decimal
) -> None:
    with pytest.raises(ValueError):
        make_intent(**{field: value})


@pytest.mark.parametrize("field", ["quantity", "reference_price"])
@pytest.mark.parametrize("value", [1, 1.0, True, "1"])
def test_order_intent_requires_decimal_values(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        make_intent(**{field: value})


@pytest.mark.parametrize("leverage", [0, -1])
def test_order_intent_rejects_non_positive_leverage(leverage: int) -> None:
    with pytest.raises(ValueError):
        make_intent(leverage=leverage)


@pytest.mark.parametrize("leverage", [True, 1.0, Decimal("1")])
def test_order_intent_requires_integer_leverage(leverage: object) -> None:
    with pytest.raises(TypeError):
        make_intent(leverage=leverage)


def test_hold_cannot_become_an_order_intent() -> None:
    with pytest.raises(TypeError):
        make_intent(side=SignalAction.HOLD)


def test_legacy_order_side_cannot_cross_the_domain_boundary() -> None:
    with pytest.raises(TypeError):
        make_intent(side=LegacyOrderSide.BUY)


def test_order_intent_rejects_invalid_type_and_naive_timestamp() -> None:
    with pytest.raises(TypeError):
        make_intent(order_type="MARKET")

    with pytest.raises(ValueError):
        make_intent(created_at=datetime(2026, 8, 11, 12, 0))


def test_submission_report_exposes_submission_semantics() -> None:
    report = make_report()

    assert report.acknowledged is True
    assert report.requires_reconciliation is False
    with pytest.raises(FrozenInstanceError):
        report.status = SubmissionStatus.AMBIGUOUS  # type: ignore[misc]


@pytest.mark.parametrize(
    "status",
    [
        SubmissionStatus.NOT_SENT,
        SubmissionStatus.VENUE_REJECTED,
        SubmissionStatus.AMBIGUOUS,
    ],
)
def test_non_submitted_report_requires_a_reason(status: SubmissionStatus) -> None:
    with pytest.raises(ValueError):
        make_report(
            status=status,
            reason=None,
            venue_order_id=None,
            venue_status=None,
        )


def test_submitted_report_requires_a_venue_order_id() -> None:
    with pytest.raises(ValueError):
        make_report(venue_order_id=None)


@pytest.mark.parametrize(
    "status", [SubmissionStatus.NOT_SENT, SubmissionStatus.VENUE_REJECTED]
)
def test_proven_non_submission_rejects_a_venue_order_id(
    status: SubmissionStatus,
) -> None:
    with pytest.raises(ValueError):
        make_report(status=status, reason="not accepted", venue_order_id="unexpected")


def test_not_sent_report_rejects_a_venue_status() -> None:
    with pytest.raises(ValueError):
        make_report(
            status=SubmissionStatus.NOT_SENT,
            reason="blocked locally",
            venue_order_id=None,
            venue_status="FILLED",
        )


@pytest.mark.parametrize(
    "status", [SubmissionStatus.SUBMITTED, SubmissionStatus.AMBIGUOUS]
)
def test_possibly_sent_report_cannot_be_marked_safe_to_retry(
    status: SubmissionStatus,
) -> None:
    with pytest.raises(ValueError):
        make_report(
            status=status,
            retry_safety=RetrySafety.SAFE,
            reason=(
                "transport outcome is unresolved"
                if status is SubmissionStatus.AMBIGUOUS
                else None
            ),
            venue_order_id=(
                "paper-order-1" if status is SubmissionStatus.SUBMITTED else None
            ),
        )


def test_ambiguous_report_requires_reconciliation() -> None:
    report = make_report(
        status=SubmissionStatus.AMBIGUOUS,
        reason="transport timed out",
        venue_order_id=None,
        venue_status=None,
    )

    assert report.acknowledged is False
    assert report.requires_reconciliation is True


@pytest.mark.parametrize(
    ("status", "retry_safety"),
    [
        (SubmissionStatus.NOT_SENT, RetrySafety.SAFE),
        (SubmissionStatus.NOT_SENT, RetrySafety.UNSAFE),
        (SubmissionStatus.VENUE_REJECTED, RetrySafety.SAFE),
        (SubmissionStatus.VENUE_REJECTED, RetrySafety.UNSAFE),
    ],
)
def test_known_non_submission_keeps_retry_safety_separate(
    status: SubmissionStatus, retry_safety: RetrySafety
) -> None:
    report = make_report(
        status=status,
        retry_safety=retry_safety,
        reason="known outcome",
        venue_order_id=None,
        venue_status=None,
    )

    assert report.retry_safety is retry_safety


@pytest.mark.parametrize("reason", ["", "   ", " padded "])
def test_submission_report_rejects_invalid_reason(reason: str) -> None:
    with pytest.raises(ValueError):
        make_report(
            status=SubmissionStatus.NOT_SENT,
            reason=reason,
            venue_order_id=None,
            venue_status=None,
        )


@pytest.mark.parametrize("field", ["client_order_id", "venue_order_id"])
def test_submission_report_rejects_blank_ids(field: str) -> None:
    with pytest.raises(ValueError):
        make_report(**{field: " "})


def test_submission_report_rejects_untyped_enums() -> None:
    with pytest.raises(TypeError):
        make_report(status="SUBMITTED")

    with pytest.raises(TypeError):
        make_report(retry_safety="UNSAFE")


def test_domain_package_has_only_standard_library_and_internal_imports() -> None:
    domain_dir = Path(__file__).parents[1] / "trading" / "domain"
    standard_library_roots = {
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "math",
        "numbers",
    }

    imported_modules: set[str] = set()
    for module_path in domain_dir.glob("*.py"):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

    unexpected = {
        module
        for module in imported_modules
        if module.split(".", 1)[0] not in standard_library_roots
        and not module.startswith("trading.domain")
    }
    assert unexpected == set()
