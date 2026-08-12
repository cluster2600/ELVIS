import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, Rounded, localcontext
from pathlib import Path

import pytest

from trading.domain import (
    CancellationConfirmed,
    CancellationRejected,
    CancellationRequested,
    ConfirmedFill,
    InvalidOrderTransition,
    OrderIntent,
    OrderLifecycle,
    OrderLifecycleState,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionAcknowledged,
    SubmissionAmbiguous,
    SubmissionFailed,
    SubmissionReport,
    SubmissionStatus,
    new_order_lifecycle,
    reduce_order_lifecycle,
    submission_event_from_report,
)
from trading.orders.base_order import OrderStatus as LegacyOrderStatus

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(seconds=1)


def make_intent(**overrides: object) -> OrderIntent:
    values = {
        "client_order_id": "elvis-order-1",
        "decision_id": "decision-1",
        "symbol": "BTCUSDT",
        "side": OrderSide.BUY,
        "quantity": Decimal("1.0"),
        "order_type": OrderType.MARKET,
        "reference_price": Decimal("50000"),
        "leverage": 3,
        "created_at": NOW,
    }
    values.update(overrides)
    return OrderIntent(**values)


def make_ack(**overrides: object) -> SubmissionAcknowledged:
    values = {
        "client_order_id": "elvis-order-1",
        "venue_order_id": "venue-order-1",
        "observed_at": LATER,
    }
    values.update(overrides)
    return SubmissionAcknowledged(**values)


def make_ambiguous(**overrides: object) -> SubmissionAmbiguous:
    values = {
        "client_order_id": "elvis-order-1",
        "reason": "transport timed out",
        "observed_at": LATER,
        "venue_order_id": None,
    }
    values.update(overrides)
    return SubmissionAmbiguous(**values)


def make_failed(**overrides: object) -> SubmissionFailed:
    values = {
        "client_order_id": "elvis-order-1",
        "status": SubmissionStatus.VENUE_REJECTED,
        "retry_safety": RetrySafety.SAFE,
        "reason": "venue rejected the order",
        "observed_at": LATER,
    }
    values.update(overrides)
    return SubmissionFailed(**values)


def make_fill(**overrides: object) -> ConfirmedFill:
    values = {
        "client_order_id": "elvis-order-1",
        "venue_order_id": "venue-order-1",
        "trade_id": "trade-1",
        "symbol": "BTCUSDT",
        "side": OrderSide.BUY,
        "quantity": Decimal("0.4"),
        "price": Decimal("50010"),
        "fee_amount": Decimal("0.2"),
        "fee_asset": "USDT",
        "executed_at": LATER,
    }
    values.update(overrides)
    return ConfirmedFill(**values)


def make_cancel_request(**overrides: object) -> CancellationRequested:
    values = {
        "client_order_id": "elvis-order-1",
        "cancel_request_id": "cancel-1",
        "requested_at": LATER,
    }
    values.update(overrides)
    return CancellationRequested(**values)


def make_cancel_confirmed(**overrides: object) -> CancellationConfirmed:
    values = {
        "client_order_id": "elvis-order-1",
        "venue_order_id": "venue-order-1",
        "cancel_request_id": "cancel-1",
        "observed_at": LATER,
    }
    values.update(overrides)
    return CancellationConfirmed(**values)


def make_cancel_rejected(**overrides: object) -> CancellationRejected:
    values = {
        "client_order_id": "elvis-order-1",
        "venue_order_id": "venue-order-1",
        "cancel_request_id": "cancel-1",
        "reason": "already completed",
        "observed_at": LATER,
    }
    values.update(overrides)
    return CancellationRejected(**values)


def apply(*events: object) -> OrderLifecycle:
    lifecycle = new_order_lifecycle(make_intent())
    for event in events:
        lifecycle = reduce_order_lifecycle(lifecycle, event)
    return lifecycle


def lifecycle_for(state: OrderLifecycleState) -> OrderLifecycle:
    if state is OrderLifecycleState.PENDING:
        return apply()
    if state is OrderLifecycleState.RECONCILING:
        return apply(make_ambiguous(venue_order_id="venue-order-1"))
    if state is OrderLifecycleState.OPEN:
        return apply(make_ack())
    if state is OrderLifecycleState.PARTIAL:
        return apply(make_ack(), make_fill())
    if state is OrderLifecycleState.CANCEL_PENDING:
        return apply(make_ack(), make_cancel_request())
    if state is OrderLifecycleState.CANCELLED:
        return apply(make_ack(), make_cancel_request(), make_cancel_confirmed())
    if state is OrderLifecycleState.FILLED:
        return apply(make_ack(), make_fill(quantity=Decimal("1.0")))
    if state is OrderLifecycleState.FAILED:
        return apply(make_failed())
    raise AssertionError(f"unhandled state {state}")


def test_new_lifecycle_is_immutable_pending_and_exact() -> None:
    lifecycle = new_order_lifecycle(make_intent())

    assert lifecycle.state is OrderLifecycleState.PENDING
    assert lifecycle.venue_order_id is None
    assert lifecycle.fills == ()
    assert lifecycle.pending_cancel_request_id is None
    assert lifecycle.filled_quantity == Decimal("0")
    assert lifecycle.remaining_quantity == Decimal("1.0")
    assert isinstance(hash(lifecycle), int)
    with pytest.raises(FrozenInstanceError):
        lifecycle.state = OrderLifecycleState.OPEN  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, {}, "intent"])
def test_new_lifecycle_requires_a_typed_intent(value: object) -> None:
    with pytest.raises(TypeError):
        new_order_lifecycle(value)  # type: ignore[arg-type]


def test_acknowledgement_is_not_a_fill_even_when_legacy_status_says_filled() -> None:
    report = SubmissionReport(
        client_order_id="elvis-order-1",
        status=SubmissionStatus.SUBMITTED,
        retry_safety=RetrySafety.UNSAFE,
        venue_order_id="venue-order-1",
        venue_status="FILLED",
    )

    event = submission_event_from_report(report, LATER)
    lifecycle = reduce_order_lifecycle(new_order_lifecycle(make_intent()), event)

    assert isinstance(event, SubmissionAcknowledged)
    assert lifecycle.state is OrderLifecycleState.OPEN
    assert lifecycle.filled_quantity == Decimal("0")
    assert lifecycle.fills == ()


@pytest.mark.parametrize(
    ("status", "expected_type", "expected_state"),
    [
        (
            SubmissionStatus.AMBIGUOUS,
            SubmissionAmbiguous,
            OrderLifecycleState.RECONCILING,
        ),
        (SubmissionStatus.NOT_SENT, SubmissionFailed, OrderLifecycleState.FAILED),
        (
            SubmissionStatus.VENUE_REJECTED,
            SubmissionFailed,
            OrderLifecycleState.FAILED,
        ),
    ],
)
def test_submission_report_mapping_preserves_outcome_semantics(
    status: SubmissionStatus,
    expected_type: type[object],
    expected_state: OrderLifecycleState,
) -> None:
    report = SubmissionReport(
        client_order_id="elvis-order-1",
        status=status,
        retry_safety=(
            RetrySafety.UNSAFE
            if status is SubmissionStatus.AMBIGUOUS
            else RetrySafety.SAFE
        ),
        reason="known submission outcome",
        venue_order_id=(
            "venue-order-1" if status is SubmissionStatus.AMBIGUOUS else None
        ),
        venue_status=(
            "REJECTED" if status is SubmissionStatus.VENUE_REJECTED else None
        ),
    )

    event = submission_event_from_report(report, LATER)
    lifecycle = reduce_order_lifecycle(new_order_lifecycle(make_intent()), event)

    assert type(event) is expected_type
    assert lifecycle.state is expected_state
    assert lifecycle.venue_order_id == (
        "venue-order-1" if status is SubmissionStatus.AMBIGUOUS else None
    )
    if isinstance(event, SubmissionFailed):
        assert event.status is status
        assert event.retry_safety is RetrySafety.SAFE


def test_failed_submission_event_preserves_retry_safety_orthogonally() -> None:
    not_sent_unsafe = submission_event_from_report(
        SubmissionReport(
            client_order_id="elvis-order-1",
            status=SubmissionStatus.NOT_SENT,
            retry_safety=RetrySafety.UNSAFE,
            reason="same reason",
        ),
        LATER,
    )
    rejected_safe = submission_event_from_report(
        SubmissionReport(
            client_order_id="elvis-order-1",
            status=SubmissionStatus.VENUE_REJECTED,
            retry_safety=RetrySafety.SAFE,
            reason="same reason",
        ),
        LATER,
    )

    assert isinstance(not_sent_unsafe, SubmissionFailed)
    assert isinstance(rejected_safe, SubmissionFailed)
    assert not_sent_unsafe != rejected_safe
    assert not_sent_unsafe.status is SubmissionStatus.NOT_SENT
    assert not_sent_unsafe.retry_safety is RetrySafety.UNSAFE
    assert rejected_safe.status is SubmissionStatus.VENUE_REJECTED
    assert rejected_safe.retry_safety is RetrySafety.SAFE


def test_submission_mapping_requires_typed_report_and_aware_time() -> None:
    with pytest.raises(TypeError):
        submission_event_from_report({}, LATER)  # type: ignore[arg-type]

    report = SubmissionReport(
        client_order_id="elvis-order-1",
        status=SubmissionStatus.SUBMITTED,
        retry_safety=RetrySafety.UNSAFE,
        venue_order_id="venue-order-1",
    )
    with pytest.raises(ValueError):
        submission_event_from_report(report, datetime(2026, 8, 11, 12, 0))


def test_fill_before_ack_is_kept_and_late_ack_does_not_regress_state() -> None:
    fill = make_fill()
    filled_first = apply(fill)

    acknowledged_later = reduce_order_lifecycle(filled_first, make_ack())

    assert filled_first.state is OrderLifecycleState.PARTIAL
    assert acknowledged_later is filled_first
    assert acknowledged_later.fills == (fill,)


def test_partial_then_full_fill_uses_exact_decimal_arithmetic() -> None:
    partial = apply(make_ack(), make_fill(quantity=Decimal("0.4")))
    completed = reduce_order_lifecycle(
        partial,
        make_fill(trade_id="trade-2", quantity=Decimal("0.6")),
    )

    assert partial.state is OrderLifecycleState.PARTIAL
    assert partial.filled_quantity == Decimal("0.4")
    assert partial.remaining_quantity == Decimal("0.6")
    assert completed.state is OrderLifecycleState.FILLED
    assert completed.filled_quantity == Decimal("1.0")
    assert completed.remaining_quantity == Decimal("0.0")


def test_decimal_arithmetic_does_not_round_a_large_exact_fill() -> None:
    quantity = Decimal("1000000000000000000000000000001")
    intent = make_intent(quantity=quantity)
    lifecycle = new_order_lifecycle(intent)
    lifecycle = reduce_order_lifecycle(
        lifecycle,
        make_ack(),
    )
    lifecycle = reduce_order_lifecycle(
        lifecycle,
        make_fill(
            trade_id="trade-a",
            quantity=Decimal("1000000000000000000000000000000"),
        ),
    )

    completed = reduce_order_lifecycle(
        lifecycle,
        make_fill(trade_id="trade-b", quantity=Decimal("1")),
    )

    assert completed.state is OrderLifecycleState.FILLED
    assert completed.filled_quantity == quantity
    assert completed.remaining_quantity == Decimal("0")


def test_decimal_arithmetic_rejects_a_large_overfill_without_context_drift() -> None:
    quantity = Decimal("1000000000000000000000000000001")

    def attempt(precision: int) -> None:
        with localcontext() as context:
            context.prec = precision
            lifecycle = new_order_lifecycle(make_intent(quantity=quantity))
            lifecycle = reduce_order_lifecycle(lifecycle, make_ack())
            lifecycle = reduce_order_lifecycle(
                lifecycle,
                make_fill(
                    trade_id="trade-a",
                    quantity=Decimal("1000000000000000000000000000000"),
                ),
            )
            with pytest.raises(InvalidOrderTransition):
                reduce_order_lifecycle(
                    lifecycle,
                    make_fill(trade_id="trade-b", quantity=Decimal("2")),
                )
            assert lifecycle.filled_quantity == Decimal(
                "1000000000000000000000000000000"
            )
            assert lifecycle.remaining_quantity == Decimal("1")

    attempt(4)
    attempt(80)


def test_decimal_arithmetic_ignores_ambient_rounding_and_traps() -> None:
    quantity = Decimal("246281948219935181000001")
    with localcontext() as ambient:
        ambient.prec = 4
        ambient.rounding = ROUND_DOWN
        ambient.traps[Rounded] = True
        lifecycle = new_order_lifecycle(make_intent(quantity=quantity))
        lifecycle = reduce_order_lifecycle(lifecycle, make_ack())
        lifecycle = reduce_order_lifecycle(
            lifecycle,
            make_fill(
                trade_id="trade-a",
                quantity=Decimal("2.46281948219935181E+23"),
            ),
        )
        lifecycle = reduce_order_lifecycle(
            lifecycle,
            make_fill(trade_id="trade-b", quantity=Decimal("1")),
        )

    assert lifecycle.state is OrderLifecycleState.FILLED
    assert lifecycle.filled_quantity == quantity
    assert lifecycle.remaining_quantity == Decimal("0")


@pytest.mark.parametrize("precision", [4, 80])
def test_unfilled_large_exponent_quantity_has_an_exact_remainder(
    precision: int,
) -> None:
    quantity = Decimal("1E+10001")
    with localcontext() as ambient:
        ambient.prec = precision
        lifecycle = new_order_lifecycle(make_intent(quantity=quantity))

        assert lifecycle.filled_quantity == Decimal("0")
        assert lifecycle.remaining_quantity == quantity


def test_lifecycle_rejects_a_quantity_outside_exact_arithmetic_bounds() -> None:
    intent = make_intent(quantity=Decimal("1E-1000000000"))

    with pytest.raises(
        ValueError,
        match="exact Decimal arithmetic exceeds the supported exponent",
    ):
        new_order_lifecycle(intent)


def test_exact_duplicate_fill_is_an_identity_noop() -> None:
    fill = make_fill()
    lifecycle = apply(make_ack(), fill)

    duplicate = reduce_order_lifecycle(lifecycle, fill)

    assert duplicate is lifecycle


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantity", Decimal("0.5")),
        ("price", Decimal("50011")),
        ("fee_amount", Decimal("0.3")),
        ("fee_asset", "BNB"),
        ("executed_at", LATER + timedelta(seconds=1)),
        ("client_order_id", "other-order"),
        ("venue_order_id", "other-venue"),
        ("symbol", "BNBUSDT"),
        ("side", OrderSide.SELL),
    ],
)
def test_same_trade_id_with_different_payload_is_a_conflict(
    field: str, value: object
) -> None:
    lifecycle = apply(make_ack(), make_fill())

    with pytest.raises(InvalidOrderTransition):
        reduce_order_lifecycle(lifecycle, make_fill(**{field: value}))


def test_overfill_is_rejected_without_mutating_source_state() -> None:
    partial = apply(make_ack(), make_fill(quantity=Decimal("0.6")))

    with pytest.raises(InvalidOrderTransition):
        reduce_order_lifecycle(
            partial,
            make_fill(trade_id="trade-2", quantity=Decimal("0.4000000001")),
        )

    assert partial.state is OrderLifecycleState.PARTIAL
    assert partial.filled_quantity == Decimal("0.6")
    assert len(partial.fills) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_order_id", "other-order"),
        ("venue_order_id", "other-venue"),
        ("symbol", "BNBUSDT"),
        ("side", OrderSide.SELL),
    ],
)
def test_fill_correlation_mismatch_is_rejected(field: str, value: object) -> None:
    lifecycle = apply(make_ack())

    with pytest.raises(InvalidOrderTransition):
        reduce_order_lifecycle(lifecycle, make_fill(**{field: value}))


def test_fills_are_canonical_across_arrival_order() -> None:
    fill_a = make_fill(
        trade_id="trade-a",
        quantity=Decimal("0.4"),
        executed_at=LATER + timedelta(seconds=10),
    )
    fill_b = make_fill(
        trade_id="trade-b",
        quantity=Decimal("0.6"),
        executed_at=LATER - timedelta(seconds=10),
    )

    a_then_b = apply(make_ack(), fill_a, fill_b)
    b_then_a = apply(make_ack(), fill_b, fill_a)

    assert a_then_b == b_then_a
    assert tuple(fill.trade_id for fill in a_then_b.fills) == (
        "trade-a",
        "trade-b",
    )
    assert a_then_b.state is OrderLifecycleState.FILLED


def test_ack_and_fill_arrival_order_converges() -> None:
    fill = make_fill()

    acknowledged_then_filled = apply(make_ack(), fill)
    filled_then_acknowledged = apply(fill, make_ack())

    assert acknowledged_then_filled == filled_then_acknowledged


@pytest.mark.parametrize("quantity", [Decimal("0.4"), Decimal("1.0")])
def test_cancel_request_and_fill_arrival_order_converges(quantity: Decimal) -> None:
    fill = make_fill(quantity=quantity)

    request_then_fill = apply(make_ack(), make_cancel_request(), fill)
    fill_then_request = apply(make_ack(), fill, make_cancel_request())

    assert request_then_fill == fill_then_request


@pytest.mark.parametrize("quantity", [Decimal("0.4"), Decimal("1.0")])
def test_cancel_rejection_and_fill_arrival_order_converges(quantity: Decimal) -> None:
    fill = make_fill(quantity=quantity)

    rejected_then_fill = apply(
        make_ack(),
        make_cancel_request(),
        make_cancel_rejected(),
        fill,
    )
    fill_then_rejected = apply(
        make_ack(),
        make_cancel_request(),
        fill,
        make_cancel_rejected(),
    )

    assert rejected_then_fill == fill_then_rejected


def test_cancel_rejection_restores_open_or_partial_state() -> None:
    open_pending = apply(make_ack(), make_cancel_request())
    partial_pending = apply(make_ack(), make_fill(), make_cancel_request())

    restored_open = reduce_order_lifecycle(open_pending, make_cancel_rejected())
    restored_partial = reduce_order_lifecycle(partial_pending, make_cancel_rejected())

    assert restored_open.state is OrderLifecycleState.OPEN
    assert restored_partial.state is OrderLifecycleState.PARTIAL
    assert restored_partial.filled_quantity == Decimal("0.4")


def test_stale_cancel_rejection_cannot_clear_a_new_pending_request() -> None:
    first_pending = apply(
        make_ack(),
        make_cancel_request(cancel_request_id="cancel-a"),
    )
    first_rejected = reduce_order_lifecycle(
        first_pending,
        make_cancel_rejected(cancel_request_id="cancel-a"),
    )
    second_pending = reduce_order_lifecycle(
        first_rejected,
        make_cancel_request(cancel_request_id="cancel-b"),
    )

    with pytest.raises(InvalidOrderTransition):
        reduce_order_lifecycle(
            second_pending,
            make_cancel_rejected(cancel_request_id="cancel-a"),
        )

    assert second_pending.state is OrderLifecycleState.CANCEL_PENDING
    assert second_pending.pending_cancel_request_id == "cancel-b"


@pytest.mark.parametrize("outcome", ["request", "confirmed"])
def test_stale_cancel_event_cannot_replace_a_pending_request(outcome: str) -> None:
    pending = apply(
        make_ack(),
        make_cancel_request(cancel_request_id="cancel-b"),
    )
    event = (
        make_cancel_request(cancel_request_id="cancel-a")
        if outcome == "request"
        else make_cancel_confirmed(cancel_request_id="cancel-a")
    )

    with pytest.raises(InvalidOrderTransition):
        reduce_order_lifecycle(pending, event)

    assert pending.state is OrderLifecycleState.CANCEL_PENDING
    assert pending.pending_cancel_request_id == "cancel-b"


def test_partial_fill_during_cancel_keeps_cancel_pending() -> None:
    cancel_pending = apply(make_ack(), make_cancel_request())

    partial = reduce_order_lifecycle(cancel_pending, make_fill())

    assert partial.state is OrderLifecycleState.CANCEL_PENDING
    assert partial.filled_quantity == Decimal("0.4")


def test_late_fills_after_cancel_are_counted_and_full_fill_wins() -> None:
    cancelled = apply(make_ack(), make_cancel_request(), make_cancel_confirmed())

    late_partial = reduce_order_lifecycle(cancelled, make_fill())
    late_full = reduce_order_lifecycle(
        late_partial,
        make_fill(trade_id="trade-2", quantity=Decimal("0.6")),
    )

    assert late_partial.state is OrderLifecycleState.CANCELLED
    assert late_partial.filled_quantity == Decimal("0.4")
    assert late_full.state is OrderLifecycleState.FILLED
    assert late_full.filled_quantity == Decimal("1.0")


@pytest.mark.parametrize("quantity", [Decimal("0.4"), Decimal("1.0")])
def test_cancel_confirmation_and_fill_order_converge(quantity: Decimal) -> None:
    fill = make_fill(quantity=quantity)

    cancelled_then_fill = apply(
        make_ack(), make_cancel_request(), make_cancel_confirmed(), fill
    )
    fill_then_cancelled = apply(
        make_ack(), make_cancel_request(), fill, make_cancel_confirmed()
    )

    assert cancelled_then_fill == fill_then_cancelled
    expected = (
        OrderLifecycleState.FILLED
        if quantity == Decimal("1.0")
        else OrderLifecycleState.CANCELLED
    )
    assert cancelled_then_fill.state is expected


def test_failed_order_never_accepts_a_fill_or_ack() -> None:
    failed = apply(make_failed())

    with pytest.raises(InvalidOrderTransition):
        reduce_order_lifecycle(failed, make_fill(quantity=Decimal("1.0")))
    with pytest.raises(InvalidOrderTransition):
        reduce_order_lifecycle(failed, make_ack())


NON_FILL_TRANSITIONS = {
    OrderLifecycleState.PENDING: {
        "ack": OrderLifecycleState.OPEN,
        "ambiguous": OrderLifecycleState.RECONCILING,
        "failed": OrderLifecycleState.FAILED,
        "request": None,
        "confirmed": None,
        "rejected": None,
    },
    OrderLifecycleState.RECONCILING: {
        "ack": OrderLifecycleState.OPEN,
        "ambiguous": OrderLifecycleState.RECONCILING,
        "failed": OrderLifecycleState.FAILED,
        "request": None,
        "confirmed": None,
        "rejected": None,
    },
    OrderLifecycleState.OPEN: {
        "ack": OrderLifecycleState.OPEN,
        "ambiguous": OrderLifecycleState.OPEN,
        "failed": None,
        "request": OrderLifecycleState.CANCEL_PENDING,
        "confirmed": None,
        "rejected": None,
    },
    OrderLifecycleState.PARTIAL: {
        "ack": OrderLifecycleState.PARTIAL,
        "ambiguous": OrderLifecycleState.PARTIAL,
        "failed": None,
        "request": OrderLifecycleState.CANCEL_PENDING,
        "confirmed": None,
        "rejected": None,
    },
    OrderLifecycleState.CANCEL_PENDING: {
        "ack": OrderLifecycleState.CANCEL_PENDING,
        "ambiguous": OrderLifecycleState.CANCEL_PENDING,
        "failed": None,
        "request": OrderLifecycleState.CANCEL_PENDING,
        "confirmed": OrderLifecycleState.CANCELLED,
        "rejected": OrderLifecycleState.OPEN,
    },
    OrderLifecycleState.CANCELLED: {
        "ack": OrderLifecycleState.CANCELLED,
        "ambiguous": OrderLifecycleState.CANCELLED,
        "failed": None,
        "request": OrderLifecycleState.CANCELLED,
        "confirmed": OrderLifecycleState.CANCELLED,
        "rejected": OrderLifecycleState.CANCELLED,
    },
    OrderLifecycleState.FILLED: {
        "ack": OrderLifecycleState.FILLED,
        "ambiguous": OrderLifecycleState.FILLED,
        "failed": None,
        "request": OrderLifecycleState.FILLED,
        "confirmed": OrderLifecycleState.FILLED,
        "rejected": OrderLifecycleState.FILLED,
    },
    OrderLifecycleState.FAILED: {
        "ack": None,
        "ambiguous": OrderLifecycleState.FAILED,
        "failed": OrderLifecycleState.FAILED,
        "request": None,
        "confirmed": None,
        "rejected": None,
    },
}

EVENT_FACTORIES = {
    "ack": make_ack,
    "ambiguous": lambda: make_ambiguous(venue_order_id="venue-order-1"),
    "failed": make_failed,
    "request": make_cancel_request,
    "confirmed": make_cancel_confirmed,
    "rejected": make_cancel_rejected,
}


@pytest.mark.parametrize(
    ("initial_state", "event_name", "expected_state"),
    [
        (initial_state, event_name, expected_state)
        for initial_state, transitions in NON_FILL_TRANSITIONS.items()
        for event_name, expected_state in transitions.items()
    ],
)
def test_non_fill_transition_matrix(
    initial_state: OrderLifecycleState,
    event_name: str,
    expected_state: OrderLifecycleState | None,
) -> None:
    lifecycle = lifecycle_for(initial_state)
    event = EVENT_FACTORIES[event_name]()

    if expected_state is None:
        with pytest.raises(InvalidOrderTransition):
            reduce_order_lifecycle(lifecycle, event)
        return

    result = reduce_order_lifecycle(lifecycle, event)
    assert result.state is expected_state


@pytest.mark.parametrize(
    ("field", "value", "expected_exception"),
    [
        ("trade_id", "", ValueError),
        ("symbol", " padded ", ValueError),
        ("side", "BUY", TypeError),
        ("quantity", Decimal("0"), ValueError),
        ("quantity", Decimal("NaN"), ValueError),
        ("quantity", 1, TypeError),
        ("price", Decimal("Infinity"), ValueError),
        ("price", True, TypeError),
        ("fee_amount", Decimal("-0.01"), ValueError),
        ("fee_amount", Decimal("sNaN"), ValueError),
        ("fee_amount", 0.0, TypeError),
        ("fee_asset", "", ValueError),
        ("executed_at", datetime(2026, 8, 11, 12, 0), ValueError),
    ],
)
def test_confirmed_fill_rejects_invalid_values(
    field: str, value: object, expected_exception: type[Exception]
) -> None:
    with pytest.raises(expected_exception):
        make_fill(**{field: value})


def test_positive_fee_requires_an_asset_but_zero_fee_does_not() -> None:
    with pytest.raises(ValueError):
        make_fill(fee_amount=Decimal("0.1"), fee_asset=None)

    fill = make_fill(fee_amount=Decimal("0"), fee_asset=None)
    assert fill.fee_asset is None


@pytest.mark.parametrize(
    "factory",
    [
        make_ack,
        make_ambiguous,
        make_failed,
        make_cancel_request,
        make_cancel_confirmed,
        make_cancel_rejected,
    ],
)
def test_lifecycle_events_require_aware_timestamps(factory: object) -> None:
    timestamp_field = (
        "requested_at" if factory is make_cancel_request else "observed_at"
    )
    kwargs = {timestamp_field: datetime(2026, 8, 11, 12, 0)}
    with pytest.raises(ValueError):
        factory(**kwargs)  # type: ignore[operator]


@pytest.mark.parametrize("value", ["", "   ", " padded ", 1, None])
def test_lifecycle_events_reject_invalid_client_ids(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_ack(client_order_id=value)


def test_failed_submission_event_requires_typed_non_submission_status() -> None:
    with pytest.raises(TypeError):
        make_failed(status="NOT_SENT")
    with pytest.raises(ValueError):
        make_failed(status=SubmissionStatus.SUBMITTED)
    with pytest.raises(TypeError):
        make_failed(retry_safety="SAFE")


@pytest.mark.parametrize(
    ("state", "venue_order_id", "fills"),
    [
        (OrderLifecycleState.PENDING, "venue-order-1", ()),
        (OrderLifecycleState.OPEN, None, ()),
        (OrderLifecycleState.OPEN, "venue-order-1", (make_fill(),)),
        (OrderLifecycleState.PARTIAL, "venue-order-1", ()),
        (
            OrderLifecycleState.PARTIAL,
            "venue-order-1",
            (make_fill(quantity=Decimal("1.0")),),
        ),
        (
            OrderLifecycleState.FILLED,
            "venue-order-1",
            (make_fill(quantity=Decimal("0.4")),),
        ),
        (OrderLifecycleState.FAILED, "venue-order-1", ()),
        (OrderLifecycleState.FAILED, None, (make_fill(),)),
        (OrderLifecycleState.RECONCILING, "venue-order-1", (make_fill(),)),
        (
            OrderLifecycleState.CANCEL_PENDING,
            "venue-order-1",
            (make_fill(quantity=Decimal("1.0")),),
        ),
        (
            OrderLifecycleState.CANCELLED,
            "venue-order-1",
            (make_fill(quantity=Decimal("1.0")),),
        ),
    ],
)
def test_direct_lifecycle_construction_rejects_impossible_states(
    state: OrderLifecycleState,
    venue_order_id: str | None,
    fills: tuple[ConfirmedFill, ...],
) -> None:
    with pytest.raises(ValueError):
        OrderLifecycle(
            intent=make_intent(),
            state=state,
            venue_order_id=venue_order_id,
            fills=fills,
        )


def test_direct_lifecycle_requires_canonical_unique_correlated_fills() -> None:
    fill_a = make_fill(trade_id="trade-a", quantity=Decimal("0.2"))
    fill_b = make_fill(trade_id="trade-b", quantity=Decimal("0.2"))

    with pytest.raises(ValueError):
        OrderLifecycle(
            make_intent(),
            OrderLifecycleState.PARTIAL,
            "venue-order-1",
            (fill_b, fill_a),
        )
    with pytest.raises(ValueError):
        OrderLifecycle(
            make_intent(),
            OrderLifecycleState.PARTIAL,
            "venue-order-1",
            (fill_a, fill_a),
        )
    with pytest.raises(ValueError):
        OrderLifecycle(
            make_intent(),
            OrderLifecycleState.PARTIAL,
            "venue-order-1",
            (make_fill(symbol="BNBUSDT"),),
        )


def test_lifecycle_rejects_legacy_or_string_state_and_mutable_fills() -> None:
    with pytest.raises(TypeError):
        OrderLifecycle(make_intent(), LegacyOrderStatus.PENDING)
    with pytest.raises(TypeError):
        OrderLifecycle(make_intent(), "PENDING")
    with pytest.raises(TypeError):
        OrderLifecycle(
            make_intent(),
            OrderLifecycleState.PARTIAL,
            "venue-order-1",
            [make_fill()],  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError):
        OrderLifecycle(
            make_intent(),
            OrderLifecycleState.CANCEL_PENDING,
            "venue-order-1",
        )
    with pytest.raises(ValueError):
        OrderLifecycle(
            make_intent(),
            OrderLifecycleState.OPEN,
            "venue-order-1",
            (),
            "cancel-1",
        )


def test_reducer_rejects_unknown_event_and_mismatched_client_id() -> None:
    lifecycle = new_order_lifecycle(make_intent())

    with pytest.raises(TypeError):
        reduce_order_lifecycle(lifecycle, object())
    with pytest.raises(InvalidOrderTransition):
        reduce_order_lifecycle(
            lifecycle,
            make_ambiguous(client_order_id="other-order"),
        )


_ORDER_LIFECYCLE_SYMBOLS = {
    "CancellationConfirmed",
    "CancellationRejected",
    "CancellationRequested",
    "ConfirmedFill",
    "InvalidOrderTransition",
    "OrderLifecycle",
    "OrderLifecycleEvent",
    "OrderLifecycleState",
    "SubmissionAcknowledged",
    "SubmissionAmbiguous",
    "SubmissionEvent",
    "SubmissionFailed",
    "new_order_lifecycle",
    "reduce_order_lifecycle",
    "submission_event_from_report",
}


def _literal_import_target(call: ast.Call) -> str | None:
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None
    target = call.args[0].value
    return target if isinstance(target, str) else None


def _uses_order_lifecycle_contract(source: str) -> bool:
    """Detect direct, facade, relative, and literal dynamic imports."""
    tree = ast.parse(source)
    builtins_aliases = {"builtins"}
    builtin_import_aliases = {"__import__"}
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"trading", "trading.domain"}:
                    return True
                if (
                    alias.name == "trading.domain.order_lifecycle"
                    or alias.name.startswith("trading.domain.order_lifecycle.")
                ):
                    return True
                if alias.name.startswith("trading.domain.") and alias.asname is None:
                    # A dotted import without ``as`` binds ``trading`` and can
                    # therefore reach lifecycle symbols through the facade.
                    return True
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
                if alias.name == "builtins":
                    builtins_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported_names = {alias.name for alias in node.names}
            module = node.module or ""
            if module == "importlib" and "import_module" in imported_names:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if module == "builtins" and "__import__" in imported_names:
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )
            if module == "trading" and "domain" in imported_names:
                return True
            if module == "trading.domain.order_lifecycle" or (
                node.level and module in {"order_lifecycle", "domain.order_lifecycle"}
            ):
                return True
            if module.startswith("trading.domain") or (
                node.level and module == "domain"
            ):
                if imported_names & (
                    _ORDER_LIFECYCLE_SYMBOLS | {"*", "order_lifecycle"}
                ):
                    return True
            if node.level and not module and "domain" in imported_names:
                return True

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            is_builtin_import = (
                isinstance(value, ast.Name) and value.id in builtin_import_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "__import__"
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            )
            is_import_module = (
                isinstance(value, ast.Name) and value.id in import_module_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "import_module"
                and isinstance(value.value, ast.Name)
                and value.value.id in importlib_aliases
            )
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for assigned in targets:
                if not isinstance(assigned, ast.Name):
                    continue
                if is_builtin_import and assigned.id not in builtin_import_aliases:
                    builtin_import_aliases.add(assigned.id)
                    changed = True
                if is_import_module and assigned.id not in import_module_aliases:
                    import_module_aliases.add(assigned.id)
                    changed = True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _literal_import_target(node)
        is_builtin_import = (
            isinstance(node.func, ast.Name) and node.func.id in builtin_import_aliases
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "__import__"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in builtins_aliases
        )
        if is_builtin_import and target is not None and target.startswith("trading"):
            return True
        if target not in {
            "trading",
            "trading.domain",
            "trading.domain.order_lifecycle",
        }:
            continue
        if isinstance(node.func, ast.Name) and node.func.id in (
            import_module_aliases | {"__import__"}
        ):
            return True
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_aliases
        ):
            return True

    return False


@pytest.mark.parametrize(
    "source",
    [
        "from trading.domain.order_lifecycle import OrderLifecycle",
        "from trading.domain import OrderLifecycle",
        "from trading.domain.positions import ConfirmedFill",
        "import trading.domain.order_lifecycle as lifecycle",
        "import trading.domain as domain\nvalue = domain.OrderLifecycle",
        "import trading as trading_alias\nvalue = trading_alias.domain.OrderLifecycle",
        "import trading.domain.orders\nvalue = trading.domain.OrderLifecycle",
        "from ..domain import OrderLifecycle",
        "from ..domain.order_lifecycle import OrderLifecycle",
        (
            "from importlib import import_module as load\n"
            "load('trading.domain.order_lifecycle')"
        ),
        (
            "import importlib as loader\n"
            "loader.import_module('trading.domain.order_lifecycle')"
        ),
        "from trading import domain as d\nvalue = getattr(d, 'OrderLifecycle')",
        (
            "from importlib import import_module as load\n"
            "domain = load('trading.domain')\n"
            "value = getattr(domain, 'OrderLifecycle')"
        ),
        "root = __import__('trading')\nvalue = root.domain.OrderLifecycle",
        ("__import__('trading.domain.orders')" ".domain.OrderLifecycle"),
        ("load = __import__\n" "load('trading.domain.orders').domain.OrderLifecycle"),
        (
            "import importlib as loader\n"
            "load = loader.import_module\n"
            "load('trading').domain.OrderLifecycle"
        ),
        (
            "from importlib import import_module as load\n"
            "root = load('trading')\n"
            "value = root.domain.OrderLifecycle"
        ),
    ],
)
def test_order_consumer_detector_rejects_facade_and_indirect_imports(
    source: str,
) -> None:
    assert _uses_order_lifecycle_contract(source)


@pytest.mark.parametrize(
    "source",
    [
        "from trading.domain import Position",
        "import trading.domain.positions as positions",
        "from trading.domain.positions import Position",
        "from importlib import import_module as load\nload('trading.domain.positions')",
    ],
)
def test_order_consumer_detector_allows_explicit_unrelated_domain_imports(
    source: str,
) -> None:
    assert not _uses_order_lifecycle_contract(source)


def test_order_lifecycle_only_approved_modules_consume_contract() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "domain" / "order_lifecycle.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    allowed_standard_library = {"dataclasses", "datetime", "decimal", "enum"}
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert {
        module
        for module in imports
        if module.split(".", 1)[0] not in allowed_standard_library
        and not module.startswith("trading.domain")
    } == set()

    consumers = []
    excluded = {
        module_path,
        root / "trading" / "domain" / "__init__.py",
        root / "trading" / "domain" / "positions.py",
    }
    for source_path in root.rglob("*.py"):
        if (
            source_path in excluded
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
        ):
            continue
        if _uses_order_lifecycle_contract(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))

    assert sorted(consumers) == [
        Path("trading/application/durable_submission.py"),
        Path("trading/application/journaled_order_service.py"),
        Path("trading/persistence/atomic_paper_account_owner.py"),
        Path("trading/persistence/atomic_paper_submission_owner.py"),
        Path("trading/persistence/journal_codec.py"),
        Path("trading/persistence/order_position_journal.py"),
        Path("trading/persistence/paper_account_journal.py"),
    ]
