"""Contract tests for the unwired durable-submission boundary."""

import ast
import copy
import importlib.util
import inspect
import pickle
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from trading.application.durable_submission import (
    DurableLifecycleReceipt,
    DurableSubmissionDisposition,
    DurableSubmissionOwner,
    DurableSubmissionReceipt,
    PaperPlannedFill,
    PaperSubmissionPlan,
    PaperSubmissionPlanner,
    SubmissionAttemptContext,
    SubmissionCommitUnknown,
    SubmissionReconciliationRequired,
)
from trading.domain.order_lifecycle import (
    CancellationRequested,
    ConfirmedFill,
    SubmissionAcknowledged,
    SubmissionAmbiguous,
    SubmissionFailed,
)
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionInstruction,
    TakeProfitProfile,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
UTC_UNDERFLOW = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
UTC_OVERFLOW = datetime.max.replace(tzinfo=timezone(-timedelta(hours=1)))
PUBLIC_EXPORTS = {
    "DurableLifecycleReceipt",
    "DurableSubmissionDisposition",
    "DurableSubmissionOwner",
    "DurableSubmissionReceipt",
    "PaperPlannedFill",
    "PaperSubmissionPlan",
    "PaperSubmissionPlanner",
    "SubmissionAttemptContext",
    "SubmissionCommitUnknown",
    "SubmissionReconciliationRequired",
}


def make_attempt(
    *,
    client_order_id: str = "order-1",
    instruction: PositionInstruction | None = None,
    execution_scope: str = "paper:test",
    observed_at: datetime = NOW,
) -> SubmissionAttemptContext:
    selected_instruction = instruction or make_instruction(
        client_order_id=client_order_id
    )
    return SubmissionAttemptContext.first(
        selected_instruction,
        execution_scope,
        observed_at,
    )


def make_ack(
    *,
    client_order_id: str = "order-1",
    venue_order_id: str = "venue-1",
    observed_at: datetime = NOW,
) -> SubmissionAcknowledged:
    return SubmissionAcknowledged(
        client_order_id=client_order_id,
        venue_order_id=venue_order_id,
        observed_at=observed_at,
    )


def make_ambiguous(
    *,
    client_order_id: str = "order-1",
    observed_at: datetime = NOW,
    venue_order_id: str | None = "venue-maybe",
    reason: str = "transport result is unknown",
) -> SubmissionAmbiguous:
    return SubmissionAmbiguous(
        client_order_id=client_order_id,
        reason=reason,
        observed_at=observed_at,
        venue_order_id=venue_order_id,
    )


def make_failed(
    *,
    client_order_id: str = "order-1",
    observed_at: datetime = NOW,
    status: SubmissionStatus = SubmissionStatus.NOT_SENT,
    reason: str = "submission was proven absent",
) -> SubmissionFailed:
    return SubmissionFailed(
        client_order_id=client_order_id,
        status=status,
        retry_safety=RetrySafety.SAFE,
        reason=reason,
        observed_at=observed_at,
    )


def make_fill(
    *,
    client_order_id: str = "order-1",
    venue_order_id: str = "venue-1",
    trade_id: str = "trade-1",
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("0.40"),
    executed_at: datetime = NOW + timedelta(seconds=1),
    fee_asset: str = "USDT",
) -> ConfirmedFill:
    return ConfirmedFill(
        client_order_id=client_order_id,
        venue_order_id=venue_order_id,
        trade_id=trade_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=Decimal("50001.25"),
        fee_amount=Decimal("0.25"),
        fee_asset=fee_asset,
        executed_at=executed_at,
    )


def lifecycle_receipt(
    event,
    *,
    event_id: str = "submission-attempt-1",
    position_version: int = 1,
) -> DurableLifecycleReceipt:
    return DurableLifecycleReceipt(
        event_id=event_id,
        position_version=position_version,
        event=event,
    )


def make_receipt(
    event=None,
    *,
    disposition: DurableSubmissionDisposition = (
        DurableSubmissionDisposition.COMMITTED
    ),
    attempt: SubmissionAttemptContext | None = None,
    fills: tuple[DurableLifecycleReceipt, ...] = (),
) -> DurableSubmissionReceipt:
    chosen_event = event if event is not None else make_ack()
    chosen_attempt = attempt or make_attempt(observed_at=chosen_event.observed_at)
    return DurableSubmissionReceipt(
        disposition=disposition,
        attempt=chosen_attempt,
        submission=lifecycle_receipt(chosen_event),
        fills=fills,
    )


def planned_fill(
    *,
    event_id: str = "fill-trade-1",
    fill: ConfirmedFill | None = None,
    **fill_changes,
) -> PaperPlannedFill:
    return PaperPlannedFill(
        event_id=event_id,
        fill=fill or make_fill(**fill_changes),
    )


def make_plan(
    *,
    attempt: SubmissionAttemptContext | None = None,
    submission: SubmissionAcknowledged | None = None,
    fills: tuple[PaperPlannedFill, ...] | None = None,
) -> PaperSubmissionPlan:
    chosen_attempt = attempt or make_attempt()
    chosen_submission = submission or make_ack(
        client_order_id=chosen_attempt.client_order_id,
        observed_at=chosen_attempt.observed_at,
    )
    chosen_fills = (
        fills
        if fills is not None
        else (
            planned_fill(
                client_order_id=chosen_attempt.client_order_id,
                symbol=chosen_attempt.instruction.order_intent.symbol,
                side=chosen_attempt.instruction.order_intent.side,
                quantity=chosen_attempt.instruction.order_intent.quantity,
                executed_at=chosen_attempt.observed_at + timedelta(seconds=1),
            ),
        )
    )
    return PaperSubmissionPlan(
        attempt=chosen_attempt,
        submission=chosen_submission,
        fills=chosen_fills,
    )


def make_instruction(
    *,
    client_order_id: str = "order-1",
    decision_id: str = "decision-1",
    position_key: str = "position-1",
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("1.00"),
    created_at: datetime = NOW,
) -> PositionInstruction:
    intent = OrderIntent(
        client_order_id=client_order_id,
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        reference_price=Decimal("50000.00"),
        leverage=2,
        created_at=created_at,
    )
    return PositionInstruction(
        position_key=position_key,
        effect=PositionEffect.OPEN,
        order_intent=intent,
        exit_context=PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.0025"),
            stop_loss_fraction=Decimal("0.005"),
        ),
    )


def test_first_attempt_factory_is_stable_aware_frozen_and_slotted() -> None:
    instruction = make_instruction()
    first = SubmissionAttemptContext.first(instruction, "paper:test", NOW)
    repeated = SubmissionAttemptContext.first(instruction, "paper:test", NOW)

    assert first == repeated
    assert first.event_id == "submission-attempt-1"
    assert first.instruction is instruction
    assert first.execution_scope == "paper:test"
    assert first.client_order_id == "order-1"
    assert first.observed_at is NOW
    assert not hasattr(first, "__dict__")
    with pytest.raises(FrozenInstanceError):
        first.event_id = "submission-attempt-2"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("instruction", object(), TypeError),
        ("execution_scope", "", ValueError),
        ("execution_scope", " paper:test", ValueError),
        ("execution_scope", "x" * 129, ValueError),
        ("execution_scope", "bad\x00scope", ValueError),
        ("execution_scope", "bad\ud800scope", ValueError),
        ("execution_scope", 1, TypeError),
        ("event_id", "", ValueError),
        ("event_id", "x" * 256, ValueError),
        ("event_id", False, TypeError),
    ],
)
def test_attempt_context_rejects_invalid_bounded_identifiers(
    field, value, error
) -> None:
    values = {
        "instruction": make_instruction(),
        "execution_scope": "paper:test",
        "event_id": "submission-attempt-1",
        "observed_at": NOW,
    }
    values[field] = value
    with pytest.raises(error):
        SubmissionAttemptContext(**values)


def test_attempt_context_accepts_the_storage_identifier_limits() -> None:
    context = SubmissionAttemptContext(
        instruction=make_instruction(),
        execution_scope="s" * 128,
        event_id="e" * 255,
        observed_at=NOW,
    )
    assert len(context.execution_scope) == 128
    assert len(context.event_id) == 255


@pytest.mark.parametrize(
    "instruction",
    [
        make_instruction(client_order_id="c" * 256),
        make_instruction(decision_id="d" * 256),
        make_instruction(position_key="p" * 256),
        make_instruction(symbol="S" * 65),
        make_instruction(client_order_id="bad\x00client"),
        make_instruction(decision_id="bad\ud800decision"),
    ],
)
def test_attempt_context_rejects_instruction_identifiers_unsafe_for_storage(
    instruction,
) -> None:
    with pytest.raises(ValueError):
        SubmissionAttemptContext.first(instruction, "paper:test", NOW)


@pytest.mark.parametrize("observed_at", [datetime(2026, 8, 12), "now", None])
def test_attempt_context_requires_an_aware_datetime(observed_at) -> None:
    with pytest.raises((TypeError, ValueError)):
        SubmissionAttemptContext.first(
            make_instruction(),
            "paper:test",
            observed_at,
        )


@pytest.mark.parametrize("observed_at", [UTC_UNDERFLOW, UTC_OVERFLOW])
def test_attempt_context_rejects_timestamps_that_cannot_normalize_to_utc(
    observed_at,
) -> None:
    with pytest.raises(ValueError, match="observed_at"):
        SubmissionAttemptContext.first(
            make_instruction(),
            "paper:test",
            observed_at,
        )


@pytest.mark.parametrize("created_at", [UTC_UNDERFLOW, UTC_OVERFLOW])
def test_attempt_context_rejects_instruction_timestamps_unrepresentable_in_utc(
    created_at,
) -> None:
    with pytest.raises(ValueError, match="created_at"):
        SubmissionAttemptContext.first(
            make_instruction(created_at=created_at),
            "paper:test",
            NOW,
        )


def test_attempt_context_rejects_observation_before_instruction_creation() -> None:
    with pytest.raises(ValueError, match="cannot predate"):
        make_attempt(
            instruction=make_instruction(created_at=NOW),
            observed_at=NOW - timedelta(microseconds=1),
        )


@pytest.mark.parametrize("position_version", [True, 1.0, "1", None])
def test_lifecycle_receipt_requires_an_integer_version(position_version) -> None:
    with pytest.raises(TypeError):
        lifecycle_receipt(make_ack(), position_version=position_version)


@pytest.mark.parametrize("position_version", [0, -1])
def test_lifecycle_receipt_requires_a_positive_version(position_version) -> None:
    with pytest.raises(ValueError):
        lifecycle_receipt(make_ack(), position_version=position_version)


def test_lifecycle_receipt_version_is_bounded_to_postgresql_bigint() -> None:
    maximum = (1 << 63) - 1
    assert (
        lifecycle_receipt(make_ack(), position_version=maximum).position_version
        == maximum
    )
    with pytest.raises(ValueError, match="position_version"):
        lifecycle_receipt(make_ack(), position_version=maximum + 1)


def test_lifecycle_receipt_is_frozen_slotted_and_requires_supported_exact_type() -> (
    None
):
    receipt = lifecycle_receipt(make_ack())
    assert not hasattr(receipt, "__dict__")
    with pytest.raises(FrozenInstanceError):
        receipt.position_version = 2
    with pytest.raises(TypeError):
        lifecycle_receipt(
            CancellationRequested(
                client_order_id="order-1",
                cancel_request_id="cancel-1",
                requested_at=NOW,
            )
        )


@pytest.mark.parametrize(
    "event",
    [
        make_ack(client_order_id="c" * 256),
        make_ack(venue_order_id="v" * 256),
        make_ambiguous(venue_order_id="v" * 256),
        make_fill(trade_id="t" * 256),
    ],
)
def test_lifecycle_receipt_enforces_storage_identifier_bounds(event) -> None:
    with pytest.raises(ValueError):
        lifecycle_receipt(event)


@pytest.mark.parametrize(
    "event",
    [
        make_ambiguous(reason="bad\x00reason"),
        make_ambiguous(reason="bad\ud800reason"),
        make_failed(reason="bad\x00reason"),
        make_failed(reason="bad\ud800reason"),
    ],
)
def test_lifecycle_receipt_rejects_submission_reasons_unrepresentable_in_json(
    event,
) -> None:
    with pytest.raises(ValueError, match="reason"):
        lifecycle_receipt(event)


@pytest.mark.parametrize("fee_asset", ["bad\x00asset", "bad\ud800asset"])
def test_lifecycle_receipt_rejects_fee_assets_unrepresentable_in_json(
    fee_asset,
) -> None:
    with pytest.raises(ValueError, match="fee_asset"):
        lifecycle_receipt(make_fill(fee_asset=fee_asset))


@pytest.mark.parametrize("observed_at", [UTC_UNDERFLOW, UTC_OVERFLOW])
def test_lifecycle_receipt_rejects_submission_timestamps_unrepresentable_in_utc(
    observed_at,
) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        lifecycle_receipt(make_ack(observed_at=observed_at))


@pytest.mark.parametrize("executed_at", [UTC_UNDERFLOW, UTC_OVERFLOW])
def test_lifecycle_receipt_rejects_fill_timestamps_unrepresentable_in_utc(
    executed_at,
) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        lifecycle_receipt(make_fill(executed_at=executed_at))


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            make_ack(),
            SubmissionReport(
                client_order_id="order-1",
                status=SubmissionStatus.SUBMITTED,
                retry_safety=RetrySafety.UNSAFE,
                venue_order_id="venue-1",
                venue_status=None,
            ),
        ),
        (
            make_failed(),
            SubmissionReport(
                client_order_id="order-1",
                status=SubmissionStatus.NOT_SENT,
                retry_safety=RetrySafety.SAFE,
                reason="submission was proven absent",
                venue_status=None,
            ),
        ),
        (
            make_ambiguous(),
            SubmissionReport(
                client_order_id="order-1",
                status=SubmissionStatus.AMBIGUOUS,
                retry_safety=RetrySafety.UNSAFE,
                reason="transport result is unknown",
                venue_order_id="venue-maybe",
                venue_status=None,
            ),
        ),
    ],
)
def test_submission_receipt_derives_the_canonical_report(event, expected) -> None:
    receipt = make_receipt(event)
    assert receipt.canonical_report == expected
    assert receipt.canonical_report.venue_status is None


def test_acknowledgement_receipt_accepts_consecutive_confirmed_fills() -> None:
    first = lifecycle_receipt(
        make_fill(trade_id="trade-1"),
        event_id="fill-trade-1",
        position_version=2,
    )
    second = lifecycle_receipt(
        make_fill(
            trade_id="trade-2",
            quantity=Decimal("0.60"),
            executed_at=NOW + timedelta(seconds=2),
        ),
        event_id="fill-trade-2",
        position_version=3,
    )

    receipt = make_receipt(fills=(first, second))

    assert receipt.fills == (first, second)
    assert receipt.canonical_report.acknowledged is True
    assert receipt.canonical_report.venue_status is None


@pytest.mark.parametrize("submission", [make_failed(), make_ambiguous()])
def test_only_an_acknowledgement_can_carry_confirmed_fills(submission) -> None:
    fill = lifecycle_receipt(
        make_fill(venue_order_id="venue-maybe"),
        event_id="fill-1",
        position_version=2,
    )
    with pytest.raises(ValueError, match="acknowledged"):
        make_receipt(submission, fills=(fill,))


def test_committed_receipt_requires_the_exact_attempt_timestamp() -> None:
    with pytest.raises(ValueError, match="retain the attempt timestamp"):
        make_receipt(
            make_ack(observed_at=NOW - timedelta(seconds=1)),
            attempt=make_attempt(observed_at=NOW),
        )


def test_replayed_receipt_accepts_the_exact_attempt_timestamp() -> None:
    receipt = make_receipt(
        make_ack(observed_at=NOW),
        disposition=DurableSubmissionDisposition.REPLAYED,
        attempt=make_attempt(observed_at=NOW),
    )
    assert receipt.submission.event.observed_at == NOW


@pytest.mark.parametrize(
    "durable_time",
    [NOW - timedelta(microseconds=1), NOW + timedelta(microseconds=1)],
)
def test_replayed_receipt_rejects_any_different_timestamp(durable_time) -> None:
    with pytest.raises(ValueError, match="retain the attempt timestamp"):
        make_receipt(
            make_ack(observed_at=durable_time),
            disposition=DurableSubmissionDisposition.REPLAYED,
            attempt=make_attempt(observed_at=NOW),
        )


def test_attempt_identity_changes_with_instruction_or_execution_scope() -> None:
    attempt = make_attempt()
    different_instruction = replace(
        attempt,
        instruction=make_instruction(client_order_id="order-2"),
    )
    different_scope = replace(attempt, execution_scope="paper:other")

    assert different_instruction != attempt
    assert different_instruction.client_order_id == "order-2"
    assert different_scope != attempt


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"disposition": "COMMITTED"}, TypeError),
        ({"attempt": object()}, TypeError),
        ({"submission": object()}, TypeError),
        ({"fills": []}, TypeError),
        ({"fills": (object(),)}, TypeError),
    ],
)
def test_submission_receipt_requires_exact_contract_types(changes, error) -> None:
    values = {
        "disposition": DurableSubmissionDisposition.COMMITTED,
        "attempt": make_attempt(),
        "submission": lifecycle_receipt(make_ack()),
        "fills": (),
    }
    values.update(changes)
    with pytest.raises(error):
        DurableSubmissionReceipt(**values)


def test_submission_receipt_requires_a_submission_event() -> None:
    with pytest.raises(TypeError, match="submission event"):
        DurableSubmissionReceipt(
            disposition=DurableSubmissionDisposition.COMMITTED,
            attempt=make_attempt(),
            submission=lifecycle_receipt(make_fill()),
        )


@pytest.mark.parametrize(
    ("submission", "attempt", "message"),
    [
        (
            lifecycle_receipt(make_ack(client_order_id="order-2")),
            make_attempt(),
            "client_order_id",
        ),
        (
            lifecycle_receipt(make_ack(), event_id="submission-attempt-2"),
            make_attempt(),
            "event_id",
        ),
    ],
)
def test_submission_identity_must_match_the_attempt(
    submission, attempt, message
) -> None:
    with pytest.raises(ValueError, match=message):
        DurableSubmissionReceipt(
            disposition=DurableSubmissionDisposition.COMMITTED,
            attempt=attempt,
            submission=submission,
        )


@pytest.mark.parametrize(
    ("fill", "message"),
    [
        (
            lifecycle_receipt(
                make_fill(client_order_id="order-2"),
                event_id="fill-1",
                position_version=2,
            ),
            "client_order_id",
        ),
        (
            lifecycle_receipt(
                make_fill(venue_order_id="venue-2"),
                event_id="fill-1",
                position_version=2,
            ),
            "venue order ID",
        ),
    ],
)
def test_fill_identity_must_match_the_submission(fill, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_receipt(fills=(fill,))


@pytest.mark.parametrize(
    "fills",
    [
        (
            lifecycle_receipt(
                make_fill(trade_id="trade-1"),
                event_id="submission-attempt-1",
                position_version=2,
            ),
        ),
        (
            lifecycle_receipt(
                make_fill(trade_id="trade-1"),
                event_id="fill-1",
                position_version=3,
            ),
        ),
        (
            lifecycle_receipt(
                make_fill(trade_id="trade-1"),
                event_id="fill-1",
                position_version=2,
            ),
            lifecycle_receipt(
                make_fill(trade_id="trade-2", quantity=Decimal("0.60")),
                event_id="fill-2",
                position_version=4,
            ),
        ),
    ],
)
def test_durable_event_ids_and_versions_must_be_unique_and_consecutive(
    fills,
) -> None:
    with pytest.raises(ValueError):
        make_receipt(fills=fills)


@pytest.mark.parametrize(
    "second_fill",
    [
        make_fill(trade_id="trade-2", symbol="ETHUSDT"),
        make_fill(trade_id="trade-2", side=OrderSide.SELL),
        make_fill(trade_id="trade-2", venue_order_id="venue-2"),
    ],
)
def test_fills_cannot_conflict_with_one_another(second_fill) -> None:
    fills = (
        lifecycle_receipt(
            make_fill(trade_id="trade-1"),
            event_id="fill-1",
            position_version=2,
        ),
        lifecycle_receipt(
            second_fill,
            event_id="fill-2",
            position_version=3,
        ),
    )
    with pytest.raises(ValueError):
        make_receipt(fills=fills)


def test_fill_cannot_predate_the_submission_attempt() -> None:
    fill = lifecycle_receipt(
        make_fill(executed_at=NOW - timedelta(microseconds=1)),
        event_id="fill-1",
        position_version=2,
    )
    with pytest.raises(ValueError, match="cannot predate"):
        make_receipt(fills=(fill,))


def test_fills_cannot_exceed_the_instruction_quantity() -> None:
    fill = lifecycle_receipt(
        make_fill(quantity=Decimal("1.01")),
        event_id="fill-1",
        position_version=2,
    )
    with pytest.raises(ValueError, match="exceed"):
        make_receipt(fills=(fill,))


def test_fill_trade_ids_cannot_repeat_even_with_distinct_event_ids() -> None:
    fills = (
        lifecycle_receipt(
            make_fill(trade_id="trade-1"),
            event_id="fill-observation-1",
            position_version=2,
        ),
        lifecycle_receipt(
            make_fill(
                trade_id="trade-1",
                quantity=Decimal("0.60"),
                executed_at=NOW + timedelta(seconds=2),
            ),
            event_id="fill-observation-2",
            position_version=3,
        ),
    )
    with pytest.raises(ValueError, match="trade IDs"):
        make_receipt(fills=fills)


def test_fill_receipts_must_contain_confirmed_fills() -> None:
    invalid_fill = lifecycle_receipt(
        make_ack(),
        event_id="fill-1",
        position_version=2,
    )
    with pytest.raises(TypeError, match="ConfirmedFill"):
        make_receipt(fills=(invalid_fill,))


def test_submission_receipt_is_frozen_and_slotted() -> None:
    receipt = make_receipt()
    assert not hasattr(receipt, "__dict__")
    with pytest.raises(FrozenInstanceError):
        receipt.fills = ()


def test_paper_plan_requires_one_exact_terminal_full_fill() -> None:
    attempt = make_attempt()
    fills = (
        planned_fill(quantity=Decimal("0.40")),
        planned_fill(
            event_id="fill-trade-2",
            trade_id="trade-2",
            quantity=Decimal("0.60"),
            executed_at=NOW + timedelta(seconds=2),
        ),
    )

    plan = make_plan(attempt=attempt, fills=fills)

    assert plan.attempt is attempt
    assert type(plan.submission) is SubmissionAcknowledged
    assert plan.fills == fills
    assert sum((candidate.fill.quantity for candidate in fills), Decimal(0)) == (
        attempt.instruction.order_intent.quantity
    )


@pytest.mark.parametrize(
    ("fills", "message"),
    [
        ((), "at least one fill"),
        ((planned_fill(quantity=Decimal("0.99")),), "exact full fills"),
        ((planned_fill(quantity=Decimal("1.01")),), "exceed"),
    ],
)
def test_paper_plan_rejects_empty_partial_and_overfill_batches(fills, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_plan(fills=fills)


@pytest.mark.parametrize("fills", [[], (object(),)])
def test_paper_plan_requires_an_exact_tuple_of_planned_fills(fills) -> None:
    with pytest.raises(TypeError):
        make_plan(fills=fills)


def test_paper_plan_rejects_a_tuple_subclass_with_mutable_iteration() -> None:
    class MutableTuple(tuple):
        def __iter__(self):
            return iter((planned_fill(quantity=Decimal("1.00")),))

    with pytest.raises(TypeError, match="fills must be a tuple"):
        make_plan(fills=MutableTuple())


@pytest.mark.parametrize("submission", [make_ambiguous(), make_failed()])
def test_paper_plan_requires_an_acknowledgement(submission) -> None:
    with pytest.raises(TypeError, match="submission"):
        PaperSubmissionPlan(
            attempt=make_attempt(),
            submission=submission,
            fills=(planned_fill(quantity=Decimal("1.00")),),
        )


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (
            planned_fill(client_order_id="order-other", quantity=Decimal("1.00")),
            "client_order_id",
        ),
        (planned_fill(symbol="ETHUSDT", quantity=Decimal("1.00")), "symbol"),
        (planned_fill(side=OrderSide.SELL, quantity=Decimal("1.00")), "side"),
        (
            planned_fill(
                executed_at=NOW - timedelta(microseconds=1),
                quantity=Decimal("1.00"),
            ),
            "cannot predate",
        ),
        (
            planned_fill(venue_order_id="venue-other", quantity=Decimal("1.00")),
            "venue order ID",
        ),
    ],
)
def test_paper_plan_binds_each_fill_to_the_exact_attempt_and_ack(
    candidate, message
) -> None:
    with pytest.raises(ValueError, match=message):
        make_plan(fills=(candidate,))


@pytest.mark.parametrize(
    ("submission", "message"),
    [
        (make_ack(client_order_id="order-other"), "client_order_id"),
        (make_ack(observed_at=NOW + timedelta(microseconds=1)), "timestamp"),
    ],
)
def test_paper_plan_binds_acknowledgement_to_the_exact_attempt(
    submission, message
) -> None:
    with pytest.raises(ValueError, match=message):
        make_plan(submission=submission)


def test_paper_plan_rejects_duplicate_event_and_trade_identities() -> None:
    with pytest.raises(ValueError, match="event IDs"):
        make_plan(
            fills=(
                planned_fill(
                    event_id="submission-attempt-1",
                    quantity=Decimal("1.00"),
                ),
            )
        )

    with pytest.raises(ValueError, match="trade IDs"):
        make_plan(
            fills=(
                planned_fill(quantity=Decimal("0.40")),
                planned_fill(
                    event_id="fill-observation-2",
                    quantity=Decimal("0.60"),
                ),
            )
        )


@pytest.mark.parametrize(
    ("event_id", "fill", "error"),
    [
        ("", make_fill(quantity=Decimal("1.00")), ValueError),
        ("f" * 256, make_fill(quantity=Decimal("1.00")), ValueError),
        ("fill-1", object(), TypeError),
    ],
)
def test_planned_fill_requires_a_durable_event_id_and_confirmed_fill(
    event_id, fill, error
) -> None:
    with pytest.raises(error):
        PaperPlannedFill(event_id=event_id, fill=fill)


def test_planner_protocol_receives_the_exact_attempt_positionally() -> None:
    attempt = make_attempt()
    expected = make_plan(attempt=attempt)

    class FakePlanner:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, candidate, /):
            self.calls.append(candidate)
            return expected

    planner = FakePlanner()

    assert planner.plan(attempt) is expected
    assert planner.calls == [attempt]
    with pytest.raises(TypeError):
        planner.plan(candidate=attempt)

    parameters = inspect.signature(PaperSubmissionPlanner.plan).parameters
    assert tuple(parameters) == ("self", "attempt")
    assert parameters["attempt"].kind is inspect.Parameter.POSITIONAL_ONLY


def test_paper_plan_value_graph_rejects_generated_setstate_mutation() -> None:
    plan = make_plan()
    original_identity = (repr(plan), hash(plan))
    visited: set[int] = set()
    protected_types: set[type] = set()

    def assert_protected(value: object) -> None:
        if id(value) in visited or not is_dataclass(value) or isinstance(value, type):
            return
        visited.add(id(value))
        state = [getattr(value, field.name) for field in fields(value)]
        assert hasattr(value, "__setstate__")
        protected_types.add(type(value))
        with pytest.raises(TypeError, match="state mutation"):
            value.__setstate__(state)
        for field in fields(value):
            child = getattr(value, field.name)
            if isinstance(child, tuple):
                for item in child:
                    assert_protected(item)
            else:
                assert_protected(child)

    assert_protected(plan)

    assert protected_types == {
        ConfirmedFill,
        OrderIntent,
        PaperPlannedFill,
        PaperSubmissionPlan,
        PositionExitContext,
        PositionInstruction,
        SubmissionAcknowledged,
        SubmissionAttemptContext,
    }
    assert (repr(plan), hash(plan)) == original_identity


def test_paper_plan_copy_and_pickle_round_trips_revalidate() -> None:
    plan = make_plan()

    for restored in (
        copy.copy(plan),
        copy.deepcopy(plan),
        pickle.loads(pickle.dumps(plan)),
    ):
        assert restored == plan
        assert hash(restored) == hash(plan)


def test_commit_unknown_copy_and_pickle_round_trips_retain_the_attempt() -> None:
    error = SubmissionCommitUnknown(make_attempt())

    for restored in (
        copy.copy(error),
        copy.deepcopy(error),
        pickle.loads(pickle.dumps(error)),
    ):
        assert type(restored) is SubmissionCommitUnknown
        assert restored.attempt == error.attempt
        assert restored.client_order_id == error.client_order_id
        assert restored.requires_reconciliation is True


def test_paper_plan_invalid_state_restore_cleans_partial_object() -> None:
    plan = make_plan()
    field_names = tuple(field.name for field in fields(plan))
    state = [getattr(plan, name) for name in field_names]
    state[field_names.index("fills")] = ()
    restored = object.__new__(PaperSubmissionPlan)

    with pytest.raises(ValueError, match="at least one fill"):
        restored.__setstate__(state)

    assert all(not hasattr(restored, name) for name in field_names)


def test_owner_protocol_fake_receives_the_exact_call_contract() -> None:
    expected = make_receipt()

    class FakeOwner:
        def __init__(self):
            self.calls = []

        def execute(self, attempt, /):
            self.calls.append(attempt)
            return expected

    owner = FakeOwner()
    attempt = make_attempt()

    actual = owner.execute(attempt)

    assert actual is expected
    assert owner.calls == [attempt]
    with pytest.raises(TypeError):
        owner.execute(attempt=attempt)


def test_owner_protocol_declares_the_complete_attempt_positional_only() -> None:
    parameters = inspect.signature(DurableSubmissionOwner.execute).parameters
    assert tuple(parameters) == ("self", "attempt")
    assert parameters["attempt"].kind is inspect.Parameter.POSITIONAL_ONLY


def test_commit_unknown_preserves_attempt_and_requires_reconciliation() -> None:
    attempt = make_attempt()
    failure = SubmissionCommitUnknown(attempt)

    assert isinstance(failure, RuntimeError)
    assert str(failure) == "durable submission commit outcome is unknown"
    assert failure.client_order_id == "order-1"
    assert failure.attempt is attempt
    assert failure.requires_reconciliation is True
    with pytest.raises(FrozenInstanceError):
        failure.attempt = make_attempt(client_order_id="order-2")


def test_commit_unknown_requires_an_exact_attempt_context() -> None:
    with pytest.raises(TypeError):
        SubmissionCommitUnknown(object())


def test_reconciliation_required_preserves_attempt_and_round_trips() -> None:
    attempt = make_attempt()
    failure = SubmissionReconciliationRequired(attempt)

    assert isinstance(failure, RuntimeError)
    assert failure.client_order_id == "order-1"
    assert failure.attempt is attempt
    assert failure.requires_reconciliation is True
    for restored in (
        copy.copy(failure),
        copy.deepcopy(failure),
        pickle.loads(pickle.dumps(failure)),
    ):
        assert type(restored) is SubmissionReconciliationRequired
        assert restored.attempt == attempt


def test_reconciliation_required_requires_an_exact_attempt_context() -> None:
    with pytest.raises(TypeError):
        SubmissionReconciliationRequired(object())


def _literal_dynamic_import(
    node: ast.Call,
    *,
    builtins_aliases: set[str],
    builtin_import_aliases: set[str],
    importlib_aliases: set[str],
    import_module_aliases: set[str],
) -> str | None:
    target = (
        node.args[0].value
        if node.args and isinstance(node.args[0], ast.Constant)
        else next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
            ),
            None,
        )
    )
    if not isinstance(target, str):
        return None
    built_in = (
        isinstance(node.func, ast.Name) and node.func.id in builtin_import_aliases
    ) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "__import__"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in builtins_aliases
    )
    import_module = (
        isinstance(node.func, ast.Name) and node.func.id in import_module_aliases
    ) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in importlib_aliases
    )
    if not built_in and not import_module:
        return None
    if built_in and target.startswith("trading."):
        return "trading"
    if not target.startswith("."):
        return target
    package = (
        node.args[1].value
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
        else next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "package" and isinstance(keyword.value, ast.Constant)
            ),
            None,
        )
    )
    if not isinstance(package, str):
        return None
    try:
        return importlib.util.resolve_name(target, package)
    except ImportError, ValueError:
        return None


def _uses_durable_submission(source: str) -> bool:
    """Conservatively detect direct, facade, relative, and dynamic consumers."""
    tree = ast.parse(source)
    module = "trading.application.durable_submission"
    builtins_aliases = {"builtins"}
    builtin_import_aliases = {"__import__"}
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "trading":
                    return True
                if alias.name == module or alias.name.startswith(f"{module}."):
                    return True
                if alias.name == "trading.application":
                    return True
                if alias.name.startswith("trading.application.") and not alias.asname:
                    return True
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
                if alias.name == "builtins":
                    builtins_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            imported_module = node.module or ""
            if imported_module == "importlib" and "import_module" in imported:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if imported_module == "builtins" and "__import__" in imported:
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )
            if imported_module == module:
                return True
            if node.level and imported_module.endswith("durable_submission"):
                return True
            if node.level and "durable_submission" in imported:
                return True
            if node.level and imported & PUBLIC_EXPORTS:
                return True
            if node.level and "application" in imported:
                return True
            if imported_module == "trading" and "application" in imported:
                return True
            if imported_module == "trading.application" and imported & (
                PUBLIC_EXPORTS | {"durable_submission", "*"}
            ):
                return True

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            built_in = (
                isinstance(value, ast.Name) and value.id in builtin_import_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "__import__"
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            )
            import_module = (
                isinstance(value, ast.Name) and value.id in import_module_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "import_module"
                and isinstance(value.value, ast.Name)
                and value.value.id in importlib_aliases
            )
            if not built_in and not import_module:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                aliases = builtin_import_aliases if built_in else import_module_aliases
                if target.id not in aliases:
                    aliases.add(target.id)
                    changed = True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _literal_dynamic_import(
            node,
            builtins_aliases=builtins_aliases,
            builtin_import_aliases=builtin_import_aliases,
            importlib_aliases=importlib_aliases,
            import_module_aliases=import_module_aliases,
        )
        if target == module or (target and target.startswith(f"{module}.")):
            return True
        if target in {"trading", "trading.application"}:
            return True
    return False


@pytest.mark.parametrize(
    "source",
    [
        (
            "from trading.application.durable_submission "
            "import DurableSubmissionReceipt"
        ),
        "import trading.application.durable_submission as durable",
        "from trading.application import DurableSubmissionOwner",
        "from trading.application import durable_submission",
        "import trading.application as app\napp.DurableSubmissionReceipt",
        "import trading as root\nroot.application.DurableSubmissionReceipt",
        "import trading.application.order_service\n"
        "trading.application.DurableSubmissionReceipt",
        "from trading import application as app\napp.DurableSubmissionOwner",
        "from . import application as app\napp.SubmissionAttemptContext",
        "from .durable_submission import SubmissionAttemptContext",
        "from . import durable_submission",
        "from ..application import DurableSubmissionReceipt",
        "from importlib import import_module as load\n"
        "load('trading.application.durable_submission')",
        "import importlib as loader\n"
        "loader.import_module('trading.application').DurableSubmissionOwner",
        "root = __import__('trading')\n" "root.application.DurableSubmissionReceipt",
        "from builtins import __import__ as load\n"
        "load('trading.application.durable_submission')",
        "load = __import__\nload('trading.application.durable_submission')",
        "import importlib\nload = importlib.import_module\n"
        "load('trading.application.durable_submission')",
        "from importlib import import_module\n"
        "import_module('.durable_submission', package='trading.application')",
        "import importlib\n"
        "importlib.import_module('..application.durable_submission', "
        "package='trading.execution')",
    ],
)
def test_consumer_detector_catches_supported_forms(source) -> None:
    assert _uses_durable_submission(source)


@pytest.mark.parametrize(
    "source",
    [
        "from trading.application.order_service import OrderService",
        "from trading.application import OrderService",
        "from trading.domain.orders import OrderIntent",
        "name = 'trading.application.durable_submission'",
    ],
)
def test_consumer_detector_allows_unrelated_imports(source) -> None:
    assert not _uses_durable_submission(source)


def test_durable_submission_has_one_persistence_consumer_and_stays_pure() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "application" / "durable_submission.py"
    facade_path = root / "trading" / "application" / "__init__.py"
    consumers = []
    scanned = []

    for source_path in root.rglob("*.py"):
        if (
            source_path in {module_path, facade_path}
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        scanned.append(source_path.relative_to(root))
        if _uses_durable_submission(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))

    assert sorted(consumers) == [
        Path("trading/persistence/atomic_paper_account_owner.py"),
        Path("trading/persistence/atomic_paper_submission_owner.py"),
    ]
    assert {Path("main.py"), Path("core/bootstrap.py")} <= set(scanned)

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            if node.module:
                imported_roots.add(node.module.split(".")[0])
                if node.module.startswith("trading."):
                    assert node.module.startswith("trading.domain.")

    assert imported_roots <= set(sys.stdlib_module_names) | {"trading"}


def test_application_facade_exports_the_complete_contract() -> None:
    facade_path = Path(__file__).parents[1] / "trading" / "application" / "__init__.py"
    namespace = {}
    exec(facade_path.read_text(encoding="utf-8"), namespace)
    assert PUBLIC_EXPORTS <= set(namespace["__all__"])
    assert all(export in namespace for export in PUBLIC_EXPORTS)
