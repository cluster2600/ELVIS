"""Contracts for atomically owned paper-account submissions."""

import copy
import inspect
import pickle
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading.application import (
    DurablePaperAccountSubmissionReceipt,
    PaperAccountSubmissionCommitUnknown,
    PaperAccountSubmissionContext,
    PaperAccountSubmissionOwner,
    PaperAccountSubmissionReconciliationRequired,
    PaperAccountSubmissionRejected,
    PaperAccountSubmissionResult,
)
from trading.application.durable_submission import (
    DurableLifecycleReceipt,
    DurableSubmissionDisposition,
    DurableSubmissionReceipt,
    SubmissionAttemptContext,
)
from trading.domain.order_lifecycle import ConfirmedFill, SubmissionAcknowledged
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.paper_settlement import PaperLinearInstrument
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionInstruction,
    TakeProfitProfile,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def _instruction(*, quantity: Decimal = Decimal("1.00")) -> PositionInstruction:
    return PositionInstruction(
        position_key="position-1",
        effect=PositionEffect.OPEN,
        order_intent=OrderIntent(
            client_order_id="order-1",
            decision_id="decision-1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            reference_price=Decimal("50000.00"),
            leverage=2,
            created_at=NOW,
        ),
        exit_context=PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.0025"),
            stop_loss_fraction=Decimal("0.005"),
        ),
    )


def _attempt(*, quantity: Decimal = Decimal("1.00")) -> SubmissionAttemptContext:
    return SubmissionAttemptContext.first(
        _instruction(quantity=quantity),
        "paper:test",
        NOW,
    )


def _context(
    *, attempt: SubmissionAttemptContext | None = None
) -> PaperAccountSubmissionContext:
    return PaperAccountSubmissionContext(
        attempt=attempt or _attempt(),
        account_key="account-1",
        instrument=PaperLinearInstrument("BTCUSDT", "BTC", "USDT"),
    )


def _submission(
    attempt: SubmissionAttemptContext,
    *,
    disposition: DurableSubmissionDisposition = DurableSubmissionDisposition.COMMITTED,
) -> DurableSubmissionReceipt:
    acknowledgement = SubmissionAcknowledged(
        client_order_id=attempt.client_order_id,
        venue_order_id="venue-1",
        observed_at=attempt.observed_at,
    )
    fill = ConfirmedFill(
        client_order_id=attempt.client_order_id,
        venue_order_id="venue-1",
        trade_id="trade-1",
        symbol=attempt.instruction.order_intent.symbol,
        side=attempt.instruction.order_intent.side,
        quantity=attempt.instruction.order_intent.quantity,
        price=Decimal("50001.25"),
        fee_amount=Decimal("0.25"),
        fee_asset="USDT",
        executed_at=attempt.observed_at + timedelta(seconds=1),
    )
    return DurableSubmissionReceipt(
        disposition=disposition,
        attempt=attempt,
        submission=DurableLifecycleReceipt(
            event_id=attempt.event_id,
            position_version=1,
            event=acknowledgement,
        ),
        fills=(
            DurableLifecycleReceipt(
                event_id="fill-1",
                position_version=2,
                event=fill,
            ),
        ),
    )


def test_context_retains_exact_attempt_and_instrument_snapshot() -> None:
    attempt = _attempt()
    instrument = PaperLinearInstrument("BTCUSDT", "BTC", "USDT")

    context = PaperAccountSubmissionContext(attempt, "account-1", instrument)

    assert context.attempt is attempt
    assert context.instrument is instrument
    assert context.execution_scope == "paper:test"
    assert context.client_order_id == "order-1"
    assert not hasattr(context, "__dict__")
    with pytest.raises(FrozenInstanceError):
        context.account_key = "account-2"


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"attempt": object()}, TypeError),
        ({"account_key": ""}, ValueError),
        ({"account_key": "x" * 256}, ValueError),
        ({"account_key": "bad\x00key"}, ValueError),
        ({"instrument": object()}, TypeError),
        (
            {"instrument": PaperLinearInstrument("ETHUSDT", "ETH", "USDT")},
            ValueError,
        ),
        (
            {
                "instrument": PaperLinearInstrument(
                    "BTCUSDT",
                    "B" * 65,
                    "USDT",
                )
            },
            ValueError,
        ),
    ],
)
def test_context_rejects_invalid_durable_identity(values, error) -> None:
    arguments = {
        "attempt": _attempt(),
        "account_key": "account-1",
        "instrument": PaperLinearInstrument("BTCUSDT", "BTC", "USDT"),
    }
    arguments.update(values)

    with pytest.raises(error):
        PaperAccountSubmissionContext(**arguments)


def test_receipt_retains_exact_context_and_consecutive_account_versions() -> None:
    context = _context()
    submission = _submission(context.attempt)

    receipt = DurablePaperAccountSubmissionReceipt(context, submission, (7,))

    assert receipt.context is context
    assert receipt.submission is submission
    assert receipt.account_versions == (7,)
    assert receipt.disposition is DurableSubmissionDisposition.COMMITTED


def test_receipt_rejects_numerically_equal_but_distinct_attempt_envelope() -> None:
    context = _context(attempt=_attempt(quantity=Decimal("1.00")))
    distinct_attempt = _attempt(quantity=Decimal("1.0"))
    assert distinct_attempt == context.attempt
    assert distinct_attempt is not context.attempt
    assert (
        distinct_attempt.instruction.order_intent.quantity.as_tuple()
        != context.attempt.instruction.order_intent.quantity.as_tuple()
    )

    with pytest.raises(ValueError, match="attempt"):
        DurablePaperAccountSubmissionReceipt(
            context,
            _submission(distinct_attempt),
            (1,),
        )


@pytest.mark.parametrize(
    ("versions", "error"),
    [
        ((), ValueError),
        ((0,), ValueError),
        (((1 << 63),), ValueError),
        ((True,), TypeError),
        ((1.0,), TypeError),
        ([1], TypeError),
    ],
)
def test_receipt_rejects_invalid_account_versions(versions, error) -> None:
    context = _context()
    with pytest.raises(error):
        DurablePaperAccountSubmissionReceipt(
            context,
            _submission(context.attempt),
            versions,
        )


def test_rejection_requires_one_durable_event_identity_and_reason() -> None:
    context = _context()
    rejected = PaperAccountSubmissionRejected(
        context,
        "fill-1",
        ("insufficient available balance for USDT",),
    )

    assert rejected.context is context
    assert rejected.rejected_event_id == "fill-1"
    assert rejected.reasons == ("insufficient available balance for USDT",)
    with pytest.raises(ValueError):
        PaperAccountSubmissionRejected(context, "fill-1", ())
    with pytest.raises(ValueError):
        PaperAccountSubmissionRejected(context, "", ("reason",))


@pytest.mark.parametrize(
    "error_type",
    [
        PaperAccountSubmissionCommitUnknown,
        PaperAccountSubmissionReconciliationRequired,
    ],
)
def test_reconciliation_exceptions_preserve_full_context(error_type) -> None:
    context = _context()
    error = error_type(context)

    assert error.context is context
    assert error.client_order_id == context.client_order_id
    assert error.requires_reconciliation is True
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(error, protocol=protocol))
        assert type(restored) is error_type
        assert restored.context == context


def test_account_owner_protocol_is_positional_only_and_result_union_is_public() -> None:
    parameters = inspect.signature(PaperAccountSubmissionOwner.execute).parameters
    assert parameters["context"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert PaperAccountSubmissionResult == (
        DurablePaperAccountSubmissionReceipt | PaperAccountSubmissionRejected
    )


def test_account_submission_contracts_are_exported_by_application_facade() -> None:
    import trading.application as application

    expected = {
        "DurablePaperAccountSubmissionReceipt",
        "PaperAccountSubmissionCommitUnknown",
        "PaperAccountSubmissionContext",
        "PaperAccountSubmissionOwner",
        "PaperAccountSubmissionReconciliationRequired",
        "PaperAccountSubmissionRejected",
        "PaperAccountSubmissionResult",
    }
    assert expected <= set(application.__all__)
    assert all(getattr(application, name) is globals()[name] for name in expected)


def test_contracts_reject_copy_protocol_state_mutation() -> None:
    context = _context()
    receipt = DurablePaperAccountSubmissionReceipt(
        context,
        _submission(context.attempt),
        (1,),
    )
    rejection = PaperAccountSubmissionRejected(context, "fill-1", ("reason",))

    for value in (context, receipt, rejection):
        restored = copy.copy(value)
        assert restored == value
        assert hash(restored) == hash(value)
        with pytest.raises(TypeError):
            value.__setstate__((None,))
