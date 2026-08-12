"""Pure contracts for one atomically durable submission attempt."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from trading.domain._decimal import exact_decimal_sum
from trading.domain.order_lifecycle import (
    ConfirmedFill,
    SubmissionAcknowledged,
    SubmissionAmbiguous,
    SubmissionEvent,
    SubmissionFailed,
)
from trading.domain.orders import RetrySafety, SubmissionReport, SubmissionStatus
from trading.domain.positions import PositionInstruction

_FIRST_SUBMISSION_EVENT_ID = "submission-attempt-1"
_IDENTIFIER_MAX_LENGTH = 255
_EXECUTION_SCOPE_MAX_LENGTH = 128
_SYMBOL_MAX_LENGTH = 64
_POSTGRES_BIGINT_MAX = (1 << 63) - 1
_SUBMISSION_EVENT_TYPES = (
    SubmissionAcknowledged,
    SubmissionAmbiguous,
    SubmissionFailed,
)
_DURABLE_EVENT_TYPES = _SUBMISSION_EVENT_TYPES + (ConfirmedFill,)


def _require_identifier(
    name: str,
    value: object,
    max_length: int = _IDENTIFIER_MAX_LENGTH,
) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")
    if len(value) > max_length:
        raise ValueError(f"{name} must contain at most {max_length} characters")
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError(f"{name} contains unsupported characters")


def _require_payload_text(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError(f"{name} is not representable in durable JSON")


def _require_durable_datetime(name: str, value: object) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    try:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")
        value.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be represented in UTC") from exc


def _validate_event_identifiers(event: SubmissionEvent | ConfirmedFill) -> None:
    _require_identifier("event.client_order_id", event.client_order_id)
    if isinstance(event, (SubmissionAcknowledged, ConfirmedFill)):
        _require_identifier("event.venue_order_id", event.venue_order_id)
    elif isinstance(event, SubmissionAmbiguous) and event.venue_order_id is not None:
        _require_identifier("event.venue_order_id", event.venue_order_id)
    if isinstance(event, (SubmissionAmbiguous, SubmissionFailed)):
        _require_payload_text("event.reason", event.reason)
    if isinstance(event, ConfirmedFill):
        _require_identifier("event.trade_id", event.trade_id)
        _require_identifier("event.symbol", event.symbol, _SYMBOL_MAX_LENGTH)
        if event.fee_asset is not None:
            _require_payload_text("event.fee_asset", event.fee_asset)


def _event_timestamp(event: SubmissionEvent | ConfirmedFill) -> datetime:
    if isinstance(event, ConfirmedFill):
        return event.executed_at
    return event.observed_at


@dataclass(frozen=True, slots=True)
class SubmissionAttemptContext:
    """Stable caller-selected identity and timestamp for one submission attempt."""

    instruction: PositionInstruction
    execution_scope: str
    event_id: str
    observed_at: datetime

    def __post_init__(self) -> None:
        if type(self.instruction) is not PositionInstruction:
            raise TypeError("instruction must be a PositionInstruction")
        _require_identifier(
            "execution_scope",
            self.execution_scope,
            _EXECUTION_SCOPE_MAX_LENGTH,
        )
        _require_identifier("event_id", self.event_id)
        _require_durable_datetime("observed_at", self.observed_at)

        intent = self.instruction.order_intent
        _require_identifier("position_key", self.instruction.position_key)
        _require_identifier("client_order_id", intent.client_order_id)
        _require_identifier("decision_id", intent.decision_id)
        _require_identifier("symbol", intent.symbol, _SYMBOL_MAX_LENGTH)
        _require_durable_datetime("instruction.created_at", intent.created_at)

    @property
    def client_order_id(self) -> str:
        """Derive the stable client identity from the complete instruction."""
        return self.instruction.order_intent.client_order_id

    @classmethod
    def first(
        cls,
        instruction: PositionInstruction,
        execution_scope: str,
        observed_at: datetime,
    ) -> "SubmissionAttemptContext":
        """Build the fixed identity for the first supported submission attempt."""
        return cls(
            instruction=instruction,
            execution_scope=execution_scope,
            event_id=_FIRST_SUBMISSION_EVENT_ID,
            observed_at=observed_at,
        )


@dataclass(frozen=True, slots=True)
class DurableLifecycleReceipt:
    """One lifecycle fact whose identity and stream version are committed."""

    event_id: str
    position_version: int
    event: SubmissionEvent | ConfirmedFill

    def __post_init__(self) -> None:
        _require_identifier("event_id", self.event_id)
        if isinstance(self.position_version, bool) or not isinstance(
            self.position_version, int
        ):
            raise TypeError("position_version must be an integer")
        if self.position_version < 1:
            raise ValueError("position_version must be positive")
        if self.position_version > _POSTGRES_BIGINT_MAX:
            raise ValueError("position_version exceeds the durable storage limit")
        if type(self.event) not in _DURABLE_EVENT_TYPES:
            raise TypeError("event must be a submission event or ConfirmedFill")
        _validate_event_identifiers(self.event)
        _require_durable_datetime("event timestamp", _event_timestamp(self.event))


class DurableSubmissionDisposition(str, Enum):
    """Whether this call committed new facts or replayed existing facts."""

    COMMITTED = "COMMITTED"
    REPLAYED = "REPLAYED"


@dataclass(frozen=True, slots=True)
class DurableSubmissionReceipt:
    """A complete, validated durable result for one submission attempt."""

    disposition: DurableSubmissionDisposition
    attempt: SubmissionAttemptContext
    submission: DurableLifecycleReceipt
    fills: tuple[DurableLifecycleReceipt, ...] = ()

    def __post_init__(self) -> None:
        if type(self.disposition) is not DurableSubmissionDisposition:
            raise TypeError("disposition must be a DurableSubmissionDisposition")
        if type(self.attempt) is not SubmissionAttemptContext:
            raise TypeError("attempt must be a SubmissionAttemptContext")
        if type(self.submission) is not DurableLifecycleReceipt:
            raise TypeError("submission must be a DurableLifecycleReceipt")
        if type(self.submission.event) not in _SUBMISSION_EVENT_TYPES:
            raise TypeError("submission receipt must contain a submission event")
        if not isinstance(self.fills, tuple):
            raise TypeError("fills must be a tuple")
        if any(type(receipt) is not DurableLifecycleReceipt for receipt in self.fills):
            raise TypeError("fills must contain only DurableLifecycleReceipt values")
        if any(type(receipt.event) is not ConfirmedFill for receipt in self.fills):
            raise TypeError("fill receipts must contain only ConfirmedFill events")

        submission_event = self.submission.event
        if submission_event.client_order_id != self.attempt.client_order_id:
            raise ValueError("submission client_order_id must match the attempt")
        if self.submission.event_id != self.attempt.event_id:
            raise ValueError("submission event_id must match the attempt")

        if submission_event.observed_at != self.attempt.observed_at:
            raise ValueError("the durable submission must retain the attempt timestamp")

        event_ids = (self.submission.event_id,) + tuple(
            receipt.event_id for receipt in self.fills
        )
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("durable event IDs must be distinct")

        expected_version = self.submission.position_version + 1
        for receipt in self.fills:
            if receipt.position_version != expected_version:
                raise ValueError(
                    "submission and fill versions must be strictly consecutive"
                )
            expected_version += 1

        fills = tuple(receipt.event for receipt in self.fills)
        if fills and not isinstance(submission_event, SubmissionAcknowledged):
            raise ValueError("only an acknowledged submission may contain fills")

        intent = self.attempt.instruction.order_intent
        for fill in fills:
            if fill.client_order_id != intent.client_order_id:
                raise ValueError("fill client_order_id must match the instruction")
            if fill.symbol != intent.symbol:
                raise ValueError("fill symbol must match the instruction")
            if fill.side is not intent.side:
                raise ValueError("fill side must match the instruction")
            if fill.executed_at < self.attempt.observed_at:
                raise ValueError("fill execution cannot predate the submission")

        if fills:
            first_fill = fills[0]
            if any(fill.venue_order_id != first_fill.venue_order_id for fill in fills):
                raise ValueError("fills must use one venue order ID")
            if first_fill.venue_order_id != submission_event.venue_order_id:
                raise ValueError(
                    "acknowledgement and fills must use one venue order ID"
                )
            trade_ids = tuple(fill.trade_id for fill in fills)
            if len(trade_ids) != len(set(trade_ids)):
                raise ValueError("fill trade IDs must be unique")
            filled_quantity = exact_decimal_sum(tuple(fill.quantity for fill in fills))
            if filled_quantity > intent.quantity:
                raise ValueError("confirmed fills exceed the instruction quantity")

    @property
    def canonical_report(self) -> SubmissionReport:
        """Derive the transport report solely from the durable submission fact."""
        event = self.submission.event
        if isinstance(event, SubmissionAcknowledged):
            return SubmissionReport(
                client_order_id=event.client_order_id,
                status=SubmissionStatus.SUBMITTED,
                retry_safety=RetrySafety.UNSAFE,
                venue_order_id=event.venue_order_id,
                venue_status=None,
            )
        if isinstance(event, SubmissionAmbiguous):
            return SubmissionReport(
                client_order_id=event.client_order_id,
                status=SubmissionStatus.AMBIGUOUS,
                retry_safety=RetrySafety.UNSAFE,
                reason=event.reason,
                venue_order_id=event.venue_order_id,
                venue_status=None,
            )
        if isinstance(event, SubmissionFailed):
            return SubmissionReport(
                client_order_id=event.client_order_id,
                status=event.status,
                retry_safety=event.retry_safety,
                reason=event.reason,
                venue_status=None,
            )
        raise AssertionError("durable submission receipt lost its submission event")


class DurableSubmissionOwner(Protocol):
    """Atomic boundary which owns reservation, submission, and durable facts."""

    def execute(
        self,
        attempt: SubmissionAttemptContext,
        /,
    ) -> DurableSubmissionReceipt:
        """Return committed facts or their exact replay without implicit retry."""
        ...


@dataclass(frozen=True, slots=True)
class SubmissionCommitUnknown(RuntimeError):
    """Preserve an attempt whose durable commit acknowledgement was lost."""

    attempt: SubmissionAttemptContext

    def __post_init__(self) -> None:
        if type(self.attempt) is not SubmissionAttemptContext:
            raise TypeError("attempt must be a SubmissionAttemptContext")
        RuntimeError.__init__(self, "durable submission commit outcome is unknown")

    @property
    def client_order_id(self) -> str:
        """Expose the derived client identity needed by reconciliation."""
        return self.attempt.client_order_id

    @property
    def requires_reconciliation(self) -> bool:
        """An unacknowledged commit must be resolved before any retry."""
        return True


__all__ = [
    "DurableLifecycleReceipt",
    "DurableSubmissionDisposition",
    "DurableSubmissionOwner",
    "DurableSubmissionReceipt",
    "SubmissionAttemptContext",
    "SubmissionCommitUnknown",
]
