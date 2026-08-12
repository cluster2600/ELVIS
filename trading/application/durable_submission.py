"""Pure contracts for one atomically durable submission attempt."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from trading.domain._decimal import exact_decimal_sum
from trading.domain._validation import protect_frozen_dataclass_state
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


@protect_frozen_dataclass_state
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
        if self.observed_at < intent.created_at:
            raise ValueError("observed_at cannot predate the instruction")

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


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperPlannedFill:
    """One non-durable fill candidate with its future event identity."""

    event_id: str
    fill: ConfirmedFill

    def __post_init__(self) -> None:
        _require_identifier("event_id", self.event_id)
        if type(self.fill) is not ConfirmedFill:
            raise TypeError("fill must be a ConfirmedFill")


def _validate_submission_facts(
    attempt: SubmissionAttemptContext,
    submission: SubmissionEvent,
    fills: tuple[tuple[str, ConfirmedFill], ...],
) -> None:
    if type(submission) not in _SUBMISSION_EVENT_TYPES:
        raise TypeError("submission must be a submission lifecycle event")
    _validate_event_identifiers(submission)
    _require_durable_datetime("submission timestamp", submission.observed_at)
    if submission.client_order_id != attempt.client_order_id:
        raise ValueError("submission client_order_id must match the attempt")
    if submission.observed_at != attempt.observed_at:
        raise ValueError("submission must retain the attempt timestamp")

    event_ids = (attempt.event_id,) + tuple(event_id for event_id, _fill in fills)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("submission and fill event IDs must be distinct")
    if fills and type(submission) is not SubmissionAcknowledged:
        raise ValueError("only an acknowledged submission may contain fills")

    intent = attempt.instruction.order_intent
    confirmed_fills = tuple(fill for _event_id, fill in fills)
    for fill in confirmed_fills:
        _validate_event_identifiers(fill)
        _require_durable_datetime("fill timestamp", fill.executed_at)
        if fill.client_order_id != intent.client_order_id:
            raise ValueError("fill client_order_id must match the instruction")
        if fill.symbol != intent.symbol:
            raise ValueError("fill symbol must match the instruction")
        if fill.side is not intent.side:
            raise ValueError("fill side must match the instruction")
        if fill.executed_at < attempt.observed_at:
            raise ValueError("fill execution cannot predate the submission")

    if confirmed_fills:
        first_fill = confirmed_fills[0]
        if any(
            fill.venue_order_id != first_fill.venue_order_id for fill in confirmed_fills
        ):
            raise ValueError("fills must use one venue order ID")
        if first_fill.venue_order_id != submission.venue_order_id:
            raise ValueError("acknowledgement and fills must use one venue order ID")
        trade_ids = tuple(fill.trade_id for fill in confirmed_fills)
        if len(trade_ids) != len(set(trade_ids)):
            raise ValueError("fill trade IDs must be unique")
        filled_quantity = exact_decimal_sum(
            tuple(fill.quantity for fill in confirmed_fills)
        )
        if filled_quantity > intent.quantity:
            raise ValueError("confirmed fills exceed the instruction quantity")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperSubmissionPlan:
    """Pure candidate facts for one future atomic paper transaction."""

    attempt: SubmissionAttemptContext
    submission: SubmissionAcknowledged
    fills: tuple[PaperPlannedFill, ...]

    def __post_init__(self) -> None:
        if type(self.attempt) is not SubmissionAttemptContext:
            raise TypeError("attempt must be a SubmissionAttemptContext")
        if type(self.submission) is not SubmissionAcknowledged:
            raise TypeError("submission must be a SubmissionAcknowledged event")
        if type(self.fills) is not tuple:
            raise TypeError("fills must be a tuple")
        if not self.fills:
            raise ValueError("a paper submission plan requires at least one fill")
        if any(type(candidate) is not PaperPlannedFill for candidate in self.fills):
            raise TypeError("fills must contain only PaperPlannedFill values")
        _validate_submission_facts(
            self.attempt,
            self.submission,
            tuple((candidate.event_id, candidate.fill) for candidate in self.fills),
        )
        filled_quantity = exact_decimal_sum(
            tuple(candidate.fill.quantity for candidate in self.fills)
        )
        if filled_quantity != self.attempt.instruction.order_intent.quantity:
            raise ValueError("paper submission plans must be exact full fills")


class PaperSubmissionPlanner(Protocol):
    """Stable source that supplies facts without claiming they are durable."""

    def plan(
        self,
        attempt: SubmissionAttemptContext,
        /,
    ) -> PaperSubmissionPlan:
        """Return a precomputed plan while retaining the exact attempt object."""
        ...


@protect_frozen_dataclass_state
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


@protect_frozen_dataclass_state
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

        if self.submission.event_id != self.attempt.event_id:
            raise ValueError("submission event_id must match the attempt")

        expected_version = self.submission.position_version + 1
        for receipt in self.fills:
            if receipt.position_version != expected_version:
                raise ValueError(
                    "submission and fill versions must be strictly consecutive"
                )
            expected_version += 1

        _validate_submission_facts(
            self.attempt,
            self.submission.event,
            tuple((receipt.event_id, receipt.event) for receipt in self.fills),
        )

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


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class SubmissionCommitUnknown(RuntimeError):
    """Preserve an attempt whose durable commit acknowledgement was lost."""

    attempt: SubmissionAttemptContext

    def __post_init__(self) -> None:
        if type(self.attempt) is not SubmissionAttemptContext:
            raise TypeError("attempt must be a SubmissionAttemptContext")
        RuntimeError.__init__(self, "durable submission commit outcome is unknown")

    def __reduce__(self) -> tuple[object, tuple[SubmissionAttemptContext]]:
        """Reconstruct the typed exception from its attempt, not its message."""
        return (type(self), (self.attempt,))

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
    "PaperPlannedFill",
    "PaperSubmissionPlan",
    "PaperSubmissionPlanner",
    "SubmissionAttemptContext",
    "SubmissionCommitUnknown",
]
