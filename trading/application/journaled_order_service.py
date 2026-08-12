"""Register-before-submit orchestration for the unwired journal path."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from trading.application.order_service import OrderService
from trading.domain._validation import require_aware_datetime, require_clean_text
from trading.domain.order_lifecycle import (
    OrderLifecycleEvent,
    SubmissionAcknowledged,
    SubmissionAmbiguous,
    SubmissionEvent,
    SubmissionFailed,
    submission_event_from_report,
)
from trading.domain.orders import SubmissionReport
from trading.domain.positions import PositionInstruction

_SUBMISSION_EVENT_ID = "submission-attempt-1"
_SUBMISSION_EVENT_TYPES = (
    SubmissionAcknowledged,
    SubmissionAmbiguous,
    SubmissionFailed,
)


class Clock(Protocol):
    """Wall-clock boundary used to timestamp one submission observation."""

    def now(self, /) -> datetime:
        """Return one timezone-aware wall-clock timestamp."""
        ...


class ReservationReceipt(Protocol):
    """Storage-neutral view of a committed instruction reservation."""

    @property
    def is_created(self) -> bool:
        """Return whether this call committed a new reservation."""
        ...


class EventReceipt(Protocol):
    """Storage-neutral view of a committed journal event."""

    @property
    def durable_event_id(self) -> str:
        """Return the event identity selected by durable storage."""
        ...

    @property
    def position_version(self) -> int:
        """Return the committed position-stream version."""
        ...


class OrderJournalPort(Protocol):
    """Minimal durable boundary needed by register-before-submit."""

    def reserve_instruction(
        self,
        *,
        execution_scope: str,
        instruction: PositionInstruction,
    ) -> ReservationReceipt:
        """Commit an instruction reservation or rediscover its exact identity."""
        ...

    def append_event(
        self,
        *,
        execution_scope: str,
        position_key: str,
        event_id: str,
        event: OrderLifecycleEvent,
    ) -> EventReceipt:
        """Commit one exact event after validating the current stream."""
        ...


class JournaledSubmissionDisposition(str, Enum):
    """Whether this invocation recorded a report or found prior work."""

    RECORDED = "RECORDED"
    EXISTING_RESERVATION = "EXISTING_RESERVATION"


@dataclass(frozen=True, slots=True)
class JournaledSubmissionResult:
    """One non-ambiguous application result from the journaled owner."""

    disposition: JournaledSubmissionDisposition
    report: SubmissionReport | None = None
    event: SubmissionEvent | None = None
    durable_event_id: str | None = None
    position_version: int | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not JournaledSubmissionDisposition:
            raise TypeError("disposition must be a JournaledSubmissionDisposition")

        details = (
            self.report,
            self.event,
            self.durable_event_id,
            self.position_version,
        )
        if self.disposition is JournaledSubmissionDisposition.EXISTING_RESERVATION:
            if any(value is not None for value in details):
                raise ValueError("an existing reservation cannot report a new attempt")
            return

        if not isinstance(self.report, SubmissionReport):
            raise TypeError("a recorded submission requires a SubmissionReport")
        if type(self.event) not in _SUBMISSION_EVENT_TYPES:
            raise TypeError("a recorded submission requires a submission event")
        if self.durable_event_id != _SUBMISSION_EVENT_ID:
            raise ValueError("recorded submission event identity is invalid")
        if (
            isinstance(self.position_version, bool)
            or not isinstance(self.position_version, int)
            or self.position_version < 1
        ):
            raise ValueError(
                "a recorded submission requires a positive position version"
            )
        expected_event = submission_event_from_report(
            self.report,
            self.event.observed_at,
        )
        if self.event != expected_event:
            raise ValueError("the durable event must exactly represent the report")

    @property
    def execution_attempted(self) -> bool:
        """Return whether this invocation called the order service."""
        return self.report is not None

    @property
    def requires_reconciliation(self) -> bool:
        """Return whether venue or prior-reservation state remains unresolved."""
        if self.disposition is JournaledSubmissionDisposition.EXISTING_RESERVATION:
            return True
        if self.report is None:
            raise AssertionError("recorded result lost its submission report")
        return self.report.requires_reconciliation


class SubmissionObservationNotRecorded(RuntimeError):
    """Preserve a transport outcome whose journal append did not return success."""

    __slots__ = ("report", "event", "event_id")

    def __init__(
        self,
        *,
        report: SubmissionReport,
        event: SubmissionEvent,
        event_id: str,
    ) -> None:
        super().__init__("submission observation was not durably recorded")
        self.report = report
        self.event = event
        self.event_id = event_id

    @property
    def requires_reconciliation(self) -> bool:
        """An external attempt without acknowledged persistence is unresolved."""
        return True


class JournaledOrderService:
    """Commit an instruction before exactly one order-service invocation."""

    __slots__ = ("_clock", "_journal", "_orders")

    def __init__(
        self,
        order_service: OrderService,
        journal: OrderJournalPort,
        clock: Clock,
    ) -> None:
        if not isinstance(order_service, OrderService):
            raise TypeError("order_service must be an OrderService")
        if not callable(getattr(journal, "reserve_instruction", None)):
            raise TypeError("journal must provide reserve_instruction")
        if not callable(getattr(journal, "append_event", None)):
            raise TypeError("journal must provide append_event")
        if not callable(getattr(clock, "now", None)):
            raise TypeError("clock must provide now")
        self._orders = order_service
        self._journal = journal
        self._clock = clock

    def submit(
        self,
        instruction: PositionInstruction,
        *,
        execution_scope: str,
    ) -> JournaledSubmissionResult:
        """Reserve, submit at most once, and persist the transport observation."""
        if type(instruction) is not PositionInstruction:
            raise TypeError("instruction must be a PositionInstruction")
        require_clean_text("execution_scope", execution_scope)

        reservation = self._journal.reserve_instruction(
            execution_scope=execution_scope,
            instruction=instruction,
        )
        created = reservation.is_created
        if type(created) is not bool:
            raise TypeError("reservation is_created must be a bool")
        if not created:
            return JournaledSubmissionResult(
                disposition=JournaledSubmissionDisposition.EXISTING_RESERVATION
            )

        observed_at = self._clock.now()
        require_aware_datetime("observed_at", observed_at)

        report = self._orders.submit(instruction.order_intent)
        event = submission_event_from_report(report, observed_at)
        try:
            committed = self._journal.append_event(
                execution_scope=execution_scope,
                position_key=instruction.position_key,
                event_id=_SUBMISSION_EVENT_ID,
                event=event,
            )
            durable_event_id = committed.durable_event_id
            position_version = committed.position_version
            if durable_event_id != _SUBMISSION_EVENT_ID:
                raise ValueError(
                    "journal returned a mismatched durable submission event ID"
                )
            return JournaledSubmissionResult(
                disposition=JournaledSubmissionDisposition.RECORDED,
                report=report,
                event=event,
                durable_event_id=durable_event_id,
                position_version=position_version,
            )
        except Exception as exc:
            raise SubmissionObservationNotRecorded(
                report=report,
                event=event,
                event_id=_SUBMISSION_EVENT_ID,
            ) from exc


__all__ = [
    "Clock",
    "EventReceipt",
    "JournaledOrderService",
    "JournaledSubmissionDisposition",
    "JournaledSubmissionResult",
    "OrderJournalPort",
    "ReservationReceipt",
    "SubmissionObservationNotRecorded",
]
