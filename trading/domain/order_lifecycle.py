"""Pure, immutable order-lifecycle state transitions."""

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum

from trading.domain._decimal import exact_decimal_sum
from trading.domain._validation import (
    require_aware_datetime,
    require_clean_text,
    require_non_negative_decimal,
    require_optional_clean_text,
    require_positive_decimal,
)
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)


class OrderLifecycleState(str, Enum):
    """What ELVIS currently knows about one correlated order."""

    PENDING = "PENDING"
    RECONCILING = "RECONCILING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    FILLED = "FILLED"
    FAILED = "FAILED"


class InvalidOrderTransition(ValueError):
    """Raised when an event contradicts the known order lifecycle."""


@dataclass(frozen=True, slots=True)
class SubmissionAcknowledged:
    """Proof that a venue accepted an order, never proof of a fill."""

    client_order_id: str
    venue_order_id: str
    observed_at: datetime

    def __post_init__(self) -> None:
        require_clean_text("client_order_id", self.client_order_id)
        require_clean_text("venue_order_id", self.venue_order_id)
        require_aware_datetime("observed_at", self.observed_at)


@dataclass(frozen=True, slots=True)
class SubmissionAmbiguous:
    """A submission which may exist at the venue and must be reconciled."""

    client_order_id: str
    reason: str
    observed_at: datetime
    venue_order_id: str | None = None

    def __post_init__(self) -> None:
        require_clean_text("client_order_id", self.client_order_id)
        require_clean_text("reason", self.reason)
        require_aware_datetime("observed_at", self.observed_at)
        require_optional_clean_text("venue_order_id", self.venue_order_id)


@dataclass(frozen=True, slots=True)
class SubmissionFailed:
    """Proof that no live venue order resulted from the submission."""

    client_order_id: str
    status: SubmissionStatus
    retry_safety: RetrySafety
    reason: str
    observed_at: datetime

    def __post_init__(self) -> None:
        require_clean_text("client_order_id", self.client_order_id)
        if not isinstance(self.status, SubmissionStatus):
            raise TypeError("status must be a SubmissionStatus")
        if self.status not in {
            SubmissionStatus.NOT_SENT,
            SubmissionStatus.VENUE_REJECTED,
        }:
            raise ValueError("a failed event requires a proven non-submission status")
        if not isinstance(self.retry_safety, RetrySafety):
            raise TypeError("retry_safety must be a RetrySafety")
        require_clean_text("reason", self.reason)
        require_aware_datetime("observed_at", self.observed_at)


@dataclass(frozen=True, slots=True)
class ConfirmedFill:
    """One exact venue fill, independently confirmed from submission."""

    client_order_id: str
    venue_order_id: str
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee_amount: Decimal
    executed_at: datetime
    fee_asset: str | None = None

    def __post_init__(self) -> None:
        require_clean_text("client_order_id", self.client_order_id)
        require_clean_text("venue_order_id", self.venue_order_id)
        require_clean_text("trade_id", self.trade_id)
        require_clean_text("symbol", self.symbol)
        if not isinstance(self.side, OrderSide):
            raise TypeError("side must be an OrderSide")
        require_positive_decimal("quantity", self.quantity)
        require_positive_decimal("price", self.price)
        require_non_negative_decimal("fee_amount", self.fee_amount)
        require_optional_clean_text("fee_asset", self.fee_asset)
        if self.fee_amount > 0 and self.fee_asset is None:
            raise ValueError("a positive fee requires a fee_asset")
        require_aware_datetime("executed_at", self.executed_at)


@dataclass(frozen=True, slots=True)
class CancellationRequested:
    """A local request to cancel a known venue order."""

    client_order_id: str
    cancel_request_id: str
    requested_at: datetime

    def __post_init__(self) -> None:
        require_clean_text("client_order_id", self.client_order_id)
        require_clean_text("cancel_request_id", self.cancel_request_id)
        require_aware_datetime("requested_at", self.requested_at)


@dataclass(frozen=True, slots=True)
class CancellationConfirmed:
    """Venue proof that the unfilled remainder was cancelled."""

    client_order_id: str
    venue_order_id: str
    cancel_request_id: str
    observed_at: datetime

    def __post_init__(self) -> None:
        require_clean_text("client_order_id", self.client_order_id)
        require_clean_text("venue_order_id", self.venue_order_id)
        require_clean_text("cancel_request_id", self.cancel_request_id)
        require_aware_datetime("observed_at", self.observed_at)


@dataclass(frozen=True, slots=True)
class CancellationRejected:
    """Venue proof that a cancellation request was not applied."""

    client_order_id: str
    venue_order_id: str
    cancel_request_id: str
    reason: str
    observed_at: datetime

    def __post_init__(self) -> None:
        require_clean_text("client_order_id", self.client_order_id)
        require_clean_text("venue_order_id", self.venue_order_id)
        require_clean_text("cancel_request_id", self.cancel_request_id)
        require_clean_text("reason", self.reason)
        require_aware_datetime("observed_at", self.observed_at)


SubmissionEvent = SubmissionAcknowledged | SubmissionAmbiguous | SubmissionFailed
OrderLifecycleEvent = (
    SubmissionEvent
    | ConfirmedFill
    | CancellationRequested
    | CancellationConfirmed
    | CancellationRejected
)

_EVENT_TYPES = (
    SubmissionAcknowledged,
    SubmissionAmbiguous,
    SubmissionFailed,
    ConfirmedFill,
    CancellationRequested,
    CancellationConfirmed,
    CancellationRejected,
)


def _sum_fill_quantities(fills: tuple[ConfirmedFill, ...]) -> Decimal:
    return exact_decimal_sum(tuple(fill.quantity for fill in fills))


@dataclass(frozen=True, slots=True)
class OrderLifecycle:
    """A validated projection derived only from immutable order events."""

    intent: OrderIntent
    state: OrderLifecycleState
    venue_order_id: str | None = None
    fills: tuple[ConfirmedFill, ...] = ()
    pending_cancel_request_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.intent, OrderIntent):
            raise TypeError("intent must be an OrderIntent")
        if not isinstance(self.state, OrderLifecycleState):
            raise TypeError("state must be an OrderLifecycleState")
        require_optional_clean_text("venue_order_id", self.venue_order_id)
        require_optional_clean_text(
            "pending_cancel_request_id", self.pending_cancel_request_id
        )
        if not isinstance(self.fills, tuple):
            raise TypeError("fills must be a tuple")
        if any(type(fill) is not ConfirmedFill for fill in self.fills):
            raise TypeError("fills must contain only ConfirmedFill values")

        trade_ids = tuple(fill.trade_id for fill in self.fills)
        if len(trade_ids) != len(set(trade_ids)):
            raise ValueError("fill trade IDs must be unique")
        if trade_ids != tuple(sorted(trade_ids)):
            raise ValueError("fills must be ordered canonically by trade_id")

        for fill in self.fills:
            if fill.client_order_id != self.intent.client_order_id:
                raise ValueError("fill client_order_id must match the intent")
            if fill.symbol != self.intent.symbol:
                raise ValueError("fill symbol must match the intent")
            if fill.side is not self.intent.side:
                raise ValueError("fill side must match the intent")
            if fill.venue_order_id != self.venue_order_id:
                raise ValueError("fill venue_order_id must match the lifecycle")

        filled_quantity = _sum_fill_quantities(self.fills)
        if filled_quantity > self.intent.quantity:
            raise ValueError("filled quantity cannot exceed the order quantity")
        remaining_quantity = exact_decimal_sum(
            (self.intent.quantity, filled_quantity.copy_negate())
        )

        has_venue = self.venue_order_id is not None
        has_fills = filled_quantity > 0
        is_full = remaining_quantity == 0

        if self.state is OrderLifecycleState.PENDING:
            if has_venue or has_fills:
                raise ValueError("a pending order cannot have venue facts")
        elif self.state is OrderLifecycleState.RECONCILING:
            if has_fills:
                raise ValueError("a reconciling submission cannot contain fills")
        elif self.state is OrderLifecycleState.OPEN:
            if not has_venue or has_fills:
                raise ValueError("an open order requires a venue ID and no fills")
        elif self.state is OrderLifecycleState.PARTIAL:
            if not has_venue or not has_fills or is_full:
                raise ValueError("a partial order requires an incomplete fill")
        elif self.state is OrderLifecycleState.CANCEL_PENDING:
            if not has_venue or is_full or self.pending_cancel_request_id is None:
                raise ValueError("a pending cancellation requires an open remainder")
        elif self.state is OrderLifecycleState.CANCELLED:
            if not has_venue or is_full:
                raise ValueError("a cancelled order must retain an unfilled remainder")
        elif self.state is OrderLifecycleState.FILLED:
            if not has_venue or not is_full:
                raise ValueError("a filled order requires the exact order quantity")
        elif self.state is OrderLifecycleState.FAILED:
            if has_venue or has_fills:
                raise ValueError("a failed submission cannot have venue facts")

        if (
            self.state is not OrderLifecycleState.CANCEL_PENDING
            and self.pending_cancel_request_id is not None
        ):
            raise ValueError("only a pending cancellation can retain its request ID")

    @property
    def filled_quantity(self) -> Decimal:
        """Return the exact sum of unique confirmed fills."""
        return _sum_fill_quantities(self.fills)

    @property
    def remaining_quantity(self) -> Decimal:
        """Return the exact unfilled order quantity."""
        return exact_decimal_sum(
            (self.intent.quantity, self.filled_quantity.copy_negate())
        )


def new_order_lifecycle(intent: OrderIntent) -> OrderLifecycle:
    """Create the only valid initial lifecycle for an approved intent."""
    if not isinstance(intent, OrderIntent):
        raise TypeError("intent must be an OrderIntent")
    return OrderLifecycle(intent=intent, state=OrderLifecycleState.PENDING)


def submission_event_from_report(
    report: SubmissionReport,
    observed_at: datetime,
) -> SubmissionEvent:
    """Translate transport certainty without ever inferring a venue fill."""
    if not isinstance(report, SubmissionReport):
        raise TypeError("report must be a SubmissionReport")
    require_aware_datetime("observed_at", observed_at)

    if report.status is SubmissionStatus.SUBMITTED:
        return SubmissionAcknowledged(
            client_order_id=report.client_order_id,
            venue_order_id=report.venue_order_id,
            observed_at=observed_at,
        )
    if report.status is SubmissionStatus.AMBIGUOUS:
        return SubmissionAmbiguous(
            client_order_id=report.client_order_id,
            reason=report.reason,
            observed_at=observed_at,
            venue_order_id=report.venue_order_id,
        )
    if report.status in {
        SubmissionStatus.NOT_SENT,
        SubmissionStatus.VENUE_REJECTED,
    }:
        return SubmissionFailed(
            client_order_id=report.client_order_id,
            status=report.status,
            retry_safety=report.retry_safety,
            reason=report.reason,
            observed_at=observed_at,
        )
    raise AssertionError("unhandled submission status")


def _resolved_venue_order_id(
    order: OrderLifecycle,
    incoming_venue_order_id: str | None,
) -> str | None:
    if incoming_venue_order_id is None:
        return order.venue_order_id
    if (
        order.venue_order_id is not None
        and order.venue_order_id != incoming_venue_order_id
    ):
        raise InvalidOrderTransition("venue_order_id conflicts with known order")
    return order.venue_order_id or incoming_venue_order_id


def reduce_order_lifecycle(
    order: OrderLifecycle,
    event: OrderLifecycleEvent,
) -> OrderLifecycle:
    """Apply one validated fact without I/O, retries, or implicit time."""
    if type(order) is not OrderLifecycle:
        raise TypeError("order must be an OrderLifecycle")
    if type(event) not in _EVENT_TYPES:
        raise TypeError("event must be a supported order lifecycle event")
    if event.client_order_id != order.intent.client_order_id:
        raise InvalidOrderTransition("client_order_id does not match the order")

    if isinstance(event, SubmissionAcknowledged):
        if order.state is OrderLifecycleState.FAILED:
            raise InvalidOrderTransition("a failed submission cannot be acknowledged")
        venue_order_id = _resolved_venue_order_id(order, event.venue_order_id)
        if order.state in {
            OrderLifecycleState.PENDING,
            OrderLifecycleState.RECONCILING,
        }:
            return replace(
                order,
                state=OrderLifecycleState.OPEN,
                venue_order_id=venue_order_id,
            )
        return order

    if isinstance(event, SubmissionAmbiguous):
        if order.state is OrderLifecycleState.FAILED:
            return order
        venue_order_id = _resolved_venue_order_id(order, event.venue_order_id)
        if order.state is OrderLifecycleState.PENDING:
            return replace(
                order,
                state=OrderLifecycleState.RECONCILING,
                venue_order_id=venue_order_id,
            )
        if (
            order.state is OrderLifecycleState.RECONCILING
            and venue_order_id != order.venue_order_id
        ):
            return replace(order, venue_order_id=venue_order_id)
        return order

    if isinstance(event, SubmissionFailed):
        if order.state in {
            OrderLifecycleState.PENDING,
            OrderLifecycleState.RECONCILING,
        }:
            return replace(
                order,
                state=OrderLifecycleState.FAILED,
                venue_order_id=None,
            )
        if order.state is OrderLifecycleState.FAILED:
            return order
        raise InvalidOrderTransition("venue facts contradict submission failure")

    if isinstance(event, ConfirmedFill):
        if event.symbol != order.intent.symbol:
            raise InvalidOrderTransition("fill symbol does not match the order")
        if event.side is not order.intent.side:
            raise InvalidOrderTransition("fill side does not match the order")
        venue_order_id = _resolved_venue_order_id(order, event.venue_order_id)

        existing = next(
            (fill for fill in order.fills if fill.trade_id == event.trade_id),
            None,
        )
        if existing is not None:
            if existing == event:
                return order
            raise InvalidOrderTransition("trade_id has conflicting fill data")
        if order.state is OrderLifecycleState.FAILED:
            raise InvalidOrderTransition("a failed submission cannot receive a fill")

        fills = tuple(sorted(order.fills + (event,), key=lambda fill: fill.trade_id))
        try:
            filled_quantity = _sum_fill_quantities(fills)
        except ValueError as exc:
            raise InvalidOrderTransition(
                "confirmed fills cannot be aggregated exactly"
            ) from exc
        if filled_quantity > order.intent.quantity:
            raise InvalidOrderTransition("confirmed fills exceed the order quantity")
        if filled_quantity == order.intent.quantity:
            next_state = OrderLifecycleState.FILLED
        elif order.state is OrderLifecycleState.CANCEL_PENDING:
            next_state = OrderLifecycleState.CANCEL_PENDING
        elif order.state is OrderLifecycleState.CANCELLED:
            next_state = OrderLifecycleState.CANCELLED
        else:
            next_state = OrderLifecycleState.PARTIAL
        return replace(
            order,
            state=next_state,
            venue_order_id=venue_order_id,
            fills=fills,
            pending_cancel_request_id=(
                order.pending_cancel_request_id
                if next_state is OrderLifecycleState.CANCEL_PENDING
                else None
            ),
        )

    if isinstance(event, CancellationRequested):
        if order.state in {OrderLifecycleState.OPEN, OrderLifecycleState.PARTIAL}:
            return replace(
                order,
                state=OrderLifecycleState.CANCEL_PENDING,
                pending_cancel_request_id=event.cancel_request_id,
            )
        if order.state is OrderLifecycleState.CANCEL_PENDING:
            if event.cancel_request_id != order.pending_cancel_request_id:
                raise InvalidOrderTransition(
                    "another cancellation request is already pending"
                )
            return order
        if order.state in {
            OrderLifecycleState.CANCELLED,
            OrderLifecycleState.FILLED,
        }:
            return order
        raise InvalidOrderTransition("order cannot be cancelled in its current state")

    if isinstance(event, CancellationConfirmed):
        if order.state is OrderLifecycleState.FAILED:
            raise InvalidOrderTransition("a failed submission cannot be cancelled")
        venue_order_id = _resolved_venue_order_id(order, event.venue_order_id)
        if order.state in {
            OrderLifecycleState.CANCELLED,
            OrderLifecycleState.FILLED,
        }:
            return order
        if order.state is not OrderLifecycleState.CANCEL_PENDING:
            raise InvalidOrderTransition(
                "cancellation confirmation has no matching pending request"
            )
        if event.cancel_request_id != order.pending_cancel_request_id:
            raise InvalidOrderTransition(
                "cancellation confirmation does not match the pending request"
            )
        return replace(
            order,
            state=OrderLifecycleState.CANCELLED,
            venue_order_id=venue_order_id,
            pending_cancel_request_id=None,
        )

    if isinstance(event, CancellationRejected):
        if order.state in {
            OrderLifecycleState.CANCELLED,
            OrderLifecycleState.FILLED,
        }:
            _resolved_venue_order_id(order, event.venue_order_id)
            return order
        if order.state is not OrderLifecycleState.CANCEL_PENDING:
            raise InvalidOrderTransition(
                "cancellation rejection has no matching pending request"
            )
        _resolved_venue_order_id(order, event.venue_order_id)
        if event.cancel_request_id != order.pending_cancel_request_id:
            raise InvalidOrderTransition(
                "cancellation rejection does not match the pending request"
            )
        restored_state = (
            OrderLifecycleState.PARTIAL
            if order.filled_quantity > 0
            else OrderLifecycleState.OPEN
        )
        return replace(
            order,
            state=restored_state,
            pending_cancel_request_id=None,
        )

    raise AssertionError("unreachable lifecycle event")
