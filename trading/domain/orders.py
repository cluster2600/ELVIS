"""Immutable order-intent and submission-outcome contracts."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from trading.domain._validation import (
    require_aware_datetime,
    require_clean_text,
    require_optional_clean_text,
    require_positive_decimal,
)


class OrderSide(str, Enum):
    """An actionable order direction; HOLD is intentionally absent."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order types currently supported by the migrated application path."""

    MARKET = "MARKET"


class SubmissionStatus(str, Enum):
    """What is known about a single order-submission attempt."""

    NOT_SENT = "NOT_SENT"
    VENUE_REJECTED = "VENUE_REJECTED"
    SUBMITTED = "SUBMITTED"
    AMBIGUOUS = "AMBIGUOUS"


class RetrySafety(str, Enum):
    """Whether another transport attempt is known not to duplicate an order."""

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A complete, pre-approved request ready for an execution adapter."""

    client_order_id: str
    decision_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    reference_price: Decimal
    leverage: int
    created_at: datetime

    def __post_init__(self) -> None:
        require_clean_text("client_order_id", self.client_order_id)
        require_clean_text("decision_id", self.decision_id)
        require_clean_text("symbol", self.symbol)

        if not isinstance(self.side, OrderSide):
            raise TypeError("side must be an OrderSide")
        if not isinstance(self.order_type, OrderType):
            raise TypeError("order_type must be an OrderType")

        require_positive_decimal("quantity", self.quantity)
        require_positive_decimal("reference_price", self.reference_price)

        if isinstance(self.leverage, bool) or not isinstance(self.leverage, int):
            raise TypeError("leverage must be an integer")
        if self.leverage < 1:
            raise ValueError("leverage must be positive")

        require_aware_datetime("created_at", self.created_at)


@dataclass(frozen=True, slots=True)
class SubmissionReport:
    """The transport-level result of one submission attempt, never a fill."""

    client_order_id: str
    status: SubmissionStatus
    retry_safety: RetrySafety
    reason: str | None = None
    venue_order_id: str | None = None
    venue_status: str | None = None

    def __post_init__(self) -> None:
        require_clean_text("client_order_id", self.client_order_id)
        if not isinstance(self.status, SubmissionStatus):
            raise TypeError("status must be a SubmissionStatus")
        if not isinstance(self.retry_safety, RetrySafety):
            raise TypeError("retry_safety must be a RetrySafety")

        require_optional_clean_text("reason", self.reason)
        require_optional_clean_text("venue_order_id", self.venue_order_id)
        require_optional_clean_text("venue_status", self.venue_status)

        if self.status is not SubmissionStatus.SUBMITTED and self.reason is None:
            raise ValueError("a non-submitted report requires a reason")
        if self.status is SubmissionStatus.SUBMITTED and self.venue_order_id is None:
            raise ValueError("a submitted report requires a venue_order_id")
        if (
            self.status in {SubmissionStatus.NOT_SENT, SubmissionStatus.VENUE_REJECTED}
            and self.venue_order_id is not None
        ):
            raise ValueError("a non-submitted report cannot have a venue_order_id")
        if self.status is SubmissionStatus.NOT_SENT and self.venue_status is not None:
            raise ValueError("a request that was not sent cannot have a venue_status")
        if (
            self.status in {SubmissionStatus.SUBMITTED, SubmissionStatus.AMBIGUOUS}
            and self.retry_safety is not RetrySafety.UNSAFE
        ):
            raise ValueError("a possibly submitted order is unsafe to retry")

    @property
    def acknowledged(self) -> bool:
        """Return whether the adapter proved submission, not whether it filled."""
        return self.status is SubmissionStatus.SUBMITTED

    @property
    def requires_reconciliation(self) -> bool:
        """Return whether venue state must resolve an uncertain submission."""
        return self.status is SubmissionStatus.AMBIGUOUS
