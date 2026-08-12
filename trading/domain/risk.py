"""Immutable pre-trade risk-decision contracts."""

from dataclasses import dataclass

from trading.domain._validation import require_clean_text
from trading.domain.orders import OrderIntent


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """One correlated pre-trade decision, never an execution result."""

    decision_id: str
    approved: bool
    reasons: tuple[str, ...] = ()
    order_intent: OrderIntent | None = None

    def __post_init__(self) -> None:
        require_clean_text("decision_id", self.decision_id)
        if type(self.approved) is not bool:
            raise TypeError("approved must be a boolean")

        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple")
        for reason in self.reasons:
            require_clean_text("reason", reason)

        if self.order_intent is not None and not isinstance(
            self.order_intent, OrderIntent
        ):
            raise TypeError("order_intent must be an OrderIntent or None")

        if self.approved:
            if self.order_intent is None:
                raise ValueError("an approved risk decision requires an intent")
            if self.order_intent.decision_id != self.decision_id:
                raise ValueError("intent decision_id must match the risk decision")
        else:
            if self.order_intent is not None:
                raise ValueError("a rejected risk decision cannot have an intent")
            if not self.reasons:
                raise ValueError("a rejected risk decision requires a reason")
