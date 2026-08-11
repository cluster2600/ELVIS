"""Immutable strategy-decision contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from trading.domain._validation import (
    require_aware_datetime,
    require_clean_text,
    require_finite_real,
)


class SignalAction(str, Enum):
    """A strategy decision, including the non-actionable HOLD state."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class Signal:
    """A validated strategy decision independent of execution infrastructure."""

    decision_id: str
    symbol: str
    action: SignalAction
    confidence: float
    reference_price: float
    observed_at: datetime
    strategy_id: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_clean_text("decision_id", self.decision_id)
        require_clean_text("symbol", self.symbol)
        require_clean_text("strategy_id", self.strategy_id)

        if not isinstance(self.action, SignalAction):
            raise TypeError("action must be a SignalAction")

        confidence = require_finite_real("confidence", self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)

        reference_price = require_finite_real("reference_price", self.reference_price)
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")
        object.__setattr__(self, "reference_price", reference_price)

        require_aware_datetime("observed_at", self.observed_at)

        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple")
        for reason in self.reasons:
            require_clean_text("reason", reason)
