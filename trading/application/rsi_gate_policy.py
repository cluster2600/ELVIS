"""Pure RSI qualification policy for the progressive signal migration."""

from dataclasses import dataclass, field

from trading.application.signal_policy import SignalPolicyResult
from trading.domain._validation import require_finite_real
from trading.domain.signals import Signal, SignalAction


@dataclass(frozen=True, slots=True)
class RsiGatePolicy:
    """Veto entries into a stretched or unobservable RSI state."""

    rsi: object
    overbought: float = 70.0
    oversold: float = 30.0
    policy_id: str = field(default="rsi-gate", init=False)

    def __post_init__(self) -> None:
        overbought = require_finite_real("overbought", self.overbought)
        oversold = require_finite_real("oversold", self.oversold)
        if not 0.0 <= oversold < overbought <= 100.0:
            raise ValueError(
                "RSI thresholds must satisfy 0 <= oversold < overbought <= 100"
            )
        object.__setattr__(self, "overbought", overbought)
        object.__setattr__(self, "oversold", oversold)

        try:
            rsi = require_finite_real("rsi", self.rsi)
        except (TypeError, ValueError):
            return
        object.__setattr__(self, "rsi", rsi)

    def evaluate(self, signal: Signal, /) -> SignalPolicyResult:
        """Apply strict legacy boundaries and fail closed on invalid readings."""
        try:
            rsi = require_finite_real("rsi", self.rsi)
        except (TypeError, ValueError):
            return SignalPolicyResult(veto=True, reasons=("unavailable",))

        if not 0.0 <= rsi <= 100.0:
            return SignalPolicyResult(veto=True, reasons=("out-of-range",))
        if signal.action is SignalAction.BUY and rsi > self.overbought:
            return SignalPolicyResult(veto=True, reasons=("overbought",))
        if signal.action is SignalAction.SELL and rsi < self.oversold:
            return SignalPolicyResult(veto=True, reasons=("oversold",))
        return SignalPolicyResult()


__all__ = ["RsiGatePolicy"]
