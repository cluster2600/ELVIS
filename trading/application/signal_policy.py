"""Pure, fail-closed qualification of immutable strategy signals.

This synchronous core converts a policy-raised ``TimeoutError`` into a veto;
it cannot pre-empt a blocking I/O call. I/O-bound policies require a bounded
adapter in a later migration slice.
"""

from dataclasses import dataclass, replace
from typing import Protocol

from trading.domain._validation import require_clean_text, require_finite_real
from trading.domain.signals import Signal, SignalAction


@dataclass(frozen=True, slots=True)
class SignalPolicyResult:
    """A policy result which cannot promote or reverse a signal side."""

    veto: bool = False
    confidence: float | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.veto, bool):
            raise TypeError("veto must be a bool")
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple")
        for reason in self.reasons:
            require_clean_text("reason", reason)

        if self.confidence is not None:
            confidence = require_finite_real("confidence", self.confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)

        if self.veto and self.confidence is not None:
            raise ValueError("a veto cannot set confidence")
        if (self.veto or self.confidence is not None) and not self.reasons:
            raise ValueError("a veto or confidence adjustment requires a reason")


class SignalPolicy(Protocol):
    """Named, side-effect-free qualification step."""

    policy_id: str

    def evaluate(self, signal: Signal, /) -> SignalPolicyResult:
        """Evaluate a signal without mutating it or executing an order."""
        ...


class SignalPolicyPipeline:
    """Apply named policies once, in order, and fail closed on faults."""

    __slots__ = ("_policies", "_policy_ids")

    def __init__(self, policies: tuple[SignalPolicy, ...] = ()) -> None:
        if not isinstance(policies, tuple):
            raise TypeError("policies must be a tuple")

        policy_ids: list[str] = []
        for policy in policies:
            if not callable(getattr(policy, "evaluate", None)):
                raise TypeError("each policy must provide a callable evaluate method")
            policy_id = getattr(policy, "policy_id", None)
            require_clean_text("policy_id", policy_id)
            if ":" in policy_id:
                raise ValueError("policy_id cannot contain ':'")
            policy_ids.append(policy_id)

        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("policy IDs must be unique")

        self._policies = policies
        self._policy_ids = tuple(policy_ids)

    @property
    def policy_ids(self) -> tuple[str, ...]:
        """Return the immutable execution order."""
        return self._policy_ids

    def evaluate(self, signal: Signal, /) -> Signal:
        """Qualify one signal; policy faults become non-actionable signals."""
        if not isinstance(signal, Signal):
            raise TypeError("signal must be a Signal")
        if signal.action is SignalAction.HOLD or not self._policies:
            return signal

        current = signal
        for policy_id, policy in zip(self._policy_ids, self._policies):
            try:
                result = policy.evaluate(current)
            except TimeoutError:
                return self._veto(current, f"policy:{policy_id}:timeout")
            except Exception:
                return self._veto(current, f"policy:{policy_id}:exception")

            if not isinstance(result, SignalPolicyResult):
                return self._veto(current, f"policy:{policy_id}:invalid-result")

            try:
                result = SignalPolicyResult(
                    veto=result.veto,
                    confidence=result.confidence,
                    reasons=result.reasons,
                )
            except Exception:
                return self._veto(current, f"policy:{policy_id}:invalid-result")

            reasons = tuple(f"policy:{policy_id}:{reason}" for reason in result.reasons)
            if result.veto:
                return replace(
                    current,
                    action=SignalAction.HOLD,
                    confidence=0.0,
                    reasons=current.reasons + reasons,
                )
            if result.confidence is not None:
                current = replace(
                    current,
                    confidence=result.confidence,
                    reasons=current.reasons + reasons,
                )
            elif reasons:
                current = replace(current, reasons=current.reasons + reasons)

        return current

    @staticmethod
    def _veto(signal: Signal, reason: str) -> Signal:
        return replace(
            signal,
            action=SignalAction.HOLD,
            confidence=0.0,
            reasons=signal.reasons + (reason,),
        )


__all__ = ["SignalPolicy", "SignalPolicyPipeline", "SignalPolicyResult"]
