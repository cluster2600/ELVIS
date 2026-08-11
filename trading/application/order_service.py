"""Single-attempt order submission orchestration."""

from typing import Protocol

from trading.domain.orders import (
    OrderIntent,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)


class ExecutionPort(Protocol):
    """Narrow boundary implemented by paper, replay, or venue adapters."""

    def submit(self, intent: OrderIntent, /) -> SubmissionReport:
        """Attempt to submit one intent and return what is known."""
        ...


class OrderService:
    """Make one adapter call without retrying or inferring fill state."""

    __slots__ = ("_execution",)

    def __init__(self, execution: ExecutionPort) -> None:
        if not callable(getattr(execution, "submit", None)):
            raise TypeError("execution must provide a callable submit method")
        self._execution = execution

    def submit(self, intent: OrderIntent) -> SubmissionReport:
        """Submit once; conservatively classify unexpected outcomes."""
        if not isinstance(intent, OrderIntent):
            raise TypeError("intent must be an OrderIntent")

        try:
            report = self._execution.submit(intent)
        except Exception as exc:
            return self._ambiguous(
                intent,
                reason=f"execution adapter raised {type(exc).__name__}",
            )

        if not isinstance(report, SubmissionReport):
            return self._ambiguous(
                intent,
                reason="execution adapter returned an invalid response",
            )

        if report.client_order_id != intent.client_order_id:
            return self._ambiguous(
                intent,
                reason="execution adapter returned a mismatched client order ID",
            )

        return report

    @staticmethod
    def _ambiguous(
        intent: OrderIntent,
        *,
        reason: str,
    ) -> SubmissionReport:
        return SubmissionReport(
            client_order_id=intent.client_order_id,
            status=SubmissionStatus.AMBIGUOUS,
            retry_safety=RetrySafety.UNSAFE,
            reason=reason,
        )
