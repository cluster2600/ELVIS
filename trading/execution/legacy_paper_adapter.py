"""Typed, paper-only boundary around the current dictionary executor."""

import math
from collections.abc import Mapping
from typing import Protocol

from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)


class _LegacyPaperExecutor(Protocol):
    default_leverage: int

    def execute_buy(self, symbol: str, quantity: float, price: float) -> object: ...

    def execute_sell(self, symbol: str, quantity: float, price: float) -> object: ...


class LegacyPaperExecutionAdapter:
    """Map the current executor's paper results to submission contracts."""

    supports_live_submission = False
    __slots__ = ("_executor", "_runtime_mode", "_default_leverage")

    def __init__(self, executor: _LegacyPaperExecutor, runtime_mode: str) -> None:
        for method_name in ("execute_buy", "execute_sell"):
            if not callable(getattr(executor, method_name, None)):
                raise TypeError(
                    f"legacy executor must provide a callable {method_name} method"
                )

        default_leverage = getattr(executor, "default_leverage", None)
        if isinstance(default_leverage, bool) or not isinstance(default_leverage, int):
            raise TypeError("legacy executor default_leverage must be an integer")
        if default_leverage < 1:
            raise ValueError("legacy executor default_leverage must be positive")
        if not isinstance(runtime_mode, str) or not runtime_mode:
            raise TypeError("runtime_mode must be a non-empty string")

        self._executor = executor
        self._runtime_mode = runtime_mode
        self._default_leverage = default_leverage

    @property
    def default_leverage(self) -> int:
        return self._default_leverage

    def submit(self, intent: OrderIntent, /) -> SubmissionReport:
        """Make one legacy call in paper mode and classify its result."""
        if not isinstance(intent, OrderIntent):
            raise TypeError("intent must be an OrderIntent")
        if self._runtime_mode != "paper":
            return self._not_sent(intent, "legacy executor is paper-only")
        if intent.leverage != self._default_leverage:
            return self._not_sent(
                intent,
                "intent leverage does not match the legacy executor leverage",
            )

        try:
            quantity = float(intent.quantity)
            reference_price = float(intent.reference_price)
        except (OverflowError, ValueError):
            return self._not_sent(intent, "order values cannot be represented as float")
        if (
            not math.isfinite(quantity)
            or not math.isfinite(reference_price)
            or quantity <= 0.0
            or reference_price <= 0.0
        ):
            return self._not_sent(intent, "order values cannot be represented as float")

        if intent.side is OrderSide.BUY:
            result = self._executor.execute_buy(
                intent.symbol, quantity, reference_price
            )
        else:
            result = self._executor.execute_sell(
                intent.symbol, quantity, reference_price
            )

        return self._map_result(intent, result)

    def _map_result(self, intent: OrderIntent, result: object) -> SubmissionReport:
        if not isinstance(result, Mapping) or not result:
            return self._ambiguous(
                intent, "legacy executor returned no usable response"
            )

        raw_status = result.get("status")
        if not isinstance(raw_status, str) or not raw_status.strip():
            return self._ambiguous(intent, "legacy executor returned no usable status")
        status = raw_status.strip().upper()

        if status == "BLOCKED":
            return self._not_sent(intent, "legacy risk management blocked the order")
        if status == "REJECTED":
            return SubmissionReport(
                client_order_id=intent.client_order_id,
                status=SubmissionStatus.VENUE_REJECTED,
                retry_safety=RetrySafety.SAFE,
                reason="legacy executor rejected the order",
                venue_status="REJECTED",
            )
        if status != "FILLED":
            return self._ambiguous(intent, "legacy executor returned an unknown status")

        if not self._matches_text(result.get("symbol"), intent.symbol):
            return self._ambiguous(intent, "legacy executor returned another symbol")
        if not self._matches_text(result.get("side"), intent.side.value):
            return self._ambiguous(intent, "legacy executor returned another side")

        raw_order_id = result.get("orderId")
        if isinstance(raw_order_id, bool) or raw_order_id is None:
            return self._ambiguous(intent, "filled legacy response has no order ID")
        venue_order_id = str(raw_order_id)
        if not venue_order_id or venue_order_id != venue_order_id.strip():
            return self._ambiguous(intent, "filled legacy response has no order ID")

        return SubmissionReport(
            client_order_id=intent.client_order_id,
            status=SubmissionStatus.SUBMITTED,
            retry_safety=RetrySafety.UNSAFE,
            venue_order_id=venue_order_id,
            venue_status="FILLED",
        )

    @staticmethod
    def _matches_text(value: object, expected: str) -> bool:
        return isinstance(value, str) and value == expected

    @staticmethod
    def _not_sent(intent: OrderIntent, reason: str) -> SubmissionReport:
        return SubmissionReport(
            client_order_id=intent.client_order_id,
            status=SubmissionStatus.NOT_SENT,
            retry_safety=RetrySafety.SAFE,
            reason=reason,
        )

    @staticmethod
    def _ambiguous(intent: OrderIntent, reason: str) -> SubmissionReport:
        return SubmissionReport(
            client_order_id=intent.client_order_id,
            status=SubmissionStatus.AMBIGUOUS,
            retry_safety=RetrySafety.UNSAFE,
            reason=reason,
        )
