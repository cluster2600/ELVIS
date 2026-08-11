from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trading.application.order_service import OrderService
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionStatus,
)
from trading.execution.legacy_paper_adapter import LegacyPaperExecutionAdapter


def make_intent(**overrides: object) -> OrderIntent:
    values = {
        "client_order_id": "ELV-0123456789abcdef0123456789abcdef",
        "decision_id": "0123456789abcdef0123456789abcdef",
        "symbol": "BTCUSDT",
        "side": OrderSide.BUY,
        "quantity": Decimal("0.001"),
        "order_type": OrderType.MARKET,
        "reference_price": Decimal("123456.75"),
        "leverage": 3,
        "created_at": datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return OrderIntent(**values)


class LegacyExecutor:
    def __init__(self, result: object, *, default_leverage: int = 3) -> None:
        self.result = result
        self.default_leverage = default_leverage
        self.buy_calls: list[tuple[str, float, float]] = []
        self.sell_calls: list[tuple[str, float, float]] = []

    def execute_buy(self, symbol: str, quantity: float, price: float) -> object:
        self.buy_calls.append((symbol, quantity, price))
        return self.result

    def execute_sell(self, symbol: str, quantity: float, price: float) -> object:
        self.sell_calls.append((symbol, quantity, price))
        return self.result


class RaisingExecutor(LegacyExecutor):
    def execute_buy(self, symbol: str, quantity: float, price: float) -> object:
        self.buy_calls.append((symbol, quantity, price))
        raise TimeoutError("uncertain paper database outcome")


def filled_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "symbol": "BTCUSDT",
        "orderId": "MOCK_BTCUSDT_1",
        "side": "BUY",
        "status": "FILLED",
    }
    result.update(overrides)
    return result


def test_buy_submission_maps_filled_result_and_converts_decimal_at_boundary() -> None:
    executor = LegacyExecutor(filled_result())
    adapter = LegacyPaperExecutionAdapter(executor, runtime_mode="paper")

    report = adapter.submit(make_intent())

    assert executor.buy_calls == [("BTCUSDT", 0.001, 123456.75)]
    assert executor.sell_calls == []
    assert report.status is SubmissionStatus.SUBMITTED
    assert report.retry_safety is RetrySafety.UNSAFE
    assert report.client_order_id == "ELV-0123456789abcdef0123456789abcdef"
    assert report.venue_order_id == "MOCK_BTCUSDT_1"
    assert report.venue_status == "FILLED"


def test_sell_submission_uses_only_the_sell_method() -> None:
    executor = LegacyExecutor(filled_result(side="SELL"))
    adapter = LegacyPaperExecutionAdapter(executor, runtime_mode="paper")

    report = adapter.submit(make_intent(side=OrderSide.SELL))

    assert report.acknowledged is True
    assert executor.buy_calls == []
    assert executor.sell_calls == [("BTCUSDT", 0.001, 123456.75)]


def test_live_mode_is_rejected_before_any_executor_call() -> None:
    executor = LegacyExecutor(filled_result())
    adapter = LegacyPaperExecutionAdapter(executor, runtime_mode="live")

    report = adapter.submit(make_intent())

    assert report.status is SubmissionStatus.NOT_SENT
    assert report.retry_safety is RetrySafety.SAFE
    assert "paper-only" in report.reason
    assert executor.buy_calls == []
    assert executor.sell_calls == []


def test_leverage_mismatch_is_rejected_before_any_executor_call() -> None:
    executor = LegacyExecutor(filled_result(), default_leverage=2)
    adapter = LegacyPaperExecutionAdapter(executor, runtime_mode="paper")

    report = adapter.submit(make_intent(leverage=3))

    assert report.status is SubmissionStatus.NOT_SENT
    assert report.retry_safety is RetrySafety.SAFE
    assert "leverage" in report.reason
    assert executor.buy_calls == []


def test_float_overflow_is_rejected_before_any_executor_call() -> None:
    executor = LegacyExecutor(filled_result())
    adapter = LegacyPaperExecutionAdapter(executor, runtime_mode="paper")

    report = adapter.submit(make_intent(quantity=Decimal("1e10000")))

    assert report.status is SubmissionStatus.NOT_SENT
    assert executor.buy_calls == []


@pytest.mark.parametrize("field", ["quantity", "reference_price"])
def test_float_underflow_is_rejected_before_any_executor_call(field: str) -> None:
    executor = LegacyExecutor(filled_result())
    adapter = LegacyPaperExecutionAdapter(executor, runtime_mode="paper")

    report = adapter.submit(make_intent(**{field: Decimal("1e-10000")}))

    assert report.status is SubmissionStatus.NOT_SENT
    assert executor.buy_calls == []


def test_truthy_blocked_result_is_not_acknowledged() -> None:
    executor = LegacyExecutor({"status": "BLOCKED", "reason": "Risk management"})

    report = LegacyPaperExecutionAdapter(executor, "paper").submit(make_intent())

    assert report.status is SubmissionStatus.NOT_SENT
    assert report.acknowledged is False
    assert report.retry_safety is RetrySafety.SAFE
    assert report.venue_order_id is None
    assert report.venue_status is None


def test_explicit_rejection_is_a_typed_venue_rejection() -> None:
    executor = LegacyExecutor({"status": "REJECTED", "reason": "bad order"})

    report = LegacyPaperExecutionAdapter(executor, "paper").submit(make_intent())

    assert report.status is SubmissionStatus.VENUE_REJECTED
    assert report.retry_safety is RetrySafety.SAFE
    assert report.venue_order_id is None
    assert report.venue_status == "REJECTED"


@pytest.mark.parametrize(
    "result",
    [None, {}, True, {"status": "UNKNOWN"}, {"status": "FILLED"}],
)
def test_uncertain_legacy_result_is_ambiguous(result: object) -> None:
    executor = LegacyExecutor(result)

    report = LegacyPaperExecutionAdapter(executor, "paper").submit(make_intent())

    assert report.status is SubmissionStatus.AMBIGUOUS
    assert report.retry_safety is RetrySafety.UNSAFE
    assert report.requires_reconciliation is True


@pytest.mark.parametrize(
    "result",
    [
        filled_result(symbol="ETHUSDT"),
        filled_result(side="SELL"),
        filled_result(orderId=" "),
        filled_result(orderId=True),
    ],
)
def test_incoherent_filled_result_is_ambiguous(result: object) -> None:
    executor = LegacyExecutor(result)

    report = LegacyPaperExecutionAdapter(executor, "paper").submit(make_intent())

    assert report.status is SubmissionStatus.AMBIGUOUS
    assert report.venue_order_id is None


@pytest.mark.parametrize("missing_field", ["symbol", "side"])
def test_incomplete_filled_result_is_ambiguous(missing_field: str) -> None:
    result = filled_result()
    result.pop(missing_field)
    executor = LegacyExecutor(result)

    report = LegacyPaperExecutionAdapter(executor, "paper").submit(make_intent())

    assert report.status is SubmissionStatus.AMBIGUOUS
    assert report.venue_order_id is None


def test_adapter_exception_is_classified_once_by_order_service() -> None:
    executor = RaisingExecutor(filled_result())
    service = OrderService(LegacyPaperExecutionAdapter(executor, "paper"))

    report = service.submit(make_intent())

    assert len(executor.buy_calls) == 1
    assert report.status is SubmissionStatus.AMBIGUOUS
    assert report.retry_safety is RetrySafety.UNSAFE


@pytest.mark.parametrize(
    "executor",
    [object(), LegacyExecutor(filled_result(), default_leverage=0)],
)
def test_adapter_rejects_an_invalid_legacy_executor(executor: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        LegacyPaperExecutionAdapter(executor, "paper")
