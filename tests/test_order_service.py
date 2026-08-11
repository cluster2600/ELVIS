import ast
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from trading.application.order_service import ExecutionPort, OrderService
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)


def make_intent(**overrides: object) -> OrderIntent:
    values = {
        "client_order_id": "elvis-order-1",
        "decision_id": "decision-1",
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


def make_report(status: SubmissionStatus) -> SubmissionReport:
    if status is SubmissionStatus.SUBMITTED:
        return SubmissionReport(
            client_order_id="elvis-order-1",
            status=status,
            retry_safety=RetrySafety.UNSAFE,
            venue_order_id="paper-order-1",
            venue_status="FILLED",
        )
    return SubmissionReport(
        client_order_id="elvis-order-1",
        status=status,
        retry_safety=(
            RetrySafety.SAFE
            if status is not SubmissionStatus.AMBIGUOUS
            else RetrySafety.UNSAFE
        ),
        reason=f"{status.value.lower()} outcome",
        venue_status=(
            "REJECTED" if status is SubmissionStatus.VENUE_REJECTED else None
        ),
    )


class FakeExecutionPort:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[OrderIntent] = []

    def submit(self, intent: OrderIntent, /) -> SubmissionReport:
        self.calls.append(intent)
        return self.response  # type: ignore[return-value]


class RaisingExecutionPort:
    def __init__(self) -> None:
        self.calls = 0

    def submit(self, intent: OrderIntent, /) -> SubmissionReport:
        self.calls += 1
        raise TimeoutError("private venue details must not leak")


@pytest.mark.parametrize("execution", [object(), {"submit": "not-callable"}])
def test_order_service_rejects_an_invalid_execution_port(execution: object) -> None:
    with pytest.raises(TypeError):
        OrderService(execution)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", list(SubmissionStatus))
def test_order_service_returns_each_typed_outcome(status: SubmissionStatus) -> None:
    intent = make_intent()
    expected = make_report(status)
    execution = FakeExecutionPort(expected)

    actual = OrderService(execution).submit(intent)

    assert actual is expected
    assert execution.calls == [intent]


def test_order_service_passes_the_same_intent_to_the_port() -> None:
    intent = make_intent()
    execution = FakeExecutionPort(make_report(SubmissionStatus.SUBMITTED))

    OrderService(execution).submit(intent)

    assert execution.calls[0] is intent


def test_repeated_invocations_remain_explicit_until_idempotency_is_durable() -> None:
    intent = make_intent()
    execution = FakeExecutionPort(make_report(SubmissionStatus.SUBMITTED))
    service = OrderService(execution)

    first = service.submit(intent)
    second = service.submit(intent)

    assert first is second
    assert execution.calls == [intent, intent]


def test_order_service_rejects_an_untyped_intent_without_calling_the_port() -> None:
    execution = FakeExecutionPort(make_report(SubmissionStatus.SUBMITTED))

    with pytest.raises(TypeError):
        OrderService(execution).submit({"symbol": "BTCUSDT"})  # type: ignore[arg-type]

    assert execution.calls == []


def test_unexpected_adapter_exception_is_ambiguous_and_not_retried() -> None:
    execution = RaisingExecutionPort()

    report = OrderService(execution).submit(make_intent())

    assert execution.calls == 1
    assert report.status is SubmissionStatus.AMBIGUOUS
    assert report.retry_safety is RetrySafety.UNSAFE
    assert report.requires_reconciliation is True
    assert "TimeoutError" in report.reason
    assert "private venue details" not in report.reason


@pytest.mark.parametrize("malformed", [None, {}, True, object()])
def test_malformed_adapter_result_is_ambiguous(malformed: object) -> None:
    execution = FakeExecutionPort(malformed)

    report = OrderService(execution).submit(make_intent())

    assert len(execution.calls) == 1
    assert report.status is SubmissionStatus.AMBIGUOUS
    assert report.retry_safety is RetrySafety.UNSAFE
    assert report.reason == "execution adapter returned an invalid response"


def test_mismatched_client_order_id_is_ambiguous() -> None:
    execution = FakeExecutionPort(
        SubmissionReport(
            client_order_id="different-order",
            status=SubmissionStatus.SUBMITTED,
            retry_safety=RetrySafety.UNSAFE,
            venue_order_id="paper-order-1",
            venue_status="FILLED",
        )
    )

    report = OrderService(execution).submit(make_intent())

    assert len(execution.calls) == 1
    assert report.client_order_id == "elvis-order-1"
    assert report.status is SubmissionStatus.AMBIGUOUS
    assert report.venue_order_id is None
    assert report.venue_status is None


def test_execution_protocol_is_structural() -> None:
    execution: ExecutionPort = FakeExecutionPort(
        make_report(SubmissionStatus.SUBMITTED)
    )

    assert execution.submit(make_intent()).acknowledged is True


def test_application_package_has_only_standard_library_and_domain_imports() -> None:
    application_dir = Path(__file__).parents[1] / "trading" / "application"
    standard_library_roots = {"typing"}

    imported_modules: set[str] = set()
    for module_path in application_dir.rglob("*.py"):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

    unexpected = {
        module
        for module in imported_modules
        if module.split(".", 1)[0] not in standard_library_roots
        and not module.startswith("trading.application")
        and not module.startswith("trading.domain")
    }
    assert unexpected == set()
