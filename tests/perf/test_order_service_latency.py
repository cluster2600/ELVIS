"""Low-overhead tripwire for the infrastructure-free order service."""

import gc
import platform
import sys
from datetime import datetime, timezone
from decimal import Decimal
from time import perf_counter_ns

import pytest

from trading.application.order_service import OrderService
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)

SAMPLE_COUNT = 10_000
WARM_UP_COUNT = 1_000
P99_CEILING_NS = 1_000_000


class NoOpExecutionPort:
    def __init__(self, report: SubmissionReport) -> None:
        self.report = report
        self.calls = 0

    def submit(self, intent: OrderIntent, /) -> SubmissionReport:
        self.calls += 1
        return self.report


def percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(quantile * (len(ordered) - 1)))
    return ordered[index]


@pytest.mark.perf
def test_order_service_p99_overhead_is_below_one_millisecond() -> None:
    intent = OrderIntent(
        client_order_id="latency-order-1",
        decision_id="latency-decision-1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("0.001"),
        order_type=OrderType.MARKET,
        reference_price=Decimal("123456.75"),
        leverage=3,
        created_at=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
    )
    report = SubmissionReport(
        client_order_id=intent.client_order_id,
        status=SubmissionStatus.SUBMITTED,
        retry_safety=RetrySafety.UNSAFE,
        venue_order_id="paper-order-1",
        venue_status="FILLED",
    )
    execution = NoOpExecutionPort(report)
    service = OrderService(execution)

    for _ in range(WARM_UP_COUNT):
        service.submit(intent)

    latencies_ns = []
    gc_enabled = gc.isenabled()
    for _ in range(SAMPLE_COUNT):
        started_ns = perf_counter_ns()
        service.submit(intent)
        latencies_ns.append(perf_counter_ns() - started_ns)

    p99_ns = percentile(latencies_ns, 0.99)
    print(
        "order-service "
        f"p99={p99_ns / 1_000:.2f}us samples={SAMPLE_COUNT} "
        f"warmup={WARM_UP_COUNT} clock=perf_counter_ns gc={gc_enabled} "
        f"python={platform.python_version()} implementation={sys.implementation.name} "
        f"os={platform.platform()} cpu={platform.processor() or platform.machine()}"
    )

    assert execution.calls == WARM_UP_COUNT + SAMPLE_COUNT
    assert p99_ns < P99_CEILING_NS
