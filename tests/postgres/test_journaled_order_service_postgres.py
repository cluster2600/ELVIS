"""PostgreSQL composition tests for the unwired journaled order service."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from threading import Barrier, Lock

import psycopg2

from trading.application.journaled_order_service import (
    JournaledOrderService,
    JournaledSubmissionDisposition,
    SubmissionObservationNotRecorded,
)
from trading.application.order_service import OrderService
from trading.domain.order_lifecycle import (
    OrderLifecycleState,
    SubmissionAcknowledged,
)
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionInstruction,
    TakeProfitProfile,
)
from trading.persistence.order_position_journal import (
    EventAppendDisposition,
    JournalCommitUnknown,
    PostgresOrderPositionJournal,
)

SCOPE = "paper:journaled-service"
POSITION_KEY = "position-journaled-service"
NOW = datetime(2026, 8, 12, 12, 0, 0, 123456, tzinfo=timezone.utc)


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _journal(dsn):
    return PostgresOrderPositionJournal(lambda: _connect(dsn))


def _instruction() -> PositionInstruction:
    return PositionInstruction(
        position_key=POSITION_KEY,
        effect=PositionEffect.OPEN,
        order_intent=OrderIntent(
            client_order_id="order-journaled-service",
            decision_id="decision-journaled-service",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.00"),
            order_type=OrderType.MARKET,
            reference_price=Decimal("50000.00"),
            leverage=3,
            created_at=NOW,
        ),
        exit_context=PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.0025"),
            stop_loss_fraction=Decimal("0.005"),
        ),
    )


class _FixedClock:
    def __init__(self) -> None:
        self.calls = 0
        self._lock = Lock()

    def now(self) -> datetime:
        with self._lock:
            self.calls += 1
        return NOW


class _InspectingExecution:
    def __init__(self, dsn, status: SubmissionStatus) -> None:
        self._dsn = dsn
        self._status = status
        self._lock = Lock()
        self.calls: list[OrderIntent] = []
        self.visible_rows: list[tuple[int, int]] = []

    def submit(self, intent: OrderIntent, /) -> SubmissionReport:
        connection = _connect(self._dsn)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM np.orders
                    WHERE client_order_id = %s
                    """,
                    (intent.client_order_id,),
                )
                order_count = cursor.fetchone()[0]
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM np.order_events
                    WHERE client_order_id = %s
                    """,
                    (intent.client_order_id,),
                )
                event_count = cursor.fetchone()[0]
        finally:
            connection.close()

        with self._lock:
            self.calls.append(intent)
            self.visible_rows.append((order_count, event_count))

        if self._status is SubmissionStatus.SUBMITTED:
            return SubmissionReport(
                client_order_id=intent.client_order_id,
                status=SubmissionStatus.SUBMITTED,
                retry_safety=RetrySafety.UNSAFE,
                venue_order_id="venue-journaled-service",
            )
        if self._status is SubmissionStatus.AMBIGUOUS:
            return SubmissionReport(
                client_order_id=intent.client_order_id,
                status=SubmissionStatus.AMBIGUOUS,
                retry_safety=RetrySafety.UNSAFE,
                reason="venue acknowledgement was not observable",
            )
        raise AssertionError("unsupported test submission status")


def _service(journal, execution, clock):
    return JournaledOrderService(
        order_service=OrderService(execution),
        journal=journal,
        clock=clock,
    )


def _event_rows(dsn):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT event_type, position_version
                FROM np.order_events
                WHERE position_key = %s
                ORDER BY position_version
                """,
                (POSITION_KEY,),
            )
            return cursor.fetchall()
    finally:
        connection.close()


def test_reservation_commits_before_submit_and_exact_service_retry_is_read_only(
    migrated_postgres_dsn,
):
    journal = _journal(migrated_postgres_dsn)
    execution = _InspectingExecution(
        migrated_postgres_dsn,
        SubmissionStatus.SUBMITTED,
    )
    clock = _FixedClock()
    service = _service(journal, execution, clock)
    instruction = _instruction()

    first = service.submit(instruction, execution_scope=SCOPE)
    second = service.submit(instruction, execution_scope=SCOPE)

    assert first.disposition is JournaledSubmissionDisposition.RECORDED
    assert first.report is not None
    assert first.report.status is SubmissionStatus.SUBMITTED
    assert first.position_version == 1
    assert second.disposition is (JournaledSubmissionDisposition.EXISTING_RESERVATION)
    assert second.report is None
    assert second.event is None
    assert second.requires_reconciliation is True
    assert execution.calls == [instruction.order_intent]
    assert execution.visible_rows == [(1, 0)]
    assert clock.calls == 1
    assert _event_rows(migrated_postgres_dsn) == [("SUBMISSION_ACKNOWLEDGED", 1)]

    projection = journal.replay_position(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
    )
    assert projection.stream_version == 1
    assert projection.orders[0].lifecycle.state is OrderLifecycleState.OPEN


def test_concurrent_exact_submissions_execute_once(migrated_postgres_dsn):
    journal = _journal(migrated_postgres_dsn)
    execution = _InspectingExecution(
        migrated_postgres_dsn,
        SubmissionStatus.SUBMITTED,
    )
    clock = _FixedClock()
    service = _service(journal, execution, clock)
    instruction = _instruction()
    ready = Barrier(2)

    def submit():
        ready.wait(timeout=10)
        return service.submit(instruction, execution_scope=SCOPE)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(submit) for _ in range(2))
        results = tuple(future.result(timeout=20) for future in futures)

    assert {result.disposition for result in results} == {
        JournaledSubmissionDisposition.RECORDED,
        JournaledSubmissionDisposition.EXISTING_RESERVATION,
    }
    assert execution.calls == [instruction.order_intent]
    assert execution.visible_rows == [(1, 0)]
    assert clock.calls == 1
    assert _event_rows(migrated_postgres_dsn) == [("SUBMISSION_ACKNOWLEDGED", 1)]


def test_ambiguous_report_is_durably_reconciling(migrated_postgres_dsn):
    journal = _journal(migrated_postgres_dsn)
    execution = _InspectingExecution(
        migrated_postgres_dsn,
        SubmissionStatus.AMBIGUOUS,
    )
    service = _service(journal, execution, _FixedClock())

    result = service.submit(_instruction(), execution_scope=SCOPE)

    assert result.disposition is JournaledSubmissionDisposition.RECORDED
    assert result.report is not None
    assert result.report.status is SubmissionStatus.AMBIGUOUS
    assert result.requires_reconciliation is True
    assert _event_rows(migrated_postgres_dsn) == [("SUBMISSION_AMBIGUOUS", 1)]
    projection = journal.replay_position(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
    )
    assert projection.orders[0].lifecycle.state is (OrderLifecycleState.RECONCILING)


class _CommitAcknowledgementLost:
    def __init__(self, connection) -> None:
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def commit(self) -> None:
        self._connection.commit()
        raise psycopg2.OperationalError("simulated lost commit acknowledgement")


class _LoseSecondCommitFactory:
    def __init__(self, dsn) -> None:
        self._dsn = dsn
        self._calls = 0

    def __call__(self):
        self._calls += 1
        connection = _connect(self._dsn)
        if self._calls == 2:
            return _CommitAcknowledgementLost(connection)
        return connection


def test_append_commit_unknown_can_be_resolved_without_resubmission(
    migrated_postgres_dsn,
):
    journal = PostgresOrderPositionJournal(
        _LoseSecondCommitFactory(migrated_postgres_dsn)
    )
    execution = _InspectingExecution(
        migrated_postgres_dsn,
        SubmissionStatus.SUBMITTED,
    )
    service = _service(journal, execution, _FixedClock())
    instruction = _instruction()

    try:
        service.submit(instruction, execution_scope=SCOPE)
    except SubmissionObservationNotRecorded as exc:
        failure = exc
    else:
        raise AssertionError("lost append acknowledgement was reported as success")

    assert isinstance(failure.__cause__, JournalCommitUnknown)
    assert failure.report.status is SubmissionStatus.SUBMITTED
    assert isinstance(failure.event, SubmissionAcknowledged)
    assert execution.calls == [instruction.order_intent]

    normal_journal = _journal(migrated_postgres_dsn)
    resolved = normal_journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id=failure.event_id,
        event=failure.event,
    )
    assert resolved.disposition is EventAppendDisposition.EXISTING_EVENT_ID
    assert resolved.position_version == 1

    existing = service.submit(instruction, execution_scope=SCOPE)
    assert existing.disposition is (JournaledSubmissionDisposition.EXISTING_RESERVATION)
    assert execution.calls == [instruction.order_intent]
    assert _event_rows(migrated_postgres_dsn) == [("SUBMISSION_ACKNOWLEDGED", 1)]
