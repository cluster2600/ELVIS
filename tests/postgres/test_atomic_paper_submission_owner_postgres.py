"""PostgreSQL 15 proofs for the atomic terminal paper-submission owner."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Barrier, Lock

import psycopg2
import pytest

from trading.application.durable_submission import (
    DurableSubmissionDisposition,
    PaperPlannedFill,
    PaperSubmissionPlan,
    SubmissionAttemptContext,
    SubmissionCommitUnknown,
    SubmissionReconciliationRequired,
)
from trading.domain.order_lifecycle import ConfirmedFill, SubmissionAcknowledged
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionInstruction,
    TakeProfitProfile,
)
from trading.persistence.atomic_paper_submission_owner import (
    PostgresAtomicPaperSubmissionOwner,
)
from trading.persistence.order_position_journal import (
    JournalConflictError,
    JournalConflictKind,
    JournalReplayError,
    JournalStorageError,
    PostgresOrderPositionJournal,
)

SCOPE = "paper:atomic-owner"
POSITION_KEY = "position-atomic-owner"
NOW = datetime(2026, 8, 12, 12, 0, 0, 123456, tzinfo=timezone.utc)
FIRST_QUANTITY = Decimal("0.40000000000000000001")
SECOND_QUANTITY = Decimal("0.59999999999999999999")


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _instruction(
    *,
    client_order_id="order-atomic-1",
    decision_id="decision-atomic-1",
    position_key=POSITION_KEY,
    quantity=Decimal("1.00000000000000000000"),
):
    return PositionInstruction(
        position_key=position_key,
        effect=PositionEffect.OPEN,
        order_intent=OrderIntent(
            client_order_id=client_order_id,
            decision_id=decision_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            reference_price=Decimal("50000.00"),
            leverage=3,
            created_at=NOW,
        ),
        exit_context=PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.0025"),
            stop_loss_fraction=Decimal("0.005"),
            trailing_stop_fraction=Decimal("0.02"),
        ),
    )


def _attempt(instruction=None):
    selected = instruction or _instruction()
    return SubmissionAttemptContext.first(selected, SCOPE, NOW)


def _ack(attempt, *, venue_order_id=None):
    return SubmissionAcknowledged(
        client_order_id=attempt.client_order_id,
        venue_order_id=venue_order_id or f"venue-{attempt.client_order_id}",
        observed_at=attempt.observed_at,
    )


def _fill(
    attempt,
    *,
    trade_id,
    quantity,
    seconds,
    venue_order_id=None,
):
    intent = attempt.instruction.order_intent
    return ConfirmedFill(
        client_order_id=intent.client_order_id,
        venue_order_id=venue_order_id or f"venue-{intent.client_order_id}",
        trade_id=trade_id,
        symbol=intent.symbol,
        side=intent.side,
        quantity=quantity,
        price=Decimal("50001.25000000000000000001") + Decimal(seconds),
        fee_amount=Decimal("0.00000000000000000001"),
        fee_asset="USDT",
        executed_at=attempt.observed_at + timedelta(seconds=seconds),
    )


def _plan(attempt):
    client_order_id = attempt.client_order_id
    return PaperSubmissionPlan(
        attempt=attempt,
        submission=_ack(attempt),
        fills=(
            PaperPlannedFill(
                event_id="fill-observation-1",
                fill=_fill(
                    attempt,
                    trade_id=f"trade-{client_order_id}-1",
                    quantity=FIRST_QUANTITY,
                    seconds=1,
                ),
            ),
            PaperPlannedFill(
                event_id="fill-observation-2",
                fill=_fill(
                    attempt,
                    trade_id=f"trade-{client_order_id}-2",
                    quantity=SECOND_QUANTITY,
                    seconds=2,
                ),
            ),
        ),
    )


class _CountingPlanner:
    def __init__(self):
        self.calls = []
        self._lock = Lock()

    def plan(self, attempt, /):
        with self._lock:
            self.calls.append(attempt)
        return _plan(attempt)


class _ExplodingPlanner:
    def __init__(self, error=None):
        self.error = error or RuntimeError("planner must not run")
        self.calls = []

    def plan(self, attempt, /):
        self.calls.append(attempt)
        raise self.error


class _TrackingConnection:
    def __init__(self, connection, *, statements=None):
        self._connection = connection
        self._statements = statements
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def cursor(self):
        cursor = self._connection.cursor()
        if self._statements is None:
            return cursor
        return _TracingCursor(cursor, self._statements)

    def commit(self):
        self.commits += 1
        return self._connection.commit()

    def rollback(self):
        self.rollbacks += 1
        return self._connection.rollback()

    def close(self):
        self.closed = True
        return self._connection.close()


class _TrackingFactory:
    def __init__(self, dsn, *, statements=None, connection_type=_TrackingConnection):
        self._dsn = dsn
        self._statements = statements
        self._connection_type = connection_type
        self.connections = []
        self._lock = Lock()

    def __call__(self):
        connection = self._connection_type(
            _connect(self._dsn),
            statements=self._statements,
        )
        with self._lock:
            self.connections.append(connection)
        return connection


class _TracingCursor:
    def __init__(self, cursor, statements):
        self._cursor = cursor
        self._statements = statements

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._cursor.__exit__(exc_type, exc_value, traceback)

    def execute(self, statement, parameters=None):
        normalized = " ".join(str(statement).split())
        self._statements.append(normalized)
        if parameters is None:
            return self._cursor.execute(statement)
        return self._cursor.execute(statement, parameters)


class _MutationFailureCursor(_TracingCursor):
    def __init__(self, cursor, statements, fail_after):
        super().__init__(cursor, statements)
        self._fail_after = fail_after
        self.mutations = 0

    def execute(self, statement, parameters=None):
        result = super().execute(statement, parameters)
        normalized = " ".join(str(statement).split()).upper()
        if normalized.startswith(("INSERT ", "UPDATE ", "DELETE ")):
            self.mutations += 1
            if self.mutations == self._fail_after:
                raise RuntimeError(
                    f"injected failure after SQL mutation {self._fail_after}"
                )
        return result


class _MutationFailureConnection(_TrackingConnection):
    def __init__(self, connection, *, statements, fail_after):
        super().__init__(connection, statements=statements)
        self._fail_after = fail_after
        self.failure_cursor = None

    def cursor(self):
        self.failure_cursor = _MutationFailureCursor(
            self._connection.cursor(),
            self._statements,
            self._fail_after,
        )
        return self.failure_cursor


class _MutationFailureFactory:
    def __init__(self, dsn, fail_after):
        self._dsn = dsn
        self._fail_after = fail_after
        self.statements = []
        self.connection = None

    def __call__(self):
        self.connection = _MutationFailureConnection(
            _connect(self._dsn),
            statements=self.statements,
            fail_after=self._fail_after,
        )
        return self.connection


class _CommitThenRaiseConnection(_TrackingConnection):
    def commit(self):
        self.commits += 1
        self._connection.commit()
        raise psycopg2.OperationalError("simulated lost commit acknowledgement")


def _owner(dsn, planner, factory=None):
    selected_factory = factory or (lambda: _connect(dsn))
    return PostgresAtomicPaperSubmissionOwner(selected_factory, planner)


def _journal(dsn):
    return PostgresOrderPositionJournal(lambda: _connect(dsn))


def _journal_snapshot(dsn):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    position_key,
                    execution_scope,
                    stream_version,
                    created_at,
                    xmin::text,
                    ctid::text
                FROM np.position_streams
                ORDER BY position_key
                """)
            streams = cursor.fetchall()
            cursor.execute("""
                SELECT
                    client_order_id,
                    decision_id,
                    position_key,
                    execution_scope,
                    instruction_payload,
                    venue_order_id,
                    registered_at,
                    xmin::text,
                    ctid::text
                FROM np.orders
                ORDER BY client_order_id
                """)
            orders = cursor.fetchall()
            cursor.execute("""
                SELECT
                    position_key,
                    position_version,
                    client_order_id,
                    event_id,
                    event_type,
                    event_payload,
                    trade_id,
                    occurred_at,
                    recorded_at,
                    xmin::text,
                    ctid::text
                FROM np.order_events
                ORDER BY position_key, position_version
                """)
            events = cursor.fetchall()
            return streams, orders, events
    finally:
        connection.close()


def _legacy_counts(dsn):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            values = []
            for table in (
                "trades",
                "open_positions",
                "liquidations",
                "margin_history",
                "trading_session_resets",
                "model_predictions",
                "account_balances",
            ):
                cursor.execute(f"SELECT COUNT(*) FROM np.{table}")
                values.append((table, cursor.fetchone()[0]))
            return tuple(values)
    finally:
        connection.close()


def _event_rows(dsn, position_key=POSITION_KEY):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    position_version,
                    client_order_id,
                    event_id,
                    event_type,
                    event_payload
                FROM np.order_events
                WHERE position_key = %s
                ORDER BY position_version
                """,
                (position_key,),
            )
            return cursor.fetchall()
    finally:
        connection.close()


def test_two_fill_batch_has_one_connection_one_commit_and_exact_versions(
    migrated_postgres_dsn,
):
    planner = _CountingPlanner()
    statements = []
    factory = _TrackingFactory(migrated_postgres_dsn, statements=statements)
    attempt = _attempt()

    receipt = _owner(migrated_postgres_dsn, planner, factory).execute(attempt)

    assert receipt.disposition is DurableSubmissionDisposition.COMMITTED
    assert receipt.attempt is attempt
    assert receipt.submission.position_version == 1
    assert tuple(fill.position_version for fill in receipt.fills) == (2, 3)
    assert tuple(fill.event.quantity for fill in receipt.fills) == (
        FIRST_QUANTITY,
        SECOND_QUANTITY,
    )
    assert planner.calls == [attempt]
    assert len(factory.connections) == 1
    assert factory.connections[0].commits == 1
    assert factory.connections[0].closed is True

    streams, orders, events = _journal_snapshot(migrated_postgres_dsn)
    assert streams[0][2] == 3
    assert orders[0][5] == "venue-order-atomic-1"
    assert tuple(event[1] for event in events) == (1, 2, 3)
    assert tuple(event[4] for event in events) == (
        "SUBMISSION_ACKNOWLEDGED",
        "CONFIRMED_FILL",
        "CONFIRMED_FILL",
    )
    assert events[1][5]["quantity"] == str(FIRST_QUANTITY)
    assert events[2][5]["quantity"] == str(SECOND_QUANTITY)


def test_exact_replay_after_later_terminal_batch_is_read_only_and_does_not_plan(
    migrated_postgres_dsn,
):
    planner = _CountingPlanner()
    owner = _owner(migrated_postgres_dsn, planner)
    first = _attempt()
    second = _attempt(
        _instruction(
            client_order_id="order-atomic-2",
            decision_id="decision-atomic-2",
        )
    )
    first_receipt = owner.execute(first)
    second_receipt = owner.execute(second)
    before = _journal_snapshot(migrated_postgres_dsn)
    replay_planner = _ExplodingPlanner()

    replay = _owner(migrated_postgres_dsn, replay_planner).execute(first)

    assert first_receipt.disposition is DurableSubmissionDisposition.COMMITTED
    assert second_receipt.disposition is DurableSubmissionDisposition.COMMITTED
    assert replay.disposition is DurableSubmissionDisposition.REPLAYED
    assert replay.submission.position_version == 1
    assert tuple(fill.position_version for fill in replay.fills) == (2, 3)
    assert replay_planner.calls == []
    assert _journal_snapshot(migrated_postgres_dsn) == before
    assert tuple(row[1] for row in _event_rows(migrated_postgres_dsn)) == (
        "order-atomic-1",
        "order-atomic-1",
        "order-atomic-1",
        "order-atomic-2",
        "order-atomic-2",
        "order-atomic-2",
    )


def test_exact_terminal_shape_from_separate_commits_is_adopted_without_planning(
    migrated_postgres_dsn,
):
    attempt = _attempt()
    plan = _plan(attempt)
    journal = _journal(migrated_postgres_dsn)
    journal.reserve_instruction(
        execution_scope=SCOPE,
        instruction=attempt.instruction,
    )
    journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id=attempt.event_id,
        event=plan.submission,
    )
    for candidate in plan.fills:
        journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id=candidate.event_id,
            event=candidate.fill,
        )
    before = _journal_snapshot(migrated_postgres_dsn)
    planner = _ExplodingPlanner()

    receipt = _owner(migrated_postgres_dsn, planner).execute(attempt)

    assert receipt.disposition is DurableSubmissionDisposition.REPLAYED
    assert planner.calls == []
    assert _journal_snapshot(migrated_postgres_dsn) == before


def test_concurrent_exact_calls_commit_once_replay_once_and_plan_once(
    migrated_postgres_dsn,
):
    planner = _CountingPlanner()
    owner = _owner(migrated_postgres_dsn, planner)
    attempt = _attempt()
    ready = Barrier(2)

    def execute():
        ready.wait(timeout=10)
        return owner.execute(attempt)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(execute) for _ in range(2))
        receipts = tuple(future.result(timeout=20) for future in futures)

    assert {receipt.disposition for receipt in receipts} == {
        DurableSubmissionDisposition.COMMITTED,
        DurableSubmissionDisposition.REPLAYED,
    }
    assert planner.calls == [attempt]
    assert tuple(row[0] for row in _event_rows(migrated_postgres_dsn)) == (1, 2, 3)


def test_concurrent_distinct_batches_never_interleave_one_position_stream(
    migrated_postgres_dsn,
):
    planner = _CountingPlanner()
    owner = _owner(migrated_postgres_dsn, planner)
    attempts = (
        _attempt(
            _instruction(
                client_order_id="order-concurrent-a",
                decision_id="decision-concurrent-a",
            )
        ),
        _attempt(
            _instruction(
                client_order_id="order-concurrent-b",
                decision_id="decision-concurrent-b",
            )
        ),
    )
    ready = Barrier(2)

    def execute(attempt):
        ready.wait(timeout=10)
        return owner.execute(attempt)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(execute, attempt) for attempt in attempts)
        receipts = tuple(future.result(timeout=20) for future in futures)

    assert {receipt.disposition for receipt in receipts} == {
        DurableSubmissionDisposition.COMMITTED
    }
    assert set(planner.calls) == set(attempts)
    rows = _event_rows(migrated_postgres_dsn)
    assert tuple(row[0] for row in rows) == (1, 2, 3, 4, 5, 6)
    clients = tuple(row[1] for row in rows)
    assert len(set(clients[:3])) == 1
    assert len(set(clients[3:])) == 1
    assert clients[0] != clients[3]


@pytest.mark.parametrize("fail_after", range(1, 8))
def test_failure_after_each_sql_mutation_rolls_back_the_complete_batch(
    migrated_postgres_dsn,
    fail_after,
):
    planner = _CountingPlanner()
    factory = _MutationFailureFactory(migrated_postgres_dsn, fail_after)

    with pytest.raises(JournalStorageError, match="failed before commit"):
        _owner(migrated_postgres_dsn, planner, factory).execute(_attempt())

    assert factory.connection is not None
    assert factory.connection.failure_cursor.mutations == fail_after
    assert factory.connection.commits == 0
    assert factory.connection.rollbacks == 1
    assert factory.connection.closed is True
    assert _journal_snapshot(migrated_postgres_dsn) == ([], [], [])
    assert _legacy_counts(migrated_postgres_dsn) == tuple(
        (table, 0)
        for table in (
            "trades",
            "open_positions",
            "liquidations",
            "margin_history",
            "trading_session_resets",
            "model_predictions",
            "account_balances",
        )
    )


def test_planner_error_rolls_back_stream_and_reservation(migrated_postgres_dsn):
    failure = RuntimeError("stable plan source failed")
    planner = _ExplodingPlanner(failure)
    attempt = _attempt()

    with pytest.raises(JournalStorageError) as storage_error:
        _owner(migrated_postgres_dsn, planner).execute(attempt)

    assert storage_error.value.__cause__ is failure
    assert planner.calls == [attempt]
    assert _journal_snapshot(migrated_postgres_dsn) == ([], [], [])


def test_commit_acknowledgement_loss_is_unknown_then_exact_retry_replays(
    migrated_postgres_dsn,
):
    attempt = _attempt()
    first_planner = _CountingPlanner()
    factory = _TrackingFactory(
        migrated_postgres_dsn,
        connection_type=_CommitThenRaiseConnection,
    )

    with pytest.raises(SubmissionCommitUnknown) as unknown:
        _owner(migrated_postgres_dsn, first_planner, factory).execute(attempt)

    assert unknown.value.attempt is attempt
    assert isinstance(unknown.value.__cause__, psycopg2.OperationalError)
    assert first_planner.calls == [attempt]
    assert tuple(row[0] for row in _event_rows(migrated_postgres_dsn)) == (1, 2, 3)
    replay_planner = _ExplodingPlanner()

    replay = _owner(migrated_postgres_dsn, replay_planner).execute(attempt)

    assert replay.disposition is DurableSubmissionDisposition.REPLAYED
    assert replay_planner.calls == []
    assert tuple(row[0] for row in _event_rows(migrated_postgres_dsn)) == (1, 2, 3)


@pytest.mark.parametrize("history", ("pending", "acknowledged", "partial"))
def test_existing_non_terminal_target_requires_reconciliation_without_planning(
    migrated_postgres_dsn,
    history,
):
    attempt = _attempt()
    journal = _journal(migrated_postgres_dsn)
    journal.reserve_instruction(
        execution_scope=SCOPE,
        instruction=attempt.instruction,
    )
    if history != "pending":
        journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id=attempt.event_id,
            event=_ack(attempt),
        )
    if history == "partial":
        journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id="fill-observation-1",
            event=_fill(
                attempt,
                trade_id=f"trade-{attempt.client_order_id}-1",
                quantity=FIRST_QUANTITY,
                seconds=1,
            ),
        )
    before = _journal_snapshot(migrated_postgres_dsn)
    planner = _ExplodingPlanner()

    with pytest.raises(SubmissionReconciliationRequired) as required:
        _owner(migrated_postgres_dsn, planner).execute(attempt)

    assert required.value.attempt is attempt
    assert planner.calls == []
    assert _journal_snapshot(migrated_postgres_dsn) == before


def test_unresolved_sibling_blocks_a_new_order_without_reserving_or_planning(
    migrated_postgres_dsn,
):
    sibling = _instruction(
        client_order_id="order-unresolved-sibling",
        decision_id="decision-unresolved-sibling",
    )
    _journal(migrated_postgres_dsn).reserve_instruction(
        execution_scope=SCOPE,
        instruction=sibling,
    )
    target = _attempt(
        _instruction(
            client_order_id="order-blocked-target",
            decision_id="decision-blocked-target",
        )
    )
    before = _journal_snapshot(migrated_postgres_dsn)
    planner = _ExplodingPlanner()

    with pytest.raises(SubmissionReconciliationRequired):
        _owner(migrated_postgres_dsn, planner).execute(target)

    assert planner.calls == []
    assert _journal_snapshot(migrated_postgres_dsn) == before
    assert all(order[0] != target.client_order_id for order in before[1])


def test_interleaved_target_history_requires_reconciliation_without_planning(
    migrated_postgres_dsn,
):
    target = _attempt()
    sibling = _attempt(
        _instruction(
            client_order_id="order-interleaved-sibling",
            decision_id="decision-interleaved-sibling",
        )
    )
    journal = _journal(migrated_postgres_dsn)
    for attempt in (target, sibling):
        journal.reserve_instruction(
            execution_scope=SCOPE,
            instruction=attempt.instruction,
        )
    journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id=target.event_id,
        event=_ack(target),
    )
    journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id=sibling.event_id,
        event=_ack(sibling),
    )
    for index, quantity in enumerate((FIRST_QUANTITY, SECOND_QUANTITY), start=1):
        journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id=f"fill-observation-{index}",
            event=_fill(
                target,
                trade_id=f"trade-{target.client_order_id}-{index}",
                quantity=quantity,
                seconds=index,
            ),
        )
    before = _journal_snapshot(migrated_postgres_dsn)
    planner = _ExplodingPlanner()

    with pytest.raises(SubmissionReconciliationRequired):
        _owner(migrated_postgres_dsn, planner).execute(target)

    assert planner.calls == []
    assert _journal_snapshot(migrated_postgres_dsn) == before


@pytest.mark.parametrize("damage", ("payload", "version-gap"))
def test_corrupt_or_gapped_stream_fails_closed_before_planner(
    migrated_postgres_dsn,
    damage,
):
    attempt = _attempt()
    _owner(migrated_postgres_dsn, _CountingPlanner()).execute(attempt)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            if damage == "payload":
                cursor.execute(
                    """
                    UPDATE np.order_events
                    SET event_payload = event_payload || '{"tampered": true}'::jsonb
                    WHERE position_key = %s AND position_version = 1
                    """,
                    (POSITION_KEY,),
                )
            else:
                cursor.execute(
                    """
                    UPDATE np.position_streams
                    SET stream_version = stream_version + 1
                    WHERE position_key = %s
                    """,
                    (POSITION_KEY,),
                )
        connection.commit()
    finally:
        connection.close()
    before = _journal_snapshot(migrated_postgres_dsn)
    planner = _ExplodingPlanner()

    with pytest.raises(JournalReplayError):
        _owner(migrated_postgres_dsn, planner).execute(attempt)

    assert planner.calls == []
    assert _journal_snapshot(migrated_postgres_dsn) == before


def test_decimal_scale_change_is_not_an_exact_replay(migrated_postgres_dsn):
    stored = _instruction(quantity=Decimal("1.0"))
    _journal(migrated_postgres_dsn).reserve_instruction(
        execution_scope=SCOPE,
        instruction=stored,
    )
    incoming = _attempt(_instruction(quantity=Decimal("1.00")))
    planner = _ExplodingPlanner()
    before = _journal_snapshot(migrated_postgres_dsn)

    with pytest.raises(JournalConflictError) as conflict:
        _owner(migrated_postgres_dsn, planner).execute(incoming)

    assert conflict.value.kind is JournalConflictKind.CLIENT_ORDER_ID
    assert planner.calls == []
    assert _journal_snapshot(migrated_postgres_dsn) == before


def test_owner_never_queries_or_mutates_legacy_or_account_tables(
    migrated_postgres_dsn,
):
    statements = []
    factory = _TrackingFactory(migrated_postgres_dsn, statements=statements)
    before = _legacy_counts(migrated_postgres_dsn)

    _owner(migrated_postgres_dsn, _CountingPlanner(), factory).execute(_attempt())

    forbidden = (
        "np.trades",
        "np.open_positions",
        "np.liquidations",
        "np.margin_history",
        "np.trading_session_resets",
        "np.model_predictions",
        "np.account_balances",
    )
    assert statements
    assert all(
        relation not in statement.lower()
        for statement in statements
        for relation in forbidden
    )
    assert _legacy_counts(migrated_postgres_dsn) == before
