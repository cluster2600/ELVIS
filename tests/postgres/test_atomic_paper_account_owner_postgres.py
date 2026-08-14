"""PostgreSQL 15 proofs for the atomic paper-account submission owner."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Barrier, Event, Lock

import psycopg2
import pytest
from psycopg2 import sql

from tests.postgres.test_paper_account_readiness_postgres import (
    _readiness_opening_role,
    _seed_fresh_opening_provenance,
)
from trading.application.durable_submission import (
    DurableSubmissionDisposition,
    PaperAccountSubmissionCommitUnknown,
    PaperAccountSubmissionContext,
    PaperAccountSubmissionReconciliationRequired,
    PaperAccountSubmissionRejected,
    PaperAccountSubmissionRuntimeUnavailable,
    PaperPlannedFill,
    PaperSubmissionPlan,
    SubmissionAttemptContext,
)
from trading.domain.order_lifecycle import ConfirmedFill, SubmissionAcknowledged
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.paper_accounting import (
    PaperAccountBalance,
    PaperAccountPolicy,
    new_paper_account,
)
from trading.domain.paper_settlement import PaperLinearInstrument
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionInstruction,
    TakeProfitProfile,
)
from trading.persistence.atomic_paper_account_owner import (
    PostgresAtomicPaperAccountOwner,
)
from trading.persistence.atomic_paper_submission_owner import (
    PostgresAtomicPaperSubmissionOwner,
)
from trading.persistence.paper_account_journal import (
    PaperAccountStorageError,
    PostgresPaperAccountJournal,
)
from trading.persistence.paper_account_journal_codec import (
    encode_paper_account_opening,
)

SCOPE = "paper:atomic-account-owner"
ACCOUNT_KEY = "atomic-account"
NOW = datetime(2026, 8, 12, 12, 0, 0, 123456, tzinfo=timezone.utc)
FIRST_QUANTITY = Decimal("0.40000000000000000001")
SECOND_QUANTITY = Decimal("0.59999999999999999999")
INSTRUMENT = PaperLinearInstrument("BTCUSDT", "BTC", "USDT")
RUNTIME_GENERATION = 1

_SNAPSHOT_TABLES = (
    "position_streams",
    "orders",
    "order_events",
    "paper_account_streams",
    "paper_account_balances",
    "paper_margin_reservations",
    "paper_account_batch_manifests",
    "paper_account_settlements",
    "paper_account_postings",
    "paper_runtime_control",
    "paper_runtime_generations",
    "trades",
    "open_positions",
    "liquidations",
    "margin_history",
    "trading_session_resets",
    "model_predictions",
    "account_balances",
)


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


@pytest.fixture(autouse=True)
def _cleanup_atomic_owner_opening_role(postgres_database_dsn):
    yield
    connection = _connect(postgres_database_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            role_name = _readiness_opening_role(cursor.fetchone()[0])
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = %s)",
                (role_name,),
            )
            if cursor.fetchone()[0]:
                cursor.execute(
                    sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role_name))
                )
                cursor.execute(
                    sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name))
                )
        connection.commit()
    finally:
        connection.close()


def _instruction(
    *,
    client_order_id="account-order-1",
    decision_id="account-decision-1",
    position_key="account-position-1",
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
            reference_price=Decimal("100.00"),
            leverage=2,
            created_at=NOW,
        ),
        exit_context=PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.0025"),
            stop_loss_fraction=Decimal("0.005"),
        ),
    )


def _context(
    *,
    account_key=ACCOUNT_KEY,
    instruction=None,
    runtime_generation=RUNTIME_GENERATION,
):
    attempt = SubmissionAttemptContext.first(
        instruction or _instruction(),
        SCOPE,
        NOW,
    )
    return PaperAccountSubmissionContext(
        attempt,
        account_key,
        INSTRUMENT,
        runtime_generation,
    )


def _fill(attempt, *, trade_id, quantity, seconds):
    intent = attempt.instruction.order_intent
    return ConfirmedFill(
        client_order_id=intent.client_order_id,
        venue_order_id=f"venue-{intent.client_order_id}",
        trade_id=trade_id,
        symbol=intent.symbol,
        side=intent.side,
        quantity=quantity,
        price=Decimal("100.00"),
        fee_amount=Decimal("0.00"),
        fee_asset="USDT",
        executed_at=attempt.observed_at + timedelta(seconds=seconds),
    )


def _plan(attempt):
    return PaperSubmissionPlan(
        attempt=attempt,
        submission=SubmissionAcknowledged(
            client_order_id=attempt.client_order_id,
            venue_order_id=f"venue-{attempt.client_order_id}",
            observed_at=attempt.observed_at,
        ),
        fills=(
            PaperPlannedFill(
                "account-fill-1",
                _fill(
                    attempt,
                    trade_id=f"trade-{attempt.client_order_id}-1",
                    quantity=FIRST_QUANTITY,
                    seconds=1,
                ),
            ),
            PaperPlannedFill(
                "account-fill-2",
                _fill(
                    attempt,
                    trade_id=f"trade-{attempt.client_order_id}-2",
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
    def __init__(self):
        self.calls = []

    def plan(self, attempt, /):
        self.calls.append(attempt)
        raise AssertionError("planner must not run")


class _BlockingPlanner(_CountingPlanner):
    def __init__(self):
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def plan(self, attempt, /):
        with self._lock:
            self.calls.append(attempt)
        self.entered.set()
        assert self.release.wait(timeout=10)
        return _plan(attempt)


def _opening(account_key=ACCOUNT_KEY, *, available=Decimal("100.00")):
    return new_paper_account(
        PaperAccountPolicy(account_key, "USDT", Decimal("0.01")),
        (PaperAccountBalance("USDT", available, Decimal("0.00")),),
    )


def _activate(dsn, opening, runtime_generation=RUNTIME_GENERATION):
    encoded = encode_paper_account_opening(
        opening.execution_scope,
        opening.owner_generation,
        opening.account,
    )
    _seed_fresh_opening_provenance(dsn, encoded)
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO np.paper_runtime_generations (
                    runtime_generation,
                    activation_id,
                    execution_scope,
                    account_key,
                    owner_generation,
                    opening_version,
                    opening_payload_sha256
                ) VALUES (%s, %s, %s, %s, %s, 1, %s)
                """,
                (
                    runtime_generation,
                    f"atomic-activation-{runtime_generation}",
                    opening.execution_scope,
                    opening.account.policy.account_key,
                    opening.owner_generation,
                    opening.current.opening_payload_sha256,
                ),
            )
            cursor.execute(
                """
                UPDATE np.paper_runtime_control
                SET mode = 'ACTIVE', runtime_generation = %s
                WHERE control_key
                """,
                (runtime_generation,),
            )
        connection.commit()
    finally:
        connection.close()


def _provision(
    dsn,
    account_key=ACCOUNT_KEY,
    *,
    available=Decimal("100.00"),
    runtime_generation=RUNTIME_GENERATION,
    activate=True,
):
    opening = PostgresPaperAccountJournal(lambda: _connect(dsn)).provision_account(
        execution_scope=SCOPE,
        owner_generation=7,
        account=_opening(account_key, available=available),
    )
    if activate:
        _activate(dsn, opening, runtime_generation)
    return opening


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
        self._statements.append(" ".join(str(statement).split()))
        if parameters is None:
            return self._cursor.execute(statement)
        return self._cursor.execute(statement, parameters)


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
        return (
            cursor
            if self._statements is None
            else _TracingCursor(cursor, self._statements)
        )

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


class _CommitThenRaiseConnection(_TrackingConnection):
    def commit(self):
        self.commits += 1
        self._connection.commit()
        raise psycopg2.OperationalError("simulated lost commit acknowledgement")


class _CommitRejectedConnection(_TrackingConnection):
    def commit(self):
        self.commits += 1
        raise psycopg2.OperationalError("simulated rejected commit")


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
                raise RuntimeError(f"injected failure after mutation {self.mutations}")
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


def _owner(
    dsn,
    planner,
    factory=None,
    *,
    runtime_generation=RUNTIME_GENERATION,
):
    return PostgresAtomicPaperAccountOwner(
        factory or (lambda: _connect(dsn)),
        planner,
        runtime_generation,
    )


def _snapshot(dsn):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            result = []
            for table in _SNAPSHOT_TABLES:
                cursor.execute(f"""
                    SELECT row_to_json(candidate)::text AS value
                    FROM np.{table} AS candidate
                    ORDER BY value
                    """)
                result.append((table, tuple(row[0] for row in cursor.fetchall())))
            return tuple(result)
    finally:
        connection.close()


def _relation_count(dsn, relation):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM np.{relation}")
            return cursor.fetchone()[0]
    finally:
        connection.close()


def _has_dml(statements):
    return any(
        statement.upper().startswith(("INSERT ", "UPDATE ", "DELETE "))
        for statement in statements
    )


def _rewrite_manifest_generation(dsn, runtime_generation):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT batch_payload
                FROM np.paper_account_batch_manifests
                WHERE account_key = %s
                """,
                (ACCOUNT_KEY,),
            )
            payload = cursor.fetchone()[0]
            if runtime_generation is None:
                payload.pop("runtime_generation")
                batch_version = 1
            else:
                payload["runtime_generation"] = runtime_generation
                batch_version = 2
            payload_text = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            payload_sha = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
            cursor.execute(
                """
                UPDATE np.paper_account_batch_manifests
                SET batch_version = %s,
                    batch_payload = %s::jsonb,
                    batch_payload_sha256 = %s,
                    runtime_generation = %s
                WHERE account_key = %s
                """,
                (
                    batch_version,
                    payload_text,
                    payload_sha,
                    runtime_generation,
                    ACCOUNT_KEY,
                ),
            )
        connection.commit()
    finally:
        connection.close()


def test_two_fill_batch_commits_every_account_and_position_fact_then_replays(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn)
    planner = _CountingPlanner()
    statements = []
    factory = _TrackingFactory(migrated_postgres_dsn, statements=statements)
    context = _context()
    owner = _owner(migrated_postgres_dsn, planner, factory)

    committed = owner.execute(context)
    replay_planner = _ExplodingPlanner()
    before_replay = _snapshot(migrated_postgres_dsn)
    replay_statements = []
    replay_factory = _TrackingFactory(
        migrated_postgres_dsn,
        statements=replay_statements,
    )
    replayed = _owner(
        migrated_postgres_dsn,
        replay_planner,
        replay_factory,
    ).execute(context)

    assert committed.disposition is DurableSubmissionDisposition.COMMITTED
    assert committed.context is context
    assert committed.account_versions == (1, 2)
    assert tuple(fill.position_version for fill in committed.submission.fills) == (
        2,
        3,
    )
    assert replayed.disposition is DurableSubmissionDisposition.REPLAYED
    assert replayed.account_versions == committed.account_versions
    assert replayed.submission.submission.event == committed.submission.submission.event
    assert replayed.submission.fills == committed.submission.fills
    assert planner.calls == [context.attempt]
    assert replay_planner.calls == []
    assert _snapshot(migrated_postgres_dsn) == before_replay
    assert not any(
        statement.upper().startswith(("INSERT ", "UPDATE ", "DELETE "))
        for statement in replay_statements
    )
    assert len(factory.connections) == 1
    assert factory.connections[0].commits == 1
    assert factory.connections[0].closed is True

    account = PostgresPaperAccountJournal(
        lambda: _connect(migrated_postgres_dsn)
    ).replay_account(execution_scope=SCOPE, account_key=ACCOUNT_KEY)
    assert len(account.batches) == 1
    assert account.batches[0].runtime_generation == RUNTIME_GENERATION
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT batch_version, runtime_generation,
                       batch_payload ->> 'runtime_generation'
                FROM np.paper_account_batch_manifests
                WHERE account_key = %s AND client_order_id = %s
                """,
                (ACCOUNT_KEY, context.client_order_id),
            )
            assert cursor.fetchone() == (2, RUNTIME_GENERATION, "1")
    finally:
        connection.close()
    assert tuple(record.account_version for record in account.account.records) == (1, 2)
    assert tuple(
        record.settlement.record.position_version for record in account.account.records
    ) == (2, 3)

    normalized = tuple(statement.upper() for statement in statements)
    transaction_start = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("SET TRANSACTION")
    )
    runtime_control_lock = next(
        index
        for index, statement in enumerate(normalized)
        if "FROM NP.PAPER_RUNTIME_CONTROL" in statement and "FOR SHARE" in statement
    )
    runtime_generation_lock = next(
        index
        for index, statement in enumerate(normalized)
        if "FROM NP.PAPER_RUNTIME_GENERATIONS" in statement and "FOR SHARE" in statement
    )
    account_lock = next(
        index
        for index, statement in enumerate(normalized)
        if "FROM NP.PAPER_ACCOUNT_STREAMS" in statement and "FOR UPDATE" in statement
    )
    position_touch = next(
        index
        for index, statement in enumerate(normalized)
        if "NP.POSITION_STREAMS" in statement
    )
    assert (
        transaction_start
        < runtime_control_lock
        < runtime_generation_lock
        < account_lock
        < position_touch
    )
    assert any(statement == "SET CONSTRAINTS ALL IMMEDIATE" for statement in normalized)
    assert all(
        "SET CONSTRAINTS ALL DEFERRED" not in statement for statement in normalized
    )


def test_rejection_on_second_fill_rolls_back_every_candidate_fact(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn, available=Decimal("30.00"))
    before = _snapshot(migrated_postgres_dsn)
    factory = _TrackingFactory(migrated_postgres_dsn)
    context = _context()

    rejected = _owner(
        migrated_postgres_dsn,
        _CountingPlanner(),
        factory,
    ).execute(context)

    assert type(rejected) is PaperAccountSubmissionRejected
    assert rejected.context is context
    assert rejected.rejected_event_id == "account-fill-2"
    assert rejected.reasons == ("insufficient available balance for USDT",)
    assert _snapshot(migrated_postgres_dsn) == before
    assert len(factory.connections) == 1
    assert factory.connections[0].commits == 0
    assert factory.connections[0].rollbacks == 1
    assert factory.connections[0].closed is True


def test_unprovisioned_account_requires_reconciliation_without_planning_or_dml(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn, "other-account")
    context = _context()
    planner = _ExplodingPlanner()
    statements = []
    factory = _TrackingFactory(migrated_postgres_dsn, statements=statements)
    before = _snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperAccountSubmissionReconciliationRequired) as required:
        _owner(migrated_postgres_dsn, planner, factory).execute(context)

    assert required.value.context is context
    assert planner.calls == []
    assert not any(
        statement.upper().startswith(("INSERT ", "UPDATE ", "DELETE "))
        for statement in statements
    )
    assert _snapshot(migrated_postgres_dsn) == before


@pytest.mark.parametrize("mode", ("LEGACY", "SHADOW", "PAUSED"))
def test_nonactive_runtime_is_unavailable_before_planner_or_dml(
    migrated_postgres_dsn,
    mode,
):
    _provision(migrated_postgres_dsn, activate=False)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE np.paper_runtime_control
                SET mode = %s, runtime_generation = 0
                WHERE control_key
                """,
                (mode,),
            )
        connection.commit()
    finally:
        connection.close()
    context = _context()
    planner = _ExplodingPlanner()
    statements = []
    before = _snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperAccountSubmissionRuntimeUnavailable) as unavailable:
        _owner(
            migrated_postgres_dsn,
            planner,
            _TrackingFactory(migrated_postgres_dsn, statements=statements),
        ).execute(context)

    assert unavailable.value.context is context
    assert planner.calls == []
    assert not _has_dml(statements)
    assert _snapshot(migrated_postgres_dsn) == before


def test_active_control_without_pinned_epoch_is_unavailable_before_planner_or_dml(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn, activate=False)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                UPDATE np.paper_runtime_control
                SET mode = 'ACTIVE', runtime_generation = 1
                WHERE control_key
                """)
        connection.commit()
    finally:
        connection.close()
    context = _context()
    planner = _ExplodingPlanner()
    statements = []
    before = _snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperAccountSubmissionRuntimeUnavailable) as unavailable:
        _owner(
            migrated_postgres_dsn,
            planner,
            _TrackingFactory(migrated_postgres_dsn, statements=statements),
        ).execute(context)

    assert unavailable.value.context is context
    assert planner.calls == []
    assert not _has_dml(statements)
    assert _snapshot(migrated_postgres_dsn) == before


def test_missing_runtime_control_is_unavailable_before_planner_or_dml(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn, activate=False)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM np.paper_runtime_control")
        connection.commit()
    finally:
        connection.close()
    context = _context()
    planner = _ExplodingPlanner()
    statements = []
    before = _snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperAccountSubmissionRuntimeUnavailable) as unavailable:
        _owner(
            migrated_postgres_dsn,
            planner,
            _TrackingFactory(migrated_postgres_dsn, statements=statements),
        ).execute(context)

    assert unavailable.value.context is context
    assert planner.calls == []
    assert not _has_dml(statements)
    assert _snapshot(migrated_postgres_dsn) == before


def test_epoch_for_another_opening_is_unavailable_before_planner_or_dml(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn, activate=False)
    _provision(migrated_postgres_dsn, "other-account")
    context = _context()
    planner = _ExplodingPlanner()
    statements = []
    before = _snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperAccountSubmissionRuntimeUnavailable) as unavailable:
        _owner(
            migrated_postgres_dsn,
            planner,
            _TrackingFactory(migrated_postgres_dsn, statements=statements),
        ).execute(context)

    assert unavailable.value.context is context
    assert planner.calls == []
    assert not _has_dml(statements)
    assert _snapshot(migrated_postgres_dsn) == before


def test_concurrent_exact_calls_commit_once_replay_once_and_plan_once(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn)
    planner = _CountingPlanner()
    owner = _owner(migrated_postgres_dsn, planner)
    context = _context()
    ready = Barrier(2)

    def execute():
        ready.wait(timeout=10)
        return owner.execute(context)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(execute) for _ in range(2))
        results = tuple(future.result(timeout=20) for future in futures)

    assert {result.disposition for result in results} == {
        DurableSubmissionDisposition.COMMITTED,
        DurableSubmissionDisposition.REPLAYED,
    }
    assert planner.calls == [context.attempt]
    assert _relation_count(migrated_postgres_dsn, "paper_account_batch_manifests") == 1
    assert _relation_count(migrated_postgres_dsn, "paper_account_settlements") == 2


def test_runtime_control_transition_waits_for_owner_share_lock_until_commit(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn)
    planner = _BlockingPlanner()
    context = _context()
    owner = _owner(migrated_postgres_dsn, planner)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(owner.execute, context)
        assert planner.entered.wait(timeout=10)

        transition = _connect(migrated_postgres_dsn)
        try:
            with transition.cursor() as cursor:
                cursor.execute("SET LOCAL lock_timeout = '250ms'")
                with pytest.raises(psycopg2.Error) as blocked:
                    cursor.execute("""
                        UPDATE np.paper_runtime_control
                        SET mode = 'PAUSED'
                        WHERE control_key
                        """)
                assert blocked.value.pgcode == "55P03"
            transition.rollback()
            assert future.done() is False

            planner.release.set()
            committed = future.result(timeout=20)
            assert committed.disposition is DurableSubmissionDisposition.COMMITTED

            with transition.cursor() as cursor:
                cursor.execute("SET LOCAL lock_timeout = '1s'")
                cursor.execute("""
                    UPDATE np.paper_runtime_control
                    SET mode = 'PAUSED'
                    WHERE control_key
                    RETURNING mode
                    """)
                assert cursor.fetchone() == ("PAUSED",)
            transition.rollback()
        finally:
            planner.release.set()
            transition.close()

    assert planner.calls == [context.attempt]


def test_account_lock_serializes_collateral_across_distinct_positions(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn, available=Decimal("50.00"))
    planner = _CountingPlanner()
    owner = _owner(migrated_postgres_dsn, planner)
    contexts = (
        _context(
            instruction=_instruction(
                client_order_id="scarce-order-a",
                decision_id="scarce-decision-a",
                position_key="scarce-position-a",
            )
        ),
        _context(
            instruction=_instruction(
                client_order_id="scarce-order-b",
                decision_id="scarce-decision-b",
                position_key="scarce-position-b",
            )
        ),
    )
    ready = Barrier(2)

    def execute(context):
        ready.wait(timeout=10)
        return owner.execute(context)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(execute, context) for context in contexts)
        results = tuple(future.result(timeout=20) for future in futures)

    committed = tuple(
        result
        for result in results
        if not isinstance(result, PaperAccountSubmissionRejected)
    )
    rejected = tuple(
        result
        for result in results
        if isinstance(result, PaperAccountSubmissionRejected)
    )
    assert len(committed) == len(rejected) == 1
    assert committed[0].disposition is DurableSubmissionDisposition.COMMITTED
    assert rejected[0].reasons == ("insufficient available balance for USDT",)
    assert set(planner.calls) == {context.attempt for context in contexts}
    assert _relation_count(migrated_postgres_dsn, "position_streams") == 1
    assert _relation_count(migrated_postgres_dsn, "orders") == 1
    assert _relation_count(migrated_postgres_dsn, "paper_account_batch_manifests") == 1


@pytest.mark.parametrize("history", ("missing", "incomplete", "corrupt"))
def test_terminal_history_without_one_exact_account_batch_requires_reconciliation(
    migrated_postgres_dsn,
    history,
):
    _provision(migrated_postgres_dsn)
    context = _context()
    if history == "missing":
        PostgresAtomicPaperSubmissionOwner(
            lambda: _connect(migrated_postgres_dsn),
            _CountingPlanner(),
        ).execute(context.attempt)
    else:
        _owner(migrated_postgres_dsn, _CountingPlanner()).execute(context)
        connection = _connect(migrated_postgres_dsn)
        try:
            with connection.cursor() as cursor:
                if history == "incomplete":
                    cursor.execute(
                        "DELETE FROM np.paper_account_postings WHERE account_key = %s",
                        (ACCOUNT_KEY,),
                    )
                    cursor.execute(
                        "DELETE FROM np.paper_account_settlements "
                        "WHERE account_key = %s",
                        (ACCOUNT_KEY,),
                    )
                    cursor.execute(
                        "DELETE FROM np.paper_margin_reservations "
                        "WHERE account_key = %s",
                        (ACCOUNT_KEY,),
                    )
                    cursor.execute(
                        """
                        UPDATE np.paper_account_balances
                        SET available_decimal = '100.00', reserved_decimal = '0.00'
                        WHERE account_key = %s AND asset = 'USDT'
                        """,
                        (ACCOUNT_KEY,),
                    )
                    cursor.execute(
                        """
                        UPDATE np.paper_account_streams
                        SET account_version = 0, account_state = 'ACTIVE'
                        WHERE account_key = %s
                        """,
                        (ACCOUNT_KEY,),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE np.paper_account_batch_manifests
                        SET batch_payload = batch_payload || '{"tampered": true}'::jsonb
                        WHERE account_key = %s
                        """,
                        (ACCOUNT_KEY,),
                    )
            connection.commit()
        finally:
            connection.close()
    before = _snapshot(migrated_postgres_dsn)
    planner = _ExplodingPlanner()

    with pytest.raises(PaperAccountSubmissionReconciliationRequired) as required:
        _owner(migrated_postgres_dsn, planner).execute(context)

    assert required.value.context is context
    assert planner.calls == []
    assert _snapshot(migrated_postgres_dsn) == before


def test_manifest_instruction_quantum_mismatch_requires_reconciliation_not_conflict(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn)
    committed = _context(instruction=_instruction(quantity=Decimal("1.0")))
    _owner(migrated_postgres_dsn, _CountingPlanner()).execute(committed)
    incoming = _context(instruction=_instruction(quantity=Decimal("1.00")))
    assert incoming.client_order_id == committed.client_order_id
    assert (
        incoming.attempt.instruction.order_intent.quantity.as_tuple()
        != committed.attempt.instruction.order_intent.quantity.as_tuple()
    )
    planner = _ExplodingPlanner()
    statements = []
    factory = _TrackingFactory(migrated_postgres_dsn, statements=statements)
    before = _snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperAccountSubmissionReconciliationRequired) as required:
        _owner(migrated_postgres_dsn, planner, factory).execute(incoming)

    assert required.value.context is incoming
    assert planner.calls == []
    assert not any(
        statement.upper().startswith(("INSERT ", "UPDATE ", "DELETE "))
        for statement in statements
    )
    assert _snapshot(migrated_postgres_dsn) == before


def test_manifest_instrument_mismatch_requires_reconciliation_without_dml(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn)
    committed = _context()
    _owner(migrated_postgres_dsn, _CountingPlanner()).execute(committed)
    incoming = PaperAccountSubmissionContext(
        committed.attempt,
        ACCOUNT_KEY,
        PaperLinearInstrument("BTCUSDT", "XBT", "USDT"),
        RUNTIME_GENERATION,
    )
    planner = _ExplodingPlanner()
    statements = []
    factory = _TrackingFactory(migrated_postgres_dsn, statements=statements)
    before = _snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperAccountSubmissionReconciliationRequired) as required:
        _owner(migrated_postgres_dsn, planner, factory).execute(incoming)

    assert required.value.context is incoming
    assert planner.calls == []
    assert not any(
        statement.upper().startswith(("INSERT ", "UPDATE ", "DELETE "))
        for statement in statements
    )
    assert _snapshot(migrated_postgres_dsn) == before


@pytest.mark.parametrize("history", ("v1", "wrong_generation"))
def test_v1_or_wrong_generation_manifest_requires_reconciliation_without_dml(
    migrated_postgres_dsn,
    history,
):
    opening = _provision(migrated_postgres_dsn)
    context = _context()
    _owner(migrated_postgres_dsn, _CountingPlanner()).execute(context)
    owner_generation = RUNTIME_GENERATION
    if history == "v1":
        _rewrite_manifest_generation(migrated_postgres_dsn, None)
    else:
        owner_generation = 2
        _activate(migrated_postgres_dsn, opening, owner_generation)
        _rewrite_manifest_generation(migrated_postgres_dsn, owner_generation)
    planner = _ExplodingPlanner()
    statements = []
    before = _snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperAccountSubmissionReconciliationRequired) as required:
        _owner(
            migrated_postgres_dsn,
            planner,
            _TrackingFactory(migrated_postgres_dsn, statements=statements),
            runtime_generation=owner_generation,
        ).execute(context)

    assert required.value.context is context
    assert planner.calls == []
    assert not _has_dml(statements)
    assert _snapshot(migrated_postgres_dsn) == before


def test_rollover_allows_exact_old_generation_replay_but_not_a_new_old_attempt(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    old_context = _context()
    _owner(migrated_postgres_dsn, _CountingPlanner()).execute(old_context)
    _activate(migrated_postgres_dsn, opening, 2)

    replay_planner = _ExplodingPlanner()
    replay_statements = []
    before_replay = _snapshot(migrated_postgres_dsn)
    replayed = _owner(
        migrated_postgres_dsn,
        replay_planner,
        _TrackingFactory(migrated_postgres_dsn, statements=replay_statements),
        runtime_generation=2,
    ).execute(old_context)

    assert replayed.disposition is DurableSubmissionDisposition.REPLAYED
    assert replayed.context is old_context
    assert replayed.context.runtime_generation == 1
    assert replay_planner.calls == []
    assert not _has_dml(replay_statements)
    assert _snapshot(migrated_postgres_dsn) == before_replay

    stale_context = _context(
        runtime_generation=1,
        instruction=_instruction(
            client_order_id="stale-new-order",
            decision_id="stale-new-decision",
            position_key="stale-new-position",
        ),
    )
    stale_planner = _ExplodingPlanner()
    stale_statements = []
    before_stale = _snapshot(migrated_postgres_dsn)
    with pytest.raises(PaperAccountSubmissionRuntimeUnavailable) as unavailable:
        _owner(
            migrated_postgres_dsn,
            stale_planner,
            _TrackingFactory(migrated_postgres_dsn, statements=stale_statements),
            runtime_generation=2,
        ).execute(stale_context)

    assert unavailable.value.context is stale_context
    assert stale_planner.calls == []
    assert not _has_dml(stale_statements)
    assert _snapshot(migrated_postgres_dsn) == before_stale


def test_old_pinned_owner_is_unavailable_after_control_rollover_without_dml(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    _activate(migrated_postgres_dsn, opening, 2)
    context = _context()
    planner = _ExplodingPlanner()
    statements = []
    before = _snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperAccountSubmissionRuntimeUnavailable) as unavailable:
        _owner(
            migrated_postgres_dsn,
            planner,
            _TrackingFactory(migrated_postgres_dsn, statements=statements),
        ).execute(context)

    assert unavailable.value.context is context
    assert planner.calls == []
    assert not _has_dml(statements)
    assert _snapshot(migrated_postgres_dsn) == before


def test_failure_after_every_sql_mutation_rolls_back_all_snapshotted_relations(
    migrated_postgres_dsn,
):
    probe_key = "mutation-probe"
    opening = _provision(migrated_postgres_dsn, probe_key)
    statements = []
    probe_factory = _TrackingFactory(
        migrated_postgres_dsn,
        statements=statements,
        connection_type=_CommitRejectedConnection,
    )
    with pytest.raises(PaperAccountSubmissionCommitUnknown):
        _owner(migrated_postgres_dsn, _CountingPlanner(), probe_factory).execute(
            _context(
                account_key=probe_key,
                instruction=_instruction(
                    client_order_id="mutation-probe-order",
                    decision_id="mutation-probe-decision",
                    position_key="mutation-probe-position",
                ),
            )
        )
    mutation_count = sum(
        statement.upper().startswith(("INSERT ", "UPDATE ", "DELETE "))
        for statement in statements
    )
    assert mutation_count >= 12

    for fail_after in range(1, mutation_count + 1):
        runtime_generation = fail_after + 1
        _activate(migrated_postgres_dsn, opening, runtime_generation)
        context = _context(
            account_key=probe_key,
            runtime_generation=runtime_generation,
            instruction=_instruction(
                client_order_id=f"fault-order-{fail_after}",
                decision_id=f"fault-decision-{fail_after}",
                position_key=f"fault-position-{fail_after}",
            ),
        )
        before = _snapshot(migrated_postgres_dsn)
        factory = _MutationFailureFactory(migrated_postgres_dsn, fail_after)

        with pytest.raises(PaperAccountStorageError) as failure:
            _owner(
                migrated_postgres_dsn,
                _CountingPlanner(),
                factory,
                runtime_generation=runtime_generation,
            ).execute(context)

        assert isinstance(failure.value.__cause__, RuntimeError)
        assert factory.connection is not None
        assert factory.connection.failure_cursor.mutations == fail_after
        assert factory.connection.commits == 0
        assert factory.connection.rollbacks == 1
        assert factory.connection.closed is True
        assert _snapshot(migrated_postgres_dsn) == before


def test_commit_acknowledgement_loss_is_unknown_then_exact_replay(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    context = _context()
    planner = _CountingPlanner()
    factory = _TrackingFactory(
        migrated_postgres_dsn,
        connection_type=_CommitThenRaiseConnection,
    )

    with pytest.raises(PaperAccountSubmissionCommitUnknown) as unknown:
        _owner(migrated_postgres_dsn, planner, factory).execute(context)

    assert unknown.value.context is context
    assert isinstance(unknown.value.__cause__, psycopg2.OperationalError)
    assert planner.calls == [context.attempt]
    _activate(migrated_postgres_dsn, opening, 2)
    replay_planner = _ExplodingPlanner()
    replay_statements = []
    before_replay = _snapshot(migrated_postgres_dsn)
    replayed = _owner(
        migrated_postgres_dsn,
        replay_planner,
        _TrackingFactory(migrated_postgres_dsn, statements=replay_statements),
        runtime_generation=2,
    ).execute(context)
    assert replayed.disposition is DurableSubmissionDisposition.REPLAYED
    assert replayed.account_versions == (1, 2)
    assert replayed.context.runtime_generation == 1
    assert replay_planner.calls == []
    assert not _has_dml(replay_statements)
    assert _snapshot(migrated_postgres_dsn) == before_replay


def test_owner_never_queries_or_mutates_legacy_tables(migrated_postgres_dsn):
    _provision(migrated_postgres_dsn)
    statements = []
    factory = _TrackingFactory(migrated_postgres_dsn, statements=statements)
    before = _snapshot(migrated_postgres_dsn)

    _owner(migrated_postgres_dsn, _CountingPlanner(), factory).execute(_context())

    forbidden = tuple(
        f"NP.{table.upper()}"
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
    assert statements
    assert all(
        relation not in statement.upper()
        for statement in statements
        for relation in forbidden
    )
    after = _snapshot(migrated_postgres_dsn)
    before_by_table = dict(before)
    after_by_table = dict(after)
    for table in _SNAPSHOT_TABLES[11:]:
        assert after_by_table[table] == before_by_table[table] == ()
