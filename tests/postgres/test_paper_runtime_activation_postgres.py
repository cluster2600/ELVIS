"""PostgreSQL 15 proofs for the dormant locked runtime activation boundary."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from threading import Event
from time import monotonic, sleep

import psycopg2
import pytest

from trading.application.paper_account_readiness import (
    PaperAccountReadinessContext,
    PaperAccountReadinessDisposition,
    PaperAccountReadinessFindingKind,
)
from trading.application.paper_runtime_activation import (
    PaperRuntimeActivationBlocked,
    PaperRuntimeActivationBusy,
    PaperRuntimeActivationCommitUnknown,
    PaperRuntimeActivationConflict,
    PaperRuntimeActivationContext,
    PaperRuntimeActivationDisposition,
    PaperRuntimeActivationSource,
)
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.paper_accounting import (
    PaperAccountBalance,
    PaperAccountPolicy,
    new_paper_account,
)
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionInstruction,
    TakeProfitProfile,
)
from trading.persistence.order_position_journal import PostgresOrderPositionJournal
from trading.persistence.paper_account_journal import PostgresPaperAccountJournal
from trading.persistence.paper_runtime_activation import (
    PaperRuntimeActivationStorageError,
    PostgresPaperRuntimeActivation,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
SCOPE = "paper:test"
ACCOUNT_KEY = "paper-main"
OWNER_GENERATION = 7


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _provision(dsn, *, account_key=ACCOUNT_KEY):
    account = new_paper_account(
        PaperAccountPolicy(account_key, "USDT", Decimal("0.01")),
        (PaperAccountBalance("USDT", Decimal("100.00"), Decimal("0.00")),),
    )
    return PostgresPaperAccountJournal(lambda: _connect(dsn)).provision_account(
        execution_scope=SCOPE,
        owner_generation=OWNER_GENERATION,
        account=account,
    )


def _context(
    opening,
    *,
    activation_id="activate-paper-1",
    source=PaperRuntimeActivationSource.LEGACY,
    expected_runtime_generation=0,
):
    readiness = PaperAccountReadinessContext(
        SCOPE,
        opening.account.policy.account_key,
        opening.owner_generation,
        opening.current.opening_payload_sha256,
    )
    return PaperRuntimeActivationContext(
        readiness,
        activation_id,
        source,
        expected_runtime_generation,
    )


def _activation(dsn, factory=None):
    return PostgresPaperRuntimeActivation(factory or (lambda: _connect(dsn)))


def _runtime_snapshot(dsn):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT mode, runtime_generation FROM np.paper_runtime_control"
            )
            control = tuple(cursor.fetchall())
            cursor.execute("""
                SELECT runtime_generation, activation_id, execution_scope,
                       account_key, owner_generation, opening_version,
                       opening_payload_sha256
                FROM np.paper_runtime_generations
                ORDER BY runtime_generation
                """)
            epochs = tuple(cursor.fetchall())
        return control, epochs
    finally:
        connection.close()


def _set_paused(dsn, generation):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE np.paper_runtime_control SET mode = 'PAUSED', "
                "runtime_generation = %s WHERE control_key",
                (generation,),
            )
        connection.commit()
    finally:
        connection.close()


class _TracingCursor:
    def __init__(self, cursor, statements):
        self._cursor = cursor
        self._statements = statements

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._cursor.__exit__(exc_type, exc, traceback)

    def execute(self, statement, parameters=None):
        self._statements.append(" ".join(str(statement).split()))
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


class _CommitThenRaiseConnection(_TrackingConnection):
    def commit(self):
        self.commits += 1
        self._connection.commit()
        raise psycopg2.OperationalError("simulated lost activation acknowledgement")


class _BlockingCommitConnection(_TrackingConnection):
    def __init__(self, connection, *, entered, release):
        super().__init__(connection)
        self._entered = entered
        self._release = release

    def commit(self):
        self.commits += 1
        self._entered.set()
        assert self._release.wait(timeout=10)
        return self._connection.commit()


class _BlockingCommitFactory:
    def __init__(self, dsn):
        self._dsn = dsn
        self.entered = Event()
        self.release = Event()

    def __call__(self):
        return _BlockingCommitConnection(
            _connect(self._dsn),
            entered=self.entered,
            release=self.release,
        )


class _MutationFailureCursor(_TracingCursor):
    def __init__(self, cursor, statements, fail_after):
        super().__init__(cursor, statements)
        self._fail_after = fail_after
        self.mutations = 0

    def execute(self, statement, parameters=None):
        result = super().execute(statement, parameters)
        normalized = " ".join(str(statement).split()).upper()
        if normalized.startswith(("INSERT ", "UPDATE ")):
            self.mutations += 1
            if self.mutations == self._fail_after:
                raise RuntimeError(f"injected failure after mutation {self.mutations}")
        if self._fail_after == 3 and normalized == "SET CONSTRAINTS ALL IMMEDIATE":
            raise RuntimeError("injected failure during pre-commit validation")
        return result


class _MutationFailureConnection(_TrackingConnection):
    def __init__(self, connection, *, statements, fail_after):
        super().__init__(connection, statements=statements)
        self._fail_after = fail_after

    def cursor(self):
        return _MutationFailureCursor(
            self._connection.cursor(),
            self._statements,
            self._fail_after,
        )


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


class _TrackingFactory:
    def __init__(self, dsn, *, statements=None, connection_type=_TrackingConnection):
        self._dsn = dsn
        self._statements = statements
        self._connection_type = connection_type
        self.connections = []

    def __call__(self):
        connection = self._connection_type(
            _connect(self._dsn),
            statements=self._statements,
        )
        self.connections.append(connection)
        return connection


def _instruction(suffix):
    intent = OrderIntent(
        client_order_id=f"order-{suffix}",
        decision_id=f"decision-{suffix}",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("1.00"),
        order_type=OrderType.MARKET,
        reference_price=Decimal("10.00"),
        leverage=2,
        created_at=NOW,
    )
    return PositionInstruction(
        f"position-{suffix}",
        PositionEffect.OPEN,
        intent,
        PositionExitContext(
            TakeProfitProfile.RANGING,
            Decimal("0.02"),
            Decimal("0.01"),
        ),
    )


def test_legacy_zero_activation_commits_epoch_one_and_exact_retry_rolls_back(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    context = _context(opening)
    first_factory = _TrackingFactory(migrated_postgres_dsn)

    activated = _activation(migrated_postgres_dsn, first_factory).activate(context)

    assert activated.disposition is PaperRuntimeActivationDisposition.ACTIVATED
    assert activated.runtime_generation == 1
    assert first_factory.connections[0].commits == 1
    assert _runtime_snapshot(migrated_postgres_dsn)[0] == (("ACTIVE", 1),)

    replay_statements = []
    replay_factory = _TrackingFactory(
        migrated_postgres_dsn,
        statements=replay_statements,
    )
    replayed = _activation(migrated_postgres_dsn, replay_factory).activate(context)
    assert replayed.disposition is PaperRuntimeActivationDisposition.REPLAYED
    assert replayed.context is context
    assert replay_factory.connections[0].commits == 0
    assert replay_factory.connections[0].rollbacks == 1
    assert not any(
        statement.upper().startswith(("INSERT ", "UPDATE ", "DELETE "))
        for statement in replay_statements
    )
    assert len(_runtime_snapshot(migrated_postgres_dsn)[1]) == 1


def test_activation_trace_uses_one_cursor_and_canonical_lock_then_mutation_order(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    statements = []
    factory = _TrackingFactory(migrated_postgres_dsn, statements=statements)

    receipt = _activation(migrated_postgres_dsn, factory).activate(_context(opening))

    assert receipt.disposition is PaperRuntimeActivationDisposition.ACTIVATED
    assert len(factory.connections) == 1
    normalized = tuple(statement.upper() for statement in statements)
    transaction = normalized.index("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
    timeout = normalized.index("SET LOCAL LOCK_TIMEOUT = '1S'")
    authority_lock = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("LOCK TABLE ONLY NP.ACCOUNT_BALANCES")
    )
    control_lock = next(
        index
        for index, statement in enumerate(normalized)
        if "FROM NP.PAPER_RUNTIME_CONTROL" in statement
        and "FOR UPDATE NOWAIT" in statement
    )
    account_lock = next(
        index
        for index, statement in enumerate(normalized)
        if "FROM NP.PAPER_ACCOUNT_STREAMS" in statement and "FOR UPDATE" in statement
    )
    epoch_insert = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("INSERT INTO NP.PAPER_RUNTIME_GENERATIONS")
    )
    control_update = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("UPDATE NP.PAPER_RUNTIME_CONTROL")
    )
    constraints = normalized.index("SET CONSTRAINTS ALL IMMEDIATE")
    assert (
        transaction
        < timeout
        < authority_lock
        < control_lock
        < account_lock
        < epoch_insert
        < control_update
        < constraints
    )
    authority_sql = normalized[authority_lock]
    for relation in (
        "NP.PAPER_RUNTIME_CONTROL",
        "NP.PAPER_RUNTIME_GENERATIONS",
        "NP.SCHEMA_MIGRATIONS",
    ):
        assert f"ONLY {relation}" in authority_sql


def test_blocked_trace_locks_account_before_position_without_activation_dml(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    PostgresOrderPositionJournal(
        lambda: _connect(migrated_postgres_dsn)
    ).reserve_instruction(
        execution_scope=SCOPE,
        instruction=_instruction("trace"),
    )
    statements = []
    result = _activation(
        migrated_postgres_dsn,
        _TrackingFactory(migrated_postgres_dsn, statements=statements),
    ).activate(_context(opening))

    assert type(result) is PaperRuntimeActivationBlocked
    normalized = tuple(statement.upper() for statement in statements)
    authority_lock = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("LOCK TABLE ONLY NP.ACCOUNT_BALANCES")
    )
    control_lock = next(
        index
        for index, statement in enumerate(normalized)
        if "FROM NP.PAPER_RUNTIME_CONTROL" in statement
        and "FOR UPDATE NOWAIT" in statement
    )
    account_lock = next(
        index
        for index, statement in enumerate(normalized)
        if "FROM NP.PAPER_ACCOUNT_STREAMS" in statement and "FOR UPDATE" in statement
    )
    position_lock = next(
        index
        for index, statement in enumerate(normalized)
        if "FROM NP.POSITION_STREAMS" in statement and "FOR UPDATE" in statement
    )
    assert authority_lock < control_lock < account_lock < position_lock
    assert not any(
        statement.startswith(("INSERT ", "UPDATE ", "DELETE "))
        for statement in normalized
    )


def test_paused_generation_reactivation_appends_exact_next_epoch(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    first = _context(opening)
    _activation(migrated_postgres_dsn).activate(first)
    _set_paused(migrated_postgres_dsn, 1)
    second = _context(
        opening,
        activation_id="activate-paper-2",
        source=PaperRuntimeActivationSource.PAUSED,
        expected_runtime_generation=1,
    )

    receipt = _activation(migrated_postgres_dsn).activate(second)

    assert receipt.disposition is PaperRuntimeActivationDisposition.ACTIVATED
    assert receipt.runtime_generation == 2
    control, epochs = _runtime_snapshot(migrated_postgres_dsn)
    assert control == (("ACTIVE", 2),)
    assert tuple(row[:2] for row in epochs) == (
        (1, "activate-paper-1"),
        (2, "activate-paper-2"),
    )
    replay_statements = []
    replayed_first = _activation(
        migrated_postgres_dsn,
        _TrackingFactory(migrated_postgres_dsn, statements=replay_statements),
    ).activate(first)
    assert replayed_first.disposition is PaperRuntimeActivationDisposition.REPLAYED
    assert replayed_first.runtime_generation == 1
    assert not any(
        statement.upper().startswith(("INSERT ", "UPDATE ", "DELETE "))
        for statement in replay_statements
    )


def test_stale_generation_or_reused_activation_identity_conflicts_without_delta(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    _activation(migrated_postgres_dsn).activate(_context(opening))
    before = _runtime_snapshot(migrated_postgres_dsn)
    conflicts = (
        _context(opening, activation_id="different-id"),
        PaperRuntimeActivationContext(
            PaperAccountReadinessContext(
                SCOPE,
                ACCOUNT_KEY,
                OWNER_GENERATION,
                "f" * 64,
            ),
            "activate-paper-1",
            PaperRuntimeActivationSource.LEGACY,
            0,
        ),
    )

    for context in conflicts:
        with pytest.raises(PaperRuntimeActivationConflict):
            _activation(migrated_postgres_dsn).activate(context)
        assert _runtime_snapshot(migrated_postgres_dsn) == before


def test_stray_matching_activation_id_under_legacy_zero_is_never_replayed(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    context = _context(opening)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO np.paper_runtime_generations (
                    runtime_generation, activation_id, execution_scope,
                    account_key, owner_generation, opening_version,
                    opening_payload_sha256
                ) VALUES (1, %s, %s, %s, %s, 1, %s)
                """,
                (
                    context.activation_id,
                    SCOPE,
                    ACCOUNT_KEY,
                    OWNER_GENERATION,
                    opening.current.opening_payload_sha256,
                ),
            )
        connection.commit()
    finally:
        connection.close()
    before = _runtime_snapshot(migrated_postgres_dsn)

    with pytest.raises(PaperRuntimeActivationConflict):
        _activation(migrated_postgres_dsn).activate(context)
    assert _runtime_snapshot(migrated_postgres_dsn) == before


def test_readiness_blocked_rolls_back_without_epoch_or_control_delta(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    writer = _connect(migrated_postgres_dsn)
    try:
        with writer.cursor() as cursor:
            cursor.execute(
                "INSERT INTO np.open_positions "
                "(symbol, side, entry_price, quantity, leverage) "
                "VALUES ('BTCUSDT', 'BUY', 10, 1, 2)"
            )
        writer.commit()
    finally:
        writer.close()
    before = _runtime_snapshot(migrated_postgres_dsn)

    result = _activation(migrated_postgres_dsn).activate(_context(opening))

    assert type(result) is PaperRuntimeActivationBlocked
    assert result.assessment.context is result.context.readiness
    assert result.assessment.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION in {
        finding.kind for finding in result.assessment.findings
    }
    assert _runtime_snapshot(migrated_postgres_dsn) == before


@pytest.mark.parametrize("corruption", ("control", "catalog"))
def test_missing_control_and_catalog_drift_return_blocked_without_activation_dml(
    migrated_postgres_dsn,
    corruption,
):
    opening = _provision(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            if corruption == "control":
                cursor.execute("DELETE FROM np.paper_runtime_control")
            else:
                cursor.execute(
                    "ALTER TABLE np.paper_runtime_control "
                    "DROP CONSTRAINT paper_runtime_control_mode"
                )
        connection.commit()
    finally:
        connection.close()
    before = _runtime_snapshot(migrated_postgres_dsn)

    result = _activation(migrated_postgres_dsn).activate(
        _context(opening, activation_id=f"blocked-{corruption}")
    )

    assert type(result) is PaperRuntimeActivationBlocked
    assert PaperAccountReadinessFindingKind.MIGRATION_DRIFT in {
        finding.kind for finding in result.assessment.findings
    }
    assert _runtime_snapshot(migrated_postgres_dsn) == before


def test_commit_unknown_is_resolved_by_exact_read_only_replay(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    context = _context(opening)
    failing_factory = _TrackingFactory(
        migrated_postgres_dsn,
        connection_type=_CommitThenRaiseConnection,
    )

    with pytest.raises(PaperRuntimeActivationCommitUnknown) as unknown:
        _activation(migrated_postgres_dsn, failing_factory).activate(context)
    assert unknown.value.context is context
    assert unknown.value.activation_id == context.activation_id

    retry_factory = _TrackingFactory(
        migrated_postgres_dsn,
        connection_type=_CommitThenRaiseConnection,
    )
    replayed = _activation(migrated_postgres_dsn, retry_factory).activate(context)
    assert replayed.disposition is PaperRuntimeActivationDisposition.REPLAYED
    assert retry_factory.connections[0].commits == 0
    assert retry_factory.connections[0].rollbacks == 1


@pytest.mark.parametrize("fail_after", (1, 2, 3))
def test_failure_after_epoch_or_control_mutation_rolls_back_complete_transition(
    migrated_postgres_dsn,
    fail_after,
):
    opening = _provision(migrated_postgres_dsn)
    before = _runtime_snapshot(migrated_postgres_dsn)
    factory = _MutationFailureFactory(migrated_postgres_dsn, fail_after)

    with pytest.raises(PaperRuntimeActivationStorageError) as failure:
        _activation(migrated_postgres_dsn, factory).activate(_context(opening))

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert factory.connection.rollbacks == 1
    assert factory.connection.commits == 0
    assert factory.connection.closed is True
    assert _runtime_snapshot(migrated_postgres_dsn) == before


@pytest.mark.parametrize(
    "relation",
    (
        "trades",
        "paper_account_streams",
        "paper_runtime_control",
        "paper_runtime_generations",
        "position_streams",
        "schema_migrations",
    ),
)
def test_concurrent_authority_writer_makes_activation_busy_without_delta(
    migrated_postgres_dsn,
    relation,
):
    opening = _provision(migrated_postgres_dsn)
    holder = _connect(migrated_postgres_dsn)
    try:
        with holder.cursor() as cursor:
            cursor.execute(f"LOCK TABLE ONLY np.{relation} IN ROW EXCLUSIVE MODE")
        before = _runtime_snapshot(migrated_postgres_dsn)
        started = monotonic()
        with pytest.raises(PaperRuntimeActivationBusy):
            _activation(migrated_postgres_dsn).activate(_context(opening))
        assert monotonic() - started < 5
        assert _runtime_snapshot(migrated_postgres_dsn) == before
    finally:
        holder.rollback()
        holder.close()


@pytest.mark.parametrize("writer_kind", ("account", "order"))
def test_real_dormant_writer_cannot_create_a_phantom_during_activation(
    migrated_postgres_dsn,
    writer_kind,
):
    opening = _provision(migrated_postgres_dsn)
    factory = _BlockingCommitFactory(migrated_postgres_dsn)

    def write():
        if writer_kind == "account":
            account = new_paper_account(
                PaperAccountPolicy("foreign-account", "USDT", Decimal("0.01")),
                (
                    PaperAccountBalance(
                        "USDT",
                        Decimal("100.00"),
                        Decimal("0.00"),
                    ),
                ),
            )
            return PostgresPaperAccountJournal(factory).provision_account(
                execution_scope=SCOPE,
                owner_generation=OWNER_GENERATION,
                account=account,
            )
        return PostgresOrderPositionJournal(factory).reserve_instruction(
            execution_scope=SCOPE,
            instruction=_instruction("phantom"),
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(write)
        assert factory.entered.wait(timeout=10)
        before = _runtime_snapshot(migrated_postgres_dsn)
        try:
            with pytest.raises(PaperRuntimeActivationBusy):
                _activation(migrated_postgres_dsn).activate(_context(opening))
            assert _runtime_snapshot(migrated_postgres_dsn) == before
        finally:
            factory.release.set()
        future.result(timeout=10)


@pytest.mark.parametrize("relation", ("paper_account_streams", "position_streams"))
def test_held_account_or_position_row_returns_bounded_busy_without_delta(
    migrated_postgres_dsn,
    relation,
):
    opening = _provision(migrated_postgres_dsn)
    if relation == "position_streams":
        journal = PostgresOrderPositionJournal(lambda: _connect(migrated_postgres_dsn))
        journal.reserve_instruction(
            execution_scope=SCOPE,
            instruction=_instruction("held"),
        )
    holder = _connect(migrated_postgres_dsn)
    try:
        with holder.cursor() as cursor:
            cursor.execute(f"SELECT * FROM np.{relation} FOR UPDATE")
        before = _runtime_snapshot(migrated_postgres_dsn)
        started = monotonic()
        with pytest.raises(PaperRuntimeActivationBusy):
            _activation(migrated_postgres_dsn).activate(_context(opening))
        assert monotonic() - started < 5
        assert _runtime_snapshot(migrated_postgres_dsn) == before
    finally:
        holder.rollback()
        holder.close()


def test_two_activators_produce_one_activation_and_one_exact_replay(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    context = _context(opening)
    gate = Event()

    def call():
        gate.wait(timeout=5)
        deadline = monotonic() + 10
        while True:
            try:
                return _activation(migrated_postgres_dsn).activate(context)
            except PaperRuntimeActivationBusy:
                if monotonic() >= deadline:
                    raise
                sleep(0.02)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(call), executor.submit(call))
        gate.set()
        results = tuple(future.result(timeout=20) for future in futures)

    assert {result.disposition for result in results} == {
        PaperRuntimeActivationDisposition.ACTIVATED,
        PaperRuntimeActivationDisposition.REPLAYED,
    }
    assert len(_runtime_snapshot(migrated_postgres_dsn)[1]) == 1
