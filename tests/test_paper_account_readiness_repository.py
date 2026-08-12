"""Fast PostgreSQL-boundary tests for paper-account readiness assessment."""

import ast
import importlib.util
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

import trading.persistence.paper_account_readiness as readiness_module
from trading.application.paper_account_readiness import (
    PaperAccountReadinessContext,
    PaperAccountReadinessDisposition,
    PaperAccountReadinessFindingKind,
)
from trading.domain.order_lifecycle import OrderLifecycleState
from trading.domain.paper_accounting import PaperAccountState
from trading.domain.positions import PositionState
from trading.persistence.migration_runner import load_migrations
from trading.persistence.order_position_journal import JournalReplayError
from trading.persistence.paper_account_journal import PaperAccountReplayError
from trading.persistence.paper_account_readiness import (
    PaperAccountReadinessError,
    PaperAccountReadinessInputError,
    PaperAccountReadinessStorageError,
    PostgresPaperAccountReadiness,
)

SCOPE = "paper:test"
ACCOUNT_KEY = "paper-main"
GENERATION = 7
OPENING_SHA256 = "a" * 64
MIGRATION_RELATION_METADATA = (
    "r",
    "p",
    False,
    False,
    False,
    False,
    False,
    False,
)
MIGRATION_COLUMNS = (
    (1, "version", "int4", "NO", "none", None),
    (2, "name", "text", "NO", "none", None),
    (3, "checksum", "bpchar", "NO", "none", 64),
    (4, "applied_at", "timestamptz", "NO", "now", None),
)
MIGRATION_CONSTRAINTS = (("p", [1], False, False, True),)
DURABLE_RELATION_ROWS = tuple(
    (relation, "r", "p", False, False, False, False, False, False)
    for relation in readiness_module._DURABLE_BUSINESS_RELATIONS
)


def context(**changes: object) -> PaperAccountReadinessContext:
    values = {
        "execution_scope": SCOPE,
        "account_key": ACCOUNT_KEY,
        "owner_generation": GENERATION,
        "opening_payload_sha256": OPENING_SHA256,
    }
    values.update(changes)
    return PaperAccountReadinessContext(**values)


class ScriptedDatabase:
    """Return isolated scripted connections and retain their observations."""

    def __init__(self, responder):
        self.responder = responder
        self.connections = []

    def connect(self):
        connection = ScriptedConnection(self, self.responder)
        self.connections.append(connection)
        return connection


class ScriptedConnection:
    autocommit = False
    status = STATUS_READY

    def __init__(self, database, responder):
        self.database = database
        self.responder = responder
        self.commands = []
        self.cursor_calls = 0
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def get_transaction_status(self):
        return TRANSACTION_STATUS_IDLE

    def cursor(self):
        self.cursor_calls += 1
        return ScriptedCursor(self, self.responder)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class ScriptedCursor:
    def __init__(self, connection, responder):
        self.connection = connection
        self.responder = responder
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, statement, parameters=None):
        sql = " ".join(statement.split())
        params = parameters or ()
        self.connection.commands.append((sql, tuple(params)))
        response = self.responder(sql, tuple(params))
        if isinstance(response, BaseException):
            raise response
        self.rows = list(response or ())

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


def finding_kinds(assessment):
    return tuple(finding.kind for finding in assessment.findings)


def migration_rows():
    return tuple(
        (migration.version, migration.name, migration.checksum)
        for migration in load_migrations()
    )


def order_reference(
    position_key,
    client_order_id,
    *,
    scope=SCOPE,
):
    return (position_key, scope, client_order_id)


def manifest_reference(
    position_key,
    client_order_id,
    *,
    account_key=ACCOUNT_KEY,
    scope=SCOPE,
):
    return (account_key, scope, position_key, client_order_id)


def replayed_account(
    *,
    scope=SCOPE,
    generation=GENERATION,
    opening_sha256=OPENING_SHA256,
    state=PaperAccountState.ACTIVE,
    reservations=(),
    records=(),
    client_order_ids=(),
):
    manifest_claims = tuple(
        value if isinstance(value, tuple) else ("position-1", value)
        for value in client_order_ids
    )
    return SimpleNamespace(
        execution_scope=scope,
        owner_generation=generation,
        opening_payload_sha256=opening_sha256,
        account=SimpleNamespace(
            state=state,
            reservations=tuple(reservations),
            records=tuple(records),
        ),
        batches=tuple(
            SimpleNamespace(position_key=position_key, client_order_id=client_order_id)
            for position_key, client_order_id in manifest_claims
        ),
    )


def replayed_position(
    *,
    orders=(),
    position_state=None,
):
    position = None if position_state is None else SimpleNamespace(state=position_state)
    return SimpleNamespace(
        projection=SimpleNamespace(
            position=position,
            orders=tuple(
                SimpleNamespace(
                    instruction=SimpleNamespace(
                        order_intent=SimpleNamespace(client_order_id=client_order_id)
                    ),
                    lifecycle=SimpleNamespace(state=state),
                )
                for client_order_id, state in orders
            ),
        )
    )


def snapshot_responder(
    *,
    relation="np.schema_migrations",
    applied=None,
    account_keys=(ACCOUNT_KEY,),
    position_keys=(),
    legacy_rows=None,
    fail_when=None,
    migration_relation_rows=(MIGRATION_RELATION_METADATA,),
    migration_columns=MIGRATION_COLUMNS,
    migration_constraints=MIGRATION_CONSTRAINTS,
    raw_order_references=(),
    raw_manifest_references=(),
    durable_relation_rows=DURABLE_RELATION_ROWS,
):
    applied_rows = migration_rows() if applied is None else tuple(applied)
    watermarks = {} if legacy_rows is None else dict(legacy_rows)

    def identity_rows(values):
        return [
            tuple(value) if isinstance(value, tuple) else (value, SCOPE)
            for value in values
        ]

    def respond(sql, params):
        if fail_when is not None and fail_when(sql, params):
            return RuntimeError("scripted database failure")
        if sql == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY":
            return ()
        if sql == "SELECT to_regclass(%s)":
            return [(relation,)]
        if sql.startswith("SELECT table_row.relkind, table_row.relpersistence"):
            return migration_relation_rows
        if "FROM information_schema.columns" in sql:
            return migration_columns
        if "FROM pg_constraint constraint_row" in sql:
            return migration_constraints
        if sql.startswith("SELECT version, name, checksum"):
            return applied_rows
        if "table_row.relname = ANY(%s)" in sql:
            return durable_relation_rows
        if "FROM np.orders" in sql:
            return raw_order_references
        if "FROM np.paper_account_batch_manifests" in sql:
            return raw_manifest_references
        if "FROM np.paper_account_streams" in sql and "account_key" in sql:
            return identity_rows(account_keys)
        if "FROM np.position_streams" in sql and "position_key" in sql:
            return identity_rows(position_keys)
        if sql.startswith("SELECT COUNT(*), MAX(id) FROM np."):
            relation_name = sql.rsplit(" ", 1)[-1]
            row_count, max_id = watermarks.get(relation_name, (0, None))
            return [(row_count, max_id)]
        raise AssertionError(f"unexpected SQL: {sql!r} {params!r}")

    return respond


def install_replayers(
    monkeypatch,
    *,
    accounts=None,
    positions=None,
):
    account_results = (
        {ACCOUNT_KEY: replayed_account()} if accounts is None else dict(accounts)
    )
    position_results = {} if positions is None else dict(positions)
    account_calls = []
    position_calls = []

    def replay_account(cursor, *, execution_scope, account_key, lock):
        account_calls.append((cursor, execution_scope, account_key, lock))
        result = account_results[account_key]
        if isinstance(result, BaseException):
            raise result
        return result

    def replay_position(cursor, *, execution_scope, position_key, lock):
        position_calls.append((cursor, execution_scope, position_key, lock))
        result = position_results[position_key]
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(readiness_module, "_replay_account_locked", replay_account)
    monkeypatch.setattr(readiness_module, "_replay_stream", replay_position)
    return account_calls, position_calls


def assess_snapshot(
    monkeypatch,
    *,
    responder=None,
    accounts=None,
    positions=None,
    assessment_context=None,
):
    database = ScriptedDatabase(responder or snapshot_responder())
    calls = install_replayers(
        monkeypatch,
        accounts=accounts,
        positions=positions,
    )
    result = PostgresPaperAccountReadiness(database.connect).assess(
        context() if assessment_context is None else assessment_context
    )
    return database, calls, result


def assert_read_only_snapshot(connection) -> None:
    assert connection.cursor_calls == 1
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True
    assert connection.commands[0][0] == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    )
    forbidden = re.compile(
        r"\b(ALTER|CREATE|DELETE|DROP|GRANT|INSERT|REVOKE|TRUNCATE|UPDATE)\b",
        re.IGNORECASE,
    )
    assert all(forbidden.search(sql) is None for sql, _params in connection.commands)


def test_repository_public_api_is_positional_and_errors_share_one_base() -> None:
    parameters = inspect.signature(PostgresPaperAccountReadiness.assess).parameters

    assert tuple(parameters) == ("self", "context")
    assert parameters["context"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert issubclass(PaperAccountReadinessInputError, PaperAccountReadinessError)
    assert issubclass(PaperAccountReadinessStorageError, PaperAccountReadinessError)


def test_invalid_context_fails_before_connect() -> None:
    database = ScriptedDatabase(lambda _sql, _params: ())
    repository = PostgresPaperAccountReadiness(database.connect)

    with pytest.raises(PaperAccountReadinessInputError):
        repository.assess(object())

    assert database.connections == []


def test_exact_empty_account_is_prepared_from_one_read_only_snapshot(
    monkeypatch,
) -> None:
    database, (account_calls, position_calls), result = assess_snapshot(monkeypatch)

    assert result.disposition is PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
    assert result.snapshot_authoritative is False
    assert result.context == context()
    assert result.applied_migrations == result.expected_migrations
    assert result.account_version == 0
    assert result.findings == ()
    assert tuple(
        (watermark.relation, watermark.row_count, watermark.max_id)
        for watermark in result.legacy_watermarks
    ) == (
        ("np.account_balances", 0, None),
        ("np.liquidations", 0, None),
        ("np.margin_history", 0, None),
        ("np.model_predictions", 0, None),
        ("np.open_positions", 0, None),
        ("np.trades", 0, None),
        ("np.trading_session_resets", 0, None),
    )
    assert len(database.connections) == 1
    connection = database.connections[0]
    assert_read_only_snapshot(connection)
    assert account_calls == [(account_calls[0][0], SCOPE, ACCOUNT_KEY, False)]
    assert position_calls == []


@pytest.mark.parametrize(
    ("relation", "applied", "kind"),
    (
        (None, (), PaperAccountReadinessFindingKind.MIGRATION_LEDGER_ABSENT),
        (
            "np.schema_migrations",
            migration_rows()[:2],
            PaperAccountReadinessFindingKind.MIGRATION_PENDING,
        ),
        (
            "np.schema_migrations",
            (*migration_rows()[:-1], (*migration_rows()[-1][:2], "d" * 64)),
            PaperAccountReadinessFindingKind.MIGRATION_DRIFT,
        ),
    ),
)
def test_migration_blockers_short_circuit_all_business_reads(
    monkeypatch,
    relation,
    applied,
    kind,
) -> None:
    database = ScriptedDatabase(snapshot_responder(relation=relation, applied=applied))
    account_calls, position_calls = install_replayers(monkeypatch)

    result = PostgresPaperAccountReadiness(database.connect).assess(context())

    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert finding_kinds(result) == (kind,)
    assert result.account_version is None
    assert result.legacy_watermarks == ()
    assert account_calls == []
    assert position_calls == []
    connection = database.connections[0]
    assert_read_only_snapshot(connection)
    assert all(
        "paper_account_streams" not in sql
        and "position_streams" not in sql
        and "COUNT(*)" not in sql
        for sql, _params in connection.commands
    )


def test_malformed_trailing_migration_row_is_canonical_drift_not_pending(
    monkeypatch,
) -> None:
    first = migration_rows()[0]
    database = ScriptedDatabase(
        snapshot_responder(applied=(first, (2, "bad name", "b" * 64)))
    )
    account_calls, position_calls = install_replayers(monkeypatch)

    result = PostgresPaperAccountReadiness(database.connect).assess(context())

    assert result.applied_migrations[0].version == 1
    assert finding_kinds(result) == (PaperAccountReadinessFindingKind.MIGRATION_DRIFT,)
    assert account_calls == []
    assert position_calls == []
    assert_read_only_snapshot(database.connections[0])


@pytest.mark.parametrize(
    "metadata",
    (
        {
            "migration_relation_rows": (
                ("v", "p", False, False, False, False, False, False),
            )
        },
        {
            "migration_relation_rows": (
                ("r", "p", False, True, False, False, False, False),
            )
        },
        {
            "migration_columns": (
                *MIGRATION_COLUMNS[:-1],
                (4, "applied_at", "timestamp", "NO", "now", None),
            )
        },
        {
            "migration_constraints": (
                *MIGRATION_CONSTRAINTS,
                ("u", [2], False, False, True),
            )
        },
    ),
)
def test_non_authoritative_migration_ledger_metadata_is_early_drift(
    monkeypatch,
    metadata,
) -> None:
    database = ScriptedDatabase(snapshot_responder(**metadata))
    account_calls, position_calls = install_replayers(monkeypatch)

    result = PostgresPaperAccountReadiness(database.connect).assess(context())

    assert finding_kinds(result) == (PaperAccountReadinessFindingKind.MIGRATION_DRIFT,)
    assert result.account_version is None
    assert result.legacy_watermarks == ()
    assert account_calls == []
    assert position_calls == []
    statements = tuple(sql for sql, _params in database.connections[0].commands)
    assert not any("FROM np.orders" in sql for sql in statements)
    assert not any("FROM np.paper_account_streams" in sql for sql in statements)
    assert_read_only_snapshot(database.connections[0])


@pytest.mark.parametrize("field_index", (1, 6))
def test_non_authoritative_durable_business_relation_is_early_drift(
    monkeypatch,
    field_index,
) -> None:
    first = list(DURABLE_RELATION_ROWS[0])
    first[field_index] = "v" if field_index == 1 else True
    drifted = (tuple(first), *DURABLE_RELATION_ROWS[1:])
    database = ScriptedDatabase(snapshot_responder(durable_relation_rows=drifted))
    account_calls, position_calls = install_replayers(monkeypatch)

    result = PostgresPaperAccountReadiness(database.connect).assess(context())

    assert result.applied_migrations == result.expected_migrations
    assert finding_kinds(result) == (PaperAccountReadinessFindingKind.MIGRATION_DRIFT,)
    assert result.account_version is None
    assert result.legacy_watermarks == ()
    assert account_calls == []
    assert position_calls == []
    statements = tuple(sql for sql, _params in database.connections[0].commands)
    assert not any("FROM np.orders" in sql for sql in statements)
    assert not any("FROM np.paper_account_streams" in sql for sql in statements)
    assert_read_only_snapshot(database.connections[0])


def test_missing_expected_account_is_blocked(monkeypatch) -> None:
    responder = snapshot_responder(account_keys=())
    database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts={},
    )

    assert result.account_version is None
    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED,
    )
    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert_read_only_snapshot(database.connections[0])


def test_extra_account_in_scope_is_blocked_and_every_account_is_replayed(
    monkeypatch,
) -> None:
    extra = "paper-extra"
    responder = snapshot_responder(account_keys=(extra, ACCOUNT_KEY))
    accounts = {
        ACCOUNT_KEY: replayed_account(),
        extra: replayed_account(),
    }

    database, (account_calls, _position_calls), result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts=accounts,
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.UNEXPECTED_ACCOUNT,
    )
    assert result.findings[0].subject_id == extra
    assert [call[2] for call in account_calls] == [extra, ACCOUNT_KEY]
    assert len({id(call[0]) for call in account_calls}) == 1
    assert_read_only_snapshot(database.connections[0])


def test_foreign_scope_account_and_position_are_globally_replayed_and_blocked(
    monkeypatch,
) -> None:
    foreign_key = "paper-foreign"
    foreign_scope = "paper:foreign"
    responder = snapshot_responder(
        account_keys=(
            (ACCOUNT_KEY, SCOPE),
            (foreign_key, foreign_scope),
        ),
        position_keys=(("position-foreign", foreign_scope),),
        raw_order_references=(
            order_reference(
                "position-foreign",
                "order-foreign",
                scope=foreign_scope,
            ),
        ),
        raw_manifest_references=(
            manifest_reference(
                "position-foreign",
                "order-foreign",
                account_key=foreign_key,
                scope=foreign_scope,
            ),
        ),
    )
    accounts = {
        ACCOUNT_KEY: replayed_account(),
        foreign_key: replayed_account(
            scope=foreign_scope,
            client_order_ids=(("position-foreign", "order-foreign"),),
        ),
    }
    positions = {
        "position-foreign": replayed_position(
            orders=(("order-foreign", OrderLifecycleState.FILLED),)
        )
    }

    _database, (account_calls, position_calls), result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts=accounts,
        positions=positions,
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.UNEXPECTED_ACCOUNT,
    )
    assert result.findings[0].subject_id == foreign_key
    assert {(call[1], call[2]) for call in account_calls} == {
        (SCOPE, ACCOUNT_KEY),
        (foreign_scope, foreign_key),
    }
    assert position_calls[0][1:] == (
        foreign_scope,
        "position-foreign",
        False,
    )


def test_expected_account_key_in_wrong_scope_is_missing_and_unexpected(
    monkeypatch,
) -> None:
    foreign_scope = "paper:foreign"
    responder = snapshot_responder(account_keys=((ACCOUNT_KEY, foreign_scope),))

    _database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts={ACCOUNT_KEY: replayed_account(scope=foreign_scope)},
    )

    assert result.account_version is None
    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED,
        PaperAccountReadinessFindingKind.UNEXPECTED_ACCOUNT,
    )
    assert tuple(finding.subject_id for finding in result.findings) == (
        ACCOUNT_KEY,
        ACCOUNT_KEY,
    )


@pytest.mark.parametrize(
    ("account", "kind", "subject_id"),
    (
        (
            replayed_account(generation=GENERATION + 1),
            PaperAccountReadinessFindingKind.ACCOUNT_PROVENANCE_MISMATCH,
            ACCOUNT_KEY,
        ),
        (
            replayed_account(state=PaperAccountState.INSOLVENT),
            PaperAccountReadinessFindingKind.ACCOUNT_INSOLVENT,
            ACCOUNT_KEY,
        ),
        (
            replayed_account(
                reservations=(SimpleNamespace(position_key="position-reserved"),)
            ),
            PaperAccountReadinessFindingKind.MARGIN_RESERVATION_PRESENT,
            "position-reserved",
        ),
    ),
)
def test_account_provenance_state_and_reservation_blockers_are_explicit(
    monkeypatch,
    account,
    kind,
    subject_id,
) -> None:
    _database, _calls, result = assess_snapshot(
        monkeypatch,
        accounts={ACCOUNT_KEY: account},
    )

    assert finding_kinds(result) == (kind,)
    assert result.findings[0].subject_id == subject_id
    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED


def test_account_replay_failure_requires_reconciliation(monkeypatch) -> None:
    database, _calls, result = assess_snapshot(
        monkeypatch,
        accounts={ACCOUNT_KEY: PaperAccountReplayError("corrupt account")},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
    )
    assert (
        result.disposition is PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
    )
    assert_read_only_snapshot(database.connections[0])


def test_position_replay_failure_requires_reconciliation(monkeypatch) -> None:
    responder = snapshot_responder(position_keys=("position-corrupt",))
    database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
        positions={"position-corrupt": JournalReplayError("corrupt position")},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
    )
    assert (
        result.disposition is PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
    )
    assert_read_only_snapshot(database.connections[0])


@pytest.mark.parametrize("detail", ("empty stream", "corrupt stream"))
def test_foreign_scope_empty_or_corrupt_position_fails_closed(
    monkeypatch,
    detail,
) -> None:
    foreign_key = f"position-{detail.split()[0]}"
    foreign_scope = "paper:foreign"
    responder = snapshot_responder(position_keys=((foreign_key, foreign_scope),))

    _database, (_account_calls, position_calls), result = assess_snapshot(
        monkeypatch,
        responder=responder,
        positions={foreign_key: JournalReplayError(detail)},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
    )
    assert result.findings[0].subject_id == foreign_key
    assert position_calls[0][1:] == (foreign_scope, foreign_key, False)


@pytest.mark.parametrize(
    "state",
    (
        OrderLifecycleState.PENDING,
        OrderLifecycleState.RECONCILING,
        OrderLifecycleState.OPEN,
        OrderLifecycleState.PARTIAL,
        OrderLifecycleState.CANCEL_PENDING,
    ),
)
def test_every_nonterminal_order_state_is_unresolved(monkeypatch, state) -> None:
    client_order_id = f"order-{state.value.lower()}"
    responder = snapshot_responder(
        position_keys=("position-1",),
        raw_order_references=(order_reference("position-1", client_order_id),),
        raw_manifest_references=(manifest_reference("position-1", client_order_id),),
    )
    account = replayed_account(client_order_ids=(client_order_id,))
    position = replayed_position(orders=((client_order_id, state),))

    _database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts={ACCOUNT_KEY: account},
        positions={"position-1": position},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.UNRESOLVED_SUBMISSION,
    )
    assert result.findings[0].subject_id == client_order_id
    assert (
        result.disposition is PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
    )


def test_terminal_order_without_account_manifest_is_unaccounted(monkeypatch) -> None:
    responder = snapshot_responder(
        position_keys=("position-1",),
        raw_order_references=(order_reference("position-1", "order-unaccounted"),),
    )
    position = replayed_position(
        orders=(("order-unaccounted", OrderLifecycleState.FILLED),)
    )

    _database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
        positions={"position-1": position},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
    )
    assert result.findings[0].subject_id == "order-unaccounted"
    assert (
        result.disposition is PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
    )


def test_foreign_scope_terminal_order_is_in_global_manifest_antijoin(
    monkeypatch,
) -> None:
    foreign_scope = "paper:foreign"
    responder = snapshot_responder(
        position_keys=(("position-foreign", foreign_scope),),
        raw_order_references=(
            order_reference(
                "position-foreign",
                "order-foreign",
                scope=foreign_scope,
            ),
        ),
    )
    position = replayed_position(
        orders=(("order-foreign", OrderLifecycleState.FAILED),)
    )

    _database, (_account_calls, position_calls), result = assess_snapshot(
        monkeypatch,
        responder=responder,
        positions={"position-foreign": position},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
    )
    assert result.findings[0].subject_id == "order-foreign"
    assert position_calls[0][1:] == (foreign_scope, "position-foreign", False)


def test_raw_orphan_order_outside_any_stream_is_visible_and_unaccounted(
    monkeypatch,
) -> None:
    responder = snapshot_responder(
        raw_order_references=(order_reference("position-orphan", "order-orphan"),),
    )

    _database, (_account_calls, position_calls), result = assess_snapshot(
        monkeypatch,
        responder=responder,
    )

    assert position_calls == []
    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
    )
    assert {
        (finding.subject_kind, finding.subject_id) for finding in result.findings
    } == {
        ("durable_relation", "np.orders"),
        ("client_order", "order-orphan"),
    }


def test_duplicate_raw_order_claim_is_not_hidden_by_set_projection(
    monkeypatch,
) -> None:
    order = order_reference("position-1", "order-duplicate")
    manifest = manifest_reference("position-1", "order-duplicate")
    responder = snapshot_responder(
        position_keys=("position-1",),
        raw_order_references=(order, order),
        raw_manifest_references=(manifest,),
    )
    account = replayed_account(client_order_ids=("order-duplicate",))
    position = replayed_position(
        orders=(("order-duplicate", OrderLifecycleState.FILLED),)
    )

    _database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts={ACCOUNT_KEY: account},
        positions={"position-1": position},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
    )
    assert {
        (finding.subject_kind, finding.subject_id) for finding in result.findings
    } == {
        ("durable_relation", "np.orders"),
        ("client_order", "order-duplicate"),
    }


def test_raw_orphan_manifest_outside_any_account_is_not_hidden(monkeypatch) -> None:
    responder = snapshot_responder(
        raw_manifest_references=(
            manifest_reference("position-orphan", "manifest-orphan"),
        ),
    )

    _database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
    )
    assert {
        (finding.subject_kind, finding.subject_id) for finding in result.findings
    } == {
        ("durable_relation", "np.paper_account_batch_manifests"),
        ("client_order", "manifest-orphan"),
    }


def test_duplicate_raw_manifest_claim_is_not_hidden_by_set_projection(
    monkeypatch,
) -> None:
    order = order_reference("position-1", "manifest-duplicate")
    manifest = manifest_reference("position-1", "manifest-duplicate")
    responder = snapshot_responder(
        position_keys=("position-1",),
        raw_order_references=(order,),
        raw_manifest_references=(manifest, manifest),
    )
    account = replayed_account(client_order_ids=("manifest-duplicate",))
    position = replayed_position(
        orders=(("manifest-duplicate", OrderLifecycleState.FILLED),)
    )

    _database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts={ACCOUNT_KEY: account},
        positions={"position-1": position},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
    )
    assert {
        (finding.subject_kind, finding.subject_id) for finding in result.findings
    } == {
        ("durable_relation", "np.paper_account_batch_manifests"),
        ("client_order", "manifest-duplicate"),
    }


def test_manifest_without_durable_order_is_account_replay_failure(monkeypatch) -> None:
    account = replayed_account(client_order_ids=("orphan-manifest",))
    responder = snapshot_responder(
        raw_manifest_references=(manifest_reference("position-1", "orphan-manifest"),),
    )

    _database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts={ACCOUNT_KEY: account},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
    )
    assert result.findings[0].subject_id == "orphan-manifest"


def test_durable_open_position_blocks_even_when_order_is_accounted(monkeypatch) -> None:
    responder = snapshot_responder(
        position_keys=("position-open",),
        raw_order_references=(order_reference("position-open", "order-filled"),),
        raw_manifest_references=(manifest_reference("position-open", "order-filled"),),
    )
    account = replayed_account(client_order_ids=(("position-open", "order-filled"),))
    position = replayed_position(
        orders=(("order-filled", OrderLifecycleState.FILLED),),
        position_state=PositionState.OPEN,
    )

    _database, _calls, result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts={ACCOUNT_KEY: account},
        positions={"position-open": position},
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.DURABLE_OPEN_POSITION,
    )
    assert result.findings[0].subject_id == "position-open"
    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED


def test_legacy_watermarks_are_exact_and_open_positions_block(monkeypatch) -> None:
    legacy_rows = {
        "np.account_balances": (2, 11),
        "np.liquidations": (1, 12),
        "np.margin_history": (3, 13),
        "np.model_predictions": (4, 14),
        "np.open_positions": (1, 15),
        "np.trades": (5, 16),
        "np.trading_session_resets": (1, 17),
    }
    responder = snapshot_responder(legacy_rows=legacy_rows)

    database, _calls, result = assess_snapshot(monkeypatch, responder=responder)

    assert {
        watermark.relation: (watermark.row_count, watermark.max_id)
        for watermark in result.legacy_watermarks
    } == legacy_rows
    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION,
    )
    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert_read_only_snapshot(database.connections[0])


def test_late_query_failure_returns_no_partial_assessment(monkeypatch) -> None:
    responder = snapshot_responder(
        fail_when=lambda sql, _params: sql.endswith("np.open_positions")
    )
    database = ScriptedDatabase(responder)
    install_replayers(monkeypatch)
    repository = PostgresPaperAccountReadiness(database.connect)

    with pytest.raises(PaperAccountReadinessStorageError) as failure:
        repository.assess(context())

    assert isinstance(failure.value.__cause__, RuntimeError)
    connection = database.connections[0]
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True


def test_connection_failure_is_typed_without_partial_result() -> None:
    def fail():
        raise RuntimeError("connect failed")

    with pytest.raises(PaperAccountReadinessStorageError) as failure:
        PostgresPaperAccountReadiness(fail).assess(context())

    assert isinstance(failure.value.__cause__, Exception)


def test_storage_boundary_accepts_exact_identifier_limits(monkeypatch) -> None:
    maximum_scope = "s" * 128
    maximum_account_key = "a" * 255
    maximum_position_key = "p" * 255
    assessment_context = context(
        execution_scope=maximum_scope,
        account_key=maximum_account_key,
    )
    responder = snapshot_responder(
        account_keys=((maximum_account_key, maximum_scope),),
        position_keys=((maximum_position_key, maximum_scope),),
        raw_order_references=(
            order_reference(
                maximum_position_key,
                "order-at-limit",
                scope=maximum_scope,
            ),
        ),
        raw_manifest_references=(
            manifest_reference(
                maximum_position_key,
                "order-at-limit",
                account_key=maximum_account_key,
                scope=maximum_scope,
            ),
        ),
    )
    account = replayed_account(
        scope=maximum_scope,
        client_order_ids=((maximum_position_key, "order-at-limit"),),
    )
    position = replayed_position(
        orders=(("order-at-limit", OrderLifecycleState.FILLED),)
    )

    database, (account_calls, position_calls), result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts={maximum_account_key: account},
        positions={maximum_position_key: position},
        assessment_context=assessment_context,
    )

    assert result.disposition is PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
    assert result.findings == ()
    assert account_calls[0][1:3] == (maximum_scope, maximum_account_key)
    assert position_calls[0][1:3] == (maximum_scope, maximum_position_key)
    assert_read_only_snapshot(database.connections[0])


def test_maximum_foreign_identities_can_be_reported_after_replay_failure(
    monkeypatch,
) -> None:
    maximum_scope = "s" * 128
    maximum_account_key = "a" * 255
    maximum_position_key = "p" * 255
    responder = snapshot_responder(
        account_keys=(
            (ACCOUNT_KEY, SCOPE),
            (maximum_account_key, maximum_scope),
        ),
        position_keys=((maximum_position_key, maximum_scope),),
    )

    database, (account_calls, position_calls), result = assess_snapshot(
        monkeypatch,
        responder=responder,
        accounts={
            ACCOUNT_KEY: replayed_account(),
            maximum_account_key: PaperAccountReplayError("corrupt account"),
        },
        positions={
            maximum_position_key: JournalReplayError("corrupt position"),
        },
    )

    assert finding_kinds(result) == (
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.UNEXPECTED_ACCOUNT,
    )
    assert {finding.subject_id for finding in result.findings} == {
        maximum_account_key,
        maximum_position_key,
    }
    assert (maximum_scope, maximum_account_key) in {
        (call[1], call[2]) for call in account_calls
    }
    assert position_calls[0][1:3] == (maximum_scope, maximum_position_key)
    assert_read_only_snapshot(database.connections[0])


def test_repository_module_contains_no_dml_statement() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "trading"
        / "persistence"
        / "paper_account_readiness.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    sql_values = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(
            isinstance(target, ast.Name) and target.id.endswith("_SQL")
            for target in targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            sql_values.append(node.value.value)

    assert sql_values
    forbidden = re.compile(
        r"\b(ALTER|CREATE|DELETE|DROP|GRANT|INSERT|REVOKE|TRUNCATE|UPDATE)\b",
        re.IGNORECASE,
    )
    assert all(forbidden.search(sql) is None for sql in sql_values)
    source = source_path.read_text(encoding="utf-8")
    assert ".replay_account(" not in source
    assert ".replay_order(" not in source
    assert ".list_accounts(" not in source
    assert ".list_unresolved_submissions(" not in source


_REPOSITORY_MODULE = "trading.persistence.paper_account_readiness"
_REPOSITORY_EXPORTS = {
    "PaperAccountReadinessError",
    "PaperAccountReadinessInputError",
    "PaperAccountReadinessStorageError",
    "PostgresPaperAccountReadiness",
}


def _attribute_path(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return (*parent, node.attr) if parent is not None else None
    return None


def _uses_readiness_repository(source):
    """Detect static, facade, relative, aliased, and literal-dynamic use."""
    tree = ast.parse(source)
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}
    builtin_import_aliases = {"__import__"}
    trading_aliases = set()
    persistence_aliases = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _REPOSITORY_MODULE or alias.name.startswith(
                    f"{_REPOSITORY_MODULE}."
                ):
                    return True
                if alias.name == "trading":
                    trading_aliases.add(alias.asname or "trading")
                elif alias.name == "trading.persistence":
                    persistence_aliases.add(alias.asname or "persistence")
                    if alias.asname is None:
                        trading_aliases.add("trading")
                elif alias.name == "importlib":
                    importlib_aliases.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            module = node.module or ""
            if module == _REPOSITORY_MODULE or (
                node.level and module.endswith("paper_account_readiness")
            ):
                return True
            if module == "trading.persistence" or (
                node.level and module == "persistence"
            ):
                if imported & (_REPOSITORY_EXPORTS | {"paper_account_readiness", "*"}):
                    return True
            if (
                node.level
                and not module
                and imported
                & {
                    "paper_account_readiness",
                    "*",
                }
            ):
                return True
            if module == "trading" and "persistence" in imported:
                persistence_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "persistence"
                )
            if module == "importlib" and "import_module" in imported:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if module == "builtins" and "__import__" in imported:
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            value = node.value
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if not names or value is None:
                continue
            path = _attribute_path(value)
            is_import_module = (
                isinstance(value, ast.Name) and value.id in import_module_aliases
            ) or (
                path is not None
                and len(path) == 2
                and path[0] in importlib_aliases
                and path[1] == "import_module"
            )
            is_builtin_import = (
                isinstance(value, ast.Name) and value.id in builtin_import_aliases
            )
            target_set = (
                import_module_aliases if is_import_module else builtin_import_aliases
            )
            if is_import_module or is_builtin_import:
                for name in names:
                    if name not in target_set:
                        target_set.add(name)
                        changed = True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = (
            node.args[0].value
            if node.args and isinstance(node.args[0], ast.Constant)
            else next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
                ),
                None,
            )
        )
        if not isinstance(target, str):
            continue
        function_path = _attribute_path(node.func)
        dynamic = (
            isinstance(node.func, ast.Name)
            and node.func.id in import_module_aliases | builtin_import_aliases
        ) or (
            function_path is not None
            and len(function_path) == 2
            and function_path[0] in importlib_aliases
            and function_path[1] == "import_module"
        )
        if dynamic and (
            target == _REPOSITORY_MODULE or target.startswith(f"{_REPOSITORY_MODULE}.")
        ):
            return True
        if dynamic and target.startswith("."):
            package = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "package"
                    and isinstance(keyword.value, ast.Constant)
                ),
                (
                    node.args[1].value
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    else None
                ),
            )
            if isinstance(package, str):
                try:
                    if (
                        importlib.util.resolve_name(target, package)
                        == _REPOSITORY_MODULE
                    ):
                        return True
                except (ImportError, ValueError):
                    pass

    for node in ast.walk(tree):
        path = _attribute_path(node)
        if path is None or path[-1] not in (
            _REPOSITORY_EXPORTS | {"paper_account_readiness"}
        ):
            continue
        if path[0] in persistence_aliases:
            return True
        if len(path) >= 3 and path[0] in trading_aliases and path[1] == "persistence":
            return True
    return False


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence.paper_account_readiness "
        "import PostgresPaperAccountReadiness",
        "import trading.persistence.paper_account_readiness as readiness",
        "from trading.persistence import PostgresPaperAccountReadiness",
        "from trading.persistence import paper_account_readiness",
        "from trading.persistence import *",
        "import trading as root\nroot.persistence.PostgresPaperAccountReadiness",
        "from trading import persistence as store\nstore.paper_account_readiness",
        "from .persistence.paper_account_readiness import "
        "PaperAccountReadinessStorageError",
        "from .persistence import paper_account_readiness",
        "from importlib import import_module as load\n"
        "load('trading.persistence.paper_account_readiness')",
        "import importlib as loader\n"
        "loader.import_module(name='trading.persistence.paper_account_readiness')",
        "__import__('trading.persistence.paper_account_readiness')",
        "load = __import__\nload('trading.persistence.paper_account_readiness')",
        "from importlib import import_module\nload = import_module\n"
        "load('.paper_account_readiness', 'trading.persistence')",
    ),
)
def test_repository_consumer_detector_catches_supported_forms(source) -> None:
    assert _uses_readiness_repository(source)


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence import apply_migrations",
        "import trading.persistence",
        "from trading.application import PaperAccountReadinessContext",
        "name = 'trading.persistence.paper_account_readiness'",
    ),
)
def test_repository_consumer_detector_allows_unrelated_forms(source) -> None:
    assert not _uses_readiness_repository(source)


def test_readiness_repository_is_unwired_and_not_facade_exported() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "persistence" / "paper_account_readiness.py"
    facade_path = root / "trading" / "persistence" / "__init__.py"
    consumers = []
    for source_path in root.rglob("*.py"):
        if (
            source_path == module_path
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        if _uses_readiness_repository(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))

    assert consumers == []
    assert not _uses_readiness_repository(facade_path.read_text(encoding="utf-8"))
