"""PostgreSQL 15 proofs for the dormant paper-account readiness snapshot."""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg2
import pytest
from psycopg2 import sql

from trading.application.paper_account_readiness import (
    PaperAccountReadinessContext,
    PaperAccountReadinessDisposition,
    PaperAccountReadinessFindingKind,
)
from trading.domain.order_lifecycle import (
    ConfirmedFill,
    SubmissionAcknowledged,
    SubmissionFailed,
)
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionStatus,
)
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
from trading.persistence.journal_codec import encode_position_instruction
from trading.persistence.migration_runner import apply_migrations, load_migrations
from trading.persistence.order_position_journal import PostgresOrderPositionJournal
from trading.persistence.paper_account_journal import PostgresPaperAccountJournal
from trading.persistence.paper_account_journal_codec import (
    encode_paper_account_opening,
)
from trading.persistence.paper_account_readiness import PostgresPaperAccountReadiness

NOW = datetime(2026, 8, 12, 12, 0, 0, 123456, tzinfo=timezone.utc)
SCOPE = "paper:test"
ACCOUNT_KEY = "paper-main"
GENERATION = 7
LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _readiness_opening_role(database_name):
    return f"{database_name}_opening"


@pytest.fixture(autouse=True)
def _cleanup_readiness_opening_role(postgres_database_dsn):
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


def _apply(dsn, count=None):
    migrations = load_migrations()
    selected = migrations if count is None else migrations[:count]
    connection = _connect(dsn)
    try:
        apply_migrations(connection, selected)
    finally:
        connection.close()
    return migrations


def _opening(account_key=ACCOUNT_KEY):
    return new_paper_account(
        PaperAccountPolicy(account_key, "USDT", Decimal("0.01")),
        (PaperAccountBalance("USDT", Decimal("100.00"), Decimal("0.00")),),
    )


def _provision(
    dsn,
    *,
    account_key=ACCOUNT_KEY,
    scope=SCOPE,
    generation=GENERATION,
    with_provenance=True,
):
    account = _opening(account_key)
    encoded = encode_paper_account_opening(scope, generation, account)
    PostgresPaperAccountJournal(lambda: _connect(dsn)).provision_account(
        execution_scope=scope,
        owner_generation=generation,
        account=account,
    )
    if with_provenance:
        _seed_fresh_opening_provenance(dsn, encoded)
    return encoded


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_fresh_opening_provenance(dsn, encoded):
    candidate_payload = _canonical_json({"schema_version": 1})
    candidate_sha256 = _sha256(candidate_payload)
    pin_sha256 = _sha256("readiness-test-pin")
    deployment_incarnation = "readiness-test-deployment"
    admission_payload = _canonical_json(
        {
            "candidate_sha256": candidate_sha256,
            "deployment_incarnation_id": deployment_incarnation,
            "pin_authority_record_sha256": pin_sha256,
            "schema_version": 1,
        }
    )
    empty_payload = _canonical_json({})
    receipt_payload = _canonical_json({"schema_version": 1})
    plain_sha256 = _sha256(empty_payload)
    intent_sha256 = hashlib.sha256(
        b"ELVIS\x00fresh-opening-intent\x00v1\x00" + empty_payload.encode("utf-8")
    ).hexdigest()
    migration = load_migrations()[-1]
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            database_name = cursor.fetchone()[0]
            opening_anchor_role = _readiness_opening_role(database_name)
            cursor.execute(
                """
                INSERT INTO np.paper_fresh_opening_admissions (
                    candidate_payload_sha256,
                    pin_authority_record_sha256,
                    deployment_incarnation_id,
                    admission_payload,
                    admission_payload_sha256
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (control_key) DO NOTHING
                """,
                (
                    candidate_sha256,
                    pin_sha256,
                    deployment_incarnation,
                    admission_payload,
                    _sha256(admission_payload),
                ),
            )
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = %s)",
                (opening_anchor_role,),
            )
            if not cursor.fetchone()[0]:
                cursor.execute(
                    sql.SQL(
                        "CREATE ROLE {} NOLOGIN NOINHERIT NOSUPERUSER "
                        "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS "
                        "CONNECTION LIMIT -1 PASSWORD NULL"
                    ).format(sql.Identifier(opening_anchor_role))
                )
            cursor.execute(
                sql.SQL("COMMENT ON ROLE {} IS %s").format(
                    sql.Identifier(opening_anchor_role)
                ),
                (
                    "elvis-postgres-bootstrap:v2:"
                    f"{database_name}:opening:{_sha256(admission_payload)}",
                ),
            )
            cursor.execute("SELECT np.paper_terminal_catalog_fingerprint()")
            terminal_catalog_sha256 = cursor.fetchone()[0]
            cursor.execute(
                "COMMENT ON SCHEMA np IS %s",
                (
                    "elvis-postgres-bootstrap-schema:v2:"
                    f"{database_name}:{terminal_catalog_sha256}",
                ),
            )
            cursor.execute(
                """
                SELECT np.paper_fresh_opening_database_incarnation(
                    current_database(),
                    (SELECT system_identifier::numeric FROM pg_control_system()),
                    %s, %s, %s, %s, current_user, %s, %s
                )
                """,
                (
                    migration.version,
                    migration.name,
                    migration.checksum,
                    terminal_catalog_sha256,
                    opening_anchor_role,
                    deployment_incarnation,
                ),
            )
            database_incarnation_sha256 = cursor.fetchone()[0]
            cursor.execute(
                """
                INSERT INTO np.paper_fresh_opening_nonces (
                    trust_domain,
                    signer_key_id,
                    nonce,
                    candidate_payload_sha256
                ) VALUES ('readiness-test', 'readiness-key', repeat('1', 64), %s)
                ON CONFLICT (trust_domain, signer_key_id, nonce) DO NOTHING
                """,
                (candidate_sha256,),
            )
            cursor.execute(
                """
                INSERT INTO np.paper_fresh_opening_provisionings (
                    trust_domain,
                    signer_key_id,
                    nonce,
                    logical_target,
                    execution_scope,
                    account_key,
                    owner_generation,
                    collateral_asset,
                    opening_version,
                    intent_payload,
                    intent_payload_sha256,
                    approval_payload,
                    approval_payload_sha256,
                    trust_policy_payload,
                    trust_policy_payload_sha256,
                    candidate_payload,
                    candidate_payload_sha256,
                    opening_payload,
                    opening_payload_sha256,
                    opening_receipt_payload,
                    opening_receipt_payload_sha256,
                    provisioning_receipt_payload,
                    provisioning_receipt_payload_sha256,
                    database_name,
                    system_identifier,
                    control_plane_role,
                    opening_anchor_role,
                    migration_version,
                    migration_name,
                    migration_checksum,
                    terminal_catalog_sha256,
                    deployment_incarnation_id,
                    database_incarnation_id,
                    pin_authority_record_sha256,
                    runtime_mode,
                    runtime_generation,
                    authority_transition_sequence,
                    writer_fence,
                    runtime_activation_authorized,
                    trading_authorized,
                    stale_on_return,
                    authority_evaluated_at
                ) VALUES (
                    'readiness-test',
                    'readiness-key',
                    repeat('1', 64),
                    'readiness-logical-target',
                    %s, %s, %s, %s, 1,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    current_database(),
                    (SELECT system_identifier::numeric FROM pg_control_system()),
                    current_user,
                    %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    'LEGACY', 0, 0, 0, FALSE, FALSE, TRUE,
                    transaction_timestamp()
                )
                ON CONFLICT (control_key) DO NOTHING
                """,
                (
                    encoded.execution_scope,
                    encoded.account_key,
                    encoded.owner_generation,
                    encoded.collateral_asset,
                    empty_payload,
                    intent_sha256,
                    empty_payload,
                    plain_sha256,
                    empty_payload,
                    plain_sha256,
                    candidate_payload,
                    candidate_sha256,
                    encoded.opening_payload,
                    encoded.opening_payload_sha256,
                    empty_payload,
                    plain_sha256,
                    receipt_payload,
                    _sha256(receipt_payload),
                    opening_anchor_role,
                    migration.version,
                    migration.name,
                    migration.checksum,
                    terminal_catalog_sha256,
                    deployment_incarnation,
                    database_incarnation_sha256,
                    pin_sha256,
                ),
            )
        connection.commit()
    finally:
        connection.close()


def _context(encoded, **changes):
    values = {
        "execution_scope": encoded.execution_scope,
        "account_key": encoded.account_key,
        "owner_generation": encoded.owner_generation,
        "opening_payload_sha256": encoded.opening_payload_sha256,
    }
    values.update(changes)
    return PaperAccountReadinessContext(**values)


def _placeholder_context():
    return PaperAccountReadinessContext(SCOPE, ACCOUNT_KEY, GENERATION, "a" * 64)


def _assess(dsn, context, *, factory=None):
    connection_factory = factory or (lambda: _connect(dsn))
    return PostgresPaperAccountReadiness(connection_factory).assess(context)


def _finding_kinds(result):
    return {finding.kind for finding in result.findings}


def _instruction(suffix, *, scope=SCOPE, client_order_id=None):
    del scope
    intent = OrderIntent(
        client_order_id=client_order_id or f"order-{suffix}",
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
        position_key=f"position-{suffix}",
        effect=PositionEffect.OPEN,
        order_intent=intent,
        exit_context=PositionExitContext(
            TakeProfitProfile.RANGING,
            Decimal("0.02"),
            Decimal("0.01"),
        ),
    )


def _reserve(dsn, instruction, *, scope=SCOPE):
    journal = PostgresOrderPositionJournal(lambda: _connect(dsn))
    journal.reserve_instruction(execution_scope=scope, instruction=instruction)
    return journal


def _insert_encoded_order(cursor, instruction, *, scope=SCOPE):
    encoded = encode_position_instruction(instruction)
    cursor.execute(
        """
        INSERT INTO np.orders (
            client_order_id,
            decision_id,
            position_key,
            execution_scope,
            symbol,
            position_effect,
            instruction_version,
            instruction_payload,
            instruction_payload_sha256
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
        """,
        (
            encoded.client_order_id,
            encoded.decision_id,
            encoded.position_key,
            scope,
            encoded.symbol,
            encoded.position_effect,
            encoded.instruction_version,
            encoded.instruction_payload,
            encoded.instruction_payload_sha256,
        ),
    )


class _TracingFactory:
    def __init__(self, dsn, *, after_execute=None):
        self.dsn = dsn
        self.after_execute = after_execute
        self.connections = []

    def __call__(self):
        value = _TracingConnection(_connect(self.dsn), after_execute=self.after_execute)
        self.connections.append(value)
        return value


class _TracingConnection:
    def __init__(self, connection, *, after_execute=None):
        self._connection = connection
        self._after_execute = after_execute
        self.commands = []
        self.cursor_calls = 0
        self.commit_calls = 0
        self.rollback_calls = 0

    @property
    def autocommit(self):
        return self._connection.autocommit

    @property
    def status(self):
        return self._connection.status

    def get_transaction_status(self):
        return self._connection.get_transaction_status()

    def cursor(self):
        self.cursor_calls += 1
        return _TracingCursor(self, self._connection.cursor())

    def commit(self):
        self.commit_calls += 1
        return self._connection.commit()

    def rollback(self):
        self.rollback_calls += 1
        return self._connection.rollback()

    def close(self):
        return self._connection.close()


class _RetainedTracingConnection(_TracingConnection):
    def close(self):
        return None


class _TracingCursor:
    def __init__(self, connection, cursor):
        self._connection = connection
        self._cursor = cursor

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._cursor.__exit__(exc_type, exc, traceback)

    def execute(self, statement, parameters=None):
        normalized = " ".join(statement.split())
        self._connection.commands.append((normalized, parameters))
        result = self._cursor.execute(statement, parameters)
        callback = self._connection._after_execute
        if callback is not None:
            callback(normalized)
        return result

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()


def test_exact_empty_account_is_prepared_with_all_legacy_watermarks(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)

    result = _assess(migrated_postgres_dsn, _context(encoded))

    assert result.disposition is PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
    assert result.snapshot_authoritative is False
    assert result.account_version == 0
    assert result.findings == ()
    assert tuple(item.relation for item in result.legacy_watermarks) == (
        LEGACY_RELATIONS
    )
    assert all(
        (item.row_count, item.max_id) == (0, None) for item in result.legacy_watermarks
    )


def test_readiness_ignores_pg_temp_catalog_relation_shadow(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    raw_connection = _connect(migrated_postgres_dsn)
    try:
        with raw_connection.cursor() as cursor:
            cursor.execute(
                "CREATE TEMP SEQUENCE fresh_opening_temp_shadow_probe START WITH 1"
            )
            cursor.execute("""
                CREATE FUNCTION pg_temp.bump_fresh_opening_shadow_probe(
                    requested_name name
                )
                RETURNS name
                LANGUAGE plpgsql
                VOLATILE
                AS $function$
                BEGIN
                    PERFORM pg_catalog.nextval(
                        'pg_temp.fresh_opening_temp_shadow_probe'
                    );
                    RETURN requested_name;
                END
                $function$
                """)
            cursor.execute("""
                CREATE TEMP VIEW pg_roles AS
                SELECT
                    role_row.oid,
                    pg_temp.bump_fresh_opening_shadow_probe(role_row.rolname)
                        AS rolname,
                    role_row.rolsuper,
                    role_row.rolinherit,
                    role_row.rolcreaterole,
                    role_row.rolcreatedb,
                    role_row.rolcanlogin,
                    role_row.rolreplication,
                    role_row.rolconnlimit,
                    role_row.rolbypassrls,
                    role_row.rolconfig
                FROM pg_catalog.pg_roles role_row
                """)
        raw_connection.commit()

        retained = _RetainedTracingConnection(raw_connection)
        result = _assess(
            migrated_postgres_dsn,
            _context(encoded),
            factory=lambda: retained,
        )

        with raw_connection.cursor() as cursor:
            cursor.execute(
                "SELECT last_value, is_called "
                "FROM pg_temp.fresh_opening_temp_shadow_probe"
            )
            assert cursor.fetchone() == (1, False)
        raw_connection.rollback()
    finally:
        raw_connection.close()

    assert result.disposition is PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
    assert result.findings == ()
    assert any(
        "np.paper_fresh_opening_target_is_current()" in statement
        for statement, _ in retained.commands
    )


@pytest.mark.parametrize(
    "drift",
    ("schema_marker", "opening_anchor", "candidate_payload", "catalog"),
)
def test_current_fresh_opening_physical_or_candidate_drift_is_blocking(
    migrated_postgres_dsn,
    drift,
):
    encoded = _provision(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            if drift == "schema_marker":
                cursor.execute("COMMENT ON SCHEMA np IS 'drifted'")
            elif drift == "opening_anchor":
                cursor.execute("SELECT current_database()")
                role_name = _readiness_opening_role(cursor.fetchone()[0])
                cursor.execute(
                    sql.SQL("COMMENT ON ROLE {} IS 'drifted'").format(
                        sql.Identifier(role_name)
                    )
                )
            elif drift == "candidate_payload":
                cursor.execute(
                    "ALTER TABLE np.paper_fresh_opening_provisionings DISABLE TRIGGER "
                    "paper_fresh_opening_provisionings_append_only"
                )
                cursor.execute(
                    "UPDATE np.paper_fresh_opening_provisionings "
                    "SET candidate_payload = '{\"schema_version\":2}'"
                )
                cursor.execute(
                    "ALTER TABLE np.paper_fresh_opening_provisionings ENABLE ALWAYS "
                    "TRIGGER paper_fresh_opening_provisionings_append_only"
                )
            else:
                cursor.execute("CREATE TABLE np.unexpected_readiness_object (id int)")
        connection.commit()
    finally:
        connection.close()

    result = _assess(migrated_postgres_dsn, _context(encoded))

    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.OPENING_PROVENANCE_MISMATCH
    }


def test_head_six_account_migrates_but_is_blocked_without_opening_provenance(
    postgres_database_dsn,
):
    _apply(postgres_database_dsn, 6)
    encoded = _provision(postgres_database_dsn, with_provenance=False)
    _apply(postgres_database_dsn)

    result = _assess(postgres_database_dsn, _context(encoded))

    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.OPENING_PROVISIONING_ABSENT
    }
    assert result.account_version == 0
    assert tuple(item.version for item in result.applied_migrations) == tuple(
        range(1, 8)
    )


@pytest.mark.parametrize("migration_count", (0, 1, 2))
def test_absent_or_pending_migration_ledger_short_circuits_new_schema_queries(
    postgres_database_dsn,
    migration_count,
):
    if migration_count:
        migrations = _apply(postgres_database_dsn, migration_count)
    else:
        migrations = load_migrations()
    trace = _TracingFactory(postgres_database_dsn)

    result = _assess(
        postgres_database_dsn,
        _placeholder_context(),
        factory=trace,
    )

    expected_kind = (
        PaperAccountReadinessFindingKind.MIGRATION_LEDGER_ABSENT
        if migration_count == 0
        else PaperAccountReadinessFindingKind.MIGRATION_PENDING
    )
    assert _finding_kinds(result) == {expected_kind}
    assert tuple(item.version for item in result.expected_migrations) == tuple(
        item.version for item in migrations
    )
    assert tuple(item.version for item in result.applied_migrations) == tuple(
        range(1, migration_count + 1)
    )
    assert result.account_version is None
    assert result.legacy_watermarks == ()
    sql = "\n".join(command for command, _ in trace.connections[0].commands)
    assert "paper_account_streams" not in sql
    assert "position_streams" not in sql
    assert "FROM np.orders" not in sql
    assert "paper_account_batch_manifests" not in sql


@pytest.mark.parametrize(
    "mutation",
    (
        "UPDATE np.schema_migrations SET checksum = repeat('f', 64) WHERE version = 2",
        "UPDATE np.schema_migrations SET name = 'Bad Name' WHERE version = 2",
    ),
)
def test_drifted_or_malformed_migration_ledger_short_circuits_new_schema_queries(
    migrated_postgres_dsn,
    mutation,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(mutation)
        connection.commit()
    finally:
        connection.close()
    trace = _TracingFactory(migrated_postgres_dsn)

    result = _assess(
        migrated_postgres_dsn,
        _placeholder_context(),
        factory=trace,
    )

    assert _finding_kinds(result) == {PaperAccountReadinessFindingKind.MIGRATION_DRIFT}
    assert result.account_version is None
    assert result.legacy_watermarks == ()
    sql = "\n".join(command for command, _ in trace.connections[0].commands)
    assert "paper_account_streams" not in sql
    assert "position_streams" not in sql


def test_exact_migration_rows_exposed_by_a_view_are_drift_and_short_circuit(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE np.schema_migrations RENAME TO schema_migrations_backing"
            )
            cursor.execute("""
                CREATE VIEW np.schema_migrations AS
                SELECT version, name, checksum, applied_at
                FROM np.schema_migrations_backing
                """)
        connection.commit()
    finally:
        connection.close()
    trace = _TracingFactory(migrated_postgres_dsn)

    result = _assess(
        migrated_postgres_dsn,
        _placeholder_context(),
        factory=trace,
    )

    assert _finding_kinds(result) == {PaperAccountReadinessFindingKind.MIGRATION_DRIFT}
    assert result.account_version is None
    assert result.legacy_watermarks == ()
    sql = "\n".join(command for command, _ in trace.connections[0].commands)
    assert "paper_account_streams" not in sql
    assert "position_streams" not in sql
    assert "FROM np.orders" not in sql
    assert "paper_account_batch_manifests" not in sql


def test_business_table_replaced_by_exact_row_view_is_schema_drift_before_replay(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE np.orders RENAME TO orders_backing")
            cursor.execute("CREATE VIEW np.orders AS SELECT * FROM np.orders_backing")
        connection.commit()
    finally:
        connection.close()
    trace = _TracingFactory(migrated_postgres_dsn)

    result = _assess(
        migrated_postgres_dsn,
        _context(encoded),
        factory=trace,
    )

    assert _finding_kinds(result) == {PaperAccountReadinessFindingKind.MIGRATION_DRIFT}
    assert result.account_version is None
    assert result.legacy_watermarks == ()
    sql = "\n".join(command for command, _ in trace.connections[0].commands)
    assert "FROM np.orders" not in sql
    assert "paper_account_streams ORDER BY account_key" not in sql
    assert "position_streams ORDER BY position_key" not in sql


@pytest.mark.parametrize("mode", ("SHADOW", "PAUSED", "ACTIVE"))
def test_nonlegacy_runtime_control_mode_is_an_explicit_blocker(
    migrated_postgres_dsn,
    mode,
):
    encoded = _provision(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            if mode == "ACTIVE":
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
                    ) VALUES (1, 'activation-1', %s, %s, %s, 1, %s)
                    """,
                    (
                        encoded.execution_scope,
                        encoded.account_key,
                        encoded.owner_generation,
                        encoded.opening_payload_sha256,
                    ),
                )
            cursor.execute(
                """
                UPDATE np.paper_runtime_control
                SET mode = %s,
                    runtime_generation = %s
                WHERE control_key
                """,
                (mode, 1 if mode == "ACTIVE" else 0),
            )
        connection.commit()
    finally:
        connection.close()

    result = _assess(migrated_postgres_dsn, _context(encoded))

    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.RUNTIME_CONTROL_NOT_LEGACY
    }
    assert result.account_version == 0
    assert len(result.legacy_watermarks) == len(LEGACY_RELATIONS)


@pytest.mark.parametrize(
    "tamper",
    (
        "DELETE FROM np.paper_runtime_control",
        """
        CREATE OR REPLACE FUNCTION np.enforce_legacy_paper_runtime_fence()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        BEGIN
            RETURN NULL;
        END
        $function$
        """,
        """
        ALTER TABLE np.trades
        DISABLE TRIGGER legacy_paper_runtime_fence_trades
        """,
        """
        ALTER TABLE np.paper_runtime_control
        DROP CONSTRAINT paper_runtime_control_mode;
        ALTER TABLE np.paper_runtime_control
        ADD CONSTRAINT paper_runtime_control_mode CHECK (TRUE)
        """,
        """
        CREATE OR REPLACE FUNCTION np.reject_paper_runtime_generation_mutation()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        BEGIN
            RETURN NULL;
        END
        $function$
        """,
        """
        ALTER TABLE np.paper_runtime_generations
        DISABLE TRIGGER paper_runtime_generations_append_only
        """,
        """
        ALTER TABLE np.paper_runtime_generations
        DROP CONSTRAINT paper_runtime_generations_activated_at_finite
        """,
    ),
)
def test_runtime_control_generation_or_fence_tamper_is_early_schema_drift(
    migrated_postgres_dsn,
    tamper,
):
    encoded = _provision(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(tamper)
        connection.commit()
    finally:
        connection.close()
    trace = _TracingFactory(migrated_postgres_dsn)

    result = _assess(
        migrated_postgres_dsn,
        _context(encoded),
        factory=trace,
    )

    assert _finding_kinds(result) == {PaperAccountReadinessFindingKind.MIGRATION_DRIFT}
    assert result.account_version is None
    assert result.legacy_watermarks == ()
    sql = "\n".join(command for command, _ in trace.connections[0].commands)
    assert "FROM np.orders" not in sql
    assert "paper_account_streams ORDER BY account_key" not in sql
    assert "position_streams ORDER BY position_key" not in sql


def test_conditional_runtime_fence_trigger_is_early_schema_drift(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "DROP TRIGGER legacy_paper_runtime_fence_trades ON np.trades"
            )
            cursor.execute("""
                CREATE TRIGGER legacy_paper_runtime_fence_trades
                BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE
                ON np.trades
                FOR EACH STATEMENT
                WHEN (false)
                EXECUTE FUNCTION np.enforce_legacy_paper_runtime_fence()
                """)
            cursor.execute("""
                ALTER TABLE np.trades
                ENABLE ALWAYS TRIGGER legacy_paper_runtime_fence_trades
                """)
        connection.commit()
    finally:
        connection.close()
    trace = _TracingFactory(migrated_postgres_dsn)

    result = _assess(
        migrated_postgres_dsn,
        _context(encoded),
        factory=trace,
    )

    assert _finding_kinds(result) == {PaperAccountReadinessFindingKind.MIGRATION_DRIFT}
    assert result.account_version is None
    assert result.legacy_watermarks == ()
    sql = "\n".join(command for command, _ in trace.connections[0].commands)
    assert "FROM np.orders" not in sql
    assert "paper_account_streams ORDER BY account_key" not in sql
    assert "position_streams ORDER BY position_key" not in sql


def test_account_provenance_and_foreign_scope_are_global_blockers(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    _provision(
        migrated_postgres_dsn,
        account_key="foreign-account",
        scope="paper:foreign",
        generation=11,
    )

    result = _assess(
        migrated_postgres_dsn,
        _context(
            encoded,
            owner_generation=GENERATION + 1,
            opening_payload_sha256="f" * 64,
        ),
    )

    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.ACCOUNT_PROVENANCE_MISMATCH,
        PaperAccountReadinessFindingKind.OPENING_PROVENANCE_MISMATCH,
        PaperAccountReadinessFindingKind.UNEXPECTED_ACCOUNT,
    }
    unexpected = tuple(
        finding
        for finding in result.findings
        if finding.kind is PaperAccountReadinessFindingKind.UNEXPECTED_ACCOUNT
    )
    assert tuple(item.subject_id for item in unexpected) == ("foreign-account",)


def test_missing_expected_account_is_a_blocker_with_no_guessed_version(
    migrated_postgres_dsn,
):
    result = _assess(migrated_postgres_dsn, _placeholder_context())

    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert result.account_version is None
    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED,
        PaperAccountReadinessFindingKind.OPENING_PROVISIONING_ABSENT,
    }


def test_orphan_empty_position_stream_requires_reconciliation(migrated_postgres_dsn):
    encoded = _provision(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO np.position_streams (position_key, execution_scope) "
                "VALUES ('position-empty', %s)",
                (SCOPE,),
            )
        connection.commit()
    finally:
        connection.close()

    result = _assess(migrated_postgres_dsn, _context(encoded))

    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED
    }


def test_orphan_order_outside_any_stream_is_not_hidden_by_replay_inventory(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    instruction = _instruction("orphan-row")
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role = replica")
            _insert_encoded_order(cursor, instruction)
        connection.commit()
    finally:
        connection.close()

    result = _assess(migrated_postgres_dsn, _context(encoded))

    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
    }
    assert (
        result.disposition is PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
    )


def test_duplicate_client_order_across_streams_is_a_replay_failure(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    duplicate_client = "order-repeated-across-streams"
    instructions = (
        _instruction("duplicate-a", client_order_id=duplicate_client),
        _instruction("duplicate-b", client_order_id=duplicate_client),
    )
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE np.orders DROP CONSTRAINT orders_pkey")
            for instruction in instructions:
                cursor.execute(
                    "INSERT INTO np.position_streams "
                    "(position_key, execution_scope) VALUES (%s, %s)",
                    (instruction.position_key, SCOPE),
                )
                _insert_encoded_order(cursor, instruction)
        connection.commit()
    finally:
        connection.close()

    result = _assess(migrated_postgres_dsn, _context(encoded))

    assert any(
        finding.kind is PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED
        and finding.subject_kind == "durable_relation"
        and finding.subject_id == "np.orders"
        for finding in result.findings
    )
    assert (
        result.disposition is PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
    )


@pytest.mark.parametrize("scope", (SCOPE, "paper:foreign"))
def test_pending_unaccounted_order_is_reported_once_per_identity_and_global_scope(
    migrated_postgres_dsn,
    scope,
):
    encoded = _provision(migrated_postgres_dsn)
    instruction = _instruction("pending")
    _reserve(migrated_postgres_dsn, instruction, scope=scope)

    result = _assess(migrated_postgres_dsn, _context(encoded))

    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.UNRESOLVED_SUBMISSION,
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
    }
    assert {finding.subject_id for finding in result.findings} == {
        instruction.order_intent.client_order_id
    }


def test_terminal_failed_order_without_manifest_is_unaccounted(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    instruction = _instruction("failed")
    journal = _reserve(migrated_postgres_dsn, instruction)
    journal.append_event(
        execution_scope=SCOPE,
        position_key=instruction.position_key,
        event_id="failed-event",
        event=SubmissionFailed(
            instruction.order_intent.client_order_id,
            SubmissionStatus.NOT_SENT,
            RetrySafety.SAFE,
            "paper planner refused submission",
            NOW + timedelta(seconds=1),
        ),
    )

    result = _assess(migrated_postgres_dsn, _context(encoded))

    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER
    }


def test_terminal_migration_0002_fill_is_open_and_unaccounted(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    instruction = _instruction("terminal-fill")
    journal = _reserve(migrated_postgres_dsn, instruction)
    client_order_id = instruction.order_intent.client_order_id
    journal.append_event(
        execution_scope=SCOPE,
        position_key=instruction.position_key,
        event_id="ack-event",
        event=SubmissionAcknowledged(
            client_order_id,
            "venue-terminal-fill",
            NOW + timedelta(seconds=1),
        ),
    )
    journal.append_event(
        execution_scope=SCOPE,
        position_key=instruction.position_key,
        event_id="fill-event",
        event=ConfirmedFill(
            client_order_id=client_order_id,
            venue_order_id="venue-terminal-fill",
            trade_id="trade-terminal-fill",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.00"),
            price=Decimal("10.00"),
            fee_amount=Decimal("0.00"),
            executed_at=NOW + timedelta(seconds=2),
        ),
    )

    result = _assess(migrated_postgres_dsn, _context(encoded))

    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.DURABLE_OPEN_POSITION,
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
    }


@pytest.mark.parametrize("corruption", ("account", "position"))
def test_strict_replay_corruption_becomes_a_stable_finding(
    migrated_postgres_dsn,
    corruption,
):
    encoded = _provision(migrated_postgres_dsn)
    instruction = None
    if corruption == "position":
        instruction = _instruction("corrupt")
        _reserve(migrated_postgres_dsn, instruction)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            if corruption == "account":
                cursor.execute(
                    "UPDATE np.paper_account_streams "
                    "SET account_version = account_version + 1 "
                    "WHERE account_key = %s",
                    (ACCOUNT_KEY,),
                )
            else:
                cursor.execute(
                    "UPDATE np.orders SET instruction_payload_sha256 = repeat('f', 64) "
                    "WHERE client_order_id = %s",
                    (instruction.order_intent.client_order_id,),
                )
        connection.commit()
    finally:
        connection.close()

    result = _assess(migrated_postgres_dsn, _context(encoded))

    expected = (
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED
        if corruption == "account"
        else PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED
    )
    assert expected in _finding_kinds(result)
    assert (
        result.disposition is PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
    )


def test_legacy_open_position_is_visible_in_watermark_and_blocks(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO np.open_positions (
                    symbol, side, entry_price, quantity, leverage
                ) VALUES ('BTCUSDT', 'BUY', 10, 1, 2)
                RETURNING id
                """)
            position_id = cursor.fetchone()[0]
        connection.commit()
    finally:
        connection.close()

    result = _assess(migrated_postgres_dsn, _context(encoded))

    watermark = next(
        item
        for item in result.legacy_watermarks
        if item.relation == "np.open_positions"
    )
    assert (watermark.row_count, watermark.max_id) == (1, position_id)
    assert _finding_kinds(result) == {
        PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION
    }


def test_repeatable_read_snapshot_is_stale_after_concurrent_legacy_commit(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    committed = False

    def insert_after_snapshot(statement):
        nonlocal committed
        if committed or statement != "SELECT to_regclass(%s)":
            return
        writer = _connect(migrated_postgres_dsn)
        try:
            with writer.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO np.open_positions (
                        symbol, side, entry_price, quantity, leverage
                    ) VALUES ('BTCUSDT', 'BUY', 10, 1, 2)
                    """)
            writer.commit()
            committed = True
        finally:
            writer.close()

    trace = _TracingFactory(
        migrated_postgres_dsn,
        after_execute=insert_after_snapshot,
    )

    result = _assess(
        migrated_postgres_dsn,
        _context(encoded),
        factory=trace,
    )

    assert committed is True
    watermark = next(
        item
        for item in result.legacy_watermarks
        if item.relation == "np.open_positions"
    )
    assert (watermark.row_count, watermark.max_id) == (0, None)
    assert result.snapshot_authoritative is False
    verifier = _connect(migrated_postgres_dsn)
    try:
        with verifier.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM np.open_positions")
            assert cursor.fetchone() == (1,)
    finally:
        verifier.close()


def test_assessment_trace_is_one_read_only_cursor_with_no_write_or_lock(
    migrated_postgres_dsn,
):
    encoded = _provision(migrated_postgres_dsn)
    trace = _TracingFactory(migrated_postgres_dsn)

    result = _assess(
        migrated_postgres_dsn,
        _context(encoded),
        factory=trace,
    )

    assert result.disposition is PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
    assert len(trace.connections) == 1
    connection = trace.connections[0]
    assert connection.cursor_calls == 1
    assert connection.commit_calls == 0
    assert connection.rollback_calls == 1
    statements = tuple(command for command, _ in connection.commands)
    assert statements[0] == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    )
    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|MERGE|TRUNCATE|ALTER|DROP|CREATE|LOCK)\b"
    )
    assert not any(forbidden.search(statement.upper()) for statement in statements)
    assert not any("FOR UPDATE" in statement.upper() for statement in statements)
