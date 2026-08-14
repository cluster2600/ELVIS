"""PostgreSQL 15 security and snapshot proofs for activation capabilities."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import psycopg2
import pytest
from psycopg2 import sql

from tests.postgres.test_paper_account_readiness_postgres import (
    _readiness_opening_role,
    _seed_fresh_opening_provenance,
)
from tests.postgres.test_paper_runtime_activation_postgres import (
    ACCOUNT_KEY,
    OWNER_GENERATION,
    SCOPE,
    _connect,
    _context,
    _instruction,
    _provision,
    _runtime_snapshot,
)
from trading.application.paper_account_readiness import (
    PaperAccountReadinessDisposition,
    PaperAccountReadinessFindingKind,
)
from trading.application.paper_runtime_activation import (
    PaperRuntimeActivationBlocked,
)
from trading.persistence.migration_runner import apply_migrations, load_migrations
from trading.persistence.order_position_journal import PostgresOrderPositionJournal
from trading.persistence.paper_account_journal_codec import (
    encode_paper_account_opening,
)
from trading.persistence.paper_account_readiness import PostgresPaperAccountReadiness
from trading.persistence.paper_runtime_activation import PostgresPaperRuntimeActivation

FENCE_FUNCTION = "np.acquire_paper_runtime_activation_fence()"
MUTATION_FUNCTION = (
    "np.activate_paper_runtime_generation(text,bigint,bigint,text,text,text,"
    "bigint,text)"
)
AUTHORITY_RELATIONS = (
    "account_balances",
    "liquidations",
    "margin_history",
    "model_predictions",
    "open_positions",
    "order_events",
    "orders",
    "paper_account_balances",
    "paper_account_batch_manifests",
    "paper_account_postings",
    "paper_account_settlements",
    "paper_account_streams",
    "paper_margin_reservations",
    "paper_runtime_control",
    "paper_runtime_generations",
    "position_streams",
    "schema_migrations",
    "trades",
    "trading_session_resets",
)


@pytest.fixture(autouse=True)
def _cleanup_capability_opening_role(postgres_database_dsn):
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


def _seed_current_provenance(dsn, opening):
    encoded = encode_paper_account_opening(
        opening.execution_scope,
        opening.owner_generation,
        opening.account,
    )
    _seed_fresh_opening_provenance(dsn, encoded)
    return encoded


def _role_connection(dsn, role_name):
    connection = _connect(dsn)
    with connection.cursor() as cursor:
        cursor.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role_name)))
    connection.commit()
    return connection


def _grant_activation_execute(connection, role_name):
    with connection.cursor() as cursor:
        role = sql.Identifier(role_name)
        cursor.execute(
            sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(
                sql.SQL(FENCE_FUNCTION),
                role,
            )
        )
        cursor.execute(
            sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(
                sql.SQL(MUTATION_FUNCTION),
                role,
            )
        )
    connection.commit()


def _direct_mutation(cursor, values):
    cursor.execute(
        """
        SELECT mode, runtime_generation
        FROM np.activate_paper_runtime_generation(
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        """,
        values,
    )
    return cursor.fetchall()


def test_version_five_upgrade_adds_exact_capabilities_without_business_rewrite(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        assert apply_migrations(connection, migrations[:5]) == (1, 2, 3, 4, 5)
    finally:
        connection.close()

    _provision(postgres_database_dsn)
    PostgresOrderPositionJournal(
        lambda: _connect(postgres_database_dsn)
    ).reserve_instruction(
        execution_scope=SCOPE,
        instruction=_instruction("upgrade-six"),
    )
    connection = _connect(postgres_database_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO np.trades (id, symbol) VALUES (940001, 'BTCUSDT')"
            )
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT xmin::text, ctid::text, account_key, execution_scope, "
                "owner_generation, opening_payload_sha256::text "
                "FROM np.paper_account_streams"
            )
            account_before = cursor.fetchall()
            cursor.execute(
                "SELECT xmin::text, ctid::text, position_key, execution_scope "
                "FROM np.position_streams"
            )
            position_before = cursor.fetchall()
            cursor.execute("SELECT xmin::text, ctid::text, id, symbol FROM np.trades")
            legacy_before = cursor.fetchall()
        connection.rollback()

        assert apply_migrations(connection, migrations) == (6, 7)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT xmin::text, ctid::text, account_key, execution_scope, "
                "owner_generation, opening_payload_sha256::text "
                "FROM np.paper_account_streams"
            )
            assert cursor.fetchall() == account_before
            cursor.execute(
                "SELECT xmin::text, ctid::text, position_key, execution_scope "
                "FROM np.position_streams"
            )
            assert cursor.fetchall() == position_before
            cursor.execute("SELECT xmin::text, ctid::text, id, symbol FROM np.trades")
            assert cursor.fetchall() == legacy_before
            cursor.execute(
                "SELECT version, name, checksum FROM np.schema_migrations "
                "WHERE version = 6"
            )
            assert cursor.fetchone() == (
                6,
                "paper_runtime_activation_capabilities",
                "e01c02d1e64b8b136e80dcf2fe365dc85df72d4e1cfa58a8a13b14e4b3f6449d",
            )
    finally:
        connection.close()


def test_capabilities_are_exact_security_definers_with_public_execute_revoked(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    routine_row.proname,
                    pg_get_function_identity_arguments(routine_row.oid),
                    pg_get_function_result(routine_row.oid),
                    routine_row.prosecdef,
                    routine_row.provolatile,
                    routine_row.proisstrict,
                    routine_row.proretset,
                    language_row.lanname,
                    routine_row.proconfig,
                    routine_row.proowner,
                    NOT EXISTS (
                        SELECT 1
                        FROM aclexplode(
                            COALESCE(
                                routine_row.proacl,
                                acldefault('f', routine_row.proowner)
                            )
                        ) AS function_acl
                        WHERE function_acl.grantee = 0
                          AND function_acl.privilege_type = 'EXECUTE'
                    )
                FROM pg_proc routine_row
                JOIN pg_namespace namespace_row
                  ON namespace_row.oid = routine_row.pronamespace
                JOIN pg_language language_row
                  ON language_row.oid = routine_row.prolang
                WHERE namespace_row.nspname = 'np'
                  AND routine_row.proname IN (
                      'acquire_paper_runtime_activation_fence',
                      'activate_paper_runtime_generation'
                  )
                ORDER BY routine_row.proname
                """)
            rows = cursor.fetchall()

        assert tuple(row[:9] for row in rows) == (
            (
                "acquire_paper_runtime_activation_fence",
                "",
                "void",
                True,
                "v",
                False,
                False,
                "plpgsql",
                ["search_path=pg_catalog, pg_temp"],
            ),
            (
                "activate_paper_runtime_generation",
                "expected_mode text, expected_generation bigint, target_generation "
                "bigint, requested_activation_id text, requested_execution_scope "
                "text, requested_account_key text, requested_owner_generation "
                "bigint, requested_opening_payload_sha256 text",
                "TABLE(mode text, runtime_generation bigint)",
                True,
                "v",
                False,
                True,
                "plpgsql",
                ["search_path=pg_catalog, pg_temp"],
            ),
        )
        assert rows[0][9] == rows[1][9]
        assert rows[0][10:] == (True,)
        assert rows[1][10:] == (True,)
    finally:
        connection.close()


def test_third_party_execute_grant_exposes_capability_but_catalog_fails_closed(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    administrator = _connect(migrated_postgres_dsn)
    role_name = None
    try:
        with administrator.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            role_name = f"{cursor.fetchone()[0]}_activation"
            role = sql.Identifier(role_name)
            cursor.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(role))
            cursor.execute(sql.SQL("GRANT USAGE ON SCHEMA np TO {}").format(role))
            cursor.execute(
                sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA np TO {}").format(role)
            )
        administrator.commit()

        denied = _role_connection(migrated_postgres_dsn, role_name)
        try:
            with denied.cursor() as cursor:
                with pytest.raises(psycopg2.Error) as raised:
                    cursor.execute("SELECT np.acquire_paper_runtime_activation_fence()")
            assert raised.value.pgcode == "42501"
            denied.rollback()
        finally:
            denied.close()

        _grant_activation_execute(administrator, role_name)
        forbidden = (
            "LOCK TABLE ONLY np.trades IN SHARE MODE NOWAIT",
            "SELECT * FROM np.paper_runtime_control FOR UPDATE NOWAIT",
            "INSERT INTO np.paper_runtime_generations "
            "(runtime_generation, activation_id, execution_scope, account_key, "
            "owner_generation, opening_version, opening_payload_sha256) VALUES "
            "(1, 'direct-epoch', 'paper:test', 'paper-main', 7, 1, repeat('a', 64))",
            "UPDATE np.paper_runtime_control SET mode = 'ACTIVE'",
        )
        for statement in forbidden:
            attacker = _role_connection(migrated_postgres_dsn, role_name)
            try:
                with attacker.cursor() as cursor:
                    with pytest.raises(psycopg2.Error) as raised:
                        cursor.execute(statement)
                assert raised.value.pgcode == "42501"
                attacker.rollback()
            finally:
                attacker.close()

        direct = _role_connection(migrated_postgres_dsn, role_name)
        try:
            with direct.cursor() as cursor:
                cursor.execute("SELECT np.acquire_paper_runtime_activation_fence()")
                assert cursor.fetchone() is not None
                with pytest.raises(psycopg2.Error) as invalid:
                    _direct_mutation(
                        cursor,
                        (
                            "LEGACY",
                            0,
                            2,
                            "third-party-call",
                            SCOPE,
                            ACCOUNT_KEY,
                            OWNER_GENERATION,
                            opening.current.opening_payload_sha256,
                        ),
                    )
            assert invalid.value.pgcode == "22023"
            direct.rollback()
        finally:
            direct.close()

        result = PostgresPaperRuntimeActivation(
            lambda: _role_connection(migrated_postgres_dsn, role_name)
        ).activate(_context(opening))
        assert type(result) is PaperRuntimeActivationBlocked
        assert {finding.kind for finding in result.assessment.findings} == {
            PaperAccountReadinessFindingKind.MIGRATION_DRIFT
        }
        readiness = PostgresPaperAccountReadiness(
            lambda: _connect(migrated_postgres_dsn)
        ).assess(_context(opening).readiness)
        assert {finding.kind for finding in readiness.findings} == {
            PaperAccountReadinessFindingKind.MIGRATION_DRIFT
        }
        assert _runtime_snapshot(migrated_postgres_dsn) == ((("LEGACY", 0),), ())
    finally:
        administrator.rollback()
        if role_name is not None:
            with administrator.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role_name))
                )
                cursor.execute(
                    sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name))
                )
            administrator.commit()
        administrator.close()


def test_fence_holds_exact_share_set_and_all_ordered_row_drains(
    migrated_postgres_dsn,
):
    _provision(migrated_postgres_dsn)
    instruction = _instruction("lock-proof")
    PostgresOrderPositionJournal(
        lambda: _connect(migrated_postgres_dsn)
    ).reserve_instruction(
        execution_scope=SCOPE,
        instruction=instruction,
    )
    holder = _connect(migrated_postgres_dsn)
    try:
        with holder.cursor() as cursor:
            cursor.execute("SELECT np.acquire_paper_runtime_activation_fence()")
            cursor.execute("""
                SELECT table_row.relname, lock_row.mode
                FROM pg_locks lock_row
                JOIN pg_class table_row ON table_row.oid = lock_row.relation
                JOIN pg_namespace namespace_row
                  ON namespace_row.oid = table_row.relnamespace
                WHERE lock_row.pid = pg_backend_pid()
                  AND lock_row.granted
                  AND namespace_row.nspname = 'np'
                  AND table_row.relkind = 'r'
                  AND lock_row.mode IN ('ShareLock', 'RowShareLock')
                ORDER BY lock_row.mode, table_row.relname
                """)
            locks = cursor.fetchall()
            cursor.execute("""
                SELECT prosrc
                FROM pg_proc routine_row
                JOIN pg_namespace namespace_row
                  ON namespace_row.oid = routine_row.pronamespace
                WHERE namespace_row.nspname = 'np'
                  AND routine_row.proname =
                      'acquire_paper_runtime_activation_fence'
                """)
            source = cursor.fetchone()[0]

        assert tuple(name for name, mode in locks if mode == "ShareLock") == (
            AUTHORITY_RELATIONS
        )
        assert tuple(name for name, mode in locks if mode == "RowShareLock") == (
            "paper_account_streams",
            "paper_runtime_control",
            "position_streams",
        )
        control_index = source.index("FROM np.paper_runtime_control")
        account_index = source.index("FROM np.paper_account_streams")
        position_index = source.index("FROM np.position_streams")
        assert control_index < account_index < position_index
        assert "ORDER BY account_key\n    FOR UPDATE NOWAIT" in source
        assert "ORDER BY position_key\n    FOR UPDATE NOWAIT" in source

        contenders = (
            "SELECT * FROM np.paper_runtime_control "
            "WHERE control_key FOR UPDATE NOWAIT",
            "SELECT * FROM np.paper_account_streams "
            "WHERE account_key = 'paper-main' FOR UPDATE NOWAIT",
            "SELECT * FROM np.position_streams "
            "WHERE position_key = 'position-lock-proof' FOR UPDATE NOWAIT",
        )
        for statement in contenders:
            contender = _connect(migrated_postgres_dsn)
            try:
                with contender.cursor() as cursor:
                    with pytest.raises(psycopg2.Error) as raised:
                        cursor.execute(statement)
                assert raised.value.pgcode == "55P03"
                contender.rollback()
            finally:
                contender.close()
    finally:
        holder.rollback()
        holder.close()


class _PreFenceSnapshotCursor:
    def __init__(self, cursor, *, snapshot_taken, writer_committed):
        self._cursor = cursor
        self._snapshot_taken = snapshot_taken
        self._writer_committed = writer_committed

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._cursor.__exit__(exc_type, exc, traceback)

    def execute(self, statement, parameters=None):
        result = self._cursor.execute(statement, parameters)
        if " ".join(str(statement).split()).upper() == (
            "SET LOCAL LOCK_TIMEOUT = '1S'"
        ):
            self._cursor.execute("SELECT COUNT(*) FROM np.open_positions")
            assert self._cursor.fetchone() == (0,)
            self._snapshot_taken.set()
            assert self._writer_committed.wait(timeout=10)
        return result


class _PreFenceSnapshotConnection:
    def __init__(self, connection, *, snapshot_taken, writer_committed):
        self._connection = connection
        self._snapshot_taken = snapshot_taken
        self._writer_committed = writer_committed

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def cursor(self):
        return _PreFenceSnapshotCursor(
            self._connection.cursor(),
            snapshot_taken=self._snapshot_taken,
            writer_committed=self._writer_committed,
        )


def test_read_committed_discards_prebody_snapshot_before_successful_fence(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    snapshot_taken = Event()
    writer_committed = Event()

    def write_open_position():
        assert snapshot_taken.wait(timeout=10)
        writer = _connect(migrated_postgres_dsn)
        try:
            with writer.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO np.open_positions "
                    "(symbol, side, entry_price, quantity, leverage) "
                    "VALUES ('BTCUSDT', 'BUY', 10, 1, 2)"
                )
            writer.commit()
            writer_committed.set()
        finally:
            writer.close()

    def factory():
        return _PreFenceSnapshotConnection(
            _connect(migrated_postgres_dsn),
            snapshot_taken=snapshot_taken,
            writer_committed=writer_committed,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(write_open_position)
        result = PostgresPaperRuntimeActivation(factory).activate(_context(opening))
        future.result(timeout=10)

    assert type(result) is PaperRuntimeActivationBlocked
    assert result.assessment.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION in {
        finding.kind for finding in result.assessment.findings
    }
    assert _runtime_snapshot(migrated_postgres_dsn) == ((("LEGACY", 0),), ())


def test_activation_mutation_rejects_missing_opening_provenance_without_delta(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    before = _runtime_snapshot(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            with pytest.raises(psycopg2.Error) as raised:
                _direct_mutation(
                    cursor,
                    (
                        "LEGACY",
                        0,
                        1,
                        "missing-provenance",
                        SCOPE,
                        ACCOUNT_KEY,
                        OWNER_GENERATION,
                        opening.current.opening_payload_sha256,
                    ),
                )
        assert raised.value.pgcode == "55000"
        connection.rollback()
    finally:
        connection.close()

    assert _runtime_snapshot(migrated_postgres_dsn) == before


def test_activation_mutation_accepts_only_the_exact_current_opening_provenance(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    encoded = _seed_current_provenance(migrated_postgres_dsn, opening)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT np.paper_fresh_opening_target_is_current()")
            assert cursor.fetchone() == (True,)
            assert _direct_mutation(
                cursor,
                (
                    "LEGACY",
                    0,
                    1,
                    "exact-provenance",
                    encoded.execution_scope,
                    encoded.account_key,
                    encoded.owner_generation,
                    encoded.opening_payload_sha256,
                ),
            ) == [("ACTIVE", 1)]
        connection.commit()
    finally:
        connection.close()

    control, generations = _runtime_snapshot(migrated_postgres_dsn)
    assert control == (("ACTIVE", 1),)
    assert tuple(row[:2] for row in generations) == ((1, "exact-provenance"),)


@pytest.mark.parametrize("drift", ("schema_marker", "opening_anchor", "catalog"))
def test_activation_mutation_rejects_current_physical_target_drift(
    migrated_postgres_dsn,
    drift,
):
    opening = _provision(migrated_postgres_dsn)
    encoded = _seed_current_provenance(migrated_postgres_dsn, opening)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            if drift == "schema_marker":
                cursor.execute("COMMENT ON SCHEMA np IS 'drifted'")
            elif drift == "opening_anchor":
                cursor.execute("SELECT current_database()")
                opening_role = _readiness_opening_role(cursor.fetchone()[0])
                cursor.execute(
                    sql.SQL("COMMENT ON ROLE {} IS 'drifted'").format(
                        sql.Identifier(opening_role)
                    )
                )
            else:
                cursor.execute("CREATE TABLE np.unexpected_activation_object (id int)")
        connection.commit()

        before = _runtime_snapshot(migrated_postgres_dsn)
        with connection.cursor() as cursor:
            cursor.execute("SELECT np.paper_fresh_opening_target_is_current()")
            assert cursor.fetchone() == (False,)
            with pytest.raises(psycopg2.Error) as raised:
                _direct_mutation(
                    cursor,
                    (
                        "LEGACY",
                        0,
                        1,
                        f"physical-drift-{drift}",
                        encoded.execution_scope,
                        encoded.account_key,
                        encoded.owner_generation,
                        encoded.opening_payload_sha256,
                    ),
                )
        assert raised.value.pgcode == "55000"
        connection.rollback()
    finally:
        connection.close()

    assert _runtime_snapshot(migrated_postgres_dsn) == before


@pytest.mark.parametrize(
    "invalid",
    (
        ("LEGACY", 0, 2, "bad-target", SCOPE, ACCOUNT_KEY, OWNER_GENERATION, "a" * 64),
        ("LEGACY", 0, 1, "", SCOPE, ACCOUNT_KEY, OWNER_GENERATION, "a" * 64),
        ("LEGACY", 0, 1, "bad-sha", SCOPE, ACCOUNT_KEY, OWNER_GENERATION, "A" * 64),
        (
            "PAUSED",
            (1 << 63) - 1,
            (1 << 63) - 1,
            "overflow",
            SCOPE,
            ACCOUNT_KEY,
            OWNER_GENERATION,
            "a" * 64,
        ),
    ),
)
def test_mutation_rejects_invalid_arguments_with_22023_and_no_orphan(
    migrated_postgres_dsn,
    invalid,
):
    opening = _provision(migrated_postgres_dsn)
    values = (*invalid[:-1], opening.current.opening_payload_sha256)
    if invalid[3] == "bad-sha":
        values = invalid
    before = _runtime_snapshot(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            with pytest.raises(psycopg2.Error) as raised:
                _direct_mutation(cursor, values)
        assert raised.value.pgcode == "22023"
        connection.rollback()
    finally:
        connection.close()
    assert _runtime_snapshot(migrated_postgres_dsn) == before


def test_mutation_pt001_cas_failure_rolls_back_inserted_epoch(
    migrated_postgres_dsn,
):
    opening = _provision(migrated_postgres_dsn)
    _seed_current_provenance(migrated_postgres_dsn, opening)
    before = _runtime_snapshot(migrated_postgres_dsn)
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            with pytest.raises(psycopg2.Error) as raised:
                _direct_mutation(
                    cursor,
                    (
                        "PAUSED",
                        1,
                        2,
                        "cas-failure",
                        SCOPE,
                        ACCOUNT_KEY,
                        OWNER_GENERATION,
                        opening.current.opening_payload_sha256,
                    ),
                )
        assert raised.value.pgcode == "PT001"
        connection.rollback()
    finally:
        connection.close()
    assert _runtime_snapshot(migrated_postgres_dsn) == before


@pytest.mark.parametrize("tamper", ("source", "acl", "owner"))
def test_function_source_acl_or_owner_tamper_is_early_readiness_drift(
    migrated_postgres_dsn,
    tamper,
):
    opening = _provision(migrated_postgres_dsn)
    administrator = _connect(migrated_postgres_dsn)
    role_name = None
    original_owner = None
    try:
        with administrator.cursor() as cursor:
            if tamper == "source":
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION
                        np.acquire_paper_runtime_activation_fence()
                    RETURNS VOID
                    LANGUAGE plpgsql
                    SECURITY DEFINER
                    SET search_path = pg_catalog
                    AS $function$
                    BEGIN
                        RETURN;
                    END
                    $function$
                    """)
            elif tamper == "acl":
                cursor.execute(
                    "GRANT EXECUTE ON FUNCTION "
                    "np.acquire_paper_runtime_activation_fence() TO PUBLIC"
                )
            else:
                cursor.execute("SELECT current_user, current_database()")
                original_owner, database_name = cursor.fetchone()
                role_name = f"{database_name}_foreign_function_owner"
                cursor.execute(
                    sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role_name))
                )
                cursor.execute(
                    sql.SQL("ALTER FUNCTION {} OWNER TO {}").format(
                        sql.SQL(FENCE_FUNCTION),
                        sql.Identifier(role_name),
                    )
                )
        administrator.commit()

        result = PostgresPaperAccountReadiness(
            lambda: _connect(migrated_postgres_dsn)
        ).assess(_context(opening).readiness)
        assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
        assert {finding.kind for finding in result.findings} == {
            PaperAccountReadinessFindingKind.MIGRATION_DRIFT
        }
        assert result.account_version is None
        assert result.legacy_watermarks == ()
    finally:
        administrator.rollback()
        if role_name is not None:
            with administrator.cursor() as cursor:
                cursor.execute(
                    sql.SQL("ALTER FUNCTION {} OWNER TO {}").format(
                        sql.SQL(FENCE_FUNCTION),
                        sql.Identifier(original_owner),
                    )
                )
                cursor.execute(
                    sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role_name))
                )
                cursor.execute(
                    sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name))
                )
            administrator.commit()
        administrator.close()
