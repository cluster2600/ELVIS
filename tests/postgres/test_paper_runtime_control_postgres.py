"""PostgreSQL 15 proofs for the dormant legacy-paper runtime fence."""

from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from threading import Event
from time import monotonic

import psycopg2
import pytest
from psycopg2 import sql

LEGACY_RELATIONS = (
    "account_balances",
    "liquidations",
    "margin_history",
    "model_predictions",
    "open_positions",
    "trades",
    "trading_session_resets",
)
FENCED_MODES = ("PAUSED", "ACTIVE")
WRITE_OPERATIONS = ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "COPY", "UPSERT")
FENCE_SQLSTATE = "55000"


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _set_mode(connection, mode):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE np.paper_runtime_control
            SET mode = %s,
                runtime_generation = runtime_generation + 1
            WHERE control_key
            """,
            (mode,),
        )
        assert cursor.rowcount == 1


def _write(cursor, relation, operation, *, row_id=900001):
    qualified = f"np.{relation}"
    if operation == "INSERT":
        if relation == "account_balances":
            cursor.execute(
                f"INSERT INTO {qualified} (id, asset, balance) "
                "VALUES (%s, 'USDT', 1)",
                (row_id,),
            )
        else:
            cursor.execute(f"INSERT INTO {qualified} (id) VALUES (%s)", (row_id,))
    elif operation == "UPDATE":
        cursor.execute(f"UPDATE {qualified} SET id = id")
    elif operation == "DELETE":
        cursor.execute(f"DELETE FROM {qualified}")
    elif operation == "TRUNCATE":
        cursor.execute(f"TRUNCATE TABLE {qualified}")
    elif operation == "COPY":
        if relation == "account_balances":
            cursor.copy_expert(
                f"COPY {qualified} (id, asset, balance) FROM STDIN",
                StringIO(f"{row_id}\tUSDT\t1\n"),
            )
        else:
            cursor.copy_expert(
                f"COPY {qualified} (id) FROM STDIN",
                StringIO(f"{row_id}\n"),
            )
    elif operation == "UPSERT":
        if relation == "account_balances":
            cursor.execute(
                f"""
                INSERT INTO {qualified} (id, asset, balance)
                VALUES (%s, 'USDT', 1)
                ON CONFLICT (id) DO UPDATE SET id = EXCLUDED.id
                """,
                (row_id,),
            )
        else:
            cursor.execute(
                f"""
                INSERT INTO {qualified} (id) VALUES (%s)
                ON CONFLICT (id) DO UPDATE SET id = EXCLUDED.id
                """,
                (row_id,),
            )
    else:  # pragma: no cover - protects the parametrized test helper
        raise AssertionError(f"unsupported write operation: {operation}")


def _assert_fenced(exc, mode):
    assert exc.pgcode == FENCE_SQLSTATE
    assert exc.diag.message_primary == f"legacy paper writes are fenced in {mode} mode"


@pytest.mark.parametrize("relation", LEGACY_RELATIONS)
@pytest.mark.parametrize("operation", WRITE_OPERATIONS)
def test_legacy_mode_allows_each_legacy_relation_write_shape(
    migrated_postgres_dsn,
    relation,
    operation,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            _write(cursor, relation, operation)
        connection.rollback()
    finally:
        connection.close()


def test_shadow_mode_keeps_all_legacy_relations_writable(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        _set_mode(connection, "SHADOW")
        connection.commit()
        with connection.cursor() as cursor:
            for offset, relation in enumerate(LEGACY_RELATIONS):
                _write(cursor, relation, "INSERT", row_id=901000 + offset)
        connection.rollback()
    finally:
        connection.close()


@pytest.mark.parametrize("mode", FENCED_MODES)
@pytest.mark.parametrize("relation", LEGACY_RELATIONS)
@pytest.mark.parametrize("operation", WRITE_OPERATIONS)
def test_fenced_modes_reject_each_legacy_relation_write_shape(
    migrated_postgres_dsn,
    mode,
    relation,
    operation,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        _set_mode(connection, mode)
        connection.commit()

        with pytest.raises(psycopg2.Error) as raised:
            with connection.cursor() as cursor:
                _write(cursor, relation, operation)

        _assert_fenced(raised.value, mode)
        connection.rollback()
    finally:
        connection.close()


@pytest.mark.parametrize("relation", LEGACY_RELATIONS)
def test_missing_runtime_control_singleton_fails_closed(
    migrated_postgres_dsn,
    relation,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM np.paper_runtime_control")
        connection.commit()

        with pytest.raises(psycopg2.Error) as raised:
            with connection.cursor() as cursor:
                _write(cursor, relation, "INSERT")

        assert raised.value.pgcode == FENCE_SQLSTATE
        assert (
            raised.value.diag.message_primary == "paper runtime control is unavailable"
        )
        connection.rollback()
    finally:
        connection.close()


def test_active_fence_survives_replica_session_role(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        _set_mode(connection, "ACTIVE")
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role = replica")
            with pytest.raises(psycopg2.Error) as raised:
                _write(cursor, "trades", "INSERT")

        _assert_fenced(raised.value, "ACTIVE")
        connection.rollback()
    finally:
        connection.close()


def test_security_definer_fence_serves_a_least_privilege_legacy_writer(
    migrated_postgres_dsn,
):
    administrator = _connect(migrated_postgres_dsn)
    role_name = None
    try:
        with administrator.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            role_name = f"{cursor.fetchone()[0]}_legacy_writer"
            role_identifier = sql.Identifier(role_name)
            cursor.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(role_identifier))
            cursor.execute(
                sql.SQL("GRANT USAGE ON SCHEMA np TO {}").format(role_identifier)
            )
            cursor.execute(
                sql.SQL(
                    "GRANT INSERT, UPDATE, DELETE, TRUNCATE ON TABLE {} TO {}"
                ).format(
                    sql.SQL(", ").join(
                        sql.Identifier("np", relation) for relation in LEGACY_RELATIONS
                    ),
                    role_identifier,
                )
            )
            cursor.execute(
                """
                SELECT
                    has_schema_privilege(%s, 'np', 'USAGE'),
                    has_schema_privilege(%s, 'np', 'CREATE'),
                    has_table_privilege(
                        %s,
                        'np.paper_runtime_control',
                        'SELECT'
                    ),
                    has_table_privilege(
                        %s,
                        'np.paper_runtime_control',
                        'UPDATE'
                    ),
                    NOT EXISTS (
                        SELECT 1
                        FROM pg_proc routine_row
                        JOIN pg_namespace namespace_row
                          ON namespace_row.oid = routine_row.pronamespace
                        CROSS JOIN LATERAL aclexplode(
                            COALESCE(
                                routine_row.proacl,
                                acldefault('f', routine_row.proowner)
                            )
                        ) acl_row
                        JOIN pg_roles role_row
                          ON role_row.oid = acl_row.grantee
                        WHERE namespace_row.nspname = 'np'
                          AND routine_row.proname =
                              'enforce_legacy_paper_runtime_fence'
                          AND role_row.rolname = %s
                    )
                """,
                (role_name, role_name, role_name, role_name, role_name),
            )
            assert cursor.fetchone() == (True, False, False, False, True)
        administrator.commit()

        legacy_writer = _connect(migrated_postgres_dsn)
        try:
            with legacy_writer.cursor() as cursor:
                cursor.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role_name)))
                _write(cursor, "trades", "INSERT", row_id=902001)
            legacy_writer.commit()
        finally:
            legacy_writer.close()

        _set_mode(administrator, "ACTIVE")
        administrator.commit()

        active_writer = _connect(migrated_postgres_dsn)
        try:
            with active_writer.cursor() as cursor:
                cursor.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role_name)))
                with pytest.raises(psycopg2.Error) as raised:
                    _write(cursor, "trades", "INSERT", row_id=902002)
            _assert_fenced(raised.value, "ACTIVE")
            active_writer.rollback()
        finally:
            active_writer.close()

        for statement, expected_sqlstate in (
            (
                "UPDATE np.paper_runtime_control SET mode = 'LEGACY'",
                "42501",
            ),
            (
                "ALTER TABLE np.trades DISABLE TRIGGER "
                "legacy_paper_runtime_fence_trades",
                "42501",
            ),
            ("TRUNCATE TABLE np.trades", FENCE_SQLSTATE),
        ):
            attacker = _connect(migrated_postgres_dsn)
            try:
                with attacker.cursor() as cursor:
                    cursor.execute(
                        sql.SQL("SET ROLE {}").format(sql.Identifier(role_name))
                    )
                    with pytest.raises(psycopg2.Error) as raised:
                        cursor.execute(statement)
                assert raised.value.pgcode == expected_sqlstate
                if expected_sqlstate == FENCE_SQLSTATE:
                    _assert_fenced(raised.value, "ACTIVE")
                attacker.rollback()
            finally:
                attacker.close()
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


def test_runtime_control_catalog_is_exact_and_fence_triggers_are_always_enabled(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    table_row.relkind,
                    table_row.relpersistence,
                    table_row.relhasrules,
                    table_row.relhastriggers,
                    table_row.relrowsecurity,
                    table_row.relforcerowsecurity,
                    EXISTS (
                        SELECT 1
                        FROM pg_inherits inheritance_row
                        WHERE inheritance_row.inhrelid = table_row.oid
                           OR inheritance_row.inhparent = table_row.oid
                    ),
                    EXISTS (
                        SELECT 1
                        FROM pg_policy policy_row
                        WHERE policy_row.polrelid = table_row.oid
                    )
                FROM pg_class table_row
                JOIN pg_namespace namespace_row
                  ON namespace_row.oid = table_row.relnamespace
                WHERE namespace_row.nspname = 'np'
                  AND table_row.relname = 'paper_runtime_control'
                """)
            assert cursor.fetchall() == [
                ("r", "p", False, False, False, False, False, False)
            ]

            cursor.execute("""
                SELECT
                    ordinal_position,
                    column_name,
                    udt_name,
                    is_nullable,
                    CASE
                        WHEN column_default IS NULL THEN 'none'
                        WHEN LOWER(column_default) IN ('true', 'true::boolean')
                            THEN 'true'
                        WHEN LOWER(column_default) IN (
                            'now()',
                            'current_timestamp'
                        ) THEN 'now'
                        ELSE 'other'
                    END
                FROM information_schema.columns
                WHERE table_schema = 'np'
                  AND table_name = 'paper_runtime_control'
                ORDER BY ordinal_position
                """)
            assert cursor.fetchall() == [
                (1, "control_key", "bool", "NO", "true"),
                (2, "mode", "text", "NO", "none"),
                (3, "runtime_generation", "int8", "NO", "none"),
                (4, "updated_at", "timestamptz", "NO", "now"),
            ]

            cursor.execute("""
                SELECT
                    constraint_row.conname,
                    constraint_row.contype,
                    constraint_row.conkey,
                    constraint_row.condeferrable,
                    constraint_row.condeferred,
                    constraint_row.convalidated,
                    pg_get_constraintdef(constraint_row.oid, TRUE)
                FROM pg_constraint constraint_row
                JOIN pg_class table_row
                  ON table_row.oid = constraint_row.conrelid
                JOIN pg_namespace namespace_row
                  ON namespace_row.oid = table_row.relnamespace
                WHERE namespace_row.nspname = 'np'
                  AND table_row.relname = 'paper_runtime_control'
                ORDER BY constraint_row.conname
                """)
            assert cursor.fetchall() == [
                (
                    "paper_runtime_control_generation_nonnegative",
                    "c",
                    [3],
                    False,
                    False,
                    True,
                    "CHECK (runtime_generation >= 0)",
                ),
                (
                    "paper_runtime_control_mode",
                    "c",
                    [2],
                    False,
                    False,
                    True,
                    "CHECK (mode = ANY (ARRAY['LEGACY'::text, "
                    "'SHADOW'::text, 'PAUSED'::text, 'ACTIVE'::text]))",
                ),
                (
                    "paper_runtime_control_pkey",
                    "p",
                    [1],
                    False,
                    False,
                    True,
                    "PRIMARY KEY (control_key)",
                ),
                (
                    "paper_runtime_control_singleton",
                    "c",
                    [1],
                    False,
                    False,
                    True,
                    "CHECK (control_key)",
                ),
            ]

            cursor.execute("""
                SELECT control_key, mode, runtime_generation
                FROM np.paper_runtime_control
                """)
            assert cursor.fetchall() == [(True, "LEGACY", 0)]

            cursor.execute("""
                SELECT
                    routine_row.prosecdef,
                    routine_row.provolatile,
                    routine_row.proleakproof,
                    routine_row.proisstrict,
                    routine_row.pronargs,
                    routine_row.prorettype = 'trigger'::regtype,
                    language_row.lanname,
                    routine_row.proconfig,
                    routine_row.prosrc
                FROM pg_proc routine_row
                JOIN pg_namespace namespace_row
                  ON namespace_row.oid = routine_row.pronamespace
                JOIN pg_language language_row
                  ON language_row.oid = routine_row.prolang
                WHERE namespace_row.nspname = 'np'
                  AND routine_row.proname = 'enforce_legacy_paper_runtime_fence'
                """)
            rows = cursor.fetchall()
            assert rows == [
                (
                    True,
                    "v",
                    False,
                    False,
                    0,
                    True,
                    "plpgsql",
                    ["search_path=pg_catalog"],
                    """
DECLARE
    current_mode TEXT;
    current_generation BIGINT;
BEGIN
    BEGIN
        SELECT mode, runtime_generation
        INTO STRICT current_mode, current_generation
        FROM np.paper_runtime_control
        WHERE control_key IS TRUE
        FOR SHARE;
    EXCEPTION
        WHEN NO_DATA_FOUND OR TOO_MANY_ROWS OR undefined_table THEN
            RAISE EXCEPTION USING
                ERRCODE = '55000',
                MESSAGE = 'paper runtime control is unavailable';
    END;

    IF current_mode IS NULL
       OR current_generation IS NULL
       OR current_generation < 0
       OR current_mode NOT IN ('LEGACY', 'SHADOW', 'PAUSED', 'ACTIVE') THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = 'paper runtime control is invalid';
    END IF;

    IF current_mode IN ('PAUSED', 'ACTIVE') THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = FORMAT(
                'legacy paper writes are fenced in %s mode',
                current_mode
            );
    END IF;

    RETURN NULL;
END
""",
                )
            ]

            cursor.execute(
                """
                SELECT
                    table_row.relname,
                    trigger_row.tgname,
                    trigger_row.tgenabled,
                    trigger_row.tgtype,
                    function_namespace.nspname,
                    routine_row.proname
                FROM pg_trigger trigger_row
                JOIN pg_class table_row
                  ON table_row.oid = trigger_row.tgrelid
                JOIN pg_namespace namespace_row
                  ON namespace_row.oid = table_row.relnamespace
                JOIN pg_proc routine_row
                  ON routine_row.oid = trigger_row.tgfoid
                JOIN pg_namespace function_namespace
                  ON function_namespace.oid = routine_row.pronamespace
                WHERE namespace_row.nspname = 'np'
                  AND NOT trigger_row.tgisinternal
                  AND table_row.relname = ANY(%s)
                ORDER BY table_row.relname, trigger_row.tgname
                """,
                (list(LEGACY_RELATIONS),),
            )
            assert cursor.fetchall() == [
                (
                    relation,
                    f"legacy_paper_runtime_fence_{relation}",
                    "A",
                    62,
                    "np",
                    "enforce_legacy_paper_runtime_fence",
                )
                for relation in LEGACY_RELATIONS
            ]
    finally:
        connection.rollback()
        connection.close()


def _backend_pid(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_backend_pid()")
        return cursor.fetchone()[0]


def _assert_backend_blocked_by(dsn, *, blocked_pid, blocker_pid):
    observer = _connect(dsn)
    deadline = monotonic() + 10
    try:
        while monotonic() < deadline:
            with observer.cursor() as cursor:
                cursor.execute("SELECT pg_blocking_pids(%s)", (blocked_pid,))
                blockers = cursor.fetchone()[0]
            observer.rollback()
            if blocker_pid in blockers:
                return
        pytest.fail(f"backend {blocked_pid} was not observed blocked by {blocker_pid}")
    finally:
        observer.close()


def test_in_flight_legacy_writer_serializes_before_fence_transition(
    migrated_postgres_dsn,
):
    writer = _connect(migrated_postgres_dsn)
    writer_pid = _backend_pid(writer)
    with writer.cursor() as cursor:
        _write(cursor, "trades", "INSERT")

    updater_started = Event()
    updater_pid = []

    def transition():
        connection = _connect(migrated_postgres_dsn)
        try:
            updater_pid.append(_backend_pid(connection))
            updater_started.set()
            _set_mode(connection, "ACTIVE")
            connection.commit()
        finally:
            connection.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(transition)
            assert updater_started.wait(timeout=10)
            _assert_backend_blocked_by(
                migrated_postgres_dsn,
                blocked_pid=updater_pid[0],
                blocker_pid=writer_pid,
            )
            writer.commit()
            future.result(timeout=10)
    finally:
        writer.close()

    verifier = _connect(migrated_postgres_dsn)
    try:
        with verifier.cursor() as cursor:
            cursor.execute(
                "SELECT mode, runtime_generation FROM np.paper_runtime_control"
            )
            assert cursor.fetchone() == ("ACTIVE", 1)
            cursor.execute("SELECT COUNT(*) FROM np.trades WHERE id = 900001")
            assert cursor.fetchone() == (1,)
    finally:
        verifier.rollback()
        verifier.close()


def test_committing_fence_transition_serializes_before_new_legacy_writer(
    migrated_postgres_dsn,
):
    transition = _connect(migrated_postgres_dsn)
    transition_pid = _backend_pid(transition)
    _set_mode(transition, "ACTIVE")

    writer_started = Event()
    writer_pid = []

    def write():
        connection = _connect(migrated_postgres_dsn)
        try:
            writer_pid.append(_backend_pid(connection))
            writer_started.set()
            try:
                with connection.cursor() as cursor:
                    _write(cursor, "trades", "INSERT")
            except psycopg2.Error as exc:
                connection.rollback()
                return exc
            connection.commit()
            return None
        finally:
            connection.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(write)
            assert writer_started.wait(timeout=10)
            _assert_backend_blocked_by(
                migrated_postgres_dsn,
                blocked_pid=writer_pid[0],
                blocker_pid=transition_pid,
            )
            transition.commit()
            error = future.result(timeout=10)
    finally:
        transition.close()

    assert isinstance(error, psycopg2.Error)
    _assert_fenced(error, "ACTIVE")

    verifier = _connect(migrated_postgres_dsn)
    try:
        with verifier.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM np.trades WHERE id = 900001")
            assert cursor.fetchone() == (0,)
    finally:
        verifier.rollback()
        verifier.close()


def test_stale_repeatable_read_writer_cannot_write_after_active_commit(
    migrated_postgres_dsn,
):
    stale_writer = _connect(migrated_postgres_dsn)
    transition = _connect(migrated_postgres_dsn)
    try:
        with stale_writer.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("SELECT mode FROM np.paper_runtime_control")
            assert cursor.fetchone() == ("LEGACY",)

        _set_mode(transition, "ACTIVE")
        transition.commit()

        with pytest.raises(psycopg2.Error) as raised:
            with stale_writer.cursor() as cursor:
                _write(cursor, "trades", "INSERT")

        assert raised.value.pgcode in {"40001", FENCE_SQLSTATE}
        if raised.value.pgcode == FENCE_SQLSTATE:
            _assert_fenced(raised.value, "ACTIVE")
        stale_writer.rollback()

        with transition.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM np.trades WHERE id = 900001")
            assert cursor.fetchone() == (0,)
        transition.rollback()
    finally:
        stale_writer.close()
        transition.close()
