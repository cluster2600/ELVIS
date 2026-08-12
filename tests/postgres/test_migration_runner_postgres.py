"""PostgreSQL 15 integration checks for the packaged migration runner."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import psycopg2
import pytest
from psycopg2 import sql

from trading.persistence import (
    Migration,
    MigrationApplyError,
    apply_migrations,
    load_migrations,
)


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def test_harness_rejects_admin_database_during_test(
    postgres_database_dsn,
    postgres_admin_dsn,
):
    with pytest.raises(pytest.fail.Exception, match="outside the disposable"):
        psycopg2.connect(postgres_admin_dsn)

    with ThreadPoolExecutor(max_workers=1) as executor:
        attempt = executor.submit(psycopg2.connect, postgres_admin_dsn)
        with pytest.raises(pytest.fail.Exception, match="outside the disposable"):
            attempt.result()


def test_fresh_database_migrates_once_without_business_seed(postgres_database_dsn):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        assert apply_migrations(connection, migrations) == (1, 2, 3, 4)
        assert apply_migrations(connection, migrations) == ()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'np'
                ORDER BY table_name
                """)
            assert tuple(row[0] for row in cursor.fetchall()) == (
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
                "position_streams",
                "schema_migrations",
                "trades",
                "trading_session_resets",
            )
            cursor.execute("SELECT version, name, checksum FROM np.schema_migrations")
            assert cursor.fetchall() == [
                (migration.version, migration.name, migration.checksum)
                for migration in migrations
            ]
            cursor.execute("SELECT COUNT(*) FROM np.account_balances")
            assert cursor.fetchone() == (0,)
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM np.position_streams),
                    (SELECT COUNT(*) FROM np.orders),
                    (SELECT COUNT(*) FROM np.order_events)
                """)
            assert cursor.fetchone() == (0, 0, 0)
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM np.paper_account_streams),
                    (SELECT COUNT(*) FROM np.paper_account_balances),
                    (SELECT COUNT(*) FROM np.paper_margin_reservations),
                    (SELECT COUNT(*) FROM np.paper_account_batch_manifests),
                    (SELECT COUNT(*) FROM np.paper_account_settlements),
                    (SELECT COUNT(*) FROM np.paper_account_postings)
                """)
            assert cursor.fetchone() == (0, 0, 0, 0, 0, 0)
    finally:
        connection.close()


def test_exact_unversioned_legacy_schema_is_adopted_without_data_loss(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(migrations[0].sql)
            cursor.execute("""
                INSERT INTO np.open_positions (
                    symbol, side, entry_price, quantity, leverage
                ) VALUES ('BTCUSDT', 'BUY', 50000, 0.01, 3)
                """)
        connection.commit()

        assert apply_migrations(connection, migrations) == (1, 2, 3, 4)
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT symbol, side, entry_price, quantity, leverage
                FROM np.open_positions
                """)
            assert cursor.fetchone() == (
                "BTCUSDT",
                "BUY",
                50000.0,
                pytest.approx(0.01),
                3.0,
            )
    finally:
        connection.close()


def test_versioned_baseline_upgrades_to_journal_without_legacy_data_loss(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        assert apply_migrations(connection, migrations[:1]) == (1,)
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO np.open_positions (
                    symbol, side, entry_price, quantity, leverage
                ) VALUES ('BTCUSDT', 'BUY', 51000, 0.02, 3)
                """)
        connection.commit()

        assert apply_migrations(connection, migrations) == (2, 3, 4)
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT symbol, side, entry_price, quantity, leverage
                FROM np.open_positions
                """)
            assert cursor.fetchone() == (
                "BTCUSDT",
                "BUY",
                51000.0,
                pytest.approx(0.02),
                3.0,
            )
            cursor.execute("SELECT to_regclass('np.order_events')")
            assert cursor.fetchone() == ("np.order_events",)
            cursor.execute("SELECT to_regclass('np.paper_account_streams')")
            assert cursor.fetchone() == ("np.paper_account_streams",)
    finally:
        connection.close()


def test_versioned_journal_upgrades_to_dormant_account_ledger(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        assert apply_migrations(connection, migrations[:2]) == (1, 2)
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO np.position_streams (
                    position_key, execution_scope
                ) VALUES ('position-existing', 'paper:upgrade')
                """)
        connection.commit()

        assert apply_migrations(connection, migrations) == (3, 4)
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT position_key, execution_scope
                FROM np.position_streams
                """)
            assert cursor.fetchone() == ("position-existing", "paper:upgrade")
            cursor.execute("SELECT to_regclass('np.paper_account_streams')")
            assert cursor.fetchone() == ("np.paper_account_streams",)
            cursor.execute("SELECT COUNT(*) FROM np.paper_account_streams")
            assert cursor.fetchone() == (0,)
            cursor.execute(
                "SELECT control_key, mode, runtime_generation "
                "FROM np.paper_runtime_control"
            )
            assert cursor.fetchone() == (True, "LEGACY", 0)
    finally:
        connection.close()


def test_versioned_account_ledger_upgrades_to_dormant_runtime_fence_without_loss(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        assert apply_migrations(connection, migrations[:3]) == (1, 2, 3)
        with connection.cursor() as cursor:
            for relation in (
                "liquidations",
                "margin_history",
                "model_predictions",
                "open_positions",
                "trades",
                "trading_session_resets",
            ):
                cursor.execute(f"INSERT INTO np.{relation} (id) VALUES (710001)")
            cursor.execute("""
                INSERT INTO np.account_balances (id, asset, balance)
                VALUES (710001, 'USDT', 100)
                """)
        connection.commit()

        assert apply_migrations(connection, migrations) == (4,)
        with connection.cursor() as cursor:
            for relation in (
                "account_balances",
                "liquidations",
                "margin_history",
                "model_predictions",
                "open_positions",
                "trades",
                "trading_session_resets",
            ):
                cursor.execute(f"SELECT id FROM np.{relation}")
                assert cursor.fetchall() == [(710001,)]
            cursor.execute(
                "SELECT control_key, mode, runtime_generation "
                "FROM np.paper_runtime_control"
            )
            assert cursor.fetchone() == (True, "LEGACY", 0)
    finally:
        connection.close()


def test_runtime_fence_function_collision_rolls_back_version_four_only(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        assert apply_migrations(connection, migrations[:3]) == (1, 2, 3)
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE FUNCTION np.enforce_legacy_paper_runtime_fence()
                RETURNS TRIGGER
                LANGUAGE plpgsql
                AS $function$
                BEGIN
                    RETURN NULL;
                END
                $function$
                """)
        connection.commit()

        with pytest.raises(MigrationApplyError, match="0004_paper_runtime_control"):
            apply_migrations(connection, migrations)

        with connection.cursor() as cursor:
            cursor.execute("SELECT version FROM np.schema_migrations ORDER BY version")
            assert cursor.fetchall() == [(1,), (2,), (3,)]
            cursor.execute("SELECT to_regclass('np.paper_runtime_control')")
            assert cursor.fetchone() == (None,)
            cursor.execute("""
                SELECT COUNT(*)
                FROM pg_trigger trigger_row
                JOIN pg_class table_row ON table_row.oid = trigger_row.tgrelid
                JOIN pg_namespace namespace_row
                  ON namespace_row.oid = table_row.relnamespace
                WHERE namespace_row.nspname = 'np'
                  AND NOT trigger_row.tgisinternal
                """)
            assert cursor.fetchone() == (0,)
            cursor.execute("""
                SELECT routine_row.prosecdef
                FROM pg_proc routine_row
                JOIN pg_namespace namespace_row
                  ON namespace_row.oid = routine_row.pronamespace
                WHERE namespace_row.nspname = 'np'
                  AND routine_row.proname = 'enforce_legacy_paper_runtime_fence'
                """)
            assert cursor.fetchone() == (False,)
    finally:
        connection.close()


def test_runtime_fence_upgrade_allows_legacy_table_owner_to_differ(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    role_name = None
    try:
        assert apply_migrations(connection, migrations[:3]) == (1, 2, 3)
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            role_name = f"{cursor.fetchone()[0]}_legacy_owner"
            cursor.execute(
                sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role_name))
            )
            cursor.execute(
                sql.SQL("ALTER TABLE np.trades OWNER TO {}").format(
                    sql.Identifier(role_name)
                )
            )
        connection.commit()

        assert apply_migrations(connection, migrations) == (4,)
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    legacy_table.relowner <> control_table.relowner,
                    fence_function.proowner = control_table.relowner
                FROM pg_class legacy_table
                JOIN pg_namespace legacy_namespace
                  ON legacy_namespace.oid = legacy_table.relnamespace
                JOIN pg_class control_table
                  ON control_table.relname = 'paper_runtime_control'
                JOIN pg_namespace control_namespace
                  ON control_namespace.oid = control_table.relnamespace
                 AND control_namespace.nspname = 'np'
                JOIN pg_proc fence_function
                  ON fence_function.proname = 'enforce_legacy_paper_runtime_fence'
                JOIN pg_namespace function_namespace
                  ON function_namespace.oid = fence_function.pronamespace
                 AND function_namespace.nspname = 'np'
                WHERE legacy_namespace.nspname = 'np'
                  AND legacy_table.relname = 'trades'
                """)
            assert cursor.fetchone() == (True, True)
    finally:
        connection.rollback()
        if role_name is not None:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role_name))
                )
                cursor.execute(
                    sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name))
                )
            connection.commit()
        connection.close()


def test_account_ledger_table_collision_rolls_back_version_three_only(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        assert apply_migrations(connection, migrations[:2]) == (1, 2)
        with connection.cursor() as cursor:
            cursor.execute("CREATE TABLE np.paper_account_streams (unexpected INTEGER)")
        connection.commit()

        with pytest.raises(MigrationApplyError, match="0003_paper_account_ledger"):
            apply_migrations(connection, migrations)

        with connection.cursor() as cursor:
            cursor.execute("SELECT version FROM np.schema_migrations ORDER BY version")
            assert cursor.fetchall() == [(1,), (2,)]
            cursor.execute("SELECT to_regclass('np.paper_account_balances')")
            assert cursor.fetchone() == (None,)
            cursor.execute("SELECT unexpected FROM np.paper_account_streams")
            assert cursor.fetchall() == []
            cursor.execute("SELECT to_regclass('np.orders_paper_account_batch_ref_uq')")
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


def test_journal_table_collision_rolls_back_version_two_only(
    postgres_database_dsn,
):
    migrations = load_migrations()
    connection = _connect(postgres_database_dsn)
    try:
        assert apply_migrations(connection, migrations[:1]) == (1,)
        with connection.cursor() as cursor:
            cursor.execute("CREATE TABLE np.orders (unexpected INTEGER)")
            cursor.execute("""
                INSERT INTO np.open_positions (
                    symbol, side, entry_price, quantity, leverage
                ) VALUES ('BNBUSDT', 'SELL', 600, 0.5, 2)
                """)
        connection.commit()

        with pytest.raises(MigrationApplyError, match="0002_order_position_journal"):
            apply_migrations(connection, migrations)

        with connection.cursor() as cursor:
            cursor.execute("SELECT version FROM np.schema_migrations ORDER BY version")
            assert cursor.fetchall() == [(1,)]
            cursor.execute("SELECT to_regclass('np.position_streams')")
            assert cursor.fetchone() == (None,)
            cursor.execute("SELECT unexpected FROM np.orders")
            assert cursor.fetchall() == []
            cursor.execute("SELECT symbol FROM np.open_positions")
            assert cursor.fetchone() == ("BNBUSDT",)
    finally:
        connection.close()


def test_broken_followup_rolls_back_baseline_and_ledger(postgres_database_dsn):
    baseline = load_migrations()[0]
    broken = Migration(
        version=2,
        name="broken_followup",
        sql="CREATE TABLE public.must_rollback (id INTEGER); SELECT 1 / 0;",
    )
    connection = _connect(postgres_database_dsn)
    try:
        with pytest.raises(MigrationApplyError, match="0002_broken_followup"):
            apply_migrations(connection, (baseline, broken))
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.must_rollback')")
            assert cursor.fetchone() == (None,)
            cursor.execute("SELECT to_regclass('np.schema_migrations')")
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


def test_concurrent_runners_apply_packaged_migrations_once(postgres_database_dsn):
    migrations = load_migrations()
    barrier = Barrier(2)

    def migrate():
        connection = _connect(postgres_database_dsn)
        try:
            barrier.wait(timeout=10)
            return apply_migrations(connection, migrations)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: migrate(), range(2)))

    assert sorted(results) == [(), (1, 2, 3, 4)]
    connection = _connect(postgres_database_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM np.schema_migrations")
            assert cursor.fetchone() == (4,)
    finally:
        connection.close()
