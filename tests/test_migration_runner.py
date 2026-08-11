import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from psycopg2.extensions import (
    STATUS_BEGIN,
    STATUS_PREPARED,
    STATUS_READY,
    TRANSACTION_STATUS_IDLE,
    TRANSACTION_STATUS_INTRANS,
)

from trading.persistence.migration_runner import (
    Migration,
    MigrationApplyError,
    MigrationDriftError,
    _statement_prefixes,
    apply_migrations,
    load_migrations,
)


def fake_connection(*, applied_rows=()):
    existing_rows = tuple(applied_rows)
    connection = MagicMock()
    connection.autocommit = False
    connection.status = STATUS_READY
    connection.get_transaction_status.return_value = TRANSACTION_STATUS_IDLE
    cursor = MagicMock()

    def recorded_rows():
        inserted = tuple(
            tuple(call.args[1])
            for call in cursor.execute.call_args_list
            if call.args and "INSERT INTO np.schema_migrations" in str(call.args[0])
        )
        return list(existing_rows + inserted)

    def returned_identity():
        call = cursor.execute.call_args
        if (
            call is not None
            and call.args
            and "INSERT INTO np.schema_migrations" in str(call.args[0])
        ):
            return tuple(call.args[1])
        return None

    cursor.fetchall.side_effect = recorded_rows
    cursor.fetchone.side_effect = returned_identity
    connection.cursor.return_value.__enter__.return_value = cursor
    connection.cursor.return_value.__exit__.return_value = False
    return connection, cursor


def test_packaged_migrations_are_ordered_additive_and_immutable() -> None:
    migrations = load_migrations()

    assert tuple(migration.version for migration in migrations) == (1, 2)
    assert migrations[0].name == "legacy_baseline"
    assert migrations[0].checksum == (
        "38d01ec919fa4a39ee28423c74326c6f" "5dd51d0e7e8216f9a8cffb9b11b5c9b1"
    )
    assert "CREATE SCHEMA IF NOT EXISTS np" in migrations[0].sql
    assert "CREATE TABLE IF NOT EXISTS np.trades" in migrations[0].sql
    assert "CREATE TABLE IF NOT EXISTS np.open_positions" in migrations[0].sql
    assert "information_schema.columns" in migrations[0].sql
    assert "legacy table layout is incompatible" in migrations[0].sql
    forbidden_statements = (
        "DROP ",
        "TRUNCATE ",
        "DELETE ",
        "UPDATE ",
        "ALTER ",
        "RENAME ",
    )
    assert not any(
        statement in migrations[0].sql.upper() for statement in forbidden_statements
    )
    assert "INSERT INTO np.account_balances" not in migrations[0].sql
    assert "np.orders" not in migrations[0].sql
    assert "np.order_events" not in migrations[0].sql
    for deferred_contract in (
        "position_key",
        "position_effect",
        "reduce_only",
        "take_profit_profile",
        "order_fills",
    ):
        assert deferred_contract not in migrations[0].sql.lower()

    journal = migrations[1]
    assert journal.name == "order_position_journal"
    assert journal.checksum == (
        "b33131cc968545de5d5fa18ea6c54a4a" "7e2da50941258a942894edda98d1e234"
    )
    assert "CREATE TABLE np.position_streams" in journal.sql
    assert "CREATE TABLE np.orders" in journal.sql
    assert "CREATE TABLE np.order_events" in journal.sql
    assert "position_version" in journal.sql
    assert "execution_scope" in journal.sql
    assert "instruction_payload JSONB" in journal.sql
    assert "event_payload JSONB" in journal.sql
    assert "NUMERIC" not in journal.sql.upper()
    assert "np.trades" not in journal.sql
    assert "np.open_positions" not in journal.sql
    assert {prefix[0] for prefix in _statement_prefixes(journal.sql)} == {"CREATE"}


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"version": True, "name": "valid", "sql": "SELECT 1"}, TypeError),
        ({"version": 0, "name": "valid", "sql": "SELECT 1"}, ValueError),
        ({"version": 1, "name": "Bad Name", "sql": "SELECT 1"}, ValueError),
        ({"version": 1, "name": "valid", "sql": "  "}, ValueError),
    ],
)
def test_migration_rejects_invalid_metadata(kwargs, exception) -> None:
    with pytest.raises(exception):
        Migration(**kwargs)


@pytest.mark.parametrize(
    "command",
    [
        "ABORT",
        "BEGIN",
        "COMMIT",
        "END TRANSACTION",
        "PREPARE TRANSACTION 'migration'",
        "RELEASE SAVEPOINT migration",
        "RESET standard_conforming_strings",
        "ROLLBACK",
        "SAVEPOINT migration",
        "SET LOCAL standard_conforming_strings = off",
        "START TRANSACTION",
    ],
)
def test_migration_rejects_top_level_transaction_control(command) -> None:
    with pytest.raises(ValueError, match="must not control transactions"):
        Migration(
            version=1,
            name="unsafe",
            sql=f"SELECT 1; /* boundary */ {command}; SELECT 2",
        )


def test_migration_line_comment_ends_on_carriage_return() -> None:
    with pytest.raises(ValueError, match="must not control transactions"):
        Migration(
            version=1,
            name="carriage_return",
            sql="SELECT 1; -- comment\r COMMIT; SELECT 2",
        )


def test_migration_lexer_ignores_quoted_or_nested_transaction_words() -> None:
    sql = """
    -- COMMIT;
    SELECT 'ROLLBACK; still text';
    SELECT E'BEGIN; escaped quote: \\'';
    SELECT CASE WHEN TRUE THEN 1 ELSE 0 END;
    CREATE TABLE "COMMIT" (value TEXT);
    DO $body$
    BEGIN
        RAISE NOTICE 'SAVEPOINT;';
    END
    $body$;
    /* outer ROLLBACK; /* nested COMMIT; */ still comment */
    SELECT 2;
    """

    migration = Migration(version=1, name="quoted", sql=sql)

    assert migration.sql == sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 'unterminated",
        'SELECT "unterminated',
        "SELECT $body$unterminated",
        "SELECT 1 /* unterminated",
    ],
)
def test_migration_rejects_unterminated_lexical_constructs(sql) -> None:
    with pytest.raises(ValueError, match="unterminated"):
        Migration(version=1, name="malformed", sql=sql)


def test_apply_migrations_commits_once_and_records_checksum() -> None:
    connection, cursor = fake_connection()
    migration = Migration(version=1, name="example", sql="SELECT 1")

    applied = apply_migrations(connection, (migration,))

    assert applied == (1,)
    connection.commit.assert_called_once_with()
    connection.rollback.assert_not_called()
    assert cursor.execute.call_args_list[0].args == (
        "SET TRANSACTION ISOLATION LEVEL READ COMMITTED",
    )
    assert cursor.execute.call_args_list[1].args == (
        "SET LOCAL standard_conforming_strings = on",
    )
    executed_sql = tuple(str(call.args[0]) for call in cursor.execute.call_args_list)
    metadata_index = next(
        index
        for index, statement in enumerate(executed_sql)
        if "CREATE TABLE IF NOT EXISTS np.schema_migrations" in statement
    )
    validation_index = next(
        index
        for index, statement in enumerate(executed_sql)
        if "migration ledger layout is incompatible" in statement
    )
    load_index = next(
        index
        for index, statement in enumerate(executed_sql)
        if "FROM np.schema_migrations" in statement
    )
    assert metadata_index < validation_index < load_index
    validation_indexes = tuple(
        index
        for index, statement in enumerate(executed_sql)
        if "migration ledger layout is incompatible" in statement
    )
    flush_index = executed_sql.index("SET CONSTRAINTS ALL IMMEDIATE")
    load_indexes = tuple(
        index
        for index, statement in enumerate(executed_sql)
        if "FROM np.schema_migrations" in statement
    )
    assert len(validation_indexes) == 2
    assert flush_index < validation_indexes[-1] < load_indexes[-1]
    migration_index = executed_sql.index(migration.sql)
    assert executed_sql[migration_index - 1] == (
        "SET LOCAL standard_conforming_strings = on"
    )
    assert any(
        call.args
        and "INSERT INTO np.schema_migrations" in str(call.args[0])
        and call.args[1] == (migration.version, migration.name, migration.checksum)
        for call in cursor.execute.call_args_list
    )


def test_apply_migrations_is_an_idempotent_noop_for_matching_rows() -> None:
    migration = Migration(version=1, name="example", sql="SELECT 1")
    connection, cursor = fake_connection(
        applied_rows=((migration.version, migration.name, migration.checksum),)
    )

    applied = apply_migrations(connection, (migration,))

    assert applied == ()
    connection.commit.assert_called_once_with()
    assert not any(
        call.args and call.args[0] == migration.sql
        for call in cursor.execute.call_args_list
    )


def test_apply_migrations_rejects_checksum_drift_and_rolls_back() -> None:
    migration = Migration(version=1, name="example", sql="SELECT 1")
    connection, _ = fake_connection(
        applied_rows=((migration.version, migration.name, "0" * 64),)
    )

    with pytest.raises(MigrationDriftError, match="version 1"):
        apply_migrations(connection, (migration,))

    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


@pytest.mark.parametrize(
    "applied_versions",
    [
        (2,),
        (1, 3),
    ],
)
def test_apply_migrations_rejects_non_prefix_history(applied_versions) -> None:
    migrations = (
        Migration(version=1, name="one", sql="SELECT 1"),
        Migration(version=2, name="two", sql="SELECT 2"),
        Migration(version=3, name="three", sql="SELECT 3"),
    )
    rows = tuple(
        (version, migrations[version - 1].name, migrations[version - 1].checksum)
        for version in applied_versions
    )
    connection, cursor = fake_connection(applied_rows=rows)

    with pytest.raises(MigrationDriftError, match="contiguous prefix"):
        apply_migrations(connection, migrations)

    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()
    assert not any(
        call.args and call.args[0] in {migration.sql for migration in migrations}
        for call in cursor.execute.call_args_list
    )


def test_apply_migrations_rolls_back_sql_and_version_on_failure() -> None:
    connection, cursor = fake_connection()
    migration = Migration(version=1, name="broken", sql="BROKEN")

    def execute(query, parameters=None):
        if query == migration.sql:
            raise RuntimeError("sensitive database detail")

    cursor.execute.side_effect = execute

    with pytest.raises(
        MigrationApplyError, match="migration 0001_broken failed"
    ) as exc:
        apply_migrations(connection, (migration,))

    assert "sensitive database detail" not in str(exc.value)
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_apply_migrations_rejects_an_incompatible_ledger_before_sql() -> None:
    connection, cursor = fake_connection()
    migration = Migration(version=1, name="example", sql="SELECT 1")

    def execute(query, parameters=None):
        if "migration ledger layout is incompatible" in query:
            raise RuntimeError("private catalogue detail")

    cursor.execute.side_effect = execute

    with pytest.raises(
        MigrationApplyError, match="metadata initialization failed"
    ) as exc:
        apply_migrations(connection, (migration,))

    assert "private catalogue detail" not in str(exc.value)
    assert not any(
        call.args and call.args[0] == migration.sql
        for call in cursor.execute.call_args_list
    )
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_apply_migrations_requires_the_ledger_insert_to_return_identity() -> None:
    connection, cursor = fake_connection()
    cursor.fetchone.side_effect = None
    cursor.fetchone.return_value = None
    migration = Migration(version=1, name="example", sql="SELECT 1")

    with pytest.raises(MigrationDriftError, match="did not record migration"):
        apply_migrations(connection, (migration,))

    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_apply_migrations_rechecks_the_complete_ledger_before_commit() -> None:
    connection, cursor = fake_connection()
    cursor.fetchall.side_effect = [[], []]
    migration = Migration(version=1, name="example", sql="SELECT 1")

    with pytest.raises(MigrationDriftError, match="history changed"):
        apply_migrations(connection, (migration,))

    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_apply_migrations_rejects_unsafe_connection_or_sequence() -> None:
    migration = Migration(version=1, name="example", sql="SELECT 1")
    autocommit_connection, _ = fake_connection()
    autocommit_connection.autocommit = True

    with pytest.raises(ValueError, match="autocommit"):
        apply_migrations(autocommit_connection, (migration,))

    connection, _ = fake_connection()
    duplicate = Migration(version=1, name="other", sql="SELECT 2")
    with pytest.raises(ValueError, match="strictly increasing"):
        apply_migrations(connection, (migration, duplicate))

    empty_connection, _ = fake_connection()
    with pytest.raises(ValueError, match="at least one"):
        apply_migrations(empty_connection, ())
    empty_connection.commit.assert_not_called()
    empty_connection.rollback.assert_not_called()


def test_apply_migrations_rejects_an_existing_caller_transaction() -> None:
    connection, cursor = fake_connection()
    connection.status = STATUS_BEGIN
    connection.get_transaction_status.return_value = TRANSACTION_STATUS_INTRANS
    migration = Migration(version=1, name="example", sql="SELECT 1")

    with pytest.raises(ValueError, match="active transaction"):
        apply_migrations(connection, (migration,))

    cursor.execute.assert_not_called()
    connection.commit.assert_not_called()
    connection.rollback.assert_not_called()


def test_apply_migrations_rejects_a_prepared_connection_without_rollback() -> None:
    connection, cursor = fake_connection()
    connection.status = STATUS_PREPARED
    migration = Migration(version=1, name="example", sql="SELECT 1")

    with pytest.raises(ValueError, match="must be ready"):
        apply_migrations(connection, (migration,))

    cursor.execute.assert_not_called()
    connection.commit.assert_not_called()
    connection.rollback.assert_not_called()


def test_migration_runner_is_not_wired_to_production_startup() -> None:
    root = Path(__file__).parents[1]
    consumers = []
    for source_path in root.rglob("*.py"):
        if (
            "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or source_path.parts[-3:-1] == ("trading", "persistence")
        ):
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports_persistence = any(
            (
                isinstance(node, ast.Import)
                and any(
                    alias.name.startswith("trading.persistence") for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("trading.persistence")
            )
            for node in ast.walk(tree)
        )
        if imports_persistence:
            consumers.append(source_path.relative_to(root))

    assert consumers == []
