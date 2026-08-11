"""Small PostgreSQL migration runner with immutable checksums."""

import hashlib
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import Iterable

from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

_MIGRATION_PACKAGE = "trading.persistence.sql_migrations"
_MIGRATION_FILENAME = re.compile(
    r"(?P<version>[0-9]{4})_(?P<name>[a-z][a-z0-9_]*)\.sql"
)
_MIGRATION_NAME = re.compile(r"[a-z][a-z0-9_]*")
_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
_ADVISORY_LOCK_ID = 4_544_865_376_849_463
_SET_TRANSACTION_SQL = "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
_SET_STRING_SYNTAX_SQL = "SET LOCAL standard_conforming_strings = on"
_FLUSH_DEFERRED_SQL = "SET CONSTRAINTS ALL IMMEDIATE"
_RUNNER_CONTROL_COMMANDS = {
    "ABORT",
    "BEGIN",
    "COMMIT",
    "END",
    "PREPARE",
    "RELEASE",
    "RESET",
    "ROLLBACK",
    "SAVEPOINT",
    "SET",
}

_CREATE_METADATA_SQL = """
CREATE SCHEMA IF NOT EXISTS np;
CREATE TABLE IF NOT EXISTS np.schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum CHAR(64) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_VALIDATE_METADATA_SQL = """
DO $migration$
DECLARE
    layout_mismatch BOOLEAN;
    constraint_mismatch BOOLEAN;
    relation_mismatch BOOLEAN;
    behavior_mismatch BOOLEAN;
BEGIN
    WITH expected (
        ordinal_position,
        column_name,
        udt_name,
        is_nullable,
        default_kind,
        character_maximum_length
    ) AS (
        VALUES
            (1, 'version', 'int4', 'NO', 'none', NULL::INTEGER),
            (2, 'name', 'text', 'NO', 'none', NULL::INTEGER),
            (3, 'checksum', 'bpchar', 'NO', 'none', 64),
            (4, 'applied_at', 'timestamptz', 'NO', 'now', NULL::INTEGER)
    ),
    actual AS (
        SELECT
            ordinal_position,
            column_name,
            udt_name,
            is_nullable,
            CASE
                WHEN column_default IS NULL THEN 'none'
                WHEN LOWER(column_default) IN ('now()', 'current_timestamp')
                    THEN 'now'
                ELSE 'other'
            END AS default_kind,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = 'np'
          AND table_name = 'schema_migrations'
    ),
    differences AS (
        (SELECT * FROM expected EXCEPT SELECT * FROM actual)
        UNION ALL
        (SELECT * FROM actual EXCEPT SELECT * FROM expected)
    )
    SELECT EXISTS (SELECT 1 FROM differences) INTO layout_mismatch;

    SELECT
        COUNT(*) <> 1
        OR NOT COALESCE(
            BOOL_OR(
                constraint_row.contype = 'p'
                AND constraint_row.conkey = ARRAY[1]::SMALLINT[]
                AND NOT constraint_row.condeferrable
                AND NOT constraint_row.condeferred
                AND constraint_row.convalidated
            ),
            FALSE
        )
    INTO constraint_mismatch
    FROM pg_constraint constraint_row
    JOIN pg_class table_row
      ON table_row.oid = constraint_row.conrelid
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = table_row.relnamespace
    WHERE namespace_row.nspname = 'np'
      AND table_row.relname = 'schema_migrations';

    SELECT NOT EXISTS (
        SELECT 1
        FROM pg_class table_row
        JOIN pg_namespace namespace_row
          ON namespace_row.oid = table_row.relnamespace
        WHERE namespace_row.nspname = 'np'
          AND table_row.relname = 'schema_migrations'
          AND table_row.relkind = 'r'
          AND table_row.relpersistence = 'p'
    ) INTO relation_mismatch;

    SELECT
        table_row.relhasrules
        OR table_row.relhastriggers
        OR table_row.relrowsecurity
        OR table_row.relforcerowsecurity
        OR EXISTS (
            SELECT 1
            FROM pg_inherits inheritance_row
            WHERE inheritance_row.inhrelid = table_row.oid
               OR inheritance_row.inhparent = table_row.oid
        )
        OR EXISTS (
            SELECT 1
            FROM pg_policy policy_row
            WHERE policy_row.polrelid = table_row.oid
        )
    INTO behavior_mismatch
    FROM pg_class table_row
    JOIN pg_namespace namespace_row
      ON namespace_row.oid = table_row.relnamespace
    WHERE namespace_row.nspname = 'np'
      AND table_row.relname = 'schema_migrations';

    IF layout_mismatch
       OR constraint_mismatch
       OR relation_mismatch
       OR behavior_mismatch THEN
        RAISE EXCEPTION 'migration ledger layout is incompatible';
    END IF;
END
$migration$;
"""

_LOAD_APPLIED_SQL = """
SELECT version, name, checksum
FROM np.schema_migrations
ORDER BY version ASC
"""

_RECORD_APPLIED_SQL = """
INSERT INTO np.schema_migrations (version, name, checksum)
VALUES (%s, %s, %s)
RETURNING version, name, checksum
"""


class MigrationApplyError(RuntimeError):
    """Raised when PostgreSQL cannot atomically apply a migration sequence."""


class MigrationDriftError(MigrationApplyError):
    """Raised when recorded migration identity differs from packaged SQL."""


def _statement_prefixes(sql: str) -> tuple[tuple[str, ...], ...]:
    """Return up to two unquoted words from each top-level SQL statement."""
    prefixes = []
    words = []
    index = 0
    length = len(sql)

    while index < length:
        character = sql[index]

        if character.isspace():
            index += 1
            continue

        if sql.startswith("--", index):
            index += 2
            while index < length and sql[index] not in "\r\n":
                index += 1
            continue

        if sql.startswith("/*", index):
            depth = 1
            index += 2
            while index < length and depth:
                if sql.startswith("/*", index):
                    depth += 1
                    index += 2
                elif sql.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            if depth:
                raise ValueError("migration sql contains an unterminated comment")
            continue

        if character == "'":
            escape_backslashes = (
                index > 0
                and sql[index - 1] in "Ee"
                and (
                    index == 1
                    or not (sql[index - 2].isalnum() or sql[index - 2] in "_$")
                )
            )
            index += 1
            while index < length:
                if escape_backslashes and sql[index] == "\\":
                    index += 2
                elif sql[index] == "'":
                    if index + 1 < length and sql[index + 1] == "'":
                        index += 2
                    else:
                        index += 1
                        break
                else:
                    index += 1
            else:
                raise ValueError("migration sql contains an unterminated string")
            continue

        if character == '"':
            index += 1
            while index < length:
                if sql[index] == '"':
                    if index + 1 < length and sql[index + 1] == '"':
                        index += 2
                    else:
                        index += 1
                        break
                else:
                    index += 1
            else:
                raise ValueError("migration sql contains an unterminated identifier")
            continue

        dollar_match = None
        if character == "$" and (
            index == 0 or not (sql[index - 1].isalnum() or sql[index - 1] in "_$")
        ):
            dollar_match = _DOLLAR_QUOTE.match(sql, index)
        if dollar_match is not None:
            delimiter = dollar_match.group(0)
            closing = sql.find(delimiter, dollar_match.end())
            if closing < 0:
                raise ValueError("migration sql contains an unterminated dollar quote")
            index = closing + len(delimiter)
            continue

        if character == ";":
            if words:
                prefixes.append(tuple(words))
            words = []
            index += 1
            continue

        if character.isalpha() or character == "_":
            end = index + 1
            while end < length and (sql[end].isalnum() or sql[end] in "_$"):
                end += 1
            if len(words) < 2:
                words.append(sql[index:end].upper())
            index = end
            continue

        index += 1

    if words:
        prefixes.append(tuple(words))
    return tuple(prefixes)


def _reject_transaction_control(sql: str) -> None:
    for prefix in _statement_prefixes(sql):
        first = prefix[0]
        second = prefix[1] if len(prefix) > 1 else None
        if first in _RUNNER_CONTROL_COMMANDS or (
            first == "START" and second == "TRANSACTION"
        ):
            raise ValueError(
                "migration sql must not control transactions or session settings"
            )


@dataclass(frozen=True, slots=True)
class Migration:
    """One immutable, ordered SQL migration."""

    version: int
    name: str
    sql: str
    checksum: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("migration version must be an integer")
        if self.version < 1:
            raise ValueError("migration version must be positive")
        if not isinstance(self.name, str) or not _MIGRATION_NAME.fullmatch(self.name):
            raise ValueError("migration name must be a lowercase identifier")
        if not isinstance(self.sql, str):
            raise TypeError("migration sql must be text")
        if not self.sql.strip():
            raise ValueError("migration sql must not be empty")
        _reject_transaction_control(self.sql)
        object.__setattr__(
            self,
            "checksum",
            hashlib.sha256(self.sql.encode("utf-8")).hexdigest(),
        )


def _validated_migrations(migrations: Iterable[Migration]) -> tuple[Migration, ...]:
    sequence = tuple(migrations)
    if not sequence:
        raise ValueError("at least one migration is required")
    if any(type(migration) is not Migration for migration in sequence):
        raise TypeError("migrations must contain only Migration values")
    versions = tuple(migration.version for migration in sequence)
    expected = tuple(range(1, len(sequence) + 1))
    if versions != expected:
        raise ValueError(
            "migration versions must be strictly increasing without gaps from 1"
        )
    return sequence


def load_migrations(
    package: str = _MIGRATION_PACKAGE,
) -> tuple[Migration, ...]:
    """Load packaged SQL by filename and verify a contiguous version sequence."""
    if not isinstance(package, str) or not package:
        raise TypeError("migration package must be a non-empty string")

    discovered = []
    for resource in resources.files(package).iterdir():
        if not resource.name.endswith(".sql"):
            continue
        match = _MIGRATION_FILENAME.fullmatch(resource.name)
        if match is None:
            raise ValueError(f"invalid migration filename: {resource.name}")
        discovered.append(
            Migration(
                version=int(match.group("version")),
                name=match.group("name"),
                sql=resource.read_text(encoding="utf-8"),
            )
        )

    discovered.sort(key=lambda migration: migration.version)
    return _validated_migrations(discovered)


def apply_migrations(
    connection: object,
    migrations: Iterable[Migration],
) -> tuple[int, ...]:
    """Apply every pending migration in one locked PostgreSQL transaction."""
    if not callable(getattr(connection, "cursor", None)):
        raise TypeError("connection must provide cursor()")
    if not callable(getattr(connection, "commit", None)):
        raise TypeError("connection must provide commit()")
    if not callable(getattr(connection, "rollback", None)):
        raise TypeError("connection must provide rollback()")
    transaction_status = getattr(connection, "get_transaction_status", None)
    if not callable(transaction_status):
        raise TypeError("connection must provide get_transaction_status()")
    if getattr(connection, "autocommit", None) is not False:
        raise ValueError("migration connection must have autocommit disabled")
    if transaction_status() != TRANSACTION_STATUS_IDLE:
        raise ValueError("migration connection must not have an active transaction")
    if getattr(connection, "status", None) != STATUS_READY:
        raise ValueError("migration connection must be ready")

    sequence = _validated_migrations(migrations)
    by_version = {migration.version: migration for migration in sequence}
    current_migration = None
    applied_versions = []

    try:
        with connection.cursor() as cursor:
            cursor.execute(_SET_TRANSACTION_SQL)
            cursor.execute(_SET_STRING_SYNTAX_SQL)
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (_ADVISORY_LOCK_ID,),
            )
            cursor.execute(_CREATE_METADATA_SQL)
            cursor.execute(_VALIDATE_METADATA_SQL)
            cursor.execute(_LOAD_APPLIED_SQL)
            recorded = tuple(cursor.fetchall())

            recorded_versions = tuple(int(row[0]) for row in recorded)
            expected_recorded_versions = tuple(range(1, len(recorded_versions) + 1))
            if recorded_versions != expected_recorded_versions:
                raise MigrationDriftError(
                    "database migration history is not a contiguous prefix"
                )

            for raw_version, raw_name, raw_checksum in recorded:
                version = int(raw_version)
                expected = by_version.get(version)
                if expected is None:
                    raise MigrationDriftError(
                        f"database contains unknown migration version {version}"
                    )
                if raw_name != expected.name or raw_checksum != expected.checksum:
                    raise MigrationDriftError(
                        f"database migration version {version} has drifted"
                    )

            recorded_version_set = set(recorded_versions)
            for migration in sequence:
                if migration.version in recorded_version_set:
                    continue
                current_migration = migration
                cursor.execute(_SET_STRING_SYNTAX_SQL)
                cursor.execute(migration.sql)
                cursor.execute(
                    _RECORD_APPLIED_SQL,
                    (migration.version, migration.name, migration.checksum),
                )
                recorded_identity = cursor.fetchone()
                expected_identity = (
                    migration.version,
                    migration.name,
                    migration.checksum,
                )
                if (
                    recorded_identity is None
                    or tuple(recorded_identity) != expected_identity
                ):
                    raise MigrationDriftError(
                        f"database did not record migration version {migration.version}"
                    )
                applied_versions.append(migration.version)

            cursor.execute(_FLUSH_DEFERRED_SQL)
            cursor.execute(_VALIDATE_METADATA_SQL)
            cursor.execute(_LOAD_APPLIED_SQL)
            verified = tuple(cursor.fetchall())
            expected_history = tuple(
                (migration.version, migration.name, migration.checksum)
                for migration in sequence
            )
            if verified != expected_history:
                raise MigrationDriftError(
                    "database migration history changed while applying migrations"
                )

        connection.commit()
    except MigrationDriftError:
        connection.rollback()
        raise
    except Exception as exc:
        connection.rollback()
        if current_migration is None:
            message = "migration metadata initialization failed"
        else:
            message = (
                f"migration {current_migration.version:04d}_"
                f"{current_migration.name} failed"
            )
        raise MigrationApplyError(message) from exc

    return tuple(applied_versions)
