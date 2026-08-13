"""Read-only PostgreSQL evidence for a fresh-target cut-over preflight."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections.abc import Callable
from decimal import Decimal

from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

from trading.application.fresh_target_cutover import (
    FreshTargetBootstrapIntent,
    FreshTargetCutoverBlocker,
    FreshTargetCutoverContext,
    FreshTargetCutoverReceipt,
    FreshTargetCutoverSourceEvidence,
    FreshTargetCutoverStatus,
    FreshTargetCutoverTargetEvidence,
    FreshTargetRelationEvidence,
)
from trading.persistence.postgres_bootstrap import (
    _SELECT_INDEX_EVIDENCE_SQL,
    _SELECT_UNEXPECTED_NP_CATALOG_OBJECTS_SQL,
    PostgresBootstrap,
    PostgresBootstrapContext,
    PostgresBootstrapRoles,
    PostgresBootstrapStorageError,
)

_READ_ONLY_SQL = "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
_UTC_SQL = "SET LOCAL TIME ZONE 'UTC'"
_SEARCH_PATH_SQL = "SET LOCAL search_path = pg_catalog"
_LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)
_EXPECTED_MIGRATION_VERSIONS = (1, 2, 3, 4, 5, 6)
_FETCH_BATCH_SIZE = 512
_LEGACY_TABLE_NAMES = tuple(
    relation.removeprefix("np.") for relation in _LEGACY_RELATIONS
)
_LEGACY_SEQUENCE_NAMES = tuple(f"{name}_id_seq" for name in _LEGACY_TABLE_NAMES)
_LEGACY_INDEX_NAMES = tuple(
    sorted(
        tuple(f"{name}_pkey" for name in _LEGACY_TABLE_NAMES)
        + (
            "account_balances_asset_key",
            "idx_model_predictions_scored",
            "idx_trades_symbol_ts",
        )
    )
)
_EXPECTED_LEGACY_COLUMNS = (
    ("account_balances", 1, "id", "int4", "NO", "serial"),
    ("account_balances", 2, "asset", "text", "NO", "none"),
    ("account_balances", 3, "balance", "float4", "NO", "zero"),
    ("account_balances", 4, "last_updated", "timestamp", "YES", "now"),
    ("liquidations", 1, "id", "int4", "NO", "serial"),
    ("liquidations", 2, "timestamp", "timestamp", "YES", "now"),
    ("liquidations", 3, "symbol", "text", "YES", "none"),
    ("liquidations", 4, "entry_price", "float4", "YES", "none"),
    ("liquidations", 5, "liquidation_price", "float4", "YES", "none"),
    ("liquidations", 6, "quantity", "float4", "YES", "none"),
    ("liquidations", 7, "leverage", "float4", "YES", "none"),
    ("liquidations", 8, "liquidation_fee", "float4", "YES", "none"),
    ("margin_history", 1, "id", "int4", "NO", "serial"),
    ("margin_history", 2, "timestamp", "timestamp", "YES", "now"),
    ("margin_history", 3, "balance", "float4", "YES", "none"),
    ("margin_history", 4, "used_margin", "float4", "YES", "none"),
    ("margin_history", 5, "open_positions", "int4", "YES", "none"),
    ("model_predictions", 1, "id", "int4", "NO", "serial"),
    ("model_predictions", 2, "created_at", "timestamp", "YES", "now"),
    ("model_predictions", 3, "symbol", "text", "YES", "none"),
    ("model_predictions", 4, "side", "text", "YES", "none"),
    ("model_predictions", 5, "model", "text", "YES", "none"),
    ("model_predictions", 6, "vote", "text", "YES", "none"),
    ("model_predictions", 7, "scored", "bool", "YES", "false"),
    ("open_positions", 1, "id", "int4", "NO", "serial"),
    ("open_positions", 2, "symbol", "text", "YES", "none"),
    ("open_positions", 3, "side", "text", "YES", "none"),
    ("open_positions", 4, "entry_price", "float4", "YES", "none"),
    ("open_positions", 5, "quantity", "float4", "YES", "none"),
    ("open_positions", 6, "leverage", "float4", "YES", "none"),
    ("open_positions", 7, "entry_time", "timestamp", "YES", "now"),
    ("trades", 1, "id", "int4", "NO", "serial"),
    ("trades", 2, "timestamp", "timestamp", "YES", "now"),
    ("trades", 3, "symbol", "text", "YES", "none"),
    ("trades", 4, "side", "text", "YES", "none"),
    ("trades", 5, "price", "float4", "YES", "none"),
    ("trades", 6, "quantity", "float4", "YES", "none"),
    ("trades", 7, "pnl", "float4", "YES", "none"),
    ("trades", 8, "fee", "float4", "YES", "none"),
    ("trading_session_resets", 1, "id", "int4", "NO", "serial"),
    ("trading_session_resets", 2, "reset_timestamp", "timestamp", "YES", "now"),
    ("trading_session_resets", 3, "reason", "text", "YES", "none"),
)
_EXPECTED_LEGACY_COLUMNS = tuple(
    row
    + (
        {
            "int4": "integer",
            "float4": "real",
            "text": "text",
            "timestamp": "timestamp without time zone",
            "bool": "boolean",
        }[row[3]],
        '"default"' if row[3] == "text" else "",
    )
    for row in _EXPECTED_LEGACY_COLUMNS
)

_SELECT_IDENTITY_SQL = """
SELECT
    current_database(),
    current_user,
    session_user,
    (SELECT usename FROM pg_stat_activity WHERE pid = pg_backend_pid()),
    (SELECT system_identifier FROM pg_control_system())
"""
_SELECT_OTHER_SESSIONS_SQL = """
SELECT COUNT(*)::bigint
FROM pg_stat_activity
WHERE datname = current_database()
  AND pid <> pg_backend_pid()
"""
_SELECT_LEGACY_COLUMNS_SQL = """
SELECT
    table_row.relname,
    column_row.attnum,
    column_row.attname,
    column_type.typname,
    CASE WHEN column_row.attnotnull THEN 'NO' ELSE 'YES' END,
    CASE
        WHEN default_row.adbin IS NULL THEN 'none'
        WHEN column_row.attname = 'id' AND pg_get_expr(
            default_row.adbin,
            default_row.adrelid
        ) = FORMAT(
            'nextval(%%L::regclass)',
            pg_get_serial_sequence(
                FORMAT('%%I.%%I', namespace_row.nspname, table_row.relname),
                'id'
            )
        ) THEN 'serial'
        WHEN LOWER(pg_get_expr(default_row.adbin, default_row.adrelid))
             IN ('now()', 'current_timestamp') THEN 'now'
        WHEN LOWER(pg_get_expr(default_row.adbin, default_row.adrelid)) = 'false'
            THEN 'false'
        WHEN pg_get_expr(default_row.adbin, default_row.adrelid) = '0' THEN 'zero'
        ELSE 'other'
    END,
    format_type(column_row.atttypid, column_row.atttypmod),
    CASE
        WHEN column_row.attcollation = 0 THEN ''
        ELSE column_row.attcollation::regcollation::text
    END
FROM pg_class table_row
JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
JOIN pg_attribute column_row ON column_row.attrelid = table_row.oid
JOIN pg_type column_type ON column_type.oid = column_row.atttypid
LEFT JOIN pg_attrdef default_row
  ON default_row.adrelid = table_row.oid
 AND default_row.adnum = column_row.attnum
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = ANY(%s)
  AND column_row.attnum > 0
  AND NOT column_row.attisdropped
ORDER BY table_row.relname, column_row.attnum
"""
_SELECT_LEGACY_RELATIONS_SQL = """
SELECT
    table_row.relname,
    table_row.relkind,
    table_row.relpersistence,
    table_row.relhasrules,
    table_row.relrowsecurity,
    table_row.relforcerowsecurity,
    EXISTS (
        SELECT 1 FROM pg_inherits inheritance
        WHERE inheritance.inhrelid = table_row.oid
           OR inheritance.inhparent = table_row.oid
    ),
    EXISTS (SELECT 1 FROM pg_policy policy WHERE policy.polrelid = table_row.oid)
FROM pg_class table_row
JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = ANY(%s)
ORDER BY table_row.relname
"""
_SELECT_LEGACY_CONSTRAINTS_SQL = """
SELECT
    table_row.relname,
    constraint_row.contype,
    constraint_row.conkey::smallint[],
    constraint_row.condeferrable,
    constraint_row.condeferred,
    constraint_row.convalidated
FROM pg_constraint constraint_row
JOIN pg_class table_row ON table_row.oid = constraint_row.conrelid
JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = ANY(%s)
ORDER BY table_row.relname, constraint_row.contype, constraint_row.conname
"""
_SELECT_LEGACY_ROOT_SQL = """
SELECT database_owner.rolname, schema_owner.rolname
FROM pg_database database_row
JOIN pg_roles database_owner ON database_owner.oid = database_row.datdba
JOIN pg_namespace namespace_row ON namespace_row.nspname = 'np'
JOIN pg_roles schema_owner ON schema_owner.oid = namespace_row.nspowner
WHERE database_row.datname = current_database()
"""
_SELECT_NP_OBJECTS_SQL = """
SELECT object_row.relname, object_row.relkind, owner_row.rolname
FROM pg_class object_row
JOIN pg_namespace namespace_row ON namespace_row.oid = object_row.relnamespace
JOIN pg_roles owner_row ON owner_row.oid = object_row.relowner
WHERE namespace_row.nspname = 'np'
ORDER BY object_row.relkind, object_row.relname
"""
_SELECT_LEGACY_SEQUENCES_SQL = """
SELECT
    sequence_row.relname,
    format_type(sequence_catalog.seqtypid, NULL),
    sequence_catalog.seqstart,
    sequence_catalog.seqincrement,
    sequence_catalog.seqmax,
    sequence_catalog.seqmin,
    sequence_catalog.seqcache,
    sequence_catalog.seqcycle,
    table_row.relname,
    attribute_row.attname
FROM pg_class sequence_row
JOIN pg_namespace namespace_row ON namespace_row.oid = sequence_row.relnamespace
JOIN pg_sequence sequence_catalog ON sequence_catalog.seqrelid = sequence_row.oid
JOIN pg_depend dependency_row
  ON dependency_row.objid = sequence_row.oid
 AND dependency_row.classid = 'pg_class'::regclass
 AND dependency_row.refclassid = 'pg_class'::regclass
 AND dependency_row.deptype = 'a'
JOIN pg_class table_row ON table_row.oid = dependency_row.refobjid
JOIN pg_attribute attribute_row
  ON attribute_row.attrelid = table_row.oid
 AND attribute_row.attnum = dependency_row.refobjsubid
WHERE namespace_row.nspname = 'np'
ORDER BY sequence_row.relname
"""
_SELECT_NP_ROUTINES_TYPES_SQL = """
SELECT
    EXISTS (
        SELECT 1 FROM pg_proc routine
        JOIN pg_namespace namespace_row ON namespace_row.oid = routine.pronamespace
        WHERE namespace_row.nspname = 'np'
    ),
    EXISTS (
        SELECT 1 FROM pg_type type_row
        JOIN pg_namespace namespace_row ON namespace_row.oid = type_row.typnamespace
        WHERE namespace_row.nspname = 'np'
          AND type_row.typrelid = 0
          AND type_row.typelem = 0
    ),
    EXISTS (
        SELECT 1 FROM pg_trigger trigger_row
        JOIN pg_class table_row ON table_row.oid = trigger_row.tgrelid
        JOIN pg_namespace namespace_row ON namespace_row.oid = table_row.relnamespace
        WHERE namespace_row.nspname = 'np'
          AND NOT trigger_row.tgisinternal
    )
"""
_SELECT_NP_ACLS_SQL = """
SELECT EXISTS (
    SELECT 1
    FROM pg_namespace namespace_row
    WHERE namespace_row.nspname = 'np'
      AND namespace_row.nspacl IS NOT NULL
    UNION ALL
    SELECT 1
    FROM pg_class relation_row
    JOIN pg_namespace namespace_row ON namespace_row.oid = relation_row.relnamespace
    WHERE namespace_row.nspname = 'np'
      AND relation_row.relacl IS NOT NULL
    UNION ALL
    SELECT 1
    FROM pg_default_acl default_acl
    LEFT JOIN pg_namespace namespace_row
      ON namespace_row.oid = default_acl.defaclnamespace
    WHERE default_acl.defaclnamespace = 0
       OR namespace_row.nspname = 'np'
    UNION ALL
    SELECT 1
    FROM pg_attribute attribute_row
    JOIN pg_class relation_row ON relation_row.oid = attribute_row.attrelid
    JOIN pg_namespace namespace_row ON namespace_row.oid = relation_row.relnamespace
    WHERE namespace_row.nspname = 'np'
      AND attribute_row.attacl IS NOT NULL
    UNION ALL
    SELECT 1
    FROM pg_proc routine
    JOIN pg_namespace namespace_row ON namespace_row.oid = routine.pronamespace
    WHERE namespace_row.nspname = 'np'
      AND routine.proacl IS NOT NULL
)
"""


class PostgresCutoverPreflightInputError(ValueError):
    """Raised before inspection when caller input is unsafe."""


class PostgresCutoverPreflightStorageError(RuntimeError):
    """Raised without driver or connection detail when an inspection fails."""


def _close_quietly(connection: object) -> None:
    try:
        connection.close()
    except Exception:
        pass


def _rollback_quietly(connection: object) -> None:
    try:
        connection.rollback()
    except Exception:
        pass


def _fresh_connection(factory: Callable[[], object], label: str) -> object:
    connection = None
    failed = False
    try:
        connection = factory()
        interface_exact = all(
            callable(getattr(connection, name, None))
            for name in ("cursor", "rollback", "close", "get_transaction_status")
        )
        exact = (
            interface_exact
            and getattr(connection, "autocommit", None) is False
            and getattr(connection, "status", None) == STATUS_READY
            and connection.get_transaction_status() == TRANSACTION_STATUS_IDLE
        )
    except Exception:
        failed = True
        exact = False
    if failed:
        raise PostgresCutoverPreflightStorageError(
            f"could not open the {label} inspection connection"
        )
    if not exact:
        _close_quietly(connection)
        raise PostgresCutoverPreflightStorageError(
            f"the {label} inspection connection is not fresh and idle"
        )
    return connection


def _one_row(value: object, length: int) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise PostgresCutoverPreflightStorageError(
            "PostgreSQL returned invalid preflight evidence"
        )
    return tuple(value)


def _tagged_value(value: object) -> list[object]:
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is Decimal:
        return ["decimal", str(value)]
    if type(value) is str:
        return ["text", value]
    if isinstance(value, dt.datetime):
        return ["timestamp", value.isoformat(timespec="microseconds")]
    if isinstance(value, dt.date):
        return ["date", value.isoformat()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    raise PostgresCutoverPreflightStorageError(
        "PostgreSQL returned an unsupported preflight value"
    )


def _canonical_row(row: tuple[object, ...]) -> bytes:
    return json.dumps(
        [_tagged_value(value) for value in row],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _row_is_semantically_valid(relation: str, row: tuple[object, ...]) -> bool:
    if not row or type(row[0]) is not int or row[0] <= 0:
        return False
    if any(type(value) is float and not math.isfinite(value) for value in row):
        return False
    text_columns = {
        "np.account_balances": (1,),
        "np.liquidations": (2,),
        "np.model_predictions": (2, 3, 4, 5),
        "np.open_positions": (1, 2),
        "np.trades": (2, 3),
        "np.trading_session_resets": (2,),
    }.get(relation, ())
    if any(
        type(row[index]) is not str
        or not row[index]
        or row[index] != row[index].strip()
        for index in text_columns
    ):
        return False
    timestamp_columns = {
        "np.account_balances": (3,),
        "np.liquidations": (1,),
        "np.margin_history": (1,),
        "np.model_predictions": (1,),
        "np.open_positions": (6,),
        "np.trades": (1,),
        "np.trading_session_resets": (1,),
    }[relation]
    if any(not isinstance(row[index], dt.datetime) for index in timestamp_columns):
        return False
    if relation == "np.account_balances":
        return row[2] >= 0
    if relation == "np.margin_history":
        return (
            row[2] is not None
            and row[2] >= 0
            and row[3] is not None
            and row[3] >= 0
            and type(row[4]) is int
            and row[4] >= 0
        )
    if relation == "np.model_predictions":
        return (
            row[3] in ("BUY", "SELL")
            and row[5] in ("BUY", "SELL", "HOLD")
            and type(row[6]) is bool
        )
    if relation == "np.open_positions" and row[2] not in ("BUY", "SELL"):
        return False
    if relation == "np.trades" and row[3] not in ("BUY", "SELL", "TEST"):
        return False
    positive_indexes = {
        "np.liquidations": (3, 4, 5, 6),
        "np.open_positions": (3, 4, 5),
        "np.trades": (4, 5),
    }.get(relation, ())
    if any(row[index] is None or row[index] <= 0 for index in positive_indexes):
        return False
    if relation == "np.liquidations" and (row[7] is None or row[7] < 0):
        return False
    if relation == "np.trades" and (row[6] is None or row[7] is None or row[7] < 0):
        return False
    return True


def _legacy_layout_is_exact(
    cursor: object,
    expected_owner: str,
) -> bool:
    table_names = list(_LEGACY_TABLE_NAMES)
    cursor.execute(_SELECT_LEGACY_ROOT_SQL)
    if tuple(tuple(row) for row in cursor.fetchall()) != (
        (expected_owner, expected_owner),
    ):
        return False
    cursor.execute(_SELECT_NP_OBJECTS_SQL)
    expected_objects = tuple(
        sorted(
            tuple((name, "r", expected_owner) for name in _LEGACY_TABLE_NAMES)
            + tuple((name, "S", expected_owner) for name in _LEGACY_SEQUENCE_NAMES)
            + tuple((name, "i", expected_owner) for name in _LEGACY_INDEX_NAMES),
            key=lambda row: (row[1], row[0]),
        )
    )
    if tuple(tuple(row) for row in cursor.fetchall()) != expected_objects:
        return False
    cursor.execute(_SELECT_LEGACY_SEQUENCES_SQL)
    expected_sequences = tuple(
        (
            sequence,
            "integer",
            1,
            1,
            2147483647,
            1,
            1,
            False,
            sequence.removesuffix("_id_seq"),
            "id",
        )
        for sequence in _LEGACY_SEQUENCE_NAMES
    )
    if tuple(tuple(row) for row in cursor.fetchall()) != expected_sequences:
        return False
    cursor.execute(_SELECT_INDEX_EVIDENCE_SQL)
    indexes = tuple(tuple(row) for row in cursor.fetchall())
    if tuple(row[0] for row in indexes) != _LEGACY_INDEX_NAMES:
        return False
    for row in indexes:
        if len(row) != 24:
            return False
        name, table = row[:2]
        expected_key = {
            "account_balances_asset_key": "[0:0]={2}",
            "idx_model_predictions_scored": "[0:1]={7,2}",
            "idx_trades_symbol_ts": "[0:1]={3,2}",
        }.get(name, "[0:0]={1}")
        expected_unique = name not in (
            "idx_model_predictions_scored",
            "idx_trades_symbol_ts",
        )
        expected_primary = name.endswith("_pkey")
        expected_width = 2 if name.startswith("idx_") else 1
        expected_opclasses = {
            "account_balances_asset_key": "{pg_catalog.text_ops}",
            "idx_model_predictions_scored": (
                "{pg_catalog.bool_ops,pg_catalog.timestamp_ops}"
            ),
            "idx_trades_symbol_ts": ("{pg_catalog.text_ops,pg_catalog.timestamp_ops}"),
        }.get(name, "{pg_catalog.int4_ops}")
        expected_collations = {
            "account_balances_asset_key": '{"pg_catalog.\\"default\\""}',
            "idx_model_predictions_scored": '{"",""}',
            "idx_trades_symbol_ts": '{"pg_catalog.\\"default\\"",""}',
        }.get(name, '{""}')
        expected_options = "[0:1]={0,0}" if expected_width == 2 else "[0:0]={0}"
        expected = (
            name,
            table,
            "btree",
            expected_unique,
            expected_primary,
            True,
            True,
            expected_width,
            expected_width,
            expected_key,
            "",
            "",
            "p",
            True,
            False,
            False,
            False,
            expected_opclasses,
            expected_collations,
            expected_options,
            True,
            True,
            True,
            True,
        )
        if row != expected:
            return False
    cursor.execute(_SELECT_NP_ROUTINES_TYPES_SQL)
    if _one_row(cursor.fetchone(), 3) != (False, False, False):
        return False
    cursor.execute(
        _SELECT_UNEXPECTED_NP_CATALOG_OBJECTS_SQL,
        (list(_LEGACY_TABLE_NAMES),),
    )
    if cursor.fetchall():
        return False
    cursor.execute(_SELECT_NP_ACLS_SQL)
    if _one_row(cursor.fetchone(), 1) != (False,):
        return False
    cursor.execute(_SELECT_LEGACY_RELATIONS_SQL, (table_names,))
    relations = tuple(tuple(row) for row in cursor.fetchall())
    expected_relations = tuple(
        (name, "r", "p", False, False, False, False, False) for name in table_names
    )
    if relations != expected_relations:
        return False
    cursor.execute(_SELECT_LEGACY_COLUMNS_SQL, (table_names,))
    if tuple(tuple(row) for row in cursor.fetchall()) != _EXPECTED_LEGACY_COLUMNS:
        return False
    cursor.execute(_SELECT_LEGACY_CONSTRAINTS_SQL, (table_names,))
    constraints = tuple(tuple(row) for row in cursor.fetchall())
    expected_constraints = []
    for name in table_names:
        expected_constraints.append((name, "p", [1], False, False, True))
        if name == "account_balances":
            expected_constraints.insert(-1, (name, "u", [2], False, False, True))
    expected_constraints.sort(key=lambda row: (row[0], row[1]))
    return constraints == tuple(expected_constraints)


def _stream_relation(
    connection: object,
    relation: str,
) -> tuple[FreshTargetRelationEvidence, int]:
    table = relation.removeprefix("np.")
    digest = hashlib.sha256()
    row_count = 0
    primary_key_min = None
    primary_key_max = None
    invalid_count = 0
    failed = False
    try:
        cursor = connection.cursor(name=f"elvis_preflight_{table}")
        cursor.itersize = _FETCH_BATCH_SIZE
        cursor.execute(f'SELECT * FROM np."{table}" ORDER BY id')
        while True:
            rows = cursor.fetchmany(_FETCH_BATCH_SIZE)
            if not rows:
                break
            for raw_row in rows:
                row = tuple(raw_row)
                encoded = _canonical_row(row)
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                row_count += 1
                primary_key = row[0] if row else None
                if type(primary_key) is int:
                    if primary_key_min is None:
                        primary_key_min = primary_key
                    primary_key_max = primary_key
                if not _row_is_semantically_valid(relation, row):
                    invalid_count += 1
        cursor.close()
    except Exception:
        failed = True
    if failed:
        raise PostgresCutoverPreflightStorageError("source relation inspection failed")
    return (
        FreshTargetRelationEvidence(
            name=relation,
            row_count=row_count,
            pk_min=primary_key_min,
            pk_max=primary_key_max,
            sha256=digest.hexdigest(),
        ),
        invalid_count,
    )


class PostgresCutoverPreflight:
    """Inspect one stopped source clone and one different fresh target."""

    def __init__(
        self,
        source_connection_factory: Callable[[], object],
        target_connection_factory: Callable[[], object],
    ) -> None:
        if not callable(source_connection_factory) or not callable(
            target_connection_factory
        ):
            raise PostgresCutoverPreflightInputError(
                "source and target connection factories must be callable"
            )
        if source_connection_factory is target_connection_factory:
            raise PostgresCutoverPreflightInputError(
                "source and target connection factories must be distinct"
            )
        self._source_connection_factory = source_connection_factory
        self._target_connection_factory = target_connection_factory

    def inspect(
        self, context: FreshTargetCutoverContext, /
    ) -> FreshTargetCutoverReceipt:
        """Return read-only, stale-on-return evidence from both databases."""
        if type(context) is not FreshTargetCutoverContext:
            raise PostgresCutoverPreflightInputError(
                "context must be a FreshTargetCutoverContext"
            )
        storage_failed = False
        try:
            source = self._inspect_source(context)
            target = self._inspect_target(context)
        except (
            PostgresCutoverPreflightStorageError,
            PostgresBootstrapStorageError,
        ):
            storage_failed = True
        if storage_failed:
            raise PostgresCutoverPreflightStorageError(
                "PostgreSQL cut-over preflight storage inspection failed"
            )
        blockers = []
        if not source.identity_exact:
            blockers.append(FreshTargetCutoverBlocker.SOURCE_IDENTITY)
        if not source.legacy_layout_exact:
            blockers.append(FreshTargetCutoverBlocker.SOURCE_SCHEMA)
        if source.other_session_count:
            blockers.append(FreshTargetCutoverBlocker.SOURCE_ACTIVE_SESSIONS)
        if source.open_position_count:
            blockers.append(FreshTargetCutoverBlocker.SOURCE_OPEN_POSITIONS)
        if source.semantic_invalid_row_count:
            blockers.append(FreshTargetCutoverBlocker.SOURCE_DATA_QUALITY)
        if source.system_identifier == target.system_identifier:
            blockers.append(FreshTargetCutoverBlocker.SAME_CLUSTER)
        if (
            not target.terminal_catalog_exact
            or target.migration_versions != _EXPECTED_MIGRATION_VERSIONS
        ):
            blockers.append(FreshTargetCutoverBlocker.TARGET_NOT_COMPLETE)
        if target.runtime_mode != "LEGACY" or target.runtime_generation != 0:
            blockers.append(FreshTargetCutoverBlocker.TARGET_MODE)
        if target.nonempty_relations:
            blockers.append(FreshTargetCutoverBlocker.TARGET_NOT_EMPTY)
        ordered = tuple(sorted(set(blockers), key=lambda value: value.value))
        status = (
            FreshTargetCutoverStatus.BLOCKED
            if ordered
            else FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET
        )
        return FreshTargetCutoverReceipt(status, ordered, source, target)

    def _inspect_source(
        self, context: FreshTargetCutoverContext
    ) -> FreshTargetCutoverSourceEvidence:
        connection = _fresh_connection(self._source_connection_factory, "source")
        result = None
        storage_failed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_ONLY_SQL)
                    cursor.execute(_UTC_SQL)
                    cursor.execute(_SEARCH_PATH_SQL)
                    cursor.execute(_SELECT_IDENTITY_SQL)
                    identity = _one_row(cursor.fetchone(), 5)
                    if type(identity[4]) is not int or identity[4] <= 0:
                        raise PostgresCutoverPreflightStorageError(
                            "PostgreSQL returned invalid cluster evidence"
                        )
                    identity_exact = identity[:4] == (
                        context.source_expected_database,
                        context.source_expected_role,
                        context.source_expected_role,
                        context.source_expected_role,
                    )
                    cursor.execute(_SELECT_OTHER_SESSIONS_SQL)
                    other_sessions = _one_row(cursor.fetchone(), 1)[0]
                    if type(other_sessions) is not int or other_sessions < 0:
                        raise PostgresCutoverPreflightStorageError(
                            "PostgreSQL returned invalid session evidence"
                        )
                    layout_exact = identity_exact and _legacy_layout_is_exact(
                        cursor, context.source_expected_role
                    )
                relations = []
                invalid_count = 0
                if layout_exact:
                    for relation in _LEGACY_RELATIONS:
                        evidence, invalid = _stream_relation(connection, relation)
                        relations.append(evidence)
                        invalid_count += invalid
                combined = None
                if layout_exact:
                    encoded = json.dumps(
                        [
                            {
                                "name": relation.name,
                                "pk_max": relation.pk_max,
                                "pk_min": relation.pk_min,
                                "row_count": relation.row_count,
                                "sha256": relation.sha256,
                            }
                            for relation in relations
                        ],
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                    combined = hashlib.sha256(encoded).hexdigest()
                open_positions = next(
                    (
                        relation.row_count
                        for relation in relations
                        if relation.name == "np.open_positions"
                    ),
                    0,
                )
                result = FreshTargetCutoverSourceEvidence(
                    system_identifier=identity[4],
                    relations=tuple(relations),
                    other_session_count=other_sessions,
                    open_position_count=open_positions,
                    semantic_invalid_row_count=invalid_count,
                    canonical_sha256=combined,
                    legacy_layout_exact=layout_exact,
                    identity_exact=identity_exact,
                )
            except Exception:
                storage_failed = True
        finally:
            _rollback_quietly(connection)
            _close_quietly(connection)
        if storage_failed or result is None:
            raise PostgresCutoverPreflightStorageError(
                "source PostgreSQL inspection failed"
            )
        return result

    def _inspect_target(
        self, context: FreshTargetCutoverContext
    ) -> FreshTargetCutoverTargetEvidence:
        intent = context.target_bootstrap_intent
        if type(intent) is not FreshTargetBootstrapIntent:
            raise PostgresCutoverPreflightInputError(
                "target bootstrap intent is invalid"
            )
        roles = intent.roles
        bootstrap_context = PostgresBootstrapContext(
            expected_database=intent.expected_database,
            admin_role=intent.admin_role,
            roles=PostgresBootstrapRoles(
                schema_owner=roles.schema_owner,
                migrator=roles.migrator,
                legacy_runtime=roles.legacy_runtime,
                atomic_runtime=roles.atomic_runtime,
                activation=roles.activation,
                readiness=roles.readiness,
                trainer=roles.trainer,
            ),
            adoption=None,
        )
        try:
            inspection = PostgresBootstrap(
                self._target_connection_factory
            ).inspect_terminal(bootstrap_context)
        except PostgresBootstrapStorageError:
            storage_failed = True
        else:
            storage_failed = False
        if storage_failed:
            raise PostgresCutoverPreflightStorageError(
                "target PostgreSQL inspection failed"
            )
        return FreshTargetCutoverTargetEvidence(
            system_identifier=inspection.system_identifier,
            terminal_catalog_exact=inspection.exact,
            migration_versions=inspection.migration_versions,
            runtime_mode=inspection.runtime_mode,
            runtime_generation=inspection.runtime_generation,
            nonempty_relations=inspection.nonempty_relations,
        )


__all__ = [
    "PostgresCutoverPreflight",
    "PostgresCutoverPreflightInputError",
    "PostgresCutoverPreflightStorageError",
]
