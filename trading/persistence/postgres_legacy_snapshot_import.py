"""Bounded, replay-safe PostgreSQL import of the exact legacy V1 snapshot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from psycopg2 import sql

from trading.application.fresh_target_cutover import (
    FreshTargetCutoverReceipt,
    FreshTargetCutoverStatus,
    FreshTargetRelationEvidence,
)
from trading.application.legacy_snapshot_import import (
    LegacySnapshotImportContext,
    LegacySnapshotImportDisposition,
    LegacySnapshotImportReceipt,
    LegacySnapshotRelationReceipt,
)
from trading.persistence.postgres_bootstrap import (
    PostgresBootstrap,
    PostgresBootstrapContext,
    PostgresBootstrapDriftError,
    PostgresBootstrapMigrationError,
    PostgresBootstrapRoles,
    PostgresBootstrapStorageError,
)
from trading.persistence.postgres_cutover_preflight import (
    _LEGACY_RELATIONS,
    _READ_ONLY_SQL,
    _SEARCH_PATH_SQL,
    _SELECT_IDENTITY_SQL,
    _SELECT_OTHER_SESSIONS_SQL,
    _UTC_SQL,
    PostgresCutoverPreflightStorageError,
    _canonical_row,
    _fresh_connection,
    _legacy_layout_is_exact,
    _one_row,
    _rollback_quietly,
    _row_is_semantically_valid,
)

_MAX_TOTAL_ROWS = 100_000
_MAX_CANONICAL_BYTES = 512 * 1024 * 1024
_MAX_CANONICAL_ROW_BYTES = 64 * 1024
_POSTGRES_INTEGER_MAX = (1 << 31) - 1
_HISTORICAL_HEAD6_AUTHORITY_TABLES = (
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
# Each target validation SELECT must acquire its snapshot after the SHARE locks.
# READ COMMITTED prevents the pre-lock identity query from pinning stale rows.
_TARGET_ISOLATION_SQL = "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
_LOCK_TIMEOUT_SQL = "SET LOCAL lock_timeout = '1s'"
_LOCK_IMPORT_TABLES_SQL = (
    "LOCK TABLE "
    + ", ".join(f"ONLY np.{table}" for table in _HISTORICAL_HEAD6_AUTHORITY_TABLES)
    + " IN SHARE MODE NOWAIT"
)
_SELECT_MIGRATOR_TARGET_IDENTITY_SQL = """
SELECT
    current_database(),
    session_user,
    current_user,
    (SELECT usename FROM pg_stat_activity WHERE pid = pg_backend_pid()),
    (SELECT system_identifier FROM pg_control_system())
"""
_LEGACY_COLUMNS = {
    "np.account_balances": ("id", "asset", "balance", "last_updated"),
    "np.liquidations": (
        "id",
        "timestamp",
        "symbol",
        "entry_price",
        "liquidation_price",
        "quantity",
        "leverage",
        "liquidation_fee",
    ),
    "np.margin_history": (
        "id",
        "timestamp",
        "balance",
        "used_margin",
        "open_positions",
    ),
    "np.model_predictions": (
        "id",
        "created_at",
        "symbol",
        "side",
        "model",
        "vote",
        "scored",
    ),
    "np.open_positions": (
        "id",
        "symbol",
        "side",
        "entry_price",
        "quantity",
        "leverage",
        "entry_time",
    ),
    "np.trades": (
        "id",
        "timestamp",
        "symbol",
        "side",
        "price",
        "quantity",
        "pnl",
        "fee",
    ),
    "np.trading_session_resets": ("id", "reset_timestamp", "reason"),
}
_LEGACY_REAL_COLUMNS = {
    "np.account_balances": frozenset({"balance"}),
    "np.liquidations": frozenset(
        {
            "entry_price",
            "liquidation_price",
            "quantity",
            "leverage",
            "liquidation_fee",
        }
    ),
    "np.margin_history": frozenset({"balance", "used_margin"}),
    "np.model_predictions": frozenset(),
    "np.open_positions": frozenset({"entry_price", "quantity", "leverage"}),
    "np.trades": frozenset({"price", "quantity", "pnl", "fee"}),
    "np.trading_session_resets": frozenset(),
}
_V2_ONLY_TABLES = (
    "order_events",
    "orders",
    "paper_account_balances",
    "paper_account_batch_manifests",
    "paper_account_postings",
    "paper_account_settlements",
    "paper_account_streams",
    "paper_margin_reservations",
    "paper_runtime_generations",
    "position_streams",
)


class PostgresLegacySnapshotImportInputError(ValueError):
    """Raised before database access when an import intent is unsafe."""


class PostgresLegacySnapshotImportBusyError(RuntimeError):
    """Raised when the declared exclusive database window is not available."""


class PostgresLegacySnapshotImportConflict(RuntimeError):
    """Raised without repair when source or target evidence is not exact."""


class PostgresLegacySnapshotImportStorageError(RuntimeError):
    """Raised without driver detail when PostgreSQL access fails."""


class PostgresLegacySnapshotImportCommitUnknown(RuntimeError):
    """Raised when row-commit durability cannot be established by readback."""


def _close_quietly(connection: object) -> None:
    try:
        connection.close()
    except Exception:
        pass


def _bootstrap_context(
    context: LegacySnapshotImportContext,
) -> PostgresBootstrapContext:
    intent = context.cutover_context.target_bootstrap_intent
    roles = intent.roles
    return PostgresBootstrapContext(
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


def _combined_sha256(relations: tuple[FreshTargetRelationEvidence, ...]) -> str:
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
    return hashlib.sha256(encoded).hexdigest()


def _sequence_name(relation: str) -> str:
    return f"{relation.removeprefix('np.')}_id_seq"


def _scan_relation(
    connection: object,
    relation: str,
    batch_size: int,
    pass_name: str,
    consume: Callable[[tuple[tuple[object, ...], ...]], None] | None = None,
    *,
    remaining_rows: int = _MAX_TOTAL_ROWS,
    remaining_bytes: int = _MAX_CANONICAL_BYTES,
) -> tuple[FreshTargetRelationEvidence, int, int]:
    table = relation.removeprefix("np.")
    digest = hashlib.sha256()
    count = 0
    minimum = None
    maximum = None
    invalid = 0
    byte_count = 0
    cursor = connection.cursor(name=f"elvis_{pass_name}_{table}")
    try:
        cursor.itersize = batch_size
        cursor.execute(
            sql.SQL("SELECT * FROM {} ORDER BY id").format(sql.Identifier("np", table))
        )
        while True:
            raw_rows = cursor.fetchmany(batch_size)
            if not raw_rows:
                break
            rows = tuple(tuple(row) for row in raw_rows)
            for row in rows:
                encoded = _canonical_row(row)
                if len(encoded) > _MAX_CANONICAL_ROW_BYTES:
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot row exceeds the compiled import bound"
                    )
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                byte_count += len(encoded) + 8
                count += 1
                if (
                    count > _MAX_TOTAL_ROWS
                    or byte_count > _MAX_CANONICAL_BYTES
                    or count > remaining_rows
                    or byte_count > remaining_bytes
                ):
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot exceeds the compiled import bound"
                    )
                key = row[0] if row else None
                if type(key) is int:
                    if minimum is None:
                        minimum = key
                    maximum = key
                if not _row_is_semantically_valid(relation, row):
                    invalid += 1
            if consume is not None:
                consume(rows)
    finally:
        try:
            cursor.close()
        except Exception:
            pass
    return (
        FreshTargetRelationEvidence(
            name=relation,
            row_count=count,
            pk_min=minimum,
            pk_max=maximum,
            sha256=digest.hexdigest(),
        ),
        invalid,
        byte_count,
    )


def _scan_relations(
    connection: object,
    batch_size: int,
    pass_name: str,
    consumers: (
        dict[str, Callable[[tuple[tuple[object, ...], ...]], None]] | None
    ) = None,
) -> tuple[tuple[FreshTargetRelationEvidence, ...], int, int]:
    relations = []
    invalid = 0
    total_bytes = 0
    total_rows = 0
    for relation in _LEGACY_RELATIONS:
        evidence, relation_invalid, relation_bytes = _scan_relation(
            connection,
            relation,
            batch_size,
            pass_name,
            None if consumers is None else consumers.get(relation),
            remaining_rows=_MAX_TOTAL_ROWS - total_rows,
            remaining_bytes=_MAX_CANONICAL_BYTES - total_bytes,
        )
        relations.append(evidence)
        invalid += relation_invalid
        total_bytes += relation_bytes
        total_rows += evidence.row_count
        if total_rows > _MAX_TOTAL_ROWS or total_bytes > _MAX_CANONICAL_BYTES:
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot exceeds the compiled import bound"
            )
    return tuple(relations), invalid, total_bytes


def _read_sequence_next(cursor: object, relation: str) -> int:
    cursor.execute(
        sql.SQL("SELECT last_value, is_called FROM {}").format(
            sql.Identifier("np", _sequence_name(relation))
        )
    )
    last_value, is_called = _one_row(cursor.fetchone(), 2)
    if type(last_value) is not int or type(is_called) is not bool:
        raise PostgresLegacySnapshotImportStorageError(
            "PostgreSQL returned invalid sequence evidence"
        )
    next_value = last_value + 1 if is_called else last_value
    if not 1 <= next_value <= _POSTGRES_INTEGER_MAX:
        raise PostgresLegacySnapshotImportConflict(
            "legacy source sequence is outside the safe integer range"
        )
    return next_value


class PostgresLegacySnapshotImport:
    """Copy an admitted raw V1 snapshot without synthesizing V2 facts."""

    def __init__(
        self,
        source_connection_factory: Callable[[], object],
        target_admin_connection_factory: Callable[[], object],
        target_migrator_connection_factory: Callable[[], object],
    ) -> None:
        factories = (
            source_connection_factory,
            target_admin_connection_factory,
            target_migrator_connection_factory,
        )
        if any(not callable(factory) for factory in factories):
            raise PostgresLegacySnapshotImportInputError(
                "all snapshot import connection factories must be callable"
            )
        if len({id(factory) for factory in factories}) != 3:
            raise PostgresLegacySnapshotImportInputError(
                "snapshot import connection factories must be distinct"
            )
        self._source_connection_factory = source_connection_factory
        self._target_admin_connection_factory = target_admin_connection_factory
        self._target_migrator_connection_factory = target_migrator_connection_factory

    def import_snapshot(
        self,
        context: LegacySnapshotImportContext,
        preflight_receipt: FreshTargetCutoverReceipt,
        /,
    ) -> LegacySnapshotImportReceipt:
        if type(context) is not LegacySnapshotImportContext:
            raise PostgresLegacySnapshotImportInputError(
                "context must be a LegacySnapshotImportContext"
            )
        if (
            type(preflight_receipt) is not FreshTargetCutoverReceipt
            or preflight_receipt.status
            is not FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET
            or preflight_receipt.blockers
        ):
            raise PostgresLegacySnapshotImportInputError(
                "preflight receipt must be an exact READY receipt"
            )
        try:
            return self._import_snapshot(context, preflight_receipt)
        except (
            PostgresLegacySnapshotImportBusyError,
            PostgresLegacySnapshotImportCommitUnknown,
            PostgresLegacySnapshotImportConflict,
            PostgresLegacySnapshotImportInputError,
            PostgresLegacySnapshotImportStorageError,
        ):
            raise
        except Exception:
            failed = True
        if failed:
            raise PostgresLegacySnapshotImportStorageError(
                "PostgreSQL legacy snapshot import failed"
            ) from None
        raise AssertionError("unreachable")

    def _import_snapshot(
        self,
        context: LegacySnapshotImportContext,
        preflight: FreshTargetCutoverReceipt,
    ) -> LegacySnapshotImportReceipt:
        source = self._open_source(context, preflight)
        try:
            relations, sequences = self._admit_source(source, context, preflight)
            desired_sequences = tuple(
                max(
                    sequences[index],
                    1 if relation.pk_max is None else relation.pk_max + 1,
                )
                for index, relation in enumerate(relations)
            )
            if any(value > _POSTGRES_INTEGER_MAX for value in desired_sequences):
                raise PostgresLegacySnapshotImportConflict(
                    "legacy snapshot has exhausted an integer sequence"
                )
            target_state = self._inspect_target(context, preflight, relations)
            disposition = self._copy_or_replay(
                source,
                context,
                preflight,
                relations,
                target_state,
            )
        finally:
            _rollback_quietly(source)
            _close_quietly(source)
        self._require_target_rows_exact(context, preflight, relations)
        self._normalize_sequences(
            context,
            relations,
            desired_sequences,
            preflight,
        )
        self._require_target_rows_exact(
            context,
            preflight,
            relations,
            desired_sequences=desired_sequences,
        )
        relation_receipts = tuple(
            LegacySnapshotRelationReceipt(
                name=relation.name,
                row_count=relation.row_count,
                pk_min=relation.pk_min,
                pk_max=relation.pk_max,
                sha256=relation.sha256,
                source_sequence_next=sequences[index],
                target_sequence_next=desired_sequences[index],
            )
            for index, relation in enumerate(relations)
        )
        return LegacySnapshotImportReceipt(
            context=context,
            disposition=disposition,
            source_system_identifier=preflight.source.system_identifier,
            target_system_identifier=preflight.target.system_identifier,
            source_canonical_sha256=preflight.source.canonical_sha256,
            relations=relation_receipts,
        )

    def _open_source(
        self,
        context: LegacySnapshotImportContext,
        preflight: FreshTargetCutoverReceipt,
    ) -> object:
        try:
            connection = _fresh_connection(
                self._source_connection_factory,
                "legacy snapshot source",
            )
        except PostgresCutoverPreflightStorageError:
            failed = True
        else:
            failed = False
        if failed:
            raise PostgresLegacySnapshotImportStorageError(
                "legacy snapshot source connection failed"
            )
        try:
            with connection.cursor() as cursor:
                cursor.execute(_READ_ONLY_SQL)
                cursor.execute(_UTC_SQL)
                cursor.execute(_SEARCH_PATH_SQL)
                cursor.execute(
                    "LOCK TABLE "
                    + ", ".join(f"ONLY {relation}" for relation in _LEGACY_RELATIONS)
                    + " IN SHARE MODE NOWAIT"
                )
                cursor.execute(_SELECT_IDENTITY_SQL)
                identity = _one_row(cursor.fetchone(), 5)
                expected = context.cutover_context
                if (
                    identity[:4]
                    != (
                        expected.source_expected_database,
                        expected.source_expected_role,
                        expected.source_expected_role,
                        expected.source_expected_role,
                    )
                    or identity[4] != preflight.source.system_identifier
                ):
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot source identity has drifted"
                    )
        except PostgresLegacySnapshotImportConflict:
            _rollback_quietly(connection)
            _close_quietly(connection)
            raise
        except Exception as error:
            busy = getattr(error, "pgcode", None) == "55P03"
            _rollback_quietly(connection)
            _close_quietly(connection)
            failed = True
        if failed:
            if busy:
                raise PostgresLegacySnapshotImportBusyError(
                    "legacy snapshot source is busy"
                )
            raise PostgresLegacySnapshotImportStorageError(
                "legacy snapshot source admission failed"
            )
        return connection

    def _admit_source(
        self,
        connection: object,
        context: LegacySnapshotImportContext,
        preflight: FreshTargetCutoverReceipt,
    ) -> tuple[tuple[FreshTargetRelationEvidence, ...], tuple[int, ...]]:
        with connection.cursor() as cursor:
            cursor.execute(_SELECT_OTHER_SESSIONS_SQL)
            other_sessions = _one_row(cursor.fetchone(), 1)[0]
            if other_sessions != 0:
                raise PostgresLegacySnapshotImportBusyError(
                    "legacy snapshot source has another database session"
                )
            expected = context.cutover_context
            if not _legacy_layout_is_exact(cursor, expected.source_expected_role):
                raise PostgresLegacySnapshotImportConflict(
                    "legacy snapshot source layout has drifted"
                )
        relations, invalid, _ = _scan_relations(
            connection,
            context.batch_size,
            "source_hash",
        )
        if invalid:
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot source contains invalid rows"
            )
        if relations != preflight.source.relations:
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot source rows have drifted"
            )
        if _combined_sha256(relations) != preflight.source.canonical_sha256:
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot source fingerprint has drifted"
            )
        if next(
            relation.row_count
            for relation in relations
            if relation.name == "np.open_positions"
        ):
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot contains an open position"
            )
        with connection.cursor() as cursor:
            sequences = tuple(
                _read_sequence_next(cursor, relation) for relation in _LEGACY_RELATIONS
            )
        return relations, sequences

    def _inspect_target(
        self,
        context: LegacySnapshotImportContext,
        preflight: FreshTargetCutoverReceipt,
        relations: tuple[FreshTargetRelationEvidence, ...],
    ) -> str:
        bootstrap_context = _bootstrap_context(context)
        try:
            inspection = PostgresBootstrap(
                self._target_admin_connection_factory
            ).inspect_historical_terminal(bootstrap_context)
        except PostgresBootstrapStorageError:
            failed = True
        else:
            failed = False
        if failed:
            raise PostgresLegacySnapshotImportStorageError(
                "legacy snapshot target inspection failed"
            )
        if (
            inspection.system_identifier != preflight.target.system_identifier
            or inspection.system_identifier == preflight.source.system_identifier
            or not inspection.exact
            or inspection.migration_versions != (1, 2, 3, 4, 5, 6)
            or inspection.runtime_mode != "LEGACY"
            or inspection.runtime_generation != 0
        ):
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot target authority has drifted"
            )
        expected_nonempty = tuple(
            sorted(relation.name for relation in relations if relation.row_count)
        )
        if not inspection.nonempty_relations:
            return "empty"
        if inspection.nonempty_relations == expected_nonempty:
            return "candidate_exact"
        raise PostgresLegacySnapshotImportConflict(
            "legacy snapshot target contains conflicting rows"
        )

    def _open_migrator(
        self,
        context: LegacySnapshotImportContext,
    ) -> object:
        try:
            connection = _fresh_connection(
                self._target_migrator_connection_factory,
                "legacy snapshot target migrator",
            )
        except PostgresCutoverPreflightStorageError:
            failed = True
        else:
            failed = False
        if failed:
            raise PostgresLegacySnapshotImportStorageError(
                "legacy snapshot target migrator connection failed"
            )
        bootstrap_context = _bootstrap_context(context)
        identity_drift = False
        identity_storage_failed = False
        try:
            PostgresBootstrap._require_historical_migrator_connection_identity(
                connection,
                bootstrap_context,
            )
        except (
            PostgresBootstrapDriftError,
            PostgresBootstrapMigrationError,
            PostgresBootstrapStorageError,
        ):
            identity_drift = True
        except Exception:
            identity_storage_failed = True
        if identity_drift or identity_storage_failed:
            _close_quietly(connection)
        if identity_drift:
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot target migrator identity has drifted"
            )
        if identity_storage_failed:
            raise PostgresLegacySnapshotImportStorageError(
                "legacy snapshot target migrator identity could not be verified"
            )
        return connection

    @staticmethod
    def _prepare_target_transaction(
        connection: object,
        context: LegacySnapshotImportContext,
        target_system_identifier: int,
    ) -> object:
        cursor = connection.cursor()
        cursor.execute(_TARGET_ISOLATION_SQL)
        cursor.execute(_UTC_SQL)
        cursor.execute(_SEARCH_PATH_SQL)
        cursor.execute(_LOCK_TIMEOUT_SQL)
        cursor.execute(_SELECT_MIGRATOR_TARGET_IDENTITY_SQL)
        identity = _one_row(cursor.fetchone(), 5)
        intent = context.cutover_context.target_bootstrap_intent
        role = intent.roles.migrator
        if identity != (
            intent.expected_database,
            role,
            role,
            role,
            target_system_identifier,
        ):
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot target migrator cluster identity has drifted"
            )
        cursor.execute(
            sql.SQL("SET LOCAL ROLE {}").format(
                sql.Identifier(
                    context.cutover_context.target_bootstrap_intent.roles.schema_owner
                )
            )
        )
        cursor.execute(_LOCK_IMPORT_TABLES_SQL)
        cursor.execute("SELECT version FROM np.schema_migrations ORDER BY version")
        if tuple(row[0] for row in cursor.fetchall()) != (1, 2, 3, 4, 5, 6):
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot target migration history has drifted"
            )
        cursor.execute(
            "SELECT mode, runtime_generation FROM np.paper_runtime_control "
            "WHERE control_key IS TRUE"
        )
        if _one_row(cursor.fetchone(), 2) != ("LEGACY", 0):
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot target runtime control has drifted"
            )
        for table in _V2_ONLY_TABLES:
            cursor.execute(
                sql.SQL("SELECT EXISTS (SELECT 1 FROM {})").format(
                    sql.Identifier("np", table)
                )
            )
            if _one_row(cursor.fetchone(), 1)[0] is not False:
                raise PostgresLegacySnapshotImportConflict(
                    "legacy snapshot target contains V2 facts"
                )
        return cursor

    def _copy_or_replay(
        self,
        source: object,
        context: LegacySnapshotImportContext,
        preflight: FreshTargetCutoverReceipt,
        expected_relations: tuple[FreshTargetRelationEvidence, ...],
        target_state: str,
    ) -> LegacySnapshotImportDisposition:
        connection = self._open_migrator(context)
        cursor = None
        commit_failed = False
        failed = False
        busy = False
        try:
            try:
                cursor = self._prepare_target_transaction(
                    connection,
                    context,
                    preflight.target.system_identifier,
                )
                locked_state = self._inspect_target(
                    context,
                    preflight,
                    expected_relations,
                )
                if locked_state != target_state:
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot target changed before locked import"
                    )
                current, invalid, _ = _scan_relations(
                    connection,
                    context.batch_size,
                    "target_before",
                )
                if invalid:
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot target contains invalid rows"
                    )
                empty = all(relation.row_count == 0 for relation in current)
                exact = current == expected_relations
                if target_state == "empty" and not empty:
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot target changed before import"
                    )
                if target_state == "candidate_exact" and not exact:
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot target rows do not match the source"
                    )
                consumers = None
                if empty:
                    consumers = {
                        relation: self._insert_consumer(cursor, relation)
                        for relation in _LEGACY_RELATIONS
                    }
                copied, copied_invalid, _ = _scan_relations(
                    source,
                    context.batch_size,
                    "source_copy",
                    consumers,
                )
                if copied_invalid or copied != expected_relations:
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot changed while it was copied"
                    )
                after, invalid_after, _ = _scan_relations(
                    connection,
                    context.batch_size,
                    "target_after",
                )
                if invalid_after or after != expected_relations:
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot target row readback is not exact"
                    )
            except (
                PostgresLegacySnapshotImportBusyError,
                PostgresLegacySnapshotImportConflict,
            ):
                _rollback_quietly(connection)
                raise
            except Exception as error:
                _rollback_quietly(connection)
                if getattr(error, "pgcode", None) == "55P03":
                    busy = True
                else:
                    busy = False
                failed = True
            if failed:
                if busy:
                    raise PostgresLegacySnapshotImportBusyError(
                        "legacy snapshot target is busy"
                    ) from None
                raise PostgresLegacySnapshotImportStorageError(
                    "legacy snapshot target copy failed before commit"
                ) from None
            try:
                connection.commit()
            except Exception:
                commit_failed = True
            if commit_failed:
                _rollback_quietly(connection)
                try:
                    readback_state = self._inspect_target(
                        context,
                        preflight,
                        expected_relations,
                    )
                    if readback_state == "candidate_exact":
                        self._require_target_rows_exact(
                            context,
                            preflight,
                            expected_relations,
                        )
                        return LegacySnapshotImportDisposition.REPLAYED
                    if readback_state != "empty":
                        raise PostgresLegacySnapshotImportConflict(
                            "legacy snapshot target commit readback conflicts"
                        )
                except PostgresLegacySnapshotImportConflict:
                    raise
                except PostgresLegacySnapshotImportStorageError:
                    pass
                raise PostgresLegacySnapshotImportCommitUnknown(
                    "legacy snapshot row commit is unknown"
                ) from None
            return (
                LegacySnapshotImportDisposition.REPLAYED
                if target_state == "candidate_exact"
                else LegacySnapshotImportDisposition.IMPORTED
            )
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            _close_quietly(connection)

    @staticmethod
    def _insert_consumer(
        cursor: object,
        relation: str,
    ) -> Callable[[tuple[tuple[object, ...], ...]], None]:
        table = relation.removeprefix("np.")
        columns = _LEGACY_COLUMNS[relation]
        statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier("np", table),
            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
            sql.SQL(", ").join(
                (
                    sql.SQL("%s::real")
                    if column in _LEGACY_REAL_COLUMNS[relation]
                    else sql.Placeholder()
                )
                for column in columns
            ),
        )

        def consume(rows: tuple[tuple[object, ...], ...]) -> None:
            if rows:
                transported = tuple(
                    tuple(
                        (
                            repr(value)
                            if column in _LEGACY_REAL_COLUMNS[relation]
                            and type(value) is float
                            else value
                        )
                        for column, value in zip(columns, row)
                    )
                    for row in rows
                )
                cursor.executemany(statement, transported)

        return consume

    def _target_rows_are_exact(
        self,
        context: LegacySnapshotImportContext,
        preflight: FreshTargetCutoverReceipt,
        expected: tuple[FreshTargetRelationEvidence, ...],
        desired_sequences: tuple[int, ...] | None = None,
    ) -> bool:
        try:
            self._require_target_rows_exact(
                context,
                preflight,
                expected,
                desired_sequences=desired_sequences,
            )
        except (
            PostgresLegacySnapshotImportConflict,
            PostgresLegacySnapshotImportStorageError,
        ):
            return False
        return True

    def _require_target_rows_exact(
        self,
        context: LegacySnapshotImportContext,
        preflight: FreshTargetCutoverReceipt,
        expected: tuple[FreshTargetRelationEvidence, ...],
        desired_sequences: tuple[int, ...] | None = None,
    ) -> None:
        inspection_state = self._inspect_target(context, preflight, expected)
        if inspection_state != "candidate_exact":
            raise PostgresLegacySnapshotImportConflict(
                "legacy snapshot target row commit is not exact"
            )
        try:
            connection = _fresh_connection(
                self._target_admin_connection_factory,
                "legacy snapshot target readback",
            )
        except PostgresCutoverPreflightStorageError:
            failed = True
        else:
            failed = False
        if failed:
            raise PostgresLegacySnapshotImportStorageError(
                "legacy snapshot target readback connection failed"
            )
        try:
            with connection.cursor() as cursor:
                cursor.execute(_READ_ONLY_SQL)
                cursor.execute(_UTC_SQL)
                cursor.execute(_SEARCH_PATH_SQL)
            actual, invalid, _ = _scan_relations(
                connection,
                context.batch_size,
                "target_readback",
            )
            if invalid or actual != expected:
                raise PostgresLegacySnapshotImportConflict(
                    "legacy snapshot target row readback is not exact"
                )
            if desired_sequences is not None:
                with connection.cursor() as cursor:
                    actual_sequences = tuple(
                        _read_sequence_next(cursor, relation)
                        for relation in _LEGACY_RELATIONS
                    )
                if actual_sequences != desired_sequences:
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot target sequence readback is not exact"
                    )
        finally:
            _rollback_quietly(connection)
            _close_quietly(connection)

    def _normalize_sequences(
        self,
        context: LegacySnapshotImportContext,
        relations: tuple[FreshTargetRelationEvidence, ...],
        desired: tuple[int, ...],
        preflight: FreshTargetCutoverReceipt,
    ) -> None:
        connection = self._open_migrator(context)
        failed = False
        busy = False
        target_system_identifier = preflight.target.system_identifier
        try:
            try:
                cursor = self._prepare_target_transaction(
                    connection,
                    context,
                    target_system_identifier,
                )
                if self._inspect_target(context, preflight, relations) != (
                    "candidate_exact"
                ):
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot target changed before sequence normalization"
                    )
                locked_relations, invalid, _ = _scan_relations(
                    connection,
                    context.batch_size,
                    "target_sequence_locked",
                )
                if invalid or locked_relations != relations:
                    raise PostgresLegacySnapshotImportConflict(
                        "legacy snapshot target rows changed before sequence normalization"
                    )
                for relation, evidence, next_value in zip(
                    _LEGACY_RELATIONS, relations, desired
                ):
                    cursor.execute(
                        sql.SQL(
                            "SELECT pg_catalog.setval({}::regclass, %s, false)"
                        ).format(sql.Literal(f"np.{_sequence_name(relation)}")),
                        (next_value,),
                    )
                    if evidence.pk_max is not None and next_value <= evidence.pk_max:
                        raise PostgresLegacySnapshotImportConflict(
                            "legacy snapshot target sequence would collide"
                        )
                cursor.close()
                connection.commit()
            except PostgresLegacySnapshotImportConflict:
                _rollback_quietly(connection)
                raise
            except Exception as error:
                _rollback_quietly(connection)
                busy = getattr(error, "pgcode", None) == "55P03"
                failed = True
        finally:
            _close_quietly(connection)
        if failed:
            if busy:
                raise PostgresLegacySnapshotImportBusyError(
                    "legacy snapshot target is busy during sequence normalization"
                ) from None
            raise PostgresLegacySnapshotImportStorageError(
                "legacy snapshot sequence normalization failed"
            ) from None


__all__ = [
    "PostgresLegacySnapshotImport",
    "PostgresLegacySnapshotImportBusyError",
    "PostgresLegacySnapshotImportCommitUnknown",
    "PostgresLegacySnapshotImportConflict",
    "PostgresLegacySnapshotImportInputError",
    "PostgresLegacySnapshotImportStorageError",
]
