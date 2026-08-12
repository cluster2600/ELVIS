"""Unwired PostgreSQL snapshot assessment before a paper-runtime fence."""

from collections import Counter
from collections.abc import Callable

import psycopg2

from trading.application.paper_account_readiness import (
    LegacyRelationWatermark,
    MigrationIdentity,
    PaperAccountReadinessAssessment,
    PaperAccountReadinessContext,
    PaperAccountReadinessFinding,
    PaperAccountReadinessFindingKind,
)
from trading.domain.order_lifecycle import OrderLifecycleState
from trading.domain.paper_accounting import PaperAccountState
from trading.domain.positions import PositionState
from trading.persistence.migration_runner import load_migrations
from trading.persistence.order_position_journal import (
    _READ_TRANSACTION_SQL,
    JournalRepositoryError,
    PostgresOrderPositionJournal,
    _replay_stream,
)
from trading.persistence.paper_account_journal import (
    PaperAccountJournalError,
    PaperAccountReplayError,
    _replay_account_locked,
)

_ACCOUNT_KEY_MAX_LENGTH = 255
_POSITION_KEY_MAX_LENGTH = 255
_CLIENT_ORDER_ID_MAX_LENGTH = 255
_SCHEMA_MIGRATION_RELATION = "np.schema_migrations"
_LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)
_DURABLE_BUSINESS_RELATIONS = tuple(
    sorted(
        _LEGACY_RELATIONS
        + (
            "np.order_events",
            "np.orders",
            "np.paper_account_balances",
            "np.paper_account_batch_manifests",
            "np.paper_account_postings",
            "np.paper_account_settlements",
            "np.paper_account_streams",
            "np.paper_margin_reservations",
            "np.position_streams",
        )
    )
)
_SCHEMA_DRIFT_SQLSTATES = frozenset(
    {
        "42703",  # undefined_column
        "42704",  # undefined_object
        "42804",  # datatype_mismatch
        "42809",  # wrong_object_type
        "42P01",  # undefined_table
    }
)
_TERMINAL_LIFECYCLE_STATES = frozenset(
    {
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.FAILED,
    }
)

_SELECT_MIGRATION_RELATION_SQL = "SELECT to_regclass(%s)"
_SELECT_APPLIED_MIGRATIONS_SQL = """
SELECT version, name, checksum
FROM np.schema_migrations
ORDER BY version
"""
_SELECT_MIGRATION_COLUMNS_SQL = """
SELECT
    ordinal_position,
    column_name,
    udt_name,
    is_nullable,
    CASE
        WHEN column_default IS NULL THEN 'none'
        WHEN LOWER(column_default) IN ('now()', 'current_timestamp') THEN 'now'
        ELSE 'other'
    END,
    character_maximum_length
FROM information_schema.columns
WHERE table_schema = 'np'
  AND table_name = 'schema_migrations'
ORDER BY ordinal_position
"""
_SELECT_MIGRATION_CONSTRAINTS_SQL = """
SELECT
    constraint_row.contype,
    constraint_row.conkey,
    constraint_row.condeferrable,
    constraint_row.condeferred,
    constraint_row.convalidated
FROM pg_constraint constraint_row
JOIN pg_class table_row
  ON table_row.oid = constraint_row.conrelid
JOIN pg_namespace namespace_row
  ON namespace_row.oid = table_row.relnamespace
WHERE namespace_row.nspname = 'np'
  AND table_row.relname = 'schema_migrations'
ORDER BY constraint_row.conname
"""
_SELECT_MIGRATION_RELATION_KIND_SQL = """
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
  AND table_row.relname = 'schema_migrations'
"""
_SELECT_DURABLE_RELATIONS_SQL = """
SELECT
    FORMAT('%%I.%%I', namespace_row.nspname, table_row.relname),
    table_row.relkind,
    table_row.relpersistence,
    table_row.relhasrules,
    table_row.relrowsecurity,
    table_row.relforcerowsecurity,
    EXISTS (
        SELECT 1
        FROM pg_trigger trigger_row
        WHERE trigger_row.tgrelid = table_row.oid
          AND NOT trigger_row.tgisinternal
    ),
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
  AND table_row.relname = ANY(%s)
ORDER BY table_row.relname
"""
_SELECT_ACCOUNT_IDENTITIES_SQL = """
SELECT account_key, execution_scope
FROM np.paper_account_streams
ORDER BY account_key
"""
_SELECT_POSITION_IDENTITIES_SQL = """
SELECT position_key, execution_scope
FROM np.position_streams
ORDER BY position_key
"""
_SELECT_ORDER_REFERENCES_SQL = """
SELECT position_key, execution_scope, client_order_id
FROM np.orders
ORDER BY position_key, client_order_id
"""
_SELECT_MANIFEST_REFERENCES_SQL = """
SELECT account_key, execution_scope, position_key, client_order_id
FROM np.paper_account_batch_manifests
ORDER BY account_key, client_order_id
"""
_SELECT_LEGACY_WATERMARK_SQL = {
    relation: f"SELECT COUNT(*), MAX(id) FROM {relation}"
    for relation in _LEGACY_RELATIONS
}


class PaperAccountReadinessError(RuntimeError):
    """Base class for dormant pre-fence assessment failures."""


class PaperAccountReadinessInputError(PaperAccountReadinessError, ValueError):
    """Raised before I/O when the assessment context is invalid."""


class PaperAccountReadinessStorageError(PaperAccountReadinessError):
    """Raised when no complete assessment can be obtained from PostgreSQL."""


def _finding(
    kind: PaperAccountReadinessFindingKind,
    subject_kind: str,
    subject_id: str,
) -> PaperAccountReadinessFinding:
    return PaperAccountReadinessFinding(kind, subject_kind, subject_id)


def _expected_migrations() -> tuple[MigrationIdentity, ...]:
    try:
        migrations = load_migrations()
        return tuple(
            MigrationIdentity(item.version, item.name, item.checksum)
            for item in migrations
        )
    except Exception as exc:
        raise PaperAccountReadinessStorageError(
            "packaged migration evidence cannot be loaded"
        ) from exc


def _one_row(raw: object, field: str, length: int) -> tuple[object, ...]:
    if not isinstance(raw, (tuple, list)) or len(raw) != length:
        raise PaperAccountReadinessStorageError(
            f"PostgreSQL returned an invalid {field} row"
        )
    return tuple(raw)


def _stored_key(value: object, field: str, maximum: int) -> str:
    if type(value) is not str:
        raise ValueError(f"stored {field} is not text")
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"stored {field} is not canonical")
    if "\x00" in value or any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        raise ValueError(f"stored {field} is not representable")
    return value


def _decode_identities(
    rows: object,
    *,
    field: str,
    maximum: int,
) -> tuple[tuple[str, str], ...]:
    try:
        values = tuple(
            (
                _stored_key(_one_row(row, field, 2)[0], field, maximum),
                _stored_key(
                    _one_row(row, field, 2)[1],
                    "execution scope",
                    128,
                ),
            )
            for row in rows
        )
    except (PaperAccountReadinessStorageError, TypeError, ValueError) as exc:
        raise PaperAccountReplayError(f"stored {field} inventory is invalid") from exc
    if len({identity[0] for identity in values}) != len(values):
        raise PaperAccountReplayError(f"stored {field} inventory repeats an identity")
    return tuple(sorted(values))


def _raw_migration_drift() -> PaperAccountReadinessFinding:
    return _finding(
        PaperAccountReadinessFindingKind.MIGRATION_DRIFT,
        "migration_ledger",
        _SCHEMA_MIGRATION_RELATION,
    )


def _migration_metadata_is_exact(cursor: object) -> bool:
    try:
        cursor.execute(_SELECT_MIGRATION_RELATION_KIND_SQL)
        relation_rows = tuple(cursor.fetchall())
        if relation_rows != (("r", "p", False, False, False, False, False, False),):
            return False

        cursor.execute(_SELECT_MIGRATION_COLUMNS_SQL)
        columns = tuple(tuple(row) for row in cursor.fetchall())
        if columns != (
            (1, "version", "int4", "NO", "none", None),
            (2, "name", "text", "NO", "none", None),
            (3, "checksum", "bpchar", "NO", "none", 64),
            (4, "applied_at", "timestamptz", "NO", "now", None),
        ):
            return False

        cursor.execute(_SELECT_MIGRATION_CONSTRAINTS_SQL)
        constraints = tuple(cursor.fetchall())
        if len(constraints) != 1:
            return False
        constraint = _one_row(
            constraints[0],
            "migration ledger constraint",
            5,
        )
        return (
            constraint[0] == "p"
            and tuple(constraint[1]) == (1,)
            and constraint[2:] == (False, False, True)
        )
    except (PaperAccountReadinessStorageError, TypeError, ValueError):
        return False


def _read_migration_evidence(
    cursor: object,
) -> tuple[tuple[MigrationIdentity, ...], tuple[PaperAccountReadinessFinding, ...]]:
    cursor.execute(_SELECT_MIGRATION_RELATION_SQL, (_SCHEMA_MIGRATION_RELATION,))
    relation_row = cursor.fetchone()
    relation = _one_row(relation_row, "migration relation", 1)[0]
    if relation is None:
        return (), ()
    if type(relation) is not str or relation != _SCHEMA_MIGRATION_RELATION:
        return (), (_raw_migration_drift(),)
    if not _migration_metadata_is_exact(cursor):
        return (), (_raw_migration_drift(),)

    try:
        cursor.execute(_SELECT_APPLIED_MIGRATIONS_SQL)
        raw_rows = tuple(cursor.fetchall())
    except psycopg2.Error as exc:
        if getattr(exc, "pgcode", None) in _SCHEMA_DRIFT_SQLSTATES:
            return (), (_raw_migration_drift(),)
        raise

    decoded = []
    for raw in raw_rows:
        try:
            row = _one_row(raw, "migration ledger", 3)
            identity = MigrationIdentity(row[0], row[1], row[2])
            if identity.version != len(decoded) + 1:
                raise ValueError("migration versions are not contiguous")
        except (PaperAccountReadinessStorageError, TypeError, ValueError):
            return tuple(decoded), (_raw_migration_drift(),)
        decoded.append(identity)
    return tuple(decoded), ()


def _durable_business_relations_are_authoritative(cursor: object) -> bool:
    cursor.execute(
        _SELECT_DURABLE_RELATIONS_SQL,
        ([relation.removeprefix("np.") for relation in _DURABLE_BUSINESS_RELATIONS],),
    )
    rows = tuple(tuple(row) for row in cursor.fetchall())
    expected = tuple(
        (relation, "r", "p", False, False, False, False, False, False)
        for relation in _DURABLE_BUSINESS_RELATIONS
    )
    return rows == expected


def _decode_order_references(rows: object) -> tuple[tuple[str, str, str], ...]:
    try:
        references = tuple(
            (
                _stored_key(
                    _one_row(row, "order reference", 3)[0],
                    "position key",
                    _POSITION_KEY_MAX_LENGTH,
                ),
                _stored_key(
                    _one_row(row, "order reference", 3)[1],
                    "execution scope",
                    128,
                ),
                _stored_key(
                    _one_row(row, "order reference", 3)[2],
                    "client order ID",
                    _CLIENT_ORDER_ID_MAX_LENGTH,
                ),
            )
            for row in rows
        )
    except (PaperAccountReadinessStorageError, TypeError, ValueError) as exc:
        raise JournalRepositoryError("stored order inventory is invalid") from exc
    return tuple(sorted(references))


def _decode_manifest_references(
    rows: object,
) -> tuple[tuple[str, str, str, str], ...]:
    try:
        references = tuple(
            (
                _stored_key(
                    _one_row(row, "manifest reference", 4)[0],
                    "account key",
                    _ACCOUNT_KEY_MAX_LENGTH,
                ),
                _stored_key(
                    _one_row(row, "manifest reference", 4)[1],
                    "execution scope",
                    128,
                ),
                _stored_key(
                    _one_row(row, "manifest reference", 4)[2],
                    "position key",
                    _POSITION_KEY_MAX_LENGTH,
                ),
                _stored_key(
                    _one_row(row, "manifest reference", 4)[3],
                    "client order ID",
                    _CLIENT_ORDER_ID_MAX_LENGTH,
                ),
            )
            for row in rows
        )
    except (PaperAccountReadinessStorageError, TypeError, ValueError) as exc:
        raise PaperAccountReplayError("stored manifest inventory is invalid") from exc
    return tuple(sorted(references))


def _migration_only_assessment(
    *,
    context: PaperAccountReadinessContext,
    expected: tuple[MigrationIdentity, ...],
    applied: tuple[MigrationIdentity, ...],
    findings: tuple[PaperAccountReadinessFinding, ...],
) -> PaperAccountReadinessAssessment:
    return PaperAccountReadinessAssessment(
        context=context,
        expected_migrations=expected,
        applied_migrations=applied,
        account_version=None,
        legacy_watermarks=(),
        findings=findings,
    )


def _schema_drift_assessment(
    context: PaperAccountReadinessContext,
    expected: tuple[MigrationIdentity, ...],
    applied: tuple[MigrationIdentity, ...],
) -> PaperAccountReadinessAssessment:
    return _migration_only_assessment(
        context=context,
        expected=expected,
        applied=applied,
        findings=(_raw_migration_drift(),),
    )


def _read_legacy_watermarks(cursor: object) -> tuple[LegacyRelationWatermark, ...]:
    result = []
    for relation in _LEGACY_RELATIONS:
        cursor.execute(_SELECT_LEGACY_WATERMARK_SQL[relation])
        row = _one_row(cursor.fetchone(), f"{relation} watermark", 2)
        try:
            result.append(LegacyRelationWatermark(relation, row[0], row[1]))
        except (TypeError, ValueError) as exc:
            raise PaperAccountReplayError(
                f"stored {relation} watermark is invalid"
            ) from exc
    return tuple(result)


def _assess_exact_schema(
    cursor: object,
    *,
    context: PaperAccountReadinessContext,
    expected: tuple[MigrationIdentity, ...],
    applied: tuple[MigrationIdentity, ...],
) -> PaperAccountReadinessAssessment:
    findings = []
    account_version = None

    raw_order_references = None
    try:
        cursor.execute(_SELECT_ORDER_REFERENCES_SQL)
        raw_order_references = _decode_order_references(cursor.fetchall())
        if len(
            {client_order_id for _, _, client_order_id in raw_order_references}
        ) != len(raw_order_references):
            raise JournalRepositoryError(
                "stored order inventory repeats a client identity"
            )
    except JournalRepositoryError:
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
                "durable_relation",
                "np.orders",
            )
        )

    raw_manifest_references = None
    try:
        cursor.execute(_SELECT_MANIFEST_REFERENCES_SQL)
        raw_manifest_references = _decode_manifest_references(cursor.fetchall())
        if len(
            {client_order_id for _, _, _, client_order_id in raw_manifest_references}
        ) != len(raw_manifest_references):
            raise PaperAccountReplayError(
                "stored manifest inventory repeats an order identity"
            )
    except PaperAccountReplayError:
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                "durable_relation",
                "np.paper_account_batch_manifests",
            )
        )

    try:
        cursor.execute(_SELECT_ACCOUNT_IDENTITIES_SQL)
        account_identities = _decode_identities(
            cursor.fetchall(), field="account key", maximum=_ACCOUNT_KEY_MAX_LENGTH
        )
    except PaperAccountReplayError:
        account_identities = ()
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                "durable_relation",
                "np.paper_account_streams",
            )
        )

    if (context.account_key, context.execution_scope) not in account_identities:
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED,
                "paper_account",
                context.account_key,
            )
        )
    for account_key, execution_scope in account_identities:
        if (account_key, execution_scope) != (
            context.account_key,
            context.execution_scope,
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.UNEXPECTED_ACCOUNT,
                    "paper_account",
                    account_key,
                )
            )

    replayed_accounts = {}
    replayed_manifest_references = []
    for account_key, execution_scope in account_identities:
        try:
            replayed = _replay_account_locked(
                cursor,
                execution_scope=execution_scope,
                account_key=account_key,
                lock=False,
            )
        except PaperAccountJournalError:
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                    "paper_account",
                    account_key,
                )
            )
            continue
        replayed_accounts[(account_key, execution_scope)] = replayed
        replayed_manifest_references.extend(
            (
                account_key,
                execution_scope,
                batch.position_key,
                batch.client_order_id,
            )
            for batch in replayed.batches
        )
        if replayed.account.state is PaperAccountState.INSOLVENT:
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.ACCOUNT_INSOLVENT,
                    "paper_account",
                    account_key,
                )
            )
        for reservation in replayed.account.reservations:
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.MARGIN_RESERVATION_PRESENT,
                    "position_stream",
                    reservation.position_key,
                )
            )

    expected_account = replayed_accounts.get(
        (context.account_key, context.execution_scope)
    )
    if expected_account is not None:
        account_version = len(expected_account.account.records)
        if (
            expected_account.owner_generation != context.owner_generation
            or expected_account.opening_payload_sha256 != context.opening_payload_sha256
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.ACCOUNT_PROVENANCE_MISMATCH,
                    "paper_account",
                    context.account_key,
                )
            )

    try:
        cursor.execute(_SELECT_POSITION_IDENTITIES_SQL)
        position_identities = _decode_identities(
            cursor.fetchall(), field="position key", maximum=_POSITION_KEY_MAX_LENGTH
        )
    except PaperAccountReplayError:
        position_identities = ()
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
                "durable_relation",
                "np.position_streams",
            )
        )

    replayed_order_references = []
    for position_key, execution_scope in position_identities:
        try:
            projection = _replay_stream(
                cursor,
                execution_scope=execution_scope,
                position_key=position_key,
                lock=False,
            ).projection
        except JournalRepositoryError:
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
                    "position_stream",
                    position_key,
                )
            )
            continue
        if projection.position is not None and (
            projection.position.state is PositionState.OPEN
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.DURABLE_OPEN_POSITION,
                    "position_stream",
                    position_key,
                )
            )
        for order in projection.orders:
            client_order_id = order.instruction.order_intent.client_order_id
            replayed_order_references.append(
                (position_key, execution_scope, client_order_id)
            )
            if order.lifecycle.state not in _TERMINAL_LIFECYCLE_STATES:
                findings.append(
                    _finding(
                        PaperAccountReadinessFindingKind.UNRESOLVED_SUBMISSION,
                        "client_order",
                        client_order_id,
                    )
                )

    replayed_orders = tuple(sorted(replayed_order_references))
    replayed_manifests = tuple(sorted(replayed_manifest_references))
    if raw_order_references is not None and Counter(raw_order_references) != Counter(
        replayed_orders
    ):
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
                "durable_relation",
                "np.orders",
            )
        )
    if raw_manifest_references is not None and Counter(
        raw_manifest_references
    ) != Counter(replayed_manifests):
        findings.append(
            _finding(
                PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                "durable_relation",
                "np.paper_account_batch_manifests",
            )
        )

    if raw_order_references is not None and raw_manifest_references is not None:
        order_claims = Counter(raw_order_references)
        manifest_claims = Counter(
            (execution_scope, position_key, client_order_id)
            for _, execution_scope, position_key, client_order_id in (
                raw_manifest_references
            )
        )
        normalized_orders = Counter(
            (execution_scope, position_key, client_order_id)
            for position_key, execution_scope, client_order_id in (
                order_claims.elements()
            )
        )
        for _, _, client_order_id in sorted(
            (normalized_orders - manifest_claims).elements()
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
                    "client_order",
                    client_order_id,
                )
            )
        for _, _, client_order_id in sorted(
            (manifest_claims - normalized_orders).elements()
        ):
            findings.append(
                _finding(
                    PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                    "client_order",
                    client_order_id,
                )
            )

    watermarks = _read_legacy_watermarks(cursor)
    return PaperAccountReadinessAssessment(
        context=context,
        expected_migrations=expected,
        applied_migrations=applied,
        account_version=account_version,
        legacy_watermarks=watermarks,
        findings=tuple(findings),
    )


class PostgresPaperAccountReadiness:
    """Collect one stale-on-return assessment from a single read-only snapshot."""

    def __init__(self, connection_factory: Callable[[], object]) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._journal_boundary = PostgresOrderPositionJournal(connection_factory)

    def assess(
        self,
        context: PaperAccountReadinessContext,
        /,
    ) -> PaperAccountReadinessAssessment:
        """Assess durable evidence without committing or granting authority."""
        if type(context) is not PaperAccountReadinessContext:
            raise PaperAccountReadinessInputError(
                "context must be a PaperAccountReadinessContext"
            )
        expected = _expected_migrations()
        try:
            connection = self._journal_boundary._connection()
        except JournalRepositoryError as exc:
            raise PaperAccountReadinessStorageError(
                "could not open a readiness assessment connection"
            ) from exc

        try:
            applied: tuple[MigrationIdentity, ...] = ()
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_TRANSACTION_SQL)
                    applied, migration_findings = _read_migration_evidence(cursor)
                    if migration_findings or applied != expected:
                        result = _migration_only_assessment(
                            context=context,
                            expected=expected,
                            applied=applied,
                            findings=migration_findings,
                        )
                    elif not _durable_business_relations_are_authoritative(cursor):
                        result = _schema_drift_assessment(
                            context,
                            expected,
                            applied,
                        )
                    else:
                        result = _assess_exact_schema(
                            cursor,
                            context=context,
                            expected=expected,
                            applied=applied,
                        )
            except psycopg2.Error as exc:
                if getattr(exc, "pgcode", None) in _SCHEMA_DRIFT_SQLSTATES:
                    result = _schema_drift_assessment(context, expected, applied)
                else:
                    raise PaperAccountReadinessStorageError(
                        "paper readiness assessment query failed"
                    ) from exc
            except PaperAccountReadinessError:
                raise
            except (PaperAccountJournalError, JournalRepositoryError) as exc:
                raise PaperAccountReadinessStorageError(
                    "paper readiness assessment could not replay its snapshot"
                ) from exc
            except Exception as exc:
                raise PaperAccountReadinessStorageError(
                    "paper readiness assessment failed"
                ) from exc

            try:
                connection.rollback()
            except Exception as exc:
                raise PaperAccountReadinessStorageError(
                    "paper readiness snapshot could not finish"
                ) from exc
            return result
        except Exception:
            self._journal_boundary._rollback(connection)
            raise
        finally:
            self._journal_boundary._close(connection)


__all__ = [
    "PaperAccountReadinessError",
    "PaperAccountReadinessInputError",
    "PaperAccountReadinessStorageError",
    "PostgresPaperAccountReadiness",
]
