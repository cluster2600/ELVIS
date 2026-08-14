"""Read-only PostgreSQL review of two legacy opening interpretations."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from decimal import Decimal

from trading.application.fresh_target_cutover import FreshTargetRelationEvidence
from trading.application.legacy_snapshot_import import LegacySnapshotImportReceipt
from trading.application.legacy_snapshot_reconciliation import (
    LegacyOpeningCandidate,
    LegacyOpeningCandidateSource,
    LegacySnapshotReconciliationContext,
    LegacySnapshotReconciliationDisposition,
    LegacySnapshotReconciliationEvidence,
    LegacySnapshotReconciliationFinding,
    LegacySnapshotReconciliationFindingKind,
    LegacySnapshotReconciliationReceipt,
    legacy_opening_candidate_sha256,
    legacy_opening_quantization_required,
    legacy_operator_equity_hypothesis_balances,
    legacy_snapshot_import_receipt_sha256,
    legacy_snapshot_relation_evidence_sha256,
)
from trading.domain.paper_accounting import PaperAccountBalance
from trading.persistence.postgres_bootstrap import (
    _HISTORICAL_EXPECTED_ROLE_ATTRIBUTES,
    _HISTORICAL_ROLE_MARKER_PREFIX,
    _SELECT_ADMIN_IDENTITY_SQL,
    _SELECT_CREDENTIAL_IDENTITY_SQL,
    PostgresBootstrap,
    PostgresBootstrapDriftError,
    PostgresBootstrapStorageError,
)
from trading.persistence.postgres_cutover_preflight import (
    _READ_ONLY_SQL,
    _SEARCH_PATH_SQL,
    _SELECT_IDENTITY_SQL,
    _UTC_SQL,
    PostgresCutoverPreflightStorageError,
    _fresh_connection,
    _one_row,
    _rollback_quietly,
)
from trading.persistence.postgres_legacy_snapshot_import import (
    _LEGACY_RELATIONS,
    _V2_ONLY_TABLES,
    PostgresLegacySnapshotImportConflict,
    _bootstrap_context,
    _read_sequence_next,
    _scan_relations,
)

_SELECT_CLIENT_SESSIONS_SQL = """
SELECT usename
FROM pg_stat_activity
WHERE datname = current_database()
  AND pid <> pg_backend_pid()
  AND backend_type = 'client backend'
ORDER BY pid
"""
_SELECT_IMPORTED_BALANCES_SQL = """
SELECT asset, encode(pg_catalog.float4send(balance), 'hex')
FROM np.account_balances
ORDER BY asset
"""
_SELECT_LATEST_RESET_SQL = """
SELECT reset_timestamp
FROM np.trading_session_resets
ORDER BY reset_timestamp DESC, id DESC
LIMIT 1
"""
_SELECT_TRADE_VALUES_SQL = """
SELECT encode(pg_catalog.float4send(pnl), 'hex'),
       encode(pg_catalog.float4send(fee), 'hex')
FROM np.trades
ORDER BY id
"""
_SELECT_TRADE_VALUES_AFTER_SQL = _SELECT_TRADE_VALUES_SQL.replace(
    "ORDER BY id", "WHERE timestamp >= %s ORDER BY id"
)
_SELECT_LIQUIDATION_FEES_SQL = """
SELECT encode(pg_catalog.float4send(liquidation_fee), 'hex')
FROM np.liquidations
ORDER BY id
"""
_SELECT_LIQUIDATION_FEES_AFTER_SQL = _SELECT_LIQUIDATION_FEES_SQL.replace(
    "ORDER BY id", "WHERE timestamp >= %s ORDER BY id"
)
_OPENING_ASSET_MAX_LENGTH = 64


class PostgresLegacySnapshotReconciliationInputError(ValueError):
    """Raised before inspection when the review contract is invalid."""


class PostgresLegacySnapshotReconciliationConflict(RuntimeError):
    """Raised when supplied evidence cannot identify the declared snapshot."""


class PostgresLegacySnapshotReconciliationStorageError(RuntimeError):
    """Raised without driver detail when safe inspection cannot complete."""


def _close_quietly(connection: object) -> None:
    try:
        connection.close()
    except Exception:
        pass


def _float4_from_hex(value: object) -> float:
    if (
        type(value) is not str
        or len(value) != 8
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("legacy numeric evidence is not PostgreSQL float4 bytes")
    decoded = struct.unpack("!f", bytes.fromhex(value))[0]
    if not math.isfinite(decoded):
        raise ValueError("legacy numeric evidence is not finite")
    return decoded


def _decimal_from_float4_hex(value: object) -> Decimal:
    return Decimal.from_float(_float4_from_hex(value))


def _asset_is_representable(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and len(value) <= _OPENING_ASSET_MAX_LENGTH
        and "\x00" not in value
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _ordered_binary64_sum_of_float4(
    rows: tuple[tuple[object, ...], ...], index: int
) -> float:
    """Apply the documented hypothesis algorithm in primary-key order."""

    total = 0.0
    for row in rows:
        total += _float4_from_hex(row[index])
    if not math.isfinite(total):
        raise ValueError("legacy numeric aggregation is not finite")
    return total


def _unavailable_candidates() -> tuple[LegacyOpeningCandidate, ...]:
    return tuple(
        LegacyOpeningCandidate(source, (), None, False)
        for source in LegacyOpeningCandidateSource
    )


def _finding(
    kind: LegacySnapshotReconciliationFindingKind,
) -> LegacySnapshotReconciliationFinding:
    return LegacySnapshotReconciliationFinding(kind)


def _blocked_receipt(
    context: LegacySnapshotReconciliationContext,
    import_receipt: LegacySnapshotImportReceipt,
    *kinds: LegacySnapshotReconciliationFindingKind,
) -> LegacySnapshotReconciliationReceipt:
    return LegacySnapshotReconciliationReceipt(
        context=context,
        import_receipt=import_receipt,
        disposition=LegacySnapshotReconciliationDisposition.BLOCKED,
        findings=tuple(_finding(kind) for kind in dict.fromkeys(kinds)),
        evidence=LegacySnapshotReconciliationEvidence(
            reset_timestamp=None,
            hypothesis_realised_pnl=Decimal("0"),
            hypothesis_trade_fees=Decimal("0"),
            hypothesis_liquidation_fees=Decimal("0"),
            candidates=_unavailable_candidates(),
        ),
        target_system_identifier=import_receipt.target_system_identifier,
        source_canonical_sha256=import_receipt.source_canonical_sha256,
        config_document_sha256=context.config_document_sha256,
        import_receipt_sha256=context.import_receipt_sha256,
    )


def _expected_relations(
    import_receipt: LegacySnapshotImportReceipt,
) -> tuple[FreshTargetRelationEvidence, ...]:
    return tuple(
        FreshTargetRelationEvidence(
            name=value.name,
            row_count=value.row_count,
            pk_min=value.pk_min,
            pk_max=value.pk_max,
            sha256=value.sha256,
        )
        for value in import_receipt.relations
    )


def _opening_candidate(
    context: LegacySnapshotReconciliationContext,
    source: LegacyOpeningCandidateSource,
    balances: tuple[PaperAccountBalance, ...],
) -> LegacyOpeningCandidate:
    return LegacyOpeningCandidate(
        source=source,
        balances=balances,
        opening_payload_sha256=legacy_opening_candidate_sha256(context, balances),
        available=True,
    )


class PostgresLegacySnapshotReconciliation:
    """Compare imported balances with one non-runtime operator hypothesis."""

    def __init__(
        self,
        target_admin_connection_factory: Callable[[], object],
        target_readiness_connection_factory: Callable[[], object],
    ) -> None:
        if not callable(target_admin_connection_factory) or not callable(
            target_readiness_connection_factory
        ):
            raise PostgresLegacySnapshotReconciliationInputError(
                "target reconciliation connection factories must be callable"
            )
        if id(target_admin_connection_factory) == id(
            target_readiness_connection_factory
        ):
            raise PostgresLegacySnapshotReconciliationInputError(
                "target reconciliation connection factories must be distinct"
            )
        self._target_admin_connection_factory = target_admin_connection_factory
        self._target_readiness_connection_factory = target_readiness_connection_factory

    def reconcile(
        self,
        context: LegacySnapshotReconciliationContext,
        import_receipt: LegacySnapshotImportReceipt,
        /,
    ) -> LegacySnapshotReconciliationReceipt:
        if type(context) is not LegacySnapshotReconciliationContext:
            raise PostgresLegacySnapshotReconciliationInputError(
                "context must be a LegacySnapshotReconciliationContext"
            )
        if type(import_receipt) is not LegacySnapshotImportReceipt:
            raise PostgresLegacySnapshotReconciliationInputError(
                "import_receipt must be a LegacySnapshotImportReceipt"
            )
        if context.import_context != import_receipt.context:
            raise PostgresLegacySnapshotReconciliationInputError(
                "context must be bound to the exact import receipt"
            )
        if context.import_receipt_sha256 != legacy_snapshot_import_receipt_sha256(
            import_receipt
        ) or import_receipt.source_canonical_sha256 != legacy_snapshot_relation_evidence_sha256(
            import_receipt.relations
        ):
            raise PostgresLegacySnapshotReconciliationInputError(
                "import receipt evidence is not internally consistent"
            )
        storage_failed = False
        try:
            return self._reconcile(context, import_receipt)
        except (
            PostgresLegacySnapshotReconciliationConflict,
            PostgresLegacySnapshotReconciliationInputError,
            PostgresLegacySnapshotReconciliationStorageError,
        ):
            raise
        except Exception:
            storage_failed = True
        if storage_failed:
            raise PostgresLegacySnapshotReconciliationStorageError(
                "PostgreSQL legacy snapshot reconciliation failed"
            ) from None
        raise AssertionError("unreachable")

    def _open(self, factory: Callable[[], object], label: str) -> object:
        failed = False
        try:
            connection = _fresh_connection(factory, label)
        except PostgresCutoverPreflightStorageError:
            failed = True
            connection = None
        if failed:
            raise PostgresLegacySnapshotReconciliationStorageError(
                f"the {label} connection could not be opened"
            )
        return connection

    @staticmethod
    def _begin(connection: object) -> None:
        with connection.cursor() as cursor:
            cursor.execute(_READ_ONLY_SQL)
            cursor.execute(_UTC_SQL)
            cursor.execute(_SEARCH_PATH_SQL)

    @staticmethod
    def _require_admin_identity(
        connection: object,
        context: LegacySnapshotReconciliationContext,
        import_receipt: LegacySnapshotImportReceipt,
    ) -> None:
        intent = context.import_context.cutover_context.target_bootstrap_intent
        with connection.cursor() as cursor:
            cursor.execute(_SELECT_ADMIN_IDENTITY_SQL)
            identity = tuple(_one_row(cursor.fetchone(), 6))
            cursor.execute(_SELECT_IDENTITY_SQL)
            cluster = tuple(_one_row(cursor.fetchone(), 5))
        if identity != (
            intent.expected_database,
            intent.admin_role,
            intent.admin_role,
            intent.admin_role,
            True,
            True,
        ) or cluster != (
            intent.expected_database,
            intent.admin_role,
            intent.admin_role,
            intent.admin_role,
            import_receipt.target_system_identifier,
        ):
            raise PostgresLegacySnapshotReconciliationConflict(
                "target admin identity does not match the import receipt"
            )

    @staticmethod
    def _require_readiness_identity(
        connection: object,
        context: LegacySnapshotReconciliationContext,
        import_receipt: LegacySnapshotImportReceipt,
    ) -> None:
        intent = context.import_context.cutover_context.target_bootstrap_intent
        bootstrap_context = _bootstrap_context(context.import_context)
        role = intent.roles.readiness
        with connection.cursor() as cursor:
            cursor.execute(_SELECT_CREDENTIAL_IDENTITY_SQL)
            identity = tuple(_one_row(cursor.fetchone(), 14))
            cursor.execute(_SELECT_IDENTITY_SQL)
            cluster = tuple(_one_row(cursor.fetchone(), 5))
        if (
            identity[:4] != (intent.expected_database, role, role, role)
            or identity[4:12] != _HISTORICAL_EXPECTED_ROLE_ATTRIBUTES["readiness"]
            or identity[12] is not None
            or identity[13]
            != (
                f"{_HISTORICAL_ROLE_MARKER_PREFIX}"
                f"{bootstrap_context.expected_database}:readiness"
            )
            or cluster
            != (
                intent.expected_database,
                role,
                role,
                role,
                import_receipt.target_system_identifier,
            )
        ):
            raise PostgresLegacySnapshotReconciliationConflict(
                "target readiness identity does not match the import receipt"
            )

    @staticmethod
    def _has_no_other_client_sessions(connection: object) -> bool:
        with connection.cursor() as cursor:
            cursor.execute(_SELECT_CLIENT_SESSIONS_SQL)
            return tuple(tuple(row) for row in cursor.fetchall()) == ()

    def _terminal_is_exact(
        self,
        context: LegacySnapshotReconciliationContext,
        import_receipt: LegacySnapshotImportReceipt,
    ) -> tuple[bool, LegacySnapshotReconciliationFindingKind | None]:
        bootstrap_context = _bootstrap_context(context.import_context)
        failed = False
        try:
            inspection = PostgresBootstrap(
                self._target_admin_connection_factory
            ).inspect_historical_terminal(bootstrap_context)
        except PostgresBootstrapDriftError:
            return (
                False,
                LegacySnapshotReconciliationFindingKind.TARGET_CATALOG_DRIFT,
            )
        except PostgresBootstrapStorageError:
            failed = True
            inspection = None
        if failed:
            raise PostgresLegacySnapshotReconciliationStorageError(
                "target terminal catalog inspection failed"
            )
        if inspection.system_identifier != import_receipt.target_system_identifier:
            return (
                False,
                LegacySnapshotReconciliationFindingKind.TARGET_IDENTITY_MISMATCH,
            )
        if not inspection.exact:
            return False, LegacySnapshotReconciliationFindingKind.TARGET_CATALOG_DRIFT
        if inspection.runtime_mode != "LEGACY" or inspection.runtime_generation != 0:
            return (
                False,
                LegacySnapshotReconciliationFindingKind.TARGET_RUNTIME_CONTROL_DRIFT,
            )
        expected_nonempty = tuple(
            sorted(
                value.name for value in import_receipt.relations if value.row_count > 0
            )
        )
        if any(
            value.removeprefix("np.") in _V2_ONLY_TABLES
            for value in inspection.nonempty_relations
        ):
            return (
                False,
                LegacySnapshotReconciliationFindingKind.TARGET_V2_STATE_PRESENT,
            )
        if (
            "np.open_positions" in inspection.nonempty_relations
            and next(
                value.row_count
                for value in import_receipt.relations
                if value.name == "np.open_positions"
            )
            == 0
        ):
            return False, LegacySnapshotReconciliationFindingKind.TARGET_OPEN_POSITION
        if inspection.nonempty_relations != expected_nonempty:
            return (
                False,
                LegacySnapshotReconciliationFindingKind.TARGET_LEGACY_ROWS_DRIFT,
            )
        return True, None

    def _reconcile(
        self,
        context: LegacySnapshotReconciliationContext,
        import_receipt: LegacySnapshotImportReceipt,
    ) -> LegacySnapshotReconciliationReceipt:
        admin = self._open(
            self._target_admin_connection_factory,
            "target admin reconciliation",
        )
        readiness = None
        try:
            self._begin(admin)
            self._require_admin_identity(admin, context, import_receipt)
            if not self._has_no_other_client_sessions(admin):
                return _blocked_receipt(
                    context,
                    import_receipt,
                    LegacySnapshotReconciliationFindingKind.TARGET_ACTIVE_SESSIONS,
                )
            readiness = self._open(
                self._target_readiness_connection_factory,
                "target readiness reconciliation",
            )
            self._begin(readiness)
            self._require_readiness_identity(readiness, context, import_receipt)
            terminal_exact, terminal_finding = self._terminal_is_exact(
                context, import_receipt
            )
            if not terminal_exact:
                return _blocked_receipt(context, import_receipt, terminal_finding)
            expected = _expected_relations(import_receipt)
            try:
                actual, invalid, _ = _scan_relations(
                    readiness,
                    context.import_context.batch_size,
                    "reconciliation",
                )
            except PostgresLegacySnapshotImportConflict:
                return _blocked_receipt(
                    context,
                    import_receipt,
                    LegacySnapshotReconciliationFindingKind.TARGET_LEGACY_ROWS_DRIFT,
                )
            if invalid or actual != expected:
                return _blocked_receipt(
                    context,
                    import_receipt,
                    LegacySnapshotReconciliationFindingKind.TARGET_LEGACY_ROWS_DRIFT,
                )
            try:
                with admin.cursor() as cursor:
                    sequences = tuple(
                        _read_sequence_next(cursor, relation)
                        for relation in _LEGACY_RELATIONS
                    )
            except PostgresLegacySnapshotImportConflict:
                return _blocked_receipt(
                    context,
                    import_receipt,
                    LegacySnapshotReconciliationFindingKind.TARGET_SEQUENCE_DRIFT,
                )
            if sequences != tuple(
                value.target_sequence_next for value in import_receipt.relations
            ):
                return _blocked_receipt(
                    context,
                    import_receipt,
                    LegacySnapshotReconciliationFindingKind.TARGET_SEQUENCE_DRIFT,
                )
            if next(
                value.row_count
                for value in import_receipt.relations
                if value.name == "np.open_positions"
            ):
                return _blocked_receipt(
                    context,
                    import_receipt,
                    LegacySnapshotReconciliationFindingKind.TARGET_OPEN_POSITION,
                )
            return self._compare_candidates(context, import_receipt, readiness)
        finally:
            if readiness is not None:
                _rollback_quietly(readiness)
                _close_quietly(readiness)
            _rollback_quietly(admin)
            _close_quietly(admin)

    def _compare_candidates(
        self,
        context: LegacySnapshotReconciliationContext,
        import_receipt: LegacySnapshotImportReceipt,
        connection: object,
    ) -> LegacySnapshotReconciliationReceipt:
        if context.collateral_asset != "USDT":
            return _blocked_receipt(
                context,
                import_receipt,
                LegacySnapshotReconciliationFindingKind.HYPOTHESIS_COLLATERAL_UNSUPPORTED,
            )
        try:
            with connection.cursor() as cursor:
                cursor.execute(_SELECT_IMPORTED_BALANCES_SQL)
                imported_rows = tuple(tuple(row) for row in cursor.fetchall())
                cursor.execute(_SELECT_LATEST_RESET_SQL)
                reset_row = cursor.fetchone()
                reset_timestamp = (
                    None if reset_row is None else _one_row(reset_row, 1)[0]
                )
                if reset_timestamp is None:
                    cursor.execute(_SELECT_TRADE_VALUES_SQL)
                    trade_rows = tuple(tuple(row) for row in cursor.fetchall())
                    cursor.execute(_SELECT_LIQUIDATION_FEES_SQL)
                    liquidation_rows = tuple(tuple(row) for row in cursor.fetchall())
                else:
                    cursor.execute(_SELECT_TRADE_VALUES_AFTER_SQL, (reset_timestamp,))
                    trade_rows = tuple(tuple(row) for row in cursor.fetchall())
                    cursor.execute(
                        _SELECT_LIQUIDATION_FEES_AFTER_SQL,
                        (reset_timestamp,),
                    )
                    liquidation_rows = tuple(tuple(row) for row in cursor.fetchall())
            if any(not _asset_is_representable(row[0]) for row in imported_rows):
                return _blocked_receipt(
                    context,
                    import_receipt,
                    LegacySnapshotReconciliationFindingKind.OPENING_EVIDENCE_UNREPRESENTABLE,
                )
            imported_balances = tuple(
                PaperAccountBalance(
                    asset,
                    _decimal_from_float4_hex(balance),
                    Decimal("0"),
                )
                for asset, balance in sorted(imported_rows, key=lambda row: row[0])
            )
            if not any(
                value.asset == context.collateral_asset for value in imported_balances
            ):
                return _blocked_receipt(
                    context,
                    import_receipt,
                    LegacySnapshotReconciliationFindingKind.COLLATERAL_MISSING,
                )
            hypothesis_pnl_float = _ordered_binary64_sum_of_float4(trade_rows, 0)
            hypothesis_trade_fees_float = _ordered_binary64_sum_of_float4(trade_rows, 1)
            hypothesis_liquidation_fees_float = _ordered_binary64_sum_of_float4(
                liquidation_rows, 0
            )
            hypothesis_realised_pnl = Decimal.from_float(hypothesis_pnl_float)
            hypothesis_trade_fees = Decimal.from_float(hypothesis_trade_fees_float)
            hypothesis_liquidation_fees = Decimal.from_float(
                hypothesis_liquidation_fees_float
            )
            starting_float = float(context.hypothesis_starting_collateral)
            if not math.isfinite(starting_float):
                raise ValueError("starting collateral is outside float semantics")
            hypothesis_balances = legacy_operator_equity_hypothesis_balances(
                context,
                hypothesis_realised_pnl,
            )
            candidates = (
                _opening_candidate(
                    context,
                    LegacyOpeningCandidateSource.IMPORTED_ACCOUNT_BALANCES,
                    imported_balances,
                ),
                _opening_candidate(
                    context,
                    LegacyOpeningCandidateSource.OPERATOR_EQUITY_HYPOTHESIS,
                    hypothesis_balances,
                ),
            )
        except TypeError, ValueError, ArithmeticError:
            return _blocked_receipt(
                context,
                import_receipt,
                LegacySnapshotReconciliationFindingKind.NUMERIC_EVIDENCE_INVALID,
            )
        findings = [
            _finding(
                LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN
            )
        ]
        try:
            quantization_required = legacy_opening_quantization_required(
                context,
                candidates,
            )
        except TypeError, ValueError, ArithmeticError:
            return _blocked_receipt(
                context,
                import_receipt,
                LegacySnapshotReconciliationFindingKind.NUMERIC_EVIDENCE_INVALID,
            )
        if quantization_required:
            findings.append(
                _finding(LegacySnapshotReconciliationFindingKind.QUANTIZATION_REQUIRED)
            )
        if (
            candidates[0].balances != candidates[1].balances
            or candidates[0].opening_payload_sha256
            != candidates[1].opening_payload_sha256
        ):
            findings.append(
                _finding(LegacySnapshotReconciliationFindingKind.CANDIDATE_MISMATCH)
            )
        findings = tuple(dict.fromkeys(findings))
        return LegacySnapshotReconciliationReceipt(
            context=context,
            import_receipt=import_receipt,
            disposition=LegacySnapshotReconciliationDisposition.DECISION_REQUIRED,
            findings=findings,
            evidence=LegacySnapshotReconciliationEvidence(
                reset_timestamp=(
                    None
                    if reset_timestamp is None
                    else reset_timestamp.isoformat(timespec="microseconds")
                ),
                hypothesis_realised_pnl=hypothesis_realised_pnl,
                hypothesis_trade_fees=hypothesis_trade_fees,
                hypothesis_liquidation_fees=hypothesis_liquidation_fees,
                candidates=candidates,
            ),
            target_system_identifier=import_receipt.target_system_identifier,
            source_canonical_sha256=import_receipt.source_canonical_sha256,
            config_document_sha256=context.config_document_sha256,
            import_receipt_sha256=context.import_receipt_sha256,
        )


__all__ = [
    "PostgresLegacySnapshotReconciliation",
    "PostgresLegacySnapshotReconciliationConflict",
    "PostgresLegacySnapshotReconciliationInputError",
    "PostgresLegacySnapshotReconciliationStorageError",
]
