"""Dormant PostgreSQL boundary for one locked paper-runtime activation."""

from collections.abc import Callable

import psycopg2

from trading.application.paper_account_readiness import (
    PaperAccountReadinessDisposition,
)
from trading.application.paper_runtime_activation import (
    PaperRuntimeActivationBlocked,
    PaperRuntimeActivationBusy,
    PaperRuntimeActivationCommitUnknown,
    PaperRuntimeActivationConflict,
    PaperRuntimeActivationContext,
    PaperRuntimeActivationDisposition,
    PaperRuntimeActivationReceipt,
    PaperRuntimeActivationResult,
)
from trading.persistence.order_position_journal import (
    JournalRepositoryError,
    PostgresOrderPositionJournal,
)
from trading.persistence.paper_account_readiness import (
    PaperAccountReadinessError,
    _activation_catalog_is_authoritative,
    _collect_paper_account_readiness,
    _runtime_generation_evidence_is_exact,
)

_ACTIVATION_TRANSACTION_SQL = "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
_SET_LOCK_TIMEOUT_SQL = "SET LOCAL lock_timeout = '1s'"
_ACQUIRE_ACTIVATION_FENCE_SQL = "SELECT np.acquire_paper_runtime_activation_fence()"
_SELECT_RUNTIME_CONTROL_SQL = """
SELECT mode, runtime_generation
FROM np.paper_runtime_control
WHERE control_key = TRUE
"""
_SELECT_ACTIVATION_ID_SQL = """
SELECT
    runtime_generation,
    activation_id,
    execution_scope,
    account_key,
    owner_generation,
    opening_version,
    opening_payload_sha256
FROM np.paper_runtime_generations
WHERE activation_id = %s
"""
_ACTIVATE_RUNTIME_GENERATION_SQL = """
SELECT mode, runtime_generation
FROM np.activate_paper_runtime_generation(%s, %s, %s, %s, %s, %s, %s, %s)
"""
_CHECK_CONSTRAINTS_SQL = "SET CONSTRAINTS ALL IMMEDIATE"

_BUSY_SQLSTATES = frozenset({"40P01", "55P03"})
_CONFLICT_SQLSTATES = frozenset({"23505", "PT001"})
_RUNTIME_CONTROL_MODES = frozenset({"LEGACY", "SHADOW", "PAUSED", "ACTIVE"})
_POSTGRES_BIGINT_MAX = (1 << 63) - 1


class PaperRuntimeActivationStorageError(RuntimeError):
    """Raised when an activation is known not to have committed."""


def _row(value: object, length: int, source: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise PaperRuntimeActivationStorageError(
            f"PostgreSQL returned an invalid {source} row"
        )
    return tuple(value)


class PostgresPaperRuntimeActivation:
    """Activate or replay one epoch through dormant database capabilities."""

    def __init__(self, connection_factory: Callable[[], object]) -> None:
        self._journal_boundary = PostgresOrderPositionJournal(connection_factory)

    def activate(
        self,
        context: PaperRuntimeActivationContext,
        /,
    ) -> PaperRuntimeActivationResult:
        """Return a committed epoch, its exact replay, or locked blocking evidence."""
        if type(context) is not PaperRuntimeActivationContext:
            raise TypeError("context must be a PaperRuntimeActivationContext")
        try:
            connection = self._journal_boundary._connection()
        except JournalRepositoryError as exc:
            raise PaperRuntimeActivationStorageError(
                "could not open a paper runtime activation connection"
            ) from exc

        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_ACTIVATION_TRANSACTION_SQL)
                    cursor.execute(_SET_LOCK_TIMEOUT_SQL)
                    cursor.execute(_ACQUIRE_ACTIVATION_FENCE_SQL)
                    result: PaperRuntimeActivationResult | None = None
                    if not _activation_catalog_is_authoritative(cursor):
                        result = self._blocked_assessment(cursor, context)
                    else:
                        control = self._runtime_control(cursor)
                        if control is None or not (
                            _activation_catalog_is_authoritative(cursor)
                        ):
                            result = self._blocked_assessment(cursor, context)
                            replay = None
                        else:
                            replay = self._activation_replay(
                                cursor,
                                context,
                                control=control,
                            )
                        if replay is not None:
                            result = replay
                        if result is None:
                            self._require_expected_control(control, context)
                            assessment = _collect_paper_account_readiness(
                                cursor,
                                context=context.readiness,
                                required_runtime_mode=context.source.value,
                                lock_replayed_state=False,
                            )
                            if assessment.disposition is not (
                                PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
                            ):
                                result = PaperRuntimeActivationBlocked(
                                    context, assessment
                                )
                            else:
                                result = self._activate_locked(cursor, context)
                    if result is None:
                        raise PaperRuntimeActivationStorageError(
                            "paper runtime activation produced no result"
                        )
            except (
                PaperRuntimeActivationBusy,
                PaperRuntimeActivationConflict,
                PaperRuntimeActivationStorageError,
            ):
                raise
            except psycopg2.Error as exc:
                if getattr(exc, "pgcode", None) in _BUSY_SQLSTATES:
                    raise PaperRuntimeActivationBusy(context) from exc
                if getattr(exc, "pgcode", None) in _CONFLICT_SQLSTATES:
                    raise PaperRuntimeActivationConflict(context) from exc
                raise PaperRuntimeActivationStorageError(
                    "PostgreSQL rejected the paper runtime activation"
                ) from exc
            except PaperAccountReadinessError as exc:
                raise PaperRuntimeActivationStorageError(
                    "paper runtime activation readiness failed"
                ) from exc
            except Exception as exc:
                raise PaperRuntimeActivationStorageError(
                    "paper runtime activation failed before commit"
                ) from exc

            if type(result) is PaperRuntimeActivationBlocked or (
                type(result) is PaperRuntimeActivationReceipt
                and result.disposition is PaperRuntimeActivationDisposition.REPLAYED
            ):
                try:
                    connection.rollback()
                except Exception as exc:
                    raise PaperRuntimeActivationStorageError(
                        "read-only paper runtime activation could not roll back"
                    ) from exc
                return result

            try:
                connection.commit()
            except Exception as exc:
                raise PaperRuntimeActivationCommitUnknown(context) from exc
            return result
        except Exception:
            self._journal_boundary._rollback(connection)
            raise
        finally:
            self._journal_boundary._close(connection)

    @staticmethod
    def _blocked_assessment(
        cursor: object,
        context: PaperRuntimeActivationContext,
    ) -> PaperRuntimeActivationBlocked:
        assessment = _collect_paper_account_readiness(
            cursor,
            context=context.readiness,
            required_runtime_mode=context.source.value,
            lock_replayed_state=False,
        )
        if (
            assessment.disposition
            is PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
        ):
            raise PaperRuntimeActivationStorageError(
                "non-authoritative activation state produced prepared evidence"
            )
        return PaperRuntimeActivationBlocked(context, assessment)

    @staticmethod
    def _runtime_control(cursor: object) -> tuple[object, ...] | None:
        cursor.execute(_SELECT_RUNTIME_CONTROL_SQL)
        raw = cursor.fetchone()
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            return None
        row = tuple(raw)
        if (
            type(row[0]) is not str
            or row[0] not in _RUNTIME_CONTROL_MODES
            or type(row[1]) is not int
            or not 0 <= row[1] <= _POSTGRES_BIGINT_MAX
        ):
            return None
        return row

    @staticmethod
    def _activation_replay(
        cursor: object,
        context: PaperRuntimeActivationContext,
        *,
        control: tuple[object, ...],
    ) -> PaperRuntimeActivationReceipt | None:
        cursor.execute(_SELECT_ACTIVATION_ID_SQL, (context.activation_id,))
        raw = cursor.fetchone()
        if raw is None:
            return None
        row = _row(raw, 7, "paper runtime generation")
        readiness = context.readiness
        if (
            type(control[0]) is not str
            or control[0] not in {"PAUSED", "ACTIVE"}
            or type(control[1]) is not int
            or control[1] < context.target_runtime_generation
            or not _runtime_generation_evidence_is_exact(
                cursor,
                context=readiness,
                runtime_mode=control[0],
                runtime_generation=control[1],
            )
        ):
            raise PaperRuntimeActivationConflict(context)
        if (
            type(row[0]) is not int
            or row[0] != context.target_runtime_generation
            or type(row[1]) is not str
            or row[1] != context.activation_id
            or type(row[2]) is not str
            or row[2] != readiness.execution_scope
            or type(row[3]) is not str
            or row[3] != readiness.account_key
            or type(row[4]) is not int
            or row[4] != readiness.owner_generation
            or type(row[5]) is not int
            or row[5] != 1
            or type(row[6]) is not str
            or row[6] != readiness.opening_payload_sha256
        ):
            raise PaperRuntimeActivationConflict(context)
        return PaperRuntimeActivationReceipt(
            context=context,
            runtime_generation=row[0],
            disposition=PaperRuntimeActivationDisposition.REPLAYED,
        )

    @staticmethod
    def _require_expected_control(
        row: tuple[object, ...],
        context: PaperRuntimeActivationContext,
    ) -> None:
        if (
            type(row[0]) is not str
            or row[0] != context.source.value
            or type(row[1]) is not int
            or row[1] != context.expected_runtime_generation
        ):
            raise PaperRuntimeActivationConflict(context)

    @staticmethod
    def _activate_locked(
        cursor: object,
        context: PaperRuntimeActivationContext,
    ) -> PaperRuntimeActivationReceipt:
        readiness = context.readiness
        target = context.target_runtime_generation
        cursor.execute(
            _ACTIVATE_RUNTIME_GENERATION_SQL,
            (
                context.source.value,
                context.expected_runtime_generation,
                target,
                context.activation_id,
                readiness.execution_scope,
                readiness.account_key,
                readiness.owner_generation,
                readiness.opening_payload_sha256,
            ),
        )
        activated = cursor.fetchone()
        if activated is None:
            raise PaperRuntimeActivationStorageError(
                "paper runtime activation capability returned no result"
            )
        activated_row = _row(activated, 2, "activated runtime control")
        if activated_row != ("ACTIVE", target):
            raise PaperRuntimeActivationStorageError(
                "paper runtime control returned an invalid activation"
            )
        cursor.execute(_CHECK_CONSTRAINTS_SQL)
        return PaperRuntimeActivationReceipt(
            context=context,
            runtime_generation=target,
            disposition=PaperRuntimeActivationDisposition.ACTIVATED,
        )


__all__ = [
    "PaperRuntimeActivationStorageError",
    "PostgresPaperRuntimeActivation",
]
