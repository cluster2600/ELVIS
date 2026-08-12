"""Unwired atomic PostgreSQL owner for journaled paper-account batches.

One fresh transaction locks the active runtime control and pinned epoch before
the preprovisioned account and its target position, derives every journal and
account fact in memory, and either rolls back a typed admission rejection or
commits the complete generation-stamped batch once.
"""

from dataclasses import dataclass
from typing import Callable

import psycopg2

from trading.application.durable_submission import (
    DurablePaperAccountSubmissionReceipt,
    DurableSubmissionDisposition,
    PaperAccountSubmissionCommitUnknown,
    PaperAccountSubmissionContext,
    PaperAccountSubmissionReconciliationRequired,
    PaperAccountSubmissionRejected,
    PaperAccountSubmissionResult,
    PaperAccountSubmissionRuntimeUnavailable,
    PaperSubmissionPlan,
    PaperSubmissionPlanner,
)
from trading.domain.order_lifecycle import (
    InvalidOrderTransition,
    new_order_lifecycle,
    reduce_order_lifecycle,
)
from trading.domain.paper_accounting import (
    InvalidPaperAccountTransition,
    PaperAccountAdmission,
    PaperAccountAdmissionDisposition,
    admit_paper_settlement,
)
from trading.domain.paper_economics import PaperFillRecord
from trading.domain.paper_settlement import (
    InvalidPaperSettlement,
    PaperSettlementCheckpoint,
    PaperSettlementDisposition,
    settle_paper_fill,
)
from trading.domain.positions import (
    InvalidPositionTransition,
    PositionFill,
    new_position,
    position_fill_from_lifecycle,
    reduce_position,
)
from trading.persistence.atomic_paper_submission_owner import (
    _receipt_for_order,
    _require_terminal_stream,
    _reservation_conflict,
)
from trading.persistence.journal_codec import (
    JournalEncodeError,
    encode_order_lifecycle_event,
    encode_position_instruction,
)
from trading.persistence.order_position_journal import (
    _ADVANCE_STREAM_SQL,
    _BIGINT_MAX,
    _INSERT_EVENT_SQL,
    _INSERT_ORDER_SQL,
    _INSERT_STREAM_SQL,
    _SELECT_VENUE_OWNER_SQL,
    _SET_VENUE_ID_SQL,
    _WRITE_TRANSACTION_SQL,
    JournalConflictError,
    JournalConflictKind,
    JournalNotFoundError,
    JournalReplayError,
    JournalRepositoryError,
    JournalStorageError,
    PostgresOrderPositionJournal,
    _checked_stored_datetime,
    _find_replayed_order,
    _replay_stream,
    _row,
    _translate_database_error,
)
from trading.persistence.paper_account_journal import (
    PaperAccountConflictError,
    PaperAccountJournalError,
    PaperAccountNotFoundError,
    PaperAccountReplayError,
    PaperAccountStorageError,
    ReplayedPaperAccount,
    _replay_account_locked,
)
from trading.persistence.paper_account_journal_codec import (
    EncodedPaperAccountBatch,
    EncodedPaperAccountSettlement,
    PaperAccountBatchFill,
    PaperAccountBatchManifest,
    encode_paper_account_batch,
    encode_paper_account_settlement,
)

_SELECT_RUNTIME_CONTROL_FOR_SHARE_SQL = """
SELECT mode, runtime_generation
FROM np.paper_runtime_control
WHERE control_key = TRUE
FOR SHARE
"""

_SELECT_RUNTIME_GENERATION_FOR_SHARE_SQL = """
SELECT
    runtime_generation,
    execution_scope,
    account_key,
    owner_generation,
    opening_version,
    opening_payload_sha256
FROM np.paper_runtime_generations
WHERE runtime_generation = %s
FOR SHARE
"""

_INSERT_ACCOUNT_SETTLEMENT_SQL = """
INSERT INTO np.paper_account_settlements (
    account_key,
    account_version,
    client_order_id,
    fill_ordinal,
    batch_first_account_version,
    batch_submission_position_version,
    batch_fill_count,
    collateral_asset,
    position_key,
    position_version,
    event_id,
    trade_id,
    event_type,
    event_payload_sha256,
    symbol,
    base_asset,
    quote_asset,
    instrument_version,
    settlement_version,
    settlement_payload,
    settlement_payload_sha256
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s::jsonb, %s
)
RETURNING account_version
"""

_INSERT_ACCOUNT_POSTING_SQL = """
INSERT INTO np.paper_account_postings (
    account_key,
    account_version,
    posting_ordinal,
    asset,
    bucket,
    amount_decimal
) VALUES (%s, %s, %s, %s, %s, %s)
RETURNING posting_ordinal
"""

_UPSERT_ACCOUNT_BALANCE_SQL = """
INSERT INTO np.paper_account_balances (
    account_key,
    asset,
    available_decimal,
    reserved_decimal
) VALUES (%s, %s, %s, %s)
ON CONFLICT (account_key, asset) DO UPDATE SET
    available_decimal = EXCLUDED.available_decimal,
    reserved_decimal = EXCLUDED.reserved_decimal,
    updated_at = clock_timestamp()
RETURNING asset
"""

_DELETE_POSITION_RESERVATION_SQL = """
DELETE FROM np.paper_margin_reservations
WHERE account_key = %s AND position_key = %s
RETURNING position_key
"""

_INSERT_POSITION_RESERVATION_SQL = """
INSERT INTO np.paper_margin_reservations (
    account_key,
    execution_scope,
    position_key,
    amount_decimal
) VALUES (%s, %s, %s, %s)
RETURNING position_key
"""

_ADVANCE_ACCOUNT_STREAM_SQL = """
UPDATE np.paper_account_streams
SET
    account_version = %s,
    account_state = %s,
    updated_at = clock_timestamp()
WHERE account_key = %s AND account_version = %s
RETURNING account_version
"""

_INSERT_ACCOUNT_BATCH_SQL = """
INSERT INTO np.paper_account_batch_manifests (
    account_key,
    client_order_id,
    execution_scope,
    owner_generation,
    opening_version,
    opening_payload_sha256,
    position_key,
    instruction_payload_sha256,
    submission_event_id,
    submission_event_type,
    submission_position_version,
    submission_observed_at,
    submission_event_payload_sha256,
    first_account_version,
    last_account_version,
    last_position_version,
    fill_count,
    batch_version,
    batch_payload,
    batch_payload_sha256,
    runtime_generation
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s::jsonb, %s, %s
)
RETURNING client_order_id
"""

_CHECK_CONSTRAINTS_SQL = "SET CONSTRAINTS ALL IMMEDIATE"


@dataclass(frozen=True, slots=True)
class _DerivedAccountBatch:
    plan: PaperSubmissionPlan
    admissions: tuple[PaperAccountAdmission, ...]
    encoded_events: tuple[tuple[str, object], ...]
    encoded_settlements: tuple[EncodedPaperAccountSettlement, ...]
    encoded_batch: EncodedPaperAccountBatch

    @property
    def account_versions(self) -> tuple[int, ...]:
        return tuple(admission.account_version for admission in self.admissions)


class PostgresAtomicPaperAccountOwner:
    """Commit, replay, or reject one pinned-generation account batch atomically."""

    def __init__(
        self,
        connection_factory: Callable[[], object],
        planner: PaperSubmissionPlanner,
        runtime_generation: int,
    ) -> None:
        if type(runtime_generation) is not int:
            raise TypeError("runtime_generation must be an integer")
        if runtime_generation < 1:
            raise ValueError("runtime_generation must be positive")
        if runtime_generation > _BIGINT_MAX:
            raise ValueError("runtime_generation exceeds the durable storage limit")
        self._journal_boundary = PostgresOrderPositionJournal(connection_factory)
        if not callable(getattr(planner, "plan", None)):
            raise TypeError("planner must provide plan(attempt)")
        self._planner = planner
        self._runtime_generation = runtime_generation

    def execute(
        self,
        context: PaperAccountSubmissionContext,
        /,
    ) -> PaperAccountSubmissionResult:
        """Own the complete account-first transaction for one paper batch."""
        if type(context) is not PaperAccountSubmissionContext:
            raise TypeError("context must be a PaperAccountSubmissionContext")
        if context.runtime_generation > self._runtime_generation:
            raise PaperAccountSubmissionRuntimeUnavailable(context)
        try:
            encoded_instruction = encode_position_instruction(
                context.attempt.instruction
            )
        except JournalEncodeError as exc:
            raise PaperAccountStorageError(
                "paper-account submission context is not representable"
            ) from exc

        try:
            connection = self._journal_boundary._connection()
        except JournalStorageError as exc:
            raise PaperAccountStorageError(
                "could not open an atomic paper-account connection"
            ) from exc
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_WRITE_TRANSACTION_SQL)
                    self._lock_runtime_control(cursor, context)
                    result = self._execute_locked(
                        cursor,
                        context=context,
                        encoded_instruction=encoded_instruction,
                    )
            except (
                JournalConflictError,
                PaperAccountSubmissionReconciliationRequired,
                PaperAccountSubmissionRuntimeUnavailable,
            ):
                raise
            except psycopg2.Error as exc:
                translated = _translate_database_error(exc)
                if isinstance(translated, JournalConflictError):
                    raise translated from exc
                raise PaperAccountStorageError(
                    "PostgreSQL rejected the atomic paper-account submission"
                ) from exc
            except Exception as exc:
                raise PaperAccountStorageError(
                    "atomic paper-account submission failed before commit"
                ) from exc

            if type(result) is PaperAccountSubmissionRejected:
                try:
                    connection.rollback()
                except Exception as exc:
                    raise PaperAccountStorageError(
                        "rejected paper-account submission could not roll back"
                    ) from exc
                return result

            try:
                connection.commit()
            except Exception as exc:
                raise PaperAccountSubmissionCommitUnknown(context) from exc
            return result
        except Exception:
            self._journal_boundary._rollback(connection)
            raise
        finally:
            self._journal_boundary._close(connection)

    def _execute_locked(
        self,
        cursor: object,
        *,
        context: PaperAccountSubmissionContext,
        encoded_instruction: object,
    ) -> PaperAccountSubmissionResult:
        runtime_generation = self._locked_runtime_generation(cursor, context)
        account = self._locked_account(cursor, context)
        self._require_runtime_generation(
            runtime_generation,
            context=context,
            account=account,
        )
        if any(
            manifest.runtime_generation is None
            or manifest.runtime_generation > self._runtime_generation
            for manifest in account.batches
        ):
            raise PaperAccountSubmissionReconciliationRequired(context)
        target_manifest = next(
            (
                manifest
                for manifest in account.batches
                if manifest.client_order_id == context.client_order_id
            ),
            None,
        )
        if (
            target_manifest is not None
            and target_manifest.runtime_generation != context.runtime_generation
        ):
            raise PaperAccountSubmissionReconciliationRequired(context)
        if (
            target_manifest is None
            and context.runtime_generation != self._runtime_generation
        ):
            raise PaperAccountSubmissionRuntimeUnavailable(context)
        replay, existing = self._locked_position(
            cursor,
            context=context,
            encoded_instruction=encoded_instruction,
            allow_create=target_manifest is None,
        )
        self._require_account_owned_position(account, replay, context)

        if existing is not None:
            if existing.encoded != encoded_instruction:
                if target_manifest is not None:
                    raise PaperAccountSubmissionReconciliationRequired(context)
                raise JournalConflictError(
                    JournalConflictKind.CLIENT_ORDER_ID,
                    "client order identity is bound to another instruction",
                )
            return self._replayed_receipt(
                account,
                replay,
                context=context,
                encoded_instruction=encoded_instruction,
            )

        if any(
            manifest.client_order_id == context.client_order_id
            for manifest in account.batches
        ):
            raise PaperAccountSubmissionReconciliationRequired(context)

        plan = self._planner.plan(context.attempt)
        if type(plan) is not PaperSubmissionPlan:
            raise TypeError("planner must return a PaperSubmissionPlan")
        if plan.attempt is not context.attempt:
            raise ValueError("planner must retain the exact attempt object")

        derived = self._derive_batch(
            account,
            replay,
            context=context,
            encoded_instruction=encoded_instruction,
            plan=plan,
        )
        if type(derived) is PaperAccountSubmissionRejected:
            return derived

        self._store_journal(
            cursor,
            replay=replay,
            encoded_instruction=encoded_instruction,
            batch=derived,
        )
        self._store_account(
            cursor,
            before=account,
            context=context,
            batch=derived,
        )
        cursor.execute(_CHECK_CONSTRAINTS_SQL)

        try:
            updated_account = _replay_account_locked(
                cursor,
                execution_scope=context.execution_scope,
                account_key=context.account_key,
                lock=True,
            )
            updated_position = _replay_stream(
                cursor,
                execution_scope=context.execution_scope,
                position_key=encoded_instruction.position_key,
                lock=True,
            )
        except (JournalRepositoryError, PaperAccountJournalError) as exc:
            raise JournalStorageError(
                "stored paper-account batch failed its pre-commit replay"
            ) from exc
        return self._receipt_from_replay(
            updated_account,
            updated_position,
            context=context,
            encoded_instruction=encoded_instruction,
            disposition=DurableSubmissionDisposition.COMMITTED,
        )

    def _lock_runtime_control(
        self,
        cursor: object,
        context: PaperAccountSubmissionContext,
    ) -> None:
        cursor.execute(_SELECT_RUNTIME_CONTROL_FOR_SHARE_SQL)
        raw = cursor.fetchone()
        if raw is None:
            raise PaperAccountSubmissionRuntimeUnavailable(context)
        try:
            row = _row(raw, 2, "paper runtime control")
        except JournalRepositoryError as exc:
            raise PaperAccountSubmissionRuntimeUnavailable(context) from exc
        if (
            type(row[0]) is not str
            or row[0] != "ACTIVE"
            or type(row[1]) is not int
            or row[1] != self._runtime_generation
        ):
            raise PaperAccountSubmissionRuntimeUnavailable(context)

    def _locked_runtime_generation(
        self,
        cursor: object,
        context: PaperAccountSubmissionContext,
    ) -> tuple[object, ...]:
        cursor.execute(
            _SELECT_RUNTIME_GENERATION_FOR_SHARE_SQL,
            (self._runtime_generation,),
        )
        raw = cursor.fetchone()
        if raw is None:
            raise PaperAccountSubmissionRuntimeUnavailable(context)
        try:
            row = _row(raw, 6, "paper runtime generation")
        except JournalRepositoryError as exc:
            raise PaperAccountSubmissionRuntimeUnavailable(context) from exc
        return row

    def _require_runtime_generation(
        self,
        row: tuple[object, ...],
        *,
        context: PaperAccountSubmissionContext,
        account: ReplayedPaperAccount,
    ) -> None:
        if (
            type(row[0]) is not int
            or row[0] != self._runtime_generation
            or type(row[1]) is not str
            or row[1] != context.execution_scope
            or type(row[2]) is not str
            or row[2] != context.account_key
            or type(row[3]) is not int
            or row[3] != account.owner_generation
            or type(row[4]) is not int
            or row[4] != 1
            or type(row[5]) is not str
            or row[5] != account.opening_payload_sha256
        ):
            raise PaperAccountSubmissionRuntimeUnavailable(context)

    @staticmethod
    def _locked_account(
        cursor: object,
        context: PaperAccountSubmissionContext,
    ) -> ReplayedPaperAccount:
        try:
            return _replay_account_locked(
                cursor,
                execution_scope=context.execution_scope,
                account_key=context.account_key,
                lock=True,
            )
        except (
            PaperAccountConflictError,
            PaperAccountNotFoundError,
            PaperAccountReplayError,
        ) as exc:
            raise PaperAccountSubmissionReconciliationRequired(context) from exc

    @staticmethod
    def _locked_position(
        cursor: object,
        *,
        context: PaperAccountSubmissionContext,
        encoded_instruction: object,
        allow_create: bool,
    ) -> tuple[object, object | None]:
        inserted = None
        if allow_create:
            cursor.execute(
                _INSERT_STREAM_SQL,
                (encoded_instruction.position_key, context.execution_scope),
            )
            inserted = cursor.fetchone()
            if (
                inserted is not None
                and _row(inserted, 1, "inserted position stream")[0]
                != encoded_instruction.position_key
            ):
                raise JournalStorageError("PostgreSQL returned another position stream")
        try:
            replay = _replay_stream(
                cursor,
                execution_scope=context.execution_scope,
                position_key=encoded_instruction.position_key,
                lock=True,
                allow_empty=inserted is not None,
            )
        except (JournalNotFoundError, JournalReplayError) as exc:
            raise PaperAccountSubmissionReconciliationRequired(context) from exc
        return replay, replay.orders_by_client.get(encoded_instruction.client_order_id)

    @staticmethod
    def _require_account_owned_position(
        account: ReplayedPaperAccount,
        replay: object,
        context: PaperAccountSubmissionContext,
    ) -> None:
        expected = {
            manifest.client_order_id
            for manifest in account.batches
            if manifest.position_key == context.attempt.instruction.position_key
        }
        if set(replay.orders_by_client) != expected:
            raise PaperAccountSubmissionReconciliationRequired(context)
        try:
            _require_terminal_stream(replay, context.attempt)
        except Exception as exc:
            raise PaperAccountSubmissionReconciliationRequired(context) from exc

    def _derive_batch(
        self,
        account: ReplayedPaperAccount,
        replay: object,
        *,
        context: PaperAccountSubmissionContext,
        encoded_instruction: object,
        plan: PaperSubmissionPlan,
    ) -> _DerivedAccountBatch | PaperAccountSubmissionRejected:
        first_position_version = replay.projection.stream_version + 1
        final_position_version = replay.projection.stream_version + 1 + len(plan.fills)
        first_account_version = len(account.account.records) + 1
        final_account_version = len(account.account.records) + len(plan.fills)
        if final_position_version > _BIGINT_MAX or final_account_version > _BIGINT_MAX:
            raise JournalConflictError(
                JournalConflictKind.STREAM_VERSION,
                "journal or account stream version is exhausted",
            )

        lifecycle = new_order_lifecycle(plan.attempt.instruction.order_intent)
        position = replay.projection.position
        current_account = account.account
        checkpoint = None
        for record in current_account.records:
            if (
                record.settlement.record.position_fill.instruction.position_key
                == encoded_instruction.position_key
            ):
                checkpoint = record.settlement.after
        if checkpoint is not None and not isinstance(
            checkpoint, PaperSettlementCheckpoint
        ):
            raise PaperAccountReplayError("account lost its position checkpoint")

        try:
            lifecycle = reduce_order_lifecycle(lifecycle, plan.submission)
            encoded_events = (
                (
                    plan.attempt.event_id,
                    encode_order_lifecycle_event(plan.submission),
                ),
            )
            admissions = []
            encoded_settlements = []
            batch_fills = []
            for ordinal, candidate in enumerate(plan.fills, start=1):
                lifecycle = reduce_order_lifecycle(lifecycle, candidate.fill)
                position_fill = position_fill_from_lifecycle(
                    plan.attempt.instruction,
                    lifecycle,
                    candidate.fill,
                )
                position = (
                    new_position(position_fill)
                    if position is None
                    else reduce_position(position, position_fill)
                )
                position_version = first_position_version + ordinal
                account_version = first_account_version + ordinal - 1
                record = PaperFillRecord(
                    position_version=position_version,
                    event_id=candidate.event_id,
                    position_fill=PositionFill(
                        plan.attempt.instruction,
                        candidate.fill,
                    ),
                )
                settlement = settle_paper_fill(
                    context.instrument,
                    checkpoint,
                    record,
                )
                if settlement.disposition is not PaperSettlementDisposition.APPLIED:
                    raise InvalidPaperSettlement(
                        "a new batch produced a replayed settlement"
                    )
                admission = admit_paper_settlement(
                    current_account,
                    account_version,
                    settlement,
                )
                if admission.disposition is PaperAccountAdmissionDisposition.REJECTED:
                    return PaperAccountSubmissionRejected(
                        context,
                        candidate.event_id,
                        admission.reasons,
                    )
                if (
                    admission.disposition
                    is not PaperAccountAdmissionDisposition.APPLIED
                ):
                    raise InvalidPaperAccountTransition(
                        "a new batch produced a replayed account admission"
                    )
                encoded_event = encode_order_lifecycle_event(candidate.fill)
                encoded_settlement = encode_paper_account_settlement(admission)
                encoded_events += ((candidate.event_id, encoded_event),)
                admissions.append(admission)
                encoded_settlements.append(encoded_settlement)
                batch_fills.append(
                    PaperAccountBatchFill(
                        position_key=encoded_instruction.position_key,
                        client_order_id=encoded_instruction.client_order_id,
                        event_id=candidate.event_id,
                        trade_id=candidate.fill.trade_id,
                        position_version=position_version,
                        account_version=account_version,
                        event_payload_sha256=encoded_event.event_payload_sha256,
                        account_settlement_payload_sha256=(
                            encoded_settlement.settlement_payload_sha256
                        ),
                    )
                )
                current_account = admission.after
                checkpoint = settlement.after
        except (
            InvalidOrderTransition,
            InvalidPaperAccountTransition,
            InvalidPaperSettlement,
            InvalidPositionTransition,
            JournalEncodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise JournalConflictError(
                JournalConflictKind.INVALID_TRANSITION,
                "planned batch contradicts the journal or account projection",
            ) from exc

        manifest = PaperAccountBatchManifest(
            execution_scope=context.execution_scope,
            account_key=context.account_key,
            owner_generation=account.owner_generation,
            position_key=encoded_instruction.position_key,
            client_order_id=encoded_instruction.client_order_id,
            instruction_payload_sha256=(encoded_instruction.instruction_payload_sha256),
            submission_event_id=plan.attempt.event_id,
            submission_position_version=first_position_version,
            submission_observed_at=plan.submission.observed_at,
            submission_event_payload_sha256=(encoded_events[0][1].event_payload_sha256),
            fills=tuple(batch_fills),
            runtime_generation=context.runtime_generation,
        )
        return _DerivedAccountBatch(
            plan=plan,
            admissions=tuple(admissions),
            encoded_events=encoded_events,
            encoded_settlements=tuple(encoded_settlements),
            encoded_batch=encode_paper_account_batch(manifest),
        )

    @staticmethod
    def _store_journal(
        cursor: object,
        *,
        replay: object,
        encoded_instruction: object,
        batch: _DerivedAccountBatch,
    ) -> None:
        cursor.execute(
            _INSERT_ORDER_SQL,
            (
                encoded_instruction.client_order_id,
                encoded_instruction.decision_id,
                encoded_instruction.position_key,
                batch.plan.attempt.execution_scope,
                encoded_instruction.symbol,
                encoded_instruction.position_effect,
                encoded_instruction.instruction_version,
                encoded_instruction.instruction_payload,
                encoded_instruction.instruction_payload_sha256,
            ),
        )
        inserted_order = cursor.fetchone()
        if inserted_order is None:
            _reservation_conflict(
                cursor,
                encoded=encoded_instruction,
                execution_scope=batch.plan.attempt.execution_scope,
            )
        _checked_stored_datetime(
            _row(inserted_order, 1, "inserted order")[0],
            "registered_at",
        )

        venue_order_id = batch.plan.submission.venue_order_id
        cursor.execute(
            _SELECT_VENUE_OWNER_SQL,
            (
                batch.plan.attempt.execution_scope,
                encoded_instruction.symbol,
                venue_order_id,
            ),
        )
        owner = cursor.fetchone()
        if owner is not None and _row(owner, 1, "venue owner")[0] != (
            encoded_instruction.client_order_id
        ):
            raise JournalConflictError(
                JournalConflictKind.VENUE_ORDER_ID,
                "venue identity belongs to another order",
            )
        cursor.execute(
            _SET_VENUE_ID_SQL,
            (venue_order_id, encoded_instruction.client_order_id),
        )
        updated_venue = cursor.fetchone()
        if (
            updated_venue is None
            or _row(
                updated_venue,
                1,
                "updated venue identity",
            )[0]
            != venue_order_id
        ):
            raise JournalStorageError("venue identity was not updated")

        first_version = replay.projection.stream_version + 1
        final_version = replay.projection.stream_version + len(batch.encoded_events)
        cursor.execute(
            _ADVANCE_STREAM_SQL,
            (
                final_version,
                encoded_instruction.position_key,
                replay.projection.stream_version,
            ),
        )
        advanced = cursor.fetchone()
        if advanced is None or _row(advanced, 1, "advanced stream")[0] != (
            final_version
        ):
            raise JournalStorageError("position stream version did not advance")

        for offset, (event_id, encoded_event) in enumerate(batch.encoded_events):
            cursor.execute(
                _INSERT_EVENT_SQL,
                (
                    encoded_instruction.position_key,
                    first_version + offset,
                    encoded_event.client_order_id,
                    event_id,
                    encoded_event.event_type,
                    encoded_event.event_version,
                    encoded_event.event_payload,
                    encoded_event.event_payload_sha256,
                    encoded_event.trade_id,
                    encoded_event.occurred_at,
                ),
            )
            recorded = cursor.fetchone()
            if recorded is None:
                raise JournalStorageError(
                    "PostgreSQL did not return the appended event"
                )
            _checked_stored_datetime(
                _row(recorded, 1, "inserted event")[0],
                "recorded_at",
            )

    @staticmethod
    def _store_account(
        cursor: object,
        *,
        before: ReplayedPaperAccount,
        context: PaperAccountSubmissionContext,
        batch: _DerivedAccountBatch,
    ) -> None:
        encoded_batch = batch.encoded_batch
        acknowledgement = batch.encoded_events[0][1]
        cursor.execute(
            _INSERT_ACCOUNT_BATCH_SQL,
            (
                encoded_batch.account_key,
                encoded_batch.client_order_id,
                encoded_batch.execution_scope,
                encoded_batch.owner_generation,
                1,
                before.opening_payload_sha256,
                encoded_batch.position_key,
                encoded_batch.instruction_payload_sha256,
                encoded_batch.submission_event_id,
                acknowledgement.event_type,
                encoded_batch.submission_position_version,
                encoded_batch.submission_observed_at,
                acknowledgement.event_payload_sha256,
                encoded_batch.first_account_version,
                encoded_batch.last_account_version,
                encoded_batch.last_position_version,
                encoded_batch.fill_count,
                encoded_batch.batch_version,
                encoded_batch.batch_payload,
                encoded_batch.batch_payload_sha256,
                encoded_batch.runtime_generation,
            ),
        )
        inserted_batch = cursor.fetchone()
        if (
            inserted_batch is None
            or _row(
                inserted_batch,
                1,
                "inserted paper-account batch",
            )[0]
            != encoded_batch.client_order_id
        ):
            raise JournalStorageError("paper-account batch was not inserted")

        first_account_version = batch.account_versions[0]
        for ordinal, (admission, encoded) in enumerate(
            zip(batch.admissions, batch.encoded_settlements),
            start=1,
        ):
            encoded_event = batch.encoded_events[ordinal][1]
            cursor.execute(
                _INSERT_ACCOUNT_SETTLEMENT_SQL,
                (
                    encoded.account_key,
                    encoded.account_version,
                    encoded.client_order_id,
                    ordinal,
                    first_account_version,
                    encoded_batch.submission_position_version,
                    len(batch.admissions),
                    encoded.collateral_asset,
                    encoded.position_key,
                    encoded.position_version,
                    encoded.event_id,
                    encoded.trade_id,
                    encoded_event.event_type,
                    encoded_event.event_payload_sha256,
                    encoded.symbol,
                    encoded.base_asset,
                    encoded.quote_asset,
                    encoded.instrument_version,
                    encoded.settlement_version,
                    encoded.settlement_payload,
                    encoded.settlement_payload_sha256,
                ),
            )
            inserted_version = cursor.fetchone()
            if (
                inserted_version is None
                or _row(
                    inserted_version,
                    1,
                    "inserted account settlement",
                )[0]
                != encoded.account_version
            ):
                raise JournalStorageError("paper-account settlement was not inserted")
            for posting_ordinal, posting in enumerate(
                admission.postings,
                start=1,
            ):
                cursor.execute(
                    _INSERT_ACCOUNT_POSTING_SQL,
                    (
                        encoded.account_key,
                        encoded.account_version,
                        posting_ordinal,
                        posting.asset,
                        posting.bucket.value,
                        str(posting.amount),
                    ),
                )
                inserted_ordinal = cursor.fetchone()
                if (
                    inserted_ordinal is None
                    or _row(
                        inserted_ordinal,
                        1,
                        "inserted account posting",
                    )[0]
                    != posting_ordinal
                ):
                    raise JournalStorageError("paper-account posting was not inserted")

        after = batch.admissions[-1].after
        for balance in after.balances:
            cursor.execute(
                _UPSERT_ACCOUNT_BALANCE_SQL,
                (
                    context.account_key,
                    balance.asset,
                    str(balance.available),
                    str(balance.reserved),
                ),
            )
            stored_asset = cursor.fetchone()
            if (
                stored_asset is None
                or _row(
                    stored_asset,
                    1,
                    "updated account balance",
                )[0]
                != balance.asset
            ):
                raise JournalStorageError("paper-account balance was not updated")

        position_key = context.attempt.instruction.position_key
        cursor.execute(
            _DELETE_POSITION_RESERVATION_SQL,
            (context.account_key, position_key),
        )
        cursor.fetchone()
        reservation = next(
            (
                candidate
                for candidate in after.reservations
                if candidate.position_key == position_key
            ),
            None,
        )
        if reservation is not None:
            cursor.execute(
                _INSERT_POSITION_RESERVATION_SQL,
                (
                    context.account_key,
                    context.execution_scope,
                    position_key,
                    str(reservation.amount),
                ),
            )
            inserted_position = cursor.fetchone()
            if (
                inserted_position is None
                or _row(
                    inserted_position,
                    1,
                    "inserted margin reservation",
                )[0]
                != position_key
            ):
                raise JournalStorageError("margin reservation was not inserted")

        cursor.execute(
            _ADVANCE_ACCOUNT_STREAM_SQL,
            (
                batch.account_versions[-1],
                after.state.value,
                context.account_key,
                len(before.account.records),
            ),
        )
        advanced = cursor.fetchone()
        if advanced is None or _row(advanced, 1, "advanced account stream")[0] != (
            batch.account_versions[-1]
        ):
            raise JournalStorageError("paper-account stream version did not advance")

    def _replayed_receipt(
        self,
        account: ReplayedPaperAccount,
        replay: object,
        *,
        context: PaperAccountSubmissionContext,
        encoded_instruction: object,
    ) -> DurablePaperAccountSubmissionReceipt:
        return self._receipt_from_replay(
            account,
            replay,
            context=context,
            encoded_instruction=encoded_instruction,
            disposition=DurableSubmissionDisposition.REPLAYED,
        )

    @staticmethod
    def _receipt_from_replay(
        account: ReplayedPaperAccount,
        replay: object,
        *,
        context: PaperAccountSubmissionContext,
        encoded_instruction: object,
        disposition: DurableSubmissionDisposition,
    ) -> DurablePaperAccountSubmissionReceipt:
        manifests = tuple(
            manifest
            for manifest in account.batches
            if manifest.client_order_id == context.client_order_id
        )
        if len(manifests) != 1:
            raise PaperAccountSubmissionReconciliationRequired(context)
        manifest = manifests[0]
        if (
            manifest.execution_scope != context.execution_scope
            or manifest.account_key != context.account_key
            or manifest.runtime_generation != context.runtime_generation
            or manifest.position_key != encoded_instruction.position_key
            or manifest.instruction_payload_sha256
            != encoded_instruction.instruction_payload_sha256
            or manifest.submission_event_id != context.attempt.event_id
            or manifest.submission_observed_at != context.attempt.observed_at
        ):
            raise PaperAccountSubmissionReconciliationRequired(context)

        stored = replay.orders_by_client.get(context.client_order_id)
        if stored is None or stored.encoded != encoded_instruction:
            raise PaperAccountSubmissionReconciliationRequired(context)
        try:
            order = _find_replayed_order(
                replay.projection,
                context.client_order_id,
            )
            submission = _receipt_for_order(
                order,
                attempt=context.attempt,
                disposition=disposition,
            )
        except (JournalRepositoryError, TypeError, ValueError) as exc:
            raise PaperAccountSubmissionReconciliationRequired(context) from exc

        records = {record.account_version: record for record in account.account.records}
        if len(records) != len(account.account.records):
            raise PaperAccountSubmissionReconciliationRequired(context)
        for expected, durable_fill in zip(manifest.fills, submission.fills):
            record = records.get(expected.account_version)
            if record is None:
                raise PaperAccountSubmissionReconciliationRequired(context)
            settlement = record.settlement
            fill_record = settlement.record
            if (
                settlement.instrument != context.instrument
                or expected.event_id != durable_fill.event_id
                or expected.position_version != durable_fill.position_version
                or expected.trade_id != durable_fill.event.trade_id
                or fill_record.event_id != expected.event_id
                or fill_record.position_version != expected.position_version
                or fill_record.position_fill.instruction != context.attempt.instruction
            ):
                raise PaperAccountSubmissionReconciliationRequired(context)
        if len(manifest.fills) != len(submission.fills):
            raise PaperAccountSubmissionReconciliationRequired(context)
        return DurablePaperAccountSubmissionReceipt(
            context=context,
            submission=submission,
            account_versions=tuple(fill.account_version for fill in manifest.fills),
        )


__all__ = ["PostgresAtomicPaperAccountOwner"]
