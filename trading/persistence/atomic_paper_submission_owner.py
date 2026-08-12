"""Unwired PostgreSQL owner for one terminal paper-submission journal batch.

The owner deliberately persists only the instruction, acknowledgement, and
confirmed fills.  It does not execute a venue action or mutate legacy/account
tables.  One fresh connection, one position-stream lock, and one commit own the
entire batch.
"""

from typing import Callable

import psycopg2

from trading.application.durable_submission import (
    DurableLifecycleReceipt,
    DurableSubmissionDisposition,
    DurableSubmissionReceipt,
    PaperPlannedFill,
    PaperSubmissionPlan,
    PaperSubmissionPlanner,
    SubmissionAttemptContext,
    SubmissionCommitUnknown,
    SubmissionReconciliationRequired,
)
from trading.domain.order_lifecycle import (
    ConfirmedFill,
    InvalidOrderTransition,
    OrderLifecycleState,
    SubmissionAcknowledged,
    new_order_lifecycle,
    reduce_order_lifecycle,
)
from trading.domain.positions import (
    InvalidPositionTransition,
    new_position,
    position_fill_from_lifecycle,
    reduce_position,
)
from trading.persistence.journal_codec import (
    encode_order_lifecycle_event,
    encode_position_instruction,
)
from trading.persistence.order_position_journal import (
    _ADVANCE_STREAM_SQL,
    _BIGINT_MAX,
    _INSERT_EVENT_SQL,
    _INSERT_ORDER_SQL,
    _INSERT_STREAM_SQL,
    _SELECT_RESERVATION_CONFLICTS_SQL,
    _SELECT_VENUE_OWNER_SQL,
    _SET_VENUE_ID_SQL,
    _WRITE_TRANSACTION_SQL,
    JournalConflictError,
    JournalConflictKind,
    JournalRepositoryError,
    JournalStorageError,
    PostgresOrderPositionJournal,
    ReplayedOrder,
    _checked_stored_datetime,
    _decode_order_row,
    _find_replayed_order,
    _replay_stream,
    _row,
    _translate_database_error,
)


def _planned_history(
    order: ReplayedOrder,
    *,
    execution_scope: str,
) -> tuple[SubmissionAttemptContext, PaperSubmissionPlan]:
    """Rebuild the only terminal batch shape supported by migration 0002."""
    records = order.events
    if len(records) < 2 or type(records[0].event) is not SubmissionAcknowledged:
        raise ValueError("order history is not an acknowledged full-fill batch")
    if any(type(record.event) is not ConfirmedFill for record in records[1:]):
        raise ValueError("order history contains a non-fill suffix")
    if any(
        current.position_version != previous.position_version + 1
        for previous, current in zip(records, records[1:])
    ):
        raise ValueError("order batch events are not consecutive")
    if order.lifecycle.state is not OrderLifecycleState.FILLED:
        raise ValueError("order history is not terminally filled")

    attempt = SubmissionAttemptContext(
        instruction=order.instruction,
        execution_scope=execution_scope,
        event_id=records[0].event_id,
        observed_at=records[0].event.observed_at,
    )
    plan = PaperSubmissionPlan(
        attempt=attempt,
        submission=records[0].event,
        fills=tuple(
            PaperPlannedFill(event_id=record.event_id, fill=record.event)
            for record in records[1:]
        ),
    )
    return attempt, plan


def _receipt_for_order(
    order: ReplayedOrder,
    *,
    attempt: SubmissionAttemptContext,
    disposition: DurableSubmissionDisposition,
) -> DurableSubmissionReceipt:
    """Return one exact receipt from order-local journal order, never fill sorting."""
    durable_attempt, plan = _planned_history(
        order,
        execution_scope=attempt.execution_scope,
    )
    if durable_attempt != attempt:
        raise ValueError("durable submission attempt differs from the requested one")
    records = order.events
    return DurableSubmissionReceipt(
        disposition=disposition,
        attempt=attempt,
        submission=DurableLifecycleReceipt(
            event_id=records[0].event_id,
            position_version=records[0].position_version,
            event=plan.submission,
        ),
        fills=tuple(
            DurableLifecycleReceipt(
                event_id=record.event_id,
                position_version=record.position_version,
                event=record.event,
            )
            for record in records[1:]
        ),
    )


def _require_terminal_stream(replay: object, attempt: SubmissionAttemptContext) -> None:
    """Reject every unresolved sibling before a new order may be planned."""
    try:
        for order in replay.projection.orders:
            _planned_history(order, execution_scope=attempt.execution_scope)
    except (TypeError, ValueError) as exc:
        raise SubmissionReconciliationRequired(attempt) from exc


def _reservation_conflict(
    cursor: object,
    *,
    encoded: object,
    execution_scope: str,
) -> None:
    cursor.execute(
        _SELECT_RESERVATION_CONFLICTS_SQL,
        (encoded.client_order_id, execution_scope, encoded.decision_id),
    )
    candidates = tuple(_decode_order_row(row) for row in cursor.fetchall())
    if any(
        candidate.encoded.client_order_id == encoded.client_order_id
        for candidate in candidates
    ):
        kind = JournalConflictKind.CLIENT_ORDER_ID
    elif any(
        candidate.execution_scope == execution_scope
        and candidate.encoded.decision_id == encoded.decision_id
        for candidate in candidates
    ):
        kind = JournalConflictKind.DECISION_ID
    else:
        raise JournalStorageError("reservation conflict could not be read back")
    raise JournalConflictError(
        kind,
        "reservation identity is already bound to different data",
    )


class PostgresAtomicPaperSubmissionOwner:
    """Commit or replay one immediate terminal paper fill batch atomically."""

    def __init__(
        self,
        connection_factory: Callable[[], object],
        planner: PaperSubmissionPlanner,
    ) -> None:
        self._journal_boundary = PostgresOrderPositionJournal(connection_factory)
        if not callable(getattr(planner, "plan", None)):
            raise TypeError("planner must provide plan(attempt)")
        self._planner = planner

    def execute(
        self,
        attempt: SubmissionAttemptContext,
        /,
    ) -> DurableSubmissionReceipt:
        """Commit one new terminal batch or replay its exact durable facts."""
        if type(attempt) is not SubmissionAttemptContext:
            raise TypeError("attempt must be a SubmissionAttemptContext")
        encoded_instruction = encode_position_instruction(attempt.instruction)
        connection = self._journal_boundary._connection()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_WRITE_TRANSACTION_SQL)
                    receipt = self._execute_locked(
                        cursor,
                        attempt=attempt,
                        encoded_instruction=encoded_instruction,
                    )
            except (
                JournalRepositoryError,
                SubmissionReconciliationRequired,
                TypeError,
                ValueError,
            ):
                raise
            except psycopg2.Error as exc:
                raise _translate_database_error(exc) from exc
            except Exception as exc:
                raise JournalStorageError(
                    "atomic paper submission failed before commit"
                ) from exc

            try:
                connection.commit()
            except Exception as exc:
                raise SubmissionCommitUnknown(attempt) from exc
            return receipt
        except Exception:
            self._journal_boundary._rollback(connection)
            raise
        finally:
            self._journal_boundary._close(connection)

    def _execute_locked(
        self,
        cursor: object,
        *,
        attempt: SubmissionAttemptContext,
        encoded_instruction: object,
    ) -> DurableSubmissionReceipt:
        scope = attempt.execution_scope
        position_key = encoded_instruction.position_key
        cursor.execute(_INSERT_STREAM_SQL, (position_key, scope))
        inserted_stream = cursor.fetchone()
        if (
            inserted_stream is not None
            and _row(
                inserted_stream,
                1,
                "inserted position stream",
            )[0]
            != position_key
        ):
            raise JournalStorageError("PostgreSQL returned another position stream")

        replay = _replay_stream(
            cursor,
            execution_scope=scope,
            position_key=position_key,
            lock=True,
            allow_empty=inserted_stream is not None,
        )
        existing = replay.orders_by_client.get(encoded_instruction.client_order_id)
        if existing is not None:
            if existing.encoded != encoded_instruction:
                raise JournalConflictError(
                    JournalConflictKind.CLIENT_ORDER_ID,
                    "client order identity is bound to another instruction",
                )
            _require_terminal_stream(replay, attempt)
            order = _find_replayed_order(
                replay.projection,
                encoded_instruction.client_order_id,
            )
            try:
                return _receipt_for_order(
                    order,
                    attempt=attempt,
                    disposition=DurableSubmissionDisposition.REPLAYED,
                )
            except (TypeError, ValueError) as exc:
                raise SubmissionReconciliationRequired(attempt) from exc

        _require_terminal_stream(replay, attempt)
        cursor.execute(
            _INSERT_ORDER_SQL,
            (
                encoded_instruction.client_order_id,
                encoded_instruction.decision_id,
                encoded_instruction.position_key,
                scope,
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
                execution_scope=scope,
            )
        _checked_stored_datetime(
            _row(inserted_order, 1, "inserted order")[0],
            "registered_at",
        )

        plan = self._planner.plan(attempt)
        if type(plan) is not PaperSubmissionPlan:
            raise TypeError("planner must return a PaperSubmissionPlan")
        if plan.attempt is not attempt:
            raise ValueError("planner must retain the exact attempt object")

        self._validate_position_transition(replay, plan)
        self._store_plan(
            cursor,
            replay=replay,
            encoded_instruction=encoded_instruction,
            plan=plan,
        )
        updated = _replay_stream(
            cursor,
            execution_scope=scope,
            position_key=position_key,
            lock=True,
        )
        order = _find_replayed_order(
            updated.projection,
            encoded_instruction.client_order_id,
        )
        return _receipt_for_order(
            order,
            attempt=attempt,
            disposition=DurableSubmissionDisposition.COMMITTED,
        )

    @staticmethod
    def _validate_position_transition(
        replay: object, plan: PaperSubmissionPlan
    ) -> None:
        lifecycle = new_order_lifecycle(plan.attempt.instruction.order_intent)
        position = replay.projection.position
        try:
            lifecycle = reduce_order_lifecycle(lifecycle, plan.submission)
            for candidate in plan.fills:
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
        except (
            InvalidOrderTransition,
            InvalidPositionTransition,
            TypeError,
            ValueError,
        ) as exc:
            raise JournalConflictError(
                JournalConflictKind.INVALID_TRANSITION,
                "planned batch contradicts the current projection",
            ) from exc

    @staticmethod
    def _store_plan(
        cursor: object,
        *,
        replay: object,
        encoded_instruction: object,
        plan: PaperSubmissionPlan,
    ) -> None:
        encoded_events = (
            (
                plan.attempt.event_id,
                encode_order_lifecycle_event(plan.submission),
            ),
        ) + tuple(
            (
                candidate.event_id,
                encode_order_lifecycle_event(candidate.fill),
            )
            for candidate in plan.fills
        )
        venue_order_id = plan.submission.venue_order_id
        cursor.execute(
            _SELECT_VENUE_OWNER_SQL,
            (
                plan.attempt.execution_scope,
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
        final_version = replay.projection.stream_version + len(encoded_events)
        if final_version > _BIGINT_MAX:
            raise JournalConflictError(
                JournalConflictKind.STREAM_VERSION,
                "position stream version is exhausted",
            )
        cursor.execute(
            _ADVANCE_STREAM_SQL,
            (
                final_version,
                encoded_instruction.position_key,
                replay.projection.stream_version,
            ),
        )
        advanced = cursor.fetchone()
        if (
            advanced is None
            or _row(
                advanced,
                1,
                "advanced stream",
            )[0]
            != final_version
        ):
            raise JournalStorageError("position stream version did not advance")

        for offset, (event_id, encoded_event) in enumerate(encoded_events):
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


__all__ = ["PostgresAtomicPaperSubmissionOwner"]
