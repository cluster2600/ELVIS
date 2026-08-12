"""Transactional PostgreSQL journal for order and position projections.

This module deliberately owns one fresh connection per public operation.  It is
not wired to the runtime yet: callers must still provide an explicit connection
factory, execution scope, position key, and stable event identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable

import psycopg2
from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

from trading.domain.order_lifecycle import (
    ConfirmedFill,
    InvalidOrderTransition,
    OrderLifecycle,
    OrderLifecycleEvent,
    new_order_lifecycle,
    reduce_order_lifecycle,
)
from trading.domain.positions import (
    InvalidPositionTransition,
    Position,
    PositionInstruction,
    new_position,
    position_fill_from_lifecycle,
    reduce_position,
)
from trading.persistence.journal_codec import (
    EncodedOrderLifecycleEvent,
    EncodedPositionInstruction,
    JournalQuarantineError,
    decode_order_lifecycle_event,
    decode_position_instruction,
    encode_order_lifecycle_event,
    encode_position_instruction,
)

_EXECUTION_SCOPE_MAX_LENGTH = 128
_POSITION_KEY_MAX_LENGTH = 255
_EVENT_ID_MAX_LENGTH = 255
_VENUE_ORDER_ID_MAX_LENGTH = 255
_BIGINT_MAX = (1 << 63) - 1

_WRITE_TRANSACTION_SQL = "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
_READ_TRANSACTION_SQL = "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"

_SELECT_STREAM_SQL = """
SELECT execution_scope, stream_version, created_at
FROM np.position_streams
WHERE position_key = %s
"""
_SELECT_STREAM_FOR_UPDATE_SQL = _SELECT_STREAM_SQL + " FOR UPDATE"

_SELECT_ORDERS_SQL = """
SELECT
    client_order_id,
    decision_id,
    position_key,
    execution_scope,
    symbol,
    position_effect,
    instruction_version,
    instruction_payload,
    instruction_payload_sha256,
    venue_order_id,
    registered_at
FROM np.orders
WHERE position_key = %s
ORDER BY client_order_id
"""

_SELECT_EVENTS_SQL = """
SELECT
    position_version,
    client_order_id,
    event_id,
    event_type,
    event_version,
    event_payload,
    event_payload_sha256,
    trade_id,
    occurred_at,
    recorded_at
FROM np.order_events
WHERE position_key = %s
ORDER BY position_version
"""

_INSERT_STREAM_SQL = """
INSERT INTO np.position_streams (position_key, execution_scope)
VALUES (%s, %s)
ON CONFLICT DO NOTHING
RETURNING position_key
"""

_INSERT_ORDER_SQL = """
INSERT INTO np.orders (
    client_order_id,
    decision_id,
    position_key,
    execution_scope,
    symbol,
    position_effect,
    instruction_version,
    instruction_payload,
    instruction_payload_sha256
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT DO NOTHING
RETURNING registered_at
"""

_SELECT_RESERVATION_CONFLICTS_SQL = """
SELECT
    client_order_id,
    decision_id,
    position_key,
    execution_scope,
    symbol,
    position_effect,
    instruction_version,
    instruction_payload,
    instruction_payload_sha256,
    venue_order_id,
    registered_at
FROM np.orders
WHERE client_order_id = %s
   OR (execution_scope = %s AND decision_id = %s)
ORDER BY client_order_id
"""

_SELECT_ORDER_LOCATION_SQL = """
SELECT position_key, execution_scope
FROM np.orders
WHERE client_order_id = %s
"""

_SELECT_VENUE_OWNER_SQL = """
SELECT client_order_id
FROM np.orders
WHERE execution_scope = %s
  AND symbol = %s
  AND venue_order_id = %s
"""

_SET_VENUE_ID_SQL = """
UPDATE np.orders
SET venue_order_id = %s
WHERE client_order_id = %s
  AND venue_order_id IS NULL
RETURNING venue_order_id
"""

_ADVANCE_STREAM_SQL = """
UPDATE np.position_streams
SET stream_version = %s
WHERE position_key = %s
  AND stream_version = %s
RETURNING stream_version
"""

_INSERT_EVENT_SQL = """
INSERT INTO np.order_events (
    position_key,
    position_version,
    client_order_id,
    event_id,
    event_type,
    event_version,
    event_payload,
    event_payload_sha256,
    trade_id,
    occurred_at
) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
RETURNING recorded_at
"""


class JournalRepositoryError(RuntimeError):
    """Base class for failures at the transactional journal boundary."""


class JournalInputError(JournalRepositoryError, ValueError):
    """Raised before I/O when repository-owned input is not representable."""


class JournalStorageError(JournalRepositoryError):
    """Raised when a repository operation is known not to have committed."""


class JournalCommitUnknown(JournalRepositoryError):
    """Raised when PostgreSQL may have committed but its acknowledgement was lost."""


class JournalNotFoundError(JournalRepositoryError):
    """Raised when the requested durable stream or order does not exist."""


class JournalReplayError(JournalRepositoryError):
    """Raised when stored rows cannot produce one complete valid projection."""


class JournalConflictKind(str, Enum):
    """Stable classification for a rejected identity or transition."""

    POSITION_SCOPE = "POSITION_SCOPE"
    CLIENT_ORDER_ID = "CLIENT_ORDER_ID"
    DECISION_ID = "DECISION_ID"
    EVENT_ID = "EVENT_ID"
    FILL_ID = "FILL_ID"
    VENUE_ORDER_ID = "VENUE_ORDER_ID"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    STREAM_VERSION = "STREAM_VERSION"


class JournalConflictError(JournalRepositoryError):
    """Raised when incoming data contradicts an existing durable identity."""

    def __init__(self, kind: JournalConflictKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class ReservationDisposition(str, Enum):
    """Whether this call created or rediscovered a durable reservation."""

    CREATED = "CREATED"
    EXISTING = "EXISTING"


class EventAppendDisposition(str, Enum):
    """How an append call relates to the durable event already present."""

    APPENDED = "APPENDED"
    EXISTING_EVENT_ID = "EXISTING_EVENT_ID"
    EXISTING_FILL_ID = "EXISTING_FILL_ID"


@dataclass(frozen=True, slots=True)
class JournalEventRecord:
    """One decoded event with its durable stream metadata."""

    event_id: str
    position_version: int
    event: OrderLifecycleEvent
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class ReplayedOrder:
    """One registered instruction and its replayed lifecycle."""

    instruction: PositionInstruction
    lifecycle: OrderLifecycle
    registered_at: datetime
    venue_order_id: str | None
    events: tuple[JournalEventRecord, ...]


@dataclass(frozen=True, slots=True)
class PositionStreamProjection:
    """A complete projection from one consistent position-stream snapshot."""

    position_key: str
    execution_scope: str
    stream_version: int
    created_at: datetime
    orders: tuple[ReplayedOrder, ...]
    events: tuple[JournalEventRecord, ...]
    position: Position | None


@dataclass(frozen=True, slots=True)
class ReservationCommit:
    """A reservation result returned only after transaction commit."""

    disposition: ReservationDisposition
    order: ReplayedOrder
    current_stream: PositionStreamProjection

    @property
    def is_created(self) -> bool:
        """Return whether this call committed a new durable reservation."""
        return self.disposition is ReservationDisposition.CREATED


@dataclass(frozen=True, slots=True)
class EventCommit:
    """An event result returned only after transaction commit."""

    disposition: EventAppendDisposition
    durable_event_id: str
    position_version: int
    current_stream: PositionStreamProjection


@dataclass(frozen=True, slots=True)
class _StoredOrder:
    encoded: EncodedPositionInstruction
    instruction: PositionInstruction
    execution_scope: str
    venue_order_id: str | None
    registered_at: datetime


@dataclass(frozen=True, slots=True)
class _StoredEvent:
    encoded: EncodedOrderLifecycleEvent
    record: JournalEventRecord


@dataclass(frozen=True, slots=True)
class _ReplayResult:
    projection: PositionStreamProjection
    orders_by_client: dict[str, _StoredOrder]
    events_by_event_id: dict[tuple[str, str], _StoredEvent]
    fills_by_trade_id: dict[tuple[str, str], _StoredEvent]


def _checked_input_text(value: object, field: str, max_length: int) -> str:
    if type(value) is not str:
        raise JournalInputError(f"{field} must be text")
    if not value or value != value.strip():
        raise JournalInputError(f"{field} must be non-empty and trimmed")
    if len(value) > max_length:
        raise JournalInputError(f"{field} exceeds its storage limit")
    if "\x00" in value or any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        raise JournalInputError(f"{field} is not representable in PostgreSQL")
    return value


def _checked_stored_text(
    value: object,
    field: str,
    max_length: int,
    *,
    optional: bool = False,
) -> str | None:
    if optional and value is None:
        return None
    try:
        return _checked_input_text(value, field, max_length)
    except JournalInputError as exc:
        raise JournalReplayError(f"stored {field} is invalid") from exc


def _checked_stored_integer(
    value: object,
    field: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise JournalReplayError(f"stored {field} is invalid")
    return value


def _checked_stored_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise JournalReplayError(f"stored {field} must be timezone-aware")
    try:
        if value.utcoffset() is None:
            raise JournalReplayError(f"stored {field} must be timezone-aware")
    except (OverflowError, TypeError, ValueError) as exc:
        raise JournalReplayError(f"stored {field} has an invalid timezone") from exc
    return value


def _row(value: object, length: int, source: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise JournalReplayError(f"stored {source} row has an unknown shape")
    return tuple(value)


def _decode_order_row(value: object) -> _StoredOrder:
    row = _row(value, 11, "order")
    try:
        instruction = decode_position_instruction(
            client_order_id=row[0],
            decision_id=row[1],
            position_key=row[2],
            symbol=row[4],
            position_effect=row[5],
            instruction_version=row[6],
            instruction_payload=row[7],
            instruction_payload_sha256=row[8],
        )
    except JournalQuarantineError as exc:
        raise JournalReplayError("stored order envelope cannot be decoded") from exc
    execution_scope = _checked_stored_text(
        row[3],
        "execution_scope",
        _EXECUTION_SCOPE_MAX_LENGTH,
    )
    venue_order_id = _checked_stored_text(
        row[9],
        "venue_order_id",
        _VENUE_ORDER_ID_MAX_LENGTH,
        optional=True,
    )
    registered_at = _checked_stored_datetime(row[10], "registered_at")
    return _StoredOrder(
        encoded=encode_position_instruction(instruction),
        instruction=instruction,
        execution_scope=execution_scope,
        venue_order_id=venue_order_id,
        registered_at=registered_at,
    )


def _decode_event_row(value: object) -> _StoredEvent:
    row = _row(value, 10, "event")
    position_version = _checked_stored_integer(
        row[0],
        "position_version",
        minimum=1,
    )
    event_id = _checked_stored_text(row[2], "event_id", _EVENT_ID_MAX_LENGTH)
    try:
        event = decode_order_lifecycle_event(
            client_order_id=row[1],
            event_type=row[3],
            event_version=row[4],
            event_payload=row[5],
            event_payload_sha256=row[6],
            trade_id=row[7],
            occurred_at=row[8],
        )
    except JournalQuarantineError as exc:
        raise JournalReplayError(
            f"stored event at position version {position_version} cannot be decoded"
        ) from exc
    return _StoredEvent(
        encoded=encode_order_lifecycle_event(event),
        record=JournalEventRecord(
            event_id=event_id,
            position_version=position_version,
            event=event,
            recorded_at=_checked_stored_datetime(row[9], "recorded_at"),
        ),
    )


def _event_venue_order_id(event: OrderLifecycleEvent) -> str | None:
    if hasattr(event, "venue_order_id"):
        return event.venue_order_id
    return None


def _find_replayed_order(
    projection: PositionStreamProjection,
    client_order_id: str,
) -> ReplayedOrder:
    for order in projection.orders:
        if order.instruction.order_intent.client_order_id == client_order_id:
            return order
    raise JournalReplayError("replayed order disappeared from its position stream")


def _replay_stream(
    cursor: object,
    *,
    execution_scope: str,
    position_key: str,
    lock: bool,
    allow_empty: bool = False,
) -> _ReplayResult:
    cursor.execute(
        _SELECT_STREAM_FOR_UPDATE_SQL if lock else _SELECT_STREAM_SQL,
        (position_key,),
    )
    raw_stream = cursor.fetchone()
    if raw_stream is None:
        raise JournalNotFoundError("position stream does not exist")
    stream = _row(raw_stream, 3, "position stream")
    stored_scope = _checked_stored_text(
        stream[0],
        "execution_scope",
        _EXECUTION_SCOPE_MAX_LENGTH,
    )
    if stored_scope != execution_scope:
        raise JournalConflictError(
            JournalConflictKind.POSITION_SCOPE,
            "position key belongs to another execution scope",
        )
    stream_version = _checked_stored_integer(
        stream[1],
        "stream_version",
        minimum=0,
    )
    created_at = _checked_stored_datetime(stream[2], "stream created_at")

    cursor.execute(_SELECT_ORDERS_SQL, (position_key,))
    stored_orders = tuple(_decode_order_row(row) for row in cursor.fetchall())
    if not stored_orders and not allow_empty:
        raise JournalReplayError("stored position stream has no registered order")
    orders_by_client: dict[str, _StoredOrder] = {}
    for order in stored_orders:
        client_order_id = order.encoded.client_order_id
        if order.execution_scope != stored_scope:
            raise JournalReplayError("stored order scope conflicts with its stream")
        if order.encoded.position_key != position_key:
            raise JournalReplayError("stored order position conflicts with its stream")
        if client_order_id in orders_by_client:
            raise JournalReplayError("stored stream repeats a client order identity")
        orders_by_client[client_order_id] = order

    cursor.execute(_SELECT_EVENTS_SQL, (position_key,))
    stored_events = tuple(_decode_event_row(row) for row in cursor.fetchall())
    if len(stored_events) != stream_version or any(
        event.record.position_version != expected_version
        for expected_version, event in enumerate(stored_events, start=1)
    ):
        raise JournalReplayError(
            "stored event versions are not the exact position-stream prefix"
        )

    lifecycles = {
        client_order_id: new_order_lifecycle(order.instruction.order_intent)
        for client_order_id, order in orders_by_client.items()
    }
    events_by_client: dict[str, list[JournalEventRecord]] = {
        client_order_id: [] for client_order_id in orders_by_client
    }
    event_ids: dict[tuple[str, str], _StoredEvent] = {}
    fill_ids: dict[tuple[str, str], _StoredEvent] = {}
    historical_venue_ids: dict[str, set[str]] = {
        client_order_id: set() for client_order_id in orders_by_client
    }
    position: Position | None = None

    for stored_event in stored_events:
        record = stored_event.record
        event = record.event
        client_order_id = event.client_order_id
        order = orders_by_client.get(client_order_id)
        if order is None:
            raise JournalReplayError(
                f"event at position version {record.position_version} has no order"
            )
        event_identity = (client_order_id, record.event_id)
        if event_identity in event_ids:
            raise JournalReplayError("stored stream repeats an event identity")
        event_ids[event_identity] = stored_event

        if type(event) is ConfirmedFill:
            fill_identity = (client_order_id, event.trade_id)
            if fill_identity in fill_ids:
                raise JournalReplayError("stored stream repeats a fill identity")
            fill_ids[fill_identity] = stored_event

        venue_order_id = _event_venue_order_id(event)
        if venue_order_id is not None:
            historical_venue_ids[client_order_id].add(venue_order_id)

        try:
            lifecycle = reduce_order_lifecycle(lifecycles[client_order_id], event)
            lifecycles[client_order_id] = lifecycle
            if type(event) is ConfirmedFill:
                position_fill = position_fill_from_lifecycle(
                    order.instruction,
                    lifecycle,
                    event,
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
            raise JournalReplayError(
                f"stored event at position version {record.position_version} "
                "violates the domain"
            ) from exc
        events_by_client[client_order_id].append(record)

    replayed_orders = []
    for client_order_id in sorted(orders_by_client):
        stored_order = orders_by_client[client_order_id]
        venue_ids = historical_venue_ids[client_order_id]
        if len(venue_ids) > 1:
            raise JournalReplayError(
                "stored order contains conflicting historical venue identities"
            )
        historical_venue_id = next(iter(venue_ids), None)
        if stored_order.venue_order_id != historical_venue_id:
            raise JournalReplayError(
                "stored order venue identity conflicts with its event history"
            )
        replayed_orders.append(
            ReplayedOrder(
                instruction=stored_order.instruction,
                lifecycle=lifecycles[client_order_id],
                registered_at=stored_order.registered_at,
                venue_order_id=historical_venue_id,
                events=tuple(events_by_client[client_order_id]),
            )
        )

    projection = PositionStreamProjection(
        position_key=position_key,
        execution_scope=stored_scope,
        stream_version=stream_version,
        created_at=created_at,
        orders=tuple(replayed_orders),
        events=tuple(event.record for event in stored_events),
        position=position,
    )
    return _ReplayResult(
        projection=projection,
        orders_by_client=orders_by_client,
        events_by_event_id=event_ids,
        fills_by_trade_id=fill_ids,
    )


def _translate_database_error(exc: psycopg2.Error) -> JournalRepositoryError:
    constraint = getattr(getattr(exc, "diag", None), "constraint_name", None)
    conflicts = {
        "position_streams_pkey": JournalConflictKind.POSITION_SCOPE,
        "orders_pkey": JournalConflictKind.CLIENT_ORDER_ID,
        "orders_scope_decision_uq": JournalConflictKind.DECISION_ID,
        "orders_venue_identity_uq": JournalConflictKind.VENUE_ORDER_ID,
        "order_events_event_identity_uq": JournalConflictKind.EVENT_ID,
        "order_events_fill_identity_uq": JournalConflictKind.FILL_ID,
        "order_events_position_version_pk": JournalConflictKind.STREAM_VERSION,
    }
    kind = conflicts.get(constraint)
    if kind is not None:
        return JournalConflictError(kind, "PostgreSQL rejected a journal identity")
    return JournalStorageError("PostgreSQL rejected the journal operation")


class PostgresOrderPositionJournal:
    """Own transactional reservation, append, and replay operations."""

    def __init__(self, connection_factory: Callable[[], object]) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connection_factory = connection_factory

    def _connection(self) -> object:
        try:
            connection = self._connection_factory()
        except Exception as exc:
            raise JournalStorageError("could not open a journal connection") from exc
        required = ("cursor", "commit", "rollback", "close")
        if any(not callable(getattr(connection, name, None)) for name in required):
            self._close(connection)
            raise JournalStorageError("journal connection has an invalid interface")
        transaction_status = getattr(connection, "get_transaction_status", None)
        if not callable(transaction_status):
            self._close(connection)
            raise JournalStorageError("journal connection has no transaction status")
        if getattr(connection, "autocommit", None) is not False:
            self._close(connection)
            raise JournalStorageError("journal connection must disable autocommit")
        try:
            is_idle = transaction_status() == TRANSACTION_STATUS_IDLE
        except Exception as exc:
            self._close(connection)
            raise JournalStorageError(
                "journal connection status could not be inspected"
            ) from exc
        if not is_idle or getattr(connection, "status", None) != STATUS_READY:
            self._close(connection)
            raise JournalStorageError("journal connection must be fresh and idle")
        return connection

    @staticmethod
    def _rollback(connection: object) -> None:
        try:
            connection.rollback()
        except Exception:
            pass

    @staticmethod
    def _close(connection: object) -> None:
        close = getattr(connection, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _write(self, operation: str, callback: Callable[[object], object]) -> object:
        connection = self._connection()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_WRITE_TRANSACTION_SQL)
                    result = callback(cursor)
            except JournalRepositoryError:
                raise
            except psycopg2.Error as exc:
                raise _translate_database_error(exc) from exc
            except Exception as exc:
                raise JournalStorageError(f"{operation} failed before commit") from exc

            try:
                connection.commit()
            except Exception as exc:
                raise JournalCommitUnknown(
                    f"{operation} commit acknowledgement is unknown"
                ) from exc
            return result
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close(connection)

    def reserve_instruction(
        self,
        *,
        execution_scope: str,
        instruction: PositionInstruction,
    ) -> ReservationCommit:
        """Commit one instruction before submission or return its exact reservation."""
        scope = _checked_input_text(
            execution_scope,
            "execution_scope",
            _EXECUTION_SCOPE_MAX_LENGTH,
        )
        encoded = encode_position_instruction(instruction)

        def reserve(cursor: object) -> ReservationCommit:
            cursor.execute(_INSERT_STREAM_SQL, (encoded.position_key, scope))
            inserted_stream = cursor.fetchone()
            if (
                inserted_stream is not None
                and _row(
                    inserted_stream,
                    1,
                    "inserted position stream",
                )[0]
                != encoded.position_key
            ):
                raise JournalStorageError("PostgreSQL returned another position stream")
            replay = _replay_stream(
                cursor,
                execution_scope=scope,
                position_key=encoded.position_key,
                lock=True,
                allow_empty=inserted_stream is not None,
            )
            cursor.execute(
                _INSERT_ORDER_SQL,
                (
                    encoded.client_order_id,
                    encoded.decision_id,
                    encoded.position_key,
                    scope,
                    encoded.symbol,
                    encoded.position_effect,
                    encoded.instruction_version,
                    encoded.instruction_payload,
                    encoded.instruction_payload_sha256,
                ),
            )
            inserted = cursor.fetchone()
            if inserted is None:
                cursor.execute(
                    _SELECT_RESERVATION_CONFLICTS_SQL,
                    (encoded.client_order_id, scope, encoded.decision_id),
                )
                candidates = tuple(_decode_order_row(row) for row in cursor.fetchall())
                exact = tuple(
                    candidate
                    for candidate in candidates
                    if candidate.execution_scope == scope
                    and candidate.encoded == encoded
                )
                if len(exact) == 1:
                    existing = replay.orders_by_client.get(encoded.client_order_id)
                    if existing is None or existing.encoded != encoded:
                        raise JournalReplayError(
                            "exact reservation is outside its expected stream"
                        )
                    return ReservationCommit(
                        disposition=ReservationDisposition.EXISTING,
                        order=_find_replayed_order(
                            replay.projection,
                            encoded.client_order_id,
                        ),
                        current_stream=replay.projection,
                    )
                if any(
                    candidate.encoded.client_order_id == encoded.client_order_id
                    for candidate in candidates
                ):
                    kind = JournalConflictKind.CLIENT_ORDER_ID
                elif any(
                    candidate.execution_scope == scope
                    and candidate.encoded.decision_id == encoded.decision_id
                    for candidate in candidates
                ):
                    kind = JournalConflictKind.DECISION_ID
                else:
                    raise JournalStorageError(
                        "reservation conflict could not be read back"
                    )
                raise JournalConflictError(
                    kind,
                    "reservation identity is already bound to different data",
                )

            _checked_stored_datetime(
                _row(inserted, 1, "inserted order")[0],
                "registered_at",
            )
            updated = _replay_stream(
                cursor,
                execution_scope=scope,
                position_key=encoded.position_key,
                lock=True,
            )
            return ReservationCommit(
                disposition=ReservationDisposition.CREATED,
                order=_find_replayed_order(
                    updated.projection,
                    encoded.client_order_id,
                ),
                current_stream=updated.projection,
            )

        return self._write("instruction reservation", reserve)

    def append_event(
        self,
        *,
        execution_scope: str,
        position_key: str,
        event_id: str,
        event: OrderLifecycleEvent,
    ) -> EventCommit:
        """Validate and atomically append one event at the next stream version."""
        scope = _checked_input_text(
            execution_scope,
            "execution_scope",
            _EXECUTION_SCOPE_MAX_LENGTH,
        )
        key = _checked_input_text(
            position_key,
            "position_key",
            _POSITION_KEY_MAX_LENGTH,
        )
        durable_event_id = _checked_input_text(
            event_id,
            "event_id",
            _EVENT_ID_MAX_LENGTH,
        )
        encoded = encode_order_lifecycle_event(event)

        def append(cursor: object) -> EventCommit:
            replay = _replay_stream(
                cursor,
                execution_scope=scope,
                position_key=key,
                lock=True,
            )
            order = replay.orders_by_client.get(encoded.client_order_id)
            if order is None:
                cursor.execute(_SELECT_ORDER_LOCATION_SQL, (encoded.client_order_id,))
                location = cursor.fetchone()
                if location is None:
                    raise JournalNotFoundError("event order does not exist")
                raise JournalConflictError(
                    JournalConflictKind.CLIENT_ORDER_ID,
                    "event order belongs to another position stream",
                )

            by_event = replay.events_by_event_id.get(
                (encoded.client_order_id, durable_event_id)
            )
            by_fill = (
                replay.fills_by_trade_id.get(
                    (encoded.client_order_id, encoded.trade_id)
                )
                if encoded.trade_id is not None
                else None
            )
            if by_event is not None and by_fill is not None and by_event != by_fill:
                raise JournalConflictError(
                    JournalConflictKind.FILL_ID,
                    "event and fill identities refer to different durable events",
                )
            existing = by_event or by_fill
            if existing is not None:
                if existing.encoded != encoded:
                    kind = (
                        JournalConflictKind.EVENT_ID
                        if by_event is not None
                        else JournalConflictKind.FILL_ID
                    )
                    raise JournalConflictError(
                        kind,
                        "durable event identity is bound to different data",
                    )
                disposition = (
                    EventAppendDisposition.EXISTING_EVENT_ID
                    if by_event is not None
                    else EventAppendDisposition.EXISTING_FILL_ID
                )
                return EventCommit(
                    disposition=disposition,
                    durable_event_id=existing.record.event_id,
                    position_version=existing.record.position_version,
                    current_stream=replay.projection,
                )

            replayed_order = _find_replayed_order(
                replay.projection,
                encoded.client_order_id,
            )
            incoming_venue_id = _event_venue_order_id(event)
            if (
                incoming_venue_id is not None
                and replayed_order.venue_order_id is not None
                and incoming_venue_id != replayed_order.venue_order_id
            ):
                raise JournalConflictError(
                    JournalConflictKind.VENUE_ORDER_ID,
                    "incoming venue identity conflicts with the order history",
                )
            try:
                next_lifecycle = reduce_order_lifecycle(
                    replayed_order.lifecycle,
                    event,
                )
                if type(event) is ConfirmedFill:
                    position_fill = position_fill_from_lifecycle(
                        order.instruction,
                        next_lifecycle,
                        event,
                    )
                    if replay.projection.position is None:
                        new_position(position_fill)
                    else:
                        reduce_position(replay.projection.position, position_fill)
            except (
                InvalidOrderTransition,
                InvalidPositionTransition,
                TypeError,
                ValueError,
            ) as exc:
                raise JournalConflictError(
                    JournalConflictKind.INVALID_TRANSITION,
                    "incoming event contradicts the current projection",
                ) from exc

            if incoming_venue_id is not None and replayed_order.venue_order_id is None:
                cursor.execute(
                    _SELECT_VENUE_OWNER_SQL,
                    (scope, order.encoded.symbol, incoming_venue_id),
                )
                owner = cursor.fetchone()
                if owner is not None and _row(owner, 1, "venue owner")[0] != (
                    encoded.client_order_id
                ):
                    raise JournalConflictError(
                        JournalConflictKind.VENUE_ORDER_ID,
                        "venue identity belongs to another order",
                    )
                cursor.execute(
                    _SET_VENUE_ID_SQL,
                    (incoming_venue_id, encoded.client_order_id),
                )
                updated_venue = cursor.fetchone()
                if (
                    updated_venue is None
                    or _row(
                        updated_venue,
                        1,
                        "updated venue identity",
                    )[0]
                    != incoming_venue_id
                ):
                    raise JournalStorageError("venue identity was not updated")

            if replay.projection.stream_version >= _BIGINT_MAX:
                raise JournalConflictError(
                    JournalConflictKind.STREAM_VERSION,
                    "position stream version is exhausted",
                )
            next_version = replay.projection.stream_version + 1
            cursor.execute(
                _ADVANCE_STREAM_SQL,
                (next_version, key, replay.projection.stream_version),
            )
            advanced = cursor.fetchone()
            if (
                advanced is None
                or _row(
                    advanced,
                    1,
                    "advanced stream",
                )[0]
                != next_version
            ):
                raise JournalStorageError("position stream version did not advance")
            cursor.execute(
                _INSERT_EVENT_SQL,
                (
                    key,
                    next_version,
                    encoded.client_order_id,
                    durable_event_id,
                    encoded.event_type,
                    encoded.event_version,
                    encoded.event_payload,
                    encoded.event_payload_sha256,
                    encoded.trade_id,
                    encoded.occurred_at,
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
            updated = _replay_stream(
                cursor,
                execution_scope=scope,
                position_key=key,
                lock=True,
            )
            return EventCommit(
                disposition=EventAppendDisposition.APPENDED,
                durable_event_id=durable_event_id,
                position_version=next_version,
                current_stream=updated.projection,
            )

        return self._write("event append", append)

    def replay_position(
        self,
        *,
        execution_scope: str,
        position_key: str,
    ) -> PositionStreamProjection:
        """Return one all-or-nothing projection from a repeatable-read snapshot."""
        scope = _checked_input_text(
            execution_scope,
            "execution_scope",
            _EXECUTION_SCOPE_MAX_LENGTH,
        )
        key = _checked_input_text(
            position_key,
            "position_key",
            _POSITION_KEY_MAX_LENGTH,
        )
        connection = self._connection()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_TRANSACTION_SQL)
                    replay = _replay_stream(
                        cursor,
                        execution_scope=scope,
                        position_key=key,
                        lock=False,
                    )
            except JournalRepositoryError:
                raise
            except psycopg2.Error as exc:
                raise JournalStorageError("position replay query failed") from exc
            except Exception as exc:
                raise JournalStorageError("position replay failed") from exc
            try:
                connection.rollback()
            except Exception as exc:
                raise JournalStorageError(
                    "position replay transaction could not finish"
                ) from exc
            return replay.projection
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close(connection)


__all__ = [
    "EventAppendDisposition",
    "EventCommit",
    "JournalCommitUnknown",
    "JournalConflictError",
    "JournalConflictKind",
    "JournalEventRecord",
    "JournalInputError",
    "JournalNotFoundError",
    "JournalReplayError",
    "JournalRepositoryError",
    "JournalStorageError",
    "PositionStreamProjection",
    "PostgresOrderPositionJournal",
    "ReplayedOrder",
    "ReservationCommit",
    "ReservationDisposition",
]
