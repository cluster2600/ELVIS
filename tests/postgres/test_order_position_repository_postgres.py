"""PostgreSQL 15 integration tests for the transactional order journal."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Barrier

import psycopg2
import pytest

from trading.domain.order_lifecycle import (
    CancellationRequested,
    ConfirmedFill,
    OrderLifecycleState,
    SubmissionAcknowledged,
)
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionInstruction,
    PositionState,
    TakeProfitProfile,
)
from trading.persistence.journal_codec import JournalQuarantineError
from trading.persistence.order_position_journal import (
    EventAppendDisposition,
    JournalCommitUnknown,
    JournalConflictError,
    JournalConflictKind,
    JournalReplayError,
    PostgresOrderPositionJournal,
    ReservationDisposition,
)

SCOPE = "paper:test"
POSITION_KEY = "position-1"
NOW = datetime(2026, 8, 12, 12, 0, 0, 123456, tzinfo=timezone.utc)


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _repository(dsn):
    return PostgresOrderPositionJournal(lambda: _connect(dsn))


def _instruction(
    *,
    client_order_id="order-1",
    decision_id="decision-1",
    position_key=POSITION_KEY,
    effect=PositionEffect.OPEN,
    side=OrderSide.BUY,
    quantity=Decimal("1.0"),
):
    intent = OrderIntent(
        client_order_id=client_order_id,
        decision_id=decision_id,
        symbol="BTCUSDT",
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        reference_price=Decimal("50000.00"),
        leverage=3,
        created_at=NOW,
    )
    exit_context = (
        PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.0025"),
            stop_loss_fraction=Decimal("0.005"),
            trailing_stop_fraction=Decimal("0.02"),
        )
        if effect is PositionEffect.OPEN
        else None
    )
    return PositionInstruction(
        position_key=position_key,
        effect=effect,
        order_intent=intent,
        exit_context=exit_context,
    )


def _ack(instruction, *, venue_order_id, observed_at=NOW):
    return SubmissionAcknowledged(
        client_order_id=instruction.order_intent.client_order_id,
        venue_order_id=venue_order_id,
        observed_at=observed_at,
    )


def _fill(
    instruction,
    *,
    venue_order_id,
    trade_id,
    quantity=None,
    price=Decimal("50000.0"),
    executed_at=NOW,
):
    intent = instruction.order_intent
    return ConfirmedFill(
        client_order_id=intent.client_order_id,
        venue_order_id=venue_order_id,
        trade_id=trade_id,
        symbol=intent.symbol,
        side=intent.side,
        quantity=intent.quantity if quantity is None else quantity,
        price=price,
        fee_amount=Decimal("0.00"),
        fee_asset=None,
        executed_at=executed_at,
    )


def _stream_rows(dsn, position_key=POSITION_KEY):
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT stream_version
                FROM np.position_streams
                WHERE position_key = %s
                """,
                (position_key,),
            )
            stream = cursor.fetchone()
            cursor.execute(
                """
                SELECT position_version, client_order_id, event_id
                FROM np.order_events
                WHERE position_key = %s
                ORDER BY position_version
                """,
                (position_key,),
            )
            return stream, cursor.fetchall()
    finally:
        connection.close()


def test_registration_is_committed_idempotent_and_conflict_safe(
    migrated_postgres_dsn,
):
    journal = _repository(migrated_postgres_dsn)
    instruction = _instruction(quantity=Decimal("1.0"))

    created = journal.reserve_instruction(
        execution_scope=SCOPE,
        instruction=instruction,
    )
    existing = journal.reserve_instruction(
        execution_scope=SCOPE,
        instruction=instruction,
    )

    assert created.disposition is ReservationDisposition.CREATED
    assert existing.disposition is ReservationDisposition.EXISTING
    assert created.order.instruction == instruction
    assert existing.order == created.order

    equal_domain_value = _instruction(quantity=Decimal("1.00"))
    assert equal_domain_value == instruction
    with pytest.raises(JournalConflictError) as formatted_conflict:
        journal.reserve_instruction(
            execution_scope=SCOPE,
            instruction=equal_domain_value,
        )
    assert formatted_conflict.value.kind is JournalConflictKind.CLIENT_ORDER_ID

    duplicate_decision = _instruction(
        client_order_id="order-2",
        decision_id=instruction.order_intent.decision_id,
        position_key="position-orphan",
    )
    with pytest.raises(JournalConflictError) as decision_conflict:
        journal.reserve_instruction(
            execution_scope=SCOPE,
            instruction=duplicate_decision,
        )
    assert decision_conflict.value.kind is JournalConflictKind.DECISION_ID

    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT position_key, execution_scope, stream_version
                FROM np.position_streams
                ORDER BY position_key
                """)
            assert cursor.fetchall() == [(POSITION_KEY, SCOPE, 0)]
            cursor.execute("SELECT client_order_id FROM np.orders")
            assert cursor.fetchall() == [("order-1",)]
    finally:
        connection.close()


def test_append_is_gapless_and_deduplicates_event_and_fill_identities(
    migrated_postgres_dsn,
):
    journal = _repository(migrated_postgres_dsn)
    instruction = _instruction()
    journal.reserve_instruction(execution_scope=SCOPE, instruction=instruction)

    acknowledgement = _ack(instruction, venue_order_id="venue-1")
    appended_ack = journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="ack-1",
        event=acknowledgement,
    )
    duplicate_ack = journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="ack-1",
        event=acknowledgement,
    )
    assert appended_ack.disposition is EventAppendDisposition.APPENDED
    assert duplicate_ack.disposition is EventAppendDisposition.EXISTING_EVENT_ID
    assert appended_ack.position_version == duplicate_ack.position_version == 1

    conflicting_ack = replace(
        acknowledgement,
        observed_at=acknowledgement.observed_at + timedelta(seconds=1),
    )
    with pytest.raises(JournalConflictError) as event_conflict:
        journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id="ack-1",
            event=conflicting_ack,
        )
    assert event_conflict.value.kind is JournalConflictKind.EVENT_ID

    fill = _fill(
        instruction,
        venue_order_id="venue-1",
        trade_id="trade-1",
        price=Decimal("50000.0"),
        executed_at=NOW + timedelta(seconds=2),
    )
    appended_fill = journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="fill-observation-1",
        event=fill,
    )
    duplicate_fill = journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="fill-observation-2",
        event=fill,
    )
    assert appended_fill.disposition is EventAppendDisposition.APPENDED
    assert duplicate_fill.disposition is EventAppendDisposition.EXISTING_FILL_ID
    assert duplicate_fill.durable_event_id == "fill-observation-1"
    assert appended_fill.position_version == duplicate_fill.position_version == 2

    equal_domain_fill = replace(fill, price=Decimal("50000.00"))
    assert equal_domain_fill == fill
    with pytest.raises(JournalConflictError) as fill_conflict:
        journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id="fill-observation-3",
            event=equal_domain_fill,
        )
    assert fill_conflict.value.kind is JournalConflictKind.FILL_ID

    assert _stream_rows(migrated_postgres_dsn) == (
        (2,),
        [
            (1, "order-1", "ack-1"),
            (2, "order-1", "fill-observation-1"),
        ],
    )


def test_concurrent_orders_allocate_one_gapless_position_stream(
    migrated_postgres_dsn,
):
    journal = _repository(migrated_postgres_dsn)
    first = _instruction(client_order_id="order-a", decision_id="decision-a")
    second = _instruction(client_order_id="order-b", decision_id="decision-b")
    for instruction in (first, second):
        journal.reserve_instruction(execution_scope=SCOPE, instruction=instruction)

    ready = Barrier(2)

    def append(instruction, event_id, venue_order_id):
        ready.wait(timeout=10)
        return journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id=event_id,
            event=_ack(instruction, venue_order_id=venue_order_id),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(append, first, "ack-a", "venue-a"),
            executor.submit(append, second, "ack-b", "venue-b"),
        )
        results = tuple(future.result(timeout=20) for future in futures)

    assert {result.disposition for result in results} == {
        EventAppendDisposition.APPENDED
    }
    assert {result.position_version for result in results} == {1, 2}
    stream, rows = _stream_rows(migrated_postgres_dsn)
    assert stream == (2,)
    assert [row[0] for row in rows] == [1, 2]
    assert {row[2] for row in rows} == {"ack-a", "ack-b"}


def test_concurrent_exact_appends_create_one_event(migrated_postgres_dsn):
    journal = _repository(migrated_postgres_dsn)
    instruction = _instruction()
    journal.reserve_instruction(execution_scope=SCOPE, instruction=instruction)
    acknowledgement = _ack(instruction, venue_order_id="venue-1")
    ready = Barrier(2)

    def append():
        ready.wait(timeout=10)
        return journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id="ack-1",
            event=acknowledgement,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(append) for _ in range(2))
        results = tuple(future.result(timeout=20) for future in futures)

    assert {result.disposition for result in results} == {
        EventAppendDisposition.APPENDED,
        EventAppendDisposition.EXISTING_EVENT_ID,
    }
    assert {result.position_version for result in results} == {1}
    assert {result.durable_event_id for result in results} == {"ack-1"}
    assert _stream_rows(migrated_postgres_dsn) == (
        (1,),
        [(1, "order-1", "ack-1")],
    )


def test_concurrent_reductions_cannot_over_reduce_one_position(
    migrated_postgres_dsn,
):
    journal = _repository(migrated_postgres_dsn)
    opening = _instruction(
        client_order_id="order-open",
        decision_id="decision-open",
    )
    first_reduction = _instruction(
        client_order_id="order-reduce-a",
        decision_id="decision-reduce-a",
        effect=PositionEffect.REDUCE_ONLY,
        side=OrderSide.SELL,
        quantity=Decimal("0.75"),
    )
    second_reduction = _instruction(
        client_order_id="order-reduce-b",
        decision_id="decision-reduce-b",
        effect=PositionEffect.REDUCE_ONLY,
        side=OrderSide.SELL,
        quantity=Decimal("0.75"),
    )
    for instruction in (opening, first_reduction, second_reduction):
        journal.reserve_instruction(execution_scope=SCOPE, instruction=instruction)

    journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="open-ack",
        event=_ack(opening, venue_order_id="venue-open"),
    )
    journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="open-fill",
        event=_fill(
            opening,
            venue_order_id="venue-open",
            trade_id="trade-open",
            executed_at=NOW + timedelta(seconds=1),
        ),
    )
    journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="reduce-ack-a",
        event=_ack(first_reduction, venue_order_id="venue-reduce-a"),
    )
    journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="reduce-ack-b",
        event=_ack(second_reduction, venue_order_id="venue-reduce-b"),
    )

    ready = Barrier(2)

    def reduce(instruction, event_id, venue_order_id, trade_id):
        ready.wait(timeout=10)
        try:
            return journal.append_event(
                execution_scope=SCOPE,
                position_key=POSITION_KEY,
                event_id=event_id,
                event=_fill(
                    instruction,
                    venue_order_id=venue_order_id,
                    trade_id=trade_id,
                    executed_at=NOW + timedelta(seconds=2),
                ),
            )
        except JournalConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(
                reduce,
                first_reduction,
                "reduce-fill-a",
                "venue-reduce-a",
                "trade-reduce-a",
            ),
            executor.submit(
                reduce,
                second_reduction,
                "reduce-fill-b",
                "venue-reduce-b",
                "trade-reduce-b",
            ),
        )
        outcomes = tuple(future.result(timeout=20) for future in futures)

    committed = tuple(
        outcome for outcome in outcomes if not isinstance(outcome, JournalConflictError)
    )
    rejected = tuple(
        outcome for outcome in outcomes if isinstance(outcome, JournalConflictError)
    )
    assert len(committed) == len(rejected) == 1
    assert committed[0].disposition is EventAppendDisposition.APPENDED
    assert committed[0].position_version == 5
    assert rejected[0].kind is JournalConflictKind.INVALID_TRANSITION

    stream, rows = _stream_rows(migrated_postgres_dsn)
    assert stream == (5,)
    assert [row[0] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[-1][2] in {"reduce-fill-a", "reduce-fill-b"}
    projection = journal.replay_position(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
    )
    assert projection.position is not None
    assert projection.position.state is PositionState.OPEN
    assert projection.position.remaining_quantity == Decimal("0.25")


def test_replay_uses_position_version_across_orders(migrated_postgres_dsn):
    journal = _repository(migrated_postgres_dsn)
    opening = _instruction(
        client_order_id="z-open",
        decision_id="decision-open",
    )
    reduction = _instruction(
        client_order_id="a-reduce",
        decision_id="decision-reduce",
        effect=PositionEffect.REDUCE_ONLY,
        side=OrderSide.SELL,
    )
    journal.reserve_instruction(execution_scope=SCOPE, instruction=opening)
    journal.reserve_instruction(execution_scope=SCOPE, instruction=reduction)

    events = (
        (
            "open-ack",
            _ack(
                opening,
                venue_order_id="venue-open",
                observed_at=NOW + timedelta(seconds=4),
            ),
        ),
        (
            "open-fill",
            _fill(
                opening,
                venue_order_id="venue-open",
                trade_id="trade-open",
                executed_at=NOW + timedelta(seconds=3),
            ),
        ),
        (
            "reduce-ack",
            _ack(
                reduction,
                venue_order_id="venue-reduce",
                observed_at=NOW + timedelta(seconds=2),
            ),
        ),
        (
            "reduce-fill",
            _fill(
                reduction,
                venue_order_id="venue-reduce",
                trade_id="trade-reduce",
                executed_at=NOW + timedelta(seconds=1),
            ),
        ),
    )
    for event_id, event in events:
        journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id=event_id,
            event=event,
        )

    projection = journal.replay_position(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
    )

    assert projection.stream_version == 4
    assert tuple(event.position_version for event in projection.events) == (1, 2, 3, 4)
    assert tuple(event.event_id for event in projection.events) == tuple(
        event_id for event_id, _ in events
    )
    assert tuple(
        order.instruction.order_intent.client_order_id for order in projection.orders
    ) == ("a-reduce", "z-open")
    assert all(
        order.lifecycle.state is OrderLifecycleState.FILLED
        for order in projection.orders
    )
    assert projection.position is not None
    assert projection.position.state is PositionState.CLOSED
    assert projection.position.opened_quantity == Decimal("1.0")
    assert projection.position.reduced_quantity == Decimal("1.0")
    assert projection.position.remaining_quantity == Decimal("0.0")


def test_invalid_transition_has_no_durable_side_effect(migrated_postgres_dsn):
    journal = _repository(migrated_postgres_dsn)
    instruction = _instruction()
    journal.reserve_instruction(execution_scope=SCOPE, instruction=instruction)

    cancellation = CancellationRequested(
        client_order_id=instruction.order_intent.client_order_id,
        cancel_request_id="cancel-1",
        requested_at=NOW,
    )
    with pytest.raises(JournalConflictError) as conflict:
        journal.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id="cancel-1",
            event=cancellation,
        )
    assert conflict.value.kind is JournalConflictKind.INVALID_TRANSITION
    assert _stream_rows(migrated_postgres_dsn) == ((0,), [])

    projection = journal.replay_position(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
    )
    assert projection.orders[0].lifecycle.state is OrderLifecycleState.PENDING
    assert projection.position is None


class _CommitAcknowledgementLost:
    def __init__(self, connection):
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def commit(self):
        self._connection.commit()
        raise psycopg2.OperationalError("simulated lost commit acknowledgement")


def test_commit_unknown_is_reconciled_by_exact_retry(migrated_postgres_dsn):
    normal = _repository(migrated_postgres_dsn)
    unknown = PostgresOrderPositionJournal(
        lambda: _CommitAcknowledgementLost(_connect(migrated_postgres_dsn))
    )
    instruction = _instruction()

    with pytest.raises(JournalCommitUnknown):
        unknown.reserve_instruction(execution_scope=SCOPE, instruction=instruction)
    reservation = normal.reserve_instruction(
        execution_scope=SCOPE,
        instruction=instruction,
    )
    assert reservation.disposition is ReservationDisposition.EXISTING

    acknowledgement = _ack(instruction, venue_order_id="venue-1")
    with pytest.raises(JournalCommitUnknown):
        unknown.append_event(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
            event_id="ack-1",
            event=acknowledgement,
        )
    event = normal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="ack-1",
        event=acknowledgement,
    )
    assert event.disposition is EventAppendDisposition.EXISTING_EVENT_ID
    assert event.position_version == 1
    assert _stream_rows(migrated_postgres_dsn) == (
        (1,),
        [(1, "order-1", "ack-1")],
    )


@pytest.mark.parametrize("damage", ("payload", "version-gap"))
def test_replay_fails_closed_on_corruption_or_version_gap(
    migrated_postgres_dsn,
    damage,
):
    journal = _repository(migrated_postgres_dsn)
    instruction = _instruction()
    journal.reserve_instruction(execution_scope=SCOPE, instruction=instruction)
    journal.append_event(
        execution_scope=SCOPE,
        position_key=POSITION_KEY,
        event_id="ack-1",
        event=_ack(instruction, venue_order_id="venue-1"),
    )

    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            if damage == "payload":
                cursor.execute(
                    """
                    UPDATE np.order_events
                    SET event_payload = event_payload || '{"tampered": true}'::jsonb
                    WHERE position_key = %s AND position_version = 1
                    """,
                    (POSITION_KEY,),
                )
            else:
                cursor.execute(
                    """
                    UPDATE np.position_streams
                    SET stream_version = 2
                    WHERE position_key = %s
                    """,
                    (POSITION_KEY,),
                )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(JournalReplayError) as replay_error:
        journal.replay_position(
            execution_scope=SCOPE,
            position_key=POSITION_KEY,
        )
    if damage == "payload":
        assert isinstance(replay_error.value.__cause__, JournalQuarantineError)
    else:
        assert replay_error.value.__cause__ is None
