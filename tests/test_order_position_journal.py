"""Fast repository tests with an in-memory PostgreSQL protocol double."""

import ast
import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

from trading.domain.order_lifecycle import (
    ConfirmedFill,
    OrderLifecycleState,
    SubmissionAcknowledged,
    SubmissionAmbiguous,
    SubmissionFailed,
)
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionStatus,
)
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionInstruction,
    PositionState,
    TakeProfitProfile,
)
from trading.persistence.journal_codec import encode_order_lifecycle_event
from trading.persistence.order_position_journal import (
    EventAppendDisposition,
    JournalCommitUnknown,
    JournalConflictError,
    JournalConflictKind,
    JournalInputError,
    JournalNotFoundError,
    JournalReplayError,
    JournalStorageError,
    PostgresOrderPositionJournal,
    ReservationDisposition,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def make_instruction(
    *,
    client_order_id="order-1",
    decision_id="decision-1",
    position_key="position-1",
    quantity=Decimal("1.00"),
) -> PositionInstruction:
    intent = OrderIntent(
        client_order_id=client_order_id,
        decision_id=decision_id,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.MARKET,
        reference_price=Decimal("50000.125"),
        leverage=3,
        created_at=NOW,
    )
    return PositionInstruction(
        position_key=position_key,
        effect=PositionEffect.OPEN,
        order_intent=intent,
        exit_context=PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.0025"),
            stop_loss_fraction=Decimal("0.005"),
        ),
    )


def make_ack(
    *,
    client_order_id="order-1",
    venue_order_id="venue-1",
    observed_at=NOW + timedelta(seconds=1),
):
    return SubmissionAcknowledged(
        client_order_id=client_order_id,
        venue_order_id=venue_order_id,
        observed_at=observed_at,
    )


def make_fill(
    *,
    trade_id="trade-1",
    quantity=Decimal("1.00"),
    price=Decimal("50001.250"),
):
    return ConfirmedFill(
        client_order_id="order-1",
        venue_order_id="venue-1",
        trade_id=trade_id,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=quantity,
        price=price,
        fee_amount=Decimal("0.25"),
        fee_asset="USDT",
        executed_at=NOW + timedelta(seconds=2),
    )


def make_ambiguous(*, client_order_id="order-1"):
    return SubmissionAmbiguous(
        client_order_id=client_order_id,
        reason="transport result is unknown",
        observed_at=NOW + timedelta(seconds=2),
    )


def make_failed(*, client_order_id="order-1"):
    return SubmissionFailed(
        client_order_id=client_order_id,
        status=SubmissionStatus.NOT_SENT,
        retry_safety=RetrySafety.SAFE,
        reason="nothing was sent",
        observed_at=NOW + timedelta(seconds=3),
    )


@dataclass
class MemoryState:
    streams: dict = field(default_factory=dict)
    orders: dict = field(default_factory=dict)
    events: list = field(default_factory=list)


class MemoryDatabase:
    def __init__(self):
        self.state = MemoryState()
        self.connections = []
        self.fail_commit = False
        self.commit_then_raise = False
        self.fail_execute = False

    def connect(self):
        connection = MemoryConnection(self)
        self.connections.append(connection)
        return connection


class MemoryConnection:
    autocommit = False
    status = STATUS_READY

    def __init__(self, database):
        self.database = database
        self.state = copy.deepcopy(database.state)
        self.commands = []
        self.cursor_calls = 0
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def get_transaction_status(self):
        return TRANSACTION_STATUS_IDLE

    def cursor(self):
        self.cursor_calls += 1
        return MemoryCursor(self)

    def commit(self):
        self.commits += 1
        if self.database.commit_then_raise:
            self.database.state = copy.deepcopy(self.state)
            raise RuntimeError("lost acknowledgement after commit")
        if self.database.fail_commit:
            raise RuntimeError("lost commit acknowledgement")
        self.database.state = copy.deepcopy(self.state)

    def rollback(self):
        self.rollbacks += 1
        self.state = copy.deepcopy(self.database.state)

    def close(self):
        self.closed = True


class MemoryCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, statement, parameters=None):
        if self.connection.database.fail_execute:
            raise RuntimeError("database unavailable")
        sql = " ".join(statement.split())
        params = parameters or ()
        self.connection.commands.append(sql)
        self.rows = []
        state = self.connection.state

        if sql.startswith("SET TRANSACTION"):
            return
        if sql.startswith("INSERT INTO np.position_streams"):
            position_key, scope = params
            if position_key not in state.streams:
                state.streams[position_key] = {
                    "execution_scope": scope,
                    "stream_version": 0,
                    "created_at": NOW,
                }
                self.rows = [(position_key,)]
            return
        if sql.startswith("SELECT position_key FROM np.position_streams"):
            scope = params[0]
            self.rows = [
                (position_key,)
                for position_key, stream in reversed(tuple(state.streams.items()))
                if stream["execution_scope"] == scope
            ]
            return
        if "FROM np.position_streams" in sql:
            stream = state.streams.get(params[0])
            if stream is not None:
                self.rows = [
                    (
                        stream["execution_scope"],
                        stream["stream_version"],
                        stream["created_at"],
                    )
                ]
            return
        if sql.startswith("INSERT INTO np.orders"):
            (
                client_order_id,
                decision_id,
                position_key,
                scope,
                symbol,
                effect,
                version,
                payload,
                checksum,
            ) = params
            duplicate_decision = any(
                row["execution_scope"] == scope and row["decision_id"] == decision_id
                for row in state.orders.values()
            )
            if client_order_id in state.orders or duplicate_decision:
                return
            state.orders[client_order_id] = {
                "client_order_id": client_order_id,
                "decision_id": decision_id,
                "position_key": position_key,
                "execution_scope": scope,
                "symbol": symbol,
                "position_effect": effect,
                "instruction_version": version,
                "instruction_payload": json.loads(payload),
                "instruction_payload_sha256": checksum,
                "venue_order_id": None,
                "registered_at": NOW,
            }
            self.rows = [(NOW,)]
            return
        if "FROM np.orders" in sql and "WHERE position_key = %s" in sql:
            position_key = params[0]
            self.rows = [
                self._order_row(row)
                for row in sorted(
                    state.orders.values(),
                    key=lambda value: value["client_order_id"],
                )
                if row["position_key"] == position_key
            ]
            return
        if "FROM np.orders" in sql and "OR (execution_scope" in sql:
            client_order_id, scope, decision_id = params
            self.rows = [
                self._order_row(row)
                for row in sorted(
                    state.orders.values(),
                    key=lambda value: value["client_order_id"],
                )
                if row["client_order_id"] == client_order_id
                or (
                    row["execution_scope"] == scope
                    and row["decision_id"] == decision_id
                )
            ]
            return
        if sql.startswith("SELECT position_key, execution_scope FROM np.orders"):
            row = state.orders.get(params[0])
            if row is not None:
                self.rows = [(row["position_key"], row["execution_scope"])]
            return
        if sql.startswith("SELECT client_order_id FROM np.orders"):
            scope, symbol, venue_order_id = params
            self.rows = [
                (row["client_order_id"],)
                for row in state.orders.values()
                if row["execution_scope"] == scope
                and row["symbol"] == symbol
                and row["venue_order_id"] == venue_order_id
            ]
            return
        if sql.startswith("UPDATE np.orders SET venue_order_id"):
            venue_order_id, client_order_id = params
            row = state.orders.get(client_order_id)
            if row is not None and row["venue_order_id"] is None:
                row["venue_order_id"] = venue_order_id
                self.rows = [(venue_order_id,)]
            return
        if sql.startswith("UPDATE np.position_streams SET stream_version"):
            next_version, position_key, expected = params
            stream = state.streams.get(position_key)
            if stream is not None and stream["stream_version"] == expected:
                stream["stream_version"] = next_version
                self.rows = [(next_version,)]
            return
        if sql.startswith("INSERT INTO np.order_events"):
            (
                position_key,
                position_version,
                client_order_id,
                event_id,
                event_type,
                event_version,
                payload,
                checksum,
                trade_id,
                occurred_at,
            ) = params
            state.events.append(
                {
                    "position_key": position_key,
                    "position_version": position_version,
                    "client_order_id": client_order_id,
                    "event_id": event_id,
                    "event_type": event_type,
                    "event_version": event_version,
                    "event_payload": json.loads(payload),
                    "event_payload_sha256": checksum,
                    "trade_id": trade_id,
                    "occurred_at": occurred_at,
                    "recorded_at": NOW + timedelta(seconds=position_version),
                }
            )
            self.rows = [(NOW + timedelta(seconds=position_version),)]
            return
        if "FROM np.order_events" in sql:
            position_key = params[0]
            self.rows = [
                self._event_row(row)
                for row in sorted(
                    state.events,
                    key=lambda value: value["position_version"],
                )
                if row["position_key"] == position_key
            ]
            return
        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    @staticmethod
    def _order_row(row):
        return (
            row["client_order_id"],
            row["decision_id"],
            row["position_key"],
            row["execution_scope"],
            row["symbol"],
            row["position_effect"],
            row["instruction_version"],
            row["instruction_payload"],
            row["instruction_payload_sha256"],
            row["venue_order_id"],
            row["registered_at"],
        )

    @staticmethod
    def _event_row(row):
        return (
            row["position_version"],
            row["client_order_id"],
            row["event_id"],
            row["event_type"],
            row["event_version"],
            row["event_payload"],
            row["event_payload_sha256"],
            row["trade_id"],
            row["occurred_at"],
            row["recorded_at"],
        )


@pytest.fixture
def journal():
    database = MemoryDatabase()
    return database, PostgresOrderPositionJournal(database.connect)


def test_reservation_commits_then_exact_retry_is_existing(journal):
    database, repository = journal
    instruction = make_instruction()

    created = repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=instruction,
    )
    existing = repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=instruction,
    )

    assert created.disposition is ReservationDisposition.CREATED
    assert existing.disposition is ReservationDisposition.EXISTING
    assert created.is_created is True
    assert existing.is_created is False
    assert existing.order.instruction == instruction
    assert existing.current_stream.stream_version == 0
    assert len(database.state.streams) == len(database.state.orders) == 1
    assert all(connection.commits == 1 for connection in database.connections)
    assert all(connection.closed for connection in database.connections)


def test_reservation_compares_canonical_envelope_not_decimal_equality(journal):
    _, repository = journal
    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(quantity=Decimal("1.0")),
    )

    with pytest.raises(JournalConflictError) as caught:
        repository.reserve_instruction(
            execution_scope="paper:test",
            instruction=make_instruction(quantity=Decimal("1.00")),
        )

    assert caught.value.kind is JournalConflictKind.CLIENT_ORDER_ID


def test_append_replays_ack_and_fill_and_deduplicates_both_identities(journal):
    _, repository = journal
    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(),
    )
    ack = repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="ack-1",
        event=make_ack(),
    )
    fill = repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="fill-source-a",
        event=make_fill(),
    )
    duplicate_event = repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="fill-source-a",
        event=make_fill(),
    )
    duplicate_fill = repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="fill-source-b",
        event=make_fill(),
    )

    assert ack.disposition is EventAppendDisposition.APPENDED
    assert fill.disposition is EventAppendDisposition.APPENDED
    assert duplicate_event.disposition is EventAppendDisposition.EXISTING_EVENT_ID
    assert duplicate_fill.disposition is EventAppendDisposition.EXISTING_FILL_ID
    assert duplicate_fill.durable_event_id == "fill-source-a"
    assert duplicate_fill.position_version == 2
    projection = repository.replay_position(
        execution_scope="paper:test",
        position_key="position-1",
    )
    assert projection.stream_version == 2
    assert tuple(record.event_id for record in projection.events) == (
        "ack-1",
        "fill-source-a",
    )
    assert projection.position is not None
    assert projection.position.state is PositionState.OPEN
    assert projection.position.remaining_quantity == Decimal("1.00")
    assert projection.orders[0].lifecycle.filled_quantity == Decimal("1.00")
    assert projection.orders[0].venue_order_id == "venue-1"


def test_identity_venue_and_transition_conflicts_do_not_advance_stream(journal):
    database, repository = journal
    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(),
    )
    repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="ack-1",
        event=make_ack(),
    )

    with pytest.raises(JournalConflictError) as identity_conflict:
        repository.append_event(
            execution_scope="paper:test",
            position_key="position-1",
            event_id="ack-1",
            event=make_ack(observed_at=NOW + timedelta(minutes=1)),
        )
    assert identity_conflict.value.kind is JournalConflictKind.EVENT_ID

    with pytest.raises(JournalConflictError) as venue_conflict:
        repository.append_event(
            execution_scope="paper:test",
            position_key="position-1",
            event_id="ack-other-venue",
            event=make_ack(venue_order_id="venue-other"),
        )
    assert venue_conflict.value.kind is JournalConflictKind.VENUE_ORDER_ID

    with pytest.raises(JournalConflictError) as transition_conflict:
        repository.append_event(
            execution_scope="paper:test",
            position_key="position-1",
            event_id="failed-1",
            event=make_failed(),
        )
    assert transition_conflict.value.kind is JournalConflictKind.INVALID_TRANSITION
    assert database.state.streams["position-1"]["stream_version"] == 1
    assert len(database.state.events) == 1


def test_replay_rejects_gap_and_historical_venue_mismatch_without_partial_result(
    journal,
):
    database, repository = journal
    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(),
    )
    repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="ack-1",
        event=make_ack(),
    )

    database.state.streams["position-1"]["stream_version"] = 2
    with pytest.raises(JournalReplayError, match="exact position-stream prefix"):
        repository.replay_position(
            execution_scope="paper:test",
            position_key="position-1",
        )

    database.state.streams["position-1"]["stream_version"] = 1
    database.state.orders["order-1"]["venue_order_id"] = "venue-other"
    with pytest.raises(JournalReplayError, match="event history"):
        repository.replay_position(
            execution_scope="paper:test",
            position_key="position-1",
        )


def test_replay_rejects_checksum_drift_and_stored_invalid_transition(journal):
    database, repository = journal
    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(),
    )
    repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="ack-1",
        event=make_ack(),
    )

    database.state.events[0]["event_payload"]["venue_order_id"] = "tampered"
    with pytest.raises(JournalReplayError, match="cannot be decoded"):
        repository.replay_position(
            execution_scope="paper:test",
            position_key="position-1",
        )

    database.state.events[0]["event_payload"]["venue_order_id"] = "venue-1"
    encoded = encode_order_lifecycle_event(make_failed())
    database.state.events.append(
        {
            "position_key": "position-1",
            "position_version": 2,
            "client_order_id": encoded.client_order_id,
            "event_id": "failed-1",
            "event_type": encoded.event_type,
            "event_version": encoded.event_version,
            "event_payload": json.loads(encoded.event_payload),
            "event_payload_sha256": encoded.event_payload_sha256,
            "trade_id": encoded.trade_id,
            "occurred_at": encoded.occurred_at,
            "recorded_at": NOW + timedelta(seconds=2),
        }
    )
    database.state.streams["position-1"]["stream_version"] = 2
    with pytest.raises(JournalReplayError, match="violates the domain"):
        repository.replay_position(
            execution_scope="paper:test",
            position_key="position-1",
        )


def test_replay_rejects_huge_stream_version_without_building_an_expected_range(
    journal,
):
    database, repository = journal
    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(),
    )
    database.state.streams["position-1"]["stream_version"] = (1 << 63) - 1

    with pytest.raises(JournalReplayError, match="exact position-stream prefix"):
        repository.replay_position(
            execution_scope="paper:test",
            position_key="position-1",
        )


def test_commit_acknowledgement_failure_never_publishes_local_state(journal):
    database, repository = journal
    database.fail_commit = True

    with pytest.raises(JournalCommitUnknown):
        repository.reserve_instruction(
            execution_scope="paper:test",
            instruction=make_instruction(),
        )

    assert database.state.streams == {}
    assert database.state.orders == {}
    connection = database.connections[-1]
    assert connection.commits == 1
    assert connection.rollbacks == 1
    assert connection.closed is True


def test_commit_unknown_may_be_durable_and_exact_retry_resolves_it(journal):
    database, repository = journal
    instruction = make_instruction()
    database.commit_then_raise = True

    with pytest.raises(JournalCommitUnknown):
        repository.reserve_instruction(
            execution_scope="paper:test",
            instruction=instruction,
        )

    assert tuple(database.state.orders) == ("order-1",)
    database.commit_then_raise = False
    resolved = repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=instruction,
    )
    assert resolved.disposition is ReservationDisposition.EXISTING


def test_known_storage_failure_rolls_back_and_closes(journal):
    database, repository = journal
    database.fail_execute = True

    with pytest.raises(JournalStorageError):
        repository.reserve_instruction(
            execution_scope="paper:test",
            instruction=make_instruction(),
        )

    connection = database.connections[-1]
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True


def test_public_replay_rejects_a_committed_empty_stream(journal):
    database, repository = journal
    database.state.streams["orphan"] = {
        "execution_scope": "paper:test",
        "stream_version": 0,
        "created_at": NOW,
    }

    with pytest.raises(JournalReplayError, match="no registered order"):
        repository.replay_position(
            execution_scope="paper:test",
            position_key="orphan",
        )


def test_replay_not_found_scope_and_transaction_modes(journal):
    database, repository = journal
    with pytest.raises(JournalNotFoundError):
        repository.replay_position(
            execution_scope="paper:test",
            position_key="missing",
        )

    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(),
    )
    with pytest.raises(JournalConflictError) as caught:
        repository.replay_position(
            execution_scope="paper:other",
            position_key="position-1",
        )
    assert caught.value.kind is JournalConflictKind.POSITION_SCOPE

    repository.replay_position(
        execution_scope="paper:test",
        position_key="position-1",
    )
    assert any(
        command == "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
        for connection in database.connections
        for command in connection.commands
    )
    assert any(
        command == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
        for connection in database.connections
        for command in connection.commands
    )


def test_replay_order_returns_exact_order_from_a_complete_stream(journal):
    database, repository = journal
    instruction = make_instruction()
    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=instruction,
    )
    repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="ack-1",
        event=make_ack(),
    )

    before = len(database.connections)
    order = repository.replay_order(
        execution_scope="paper:test",
        client_order_id="order-1",
    )

    assert order.instruction == instruction
    assert order.lifecycle.state is OrderLifecycleState.OPEN
    assert order.venue_order_id == "venue-1"
    assert tuple(record.event_id for record in order.events) == ("ack-1",)
    assert len(database.connections) == before + 1
    connection = database.connections[-1]
    assert connection.cursor_calls == 1
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True


def test_replay_order_reports_not_found_and_scope_conflict(journal):
    _, repository = journal

    with pytest.raises(JournalNotFoundError, match="order does not exist"):
        repository.replay_order(
            execution_scope="paper:test",
            client_order_id="missing-order",
        )

    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(),
    )
    with pytest.raises(JournalConflictError) as caught:
        repository.replay_order(
            execution_scope="paper:other",
            client_order_id="order-1",
        )
    assert caught.value.kind is JournalConflictKind.POSITION_SCOPE


def test_replay_order_quarantines_corrupt_stream_data(journal):
    database, repository = journal
    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(),
    )
    repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="ack-1",
        event=make_ack(),
    )
    database.state.events[0]["event_payload"]["venue_order_id"] = "tampered"

    with pytest.raises(JournalReplayError, match="cannot be decoded"):
        repository.replay_order(
            execution_scope="paper:test",
            client_order_id="order-1",
        )

    connection = database.connections[-1]
    assert connection.rollbacks == 1
    assert connection.closed is True


def test_list_unresolved_submissions_filters_states_orders_and_uses_one_snapshot(
    journal,
):
    database, repository = journal
    instructions = (
        make_instruction(
            client_order_id="z-pending",
            decision_id="decision-z",
            position_key="position-z",
        ),
        make_instruction(
            client_order_id="a-reconciling",
            decision_id="decision-a",
            position_key="position-a",
        ),
        make_instruction(
            client_order_id="m-open",
            decision_id="decision-m",
            position_key="position-m",
        ),
        make_instruction(
            client_order_id="b-failed",
            decision_id="decision-b",
            position_key="position-b",
        ),
    )
    for instruction in instructions:
        repository.reserve_instruction(
            execution_scope="paper:test",
            instruction=instruction,
        )
    repository.append_event(
        execution_scope="paper:test",
        position_key="position-a",
        event_id="ambiguous-1",
        event=make_ambiguous(client_order_id="a-reconciling"),
    )
    repository.append_event(
        execution_scope="paper:test",
        position_key="position-m",
        event_id="ack-1",
        event=make_ack(client_order_id="m-open", venue_order_id="venue-m"),
    )
    repository.append_event(
        execution_scope="paper:test",
        position_key="position-b",
        event_id="failed-1",
        event=make_failed(client_order_id="b-failed"),
    )
    repository.reserve_instruction(
        execution_scope="paper:other",
        instruction=make_instruction(
            client_order_id="other-pending",
            decision_id="decision-other",
            position_key="position-other",
        ),
    )

    before = len(database.connections)
    unresolved = repository.list_unresolved_submissions(execution_scope="paper:test")

    assert tuple(
        order.instruction.order_intent.client_order_id for order in unresolved
    ) == ("a-reconciling", "z-pending")
    assert tuple(order.lifecycle.state for order in unresolved) == (
        OrderLifecycleState.RECONCILING,
        OrderLifecycleState.PENDING,
    )
    assert len(database.connections) == before + 1
    connection = database.connections[-1]
    assert connection.cursor_calls == 1
    assert (
        connection.commands.count(
            "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
        )
        == 1
    )
    assert (
        sum(
            command.startswith("SELECT position_key FROM np.position_streams")
            for command in connection.commands
        )
        == 1
    )
    assert all("FOR UPDATE" not in command for command in connection.commands)
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True


def test_list_unresolved_submissions_replays_resolved_streams_before_filtering(
    journal,
):
    database, repository = journal
    repository.reserve_instruction(
        execution_scope="paper:test",
        instruction=make_instruction(),
    )
    repository.append_event(
        execution_scope="paper:test",
        position_key="position-1",
        event_id="ack-1",
        event=make_ack(),
    )
    database.state.events[0]["event_payload"]["venue_order_id"] = "tampered"

    with pytest.raises(JournalReplayError, match="cannot be decoded"):
        repository.list_unresolved_submissions(execution_scope="paper:test")


def test_list_unresolved_submissions_rejects_an_orphan_scope_stream(journal):
    database, repository = journal
    database.state.streams["orphan-position"] = {
        "execution_scope": "paper:test",
        "stream_version": 0,
        "created_at": NOW,
    }

    with pytest.raises(JournalReplayError, match="no registered order"):
        repository.list_unresolved_submissions(execution_scope="paper:test")


@pytest.mark.parametrize(
    "client_order_id",
    (" padded ", "x" * 256, "bad\x00id", "bad\ud800id"),
)
def test_replay_order_rejects_unrepresentable_identity_before_connect(
    client_order_id,
):
    database = MemoryDatabase()
    repository = PostgresOrderPositionJournal(database.connect)

    with pytest.raises(JournalInputError):
        repository.replay_order(
            execution_scope="paper:test",
            client_order_id=client_order_id,
        )

    assert database.connections == []


def test_read_query_failure_is_typed_and_closes_connection(journal):
    database, repository = journal
    database.fail_execute = True

    with pytest.raises(JournalStorageError, match="order replay failed"):
        repository.replay_order(
            execution_scope="paper:test",
            client_order_id="order-1",
        )

    connection = database.connections[-1]
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("execution_scope", " padded "),
        ("execution_scope", "x" * 129),
        ("position_key", "x" * 256),
        ("event_id", "bad\x00id"),
    ),
)
def test_repository_owned_invalid_input_fails_before_connect(field, value):
    database = MemoryDatabase()
    repository = PostgresOrderPositionJournal(database.connect)
    kwargs = {
        "execution_scope": "paper:test",
        "position_key": "position-1",
        "event_id": "ack-1",
        "event": make_ack(),
    }
    kwargs[field] = value

    with pytest.raises(JournalInputError):
        repository.append_event(**kwargs)

    assert database.connections == []


_REPOSITORY_MODULE = "trading.persistence.order_position_journal"
_REPOSITORY_EXPORTS = {
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
}


def _literal_import_target(call):
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None
    target = call.args[0].value
    return target if isinstance(target, str) else None


def _assigned_names(node):
    targets = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    result = set()
    for target in targets:
        if isinstance(target, ast.Name):
            result.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            result.update(item.id for item in target.elts if isinstance(item, ast.Name))
    return result


def _uses_order_position_journal(source):
    """Detect direct, facade, aliased, relative, and literal dynamic use."""
    tree = ast.parse(source)
    root_aliases = set()
    persistence_aliases = set()
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _REPOSITORY_MODULE or alias.name.startswith(
                    _REPOSITORY_MODULE + "."
                ):
                    return True
                if alias.name == "trading":
                    root_aliases.add(alias.asname or "trading")
                elif alias.name == "trading.persistence":
                    if alias.asname:
                        persistence_aliases.add(alias.asname)
                    else:
                        root_aliases.add("trading")
                elif alias.name.startswith("trading.") and alias.asname is None:
                    root_aliases.add("trading")
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported = {alias.name for alias in node.names}
            if module == _REPOSITORY_MODULE or (
                node.level
                and module
                in {"order_position_journal", "persistence.order_position_journal"}
            ):
                return True
            if node.level and not module and imported & {"order_position_journal", "*"}:
                return True
            if module == "trading.persistence" or (
                node.level and module == "persistence"
            ):
                if imported & (_REPOSITORY_EXPORTS | {"order_position_journal", "*"}):
                    return True
            if module == "trading" and "persistence" in imported:
                persistence_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "persistence"
                )
            if module == "importlib" and "import_module" in imported:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )

    def dynamic_target(node):
        if not isinstance(node, ast.Call):
            return None
        target = _literal_import_target(node)
        if isinstance(node.func, ast.Name) and node.func.id in (
            import_module_aliases | {"__import__"}
        ):
            return target
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_aliases
        ):
            return target
        return None

    for node in ast.walk(tree):
        if dynamic_target(node) == _REPOSITORY_MODULE:
            return True

    def path(node):
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            prefix = path(node.value)
            return prefix + (node.attr,) if prefix else ()
        return ()

    def is_root_reference(node):
        node_path = path(node)
        if len(node_path) == 1 and node_path[0] in root_aliases:
            return True
        return dynamic_target(node) == "trading"

    def is_persistence_reference(node):
        node_path = path(node)
        if len(node_path) == 1 and node_path[0] in persistence_aliases:
            return True
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "persistence"
            and is_root_reference(node.value)
        ):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and is_root_reference(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "persistence"
        ):
            return True
        return dynamic_target(node) == "trading.persistence"

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            names = _assigned_names(node)
            if is_root_reference(value):
                before = len(root_aliases)
                root_aliases.update(names)
                changed = changed or len(root_aliases) != before
            if is_persistence_reference(value):
                before = len(persistence_aliases)
                persistence_aliases.update(names)
                changed = changed or len(persistence_aliases) != before

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in (
            _REPOSITORY_EXPORTS | {"order_position_journal"}
        ):
            if is_persistence_reference(node.value):
                return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and is_persistence_reference(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in (_REPOSITORY_EXPORTS | {"order_position_journal"})
        ):
            return True
    return False


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence.order_position_journal import EventCommit",
        "import trading.persistence.order_position_journal as journal",
        "from trading.persistence import PostgresOrderPositionJournal",
        "import trading.persistence as store\nstore.EventCommit",
        "from trading import persistence as store\nstore.ReservationCommit",
        "import trading as root\nroot.persistence.order_position_journal",
        "from ..persistence.order_position_journal import JournalStorageError",
        "from .order_position_journal import JournalStorageError",
        "from . import order_position_journal",
        (
            "from importlib import import_module as load\n"
            "load('trading.persistence.order_position_journal')"
        ),
        (
            "import importlib as loader\n"
            "loader.import_module('trading.persistence').EventCommit"
        ),
        (
            "from importlib import import_module as load\n"
            "load('trading').persistence.PostgresOrderPositionJournal"
        ),
        (
            "root = __import__('trading')\n"
            "store = getattr(root, 'persistence')\n"
            "store.JournalCommitUnknown"
        ),
    ),
)
def test_repository_consumer_detector_catches_supported_forms(source):
    assert _uses_order_position_journal(source)


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence import apply_migrations",
        "import trading.persistence",
        "from trading.domain.positions import Position",
        "name = 'trading.persistence.order_position_journal'",
    ),
)
def test_repository_consumer_detector_allows_unrelated_imports(source):
    assert not _uses_order_position_journal(source)


def test_repository_has_one_persistence_consumer_and_stays_unwired():
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "persistence" / "order_position_journal.py"
    consumers = []
    scanned = []
    for source_path in root.rglob("*.py"):
        if (
            source_path == module_path
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        scanned.append(source_path.relative_to(root))
        if _uses_order_position_journal(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))
    assert sorted(consumers) == [
        Path("trading/persistence/atomic_paper_account_owner.py"),
        Path("trading/persistence/atomic_paper_submission_owner.py"),
        Path("trading/persistence/paper_account_journal.py"),
        Path("trading/persistence/paper_account_readiness.py"),
        Path("trading/persistence/paper_runtime_activation.py"),
    ]
    assert {
        Path("main.py"),
        Path("core/bootstrap.py"),
        Path("utils/paper_trade_db.py"),
    } <= set(scanned)

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    allowed_stdlib = {
        "__future__",
        "dataclasses",
        "datetime",
        "enum",
        "typing",
    }
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.append(node.module or "")
    assert all(
        module in allowed_stdlib
        or module == "psycopg2"
        or module.startswith("psycopg2.")
        or module.startswith("trading.domain.")
        or module == "trading.persistence.journal_codec"
        for module in imports
    )
