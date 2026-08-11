"""PostgreSQL contract tests for the prepared order/position journal schema."""

import hashlib
import json
from datetime import datetime, timezone

import psycopg2
import pytest

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _canonical_payload(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _instruction_payload(
    *,
    client_order_id="order-1",
    decision_id="decision-1",
    position_key="position-1",
    effect="OPEN",
):
    value = {
        "position_key": position_key,
        "effect": effect,
        "order_intent": {
            "client_order_id": client_order_id,
            "decision_id": decision_id,
            "symbol": "BTCUSDT",
            "side": "BUY" if effect == "OPEN" else "SELL",
            "quantity": "1E-20000",
            "order_type": "MARKET",
            "reference_price": "50000.125",
            "leverage": 3,
            "created_at": "2026-08-12T12:00:00+00:00",
        },
        "exit_context": (
            {
                "take_profit_profile": "RANGING",
                "take_profit_fraction": "0.0025",
                "stop_loss_fraction": "0.005",
                "trailing_stop_fraction": None,
            }
            if effect == "OPEN"
            else None
        ),
    }
    return _canonical_payload(value)


def _default_event_payload(*, event_type, client_order_id, trade_id):
    observed_at = NOW.isoformat()
    if event_type == "SUBMISSION_ACKNOWLEDGED":
        return {
            "client_order_id": client_order_id,
            "venue_order_id": "venue-1",
            "observed_at": observed_at,
        }
    if event_type == "SUBMISSION_AMBIGUOUS":
        return {
            "client_order_id": client_order_id,
            "reason": "transport-timeout",
            "observed_at": observed_at,
            "venue_order_id": None,
        }
    if event_type == "SUBMISSION_FAILED":
        return {
            "client_order_id": client_order_id,
            "status": "NOT_SENT",
            "retry_safety": "SAFE",
            "reason": "pre-submit-failure",
            "observed_at": observed_at,
        }
    if event_type == "CONFIRMED_FILL":
        return {
            "client_order_id": client_order_id,
            "venue_order_id": "venue-1",
            "trade_id": trade_id,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": "1E-20000",
            "price": "50000.125",
            "fee_amount": "0",
            "executed_at": observed_at,
            "fee_asset": None,
        }
    if event_type == "CANCELLATION_REQUESTED":
        return {
            "client_order_id": client_order_id,
            "cancel_request_id": "cancel-1",
            "requested_at": observed_at,
        }
    if event_type == "CANCELLATION_CONFIRMED":
        return {
            "client_order_id": client_order_id,
            "venue_order_id": "venue-1",
            "cancel_request_id": "cancel-1",
            "observed_at": observed_at,
        }
    if event_type == "CANCELLATION_REJECTED":
        return {
            "client_order_id": client_order_id,
            "venue_order_id": "venue-1",
            "cancel_request_id": "cancel-1",
            "reason": "already-filled",
            "observed_at": observed_at,
        }
    return {"client_order_id": client_order_id, "observed_at": observed_at}


def _insert_stream(connection, *, position_key="position-1", scope="paper:test"):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO np.position_streams (position_key, execution_scope)
            VALUES (%s, %s)
            """,
            (position_key, scope),
        )


def _insert_order(
    connection,
    *,
    client_order_id="order-1",
    decision_id="decision-1",
    position_key="position-1",
    scope="paper:test",
    effect="OPEN",
    venue_order_id=None,
):
    payload, checksum = _instruction_payload(
        client_order_id=client_order_id,
        decision_id=decision_id,
        position_key=position_key,
        effect=effect,
    )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO np.orders (
                client_order_id,
                decision_id,
                position_key,
                execution_scope,
                symbol,
                position_effect,
                instruction_version,
                instruction_payload,
                instruction_payload_sha256,
                venue_order_id
            ) VALUES (%s, %s, %s, %s, 'BTCUSDT', %s, 1, %s::jsonb, %s, %s)
            """,
            (
                client_order_id,
                decision_id,
                position_key,
                scope,
                effect,
                payload,
                checksum,
                venue_order_id,
            ),
        )


def _insert_event(
    connection,
    *,
    position_version,
    event_id,
    event_type,
    trade_id=None,
    client_order_id="order-1",
    position_key="position-1",
    payload=None,
    payload_sha256=None,
    event_version=1,
):
    event_payload = (
        _default_event_payload(
            event_type=event_type,
            client_order_id=client_order_id,
            trade_id=trade_id,
        )
        if payload is None
        else payload
    )
    encoded, generated_checksum = _canonical_payload(event_payload)
    checksum = generated_checksum if payload_sha256 is None else payload_sha256
    with connection.cursor() as cursor:
        cursor.execute(
            """
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
            """,
            (
                position_key,
                position_version,
                client_order_id,
                event_id,
                event_type,
                event_version,
                encoded,
                checksum,
                trade_id,
                NOW,
            ),
        )


def _assert_rejected(connection, statement, parameters):
    with pytest.raises(psycopg2.Error):
        with connection.cursor() as cursor:
            cursor.execute(statement, parameters)
    connection.rollback()


def test_schema_preserves_versioned_string_payloads_and_per_position_versions(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        _insert_stream(connection)
        _insert_order(connection)
        _insert_event(
            connection,
            position_version=1,
            event_id="submission-1",
            event_type="SUBMISSION_ACKNOWLEDGED",
        )
        _insert_event(
            connection,
            position_version=2,
            event_id="fill-1",
            event_type="CONFIRMED_FILL",
            trade_id="trade-1",
        )
        with connection.cursor() as cursor:
            cursor.execute("""
                UPDATE np.position_streams
                SET stream_version = 2
                WHERE position_key = 'position-1'
                """)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    instruction_payload,
                    instruction_payload_sha256,
                    registered_at
                FROM np.orders
                WHERE client_order_id = 'order-1'
                """)
            payload, checksum, registered_at = cursor.fetchone()
            assert payload["order_intent"]["quantity"] == "1E-20000"
            assert checksum == _canonical_payload(payload)[1]
            assert registered_at.tzinfo is not None

            cursor.execute("""
                SELECT
                    position_version,
                    event_type,
                    event_payload,
                    event_payload_sha256,
                    occurred_at
                FROM np.order_events
                WHERE client_order_id = 'order-1'
                ORDER BY position_version
                """)
            events = cursor.fetchall()
            assert tuple(row[0] for row in events) == (1, 2)
            assert events[0][1] == "SUBMISSION_ACKNOWLEDGED"
            assert events[0][2]["venue_order_id"] == "venue-1"
            assert events[1][2]["quantity"] == "1E-20000"
            assert all(row[3] == _canonical_payload(row[2])[1] for row in events)
            assert all(row[4].tzinfo is not None for row in events)

            cursor.execute("""
                SELECT stream_version
                FROM np.position_streams
                WHERE position_key = 'position-1'
                """)
            assert cursor.fetchone() == (2,)
    finally:
        connection.close()


def test_one_position_version_stream_orders_facts_across_multiple_orders(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        _insert_stream(connection, position_key="position-shared")
        _insert_order(
            connection,
            client_order_id="order-open",
            decision_id="decision-open",
            position_key="position-shared",
        )
        _insert_order(
            connection,
            client_order_id="order-reduce",
            decision_id="decision-reduce",
            position_key="position-shared",
            effect="REDUCE_ONLY",
        )
        _insert_event(
            connection,
            position_key="position-shared",
            client_order_id="order-open",
            position_version=1,
            event_id="open-ack",
            event_type="SUBMISSION_ACKNOWLEDGED",
        )
        _insert_event(
            connection,
            position_key="position-shared",
            client_order_id="order-reduce",
            position_version=2,
            event_id="reduce-ack",
            event_type="SUBMISSION_ACKNOWLEDGED",
        )
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT position_version, client_order_id
                FROM np.order_events
                WHERE position_key = 'position-shared'
                ORDER BY position_version
                """)
            assert cursor.fetchall() == [
                (1, "order-open"),
                (2, "order-reduce"),
            ]
    finally:
        connection.close()


def test_schema_rejects_invalid_instruction_envelopes(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        _insert_stream(connection)
        connection.commit()
        valid_payload, valid_hash = _instruction_payload()
        insert_sql = """
            INSERT INTO np.orders (
                client_order_id, decision_id, position_key, execution_scope,
                symbol, position_effect, instruction_version,
                instruction_payload, instruction_payload_sha256
            ) VALUES (%s, %s, 'position-1', 'paper:test', 'BTCUSDT', %s, %s,
                      %s::jsonb, %s)
        """
        invalid_rows = (
            (" padded ", "decision-a", "OPEN", 1, valid_payload, valid_hash),
            ("order-a", "decision-a", "FLIP", 1, valid_payload, valid_hash),
            ("order-a", "decision-a", "OPEN", 2, valid_payload, valid_hash),
            ("order-a", "decision-a", "OPEN", 1, "[]", valid_hash),
            ("order-a", "decision-a", "OPEN", 1, valid_payload, "not-a-hash"),
        )
        for row in invalid_rows:
            _assert_rejected(connection, insert_sql, row)

        with pytest.raises(psycopg2.Error):
            _insert_order(
                connection,
                client_order_id="order-wrong-scope",
                decision_id="decision-wrong-scope",
                scope="paper:other",
            )
        connection.rollback()

        _insert_order(connection)
        connection.commit()

        duplicate_decision = (
            "order-2",
            "decision-1",
            "OPEN",
            1,
            valid_payload,
            valid_hash,
        )
        _assert_rejected(connection, insert_sql, duplicate_decision)
    finally:
        connection.close()


def test_schema_scopes_venue_identity_and_restricts_event_identity(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        _insert_stream(connection, position_key="position-a", scope="paper:a")
        _insert_stream(connection, position_key="position-b", scope="paper:b")
        _insert_stream(connection, position_key="position-c", scope="paper:a")
        _insert_order(
            connection,
            client_order_id="order-a",
            decision_id="decision-a",
            position_key="position-a",
            scope="paper:a",
            venue_order_id="venue-1",
        )
        _insert_order(
            connection,
            client_order_id="order-b",
            decision_id="decision-a",
            position_key="position-b",
            scope="paper:b",
            venue_order_id="venue-1",
        )
        connection.commit()

        with pytest.raises(psycopg2.Error):
            _insert_order(
                connection,
                client_order_id="order-c",
                decision_id="decision-c",
                position_key="position-c",
                scope="paper:a",
                venue_order_id="venue-1",
            )
        connection.rollback()

        with pytest.raises(psycopg2.Error):
            _insert_stream(
                connection,
                position_key="position-a",
                scope="paper:b",
            )
        connection.rollback()

        _insert_event(
            connection,
            position_key="position-a",
            client_order_id="order-a",
            position_version=1,
            event_id="fill-a",
            event_type="CONFIRMED_FILL",
            trade_id="trade-a",
        )
        connection.commit()

        _insert_event(
            connection,
            position_key="position-b",
            client_order_id="order-b",
            position_version=1,
            event_id="fill-a",
            event_type="CONFIRMED_FILL",
            trade_id="trade-a",
        )
        connection.commit()

        with pytest.raises(psycopg2.Error):
            _insert_event(
                connection,
                position_key="position-a",
                client_order_id="order-b",
                position_version=2,
                event_id="wrong-stream",
                event_type="SUBMISSION_ACKNOWLEDGED",
            )
        connection.rollback()

        invalid_events = (
            (1, "another-event", "CONFIRMED_FILL", "trade-b"),
            (2, "fill-a", "CONFIRMED_FILL", "trade-b"),
            (2, "fill-b", "CONFIRMED_FILL", "trade-a"),
            (2, "fill-b", "CONFIRMED_FILL", None),
            (2, "cancel-a", "CANCELLATION_REQUESTED", "unexpected-trade"),
        )
        for version, event_id, event_type, trade_id in invalid_events:
            with pytest.raises(psycopg2.Error):
                _insert_event(
                    connection,
                    position_key="position-a",
                    client_order_id="order-a",
                    position_version=version,
                    event_id=event_id,
                    event_type=event_type,
                    trade_id=trade_id,
                )
            connection.rollback()
    finally:
        connection.close()


@pytest.mark.parametrize(
    "overrides",
    (
        {"position_version": 0},
        {"event_id": " padded "},
        {"event_type": "UNKNOWN"},
        {"event_version": 2},
        {"payload": []},
        {"payload_sha256": "not-a-hash"},
    ),
)
def test_schema_rejects_invalid_event_envelopes(
    migrated_postgres_dsn,
    overrides,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        _insert_stream(connection)
        _insert_order(connection)
        connection.commit()

        values = {
            "position_version": 1,
            "event_id": "submission-1",
            "event_type": "SUBMISSION_ACKNOWLEDGED",
        }
        values.update(overrides)
        with pytest.raises(psycopg2.Error):
            _insert_event(connection, **values)
        connection.rollback()
    finally:
        connection.close()
