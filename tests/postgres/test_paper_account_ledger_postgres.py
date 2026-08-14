"""PostgreSQL 15 proofs for the dormant paper-account ledger schema."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg2
import pytest

from trading.domain.order_lifecycle import ConfirmedFill, SubmissionAcknowledged
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.paper_accounting import (
    PaperAccountAdmissionDisposition,
    PaperAccountBalance,
    PaperAccountPolicy,
    admit_paper_settlement,
    new_paper_account,
)
from trading.domain.paper_economics import PaperFillRecord
from trading.domain.paper_settlement import PaperLinearInstrument, settle_paper_fill
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionFill,
    PositionInstruction,
    TakeProfitProfile,
)
from trading.persistence.journal_codec import (
    encode_order_lifecycle_event,
    encode_position_instruction,
)
from trading.persistence.paper_account_journal_codec import (
    PaperAccountBatchFill,
    PaperAccountBatchManifest,
    decode_paper_account_batch,
    decode_paper_account_opening,
    decode_paper_account_settlement,
    encode_paper_account_batch,
    encode_paper_account_opening,
    encode_paper_account_settlement,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _payload(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _facts(*, account_key="account-1", generation=7):
    opening = _payload(
        {
            "policy": {"margin_quantum": "0.0100"},
            "balances": [{"asset": "USDT", "available": "10.00"}],
        }
    )
    instruction = _payload({"position_key": "position-1", "quantity": "1E-20000"})
    acknowledgement = _payload(
        {"client_order_id": "order-1", "observed_at": NOW.isoformat()}
    )
    fill = _payload({"trade_id": "trade-1", "quantity": "1.2300", "price": "2.500"})
    settlement = _payload({"account_version": 1, "margin": "0.6200", "fee": "0.010"})
    batch = _payload(
        {
            "account_key": account_key,
            "owner_generation": generation,
            "first_account_version": 1,
            "fill_count": 1,
        }
    )
    return {
        "account_key": account_key,
        "generation": generation,
        "opening": opening,
        "instruction": instruction,
        "ack": acknowledgement,
        "fill": fill,
        "settlement": settlement,
        "batch": batch,
    }


def _insert_opening(connection, facts, *, scope="paper:test"):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO np.paper_account_streams (
                account_key, execution_scope, owner_generation,
                collateral_asset, opening_version, opening_payload,
                opening_payload_sha256
            ) VALUES (%s, %s, %s, 'USDT', 1, %s::jsonb, %s)
            """,
            (
                facts["account_key"],
                scope,
                facts["generation"],
                facts["opening"][0],
                facts["opening"][1],
            ),
        )
        cursor.execute(
            """
            INSERT INTO np.paper_account_balances (
                account_key, asset, available_decimal, reserved_decimal
            ) VALUES (%s, 'USDT', '10.00', '0.000')
            """,
            (facts["account_key"],),
        )


def _insert_journal(connection, facts, *, include_fill=True):
    with connection.cursor() as cursor:
        cursor.execute("""
            INSERT INTO np.position_streams (position_key, execution_scope)
            VALUES ('position-1', 'paper:test')
            """)
        cursor.execute(
            """
            INSERT INTO np.orders (
                client_order_id, decision_id, position_key, execution_scope,
                symbol, position_effect, instruction_version,
                instruction_payload, instruction_payload_sha256
            ) VALUES (
                'order-1', 'decision-1', 'position-1', 'paper:test',
                'BTCUSDT', 'OPEN', 1, %s::jsonb, %s
            )
            """,
            facts["instruction"],
        )
        cursor.execute(
            """
            INSERT INTO np.order_events (
                position_key, position_version, client_order_id, event_id,
                event_type, event_version, event_payload,
                event_payload_sha256, trade_id, occurred_at
            ) VALUES (
                'position-1', 1, 'order-1', 'ack-1',
                'SUBMISSION_ACKNOWLEDGED', 1, %s::jsonb, %s, NULL, %s
            )
            """,
            (facts["ack"][0], facts["ack"][1], NOW),
        )
        if include_fill:
            cursor.execute(
                """
                INSERT INTO np.order_events (
                    position_key, position_version, client_order_id, event_id,
                    event_type, event_version, event_payload,
                    event_payload_sha256, trade_id, occurred_at
                ) VALUES (
                    'position-1', 2, 'order-1', 'fill-1',
                    'CONFIRMED_FILL', 1, %s::jsonb, %s, 'trade-1', %s
                )
                """,
                (facts["fill"][0], facts["fill"][1], NOW + timedelta(seconds=1)),
            )


def _prepare(connection, *, account_key="account-1", include_fill=True):
    facts = _facts(account_key=account_key)
    _insert_opening(connection, facts)
    _insert_journal(connection, facts, include_fill=include_fill)
    return facts


def _insert_manifest(connection, facts, **overrides):
    values = {
        "account_key": facts["account_key"],
        "execution_scope": "paper:test",
        "owner_generation": facts["generation"],
        "opening_version": 1,
        "opening_sha": facts["opening"][1],
        "instruction_sha": facts["instruction"][1],
        "submission_sha": facts["ack"][1],
        "submission_observed_at": NOW,
        "first_account_version": 1,
        "last_account_version": 1,
        "last_position_version": 2,
        "fill_count": 1,
    }
    values.update(overrides)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO np.paper_account_batch_manifests (
                account_key, client_order_id, execution_scope,
                owner_generation, opening_version, opening_payload_sha256,
                position_key, instruction_payload_sha256,
                submission_event_id, submission_event_type,
                submission_position_version, submission_observed_at,
                submission_event_payload_sha256, first_account_version,
                last_account_version, last_position_version, fill_count,
                batch_version, batch_payload, batch_payload_sha256
            ) VALUES (
                %(account_key)s, 'order-1', %(execution_scope)s,
                %(owner_generation)s, %(opening_version)s, %(opening_sha)s,
                'position-1', %(instruction_sha)s,
                'ack-1', 'SUBMISSION_ACKNOWLEDGED', 1,
                %(submission_observed_at)s, %(submission_sha)s,
                %(first_account_version)s, %(last_account_version)s,
                %(last_position_version)s, %(fill_count)s,
                1, %(batch_payload)s::jsonb, %(batch_sha)s
            )
            """,
            {
                **values,
                "batch_payload": facts["batch"][0],
                "batch_sha": facts["batch"][1],
            },
        )


def _insert_settlement(connection, facts, **overrides):
    values = {
        "account_key": facts["account_key"],
        "account_version": 1,
        "fill_ordinal": 1,
        "batch_first_account_version": 1,
        "batch_submission_position_version": 1,
        "batch_fill_count": 1,
        "position_version": 2,
        "event_sha": facts["fill"][1],
        "symbol": "BTCUSDT",
        "base_asset": "BTC",
        "quote_asset": "USDT",
    }
    values.update(overrides)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO np.paper_account_settlements (
                account_key, account_version, client_order_id, fill_ordinal,
                batch_first_account_version,
                batch_submission_position_version, batch_fill_count,
                collateral_asset, position_key, position_version, event_id,
                trade_id, event_type, event_payload_sha256, symbol,
                base_asset, quote_asset, instrument_version,
                settlement_version, settlement_payload,
                settlement_payload_sha256
            ) VALUES (
                %(account_key)s, %(account_version)s, 'order-1',
                %(fill_ordinal)s, %(batch_first_account_version)s,
                %(batch_submission_position_version)s, %(batch_fill_count)s,
                'USDT', 'position-1', %(position_version)s, 'fill-1',
                'trade-1', 'CONFIRMED_FILL', %(event_sha)s, %(symbol)s,
                %(base_asset)s, %(quote_asset)s, 1, 1,
                %(settlement_payload)s::jsonb,
                %(settlement_sha)s
            )
            """,
            {
                **values,
                "settlement_payload": facts["settlement"][0],
                "settlement_sha": facts["settlement"][1],
            },
        )


def _assert_rejected(connection, action):
    with pytest.raises(psycopg2.IntegrityError):
        action()
        connection.commit()
    connection.rollback()


def _codec_artifacts():
    opening = new_paper_account(
        PaperAccountPolicy("codec-account", "USDT", Decimal("0.010")),
        (
            PaperAccountBalance("BNB", Decimal("1.500"), Decimal("0")),
            PaperAccountBalance("USDT", Decimal("10.00"), Decimal("0.00")),
        ),
    )
    intent = OrderIntent(
        client_order_id="codec-order",
        decision_id="codec-decision",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("1.2300"),
        order_type=OrderType.MARKET,
        reference_price=Decimal("2.5000"),
        leverage=5,
        created_at=NOW,
    )
    instruction = PositionInstruction(
        position_key="codec-position",
        effect=PositionEffect.OPEN,
        order_intent=intent,
        exit_context=PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.002500"),
            stop_loss_fraction=Decimal("0.00500"),
            trailing_stop_fraction=None,
        ),
    )
    acknowledgement = SubmissionAcknowledged("codec-order", "codec-venue", NOW)
    fill = ConfirmedFill(
        client_order_id="codec-order",
        venue_order_id="codec-venue",
        trade_id="codec-trade",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("1.2300"),
        price=Decimal("2.5000"),
        fee_amount=Decimal("0.1250"),
        fee_asset="BNB",
        executed_at=NOW + timedelta(seconds=1),
    )
    record = PaperFillRecord(
        position_version=2,
        event_id="codec-fill-event",
        position_fill=PositionFill(instruction, fill),
    )
    settlement = settle_paper_fill(
        PaperLinearInstrument("BTCUSDT", "BTC", "USDT"),
        None,
        record,
    )
    admission = admit_paper_settlement(opening, 1, settlement)
    assert admission.disposition is PaperAccountAdmissionDisposition.APPLIED

    encoded_opening = encode_paper_account_opening("paper:codec", 11, opening)
    encoded_instruction = encode_position_instruction(instruction)
    encoded_ack = encode_order_lifecycle_event(acknowledgement)
    encoded_fill = encode_order_lifecycle_event(fill)
    encoded_settlement = encode_paper_account_settlement(admission)
    manifest = PaperAccountBatchManifest(
        execution_scope="paper:codec",
        account_key="codec-account",
        owner_generation=11,
        position_key="codec-position",
        client_order_id="codec-order",
        instruction_payload_sha256=(encoded_instruction.instruction_payload_sha256),
        submission_event_id="codec-submission-event",
        submission_position_version=1,
        submission_observed_at=NOW,
        submission_event_payload_sha256=encoded_ack.event_payload_sha256,
        fills=(
            PaperAccountBatchFill(
                position_key="codec-position",
                client_order_id="codec-order",
                event_id="codec-fill-event",
                trade_id="codec-trade",
                position_version=2,
                account_version=1,
                event_payload_sha256=encoded_fill.event_payload_sha256,
                account_settlement_payload_sha256=(
                    encoded_settlement.settlement_payload_sha256
                ),
            ),
        ),
    )
    return (
        opening,
        admission,
        manifest,
        encoded_opening,
        encoded_instruction,
        encoded_ack,
        encoded_fill,
        encoded_settlement,
        encode_paper_account_batch(manifest),
    )


def test_codec_envelopes_round_trip_through_the_relational_schema(
    migrated_postgres_dsn,
):
    (
        opening,
        admission,
        manifest,
        encoded_opening,
        encoded_instruction,
        encoded_ack,
        encoded_fill,
        encoded_settlement,
        encoded_batch,
    ) = _codec_artifacts()
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO np.position_streams (
                    position_key, execution_scope, stream_version
                ) VALUES (%s, %s, %s)
                """,
                (manifest.position_key, manifest.execution_scope, 2),
            )
            cursor.execute(
                """
                INSERT INTO np.orders (
                    client_order_id, decision_id, position_key,
                    execution_scope, symbol, position_effect,
                    instruction_version, instruction_payload,
                    instruction_payload_sha256, venue_order_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                (
                    encoded_instruction.client_order_id,
                    encoded_instruction.decision_id,
                    encoded_instruction.position_key,
                    manifest.execution_scope,
                    encoded_instruction.symbol,
                    encoded_instruction.position_effect,
                    encoded_instruction.instruction_version,
                    encoded_instruction.instruction_payload,
                    encoded_instruction.instruction_payload_sha256,
                    "codec-venue",
                ),
            )
            for position_version, event_id, encoded_event in (
                (1, manifest.submission_event_id, encoded_ack),
                (2, manifest.fills[0].event_id, encoded_fill),
            ):
                cursor.execute(
                    """
                    INSERT INTO np.order_events (
                        position_key, position_version, client_order_id,
                        event_id, event_type, event_version, event_payload,
                        event_payload_sha256, trade_id, occurred_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                    """,
                    (
                        manifest.position_key,
                        position_version,
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
            cursor.execute(
                """
                INSERT INTO np.paper_account_streams (
                    account_key, execution_scope, owner_generation,
                    collateral_asset, account_version, account_state,
                    opening_version, opening_payload,
                    opening_payload_sha256
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    encoded_opening.account_key,
                    encoded_opening.execution_scope,
                    encoded_opening.owner_generation,
                    encoded_opening.collateral_asset,
                    admission.account_version,
                    admission.after.state.value,
                    encoded_opening.opening_version,
                    encoded_opening.opening_payload,
                    encoded_opening.opening_payload_sha256,
                ),
            )
            for balance in admission.after.balances:
                cursor.execute(
                    """
                    INSERT INTO np.paper_account_balances (
                        account_key, asset, available_decimal, reserved_decimal
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (
                        encoded_opening.account_key,
                        balance.asset,
                        str(balance.available),
                        str(balance.reserved),
                    ),
                )
            for reservation in admission.after.reservations:
                cursor.execute(
                    """
                    INSERT INTO np.paper_margin_reservations (
                        account_key, execution_scope, position_key,
                        amount_decimal
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (
                        encoded_opening.account_key,
                        encoded_opening.execution_scope,
                        reservation.position_key,
                        str(reservation.amount),
                    ),
                )
            cursor.execute(
                """
                INSERT INTO np.paper_account_batch_manifests (
                    account_key, client_order_id, execution_scope,
                    owner_generation, opening_version,
                    opening_payload_sha256, position_key,
                    instruction_payload_sha256, submission_event_id,
                    submission_event_type, submission_position_version,
                    submission_observed_at,
                    submission_event_payload_sha256,
                    first_account_version, last_account_version,
                    last_position_version, fill_count, batch_version,
                    batch_payload, batch_payload_sha256
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
                )
                """,
                (
                    encoded_batch.account_key,
                    encoded_batch.client_order_id,
                    encoded_batch.execution_scope,
                    encoded_batch.owner_generation,
                    encoded_opening.opening_version,
                    encoded_opening.opening_payload_sha256,
                    encoded_batch.position_key,
                    encoded_batch.instruction_payload_sha256,
                    encoded_batch.submission_event_id,
                    encoded_ack.event_type,
                    encoded_batch.submission_position_version,
                    encoded_batch.submission_observed_at,
                    encoded_ack.event_payload_sha256,
                    encoded_batch.first_account_version,
                    encoded_batch.last_account_version,
                    encoded_batch.last_position_version,
                    encoded_batch.fill_count,
                    encoded_batch.batch_version,
                    encoded_batch.batch_payload,
                    encoded_batch.batch_payload_sha256,
                ),
            )
            cursor.execute(
                """
                INSERT INTO np.paper_account_settlements (
                    account_key, account_version, client_order_id,
                    fill_ordinal, batch_first_account_version,
                    batch_submission_position_version, batch_fill_count,
                    collateral_asset, position_key, position_version,
                    event_id, trade_id, event_type, event_payload_sha256,
                    symbol, base_asset, quote_asset, instrument_version,
                    settlement_version, settlement_payload,
                    settlement_payload_sha256
                ) VALUES (
                    %s, %s, %s, 1, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
                )
                """,
                (
                    encoded_settlement.account_key,
                    encoded_settlement.account_version,
                    encoded_settlement.client_order_id,
                    encoded_batch.first_account_version,
                    encoded_batch.submission_position_version,
                    encoded_batch.fill_count,
                    encoded_settlement.collateral_asset,
                    encoded_settlement.position_key,
                    encoded_settlement.position_version,
                    encoded_settlement.event_id,
                    encoded_settlement.trade_id,
                    encoded_fill.event_type,
                    encoded_fill.event_payload_sha256,
                    encoded_settlement.symbol,
                    encoded_settlement.base_asset,
                    encoded_settlement.quote_asset,
                    encoded_settlement.instrument_version,
                    encoded_settlement.settlement_version,
                    encoded_settlement.settlement_payload,
                    encoded_settlement.settlement_payload_sha256,
                ),
            )
            for ordinal, posting in enumerate(admission.postings, start=1):
                cursor.execute(
                    """
                    INSERT INTO np.paper_account_postings (
                        account_key, account_version, posting_ordinal,
                        asset, bucket, amount_decimal
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        encoded_settlement.account_key,
                        encoded_settlement.account_version,
                        ordinal,
                        posting.asset,
                        posting.bucket.value,
                        str(posting.amount),
                    ),
                )
            cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT execution_scope, account_key, owner_generation,
                       collateral_asset, opening_version, opening_payload,
                       opening_payload_sha256
                FROM np.paper_account_streams
                WHERE account_key = %s
                """,
                (encoded_opening.account_key,),
            )
            opening_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT account_key, collateral_asset, account_version,
                       position_key, position_version, client_order_id,
                       event_id, trade_id, symbol, base_asset, quote_asset,
                       instrument_version, settlement_version,
                       settlement_payload, settlement_payload_sha256
                FROM np.paper_account_settlements
                WHERE account_key = %s AND account_version = %s
                """,
                (encoded_settlement.account_key, encoded_settlement.account_version),
            )
            settlement_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT execution_scope, account_key, owner_generation,
                       position_key, client_order_id,
                       instruction_payload_sha256, submission_event_id,
                       submission_position_version, submission_observed_at,
                       first_account_version, last_account_version,
                       last_position_version, fill_count, batch_version,
                       batch_payload, batch_payload_sha256
                FROM np.paper_account_batch_manifests
                WHERE account_key = %s AND client_order_id = %s
                """,
                (encoded_batch.account_key, encoded_batch.client_order_id),
            )
            batch_row = cursor.fetchone()

        decoded_opening = decode_paper_account_opening(
            **dict(
                zip(
                    (
                        "execution_scope",
                        "account_key",
                        "owner_generation",
                        "collateral_asset",
                        "opening_version",
                        "opening_payload",
                        "opening_payload_sha256",
                    ),
                    opening_row,
                )
            )
        )
        decoded_settlement = decode_paper_account_settlement(
            opening,
            admission.settlement,
            **dict(
                zip(
                    (
                        "account_key",
                        "collateral_asset",
                        "account_version",
                        "position_key",
                        "position_version",
                        "client_order_id",
                        "event_id",
                        "trade_id",
                        "symbol",
                        "base_asset",
                        "quote_asset",
                        "instrument_version",
                        "settlement_version",
                        "settlement_payload",
                        "settlement_payload_sha256",
                    ),
                    settlement_row,
                )
            ),
        )
        decoded_batch = decode_paper_account_batch(
            **dict(
                zip(
                    (
                        "execution_scope",
                        "account_key",
                        "owner_generation",
                        "position_key",
                        "client_order_id",
                        "instruction_payload_sha256",
                        "submission_event_id",
                        "submission_position_version",
                        "submission_observed_at",
                        "first_account_version",
                        "last_account_version",
                        "last_position_version",
                        "fill_count",
                        "batch_version",
                        "batch_payload",
                        "batch_payload_sha256",
                    ),
                    batch_row,
                )
            )
        )

        assert decoded_opening == opening
        assert decoded_settlement == admission
        assert decoded_batch == manifest
    finally:
        connection.close()


def test_fresh_ledger_is_dormant_and_contains_no_business_seed(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        tables = (
            "paper_account_streams",
            "paper_account_balances",
            "paper_margin_reservations",
            "paper_account_batch_manifests",
            "paper_account_settlements",
            "paper_account_postings",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'np' AND table_name = ANY(%s)
                ORDER BY table_name
                """,
                (list(tables),),
            )
            assert tuple(row[0] for row in cursor.fetchall()) == tuple(sorted(tables))
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM np.{table}")
                assert cursor.fetchone() == (0,)
    finally:
        connection.close()


def test_valid_deferred_batch_preserves_json_hashes_and_decimal_strings(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _insert_settlement(connection, facts)
        _insert_manifest(connection, facts)
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO np.paper_account_postings (
                    account_key, account_version, posting_ordinal,
                    asset, bucket, amount_decimal
                ) VALUES ('account-1', 1, 1, 'USDT', 'AVAILABLE', '-0.6200')
                """)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT opening_payload, opening_payload_sha256
                FROM np.paper_account_streams
                """)
            opening, opening_sha = cursor.fetchone()
            assert opening["policy"]["margin_quantum"] == "0.0100"
            assert opening_sha == facts["opening"][1]
            cursor.execute("""
                SELECT settlement_payload, settlement_payload_sha256
                FROM np.paper_account_settlements
                """)
            settlement, settlement_sha = cursor.fetchone()
            assert settlement["margin"] == "0.6200"
            assert settlement_sha == facts["settlement"][1]
            cursor.execute("""
                SELECT available_decimal, reserved_decimal
                FROM np.paper_account_balances
                """)
            assert cursor.fetchone() == ("10.00", "0.000")
            cursor.execute("SELECT amount_decimal FROM np.paper_account_postings")
            assert cursor.fetchone() == ("-0.6200",)
    finally:
        connection.close()


def test_manifest_requires_exact_opening_generation(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _assert_rejected(
            connection,
            lambda: _insert_manifest(connection, facts, owner_generation=8),
        )
    finally:
        connection.close()


def test_manifest_freezes_the_exact_opening_envelope(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _insert_manifest(connection, facts)
        connection.commit()

        with pytest.raises(psycopg2.Error) as raised:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE np.paper_account_streams
                    SET opening_payload = '{}'::jsonb,
                        opening_payload_sha256 = %s
                    WHERE account_key = %s
                    """,
                    ("f" * 64, facts["account_key"]),
                )
            connection.commit()
        assert raised.value.pgcode == "55000"
        connection.rollback()
    finally:
        connection.close()


def test_manifest_requires_exact_ack_hash_and_observed_time(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _assert_rejected(
            connection,
            lambda: _insert_manifest(connection, facts, submission_sha="f" * 64),
        )
    finally:
        connection.close()


def test_manifest_requires_exact_order_instruction_hash(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _assert_rejected(
            connection,
            lambda: _insert_manifest(connection, facts, instruction_sha="f" * 64),
        )
    finally:
        connection.close()


@pytest.mark.parametrize(
    "overrides",
    (
        {"fill_ordinal": 2},
        {"account_version": 2},
        {"position_version": 3},
    ),
)
def test_settlement_ordinal_must_match_manifest_account_and_position_ranges(
    migrated_postgres_dsn,
    overrides,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _insert_manifest(connection, facts)
        _assert_rejected(
            connection,
            lambda: _insert_settlement(connection, facts, **overrides),
        )
    finally:
        connection.close()


def test_settlement_requires_exact_fill_hash_and_order_symbol(migrated_postgres_dsn):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _insert_manifest(connection, facts)
        _assert_rejected(
            connection,
            lambda: _insert_settlement(connection, facts, event_sha="f" * 64),
        )
    finally:
        connection.close()


@pytest.mark.parametrize(
    "overrides",
    (
        {"symbol": "ETHUSDT"},
        {"base_asset": "USDT"},
        {"quote_asset": "BTC"},
    ),
)
def test_settlement_rejects_order_or_instrument_denomination_drift(
    migrated_postgres_dsn,
    overrides,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _insert_manifest(connection, facts)
        _assert_rejected(
            connection,
            lambda: _insert_settlement(connection, facts, **overrides),
        )
    finally:
        connection.close()


def test_one_order_manifest_cannot_be_claimed_by_two_accounts(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _insert_manifest(connection, facts)
        other = _facts(account_key="account-2")
        _insert_opening(connection, other)
        _assert_rejected(connection, lambda: _insert_manifest(connection, other))
    finally:
        connection.close()


def test_missing_deferred_manifest_rolls_back_the_whole_transaction(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _insert_settlement(connection, facts)
        with pytest.raises(psycopg2.IntegrityError):
            connection.commit()
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM np.paper_account_settlements")
            assert cursor.fetchone() == (0,)
            cursor.execute("SELECT COUNT(*) FROM np.paper_account_streams")
            assert cursor.fetchone() == (0,)
    finally:
        connection.close()


def test_manifest_without_settlement_is_explicitly_allowed_for_row_level_schema(
    migrated_postgres_dsn,
):
    connection = _connect(migrated_postgres_dsn)
    try:
        facts = _prepare(connection)
        _insert_manifest(connection, facts)
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM np.paper_account_batch_manifests),
                    (SELECT COUNT(*) FROM np.paper_account_settlements)
                """)
            assert cursor.fetchone() == (1, 0)
    finally:
        connection.close()
