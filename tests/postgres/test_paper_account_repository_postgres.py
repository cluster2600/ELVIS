"""PostgreSQL 15 proofs for paper-account provision and strict replay."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Event

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
from trading.persistence.paper_account_journal import (
    PaperAccountCommitUnknown,
    PaperAccountConflictError,
    PaperAccountConflictKind,
    PaperAccountReplayError,
    PostgresPaperAccountJournal,
    ProvisionDisposition,
)
from trading.persistence.paper_account_journal_codec import (
    PaperAccountBatchFill,
    PaperAccountBatchManifest,
    encode_paper_account_batch,
    encode_paper_account_opening,
    encode_paper_account_settlement,
)

NOW = datetime(2026, 8, 12, 12, 0, 0, 123456, tzinfo=timezone.utc)
SCOPE = "paper:test"


def _connect(dsn):
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    return connection


def _repository(dsn):
    return PostgresPaperAccountJournal(lambda: _connect(dsn))


def _opening(
    *,
    account_key="account-1",
    available=Decimal("100.00"),
    margin_quantum=Decimal("0.0100"),
):
    return new_paper_account(
        PaperAccountPolicy(account_key, "USDT", margin_quantum),
        (
            PaperAccountBalance("BNB", Decimal("2.500"), Decimal("0")),
            PaperAccountBalance("USDT", available, Decimal("0.00")),
        ),
    )


def _artifacts(*, fill_quantities=(Decimal("1.2300"),)):
    opening = _opening(account_key="ledger-account")
    intent = OrderIntent(
        client_order_id="ledger-order",
        decision_id="ledger-decision",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("1.2300"),
        order_type=OrderType.MARKET,
        reference_price=Decimal("2.5000"),
        leverage=5,
        created_at=NOW,
    )
    instruction = PositionInstruction(
        position_key="ledger-position",
        effect=PositionEffect.OPEN,
        order_intent=intent,
        exit_context=PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.002500"),
            stop_loss_fraction=Decimal("0.00500"),
            trailing_stop_fraction=None,
        ),
    )
    acknowledgement = SubmissionAcknowledged("ledger-order", "ledger-venue", NOW)
    encoded_opening = encode_paper_account_opening(SCOPE, 7, opening)
    encoded_instruction = encode_position_instruction(instruction)
    encoded_ack = encode_order_lifecycle_event(acknowledgement)
    fills = tuple(
        ConfirmedFill(
            client_order_id="ledger-order",
            venue_order_id="ledger-venue",
            trade_id=f"ledger-trade-{ordinal}",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=quantity,
            price=Decimal("2.5000") + Decimal(ordinal - 1),
            fee_amount=Decimal("0.1250"),
            fee_asset="BNB",
            executed_at=NOW + timedelta(seconds=ordinal),
        )
        for ordinal, quantity in enumerate(fill_quantities, start=1)
    )
    records = tuple(
        PaperFillRecord(
            position_version=ordinal + 1,
            event_id=f"ledger-fill-event-{ordinal}",
            position_fill=PositionFill(instruction, fill),
        )
        for ordinal, fill in enumerate(fills, start=1)
    )
    settlements = []
    admissions = []
    current = opening
    prior = None
    instrument = PaperLinearInstrument("BTCUSDT", "BTC", "USDT")
    for account_version, record in enumerate(records, start=1):
        settlement = settle_paper_fill(instrument, prior, record)
        admission = admit_paper_settlement(current, account_version, settlement)
        assert admission.disposition is PaperAccountAdmissionDisposition.APPLIED
        settlements.append(settlement)
        admissions.append(admission)
        current = admission.after
        prior = settlement.after
    admissions = tuple(admissions)
    encoded_fills = tuple(map(encode_order_lifecycle_event, fills))
    encoded_settlements = tuple(map(encode_paper_account_settlement, admissions))
    manifest = PaperAccountBatchManifest(
        execution_scope=SCOPE,
        account_key=encoded_opening.account_key,
        owner_generation=encoded_opening.owner_generation,
        position_key=instruction.position_key,
        client_order_id=intent.client_order_id,
        instruction_payload_sha256=(encoded_instruction.instruction_payload_sha256),
        submission_event_id="ledger-ack-event",
        submission_position_version=1,
        submission_observed_at=NOW,
        submission_event_payload_sha256=encoded_ack.event_payload_sha256,
        fills=tuple(
            PaperAccountBatchFill(
                position_key=instruction.position_key,
                client_order_id=intent.client_order_id,
                event_id=record.event_id,
                trade_id=fill.trade_id,
                position_version=record.position_version,
                account_version=ordinal,
                event_payload_sha256=encoded_fill.event_payload_sha256,
                account_settlement_payload_sha256=(
                    encoded_settlement.settlement_payload_sha256
                ),
            )
            for ordinal, (record, fill, encoded_fill, encoded_settlement) in enumerate(
                zip(records, fills, encoded_fills, encoded_settlements), start=1
            )
        ),
    )
    return {
        "opening": opening,
        "admission": admissions[-1],
        "admissions": admissions,
        "manifest": manifest,
        "encoded_opening": encoded_opening,
        "encoded_instruction": encoded_instruction,
        "encoded_ack": encoded_ack,
        "encoded_fill": encoded_fills[-1],
        "encoded_fills": encoded_fills,
        "encoded_settlement": encoded_settlements[-1],
        "encoded_settlements": encoded_settlements,
        "encoded_batch": encode_paper_account_batch(manifest),
    }


def _insert_journal(connection, artifacts):
    manifest = artifacts["manifest"]
    instruction = artifacts["encoded_instruction"]
    acknowledgement = artifacts["encoded_ack"]
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO np.position_streams (
                position_key, execution_scope, stream_version
            ) VALUES (%s, %s, %s)
            """,
            (
                manifest.position_key,
                manifest.execution_scope,
                manifest.fills[-1].position_version,
            ),
        )
        cursor.execute(
            """
            INSERT INTO np.orders (
                client_order_id, decision_id, position_key, execution_scope,
                symbol, position_effect, instruction_version,
                instruction_payload, instruction_payload_sha256,
                venue_order_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            """,
            (
                instruction.client_order_id,
                instruction.decision_id,
                instruction.position_key,
                manifest.execution_scope,
                instruction.symbol,
                instruction.position_effect,
                instruction.instruction_version,
                instruction.instruction_payload,
                instruction.instruction_payload_sha256,
                "ledger-venue",
            ),
        )
        events = ((1, manifest.submission_event_id, acknowledgement),) + tuple(
            (
                fill_ref.position_version,
                fill_ref.event_id,
                encoded_fill,
            )
            for fill_ref, encoded_fill in zip(
                manifest.fills, artifacts["encoded_fills"]
            )
        )
        for version, event_id, event in events:
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
                    version,
                    event.client_order_id,
                    event_id,
                    event.event_type,
                    event.event_version,
                    event.event_payload,
                    event.event_payload_sha256,
                    event.trade_id,
                    event.occurred_at,
                ),
            )


def _insert_manifest(connection, artifacts):
    opening = artifacts["encoded_opening"]
    acknowledgement = artifacts["encoded_ack"]
    batch = artifacts["encoded_batch"]
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO np.paper_account_batch_manifests (
                account_key, client_order_id, execution_scope,
                owner_generation, opening_version,
                opening_payload_sha256, position_key,
                instruction_payload_sha256, submission_event_id,
                submission_event_type, submission_position_version,
                submission_observed_at, submission_event_payload_sha256,
                first_account_version, last_account_version,
                last_position_version, fill_count, batch_version,
                batch_payload, batch_payload_sha256
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
            )
            """,
            (
                batch.account_key,
                batch.client_order_id,
                batch.execution_scope,
                batch.owner_generation,
                opening.opening_version,
                opening.opening_payload_sha256,
                batch.position_key,
                batch.instruction_payload_sha256,
                batch.submission_event_id,
                acknowledgement.event_type,
                batch.submission_position_version,
                batch.submission_observed_at,
                acknowledgement.event_payload_sha256,
                batch.first_account_version,
                batch.last_account_version,
                batch.last_position_version,
                batch.fill_count,
                batch.batch_version,
                batch.batch_payload,
                batch.batch_payload_sha256,
            ),
        )


def _insert_full_account(connection, artifacts):
    opening = artifacts["encoded_opening"]
    admission = artifacts["admission"]
    batch = artifacts["encoded_batch"]
    _insert_journal(connection, artifacts)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO np.paper_account_streams (
                account_key, execution_scope, owner_generation,
                collateral_asset, account_version, account_state,
                opening_version, opening_payload, opening_payload_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            """,
            (
                opening.account_key,
                opening.execution_scope,
                opening.owner_generation,
                opening.collateral_asset,
                len(artifacts["admissions"]),
                admission.after.state.value,
                opening.opening_version,
                opening.opening_payload,
                opening.opening_payload_sha256,
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
                    opening.account_key,
                    balance.asset,
                    str(balance.available),
                    str(balance.reserved),
                ),
            )
        for reservation in admission.after.reservations:
            cursor.execute(
                """
                INSERT INTO np.paper_margin_reservations (
                    account_key, execution_scope, position_key, amount_decimal
                ) VALUES (%s, %s, %s, %s)
                """,
                (
                    opening.account_key,
                    opening.execution_scope,
                    reservation.position_key,
                    str(reservation.amount),
                ),
            )
    _insert_manifest(connection, artifacts)
    with connection.cursor() as cursor:
        for fill_ordinal, (settlement, fill, applied) in enumerate(
            zip(
                artifacts["encoded_settlements"],
                artifacts["encoded_fills"],
                artifacts["admissions"],
            ),
            start=1,
        ):
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
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
                )
                """,
                (
                    settlement.account_key,
                    settlement.account_version,
                    settlement.client_order_id,
                    fill_ordinal,
                    batch.first_account_version,
                    batch.submission_position_version,
                    batch.fill_count,
                    settlement.collateral_asset,
                    settlement.position_key,
                    settlement.position_version,
                    settlement.event_id,
                    settlement.trade_id,
                    fill.event_type,
                    fill.event_payload_sha256,
                    settlement.symbol,
                    settlement.base_asset,
                    settlement.quote_asset,
                    settlement.instrument_version,
                    settlement.settlement_version,
                    settlement.settlement_payload,
                    settlement.settlement_payload_sha256,
                ),
            )
            for posting_ordinal, posting in enumerate(applied.postings, start=1):
                cursor.execute(
                    """
                    INSERT INTO np.paper_account_postings (
                        account_key, account_version, posting_ordinal,
                        asset, bucket, amount_decimal
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        opening.account_key,
                        settlement.account_version,
                        posting_ordinal,
                        posting.asset,
                        posting.bucket.value,
                        str(posting.amount),
                    ),
                )
        cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")


def _canonical_payload(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_provision_replay_list_and_exact_retry_preserve_decimal_identity(
    migrated_postgres_dsn,
):
    repository = _repository(migrated_postgres_dsn)
    account = _opening()

    created = repository.provision_account(
        execution_scope=SCOPE,
        owner_generation=7,
        account=account,
    )
    existing = repository.provision_account(
        execution_scope=SCOPE,
        owner_generation=7,
        account=account,
    )

    assert created.disposition is ProvisionDisposition.CREATED
    assert existing.disposition is ProvisionDisposition.EXISTING
    assert created.account == existing.account == account
    assert (
        repository.replay_account(
            execution_scope=SCOPE,
            account_key="account-1",
        )
        == existing.current
    )
    assert repository.list_accounts(execution_scope=SCOPE) == (existing.current,)

    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT asset, available_decimal, reserved_decimal
                FROM np.paper_account_balances
                WHERE account_key = 'account-1'
                ORDER BY asset
                """)
            assert cursor.fetchall() == [
                ("BNB", "2.500", "0"),
                ("USDT", "100.00", "0.00"),
            ]
            cursor.execute("SELECT count(*) FROM np.paper_account_streams")
            assert cursor.fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("scope", "generation", "account", "kind"),
    (
        (
            "paper:other",
            7,
            _opening(),
            PaperAccountConflictKind.EXECUTION_SCOPE,
        ),
        (
            SCOPE,
            8,
            _opening(),
            PaperAccountConflictKind.OWNER_GENERATION,
        ),
        (
            SCOPE,
            7,
            _opening(margin_quantum=Decimal("0.01")),
            PaperAccountConflictKind.OPENING_IDENTITY,
        ),
    ),
)
def test_provision_conflicts_leave_one_immutable_opening(
    migrated_postgres_dsn,
    scope,
    generation,
    account,
    kind,
):
    repository = _repository(migrated_postgres_dsn)
    repository.provision_account(
        execution_scope=SCOPE,
        owner_generation=7,
        account=_opening(),
    )

    with pytest.raises(PaperAccountConflictError) as caught:
        repository.provision_account(
            execution_scope=scope,
            owner_generation=generation,
            account=account,
        )
    assert caught.value.kind is kind

    assert repository.list_accounts(execution_scope=SCOPE)[0].account == _opening()


def test_concurrent_exact_provision_creates_one_account(migrated_postgres_dsn):
    repository = _repository(migrated_postgres_dsn)

    def provision():
        return repository.provision_account(
            execution_scope=SCOPE,
            owner_generation=7,
            account=_opening(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: provision(), range(2)))

    assert {result.disposition for result in results} == {
        ProvisionDisposition.CREATED,
        ProvisionDisposition.EXISTING,
    }
    connection = _connect(migrated_postgres_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM np.paper_account_streams")
            assert cursor.fetchone() == (1,)
            cursor.execute("SELECT count(*) FROM np.paper_account_balances")
            assert cursor.fetchone() == (2,)
    finally:
        connection.close()


class _CommitThenRaise:
    def __init__(self, connection):
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def commit(self):
        self._connection.commit()
        raise psycopg2.OperationalError("simulated lost commit acknowledgement")


def test_commit_unknown_reconciles_by_exact_provision_retry(migrated_postgres_dsn):
    unknown = PostgresPaperAccountJournal(
        lambda: _CommitThenRaise(_connect(migrated_postgres_dsn))
    )
    with pytest.raises(PaperAccountCommitUnknown):
        unknown.provision_account(
            execution_scope=SCOPE,
            owner_generation=7,
            account=_opening(),
        )

    retry = _repository(migrated_postgres_dsn).provision_account(
        execution_scope=SCOPE,
        owner_generation=7,
        account=_opening(),
    )
    assert retry.disposition is ProvisionDisposition.EXISTING


def test_full_ledger_replay_rederives_manifest_settlement_and_projections(
    migrated_postgres_dsn,
):
    artifacts = _artifacts(fill_quantities=(Decimal("0.5000"), Decimal("0.7300")))
    connection = _connect(migrated_postgres_dsn)
    try:
        _insert_full_account(connection, artifacts)
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT account_version, posting_ordinal, asset, bucket,
                       amount_decimal
                FROM np.paper_account_postings
                WHERE account_key = 'ledger-account'
                ORDER BY account_version, posting_ordinal
                """)
            stored_postings = cursor.fetchall()
    finally:
        connection.close()

    replay = _repository(migrated_postgres_dsn).replay_account(
        execution_scope=SCOPE,
        account_key="ledger-account",
    )
    assert replay.account == artifacts["admission"].after
    assert replay.batches == (artifacts["manifest"],)
    assert tuple(record.account_version for record in replay.account.records) == (1, 2)
    assert tuple(
        record.settlement.record.position_version for record in replay.account.records
    ) == (2, 3)
    assert tuple(record.settlement for record in replay.account.records) == tuple(
        admission.settlement for admission in artifacts["admissions"]
    )
    assert stored_postings == [
        (
            admission.account_version,
            ordinal,
            posting.asset,
            posting.bucket.value,
            str(posting.amount),
        )
        for admission in artifacts["admissions"]
        for ordinal, posting in enumerate(admission.postings, start=1)
    ]


def test_schema_permitted_incomplete_manifest_is_quarantined_on_replay_and_list(
    migrated_postgres_dsn,
):
    artifacts = _artifacts()
    repository = _repository(migrated_postgres_dsn)
    repository.provision_account(
        execution_scope=SCOPE,
        owner_generation=7,
        account=artifacts["opening"],
    )
    connection = _connect(migrated_postgres_dsn)
    try:
        _insert_journal(connection, artifacts)
        _insert_manifest(connection, artifacts)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PaperAccountReplayError, match="missing a settlement"):
        repository.replay_account(
            execution_scope=SCOPE,
            account_key="ledger-account",
        )
    with pytest.raises(PaperAccountReplayError):
        repository.list_accounts(execution_scope=SCOPE)


@pytest.mark.parametrize(
    "mutation",
    (
        "stream_tail",
        "balance",
        "reservation",
        "posting",
        "manifest_settlement_hash",
        "settlement_payload",
    ),
)
def test_replay_fails_closed_on_schema_permitted_corruption(
    migrated_postgres_dsn,
    mutation,
):
    artifacts = _artifacts()
    connection = _connect(migrated_postgres_dsn)
    try:
        _insert_full_account(connection, artifacts)
        connection.commit()
        with connection.cursor() as cursor:
            if mutation == "stream_tail":
                cursor.execute("""
                    UPDATE np.paper_account_streams
                    SET account_version = 0
                    WHERE account_key = 'ledger-account'
                    """)
            elif mutation == "balance":
                cursor.execute("""
                    UPDATE np.paper_account_balances
                    SET available_decimal = '01.00'
                    WHERE account_key = 'ledger-account' AND asset = 'USDT'
                    """)
            elif mutation == "reservation":
                cursor.execute("""
                    UPDATE np.paper_margin_reservations
                    SET amount_decimal = '9.99'
                    WHERE account_key = 'ledger-account'
                    """)
            elif mutation == "posting":
                cursor.execute("""
                    DELETE FROM np.paper_account_postings
                    WHERE account_key = 'ledger-account'
                      AND posting_ordinal = 1
                    """)
            elif mutation == "manifest_settlement_hash":
                cursor.execute("""
                    SELECT batch_payload
                    FROM np.paper_account_batch_manifests
                    WHERE account_key = 'ledger-account'
                    """)
                payload = cursor.fetchone()[0]
                payload["fills"][0]["account_settlement_payload_sha256"] = "0" * 64
                encoded, checksum = _canonical_payload(payload)
                cursor.execute(
                    """
                    UPDATE np.paper_account_batch_manifests
                    SET batch_payload = %s::jsonb,
                        batch_payload_sha256 = %s
                    WHERE account_key = 'ledger-account'
                    """,
                    (encoded, checksum),
                )
            else:
                cursor.execute("""
                    SELECT settlement_payload
                    FROM np.paper_account_settlements
                    WHERE account_key = 'ledger-account'
                    """)
                settlement_payload = cursor.fetchone()[0]
                settlement_payload["account_state_after"] = "INSOLVENT"
                encoded_settlement, settlement_sha = _canonical_payload(
                    settlement_payload
                )
                cursor.execute("""
                    SELECT batch_payload
                    FROM np.paper_account_batch_manifests
                    WHERE account_key = 'ledger-account'
                    """)
                batch_payload = cursor.fetchone()[0]
                batch_payload["fills"][0][
                    "account_settlement_payload_sha256"
                ] = settlement_sha
                encoded_batch, batch_sha = _canonical_payload(batch_payload)
                cursor.execute(
                    """
                    UPDATE np.paper_account_settlements
                    SET settlement_payload = %s::jsonb,
                        settlement_payload_sha256 = %s
                    WHERE account_key = 'ledger-account'
                    """,
                    (encoded_settlement, settlement_sha),
                )
                cursor.execute(
                    """
                    UPDATE np.paper_account_batch_manifests
                    SET batch_payload = %s::jsonb,
                        batch_payload_sha256 = %s
                    WHERE account_key = 'ledger-account'
                    """,
                    (encoded_batch, batch_sha),
                )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PaperAccountReplayError):
        _repository(migrated_postgres_dsn).replay_account(
            execution_scope=SCOPE,
            account_key="ledger-account",
        )


class _PauseAfterStreamCursor:
    def __init__(self, cursor, stream_selected, writer_committed):
        self._cursor = cursor
        self._stream_selected = stream_selected
        self._writer_committed = writer_committed

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._cursor.__exit__(exc_type, exc_value, traceback)

    def execute(self, query, params=None):
        result = (
            self._cursor.execute(query)
            if params is None
            else self._cursor.execute(query, params)
        )
        normalized = " ".join(str(query).split())
        if (
            "FROM np.paper_account_streams" in normalized
            and "WHERE account_key = %s" in normalized
            and "FOR UPDATE" not in normalized
        ):
            self._stream_selected.set()
            if not self._writer_committed.wait(timeout=10):
                raise TimeoutError("concurrent projection change did not commit")
        return result


class _PauseAfterStreamConnection:
    def __init__(self, connection, stream_selected, writer_committed):
        self._connection = connection
        self._stream_selected = stream_selected
        self._writer_committed = writer_committed

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def cursor(self):
        return _PauseAfterStreamCursor(
            self._connection.cursor(),
            self._stream_selected,
            self._writer_committed,
        )


def test_replay_uses_one_repeatable_read_snapshot(migrated_postgres_dsn):
    normal = _repository(migrated_postgres_dsn)
    normal.provision_account(
        execution_scope=SCOPE,
        owner_generation=7,
        account=_opening(),
    )
    stream_selected = Event()
    writer_committed = Event()
    snapshot = PostgresPaperAccountJournal(
        lambda: _PauseAfterStreamConnection(
            _connect(migrated_postgres_dsn),
            stream_selected,
            writer_committed,
        )
    )

    def corrupt_projection_after_stream_select():
        if not stream_selected.wait(timeout=10):
            raise TimeoutError("stream was not selected")
        connection = _connect(migrated_postgres_dsn)
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE np.paper_account_balances
                    SET available_decimal = '99.00'
                    WHERE account_key = 'account-1' AND asset = 'USDT'
                    """)
            connection.commit()
        finally:
            connection.close()
            writer_committed.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(corrupt_projection_after_stream_select)
        first = snapshot.replay_account(
            execution_scope=SCOPE,
            account_key="account-1",
        )
        future.result(timeout=20)

    assert first.account == _opening()
    with pytest.raises(PaperAccountReplayError):
        normal.replay_account(
            execution_scope=SCOPE,
            account_key="account-1",
        )
