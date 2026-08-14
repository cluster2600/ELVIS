"""Contract tests for the pure paper-account journal codecs."""

import ast
import copy
import hashlib
import importlib.util
import json
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone, tzinfo
from decimal import ROUND_DOWN, Decimal, Inexact, Rounded, localcontext
from pathlib import Path

import pytest

from trading.domain.order_lifecycle import ConfirmedFill, SubmissionAcknowledged
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.paper_accounting import (
    PaperAccount,
    PaperAccountAdmission,
    PaperAccountAdmissionDisposition,
    PaperAccountBalance,
    PaperAccountPolicy,
    admit_paper_settlement,
    new_paper_account,
)
from trading.domain.paper_economics import PaperFillRecord
from trading.domain.paper_settlement import (
    PaperLinearInstrument,
    PaperSettlement,
    settle_paper_fill,
)
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionFill,
    PositionInstruction,
    TakeProfitProfile,
)
from trading.persistence.journal_codec import (
    JournalEncodeError,
    JournalQuarantineError,
    encode_order_lifecycle_event,
    encode_position_instruction,
)
from trading.persistence.paper_account_journal_codec import (
    EncodedPaperAccountBatch,
    EncodedPaperAccountOpening,
    EncodedPaperAccountSettlement,
    PaperAccountBatchFill,
    PaperAccountBatchManifest,
    decode_paper_account_batch,
    decode_paper_account_opening,
    decode_paper_account_settlement,
    encode_paper_account_batch,
    encode_paper_account_opening,
    encode_paper_account_settlement,
)

LOCAL_TIME = datetime(
    2026,
    8,
    12,
    14,
    34,
    56,
    123456,
    tzinfo=timezone(timedelta(hours=2)),
)
UTC_TIME = datetime(2026, 8, 12, 12, 34, 56, 123456, tzinfo=timezone.utc)
INSTRUMENT = PaperLinearInstrument("BTCUSDT", "BTC", "USDT")
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


class _TupleSubclass(tuple):
    pass


class _ExplodingOffset(tzinfo):
    def utcoffset(self, value: datetime | None) -> timedelta:
        raise RuntimeError("boom-offset")


def _canonical(payload: object) -> tuple[str, str]:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _row_kwargs(value: object) -> dict[str, object]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _opening_account(
    *,
    margin_quantum: Decimal = Decimal("0.010"),
    bnb: Decimal = Decimal("1.500"),
    usdt: Decimal = Decimal("10.00"),
) -> PaperAccount:
    return new_paper_account(
        PaperAccountPolicy("paper-main", "USDT", margin_quantum),
        (
            PaperAccountBalance("BNB", bnb, Decimal("0")),
            PaperAccountBalance("USDT", usdt, Decimal("0.00")),
        ),
    )


def _instruction(
    *,
    client_order_id: str = "order-1",
    position_key: str = "position-1",
    effect: PositionEffect = PositionEffect.OPEN,
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("2.0000"),
    reference_price: Decimal = Decimal("2.5000"),
    leverage: int = 5,
) -> PositionInstruction:
    intent = OrderIntent(
        client_order_id=client_order_id,
        decision_id=f"decision-{client_order_id}",
        symbol="BTCUSDT",
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        reference_price=reference_price,
        leverage=leverage,
        created_at=LOCAL_TIME,
    )
    return PositionInstruction(
        position_key=position_key,
        effect=effect,
        order_intent=intent,
        exit_context=(
            PositionExitContext(
                take_profit_profile=TakeProfitProfile.RANGING,
                take_profit_fraction=Decimal("0.002500"),
                stop_loss_fraction=Decimal("0.00500"),
                trailing_stop_fraction=None,
            )
            if effect is PositionEffect.OPEN
            else None
        ),
    )


def _fill_record(
    instruction: PositionInstruction,
    *,
    position_version: int,
    event_id: str,
    trade_id: str,
    quantity: Decimal,
    price: Decimal = Decimal("2.5000"),
    fee_amount: Decimal = Decimal("0"),
    fee_asset: str | None = None,
) -> PaperFillRecord:
    fill = ConfirmedFill(
        client_order_id=instruction.order_intent.client_order_id,
        venue_order_id="venue-1",
        trade_id=trade_id,
        symbol=instruction.order_intent.symbol,
        side=instruction.order_intent.side,
        quantity=quantity,
        price=price,
        fee_amount=fee_amount,
        fee_asset=fee_asset,
        executed_at=LOCAL_TIME + timedelta(seconds=position_version),
    )
    return PaperFillRecord(
        position_version=position_version,
        event_id=event_id,
        position_fill=PositionFill(instruction, fill),
    )


def _applied(
    before: PaperAccount,
    account_version: int,
    record: PaperFillRecord,
    *,
    prior_settlement: PaperSettlement | None = None,
) -> PaperAccountAdmission:
    settlement = settle_paper_fill(
        INSTRUMENT,
        None if prior_settlement is None else prior_settlement.after,
        record,
    )
    admission = admit_paper_settlement(before, account_version, settlement)
    assert admission.disposition is PaperAccountAdmissionDisposition.APPLIED
    return admission


def _single_admission(
    *,
    account: PaperAccount | None = None,
    fee_amount: Decimal = Decimal("0.1250"),
    fee_asset: str | None = "BNB",
    price: Decimal = Decimal("2.5000"),
    quantity: Decimal = Decimal("1.2300"),
) -> PaperAccountAdmission:
    before = account or _opening_account()
    instruction = _instruction(quantity=quantity, reference_price=price)
    record = _fill_record(
        instruction,
        position_version=2,
        event_id="fill-event-1",
        trade_id="trade-1",
        quantity=quantity,
        price=price,
        fee_amount=fee_amount,
        fee_asset=fee_asset,
    )
    return _applied(before, 1, record)


def _batch_artifacts() -> tuple[
    PaperAccountBatchManifest,
    tuple[PaperAccountAdmission, PaperAccountAdmission],
]:
    account = _opening_account(usdt=Decimal("100.00"))
    instruction = _instruction()
    first_record = _fill_record(
        instruction,
        position_version=2,
        event_id="fill-event-1",
        trade_id="trade-1",
        quantity=Decimal("0.7500"),
        fee_amount=Decimal("0.0100"),
        fee_asset="BNB",
    )
    first = _applied(account, 1, first_record)
    second_record = _fill_record(
        instruction,
        position_version=3,
        event_id="fill-event-2",
        trade_id="trade-2",
        quantity=Decimal("1.2500"),
        fee_amount=Decimal("0.0200"),
        fee_asset="BNB",
    )
    second = _applied(
        first.after,
        2,
        second_record,
        prior_settlement=first.settlement,
    )
    encoded_instruction = encode_position_instruction(instruction)
    acknowledgement = SubmissionAcknowledged("order-1", "venue-1", LOCAL_TIME)
    encoded_ack = encode_order_lifecycle_event(acknowledgement)
    encoded_first_fill = encode_order_lifecycle_event(first_record.position_fill.fill)
    encoded_second_fill = encode_order_lifecycle_event(second_record.position_fill.fill)
    encoded_first_account = encode_paper_account_settlement(first)
    encoded_second_account = encode_paper_account_settlement(second)
    manifest = PaperAccountBatchManifest(
        execution_scope="paper:test",
        account_key="paper-main",
        owner_generation=7,
        position_key="position-1",
        client_order_id="order-1",
        instruction_payload_sha256=(encoded_instruction.instruction_payload_sha256),
        submission_event_id="submission-attempt",
        submission_position_version=1,
        submission_observed_at=LOCAL_TIME,
        submission_event_payload_sha256=encoded_ack.event_payload_sha256,
        fills=(
            PaperAccountBatchFill(
                position_key="position-1",
                client_order_id="order-1",
                event_id="fill-event-1",
                trade_id="trade-1",
                position_version=2,
                account_version=1,
                event_payload_sha256=(encoded_first_fill.event_payload_sha256),
                account_settlement_payload_sha256=(
                    encoded_first_account.settlement_payload_sha256
                ),
            ),
            PaperAccountBatchFill(
                position_key="position-1",
                client_order_id="order-1",
                event_id="fill-event-2",
                trade_id="trade-2",
                position_version=3,
                account_version=2,
                event_payload_sha256=(encoded_second_fill.event_payload_sha256),
                account_settlement_payload_sha256=(
                    encoded_second_account.settlement_payload_sha256
                ),
            ),
        ),
    )
    return manifest, (first, second)


def _decode_opening(
    encoded: EncodedPaperAccountOpening,
    **overrides: object,
) -> PaperAccount:
    values = _row_kwargs(encoded)
    values.update(overrides)
    return decode_paper_account_opening(**values)


def _decode_settlement(
    admission: PaperAccountAdmission,
    encoded: EncodedPaperAccountSettlement,
    **overrides: object,
) -> PaperAccountAdmission:
    values = _row_kwargs(encoded)
    values.update(overrides)
    return decode_paper_account_settlement(
        admission.before,
        admission.settlement,
        **values,
    )


def _decode_batch(
    encoded: EncodedPaperAccountBatch,
    **overrides: object,
) -> PaperAccountBatchManifest:
    values = _row_kwargs(encoded)
    values.update(overrides)
    return decode_paper_account_batch(**values)


def test_opening_golden_vector_and_round_trip_preserve_decimal_identity() -> None:
    account = _opening_account()

    encoded = encode_paper_account_opening("paper:test", 7, account)
    decoded = _decode_opening(encoded)

    assert encoded.opening_version == 1
    assert encoded.opening_payload == OPENING_GOLDEN_JSON
    assert encoded.opening_payload_sha256 == OPENING_GOLDEN_SHA
    assert decoded == account
    assert decoded.policy.margin_quantum.as_tuple() == Decimal("0.010").as_tuple()
    assert decoded.opening_balances[0].available.as_tuple() == (
        Decimal("1.500").as_tuple()
    )
    assert decoded.opening_balances[1].available.as_tuple() == (
        Decimal("10.00").as_tuple()
    )


def test_settlement_golden_vector_is_compact_and_rederived_on_decode() -> None:
    admission = _single_admission()

    encoded = encode_paper_account_settlement(admission)
    decoded = _decode_settlement(admission, encoded)

    assert encoded.instrument_version == 1
    assert encoded.settlement_version == 1
    assert encoded.settlement_payload == SETTLEMENT_GOLDEN_JSON
    assert encoded.settlement_payload_sha256 == SETTLEMENT_GOLDEN_SHA
    assert decoded == admission
    payload = json.loads(encoded.settlement_payload)
    assert set(payload) == {
        "account",
        "account_state_after",
        "disposition",
        "instrument",
        "position_margin_after",
        "postings",
        "settlement_deltas",
        "settlement_ref",
    }
    assert "before" not in payload
    assert "after" not in payload
    assert "records" not in encoded.settlement_payload


def test_batch_golden_vector_round_trips_all_provenance_links() -> None:
    manifest, admissions = _batch_artifacts()

    encoded = encode_paper_account_batch(manifest)
    decoded = _decode_batch(encoded)

    assert encoded.batch_version == 1
    assert encoded.runtime_generation is None
    assert encoded.batch_payload == BATCH_GOLDEN_JSON
    assert encoded.batch_payload_sha256 == BATCH_GOLDEN_SHA
    assert decoded == replace(manifest, submission_observed_at=UTC_TIME)
    assert encoded.submission_observed_at == UTC_TIME
    assert encoded.first_account_version == 1
    assert encoded.last_account_version == 2
    assert encoded.last_position_version == 3
    assert encoded.fill_count == 2
    assert tuple(fill.account_settlement_payload_sha256 for fill in decoded.fills) == (
        encode_paper_account_settlement(admissions[0]).settlement_payload_sha256,
        encode_paper_account_settlement(admissions[1]).settlement_payload_sha256,
    )


def test_runtime_generation_batch_v2_has_exact_golden_vector_and_round_trip() -> None:
    manifest, admissions = _batch_artifacts()
    versioned = replace(manifest, runtime_generation=42)

    encoded = encode_paper_account_batch(versioned)
    decoded = _decode_batch(encoded)

    assert encoded.batch_version == 2
    assert encoded.runtime_generation == 42
    assert encoded.batch_payload == BATCH_V2_GOLDEN_JSON
    assert encoded.batch_payload_sha256 == BATCH_V2_GOLDEN_SHA
    assert decoded == replace(versioned, submission_observed_at=UTC_TIME)
    assert decoded.runtime_generation == 42
    assert tuple(fill.account_settlement_payload_sha256 for fill in decoded.fills) == (
        encode_paper_account_settlement(admissions[0]).settlement_payload_sha256,
        encode_paper_account_settlement(admissions[1]).settlement_payload_sha256,
    )


def test_adding_runtime_generation_changes_only_the_v2_envelope_field() -> None:
    manifest, _ = _batch_artifacts()

    legacy = encode_paper_account_batch(manifest)
    active = encode_paper_account_batch(replace(manifest, runtime_generation=42))
    legacy_payload = json.loads(legacy.batch_payload)
    active_payload = json.loads(active.batch_payload)

    assert active_payload.pop("runtime_generation") == 42
    assert active_payload == legacy_payload
    assert legacy.batch_version == 1
    assert legacy.runtime_generation is None
    assert legacy.batch_payload == BATCH_GOLDEN_JSON
    assert legacy.batch_payload_sha256 == BATCH_GOLDEN_SHA


@pytest.mark.parametrize("codec", ("opening", "settlement", "batch"))
def test_payload_accepts_jsonb_objects_and_noncanonical_json_text(codec: str) -> None:
    if codec == "opening":
        encoded = encode_paper_account_opening("paper:test", 7, _opening_account())
        expected = _decode_opening(encoded)
        decode = _decode_opening
        payload_field = "opening_payload"
    elif codec == "settlement":
        admission = _single_admission()
        encoded = encode_paper_account_settlement(admission)
        expected = admission

        def decode(value, **overrides):
            return _decode_settlement(admission, value, **overrides)

        payload_field = "settlement_payload"
    else:
        manifest, _ = _batch_artifacts()
        encoded = encode_paper_account_batch(manifest)
        expected = replace(manifest, submission_observed_at=UTC_TIME)
        decode = _decode_batch
        payload_field = "batch_payload"

    payload = json.loads(getattr(encoded, payload_field))

    assert decode(encoded, **{payload_field: payload}) == expected
    assert decode(encoded, **{payload_field: json.dumps(payload, indent=2)}) == expected


def test_opening_extreme_decimals_ignore_hostile_context() -> None:
    account = _opening_account(
        margin_quantum=Decimal("1E-20000"),
        bnb=Decimal("1E+200000"),
        usdt=Decimal("9.9900E+99999"),
    )

    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        decoded = _decode_opening(
            encode_paper_account_opening("paper:test", 7, account)
        )

    assert decoded.policy.margin_quantum.as_tuple() == (
        account.policy.margin_quantum.as_tuple()
    )
    assert tuple(value.available.as_tuple() for value in decoded.balances) == tuple(
        value.available.as_tuple() for value in account.balances
    )


@pytest.mark.parametrize(
    ("fee_amount", "fee_asset"),
    (
        (Decimal("0"), None),
        (Decimal("0.1250"), "USDT"),
        (Decimal("0.1250"), "BNB"),
    ),
)
def test_applied_settlement_variants_round_trip(
    fee_amount: Decimal,
    fee_asset: str | None,
) -> None:
    admission = _single_admission(fee_amount=fee_amount, fee_asset=fee_asset)

    decoded = _decode_settlement(
        admission,
        encode_paper_account_settlement(admission),
    )

    assert decoded == admission
    assert tuple(posting.amount.as_tuple() for posting in decoded.postings) == tuple(
        posting.amount.as_tuple() for posting in admission.postings
    )


def test_scale_in_partial_and_full_reduction_rows_round_trip() -> None:
    account = _opening_account(usdt=Decimal("100"))
    opened_record = _fill_record(
        _instruction(quantity=Decimal("1"), reference_price=Decimal("3")),
        position_version=2,
        event_id="open-fill",
        trade_id="open-trade",
        quantity=Decimal("1"),
        price=Decimal("3"),
    )
    opened = _applied(account, 1, opened_record)
    scaled_record = _fill_record(
        _instruction(
            client_order_id="scale-1",
            quantity=Decimal("2"),
            reference_price=Decimal("4"),
        ),
        position_version=4,
        event_id="scale-fill",
        trade_id="scale-trade",
        quantity=Decimal("2"),
        price=Decimal("4"),
    )
    scaled = _applied(
        opened.after,
        2,
        scaled_record,
        prior_settlement=opened.settlement,
    )
    partial_record = _fill_record(
        _instruction(
            client_order_id="reduce-1",
            effect=PositionEffect.REDUCE_ONLY,
            side=OrderSide.SELL,
            quantity=Decimal("1.5"),
            reference_price=Decimal("5"),
        ),
        position_version=6,
        event_id="partial-fill",
        trade_id="partial-trade",
        quantity=Decimal("1.5"),
        price=Decimal("5"),
    )
    partial = _applied(
        scaled.after,
        3,
        partial_record,
        prior_settlement=scaled.settlement,
    )
    full_record = _fill_record(
        _instruction(
            client_order_id="reduce-2",
            effect=PositionEffect.REDUCE_ONLY,
            side=OrderSide.SELL,
            quantity=Decimal("1.5"),
            reference_price=Decimal("2"),
        ),
        position_version=8,
        event_id="full-fill",
        trade_id="full-trade",
        quantity=Decimal("1.5"),
        price=Decimal("2"),
    )
    full = _applied(
        partial.after,
        4,
        full_record,
        prior_settlement=partial.settlement,
    )

    for admission in (opened, scaled, partial, full):
        encoded = encode_paper_account_settlement(admission)
        assert _decode_settlement(admission, encoded) == admission

    assert (
        json.loads(encode_paper_account_settlement(full).settlement_payload)[
            "position_margin_after"
        ]
        is None
    )


def test_opening_encoder_requires_an_empty_account() -> None:
    admission = _single_admission()

    with pytest.raises(JournalEncodeError, match="empty paper account"):
        encode_paper_account_opening("paper:test", 7, admission.after)


def test_settlement_encoder_rejects_replayed_and_rejected_admissions() -> None:
    applied = _single_admission()
    replayed = admit_paper_settlement(
        applied.after,
        1,
        applied.settlement,
    )
    rejected_account = _opening_account(usdt=Decimal("0.01"))
    rejected = admit_paper_settlement(
        rejected_account,
        1,
        _single_admission(account=_opening_account()).settlement,
    )

    assert replayed.disposition is PaperAccountAdmissionDisposition.REPLAYED
    assert rejected.disposition is PaperAccountAdmissionDisposition.REJECTED
    for admission in (replayed, rejected):
        with pytest.raises(JournalEncodeError, match="only APPLIED"):
            encode_paper_account_settlement(admission)


@pytest.mark.parametrize("value", (object(), None, ()))
def test_encoders_reject_wrong_root_types(value: object) -> None:
    with pytest.raises(JournalEncodeError):
        encode_paper_account_opening("paper:test", 7, value)
    with pytest.raises(JournalEncodeError):
        encode_paper_account_settlement(value)
    with pytest.raises(JournalEncodeError):
        encode_paper_account_batch(value)


@pytest.mark.parametrize("version", (True, 0, 2, "1", None))
def test_unknown_or_ill_typed_envelope_versions_are_quarantined(
    version: object,
) -> None:
    opening = encode_paper_account_opening("paper:test", 7, _opening_account())
    admission = _single_admission()
    settlement = encode_paper_account_settlement(admission)
    with pytest.raises(JournalQuarantineError, match="version is unknown"):
        _decode_opening(opening, opening_version=version)
    with pytest.raises(JournalQuarantineError, match="version is unknown"):
        _decode_settlement(admission, settlement, settlement_version=version)
    with pytest.raises(JournalQuarantineError, match="version is unknown"):
        _decode_settlement(admission, settlement, instrument_version=version)


@pytest.mark.parametrize("version", (True, 0, 3, "1", None))
def test_unknown_or_ill_typed_batch_versions_are_quarantined(
    version: object,
) -> None:
    manifest, _ = _batch_artifacts()
    batch = encode_paper_account_batch(manifest)

    with pytest.raises(JournalQuarantineError, match="version is unknown"):
        _decode_batch(batch, batch_version=version)


@pytest.mark.parametrize("value", (True, 0, -1, 1 << 63, 1.0, "1"))
def test_manifest_runtime_generation_rejects_bool_bounds_and_wrong_types(
    value: object,
) -> None:
    manifest, _ = _batch_artifacts()

    with pytest.raises((TypeError, ValueError), match="storage bounds|integer"):
        replace(manifest, runtime_generation=value)


@pytest.mark.parametrize(
    ("batch_version", "runtime_generation"),
    (
        (1, 1),
        (2, None),
        (2, True),
        (2, 0),
        (2, -1),
        (2, 1 << 63),
        (2, 1.0),
        (3, None),
    ),
)
def test_encoded_batch_rejects_incoherent_version_generation_pairs(
    batch_version: object,
    runtime_generation: object,
) -> None:
    encoded = encode_paper_account_batch(_manifest())

    with pytest.raises(JournalEncodeError):
        replace(
            encoded,
            batch_version=batch_version,
            runtime_generation=runtime_generation,
        )


@pytest.mark.parametrize(
    ("batch_version", "runtime_generation"),
    (
        (1, 1),
        (2, None),
        (2, True),
        (2, 0),
        (2, -1),
        (2, 1 << 63),
        (2, 1.0),
        (2, "1"),
    ),
)
def test_batch_decoder_quarantines_incoherent_version_generation_pairs(
    batch_version: object,
    runtime_generation: object,
) -> None:
    manifest, _ = _batch_artifacts()
    encoded = encode_paper_account_batch(manifest)

    with pytest.raises(JournalQuarantineError):
        _decode_batch(
            encoded,
            batch_version=batch_version,
            runtime_generation=runtime_generation,
        )


@pytest.mark.parametrize("mutation", ("missing", "indexed_drift", "payload_drift"))
def test_batch_v2_runtime_generation_is_required_and_cross_checked(
    mutation: str,
) -> None:
    manifest, _ = _batch_artifacts()
    encoded = encode_paper_account_batch(replace(manifest, runtime_generation=42))
    overrides: dict[str, object] = {}
    if mutation == "missing":
        payload = json.loads(encoded.batch_payload)
        payload.pop("runtime_generation")
        overrides["batch_payload"], overrides["batch_payload_sha256"] = _canonical(
            payload
        )
    elif mutation == "indexed_drift":
        overrides["runtime_generation"] = 43
    else:
        payload = json.loads(encoded.batch_payload)
        payload["runtime_generation"] = 43
        overrides["batch_payload"], overrides["batch_payload_sha256"] = _canonical(
            payload
        )

    with pytest.raises(JournalQuarantineError):
        _decode_batch(encoded, **overrides)


@pytest.mark.parametrize("value", (True, 0, -1, 1 << 63, 1.0, "1"))
def test_durable_integer_columns_reject_bool_bounds_and_wrong_types(
    value: object,
) -> None:
    opening = encode_paper_account_opening("paper:test", 7, _opening_account())
    admission = _single_admission()
    settlement = encode_paper_account_settlement(admission)
    manifest, _ = _batch_artifacts()
    batch = encode_paper_account_batch(manifest)

    with pytest.raises(JournalQuarantineError, match="durable storage bounds|integer"):
        _decode_opening(opening, owner_generation=value)
    with pytest.raises(JournalQuarantineError, match="durable storage bounds|integer"):
        _decode_settlement(admission, settlement, account_version=value)
    with pytest.raises(JournalQuarantineError, match="durable storage bounds|integer"):
        _decode_settlement(admission, settlement, position_version=value)
    with pytest.raises(JournalQuarantineError, match="durable storage bounds|integer"):
        _decode_batch(batch, fill_count=value)


@pytest.mark.parametrize("field", ("opening", "settlement", "batch"))
@pytest.mark.parametrize("checksum", ("0" * 64, "A" * 64, "0" * 63, None))
def test_payload_checksum_drift_and_noncanonical_hashes_are_quarantined(
    field: str,
    checksum: object,
) -> None:
    if field == "opening":
        encoded = encode_paper_account_opening("paper:test", 7, _opening_account())
        with pytest.raises(JournalQuarantineError, match="SHA-256"):
            _decode_opening(encoded, opening_payload_sha256=checksum)
    elif field == "settlement":
        admission = _single_admission()
        encoded = encode_paper_account_settlement(admission)
        with pytest.raises(JournalQuarantineError, match="SHA-256"):
            _decode_settlement(
                admission,
                encoded,
                settlement_payload_sha256=checksum,
            )
    else:
        manifest, _ = _batch_artifacts()
        encoded = encode_paper_account_batch(manifest)
        with pytest.raises(JournalQuarantineError, match="SHA-256"):
            _decode_batch(encoded, batch_payload_sha256=checksum)


@pytest.mark.parametrize("codec", ("opening", "settlement", "batch"))
def test_duplicate_json_keys_are_quarantined_before_hash_comparison(
    codec: str,
) -> None:
    if codec == "opening":
        encoded = encode_paper_account_opening("paper:test", 7, _opening_account())
        duplicate = '{"execution_scope":"paper:test",' + encoded.opening_payload[1:]
        with pytest.raises(JournalQuarantineError, match="strict JSON"):
            _decode_opening(encoded, opening_payload=duplicate)
    elif codec == "settlement":
        admission = _single_admission()
        encoded = encode_paper_account_settlement(admission)
        duplicate = '{"disposition":"APPLIED",' + encoded.settlement_payload[1:]
        with pytest.raises(JournalQuarantineError, match="strict JSON"):
            _decode_settlement(admission, encoded, settlement_payload=duplicate)
    else:
        manifest, _ = _batch_artifacts()
        encoded = encode_paper_account_batch(manifest)
        duplicate = '{"account_key":"paper-main",' + encoded.batch_payload[1:]
        with pytest.raises(JournalQuarantineError, match="strict JSON"):
            _decode_batch(encoded, batch_payload=duplicate)


@pytest.mark.parametrize("mutation", ("missing", "extra", "nested_extra"))
def test_opening_unknown_missing_and_nested_payload_keys_are_quarantined(
    mutation: str,
) -> None:
    encoded = encode_paper_account_opening("paper:test", 7, _opening_account())
    payload = json.loads(encoded.opening_payload)
    if mutation == "missing":
        payload.pop("owner_generation")
    elif mutation == "extra":
        payload["unknown"] = None
    else:
        payload["policy"]["unknown"] = None
    payload_text, payload_sha = _canonical(payload)

    with pytest.raises(JournalQuarantineError, match="payload shape"):
        _decode_opening(
            encoded,
            opening_payload=payload_text,
            opening_payload_sha256=payload_sha,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("policy", "margin_quantum"), "0.0100"),
        (("opening_balances", 0, "available"), "1.50"),
        (("opening_balances", 0, "reserved"), "0.000"),
    ),
)
def test_opening_payload_is_authoritative_for_exact_decimal_identity(
    path: tuple[object, ...],
    value: object,
) -> None:
    encoded = encode_paper_account_opening("paper:test", 7, _opening_account())
    payload = json.loads(encoded.opening_payload)
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    payload_text, payload_sha = _canonical(payload)

    decoded = _decode_opening(
        encoded,
        opening_payload=payload_text,
        opening_payload_sha256=payload_sha,
    )

    if path == ("policy", "margin_quantum"):
        actual = decoded.policy.margin_quantum
    elif path[-1] == "available":
        actual = decoded.opening_balances[0].available
    else:
        actual = decoded.opening_balances[0].reserved
    assert actual.as_tuple() == Decimal(value).as_tuple()


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("owner_generation",), 8),
        (("execution_scope",), "paper:other"),
    ),
)
def test_opening_payload_cannot_drift_from_indexed_provenance(
    path: tuple[object, ...],
    value: object,
) -> None:
    encoded = encode_paper_account_opening("paper:test", 7, _opening_account())
    payload = json.loads(encoded.opening_payload)
    payload[path[0]] = value
    payload_text, payload_sha = _canonical(payload)

    with pytest.raises(JournalQuarantineError, match="indexed columns conflict"):
        _decode_opening(
            encoded,
            opening_payload=payload_text,
            opening_payload_sha256=payload_sha,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("execution_scope", "paper:other"),
        ("account_key", "other-account"),
        ("owner_generation", 8),
        ("collateral_asset", "BTC"),
    ),
)
def test_opening_indexed_column_drift_is_quarantined(
    field: str,
    value: object,
) -> None:
    encoded = encode_paper_account_opening("paper:test", 7, _opening_account())

    with pytest.raises(JournalQuarantineError, match="indexed columns conflict"):
        _decode_opening(encoded, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("account_key", "other-account"),
        ("collateral_asset", "BTC"),
        ("account_version", 2),
        ("position_key", "other-position"),
        ("position_version", 3),
        ("client_order_id", "other-order"),
        ("event_id", "other-event"),
        ("trade_id", "other-trade"),
        ("symbol", "ETHUSDT"),
        ("base_asset", "ETH"),
        ("quote_asset", "BTC"),
    ),
)
def test_settlement_indexed_column_drift_is_quarantined(
    field: str,
    value: object,
) -> None:
    admission = _single_admission()
    encoded = encode_paper_account_settlement(admission)

    with pytest.raises(JournalQuarantineError):
        _decode_settlement(admission, encoded, **{field: value})


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("disposition",), "REPLAYED"),
        (("account", "account_key"), "other-account"),
        (("settlement_ref", "account_version"), 2),
        (("instrument", "version"), 2),
        (("instrument", "symbol"), "ETHUSDT"),
        (("settlement_deltas", "fee_debits"), []),
        (("postings",), []),
        (("account_state_after",), "INSOLVENT"),
        (("position_margin_after",), "0.610"),
    ),
)
def test_settlement_payload_tampering_with_a_valid_hash_is_quarantined(
    path: tuple[object, ...],
    value: object,
) -> None:
    admission = _single_admission()
    encoded = encode_paper_account_settlement(admission)
    payload = json.loads(encoded.settlement_payload)
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    payload_text, payload_sha = _canonical(payload)

    with pytest.raises(JournalQuarantineError, match="conflicts with the domain"):
        _decode_settlement(
            admission,
            encoded,
            settlement_payload=payload_text,
            settlement_payload_sha256=payload_sha,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "nested_extra"))
def test_settlement_payload_shape_tampering_is_quarantined(mutation: str) -> None:
    admission = _single_admission()
    encoded = encode_paper_account_settlement(admission)
    payload = json.loads(encoded.settlement_payload)
    if mutation == "missing":
        payload.pop("postings")
    elif mutation == "extra":
        payload["unknown"] = None
    else:
        payload["instrument"]["unknown"] = None
    payload_text, payload_sha = _canonical(payload)

    with pytest.raises(JournalQuarantineError):
        _decode_settlement(
            admission,
            encoded,
            settlement_payload=payload_text,
            settlement_payload_sha256=payload_sha,
        )


def test_settlement_decoder_requires_exact_validated_domain_inputs() -> None:
    admission = _single_admission()
    encoded = encode_paper_account_settlement(admission)
    values = _row_kwargs(encoded)

    with pytest.raises(JournalQuarantineError, match="validated domain values"):
        decode_paper_account_settlement(object(), admission.settlement, **values)
    with pytest.raises(JournalQuarantineError, match="validated domain values"):
        decode_paper_account_settlement(admission.before, object(), **values)


def test_settlement_decoder_rejects_a_replay_or_wrong_prior_account() -> None:
    admission = _single_admission()
    encoded = encode_paper_account_settlement(admission)
    values = _row_kwargs(encoded)

    with pytest.raises(JournalQuarantineError):
        decode_paper_account_settlement(
            admission.after,
            admission.settlement,
            **values,
        )


def _manifest(**overrides: object) -> PaperAccountBatchManifest:
    values = {
        "execution_scope": "paper:test",
        "account_key": "paper-main",
        "owner_generation": 7,
        "position_key": "position-1",
        "client_order_id": "order-1",
        "instruction_payload_sha256": SHA_A,
        "submission_event_id": "submission-attempt",
        "submission_position_version": 10,
        "submission_observed_at": LOCAL_TIME,
        "submission_event_payload_sha256": SHA_B,
        "fills": (
            PaperAccountBatchFill(
                "position-1",
                "order-1",
                "fill-1",
                "trade-1",
                11,
                20,
                SHA_C,
                SHA_D,
            ),
            PaperAccountBatchFill(
                "position-1",
                "order-1",
                "fill-2",
                "trade-2",
                12,
                21,
                SHA_D,
                SHA_C,
            ),
        ),
    }
    values.update(overrides)
    return PaperAccountBatchManifest(**values)


@pytest.mark.parametrize(
    "fills",
    (
        (),
        [],
        _TupleSubclass(
            (
                PaperAccountBatchFill(
                    "position-1",
                    "order-1",
                    "fill-1",
                    "trade-1",
                    11,
                    20,
                    SHA_C,
                    SHA_D,
                ),
            )
        ),
        (object(),),
    ),
)
def test_manifest_requires_a_nonempty_exact_tuple_of_exact_fill_refs(
    fills: object,
) -> None:
    with pytest.raises(TypeError):
        _manifest(fills=fills)


@pytest.mark.parametrize(
    "fills",
    (
        (
            PaperAccountBatchFill(
                "other-position", "order-1", "fill-1", "trade-1", 11, 20, SHA_C, SHA_D
            ),
        ),
        (
            PaperAccountBatchFill(
                "position-1", "other-order", "fill-1", "trade-1", 11, 20, SHA_C, SHA_D
            ),
        ),
        (
            PaperAccountBatchFill(
                "position-1", "order-1", "fill-1", "trade-1", 12, 20, SHA_C, SHA_D
            ),
        ),
        (
            PaperAccountBatchFill(
                "position-1", "order-1", "fill-1", "trade-1", 11, 20, SHA_C, SHA_D
            ),
            PaperAccountBatchFill(
                "position-1", "order-1", "fill-2", "trade-2", 12, 22, SHA_D, SHA_C
            ),
        ),
        (
            PaperAccountBatchFill(
                "position-1",
                "order-1",
                "submission-attempt",
                "trade-1",
                11,
                20,
                SHA_C,
                SHA_D,
            ),
        ),
        (
            PaperAccountBatchFill(
                "position-1", "order-1", "fill-1", "trade-1", 11, 20, SHA_C, SHA_D
            ),
            PaperAccountBatchFill(
                "position-1", "order-1", "fill-2", "trade-1", 12, 21, SHA_D, SHA_C
            ),
        ),
    ),
)
def test_manifest_rejects_wrong_owner_noncontiguous_versions_and_duplicate_ids(
    fills: tuple[PaperAccountBatchFill, ...],
) -> None:
    with pytest.raises(ValueError):
        _manifest(fills=fills)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("execution_scope", "paper:other"),
        ("account_key", "other-account"),
        ("owner_generation", 8),
        ("position_key", "other-position"),
        ("client_order_id", "other-order"),
        ("instruction_payload_sha256", "e" * 64),
        ("submission_event_id", "other-submission"),
        ("submission_position_version", 2),
        ("submission_observed_at", UTC_TIME + timedelta(seconds=1)),
        ("first_account_version", 2),
        ("last_account_version", 3),
        ("last_position_version", 4),
        ("fill_count", 1),
    ),
)
def test_batch_indexed_column_drift_is_quarantined(
    field: str,
    value: object,
) -> None:
    manifest, _ = _batch_artifacts()
    encoded = encode_paper_account_batch(manifest)

    with pytest.raises(JournalQuarantineError):
        _decode_batch(encoded, **{field: value})


def test_batch_decoder_quarantines_a_timezone_failure() -> None:
    manifest, _ = _batch_artifacts()
    encoded = encode_paper_account_batch(manifest)
    hostile = datetime(2026, 8, 12, tzinfo=_ExplodingOffset())

    with pytest.raises(JournalQuarantineError, match="represented in UTC"):
        _decode_batch(encoded, submission_observed_at=hostile)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("execution_scope",), "paper:other"),
        (("instruction_payload_sha256",), "e" * 64),
        (("submission", "event_id"), "other-submission"),
        (("submission", "observed_at"), "2026-08-12T12:34:57.123456+00:00"),
        (("fills", 0, "event_payload_sha256"), "e" * 64),
        (("fills", 0, "account_settlement_payload_sha256"), "e" * 64),
    ),
)
def test_batch_payload_links_cannot_drift_from_indexed_provenance(
    path: tuple[object, ...],
    value: object,
) -> None:
    manifest, _ = _batch_artifacts()
    encoded = encode_paper_account_batch(manifest)
    payload = json.loads(encoded.batch_payload)
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    payload_text, payload_sha = _canonical(payload)

    if path[0] == "fills":
        decoded = _decode_batch(
            encoded,
            batch_payload=payload_text,
            batch_payload_sha256=payload_sha,
        )
        assert decoded != replace(manifest, submission_observed_at=UTC_TIME)
        assert encode_paper_account_batch(decoded).batch_payload_sha256 == payload_sha
    else:
        with pytest.raises(JournalQuarantineError, match="indexed columns conflict"):
            _decode_batch(
                encoded,
                batch_payload=payload_text,
                batch_payload_sha256=payload_sha,
            )


@pytest.mark.parametrize("mutation", ("missing", "extra", "nested_extra", "fill_extra"))
def test_batch_unknown_missing_and_nested_payload_keys_are_quarantined(
    mutation: str,
) -> None:
    manifest, _ = _batch_artifacts()
    encoded = encode_paper_account_batch(manifest)
    payload = json.loads(encoded.batch_payload)
    if mutation == "missing":
        payload.pop("fills")
    elif mutation == "extra":
        payload["unknown"] = None
    elif mutation == "nested_extra":
        payload["submission"]["unknown"] = None
    else:
        payload["fills"][0]["unknown"] = None
    payload_text, payload_sha = _canonical(payload)

    with pytest.raises(JournalQuarantineError, match="payload shape"):
        _decode_batch(
            encoded,
            batch_payload=payload_text,
            batch_payload_sha256=payload_sha,
        )


def test_manifest_sha_links_are_identity_bearing_not_numeric_or_lossy() -> None:
    manifest = _manifest()

    encoded = encode_paper_account_batch(manifest)
    changed = replace(
        manifest,
        fills=(replace(manifest.fills[0], event_payload_sha256="e" * 64),)
        + manifest.fills[1:],
    )

    assert encode_paper_account_batch(changed).batch_payload_sha256 != (
        encoded.batch_payload_sha256
    )
    assert json.loads(encoded.batch_payload)["instruction_payload_sha256"] == SHA_A
    assert (
        json.loads(encoded.batch_payload)["fills"][0][
            "account_settlement_payload_sha256"
        ]
        == SHA_D
    )


def test_encoded_batch_direct_construction_rejects_incoherent_ranges() -> None:
    encoded = encode_paper_account_batch(_manifest())

    with pytest.raises(JournalEncodeError, match="position-version range"):
        replace(encoded, last_position_version=13)
    with pytest.raises(JournalEncodeError, match="account-version range"):
        replace(encoded, last_account_version=22)
    with pytest.raises(JournalEncodeError):
        replace(encoded, fill_count=True)


def test_manifest_and_fill_direct_construction_validate_sha_and_time() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="SHA-256"):
        replace(manifest, instruction_payload_sha256="A" * 64)
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(manifest, submission_observed_at=LOCAL_TIME.replace(tzinfo=None))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(manifest.fills[0], event_payload_sha256="short")
    with pytest.raises(ValueError, match="integer"):
        replace(manifest.fills[0], account_version=True)


def test_encoded_and_manifest_values_are_frozen_slotted_and_hashable() -> None:
    account = _opening_account()
    opening = encode_paper_account_opening("paper:test", 7, account)
    admission = _single_admission()
    settlement = encode_paper_account_settlement(admission)
    manifest, _ = _batch_artifacts()
    batch = encode_paper_account_batch(manifest)
    values = (opening, settlement, manifest.fills[0], manifest, batch)

    assert all(not hasattr(value, "__dict__") for value in values)
    assert all(isinstance(hash(value), int) for value in values)
    for value in values:
        with pytest.raises(FrozenInstanceError):
            setattr(value, fields(value)[0].name, None)


def test_reachable_codec_values_reject_setstate_mutation() -> None:
    manifest, _ = _batch_artifacts()
    values = (
        encode_paper_account_opening("paper:test", 7, _opening_account()),
        encode_paper_account_settlement(_single_admission()),
        manifest.fills[0],
        manifest,
        encode_paper_account_batch(manifest),
    )

    for value in values:
        state = [getattr(value, field.name) for field in fields(value)]
        assert hasattr(value, "__setstate__")
        with pytest.raises(TypeError, match="state mutation"):
            value.__setstate__(state)


def test_copy_and_pickle_round_trips_revalidate_codec_values() -> None:
    manifest, _ = _batch_artifacts()
    values = (
        encode_paper_account_opening("paper:test", 7, _opening_account()),
        encode_paper_account_settlement(_single_admission()),
        manifest.fills[0],
        manifest,
        encode_paper_account_batch(manifest),
    )

    for value in values:
        for restored in (
            copy.copy(value),
            copy.deepcopy(value),
            pickle.loads(pickle.dumps(value)),
        ):
            assert restored == value
            assert hash(restored) == hash(value)


@pytest.mark.parametrize("protocol", range(6))
def test_v2_manifest_and_encoded_batch_copy_and_pickle_protocols(protocol: int) -> None:
    manifest, _ = _batch_artifacts()
    manifest = replace(manifest, runtime_generation=42)
    encoded = encode_paper_account_batch(manifest)

    for value in (manifest, encoded):
        for restored in (
            copy.copy(value),
            copy.deepcopy(value),
            pickle.loads(pickle.dumps(value, protocol=protocol)),
        ):
            assert restored == value
            assert restored.runtime_generation == 42
            assert hash(restored) == hash(value)


def test_v2_manifest_and_encoded_batch_reject_setstate_mutation() -> None:
    manifest, _ = _batch_artifacts()
    manifest = replace(manifest, runtime_generation=42)

    for value in (manifest, encode_paper_account_batch(manifest)):
        state = [getattr(value, field.name) for field in fields(value)]
        assert hasattr(value, "__setstate__")
        with pytest.raises(TypeError, match="state mutation"):
            value.__setstate__(state)


def test_non_objects_invalid_constants_and_excessive_nesting_are_quarantined() -> None:
    opening = encode_paper_account_opening("paper:test", 7, _opening_account())
    for payload in ("NaN", "[]", "null"):
        with pytest.raises(JournalQuarantineError):
            _decode_opening(opening, opening_payload=payload)

    depth = 150_000
    nested_text = ("[" * depth) + ("]" * depth)
    with pytest.raises(JournalQuarantineError, match="strict JSON"):
        _decode_opening(opening, opening_payload=nested_text)

    nested_object: object = []
    for _ in range(depth):
        nested_object = [nested_object]
    with pytest.raises(JournalQuarantineError, match="canonical JSON"):
        _decode_opening(opening, opening_payload={"policy": nested_object})


CODEC_EXPORTS = {
    "EncodedPaperAccountBatch",
    "EncodedPaperAccountOpening",
    "EncodedPaperAccountSettlement",
    "PaperAccountBatchFill",
    "PaperAccountBatchManifest",
    "decode_paper_account_batch",
    "decode_paper_account_opening",
    "decode_paper_account_settlement",
    "encode_paper_account_batch",
    "encode_paper_account_opening",
    "encode_paper_account_settlement",
}


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return (*parent, node.attr) if parent is not None else None
    return None


def _uses_account_codec(source: str) -> bool:
    """Detect direct, facade, aliased, relative, and literal dynamic imports."""
    tree = ast.parse(source)
    module = "trading.persistence.paper_account_journal_codec"
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}
    builtin_import_aliases = {"__import__"}
    trading_aliases: set[str] = set()
    persistence_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module or alias.name.startswith(f"{module}."):
                    return True
                if alias.name == "trading":
                    trading_aliases.add(alias.asname or alias.name)
                elif alias.name == "trading.persistence":
                    persistence_aliases.add(alias.asname or "persistence")
                    if alias.asname is None:
                        trading_aliases.add("trading")
                elif alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            imported_module = node.module or ""
            if imported_module == module or (
                node.level and imported_module.endswith("paper_account_journal_codec")
            ):
                return True
            if imported_module == "trading.persistence" or (
                node.level and imported_module == "persistence"
            ):
                if imported & (CODEC_EXPORTS | {"paper_account_journal_codec", "*"}):
                    return True
            if imported_module == "trading" and "persistence" in imported:
                persistence_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "persistence"
                )
            if imported_module == "importlib" and "import_module" in imported:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if imported_module == "builtins" and "__import__" in imported:
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = (
            node.args[0].value
            if node.args and isinstance(node.args[0], ast.Constant)
            else next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
                ),
                None,
            )
        )
        if not isinstance(target, str):
            continue
        function_path = _attribute_path(node.func)
        dynamic = (
            isinstance(node.func, ast.Name)
            and node.func.id in import_module_aliases | builtin_import_aliases
        ) or (
            function_path is not None
            and len(function_path) == 2
            and function_path[0] in importlib_aliases
            and function_path[1] == "import_module"
        )
        if dynamic and (target == module or target.startswith(f"{module}.")):
            return True
        if dynamic and target.startswith("."):
            package = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "package"
                    and isinstance(keyword.value, ast.Constant)
                ),
                (
                    node.args[1].value
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    else None
                ),
            )
            if package:
                try:
                    if importlib.util.resolve_name(target, package) == module:
                        return True
                except ImportError, ValueError:
                    pass

    for node in ast.walk(tree):
        path = _attribute_path(node)
        if path is None:
            continue
        if len(path) >= 2 and path[-1] in CODEC_EXPORTS | {
            "paper_account_journal_codec"
        }:
            if path[0] in persistence_aliases:
                return True
            if (
                len(path) >= 3
                and path[0] in trading_aliases
                and path[1] == "persistence"
            ):
                return True
    return False


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence.paper_account_journal_codec import "
        "encode_paper_account_opening",
        "import trading.persistence.paper_account_journal_codec as codec",
        "from trading.persistence import encode_paper_account_batch",
        "import trading as root\nroot.persistence.PaperAccountBatchManifest",
        "from trading import persistence as store\n"
        "store.decode_paper_account_settlement",
        "from .persistence import paper_account_journal_codec",
        "from importlib import import_module as load\n"
        "load('trading.persistence.paper_account_journal_codec')",
        "import importlib as loader\n"
        "loader.import_module('trading.persistence.paper_account_journal_codec')",
        "__import__('trading.persistence.paper_account_journal_codec')",
        "from importlib import import_module\n"
        "import_module('.paper_account_journal_codec', "
        "package='trading.persistence')",
    ),
)
def test_account_codec_consumer_detector_catches_supported_forms(source: str) -> None:
    assert _uses_account_codec(source)


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence import apply_migrations",
        "import trading.persistence",
        "from trading.persistence.journal_codec import JournalEncodeError",
        "name = 'trading.persistence.paper_account_journal_codec'",
    ),
)
def test_account_codec_consumer_detector_allows_unrelated_forms(source: str) -> None:
    assert not _uses_account_codec(source)


def test_account_codec_has_only_explicit_non_runtime_consumers() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "persistence" / "paper_account_journal_codec.py"
    facade_path = root / "trading" / "persistence" / "__init__.py"
    repository_path = root / "trading" / "persistence" / "paper_account_journal.py"
    owner_path = root / "trading" / "persistence" / "atomic_paper_account_owner.py"
    opening_plan_composition_path = root / "scripts" / "v2_opening_plan.py"
    allowed_consumers = {
        facade_path,
        repository_path,
        owner_path,
        opening_plan_composition_path,
    }
    ignored_parts = {".git", ".venv", "__pycache__", "build", "dist", "tests"}
    consumers = []

    for source_path in root.rglob("*.py"):
        relative = source_path.relative_to(root)
        if (
            source_path == module_path
            or source_path in allowed_consumers
            or any(part in ignored_parts for part in relative.parts)
        ):
            continue
        if _uses_account_codec(source_path.read_text(encoding="utf-8")):
            consumers.append(relative)

    assert consumers == []


def test_account_codec_imports_only_stdlib_domain_and_journal_errors() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "persistence" / "paper_account_journal_codec.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    allowed_stdlib = {
        "__future__",
        "dataclasses",
        "datetime",
        "decimal",
        "hashlib",
        "hmac",
        "json",
    }
    allowed_persistence = {"trading.persistence.journal_codec"}
    imported_modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported_modules.append(node.module or "")

    assert all(
        module in allowed_stdlib
        or module.startswith("trading.domain.")
        or module in allowed_persistence
        for module in imported_modules
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "__import__"
        for node in ast.walk(tree)
    )


# Frozen after the implementation's canonical-JSON contract is known.
OPENING_GOLDEN_JSON = (
    '{"execution_scope":"paper:test","opening_balances":['
    '{"asset":"BNB","available":"1.500","reserved":"0"},'
    '{"asset":"USDT","available":"10.00","reserved":"0.00"}],'
    '"owner_generation":7,"policy":{"account_key":"paper-main",'
    '"collateral_asset":"USDT","margin_quantum":"0.010"}}'
)
OPENING_GOLDEN_SHA = "9468f628feaccf886ba2d56786529eabc569a12f2f91ddf6a8ddacee38710115"
SETTLEMENT_GOLDEN_JSON = (
    '{"account":{"account_key":"paper-main","collateral_asset":"USDT"},'
    '"account_state_after":"ACTIVE","disposition":"APPLIED",'
    '"instrument":{"base_asset":"BTC",'
    '"kind":"LINEAR_QUOTE_MULTIPLIER_ONE","quote_asset":"USDT",'
    '"symbol":"BTCUSDT","version":1},"position_margin_after":"0.620",'
    '"postings":[{"amount":"-0.1250","asset":"BNB",'
    '"bucket":"AVAILABLE"},{"amount":"-0.620","asset":"USDT",'
    '"bucket":"AVAILABLE"},{"amount":"0.620","asset":"USDT",'
    '"bucket":"RESERVED_MARGIN"}],"settlement_deltas":{'
    '"cash_deltas":[{"amount":"-0.1250","asset":"BNB"}],'
    '"fee_debits":[{"amount":"0.1250","asset":"BNB"}],'
    '"gross_realized_pnl_delta":{"amount":"0","asset":"USDT"}},'
    '"settlement_ref":{"account_version":1,"client_order_id":"order-1",'
    '"event_id":"fill-event-1","position_key":"position-1",'
    '"position_version":2,"trade_id":"trade-1"}}'
)
SETTLEMENT_GOLDEN_SHA = (
    "e36ac5f02fd82a3c437b337ef747579f45ecb5dd018561830c28c680ab356724"
)
BATCH_GOLDEN_JSON = (
    '{"account_key":"paper-main","client_order_id":"order-1",'
    '"execution_scope":"paper:test","fills":['
    '{"account_settlement_payload_sha256":'
    '"0d467c621b5fc4190881bab239ed2199de0c901b29de7c116ff3bf81cc4367e7",'
    '"account_version":1,"client_order_id":"order-1",'
    '"event_id":"fill-event-1","event_payload_sha256":'
    '"9044931e0ed47e97d6361304294e5fda60b950efcd28534209577eac4bef25a9",'
    '"position_key":"position-1","position_version":2,'
    '"trade_id":"trade-1"},{"account_settlement_payload_sha256":'
    '"4116023b0146a512a1ce9b6b8776555e17e27e83e87ec70514c27a49d682b083",'
    '"account_version":2,"client_order_id":"order-1",'
    '"event_id":"fill-event-2","event_payload_sha256":'
    '"aefd8df3bfe9c8ff8555907c3021ebe06373d6209cf197a71ec0b6effca11954",'
    '"position_key":"position-1","position_version":3,'
    '"trade_id":"trade-2"}],"instruction_payload_sha256":'
    '"28abe53ad03224843af578347142b32235845d4669b71d139c347aef41f5f08d",'
    '"owner_generation":7,"position_key":"position-1","submission":{'
    '"event_id":"submission-attempt","event_payload_sha256":'
    '"436a31d35023a7e9ffc3707280ce9c161498ce1ac740ca30c5dc0995365b1ec7",'
    '"observed_at":"2026-08-12T12:34:56.123456+00:00",'
    '"position_version":1}}'
)
BATCH_GOLDEN_SHA = "af498a278a3b37017225518237c911914f9aeba8d13532b969b0fa4b3971595e"
BATCH_V2_GOLDEN_JSON = BATCH_GOLDEN_JSON.replace(
    ',"submission":',
    ',"runtime_generation":42,"submission":',
)
BATCH_V2_GOLDEN_SHA = "f647482b66591e350f9b8155d0b316f65939aa7dddd84ad954c0df18901f4948"
