"""Contract tests for the pure version-1 order and position journal codecs."""

import ast
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from trading.domain.order_lifecycle import (
    CancellationConfirmed,
    CancellationRejected,
    CancellationRequested,
    ConfirmedFill,
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
    TakeProfitProfile,
)
from trading.persistence.journal_codec import (
    EncodedOrderLifecycleEvent,
    EncodedPositionInstruction,
    JournalEncodeError,
    JournalQuarantineError,
    decode_order_lifecycle_event,
    decode_position_instruction,
    encode_order_lifecycle_event,
    encode_position_instruction,
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


class _InvalidOffset(tzinfo):
    def utcoffset(self, value: datetime | None) -> object:
        return "invalid"


def _intent(**overrides: object) -> OrderIntent:
    values = {
        "client_order_id": "order-1",
        "decision_id": "decision-1",
        "symbol": "BTCUSDT",
        "side": OrderSide.BUY,
        "quantity": Decimal("1.2300"),
        "order_type": OrderType.MARKET,
        "reference_price": Decimal("50000.1250"),
        "leverage": 3,
        "created_at": LOCAL_TIME,
    }
    values.update(overrides)
    return OrderIntent(**values)


def _instruction(**overrides: object) -> PositionInstruction:
    values = {
        "position_key": "position-1",
        "effect": PositionEffect.OPEN,
        "order_intent": _intent(),
        "exit_context": PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.002500"),
            stop_loss_fraction=Decimal("0.00500"),
            trailing_stop_fraction=None,
        ),
    }
    values.update(overrides)
    return PositionInstruction(**values)


def _events() -> tuple[object, ...]:
    return (
        SubmissionAcknowledged("order-1", "venue-1", LOCAL_TIME),
        SubmissionAmbiguous("order-1", "transport-timeout", LOCAL_TIME, None),
        SubmissionFailed(
            "order-1",
            SubmissionStatus.NOT_SENT,
            RetrySafety.SAFE,
            "pre-submit-failure",
            LOCAL_TIME,
        ),
        ConfirmedFill(
            "order-1",
            "venue-1",
            "trade-1",
            "BTCUSDT",
            OrderSide.BUY,
            Decimal("0.4000"),
            Decimal("50010.2500"),
            Decimal("-0.000"),
            LOCAL_TIME,
            None,
        ),
        CancellationRequested("order-1", "cancel-1", LOCAL_TIME),
        CancellationConfirmed("order-1", "venue-1", "cancel-1", LOCAL_TIME),
        CancellationRejected(
            "order-1",
            "venue-1",
            "cancel-1",
            "already-filled",
            LOCAL_TIME,
        ),
    )


def _canonical(payload: object) -> tuple[str, str]:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decode_instruction(
    encoded: EncodedPositionInstruction,
    **overrides: object,
) -> PositionInstruction:
    values = {
        "client_order_id": encoded.client_order_id,
        "decision_id": encoded.decision_id,
        "position_key": encoded.position_key,
        "symbol": encoded.symbol,
        "position_effect": encoded.position_effect,
        "instruction_version": encoded.instruction_version,
        "instruction_payload": encoded.instruction_payload,
        "instruction_payload_sha256": encoded.instruction_payload_sha256,
    }
    values.update(overrides)
    return decode_position_instruction(**values)


def _decode_event(
    encoded: EncodedOrderLifecycleEvent,
    **overrides: object,
):
    values = {
        "client_order_id": encoded.client_order_id,
        "event_type": encoded.event_type,
        "event_version": encoded.event_version,
        "event_payload": encoded.event_payload,
        "event_payload_sha256": encoded.event_payload_sha256,
        "trade_id": encoded.trade_id,
        "occurred_at": encoded.occurred_at,
    }
    values.update(overrides)
    return decode_order_lifecycle_event(**values)


INSTRUCTION_GOLDEN_JSON = (
    '{"effect":"OPEN","exit_context":{"stop_loss_fraction":"0.00500",'
    '"take_profit_fraction":"0.002500","take_profit_profile":"RANGING",'
    '"trailing_stop_fraction":null},"order_intent":{"client_order_id":"order-1",'
    '"created_at":"2026-08-12T12:34:56.123456+00:00",'
    '"decision_id":"decision-1","leverage":"3","order_type":"MARKET",'
    '"quantity":"1.2300","reference_price":"50000.1250","side":"BUY",'
    '"symbol":"BTCUSDT"},"position_key":"position-1"}'
)
INSTRUCTION_GOLDEN_SHA = (
    "8998f71a1303b50eb9c70ec31836583155ede22281585a0de88f2f3b9685e69f"
)

EVENT_GOLDENS = {
    "SUBMISSION_ACKNOWLEDGED": (
        '{"client_order_id":"order-1",'
        '"observed_at":"2026-08-12T12:34:56.123456+00:00",'
        '"venue_order_id":"venue-1"}',
        "436a31d35023a7e9ffc3707280ce9c161498ce1ac740ca30c5dc0995365b1ec7",
    ),
    "SUBMISSION_AMBIGUOUS": (
        '{"client_order_id":"order-1",'
        '"observed_at":"2026-08-12T12:34:56.123456+00:00",'
        '"reason":"transport-timeout","venue_order_id":null}',
        "eef9894e171d67bee2568ceff55c906681197b25e86344ac699dc26ecccfc3b5",
    ),
    "SUBMISSION_FAILED": (
        '{"client_order_id":"order-1",'
        '"observed_at":"2026-08-12T12:34:56.123456+00:00",'
        '"reason":"pre-submit-failure","retry_safety":"SAFE",'
        '"status":"NOT_SENT"}',
        "ec7e2d71acc4215a4d97ed6d15d1330906e141104e0c906438a7fc7a1785e494",
    ),
    "CONFIRMED_FILL": (
        '{"client_order_id":"order-1",'
        '"executed_at":"2026-08-12T12:34:56.123456+00:00",'
        '"fee_amount":"-0.000","fee_asset":null,"price":"50010.2500",'
        '"quantity":"0.4000","side":"BUY","symbol":"BTCUSDT",'
        '"trade_id":"trade-1","venue_order_id":"venue-1"}',
        "f2009a391cb9c1c38e124329d40219027ae7076765c8d520b0b445a94303bb28",
    ),
    "CANCELLATION_REQUESTED": (
        '{"cancel_request_id":"cancel-1","client_order_id":"order-1",'
        '"requested_at":"2026-08-12T12:34:56.123456+00:00"}',
        "8465171e4b14ca5516d0b85eb8d4e8f426190f1310bb6e72930023322b5ebc8b",
    ),
    "CANCELLATION_CONFIRMED": (
        '{"cancel_request_id":"cancel-1","client_order_id":"order-1",'
        '"observed_at":"2026-08-12T12:34:56.123456+00:00",'
        '"venue_order_id":"venue-1"}',
        "0e9fcce48750b29a1603f300b323c0aadab54682010b93f2fd1a43fa15dbaa8e",
    ),
    "CANCELLATION_REJECTED": (
        '{"cancel_request_id":"cancel-1","client_order_id":"order-1",'
        '"observed_at":"2026-08-12T12:34:56.123456+00:00",'
        '"reason":"already-filled","venue_order_id":"venue-1"}',
        "ba94befb585e3f02412ef5e1e3be505179ec07125e2766b6234614b60c937c5e",
    ),
}

_CODEC_EXPORTS = {
    "EncodedOrderLifecycleEvent",
    "EncodedPositionInstruction",
    "JournalCodecError",
    "JournalEncodeError",
    "JournalQuarantineError",
    "decode_order_lifecycle_event",
    "decode_position_instruction",
    "encode_order_lifecycle_event",
    "encode_position_instruction",
}


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return (*parent, node.attr) if parent is not None else None
    return None


def _uses_journal_codec(source: str) -> bool:
    """Detect direct, facade, aliased, relative, and literal dynamic imports."""
    tree = ast.parse(source)
    trading_aliases: set[str] = set()
    persistence_aliases: set[str] = set()
    importlib_aliases: set[str] = set()
    import_module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "trading.persistence.journal_codec":
                    return True
                if alias.name == "trading":
                    trading_aliases.add(alias.asname or "trading")
                elif alias.name == "trading.persistence":
                    if alias.asname:
                        persistence_aliases.add(alias.asname)
                    else:
                        trading_aliases.add("trading")
                elif alias.name.startswith("trading.") and alias.asname is None:
                    # A dotted import without ``as`` binds the top-level
                    # package and can reach the persistence facade later.
                    trading_aliases.add("trading")
                elif alias.name == "importlib":
                    importlib_aliases.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported = {alias.name for alias in node.names}
            if module == "trading.persistence.journal_codec" or (
                node.level and module.endswith("persistence.journal_codec")
            ):
                return True
            if module == "trading.persistence" or (
                node.level and module == "persistence"
            ):
                if imported & (_CODEC_EXPORTS | {"journal_codec", "*"}):
                    return True
            if module == "trading":
                persistence_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "persistence"
                )
            if node.level and not module:
                persistence_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "persistence"
                )
            if module == "importlib":
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )

    def dynamic_import_target(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call) or not node.args:
            return None
        argument = node.args[0]
        if not isinstance(argument, ast.Constant) or not isinstance(
            argument.value, str
        ):
            return None
        if isinstance(node.func, ast.Name) and node.func.id in (
            import_module_aliases | {"__import__"}
        ):
            return argument.value
        path = _attribute_path(node.func)
        if (
            path is not None
            and len(path) == 2
            and path[0] in importlib_aliases
            and path[1] == "import_module"
        ):
            return argument.value
        return None

    def imports_trading_facade(node: ast.AST) -> bool:
        target = dynamic_import_target(node)
        return bool(
            target == "trading"
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                and target is not None
                and target.startswith("trading.")
            )
        )

    def assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
        if isinstance(node, ast.Assign):
            return {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
        if isinstance(node.target, ast.Name):
            return {node.target.id}
        return set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and imports_trading_facade(
            node.value
        ):
            trading_aliases.update(assigned_names(node))

    def is_trading_reference(node: ast.AST) -> bool:
        path = _attribute_path(node)
        return bool(
            (path is not None and len(path) == 1 and path[0] in trading_aliases)
            or imports_trading_facade(node)
        )

    def is_dynamic_persistence_reference(node: ast.AST) -> bool:
        if dynamic_import_target(node) == "trading.persistence":
            return True
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "persistence"
            and is_trading_reference(node.value)
        ):
            return True
        return bool(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and is_trading_reference(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "persistence"
        )

    for node in ast.walk(tree):
        target = dynamic_import_target(node)
        if target == "trading.persistence.journal_codec":
            return True
        if isinstance(node, ast.Assign) and is_dynamic_persistence_reference(
            node.value
        ):
            persistence_aliases.update(assigned_names(node))
        elif isinstance(node, ast.AnnAssign) and is_dynamic_persistence_reference(
            node.value
        ):
            persistence_aliases.update(assigned_names(node))

    def is_persistence_reference(node: ast.AST) -> bool:
        path = _attribute_path(node)
        if path is not None:
            if len(path) == 1 and path[0] in persistence_aliases:
                return True
            if (
                len(path) == 2
                and path[0] in trading_aliases
                and path[1] == "persistence"
            ):
                return True
        return is_dynamic_persistence_reference(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in (_CODEC_EXPORTS | {"journal_codec"}) and (
                is_persistence_reference(node.value)
            ):
                return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and is_persistence_reference(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in (_CODEC_EXPORTS | {"journal_codec"})
        ):
            return True
    return False


def test_instruction_golden_vector_and_round_trip() -> None:
    instruction = _instruction()

    encoded = encode_position_instruction(instruction)
    decoded = _decode_instruction(encoded)

    assert encoded.instruction_payload == INSTRUCTION_GOLDEN_JSON
    assert encoded.instruction_payload_sha256 == INSTRUCTION_GOLDEN_SHA
    assert encoded.instruction_version == 1
    assert encoded.position_effect == "OPEN"
    assert decoded == instruction
    assert decoded.order_intent.quantity.as_tuple() == (
        instruction.order_intent.quantity.as_tuple()
    )
    assert decoded.order_intent.created_at == UTC_TIME
    assert json.loads(encoded.instruction_payload)["order_intent"]["leverage"] == "3"


def test_reduce_only_instruction_uses_explicit_null_exit_context() -> None:
    instruction = _instruction(
        effect=PositionEffect.REDUCE_ONLY,
        order_intent=_intent(side=OrderSide.SELL),
        exit_context=None,
    )

    encoded = encode_position_instruction(instruction)

    assert json.loads(encoded.instruction_payload)["exit_context"] is None
    assert _decode_instruction(encoded) == instruction


@pytest.mark.parametrize("event", _events(), ids=lambda event: type(event).__name__)
def test_all_seven_event_golden_vectors_and_round_trip(event: object) -> None:
    encoded = encode_order_lifecycle_event(event)
    decoded = _decode_event(encoded)
    expected_payload, expected_hash = EVENT_GOLDENS[encoded.event_type]

    assert encoded.event_payload == expected_payload
    assert encoded.event_payload_sha256 == expected_hash
    assert encoded.event_version == 1
    assert encoded.occurred_at == UTC_TIME
    assert decoded == event
    if isinstance(event, ConfirmedFill):
        assert decoded.quantity.as_tuple() == event.quantity.as_tuple()
        assert decoded.price.as_tuple() == event.price.as_tuple()
        assert decoded.fee_amount.as_tuple() == event.fee_amount.as_tuple()


@pytest.mark.parametrize(
    "status",
    (SubmissionStatus.NOT_SENT, SubmissionStatus.VENUE_REJECTED),
)
@pytest.mark.parametrize("retry_safety", (RetrySafety.SAFE, RetrySafety.UNSAFE))
def test_all_submission_failed_status_and_retry_combinations_round_trip(
    status: SubmissionStatus,
    retry_safety: RetrySafety,
) -> None:
    event = SubmissionFailed(
        "order-1",
        status,
        retry_safety,
        "proven-non-submission",
        LOCAL_TIME,
    )

    encoded = encode_order_lifecycle_event(event)
    decoded = _decode_event(encoded)

    assert decoded == event
    payload = json.loads(encoded.event_payload)
    assert payload["status"] == status.value
    assert payload["retry_safety"] == retry_safety.value


def test_optional_present_values_round_trip() -> None:
    ambiguous = SubmissionAmbiguous(
        "order-1",
        "timeout",
        LOCAL_TIME,
        "venue-1",
    )
    fill = ConfirmedFill(
        "order-1",
        "venue-1",
        "trade-1",
        "BTCUSDT",
        OrderSide.BUY,
        Decimal("0.4"),
        Decimal("50010"),
        Decimal("0.25"),
        LOCAL_TIME,
        "USDT",
    )
    instruction = _instruction(
        exit_context=replace(
            _instruction().exit_context,
            trailing_stop_fraction=Decimal("0.0100"),
        )
    )

    assert _decode_event(encode_order_lifecycle_event(ambiguous)) == ambiguous
    assert _decode_event(encode_order_lifecycle_event(fill)) == fill
    assert _decode_instruction(encode_position_instruction(instruction)) == instruction


def test_payload_may_arrive_as_jsonb_object_or_noncanonical_json_text() -> None:
    encoded = encode_position_instruction(_instruction())
    payload = json.loads(encoded.instruction_payload)

    assert _decode_instruction(encoded, instruction_payload=payload) == _instruction()
    assert (
        _decode_instruction(
            encoded,
            instruction_payload=json.dumps(payload, indent=2),
        )
        == _instruction()
    )


def test_decimal_tuple_and_extreme_exponents_round_trip_without_ambient_context() -> (
    None
):
    instruction = _instruction(
        order_intent=_intent(
            quantity=Decimal("1E-20000"),
            reference_price=Decimal("1E+200000"),
            leverage=125,
        )
    )
    fill = ConfirmedFill(
        "order-1",
        "venue-1",
        "trade-1",
        "BTCUSDT",
        OrderSide.BUY,
        Decimal("1E-20000"),
        Decimal("1E+200000"),
        Decimal("-0.0000"),
        LOCAL_TIME,
        None,
    )

    with localcontext() as context:
        context.prec = 2
        decoded_instruction = _decode_instruction(
            encode_position_instruction(instruction)
        )
        decoded_fill = _decode_event(encode_order_lifecycle_event(fill))

    assert decoded_instruction.order_intent.quantity.as_tuple() == (
        instruction.order_intent.quantity.as_tuple()
    )
    assert decoded_instruction.order_intent.reference_price.as_tuple() == (
        instruction.order_intent.reference_price.as_tuple()
    )
    assert decoded_fill.quantity.as_tuple() == fill.quantity.as_tuple()
    assert decoded_fill.price.as_tuple() == fill.price.as_tuple()
    assert decoded_fill.fee_amount.as_tuple() == fill.fee_amount.as_tuple()


def test_leverage_over_int_string_conversion_limit_round_trips() -> None:
    leverage = 10**5000 + 7
    leverage_text = "1" + ("0" * 4999) + "7"
    instruction = _instruction(order_intent=_intent(leverage=leverage))

    encoded = encode_position_instruction(instruction)
    decoded = _decode_instruction(encoded)

    assert json.loads(encoded.instruction_payload)["order_intent"]["leverage"] == (
        leverage_text
    )
    assert decoded.order_intent.leverage == leverage


def test_equal_instants_have_one_canonical_utc_representation() -> None:
    utc_event = SubmissionAcknowledged("order-1", "venue-1", UTC_TIME)
    offset_event = SubmissionAcknowledged("order-1", "venue-1", LOCAL_TIME)

    utc_encoded = encode_order_lifecycle_event(utc_event)
    offset_encoded = encode_order_lifecycle_event(offset_event)

    assert utc_encoded.event_payload == offset_encoded.event_payload
    assert utc_encoded.event_payload_sha256 == offset_encoded.event_payload_sha256
    assert offset_encoded.occurred_at.isoformat() == "2026-08-12T12:34:56.123456+00:00"


def test_encoded_records_are_frozen_and_slotted() -> None:
    instruction = encode_position_instruction(_instruction())
    event = encode_order_lifecycle_event(_events()[0])

    assert not hasattr(instruction, "__dict__")
    assert not hasattr(event, "__dict__")
    with pytest.raises(FrozenInstanceError):
        instruction.instruction_version = 2
    with pytest.raises(FrozenInstanceError):
        event.event_version = 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("client_order_id", "other-order"),
        ("decision_id", "other-decision"),
        ("position_key", "other-position"),
        ("symbol", "ETHUSDT"),
        ("position_effect", "REDUCE_ONLY"),
    ),
)
def test_instruction_decoder_rejects_indexed_column_drift(
    field: str,
    value: object,
) -> None:
    encoded = encode_position_instruction(_instruction())

    with pytest.raises(JournalQuarantineError, match="indexed columns conflict"):
        _decode_instruction(encoded, **{field: value})


def test_event_decoder_rejects_indexed_column_drift() -> None:
    encoded = encode_order_lifecycle_event(_events()[3])
    mismatches = (
        {"client_order_id": "other-order"},
        {"trade_id": "other-trade"},
        {"occurred_at": UTC_TIME + timedelta(microseconds=1)},
    )

    for mismatch in mismatches:
        with pytest.raises(JournalQuarantineError, match="indexed columns conflict"):
            _decode_event(encoded, **mismatch)

    acknowledgement = encode_order_lifecycle_event(_events()[0])
    with pytest.raises(JournalQuarantineError, match="indexed columns conflict"):
        _decode_event(acknowledgement, trade_id="unexpected")


def test_event_decoder_quarantines_invalid_indexed_timezone() -> None:
    encoded = encode_order_lifecycle_event(_events()[0])
    invalid_time = datetime(2026, 8, 12, tzinfo=_InvalidOffset())

    with pytest.raises(JournalQuarantineError, match="represented in UTC"):
        _decode_event(encoded, occurred_at=invalid_time)


@pytest.mark.parametrize("version", (True, 0, 2, "1"))
def test_unknown_or_ill_typed_versions_are_quarantined(version: object) -> None:
    instruction = encode_position_instruction(_instruction())
    event = encode_order_lifecycle_event(_events()[0])

    with pytest.raises(JournalQuarantineError, match="version is unknown"):
        _decode_instruction(instruction, instruction_version=version)
    with pytest.raises(JournalQuarantineError, match="version is unknown"):
        _decode_event(event, event_version=version)


@pytest.mark.parametrize("event_type", ("UNKNOWN", None, 1))
def test_unknown_or_ill_typed_event_types_are_quarantined(
    event_type: object,
) -> None:
    encoded = encode_order_lifecycle_event(_events()[0])

    with pytest.raises(JournalQuarantineError, match="event_type"):
        _decode_event(encoded, event_type=event_type)


@pytest.mark.parametrize("checksum", ("0" * 64, "A" * 64, "0" * 63, None))
def test_hash_drift_and_noncanonical_hashes_are_quarantined(checksum: object) -> None:
    encoded = encode_position_instruction(_instruction())

    with pytest.raises(JournalQuarantineError, match="SHA-256"):
        _decode_instruction(encoded, instruction_payload_sha256=checksum)


def test_duplicate_json_keys_are_quarantined_before_hash_comparison() -> None:
    encoded = encode_position_instruction(_instruction())
    duplicate = '{"effect":"OPEN",' + encoded.instruction_payload[1:]

    with pytest.raises(JournalQuarantineError, match="strict JSON"):
        _decode_instruction(encoded, instruction_payload=duplicate)


def test_instruction_decoder_rejects_unknown_missing_and_nested_keys() -> None:
    encoded = encode_position_instruction(_instruction())
    original = json.loads(encoded.instruction_payload)
    invalid_payloads = []

    extra = dict(original)
    extra["unknown"] = True
    invalid_payloads.append(extra)

    missing = dict(original)
    missing.pop("effect")
    invalid_payloads.append(missing)

    nested_extra = json.loads(encoded.instruction_payload)
    nested_extra["order_intent"]["unknown"] = None
    invalid_payloads.append(nested_extra)

    missing_optional = json.loads(encoded.instruction_payload)
    missing_optional["exit_context"].pop("trailing_stop_fraction")
    invalid_payloads.append(missing_optional)

    for payload in invalid_payloads:
        payload_json, payload_hash = _canonical(payload)
        with pytest.raises(JournalQuarantineError, match="payload shape"):
            _decode_instruction(
                encoded,
                instruction_payload=payload_json,
                instruction_payload_sha256=payload_hash,
            )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("order_intent", "quantity"), 1.23),
        (("order_intent", "leverage"), 3),
        (("order_intent", "side"), "HOLD"),
        (("effect",), "FLIP"),
        (("order_intent", "created_at"), "2026-08-12T12:34:56+00:00"),
    ),
)
def test_instruction_decoder_rejects_noncanonical_scalars(
    path: tuple[str, ...],
    value: object,
) -> None:
    encoded = encode_position_instruction(_instruction())
    payload = json.loads(encoded.instruction_payload)
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    payload_json, payload_hash = _canonical(payload)

    with pytest.raises(JournalQuarantineError):
        _decode_instruction(
            encoded,
            instruction_payload=payload_json,
            instruction_payload_sha256=payload_hash,
        )


def test_event_decoder_rejects_payload_shape_enum_and_timestamp_drift() -> None:
    encoded = encode_order_lifecycle_event(_events()[2])
    original = json.loads(encoded.event_payload)
    invalid_payloads = []

    extra = dict(original)
    extra["venue_status"] = "FILLED"
    invalid_payloads.append(extra)

    missing = dict(original)
    missing.pop("reason")
    invalid_payloads.append(missing)

    unknown_status = dict(original)
    unknown_status["status"] = "SUBMITTED"
    invalid_payloads.append(unknown_status)

    noncanonical_time = dict(original)
    noncanonical_time["observed_at"] = "2026-08-12T12:34:56+00:00"
    invalid_payloads.append(noncanonical_time)

    for payload in invalid_payloads:
        payload_json, payload_hash = _canonical(payload)
        with pytest.raises(JournalQuarantineError):
            _decode_event(
                encoded,
                event_payload=payload_json,
                event_payload_sha256=payload_hash,
            )


@pytest.mark.parametrize("event", _events(), ids=lambda event: type(event).__name__)
def test_every_event_decoder_rejects_additional_payload_keys(event: object) -> None:
    encoded = encode_order_lifecycle_event(event)
    payload = json.loads(encoded.event_payload)
    payload["unexpected"] = None
    payload_json, payload_hash = _canonical(payload)

    with pytest.raises(JournalQuarantineError, match="payload shape"):
        _decode_event(
            encoded,
            event_payload=payload_json,
            event_payload_sha256=payload_hash,
        )


def test_decoder_rejects_noncanonical_decimal_text_with_valid_hash() -> None:
    encoded = encode_order_lifecycle_event(_events()[3])
    payload = json.loads(encoded.event_payload)
    payload["quantity"] = "00.4000"
    payload_json, payload_hash = _canonical(payload)

    with pytest.raises(JournalQuarantineError, match="canonical finite Decimal"):
        _decode_event(
            encoded,
            event_payload=payload_json,
            event_payload_sha256=payload_hash,
        )


def test_storage_length_boundaries_accept_exact_limits() -> None:
    instruction = _instruction(
        position_key="p" * 255,
        order_intent=_intent(
            client_order_id="c" * 255,
            decision_id="d" * 255,
            symbol="s" * 64,
        ),
    )
    fill = ConfirmedFill(
        "c" * 255,
        "v" * 255,
        "t" * 255,
        "s" * 64,
        OrderSide.BUY,
        Decimal("1"),
        Decimal("1"),
        Decimal("0"),
        LOCAL_TIME,
        None,
    )

    assert _decode_instruction(encode_position_instruction(instruction)) == instruction
    assert _decode_event(encode_order_lifecycle_event(fill)) == fill


@pytest.mark.parametrize(
    "instruction",
    (
        _instruction(position_key="p" * 256),
        _instruction(order_intent=_intent(client_order_id="c" * 256)),
        _instruction(order_intent=_intent(decision_id="d" * 256)),
        _instruction(order_intent=_intent(symbol="s" * 65)),
    ),
)
def test_instruction_encoder_rejects_values_over_index_limits(
    instruction: PositionInstruction,
) -> None:
    with pytest.raises(JournalEncodeError, match="storage limit"):
        encode_position_instruction(instruction)


@pytest.mark.parametrize(
    "event",
    (
        SubmissionAcknowledged("c" * 256, "venue-1", LOCAL_TIME),
        SubmissionAcknowledged("order-1", "v" * 256, LOCAL_TIME),
        ConfirmedFill(
            "order-1",
            "venue-1",
            "t" * 256,
            "BTCUSDT",
            OrderSide.BUY,
            Decimal("1"),
            Decimal("1"),
            Decimal("0"),
            LOCAL_TIME,
            None,
        ),
        ConfirmedFill(
            "order-1",
            "venue-1",
            "trade-1",
            "s" * 65,
            OrderSide.BUY,
            Decimal("1"),
            Decimal("1"),
            Decimal("0"),
            LOCAL_TIME,
            None,
        ),
    ),
)
def test_event_encoder_rejects_values_over_index_limits(event: object) -> None:
    with pytest.raises(JournalEncodeError, match="storage limit"):
        encode_order_lifecycle_event(event)


@pytest.mark.parametrize("bad_text", ("left\x00right", "left\ud800right"))
def test_encoder_rejects_postgres_unrepresentable_unicode(bad_text: str) -> None:
    instruction = _instruction(order_intent=_intent(client_order_id=bad_text))
    event = SubmissionAmbiguous("order-1", bad_text, LOCAL_TIME, None)

    with pytest.raises(JournalEncodeError, match="not representable"):
        encode_position_instruction(instruction)
    with pytest.raises(JournalEncodeError, match="not representable"):
        encode_order_lifecycle_event(event)


@pytest.mark.parametrize("bad_text", ("left\x00right", "left\ud800right"))
def test_decoder_quarantines_postgres_unrepresentable_unicode(bad_text: str) -> None:
    encoded = encode_order_lifecycle_event(_events()[0])
    payload = json.loads(encoded.event_payload)
    payload["venue_order_id"] = bad_text
    payload_json, payload_hash = _canonical(payload)

    with pytest.raises(JournalQuarantineError, match="not representable"):
        _decode_event(
            encoded,
            event_payload=payload_json,
            event_payload_sha256=payload_hash,
        )


def test_invalid_json_constants_and_non_objects_are_quarantined() -> None:
    encoded = encode_position_instruction(_instruction())

    for payload in ("NaN", "[]", "null"):
        with pytest.raises(JournalQuarantineError):
            _decode_instruction(encoded, instruction_payload=payload)


def test_excessive_json_nesting_is_quarantined_for_text_and_jsonb_objects() -> None:
    encoded = encode_position_instruction(_instruction())
    depth = 150_000
    nested_text = ("[" * depth) + ("]" * depth)

    with pytest.raises(JournalQuarantineError, match="strict JSON"):
        _decode_instruction(encoded, instruction_payload=nested_text)

    nested_object: object = []
    for _ in range(depth):
        nested_object = [nested_object]
    with pytest.raises(JournalQuarantineError, match="canonical JSON"):
        _decode_instruction(
            encoded,
            instruction_payload={"effect": nested_object},
        )


def test_unsupported_encode_values_raise_the_typed_error() -> None:
    with pytest.raises(JournalEncodeError):
        encode_position_instruction(object())
    with pytest.raises(JournalEncodeError, match="OrderLifecycleEvent"):
        encode_order_lifecycle_event(object())


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence.journal_codec import encode_position_instruction",
        "import trading.persistence.journal_codec as codec",
        "from trading.persistence import encode_position_instruction",
        "from trading import persistence as store\nstore.decode_position_instruction",
        "import trading as t\nt.persistence.JournalQuarantineError",
        (
            "import trading.domain.orders\n"
            "trading.persistence.encode_position_instruction"
        ),
        "from ..persistence import journal_codec",
        (
            "from importlib import import_module as load\n"
            "load('trading.persistence.journal_codec')"
        ),
        (
            "import importlib as loader\n"
            "loader.import_module('trading.persistence.journal_codec')"
        ),
        (
            "from importlib import import_module as load\n"
            "store = load('trading.persistence')\n"
            "store.encode_order_lifecycle_event"
        ),
        (
            "import trading.persistence\n"
            "getattr(trading.persistence, 'decode_order_lifecycle_event')"
        ),
        (
            "from importlib import import_module as load\n"
            "load('trading').persistence.journal_codec"
        ),
        (
            "import importlib as i\n"
            "i.import_module('trading').persistence.encode_position_instruction"
        ),
        (
            "getattr(__import__('trading'), 'persistence')"
            ".decode_position_instruction"
        ),
        ("store = __import__('trading').persistence\n" "store.journal_codec"),
        (
            "from importlib import import_module as load\n"
            "root = load('trading')\n"
            "root.persistence.journal_codec"
        ),
        (
            "root = __import__('trading')\n"
            "getattr(root, 'persistence').decode_position_instruction"
        ),
        (
            "from importlib import import_module as load\n"
            "root = load('trading')\n"
            "store = root.persistence\n"
            "store.encode_order_lifecycle_event"
        ),
        (
            "root = __import__('trading.domain.orders')\n"
            "root.persistence.journal_codec"
        ),
        "__import__('trading.persistence.journal_codec')",
    ),
)
def test_codec_consumer_detector_catches_supported_import_forms(source: str) -> None:
    assert _uses_journal_codec(source)


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence import apply_migrations",
        "import trading.persistence",
        (
            "from importlib import import_module as load\n"
            "load('trading.persistence.migration_runner')"
        ),
        "name = 'trading.persistence.journal_codec'",
    ),
)
def test_codec_consumer_detector_allows_unrelated_persistence_use(source: str) -> None:
    assert not _uses_journal_codec(source)


def test_codec_has_no_production_consumer_outside_persistence() -> None:
    root = Path(__file__).parents[1]
    ignored_parts = {
        ".git",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "tests",
    }
    consumers = []

    for source_path in root.rglob("*.py"):
        relative = source_path.relative_to(root)
        if any(part in ignored_parts for part in relative.parts) or relative.parts[
            :2
        ] == ("trading", "persistence"):
            continue
        if _uses_journal_codec(source_path.read_text(encoding="utf-8")):
            consumers.append(relative)

    assert consumers == []


def test_codec_imports_only_stdlib_and_domain_contracts() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "persistence" / "journal_codec.py"
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
    imported_modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported_modules.append(node.module or "")

    assert all(
        module in allowed_stdlib or module.startswith("trading.domain.")
        for module in imported_modules
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "__import__"
        for node in ast.walk(tree)
    )
