"""Pure, strict codecs for the version-1 order and position journal payloads."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from trading.domain.order_lifecycle import (
    CancellationConfirmed,
    CancellationRejected,
    CancellationRequested,
    ConfirmedFill,
    OrderLifecycleEvent,
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

_PAYLOAD_VERSION = 1
_CLIENT_ORDER_ID_MAX_LENGTH = 255
_DECISION_ID_MAX_LENGTH = 255
_POSITION_KEY_MAX_LENGTH = 255
_SYMBOL_MAX_LENGTH = 64
_VENUE_ORDER_ID_MAX_LENGTH = 255
_TRADE_ID_MAX_LENGTH = 255

_EVENT_TYPES = {
    "SUBMISSION_ACKNOWLEDGED",
    "SUBMISSION_AMBIGUOUS",
    "SUBMISSION_FAILED",
    "CONFIRMED_FILL",
    "CANCELLATION_REQUESTED",
    "CANCELLATION_CONFIRMED",
    "CANCELLATION_REJECTED",
}


class JournalCodecError(ValueError):
    """Base class for a journal value which cannot cross the codec boundary."""


class JournalEncodeError(JournalCodecError):
    """Raised before persistence when a domain value is not representable."""


class JournalQuarantineError(JournalCodecError):
    """Raised when persisted journal data is unknown, corrupt, or inconsistent."""


@dataclass(frozen=True, slots=True)
class EncodedPositionInstruction:
    """Canonical payload and derived columns for one position instruction."""

    client_order_id: str
    decision_id: str
    position_key: str
    symbol: str
    position_effect: str
    instruction_version: int
    instruction_payload: str
    instruction_payload_sha256: str


@dataclass(frozen=True, slots=True)
class EncodedOrderLifecycleEvent:
    """Canonical payload and derived columns for one order lifecycle event."""

    client_order_id: str
    event_type: str
    event_version: int
    event_payload: str
    event_payload_sha256: str
    trade_id: str | None
    occurred_at: datetime


class _DuplicateJsonKey(ValueError):
    pass


def _checked_text(
    value: object,
    field: str,
    *,
    error_type: type[JournalCodecError],
    max_length: int | None = None,
) -> str:
    if type(value) is not str:
        raise error_type(f"{field} must be text")
    if not value or value != value.strip():
        raise error_type(f"{field} must be non-empty and trimmed")
    if max_length is not None and len(value) > max_length:
        raise error_type(f"{field} exceeds its storage limit")
    if "\x00" in value or any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        raise error_type(f"{field} is not representable in PostgreSQL JSONB")
    return value


def _optional_text(
    value: object,
    field: str,
    *,
    error_type: type[JournalCodecError],
    max_length: int | None = None,
) -> str | None:
    if value is None:
        return None
    return _checked_text(
        value,
        field,
        error_type=error_type,
        max_length=max_length,
    )


def _decimal_text(value: object, field: str) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise JournalEncodeError(f"{field} must be a finite Decimal")
    encoded = str(value)
    _checked_text(encoded, field, error_type=JournalEncodeError)
    return encoded


def _decode_decimal(value: object, field: str) -> Decimal:
    encoded = _checked_text(value, field, error_type=JournalQuarantineError)
    try:
        decoded = Decimal(encoded)
    except InvalidOperation as exc:
        raise JournalQuarantineError(f"{field} is not a Decimal") from exc
    if not decoded.is_finite() or str(decoded) != encoded:
        raise JournalQuarantineError(f"{field} is not a canonical finite Decimal")
    return decoded


def _positive_integer_text(value: object, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise JournalEncodeError(f"{field} must be a positive integer")
    encoded = str(Decimal(value))
    _checked_text(encoded, field, error_type=JournalEncodeError)
    return encoded


def _decode_positive_integer(value: object, field: str) -> int:
    encoded = _checked_text(value, field, error_type=JournalQuarantineError)
    if encoded[0] == "0" or any(
        character < "0" or character > "9" for character in encoded
    ):
        raise JournalQuarantineError(f"{field} is not a canonical positive integer")
    try:
        decoded = int(Decimal(encoded))
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise JournalQuarantineError(f"{field} is not a positive integer") from exc
    if decoded < 1:
        raise JournalQuarantineError(f"{field} is not a canonical positive integer")
    return decoded


def _utc_datetime(
    value: object, field: str, error_type: type[JournalCodecError]
) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise error_type(f"{field} must be a timezone-aware datetime")
    try:
        if value.utcoffset() is None:
            raise error_type(f"{field} must be a timezone-aware datetime")
        return value.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as exc:
        raise error_type(f"{field} cannot be represented in UTC") from exc


def _datetime_text(value: object, field: str) -> tuple[str, datetime]:
    normalized = _utc_datetime(value, field, JournalEncodeError)
    return normalized.isoformat(timespec="microseconds"), normalized


def _decode_datetime(value: object, field: str) -> datetime:
    encoded = _checked_text(value, field, error_type=JournalQuarantineError)
    try:
        decoded = datetime.fromisoformat(encoded)
    except ValueError as exc:
        raise JournalQuarantineError(f"{field} is not an ISO datetime") from exc
    normalized = _utc_datetime(decoded, field, JournalQuarantineError)
    if normalized.isoformat(timespec="microseconds") != encoded:
        raise JournalQuarantineError(f"{field} is not a canonical UTC datetime")
    return normalized


def _canonical_json(payload: object, error_type: type[JournalCodecError]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise error_type("payload is not canonical JSON data") from exc


def _payload_sha256(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"unsupported JSON constant: {value}")


def _payload_object(value: object, field: str) -> dict[str, object]:
    if type(value) is str:
        try:
            decoded = json.loads(
                value,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (
            _DuplicateJsonKey,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ) as exc:
            raise JournalQuarantineError(f"{field} is not valid strict JSON") from exc
    elif type(value) is dict:
        decoded = value
    else:
        raise JournalQuarantineError(f"{field} must be a JSON object")
    if type(decoded) is not dict:
        raise JournalQuarantineError(f"{field} must be a JSON object")
    return decoded


def _verified_payload(value: object, checksum: object, field: str) -> dict[str, object]:
    if (
        type(checksum) is not str
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise JournalQuarantineError(f"{field} SHA-256 is invalid")
    payload = _payload_object(value, field)
    canonical = _canonical_json(payload, JournalQuarantineError)
    actual = _payload_sha256(canonical)
    if not hmac.compare_digest(actual, checksum):
        raise JournalQuarantineError(f"{field} SHA-256 does not match")
    return payload


def _exact_keys(value: object, expected: set[str], field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise JournalQuarantineError(f"{field} has an unknown payload shape")
    return value


def _decode_enum(enum_type: type, value: object, field: str):
    encoded = _checked_text(value, field, error_type=JournalQuarantineError)
    try:
        return enum_type(encoded)
    except ValueError as exc:
        raise JournalQuarantineError(f"{field} has an unknown value") from exc


def _instruction_payload(value: PositionInstruction) -> dict[str, object]:
    intent = value.order_intent
    client_order_id = _checked_text(
        intent.client_order_id,
        "client_order_id",
        error_type=JournalEncodeError,
        max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
    )
    decision_id = _checked_text(
        intent.decision_id,
        "decision_id",
        error_type=JournalEncodeError,
        max_length=_DECISION_ID_MAX_LENGTH,
    )
    symbol = _checked_text(
        intent.symbol,
        "symbol",
        error_type=JournalEncodeError,
        max_length=_SYMBOL_MAX_LENGTH,
    )
    position_key = _checked_text(
        value.position_key,
        "position_key",
        error_type=JournalEncodeError,
        max_length=_POSITION_KEY_MAX_LENGTH,
    )
    created_at, _ = _datetime_text(intent.created_at, "created_at")

    exit_context = None
    if value.exit_context is not None:
        context = value.exit_context
        exit_context = {
            "take_profit_profile": context.take_profit_profile.value,
            "take_profit_fraction": _decimal_text(
                context.take_profit_fraction,
                "take_profit_fraction",
            ),
            "stop_loss_fraction": _decimal_text(
                context.stop_loss_fraction,
                "stop_loss_fraction",
            ),
            "trailing_stop_fraction": (
                _decimal_text(
                    context.trailing_stop_fraction,
                    "trailing_stop_fraction",
                )
                if context.trailing_stop_fraction is not None
                else None
            ),
        }

    return {
        "position_key": position_key,
        "effect": value.effect.value,
        "order_intent": {
            "client_order_id": client_order_id,
            "decision_id": decision_id,
            "symbol": symbol,
            "side": intent.side.value,
            "quantity": _decimal_text(intent.quantity, "quantity"),
            "order_type": intent.order_type.value,
            "reference_price": _decimal_text(
                intent.reference_price,
                "reference_price",
            ),
            "leverage": _positive_integer_text(intent.leverage, "leverage"),
            "created_at": created_at,
        },
        "exit_context": exit_context,
    }


def encode_position_instruction(
    value: PositionInstruction,
    /,
) -> EncodedPositionInstruction:
    """Encode one validated instruction without consulting I/O or ambient context."""
    if type(value) is not PositionInstruction:
        raise JournalEncodeError("value must be a PositionInstruction")
    payload = _instruction_payload(value)
    payload_json = _canonical_json(payload, JournalEncodeError)
    intent = value.order_intent
    return EncodedPositionInstruction(
        client_order_id=intent.client_order_id,
        decision_id=intent.decision_id,
        position_key=value.position_key,
        symbol=intent.symbol,
        position_effect=value.effect.value,
        instruction_version=_PAYLOAD_VERSION,
        instruction_payload=payload_json,
        instruction_payload_sha256=_payload_sha256(payload_json),
    )


def _decode_order_intent(value: object) -> OrderIntent:
    payload = _exact_keys(
        value,
        {
            "client_order_id",
            "decision_id",
            "symbol",
            "side",
            "quantity",
            "order_type",
            "reference_price",
            "leverage",
            "created_at",
        },
        "order_intent",
    )
    return OrderIntent(
        client_order_id=_checked_text(
            payload["client_order_id"],
            "client_order_id",
            error_type=JournalQuarantineError,
            max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
        ),
        decision_id=_checked_text(
            payload["decision_id"],
            "decision_id",
            error_type=JournalQuarantineError,
            max_length=_DECISION_ID_MAX_LENGTH,
        ),
        symbol=_checked_text(
            payload["symbol"],
            "symbol",
            error_type=JournalQuarantineError,
            max_length=_SYMBOL_MAX_LENGTH,
        ),
        side=_decode_enum(OrderSide, payload["side"], "side"),
        quantity=_decode_decimal(payload["quantity"], "quantity"),
        order_type=_decode_enum(OrderType, payload["order_type"], "order_type"),
        reference_price=_decode_decimal(
            payload["reference_price"],
            "reference_price",
        ),
        leverage=_decode_positive_integer(payload["leverage"], "leverage"),
        created_at=_decode_datetime(payload["created_at"], "created_at"),
    )


def _decode_exit_context(value: object) -> PositionExitContext | None:
    if value is None:
        return None
    payload = _exact_keys(
        value,
        {
            "take_profit_profile",
            "take_profit_fraction",
            "stop_loss_fraction",
            "trailing_stop_fraction",
        },
        "exit_context",
    )
    trailing = payload["trailing_stop_fraction"]
    return PositionExitContext(
        take_profit_profile=_decode_enum(
            TakeProfitProfile,
            payload["take_profit_profile"],
            "take_profit_profile",
        ),
        take_profit_fraction=_decode_decimal(
            payload["take_profit_fraction"],
            "take_profit_fraction",
        ),
        stop_loss_fraction=_decode_decimal(
            payload["stop_loss_fraction"],
            "stop_loss_fraction",
        ),
        trailing_stop_fraction=(
            _decode_decimal(trailing, "trailing_stop_fraction")
            if trailing is not None
            else None
        ),
    )


def decode_position_instruction(
    *,
    client_order_id: object,
    decision_id: object,
    position_key: object,
    symbol: object,
    position_effect: object,
    instruction_version: object,
    instruction_payload: object,
    instruction_payload_sha256: object,
) -> PositionInstruction:
    """Decode and cross-check one untrusted persisted instruction envelope."""
    if type(instruction_version) is not int or instruction_version != _PAYLOAD_VERSION:
        raise JournalQuarantineError("instruction_version is unknown")
    payload = _verified_payload(
        instruction_payload,
        instruction_payload_sha256,
        "instruction_payload",
    )
    payload = _exact_keys(
        payload,
        {"position_key", "effect", "order_intent", "exit_context"},
        "instruction_payload",
    )
    try:
        instruction = PositionInstruction(
            position_key=_checked_text(
                payload["position_key"],
                "position_key",
                error_type=JournalQuarantineError,
                max_length=_POSITION_KEY_MAX_LENGTH,
            ),
            effect=_decode_enum(PositionEffect, payload["effect"], "effect"),
            order_intent=_decode_order_intent(payload["order_intent"]),
            exit_context=_decode_exit_context(payload["exit_context"]),
        )
    except JournalQuarantineError:
        raise
    except (TypeError, ValueError) as exc:
        raise JournalQuarantineError("instruction payload violates the domain") from exc

    indexed_client_order_id = _checked_text(
        client_order_id,
        "indexed client_order_id",
        error_type=JournalQuarantineError,
        max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
    )
    indexed_decision_id = _checked_text(
        decision_id,
        "indexed decision_id",
        error_type=JournalQuarantineError,
        max_length=_DECISION_ID_MAX_LENGTH,
    )
    indexed_position_key = _checked_text(
        position_key,
        "indexed position_key",
        error_type=JournalQuarantineError,
        max_length=_POSITION_KEY_MAX_LENGTH,
    )
    indexed_symbol = _checked_text(
        symbol,
        "indexed symbol",
        error_type=JournalQuarantineError,
        max_length=_SYMBOL_MAX_LENGTH,
    )
    indexed_effect = _checked_text(
        position_effect,
        "indexed position_effect",
        error_type=JournalQuarantineError,
    )
    expected = (
        instruction.order_intent.client_order_id,
        instruction.order_intent.decision_id,
        instruction.position_key,
        instruction.order_intent.symbol,
        instruction.effect.value,
    )
    actual = (
        indexed_client_order_id,
        indexed_decision_id,
        indexed_position_key,
        indexed_symbol,
        indexed_effect,
    )
    if actual != expected:
        raise JournalQuarantineError(
            "instruction indexed columns conflict with payload"
        )
    return instruction


def _encoded_event(
    value: OrderLifecycleEvent,
) -> tuple[str, dict[str, object], str | None, datetime]:
    if type(value) not in (
        SubmissionAcknowledged,
        SubmissionAmbiguous,
        SubmissionFailed,
        ConfirmedFill,
        CancellationRequested,
        CancellationConfirmed,
        CancellationRejected,
    ):
        raise JournalEncodeError("value must be an OrderLifecycleEvent")
    client_order_id = _checked_text(
        value.client_order_id,
        "client_order_id",
        error_type=JournalEncodeError,
        max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
    )
    if type(value) is SubmissionAcknowledged:
        occurred_text, occurred_at = _datetime_text(value.observed_at, "observed_at")
        return (
            "SUBMISSION_ACKNOWLEDGED",
            {
                "client_order_id": client_order_id,
                "venue_order_id": _checked_text(
                    value.venue_order_id,
                    "venue_order_id",
                    error_type=JournalEncodeError,
                    max_length=_VENUE_ORDER_ID_MAX_LENGTH,
                ),
                "observed_at": occurred_text,
            },
            None,
            occurred_at,
        )
    if type(value) is SubmissionAmbiguous:
        occurred_text, occurred_at = _datetime_text(value.observed_at, "observed_at")
        return (
            "SUBMISSION_AMBIGUOUS",
            {
                "client_order_id": client_order_id,
                "reason": _checked_text(
                    value.reason,
                    "reason",
                    error_type=JournalEncodeError,
                ),
                "observed_at": occurred_text,
                "venue_order_id": _optional_text(
                    value.venue_order_id,
                    "venue_order_id",
                    error_type=JournalEncodeError,
                    max_length=_VENUE_ORDER_ID_MAX_LENGTH,
                ),
            },
            None,
            occurred_at,
        )
    if type(value) is SubmissionFailed:
        occurred_text, occurred_at = _datetime_text(value.observed_at, "observed_at")
        return (
            "SUBMISSION_FAILED",
            {
                "client_order_id": client_order_id,
                "status": value.status.value,
                "retry_safety": value.retry_safety.value,
                "reason": _checked_text(
                    value.reason,
                    "reason",
                    error_type=JournalEncodeError,
                ),
                "observed_at": occurred_text,
            },
            None,
            occurred_at,
        )
    if type(value) is ConfirmedFill:
        occurred_text, occurred_at = _datetime_text(value.executed_at, "executed_at")
        trade_id = _checked_text(
            value.trade_id,
            "trade_id",
            error_type=JournalEncodeError,
            max_length=_TRADE_ID_MAX_LENGTH,
        )
        return (
            "CONFIRMED_FILL",
            {
                "client_order_id": client_order_id,
                "venue_order_id": _checked_text(
                    value.venue_order_id,
                    "venue_order_id",
                    error_type=JournalEncodeError,
                    max_length=_VENUE_ORDER_ID_MAX_LENGTH,
                ),
                "trade_id": trade_id,
                "symbol": _checked_text(
                    value.symbol,
                    "symbol",
                    error_type=JournalEncodeError,
                    max_length=_SYMBOL_MAX_LENGTH,
                ),
                "side": value.side.value,
                "quantity": _decimal_text(value.quantity, "quantity"),
                "price": _decimal_text(value.price, "price"),
                "fee_amount": _decimal_text(value.fee_amount, "fee_amount"),
                "executed_at": occurred_text,
                "fee_asset": _optional_text(
                    value.fee_asset,
                    "fee_asset",
                    error_type=JournalEncodeError,
                ),
            },
            trade_id,
            occurred_at,
        )
    if type(value) is CancellationRequested:
        occurred_text, occurred_at = _datetime_text(value.requested_at, "requested_at")
        return (
            "CANCELLATION_REQUESTED",
            {
                "client_order_id": client_order_id,
                "cancel_request_id": _checked_text(
                    value.cancel_request_id,
                    "cancel_request_id",
                    error_type=JournalEncodeError,
                ),
                "requested_at": occurred_text,
            },
            None,
            occurred_at,
        )
    if type(value) is CancellationConfirmed:
        occurred_text, occurred_at = _datetime_text(value.observed_at, "observed_at")
        return (
            "CANCELLATION_CONFIRMED",
            {
                "client_order_id": client_order_id,
                "venue_order_id": _checked_text(
                    value.venue_order_id,
                    "venue_order_id",
                    error_type=JournalEncodeError,
                    max_length=_VENUE_ORDER_ID_MAX_LENGTH,
                ),
                "cancel_request_id": _checked_text(
                    value.cancel_request_id,
                    "cancel_request_id",
                    error_type=JournalEncodeError,
                ),
                "observed_at": occurred_text,
            },
            None,
            occurred_at,
        )
    if type(value) is CancellationRejected:
        occurred_text, occurred_at = _datetime_text(value.observed_at, "observed_at")
        return (
            "CANCELLATION_REJECTED",
            {
                "client_order_id": client_order_id,
                "venue_order_id": _checked_text(
                    value.venue_order_id,
                    "venue_order_id",
                    error_type=JournalEncodeError,
                    max_length=_VENUE_ORDER_ID_MAX_LENGTH,
                ),
                "cancel_request_id": _checked_text(
                    value.cancel_request_id,
                    "cancel_request_id",
                    error_type=JournalEncodeError,
                ),
                "reason": _checked_text(
                    value.reason,
                    "reason",
                    error_type=JournalEncodeError,
                ),
                "observed_at": occurred_text,
            },
            None,
            occurred_at,
        )
    raise JournalEncodeError("value must be an OrderLifecycleEvent")


def encode_order_lifecycle_event(
    value: OrderLifecycleEvent,
    /,
) -> EncodedOrderLifecycleEvent:
    """Encode one lifecycle fact and every column derivable from that fact."""
    event_type, payload, trade_id, occurred_at = _encoded_event(value)
    payload_json = _canonical_json(payload, JournalEncodeError)
    return EncodedOrderLifecycleEvent(
        client_order_id=value.client_order_id,
        event_type=event_type,
        event_version=_PAYLOAD_VERSION,
        event_payload=payload_json,
        event_payload_sha256=_payload_sha256(payload_json),
        trade_id=trade_id,
        occurred_at=occurred_at,
    )


def _decode_event_payload(
    event_type: str, payload: dict[str, object]
) -> OrderLifecycleEvent:
    if event_type == "SUBMISSION_ACKNOWLEDGED":
        value = _exact_keys(
            payload,
            {"client_order_id", "venue_order_id", "observed_at"},
            event_type,
        )
        return SubmissionAcknowledged(
            client_order_id=_checked_text(
                value["client_order_id"],
                "client_order_id",
                error_type=JournalQuarantineError,
                max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
            ),
            venue_order_id=_checked_text(
                value["venue_order_id"],
                "venue_order_id",
                error_type=JournalQuarantineError,
                max_length=_VENUE_ORDER_ID_MAX_LENGTH,
            ),
            observed_at=_decode_datetime(value["observed_at"], "observed_at"),
        )
    if event_type == "SUBMISSION_AMBIGUOUS":
        value = _exact_keys(
            payload,
            {"client_order_id", "reason", "observed_at", "venue_order_id"},
            event_type,
        )
        return SubmissionAmbiguous(
            client_order_id=_checked_text(
                value["client_order_id"],
                "client_order_id",
                error_type=JournalQuarantineError,
                max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
            ),
            reason=_checked_text(
                value["reason"],
                "reason",
                error_type=JournalQuarantineError,
            ),
            observed_at=_decode_datetime(value["observed_at"], "observed_at"),
            venue_order_id=_optional_text(
                value["venue_order_id"],
                "venue_order_id",
                error_type=JournalQuarantineError,
                max_length=_VENUE_ORDER_ID_MAX_LENGTH,
            ),
        )
    if event_type == "SUBMISSION_FAILED":
        value = _exact_keys(
            payload,
            {"client_order_id", "status", "retry_safety", "reason", "observed_at"},
            event_type,
        )
        return SubmissionFailed(
            client_order_id=_checked_text(
                value["client_order_id"],
                "client_order_id",
                error_type=JournalQuarantineError,
                max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
            ),
            status=_decode_enum(SubmissionStatus, value["status"], "status"),
            retry_safety=_decode_enum(
                RetrySafety,
                value["retry_safety"],
                "retry_safety",
            ),
            reason=_checked_text(
                value["reason"],
                "reason",
                error_type=JournalQuarantineError,
            ),
            observed_at=_decode_datetime(value["observed_at"], "observed_at"),
        )
    if event_type == "CONFIRMED_FILL":
        value = _exact_keys(
            payload,
            {
                "client_order_id",
                "venue_order_id",
                "trade_id",
                "symbol",
                "side",
                "quantity",
                "price",
                "fee_amount",
                "executed_at",
                "fee_asset",
            },
            event_type,
        )
        return ConfirmedFill(
            client_order_id=_checked_text(
                value["client_order_id"],
                "client_order_id",
                error_type=JournalQuarantineError,
                max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
            ),
            venue_order_id=_checked_text(
                value["venue_order_id"],
                "venue_order_id",
                error_type=JournalQuarantineError,
                max_length=_VENUE_ORDER_ID_MAX_LENGTH,
            ),
            trade_id=_checked_text(
                value["trade_id"],
                "trade_id",
                error_type=JournalQuarantineError,
                max_length=_TRADE_ID_MAX_LENGTH,
            ),
            symbol=_checked_text(
                value["symbol"],
                "symbol",
                error_type=JournalQuarantineError,
                max_length=_SYMBOL_MAX_LENGTH,
            ),
            side=_decode_enum(OrderSide, value["side"], "side"),
            quantity=_decode_decimal(value["quantity"], "quantity"),
            price=_decode_decimal(value["price"], "price"),
            fee_amount=_decode_decimal(value["fee_amount"], "fee_amount"),
            executed_at=_decode_datetime(value["executed_at"], "executed_at"),
            fee_asset=_optional_text(
                value["fee_asset"],
                "fee_asset",
                error_type=JournalQuarantineError,
            ),
        )
    if event_type == "CANCELLATION_REQUESTED":
        value = _exact_keys(
            payload,
            {"client_order_id", "cancel_request_id", "requested_at"},
            event_type,
        )
        return CancellationRequested(
            client_order_id=_checked_text(
                value["client_order_id"],
                "client_order_id",
                error_type=JournalQuarantineError,
                max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
            ),
            cancel_request_id=_checked_text(
                value["cancel_request_id"],
                "cancel_request_id",
                error_type=JournalQuarantineError,
            ),
            requested_at=_decode_datetime(value["requested_at"], "requested_at"),
        )
    if event_type == "CANCELLATION_CONFIRMED":
        value = _exact_keys(
            payload,
            {"client_order_id", "venue_order_id", "cancel_request_id", "observed_at"},
            event_type,
        )
        return CancellationConfirmed(
            client_order_id=_checked_text(
                value["client_order_id"],
                "client_order_id",
                error_type=JournalQuarantineError,
                max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
            ),
            venue_order_id=_checked_text(
                value["venue_order_id"],
                "venue_order_id",
                error_type=JournalQuarantineError,
                max_length=_VENUE_ORDER_ID_MAX_LENGTH,
            ),
            cancel_request_id=_checked_text(
                value["cancel_request_id"],
                "cancel_request_id",
                error_type=JournalQuarantineError,
            ),
            observed_at=_decode_datetime(value["observed_at"], "observed_at"),
        )
    if event_type == "CANCELLATION_REJECTED":
        value = _exact_keys(
            payload,
            {
                "client_order_id",
                "venue_order_id",
                "cancel_request_id",
                "reason",
                "observed_at",
            },
            event_type,
        )
        return CancellationRejected(
            client_order_id=_checked_text(
                value["client_order_id"],
                "client_order_id",
                error_type=JournalQuarantineError,
                max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
            ),
            venue_order_id=_checked_text(
                value["venue_order_id"],
                "venue_order_id",
                error_type=JournalQuarantineError,
                max_length=_VENUE_ORDER_ID_MAX_LENGTH,
            ),
            cancel_request_id=_checked_text(
                value["cancel_request_id"],
                "cancel_request_id",
                error_type=JournalQuarantineError,
            ),
            reason=_checked_text(
                value["reason"],
                "reason",
                error_type=JournalQuarantineError,
            ),
            observed_at=_decode_datetime(value["observed_at"], "observed_at"),
        )
    raise JournalQuarantineError("event_type is unknown")


def _event_time(value: OrderLifecycleEvent) -> datetime:
    if type(value) is CancellationRequested:
        return value.requested_at
    if type(value) is ConfirmedFill:
        return value.executed_at
    return value.observed_at


def decode_order_lifecycle_event(
    *,
    client_order_id: object,
    event_type: object,
    event_version: object,
    event_payload: object,
    event_payload_sha256: object,
    trade_id: object,
    occurred_at: object,
) -> OrderLifecycleEvent:
    """Decode and cross-check one untrusted persisted lifecycle envelope."""
    if type(event_version) is not int or event_version != _PAYLOAD_VERSION:
        raise JournalQuarantineError("event_version is unknown")
    indexed_event_type = _checked_text(
        event_type,
        "indexed event_type",
        error_type=JournalQuarantineError,
    )
    if indexed_event_type not in _EVENT_TYPES:
        raise JournalQuarantineError("event_type is unknown")
    payload = _verified_payload(
        event_payload,
        event_payload_sha256,
        "event_payload",
    )
    try:
        event = _decode_event_payload(indexed_event_type, payload)
    except JournalQuarantineError:
        raise
    except (TypeError, ValueError) as exc:
        raise JournalQuarantineError("event payload violates the domain") from exc

    indexed_client_order_id = _checked_text(
        client_order_id,
        "indexed client_order_id",
        error_type=JournalQuarantineError,
        max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
    )
    indexed_trade_id = _optional_text(
        trade_id,
        "indexed trade_id",
        error_type=JournalQuarantineError,
        max_length=_TRADE_ID_MAX_LENGTH,
    )
    indexed_occurred_at = _utc_datetime(
        occurred_at,
        "indexed occurred_at",
        JournalQuarantineError,
    )
    expected_trade_id = event.trade_id if type(event) is ConfirmedFill else None
    if (
        indexed_client_order_id != event.client_order_id
        or indexed_trade_id != expected_trade_id
        or indexed_occurred_at != _event_time(event)
    ):
        raise JournalQuarantineError("event indexed columns conflict with payload")
    return event


__all__ = [
    "EncodedOrderLifecycleEvent",
    "EncodedPositionInstruction",
    "JournalCodecError",
    "JournalEncodeError",
    "JournalQuarantineError",
    "decode_order_lifecycle_event",
    "decode_position_instruction",
    "encode_order_lifecycle_event",
    "encode_position_instruction",
]
