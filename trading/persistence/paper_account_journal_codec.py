"""Strict, compact codecs for the paper-account journal boundary."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from trading.domain._validation import protect_frozen_dataclass_state
from trading.domain.paper_accounting import (
    InvalidPaperAccountTransition,
    PaperAccount,
    PaperAccountAdmission,
    PaperAccountAdmissionDisposition,
    PaperAccountBalance,
    PaperAccountPolicy,
    PaperAccountPosting,
    admit_paper_settlement,
    new_paper_account,
)
from trading.domain.paper_settlement import PaperSettlement
from trading.persistence.journal_codec import (
    JournalEncodeError,
    JournalQuarantineError,
)

_PAYLOAD_VERSION = 1
_BIGINT_MAX = (1 << 63) - 1
_EXECUTION_SCOPE_MAX_LENGTH = 128
_ACCOUNT_KEY_MAX_LENGTH = 255
_POSITION_KEY_MAX_LENGTH = 255
_CLIENT_ORDER_ID_MAX_LENGTH = 255
_EVENT_ID_MAX_LENGTH = 255
_TRADE_ID_MAX_LENGTH = 255
_SYMBOL_MAX_LENGTH = 64
_ASSET_MAX_LENGTH = 64


class _DuplicateJsonKey(ValueError):
    pass


def _checked_text(
    value: object,
    field: str,
    *,
    error_type: type[ValueError],
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


def _checked_bigint(
    value: object,
    field: str,
    *,
    error_type: type[ValueError],
) -> int:
    if type(value) is not int:
        raise error_type(f"{field} must be an integer")
    if value < 1 or value > _BIGINT_MAX:
        raise error_type(f"{field} is outside durable storage bounds")
    return value


def _checked_sha256(
    value: object,
    field: str,
    *,
    error_type: type[ValueError],
) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise error_type(f"{field} SHA-256 is invalid")
    return value


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


def _utc_datetime(
    value: object,
    field: str,
    error_type: type[ValueError],
) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise error_type(f"{field} must be a timezone-aware datetime")
    try:
        offset = value.utcoffset()
        normalized = value.astimezone(timezone.utc)
    except Exception as exc:
        raise error_type(f"{field} cannot be represented in UTC") from exc
    if offset is None:
        raise error_type(f"{field} must be a timezone-aware datetime")
    return normalized


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


def _canonical_json(payload: object, error_type: type[ValueError]) -> str:
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
    result: dict[str, object] = {}
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
    expected = _checked_sha256(
        checksum,
        field,
        error_type=JournalQuarantineError,
    )
    payload = _payload_object(value, field)
    canonical = _canonical_json(payload, JournalQuarantineError)
    if not hmac.compare_digest(_payload_sha256(canonical), expected):
        raise JournalQuarantineError(f"{field} SHA-256 does not match")
    return payload


def _exact_keys(value: object, expected: set[str], field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise JournalQuarantineError(f"{field} has an unknown payload shape")
    return value


def _exact_list(value: object, field: str) -> list[object]:
    if type(value) is not list:
        raise JournalQuarantineError(f"{field} must be a JSON array")
    return value


def _balance_payload(balance: PaperAccountBalance) -> dict[str, object]:
    return {
        "asset": _checked_text(
            balance.asset,
            "asset",
            error_type=JournalEncodeError,
            max_length=_ASSET_MAX_LENGTH,
        ),
        "available": _decimal_text(balance.available, "available"),
        "reserved": _decimal_text(balance.reserved, "reserved"),
    }


def _posting_payload(posting: PaperAccountPosting) -> dict[str, object]:
    return {
        "asset": _checked_text(
            posting.asset,
            "posting asset",
            error_type=JournalEncodeError,
            max_length=_ASSET_MAX_LENGTH,
        ),
        "bucket": posting.bucket.value,
        "amount": _decimal_text(posting.amount, "posting amount"),
    }


def _amount_payload(amount: object) -> dict[str, object]:
    return {
        "asset": _checked_text(
            amount.asset,
            "settlement asset",
            error_type=JournalEncodeError,
            max_length=_ASSET_MAX_LENGTH,
        ),
        "amount": _decimal_text(amount.amount, "settlement amount"),
    }


def _opening_payload(
    execution_scope: str,
    owner_generation: int,
    account: PaperAccount,
) -> dict[str, object]:
    return {
        "execution_scope": execution_scope,
        "owner_generation": owner_generation,
        "policy": {
            "account_key": account.policy.account_key,
            "collateral_asset": account.policy.collateral_asset,
            "margin_quantum": _decimal_text(
                account.policy.margin_quantum,
                "margin_quantum",
            ),
        },
        "opening_balances": [
            _balance_payload(balance) for balance in account.opening_balances
        ],
    }


def _settlement_payload(
    admission: PaperAccountAdmission,
) -> dict[str, object]:
    settlement = admission.settlement
    record = settlement.record
    instruction = record.position_fill.instruction
    fill = record.position_fill.fill
    position_margin = next(
        (
            reservation.amount
            for reservation in admission.after.reservations
            if reservation.position_key == instruction.position_key
        ),
        None,
    )
    return {
        "disposition": "APPLIED",
        "account": {
            "account_key": admission.before.policy.account_key,
            "collateral_asset": admission.before.policy.collateral_asset,
        },
        "settlement_ref": {
            "account_version": admission.account_version,
            "position_key": instruction.position_key,
            "position_version": record.position_version,
            "client_order_id": fill.client_order_id,
            "event_id": record.event_id,
            "trade_id": fill.trade_id,
        },
        "instrument": {
            "kind": "LINEAR_QUOTE_MULTIPLIER_ONE",
            "version": 1,
            "symbol": settlement.instrument.symbol,
            "base_asset": settlement.instrument.base_asset,
            "quote_asset": settlement.instrument.quote_asset,
        },
        "settlement_deltas": {
            "gross_realized_pnl_delta": _amount_payload(
                settlement.gross_realized_pnl_delta
            ),
            "fee_debits": [_amount_payload(amount) for amount in settlement.fee_debits],
            "cash_deltas": [
                _amount_payload(amount) for amount in settlement.cash_deltas
            ],
        },
        "postings": [_posting_payload(posting) for posting in admission.postings],
        "account_state_after": admission.after.state.value,
        "position_margin_after": (
            _decimal_text(position_margin, "position_margin_after")
            if position_margin is not None
            else None
        ),
    }


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class EncodedPaperAccountOpening:
    """Canonical opening policy and balances for one scoped account generation."""

    execution_scope: str
    account_key: str
    owner_generation: int
    collateral_asset: str
    opening_version: int
    opening_payload: str
    opening_payload_sha256: str

    def __post_init__(self) -> None:
        _checked_text(
            self.execution_scope,
            "execution_scope",
            error_type=JournalEncodeError,
            max_length=_EXECUTION_SCOPE_MAX_LENGTH,
        )
        _checked_text(
            self.account_key,
            "account_key",
            error_type=JournalEncodeError,
            max_length=_ACCOUNT_KEY_MAX_LENGTH,
        )
        _checked_bigint(
            self.owner_generation,
            "owner_generation",
            error_type=JournalEncodeError,
        )
        _checked_text(
            self.collateral_asset,
            "collateral_asset",
            error_type=JournalEncodeError,
            max_length=_ASSET_MAX_LENGTH,
        )
        if type(self.opening_version) is not int or self.opening_version != 1:
            raise JournalEncodeError("opening_version is unknown")
        if type(self.opening_payload) is not str:
            raise JournalEncodeError("opening_payload must be text")
        _checked_sha256(
            self.opening_payload_sha256,
            "opening_payload",
            error_type=JournalEncodeError,
        )


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class EncodedPaperAccountSettlement:
    """Compact account ledger payload for one newly applied fill settlement."""

    account_key: str
    collateral_asset: str
    account_version: int
    position_key: str
    position_version: int
    client_order_id: str
    event_id: str
    trade_id: str
    symbol: str
    base_asset: str
    quote_asset: str
    instrument_version: int
    settlement_version: int
    settlement_payload: str
    settlement_payload_sha256: str

    def __post_init__(self) -> None:
        for field, value, maximum in (
            ("account_key", self.account_key, _ACCOUNT_KEY_MAX_LENGTH),
            ("collateral_asset", self.collateral_asset, _ASSET_MAX_LENGTH),
            ("position_key", self.position_key, _POSITION_KEY_MAX_LENGTH),
            ("client_order_id", self.client_order_id, _CLIENT_ORDER_ID_MAX_LENGTH),
            ("event_id", self.event_id, _EVENT_ID_MAX_LENGTH),
            ("trade_id", self.trade_id, _TRADE_ID_MAX_LENGTH),
            ("symbol", self.symbol, _SYMBOL_MAX_LENGTH),
            ("base_asset", self.base_asset, _ASSET_MAX_LENGTH),
            ("quote_asset", self.quote_asset, _ASSET_MAX_LENGTH),
        ):
            _checked_text(
                value,
                field,
                error_type=JournalEncodeError,
                max_length=maximum,
            )
        _checked_bigint(
            self.account_version,
            "account_version",
            error_type=JournalEncodeError,
        )
        _checked_bigint(
            self.position_version,
            "position_version",
            error_type=JournalEncodeError,
        )
        if type(self.instrument_version) is not int or self.instrument_version != 1:
            raise JournalEncodeError("instrument_version is unknown")
        if type(self.settlement_version) is not int or self.settlement_version != 1:
            raise JournalEncodeError("settlement_version is unknown")
        if type(self.settlement_payload) is not str:
            raise JournalEncodeError("settlement_payload must be text")
        _checked_sha256(
            self.settlement_payload_sha256,
            "settlement_payload",
            error_type=JournalEncodeError,
        )


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountBatchFill:
    """One journal fill and account settlement bound into an owner manifest."""

    position_key: str
    client_order_id: str
    event_id: str
    trade_id: str
    position_version: int
    account_version: int
    event_payload_sha256: str
    account_settlement_payload_sha256: str

    def __post_init__(self) -> None:
        for field, value, maximum in (
            ("position_key", self.position_key, _POSITION_KEY_MAX_LENGTH),
            ("client_order_id", self.client_order_id, _CLIENT_ORDER_ID_MAX_LENGTH),
            ("event_id", self.event_id, _EVENT_ID_MAX_LENGTH),
            ("trade_id", self.trade_id, _TRADE_ID_MAX_LENGTH),
        ):
            _checked_text(
                value,
                field,
                error_type=ValueError,
                max_length=maximum,
            )
        _checked_bigint(
            self.position_version,
            "position_version",
            error_type=ValueError,
        )
        _checked_bigint(
            self.account_version,
            "account_version",
            error_type=ValueError,
        )
        _checked_sha256(
            self.event_payload_sha256,
            "event_payload",
            error_type=ValueError,
        )
        _checked_sha256(
            self.account_settlement_payload_sha256,
            "account_settlement_payload",
            error_type=ValueError,
        )


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountBatchManifest:
    """Exact owner provenance for one ACK plus terminal full-fill account batch."""

    execution_scope: str
    account_key: str
    owner_generation: int
    position_key: str
    client_order_id: str
    instruction_payload_sha256: str
    submission_event_id: str
    submission_position_version: int
    submission_observed_at: datetime
    submission_event_payload_sha256: str
    fills: tuple[PaperAccountBatchFill, ...]

    def __post_init__(self) -> None:
        for field, value, maximum in (
            ("execution_scope", self.execution_scope, _EXECUTION_SCOPE_MAX_LENGTH),
            ("account_key", self.account_key, _ACCOUNT_KEY_MAX_LENGTH),
            ("position_key", self.position_key, _POSITION_KEY_MAX_LENGTH),
            ("client_order_id", self.client_order_id, _CLIENT_ORDER_ID_MAX_LENGTH),
            ("submission_event_id", self.submission_event_id, _EVENT_ID_MAX_LENGTH),
        ):
            _checked_text(
                value,
                field,
                error_type=ValueError,
                max_length=maximum,
            )
        _checked_bigint(
            self.owner_generation,
            "owner_generation",
            error_type=ValueError,
        )
        _checked_bigint(
            self.submission_position_version,
            "submission_position_version",
            error_type=ValueError,
        )
        _checked_sha256(
            self.instruction_payload_sha256,
            "instruction_payload",
            error_type=ValueError,
        )
        object.__setattr__(
            self,
            "submission_observed_at",
            _utc_datetime(
                self.submission_observed_at,
                "submission_observed_at",
                ValueError,
            ),
        )
        _checked_sha256(
            self.submission_event_payload_sha256,
            "submission_event_payload",
            error_type=ValueError,
        )
        if type(self.fills) is not tuple or not self.fills:
            raise TypeError("fills must be a non-empty exact tuple")
        if any(type(fill) is not PaperAccountBatchFill for fill in self.fills):
            raise TypeError("fills must contain PaperAccountBatchFill values")
        if any(
            fill.position_key != self.position_key
            or fill.client_order_id != self.client_order_id
            for fill in self.fills
        ):
            raise ValueError("fill references must belong to the manifest order")
        position_versions = tuple(fill.position_version for fill in self.fills)
        if position_versions != tuple(
            range(
                self.submission_position_version + 1,
                self.submission_position_version + len(self.fills) + 1,
            )
        ):
            raise ValueError("fill position versions must follow the ACK consecutively")
        first_account_version = self.fills[0].account_version
        if tuple(fill.account_version for fill in self.fills) != tuple(
            range(first_account_version, first_account_version + len(self.fills))
        ):
            raise ValueError("fill account versions must be consecutive")
        event_ids = (self.submission_event_id,) + tuple(
            fill.event_id for fill in self.fills
        )
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("manifest event IDs must be unique")
        trade_ids = tuple(fill.trade_id for fill in self.fills)
        if len(trade_ids) != len(set(trade_ids)):
            raise ValueError("manifest trade IDs must be unique")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class EncodedPaperAccountBatch:
    """Canonical manifest and indexed range columns for one atomic owner batch."""

    execution_scope: str
    account_key: str
    owner_generation: int
    position_key: str
    client_order_id: str
    instruction_payload_sha256: str
    submission_event_id: str
    submission_position_version: int
    submission_observed_at: datetime
    first_account_version: int
    last_account_version: int
    last_position_version: int
    fill_count: int
    batch_version: int
    batch_payload: str
    batch_payload_sha256: str

    def __post_init__(self) -> None:
        for field, value, maximum in (
            ("execution_scope", self.execution_scope, _EXECUTION_SCOPE_MAX_LENGTH),
            ("account_key", self.account_key, _ACCOUNT_KEY_MAX_LENGTH),
            ("position_key", self.position_key, _POSITION_KEY_MAX_LENGTH),
            ("client_order_id", self.client_order_id, _CLIENT_ORDER_ID_MAX_LENGTH),
            ("submission_event_id", self.submission_event_id, _EVENT_ID_MAX_LENGTH),
        ):
            _checked_text(
                value,
                field,
                error_type=JournalEncodeError,
                max_length=maximum,
            )
        for field, value in (
            ("owner_generation", self.owner_generation),
            ("submission_position_version", self.submission_position_version),
            ("first_account_version", self.first_account_version),
            ("last_account_version", self.last_account_version),
            ("last_position_version", self.last_position_version),
            ("fill_count", self.fill_count),
        ):
            _checked_bigint(value, field, error_type=JournalEncodeError)
        _checked_sha256(
            self.instruction_payload_sha256,
            "instruction_payload",
            error_type=JournalEncodeError,
        )
        object.__setattr__(
            self,
            "submission_observed_at",
            _utc_datetime(
                self.submission_observed_at,
                "submission_observed_at",
                JournalEncodeError,
            ),
        )
        if self.last_position_version - self.submission_position_version != (
            self.fill_count
        ):
            raise JournalEncodeError("position-version range conflicts with fill_count")
        if self.last_account_version - self.first_account_version + 1 != (
            self.fill_count
        ):
            raise JournalEncodeError("account-version range conflicts with fill_count")
        if type(self.batch_version) is not int or self.batch_version != 1:
            raise JournalEncodeError("batch_version is unknown")
        if type(self.batch_payload) is not str:
            raise JournalEncodeError("batch_payload must be text")
        _checked_sha256(
            self.batch_payload_sha256,
            "batch_payload",
            error_type=JournalEncodeError,
        )


def encode_paper_account_opening(
    execution_scope: str,
    owner_generation: int,
    account: PaperAccount,
    /,
) -> EncodedPaperAccountOpening:
    """Encode an empty account and its explicit scoped owner provenance."""
    scope = _checked_text(
        execution_scope,
        "execution_scope",
        error_type=JournalEncodeError,
        max_length=_EXECUTION_SCOPE_MAX_LENGTH,
    )
    generation = _checked_bigint(
        owner_generation,
        "owner_generation",
        error_type=JournalEncodeError,
    )
    if type(account) is not PaperAccount:
        raise JournalEncodeError("account must be a PaperAccount")
    if (
        account.records
        or account.reservations
        or account.balances != (account.opening_balances)
    ):
        raise JournalEncodeError("opening codec requires an empty paper account")
    payload = _opening_payload(scope, generation, account)
    payload_json = _canonical_json(payload, JournalEncodeError)
    return EncodedPaperAccountOpening(
        execution_scope=scope,
        account_key=_checked_text(
            account.policy.account_key,
            "account_key",
            error_type=JournalEncodeError,
            max_length=_ACCOUNT_KEY_MAX_LENGTH,
        ),
        owner_generation=generation,
        collateral_asset=_checked_text(
            account.policy.collateral_asset,
            "collateral_asset",
            error_type=JournalEncodeError,
            max_length=_ASSET_MAX_LENGTH,
        ),
        opening_version=_PAYLOAD_VERSION,
        opening_payload=payload_json,
        opening_payload_sha256=_payload_sha256(payload_json),
    )


def decode_paper_account_opening(
    *,
    execution_scope: object,
    account_key: object,
    owner_generation: object,
    collateral_asset: object,
    opening_version: object,
    opening_payload: object,
    opening_payload_sha256: object,
) -> PaperAccount:
    """Decode and cross-check an untrusted persisted account opening."""
    if type(opening_version) is not int or opening_version != _PAYLOAD_VERSION:
        raise JournalQuarantineError("opening_version is unknown")
    payload = _verified_payload(
        opening_payload,
        opening_payload_sha256,
        "opening_payload",
    )
    payload = _exact_keys(
        payload,
        {"execution_scope", "owner_generation", "policy", "opening_balances"},
        "opening_payload",
    )
    policy_payload = _exact_keys(
        payload["policy"],
        {"account_key", "collateral_asset", "margin_quantum"},
        "policy",
    )
    balances_payload = _exact_list(payload["opening_balances"], "opening_balances")
    try:
        policy = PaperAccountPolicy(
            account_key=_checked_text(
                policy_payload["account_key"],
                "account_key",
                error_type=JournalQuarantineError,
                max_length=_ACCOUNT_KEY_MAX_LENGTH,
            ),
            collateral_asset=_checked_text(
                policy_payload["collateral_asset"],
                "collateral_asset",
                error_type=JournalQuarantineError,
                max_length=_ASSET_MAX_LENGTH,
            ),
            margin_quantum=_decode_decimal(
                policy_payload["margin_quantum"],
                "margin_quantum",
            ),
        )
        balances = tuple(
            PaperAccountBalance(
                asset=_checked_text(
                    balance_payload["asset"],
                    "asset",
                    error_type=JournalQuarantineError,
                    max_length=_ASSET_MAX_LENGTH,
                ),
                available=_decode_decimal(
                    balance_payload["available"],
                    "available",
                ),
                reserved=_decode_decimal(
                    balance_payload["reserved"],
                    "reserved",
                ),
            )
            for raw_balance in balances_payload
            for balance_payload in (
                _exact_keys(
                    raw_balance,
                    {"asset", "available", "reserved"},
                    "opening balance",
                ),
            )
        )
        account = new_paper_account(policy, balances)
    except JournalQuarantineError:
        raise
    except (TypeError, ValueError) as exc:
        raise JournalQuarantineError("opening payload violates the domain") from exc

    indexed = (
        _checked_text(
            execution_scope,
            "indexed execution_scope",
            error_type=JournalQuarantineError,
            max_length=_EXECUTION_SCOPE_MAX_LENGTH,
        ),
        _checked_text(
            account_key,
            "indexed account_key",
            error_type=JournalQuarantineError,
            max_length=_ACCOUNT_KEY_MAX_LENGTH,
        ),
        _checked_bigint(
            owner_generation,
            "indexed owner_generation",
            error_type=JournalQuarantineError,
        ),
        _checked_text(
            collateral_asset,
            "indexed collateral_asset",
            error_type=JournalQuarantineError,
            max_length=_ASSET_MAX_LENGTH,
        ),
    )
    embedded = (
        _checked_text(
            payload["execution_scope"],
            "execution_scope",
            error_type=JournalQuarantineError,
            max_length=_EXECUTION_SCOPE_MAX_LENGTH,
        ),
        account.policy.account_key,
        _checked_bigint(
            payload["owner_generation"],
            "owner_generation",
            error_type=JournalQuarantineError,
        ),
        account.policy.collateral_asset,
    )
    if indexed != embedded:
        raise JournalQuarantineError("opening indexed columns conflict with payload")
    return account


def encode_paper_account_settlement(
    admission: PaperAccountAdmission,
    /,
) -> EncodedPaperAccountSettlement:
    """Encode one newly applied account settlement without recursive history."""
    if type(admission) is not PaperAccountAdmission:
        raise JournalEncodeError("admission must be a PaperAccountAdmission")
    if admission.disposition is not PaperAccountAdmissionDisposition.APPLIED:
        raise JournalEncodeError("only APPLIED admissions are durable settlements")
    settlement = admission.settlement
    record = settlement.record
    fill = record.position_fill.fill
    instruction = record.position_fill.instruction
    payload = _settlement_payload(admission)
    payload_json = _canonical_json(payload, JournalEncodeError)
    return EncodedPaperAccountSettlement(
        account_key=admission.before.policy.account_key,
        collateral_asset=admission.before.policy.collateral_asset,
        account_version=admission.account_version,
        position_key=instruction.position_key,
        position_version=record.position_version,
        client_order_id=fill.client_order_id,
        event_id=record.event_id,
        trade_id=fill.trade_id,
        symbol=settlement.instrument.symbol,
        base_asset=settlement.instrument.base_asset,
        quote_asset=settlement.instrument.quote_asset,
        instrument_version=_PAYLOAD_VERSION,
        settlement_version=_PAYLOAD_VERSION,
        settlement_payload=payload_json,
        settlement_payload_sha256=_payload_sha256(payload_json),
    )


def decode_paper_account_settlement(
    before: PaperAccount,
    settlement: PaperSettlement,
    /,
    *,
    account_key: object,
    collateral_asset: object,
    account_version: object,
    position_key: object,
    position_version: object,
    client_order_id: object,
    event_id: object,
    trade_id: object,
    symbol: object,
    base_asset: object,
    quote_asset: object,
    instrument_version: object,
    settlement_version: object,
    settlement_payload: object,
    settlement_payload_sha256: object,
) -> PaperAccountAdmission:
    """Re-derive and cross-check one untrusted account-settlement row."""
    if type(before) is not PaperAccount or type(settlement) is not PaperSettlement:
        raise JournalQuarantineError(
            "before and settlement must be validated domain values"
        )
    if type(instrument_version) is not int or instrument_version != _PAYLOAD_VERSION:
        raise JournalQuarantineError("instrument_version is unknown")
    if type(settlement_version) is not int or settlement_version != _PAYLOAD_VERSION:
        raise JournalQuarantineError("settlement_version is unknown")
    version = _checked_bigint(
        account_version,
        "indexed account_version",
        error_type=JournalQuarantineError,
    )
    payload = _verified_payload(
        settlement_payload,
        settlement_payload_sha256,
        "settlement_payload",
    )
    try:
        admission = admit_paper_settlement(before, version, settlement)
    except (InvalidPaperAccountTransition, TypeError, ValueError) as exc:
        raise JournalQuarantineError(
            "settlement contradicts the prior account"
        ) from exc
    if admission.disposition is not PaperAccountAdmissionDisposition.APPLIED:
        raise JournalQuarantineError("durable settlement must be newly APPLIED")
    try:
        expected_payload = _settlement_payload(admission)
        expected_json = _canonical_json(expected_payload, JournalEncodeError)
        encoded = encode_paper_account_settlement(admission)
    except JournalEncodeError as exc:
        raise JournalQuarantineError(
            "settlement domain value is not representable"
        ) from exc
    if _canonical_json(payload, JournalQuarantineError) != expected_json:
        raise JournalQuarantineError("settlement payload conflicts with the domain")
    indexed = (
        _checked_text(
            account_key,
            "indexed account_key",
            error_type=JournalQuarantineError,
            max_length=_ACCOUNT_KEY_MAX_LENGTH,
        ),
        _checked_text(
            collateral_asset,
            "indexed collateral_asset",
            error_type=JournalQuarantineError,
            max_length=_ASSET_MAX_LENGTH,
        ),
        version,
        _checked_text(
            position_key,
            "indexed position_key",
            error_type=JournalQuarantineError,
            max_length=_POSITION_KEY_MAX_LENGTH,
        ),
        _checked_bigint(
            position_version,
            "indexed position_version",
            error_type=JournalQuarantineError,
        ),
        _checked_text(
            client_order_id,
            "indexed client_order_id",
            error_type=JournalQuarantineError,
            max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
        ),
        _checked_text(
            event_id,
            "indexed event_id",
            error_type=JournalQuarantineError,
            max_length=_EVENT_ID_MAX_LENGTH,
        ),
        _checked_text(
            trade_id,
            "indexed trade_id",
            error_type=JournalQuarantineError,
            max_length=_TRADE_ID_MAX_LENGTH,
        ),
        _checked_text(
            symbol,
            "indexed symbol",
            error_type=JournalQuarantineError,
            max_length=_SYMBOL_MAX_LENGTH,
        ),
        _checked_text(
            base_asset,
            "indexed base_asset",
            error_type=JournalQuarantineError,
            max_length=_ASSET_MAX_LENGTH,
        ),
        _checked_text(
            quote_asset,
            "indexed quote_asset",
            error_type=JournalQuarantineError,
            max_length=_ASSET_MAX_LENGTH,
        ),
    )
    expected = (
        encoded.account_key,
        encoded.collateral_asset,
        encoded.account_version,
        encoded.position_key,
        encoded.position_version,
        encoded.client_order_id,
        encoded.event_id,
        encoded.trade_id,
        encoded.symbol,
        encoded.base_asset,
        encoded.quote_asset,
    )
    if indexed != expected:
        raise JournalQuarantineError("settlement indexed columns conflict with payload")
    return admission


def _batch_payload(manifest: PaperAccountBatchManifest) -> dict[str, object]:
    observed_at, _ = _datetime_text(
        manifest.submission_observed_at,
        "submission_observed_at",
    )
    return {
        "execution_scope": manifest.execution_scope,
        "account_key": manifest.account_key,
        "owner_generation": manifest.owner_generation,
        "position_key": manifest.position_key,
        "client_order_id": manifest.client_order_id,
        "instruction_payload_sha256": manifest.instruction_payload_sha256,
        "submission": {
            "event_id": manifest.submission_event_id,
            "position_version": manifest.submission_position_version,
            "observed_at": observed_at,
            "event_payload_sha256": manifest.submission_event_payload_sha256,
        },
        "fills": [
            {
                "position_key": fill.position_key,
                "client_order_id": fill.client_order_id,
                "event_id": fill.event_id,
                "trade_id": fill.trade_id,
                "position_version": fill.position_version,
                "account_version": fill.account_version,
                "event_payload_sha256": fill.event_payload_sha256,
                "account_settlement_payload_sha256": (
                    fill.account_settlement_payload_sha256
                ),
            }
            for fill in manifest.fills
        ],
    }


def encode_paper_account_batch(
    manifest: PaperAccountBatchManifest,
    /,
) -> EncodedPaperAccountBatch:
    """Encode one exact owner manifest for an ACK and its accounted fills."""
    if type(manifest) is not PaperAccountBatchManifest:
        raise JournalEncodeError("manifest must be a PaperAccountBatchManifest")
    payload_json = _canonical_json(_batch_payload(manifest), JournalEncodeError)
    return EncodedPaperAccountBatch(
        execution_scope=manifest.execution_scope,
        account_key=manifest.account_key,
        owner_generation=manifest.owner_generation,
        position_key=manifest.position_key,
        client_order_id=manifest.client_order_id,
        instruction_payload_sha256=manifest.instruction_payload_sha256,
        submission_event_id=manifest.submission_event_id,
        submission_position_version=manifest.submission_position_version,
        submission_observed_at=_utc_datetime(
            manifest.submission_observed_at,
            "submission_observed_at",
            JournalEncodeError,
        ),
        first_account_version=manifest.fills[0].account_version,
        last_account_version=manifest.fills[-1].account_version,
        last_position_version=manifest.fills[-1].position_version,
        fill_count=len(manifest.fills),
        batch_version=_PAYLOAD_VERSION,
        batch_payload=payload_json,
        batch_payload_sha256=_payload_sha256(payload_json),
    )


def decode_paper_account_batch(
    *,
    execution_scope: object,
    account_key: object,
    owner_generation: object,
    position_key: object,
    client_order_id: object,
    instruction_payload_sha256: object,
    submission_event_id: object,
    submission_position_version: object,
    submission_observed_at: object,
    first_account_version: object,
    last_account_version: object,
    last_position_version: object,
    fill_count: object,
    batch_version: object,
    batch_payload: object,
    batch_payload_sha256: object,
) -> PaperAccountBatchManifest:
    """Decode and cross-check one untrusted atomic-owner batch manifest."""
    if type(batch_version) is not int or batch_version != _PAYLOAD_VERSION:
        raise JournalQuarantineError("batch_version is unknown")
    payload = _verified_payload(batch_payload, batch_payload_sha256, "batch_payload")
    payload = _exact_keys(
        payload,
        {
            "execution_scope",
            "account_key",
            "owner_generation",
            "position_key",
            "client_order_id",
            "instruction_payload_sha256",
            "submission",
            "fills",
        },
        "batch_payload",
    )
    submission = _exact_keys(
        payload["submission"],
        {"event_id", "position_version", "observed_at", "event_payload_sha256"},
        "submission",
    )
    fills_payload = _exact_list(payload["fills"], "fills")
    try:
        manifest = PaperAccountBatchManifest(
            execution_scope=_checked_text(
                payload["execution_scope"],
                "execution_scope",
                error_type=JournalQuarantineError,
                max_length=_EXECUTION_SCOPE_MAX_LENGTH,
            ),
            account_key=_checked_text(
                payload["account_key"],
                "account_key",
                error_type=JournalQuarantineError,
                max_length=_ACCOUNT_KEY_MAX_LENGTH,
            ),
            owner_generation=_checked_bigint(
                payload["owner_generation"],
                "owner_generation",
                error_type=JournalQuarantineError,
            ),
            position_key=_checked_text(
                payload["position_key"],
                "position_key",
                error_type=JournalQuarantineError,
                max_length=_POSITION_KEY_MAX_LENGTH,
            ),
            client_order_id=_checked_text(
                payload["client_order_id"],
                "client_order_id",
                error_type=JournalQuarantineError,
                max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
            ),
            instruction_payload_sha256=_checked_sha256(
                payload["instruction_payload_sha256"],
                "instruction_payload",
                error_type=JournalQuarantineError,
            ),
            submission_event_id=_checked_text(
                submission["event_id"],
                "submission_event_id",
                error_type=JournalQuarantineError,
                max_length=_EVENT_ID_MAX_LENGTH,
            ),
            submission_position_version=_checked_bigint(
                submission["position_version"],
                "submission_position_version",
                error_type=JournalQuarantineError,
            ),
            submission_observed_at=_decode_datetime(
                submission["observed_at"],
                "submission_observed_at",
            ),
            submission_event_payload_sha256=_checked_sha256(
                submission["event_payload_sha256"],
                "submission_event_payload",
                error_type=JournalQuarantineError,
            ),
            fills=tuple(
                PaperAccountBatchFill(
                    position_key=_checked_text(
                        fill["position_key"],
                        "fill position_key",
                        error_type=JournalQuarantineError,
                        max_length=_POSITION_KEY_MAX_LENGTH,
                    ),
                    client_order_id=_checked_text(
                        fill["client_order_id"],
                        "fill client_order_id",
                        error_type=JournalQuarantineError,
                        max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
                    ),
                    event_id=_checked_text(
                        fill["event_id"],
                        "fill event_id",
                        error_type=JournalQuarantineError,
                        max_length=_EVENT_ID_MAX_LENGTH,
                    ),
                    trade_id=_checked_text(
                        fill["trade_id"],
                        "fill trade_id",
                        error_type=JournalQuarantineError,
                        max_length=_TRADE_ID_MAX_LENGTH,
                    ),
                    position_version=_checked_bigint(
                        fill["position_version"],
                        "fill position_version",
                        error_type=JournalQuarantineError,
                    ),
                    account_version=_checked_bigint(
                        fill["account_version"],
                        "fill account_version",
                        error_type=JournalQuarantineError,
                    ),
                    event_payload_sha256=_checked_sha256(
                        fill["event_payload_sha256"],
                        "fill event_payload",
                        error_type=JournalQuarantineError,
                    ),
                    account_settlement_payload_sha256=_checked_sha256(
                        fill["account_settlement_payload_sha256"],
                        "fill account_settlement_payload",
                        error_type=JournalQuarantineError,
                    ),
                )
                for raw_fill in fills_payload
                for fill in (
                    _exact_keys(
                        raw_fill,
                        {
                            "position_key",
                            "client_order_id",
                            "event_id",
                            "trade_id",
                            "position_version",
                            "account_version",
                            "event_payload_sha256",
                            "account_settlement_payload_sha256",
                        },
                        "batch fill",
                    ),
                )
            ),
        )
    except JournalQuarantineError:
        raise
    except (TypeError, ValueError) as exc:
        raise JournalQuarantineError("batch payload violates its contract") from exc

    try:
        encoded = encode_paper_account_batch(manifest)
    except JournalEncodeError as exc:
        raise JournalQuarantineError("batch domain value is not representable") from exc
    indexed = (
        _checked_text(
            execution_scope,
            "indexed execution_scope",
            error_type=JournalQuarantineError,
            max_length=_EXECUTION_SCOPE_MAX_LENGTH,
        ),
        _checked_text(
            account_key,
            "indexed account_key",
            error_type=JournalQuarantineError,
            max_length=_ACCOUNT_KEY_MAX_LENGTH,
        ),
        _checked_bigint(
            owner_generation,
            "indexed owner_generation",
            error_type=JournalQuarantineError,
        ),
        _checked_text(
            position_key,
            "indexed position_key",
            error_type=JournalQuarantineError,
            max_length=_POSITION_KEY_MAX_LENGTH,
        ),
        _checked_text(
            client_order_id,
            "indexed client_order_id",
            error_type=JournalQuarantineError,
            max_length=_CLIENT_ORDER_ID_MAX_LENGTH,
        ),
        _checked_sha256(
            instruction_payload_sha256,
            "indexed instruction_payload",
            error_type=JournalQuarantineError,
        ),
        _checked_text(
            submission_event_id,
            "indexed submission_event_id",
            error_type=JournalQuarantineError,
            max_length=_EVENT_ID_MAX_LENGTH,
        ),
        _checked_bigint(
            submission_position_version,
            "indexed submission_position_version",
            error_type=JournalQuarantineError,
        ),
        _utc_datetime(
            submission_observed_at,
            "indexed submission_observed_at",
            JournalQuarantineError,
        ),
        _checked_bigint(
            first_account_version,
            "indexed first_account_version",
            error_type=JournalQuarantineError,
        ),
        _checked_bigint(
            last_account_version,
            "indexed last_account_version",
            error_type=JournalQuarantineError,
        ),
        _checked_bigint(
            last_position_version,
            "indexed last_position_version",
            error_type=JournalQuarantineError,
        ),
        _checked_bigint(
            fill_count,
            "indexed fill_count",
            error_type=JournalQuarantineError,
        ),
    )
    expected = (
        encoded.execution_scope,
        encoded.account_key,
        encoded.owner_generation,
        encoded.position_key,
        encoded.client_order_id,
        encoded.instruction_payload_sha256,
        encoded.submission_event_id,
        encoded.submission_position_version,
        encoded.submission_observed_at,
        encoded.first_account_version,
        encoded.last_account_version,
        encoded.last_position_version,
        encoded.fill_count,
    )
    if indexed != expected:
        raise JournalQuarantineError("batch indexed columns conflict with payload")
    return manifest


__all__ = [
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
]
