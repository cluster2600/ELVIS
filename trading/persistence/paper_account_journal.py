"""Unwired PostgreSQL provision and replay boundary for paper accounts.

This repository owns only immutable account provisioning and strict read-side
reconstruction.  It never appends settlements, executes an order, or mutates a
legacy/account projection after provisioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Callable

import psycopg2

from trading.domain.order_lifecycle import (
    ConfirmedFill,
    OrderLifecycleState,
    SubmissionAcknowledged,
)
from trading.domain.paper_accounting import (
    PaperAccount,
    PaperAccountAdmissionDisposition,
    PaperAccountPosting,
    PaperAccountPostingBucket,
)
from trading.domain.paper_economics import PaperFillRecord
from trading.domain.paper_settlement import (
    InvalidPaperSettlement,
    PaperLinearInstrument,
    PaperSettlementCheckpoint,
    settle_paper_fill,
)
from trading.domain.positions import PositionFill
from trading.persistence.journal_codec import (
    JournalEncodeError,
    JournalQuarantineError,
    encode_order_lifecycle_event,
)
from trading.persistence.order_position_journal import (
    _READ_TRANSACTION_SQL,
    _WRITE_TRANSACTION_SQL,
    JournalRepositoryError,
    PostgresOrderPositionJournal,
    _replay_stream,
)
from trading.persistence.paper_account_journal_codec import (
    PaperAccountBatchManifest,
    decode_paper_account_batch,
    decode_paper_account_opening,
    decode_paper_account_settlement,
    encode_paper_account_opening,
)

_EXECUTION_SCOPE_MAX_LENGTH = 128
_ACCOUNT_KEY_MAX_LENGTH = 255
_BIGINT_MAX = (1 << 63) - 1

_INSERT_ACCOUNT_STREAM_SQL = """
INSERT INTO np.paper_account_streams (
    account_key,
    execution_scope,
    owner_generation,
    collateral_asset,
    opening_version,
    opening_payload,
    opening_payload_sha256
) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT DO NOTHING
RETURNING account_key
"""

_INSERT_ACCOUNT_BALANCE_SQL = """
INSERT INTO np.paper_account_balances (
    account_key,
    asset,
    available_decimal,
    reserved_decimal
) VALUES (%s, %s, %s, %s)
"""

_SELECT_ACCOUNT_STREAM_SQL = """
SELECT
    account_key,
    execution_scope,
    owner_generation,
    collateral_asset,
    account_version,
    account_state,
    opening_version,
    opening_payload,
    opening_payload_sha256,
    created_at,
    updated_at
FROM np.paper_account_streams
WHERE account_key = %s
"""
_SELECT_ACCOUNT_STREAM_FOR_UPDATE_SQL = _SELECT_ACCOUNT_STREAM_SQL + " FOR UPDATE"

_SELECT_SCOPE_ACCOUNT_KEYS_SQL = """
SELECT account_key
FROM np.paper_account_streams
WHERE execution_scope = %s
"""

_SELECT_ACCOUNT_BALANCES_SQL = """
SELECT asset, available_decimal, reserved_decimal, updated_at
FROM np.paper_account_balances
WHERE account_key = %s
"""

_SELECT_ACCOUNT_RESERVATIONS_SQL = """
SELECT execution_scope, position_key, amount_decimal, updated_at
FROM np.paper_margin_reservations
WHERE account_key = %s
"""

_SELECT_ACCOUNT_BATCHES_SQL = """
SELECT
    account_key,
    client_order_id,
    execution_scope,
    owner_generation,
    opening_version,
    opening_payload_sha256,
    position_key,
    instruction_payload_sha256,
    submission_event_id,
    submission_event_type,
    submission_position_version,
    submission_observed_at,
    submission_event_payload_sha256,
    first_account_version,
    last_account_version,
    last_position_version,
    fill_count,
    batch_version,
    batch_payload,
    batch_payload_sha256,
    recorded_at
FROM np.paper_account_batch_manifests
WHERE account_key = %s
"""

_SELECT_ACCOUNT_SETTLEMENTS_SQL = """
SELECT
    account_key,
    account_version,
    client_order_id,
    fill_ordinal,
    batch_first_account_version,
    batch_submission_position_version,
    batch_fill_count,
    collateral_asset,
    position_key,
    position_version,
    event_id,
    trade_id,
    event_type,
    event_payload_sha256,
    symbol,
    base_asset,
    quote_asset,
    instrument_version,
    settlement_version,
    settlement_payload,
    settlement_payload_sha256,
    recorded_at
FROM np.paper_account_settlements
WHERE account_key = %s
"""

_SELECT_ACCOUNT_POSTINGS_SQL = """
SELECT account_version, posting_ordinal, asset, bucket, amount_decimal
FROM np.paper_account_postings
WHERE account_key = %s
ORDER BY account_version, posting_ordinal
"""


class PaperAccountJournalError(RuntimeError):
    """Base class for paper-account repository failures."""


class PaperAccountInputError(PaperAccountJournalError, ValueError):
    """Raised before I/O when provisioning or lookup input is invalid."""


class PaperAccountStorageError(PaperAccountJournalError):
    """Raised when a repository operation is known not to have committed."""


class PaperAccountCommitUnknown(PaperAccountJournalError):
    """Raised when account provisioning may have committed without an ACK."""

    requires_reconciliation = True

    def __init__(
        self,
        execution_scope: str,
        account_key: str,
        owner_generation: int,
    ) -> None:
        self.execution_scope = execution_scope
        self.account_key = account_key
        self.owner_generation = owner_generation
        super().__init__(
            f"paper account {account_key!r} provisioning commit is unknown"
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (self.execution_scope, self.account_key, self.owner_generation),
        )


class PaperAccountNotFoundError(PaperAccountJournalError):
    """Raised when an account does not exist in the requested scope."""


class PaperAccountReplayError(PaperAccountJournalError):
    """Raised when durable rows cannot reconstruct one exact account."""


class PaperAccountConflictKind(str, Enum):
    """Stable classification for immutable account-opening conflicts."""

    EXECUTION_SCOPE = "EXECUTION_SCOPE"
    OWNER_GENERATION = "OWNER_GENERATION"
    OPENING_IDENTITY = "OPENING_IDENTITY"


class PaperAccountConflictError(PaperAccountJournalError):
    """Raised when an account key is already bound to different provenance."""

    def __init__(self, kind: PaperAccountConflictKind, message: str) -> None:
        self.kind = kind
        super().__init__(message)


class ProvisionDisposition(str, Enum):
    """Whether this call created or rediscovered an exact account opening."""

    CREATED = "CREATED"
    EXISTING = "EXISTING"


@dataclass(frozen=True, slots=True)
class ReplayedPaperAccount:
    """One complete account projection from a single database snapshot."""

    execution_scope: str
    owner_generation: int
    opening_payload_sha256: str
    account: PaperAccount
    batches: tuple[PaperAccountBatchManifest, ...]


@dataclass(frozen=True, slots=True)
class ProvisionedPaperAccount:
    """A provision result returned only after the transaction commits."""

    disposition: ProvisionDisposition
    current: ReplayedPaperAccount

    @property
    def is_created(self) -> bool:
        """Return whether this call inserted the immutable opening."""
        return self.disposition is ProvisionDisposition.CREATED

    @property
    def account(self) -> PaperAccount:
        """Return the current fully replayed account projection."""
        return self.current.account

    @property
    def execution_scope(self) -> str:
        """Return the durable execution scope."""
        return self.current.execution_scope

    @property
    def owner_generation(self) -> int:
        """Return the immutable provisioning generation."""
        return self.current.owner_generation


@dataclass(frozen=True, slots=True)
class _AccountStreamRow:
    account_key: str
    execution_scope: str
    owner_generation: int
    collateral_asset: str
    account_version: int
    account_state: str
    opening_version: int
    opening_payload: object
    opening_payload_sha256: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class _BatchRow:
    manifest: PaperAccountBatchManifest
    opening_version: int
    opening_payload_sha256: str
    submission_event_type: str
    submission_event_payload_sha256: str
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class _SettlementRow:
    account_key: str
    account_version: int
    client_order_id: str
    fill_ordinal: int
    batch_first_account_version: int
    batch_submission_position_version: int
    batch_fill_count: int
    collateral_asset: str
    position_key: str
    position_version: int
    event_id: str
    trade_id: str
    event_type: str
    event_payload_sha256: str
    symbol: str
    base_asset: str
    quote_asset: str
    instrument_version: int
    settlement_version: int
    settlement_payload: object
    settlement_payload_sha256: str
    recorded_at: datetime


def _checked_input_text(value: object, field: str, maximum: int) -> str:
    if type(value) is not str:
        raise PaperAccountInputError(f"{field} must be text")
    if not value or value != value.strip():
        raise PaperAccountInputError(f"{field} must be non-empty and trimmed")
    if len(value) > maximum:
        raise PaperAccountInputError(f"{field} exceeds its storage limit")
    if "\x00" in value or any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        raise PaperAccountInputError(f"{field} is not representable in PostgreSQL")
    return value


def _checked_input_generation(value: object) -> int:
    if type(value) is not int or value < 1 or value > _BIGINT_MAX:
        raise PaperAccountInputError(
            "owner_generation is outside durable storage bounds"
        )
    return value


def _stored_text(value: object, field: str, maximum: int) -> str:
    try:
        return _checked_input_text(value, field, maximum)
    except PaperAccountInputError as exc:
        raise PaperAccountReplayError(f"stored {field} is invalid") from exc


def _stored_row(value: object, length: int, source: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise PaperAccountReplayError(f"stored {source} row has an unknown shape")
    return tuple(value)


def _stored_integer(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum or value > _BIGINT_MAX:
        raise PaperAccountReplayError(f"stored {field} is invalid")
    return value


def _stored_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PaperAccountReplayError(f"stored {field} must be timezone-aware")
    try:
        if value.utcoffset() is None:
            raise PaperAccountReplayError(f"stored {field} must be timezone-aware")
        normalized = value.astimezone(timezone.utc)
    except Exception as exc:
        if isinstance(exc, PaperAccountReplayError):
            raise
        raise PaperAccountReplayError(
            f"stored {field} has an invalid timezone"
        ) from exc
    return normalized


def _stored_decimal(value: object, field: str) -> Decimal:
    if type(value) is not str or not value or value != value.strip():
        raise PaperAccountReplayError(f"stored {field} is not canonical Decimal text")
    try:
        decoded = Decimal(value)
    except InvalidOperation as exc:
        raise PaperAccountReplayError(f"stored {field} is not a Decimal") from exc
    if not decoded.is_finite() or str(decoded) != value:
        raise PaperAccountReplayError(f"stored {field} is not canonical Decimal text")
    return decoded


def _decimal_identity(value: Decimal) -> tuple[object, ...]:
    parts = value.as_tuple()
    return (parts.sign, parts.digits, parts.exponent)


def _balance_identity(value: object) -> tuple[object, ...]:
    return (
        value.asset,
        _decimal_identity(value.available),
        _decimal_identity(value.reserved),
    )


def _reservation_identity(value: object) -> tuple[object, ...]:
    return (value.position_key, _decimal_identity(value.amount))


def _posting_identity(value: PaperAccountPosting) -> tuple[object, ...]:
    return (value.asset, value.bucket.value, _decimal_identity(value.amount))


def _decode_stream_row(value: object) -> _AccountStreamRow:
    row = _stored_row(value, 11, "paper account stream")
    account_key = _stored_text(row[0], "account_key", _ACCOUNT_KEY_MAX_LENGTH)
    execution_scope = _stored_text(
        row[1], "execution_scope", _EXECUTION_SCOPE_MAX_LENGTH
    )
    owner_generation = _stored_integer(row[2], "owner_generation", minimum=1)
    collateral_asset = _stored_text(row[3], "collateral_asset", 64)
    account_version = _stored_integer(row[4], "account_version", minimum=0)
    account_state = _stored_text(row[5], "account_state", 16)
    opening_version = _stored_integer(row[6], "opening_version", minimum=1)
    opening_payload_sha256 = _stored_text(row[8], "opening_payload_sha256", 64)
    created_at = _stored_datetime(row[9], "account created_at")
    updated_at = _stored_datetime(row[10], "account updated_at")
    if updated_at < created_at:
        raise PaperAccountReplayError("stored account updated_at predates created_at")
    return _AccountStreamRow(
        account_key=account_key,
        execution_scope=execution_scope,
        owner_generation=owner_generation,
        collateral_asset=collateral_asset,
        account_version=account_version,
        account_state=account_state,
        opening_version=opening_version,
        opening_payload=row[7],
        opening_payload_sha256=opening_payload_sha256,
        created_at=created_at,
        updated_at=updated_at,
    )


def _decode_batch_row(value: object) -> _BatchRow:
    row = _stored_row(value, 21, "paper account batch")
    try:
        manifest = decode_paper_account_batch(
            execution_scope=row[2],
            account_key=row[0],
            owner_generation=row[3],
            position_key=row[6],
            client_order_id=row[1],
            instruction_payload_sha256=row[7],
            submission_event_id=row[8],
            submission_position_version=row[10],
            submission_observed_at=row[11],
            first_account_version=row[13],
            last_account_version=row[14],
            last_position_version=row[15],
            fill_count=row[16],
            batch_version=row[17],
            batch_payload=row[18],
            batch_payload_sha256=row[19],
        )
    except JournalQuarantineError as exc:
        raise PaperAccountReplayError("stored account batch cannot be decoded") from exc
    return _BatchRow(
        manifest=manifest,
        opening_version=_stored_integer(row[4], "batch opening_version", minimum=1),
        opening_payload_sha256=_stored_text(row[5], "batch opening_payload_sha256", 64),
        submission_event_type=_stored_text(row[9], "batch submission_event_type", 32),
        submission_event_payload_sha256=_stored_text(
            row[12], "batch submission_event_payload_sha256", 64
        ),
        recorded_at=_stored_datetime(row[20], "batch recorded_at"),
    )


def _decode_settlement_row(value: object) -> _SettlementRow:
    row = _stored_row(value, 22, "paper account settlement")
    return _SettlementRow(
        account_key=_stored_text(row[0], "settlement account_key", 255),
        account_version=_stored_integer(row[1], "account_version", minimum=1),
        client_order_id=_stored_text(row[2], "client_order_id", 255),
        fill_ordinal=_stored_integer(row[3], "fill_ordinal", minimum=1),
        batch_first_account_version=_stored_integer(
            row[4], "batch_first_account_version", minimum=1
        ),
        batch_submission_position_version=_stored_integer(
            row[5], "batch_submission_position_version", minimum=1
        ),
        batch_fill_count=_stored_integer(row[6], "batch_fill_count", minimum=1),
        collateral_asset=_stored_text(row[7], "collateral_asset", 64),
        position_key=_stored_text(row[8], "position_key", 255),
        position_version=_stored_integer(row[9], "position_version", minimum=1),
        event_id=_stored_text(row[10], "event_id", 255),
        trade_id=_stored_text(row[11], "trade_id", 255),
        event_type=_stored_text(row[12], "event_type", 32),
        event_payload_sha256=_stored_text(row[13], "event_payload_sha256", 64),
        symbol=_stored_text(row[14], "symbol", 64),
        base_asset=_stored_text(row[15], "base_asset", 64),
        quote_asset=_stored_text(row[16], "quote_asset", 64),
        instrument_version=_stored_integer(row[17], "instrument_version", minimum=1),
        settlement_version=_stored_integer(row[18], "settlement_version", minimum=1),
        settlement_payload=row[19],
        settlement_payload_sha256=_stored_text(
            row[20], "settlement_payload_sha256", 64
        ),
        recorded_at=_stored_datetime(row[21], "settlement recorded_at"),
    )


def _fetch_account_stream(
    cursor: object,
    *,
    account_key: str,
    lock: bool,
) -> _AccountStreamRow | None:
    cursor.execute(
        _SELECT_ACCOUNT_STREAM_FOR_UPDATE_SQL if lock else _SELECT_ACCOUNT_STREAM_SQL,
        (account_key,),
    )
    raw = cursor.fetchone()
    return None if raw is None else _decode_stream_row(raw)


def _position_replays(
    cursor: object,
    *,
    execution_scope: str,
    manifests: tuple[PaperAccountBatchManifest, ...],
) -> dict[str, object]:
    replays = {}
    for position_key in sorted({manifest.position_key for manifest in manifests}):
        try:
            replays[position_key] = _replay_stream(
                cursor,
                execution_scope=execution_scope,
                position_key=position_key,
                lock=False,
            )
        except JournalRepositoryError as exc:
            raise PaperAccountReplayError(
                "referenced position stream cannot be replayed"
            ) from exc
    return replays


def _validate_batch_history(
    rows: tuple[_BatchRow, ...],
    *,
    stream: _AccountStreamRow,
    position_replays: dict[str, object],
) -> dict[tuple[str, int], object]:
    manifests_by_client = {}
    events_by_ref = {}
    for batch in rows:
        manifest = batch.manifest
        if (
            manifest.account_key != stream.account_key
            or manifest.execution_scope != stream.execution_scope
            or manifest.owner_generation != stream.owner_generation
            or batch.opening_version != stream.opening_version
            or batch.opening_payload_sha256 != stream.opening_payload_sha256
        ):
            raise PaperAccountReplayError(
                "stored batch conflicts with its immutable account opening"
            )
        if batch.submission_event_type != "SUBMISSION_ACKNOWLEDGED":
            raise PaperAccountReplayError("stored batch submission type is invalid")
        if (
            batch.submission_event_payload_sha256
            != manifest.submission_event_payload_sha256
        ):
            raise PaperAccountReplayError(
                "stored batch ACK hash conflicts with payload"
            )
        if manifest.client_order_id in manifests_by_client:
            raise PaperAccountReplayError("stored account repeats a batch order")
        manifests_by_client[manifest.client_order_id] = manifest

    for position_key, replay in position_replays.items():
        expected_clients = {
            manifest.client_order_id
            for manifest in manifests_by_client.values()
            if manifest.position_key == position_key
        }
        actual_clients = set(replay.orders_by_client)
        if actual_clients != expected_clients:
            raise PaperAccountReplayError(
                "referenced position history is not wholly owned by account batches"
            )

    for client_order_id, manifest in manifests_by_client.items():
        replay = position_replays.get(manifest.position_key)
        if replay is None:
            raise PaperAccountReplayError("stored batch references no position stream")
        stored_order = replay.orders_by_client.get(client_order_id)
        if stored_order is None:
            raise PaperAccountReplayError("stored batch references no durable order")
        if (
            stored_order.encoded.instruction_payload_sha256
            != manifest.instruction_payload_sha256
        ):
            raise PaperAccountReplayError("stored batch instruction hash conflicts")
        order = next(
            (
                candidate
                for candidate in replay.projection.orders
                if candidate.instruction.order_intent.client_order_id == client_order_id
            ),
            None,
        )
        if order is None or order.lifecycle.state is not OrderLifecycleState.FILLED:
            raise PaperAccountReplayError("stored batch order is not terminally filled")
        if len(order.events) != len(manifest.fills) + 1:
            raise PaperAccountReplayError("stored batch event count is incomplete")
        acknowledgement = order.events[0]
        if type(acknowledgement.event) is not SubmissionAcknowledged:
            raise PaperAccountReplayError("stored batch does not start with an ACK")
        try:
            encoded_ack = encode_order_lifecycle_event(acknowledgement.event)
        except JournalEncodeError as exc:
            raise PaperAccountReplayError(
                "stored batch ACK is not durably representable"
            ) from exc
        if (
            acknowledgement.event_id != manifest.submission_event_id
            or acknowledgement.position_version != manifest.submission_position_version
            or acknowledgement.event.observed_at != manifest.submission_observed_at
            or encoded_ack.event_payload_sha256
            != manifest.submission_event_payload_sha256
        ):
            raise PaperAccountReplayError(
                "stored batch ACK conflicts with its manifest"
            )

        for expected, record in zip(manifest.fills, order.events[1:]):
            if type(record.event) is not ConfirmedFill:
                raise PaperAccountReplayError("stored batch suffix is not all fills")
            try:
                encoded_fill = encode_order_lifecycle_event(record.event)
            except JournalEncodeError as exc:
                raise PaperAccountReplayError(
                    "stored batch fill is not durably representable"
                ) from exc
            if (
                record.event_id != expected.event_id
                or record.position_version != expected.position_version
                or record.event.trade_id != expected.trade_id
                or encoded_fill.event_payload_sha256 != expected.event_payload_sha256
            ):
                raise PaperAccountReplayError(
                    "stored journal fill conflicts with its batch manifest"
                )
            ref = (manifest.position_key, record.position_version)
            if ref in events_by_ref:
                raise PaperAccountReplayError("stored account repeats a journal fill")
            events_by_ref[ref] = (stored_order.instruction, record)
    return events_by_ref


def _validate_settlement_membership(
    batches: tuple[_BatchRow, ...],
    settlements: tuple[_SettlementRow, ...],
) -> None:
    by_version = {row.account_version: row for row in settlements}
    if len(by_version) != len(settlements):
        raise PaperAccountReplayError("stored account repeats an account version")
    expected_versions = []
    for batch in batches:
        manifest = batch.manifest
        for ordinal, fill in enumerate(manifest.fills, start=1):
            row = by_version.get(fill.account_version)
            if row is None:
                raise PaperAccountReplayError("stored batch is missing a settlement")
            if (
                row.account_key != manifest.account_key
                or row.client_order_id != manifest.client_order_id
                or row.fill_ordinal != ordinal
                or row.batch_first_account_version != manifest.fills[0].account_version
                or row.batch_submission_position_version
                != manifest.submission_position_version
                or row.batch_fill_count != len(manifest.fills)
                or row.position_key != fill.position_key
                or row.position_version != fill.position_version
                or row.event_id != fill.event_id
                or row.trade_id != fill.trade_id
                or row.event_type != "CONFIRMED_FILL"
                or row.event_payload_sha256 != fill.event_payload_sha256
                or row.settlement_payload_sha256
                != fill.account_settlement_payload_sha256
            ):
                raise PaperAccountReplayError(
                    "stored settlement conflicts with batch membership"
                )
            expected_versions.append(fill.account_version)
    if set(expected_versions) != set(by_version):
        raise PaperAccountReplayError("stored account contains an orphan settlement")


def _decode_projection_rows(
    cursor: object,
    account_key: str,
) -> tuple[tuple[object, ...], tuple[object, ...], dict[int, tuple[object, ...]]]:
    cursor.execute(_SELECT_ACCOUNT_BALANCES_SQL, (account_key,))
    balances = []
    for raw in cursor.fetchall():
        row = _stored_row(raw, 4, "paper account balance")
        balances.append(
            (
                _stored_text(row[0], "balance asset", 64),
                _stored_decimal(row[1], "available_decimal"),
                _stored_decimal(row[2], "reserved_decimal"),
            )
        )
        _stored_datetime(row[3], "balance updated_at")

    cursor.execute(_SELECT_ACCOUNT_RESERVATIONS_SQL, (account_key,))
    reservations = []
    for raw in cursor.fetchall():
        row = _stored_row(raw, 4, "paper margin reservation")
        reservations.append(
            (
                _stored_text(row[0], "reservation execution_scope", 128),
                _stored_text(row[1], "reservation position_key", 255),
                _stored_decimal(row[2], "reservation amount_decimal"),
            )
        )
        _stored_datetime(row[3], "reservation updated_at")

    cursor.execute(_SELECT_ACCOUNT_POSTINGS_SQL, (account_key,))
    postings: dict[int, list[object]] = {}
    for raw in cursor.fetchall():
        row = _stored_row(raw, 5, "paper account posting")
        version = _stored_integer(row[0], "posting account_version", minimum=1)
        ordinal = _stored_integer(row[1], "posting ordinal", minimum=1)
        asset = _stored_text(row[2], "posting asset", 64)
        bucket_text = _stored_text(row[3], "posting bucket", 32)
        try:
            bucket = PaperAccountPostingBucket(bucket_text)
            posting = PaperAccountPosting(
                asset=asset,
                bucket=bucket,
                amount=_stored_decimal(row[4], "posting amount_decimal"),
            )
        except (TypeError, ValueError) as exc:
            raise PaperAccountReplayError("stored posting violates the domain") from exc
        postings.setdefault(version, []).append((ordinal, posting))
    return (
        tuple(balances),
        tuple(reservations),
        {version: tuple(values) for version, values in postings.items()},
    )


def _compare_materialized_projection(
    *,
    stream: _AccountStreamRow,
    account: PaperAccount,
    balances: tuple[object, ...],
    reservations: tuple[object, ...],
) -> None:
    if stream.account_version != len(account.records):
        raise PaperAccountReplayError("account stream version conflicts with replay")
    if stream.account_state != account.state.value:
        raise PaperAccountReplayError("account stream state conflicts with replay")

    stored_balances = tuple(
        sorted(
            (
                asset,
                _decimal_identity(available),
                _decimal_identity(reserved),
            )
            for asset, available, reserved in balances
        )
    )
    if len({value[0] for value in stored_balances}) != len(stored_balances):
        raise PaperAccountReplayError("stored account repeats a balance asset")
    expected_balances = tuple(sorted(map(_balance_identity, account.balances)))
    if stored_balances != expected_balances:
        raise PaperAccountReplayError("materialized balances conflict with replay")

    stored_reservations = tuple(
        sorted(
            (position_key, _decimal_identity(amount))
            for scope, position_key, amount in reservations
            if scope == stream.execution_scope
        )
    )
    if any(scope != stream.execution_scope for scope, _, _ in reservations):
        raise PaperAccountReplayError("materialized reservation scope conflicts")
    if len({value[0] for value in stored_reservations}) != len(stored_reservations):
        raise PaperAccountReplayError("stored account repeats a reservation")
    expected_reservations = tuple(
        sorted(map(_reservation_identity, account.reservations))
    )
    if stored_reservations != expected_reservations:
        raise PaperAccountReplayError("materialized reservations conflict with replay")


def _replay_account_locked(
    cursor: object,
    *,
    execution_scope: str,
    account_key: str,
    lock: bool,
) -> ReplayedPaperAccount:
    stream = _fetch_account_stream(cursor, account_key=account_key, lock=lock)
    if stream is None:
        raise PaperAccountNotFoundError(f"paper account {account_key!r} does not exist")
    if stream.execution_scope != execution_scope:
        raise PaperAccountConflictError(
            PaperAccountConflictKind.EXECUTION_SCOPE,
            "paper account belongs to another execution scope",
        )
    try:
        account = decode_paper_account_opening(
            execution_scope=stream.execution_scope,
            account_key=stream.account_key,
            owner_generation=stream.owner_generation,
            collateral_asset=stream.collateral_asset,
            opening_version=stream.opening_version,
            opening_payload=stream.opening_payload,
            opening_payload_sha256=stream.opening_payload_sha256,
        )
    except JournalQuarantineError as exc:
        raise PaperAccountReplayError(
            "stored account opening cannot be decoded"
        ) from exc

    cursor.execute(_SELECT_ACCOUNT_BATCHES_SQL, (account_key,))
    batches = tuple(_decode_batch_row(row) for row in cursor.fetchall())
    if len({batch.manifest.client_order_id for batch in batches}) != len(batches):
        raise PaperAccountReplayError("stored account repeats a batch identity")

    cursor.execute(_SELECT_ACCOUNT_SETTLEMENTS_SQL, (account_key,))
    settlements = tuple(_decode_settlement_row(row) for row in cursor.fetchall())
    settlements = tuple(sorted(settlements, key=lambda row: row.account_version))
    expected_versions = tuple(range(1, len(settlements) + 1))
    if tuple(row.account_version for row in settlements) != expected_versions:
        raise PaperAccountReplayError(
            "stored account versions are not the exact contiguous prefix"
        )

    manifests = tuple(batch.manifest for batch in batches)
    position_replays = _position_replays(
        cursor,
        execution_scope=execution_scope,
        manifests=manifests,
    )
    events_by_ref = _validate_batch_history(
        batches,
        stream=stream,
        position_replays=position_replays,
    )
    _validate_settlement_membership(batches, settlements)
    balances, reservations, postings_by_version = _decode_projection_rows(
        cursor, account_key
    )

    prior_by_position: dict[str, PaperSettlementCheckpoint] = {}
    for row in settlements:
        event_ref = (row.position_key, row.position_version)
        durable = events_by_ref.get(event_ref)
        if durable is None:
            raise PaperAccountReplayError(
                "stored settlement has no exact confirmed journal fill"
            )
        instruction, event_record = durable
        fill = event_record.event
        if type(fill) is not ConfirmedFill:
            raise PaperAccountReplayError("stored settlement event is not a fill")
        try:
            record = PaperFillRecord(
                position_version=row.position_version,
                event_id=row.event_id,
                position_fill=PositionFill(instruction, fill),
            )
            instrument = PaperLinearInstrument(
                symbol=row.symbol,
                base_asset=row.base_asset,
                quote_asset=row.quote_asset,
            )
            settlement = settle_paper_fill(
                instrument,
                prior_by_position.get(row.position_key),
                record,
            )
            admission = decode_paper_account_settlement(
                account,
                settlement,
                account_key=row.account_key,
                collateral_asset=row.collateral_asset,
                account_version=row.account_version,
                position_key=row.position_key,
                position_version=row.position_version,
                client_order_id=row.client_order_id,
                event_id=row.event_id,
                trade_id=row.trade_id,
                symbol=row.symbol,
                base_asset=row.base_asset,
                quote_asset=row.quote_asset,
                instrument_version=row.instrument_version,
                settlement_version=row.settlement_version,
                settlement_payload=row.settlement_payload,
                settlement_payload_sha256=row.settlement_payload_sha256,
            )
        except (
            InvalidPaperSettlement,
            JournalEncodeError,
            JournalQuarantineError,
            TypeError,
            ValueError,
        ) as exc:
            raise PaperAccountReplayError(
                "stored settlement cannot be causally replayed"
            ) from exc
        if admission.disposition is not PaperAccountAdmissionDisposition.APPLIED:
            raise PaperAccountReplayError("durable settlement is not newly applied")

        stored_postings = postings_by_version.pop(row.account_version, ())
        if tuple(ordinal for ordinal, _ in stored_postings) != tuple(
            range(1, len(stored_postings) + 1)
        ):
            raise PaperAccountReplayError("stored posting ordinals are not contiguous")
        if tuple(_posting_identity(posting) for _, posting in stored_postings) != tuple(
            map(_posting_identity, admission.postings)
        ):
            raise PaperAccountReplayError("stored postings conflict with replay")
        account = admission.after
        prior_by_position[row.position_key] = settlement.after

    if postings_by_version:
        raise PaperAccountReplayError("stored account contains orphan postings")
    _compare_materialized_projection(
        stream=stream,
        account=account,
        balances=balances,
        reservations=reservations,
    )
    ordered_batches = tuple(
        batch.manifest
        for batch in sorted(
            batches,
            key=lambda item: (
                item.manifest.fills[0].account_version,
                item.manifest.client_order_id,
            ),
        )
    )
    return ReplayedPaperAccount(
        execution_scope=stream.execution_scope,
        owner_generation=stream.owner_generation,
        opening_payload_sha256=stream.opening_payload_sha256,
        account=account,
        batches=ordered_batches,
    )


class PostgresPaperAccountJournal:
    """Provision and strictly replay dormant paper-account state."""

    def __init__(self, connection_factory: Callable[[], object]) -> None:
        self._journal_boundary = PostgresOrderPositionJournal(connection_factory)

    def _connection(self) -> object:
        try:
            return self._journal_boundary._connection()
        except JournalRepositoryError as exc:
            raise PaperAccountStorageError(
                "could not open a paper-account connection"
            ) from exc

    def provision_account(
        self,
        *,
        execution_scope: str,
        owner_generation: int,
        account: PaperAccount,
    ) -> ProvisionedPaperAccount:
        """Create one immutable opening or return its exact durable account."""
        scope = _checked_input_text(
            execution_scope, "execution_scope", _EXECUTION_SCOPE_MAX_LENGTH
        )
        generation = _checked_input_generation(owner_generation)
        try:
            encoded = encode_paper_account_opening(scope, generation, account)
        except JournalEncodeError as exc:
            raise PaperAccountInputError(
                "paper account opening is not representable"
            ) from exc

        connection = self._connection()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_WRITE_TRANSACTION_SQL)
                    cursor.execute(
                        _INSERT_ACCOUNT_STREAM_SQL,
                        (
                            encoded.account_key,
                            encoded.execution_scope,
                            encoded.owner_generation,
                            encoded.collateral_asset,
                            encoded.opening_version,
                            encoded.opening_payload,
                            encoded.opening_payload_sha256,
                        ),
                    )
                    inserted = cursor.fetchone()
                    is_created = inserted is not None
                    if is_created:
                        if _stored_row(inserted, 1, "inserted paper account")[0] != (
                            encoded.account_key
                        ):
                            raise PaperAccountStorageError(
                                "PostgreSQL returned another paper account"
                            )
                        for balance in account.opening_balances:
                            cursor.execute(
                                _INSERT_ACCOUNT_BALANCE_SQL,
                                (
                                    encoded.account_key,
                                    balance.asset,
                                    str(balance.available),
                                    str(balance.reserved),
                                ),
                            )
                    else:
                        stored = _fetch_account_stream(
                            cursor,
                            account_key=encoded.account_key,
                            lock=True,
                        )
                        if stored is None:
                            raise PaperAccountStorageError(
                                "provisioning conflict disappeared"
                            )
                        if stored.execution_scope != scope:
                            raise PaperAccountConflictError(
                                PaperAccountConflictKind.EXECUTION_SCOPE,
                                "account key belongs to another execution scope",
                            )
                        if stored.owner_generation != generation:
                            raise PaperAccountConflictError(
                                PaperAccountConflictKind.OWNER_GENERATION,
                                "account key belongs to another owner generation",
                            )

                    current = _replay_account_locked(
                        cursor,
                        execution_scope=scope,
                        account_key=encoded.account_key,
                        lock=True,
                    )
                    if current.opening_payload_sha256 != (
                        encoded.opening_payload_sha256
                    ):
                        raise PaperAccountConflictError(
                            PaperAccountConflictKind.OPENING_IDENTITY,
                            "account key is bound to another opening envelope",
                        )
                    result = ProvisionedPaperAccount(
                        disposition=(
                            ProvisionDisposition.CREATED
                            if is_created
                            else ProvisionDisposition.EXISTING
                        ),
                        current=current,
                    )
            except PaperAccountJournalError:
                raise
            except psycopg2.Error as exc:
                raise PaperAccountStorageError(
                    "paper account provisioning failed before commit"
                ) from exc
            except Exception as exc:
                raise PaperAccountStorageError(
                    "paper account provisioning failed before commit"
                ) from exc

            try:
                connection.commit()
            except Exception as exc:
                raise PaperAccountCommitUnknown(
                    scope, encoded.account_key, generation
                ) from exc
            return result
        except Exception:
            self._journal_boundary._rollback(connection)
            raise
        finally:
            self._journal_boundary._close(connection)

    def replay_account(
        self,
        *,
        execution_scope: str,
        account_key: str,
    ) -> ReplayedPaperAccount:
        """Replay one account from a single read-only repeatable-read snapshot."""
        scope = _checked_input_text(
            execution_scope, "execution_scope", _EXECUTION_SCOPE_MAX_LENGTH
        )
        key = _checked_input_text(account_key, "account_key", _ACCOUNT_KEY_MAX_LENGTH)
        return self._read(
            "paper account replay",
            lambda cursor: _replay_account_locked(
                cursor,
                execution_scope=scope,
                account_key=key,
                lock=False,
            ),
        )

    def list_accounts(
        self,
        *,
        execution_scope: str,
    ) -> tuple[ReplayedPaperAccount, ...]:
        """Replay every scoped account all-or-nothing in one stable snapshot."""
        scope = _checked_input_text(
            execution_scope, "execution_scope", _EXECUTION_SCOPE_MAX_LENGTH
        )

        def list_in_snapshot(cursor: object) -> tuple[ReplayedPaperAccount, ...]:
            cursor.execute(_SELECT_SCOPE_ACCOUNT_KEYS_SQL, (scope,))
            keys = tuple(
                _stored_text(
                    _stored_row(row, 1, "paper account key")[0],
                    "account_key",
                    _ACCOUNT_KEY_MAX_LENGTH,
                )
                for row in cursor.fetchall()
            )
            if len(keys) != len(set(keys)):
                raise PaperAccountReplayError("scope repeats an account key")
            return tuple(
                _replay_account_locked(
                    cursor,
                    execution_scope=scope,
                    account_key=key,
                    lock=False,
                )
                for key in sorted(keys)
            )

        return self._read("paper account inventory", list_in_snapshot)

    def _read(self, operation: str, callback: Callable[[object], object]) -> object:
        connection = self._connection()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_TRANSACTION_SQL)
                    result = callback(cursor)
            except PaperAccountJournalError:
                raise
            except psycopg2.Error as exc:
                raise PaperAccountStorageError(f"{operation} query failed") from exc
            except Exception as exc:
                raise PaperAccountStorageError(f"{operation} failed") from exc
            try:
                connection.rollback()
            except Exception as exc:
                raise PaperAccountStorageError(
                    f"{operation} transaction could not finish"
                ) from exc
            return result
        except Exception:
            self._journal_boundary._rollback(connection)
            raise
        finally:
            self._journal_boundary._close(connection)


__all__ = [
    "PaperAccountCommitUnknown",
    "PaperAccountConflictError",
    "PaperAccountConflictKind",
    "PaperAccountInputError",
    "PaperAccountJournalError",
    "PaperAccountNotFoundError",
    "PaperAccountReplayError",
    "PaperAccountStorageError",
    "PostgresPaperAccountJournal",
    "ProvisionDisposition",
    "ProvisionedPaperAccount",
    "ReplayedPaperAccount",
]
