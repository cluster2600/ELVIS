"""Dormant PostgreSQL adapter for one durable trajectory-B fresh opening.

The database capabilities own the target-local nonce fence and all writes.
This adapter owns strict row decoding, replay-before-authority ordering, exact
receipt bytes, and conservative resolution of a lost commit acknowledgement.
It never grants runtime or trading authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

from trading.application.fresh_opening import (
    CanonicalFreshOpeningDocument,
    FreshOpeningPreparationDisposition,
    ProspectiveFreshOpeningCandidate,
)
from trading.application.fresh_opening_provisioning import (
    FreshOpeningCurrentAuthorityPort,
    FreshOpeningPhysicalTarget,
    FreshOpeningProvisioningDisposition,
    FreshOpeningProvisioningReceipt,
    FreshOpeningProvisioningRequest,
    FreshOpeningProvisioningResult,
)

_WRITE_TRANSACTION_SQL = "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
_READ_TRANSACTION_SQL = "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
_SET_LOCK_TIMEOUT_SQL = "SET LOCAL lock_timeout = '1s'"
_SET_TIME_ZONE_SQL = "SET LOCAL TIME ZONE 'UTC'"

_ACQUIRE_FENCE_SQL = """
SELECT *
FROM np.acquire_paper_fresh_opening_fence(%s, %s, %s, %s)
"""
_COMMIT_OPENING_SQL = """
SELECT *
FROM np.commit_paper_fresh_opening(
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s
)
"""
_READ_OPENING_SQL = """
SELECT *
FROM np.read_paper_fresh_opening(%s, %s, %s)
"""

_FENCE_ROW_LENGTH = 34
_COMMIT_ROW_LENGTH = 6
_PRESENT_RESOLUTIONS = frozenset({"EXACT_REPLAY", "NONCE_CONFLICT", "TARGET_CONFLICT"})
_ADMISSION_CONFLICT_RESOLUTION = "ADMISSION_CONFLICT"
_APPROVAL_EXPIRED_SQLSTATE = "PT004"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_POSTGRES_SYSTEM_IDENTIFIER_MAX = (1 << 64) - 1


class PostgresFreshOpeningProvisioningStorageError(RuntimeError):
    """Raised when the target outcome is known not to have committed."""


@dataclass(frozen=True, slots=True)
class _StoredOpening:
    authority_evaluated_at: datetime
    committed_at: datetime
    intent_payload: str
    intent_sha256: str
    approval_payload: str
    approval_sha256: str
    trust_policy_payload: str
    trust_policy_sha256: str
    candidate_payload: str
    candidate_sha256: str
    opening_payload: str
    opening_sha256: str
    opening_receipt_payload: str
    opening_receipt_sha256: str
    provisioning_receipt_payload: str
    provisioning_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _FenceEvidence:
    resolution: str
    evaluated_at: datetime
    database_name: str
    system_identifier: int
    control_plane_role: str
    opening_anchor_role: str
    migration_version: int
    migration_name: str
    migration_checksum: str
    terminal_catalog_sha256: str
    pin_authority_record_sha256: str
    deployment_incarnation_id: str
    database_incarnation_id: str | None
    runtime_mode: str
    runtime_generation: int
    authority_transition_sequence: int
    writer_fence: int
    v2_empty: bool
    stored: _StoredOpening | None


@dataclass(frozen=True, slots=True)
class _ReceiptDocuments:
    opening: CanonicalFreshOpeningDocument
    provisioning: CanonicalFreshOpeningDocument


def _canonical_json(value: dict[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _document(payload: str, digest: str) -> CanonicalFreshOpeningDocument:
    document = CanonicalFreshOpeningDocument(payload, digest)
    if _sha256(payload) != digest:
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL returned an inconsistent receipt digest"
        )
    return document


def _utc_datetime(value: object, label: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None:
        raise PostgresFreshOpeningProvisioningStorageError(
            f"PostgreSQL returned invalid {label}"
        )
    normalized = value.astimezone(timezone.utc)
    if normalized.utcoffset() is None:
        raise PostgresFreshOpeningProvisioningStorageError(
            f"PostgreSQL returned invalid {label}"
        )
    return normalized


def _canonical_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _text(value: object, label: str, maximum: int = 255) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > maximum
        or not value.isascii()
        or any(not 0x21 <= ord(character) <= 0x7E for character in value)
    ):
        raise PostgresFreshOpeningProvisioningStorageError(
            f"PostgreSQL returned invalid {label}"
        )
    return value


def _digest(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise PostgresFreshOpeningProvisioningStorageError(
            f"PostgreSQL returned invalid {label}"
        )
    return value


def _optional_digest(value: object, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _integer(value: object, label: str, *, maximum: int = (1 << 63) - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise PostgresFreshOpeningProvisioningStorageError(
            f"PostgreSQL returned invalid {label}"
        )
    return value


def _system_identifier(value: object) -> int:
    if type(value) is int:
        result = value
    elif type(value) is Decimal and value == value.to_integral_value():
        result = int(value)
    else:
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL returned invalid system identifier"
        )
    if not 1 <= result <= _POSTGRES_SYSTEM_IDENTIFIER_MAX:
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL returned invalid system identifier"
        )
    return result


def _row(value: object, length: int, label: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise PostgresFreshOpeningProvisioningStorageError(
            f"PostgreSQL returned an invalid {label} row"
        )
    return tuple(value)


def _postgres_sqlstate(error: BaseException) -> str | None:
    try:
        value = getattr(error, "pgcode", None)
    except Exception:
        return None
    return value if type(value) is str else None


def _stored_opening(values: tuple[object, ...]) -> _StoredOpening | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL returned a partial fresh-opening record"
        )
    authority_evaluated_at = _utc_datetime(
        values[0], "fresh-opening authority evaluation time"
    )
    committed_at = _utc_datetime(values[1], "fresh-opening commit time")
    if authority_evaluated_at > committed_at:
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL returned invalid fresh-opening authority time"
        )
    payloads = []
    digests = []
    for index in range(2, len(values), 2):
        payload = values[index]
        digest = values[index + 1]
        if type(payload) is not str or not payload or not payload.isascii():
            raise PostgresFreshOpeningProvisioningStorageError(
                "PostgreSQL returned invalid fresh-opening bytes"
            )
        payloads.append(payload)
        digests.append(_digest(digest, "fresh-opening digest"))
    return _StoredOpening(
        authority_evaluated_at,
        committed_at,
        payloads[0],
        digests[0],
        payloads[1],
        digests[1],
        payloads[2],
        digests[2],
        payloads[3],
        digests[3],
        payloads[4],
        digests[4],
        payloads[5],
        digests[5],
        payloads[6],
        digests[6],
    )


def _fence_evidence(raw: object) -> _FenceEvidence:
    row = _row(raw, _FENCE_ROW_LENGTH, "fresh-opening fence")
    resolution = row[0]
    if resolution not in {
        "ABSENT",
        _ADMISSION_CONFLICT_RESOLUTION,
        *_PRESENT_RESOLUTIONS,
    }:
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL returned an invalid fresh-opening resolution"
        )
    if type(row[17]) is not bool:
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL returned invalid V2 emptiness evidence"
        )
    stored = _stored_opening(row[18:])
    if (resolution in _PRESENT_RESOLUTIONS) != (stored is not None):
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL fresh-opening resolution conflicts with durable evidence"
        )
    if stored is not None and row[17]:
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL durable fresh-opening evidence cannot be empty"
        )
    if (row[12] is not None) != (stored is not None):
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL database incarnation conflicts with durable evidence"
        )
    return _FenceEvidence(
        resolution=resolution,
        evaluated_at=_utc_datetime(row[1], "database evaluation time"),
        database_name=_text(row[2], "database name", 63),
        system_identifier=_system_identifier(row[3]),
        control_plane_role=_text(row[4], "control-plane role", 63),
        opening_anchor_role=_text(row[5], "opening anchor role", 63),
        migration_version=_integer(row[6], "migration version"),
        migration_name=_text(row[7], "migration name", 255),
        migration_checksum=_digest(row[8], "migration checksum"),
        terminal_catalog_sha256=_digest(row[9], "terminal catalog digest"),
        pin_authority_record_sha256=_digest(row[10], "pin authority digest"),
        deployment_incarnation_id=_text(row[11], "deployment incarnation"),
        database_incarnation_id=_optional_digest(row[12], "database incarnation"),
        runtime_mode=_text(row[13], "runtime mode", 16),
        runtime_generation=_integer(row[14], "runtime generation"),
        authority_transition_sequence=_integer(
            row[15], "authority transition sequence"
        ),
        writer_fence=_integer(row[16], "writer fence"),
        v2_empty=row[17],
        stored=stored,
    )


def _physical_target_is_exact(
    evidence: _FenceEvidence,
    target: FreshOpeningPhysicalTarget,
) -> bool:
    snapshot_exact = (
        evidence.database_name == target.expected_database
        and evidence.system_identifier == target.expected_system_identifier
        and evidence.control_plane_role == target.control_plane_role
        and evidence.opening_anchor_role == target.opening_anchor_role
        and evidence.migration_version == 7
        and evidence.terminal_catalog_sha256 == target.terminal_catalog_sha256
        and evidence.pin_authority_record_sha256 == target.pin_authority_record_sha256
        and evidence.deployment_incarnation_id == target.deployment_incarnation_id
        and evidence.runtime_mode == "LEGACY"
        and evidence.runtime_generation == 0
        and evidence.authority_transition_sequence == 0
        and evidence.writer_fence == 0
    )
    if not snapshot_exact:
        return False
    if evidence.stored is None:
        return evidence.database_incarnation_id is None
    return evidence.database_incarnation_id == _database_incarnation_id(
        target,
        evidence,
    )


def _database_incarnation_id(
    target: FreshOpeningPhysicalTarget,
    evidence: _FenceEvidence,
) -> str:
    core = _canonical_json(
        {
            "database_name": evidence.database_name,
            "deployment_incarnation_id": target.deployment_incarnation_id,
            "migration_checksum": evidence.migration_checksum,
            "migration_head": evidence.migration_version,
            "migration_name": evidence.migration_name,
            "control_plane_role": evidence.control_plane_role,
            "opening_anchor_role": evidence.opening_anchor_role,
            "system_identifier": str(evidence.system_identifier),
            "terminal_catalog_sha256": evidence.terminal_catalog_sha256,
        }
    )
    return _sha256(core)


def _receipt_documents(
    request: FreshOpeningProvisioningRequest,
    candidate: ProspectiveFreshOpeningCandidate,
    evidence: _FenceEvidence,
) -> _ReceiptDocuments:
    authority_evaluated_at = (
        evidence.evaluated_at
        if evidence.stored is None
        else evidence.stored.authority_evaluated_at
    )
    opening_payload = _canonical_json(
        {
            "account_key": request.intent.account_key,
            "collateral_asset": request.intent.collateral_asset,
            "execution_scope": request.intent.execution_scope,
            "opening_payload_sha256": candidate.opening.opening_payload_sha256,
            "opening_version": candidate.opening.opening_version,
            "owner_generation": request.intent.owner_generation,
            "result": "CREATED",
            "schema_version": 1,
        }
    )
    opening = CanonicalFreshOpeningDocument(
        opening_payload,
        _sha256(opening_payload),
    )
    provisioning_payload = _canonical_json(
        {
            "approval_sha256": candidate.approval_document.sha256,
            "authority_evaluated_at": _canonical_timestamp(authority_evaluated_at),
            "authority_transition_sequence": evidence.authority_transition_sequence,
            "candidate_sha256": candidate.candidate_document.sha256,
            "database_incarnation_id": _database_incarnation_id(
                request.target, evidence
            ),
            "database_name": evidence.database_name,
            "deployment_incarnation_id": (request.target.deployment_incarnation_id),
            "intent_sha256": candidate.intent_document.sha256,
            "migration_checksum": evidence.migration_checksum,
            "migration_head": evidence.migration_version,
            "migration_name": evidence.migration_name,
            "opening_payload_sha256": candidate.opening.opening_payload_sha256,
            "opening_receipt_sha256": opening.sha256,
            "control_plane_role": evidence.control_plane_role,
            "opening_anchor_role": evidence.opening_anchor_role,
            "pin_authority_record_sha256": (request.target.pin_authority_record_sha256),
            "runtime_activation_authorized": False,
            "runtime_generation": evidence.runtime_generation,
            "runtime_mode": evidence.runtime_mode,
            "schema_version": 1,
            "stale_on_return": True,
            "system_identifier": str(evidence.system_identifier),
            "terminal_catalog_sha256": evidence.terminal_catalog_sha256,
            "trading_authorized": False,
            "trust_policy_sha256": candidate.trust_policy_document.sha256,
            "writer_fence": evidence.writer_fence,
        }
    )
    provisioning = CanonicalFreshOpeningDocument(
        provisioning_payload,
        _sha256(provisioning_payload),
    )
    return _ReceiptDocuments(opening=opening, provisioning=provisioning)


def _stored_candidate_is_exact(
    stored: _StoredOpening,
    candidate: ProspectiveFreshOpeningCandidate,
) -> bool:
    return (
        stored.intent_payload == candidate.intent_document.payload
        and stored.intent_sha256 == candidate.intent_document.sha256
        and stored.approval_payload == candidate.approval_document.payload
        and stored.approval_sha256 == candidate.approval_document.sha256
        and stored.trust_policy_payload == candidate.trust_policy_document.payload
        and stored.trust_policy_sha256 == candidate.trust_policy_document.sha256
        and stored.candidate_payload == candidate.candidate_document.payload
        and stored.candidate_sha256 == candidate.candidate_document.sha256
        and stored.opening_payload == candidate.opening.opening_payload
        and stored.opening_sha256 == candidate.opening.opening_payload_sha256
    )


def _receipt_from_stored(
    request: FreshOpeningProvisioningRequest,
    candidate: ProspectiveFreshOpeningCandidate,
    evidence: _FenceEvidence,
) -> FreshOpeningProvisioningReceipt:
    stored = evidence.stored
    if stored is None or not _stored_candidate_is_exact(stored, candidate):
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL fresh-opening replay is not exact"
        )
    expected = _receipt_documents(request, candidate, evidence)
    if (
        stored.opening_receipt_payload != expected.opening.payload
        or stored.opening_receipt_sha256 != expected.opening.sha256
        or stored.provisioning_receipt_payload != expected.provisioning.payload
        or stored.provisioning_receipt_sha256 != expected.provisioning.sha256
    ):
        raise PostgresFreshOpeningProvisioningStorageError(
            "PostgreSQL fresh-opening receipt readback is not exact"
        )
    _document(stored.opening_receipt_payload, stored.opening_receipt_sha256)
    provisioning = _document(
        stored.provisioning_receipt_payload,
        stored.provisioning_receipt_sha256,
    )
    return FreshOpeningProvisioningReceipt(
        document=provisioning,
        target=request.target,
        intent_sha256=stored.intent_sha256,
        approval_sha256=stored.approval_sha256,
        trust_policy_sha256=stored.trust_policy_sha256,
        candidate_sha256=stored.candidate_sha256,
        opening_payload_sha256=stored.opening_sha256,
    )


def _result(
    disposition: FreshOpeningProvisioningDisposition,
    reason: str,
    *,
    receipt: FreshOpeningProvisioningReceipt | None = None,
    current_authority_evaluated: bool,
) -> FreshOpeningProvisioningResult:
    return FreshOpeningProvisioningResult(
        disposition=disposition,
        primary_reason_code=reason,
        receipt=receipt,
        current_authority_evaluated=current_authority_evaluated,
    )


class PostgresFreshOpeningProvisioning:
    """Commit or exactly replay one target-local fresh opening."""

    def __init__(self, connection_factory: Callable[[], object]) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connection_factory = connection_factory

    def provision(
        self,
        request: FreshOpeningProvisioningRequest,
        candidate: ProspectiveFreshOpeningCandidate,
        current_authority: FreshOpeningCurrentAuthorityPort,
        /,
    ) -> FreshOpeningProvisioningResult:
        if type(request) is not FreshOpeningProvisioningRequest:
            raise TypeError("request must be a FreshOpeningProvisioningRequest")
        if type(candidate) is not ProspectiveFreshOpeningCandidate:
            raise TypeError("candidate must be a ProspectiveFreshOpeningCandidate")
        try:
            authority_evaluate = getattr(current_authority, "evaluate", None)
        except Exception:
            authority_evaluate = None
        if not callable(authority_evaluate):
            raise TypeError(
                "current_authority must implement FreshOpeningCurrentAuthorityPort"
            )

        connection = self._connection()
        closed = False
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_WRITE_TRANSACTION_SQL)
                    cursor.execute(_SET_LOCK_TIMEOUT_SQL)
                    cursor.execute(_SET_TIME_ZONE_SQL)
                    evidence = self._acquire(cursor, request, candidate)
                    result, documents = self._resolve_locked(
                        cursor,
                        request,
                        candidate,
                        current_authority,
                        evidence,
                    )
            except PostgresFreshOpeningProvisioningStorageError:
                raise
            except Exception:
                raise PostgresFreshOpeningProvisioningStorageError(
                    "PostgreSQL fresh-opening provisioning failed before commit"
                ) from None

            if documents is None:
                try:
                    connection.rollback()
                except Exception:
                    raise PostgresFreshOpeningProvisioningStorageError(
                        "PostgreSQL fresh-opening read transaction could not finish"
                    ) from None
                return result

            try:
                connection.commit()
            except Exception:
                self._rollback_quietly(connection)
                self._close_quietly(connection)
                closed = True
                return self._resolve_commit_unknown(
                    request,
                    candidate,
                    documents,
                )
            return result
        except Exception:
            self._rollback_quietly(connection)
            raise
        finally:
            if not closed:
                self._close_quietly(connection)

    def _resolve_locked(
        self,
        cursor: object,
        request: FreshOpeningProvisioningRequest,
        candidate: ProspectiveFreshOpeningCandidate,
        current_authority: FreshOpeningCurrentAuthorityPort,
        evidence: _FenceEvidence,
    ) -> tuple[FreshOpeningProvisioningResult, _ReceiptDocuments | None]:
        if evidence.resolution in {"NONCE_CONFLICT", "TARGET_CONFLICT"}:
            if evidence.stored is None:
                raise PostgresFreshOpeningProvisioningStorageError(
                    "PostgreSQL conflict resolution has no durable evidence"
                )
            if _stored_candidate_is_exact(evidence.stored, candidate):
                raise PostgresFreshOpeningProvisioningStorageError(
                    "PostgreSQL conflict resolution contradicts durable evidence"
                )
            return (
                _result(
                    FreshOpeningProvisioningDisposition.CONFLICT,
                    (
                        "FRESH_OPENING_NONCE_CONFLICT"
                        if evidence.resolution == "NONCE_CONFLICT"
                        else "FRESH_OPENING_TARGET_CONFLICT"
                    ),
                    current_authority_evaluated=False,
                ),
                None,
            )

        if not _physical_target_is_exact(evidence, request.target):
            return (
                _result(
                    FreshOpeningProvisioningDisposition.BLOCKED,
                    "TARGET_ADMISSION_BLOCKED",
                    current_authority_evaluated=False,
                ),
                None,
            )

        if evidence.resolution == _ADMISSION_CONFLICT_RESOLUTION:
            return (
                _result(
                    FreshOpeningProvisioningDisposition.BLOCKED,
                    "TARGET_ADMISSION_BLOCKED",
                    current_authority_evaluated=False,
                ),
                None,
            )

        if evidence.stored is not None:
            exact = _stored_candidate_is_exact(evidence.stored, candidate)
            if evidence.resolution == "EXACT_REPLAY" and not exact:
                raise PostgresFreshOpeningProvisioningStorageError(
                    "PostgreSQL replay resolution contradicts durable evidence"
                )
            if not exact:
                raise PostgresFreshOpeningProvisioningStorageError(
                    "PostgreSQL replay resolution contradicts durable evidence"
                )
            receipt = _receipt_from_stored(request, candidate, evidence)
            return (
                _result(
                    FreshOpeningProvisioningDisposition.REPLAYED,
                    "EXACT_DURABLE_REPLAY",
                    receipt=receipt,
                    current_authority_evaluated=False,
                ),
                None,
            )

        if not evidence.v2_empty:
            return (
                _result(
                    FreshOpeningProvisioningDisposition.BLOCKED,
                    "TARGET_ADMISSION_BLOCKED",
                    current_authority_evaluated=False,
                ),
                None,
            )

        preparation = current_authority.evaluate(evidence.evaluated_at)
        if preparation.disposition is not FreshOpeningPreparationDisposition.PREPARED:
            return (
                _result(
                    FreshOpeningProvisioningDisposition.BLOCKED,
                    preparation.disposition.value,
                    current_authority_evaluated=True,
                ),
                None,
            )
        if preparation.candidate != candidate:
            return (
                _result(
                    FreshOpeningProvisioningDisposition.BLOCKED,
                    "TARGET_ADMISSION_BLOCKED",
                    current_authority_evaluated=True,
                ),
                None,
            )

        documents = _receipt_documents(
            request,
            candidate,
            evidence,
        )
        try:
            cursor.execute(
                _COMMIT_OPENING_SQL,
                (
                    candidate.intent_document.payload,
                    candidate.intent_document.sha256,
                    candidate.approval_document.payload,
                    candidate.approval_document.sha256,
                    candidate.trust_policy_document.payload,
                    candidate.trust_policy_document.sha256,
                    candidate.candidate_document.payload,
                    candidate.candidate_document.sha256,
                    candidate.opening.opening_payload,
                    candidate.opening.opening_payload_sha256,
                    documents.opening.payload,
                    documents.opening.sha256,
                    documents.provisioning.payload,
                    documents.provisioning.sha256,
                ),
            )
        except Exception as error:
            if _postgres_sqlstate(error) == _APPROVAL_EXPIRED_SQLSTATE:
                return (
                    _result(
                        FreshOpeningProvisioningDisposition.BLOCKED,
                        "BLOCKED_APPROVAL_EXPIRED",
                        current_authority_evaluated=True,
                    ),
                    None,
                )
            raise
        committed = _row(
            cursor.fetchone(),
            _COMMIT_ROW_LENGTH,
            "fresh-opening commit",
        )
        committed_at = _utc_datetime(committed[1], "fresh-opening commit time")
        if (
            committed[0] != "CREATED"
            or committed_at < evidence.evaluated_at
            or committed[2:]
            != (
                documents.opening.payload,
                documents.opening.sha256,
                documents.provisioning.payload,
                documents.provisioning.sha256,
            )
        ):
            raise PostgresFreshOpeningProvisioningStorageError(
                "PostgreSQL returned an invalid fresh-opening commit receipt"
            )
        receipt = FreshOpeningProvisioningReceipt(
            document=documents.provisioning,
            target=request.target,
            intent_sha256=candidate.intent_document.sha256,
            approval_sha256=candidate.approval_document.sha256,
            trust_policy_sha256=candidate.trust_policy_document.sha256,
            candidate_sha256=candidate.candidate_document.sha256,
            opening_payload_sha256=candidate.opening.opening_payload_sha256,
        )
        return (
            _result(
                FreshOpeningProvisioningDisposition.CREATED,
                "FRESH_OPENING_CREATED",
                receipt=receipt,
                current_authority_evaluated=True,
            ),
            documents,
        )

    @staticmethod
    def _acquire(
        cursor: object,
        request: FreshOpeningProvisioningRequest,
        candidate: ProspectiveFreshOpeningCandidate,
    ) -> _FenceEvidence:
        cursor.execute(
            _ACQUIRE_FENCE_SQL,
            (
                request.intent.trust_domain,
                request.intent.signer_key_id,
                request.intent.nonce,
                candidate.candidate_document.sha256,
            ),
        )
        return _fence_evidence(cursor.fetchone())

    def _resolve_commit_unknown(
        self,
        request: FreshOpeningProvisioningRequest,
        candidate: ProspectiveFreshOpeningCandidate,
        documents: _ReceiptDocuments,
    ) -> FreshOpeningProvisioningResult:
        try:
            evidence = self._readback(request)
            if (
                evidence is not None
                and evidence.resolution == "EXACT_REPLAY"
                and _physical_target_is_exact(evidence, request.target)
                and evidence.stored is not None
                and evidence.stored.opening_receipt_payload == documents.opening.payload
                and evidence.stored.opening_receipt_sha256 == documents.opening.sha256
                and evidence.stored.provisioning_receipt_payload
                == documents.provisioning.payload
                and evidence.stored.provisioning_receipt_sha256
                == documents.provisioning.sha256
            ):
                receipt = _receipt_from_stored(request, candidate, evidence)
                return _result(
                    FreshOpeningProvisioningDisposition.CREATED,
                    "FRESH_OPENING_CREATED",
                    receipt=receipt,
                    current_authority_evaluated=True,
                )
        except Exception:
            pass
        return _result(
            FreshOpeningProvisioningDisposition.COMMIT_UNKNOWN,
            "FRESH_OPENING_COMMIT_UNKNOWN",
            current_authority_evaluated=True,
        )

    def _readback(
        self,
        request: FreshOpeningProvisioningRequest,
    ) -> _FenceEvidence | None:
        connection = self._connection()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_READ_TRANSACTION_SQL)
                    cursor.execute(_SET_LOCK_TIMEOUT_SQL)
                    cursor.execute(_SET_TIME_ZONE_SQL)
                    cursor.execute(
                        _READ_OPENING_SQL,
                        (
                            request.intent.trust_domain,
                            request.intent.signer_key_id,
                            request.intent.nonce,
                        ),
                    )
                    raw = cursor.fetchone()
                    evidence = None if raw is None else _fence_evidence(raw)
                connection.rollback()
                return evidence
            except Exception:
                self._rollback_quietly(connection)
                raise
        finally:
            self._close_quietly(connection)

    def _connection(self) -> object:
        try:
            connection = self._connection_factory()
        except Exception:
            raise PostgresFreshOpeningProvisioningStorageError(
                "could not open a fresh-opening connection"
            ) from None
        required = ("cursor", "commit", "rollback", "close")
        try:
            valid_interface = all(
                callable(getattr(connection, name, None)) for name in required
            )
        except Exception:
            valid_interface = False
        if not valid_interface:
            self._close_quietly(connection)
            raise PostgresFreshOpeningProvisioningStorageError(
                "fresh-opening connection has an invalid interface"
            )
        try:
            transaction_status = getattr(connection, "get_transaction_status", None)
        except Exception:
            transaction_status = None
        if not callable(transaction_status):
            self._close_quietly(connection)
            raise PostgresFreshOpeningProvisioningStorageError(
                "fresh-opening connection has no transaction status"
            )
        try:
            fresh = (
                getattr(connection, "autocommit", None) is False
                and getattr(connection, "status", None) == STATUS_READY
                and transaction_status() == TRANSACTION_STATUS_IDLE
            )
        except Exception:
            fresh = False
        if not fresh:
            self._close_quietly(connection)
            raise PostgresFreshOpeningProvisioningStorageError(
                "fresh-opening connection must be fresh and idle"
            )
        return connection

    @staticmethod
    def _rollback_quietly(connection: object) -> None:
        try:
            connection.rollback()
        except Exception:
            pass

    @staticmethod
    def _close_quietly(connection: object) -> None:
        try:
            close = getattr(connection, "close", None)
            if callable(close):
                close()
        except Exception:
            pass


__all__ = [
    "PostgresFreshOpeningProvisioning",
    "PostgresFreshOpeningProvisioningStorageError",
]
