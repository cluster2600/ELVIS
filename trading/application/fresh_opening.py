"""Pure, non-authoritative fresh-opening intent and approval preparation.

This module deliberately performs no I/O.  It can authenticate an offline
trajectory-B business intent and derive the bytes that a later, separately
authorised provisioning transaction may consume, but it cannot reserve a
nonce, bind a physical database, create an account, activate a runtime, or
authorise trading.
"""

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Protocol

from trading.domain._validation import protect_frozen_dataclass_state

_SCHEMA_VERSION = 1
_PURPOSE = "ELVIS_V2_FRESH_PAPER_OPENING"
_TRAJECTORY = "B"
_CONTINUITY = "NO_V1_CONTINUITY"
_OPENING_CODEC = "paper-account-opening"
_OPENING_VERSION = 1
_SIGNING_PREFIX = b"ELVIS\x00fresh-opening-intent\x00v1\x00"

_POSTGRES_BIGINT_MAX = (1 << 63) - 1
_ED25519_FIELD_PRIME = (1 << 255) - 19
_ED25519_GROUP_ORDER = (1 << 252) + 27742317777372353535851937790883648493
_MAX_DECIMAL_DIGITS = 128
_MAX_DECIMAL_EXPONENT = 128
_MAX_APPROVAL_LIFETIME_SECONDS = 31 * 24 * 60 * 60

_EXECUTION_SCOPE_MAX_LENGTH = 128
_ACCOUNT_KEY_MAX_LENGTH = 255
_ASSET_MAX_LENGTH = 64
_IDENTITY_MAX_LENGTH = 255
_LOGICAL_TARGET_MAX_LENGTH = 255
_TRUST_DOMAIN_MAX_LENGTH = 128
_SIGNER_KEY_ID_MAX_LENGTH = 255
_NONCE_HEX_LENGTH = 64

_LOWER_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_PLACEHOLDERS = frozenset(
    {
        "changeme",
        "default",
        "example",
        "placeholder",
        "todo",
        "unset",
        "unknown",
    }
)

# The sign bit is masked before comparison.  These are the seven encodings
# used by the established Ed25519 small-order blacklist; the two sign variants
# of the order-eight points therefore share one entry.  Canonical-y validation
# independently rejects the p and p+1 aliases in the last two entries.
_ED25519_SMALL_ORDER_BASE = frozenset(
    bytes.fromhex(value)
    for value in (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0100000000000000000000000000000000000000000000000000000000000000",
        "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",
        "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",
        "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    )
)


class FreshOpeningPolicy(str, Enum):
    """The sole opening policy accepted by the trajectory-B slice."""

    EXPLICIT_FRESH_SINGLE_COLLATERAL = "EXPLICIT_FRESH_SINGLE_COLLATERAL"


class FreshOpeningPreparationDisposition(str, Enum):
    """Stable result codes; PREPARED still grants no mutation authority."""

    PREPARED = "PREPARED"
    BLOCKED_AUTHORITY_UNCONFIGURED = "BLOCKED_AUTHORITY_UNCONFIGURED"
    BLOCKED_APPROVAL_MISSING = "BLOCKED_APPROVAL_MISSING"
    BLOCKED_TRUST_POLICY_MISMATCH = "BLOCKED_TRUST_POLICY_MISMATCH"
    BLOCKED_TRUST_DOMAIN_MISMATCH = "BLOCKED_TRUST_DOMAIN_MISMATCH"
    BLOCKED_SIGNER_UNKNOWN = "BLOCKED_SIGNER_UNKNOWN"
    BLOCKED_SIGNER_REVOKED = "BLOCKED_SIGNER_REVOKED"
    BLOCKED_APPROVER_MISMATCH = "BLOCKED_APPROVER_MISMATCH"
    BLOCKED_APPROVAL_BINDING_MISMATCH = "BLOCKED_APPROVAL_BINDING_MISMATCH"
    BLOCKED_APPROVAL_NOT_YET_VALID = "BLOCKED_APPROVAL_NOT_YET_VALID"
    BLOCKED_APPROVAL_EXPIRED = "BLOCKED_APPROVAL_EXPIRED"
    BLOCKED_SIGNATURE_INVALID = "BLOCKED_SIGNATURE_INVALID"


class FreshOpeningEncodedOpening(Protocol):
    """Structural value returned by the injected canonical opening codec."""

    execution_scope: str
    account_key: str
    owner_generation: int
    collateral_asset: str
    opening_version: int
    opening_payload: str
    opening_payload_sha256: str


class FreshOpeningCodecPort(Protocol):
    """Pure port for the existing canonical paper-opening codec."""

    def encode(
        self,
        *,
        execution_scope: str,
        account_key: str,
        owner_generation: int,
        collateral_asset: str,
        collateral_amount: Decimal,
        margin_quantum: Decimal,
    ) -> FreshOpeningEncodedOpening:
        """Return the exact existing-codec opening for one signed balance."""
        ...


class FreshOpeningSignatureVerifierPort(Protocol):
    """Pure port for detached Ed25519 verification only."""

    def verify(
        self,
        *,
        public_key: bytes,
        signature: bytes,
        message: bytes,
    ) -> bool:
        """Return whether the detached signature verifies exactly."""
        ...


def _require_ascii_token(name: str, value: object, maximum: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")
    if len(value) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    if not value.isascii() or any(
        not 0x21 <= ord(character) <= 0x7E for character in value
    ):
        raise ValueError(f"{name} must contain printable non-whitespace ASCII")
    if value.casefold() in _FORBIDDEN_PLACEHOLDERS:
        raise ValueError(f"{name} must not be a placeholder")
    return value


def _require_exact_text(name: str, value: object, expected: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if value != expected:
        raise ValueError(f"{name} must be {expected}")


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_positive_bigint(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= _POSTGRES_BIGINT_MAX:
        raise ValueError(f"{name} is outside durable storage bounds")
    return value


def _require_positive_seconds(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= _MAX_APPROVAL_LIFETIME_SECONDS:
        raise ValueError(f"{name} is outside the supported review window")
    return value


def _normalized_utc(name: str, value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError(f"{name} must be a timezone-aware datetime")
    try:
        if value.utcoffset() is None:
            raise TypeError(f"{name} must be a timezone-aware datetime")
        normalized = value.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be represented in UTC") from exc
    return normalized


def _decimal_components(name: str, value: object) -> tuple[int, int]:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    components = value.as_tuple()
    if len(components.digits) > _MAX_DECIMAL_DIGITS:
        raise ValueError(f"{name} exceeds the supported precision")
    exponent = int(components.exponent)
    if abs(exponent) > _MAX_DECIMAL_EXPONENT:
        raise ValueError(f"{name} exceeds the supported exponent")
    if str(value) != format(value, "f"):
        raise ValueError(f"{name} must use canonical fixed-point notation")
    coefficient = 0
    for digit in components.digits:
        coefficient = coefficient * 10 + digit
    if components.sign:
        coefficient = -coefficient
    return coefficient, exponent


def _is_exact_multiple(value: Decimal, quantum: Decimal) -> bool:
    value_coefficient, value_exponent = _decimal_components("collateral_amount", value)
    quantum_coefficient, quantum_exponent = _decimal_components(
        "margin_quantum", quantum
    )
    exponent_delta = value_exponent - quantum_exponent
    if exponent_delta >= 0:
        numerator = value_coefficient * (10**exponent_delta)
        denominator = quantum_coefficient
    else:
        numerator = value_coefficient
        denominator = quantum_coefficient * (10 ** (-exponent_delta))
    return numerator % denominator == 0


def _require_lower_hex(name: str, value: object, length: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != length or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _canonical_nonweak_ed25519_point(value: bytes) -> bool:
    if type(value) is not bytes or len(value) != 32:
        return False
    encoded_y = bytearray(value)
    encoded_y[31] &= 0x7F
    if int.from_bytes(encoded_y, "little") >= _ED25519_FIELD_PRIME:
        return False
    return bytes(encoded_y) not in _ED25519_SMALL_ORDER_BASE


def _require_public_key(value: object) -> bytes:
    if type(value) is not bytes:
        raise TypeError("ed25519_public_key must be bytes")
    if len(value) != 32:
        raise ValueError("ed25519_public_key must contain exactly 32 bytes")
    if not _canonical_nonweak_ed25519_point(value):
        raise ValueError("ed25519_public_key must be canonical and non-weak")
    return value


def _require_signature(value: object) -> bytes:
    if type(value) is not bytes:
        raise TypeError("signature must be bytes")
    if len(value) != 64:
        raise ValueError("signature must contain exactly 64 bytes")
    encoded_r = value[:32]
    scalar_s = int.from_bytes(value[32:], "little")
    if not _canonical_nonweak_ed25519_point(encoded_r):
        raise ValueError("signature R must be canonical and non-weak")
    if scalar_s >= _ED25519_GROUP_ORDER:
        raise ValueError("signature S must be canonical")
    return value


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError("payload is not canonical JSON data") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_payload(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _datetime_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class CanonicalFreshOpeningDocument:
    """One canonical, secret-free JSON document and its defined digest."""

    payload: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.payload) is not str:
            raise TypeError("payload must be a string")
        if not self.payload or not self.payload.isascii():
            raise ValueError("payload must be non-empty canonical ASCII JSON")
        try:
            decoded = json.loads(self.payload)
            canonical = _canonical_json(decoded)
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
            raise ValueError("payload must be canonical JSON") from exc
        if type(decoded) is not dict or canonical != self.payload:
            raise ValueError("payload must be one canonical JSON object")
        _require_sha256("sha256", self.sha256)


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshOpeningTrustAnchor:
    """One pinned Ed25519 verification key and approval identity."""

    signer_key_id: str
    approver_identity: str
    ed25519_public_key: bytes
    revoked: bool

    def __post_init__(self) -> None:
        _require_ascii_token(
            "signer_key_id", self.signer_key_id, _SIGNER_KEY_ID_MAX_LENGTH
        )
        _require_ascii_token(
            "approver_identity", self.approver_identity, _IDENTITY_MAX_LENGTH
        )
        _require_public_key(self.ed25519_public_key)
        if type(self.revoked) is not bool:
            raise TypeError("revoked must be a boolean")

    @property
    def public_key_sha256(self) -> str:
        """Return the raw-key fingerprint bound into the signed intent."""

        return _sha256_bytes(self.ed25519_public_key)


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshOpeningTrustPolicy:
    """Current out-of-band policy presented for offline verification."""

    schema_version: int
    purpose: str
    trust_domain: str
    max_approval_lifetime_seconds: int
    anchors: tuple[FreshOpeningTrustAnchor, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("schema_version is unknown")
        _require_exact_text("purpose", self.purpose, _PURPOSE)
        _require_ascii_token(
            "trust_domain", self.trust_domain, _TRUST_DOMAIN_MAX_LENGTH
        )
        _require_positive_seconds(
            "max_approval_lifetime_seconds", self.max_approval_lifetime_seconds
        )
        if type(self.anchors) is not tuple or any(
            type(anchor) is not FreshOpeningTrustAnchor for anchor in self.anchors
        ):
            raise TypeError("anchors must contain FreshOpeningTrustAnchor values")
        if not self.anchors:
            raise ValueError("anchors must not be empty")
        key_ids = tuple(anchor.signer_key_id for anchor in self.anchors)
        if key_ids != tuple(sorted(key_ids)) or len(key_ids) != len(set(key_ids)):
            raise ValueError("anchors must have unique sorted signer_key_id values")
        public_keys = tuple(anchor.ed25519_public_key for anchor in self.anchors)
        if len(public_keys) != len(set(public_keys)):
            raise ValueError("anchors must not alias one public key under two IDs")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshOpeningIntent:
    """Exact trajectory-B business intent; never a physical-target receipt."""

    schema_version: int
    purpose: str
    trajectory: str
    continuity: str
    logical_target: str
    execution_scope: str
    account_key: str
    owner_generation: int
    opening_codec: str
    opening_version: int
    collateral_asset: str
    collateral_amount: Decimal
    margin_quantum: Decimal
    opening_policy: FreshOpeningPolicy
    operator_identity: str
    approval_id: str
    approver_identity: str
    approval_issued_at: datetime
    approval_expires_at: datetime
    trust_policy_sha256: str
    trust_domain: str
    signer_key_id: str
    signer_public_key_sha256: str
    nonce: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("schema_version is unknown")
        _require_exact_text("purpose", self.purpose, _PURPOSE)
        _require_exact_text("trajectory", self.trajectory, _TRAJECTORY)
        _require_exact_text("continuity", self.continuity, _CONTINUITY)
        _require_ascii_token(
            "logical_target", self.logical_target, _LOGICAL_TARGET_MAX_LENGTH
        )
        _require_ascii_token(
            "execution_scope", self.execution_scope, _EXECUTION_SCOPE_MAX_LENGTH
        )
        _require_ascii_token("account_key", self.account_key, _ACCOUNT_KEY_MAX_LENGTH)
        _require_positive_bigint("owner_generation", self.owner_generation)
        _require_exact_text("opening_codec", self.opening_codec, _OPENING_CODEC)
        if type(self.opening_version) is not int:
            raise TypeError("opening_version must be an integer")
        if self.opening_version != _OPENING_VERSION:
            raise ValueError("opening_version is unknown")
        _require_ascii_token(
            "collateral_asset", self.collateral_asset, _ASSET_MAX_LENGTH
        )
        _decimal_components("collateral_amount", self.collateral_amount)
        _decimal_components("margin_quantum", self.margin_quantum)
        if not _is_exact_multiple(self.collateral_amount, self.margin_quantum):
            raise ValueError("collateral_amount must be quantized by margin_quantum")
        if type(self.opening_policy) is not FreshOpeningPolicy:
            raise TypeError("opening_policy must be a FreshOpeningPolicy")
        _require_ascii_token(
            "operator_identity", self.operator_identity, _IDENTITY_MAX_LENGTH
        )
        _require_ascii_token("approval_id", self.approval_id, _IDENTITY_MAX_LENGTH)
        _require_ascii_token(
            "approver_identity", self.approver_identity, _IDENTITY_MAX_LENGTH
        )
        if self.operator_identity == self.approver_identity:
            raise ValueError("operator and approver identities must be independent")
        issued = _normalized_utc("approval_issued_at", self.approval_issued_at)
        expires = _normalized_utc("approval_expires_at", self.approval_expires_at)
        if expires <= issued:
            raise ValueError("approval_expires_at must be after approval_issued_at")
        object.__setattr__(self, "approval_issued_at", issued)
        object.__setattr__(self, "approval_expires_at", expires)
        _require_sha256("trust_policy_sha256", self.trust_policy_sha256)
        _require_ascii_token(
            "trust_domain", self.trust_domain, _TRUST_DOMAIN_MAX_LENGTH
        )
        _require_ascii_token(
            "signer_key_id", self.signer_key_id, _SIGNER_KEY_ID_MAX_LENGTH
        )
        _require_sha256("signer_public_key_sha256", self.signer_public_key_sha256)
        _require_lower_hex("nonce", self.nonce, _NONCE_HEX_LENGTH)
        if not any(character != "0" for character in self.nonce):
            raise ValueError("nonce must not be the all-zero value")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class DetachedFreshOpeningApproval:
    """A detached Ed25519 signature over the canonical intent signing bytes."""

    schema_version: int
    intent_sha256: str
    signature: bytes

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("schema_version is unknown")
        _require_sha256("intent_sha256", self.intent_sha256)
        _require_signature(self.signature)


def _require_opening_codec(value: object) -> FreshOpeningCodecPort:
    if not callable(getattr(value, "encode", None)):
        raise TypeError("opening_codec must implement FreshOpeningCodecPort")
    return value


def _require_signature_verifier(
    value: object,
) -> FreshOpeningSignatureVerifierPort:
    if not callable(getattr(value, "verify", None)):
        raise TypeError(
            "signature_verifier must implement FreshOpeningSignatureVerifierPort"
        )
    return value


def _require_encoded_opening(value: object) -> FreshOpeningEncodedOpening:
    try:
        execution_scope = value.execution_scope
        account_key = value.account_key
        owner_generation = value.owner_generation
        collateral_asset = value.collateral_asset
        opening_version = value.opening_version
        opening_payload = value.opening_payload
        opening_payload_sha256 = value.opening_payload_sha256
    except AttributeError as exc:
        raise TypeError("opening codec returned an invalid value") from exc
    _require_ascii_token(
        "opening execution_scope", execution_scope, _EXECUTION_SCOPE_MAX_LENGTH
    )
    _require_ascii_token("opening account_key", account_key, _ACCOUNT_KEY_MAX_LENGTH)
    _require_positive_bigint("opening owner_generation", owner_generation)
    _require_ascii_token(
        "opening collateral_asset", collateral_asset, _ASSET_MAX_LENGTH
    )
    if type(opening_version) is not int or opening_version != _OPENING_VERSION:
        raise ValueError("opening codec returned an unknown opening_version")
    CanonicalFreshOpeningDocument(opening_payload, opening_payload_sha256)
    if not hmac.compare_digest(
        _sha256_payload(opening_payload),
        opening_payload_sha256,
    ):
        raise ValueError("opening codec returned an inconsistent opening digest")
    return value


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class ProspectiveFreshOpeningCandidate:
    """Authenticated prospective bytes, still lacking all mutation authority."""

    intent_document: CanonicalFreshOpeningDocument
    trust_policy_document: CanonicalFreshOpeningDocument
    approval_document: CanonicalFreshOpeningDocument
    opening: FreshOpeningEncodedOpening
    candidate_document: CanonicalFreshOpeningDocument

    def __post_init__(self) -> None:
        for name, value in (
            ("intent_document", self.intent_document),
            ("trust_policy_document", self.trust_policy_document),
            ("approval_document", self.approval_document),
            ("candidate_document", self.candidate_document),
        ):
            if type(value) is not CanonicalFreshOpeningDocument:
                raise TypeError(f"{name} must be a CanonicalFreshOpeningDocument")
        _require_encoded_opening(self.opening)
        expected_digests = (
            (
                "intent_document",
                self.intent_document.sha256,
                _sha256_bytes(
                    _SIGNING_PREFIX + self.intent_document.payload.encode("utf-8")
                ),
            ),
            (
                "trust_policy_document",
                self.trust_policy_document.sha256,
                _sha256_payload(self.trust_policy_document.payload),
            ),
            (
                "approval_document",
                self.approval_document.sha256,
                _sha256_payload(self.approval_document.payload),
            ),
            (
                "opening",
                self.opening.opening_payload_sha256,
                _sha256_payload(self.opening.opening_payload),
            ),
        )
        for name, actual, expected in expected_digests:
            if not hmac.compare_digest(actual, expected):
                raise ValueError(f"{name} digest is inconsistent")
        expected_candidate_payload = _canonical_json(
            {
                "approval_sha256": self.approval_document.sha256,
                "intent_sha256": self.intent_document.sha256,
                "opening_codec": _OPENING_CODEC,
                "opening_payload_sha256": self.opening.opening_payload_sha256,
                "opening_version": self.opening.opening_version,
                "schema_version": _SCHEMA_VERSION,
                "trust_policy_sha256": self.trust_policy_document.sha256,
            }
        )
        if not (
            hmac.compare_digest(
                self.candidate_document.payload,
                expected_candidate_payload,
            )
            and hmac.compare_digest(
                self.candidate_document.sha256,
                _sha256_payload(expected_candidate_payload),
            )
        ):
            raise ValueError("candidate_document is not derived from its evidence")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshOpeningPreparation:
    """Offline result whose prospective opening is never a durable receipt."""

    disposition: FreshOpeningPreparationDisposition
    intent_document: CanonicalFreshOpeningDocument
    prospective_opening: FreshOpeningEncodedOpening
    candidate: ProspectiveFreshOpeningCandidate | None
    nonce_replay_authority_available: bool = False
    physical_target_bound: bool = False
    opening_authorized: bool = False
    provisioning_authorized: bool = False
    runtime_authorized: bool = False
    trading_authorized: bool = False
    pin_source_authenticated: bool = False
    stale_on_return: bool = True

    def __post_init__(self) -> None:
        if type(self.disposition) is not FreshOpeningPreparationDisposition:
            raise TypeError("disposition must be a FreshOpeningPreparationDisposition")
        if type(self.intent_document) is not CanonicalFreshOpeningDocument:
            raise TypeError("intent_document must be a CanonicalFreshOpeningDocument")
        _require_encoded_opening(self.prospective_opening)
        if self.disposition is FreshOpeningPreparationDisposition.PREPARED:
            if type(self.candidate) is not ProspectiveFreshOpeningCandidate:
                raise ValueError("PREPARED requires a prospective candidate")
            if self.candidate.intent_document != self.intent_document:
                raise ValueError("PREPARED candidate must retain the exact intent")
            if self.candidate.opening != self.prospective_opening:
                raise ValueError("PREPARED candidate must retain the exact opening")
        elif self.candidate is not None:
            raise ValueError("a blocked preparation cannot expose a candidate")
        unavailable = (
            self.nonce_replay_authority_available,
            self.physical_target_bound,
            self.opening_authorized,
            self.provisioning_authorized,
            self.runtime_authorized,
            self.trading_authorized,
            self.pin_source_authenticated,
        )
        if any(value is not False for value in unavailable):
            raise ValueError("offline preparation flags must remain false")
        if self.stale_on_return is not True:
            raise ValueError("offline preparation must remain stale on return")


def encode_fresh_opening_trust_policy(
    policy: FreshOpeningTrustPolicy, /
) -> CanonicalFreshOpeningDocument:
    """Encode one current policy; its digest is meaningful only when pinned."""

    if type(policy) is not FreshOpeningTrustPolicy:
        raise TypeError("policy must be a FreshOpeningTrustPolicy")
    payload = _canonical_json(
        {
            "anchors": [
                {
                    "approver_identity": anchor.approver_identity,
                    "ed25519_public_key": anchor.ed25519_public_key.hex(),
                    "revoked": anchor.revoked,
                    "signer_key_id": anchor.signer_key_id,
                }
                for anchor in policy.anchors
            ],
            "max_approval_lifetime_seconds": (policy.max_approval_lifetime_seconds),
            "purpose": policy.purpose,
            "schema_version": policy.schema_version,
            "trust_domain": policy.trust_domain,
        }
    )
    return CanonicalFreshOpeningDocument(payload, _sha256_payload(payload))


def encode_fresh_opening_intent(
    intent: FreshOpeningIntent, /
) -> CanonicalFreshOpeningDocument:
    """Encode the signed intent and hash its domain-separated signing bytes."""

    if type(intent) is not FreshOpeningIntent:
        raise TypeError("intent must be a FreshOpeningIntent")
    payload = _canonical_json(
        {
            "account_key": intent.account_key,
            "approval_expires_at": _datetime_text(intent.approval_expires_at),
            "approval_id": intent.approval_id,
            "approval_issued_at": _datetime_text(intent.approval_issued_at),
            "approver_identity": intent.approver_identity,
            "collateral_amount": str(intent.collateral_amount),
            "collateral_asset": intent.collateral_asset,
            "continuity": intent.continuity,
            "execution_scope": intent.execution_scope,
            "logical_target": intent.logical_target,
            "margin_quantum": str(intent.margin_quantum),
            "nonce": intent.nonce,
            "opening_codec": intent.opening_codec,
            "opening_policy": intent.opening_policy.value,
            "opening_version": intent.opening_version,
            "operator_identity": intent.operator_identity,
            "owner_generation": intent.owner_generation,
            "purpose": intent.purpose,
            "schema_version": intent.schema_version,
            "signer_key_id": intent.signer_key_id,
            "signer_public_key_sha256": intent.signer_public_key_sha256,
            "trajectory": intent.trajectory,
            "trust_domain": intent.trust_domain,
            "trust_policy_sha256": intent.trust_policy_sha256,
        }
    )
    signing_bytes = _SIGNING_PREFIX + payload.encode("utf-8")
    return CanonicalFreshOpeningDocument(payload, _sha256_bytes(signing_bytes))


def fresh_opening_signing_bytes(
    intent_document: CanonicalFreshOpeningDocument, /
) -> bytes:
    """Return the exact public bytes an external ceremony must sign."""

    if type(intent_document) is not CanonicalFreshOpeningDocument:
        raise TypeError("intent_document must be a CanonicalFreshOpeningDocument")
    signing_bytes = _SIGNING_PREFIX + intent_document.payload.encode("utf-8")
    if not hmac.compare_digest(_sha256_bytes(signing_bytes), intent_document.sha256):
        raise ValueError("intent_document digest does not match signing bytes")
    return signing_bytes


def encode_detached_fresh_opening_approval(
    approval: DetachedFreshOpeningApproval, /
) -> CanonicalFreshOpeningDocument:
    """Encode the exact three-field detached approval envelope."""

    if type(approval) is not DetachedFreshOpeningApproval:
        raise TypeError("approval must be a DetachedFreshOpeningApproval")
    payload = _canonical_json(
        {
            "intent_sha256": approval.intent_sha256,
            "schema_version": approval.schema_version,
            "signature": approval.signature.hex(),
        }
    )
    return CanonicalFreshOpeningDocument(payload, _sha256_payload(payload))


def _derive_prospective_opening(
    intent: FreshOpeningIntent,
    opening_codec: FreshOpeningCodecPort,
) -> FreshOpeningEncodedOpening:
    opening = _require_encoded_opening(
        opening_codec.encode(
            execution_scope=intent.execution_scope,
            account_key=intent.account_key,
            owner_generation=intent.owner_generation,
            collateral_asset=intent.collateral_asset,
            collateral_amount=intent.collateral_amount,
            margin_quantum=intent.margin_quantum,
        )
    )
    if (
        opening.opening_version != intent.opening_version
        or opening.execution_scope != intent.execution_scope
        or opening.account_key != intent.account_key
        or opening.owner_generation != intent.owner_generation
        or opening.collateral_asset != intent.collateral_asset
    ):
        raise ValueError("opening codec output conflicts with the signed intent")
    return opening


def _blocked(
    disposition: FreshOpeningPreparationDisposition,
    intent_document: CanonicalFreshOpeningDocument,
    prospective_opening: FreshOpeningEncodedOpening,
) -> FreshOpeningPreparation:
    return FreshOpeningPreparation(
        disposition=disposition,
        intent_document=intent_document,
        prospective_opening=prospective_opening,
        candidate=None,
    )


def prepare_fresh_opening(
    intent: FreshOpeningIntent,
    approval: DetachedFreshOpeningApproval | None,
    /,
    *,
    opening_codec: FreshOpeningCodecPort,
    signature_verifier: FreshOpeningSignatureVerifierPort,
    trust_policy: FreshOpeningTrustPolicy | None,
    expected_trust_policy_sha256: str | None,
    expected_signer_public_key_sha256: str | None,
    evaluated_at: datetime,
) -> FreshOpeningPreparation:
    """Authenticate and prepare prospective bytes without granting authority."""

    if type(intent) is not FreshOpeningIntent:
        raise TypeError("intent must be a FreshOpeningIntent")
    codec = _require_opening_codec(opening_codec)
    verifier = _require_signature_verifier(signature_verifier)
    evaluated = _normalized_utc("evaluated_at", evaluated_at)
    intent_document = encode_fresh_opening_intent(intent)
    prospective_opening = _derive_prospective_opening(intent, codec)

    if (
        trust_policy is None
        or expected_trust_policy_sha256 is None
        or expected_signer_public_key_sha256 is None
    ):
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_AUTHORITY_UNCONFIGURED,
            intent_document,
            prospective_opening,
        )
    if type(trust_policy) is not FreshOpeningTrustPolicy:
        raise TypeError("trust_policy must be a FreshOpeningTrustPolicy or None")
    expected_policy = _require_sha256(
        "expected_trust_policy_sha256", expected_trust_policy_sha256
    )
    expected_key = _require_sha256(
        "expected_signer_public_key_sha256",
        expected_signer_public_key_sha256,
    )
    policy_document = encode_fresh_opening_trust_policy(trust_policy)
    if not (
        hmac.compare_digest(policy_document.sha256, expected_policy)
        and hmac.compare_digest(intent.trust_policy_sha256, expected_policy)
    ):
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_TRUST_POLICY_MISMATCH,
            intent_document,
            prospective_opening,
        )
    if trust_policy.trust_domain != intent.trust_domain:
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_TRUST_DOMAIN_MISMATCH,
            intent_document,
            prospective_opening,
        )
    anchor = next(
        (
            value
            for value in trust_policy.anchors
            if value.signer_key_id == intent.signer_key_id
        ),
        None,
    )
    if anchor is None:
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_SIGNER_UNKNOWN,
            intent_document,
            prospective_opening,
        )
    if anchor.revoked:
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_SIGNER_REVOKED,
            intent_document,
            prospective_opening,
        )
    if anchor.approver_identity != intent.approver_identity:
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_APPROVER_MISMATCH,
            intent_document,
            prospective_opening,
        )
    anchor_fingerprint = anchor.public_key_sha256
    if not (
        hmac.compare_digest(anchor_fingerprint, expected_key)
        and hmac.compare_digest(intent.signer_public_key_sha256, expected_key)
    ):
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_BINDING_MISMATCH,
            intent_document,
            prospective_opening,
        )
    if approval is None:
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_MISSING,
            intent_document,
            prospective_opening,
        )
    if type(approval) is not DetachedFreshOpeningApproval:
        raise TypeError("approval must be a DetachedFreshOpeningApproval or None")
    if not hmac.compare_digest(approval.intent_sha256, intent_document.sha256):
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_BINDING_MISMATCH,
            intent_document,
            prospective_opening,
        )
    if evaluated < intent.approval_issued_at:
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_NOT_YET_VALID,
            intent_document,
            prospective_opening,
        )
    lifetime_seconds = (
        intent.approval_expires_at - intent.approval_issued_at
    ).total_seconds()
    if (
        evaluated >= intent.approval_expires_at
        or lifetime_seconds > trust_policy.max_approval_lifetime_seconds
    ):
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_EXPIRED,
            intent_document,
            prospective_opening,
        )
    signing_bytes = fresh_opening_signing_bytes(intent_document)
    signature_valid = verifier.verify(
        public_key=anchor.ed25519_public_key,
        signature=approval.signature,
        message=signing_bytes,
    )
    if type(signature_valid) is not bool:
        raise TypeError("signature_verifier must return a boolean")
    if not signature_valid:
        return _blocked(
            FreshOpeningPreparationDisposition.BLOCKED_SIGNATURE_INVALID,
            intent_document,
            prospective_opening,
        )

    approval_document = encode_detached_fresh_opening_approval(approval)
    candidate_payload = _canonical_json(
        {
            "approval_sha256": approval_document.sha256,
            "intent_sha256": intent_document.sha256,
            "opening_codec": intent.opening_codec,
            "opening_payload_sha256": prospective_opening.opening_payload_sha256,
            "opening_version": prospective_opening.opening_version,
            "schema_version": _SCHEMA_VERSION,
            "trust_policy_sha256": policy_document.sha256,
        }
    )
    candidate_document = CanonicalFreshOpeningDocument(
        candidate_payload,
        _sha256_payload(candidate_payload),
    )
    candidate = ProspectiveFreshOpeningCandidate(
        intent_document=intent_document,
        trust_policy_document=policy_document,
        approval_document=approval_document,
        opening=prospective_opening,
        candidate_document=candidate_document,
    )
    return FreshOpeningPreparation(
        disposition=FreshOpeningPreparationDisposition.PREPARED,
        intent_document=intent_document,
        prospective_opening=prospective_opening,
        candidate=candidate,
    )


__all__ = [
    "CanonicalFreshOpeningDocument",
    "DetachedFreshOpeningApproval",
    "FreshOpeningCodecPort",
    "FreshOpeningEncodedOpening",
    "FreshOpeningIntent",
    "FreshOpeningPolicy",
    "FreshOpeningPreparation",
    "FreshOpeningPreparationDisposition",
    "FreshOpeningSignatureVerifierPort",
    "FreshOpeningTrustAnchor",
    "FreshOpeningTrustPolicy",
    "ProspectiveFreshOpeningCandidate",
    "encode_detached_fresh_opening_approval",
    "encode_fresh_opening_intent",
    "encode_fresh_opening_trust_policy",
    "fresh_opening_signing_bytes",
    "prepare_fresh_opening",
]
