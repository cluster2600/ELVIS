"""Pure orchestration contract for one durable trajectory-B fresh opening.

The application layer derives exact replay-comparison bytes before evaluating
current authority.  A persistence adapter must first inspect the target-local
nonce under its database fence.  It may call the supplied authority evaluator
only when the nonce and target opening are absent, and it must commit the
approved evidence, opening, nonce, and physical receipt atomically.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from trading.application.fresh_opening import (
    CanonicalFreshOpeningDocument,
    DetachedFreshOpeningApproval,
    FreshOpeningCodecPort,
    FreshOpeningIntent,
    FreshOpeningPreparation,
    FreshOpeningSignatureVerifierPort,
    FreshOpeningTrustPolicy,
    ProspectiveFreshOpeningCandidate,
    derive_prospective_fresh_opening_candidate,
    prepare_fresh_opening,
)
from trading.domain._validation import protect_frozen_dataclass_state

_LOWER_ASCII = frozenset("abcdefghijklmnopqrstuvwxyz")
_IDENTIFIER_TAIL = _LOWER_ASCII | frozenset("0123456789_")
_LOWER_HEX = frozenset("0123456789abcdef")
_POSTGRES_SYSTEM_IDENTIFIER_MAX = (1 << 64) - 1
_DEPLOYMENT_INCARNATION_MAX_LENGTH = 255


def _require_identifier(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if (
        not 1 <= len(value) <= 63
        or value[0] not in _LOWER_ASCII
        or any(character not in _IDENTIFIER_TAIL for character in value[1:])
    ):
        raise ValueError(f"{name} must be a lowercase PostgreSQL identifier")
    return value


def _require_token(name: str, value: object, maximum: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if (
        not value
        or value != value.strip()
        or len(value) > maximum
        or not value.isascii()
        or any(not 0x21 <= ord(character) <= 0x7E for character in value)
    ):
        raise ValueError(f"{name} must be bounded printable ASCII")
    return value


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshOpeningPhysicalTarget:
    """Expected physical target and independently authenticated admission pins."""

    expected_database: str
    expected_system_identifier: int
    control_plane_role: str
    opening_anchor_role: str
    deployment_incarnation_id: str
    terminal_catalog_sha256: str
    pin_authority_record_sha256: str

    def __post_init__(self) -> None:
        _require_identifier("expected_database", self.expected_database)
        if (
            type(self.expected_system_identifier) is not int
            or not 1
            <= self.expected_system_identifier
            <= _POSTGRES_SYSTEM_IDENTIFIER_MAX
        ):
            raise ValueError(
                "expected_system_identifier must be a positive PostgreSQL system ID"
            )
        _require_identifier("control_plane_role", self.control_plane_role)
        _require_identifier("opening_anchor_role", self.opening_anchor_role)
        if self.opening_anchor_role == self.control_plane_role:
            raise ValueError("opening_anchor_role must differ from control_plane_role")
        _require_token(
            "deployment_incarnation_id",
            self.deployment_incarnation_id,
            _DEPLOYMENT_INCARNATION_MAX_LENGTH,
        )
        _require_sha256("terminal_catalog_sha256", self.terminal_catalog_sha256)
        _require_sha256("pin_authority_record_sha256", self.pin_authority_record_sha256)


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshOpeningProvisioningRequest:
    """Typed evidence and target pins for one apply-or-exact-replay attempt."""

    intent: FreshOpeningIntent
    approval: DetachedFreshOpeningApproval
    trust_policy: FreshOpeningTrustPolicy
    expected_trust_policy_sha256: str
    expected_signer_public_key_sha256: str
    target: FreshOpeningPhysicalTarget

    def __post_init__(self) -> None:
        if type(self.intent) is not FreshOpeningIntent:
            raise TypeError("intent must be a FreshOpeningIntent")
        if type(self.approval) is not DetachedFreshOpeningApproval:
            raise TypeError("approval must be a DetachedFreshOpeningApproval")
        if type(self.trust_policy) is not FreshOpeningTrustPolicy:
            raise TypeError("trust_policy must be a FreshOpeningTrustPolicy")
        _require_sha256(
            "expected_trust_policy_sha256", self.expected_trust_policy_sha256
        )
        _require_sha256(
            "expected_signer_public_key_sha256",
            self.expected_signer_public_key_sha256,
        )
        if type(self.target) is not FreshOpeningPhysicalTarget:
            raise TypeError("target must be a FreshOpeningPhysicalTarget")


class FreshOpeningProvisioningDisposition(str, Enum):
    """Stable operator outcomes from the target transaction boundary."""

    CREATED = "CREATED"
    REPLAYED = "REPLAYED"
    BLOCKED = "BLOCKED"
    CONFLICT = "CONFLICT"
    COMMIT_UNKNOWN = "COMMIT_UNKNOWN"


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshOpeningProvisioningReceipt:
    """Exact committed physical-target receipt, never runtime authority."""

    document: CanonicalFreshOpeningDocument
    target: FreshOpeningPhysicalTarget
    intent_sha256: str
    approval_sha256: str
    trust_policy_sha256: str
    candidate_sha256: str
    opening_payload_sha256: str
    migration_head: int = 7
    runtime_mode: str = "LEGACY"
    runtime_generation: int = 0
    authority_transition_sequence: int = 0
    runtime_activation_authorized: bool = False
    trading_authorized: bool = False
    stale_on_return: bool = True

    def __post_init__(self) -> None:
        if type(self.document) is not CanonicalFreshOpeningDocument:
            raise TypeError("document must be a CanonicalFreshOpeningDocument")
        if type(self.target) is not FreshOpeningPhysicalTarget:
            raise TypeError("target must be a FreshOpeningPhysicalTarget")
        for name, value in (
            ("intent_sha256", self.intent_sha256),
            ("approval_sha256", self.approval_sha256),
            ("trust_policy_sha256", self.trust_policy_sha256),
            ("candidate_sha256", self.candidate_sha256),
            ("opening_payload_sha256", self.opening_payload_sha256),
        ):
            _require_sha256(name, value)
        if self.migration_head != 7:
            raise ValueError("migration_head must be the PR3 terminal migration")
        if self.runtime_mode != "LEGACY":
            raise ValueError("a fresh opening receipt must remain LEGACY")
        if self.runtime_generation != 0 or self.authority_transition_sequence != 0:
            raise ValueError(
                "a fresh opening receipt must remain generation/sequence 0"
            )
        if self.runtime_activation_authorized is not False:
            raise ValueError("a fresh opening receipt cannot authorize activation")
        if self.trading_authorized is not False:
            raise ValueError("a fresh opening receipt cannot authorize trading")
        if self.stale_on_return is not True:
            raise ValueError("a fresh opening receipt must remain stale on return")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshOpeningProvisioningResult:
    """Typed safe result returned by the durable adapter."""

    disposition: FreshOpeningProvisioningDisposition
    primary_reason_code: str
    receipt: FreshOpeningProvisioningReceipt | None
    current_authority_evaluated: bool
    database_contact: bool = True
    nonce_registry_checked: bool = True
    runtime_activation_authorized: bool = False
    trading_authorized: bool = False
    stale_on_return: bool = True

    def __post_init__(self) -> None:
        if type(self.disposition) is not FreshOpeningProvisioningDisposition:
            raise TypeError("disposition must be a FreshOpeningProvisioningDisposition")
        _require_token("primary_reason_code", self.primary_reason_code, 255)
        committed = self.disposition in {
            FreshOpeningProvisioningDisposition.CREATED,
            FreshOpeningProvisioningDisposition.REPLAYED,
        }
        if committed != (type(self.receipt) is FreshOpeningProvisioningReceipt):
            raise ValueError("only committed outcomes carry a provisioning receipt")
        if type(self.current_authority_evaluated) is not bool:
            raise TypeError("current_authority_evaluated must be a boolean")
        if self.disposition is FreshOpeningProvisioningDisposition.REPLAYED:
            if self.current_authority_evaluated:
                raise ValueError("exact replay must precede current authority checks")
        elif self.disposition is FreshOpeningProvisioningDisposition.CREATED:
            if not self.current_authority_evaluated:
                raise ValueError("a created opening requires current authority")
        for name, value, expected in (
            ("database_contact", self.database_contact, True),
            ("nonce_registry_checked", self.nonce_registry_checked, True),
            (
                "runtime_activation_authorized",
                self.runtime_activation_authorized,
                False,
            ),
            ("trading_authorized", self.trading_authorized, False),
            ("stale_on_return", self.stale_on_return, True),
        ):
            if value is not expected:
                raise ValueError(f"{name} must remain {expected}")

    @property
    def side_effect_state(self) -> str:
        if self.disposition in {
            FreshOpeningProvisioningDisposition.CREATED,
            FreshOpeningProvisioningDisposition.REPLAYED,
        }:
            return "COMMITTED"
        if self.disposition is FreshOpeningProvisioningDisposition.COMMIT_UNKNOWN:
            return "UNKNOWN"
        return "NONE"


class FreshOpeningCurrentAuthorityPort(Protocol):
    """Evaluate current signature, pins, freshness and revocation at DB time."""

    def evaluate(self, evaluated_at: datetime, /) -> FreshOpeningPreparation:
        """Return current non-authoritative preparation at the locked DB time."""
        ...


class FreshOpeningProvisioningPort(Protocol):
    """Target adapter that owns replay-first locking and the atomic write."""

    def provision(
        self,
        request: FreshOpeningProvisioningRequest,
        candidate: ProspectiveFreshOpeningCandidate,
        current_authority: FreshOpeningCurrentAuthorityPort,
        /,
    ) -> FreshOpeningProvisioningResult:
        """Resolve an exact replay or conditionally commit one absent opening."""
        ...


class _CurrentFreshOpeningAuthority:
    def __init__(
        self,
        request: FreshOpeningProvisioningRequest,
        opening_codec: FreshOpeningCodecPort,
        signature_verifier: FreshOpeningSignatureVerifierPort,
    ) -> None:
        self._request = request
        self._opening_codec = opening_codec
        self._signature_verifier = signature_verifier

    def evaluate(self, evaluated_at: datetime, /) -> FreshOpeningPreparation:
        return prepare_fresh_opening(
            self._request.intent,
            self._request.approval,
            opening_codec=self._opening_codec,
            signature_verifier=self._signature_verifier,
            trust_policy=self._request.trust_policy,
            expected_trust_policy_sha256=(self._request.expected_trust_policy_sha256),
            expected_signer_public_key_sha256=(
                self._request.expected_signer_public_key_sha256
            ),
            evaluated_at=evaluated_at,
        )


class FreshOpeningProvisioningService:
    """Compose pure candidate reconstruction with one durable adapter."""

    def __init__(
        self,
        provisioning: FreshOpeningProvisioningPort,
        opening_codec: FreshOpeningCodecPort,
        signature_verifier: FreshOpeningSignatureVerifierPort,
    ) -> None:
        if not callable(getattr(provisioning, "provision", None)):
            raise TypeError("provisioning must implement FreshOpeningProvisioningPort")
        if not callable(getattr(opening_codec, "encode", None)):
            raise TypeError("opening_codec must implement FreshOpeningCodecPort")
        if not callable(getattr(signature_verifier, "verify", None)):
            raise TypeError(
                "signature_verifier must implement FreshOpeningSignatureVerifierPort"
            )
        self._provisioning = provisioning
        self._opening_codec = opening_codec
        self._signature_verifier = signature_verifier

    def provision(
        self, request: FreshOpeningProvisioningRequest, /
    ) -> FreshOpeningProvisioningResult:
        if type(request) is not FreshOpeningProvisioningRequest:
            raise TypeError("request must be a FreshOpeningProvisioningRequest")
        candidate = derive_prospective_fresh_opening_candidate(
            request.intent,
            request.approval,
            request.trust_policy,
            opening_codec=self._opening_codec,
        )
        authority = _CurrentFreshOpeningAuthority(
            request,
            self._opening_codec,
            self._signature_verifier,
        )
        result = self._provisioning.provision(
            request,
            candidate,
            authority,
        )
        if type(result) is not FreshOpeningProvisioningResult:
            raise TypeError("provisioning adapter returned an invalid result")
        if result.receipt is not None:
            receipt = result.receipt
            if receipt.target != request.target:
                raise ValueError("provisioning receipt targets another database")
            expected = (
                candidate.intent_document.sha256,
                candidate.approval_document.sha256,
                candidate.trust_policy_document.sha256,
                candidate.candidate_document.sha256,
                candidate.opening.opening_payload_sha256,
            )
            actual = (
                receipt.intent_sha256,
                receipt.approval_sha256,
                receipt.trust_policy_sha256,
                receipt.candidate_sha256,
                receipt.opening_payload_sha256,
            )
            if actual != expected:
                raise ValueError("provisioning receipt conflicts with exact evidence")
        return result


__all__ = [
    "FreshOpeningCurrentAuthorityPort",
    "FreshOpeningPhysicalTarget",
    "FreshOpeningProvisioningDisposition",
    "FreshOpeningProvisioningPort",
    "FreshOpeningProvisioningReceipt",
    "FreshOpeningProvisioningRequest",
    "FreshOpeningProvisioningResult",
    "FreshOpeningProvisioningService",
]
