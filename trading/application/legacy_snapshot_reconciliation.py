"""Pure contract for one dormant legacy-snapshot reconciliation review."""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol

from trading.application.legacy_snapshot_import import (
    LegacySnapshotImportContext,
    LegacySnapshotImportReceipt,
    LegacySnapshotRelationReceipt,
)
from trading.domain._validation import (
    protect_frozen_dataclass_state,
    require_clean_text,
    require_non_negative_decimal,
    require_positive_decimal,
)
from trading.domain.paper_accounting import PaperAccountBalance

_LOWER_HEX = frozenset("0123456789abcdef")
_POSTGRES_BIGINT_MAX = (1 << 63) - 1
_EXECUTION_SCOPE_MAX_LENGTH = 128
_ACCOUNT_KEY_MAX_LENGTH = 255
_ASSET_MAX_LENGTH = 64
_MAX_QUANTIZATION_DIGITS = 10_000


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def legacy_snapshot_relation_evidence_sha256(
    relations: tuple[LegacySnapshotRelationReceipt, ...],
) -> str:
    """Recompute the c3c3a source digest from its seven relation receipts."""

    if type(relations) is not tuple or any(
        type(value) is not LegacySnapshotRelationReceipt for value in relations
    ):
        raise TypeError("relations must contain LegacySnapshotRelationReceipt values")
    return _canonical_sha256(
        [
            {
                "name": relation.name,
                "pk_max": relation.pk_max,
                "pk_min": relation.pk_min,
                "row_count": relation.row_count,
                "sha256": relation.sha256,
            }
            for relation in relations
        ]
    )


def legacy_snapshot_import_receipt_sha256(
    receipt: LegacySnapshotImportReceipt,
) -> str:
    """Hash the canonical public c3c3a receipt document, excluding its context."""

    if type(receipt) is not LegacySnapshotImportReceipt:
        raise TypeError("receipt must be a LegacySnapshotImportReceipt")
    return _canonical_sha256(
        {
            "status": receipt.disposition.value,
            "source_system_identifier": str(receipt.source_system_identifier),
            "target_system_identifier": str(receipt.target_system_identifier),
            "source_canonical_sha256": receipt.source_canonical_sha256,
            "relations": [
                {
                    "name": value.name,
                    "row_count": value.row_count,
                    "pk_min": value.pk_min,
                    "pk_max": value.pk_max,
                    "sha256": value.sha256,
                    "source_sequence_next": value.source_sequence_next,
                    "target_sequence_next": value.target_sequence_next,
                }
                for value in receipt.relations
            ],
            "target_exact": receipt.target_exact,
            "runtime_activation_authorized": (receipt.runtime_activation_authorized),
            "stale_on_return": receipt.stale_on_return,
            "snapshot_authoritative": receipt.snapshot_authoritative,
        }
    )


def legacy_opening_candidate_sha256(
    context: "LegacySnapshotReconciliationContext",
    balances: tuple[PaperAccountBalance, ...],
) -> str:
    """Hash the exact prospective opening identity without persistence imports."""

    if type(context) is not LegacySnapshotReconciliationContext:
        raise TypeError("context must be a LegacySnapshotReconciliationContext")
    if type(balances) is not tuple or any(
        type(value) is not PaperAccountBalance for value in balances
    ):
        raise TypeError("balances must contain PaperAccountBalance values")
    for balance in balances:
        _require_bounded_text("balance asset", balance.asset, _ASSET_MAX_LENGTH)
    return _canonical_sha256(
        {
            "execution_scope": context.execution_scope,
            "owner_generation": context.owner_generation,
            "policy": {
                "account_key": context.account_key,
                "collateral_asset": context.collateral_asset,
                "margin_quantum": str(context.margin_quantum),
            },
            "opening_balances": [
                {
                    "asset": balance.asset,
                    "available": str(balance.available),
                    "reserved": str(balance.reserved),
                }
                for balance in balances
            ],
        }
    )


def legacy_operator_equity_hypothesis_balances(
    context: "LegacySnapshotReconciliationContext",
    hypothesis_realised_pnl: Decimal,
) -> tuple[PaperAccountBalance, ...]:
    """Derive the explicit binary64 hypothesis without claiming runtime parity."""

    if type(context) is not LegacySnapshotReconciliationContext:
        raise TypeError("context must be a LegacySnapshotReconciliationContext")
    if not isinstance(hypothesis_realised_pnl, Decimal):
        raise TypeError("hypothesis_realised_pnl must be a Decimal")
    if not hypothesis_realised_pnl.is_finite():
        raise ValueError("hypothesis_realised_pnl must be finite")
    if context.collateral_asset != "USDT":
        raise ValueError("the operator hypothesis currently requires USDT")
    starting = float(context.hypothesis_starting_collateral)
    realised = float(hypothesis_realised_pnl)
    if not math.isfinite(starting) or not math.isfinite(realised):
        raise ValueError("operator hypothesis is outside binary64 bounds")
    hypothesis = max(0.0, starting + realised)
    if not math.isfinite(hypothesis):
        raise ValueError("operator hypothesis is outside binary64 bounds")
    return (
        PaperAccountBalance("BNB", Decimal("0"), Decimal("0")),
        PaperAccountBalance("BTC", Decimal("0"), Decimal("0")),
        PaperAccountBalance("USDT", Decimal.from_float(hypothesis), Decimal("0")),
    )


def _require_canonical_reset_timestamp(value: str | None) -> None:
    if value is None:
        return
    require_clean_text("reset_timestamp", value)
    try:
        decoded = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("reset_timestamp must be canonical ISO text") from None
    if (
        decoded.tzinfo is not None
        or decoded.isoformat(timespec="microseconds") != value
    ):
        raise ValueError("reset_timestamp must be a canonical naive timestamp")


def _decimal_coefficient(value: Decimal) -> tuple[int, int]:
    components = value.as_tuple()
    if len(components.digits) > _MAX_QUANTIZATION_DIGITS:
        raise ValueError("decimal evidence exceeds the reconciliation bound")
    coefficient = 0
    for digit in components.digits:
        coefficient = coefficient * 10 + digit
    if components.sign:
        coefficient = -coefficient
    return coefficient, int(components.exponent)


def _is_quantized(value: Decimal, quantum: Decimal) -> bool:
    value_coefficient, value_exponent = _decimal_coefficient(value)
    quantum_coefficient, quantum_exponent = _decimal_coefficient(quantum)
    exponent_delta = value_exponent - quantum_exponent
    if abs(exponent_delta) > _MAX_QUANTIZATION_DIGITS:
        raise ValueError("decimal exponent exceeds the reconciliation bound")
    if exponent_delta >= 0:
        numerator = value_coefficient * (10**exponent_delta)
        denominator = quantum_coefficient
    else:
        numerator = value_coefficient
        denominator = quantum_coefficient * (10 ** (-exponent_delta))
    return numerator % denominator == 0


def legacy_opening_quantization_required(
    context: "LegacySnapshotReconciliationContext",
    candidates: tuple["LegacyOpeningCandidate", ...],
) -> bool:
    """Derive the bounded quantization finding from the exact typed evidence."""

    if type(context) is not LegacySnapshotReconciliationContext:
        raise TypeError("context must be a LegacySnapshotReconciliationContext")
    if type(candidates) is not tuple or any(
        type(value) is not LegacyOpeningCandidate for value in candidates
    ):
        raise TypeError("candidates must contain LegacyOpeningCandidate values")
    starting = float(context.hypothesis_starting_collateral)
    if not math.isfinite(starting):
        raise ValueError("hypothesis starting collateral is outside binary64 bounds")
    if Decimal.from_float(starting) != context.hypothesis_starting_collateral:
        return True
    return any(
        not _is_quantized(balance.available, context.margin_quantum)
        for candidate in candidates
        for balance in candidate.balances
    )


def _require_sha256(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_generation(value: object) -> None:
    if type(value) is not int:
        raise TypeError("owner_generation must be an integer")
    if not 1 <= value <= _POSTGRES_BIGINT_MAX:
        raise ValueError("owner_generation is outside its durable storage range")


def _require_bounded_text(name: str, value: object, maximum: int) -> None:
    require_clean_text(name, value)
    if "\x00" in value or any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        raise ValueError(f"{name} is not representable in durable storage")
    if len(value) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class LegacySnapshotReconciliationContext:
    """Explicit hypothesis intent bound to one canonical import receipt."""

    import_context: LegacySnapshotImportContext
    config_document_sha256: str
    import_receipt_sha256: str
    execution_scope: str
    account_key: str
    owner_generation: int
    collateral_asset: str
    margin_quantum: Decimal
    hypothesis_starting_collateral: Decimal

    def __post_init__(self) -> None:
        if type(self.import_context) is not LegacySnapshotImportContext:
            raise TypeError("import_context must be a LegacySnapshotImportContext")
        _require_sha256("config_document_sha256", self.config_document_sha256)
        _require_sha256("import_receipt_sha256", self.import_receipt_sha256)
        _require_bounded_text(
            "execution_scope", self.execution_scope, _EXECUTION_SCOPE_MAX_LENGTH
        )
        _require_bounded_text("account_key", self.account_key, _ACCOUNT_KEY_MAX_LENGTH)
        _require_generation(self.owner_generation)
        _require_bounded_text(
            "collateral_asset", self.collateral_asset, _ASSET_MAX_LENGTH
        )
        require_positive_decimal("margin_quantum", self.margin_quantum)
        require_non_negative_decimal(
            "hypothesis_starting_collateral",
            self.hypothesis_starting_collateral,
        )


class LegacyOpeningCandidateSource(str, Enum):
    """The imported balances and one explicitly non-runtime hypothesis."""

    IMPORTED_ACCOUNT_BALANCES = "IMPORTED_ACCOUNT_BALANCES"
    OPERATOR_EQUITY_HYPOTHESIS = "OPERATOR_EQUITY_HYPOTHESIS"


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class LegacyOpeningCandidate:
    """One exact prospective opening payload, or an unavailable candidate."""

    source: LegacyOpeningCandidateSource
    balances: tuple[PaperAccountBalance, ...]
    opening_payload_sha256: str | None
    available: bool

    def __post_init__(self) -> None:
        if type(self.source) is not LegacyOpeningCandidateSource:
            raise TypeError("source must be a LegacyOpeningCandidateSource")
        if type(self.balances) is not tuple or any(
            type(value) is not PaperAccountBalance for value in self.balances
        ):
            raise TypeError("balances must contain PaperAccountBalance values")
        if type(self.available) is not bool:
            raise TypeError("available must be a boolean")
        if self.available:
            if not self.balances:
                raise ValueError("an available candidate requires balances")
            assets = tuple(value.asset for value in self.balances)
            for asset in assets:
                _require_bounded_text("balance asset", asset, _ASSET_MAX_LENGTH)
            if assets != tuple(sorted(assets)) or len(set(assets)) != len(assets):
                raise ValueError("candidate balances must have unique sorted assets")
            if any(
                value.available < 0 or value.reserved != 0 for value in self.balances
            ):
                raise ValueError(
                    "candidate balances must be solvent and have no reservations"
                )
            _require_sha256("opening_payload_sha256", self.opening_payload_sha256)
        elif self.balances or self.opening_payload_sha256 is not None:
            raise ValueError("an unavailable candidate cannot expose an opening")


class LegacySnapshotReconciliationFindingKind(str, Enum):
    """Stable, secret-free reasons for review or refusal."""

    TARGET_IDENTITY_MISMATCH = "TARGET_IDENTITY_MISMATCH"
    TARGET_CATALOG_DRIFT = "TARGET_CATALOG_DRIFT"
    TARGET_RUNTIME_CONTROL_DRIFT = "TARGET_RUNTIME_CONTROL_DRIFT"
    TARGET_ACTIVE_SESSIONS = "TARGET_ACTIVE_SESSIONS"
    TARGET_LEGACY_ROWS_DRIFT = "TARGET_LEGACY_ROWS_DRIFT"
    TARGET_SEQUENCE_DRIFT = "TARGET_SEQUENCE_DRIFT"
    TARGET_OPEN_POSITION = "TARGET_OPEN_POSITION"
    TARGET_V2_STATE_PRESENT = "TARGET_V2_STATE_PRESENT"
    COLLATERAL_MISSING = "COLLATERAL_MISSING"
    HYPOTHESIS_COLLATERAL_UNSUPPORTED = "HYPOTHESIS_COLLATERAL_UNSUPPORTED"
    NUMERIC_EVIDENCE_INVALID = "NUMERIC_EVIDENCE_INVALID"
    OPENING_EVIDENCE_UNREPRESENTABLE = "OPENING_EVIDENCE_UNREPRESENTABLE"
    RUNTIME_PROVENANCE_UNPROVEN = "RUNTIME_PROVENANCE_UNPROVEN"
    QUANTIZATION_REQUIRED = "QUANTIZATION_REQUIRED"
    CANDIDATE_MISMATCH = "CANDIDATE_MISMATCH"


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class LegacySnapshotReconciliationFinding:
    """One bounded classification without arbitrary database or error text."""

    kind: LegacySnapshotReconciliationFindingKind

    def __post_init__(self) -> None:
        if type(self.kind) is not LegacySnapshotReconciliationFindingKind:
            raise TypeError("kind must be a LegacySnapshotReconciliationFindingKind")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class LegacySnapshotReconciliationEvidence:
    """Stale observations and deterministic operator-hypothesis aggregates."""

    reset_timestamp: str | None
    hypothesis_realised_pnl: Decimal
    hypothesis_trade_fees: Decimal
    hypothesis_liquidation_fees: Decimal
    candidates: tuple[LegacyOpeningCandidate, ...]

    def __post_init__(self) -> None:
        _require_canonical_reset_timestamp(self.reset_timestamp)
        for name, value in (
            ("hypothesis_realised_pnl", self.hypothesis_realised_pnl),
            ("hypothesis_trade_fees", self.hypothesis_trade_fees),
            ("hypothesis_liquidation_fees", self.hypothesis_liquidation_fees),
        ):
            if not isinstance(value, Decimal):
                raise TypeError(f"{name} must be a Decimal")
            if not value.is_finite():
                raise ValueError(f"{name} must be finite")
        if self.hypothesis_trade_fees < 0 or self.hypothesis_liquidation_fees < 0:
            raise ValueError("fee evidence must be non-negative")
        if type(self.candidates) is not tuple or any(
            type(value) is not LegacyOpeningCandidate for value in self.candidates
        ):
            raise TypeError("candidates must contain LegacyOpeningCandidate values")
        if tuple(value.source for value in self.candidates) != tuple(
            LegacyOpeningCandidateSource
        ):
            raise ValueError("candidates must be the exact ordered source set")


class LegacySnapshotReconciliationDisposition(str, Enum):
    """Read-only result; no value grants opening or runtime authority."""

    DECISION_REQUIRED = "DECISION_REQUIRED"
    BLOCKED = "BLOCKED"


_DECISION_FINDINGS = frozenset(
    {
        LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN,
        LegacySnapshotReconciliationFindingKind.QUANTIZATION_REQUIRED,
        LegacySnapshotReconciliationFindingKind.CANDIDATE_MISMATCH,
    }
)


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class LegacySnapshotReconciliationReceipt:
    """Stale review evidence that can never authorise a state transition."""

    context: LegacySnapshotReconciliationContext
    import_receipt: LegacySnapshotImportReceipt
    disposition: LegacySnapshotReconciliationDisposition
    findings: tuple[LegacySnapshotReconciliationFinding, ...]
    evidence: LegacySnapshotReconciliationEvidence
    target_system_identifier: int
    source_canonical_sha256: str
    config_document_sha256: str
    import_receipt_sha256: str
    stale_on_return: bool = True
    snapshot_authoritative: bool = False
    coherent_snapshot_observed: bool = False
    source_provenance_authenticated: bool = False
    target_observations_authenticated: bool = False
    database_window_enforced: bool = False
    account_opening_authorized: bool = False
    account_provisioning_authorized: bool = False
    runtime_activation_authorized: bool = False

    def __post_init__(self) -> None:
        if type(self.context) is not LegacySnapshotReconciliationContext:
            raise TypeError("context must be a LegacySnapshotReconciliationContext")
        if type(self.import_receipt) is not LegacySnapshotImportReceipt:
            raise TypeError("import_receipt must be a LegacySnapshotImportReceipt")
        if self.context.import_context != self.import_receipt.context:
            raise ValueError("context is not bound to the import receipt")
        if self.context.import_receipt_sha256 != legacy_snapshot_import_receipt_sha256(
            self.import_receipt
        ):
            raise ValueError("context is not bound to the canonical import receipt")
        if (
            self.import_receipt.source_canonical_sha256
            != legacy_snapshot_relation_evidence_sha256(self.import_receipt.relations)
        ):
            raise ValueError("import receipt relation evidence is inconsistent")
        if type(self.disposition) is not LegacySnapshotReconciliationDisposition:
            raise TypeError("disposition must be a reconciliation disposition")
        if type(self.findings) is not tuple or any(
            type(value) is not LegacySnapshotReconciliationFinding
            for value in self.findings
        ):
            raise TypeError("findings must contain reconciliation findings")
        if len({value.kind for value in self.findings}) != len(self.findings):
            raise ValueError("findings must be distinct")
        if type(self.evidence) is not LegacySnapshotReconciliationEvidence:
            raise TypeError("evidence must be LegacySnapshotReconciliationEvidence")
        if type(self.target_system_identifier) is not int or (
            self.target_system_identifier <= 0
        ):
            raise ValueError("target_system_identifier must be positive")
        if (
            self.target_system_identifier
            != self.import_receipt.target_system_identifier
        ):
            raise ValueError(
                "target system identifier does not match the import receipt"
            )
        _require_sha256("source_canonical_sha256", self.source_canonical_sha256)
        if self.source_canonical_sha256 != self.import_receipt.source_canonical_sha256:
            raise ValueError("source canonical hash does not match the import receipt")
        _require_sha256("import_receipt_sha256", self.import_receipt_sha256)
        if self.import_receipt_sha256 != self.context.import_receipt_sha256:
            raise ValueError("receipt does not expose its canonical input binding")
        _require_sha256("config_document_sha256", self.config_document_sha256)
        if self.config_document_sha256 != self.context.config_document_sha256:
            raise ValueError("receipt does not expose its canonical config binding")
        candidates = self.evidence.candidates
        kinds = frozenset(value.kind for value in self.findings)
        if (
            self.disposition
            is LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
        ):
            if not all(value.available for value in candidates):
                raise ValueError("DECISION_REQUIRED requires two available candidates")
            if (
                LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN
                not in kinds
                or not kinds.issubset(_DECISION_FINDINGS)
            ):
                raise ValueError("DECISION_REQUIRED requires only decision findings")
            for candidate in candidates:
                if (
                    sum(
                        balance.asset == self.context.collateral_asset
                        for balance in candidate.balances
                    )
                    != 1
                ):
                    raise ValueError("candidate must contain the collateral asset")
                if candidate.opening_payload_sha256 != legacy_opening_candidate_sha256(
                    self.context,
                    candidate.balances,
                ):
                    raise ValueError("candidate opening payload hash is inconsistent")
            if candidates[1].balances != legacy_operator_equity_hypothesis_balances(
                self.context,
                self.evidence.hypothesis_realised_pnl,
            ):
                raise ValueError("operator hypothesis candidate is inconsistent")
            mismatch = (
                candidates[0].balances != candidates[1].balances
                or candidates[0].opening_payload_sha256
                != candidates[1].opening_payload_sha256
            )
            if mismatch != (
                LegacySnapshotReconciliationFindingKind.CANDIDATE_MISMATCH in kinds
            ):
                raise ValueError("candidate mismatch finding is inconsistent")
            if legacy_opening_quantization_required(
                self.context,
                candidates,
            ) != (
                LegacySnapshotReconciliationFindingKind.QUANTIZATION_REQUIRED in kinds
            ):
                raise ValueError("candidate quantization finding is inconsistent")
        else:
            if not self.findings or kinds.issubset(_DECISION_FINDINGS):
                raise ValueError("BLOCKED requires at least one blocking finding")
            if any(value.available for value in candidates) or any(
                value.balances or value.opening_payload_sha256 is not None
                for value in candidates
            ):
                raise ValueError("BLOCKED cannot expose opening candidates")
            if (
                self.evidence.reset_timestamp is not None
                or self.evidence.hypothesis_realised_pnl != 0
                or self.evidence.hypothesis_trade_fees != 0
                or self.evidence.hypothesis_liquidation_fees != 0
            ):
                raise ValueError("BLOCKED cannot expose partial numeric evidence")
        if (
            self.stale_on_return is not True
            or self.snapshot_authoritative is not False
            or self.coherent_snapshot_observed is not False
            or self.source_provenance_authenticated is not False
            or self.target_observations_authenticated is not False
            or self.database_window_enforced is not False
            or self.account_opening_authorized is not False
            or self.account_provisioning_authorized is not False
            or self.runtime_activation_authorized is not False
        ):
            raise ValueError("reconciliation evidence cannot grant authority")


class LegacySnapshotReconciliationPort(Protocol):
    """Port for one read-only, import-bound reconciliation review."""

    def reconcile(
        self,
        context: LegacySnapshotReconciliationContext,
        import_receipt: LegacySnapshotImportReceipt,
        /,
    ) -> LegacySnapshotReconciliationReceipt:
        """Revalidate the target and compare two prospective openings."""
        ...


__all__ = [
    "LegacyOpeningCandidate",
    "LegacyOpeningCandidateSource",
    "LegacySnapshotReconciliationContext",
    "LegacySnapshotReconciliationDisposition",
    "LegacySnapshotReconciliationEvidence",
    "LegacySnapshotReconciliationFinding",
    "LegacySnapshotReconciliationFindingKind",
    "LegacySnapshotReconciliationPort",
    "LegacySnapshotReconciliationReceipt",
    "legacy_snapshot_import_receipt_sha256",
    "legacy_opening_candidate_sha256",
    "legacy_opening_quantization_required",
    "legacy_operator_equity_hypothesis_balances",
    "legacy_snapshot_relation_evidence_sha256",
]
