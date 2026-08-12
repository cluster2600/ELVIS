"""Pure evidence contract for a dormant paper-account cut-over assessment."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from trading.domain._validation import protect_frozen_dataclass_state

_EXECUTION_SCOPE_MAX_LENGTH = 128
_ACCOUNT_KEY_MAX_LENGTH = 255
_SUBJECT_KIND_MAX_LENGTH = 64
_SUBJECT_ID_MAX_LENGTH = 255
_POSTGRES_INTEGER_MIN = -(1 << 31)
_POSTGRES_INTEGER_MAX = (1 << 31) - 1
_POSTGRES_BIGINT_MAX = (1 << 63) - 1
_LOWER_HEX = frozenset("0123456789abcdef")
_LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)


def _require_clean_text(name: str, value: object, max_length: int) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")
    if len(value) > max_length:
        raise ValueError(f"{name} must contain at most {max_length} characters")
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError(f"{name} contains unsupported characters")


def _is_migration_name(value: str) -> bool:
    return (
        bool(value)
        and value[0].islower()
        and value[0].isascii()
        and all(
            (character.islower() and character.isascii())
            or character.isdigit()
            or character == "_"
            for character in value
        )
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _LOWER_HEX for character in value)


def _require_integer(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its durable storage range")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountReadinessContext:
    """Approved account provenance against which one snapshot is assessed."""

    execution_scope: str
    account_key: str
    owner_generation: int
    opening_payload_sha256: str

    def __post_init__(self) -> None:
        _require_clean_text(
            "execution_scope", self.execution_scope, _EXECUTION_SCOPE_MAX_LENGTH
        )
        _require_clean_text("account_key", self.account_key, _ACCOUNT_KEY_MAX_LENGTH)
        _require_integer(
            "owner_generation",
            self.owner_generation,
            minimum=1,
            maximum=_POSTGRES_BIGINT_MAX,
        )
        if type(self.opening_payload_sha256) is not str:
            raise TypeError("opening_payload_sha256 must be a string")
        if not _is_sha256(self.opening_payload_sha256):
            raise ValueError("opening_payload_sha256 must be a lowercase SHA-256")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class MigrationIdentity:
    """One immutable expected or applied migration-ledger identity."""

    version: int
    name: str
    checksum: str

    def __post_init__(self) -> None:
        _require_integer(
            "version",
            self.version,
            minimum=1,
            maximum=_POSTGRES_INTEGER_MAX,
        )
        if type(self.name) is not str:
            raise TypeError("name must be a string")
        if not _is_migration_name(self.name):
            raise ValueError("name must be a canonical migration name")
        if type(self.checksum) is not str:
            raise TypeError("checksum must be a string")
        if not _is_sha256(self.checksum):
            raise ValueError("checksum must be a lowercase SHA-256")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class LegacyRelationWatermark:
    """Bounded inventory evidence for one migration-0001 legacy relation."""

    relation: str
    row_count: int
    max_id: int | None

    def __post_init__(self) -> None:
        if type(self.relation) is not str:
            raise TypeError("relation must be a string")
        if self.relation not in _LEGACY_RELATIONS:
            raise ValueError("relation must name a migration-0001 legacy table")
        _require_integer(
            "row_count",
            self.row_count,
            minimum=0,
            maximum=_POSTGRES_BIGINT_MAX,
        )
        if self.max_id is None:
            if self.row_count != 0:
                raise ValueError("a non-empty relation requires max_id")
        else:
            _require_integer(
                "max_id",
                self.max_id,
                minimum=_POSTGRES_INTEGER_MIN,
                maximum=_POSTGRES_INTEGER_MAX,
            )
            if self.row_count == 0:
                raise ValueError("an empty relation cannot have max_id")


class PaperAccountReadinessFindingKind(str, Enum):
    """Stable reason why a snapshot cannot yet support a fence transition."""

    MIGRATION_LEDGER_ABSENT = "MIGRATION_LEDGER_ABSENT"
    MIGRATION_PENDING = "MIGRATION_PENDING"
    MIGRATION_DRIFT = "MIGRATION_DRIFT"
    ACCOUNT_NOT_PROVISIONED = "ACCOUNT_NOT_PROVISIONED"
    ACCOUNT_PROVENANCE_MISMATCH = "ACCOUNT_PROVENANCE_MISMATCH"
    UNEXPECTED_ACCOUNT = "UNEXPECTED_ACCOUNT"
    ACCOUNT_REPLAY_FAILED = "ACCOUNT_REPLAY_FAILED"
    ACCOUNT_INSOLVENT = "ACCOUNT_INSOLVENT"
    POSITION_REPLAY_FAILED = "POSITION_REPLAY_FAILED"
    UNRESOLVED_SUBMISSION = "UNRESOLVED_SUBMISSION"
    UNACCOUNTED_ORDER = "UNACCOUNTED_ORDER"
    MARGIN_RESERVATION_PRESENT = "MARGIN_RESERVATION_PRESENT"
    DURABLE_OPEN_POSITION = "DURABLE_OPEN_POSITION"
    LEGACY_OPEN_POSITION = "LEGACY_OPEN_POSITION"


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountReadinessFinding:
    """One deterministic finding without unstable exception text."""

    kind: PaperAccountReadinessFindingKind
    subject_kind: str
    subject_id: str

    def __post_init__(self) -> None:
        if type(self.kind) is not PaperAccountReadinessFindingKind:
            raise TypeError("kind must be a PaperAccountReadinessFindingKind")
        _require_clean_text("subject_kind", self.subject_kind, _SUBJECT_KIND_MAX_LENGTH)
        _require_clean_text("subject_id", self.subject_id, _SUBJECT_ID_MAX_LENGTH)


class PaperAccountReadinessDisposition(str, Enum):
    """Assessment only; none of these values grants runtime authority."""

    PREPARED_FOR_FENCE = "PREPARED_FOR_FENCE"
    BLOCKED = "BLOCKED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


_RECONCILIATION_FINDINGS = frozenset(
    {
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.UNRESOLVED_SUBMISSION,
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
    }
)
_MIGRATION_FINDINGS = frozenset(
    {
        PaperAccountReadinessFindingKind.MIGRATION_LEDGER_ABSENT,
        PaperAccountReadinessFindingKind.MIGRATION_PENDING,
        PaperAccountReadinessFindingKind.MIGRATION_DRIFT,
    }
)


def _validate_migrations(name: str, values: object, *, allow_empty: bool) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} cannot be empty")
    if any(type(value) is not MigrationIdentity for value in values):
        raise TypeError(f"{name} must contain only MigrationIdentity values")
    versions = tuple(value.version for value in values)
    if versions != tuple(range(1, len(values) + 1)):
        raise ValueError(
            f"{name} must have unique increasing exact contiguous versions"
        )


def _migration_finding_kind(
    expected: tuple[MigrationIdentity, ...],
    applied: tuple[MigrationIdentity, ...],
) -> PaperAccountReadinessFindingKind | None:
    if not applied:
        return PaperAccountReadinessFindingKind.MIGRATION_LEDGER_ABSENT
    if len(applied) < len(expected) and applied == expected[: len(applied)]:
        return PaperAccountReadinessFindingKind.MIGRATION_PENDING
    if applied != expected:
        return PaperAccountReadinessFindingKind.MIGRATION_DRIFT
    return None


def _finding_identity(
    finding: PaperAccountReadinessFinding,
) -> tuple[str, str, str]:
    return finding.kind.value, finding.subject_kind, finding.subject_id


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountReadinessAssessment:
    """A stale-on-return snapshot report that can never activate a runtime."""

    context: PaperAccountReadinessContext
    expected_migrations: tuple[MigrationIdentity, ...]
    applied_migrations: tuple[MigrationIdentity, ...]
    account_version: int | None
    legacy_watermarks: tuple[LegacyRelationWatermark, ...]
    findings: tuple[PaperAccountReadinessFinding, ...]

    def __post_init__(self) -> None:
        if type(self.context) is not PaperAccountReadinessContext:
            raise TypeError("context must be a PaperAccountReadinessContext")
        _validate_migrations(
            "expected_migrations", self.expected_migrations, allow_empty=False
        )
        _validate_migrations(
            "applied_migrations", self.applied_migrations, allow_empty=True
        )
        if self.account_version is not None:
            _require_integer(
                "account_version",
                self.account_version,
                minimum=0,
                maximum=_POSTGRES_BIGINT_MAX,
            )

        if type(self.legacy_watermarks) is not tuple:
            raise TypeError("legacy_watermarks must be a tuple")
        if any(
            type(value) is not LegacyRelationWatermark
            for value in self.legacy_watermarks
        ):
            raise TypeError(
                "legacy_watermarks must contain only LegacyRelationWatermark values"
            )
        watermarks = tuple(
            sorted(self.legacy_watermarks, key=lambda value: value.relation)
        )
        watermark_relations = tuple(value.relation for value in watermarks)
        if len(watermark_relations) != len(set(watermark_relations)):
            raise ValueError("legacy_watermarks must not repeat a relation")

        if type(self.findings) is not tuple:
            raise TypeError("findings must be a tuple")
        if any(
            type(value) is not PaperAccountReadinessFinding for value in self.findings
        ):
            raise TypeError(
                "findings must contain only PaperAccountReadinessFinding values"
            )
        findings_by_identity = {
            _finding_identity(finding): finding for finding in self.findings
        }

        expected_migration_kind = _migration_finding_kind(
            self.expected_migrations,
            self.applied_migrations,
        )
        supplied_migration_kinds = {
            finding.kind
            for finding in findings_by_identity.values()
            if finding.kind in _MIGRATION_FINDINGS
        }
        raw_drift = (
            PaperAccountReadinessFindingKind.MIGRATION_DRIFT in supplied_migration_kinds
        )
        if expected_migration_kind is None:
            if supplied_migration_kinds - {
                PaperAccountReadinessFindingKind.MIGRATION_DRIFT
            }:
                raise ValueError("migration findings conflict with matching ledgers")
            if raw_drift:
                expected_migration_kind = (
                    PaperAccountReadinessFindingKind.MIGRATION_DRIFT
                )
            elif watermark_relations != _LEGACY_RELATIONS:
                raise ValueError(
                    "legacy_watermarks must cover every legacy relation once"
                )
        if expected_migration_kind is not None:
            if raw_drift:
                expected_migration_kind = (
                    PaperAccountReadinessFindingKind.MIGRATION_DRIFT
                )
            elif supplied_migration_kinds - {expected_migration_kind}:
                raise ValueError("migration finding conflicts with migration evidence")
            findings_by_identity = {
                identity: finding
                for identity, finding in findings_by_identity.items()
                if finding.kind not in _MIGRATION_FINDINGS
            }
            migration_finding = PaperAccountReadinessFinding(
                kind=expected_migration_kind,
                subject_kind="migration_ledger",
                subject_id="np.schema_migrations",
            )
            findings_by_identity[_finding_identity(migration_finding)] = (
                migration_finding
            )
        object.__setattr__(self, "legacy_watermarks", watermarks)

        open_positions = next(
            (
                watermark
                for watermark in watermarks
                if watermark.relation == "np.open_positions"
            ),
            None,
        )
        supplied_legacy_open = any(
            finding.kind is PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION
            for finding in findings_by_identity.values()
        )
        if open_positions is not None and open_positions.row_count:
            findings_by_identity = {
                identity: finding
                for identity, finding in findings_by_identity.items()
                if finding.kind
                is not PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION
            }
            legacy_finding = PaperAccountReadinessFinding(
                kind=PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION,
                subject_kind="legacy_relation",
                subject_id="np.open_positions",
            )
            findings_by_identity[_finding_identity(legacy_finding)] = legacy_finding
        elif open_positions is not None and supplied_legacy_open:
            raise ValueError(
                "legacy open-position finding conflicts with its watermark"
            )

        findings = tuple(sorted(findings_by_identity.values(), key=_finding_identity))
        if (
            any(
                finding.kind is PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED
                for finding in findings
            )
            and self.account_version is not None
        ):
            raise ValueError("a missing account cannot have an account version")
        if (
            self.account_version is None
            and expected_migration_kind is None
            and not any(
                finding.kind
                in {
                    PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED,
                    PaperAccountReadinessFindingKind.ACCOUNT_PROVENANCE_MISMATCH,
                    PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
                }
                for finding in findings
            )
        ):
            raise ValueError("missing account evidence requires an account finding")
        object.__setattr__(self, "findings", findings)

    @property
    def disposition(self) -> PaperAccountReadinessDisposition:
        """Derive the required next action without granting activation authority."""
        if any(finding.kind in _RECONCILIATION_FINDINGS for finding in self.findings):
            return PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
        if self.findings:
            return PaperAccountReadinessDisposition.BLOCKED
        return PaperAccountReadinessDisposition.PREPARED_FOR_FENCE

    @property
    def snapshot_authoritative(self) -> bool:
        """Remain false because activation must re-check under its own locks."""
        return False


class PaperAccountReadinessPort(Protocol):
    """Application port for one non-authoritative readiness assessment."""

    def assess(
        self,
        context: PaperAccountReadinessContext,
        /,
    ) -> PaperAccountReadinessAssessment:
        """Assess one context without authorising any runtime transition."""
        ...


__all__ = [
    "LegacyRelationWatermark",
    "MigrationIdentity",
    "PaperAccountReadinessAssessment",
    "PaperAccountReadinessContext",
    "PaperAccountReadinessDisposition",
    "PaperAccountReadinessFinding",
    "PaperAccountReadinessFindingKind",
    "PaperAccountReadinessPort",
]
