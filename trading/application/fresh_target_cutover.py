"""Pure evidence contract for a read-only fresh-target cut-over preflight."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from trading.domain._validation import protect_frozen_dataclass_state

_LOWER_HEX = frozenset("0123456789abcdef")
_LOWER_ASCII = frozenset("abcdefghijklmnopqrstuvwxyz")
_IDENTIFIER_TAIL = _LOWER_ASCII | frozenset("0123456789_")
_LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)


def _require_text(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be non-empty, trimmed text")


def _require_count(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_role(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if (
        not 1 <= len(value) <= 63
        or value[0] not in _LOWER_ASCII
        or any(character not in _IDENTIFIER_TAIL for character in value[1:])
    ):
        raise ValueError(f"{name} must be a lowercase PostgreSQL identifier")


def _require_sha256(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshTargetRoleManifest:
    """Seven non-secret target role identifiers."""

    schema_owner: str
    migrator: str
    legacy_runtime: str
    atomic_runtime: str
    activation: str
    readiness: str
    trainer: str

    def __post_init__(self) -> None:
        values = (
            self.schema_owner,
            self.migrator,
            self.legacy_runtime,
            self.atomic_runtime,
            self.activation,
            self.readiness,
            self.trainer,
        )
        for name, value in zip(
            (
                "schema_owner",
                "migrator",
                "legacy_runtime",
                "atomic_runtime",
                "activation",
                "readiness",
                "trainer",
            ),
            values,
        ):
            _require_role(name, value)
        if len(set(values)) != len(values):
            raise ValueError("target role identifiers must be pairwise distinct")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshTargetBootstrapIntent:
    """Pure target identity expected to have completed bootstrap."""

    expected_database: str
    admin_role: str
    roles: FreshTargetRoleManifest

    def __post_init__(self) -> None:
        _require_role("expected_database", self.expected_database)
        _require_role("admin_role", self.admin_role)
        if type(self.roles) is not FreshTargetRoleManifest:
            raise TypeError("roles must be a FreshTargetRoleManifest")
        if self.admin_role in (
            self.roles.schema_owner,
            self.roles.migrator,
            self.roles.legacy_runtime,
            self.roles.atomic_runtime,
            self.roles.activation,
            self.roles.readiness,
            self.roles.trainer,
        ):
            raise ValueError("admin_role must differ from every managed role")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshTargetCutoverContext:
    """Non-secret identities and target authority expected by one inspection."""

    source_expected_database: str
    source_expected_role: str
    target_bootstrap_intent: FreshTargetBootstrapIntent

    def __post_init__(self) -> None:
        _require_role("source_expected_database", self.source_expected_database)
        _require_role("source_expected_role", self.source_expected_role)
        if type(self.target_bootstrap_intent) is not FreshTargetBootstrapIntent:
            raise TypeError(
                "target_bootstrap_intent must be a FreshTargetBootstrapIntent"
            )


class FreshTargetCutoverStatus(str, Enum):
    """Read-only decision; READY never grants runtime authority."""

    READY_FOR_FRESH_TARGET = "READY_FOR_FRESH_TARGET"
    BLOCKED = "BLOCKED"


class FreshTargetCutoverBlocker(str, Enum):
    """Stable, secret-free reasons why a cut-over cannot proceed."""

    SOURCE_IDENTITY = "SOURCE_IDENTITY"
    SOURCE_ACTIVE_SESSIONS = "SOURCE_ACTIVE_SESSIONS"
    SOURCE_SCHEMA = "SOURCE_SCHEMA"
    SOURCE_OPEN_POSITIONS = "SOURCE_OPEN_POSITIONS"
    SOURCE_DATA_QUALITY = "SOURCE_DATA_QUALITY"
    SAME_CLUSTER = "SAME_CLUSTER"
    TARGET_NOT_COMPLETE = "TARGET_NOT_COMPLETE"
    TARGET_MODE = "TARGET_MODE"
    TARGET_NOT_EMPTY = "TARGET_NOT_EMPTY"


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshTargetRelationEvidence:
    """Deterministic inventory for one exact migration-0001 relation."""

    name: str
    row_count: int
    pk_min: int | None
    pk_max: int | None
    sha256: str

    def __post_init__(self) -> None:
        if self.name not in _LEGACY_RELATIONS:
            raise ValueError("name must identify an exact legacy relation")
        _require_count("row_count", self.row_count)
        _require_sha256("sha256", self.sha256)
        if self.row_count == 0:
            if self.pk_min is not None or self.pk_max is not None:
                raise ValueError("an empty relation cannot have primary-key bounds")
            return
        if type(self.pk_min) is not int or type(self.pk_max) is not int:
            raise TypeError("non-empty relation bounds must be integers")
        if self.pk_min > self.pk_max:
            raise ValueError("primary-key bounds are reversed")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshTargetCutoverSourceEvidence:
    """One repeatable-read source-clone snapshot."""

    system_identifier: int
    relations: tuple[FreshTargetRelationEvidence, ...]
    other_session_count: int
    open_position_count: int
    semantic_invalid_row_count: int
    canonical_sha256: str | None
    legacy_layout_exact: bool
    identity_exact: bool

    def __post_init__(self) -> None:
        if type(self.system_identifier) is not int or self.system_identifier <= 0:
            raise ValueError("system_identifier must be a positive integer")
        if type(self.relations) is not tuple or any(
            type(value) is not FreshTargetRelationEvidence for value in self.relations
        ):
            raise TypeError("relations must contain FreshTargetRelationEvidence values")
        relation_names = tuple(value.name for value in self.relations)
        if relation_names not in ((), _LEGACY_RELATIONS):
            raise ValueError("relations must be empty or the exact ordered legacy set")
        _require_count("other_session_count", self.other_session_count)
        _require_count("open_position_count", self.open_position_count)
        _require_count("semantic_invalid_row_count", self.semantic_invalid_row_count)
        if self.canonical_sha256 is not None:
            _require_sha256("canonical_sha256", self.canonical_sha256)
        if (
            type(self.legacy_layout_exact) is not bool
            or type(self.identity_exact) is not bool
        ):
            raise TypeError("source exactness fields must be booleans")
        if self.legacy_layout_exact:
            if not self.identity_exact:
                raise ValueError("an exact source layout requires exact identity")
            if relation_names != _LEGACY_RELATIONS or self.canonical_sha256 is None:
                raise ValueError(
                    "an exact source layout requires complete hashed evidence"
                )
            open_positions = next(
                value.row_count
                for value in self.relations
                if value.name == "np.open_positions"
            )
            if self.open_position_count != open_positions:
                raise ValueError("open-position evidence must match its relation")
        elif self.relations or self.canonical_sha256 is not None:
            raise ValueError("an inexact source layout cannot carry trusted rows")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshTargetCutoverTargetEvidence:
    """Read-only evidence for one fully bootstrapped empty target."""

    system_identifier: int
    terminal_catalog_exact: bool
    migration_versions: tuple[int, ...]
    runtime_mode: str | None
    runtime_generation: int | None
    nonempty_relations: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.system_identifier) is not int or self.system_identifier <= 0:
            raise ValueError("system_identifier must be a positive integer")
        if type(self.terminal_catalog_exact) is not bool:
            raise TypeError("terminal_catalog_exact must be a boolean")
        if type(self.migration_versions) is not tuple or any(
            type(version) is not int for version in self.migration_versions
        ):
            raise TypeError("migration_versions must contain integers")
        if self.runtime_mode is not None and type(self.runtime_mode) is not str:
            raise TypeError("runtime_mode must be text or None")
        if (
            self.runtime_generation is not None
            and type(self.runtime_generation) is not int
        ):
            raise TypeError("runtime_generation must be an integer or None")
        if type(self.nonempty_relations) is not tuple or any(
            type(relation) is not str for relation in self.nonempty_relations
        ):
            raise TypeError("nonempty_relations must contain strings")
        if self.nonempty_relations != tuple(sorted(set(self.nonempty_relations))):
            raise ValueError("nonempty_relations must be unique and sorted")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class FreshTargetCutoverReceipt:
    """Stale-on-return evidence which can never authorize a cut-over alone."""

    status: FreshTargetCutoverStatus
    blockers: tuple[FreshTargetCutoverBlocker, ...]
    source: FreshTargetCutoverSourceEvidence
    target: FreshTargetCutoverTargetEvidence
    stale_on_return: bool = True
    snapshot_authoritative: bool = False

    def __post_init__(self) -> None:
        if type(self.status) is not FreshTargetCutoverStatus:
            raise TypeError("status must be a FreshTargetCutoverStatus")
        if type(self.blockers) is not tuple or any(
            type(value) is not FreshTargetCutoverBlocker for value in self.blockers
        ):
            raise TypeError("blockers must contain FreshTargetCutoverBlocker values")
        if self.blockers != tuple(
            sorted(set(self.blockers), key=lambda item: item.value)
        ):
            raise ValueError("blockers must be unique and sorted")
        if (
            self.status is FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET
            and self.blockers
        ):
            raise ValueError("a ready receipt cannot contain blockers")
        if self.status is FreshTargetCutoverStatus.BLOCKED and not self.blockers:
            raise ValueError("a blocked receipt requires blockers")
        if type(self.source) is not FreshTargetCutoverSourceEvidence:
            raise TypeError("source must be FreshTargetCutoverSourceEvidence")
        if type(self.target) is not FreshTargetCutoverTargetEvidence:
            raise TypeError("target must be FreshTargetCutoverTargetEvidence")
        if self.stale_on_return is not True or self.snapshot_authoritative is not False:
            raise ValueError(
                "cut-over evidence must remain stale and non-authoritative"
            )
        if self.status is FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET:
            source_ready = (
                self.source.identity_exact
                and self.source.legacy_layout_exact
                and self.source.other_session_count == 0
                and self.source.open_position_count == 0
                and self.source.semantic_invalid_row_count == 0
                and self.source.canonical_sha256 is not None
            )
            target_ready = (
                self.target.terminal_catalog_exact
                and self.target.migration_versions == (1, 2, 3, 4, 5, 6)
                and self.target.runtime_mode == "LEGACY"
                and self.target.runtime_generation == 0
                and not self.target.nonempty_relations
                and self.source.system_identifier != self.target.system_identifier
            )
            if not source_ready or not target_ready:
                raise ValueError("a ready receipt requires complete exact evidence")


class FreshTargetCutoverPreflightPort(Protocol):
    """Read-only port for one source-clone and fresh-target inspection."""

    def inspect(
        self, context: FreshTargetCutoverContext, /
    ) -> FreshTargetCutoverReceipt:
        """Return deterministic evidence without mutating either database."""
        ...


__all__ = [
    "FreshTargetCutoverBlocker",
    "FreshTargetCutoverContext",
    "FreshTargetCutoverPreflightPort",
    "FreshTargetCutoverReceipt",
    "FreshTargetCutoverSourceEvidence",
    "FreshTargetCutoverStatus",
    "FreshTargetCutoverTargetEvidence",
    "FreshTargetBootstrapIntent",
    "FreshTargetRoleManifest",
    "FreshTargetRelationEvidence",
]
