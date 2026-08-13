"""Pure contract for one dormant V1 legacy snapshot import."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from trading.application.fresh_target_cutover import (
    FreshTargetCutoverContext,
    FreshTargetCutoverReceipt,
)
from trading.domain._validation import protect_frozen_dataclass_state

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
_MAX_BATCH_SIZE = 512
_POSTGRES_INTEGER_MAX = (1 << 31) - 1


def _require_count(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_sha256(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_system_identifier(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_sequence_next(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= _POSTGRES_INTEGER_MAX:
        raise ValueError(f"{name} is outside the PostgreSQL integer sequence range")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class LegacySnapshotImportContext:
    """Pure authority and bounded-copy intent for one import attempt."""

    cutover_context: FreshTargetCutoverContext
    batch_size: int = _MAX_BATCH_SIZE

    def __post_init__(self) -> None:
        if type(self.cutover_context) is not FreshTargetCutoverContext:
            raise TypeError("cutover_context must be a FreshTargetCutoverContext")
        if type(self.batch_size) is not int:
            raise TypeError("batch_size must be an integer")
        if not 1 <= self.batch_size <= _MAX_BATCH_SIZE:
            raise ValueError("batch_size must be between 1 and 512")


class LegacySnapshotImportDisposition(str, Enum):
    """Durable result after exact target and sequence readback."""

    IMPORTED = "IMPORTED"
    REPLAYED = "REPLAYED"


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class LegacySnapshotRelationReceipt:
    """Exact row and sequence evidence for one copied V1 relation."""

    name: str
    row_count: int
    pk_min: int | None
    pk_max: int | None
    sha256: str
    source_sequence_next: int
    target_sequence_next: int

    def __post_init__(self) -> None:
        if self.name not in _LEGACY_RELATIONS:
            raise ValueError("name must identify an exact legacy relation")
        _require_count("row_count", self.row_count)
        _require_sha256("sha256", self.sha256)
        _require_sequence_next("source_sequence_next", self.source_sequence_next)
        _require_sequence_next("target_sequence_next", self.target_sequence_next)
        if self.row_count == 0:
            if self.pk_min is not None or self.pk_max is not None:
                raise ValueError("an empty relation cannot have primary-key bounds")
        else:
            if type(self.pk_min) is not int or type(self.pk_max) is not int:
                raise TypeError("non-empty relation bounds must be integers")
            if not 1 <= self.pk_min <= self.pk_max <= _POSTGRES_INTEGER_MAX:
                raise ValueError("relation bounds are outside the valid ID range")
        minimum_target = 1 if self.pk_max is None else self.pk_max + 1
        if minimum_target > _POSTGRES_INTEGER_MAX:
            raise ValueError("relation has exhausted its PostgreSQL integer sequence")
        if self.target_sequence_next != max(self.source_sequence_next, minimum_target):
            raise ValueError("target_sequence_next is not the safe normalized value")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class LegacySnapshotImportReceipt:
    """Exact but stale evidence; this never authorizes runtime activation."""

    context: LegacySnapshotImportContext
    disposition: LegacySnapshotImportDisposition
    source_system_identifier: int
    target_system_identifier: int
    source_canonical_sha256: str
    relations: tuple[LegacySnapshotRelationReceipt, ...]
    target_exact: bool = True
    runtime_activation_authorized: bool = False
    stale_on_return: bool = True
    snapshot_authoritative: bool = False

    def __post_init__(self) -> None:
        if type(self.context) is not LegacySnapshotImportContext:
            raise TypeError("context must be a LegacySnapshotImportContext")
        if type(self.disposition) is not LegacySnapshotImportDisposition:
            raise TypeError("disposition must be a LegacySnapshotImportDisposition")
        _require_system_identifier(
            "source_system_identifier", self.source_system_identifier
        )
        _require_system_identifier(
            "target_system_identifier", self.target_system_identifier
        )
        if self.source_system_identifier == self.target_system_identifier:
            raise ValueError("source and target must be different PostgreSQL clusters")
        _require_sha256("source_canonical_sha256", self.source_canonical_sha256)
        if type(self.relations) is not tuple or any(
            type(value) is not LegacySnapshotRelationReceipt for value in self.relations
        ):
            raise TypeError(
                "relations must contain LegacySnapshotRelationReceipt values"
            )
        if tuple(value.name for value in self.relations) != _LEGACY_RELATIONS:
            raise ValueError("relations must be the exact ordered legacy set")
        if self.target_exact is not True:
            raise ValueError("an import receipt requires exact target readback")
        if self.runtime_activation_authorized is not False:
            raise ValueError("an import receipt cannot authorize runtime activation")
        if self.stale_on_return is not True or self.snapshot_authoritative is not False:
            raise ValueError("import evidence must remain stale and non-authoritative")


class LegacySnapshotImportPort(Protocol):
    """Port for one bounded, replay-safe legacy snapshot import."""

    def import_snapshot(
        self,
        context: LegacySnapshotImportContext,
        preflight_receipt: FreshTargetCutoverReceipt,
        /,
    ) -> LegacySnapshotImportReceipt:
        """Import or exactly replay one preflight-bound V1 snapshot."""
        ...


__all__ = [
    "LegacySnapshotImportContext",
    "LegacySnapshotImportDisposition",
    "LegacySnapshotImportPort",
    "LegacySnapshotImportReceipt",
    "LegacySnapshotRelationReceipt",
]
