"""Pure contracts for one dormant paper-runtime activation boundary."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from trading.application.paper_account_readiness import (
    PaperAccountReadinessAssessment,
    PaperAccountReadinessContext,
    PaperAccountReadinessDisposition,
)
from trading.domain._validation import protect_frozen_dataclass_state

_ACTIVATION_ID_MAX_LENGTH = 255
_POSTGRES_BIGINT_MAX = (1 << 63) - 1
_BASE_EXCEPTION_STATE_NAMES = frozenset(
    {
        "__cause__",
        "__context__",
        "__notes__",
        "__suppress_context__",
        "__traceback__",
    }
)


def _protect_frozen_activation_exception(exception_type: type) -> type:
    """Freeze business fields while retaining Python exception propagation state."""
    protect_frozen_dataclass_state(exception_type)
    frozen_setattr = exception_type.__setattr__

    def _setattr(instance: BaseException, name: str, value: object) -> None:
        if name in _BASE_EXCEPTION_STATE_NAMES:
            BaseException.__setattr__(instance, name, value)
            return
        frozen_setattr(instance, name, value)

    exception_type.__setattr__ = _setattr
    return exception_type


def _require_activation_id(value: object) -> None:
    if type(value) is not str:
        raise TypeError("activation_id must be a string")
    if not value or value != value.strip():
        raise ValueError("activation_id must be non-empty and trimmed")
    if len(value) > _ACTIVATION_ID_MAX_LENGTH:
        raise ValueError("activation_id must contain at most 255 characters")
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError("activation_id contains unsupported characters")


class PaperRuntimeActivationSource(str, Enum):
    """Control mode from which a locked activation may proceed."""

    LEGACY = "LEGACY"
    PAUSED = "PAUSED"


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperRuntimeActivationContext:
    """Caller-selected identity and expected control state for one activation."""

    readiness: PaperAccountReadinessContext
    activation_id: str
    source: PaperRuntimeActivationSource
    expected_runtime_generation: int

    def __post_init__(self) -> None:
        if type(self.readiness) is not PaperAccountReadinessContext:
            raise TypeError("readiness must be a PaperAccountReadinessContext")
        _require_activation_id(self.activation_id)
        if type(self.source) is not PaperRuntimeActivationSource:
            raise TypeError("source must be a PaperRuntimeActivationSource")
        if type(self.expected_runtime_generation) is not int:
            raise TypeError("expected_runtime_generation must be an integer")
        if not 0 <= self.expected_runtime_generation < _POSTGRES_BIGINT_MAX:
            raise ValueError(
                "expected_runtime_generation cannot produce a durable target"
            )
        if (
            self.source is PaperRuntimeActivationSource.LEGACY
            and self.expected_runtime_generation != 0
        ):
            raise ValueError("LEGACY activation must expect generation 0")
        if (
            self.source is PaperRuntimeActivationSource.PAUSED
            and self.expected_runtime_generation < 1
        ):
            raise ValueError("PAUSED activation must expect a positive generation")

    @property
    def target_runtime_generation(self) -> int:
        """Return the next positive activation epoch without mutating context."""
        return self.expected_runtime_generation + 1


class PaperRuntimeActivationDisposition(str, Enum):
    """Whether the exact activation was newly committed or rediscovered."""

    ACTIVATED = "ACTIVATED"
    REPLAYED = "REPLAYED"


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperRuntimeActivationReceipt:
    """One committed or exactly replayed activation epoch."""

    context: PaperRuntimeActivationContext
    runtime_generation: int
    disposition: PaperRuntimeActivationDisposition

    def __post_init__(self) -> None:
        if type(self.context) is not PaperRuntimeActivationContext:
            raise TypeError("context must be a PaperRuntimeActivationContext")
        if type(self.runtime_generation) is not int:
            raise TypeError("runtime_generation must be an integer")
        if self.runtime_generation != self.context.target_runtime_generation:
            raise ValueError("runtime_generation must equal the context target")
        if type(self.disposition) is not PaperRuntimeActivationDisposition:
            raise TypeError("disposition must be a PaperRuntimeActivationDisposition")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperRuntimeActivationBlocked:
    """Locked evidence which refused an activation mutation."""

    context: PaperRuntimeActivationContext
    assessment: PaperAccountReadinessAssessment

    def __post_init__(self) -> None:
        if type(self.context) is not PaperRuntimeActivationContext:
            raise TypeError("context must be a PaperRuntimeActivationContext")
        if type(self.assessment) is not PaperAccountReadinessAssessment:
            raise TypeError("assessment must be a PaperAccountReadinessAssessment")
        if self.assessment.context is not self.context.readiness:
            raise ValueError("assessment must retain the exact readiness context")
        if (
            self.assessment.disposition
            is PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
        ):
            raise ValueError("a prepared assessment cannot block activation")


PaperRuntimeActivationResult = (
    PaperRuntimeActivationReceipt | PaperRuntimeActivationBlocked
)


class PaperRuntimeActivationPort(Protocol):
    """Dormant application port for one locked activation transaction."""

    def activate(
        self,
        context: PaperRuntimeActivationContext,
        /,
    ) -> PaperRuntimeActivationResult:
        """Activate, exactly replay, or return locked blocking evidence."""
        ...


@_protect_frozen_activation_exception
@dataclass(frozen=True, slots=True)
class PaperRuntimeActivationBusy(RuntimeError):
    """Preserve an activation that could not acquire its required locks."""

    context: PaperRuntimeActivationContext

    def __post_init__(self) -> None:
        if type(self.context) is not PaperRuntimeActivationContext:
            raise TypeError("context must be a PaperRuntimeActivationContext")
        RuntimeError.__init__(self, "paper runtime activation is busy")

    def __reduce__(self) -> tuple[object, tuple[PaperRuntimeActivationContext]]:
        return (type(self), (self.context,))

    @property
    def activation_id(self) -> str:
        """Expose the stable retry identity."""
        return self.context.activation_id

    @property
    def requires_reconciliation(self) -> bool:
        """No activation mutation can precede failure to acquire its locks."""
        return False


@_protect_frozen_activation_exception
@dataclass(frozen=True, slots=True)
class PaperRuntimeActivationConflict(RuntimeError):
    """Preserve a stale expectation or conflicting activation identity."""

    context: PaperRuntimeActivationContext

    def __post_init__(self) -> None:
        if type(self.context) is not PaperRuntimeActivationContext:
            raise TypeError("context must be a PaperRuntimeActivationContext")
        RuntimeError.__init__(self, "paper runtime activation conflicts")

    def __reduce__(self) -> tuple[object, tuple[PaperRuntimeActivationContext]]:
        return (type(self), (self.context,))

    @property
    def activation_id(self) -> str:
        """Expose the stable conflicting activation identity."""
        return self.context.activation_id

    @property
    def requires_reconciliation(self) -> bool:
        """A stale or reused identity must be resolved before another attempt."""
        return True


@_protect_frozen_activation_exception
@dataclass(frozen=True, slots=True)
class PaperRuntimeActivationCommitUnknown(RuntimeError):
    """Preserve an activation whose commit acknowledgement was lost."""

    context: PaperRuntimeActivationContext

    def __post_init__(self) -> None:
        if type(self.context) is not PaperRuntimeActivationContext:
            raise TypeError("context must be a PaperRuntimeActivationContext")
        RuntimeError.__init__(self, "paper runtime activation commit is unknown")

    def __reduce__(self) -> tuple[object, tuple[PaperRuntimeActivationContext]]:
        return (type(self), (self.context,))

    @property
    def activation_id(self) -> str:
        """Expose the identity used to resolve the unacknowledged commit."""
        return self.context.activation_id

    @property
    def requires_reconciliation(self) -> bool:
        """The exact activation ID must be replayed before any new transition."""
        return True


__all__ = [
    "PaperRuntimeActivationBlocked",
    "PaperRuntimeActivationBusy",
    "PaperRuntimeActivationCommitUnknown",
    "PaperRuntimeActivationConflict",
    "PaperRuntimeActivationContext",
    "PaperRuntimeActivationDisposition",
    "PaperRuntimeActivationPort",
    "PaperRuntimeActivationReceipt",
    "PaperRuntimeActivationResult",
    "PaperRuntimeActivationSource",
]
