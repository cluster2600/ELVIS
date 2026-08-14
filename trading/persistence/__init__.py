"""Versioned persistence infrastructure for ELVIS."""

import importlib

_MIGRATION_EXPORTS = frozenset(
    {
        "Migration",
        "MigrationApplyError",
        "MigrationDriftError",
        "apply_migrations",
        "load_migrations",
    }
)


def __getattr__(name: str) -> object:
    """Load migration authority only when its facade symbols are requested."""

    if name not in _MIGRATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    migration_runner = importlib.import_module("trading.persistence.migration_runner")
    value = getattr(migration_runner, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _MIGRATION_EXPORTS)


__all__ = [
    "Migration",
    "MigrationApplyError",
    "MigrationDriftError",
    "apply_migrations",
    "load_migrations",
]
