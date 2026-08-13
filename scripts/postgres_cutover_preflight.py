"""One-shot, read-only CLI for a fresh PostgreSQL target preflight."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn, TextIO

import psycopg2

from trading.application.fresh_target_cutover import (
    FreshTargetBootstrapIntent,
    FreshTargetCutoverContext,
    FreshTargetCutoverReceipt,
    FreshTargetCutoverStatus,
    FreshTargetRoleManifest,
)
from trading.persistence.postgres_bootstrap import (
    PostgresBootstrapInputError,
    PostgresBootstrapStorageError,
)
from trading.persistence.postgres_cutover_preflight import (
    PostgresCutoverPreflight,
    PostgresCutoverPreflightInputError,
    PostgresCutoverPreflightStorageError,
)

_APPLICATION_NAME = "elvis-fresh-target-preflight-v1"
_CONNECT_TIMEOUT_SECONDS = 5
_MAX_CONFIG_BYTES = 65_536
_SERVICE_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,62}")
_ROLE_KEYS = (
    "schema_owner",
    "migrator",
    "legacy_runtime",
    "atomic_runtime",
    "activation",
    "readiness",
    "trainer",
)
_TOP_LEVEL_KEYS = {"schema_version", "source", "target"}
_SOURCE_KEYS = {"expected_database", "expected_role", "service"}
_TARGET_KEYS = {"admin_service", "bootstrap_context"}
_BOOTSTRAP_CONTEXT_KEYS = {
    "expected_database",
    "admin_role",
    "roles",
    "adoption",
}

_EXIT_READY = 0
_EXIT_INPUT = 2
_EXIT_STORAGE = 20
_EXIT_BLOCKED = 21
_EXIT_INTERNAL = 70


class _CliInputError(ValueError):
    """Secret-free signal for invalid invocation or configuration."""


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _CliInputError("invalid invocation")


def _reject_json_constant(value: str) -> NoReturn:
    del value
    raise _CliInputError("invalid configuration")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _CliInputError("invalid configuration")
        result[key] = value
    return result


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as config_file:
            payload = config_file.read(_MAX_CONFIG_BYTES + 1)
        if len(payload) > _MAX_CONFIG_BYTES:
            raise _CliInputError("invalid configuration")
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except _CliInputError:
        raise
    except OSError, UnicodeError, ValueError, RecursionError:
        document = None
    if type(document) is not dict:
        raise _CliInputError("invalid configuration")
    return document


def _require_exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise _CliInputError("invalid configuration")


def _require_text(value: object) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise _CliInputError("invalid configuration")
    return value


def _require_service(value: object) -> str:
    if type(value) is not str or _SERVICE_IDENTIFIER.fullmatch(value) is None:
        raise _CliInputError("invalid configuration")
    return value


def _parse_roles(value: object) -> FreshTargetRoleManifest:
    if type(value) is not dict:
        raise _CliInputError("invalid configuration")
    _require_exact_keys(value, set(_ROLE_KEYS))
    if any(type(value[key]) is not str for key in _ROLE_KEYS):
        raise _CliInputError("invalid configuration")
    return FreshTargetRoleManifest(**{key: value[key] for key in _ROLE_KEYS})


def _parse_config(
    document: dict[str, Any],
) -> tuple[FreshTargetCutoverContext, str, str]:
    _require_exact_keys(document, _TOP_LEVEL_KEYS)
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise _CliInputError("invalid configuration")
    source = document["source"]
    target = document["target"]
    if type(source) is not dict or type(target) is not dict:
        raise _CliInputError("invalid configuration")
    _require_exact_keys(source, _SOURCE_KEYS)
    _require_exact_keys(target, _TARGET_KEYS)
    bootstrap = target["bootstrap_context"]
    if type(bootstrap) is not dict:
        raise _CliInputError("invalid configuration")
    _require_exact_keys(bootstrap, _BOOTSTRAP_CONTEXT_KEYS)
    if bootstrap["adoption"] is not None:
        raise _CliInputError("invalid configuration")
    bootstrap_intent = FreshTargetBootstrapIntent(
        expected_database=_require_text(bootstrap["expected_database"]),
        admin_role=_require_text(bootstrap["admin_role"]),
        roles=_parse_roles(bootstrap["roles"]),
    )
    context = FreshTargetCutoverContext(
        source_expected_database=_require_text(source["expected_database"]),
        source_expected_role=_require_text(source["expected_role"]),
        target_bootstrap_intent=bootstrap_intent,
    )
    return (
        context,
        _require_service(source["service"]),
        _require_service(target["admin_service"]),
    )


def _connection_factory_for_service(service_name: str) -> Callable[[], object]:
    def connect() -> object:
        return psycopg2.connect(
            service=service_name,
            application_name=_APPLICATION_NAME,
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        )

    return connect


def _write_json(stream: TextIO, payload: dict[str, object]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    stream.write("\n")


def _write_receipt(receipt: FreshTargetCutoverReceipt) -> None:
    _write_json(
        sys.stdout,
        {
            "status": receipt.status.value,
            "blockers": [value.value for value in receipt.blockers],
            "stale_on_return": receipt.stale_on_return,
            "snapshot_authoritative": receipt.snapshot_authoritative,
            "source": {
                "system_identifier": str(receipt.source.system_identifier),
                "relations": [
                    {
                        "name": relation.name,
                        "row_count": relation.row_count,
                        "pk_min": relation.pk_min,
                        "pk_max": relation.pk_max,
                        "sha256": relation.sha256,
                    }
                    for relation in receipt.source.relations
                ],
                "other_session_count": receipt.source.other_session_count,
                "open_position_count": receipt.source.open_position_count,
                "semantic_invalid_row_count": (
                    receipt.source.semantic_invalid_row_count
                ),
                "canonical_sha256": receipt.source.canonical_sha256,
                "legacy_layout_exact": receipt.source.legacy_layout_exact,
                "identity_exact": receipt.source.identity_exact,
            },
            "target": {
                "system_identifier": str(receipt.target.system_identifier),
                "terminal_catalog_exact": receipt.target.terminal_catalog_exact,
                "migration_versions": list(receipt.target.migration_versions),
                "runtime_mode": receipt.target.runtime_mode,
                "runtime_generation": receipt.target.runtime_generation,
                "nonempty_relations": list(receipt.target.nonempty_relations),
            },
        },
    )


def _write_error(code: str) -> None:
    _write_json(sys.stdout, {"status": "ERROR", "code": code})


def _argument_parser() -> _StrictArgumentParser:
    parser = _StrictArgumentParser(
        prog="python -m scripts.postgres_cutover_preflight",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--confirm-stopped-source-clone", action="store_true")
    parser.add_argument("--confirm-exclusive-database-window", action="store_true")
    return parser


def _run(
    argv: Sequence[str] | None,
    service_connection_factory: Callable[[str], Callable[[], object]],
) -> FreshTargetCutoverReceipt:
    arguments = _argument_parser().parse_args(argv)
    if not (
        arguments.inspect
        and arguments.confirm_stopped_source_clone
        and arguments.confirm_exclusive_database_window
    ):
        raise _CliInputError("invalid invocation")
    try:
        context, source_service, target_service = _parse_config(
            _read_config(Path(arguments.config))
        )
    except _CliInputError:
        raise
    except PostgresBootstrapInputError, TypeError, ValueError:
        context = None
    if context is None:
        raise _CliInputError("invalid configuration")
    source_factory = service_connection_factory(source_service)
    target_factory = service_connection_factory(target_service)
    preflight = PostgresCutoverPreflight(source_factory, target_factory)
    return preflight.inspect(context)


def main(
    argv: Sequence[str] | None = None,
    *,
    service_connection_factory: Callable[[str], Callable[[], object]] | None = None,
) -> int:
    """Run exactly one inspection and emit one compact, secret-free JSON value."""
    try:
        receipt = _run(
            argv,
            service_connection_factory or _connection_factory_for_service,
        )
    except (
        _CliInputError,
        PostgresBootstrapInputError,
        PostgresCutoverPreflightInputError,
    ):
        _write_error("INPUT")
        return _EXIT_INPUT
    except (
        PostgresBootstrapStorageError,
        PostgresCutoverPreflightStorageError,
    ):
        _write_error("STORAGE")
        return _EXIT_STORAGE
    except Exception:
        _write_error("INTERNAL")
        return _EXIT_INTERNAL
    _write_receipt(receipt)
    if receipt.status is FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET:
        return _EXIT_READY
    return _EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
