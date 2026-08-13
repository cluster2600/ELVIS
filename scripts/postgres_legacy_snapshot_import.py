"""One-shot CLI for a bounded, replay-safe legacy snapshot import."""

from __future__ import annotations

import argparse
import json
import re
import stat
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn, TextIO

import psycopg2

from trading.application.fresh_target_cutover import (
    FreshTargetBootstrapIntent,
    FreshTargetCutoverBlocker,
    FreshTargetCutoverContext,
    FreshTargetCutoverReceipt,
    FreshTargetCutoverSourceEvidence,
    FreshTargetCutoverStatus,
    FreshTargetCutoverTargetEvidence,
    FreshTargetRelationEvidence,
    FreshTargetRoleManifest,
)
from trading.application.legacy_snapshot_import import (
    LegacySnapshotImportContext,
    LegacySnapshotImportReceipt,
)
from trading.persistence.postgres_bootstrap import PostgresBootstrapInputError
from trading.persistence.postgres_legacy_snapshot_import import (
    PostgresLegacySnapshotImport,
    PostgresLegacySnapshotImportBusyError,
    PostgresLegacySnapshotImportCommitUnknown,
    PostgresLegacySnapshotImportConflict,
    PostgresLegacySnapshotImportInputError,
    PostgresLegacySnapshotImportStorageError,
)

_APPLICATION_NAME = "elvis-legacy-snapshot-import-v1"
_CONNECT_TIMEOUT_SECONDS = 5
_MAX_FILE_BYTES = 65_536
_SERVICE_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,62}")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROLE_KEYS = (
    "schema_owner",
    "migrator",
    "legacy_runtime",
    "atomic_runtime",
    "activation",
    "readiness",
    "trainer",
)
_RELATION_NAMES = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)

_EXIT_OK = 0
_EXIT_INPUT = 2
_EXIT_STORAGE = 20
_EXIT_BUSY = 22
_EXIT_CONFLICT = 23
_EXIT_COMMIT_UNKNOWN = 24
_EXIT_INTERNAL = 70


class _CliInputError(ValueError):
    """Secret-free invalid invocation or file signal."""


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _CliInputError("invalid invocation")


def _reject_json_constant(value: str) -> NoReturn:
    del value
    raise _CliInputError("invalid JSON")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _CliInputError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise _CliInputError("invalid configuration file")
        with path.open("rb") as input_file:
            payload = input_file.read(_MAX_FILE_BYTES + 1)
        if len(payload) > _MAX_FILE_BYTES:
            raise _CliInputError("input file is too large")
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except _CliInputError:
        raise
    except OSError, UnicodeError, ValueError, RecursionError:
        raise _CliInputError("invalid input file") from None
    if type(document) is not dict:
        raise _CliInputError("input file must contain an object")
    return document


def _exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise _CliInputError("invalid object shape")


def _text(value: object) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise _CliInputError("invalid text")
    return value


def _service(value: object) -> str:
    if type(value) is not str or _SERVICE_IDENTIFIER.fullmatch(value) is None:
        raise _CliInputError("invalid service")
    return value


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _CliInputError("invalid integer")
    return value


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    return _integer(value, minimum=1)


def _sha256(value: object) -> str:
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        raise _CliInputError("invalid fingerprint")
    return value


def _parse_roles(value: object) -> FreshTargetRoleManifest:
    if type(value) is not dict:
        raise _CliInputError("invalid roles")
    _exact_keys(value, set(_ROLE_KEYS))
    return FreshTargetRoleManifest(**{name: _text(value[name]) for name in _ROLE_KEYS})


def _parse_config(
    document: dict[str, Any],
) -> tuple[LegacySnapshotImportContext, tuple[str, str, str]]:
    _exact_keys(document, {"schema_version", "batch_size", "source", "target"})
    if document["schema_version"] != 1 or type(document["schema_version"]) is not int:
        raise _CliInputError("invalid schema version")
    batch_size = _integer(document["batch_size"], minimum=1)
    if batch_size > 512:
        raise _CliInputError("invalid batch size")
    source = document["source"]
    target = document["target"]
    if type(source) is not dict or type(target) is not dict:
        raise _CliInputError("invalid connection intent")
    _exact_keys(source, {"service", "expected_database", "expected_role"})
    _exact_keys(target, {"admin_service", "migrator_service", "bootstrap_context"})
    bootstrap = target["bootstrap_context"]
    if type(bootstrap) is not dict:
        raise _CliInputError("invalid bootstrap context")
    _exact_keys(
        bootstrap,
        {"expected_database", "admin_role", "roles", "adoption"},
    )
    if bootstrap["adoption"] is not None:
        raise _CliInputError("adoption is forbidden")
    context = FreshTargetCutoverContext(
        source_expected_database=_text(source["expected_database"]),
        source_expected_role=_text(source["expected_role"]),
        target_bootstrap_intent=FreshTargetBootstrapIntent(
            expected_database=_text(bootstrap["expected_database"]),
            admin_role=_text(bootstrap["admin_role"]),
            roles=_parse_roles(bootstrap["roles"]),
        ),
    )
    services = (
        _service(source["service"]),
        _service(target["admin_service"]),
        _service(target["migrator_service"]),
    )
    if len(set(services)) != 3:
        raise _CliInputError("services must be distinct")
    return LegacySnapshotImportContext(context, batch_size), services


def _parse_relation(value: object) -> FreshTargetRelationEvidence:
    if type(value) is not dict:
        raise _CliInputError("invalid relation evidence")
    _exact_keys(value, {"name", "row_count", "pk_min", "pk_max", "sha256"})
    return FreshTargetRelationEvidence(
        name=_text(value["name"]),
        row_count=_integer(value["row_count"]),
        pk_min=_optional_integer(value["pk_min"]),
        pk_max=_optional_integer(value["pk_max"]),
        sha256=_sha256(value["sha256"]),
    )


def _parse_receipt(document: dict[str, Any]) -> FreshTargetCutoverReceipt:
    _exact_keys(
        document,
        {
            "status",
            "blockers",
            "stale_on_return",
            "snapshot_authoritative",
            "source",
            "target",
        },
    )
    if document["status"] != FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET.value:
        raise _CliInputError("receipt is not ready")
    if document["blockers"] != []:
        raise _CliInputError("receipt contains blockers")
    if (
        document["stale_on_return"] is not True
        or document["snapshot_authoritative"] is not False
    ):
        raise _CliInputError("receipt authority fields are invalid")
    source = document["source"]
    target = document["target"]
    if type(source) is not dict or type(target) is not dict:
        raise _CliInputError("invalid receipt evidence")
    _exact_keys(
        source,
        {
            "system_identifier",
            "relations",
            "other_session_count",
            "open_position_count",
            "semantic_invalid_row_count",
            "canonical_sha256",
            "legacy_layout_exact",
            "identity_exact",
        },
    )
    _exact_keys(
        target,
        {
            "system_identifier",
            "terminal_catalog_exact",
            "migration_versions",
            "runtime_mode",
            "runtime_generation",
            "nonempty_relations",
        },
    )
    relations_value = source["relations"]
    if type(relations_value) is not list:
        raise _CliInputError("invalid relation evidence")
    relations = tuple(_parse_relation(value) for value in relations_value)
    if tuple(value.name for value in relations) != _RELATION_NAMES:
        raise _CliInputError("receipt relation order is invalid")
    if (
        type(target["migration_versions"]) is not list
        or type(target["nonempty_relations"]) is not list
    ):
        raise _CliInputError("invalid target evidence")
    return FreshTargetCutoverReceipt(
        status=FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET,
        blockers=tuple(
            FreshTargetCutoverBlocker(value) for value in document["blockers"]
        ),
        source=FreshTargetCutoverSourceEvidence(
            system_identifier=(
                _integer(int(source["system_identifier"]), minimum=1)
                if type(source["system_identifier"]) is str
                and source["system_identifier"].isdigit()
                else (_ for _ in ()).throw(_CliInputError("invalid system identifier"))
            ),
            relations=relations,
            other_session_count=_integer(source["other_session_count"]),
            open_position_count=_integer(source["open_position_count"]),
            semantic_invalid_row_count=_integer(source["semantic_invalid_row_count"]),
            canonical_sha256=_sha256(source["canonical_sha256"]),
            legacy_layout_exact=source["legacy_layout_exact"],
            identity_exact=source["identity_exact"],
        ),
        target=FreshTargetCutoverTargetEvidence(
            system_identifier=(
                _integer(int(target["system_identifier"]), minimum=1)
                if type(target["system_identifier"]) is str
                and target["system_identifier"].isdigit()
                else (_ for _ in ()).throw(_CliInputError("invalid system identifier"))
            ),
            terminal_catalog_exact=target["terminal_catalog_exact"],
            migration_versions=tuple(target["migration_versions"]),
            runtime_mode=target["runtime_mode"],
            runtime_generation=target["runtime_generation"],
            nonempty_relations=tuple(target["nonempty_relations"]),
        ),
        stale_on_return=True,
        snapshot_authoritative=False,
    )


def _connection_factory_for_service(service_name: str) -> Callable[[], object]:
    def connect() -> object:
        return psycopg2.connect(
            service=service_name,
            application_name=_APPLICATION_NAME,
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        )

    return connect


def _write_json(stream: TextIO, value: dict[str, object]) -> None:
    stream.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")))
    stream.write("\n")


def _write_error(code: str) -> None:
    _write_json(sys.stdout, {"status": "ERROR", "code": code})


def _write_receipt(receipt: LegacySnapshotImportReceipt) -> None:
    _write_json(
        sys.stdout,
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
            "runtime_activation_authorized": receipt.runtime_activation_authorized,
            "stale_on_return": receipt.stale_on_return,
            "snapshot_authoritative": receipt.snapshot_authoritative,
        },
    )


def _argument_parser() -> _StrictArgumentParser:
    parser = _StrictArgumentParser(
        prog="python -m scripts.postgres_legacy_snapshot_import",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--import-snapshot", action="store_true")
    parser.add_argument("--confirm-stopped-source-clone", action="store_true")
    parser.add_argument("--confirm-exclusive-database-window", action="store_true")
    parser.add_argument("--confirm-disposable-target", action="store_true")
    return parser


def _run(
    argv: Sequence[str] | None,
    service_connection_factory: Callable[[str], Callable[[], object]],
) -> LegacySnapshotImportReceipt:
    arguments = _argument_parser().parse_args(argv)
    if not (
        arguments.import_snapshot
        and arguments.confirm_stopped_source_clone
        and arguments.confirm_exclusive_database_window
        and arguments.confirm_disposable_target
    ):
        raise _CliInputError("all confirmations are required")
    context, services = _parse_config(_read_json(Path(arguments.config)))
    preflight = _parse_receipt(_read_json(Path(arguments.preflight_receipt)))
    factories = tuple(service_connection_factory(service) for service in services)
    if len({id(factory) for factory in factories}) != 3:
        raise _CliInputError("resolved services must be distinct")
    return PostgresLegacySnapshotImport(*factories).import_snapshot(context, preflight)


def main(
    argv: Sequence[str] | None = None,
    *,
    service_connection_factory: Callable[[str], Callable[[], object]] | None = None,
) -> int:
    """Run one import and emit exactly one compact secret-free JSON value."""
    try:
        receipt = _run(
            argv,
            service_connection_factory or _connection_factory_for_service,
        )
    except (
        _CliInputError,
        PostgresBootstrapInputError,
        PostgresLegacySnapshotImportInputError,
        TypeError,
        ValueError,
    ):
        _write_error("INPUT")
        return _EXIT_INPUT
    except PostgresLegacySnapshotImportBusyError:
        _write_error("BUSY")
        return _EXIT_BUSY
    except PostgresLegacySnapshotImportConflict:
        _write_error("CONFLICT")
        return _EXIT_CONFLICT
    except PostgresLegacySnapshotImportCommitUnknown:
        _write_error("COMMIT_UNKNOWN")
        return _EXIT_COMMIT_UNKNOWN
    except PostgresLegacySnapshotImportStorageError:
        _write_error("STORAGE")
        return _EXIT_STORAGE
    except Exception:
        _write_error("INTERNAL")
        return _EXIT_INTERNAL
    _write_receipt(receipt)
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
