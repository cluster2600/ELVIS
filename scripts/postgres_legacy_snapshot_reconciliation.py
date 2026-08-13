"""One-shot CLI for a read-only legacy-snapshot reconciliation review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NoReturn, TextIO

import psycopg2

from trading.application.fresh_target_cutover import (
    FreshTargetBootstrapIntent,
    FreshTargetCutoverContext,
    FreshTargetRoleManifest,
)
from trading.application.legacy_snapshot_import import (
    LegacySnapshotImportContext,
    LegacySnapshotImportDisposition,
    LegacySnapshotImportReceipt,
    LegacySnapshotRelationReceipt,
)
from trading.application.legacy_snapshot_reconciliation import (
    LegacySnapshotReconciliationContext,
    LegacySnapshotReconciliationDisposition,
    LegacySnapshotReconciliationReceipt,
    legacy_snapshot_import_receipt_sha256,
)
from trading.persistence.postgres_legacy_snapshot_reconciliation import (
    PostgresLegacySnapshotReconciliation,
    PostgresLegacySnapshotReconciliationConflict,
    PostgresLegacySnapshotReconciliationInputError,
    PostgresLegacySnapshotReconciliationStorageError,
)

_APPLICATION_NAME = "elvis-legacy-snapshot-reconciliation-v1"
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

_EXIT_INPUT = 2
_EXIT_DECISION_REQUIRED = 10
_EXIT_STORAGE = 20
_EXIT_BLOCKED = 21
_EXIT_CONFLICT = 23
_EXIT_INTERNAL = 70


class _CliInputError(ValueError):
    """Secret-free invalid invocation or input-file signal."""


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
    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise _CliInputError("safe input-file access is unavailable")
        descriptor = os.open(path, flags | nofollow)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _CliInputError("invalid input file")
        with os.fdopen(descriptor, "rb", closefd=True) as input_file:
            descriptor = None
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
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if type(document) is not dict:
        raise _CliInputError("input file must contain an object")
    return document


def _canonical_document_sha256(document: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except RecursionError, TypeError, ValueError:
        raise _CliInputError("input document cannot be canonicalized") from None
    return hashlib.sha256(encoded).hexdigest()


def _exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise _CliInputError("invalid object shape")


def _text(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
        or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    ):
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


def _system_identifier(value: object) -> int:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdigit()
        or str(int(value)) != value
    ):
        raise _CliInputError("invalid system identifier")
    return _integer(int(value), minimum=1)


def _sha256(value: object) -> str:
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        raise _CliInputError("invalid fingerprint")
    return value


def _decimal(value: object) -> Decimal:
    encoded = _text(value)
    try:
        decoded = Decimal(encoded)
    except InvalidOperation:
        raise _CliInputError("invalid decimal") from None
    if not decoded.is_finite() or str(decoded) != encoded:
        raise _CliInputError("invalid decimal")
    return decoded


def _parse_roles(value: object) -> FreshTargetRoleManifest:
    if type(value) is not dict:
        raise _CliInputError("invalid roles")
    _exact_keys(value, set(_ROLE_KEYS))
    return FreshTargetRoleManifest(**{name: _text(value[name]) for name in _ROLE_KEYS})


def _parse_config(
    document: dict[str, Any],
    config_document_sha256: str,
    import_receipt_sha256: str,
) -> tuple[LegacySnapshotReconciliationContext, tuple[str, str]]:
    _exact_keys(
        document,
        {"schema_version", "batch_size", "source", "target", "opening"},
    )
    if document["schema_version"] != 1 or type(document["schema_version"]) is not int:
        raise _CliInputError("invalid schema version")
    batch_size = _integer(document["batch_size"], minimum=1)
    if batch_size > 512:
        raise _CliInputError("invalid batch size")
    source = document["source"]
    target = document["target"]
    opening = document["opening"]
    if (
        type(source) is not dict
        or type(target) is not dict
        or type(opening) is not dict
    ):
        raise _CliInputError("invalid reconciliation intent")
    _exact_keys(source, {"expected_database", "expected_role"})
    _exact_keys(
        target,
        {"admin_service", "readiness_service", "bootstrap_context"},
    )
    bootstrap = target["bootstrap_context"]
    if type(bootstrap) is not dict:
        raise _CliInputError("invalid bootstrap context")
    _exact_keys(
        bootstrap,
        {"expected_database", "admin_role", "roles", "adoption"},
    )
    if bootstrap["adoption"] is not None:
        raise _CliInputError("adoption is forbidden")
    _exact_keys(
        opening,
        {
            "execution_scope",
            "account_key",
            "owner_generation",
            "collateral_asset",
            "margin_quantum_decimal",
            "hypothesis_starting_collateral_decimal",
        },
    )
    cutover_context = FreshTargetCutoverContext(
        source_expected_database=_text(source["expected_database"]),
        source_expected_role=_text(source["expected_role"]),
        target_bootstrap_intent=FreshTargetBootstrapIntent(
            expected_database=_text(bootstrap["expected_database"]),
            admin_role=_text(bootstrap["admin_role"]),
            roles=_parse_roles(bootstrap["roles"]),
        ),
    )
    context = LegacySnapshotReconciliationContext(
        import_context=LegacySnapshotImportContext(
            cutover_context,
            batch_size,
        ),
        config_document_sha256=config_document_sha256,
        import_receipt_sha256=import_receipt_sha256,
        execution_scope=_text(opening["execution_scope"]),
        account_key=_text(opening["account_key"]),
        owner_generation=_integer(opening["owner_generation"], minimum=1),
        collateral_asset=_text(opening["collateral_asset"]),
        margin_quantum=_decimal(opening["margin_quantum_decimal"]),
        hypothesis_starting_collateral=_decimal(
            opening["hypothesis_starting_collateral_decimal"]
        ),
    )
    services = (
        _service(target["admin_service"]),
        _service(target["readiness_service"]),
    )
    if len(set(services)) != 2:
        raise _CliInputError("services must be distinct")
    return context, services


def _parse_relation(value: object) -> LegacySnapshotRelationReceipt:
    if type(value) is not dict:
        raise _CliInputError("invalid relation evidence")
    _exact_keys(
        value,
        {
            "name",
            "row_count",
            "pk_min",
            "pk_max",
            "sha256",
            "source_sequence_next",
            "target_sequence_next",
        },
    )
    return LegacySnapshotRelationReceipt(
        name=_text(value["name"]),
        row_count=_integer(value["row_count"]),
        pk_min=_optional_integer(value["pk_min"]),
        pk_max=_optional_integer(value["pk_max"]),
        sha256=_sha256(value["sha256"]),
        source_sequence_next=_integer(
            value["source_sequence_next"],
            minimum=1,
        ),
        target_sequence_next=_integer(
            value["target_sequence_next"],
            minimum=1,
        ),
    )


def _parse_import_receipt(
    document: dict[str, Any],
    context: LegacySnapshotReconciliationContext,
) -> LegacySnapshotImportReceipt:
    _exact_keys(
        document,
        {
            "status",
            "source_system_identifier",
            "target_system_identifier",
            "source_canonical_sha256",
            "relations",
            "target_exact",
            "runtime_activation_authorized",
            "stale_on_return",
            "snapshot_authoritative",
        },
    )
    try:
        disposition = LegacySnapshotImportDisposition(document["status"])
    except TypeError, ValueError:
        raise _CliInputError("invalid import disposition") from None
    if (
        document["target_exact"] is not True
        or document["runtime_activation_authorized"] is not False
        or document["stale_on_return"] is not True
        or document["snapshot_authoritative"] is not False
    ):
        raise _CliInputError("invalid import authority fields")
    relations_value = document["relations"]
    if type(relations_value) is not list:
        raise _CliInputError("invalid relation evidence")
    relations = tuple(_parse_relation(value) for value in relations_value)
    if tuple(value.name for value in relations) != _RELATION_NAMES:
        raise _CliInputError("invalid relation order")
    return LegacySnapshotImportReceipt(
        context=context.import_context,
        disposition=disposition,
        source_system_identifier=_system_identifier(
            document["source_system_identifier"]
        ),
        target_system_identifier=_system_identifier(
            document["target_system_identifier"]
        ),
        source_canonical_sha256=_sha256(document["source_canonical_sha256"]),
        relations=relations,
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
    stream.write(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    stream.write("\n")


def _write_error(code: str) -> None:
    _write_json(sys.stdout, {"status": "ERROR", "code": code})


def _balance_json(balance: object) -> dict[str, object]:
    return {
        "asset": balance.asset,
        "available_decimal": str(balance.available),
        "reserved_decimal": str(balance.reserved),
    }


def _write_receipt(receipt: LegacySnapshotReconciliationReceipt) -> None:
    evidence = receipt.evidence
    _write_json(
        sys.stdout,
        {
            "status": receipt.disposition.value,
            "import_disposition": receipt.import_receipt.disposition.value,
            "declared_source_system_identifier": str(
                receipt.import_receipt.source_system_identifier
            ),
            "target_system_identifier": str(receipt.target_system_identifier),
            "source_canonical_sha256": receipt.source_canonical_sha256,
            "config_document_sha256": receipt.config_document_sha256,
            "import_receipt_sha256": receipt.import_receipt_sha256,
            "findings": [value.kind.value for value in receipt.findings],
            "evidence": {
                "reset_timestamp": evidence.reset_timestamp,
                "hypothesis_realised_pnl_decimal": str(
                    evidence.hypothesis_realised_pnl
                ),
                "hypothesis_trade_fees_decimal": str(evidence.hypothesis_trade_fees),
                "hypothesis_liquidation_fees_decimal": str(
                    evidence.hypothesis_liquidation_fees
                ),
                "candidates": [
                    {
                        "source": candidate.source.value,
                        "available": candidate.available,
                        "balances": [
                            _balance_json(balance) for balance in candidate.balances
                        ],
                        "opening_payload_sha256": (candidate.opening_payload_sha256),
                    }
                    for candidate in evidence.candidates
                ],
            },
            "account_opening_authorized": receipt.account_opening_authorized,
            "account_provisioning_authorized": (
                receipt.account_provisioning_authorized
            ),
            "runtime_activation_authorized": (receipt.runtime_activation_authorized),
            "stale_on_return": receipt.stale_on_return,
            "snapshot_authoritative": receipt.snapshot_authoritative,
            "coherent_snapshot_observed": receipt.coherent_snapshot_observed,
            "source_provenance_authenticated": (
                receipt.source_provenance_authenticated
            ),
            "target_observations_authenticated": (
                receipt.target_observations_authenticated
            ),
            "database_window_enforced": receipt.database_window_enforced,
        },
    )


def _argument_parser() -> _StrictArgumentParser:
    parser = _StrictArgumentParser(
        prog="python -m scripts.postgres_legacy_snapshot_reconciliation",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--import-receipt", required=True)
    parser.add_argument("--assess", action="store_true")
    parser.add_argument(
        "--confirm-reviewed-database-window",
        action="store_true",
    )
    parser.add_argument("--confirm-disposable-target", action="store_true")
    return parser


def _run(
    argv: Sequence[str] | None,
    service_connection_factory: Callable[[str], Callable[[], object]],
) -> LegacySnapshotReconciliationReceipt:
    arguments = _argument_parser().parse_args(argv)
    if not (
        arguments.assess
        and arguments.confirm_reviewed_database_window
        and arguments.confirm_disposable_target
    ):
        raise _CliInputError("all confirmations are required")
    config_document = _read_json(Path(arguments.config))
    import_receipt_document = _read_json(Path(arguments.import_receipt))
    context, services = _parse_config(
        config_document,
        _canonical_document_sha256(config_document),
        _canonical_document_sha256(import_receipt_document),
    )
    import_receipt = _parse_import_receipt(
        import_receipt_document,
        context,
    )
    if context.import_receipt_sha256 != legacy_snapshot_import_receipt_sha256(
        import_receipt
    ):
        raise _CliInputError("import receipt is not canonical")
    factories = tuple(service_connection_factory(service) for service in services)
    if len({id(factory) for factory in factories}) != 2:
        raise _CliInputError("resolved services must be distinct")
    return PostgresLegacySnapshotReconciliation(*factories).reconcile(
        context,
        import_receipt,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    service_connection_factory: Callable[[str], Callable[[], object]] | None = None,
) -> int:
    """Emit exactly one compact, secret-free JSON result on stdout."""
    try:
        receipt = _run(
            argv,
            service_connection_factory or _connection_factory_for_service,
        )
    except (
        _CliInputError,
        PostgresLegacySnapshotReconciliationInputError,
        TypeError,
        ValueError,
    ):
        _write_error("INPUT")
        return _EXIT_INPUT
    except PostgresLegacySnapshotReconciliationConflict:
        _write_error("CONFLICT")
        return _EXIT_CONFLICT
    except PostgresLegacySnapshotReconciliationStorageError:
        _write_error("STORAGE")
        return _EXIT_STORAGE
    except Exception:
        _write_error("INTERNAL")
        return _EXIT_INTERNAL
    _write_receipt(receipt)
    return {
        LegacySnapshotReconciliationDisposition.DECISION_REQUIRED: (
            _EXIT_DECISION_REQUIRED
        ),
        LegacySnapshotReconciliationDisposition.BLOCKED: _EXIT_BLOCKED,
    }[receipt.disposition]


if __name__ == "__main__":
    raise SystemExit(main())
