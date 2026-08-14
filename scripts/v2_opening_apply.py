"""One-shot CLI for one durable ELVIS V2 fresh paper-account opening.

The command consumes strict local evidence and delegates the replay-first,
target-local transaction to the injected PostgreSQL provisioning adapter.  A
successful opening never changes runtime or trading authority.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn, TextIO

from scripts import v2_opening_plan as plan_cli
from trading.application.fresh_opening import FreshOpeningPreparationDisposition
from trading.application.fresh_opening_provisioning import (
    FreshOpeningPhysicalTarget,
    FreshOpeningProvisioningDisposition,
    FreshOpeningProvisioningPort,
    FreshOpeningProvisioningRequest,
    FreshOpeningProvisioningResult,
    FreshOpeningProvisioningService,
)

_APPLICATION_NAME = "elvis-v2-opening-apply-v1"
_CONNECT_TIMEOUT_SECONDS = 5
_SERVICE_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,62}")
_REASON_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,127}")
_TARGET_KEYS = {
    "schema_version",
    "expected_database",
    "expected_system_identifier",
    "control_plane_role",
    "opening_anchor_role",
    "deployment_incarnation_id",
    "terminal_catalog_sha256",
    "pin_authority_record_sha256",
}

_BLOCKED_REASON_CODES = frozenset(
    {
        disposition.value
        for disposition in FreshOpeningPreparationDisposition
        if disposition is not FreshOpeningPreparationDisposition.PREPARED
    }
    | {"TARGET_ADMISSION_BLOCKED"}
)
_REASON_CODES = {
    FreshOpeningProvisioningDisposition.CREATED: frozenset({"FRESH_OPENING_CREATED"}),
    FreshOpeningProvisioningDisposition.REPLAYED: frozenset({"EXACT_DURABLE_REPLAY"}),
    FreshOpeningProvisioningDisposition.BLOCKED: _BLOCKED_REASON_CODES,
    FreshOpeningProvisioningDisposition.CONFLICT: frozenset(
        {
            "FRESH_OPENING_NONCE_CONFLICT",
            "FRESH_OPENING_TARGET_CONFLICT",
        }
    ),
    FreshOpeningProvisioningDisposition.COMMIT_UNKNOWN: frozenset(
        {"FRESH_OPENING_COMMIT_UNKNOWN"}
    ),
}

_EXIT_INPUT = 2
_EXIT_BLOCKED = 10
_EXIT_CONFLICT = 20
_EXIT_COMMIT_UNKNOWN = 21
_EXIT_INTERNAL = 70
_EXIT_BY_DISPOSITION = {
    FreshOpeningProvisioningDisposition.CREATED: 0,
    FreshOpeningProvisioningDisposition.REPLAYED: 0,
    FreshOpeningProvisioningDisposition.BLOCKED: _EXIT_BLOCKED,
    FreshOpeningProvisioningDisposition.CONFLICT: _EXIT_CONFLICT,
    FreshOpeningProvisioningDisposition.COMMIT_UNKNOWN: _EXIT_COMMIT_UNKNOWN,
}


class _CliInputError(ValueError):
    """Secret-free signal for malformed local input or invocation."""


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _CliInputError("invalid invocation")


def _write_json(stream: TextIO, value: dict[str, object]) -> None:
    stream.write(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    )
    stream.write("\n")


def _emit_json(stream: TextIO, value: dict[str, object]) -> bool:
    try:
        _write_json(stream, value)
    except BrokenPipeError, OSError:
        return False
    return True


def _service_name(value: object) -> str:
    if type(value) is not str or _SERVICE_IDENTIFIER.fullmatch(value) is None:
        raise _CliInputError("invalid libpq service")
    return value


def _system_identifier(value: object) -> int:
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or not value.isdigit()
        or value.startswith("0")
    ):
        raise _CliInputError("invalid PostgreSQL system identifier")
    return int(value)


def _parse_target(document: dict[str, object]) -> FreshOpeningPhysicalTarget:
    plan_cli._exact_keys(document, _TARGET_KEYS)
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise _CliInputError("invalid target schema version")
    return FreshOpeningPhysicalTarget(
        expected_database=document["expected_database"],
        expected_system_identifier=_system_identifier(
            document["expected_system_identifier"]
        ),
        control_plane_role=document["control_plane_role"],
        opening_anchor_role=document["opening_anchor_role"],
        deployment_incarnation_id=document["deployment_incarnation_id"],
        terminal_catalog_sha256=document["terminal_catalog_sha256"],
        pin_authority_record_sha256=document["pin_authority_record_sha256"],
    )


def _canonical_document_sha256(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _argument_parser() -> _StrictArgumentParser:
    parser = _StrictArgumentParser(
        prog="python3.14 -m scripts.v2_opening_apply",
        allow_abbrev=False,
        description=(
            "Apply or exactly replay one signed fresh opening while leaving "
            "runtime and trading authority disabled."
        ),
    )
    parser.add_argument("--intent", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--trust-policy", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--pinned-target-document-sha256", required=True)
    parser.add_argument("--pinned-trust-policy-sha256", required=True)
    parser.add_argument("--pinned-signer-public-key-sha256", required=True)
    parser.add_argument("--admin-service", required=True)
    parser.add_argument("--apply-opening", action="store_true")
    parser.add_argument("--confirm-dedicated-fresh-target", action="store_true")
    parser.add_argument("--confirm-exclusive-opening-window", action="store_true")
    return parser


def _parse_request(
    argv: Sequence[str] | None,
) -> tuple[str, FreshOpeningProvisioningRequest]:
    arguments = _argument_parser().parse_args(argv)
    if not (
        arguments.apply_opening
        and arguments.confirm_dedicated_fresh_target
        and arguments.confirm_exclusive_opening_window
    ):
        raise _CliInputError("all explicit confirmations are required")
    service = _service_name(arguments.admin_service)
    target_document = plan_cli._read_json(Path(arguments.target))
    expected_target_sha256 = plan_cli._sha256(arguments.pinned_target_document_sha256)
    if not hmac.compare_digest(
        _canonical_document_sha256(target_document),
        expected_target_sha256,
    ):
        raise _CliInputError("target admission record pin mismatch")
    request = FreshOpeningProvisioningRequest(
        intent=plan_cli._parse_intent(plan_cli._read_json(Path(arguments.intent))),
        approval=plan_cli._parse_approval(
            plan_cli._read_json(Path(arguments.approval))
        ),
        trust_policy=plan_cli._parse_trust_policy(
            plan_cli._read_json(Path(arguments.trust_policy))
        ),
        expected_trust_policy_sha256=plan_cli._sha256(
            arguments.pinned_trust_policy_sha256
        ),
        expected_signer_public_key_sha256=plan_cli._sha256(
            arguments.pinned_signer_public_key_sha256
        ),
        target=_parse_target(target_document),
    )
    return service, request


def _default_provisioning_factory(
    service_name: str,
) -> FreshOpeningProvisioningPort:
    """Compose the admin-only adapter lazily so help stays dependency-light."""

    import psycopg2

    from trading.persistence.postgres_fresh_opening_provisioning import (
        PostgresFreshOpeningProvisioning,
    )

    def connect() -> object:
        return psycopg2.connect(
            service=service_name,
            application_name=_APPLICATION_NAME,
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        )

    return PostgresFreshOpeningProvisioning(connect)


def _safe_reason_code(result: FreshOpeningProvisioningResult) -> str:
    reason_code = result.primary_reason_code
    if (
        _REASON_CODE.fullmatch(reason_code) is None
        or reason_code not in _REASON_CODES[result.disposition]
    ):
        raise TypeError("provisioning adapter returned an unsupported reason code")
    return reason_code


def _result_document(
    result: FreshOpeningProvisioningResult,
) -> dict[str, object]:
    if type(result) is not FreshOpeningProvisioningResult:
        raise TypeError("provisioning adapter returned an invalid result")
    value: dict[str, object] = {
        "schema_version": 1,
        "result": result.disposition.value,
        "primary_reason_code": _safe_reason_code(result),
        "side_effect_state": result.side_effect_state,
        "database_contact": result.database_contact,
        "nonce_registry_checked": result.nonce_registry_checked,
        "current_authority_evaluated": result.current_authority_evaluated,
        "runtime_activation_authorized": result.runtime_activation_authorized,
        "trading_authorized": result.trading_authorized,
        "stale_on_return": result.stale_on_return,
    }
    if result.receipt is not None:
        receipt = result.receipt
        value.update(
            {
                "receipt_sha256": receipt.document.sha256,
                "intent_sha256": receipt.intent_sha256,
                "approval_sha256": receipt.approval_sha256,
                "trust_policy_sha256": receipt.trust_policy_sha256,
                "candidate_sha256": receipt.candidate_sha256,
                "opening_payload_sha256": receipt.opening_payload_sha256,
                "migration_head": receipt.migration_head,
                "runtime_mode": receipt.runtime_mode,
                "runtime_generation": receipt.runtime_generation,
                "authority_transition_sequence": (
                    receipt.authority_transition_sequence
                ),
            }
        )
    return value


def _error_document(result: str, *, internal: bool) -> dict[str, object]:
    return {
        "schema_version": 1,
        "result": result,
        "primary_reason_code": result,
        "side_effect_state": "UNKNOWN" if internal else "NONE",
        "database_contact": None if internal else False,
        "nonce_registry_checked": None if internal else False,
        "current_authority_evaluated": None if internal else False,
        "runtime_activation_authorized": False,
        "trading_authorized": False,
        "stale_on_return": True,
    }


def _execute(
    service_name: str,
    request: FreshOpeningProvisioningRequest,
    provisioning_factory: Callable[[str], FreshOpeningProvisioningPort],
) -> FreshOpeningProvisioningResult:
    provisioning = provisioning_factory(service_name)
    service = FreshOpeningProvisioningService(
        provisioning,
        plan_cli._OPENING_CODEC,
        plan_cli._SIGNATURE_VERIFIER,
    )
    return service.provision(request)


def main(
    argv: Sequence[str] | None = None,
    *,
    provisioning_factory: Callable[[str], FreshOpeningProvisioningPort] | None = None,
) -> int:
    """Apply once and emit exactly one compact, secret-free JSON result."""

    try:
        service_name, request = _parse_request(argv)
    except _CliInputError, plan_cli._CliInputError, TypeError, ValueError:
        emitted = _emit_json(
            sys.stdout, _error_document("INVALID_INPUT", internal=False)
        )
        return _EXIT_INPUT if emitted else _EXIT_INTERNAL
    try:
        result = _execute(
            service_name,
            request,
            provisioning_factory or _default_provisioning_factory,
        )
        document = _result_document(result)
    except Exception:
        _emit_json(sys.stdout, _error_document("INTERNAL_ERROR", internal=True))
        return _EXIT_INTERNAL
    if not _emit_json(sys.stdout, document):
        return _EXIT_INTERNAL
    return _EXIT_BY_DISPOSITION[result.disposition]


if __name__ == "__main__":
    raise SystemExit(main())
