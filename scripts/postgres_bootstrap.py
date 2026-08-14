"""Offline, one-shot operator CLI for the dormant PostgreSQL bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, TextIO

import psycopg2

from scripts import v2_opening_plan as opening_plan_cli
from trading.application.fresh_opening import (
    FreshOpeningPreparationDisposition,
    derive_prospective_fresh_opening_candidate,
    prepare_fresh_opening,
)
from trading.persistence.postgres_bootstrap import (
    PostgresBootstrap,
    PostgresBootstrapAdoption,
    PostgresBootstrapCommitUnknownError,
    PostgresBootstrapContext,
    PostgresBootstrapDriftError,
    PostgresBootstrapInputError,
    PostgresBootstrapMigrationError,
    PostgresBootstrapReceipt,
    PostgresBootstrapRoles,
    PostgresBootstrapStatus,
    PostgresBootstrapStorageError,
    PostgresFreshOpeningAdmission,
)

_APPLICATION_NAME = "elvis-postgres-bootstrap-v2"
_CONNECT_TIMEOUT_SECONDS = 5
_MAX_CONFIG_BYTES = 65_536
_SERVICE_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,62}")

_ROLE_KEYS = (
    "schema_owner",
    "migrator",
    "opening",
    "legacy_runtime",
    "atomic_runtime",
    "activation",
    "readiness",
    "trainer",
)
_CREDENTIAL_ROLE_KEYS = (
    "migrator",
    "readiness",
    "trainer",
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "expected_database",
    "admin_role",
    "roles",
    "services",
    "adoption",
    "opening_admission",
}
_ADOPTION_KEYS = {
    "migration_authority_role",
    "allowed_historical_owner_roles",
    "old_shared_runtime_role",
    "demote_old_shared_runtime",
}
_OPENING_ADMISSION_KEYS = {
    "candidate_sha256",
    "pin_authority_record_sha256",
    "deployment_incarnation_id",
}

_EXIT_COMPLETE = 0
_EXIT_CREDENTIALS_REQUIRED = 10
_EXIT_DEMOTION_REQUIRED = 11
_EXIT_INPUT = 2
_EXIT_STORAGE = 20
_EXIT_DRIFT = 21
_EXIT_MIGRATION = 22
_EXIT_COMMIT_UNKNOWN = 23
_EXIT_INTERNAL = 70

_STATUS_EXIT_CODES = {
    PostgresBootstrapStatus.COMPLETE: _EXIT_COMPLETE,
    PostgresBootstrapStatus.CREDENTIALS_REQUIRED: _EXIT_CREDENTIALS_REQUIRED,
    PostgresBootstrapStatus.DEMOTION_REQUIRED: _EXIT_DEMOTION_REQUIRED,
}


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
        document = opening_plan_cli._read_json(path)
    except opening_plan_cli._CliInputError as error:
        raise _CliInputError("invalid configuration") from error
    if type(document) is not dict:
        raise _CliInputError("invalid configuration")
    return document


def _canonical_sha256(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
) -> None:
    if set(value) != expected:
        raise _CliInputError("invalid configuration")


def _require_service_identifier(value: object) -> str:
    if not isinstance(value, str) or _SERVICE_IDENTIFIER.fullmatch(value) is None:
        raise _CliInputError("invalid configuration")
    return value


def _parse_roles(value: object) -> PostgresBootstrapRoles:
    if type(value) is not dict:
        raise _CliInputError("invalid configuration")
    _require_exact_keys(value, set(_ROLE_KEYS))
    if any(type(value[key]) is not str for key in _ROLE_KEYS):
        raise _CliInputError("invalid configuration")
    return PostgresBootstrapRoles(**{key: value[key] for key in _ROLE_KEYS})


def _parse_adoption(value: object) -> PostgresBootstrapAdoption | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise _CliInputError("invalid configuration")
    _require_exact_keys(value, _ADOPTION_KEYS)

    migration_authority_role = value["migration_authority_role"]
    historical_roles = value["allowed_historical_owner_roles"]
    old_shared_runtime_role = value["old_shared_runtime_role"]
    demote_old_shared_runtime = value["demote_old_shared_runtime"]
    if type(migration_authority_role) is not str:
        raise _CliInputError("invalid configuration")
    if type(historical_roles) is not list or any(
        type(role) is not str for role in historical_roles
    ):
        raise _CliInputError("invalid configuration")
    if old_shared_runtime_role is not None and type(old_shared_runtime_role) is not str:
        raise _CliInputError("invalid configuration")
    if type(demote_old_shared_runtime) is not bool:
        raise _CliInputError("invalid configuration")

    return PostgresBootstrapAdoption(
        migration_authority_role=migration_authority_role,
        allowed_historical_owner_roles=tuple(historical_roles),
        old_shared_runtime_role=old_shared_runtime_role,
        demote_old_shared_runtime=demote_old_shared_runtime,
    )


def _parse_opening_admission(value: object) -> PostgresFreshOpeningAdmission:
    if type(value) is not dict:
        raise _CliInputError("invalid configuration")
    _require_exact_keys(value, _OPENING_ADMISSION_KEYS)
    if any(type(value[key]) is not str for key in _OPENING_ADMISSION_KEYS):
        raise _CliInputError("invalid configuration")
    try:
        return PostgresFreshOpeningAdmission(
            candidate_sha256=value["candidate_sha256"],
            pin_authority_record_sha256=value["pin_authority_record_sha256"],
            deployment_incarnation_id=value["deployment_incarnation_id"],
        )
    except PostgresBootstrapInputError as exc:
        raise _CliInputError("invalid configuration") from exc


def _parse_services(value: object) -> dict[str, str | None]:
    if type(value) is not dict:
        raise _CliInputError("invalid configuration")
    allowed_keys = {"admin", *_ROLE_KEYS}
    _require_exact_keys(value, allowed_keys)

    services: dict[str, str | None] = {key: None for key in _ROLE_KEYS[1:]}
    services["admin"] = _require_service_identifier(value["admin"])
    if value["schema_owner"] is not None:
        raise _CliInputError("invalid configuration")
    if (
        value["activation"] is not None
        or value["opening"] is not None
        or value["legacy_runtime"] is not None
        or value["atomic_runtime"] is not None
    ):
        raise _CliInputError("invalid configuration")
    for key in _CREDENTIAL_ROLE_KEYS:
        candidate = value.get(key)
        if candidate is not None:
            services[key] = _require_service_identifier(candidate)
    return services


def _parse_config(
    document: dict[str, Any],
) -> tuple[PostgresBootstrapContext, dict[str, str | None]]:
    _require_exact_keys(document, _TOP_LEVEL_KEYS)
    if type(document["schema_version"]) is not int or document["schema_version"] != 2:
        raise _CliInputError("invalid configuration")
    if type(document["expected_database"]) is not str:
        raise _CliInputError("invalid configuration")
    if type(document["admin_role"]) is not str:
        raise _CliInputError("invalid configuration")

    context = PostgresBootstrapContext(
        expected_database=document["expected_database"],
        admin_role=document["admin_role"],
        roles=_parse_roles(document["roles"]),
        adoption=_parse_adoption(document["adoption"]),
        opening_admission=_parse_opening_admission(document["opening_admission"]),
    )
    return context, _parse_services(document["services"])


def _connection_factory_for_service(service_name: str) -> Callable[[], object]:
    def connect() -> object:
        return psycopg2.connect(
            service=service_name,
            application_name=_APPLICATION_NAME,
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        )

    return connect


def _build_bootstrap(
    services: dict[str, str | None],
    service_connection_factory: Callable[[str], Callable[[], object]],
) -> PostgresBootstrap:
    def optional_factory(key: str) -> Callable[[], object] | None:
        service_name = services[key]
        if service_name is None:
            return None
        return service_connection_factory(service_name)

    return PostgresBootstrap(
        service_connection_factory(services["admin"]),
        migrator_connection_factory=optional_factory("migrator"),
        activation_connection_factory=None,
        readiness_connection_factory=optional_factory("readiness"),
        trainer_connection_factory=optional_factory("trainer"),
    )


def _write_json(stream: TextIO, payload: dict[str, object]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    stream.write("\n")


def _emit_json(stream: TextIO, payload: dict[str, object]) -> bool:
    try:
        _write_json(stream, payload)
    except BrokenPipeError, OSError:
        return False
    return True


def _write_receipt(receipt: PostgresBootstrapReceipt) -> bool:
    return _emit_json(
        sys.stdout,
        {
            "status": receipt.status.value,
            "migration_versions": list(receipt.migration_versions),
            "verified_role_probes": list(receipt.verified_role_probes),
            "pending_role_credentials": list(receipt.pending_role_credentials),
            "old_shared_runtime_demoted": receipt.old_shared_runtime_demoted,
        },
    )


def _write_error(code: str, *, phase: str | None = None) -> bool:
    payload: dict[str, object] = {"status": "ERROR", "code": code}
    if phase is not None:
        payload["phase"] = phase
    return _emit_json(sys.stdout, payload)


def _argument_parser() -> _StrictArgumentParser:
    parser = _StrictArgumentParser(
        prog="python3.14 -m scripts.postgres_bootstrap",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pinned-config-sha256", required=True)
    parser.add_argument("--opening-intent", required=True)
    parser.add_argument("--opening-approval", required=True)
    parser.add_argument("--opening-trust-policy", required=True)
    parser.add_argument("--pinned-trust-policy-sha256", required=True)
    parser.add_argument("--pinned-signer-public-key-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-exclusive-ddl-role-window",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-old-runtime-demotion",
        action="store_true",
    )
    return parser


def _run(
    argv: Sequence[str] | None,
    service_connection_factory: Callable[[str], Callable[[], object]],
) -> PostgresBootstrapReceipt:
    arguments = _argument_parser().parse_args(argv)
    if not arguments.apply or not arguments.confirm_exclusive_ddl_role_window:
        raise _CliInputError("invalid invocation")

    config_document = _read_config(Path(arguments.config))
    expected_config_sha256 = opening_plan_cli._sha256(arguments.pinned_config_sha256)
    if not hmac.compare_digest(
        _canonical_sha256(config_document),
        expected_config_sha256,
    ):
        raise _CliInputError("bootstrap configuration pin mismatch")
    context, services = _parse_config(config_document)
    intent = opening_plan_cli._parse_intent(
        opening_plan_cli._read_json(Path(arguments.opening_intent))
    )
    approval = opening_plan_cli._parse_approval(
        opening_plan_cli._read_json(Path(arguments.opening_approval))
    )
    trust_policy = opening_plan_cli._parse_trust_policy(
        opening_plan_cli._read_json(Path(arguments.opening_trust_policy))
    )
    expected_policy = opening_plan_cli._sha256(arguments.pinned_trust_policy_sha256)
    expected_key = opening_plan_cli._sha256(arguments.pinned_signer_public_key_sha256)
    candidate = derive_prospective_fresh_opening_candidate(
        intent,
        approval,
        trust_policy,
        opening_codec=opening_plan_cli._OPENING_CODEC,
    )
    admission = context.opening_admission
    if (
        admission is None
        or candidate.candidate_document.sha256 != admission.candidate_sha256
    ):
        raise _CliInputError("opening admission is not an authenticated candidate")

    def authorize_opening(evaluated_at: datetime) -> str:
        preparation = prepare_fresh_opening(
            intent,
            approval,
            opening_codec=opening_plan_cli._OPENING_CODEC,
            signature_verifier=opening_plan_cli._SIGNATURE_VERIFIER,
            trust_policy=trust_policy,
            expected_trust_policy_sha256=expected_policy,
            expected_signer_public_key_sha256=expected_key,
            evaluated_at=evaluated_at,
        )
        if (
            preparation.disposition is not FreshOpeningPreparationDisposition.PREPARED
            or preparation.candidate is None
            or preparation.candidate.candidate_document.sha256
            != candidate.candidate_document.sha256
        ):
            raise _CliInputError("opening admission is not currently authorized")
        return admission.document_sha256

    demotion_requested = bool(
        context.adoption is not None and context.adoption.demote_old_shared_runtime
    )
    if arguments.confirm_old_runtime_demotion != demotion_requested:
        raise _CliInputError("invalid invocation")

    bootstrap = _build_bootstrap(services, service_connection_factory)
    return bootstrap.reconcile(
        context,
        admission_authorizer=authorize_opening,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    service_connection_factory: Callable[[str], Callable[[], object]] | None = None,
) -> int:
    """Run one non-interactive reconciliation and emit one compact JSON value."""
    try:
        receipt = _run(
            argv,
            service_connection_factory or _connection_factory_for_service,
        )
        exit_code = _STATUS_EXIT_CODES[receipt.status]
    except PostgresBootstrapInputError, _CliInputError:
        return _EXIT_INPUT if _write_error("INPUT") else _EXIT_INTERNAL
    except PostgresBootstrapStorageError:
        return _EXIT_STORAGE if _write_error("STORAGE") else _EXIT_INTERNAL
    except PostgresBootstrapDriftError:
        return _EXIT_DRIFT if _write_error("DRIFT") else _EXIT_INTERNAL
    except PostgresBootstrapMigrationError:
        return _EXIT_MIGRATION if _write_error("MIGRATION") else _EXIT_INTERNAL
    except PostgresBootstrapCommitUnknownError as error:
        return (
            _EXIT_COMMIT_UNKNOWN
            if _write_error("COMMIT_UNKNOWN", phase=error.phase.value)
            else _EXIT_INTERNAL
        )
    except Exception:
        _write_error("INTERNAL")
        return _EXIT_INTERNAL

    return exit_code if _write_receipt(receipt) else _EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
