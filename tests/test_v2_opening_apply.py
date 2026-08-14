"""Contract tests for the source-only durable V2 fresh-opening CLI."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts import v2_opening_apply as cli
from trading.application.fresh_opening import (
    CanonicalFreshOpeningDocument,
    FreshOpeningPreparationDisposition,
)
from trading.application.fresh_opening_provisioning import (
    FreshOpeningProvisioningDisposition,
    FreshOpeningProvisioningReceipt,
    FreshOpeningProvisioningResult,
)
from trading.persistence.postgres_fresh_opening_provisioning import (
    PostgresFreshOpeningProvisioning,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "v2_opening_apply.py"
RUNBOOK = ROOT / "docs" / "V2_FRESH_OPENING_APPLY.md"
TARGET_TEMPLATE = (
    ROOT / "docs" / "examples" / "v2" / "fresh-opening-target-v1.template.json"
)

PUBLIC_KEY = "c9dfd699dd6924e6eb5949f8a6da049c0853b248a1a9e4de0272317f8769dee3"
PUBLIC_KEY_SHA256 = "ae147ec67e99458a4dab6369d4e4b2f1da78139c24039af6350189c29cec104e"
POLICY_SHA256 = "0bf6cce716d1bb98dc3e73bff1a42698bdf209710530b019b2e6eac8780cfa84"
INTENT_SHA256 = "f191eb63f25d300b7e8cdf7291d8eece349a6062bd5cd774c98e3ec9a57392ca"
SIGNATURE = (
    "8cc97b95df5d9e88a025c797145b891a1f1803f1566354ef65ba9152407c856d"
    "d299acecc4d48d3ac320b6543e1cf522dd768f9008acbe662ada5f753232b804"
)
EVALUATED_AT = datetime(2030, 1, 1, 0, 30, tzinfo=timezone.utc)
SERVICE = "elvis_target_opening"


def _documents() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    policy = {
        "schema_version": 1,
        "purpose": "ELVIS_V2_FRESH_PAPER_OPENING",
        "trust_domain": "trust:test",
        "max_approval_lifetime_seconds": 3600,
        "anchors": [
            {
                "signer_key_id": "key:test:1",
                "approver_identity": "approver:test",
                "ed25519_public_key": PUBLIC_KEY,
                "revoked": False,
            }
        ],
    }
    intent = {
        "schema_version": 1,
        "purpose": "ELVIS_V2_FRESH_PAPER_OPENING",
        "trajectory": "B",
        "continuity": "NO_V1_CONTINUITY",
        "logical_target": "paper:test:logical",
        "execution_scope": "paper:test",
        "account_key": "paper:primary",
        "owner_generation": 1,
        "opening_codec": "paper-account-opening",
        "opening_version": 1,
        "collateral_asset": "USDT",
        "collateral_amount": "100",
        "margin_quantum": "0.01",
        "opening_policy": "EXPLICIT_FRESH_SINGLE_COLLATERAL",
        "operator_identity": "operator:test",
        "approval_id": "approval:test:1",
        "approver_identity": "approver:test",
        "approval_issued_at": "2030-01-01T00:00:00.000000+00:00",
        "approval_expires_at": "2030-01-01T01:00:00.000000+00:00",
        "trust_policy_sha256": POLICY_SHA256,
        "trust_domain": "trust:test",
        "signer_key_id": "key:test:1",
        "signer_public_key_sha256": PUBLIC_KEY_SHA256,
        "nonce": "01" * 32,
    }
    approval = {
        "schema_version": 1,
        "intent_sha256": INTENT_SHA256,
        "signature": SIGNATURE,
    }
    target = {
        "schema_version": 1,
        "expected_database": "elvis_paper_v2",
        "expected_system_identifier": "7340000000000000002",
        "control_plane_role": "elvis_bootstrap_admin",
        "opening_anchor_role": "elvis_opening_anchor",
        "deployment_incarnation_id": "deployment:test:1",
        "terminal_catalog_sha256": "c" * 64,
        "pin_authority_record_sha256": "d" * 64,
    }
    return intent, approval, policy, target


def _write_documents(
    tmp_path: Path,
    documents: (
        tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]
        | None
    ) = None,
) -> tuple[Path, Path, Path, Path]:
    values = _documents() if documents is None else documents
    paths = tuple(
        tmp_path / name
        for name in ("intent.json", "approval.json", "policy.json", "target.json")
    )
    for path, document in zip(paths, values, strict=True):
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return paths


def _arguments(paths: tuple[Path, Path, Path, Path]) -> list[str]:
    intent, approval, policy, target = paths
    target_payload = json.dumps(
        _documents()[3],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    target_sha256 = hashlib.sha256(target_payload).hexdigest()
    return [
        "--intent",
        str(intent),
        "--approval",
        str(approval),
        "--trust-policy",
        str(policy),
        "--target",
        str(target),
        "--pinned-target-document-sha256",
        target_sha256,
        "--pinned-trust-policy-sha256",
        POLICY_SHA256,
        "--pinned-signer-public-key-sha256",
        PUBLIC_KEY_SHA256,
        "--admin-service",
        SERVICE,
        "--apply-opening",
        "--confirm-dedicated-fresh-target",
        "--confirm-exclusive-opening-window",
    ]


def _load_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    assert captured.out.count("\n") == 1
    assert " " not in captured.out
    document = json.loads(captured.out)
    assert type(document) is dict
    return document


class _FakeProvisioning:
    def __init__(
        self,
        disposition: FreshOpeningProvisioningDisposition,
        reason_code: str,
        *,
        evaluate_authority: bool,
    ) -> None:
        self.disposition = disposition
        self.reason_code = reason_code
        self.evaluate_authority = evaluate_authority
        self.calls: list[tuple[object, object]] = []
        self.authority_disposition: FreshOpeningPreparationDisposition | None = None

    def provision(
        self, request: object, candidate: object, authority: object
    ) -> object:
        self.calls.append((request, candidate))
        if self.evaluate_authority:
            preparation = authority.evaluate(EVALUATED_AT)
            self.authority_disposition = preparation.disposition

        receipt = None
        if self.disposition in {
            FreshOpeningProvisioningDisposition.CREATED,
            FreshOpeningProvisioningDisposition.REPLAYED,
        }:
            payload = json.dumps(
                {
                    "candidate_sha256": candidate.candidate_document.sha256,
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            receipt = FreshOpeningProvisioningReceipt(
                document=CanonicalFreshOpeningDocument(
                    payload,
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                ),
                target=request.target,
                intent_sha256=candidate.intent_document.sha256,
                approval_sha256=candidate.approval_document.sha256,
                trust_policy_sha256=candidate.trust_policy_document.sha256,
                candidate_sha256=candidate.candidate_document.sha256,
                opening_payload_sha256=candidate.opening.opening_payload_sha256,
            )
        return FreshOpeningProvisioningResult(
            disposition=self.disposition,
            primary_reason_code=self.reason_code,
            receipt=receipt,
            current_authority_evaluated=self.evaluate_authority,
        )


def _factory_for(
    provisioning: object,
    observed_services: list[str] | None = None,
) -> Any:
    def factory(service: str) -> object:
        if observed_services is not None:
            observed_services.append(service)
        return provisioning

    return factory


@pytest.mark.parametrize(
    ("disposition", "reason_code", "evaluate_authority"),
    (
        (
            FreshOpeningProvisioningDisposition.CREATED,
            "FRESH_OPENING_CREATED",
            True,
        ),
        (
            FreshOpeningProvisioningDisposition.REPLAYED,
            "EXACT_DURABLE_REPLAY",
            False,
        ),
    ),
)
def test_created_and_exact_replay_exit_zero_with_committed_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    disposition: FreshOpeningProvisioningDisposition,
    reason_code: str,
    evaluate_authority: bool,
) -> None:
    paths = _write_documents(tmp_path)
    provisioning = _FakeProvisioning(
        disposition,
        reason_code,
        evaluate_authority=evaluate_authority,
    )
    observed_services: list[str] = []

    assert (
        cli.main(
            _arguments(paths),
            provisioning_factory=_factory_for(provisioning, observed_services),
        )
        == 0
    )

    result = _load_stdout(capsys)
    assert result["result"] == disposition.value
    assert result["primary_reason_code"] == reason_code
    assert result["side_effect_state"] == "COMMITTED"
    assert result["database_contact"] is True
    assert result["nonce_registry_checked"] is True
    assert result["current_authority_evaluated"] is evaluate_authority
    assert result["runtime_mode"] == "LEGACY"
    assert result["runtime_generation"] == 0
    assert result["authority_transition_sequence"] == 0
    assert result["migration_head"] == 7
    assert result["runtime_activation_authorized"] is False
    assert result["trading_authorized"] is False
    assert result["stale_on_return"] is True
    for name in (
        "receipt_sha256",
        "intent_sha256",
        "approval_sha256",
        "trust_policy_sha256",
        "candidate_sha256",
        "opening_payload_sha256",
    ):
        assert len(result[name]) == 64
    assert observed_services == [SERVICE]
    assert len(provisioning.calls) == 1
    if disposition is FreshOpeningProvisioningDisposition.CREATED:
        assert provisioning.authority_disposition is (
            FreshOpeningPreparationDisposition.PREPARED
        )
    else:
        assert provisioning.authority_disposition is None


@pytest.mark.parametrize(
    ("disposition", "reason_code", "exit_code", "side_effect_state"),
    (
        (
            FreshOpeningProvisioningDisposition.BLOCKED,
            "TARGET_ADMISSION_BLOCKED",
            10,
            "NONE",
        ),
        (
            FreshOpeningProvisioningDisposition.BLOCKED,
            "BLOCKED_APPROVAL_EXPIRED",
            10,
            "NONE",
        ),
        (
            FreshOpeningProvisioningDisposition.CONFLICT,
            "FRESH_OPENING_NONCE_CONFLICT",
            20,
            "NONE",
        ),
        (
            FreshOpeningProvisioningDisposition.CONFLICT,
            "FRESH_OPENING_TARGET_CONFLICT",
            20,
            "NONE",
        ),
        (
            FreshOpeningProvisioningDisposition.COMMIT_UNKNOWN,
            "FRESH_OPENING_COMMIT_UNKNOWN",
            21,
            "UNKNOWN",
        ),
    ),
)
def test_noncommitted_typed_outcomes_have_exact_exit_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    disposition: FreshOpeningProvisioningDisposition,
    reason_code: str,
    exit_code: int,
    side_effect_state: str,
) -> None:
    provisioning = _FakeProvisioning(
        disposition,
        reason_code,
        evaluate_authority=False,
    )

    assert (
        cli.main(
            _arguments(_write_documents(tmp_path)),
            provisioning_factory=_factory_for(provisioning),
        )
        == exit_code
    )

    result = _load_stdout(capsys)
    assert result["result"] == disposition.value
    assert result["primary_reason_code"] == reason_code
    assert result["side_effect_state"] == side_effect_state
    assert result["runtime_activation_authorized"] is False
    assert result["trading_authorized"] is False
    assert "receipt_sha256" not in result


def test_output_never_contains_input_paths_or_sensitive_raw_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_documents(tmp_path)
    intent, _, _, target = _documents()
    provisioning = _FakeProvisioning(
        FreshOpeningProvisioningDisposition.CREATED,
        "FRESH_OPENING_CREATED",
        evaluate_authority=True,
    )

    assert (
        cli.main(
            _arguments(paths),
            provisioning_factory=_factory_for(provisioning),
        )
        == 0
    )

    result = _load_stdout(capsys)
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in (
        PUBLIC_KEY,
        SIGNATURE,
        intent["nonce"],
        SERVICE,
        target["expected_database"],
        target["expected_system_identifier"],
        target["control_plane_role"],
        target["opening_anchor_role"],
        target["deployment_incarnation_id"],
        *(str(path) for path in paths),
    ):
        assert forbidden not in serialized


def test_committed_result_with_broken_stdout_returns_internal_for_exact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioning = _FakeProvisioning(
        FreshOpeningProvisioningDisposition.CREATED,
        "FRESH_OPENING_CREATED",
        evaluate_authority=True,
    )

    class BrokenOutput:
        def write(self, value: str) -> int:
            del value
            raise BrokenPipeError

    monkeypatch.setattr(cli.sys, "stdout", BrokenOutput())

    assert (
        cli.main(
            _arguments(_write_documents(tmp_path)),
            provisioning_factory=_factory_for(provisioning),
        )
        == cli._EXIT_INTERNAL
    )
    assert len(provisioning.calls) == 1


def test_unsupported_adapter_reason_is_internal_and_never_echoed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_reason = "SECRET/path/key/signature/nonce"
    provisioning = _FakeProvisioning(
        FreshOpeningProvisioningDisposition.BLOCKED,
        secret_reason,
        evaluate_authority=False,
    )

    assert (
        cli.main(
            _arguments(_write_documents(tmp_path)),
            provisioning_factory=_factory_for(provisioning),
        )
        == cli._EXIT_INTERNAL
    )

    result = _load_stdout(capsys)
    assert result == cli._error_document("INTERNAL_ERROR", internal=True)
    assert secret_reason not in json.dumps(result)


def test_internal_factory_failure_reports_unknown_side_effects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(service: str) -> object:
        del service
        raise RuntimeError("private DSN and password")

    assert (
        cli.main(
            _arguments(_write_documents(tmp_path)),
            provisioning_factory=fail,
        )
        == cli._EXIT_INTERNAL
    )

    result = _load_stdout(capsys)
    assert result["result"] == "INTERNAL_ERROR"
    assert result["side_effect_state"] == "UNKNOWN"
    assert result["database_contact"] is None
    assert result["nonce_registry_checked"] is None
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize(
    "missing_confirmation",
    (
        "--apply-opening",
        "--confirm-dedicated-fresh-target",
        "--confirm-exclusive-opening-window",
    ),
)
def test_missing_confirmation_never_reads_files_or_builds_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    missing_confirmation: str,
) -> None:
    paths = _write_documents(tmp_path)
    factory_calls: list[str] = []
    read_calls: list[Path] = []
    arguments = _arguments(paths)
    arguments.remove(missing_confirmation)

    def forbidden_read(path: Path) -> dict[str, Any]:
        read_calls.append(path)
        raise AssertionError("confirmations must precede file reads")

    monkeypatch.setattr(cli.plan_cli, "_read_json", forbidden_read)

    assert (
        cli.main(
            arguments,
            provisioning_factory=lambda service: factory_calls.append(service),
        )
        == cli._EXIT_INPUT
    )
    assert _load_stdout(capsys) == cli._error_document(
        "INVALID_INPUT",
        internal=False,
    )
    assert read_calls == []
    assert factory_calls == []


def test_invalid_target_shape_never_builds_adapter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_documents(tmp_path)
    calls: list[str] = []

    target = json.loads(paths[3].read_text(encoding="utf-8"))
    target["unexpected"] = True
    paths[3].write_text(json.dumps(target), encoding="utf-8")
    assert (
        cli.main(
            _arguments(paths),
            provisioning_factory=lambda service: calls.append(service),
        )
        == cli._EXIT_INPUT
    )
    assert _load_stdout(capsys)["result"] == "INVALID_INPUT"
    assert calls == []


def test_target_document_pin_is_checked_before_adapter_construction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_documents(tmp_path)
    target = json.loads(paths[3].read_text(encoding="utf-8"))
    target["deployment_incarnation_id"] = "deployment:test:other"
    paths[3].write_text(json.dumps(target), encoding="utf-8")
    calls: list[str] = []

    assert (
        cli.main(
            _arguments(paths),
            provisioning_factory=lambda service: calls.append(service),
        )
        == cli._EXIT_INPUT
    )
    assert _load_stdout(capsys) == cli._error_document(
        "INVALID_INPUT",
        internal=False,
    )
    assert calls == []


@pytest.mark.parametrize(
    "encoded",
    (0, 1, "", "0", "01", "+1", "1.0", "18446744073709551616"),
)
def test_target_requires_canonical_postgres_system_identifier(encoded: object) -> None:
    target = _documents()[3]
    target["expected_system_identifier"] = encoded

    with pytest.raises((TypeError, ValueError, cli._CliInputError)):
        cli._parse_target(target)


def test_apply_reuses_single_descriptor_safe_reader_for_every_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_documents(tmp_path)
    observed: list[Path] = []
    real_reader = cli.plan_cli._read_json

    def tracked(path: Path) -> dict[str, Any]:
        observed.append(path)
        return real_reader(path)

    monkeypatch.setattr(cli.plan_cli, "_read_json", tracked)
    provisioning = _FakeProvisioning(
        FreshOpeningProvisioningDisposition.REPLAYED,
        "EXACT_DURABLE_REPLAY",
        evaluate_authority=False,
    )

    assert (
        cli.main(
            _arguments(paths),
            provisioning_factory=_factory_for(provisioning),
        )
        == 0
    )
    _load_stdout(capsys)
    assert observed == [paths[3], *paths[:3]]


def test_apply_rejects_duplicate_symlink_fifo_and_oversize_before_factory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_paths = _write_documents(tmp_path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema_version":1,"\\u0073chema_version":1}')
    symlink = tmp_path / "target-link.json"
    symlink.symlink_to(base_paths[3])
    fifo = tmp_path / "target.fifo"
    os.mkfifo(fifo)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (cli.plan_cli._MAX_FILE_BYTES + 1))
    calls: list[str] = []

    for unsafe in (duplicate, symlink, fifo, oversized):
        paths = (*base_paths[:3], unsafe)
        assert (
            cli.main(
                _arguments(paths),
                provisioning_factory=lambda service: calls.append(service),
            )
            == cli._EXIT_INPUT
        )
        assert _load_stdout(capsys)["result"] == "INVALID_INPUT"
    assert calls == []


def test_target_template_is_valid_json_but_deliberately_nonoperational() -> None:
    target = json.loads(TARGET_TEMPLATE.read_text(encoding="utf-8"))

    with pytest.raises((TypeError, ValueError, cli._CliInputError)):
        cli._parse_target(target)


def test_source_has_no_signing_env_password_dsn_or_unbounded_reason_output() -> None:
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "Ed25519PrivateKey" not in names
    assert "subprocess" not in imported
    assert "socket" not in imported
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "password=" not in source
    assert "dsn=" not in source
    assert "str(exc)" not in source
    assert "repr(exc)" not in source
    assert source.count("plan_cli._read_json") == 4


def test_default_factory_composes_public_adapter_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_connect(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("factory construction must not contact PostgreSQL")

    monkeypatch.setattr("psycopg2.connect", forbidden_connect)

    provisioning = cli._default_provisioning_factory(SERVICE)

    assert type(provisioning) is PostgresFreshOpeningProvisioning


def test_source_cli_is_not_in_alpha2_dispatcher_image_or_release_smokes() -> None:
    surfaces = (
        ROOT / "scripts" / "v2_operator.py",
        ROOT / "deploy" / "v2" / "operator.Dockerfile",
        ROOT / ".github" / "workflows" / "release.yml",
    )

    for surface in surfaces:
        source = surface.read_text(encoding="utf-8")
        assert "v2_opening_apply" not in source
        assert "opening-apply" not in source


def test_runbook_freezes_replay_first_and_non_authority_contract() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "(trust_domain, signer_key_id, nonce)" in source
    assert "replay" in source.lower()
    assert "COMMIT_UNKNOWN" in source
    assert "LEGACY/0/S0" in source
    assert "runtime_activation_authorized=false" in source
    assert "trading_authorized=false" in source
