"""CLI and source-boundary tests for the read-only V2 opening planner."""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts import v2_opening_plan as cli
from trading.application.fresh_opening import encode_fresh_opening_trust_policy

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "v2_opening_plan.py"
RUNBOOK = ROOT / "docs" / "V2_FRESH_OPENING_PLAN.md"
TEMPLATES = ROOT / "docs" / "examples" / "v2"

PUBLIC_KEY = "c9dfd699dd6924e6eb5949f8a6da049c0853b248a1a9e4de0272317f8769dee3"
PUBLIC_KEY_SHA256 = "ae147ec67e99458a4dab6369d4e4b2f1da78139c24039af6350189c29cec104e"
POLICY_SHA256 = "0bf6cce716d1bb98dc3e73bff1a42698bdf209710530b019b2e6eac8780cfa84"
INTENT_SHA256 = "f191eb63f25d300b7e8cdf7291d8eece349a6062bd5cd774c98e3ec9a57392ca"
SIGNATURE = (
    "8cc97b95df5d9e88a025c797145b891a1f1803f1566354ef65ba9152407c856d"
    "d299acecc4d48d3ac320b6543e1cf522dd768f9008acbe662ada5f753232b804"
)
APPROVAL_SHA256 = "d982e1342e113cdcc10a506f486c86543eae9a2bf14097793cd7247ad0a058ab"
OPENING_SHA256 = "814af27fad2015dd2eb88b10e04e3921054a8714c8710fa3bad8b2c967500b00"
CANDIDATE_SHA256 = "0731060ef1b8b4af47cc93dd213cf66e7d59b55c93a8d801c757e05efc888d09"
EVALUATED_AT = datetime(2030, 1, 1, 0, 30, tzinfo=timezone.utc)


def _documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    return intent, approval, policy


def _write_documents(
    tmp_path: Path,
    intent: dict[str, Any],
    approval: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[Path, Path, Path]:
    paths = tuple(
        tmp_path / name for name in ("intent.json", "approval.json", "policy.json")
    )
    for path, document in zip(paths, (intent, approval, policy), strict=True):
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return paths


def _arguments(paths: tuple[Path, Path, Path]) -> list[str]:
    intent, approval, policy = paths
    return [
        "--intent",
        str(intent),
        "--approval",
        str(approval),
        "--trust-policy",
        str(policy),
        "--pinned-trust-policy-sha256",
        POLICY_SHA256,
        "--pinned-signer-public-key-sha256",
        PUBLIC_KEY_SHA256,
    ]


def _load_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    assert captured.out.count("\n") == 1
    document = json.loads(captured.out)
    assert isinstance(document, dict)
    return document


def test_safe_reader_rejects_duplicate_and_escaped_equivalent_keys(
    tmp_path: Path,
) -> None:
    for index, payload in enumerate(
        (
            b'{"schema_version":1,"schema_version":1}',
            b'{"schema_version":1,"\\u0073chema_version":1}',
        )
    ):
        path = tmp_path / f"duplicate-{index}.json"
        path.write_bytes(payload)
        with pytest.raises(cli._CliInputError, match="duplicate JSON key"):
            cli._read_json(path)


def test_safe_reader_rejects_symlink_fifo_and_oversize_without_blocking(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (cli._MAX_FILE_BYTES + 1))

    for path in (link, fifo, oversized):
        with pytest.raises(cli._CliInputError):
            cli._read_json(path)


def test_safe_reader_uses_one_opened_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"schema_version":1}', encoding="utf-8")
    real_open = os.open
    opened: list[Path] = []

    def tracked_open(target: object, flags: int, mode: int = 0o777) -> int:
        opened.append(Path(target))
        return real_open(target, flags, mode)

    monkeypatch.setattr(os, "open", tracked_open)

    assert cli._read_json(path) == {"schema_version": 1}
    assert opened == [path]


@pytest.mark.parametrize(
    "payload",
    (
        b"[]",
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b"\xff",
        b'{"unterminated":',
    ),
)
def test_safe_reader_rejects_non_object_or_non_strict_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(payload)

    with pytest.raises(cli._CliInputError):
        cli._read_json(path)


def test_invalid_invocation_is_one_secret_free_json_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_path = "/private/operator/SECRET-token.json"

    assert cli.main(["--unknown", secret_path]) == cli._EXIT_INPUT

    result = _load_stdout(capsys)
    assert result == cli._safe_result("INVALID_INPUT", "INVALID_INPUT")
    assert secret_path not in json.dumps(result)


def test_valid_signed_plan_is_prepared_but_never_authorised(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_documents(tmp_path, *_documents())

    assert cli.main(_arguments(paths), evaluated_at=EVALUATED_AT) == (
        cli._EXIT_PREPARED
    )

    result = _load_stdout(capsys)
    assert result["result"] == "PREPARED"
    assert result["primary_reason_code"] == "PREPARED"
    assert result["intent_sha256"] == INTENT_SHA256
    assert result["trust_policy_sha256"] == POLICY_SHA256
    assert result["signer_public_key_sha256"] == PUBLIC_KEY_SHA256
    assert result["approval_sha256"] == APPROVAL_SHA256
    assert result["opening_payload_sha256"] == OPENING_SHA256
    assert result["candidate_sha256"] == CANDIDATE_SHA256
    assert result["opening_version"] == 1
    assert result["stale_on_return"] is True
    assert result["pin_source_authenticated"] is False
    assert result["physical_target_bound"] is False
    assert result["database_contact"] is False
    assert result["nonce_registry_checked"] is False
    assert result["account_opening_authorized"] is False
    assert result["account_provisioning_authorized"] is False
    assert result["runtime_activation_authorized"] is False
    assert result["trading_authorized"] is False

    serialized = json.dumps(result, sort_keys=True)
    for forbidden in (
        PUBLIC_KEY,
        SIGNATURE,
        _documents()[0]["nonce"],
        *(str(path) for path in paths),
    ):
        assert forbidden not in serialized


def test_missing_authority_and_missing_approval_are_typed_blocks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    intent_document, approval_document, policy_document = _documents()
    intent, approval, policy = _write_documents(
        tmp_path,
        intent_document,
        approval_document,
        policy_document,
    )

    assert cli.main(["--intent", str(intent)], evaluated_at=EVALUATED_AT) == (
        cli._EXIT_BLOCKED
    )
    assert _load_stdout(capsys)["primary_reason_code"] == (
        "BLOCKED_AUTHORITY_UNCONFIGURED"
    )

    without_approval = [
        "--intent",
        str(intent),
        "--trust-policy",
        str(policy),
        "--pinned-trust-policy-sha256",
        POLICY_SHA256,
        "--pinned-signer-public-key-sha256",
        PUBLIC_KEY_SHA256,
    ]
    assert cli.main(without_approval, evaluated_at=EVALUATED_AT) == (cli._EXIT_BLOCKED)
    assert _load_stdout(capsys)["primary_reason_code"] == ("BLOCKED_APPROVAL_MISSING")
    assert approval.is_file()


@pytest.mark.parametrize(
    ("evaluated_at", "reason"),
    (
        (
            datetime(2029, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
            "BLOCKED_APPROVAL_NOT_YET_VALID",
        ),
        (
            datetime(2030, 1, 1, 1, 0, tzinfo=timezone.utc),
            "BLOCKED_APPROVAL_EXPIRED",
        ),
    ),
)
def test_approval_time_boundaries_block_deterministically(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    evaluated_at: datetime,
    reason: str,
) -> None:
    paths = _write_documents(tmp_path, *_documents())

    assert cli.main(_arguments(paths), evaluated_at=evaluated_at) == (cli._EXIT_BLOCKED)
    assert _load_stdout(capsys)["primary_reason_code"] == reason


def test_cryptographically_invalid_signature_is_blocked_not_input_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    intent, approval, policy = _documents()
    approval["signature"] = SIGNATURE[:-1] + ("5" if SIGNATURE[-1] != "5" else "4")
    paths = _write_documents(tmp_path, intent, approval, policy)

    assert cli.main(_arguments(paths), evaluated_at=EVALUATED_AT) == (cli._EXIT_BLOCKED)
    assert _load_stdout(capsys)["primary_reason_code"] == ("BLOCKED_SIGNATURE_INVALID")


def test_weak_small_order_public_key_and_signature_are_invalid_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    intent, approval, policy = _documents()
    policy["anchors"][0]["ed25519_public_key"] = "01" + "00" * 31
    paths = _write_documents(tmp_path, intent, approval, policy)

    assert cli.main(_arguments(paths), evaluated_at=EVALUATED_AT) == cli._EXIT_INPUT
    assert _load_stdout(capsys)["result"] == "INVALID_INPUT"

    intent, approval, policy = _documents()
    approval["signature"] = "01" + "00" * 63
    paths = _write_documents(tmp_path, intent, approval, policy)
    assert cli.main(_arguments(paths), evaluated_at=EVALUATED_AT) == cli._EXIT_INPUT
    assert _load_stdout(capsys)["result"] == "INVALID_INPUT"


@pytest.mark.parametrize(
    "encoded",
    (
        "2030-01-01T00:00:00Z",
        "2030-01-01T00:00:00+00:00",
        "2030-01-01T01:00:00.000000+01:00",
        "2030-01-01T00:00:00.000000",
    ),
)
def test_cli_requires_one_canonical_utc_datetime_encoding(encoded: str) -> None:
    with pytest.raises(cli._CliInputError, match="invalid datetime"):
        cli._utc_datetime(encoded)


@pytest.mark.parametrize(
    "encoded",
    ("+1", "01", "1.", ".1", "1e2", "1E+2", "nan", "infinity"),
)
def test_cli_rejects_noncanonical_decimal_text(encoded: str) -> None:
    with pytest.raises(cli._CliInputError, match="invalid decimal"):
        cli._decimal(encoded)


def test_approval_rejects_algorithm_or_embedded_key_confusion() -> None:
    _, approval, _ = _documents()
    for field, value in (
        ("algorithm", "Ed25519"),
        ("public_key", PUBLIC_KEY),
        ("jwk", {}),
        ("jwt", "header.payload.signature"),
        ("pem", "-----BEGIN PUBLIC KEY-----"),
    ):
        confused = {**approval, field: value}
        with pytest.raises(cli._CliInputError, match="invalid object shape"):
            cli._parse_approval(confused)


def test_repository_templates_are_valid_json_but_deliberately_nonoperational() -> None:
    intent = json.loads(
        (TEMPLATES / "fresh-opening-intent-v1.template.json").read_text(
            encoding="utf-8"
        )
    )
    approval = json.loads(
        (TEMPLATES / "fresh-opening-approval-v1.template.json").read_text(
            encoding="utf-8"
        )
    )
    policy = json.loads(
        (TEMPLATES / "fresh-opening-trust-policy-v1.template.json").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises((TypeError, ValueError, cli._CliInputError)):
        cli._parse_intent(intent)
    with pytest.raises((TypeError, ValueError, cli._CliInputError)):
        cli._parse_approval(approval)
    with pytest.raises((TypeError, ValueError, cli._CliInputError)):
        cli._parse_trust_policy(policy)


def test_runbook_requires_controlled_key_ceremony_and_keeps_pr3_boundary() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "controlled key-generation ceremony" in source
    assert "fingerprint must be frozen and authenticated out of band" in source
    assert "(trust_domain, signer_key_id, nonce)" in source
    assert "logical_target, nonce)" not in source


def test_multi_anchor_policy_must_be_sorted_and_fingerprint_is_canonical() -> None:
    _, _, policy = _documents()
    second = {
        "signer_key_id": "key:test:0",
        "approver_identity": "approver:other",
        "ed25519_public_key": (
            "5866666666666666666666666666666666666666666666666666666666666666"
        ),
        "revoked": True,
    }
    policy["anchors"].append(second)
    with pytest.raises(ValueError, match="unique sorted"):
        cli._parse_trust_policy(policy)

    policy["anchors"].sort(key=lambda value: value["signer_key_id"])
    parsed = cli._parse_trust_policy(policy)
    encoded = encode_fresh_opening_trust_policy(parsed)
    assert len(encoded.sha256) == 64


def test_cli_source_has_no_database_network_process_or_environment_path() -> None:
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])
    forbidden_imports = {
        "asyncpg",
        "httpx",
        "psycopg2",
        "requests",
        "socket",
        "sqlalchemy",
        "subprocess",
        "urllib",
    }

    assert imported_roots.isdisjoint(forbidden_imports)
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "PGSERVICE" not in source
    assert "system_identifier" not in source
    assert "activate" not in source.lower()
    assert "opening-apply" not in source


def test_internal_error_is_generic_and_does_not_echo_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(argv: object, evaluated_at: datetime) -> dict[str, object]:
        del argv, evaluated_at
        raise RuntimeError("SECRET private/key/path")

    monkeypatch.setattr(cli, "_run", fail)

    assert cli.main(["--intent", "unused"]) == cli._EXIT_INTERNAL
    result = _load_stdout(capsys)
    assert result == cli._safe_result("INTERNAL_ERROR", "INTERNAL_ERROR")
    assert "SECRET" not in json.dumps(result)


def test_safe_result_is_permanently_non_authorising() -> None:
    result = cli._safe_result(
        "PREPARED",
        "OPENING_PREPARED_READ_ONLY",
        details={"intent_sha256": "a" * 64},
    )

    assert result["side_effect_state"] == "NONE"
    assert result["database_contact"] is False
    assert result["nonce_registry_checked"] is False
    assert result["target_local_replay_authority"] == "UNAVAILABLE_UNTIL_PR3"
    assert result["account_opening_authorized"] is False
    assert result["account_provisioning_authorized"] is False
    assert result["runtime_activation_authorized"] is False
    assert result["trading_authorized"] is False
    assert "nonce" not in result
    assert "signature" not in result
    assert "public_key" not in result


def test_main_uses_one_explicit_utc_evaluation_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    intent = tmp_path / "intent.json"
    intent.write_text("{}", encoding="utf-8")
    observed = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    received: list[datetime] = []

    def fake_run(argv: object, evaluated_at: datetime) -> dict[str, object]:
        del argv
        received.append(evaluated_at)
        return cli._safe_result("BLOCKED", "BLOCKED_APPROVAL_MISSING")

    monkeypatch.setattr(cli, "_run", fake_run)

    assert cli.main(["--intent", str(intent)], evaluated_at=observed) == (
        cli._EXIT_BLOCKED
    )
    assert received == [observed]
    assert _load_stdout(capsys)["result"] == "BLOCKED"
