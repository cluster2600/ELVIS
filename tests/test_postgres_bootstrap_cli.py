"""Offline contract checks for ``python -m scripts.postgres_bootstrap``."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import postgres_bootstrap as cli
from trading.persistence.postgres_bootstrap import (
    PostgresBootstrapCommitUnknownError,
    PostgresBootstrapDriftError,
    PostgresBootstrapInputError,
    PostgresBootstrapMigrationError,
    PostgresBootstrapPhase,
    PostgresBootstrapReceipt,
    PostgresBootstrapStatus,
    PostgresBootstrapStorageError,
)

_ROLE_KEYS = (
    "schema_owner",
    "migrator",
    "legacy_runtime",
    "atomic_runtime",
    "activation",
    "readiness",
    "trainer",
)
_LOGIN_ROLE_KEYS = _ROLE_KEYS[1:]
_NO_RESOLUTION = AssertionError("must not resolve a service")


def _config(*, demote: bool = False) -> dict[str, object]:
    adoption = None
    if demote:
        adoption = {
            "migration_authority_role": "elvis_history_owner",
            "allowed_historical_owner_roles": ["elvis_history_owner"],
            "old_shared_runtime_role": "elvis_history_owner",
            "demote_old_shared_runtime": True,
        }
    return {
        "schema_version": 1,
        "expected_database": "elvis_cli_database",
        "admin_role": "elvis_cli_admin",
        "roles": {
            "schema_owner": "elvis_cli_owner",
            "migrator": "elvis_cli_migrator",
            "legacy_runtime": "elvis_cli_legacy",
            "atomic_runtime": "elvis_cli_atomic",
            "activation": "elvis_cli_activation",
            "readiness": "elvis_cli_readiness",
            "trainer": "elvis_cli_trainer",
        },
        "services": {
            "admin": "elvis_admin_service",
            "schema_owner": None,
            "migrator": "elvis_migrator_service",
            "legacy_runtime": "elvis_legacy_service",
            "atomic_runtime": "elvis_atomic_service",
            "activation": "elvis_activation_service",
            "readiness": "elvis_readiness_service",
            "trainer": "elvis_trainer_service",
        },
        "adoption": adoption,
    }


def _write_config(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _apply_args(path: Path, *extra: str) -> list[str]:
    return [
        "--config",
        str(path),
        "--apply",
        "--confirm-exclusive-ddl-role-window",
        *extra,
    ]


def _receipt(
    status: PostgresBootstrapStatus = PostgresBootstrapStatus.COMPLETE,
) -> PostgresBootstrapReceipt:
    return PostgresBootstrapReceipt(
        status=status,
        migration_versions=(1, 2, 3, 4, 5, 6),
        verified_role_probes=("activation", "trainer"),
        pending_role_credentials=(
            ("migrator",)
            if status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
            else ()
        ),
        old_shared_runtime_demoted=False,
    )


def _assert_input_error(capsys) -> None:
    captured = capsys.readouterr()
    assert captured.out == '{"status":"ERROR","code":"INPUT"}\n'
    assert captured.err == ""


def test_apply_and_exclusive_confirmation_precede_service_resolution(
    tmp_path, capsys
) -> None:
    ordinary_path = _write_config(tmp_path, _config())

    for arguments in (
        ["--config", str(ordinary_path)],
        ["--config", str(ordinary_path), "--apply"],
        [
            "--config",
            str(ordinary_path),
            "--confirm-exclusive-ddl-role-window",
        ],
        _apply_args(ordinary_path, "--confirm-old-runtime-demotion"),
    ):
        resolver = MagicMock(side_effect=_NO_RESOLUTION)
        assert cli.main(arguments, service_connection_factory=resolver) == 2
        resolver.assert_not_called()
        _assert_input_error(capsys)

    demotion_path = _write_config(tmp_path, _config(demote=True))
    resolver = MagicMock(side_effect=_NO_RESOLUTION)
    assert (
        cli.main(
            _apply_args(demotion_path),
            service_connection_factory=resolver,
        )
        == 2
    )
    resolver.assert_not_called()
    _assert_input_error(capsys)


def test_demotion_requires_both_exclusive_and_specific_confirmation(
    tmp_path, capsys
) -> None:
    path = _write_config(tmp_path, _config(demote=True))
    bootstrap = MagicMock()
    bootstrap.reconcile.return_value = _receipt()
    resolver = MagicMock(return_value=MagicMock())

    with patch.object(cli, "PostgresBootstrap", return_value=bootstrap):
        exit_code = cli.main(
            _apply_args(path, "--confirm-old-runtime-demotion"),
            service_connection_factory=resolver,
        )

    assert exit_code == 0
    bootstrap.reconcile.assert_called_once()
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"


def test_help_subprocess_is_offline_and_successful() -> None:
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.postgres_bootstrap", "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "--confirm-exclusive-ddl-role-window" in result.stdout
    assert "--confirm-old-runtime-demotion" in result.stdout


def test_config_schema_is_closed_and_rejected_before_service_resolution(
    tmp_path, capsys
) -> None:
    invalid_documents = []

    document = _config()
    document["unknown"] = "secret-value"
    invalid_documents.append(document)

    document = _config()
    del document["adoption"]
    invalid_documents.append(document)

    document = _config()
    document["schema_version"] = True
    invalid_documents.append(document)

    document = _config()
    del document["roles"]["trainer"]
    invalid_documents.append(document)

    document = _config()
    document["roles"]["password"] = "secret-value"
    invalid_documents.append(document)

    document = _config()
    del document["services"]["trainer"]
    invalid_documents.append(document)

    document = _config()
    document["services"]["password"] = "secret-value"
    invalid_documents.append(document)

    document = _config()
    document["services"]["schema_owner"] = "forbidden_service"
    invalid_documents.append(document)

    adoption = _config(demote=True)
    adoption["adoption"]["unknown"] = "secret-value"
    invalid_documents.append(adoption)

    for index, invalid in enumerate(invalid_documents):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(invalid), encoding="utf-8")
        resolver = MagicMock(side_effect=_NO_RESOLUTION)
        assert cli.main(_apply_args(path), service_connection_factory=resolver) == 2
        resolver.assert_not_called()
        _assert_input_error(capsys)


@pytest.mark.parametrize(
    "service_name",
    (
        "postgresql://user:password@db/elvis",
        "host=db user=admin password=secret",
        "Uppercase",
        "contains-dash",
        "contains space",
        "/tmp/postgresql.sock",
        "a" * 64,
        "",
    ),
)
def test_services_accept_names_only_and_never_dsn_material(
    tmp_path, capsys, service_name
) -> None:
    document = _config()
    document["services"]["admin"] = service_name
    path = _write_config(tmp_path, document)
    resolver = MagicMock(side_effect=_NO_RESOLUTION)

    assert cli.main(_apply_args(path), service_connection_factory=resolver) == 2

    resolver.assert_not_called()
    _assert_input_error(capsys)


def test_exact_factory_mapping_and_one_reconcile(tmp_path, capsys) -> None:
    document = _config()
    path = _write_config(tmp_path, document)
    tokens: dict[str, MagicMock] = {}

    def resolve(service_name: str) -> MagicMock:
        token = MagicMock(name=f"factory_for_{service_name}")
        tokens[service_name] = token
        return token

    bootstrap = MagicMock()
    bootstrap.reconcile.return_value = _receipt()
    bootstrap_type = MagicMock(return_value=bootstrap)

    with patch.object(cli, "PostgresBootstrap", bootstrap_type):
        exit_code = cli.main(_apply_args(path), service_connection_factory=resolve)

    assert exit_code == 0
    assert set(tokens) == {
        value
        for key, value in document["services"].items()
        if key != "schema_owner" and value is not None
    }
    constructor_args, constructor_kwargs = bootstrap_type.call_args
    assert constructor_args == (tokens[document["services"]["admin"]],)
    for role_key in _LOGIN_ROLE_KEYS:
        service_name = document["services"][role_key]
        factory = constructor_kwargs[f"{role_key}_connection_factory"]
        assert factory is tokens[service_name]
    context = bootstrap.reconcile.call_args.args[0]
    assert context.expected_database == document["expected_database"]
    assert context.admin_role == document["admin_role"]
    expected_roles = tuple(document["roles"][key] for key in _ROLE_KEYS)
    assert context.roles.all == expected_roles
    bootstrap.reconcile.assert_called_once_with(context)
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"


def test_null_login_services_are_not_resolved(tmp_path, capsys) -> None:
    document = _config()
    for role_key in _LOGIN_ROLE_KEYS:
        document["services"][role_key] = None
    path = _write_config(tmp_path, document)
    resolved = []

    def resolve(service_name):
        resolved.append(service_name)
        return MagicMock()

    bootstrap = MagicMock()
    bootstrap.reconcile.return_value = _receipt(
        PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    )
    with patch.object(cli, "PostgresBootstrap", return_value=bootstrap):
        assert cli.main(_apply_args(path), service_connection_factory=resolve) == 10

    assert resolved == [document["services"]["admin"]]
    _ = capsys.readouterr()


def test_default_factory_passes_only_libpq_service_name(monkeypatch) -> None:
    connect = MagicMock(return_value=object())
    monkeypatch.setattr(cli.psycopg2, "connect", connect)

    connection = cli._connection_factory_for_service("elvis_admin_service")()

    assert connection is connect.return_value
    connect.assert_called_once_with(
        service="elvis_admin_service",
        application_name="elvis-postgres-bootstrap-v1",
        connect_timeout=5,
    )


@pytest.mark.parametrize(
    ("status", "exit_code"),
    (
        (PostgresBootstrapStatus.COMPLETE, 0),
        (PostgresBootstrapStatus.CREDENTIALS_REQUIRED, 10),
        (PostgresBootstrapStatus.DEMOTION_REQUIRED, 11),
    ),
)
def test_receipt_json_and_status_exit_codes(
    tmp_path, capsys, status, exit_code
) -> None:
    path = _write_config(tmp_path, _config())
    receipt = _receipt(status)
    bootstrap = MagicMock()
    bootstrap.reconcile.return_value = receipt

    with patch.object(cli, "PostgresBootstrap", return_value=bootstrap):
        assert (
            cli.main(
                _apply_args(path),
                service_connection_factory=lambda _name: MagicMock(),
            )
            == exit_code
        )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert (
        captured.out
        == json.dumps(
            {
                "status": status.value,
                "migration_versions": [1, 2, 3, 4, 5, 6],
                "verified_role_probes": ["activation", "trainer"],
                "pending_role_credentials": (
                    ["migrator"]
                    if status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
                    else []
                ),
                "old_shared_runtime_demoted": False,
            },
            separators=(",", ":"),
        )
        + "\n"
    )


@pytest.mark.parametrize(
    ("error", "exit_code", "code", "phase"),
    (
        (PostgresBootstrapInputError("unsafe input"), 2, "INPUT", None),
        (
            PostgresBootstrapStorageError("connection failed"),
            20,
            "STORAGE",
            None,
        ),
        (PostgresBootstrapDriftError("catalog details"), 21, "DRIFT", None),
        (
            PostgresBootstrapMigrationError("migration details"),
            22,
            "MIGRATION",
            None,
        ),
        (
            PostgresBootstrapCommitUnknownError(PostgresBootstrapPhase.CATALOG),
            23,
            "COMMIT_UNKNOWN",
            "CATALOG",
        ),
        (RuntimeError("internal details"), 70, "INTERNAL", None),
    ),
)
def test_error_exit_mapping_redacts_exception_graph(
    tmp_path, capsys, error, exit_code, code, phase
) -> None:
    secret = "postgresql://admin:never-print@database/elvis"
    error.__cause__ = RuntimeError(f"cause {secret}")
    error.__context__ = ValueError(f"context {secret}")
    path = _write_config(tmp_path, _config())
    bootstrap = MagicMock()
    bootstrap.reconcile.side_effect = error

    with patch.object(cli, "PostgresBootstrap", return_value=bootstrap):
        assert (
            cli.main(
                _apply_args(path),
                service_connection_factory=lambda _name: MagicMock(),
            )
            == exit_code
        )

    captured = capsys.readouterr()
    assert captured.err == ""
    expected = {"status": "ERROR", "code": code}
    if phase is not None:
        expected["phase"] = phase
    assert captured.out == json.dumps(expected, separators=(",", ":")) + "\n"
    assert secret not in captured.out
    assert str(error) not in captured.out


def test_malformed_config_does_not_echo_payload(tmp_path, capsys) -> None:
    secret = "postgresql://admin:never-print@database/elvis"
    path = tmp_path / "malformed.json"
    payload = '{"schema_version":1,"secret":"' + secret
    path.write_text(payload, encoding="utf-8")

    assert cli.main(_apply_args(path)) == 2

    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert captured.out == '{"status":"ERROR","code":"INPUT"}\n'
    assert captured.err == ""


@pytest.mark.parametrize(
    "payload",
    (
        "[" * 1_200 + "]" * 1_200,
        '{"schema_version":' + "9" * 5_000 + "}",
    ),
)
def test_pathological_json_is_classified_as_input(tmp_path, capsys, payload) -> None:
    path = tmp_path / "pathological.json"
    path.write_text(payload, encoding="utf-8")

    assert cli.main(_apply_args(path)) == 2

    captured = capsys.readouterr()
    assert captured.out == '{"status":"ERROR","code":"INPUT"}\n'
    assert captured.err == ""


def test_oversized_config_is_rejected_before_service_resolution(
    tmp_path, capsys
) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * 65_537)
    resolver = MagicMock(side_effect=AssertionError("must not resolve a service"))

    assert cli.main(_apply_args(path), service_connection_factory=resolver) == 2

    resolver.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == '{"status":"ERROR","code":"INPUT"}\n'
    assert captured.err == ""


def test_cli_remains_offline_and_unwired_from_runtime_or_compose() -> None:
    repository = Path(__file__).resolve().parents[1]
    source_path = repository / "scripts" / "postgres_bootstrap.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "os" not in imported_roots
    assert "core" not in imported_roots
    assert "main" not in imported_roots
    assert "getenv" not in source

    runtime_paths = [
        repository / "main.py",
        repository / "core" / "bootstrap.py",
    ]
    runtime_paths.extend(repository.glob("docker-compose*.yml"))
    runtime_paths.extend(repository.glob("docker-compose*.yaml"))
    for path in runtime_paths:
        text = path.read_text(encoding="utf-8")
        assert "scripts.postgres_bootstrap" not in text
        assert "python -m scripts.postgres_bootstrap" not in text
