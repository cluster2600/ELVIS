"""Adversarial contracts for the read-only fresh-target cut-over preflight."""

from __future__ import annotations

import ast
import copy
import json
import pickle
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

from scripts import postgres_cutover_preflight as cli
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
from trading.persistence.postgres_bootstrap import (
    PostgresBootstrap,
    PostgresBootstrapContext,
    PostgresBootstrapRoles,
    PostgresBootstrapStorageError,
    PostgresBootstrapTerminalInspection,
)
from trading.persistence.postgres_cutover_preflight import (
    PostgresCutoverPreflight,
    PostgresCutoverPreflightStorageError,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
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


def _roles() -> PostgresBootstrapRoles:
    return PostgresBootstrapRoles(
        schema_owner="elvis_v2_owner",
        migrator="elvis_v2_migrator",
        legacy_runtime="elvis_v2_legacy",
        atomic_runtime="elvis_v2_atomic",
        activation="elvis_v2_activation",
        readiness="elvis_v2_readiness",
        trainer="elvis_v2_trainer",
    )


def _role_manifest() -> FreshTargetRoleManifest:
    roles = _roles()
    return FreshTargetRoleManifest(
        **{name: getattr(roles, name) for name in _ROLE_KEYS}
    )


def _bootstrap_intent() -> FreshTargetBootstrapIntent:
    return FreshTargetBootstrapIntent(
        expected_database="elvis_fresh_target",
        admin_role="elvis_bootstrap_admin",
        roles=_role_manifest(),
    )


def _bootstrap_context() -> PostgresBootstrapContext:
    intent = _bootstrap_intent()
    return PostgresBootstrapContext(
        expected_database=intent.expected_database,
        admin_role=intent.admin_role,
        roles=_roles(),
        adoption=None,
    )


def _context(**overrides: object) -> FreshTargetCutoverContext:
    values = {
        "source_expected_database": "elvis_source_clone",
        "source_expected_role": "elvis_source_inspector",
        "target_bootstrap_intent": _bootstrap_intent(),
    }
    values.update(overrides)
    return FreshTargetCutoverContext(**values)


def _relations(*, populated: bool = True) -> tuple[FreshTargetRelationEvidence, ...]:
    return tuple(
        FreshTargetRelationEvidence(
            name=name,
            row_count=1 if populated and name == "np.trades" else 0,
            pk_min=7 if populated and name == "np.trades" else None,
            pk_max=7 if populated and name == "np.trades" else None,
            sha256=_SHA_A if name == "np.trades" else _SHA_B,
        )
        for name in _LEGACY_RELATIONS
    )


def _source(**overrides: object) -> FreshTargetCutoverSourceEvidence:
    values = {
        "system_identifier": 11,
        "relations": _relations(),
        "other_session_count": 0,
        "open_position_count": 0,
        "semantic_invalid_row_count": 0,
        "canonical_sha256": _SHA_A,
        "legacy_layout_exact": True,
        "identity_exact": True,
    }
    values.update(overrides)
    return FreshTargetCutoverSourceEvidence(**values)


def _target(**overrides: object) -> FreshTargetCutoverTargetEvidence:
    values = {
        "system_identifier": 22,
        "terminal_catalog_exact": True,
        "migration_versions": (1, 2, 3, 4, 5, 6),
        "runtime_mode": "LEGACY",
        "runtime_generation": 0,
        "nonempty_relations": (),
    }
    values.update(overrides)
    return FreshTargetCutoverTargetEvidence(**values)


def _receipt(
    *,
    status: FreshTargetCutoverStatus = FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET,
    blockers: tuple[FreshTargetCutoverBlocker, ...] = (),
) -> FreshTargetCutoverReceipt:
    return FreshTargetCutoverReceipt(
        status=status,
        blockers=blockers,
        source=_source(),
        target=_target(),
    )


def _config() -> dict[str, object]:
    roles = _roles()
    return {
        "schema_version": 1,
        "source": {
            "expected_database": "elvis_source_clone",
            "expected_role": "elvis_source_inspector",
            "service": "elvis_source_inspector_service",
        },
        "target": {
            "admin_service": "elvis_target_admin_service",
            "bootstrap_context": {
                "expected_database": "elvis_fresh_target",
                "admin_role": "elvis_bootstrap_admin",
                "roles": {name: getattr(roles, name) for name in _ROLE_KEYS},
                "adoption": None,
            },
        },
    }


def _write_config(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "fresh-target-preflight.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _arguments(path: Path, *extra: str) -> list[str]:
    return [
        "--config",
        str(path),
        "--inspect",
        "--confirm-stopped-source-clone",
        "--confirm-exclusive-database-window",
        *extra,
    ]


def _assert_input(capsys) -> None:
    captured = capsys.readouterr()
    assert captured.out == '{"status":"ERROR","code":"INPUT"}\n'
    assert captured.err == ""


def test_contract_schema_and_enum_are_exact_and_non_authoritative() -> None:
    assert tuple(field.name for field in fields(FreshTargetRoleManifest)) == _ROLE_KEYS
    assert tuple(field.name for field in fields(FreshTargetBootstrapIntent)) == (
        "expected_database",
        "admin_role",
        "roles",
    )
    assert tuple(field.name for field in fields(FreshTargetCutoverContext)) == (
        "source_expected_database",
        "source_expected_role",
        "target_bootstrap_intent",
    )
    assert tuple(member.value for member in FreshTargetCutoverStatus) == (
        "READY_FOR_FRESH_TARGET",
        "BLOCKED",
    )
    assert tuple(member.value for member in FreshTargetCutoverBlocker) == (
        "SOURCE_IDENTITY",
        "SOURCE_ACTIVE_SESSIONS",
        "SOURCE_SCHEMA",
        "SOURCE_OPEN_POSITIONS",
        "SOURCE_DATA_QUALITY",
        "SAME_CLUSTER",
        "TARGET_NOT_COMPLETE",
        "TARGET_MODE",
        "TARGET_NOT_EMPTY",
    )
    assert tuple(field.name for field in fields(FreshTargetRelationEvidence)) == (
        "name",
        "row_count",
        "pk_min",
        "pk_max",
        "sha256",
    )
    assert tuple(field.name for field in fields(FreshTargetCutoverReceipt)) == (
        "status",
        "blockers",
        "source",
        "target",
        "stale_on_return",
        "snapshot_authoritative",
    )

    receipt = _receipt()
    assert receipt.stale_on_return is True
    assert receipt.snapshot_authoritative is False


def test_contract_values_are_frozen_slotted_copyable_and_pickleable() -> None:
    values = (
        _role_manifest(),
        _bootstrap_intent(),
        _context(),
        _relations()[0],
        _source(),
        _target(),
        _receipt(),
    )

    for value in values:
        assert not hasattr(value, "__dict__")
        assert copy.copy(value) == value
        assert copy.deepcopy(value) == value
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, next(iter(value.__dataclass_fields__)), None)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            assert pickle.loads(pickle.dumps(value, protocol=protocol)) == value


@pytest.mark.parametrize(
    ("overrides", "error"),
    (
        ({"name": "trades"}, ValueError),
        ({"name": "np.unknown"}, ValueError),
        ({"row_count": True}, TypeError),
        ({"row_count": -1}, ValueError),
        ({"pk_min": 2, "pk_max": 1}, ValueError),
        ({"pk_min": None, "pk_max": 1}, TypeError),
        ({"pk_min": 1, "pk_max": None}, TypeError),
        ({"sha256": "A" * 64}, ValueError),
        ({"sha256": "a" * 63}, ValueError),
    ),
)
def test_relation_evidence_rejects_noncanonical_shapes(overrides, error) -> None:
    values = {
        "name": "np.trades",
        "row_count": 1,
        "pk_min": 1,
        "pk_max": 1,
        "sha256": _SHA_A,
    }
    values.update(overrides)
    with pytest.raises(error):
        FreshTargetRelationEvidence(**values)


def test_source_evidence_requires_the_seven_relations_once_in_order() -> None:
    relations = _relations()
    assert tuple(value.name for value in _source().relations) == _LEGACY_RELATIONS

    for invalid in (
        tuple(reversed(relations)),
        relations[:-1] + (relations[0],),
    ):
        with pytest.raises(ValueError):
            _source(relations=invalid)


def test_ready_and_blocked_receipts_are_coherent() -> None:
    with pytest.raises(ValueError):
        _receipt(blockers=(FreshTargetCutoverBlocker.SAME_CLUSTER,))
    with pytest.raises(ValueError):
        _receipt(status=FreshTargetCutoverStatus.BLOCKED)

    blocked = _receipt(
        status=FreshTargetCutoverStatus.BLOCKED,
        blockers=(FreshTargetCutoverBlocker.SAME_CLUSTER,),
    )
    assert blocked.status is FreshTargetCutoverStatus.BLOCKED
    assert blocked.stale_on_return is True
    assert blocked.snapshot_authoritative is False


def test_all_three_confirmations_precede_config_or_service_resolution(
    tmp_path, capsys
) -> None:
    path = _write_config(tmp_path, _config())
    required_flags = (
        "--inspect",
        "--confirm-stopped-source-clone",
        "--confirm-exclusive-database-window",
    )

    for missing in required_flags:
        arguments = [value for value in _arguments(path) if value != missing]
        resolver = MagicMock(side_effect=AssertionError("must remain offline"))
        assert cli.main(arguments, service_connection_factory=resolver) == 2
        resolver.assert_not_called()
        _assert_input(capsys)

    resolver = MagicMock(side_effect=AssertionError("must remain offline"))
    assert cli.main(["--config", str(path)], service_connection_factory=resolver) == 2
    resolver.assert_not_called()
    _assert_input(capsys)


def test_config_is_closed_nonsecret_and_validated_before_resolution(
    tmp_path, capsys
) -> None:
    invalid_documents: list[object] = []

    document = _config()
    document["secret"] = "postgresql://admin:password@host/elvis"
    invalid_documents.append(document)

    document = _config()
    document["schema_version"] = True
    invalid_documents.append(document)

    document = _config()
    del document["source"]["expected_role"]
    invalid_documents.append(document)

    document = _config()
    document["source"]["service"] = "postgresql://user:password@db/elvis"
    invalid_documents.append(document)

    document = _config()
    document["source"]["expected_database"] = "postgresql://source:secret@db/elvis"
    invalid_documents.append(document)

    document = _config()
    document["source"]["expected_role"] = "Élvis_source"
    invalid_documents.append(document)

    document = _config()
    document["target"]["admin_service"] = "host=db password=secret"
    invalid_documents.append(document)

    document = _config()
    document["target"]["bootstrap_context"]["expected_database"] = "Target-DB"
    invalid_documents.append(document)

    document = _config()
    document["target"]["bootstrap_context"]["admin_role"] = "host=db user=admin"
    invalid_documents.append(document)

    document = _config()
    document["target"]["bootstrap_context"]["adoption"] = {}
    invalid_documents.append(document)

    document = _config()
    document["target"]["bootstrap_context"]["roles"]["password"] = "secret"
    invalid_documents.append(document)

    for index, document in enumerate(invalid_documents):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        resolver = MagicMock(side_effect=AssertionError("must remain offline"))
        assert cli.main(_arguments(path), service_connection_factory=resolver) == 2
        resolver.assert_not_called()
        _assert_input(capsys)


@pytest.mark.parametrize(
    "payload",
    (
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":NaN}',
        "[" * 1_200 + "]" * 1_200,
    ),
)
def test_pathological_or_ambiguous_json_is_input_before_resolution(
    tmp_path, capsys, payload
) -> None:
    path = tmp_path / "pathological.json"
    path.write_text(payload, encoding="utf-8")
    resolver = MagicMock(side_effect=AssertionError("must remain offline"))

    assert cli.main(_arguments(path), service_connection_factory=resolver) == 2

    resolver.assert_not_called()
    _assert_input(capsys)


def test_exact_service_mapping_and_one_inspection(tmp_path, capsys) -> None:
    document = _config()
    path = _write_config(tmp_path, document)
    tokens: dict[str, MagicMock] = {}

    def resolve(service_name: str) -> MagicMock:
        token = MagicMock(name=f"factory_for_{service_name}")
        tokens[service_name] = token
        return token

    inspector = MagicMock()
    inspector.inspect.return_value = _receipt()
    inspector_type = MagicMock(return_value=inspector)

    with patch.object(cli, "PostgresCutoverPreflight", inspector_type):
        assert cli.main(_arguments(path), service_connection_factory=resolve) == 0

    assert tuple(tokens) == (
        document["source"]["service"],
        document["target"]["admin_service"],
    )
    inspector_type.assert_called_once_with(
        tokens[document["source"]["service"]],
        tokens[document["target"]["admin_service"]],
    )
    inspector.inspect.assert_called_once()
    context = inspector.inspect.call_args.args[0]
    assert context == _context()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "READY_FOR_FRESH_TARGET"
    assert payload["stale_on_return"] is True
    assert payload["snapshot_authoritative"] is False
    assert payload["source"]["system_identifier"] == "11"
    assert payload["target"]["system_identifier"] == "22"
    assert "service" not in json.dumps(payload)
    assert "role" not in json.dumps(payload)


def test_default_service_factory_connects_once_with_bounded_libpq_options() -> None:
    connection = object()
    connect = MagicMock(return_value=connection)

    with patch.object(cli.psycopg2, "connect", connect):
        factory = cli._connection_factory_for_service("source_service")
        connect.assert_not_called()
        assert factory() is connection

    connect.assert_called_once_with(
        service="source_service",
        application_name="elvis-fresh-target-preflight-v1",
        connect_timeout=5,
    )


def test_application_contract_has_no_persistence_dependency() -> None:
    source_path = (
        Path(__file__).parents[1] / "trading/application/fresh_target_cutover.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(name.startswith("trading.persistence") for name in imported_modules)


@pytest.mark.parametrize(
    ("receipt", "exit_code"),
    (
        (_receipt(), 0),
        (
            _receipt(
                status=FreshTargetCutoverStatus.BLOCKED,
                blockers=(FreshTargetCutoverBlocker.SAME_CLUSTER,),
            ),
            21,
        ),
    ),
)
def test_status_exit_mapping(tmp_path, capsys, receipt, exit_code) -> None:
    path = _write_config(tmp_path, _config())
    inspector = MagicMock()
    inspector.inspect.return_value = receipt

    with patch.object(cli, "PostgresCutoverPreflight", return_value=inspector):
        assert (
            cli.main(
                _arguments(path),
                service_connection_factory=lambda _service: MagicMock(),
            )
            == exit_code
        )

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == receipt.status.value
    assert payload["blockers"] == [value.value for value in receipt.blockers]


@pytest.mark.parametrize(
    ("error", "exit_code", "code"),
    (
        (PostgresBootstrapStorageError("storage detail"), 20, "STORAGE"),
        (RuntimeError("internal detail"), 70, "INTERNAL"),
    ),
)
def test_error_exit_mapping_redacts_the_exception_graph(
    tmp_path, capsys, error, exit_code, code
) -> None:
    secret = "postgresql://operator:never-print@example.invalid/elvis"
    error.__cause__ = RuntimeError(f"cause {secret}")
    error.__context__ = ValueError(f"context {secret}")
    path = _write_config(tmp_path, _config())
    inspector = MagicMock()
    inspector.inspect.side_effect = error

    with patch.object(cli, "PostgresCutoverPreflight", return_value=inspector):
        assert (
            cli.main(
                _arguments(path),
                service_connection_factory=lambda _service: MagicMock(),
            )
            == exit_code
        )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == f'{{"status":"ERROR","code":"{code}"}}\n'
    assert secret not in captured.out
    assert str(error) not in captured.out


def test_target_inspection_uses_the_historical_head6_boundary(monkeypatch) -> None:
    target_factory = MagicMock()
    current_terminal = MagicMock(
        side_effect=AssertionError("historical workflow must not claim V2 terminal")
    )
    historical_terminal = MagicMock(
        return_value=PostgresBootstrapTerminalInspection(
            system_identifier=22,
            exact=True,
            migration_versions=(1, 2, 3, 4, 5, 6),
            runtime_mode="LEGACY",
            runtime_generation=0,
            nonempty_relations=(),
        )
    )
    monkeypatch.setattr(PostgresBootstrap, "inspect_terminal", current_terminal)
    monkeypatch.setattr(
        PostgresBootstrap,
        "inspect_historical_terminal",
        historical_terminal,
    )

    target = PostgresCutoverPreflight(
        MagicMock(),
        target_factory,
    )._inspect_target(_context())

    assert target == _target()
    current_terminal.assert_not_called()
    historical_terminal.assert_called_once()
    historical_context = historical_terminal.call_args.args[0]
    assert historical_context.roles == _roles()
    assert historical_context.roles.opening is None


def test_historical_terminal_inspection_severs_driver_exception_graph() -> None:
    secret = "postgresql://operator:never-print@example.invalid/elvis"
    connection = MagicMock()
    connection.autocommit = False
    connection.status = STATUS_READY
    connection.get_transaction_status.return_value = TRANSACTION_STATUS_IDLE
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = RuntimeError(secret)

    with pytest.raises(PostgresBootstrapStorageError) as raised:
        PostgresBootstrap(lambda: connection).inspect_historical_terminal(
            _bootstrap_context()
        )

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)
    connection.commit.assert_not_called()
    connection.rollback.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_public_preflight_severs_source_driver_exception_graph() -> None:
    secret = "postgresql://source:never-print@example.invalid/elvis"

    def broken_source():
        raise RuntimeError(secret)

    inspector = PostgresCutoverPreflight(broken_source, lambda: MagicMock())

    with pytest.raises(PostgresCutoverPreflightStorageError) as raised:
        inspector.inspect(_context())

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)


def test_help_is_offline_and_lists_all_barriers(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli.psycopg2,
        "connect",
        MagicMock(side_effect=AssertionError("help must remain offline")),
    )

    with pytest.raises(SystemExit, match="0"):
        cli.main(["--help"])

    output = capsys.readouterr().out
    assert "--inspect" in output
    assert "--confirm-stopped-source-clone" in output
    assert "--confirm-exclusive-database-window" in output
