"""Adversarial contracts for the bounded V1 legacy snapshot importer."""

from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

from scripts import postgres_legacy_snapshot_import as cli
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
    LegacySnapshotImportDisposition,
    LegacySnapshotImportReceipt,
    LegacySnapshotRelationReceipt,
)
from trading.persistence import postgres_legacy_snapshot_import as importer_module
from trading.persistence.postgres_legacy_snapshot_import import (
    PostgresLegacySnapshotImport,
    PostgresLegacySnapshotImportBusyError,
    PostgresLegacySnapshotImportCommitUnknown,
    PostgresLegacySnapshotImportConflict,
    PostgresLegacySnapshotImportInputError,
    PostgresLegacySnapshotImportStorageError,
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


def _role_manifest() -> FreshTargetRoleManifest:
    return FreshTargetRoleManifest(
        schema_owner="elvis_v2_owner",
        migrator="elvis_v2_migrator",
        legacy_runtime="elvis_v2_legacy",
        atomic_runtime="elvis_v2_atomic",
        activation="elvis_v2_activation",
        readiness="elvis_v2_readiness",
        trainer="elvis_v2_trainer",
    )


def _cutover_context() -> FreshTargetCutoverContext:
    return FreshTargetCutoverContext(
        source_expected_database="elvis_source_clone",
        source_expected_role="elvis_source_inspector",
        target_bootstrap_intent=FreshTargetBootstrapIntent(
            expected_database="elvis_fresh_target",
            admin_role="elvis_bootstrap_admin",
            roles=_role_manifest(),
        ),
    )


def _source_relations() -> tuple[FreshTargetRelationEvidence, ...]:
    return tuple(
        FreshTargetRelationEvidence(
            name=name,
            row_count=2 if name == "np.trades" else 0,
            pk_min=7 if name == "np.trades" else None,
            pk_max=19 if name == "np.trades" else None,
            sha256=_SHA_A if name == "np.trades" else _SHA_B,
        )
        for name in _LEGACY_RELATIONS
    )


def _preflight_receipt(
    *,
    status: FreshTargetCutoverStatus = FreshTargetCutoverStatus.READY_FOR_FRESH_TARGET,
    blockers: tuple[FreshTargetCutoverBlocker, ...] = (),
) -> FreshTargetCutoverReceipt:
    return FreshTargetCutoverReceipt(
        status=status,
        blockers=blockers,
        source=FreshTargetCutoverSourceEvidence(
            system_identifier=11,
            relations=_source_relations(),
            other_session_count=0,
            open_position_count=0,
            semantic_invalid_row_count=0,
            canonical_sha256=_SHA_A,
            legacy_layout_exact=True,
            identity_exact=True,
        ),
        target=FreshTargetCutoverTargetEvidence(
            system_identifier=22,
            terminal_catalog_exact=True,
            migration_versions=(1, 2, 3, 4, 5, 6),
            runtime_mode="LEGACY",
            runtime_generation=0,
            nonempty_relations=(),
        ),
    )


def _context(*, batch_size: int = 512) -> LegacySnapshotImportContext:
    return LegacySnapshotImportContext(_cutover_context(), batch_size=batch_size)


def _relation_receipts() -> tuple[LegacySnapshotRelationReceipt, ...]:
    return tuple(
        LegacySnapshotRelationReceipt(
            name=name,
            row_count=2 if name == "np.trades" else 0,
            pk_min=7 if name == "np.trades" else None,
            pk_max=19 if name == "np.trades" else None,
            sha256=_SHA_A if name == "np.trades" else _SHA_B,
            source_sequence_next=31,
            target_sequence_next=31,
        )
        for name in _LEGACY_RELATIONS
    )


def _import_receipt(
    *,
    disposition: LegacySnapshotImportDisposition = (
        LegacySnapshotImportDisposition.IMPORTED
    ),
) -> LegacySnapshotImportReceipt:
    return LegacySnapshotImportReceipt(
        context=_context(),
        disposition=disposition,
        source_system_identifier=11,
        target_system_identifier=22,
        source_canonical_sha256=_SHA_A,
        relations=_relation_receipts(),
        target_exact=True,
        runtime_activation_authorized=False,
    )


def _config_document(*, batch_size: object = 512) -> dict[str, object]:
    roles = _role_manifest()
    return {
        "schema_version": 1,
        "batch_size": batch_size,
        "source": {
            "service": "elvis_source_clone",
            "expected_database": "elvis_source_clone",
            "expected_role": "elvis_source_inspector",
        },
        "target": {
            "admin_service": "elvis_target_admin",
            "migrator_service": "elvis_target_migrator",
            "bootstrap_context": {
                "expected_database": "elvis_fresh_target",
                "admin_role": "elvis_bootstrap_admin",
                "roles": {
                    field.name: getattr(roles, field.name)
                    for field in fields(FreshTargetRoleManifest)
                },
                "adoption": None,
            },
        },
    }


def _preflight_document() -> dict[str, object]:
    receipt = _preflight_receipt()
    return {
        "status": receipt.status.value,
        "blockers": [item.value for item in receipt.blockers],
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
            "semantic_invalid_row_count": receipt.source.semantic_invalid_row_count,
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
    }


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")
    path.chmod(0o600)


def _cli_paths(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "legacy-snapshot-import.json"
    receipt = tmp_path / "fresh-target-preflight-receipt.json"
    _write_json(config, _config_document())
    _write_json(receipt, _preflight_document())
    return config, receipt


def _cli_arguments(config: Path, receipt: Path) -> list[str]:
    return [
        "--config",
        str(config),
        "--preflight-receipt",
        str(receipt),
        "--import-snapshot",
        "--confirm-stopped-source-clone",
        "--confirm-exclusive-database-window",
        "--confirm-disposable-target",
    ]


def _assert_cli_error(capsys, code: str) -> None:
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "ERROR", "code": code}
    assert captured.err == ""


def _assert_secret_absent_from_exception_graph(
    error: BaseException,
    secret: str,
) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert secret not in str(current)
        assert secret not in repr(current)
        pending.extend(
            linked
            for linked in (current.__cause__, current.__context__)
            if linked is not None
        )


def _fresh_connection_double() -> MagicMock:
    connection = MagicMock()
    connection.autocommit = False
    connection.status = STATUS_READY
    connection.get_transaction_status.return_value = TRANSACTION_STATUS_IDLE
    connection.cursor.return_value.__exit__.return_value = False
    return connection


def test_application_contract_is_exact_bounded_and_non_authoritative() -> None:
    assert tuple(field.name for field in fields(LegacySnapshotImportContext)) == (
        "cutover_context",
        "batch_size",
    )
    assert tuple(member.value for member in LegacySnapshotImportDisposition) == (
        "IMPORTED",
        "REPLAYED",
    )
    assert tuple(field.name for field in fields(LegacySnapshotRelationReceipt)) == (
        "name",
        "row_count",
        "pk_min",
        "pk_max",
        "sha256",
        "source_sequence_next",
        "target_sequence_next",
    )
    assert tuple(field.name for field in fields(LegacySnapshotImportReceipt)) == (
        "context",
        "disposition",
        "source_system_identifier",
        "target_system_identifier",
        "source_canonical_sha256",
        "relations",
        "target_exact",
        "runtime_activation_authorized",
        "stale_on_return",
        "snapshot_authoritative",
    )

    receipt = _import_receipt()
    assert receipt.target_exact is True
    assert receipt.runtime_activation_authorized is False
    assert receipt.stale_on_return is True
    assert receipt.snapshot_authoritative is False
    assert receipt.source_system_identifier != receipt.target_system_identifier
    assert tuple(item.name for item in receipt.relations) == _LEGACY_RELATIONS


@pytest.mark.parametrize("batch_size", (1, 512))
def test_context_accepts_only_the_declared_batch_boundaries(batch_size: int) -> None:
    assert _context(batch_size=batch_size).batch_size == batch_size


@pytest.mark.parametrize("batch_size", (True, 0, 513, -1, 1.0, "1"))
def test_context_rejects_invalid_batch_sizes(batch_size: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        LegacySnapshotImportContext(_cutover_context(), batch_size=batch_size)


def test_contract_values_are_frozen_slotted_copyable_and_pickleable() -> None:
    values = (
        _context(),
        _relation_receipts()[0],
        _import_receipt(),
        _import_receipt(disposition=LegacySnapshotImportDisposition.REPLAYED),
    )

    for value in values:
        assert not hasattr(value, "__dict__")
        assert copy.copy(value) == value
        assert copy.deepcopy(value) == value
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, next(iter(value.__dataclass_fields__)), None)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            assert pickle.loads(pickle.dumps(value, protocol=protocol)) == value


def test_relation_receipt_requires_exact_name_hash_bounds_and_sequence_parity() -> None:
    valid = _relation_receipts()[5]
    assert valid.name == "np.trades"
    assert valid.source_sequence_next == valid.target_sequence_next == 31

    invalid = (
        {"name": "np.unknown"},
        {"row_count": True},
        {"row_count": -1},
        {"pk_min": 20, "pk_max": 19},
        {"sha256": "A" * 64},
        {"source_sequence_next": 0},
        {"source_sequence_next": 2_147_483_648},
        {"target_sequence_next": 6},
    )
    for overrides in invalid:
        with pytest.raises((TypeError, ValueError)):
            replace(valid, **overrides)

    normalized = replace(
        valid,
        source_sequence_next=5,
        target_sequence_next=20,
    )
    assert normalized.pk_max == 19
    assert normalized.source_sequence_next == 5
    assert normalized.target_sequence_next == 20


def test_import_receipt_requires_all_relations_and_exact_safe_target() -> None:
    receipt = _import_receipt()
    invalid = (
        {"relations": tuple(reversed(receipt.relations))},
        {"relations": receipt.relations[:-1]},
        {"source_system_identifier": 22},
        {"source_canonical_sha256": "A" * 64},
        {"target_exact": False},
        {"runtime_activation_authorized": True},
        {"stale_on_return": False},
        {"snapshot_authoritative": True},
    )
    for overrides in invalid:
        with pytest.raises((TypeError, ValueError)):
            replace(receipt, **overrides)


def test_adapter_rejects_invalid_factories_context_and_preflight_before_io() -> None:
    source_factory = MagicMock(
        side_effect=AssertionError("database access is forbidden")
    )
    admin_factory = MagicMock(
        side_effect=AssertionError("database access is forbidden")
    )
    migrator_factory = MagicMock(
        side_effect=AssertionError("database access is forbidden")
    )
    with pytest.raises(PostgresLegacySnapshotImportInputError):
        PostgresLegacySnapshotImport(None, admin_factory, migrator_factory)
    with pytest.raises(PostgresLegacySnapshotImportInputError):
        PostgresLegacySnapshotImport(source_factory, None, migrator_factory)
    with pytest.raises(PostgresLegacySnapshotImportInputError):
        PostgresLegacySnapshotImport(source_factory, admin_factory, None)
    for factories in (
        (source_factory, source_factory, migrator_factory),
        (source_factory, admin_factory, source_factory),
        (source_factory, admin_factory, admin_factory),
    ):
        with pytest.raises(PostgresLegacySnapshotImportInputError):
            PostgresLegacySnapshotImport(*factories)

    importer = PostgresLegacySnapshotImport(
        source_factory,
        admin_factory,
        migrator_factory,
    )
    with pytest.raises(PostgresLegacySnapshotImportInputError):
        importer.import_snapshot(object(), _preflight_receipt())
    with pytest.raises(PostgresLegacySnapshotImportInputError):
        importer.import_snapshot(_context(), object())
    source_factory.assert_not_called()
    admin_factory.assert_not_called()
    migrator_factory.assert_not_called()


@pytest.mark.parametrize(
    "blocker",
    tuple(FreshTargetCutoverBlocker),
)
def test_every_blocked_preflight_is_rejected_before_any_connection(blocker) -> None:
    source_factory = MagicMock(side_effect=AssertionError("source must stay closed"))
    admin_factory = MagicMock(side_effect=AssertionError("admin must stay closed"))
    migrator_factory = MagicMock(
        side_effect=AssertionError("migrator must stay closed")
    )
    importer = PostgresLegacySnapshotImport(
        source_factory,
        admin_factory,
        migrator_factory,
    )
    receipt = _preflight_receipt(
        status=FreshTargetCutoverStatus.BLOCKED,
        blockers=(blocker,),
    )

    with pytest.raises(PostgresLegacySnapshotImportInputError):
        importer.import_snapshot(_context(), receipt)

    source_factory.assert_not_called()
    admin_factory.assert_not_called()
    migrator_factory.assert_not_called()


def test_source_lock_driver_detail_is_redacted_and_connection_is_closed() -> None:
    secret = "postgresql://source:never-print@example.invalid/elvis"

    class RawLockBusyError(RuntimeError):
        pgcode = "55P03"

    connection = _fresh_connection_double()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = (
        None,
        None,
        None,
        RawLockBusyError(secret),
    )
    importer = PostgresLegacySnapshotImport(
        lambda: connection,
        lambda: MagicMock(),
        lambda: MagicMock(),
    )

    with pytest.raises(PostgresLegacySnapshotImportBusyError) as raised:
        importer._open_source(_context(), _preflight_receipt())

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    _assert_secret_absent_from_exception_graph(raised.value, secret)
    connection.rollback.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_raw_migrator_identity_failure_is_redacted_and_connection_is_closed(
    monkeypatch,
) -> None:
    secret = "postgresql://migrator:never-print@example.invalid/elvis"
    connection = _fresh_connection_double()

    def fail_identity(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(secret)

    monkeypatch.setattr(
        "trading.persistence.postgres_legacy_snapshot_import."
        "PostgresBootstrap._require_migrator_connection_identity",
        fail_identity,
    )
    importer = PostgresLegacySnapshotImport(
        lambda: MagicMock(),
        lambda: MagicMock(),
        lambda: connection,
    )

    with pytest.raises(PostgresLegacySnapshotImportStorageError) as raised:
        importer._open_migrator(_context())

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    _assert_secret_absent_from_exception_graph(raised.value, secret)
    connection.rollback.assert_not_called()
    connection.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("rows", "scan_limits"),
    (
        (((1, "x" * (64 * 1024), 0.0, dt.datetime(2026, 8, 13)),), {}),
        (
            (
                (1, "asset-a", 0.0, dt.datetime(2026, 8, 13)),
                (2, "asset-b", 1.0, dt.datetime(2026, 8, 13)),
            ),
            {"remaining_rows": 1},
        ),
    ),
)
def test_relation_scan_rejects_row_or_total_bound_before_batch_consumer(
    rows,
    scan_limits,
) -> None:
    cursor = MagicMock()
    cursor.fetchmany.side_effect = (rows, ())
    connection = MagicMock()
    connection.cursor.return_value = cursor
    consumer = MagicMock()

    with pytest.raises(PostgresLegacySnapshotImportConflict):
        importer_module._scan_relation(
            connection,
            "np.account_balances",
            512,
            "bounded_canary",
            consumer,
            **scan_limits,
        )

    consumer.assert_not_called()
    cursor.close.assert_called_once_with()


def test_application_contract_has_no_persistence_dependency() -> None:
    path = Path("trading/application/legacy_snapshot_import.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name.startswith("trading.persistence") for name in imports)


def test_adapter_has_no_business_data_spool_or_forbidden_repair_sql() -> None:
    path = Path("trading/persistence/postgres_legacy_snapshot_import.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert imports.isdisjoint({"shelve", "sqlite3", "tempfile"})

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "open" not in called_names
    assert called_attributes.isdisjoint(
        {"dump", "dumps_to_file", "write", "write_bytes", "write_text"}
    )

    string_literals = "\n".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ).upper()
    for forbidden in (
        "ALTER TABLE",
        "CREATE TABLE",
        "DELETE FROM",
        "DISABLE TRIGGER",
        "DROP TABLE",
        "GRANT ",
        "ON CONFLICT",
        "REVOKE ",
        "TRUNCATE",
    ):
        assert forbidden not in string_literals


def test_commit_acknowledgement_recovery_contract_is_replay_not_imported() -> None:
    path = Path("trading/persistence/postgres_legacy_snapshot_import.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PostgresLegacySnapshotImport"
    )
    import_method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_import_snapshot"
    )
    copy_method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_copy_or_replay"
    )
    import_source = ast.unparse(import_method)
    copy_source = ast.unparse(copy_method)

    assert "disposition = self._copy_or_replay" in import_source
    assert "LegacySnapshotImportDisposition.REPLAYED" in copy_source
    assert "commit" in copy_source.lower()
    assert "PostgresLegacySnapshotImportCommitUnknown" in copy_source
    # Commit-exception readback must communicate exact recovery back to the
    # outer disposition; a bare return would mislabel durable recovery IMPORTED.
    commit_try = next(
        node
        for node in ast.walk(copy_method)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "commit"
            for child in ast.walk(node)
        )
    )
    assert any(
        isinstance(node, ast.Return) and node.value is not None
        for node in ast.walk(commit_try)
    )


def test_postcommit_nonexact_readback_is_a_conflict_without_repair() -> None:
    path = Path("trading/persistence/postgres_legacy_snapshot_import.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PostgresLegacySnapshotImport"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_require_target_rows_exact"
    )
    method_source = ast.unparse(method)
    assert "PostgresLegacySnapshotImportConflict" in method_source
    assert "actual != expected" in method_source
    assert "DELETE FROM" not in source.upper()
    assert "TRUNCATE" not in source.upper()
    assert "ON CONFLICT" not in source.upper()


def test_target_cluster_binding_and_locked_terminal_recheck_precede_copy() -> None:
    path = Path("trading/persistence/postgres_legacy_snapshot_import.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PostgresLegacySnapshotImport"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
    }
    prepare_source = methods["_prepare_target_transaction"]
    copy_source = methods["_copy_or_replay"]
    normalize_source = methods["_normalize_sequences"]

    assert "target_system_identifier" in prepare_source
    assert prepare_source.index("_SELECT_MIGRATOR_TARGET_IDENTITY_SQL") < (
        prepare_source.index("SET LOCAL ROLE")
    )
    assert prepare_source.index("SET LOCAL ROLE") < prepare_source.index(
        "_LOCK_IMPORT_TABLES_SQL"
    )
    assert "preflight.target.system_identifier" in copy_source
    assert copy_source.index("self._prepare_target_transaction") < copy_source.index(
        "locked_state = self._inspect_target"
    )
    assert copy_source.index("locked_state = self._inspect_target") < (
        copy_source.index("consumers =")
    )
    assert "target_system_identifier" in normalize_source
    assert "self._prepare_target_transaction" in normalize_source
    assert normalize_source.index("self._inspect_target") < normalize_source.index(
        "target_sequence_locked"
    )
    assert normalize_source.index("target_sequence_locked") < normalize_source.index(
        "setval"
    )

    import_source = methods["_import_snapshot"]
    assert import_source.index("desired_sequences =") < import_source.index(
        "target_state = self._inspect_target"
    )
    assert import_source.index("_POSTGRES_INTEGER_MAX") < import_source.index(
        "target_state = self._inspect_target"
    )


def test_cli_requires_all_operator_confirmations_before_file_or_service_access(
    tmp_path,
    capsys,
) -> None:
    config, receipt = _cli_paths(tmp_path)
    required_flags = (
        "--import-snapshot",
        "--confirm-stopped-source-clone",
        "--confirm-exclusive-database-window",
        "--confirm-disposable-target",
    )

    for missing in required_flags:
        resolver = MagicMock(side_effect=AssertionError("services must stay offline"))
        arguments = [
            value for value in _cli_arguments(config, receipt) if value != missing
        ]
        assert cli.main(arguments, service_connection_factory=resolver) == 2
        resolver.assert_not_called()
        _assert_cli_error(capsys, "INPUT")


def test_deploy_example_matches_the_closed_three_service_manifest() -> None:
    path = Path("deploy/v2/legacy-snapshot-import-v1.example.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document) == {"schema_version", "batch_size", "source", "target"}
    assert document["schema_version"] == 1
    assert type(document["batch_size"]) is int
    assert 1 <= document["batch_size"] <= 512
    assert set(document["source"]) == {
        "service",
        "expected_database",
        "expected_role",
    }
    assert set(document["target"]) == {
        "admin_service",
        "migrator_service",
        "bootstrap_context",
    }
    services = (
        document["source"]["service"],
        document["target"]["admin_service"],
        document["target"]["migrator_service"],
    )
    assert len(set(services)) == 3
    serialized = json.dumps(document, sort_keys=True).lower()
    for forbidden in ("password", "passfile", "postgresql://", '"dsn"'):
        assert forbidden not in serialized


@pytest.mark.parametrize("batch_size", (1, 512))
def test_cli_accepts_batch_boundaries_and_emits_non_authoritative_receipt(
    tmp_path,
    capsys,
    batch_size,
) -> None:
    config, receipt_path = _cli_paths(tmp_path)
    _write_json(config, _config_document(batch_size=batch_size))
    factories = {
        "elvis_source_clone": MagicMock(name="source_factory"),
        "elvis_target_admin": MagicMock(name="admin_factory"),
        "elvis_target_migrator": MagicMock(name="migrator_factory"),
    }
    resolver = MagicMock(side_effect=factories.__getitem__)
    adapter = MagicMock()
    adapter.import_snapshot.return_value = replace(
        _import_receipt(),
        context=_context(batch_size=batch_size),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        constructor = MagicMock(return_value=adapter)
        monkeypatch.setattr(cli, "PostgresLegacySnapshotImport", constructor)
        assert (
            cli.main(
                _cli_arguments(config, receipt_path),
                service_connection_factory=resolver,
            )
            == 0
        )

    constructor.assert_called_once_with(
        factories["elvis_source_clone"],
        factories["elvis_target_admin"],
        factories["elvis_target_migrator"],
    )
    imported_context, parsed_preflight = adapter.import_snapshot.call_args.args
    assert imported_context.batch_size == batch_size
    assert parsed_preflight == _preflight_receipt()
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "IMPORTED"
    assert output["stale_on_return"] is True
    assert output["snapshot_authoritative"] is False
    assert output["target_exact"] is True
    assert output["runtime_activation_authorized"] is False
    assert len(output["relations"]) == 7


def test_cli_rejects_strict_config_failures_before_service_resolution(
    tmp_path,
    capsys,
) -> None:
    valid_config, receipt = _cli_paths(tmp_path)
    invalid_documents = []
    for invalid_batch in (True, 0, 513, -1, "512"):
        invalid_documents.append(_config_document(batch_size=invalid_batch))
    missing = _config_document()
    del missing["batch_size"]
    invalid_documents.append(missing)
    unknown = _config_document()
    unknown["unknown"] = "closed-schema"
    invalid_documents.append(unknown)
    missing_service = _config_document()
    del missing_service["target"]["migrator_service"]
    invalid_documents.append(missing_service)
    duplicate_service = _config_document()
    duplicate_service["target"]["migrator_service"] = duplicate_service["target"][
        "admin_service"
    ]
    invalid_documents.append(duplicate_service)
    secret_key = _config_document()
    secret_key["source"]["dsn"] = "postgresql://user:password@host/database"
    invalid_documents.append(secret_key)

    for index, document in enumerate(invalid_documents):
        path = tmp_path / f"invalid-config-{index}.json"
        _write_json(path, document)
        resolver = MagicMock(side_effect=AssertionError("services must stay offline"))
        assert (
            cli.main(
                _cli_arguments(path, receipt),
                service_connection_factory=resolver,
            )
            == 2
        )
        resolver.assert_not_called()
        _assert_cli_error(capsys, "INPUT")

    for index, raw in enumerate(
        (
            '{"schema_version":1,"schema_version":1}',
            '{"schema_version":NaN}',
            "{" + '"padding":"' + ("x" * 65_536) + '"}',
        )
    ):
        path = tmp_path / f"invalid-config-raw-{index}.json"
        path.write_text(raw, encoding="utf-8")
        path.chmod(0o600)
        resolver = MagicMock(side_effect=AssertionError("services must stay offline"))
        assert (
            cli.main(
                _cli_arguments(path, receipt),
                service_connection_factory=resolver,
            )
            == 2
        )
        resolver.assert_not_called()
        _assert_cli_error(capsys, "INPUT")

    assert valid_config.exists()


def test_cli_rejects_strict_or_nonready_preflight_before_service_resolution(
    tmp_path,
    capsys,
) -> None:
    config, _receipt = _cli_paths(tmp_path)
    invalid_documents = []
    blocked = _preflight_document()
    blocked["status"] = "BLOCKED"
    blocked["blockers"] = ["SOURCE_SCHEMA"]
    invalid_documents.append(blocked)
    authoritative = _preflight_document()
    authoritative["snapshot_authoritative"] = True
    invalid_documents.append(authoritative)
    not_stale = _preflight_document()
    not_stale["stale_on_return"] = False
    invalid_documents.append(not_stale)
    uppercase_hash = _preflight_document()
    uppercase_hash["source"]["canonical_sha256"] = "A" * 64
    invalid_documents.append(uppercase_hash)
    numeric_identifier = _preflight_document()
    numeric_identifier["source"]["system_identifier"] = 11
    invalid_documents.append(numeric_identifier)
    reversed_relations = _preflight_document()
    reversed_relations["source"]["relations"] = list(
        reversed(reversed_relations["source"]["relations"])
    )
    invalid_documents.append(reversed_relations)
    unknown = _preflight_document()
    unknown["unknown"] = []
    invalid_documents.append(unknown)
    missing = _preflight_document()
    del missing["target"]
    invalid_documents.append(missing)

    for index, document in enumerate(invalid_documents):
        path = tmp_path / f"invalid-receipt-{index}.json"
        _write_json(path, document)
        resolver = MagicMock(side_effect=AssertionError("services must stay offline"))
        assert (
            cli.main(
                _cli_arguments(config, path),
                service_connection_factory=resolver,
            )
            == 2
        )
        resolver.assert_not_called()
        _assert_cli_error(capsys, "INPUT")

    for index, raw in enumerate(
        (
            '{"status":"READY_FOR_FRESH_TARGET","status":"READY_FOR_FRESH_TARGET"}',
            '{"status":NaN}',
            "{" + '"padding":"' + ("x" * 65_536) + '"}',
        )
    ):
        path = tmp_path / f"invalid-receipt-raw-{index}.json"
        path.write_text(raw, encoding="utf-8")
        path.chmod(0o600)
        resolver = MagicMock(side_effect=AssertionError("services must stay offline"))
        assert (
            cli.main(
                _cli_arguments(config, path),
                service_connection_factory=resolver,
            )
            == 2
        )
        resolver.assert_not_called()
        _assert_cli_error(capsys, "INPUT")


@pytest.mark.parametrize(
    ("error", "exit_code", "error_code"),
    (
        (PostgresLegacySnapshotImportStorageError("storage"), 20, "STORAGE"),
        (PostgresLegacySnapshotImportBusyError("busy"), 22, "BUSY"),
        (PostgresLegacySnapshotImportConflict("conflict"), 23, "CONFLICT"),
        (
            PostgresLegacySnapshotImportCommitUnknown("commit unknown"),
            24,
            "COMMIT_UNKNOWN",
        ),
        (RuntimeError("internal"), 70, "INTERNAL"),
    ),
)
def test_cli_maps_sanitized_failures_without_exposing_exception_text(
    tmp_path,
    capsys,
    error,
    exit_code,
    error_code,
) -> None:
    config, receipt = _cli_paths(tmp_path)
    resolver = MagicMock(side_effect=(MagicMock(), MagicMock(), MagicMock()))
    adapter = MagicMock()
    adapter.import_snapshot.side_effect = error

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            cli,
            "PostgresLegacySnapshotImport",
            MagicMock(return_value=adapter),
        )
        assert (
            cli.main(
                _cli_arguments(config, receipt),
                service_connection_factory=resolver,
            )
            == exit_code
        )

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "ERROR", "code": error_code}
    assert str(error) not in captured.out
    assert captured.err == ""
