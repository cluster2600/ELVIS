"""Adversarial contracts for the read-only legacy snapshot reconciliation."""

from __future__ import annotations

import ast
import json
import os
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import postgres_legacy_snapshot_reconciliation as cli
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
    LegacyOpeningCandidate,
    LegacyOpeningCandidateSource,
    LegacySnapshotReconciliationContext,
    LegacySnapshotReconciliationDisposition,
    LegacySnapshotReconciliationEvidence,
    LegacySnapshotReconciliationFinding,
    LegacySnapshotReconciliationFindingKind,
    LegacySnapshotReconciliationReceipt,
    legacy_opening_candidate_sha256,
    legacy_opening_quantization_required,
    legacy_operator_equity_hypothesis_balances,
    legacy_snapshot_import_receipt_sha256,
    legacy_snapshot_relation_evidence_sha256,
)
from trading.domain.paper_accounting import PaperAccountBalance
from trading.persistence.postgres_legacy_snapshot_reconciliation import (
    PostgresLegacySnapshotReconciliationConflict,
    PostgresLegacySnapshotReconciliationInputError,
    PostgresLegacySnapshotReconciliationStorageError,
)

# flake8: noqa: E501


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


def _import_context() -> LegacySnapshotImportContext:
    return LegacySnapshotImportContext(
        FreshTargetCutoverContext(
            source_expected_database="elvis_source_clone",
            source_expected_role="elvis_source_inspector",
            target_bootstrap_intent=FreshTargetBootstrapIntent(
                expected_database="elvis_fresh_target",
                admin_role="elvis_bootstrap_admin",
                roles=_role_manifest(),
            ),
        ),
        batch_size=512,
    )


def _import_receipt() -> LegacySnapshotImportReceipt:
    relations = tuple(
        LegacySnapshotRelationReceipt(
            name=name,
            row_count=0,
            pk_min=None,
            pk_max=None,
            sha256=_SHA_B,
            source_sequence_next=1,
            target_sequence_next=1,
        )
        for name in _LEGACY_RELATIONS
    )
    return LegacySnapshotImportReceipt(
        context=_import_context(),
        disposition=LegacySnapshotImportDisposition.IMPORTED,
        source_system_identifier=11,
        target_system_identifier=22,
        source_canonical_sha256=legacy_snapshot_relation_evidence_sha256(relations),
        relations=relations,
        target_exact=True,
        runtime_activation_authorized=False,
    )


def _context() -> LegacySnapshotReconciliationContext:
    import_receipt = _import_receipt()
    return LegacySnapshotReconciliationContext(
        import_context=_import_context(),
        config_document_sha256=_SHA_A,
        import_receipt_sha256=legacy_snapshot_import_receipt_sha256(import_receipt),
        execution_scope="paper:compatibility",
        account_key="paper:primary",
        owner_generation=1,
        collateral_asset="USDT",
        margin_quantum=Decimal("0.01"),
        hypothesis_starting_collateral=Decimal("1000"),
    )


def _candidate(
    source: LegacyOpeningCandidateSource,
    *,
    balances: tuple[PaperAccountBalance, ...] | None = None,
    digest: str | None = None,
    context: LegacySnapshotReconciliationContext | None = None,
) -> LegacyOpeningCandidate:
    if balances is None:
        balances = (PaperAccountBalance("USDT", Decimal("1000")),)
    if context is None:
        context = _context()
    return LegacyOpeningCandidate(
        source=source,
        balances=balances,
        opening_payload_sha256=(
            legacy_opening_candidate_sha256(context, balances)
            if digest is None
            else digest
        ),
        available=True,
    )


def _evidence(
    *,
    context: LegacySnapshotReconciliationContext | None = None,
    imported_balances: tuple[PaperAccountBalance, ...] | None = None,
    hypothesis_balances: tuple[PaperAccountBalance, ...] | None = None,
    imported_digest: str | None = None,
    hypothesis_digest: str | None = None,
) -> LegacySnapshotReconciliationEvidence:
    if context is None:
        context = _context()
    canonical_hypothesis = legacy_operator_equity_hypothesis_balances(
        context, Decimal("0")
    )
    if imported_balances is None:
        imported_balances = canonical_hypothesis
    if hypothesis_balances is None:
        hypothesis_balances = canonical_hypothesis
    return LegacySnapshotReconciliationEvidence(
        reset_timestamp="2026-08-13T10:11:12.123456",
        hypothesis_realised_pnl=Decimal("0"),
        hypothesis_trade_fees=Decimal("0.25"),
        hypothesis_liquidation_fees=Decimal("1.5"),
        candidates=(
            _candidate(
                LegacyOpeningCandidateSource.IMPORTED_ACCOUNT_BALANCES,
                balances=imported_balances,
                digest=imported_digest,
                context=context,
            ),
            _candidate(
                LegacyOpeningCandidateSource.OPERATOR_EQUITY_HYPOTHESIS,
                balances=hypothesis_balances,
                digest=hypothesis_digest,
                context=context,
            ),
        ),
    )


def _blocked_evidence() -> LegacySnapshotReconciliationEvidence:
    return LegacySnapshotReconciliationEvidence(
        reset_timestamp=None,
        hypothesis_realised_pnl=Decimal("0"),
        hypothesis_trade_fees=Decimal("0"),
        hypothesis_liquidation_fees=Decimal("0"),
        candidates=tuple(
            LegacyOpeningCandidate(source, (), None, False)
            for source in LegacyOpeningCandidateSource
        ),
    )


def _receipt(
    *,
    disposition: LegacySnapshotReconciliationDisposition = (
        LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
    ),
    findings: tuple[LegacySnapshotReconciliationFinding, ...] | None = None,
    evidence: LegacySnapshotReconciliationEvidence | None = None,
    context: LegacySnapshotReconciliationContext | None = None,
    import_receipt: LegacySnapshotImportReceipt | None = None,
) -> LegacySnapshotReconciliationReceipt:
    if import_receipt is None:
        import_receipt = _import_receipt()
    if context is None:
        context = _context()
    if findings is None:
        findings = (
            _finding(
                LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN
            ),
        )
    return LegacySnapshotReconciliationReceipt(
        context=context,
        import_receipt=import_receipt,
        disposition=disposition,
        findings=findings,
        evidence=_evidence() if evidence is None else evidence,
        target_system_identifier=import_receipt.target_system_identifier,
        source_canonical_sha256=import_receipt.source_canonical_sha256,
        config_document_sha256=context.config_document_sha256,
        import_receipt_sha256=context.import_receipt_sha256,
    )


def _finding(
    kind: LegacySnapshotReconciliationFindingKind,
) -> LegacySnapshotReconciliationFinding:
    return LegacySnapshotReconciliationFinding(kind)


def _config_document() -> dict[str, object]:
    roles = _role_manifest()
    return {
        "schema_version": 1,
        "batch_size": 512,
        "source": {
            "expected_database": "elvis_source_clone",
            "expected_role": "elvis_source_inspector",
        },
        "target": {
            "admin_service": "elvis_target_admin",
            "readiness_service": "elvis_target_readiness",
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
        "opening": {
            "execution_scope": "paper:compatibility",
            "account_key": "paper:primary",
            "owner_generation": 1,
            "collateral_asset": "USDT",
            "margin_quantum_decimal": "0.01",
            "hypothesis_starting_collateral_decimal": "1000",
        },
    }


def _import_receipt_document() -> dict[str, object]:
    receipt = _import_receipt()
    return {
        "status": receipt.disposition.value,
        "source_system_identifier": str(receipt.source_system_identifier),
        "target_system_identifier": str(receipt.target_system_identifier),
        "source_canonical_sha256": receipt.source_canonical_sha256,
        "relations": [
            {
                "name": relation.name,
                "row_count": relation.row_count,
                "pk_min": relation.pk_min,
                "pk_max": relation.pk_max,
                "sha256": relation.sha256,
                "source_sequence_next": relation.source_sequence_next,
                "target_sequence_next": relation.target_sequence_next,
            }
            for relation in receipt.relations
        ],
        "target_exact": receipt.target_exact,
        "runtime_activation_authorized": receipt.runtime_activation_authorized,
        "stale_on_return": receipt.stale_on_return,
        "snapshot_authoritative": receipt.snapshot_authoritative,
    }


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")
    path.chmod(0o600)


def _cli_paths(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "legacy-snapshot-reconciliation-v1.json"
    receipt = tmp_path / "legacy-snapshot-import.json"
    _write_json(config, _config_document())
    _write_json(receipt, _import_receipt_document())
    return config, receipt


def _cli_arguments(config: Path, receipt: Path) -> list[str]:
    return [
        "--config",
        str(config),
        "--import-receipt",
        str(receipt),
        "--assess",
        "--confirm-reviewed-database-window",
        "--confirm-disposable-target",
    ]


def test_context_binds_exact_import_intent_and_decimal_policy() -> None:
    context = _context()

    assert context.import_context == _import_context()
    assert context.config_document_sha256 == _SHA_A
    assert context.import_receipt_sha256 == legacy_snapshot_import_receipt_sha256(
        _import_receipt()
    )
    assert context.margin_quantum == Decimal("0.01")
    assert context.hypothesis_starting_collateral == Decimal("1000")
    assert context.collateral_asset == "USDT"

    for field_name, bad_value in (
        ("import_context", object()),
        ("config_document_sha256", "A" * 64),
        ("import_receipt_sha256", _SHA_A[:-1]),
        ("execution_scope", " paper"),
        ("account_key", ""),
        ("owner_generation", True),
        ("owner_generation", 0),
        ("owner_generation", 1 << 63),
        ("collateral_asset", "USDT "),
        ("margin_quantum", "0.01"),
        ("margin_quantum", Decimal("0")),
        ("margin_quantum", Decimal("NaN")),
        ("hypothesis_starting_collateral", 1000.0),
        ("hypothesis_starting_collateral", Decimal("-0.01")),
        ("hypothesis_starting_collateral", Decimal("Infinity")),
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(context, **{field_name: bad_value})


@pytest.mark.parametrize(
    ("field_name", "length"),
    (("execution_scope", 129), ("account_key", 256), ("collateral_asset", 65)),
)
def test_context_rejects_values_outside_durable_text_bounds(
    field_name: str,
    length: int,
) -> None:
    with pytest.raises(ValueError):
        replace(_context(), **{field_name: "x" * length})


def test_candidate_requires_canonical_complete_opening_payload() -> None:
    imported = _candidate(
        LegacyOpeningCandidateSource.IMPORTED_ACCOUNT_BALANCES,
        balances=(
            PaperAccountBalance("BTC", Decimal("0")),
            PaperAccountBalance("USDT", Decimal("1000")),
        ),
    )
    assert tuple(balance.asset for balance in imported.balances) == ("BTC", "USDT")

    with pytest.raises(ValueError):
        replace(imported, balances=tuple(reversed(imported.balances)))
    with pytest.raises(ValueError):
        replace(
            imported,
            balances=(
                PaperAccountBalance("USDT", Decimal("1")),
                PaperAccountBalance("USDT", Decimal("2")),
            ),
        )
    with pytest.raises(ValueError):
        replace(
            imported,
            balances=(PaperAccountBalance("USDT", Decimal("-0.01")),),
        )
    with pytest.raises(ValueError):
        replace(
            imported,
            balances=(
                PaperAccountBalance(
                    "USDT",
                    Decimal("999"),
                    reserved=Decimal("1"),
                ),
            ),
        )
    with pytest.raises(ValueError):
        replace(imported, opening_payload_sha256="A" * 64)
    with pytest.raises(ValueError):
        replace(imported, available=False)

    unavailable = LegacyOpeningCandidate(
        source=LegacyOpeningCandidateSource.OPERATOR_EQUITY_HYPOTHESIS,
        balances=(),
        opening_payload_sha256=None,
        available=False,
    )
    assert unavailable.available is False


def test_evidence_keeps_pnl_and_the_two_fee_interpretations_separate() -> None:
    evidence = _evidence()

    assert evidence.hypothesis_realised_pnl == Decimal("0")
    assert evidence.hypothesis_trade_fees == Decimal("0.25")
    assert evidence.hypothesis_liquidation_fees == Decimal("1.5")
    assert tuple(candidate.source for candidate in evidence.candidates) == tuple(
        LegacyOpeningCandidateSource
    )

    for field_name, bad_value in (
        ("hypothesis_realised_pnl", float("nan")),
        ("hypothesis_trade_fees", Decimal("NaN")),
        ("hypothesis_trade_fees", Decimal("-0.01")),
        ("hypothesis_liquidation_fees", Decimal("Infinity")),
        ("hypothesis_liquidation_fees", Decimal("-0.01")),
        ("candidates", tuple(reversed(evidence.candidates))),
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(evidence, **{field_name: bad_value})

    for reset_timestamp in (
        "2026-08-13T10:11:12",
        "2026-08-13T10:11:12.123456+00:00",
        "not-a-timestamp",
    ):
        with pytest.raises(ValueError):
            replace(evidence, reset_timestamp=reset_timestamp)


def test_decision_receipt_is_import_bound_stale_and_cannot_grant_authority() -> None:
    receipt = _receipt()

    assert tuple(LegacySnapshotReconciliationDisposition) == (
        LegacySnapshotReconciliationDisposition.DECISION_REQUIRED,
        LegacySnapshotReconciliationDisposition.BLOCKED,
    )
    assert (
        receipt.disposition is LegacySnapshotReconciliationDisposition.DECISION_REQUIRED
    )
    assert {finding.kind for finding in receipt.findings} == {
        LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN
    }
    assert receipt.stale_on_return is True
    assert receipt.snapshot_authoritative is False
    assert receipt.coherent_snapshot_observed is False
    assert receipt.source_provenance_authenticated is False
    assert receipt.target_observations_authenticated is False
    assert receipt.database_window_enforced is False
    assert receipt.account_opening_authorized is False
    assert receipt.account_provisioning_authorized is False
    assert receipt.runtime_activation_authorized is False
    assert (
        receipt.target_system_identifier
        == receipt.import_receipt.target_system_identifier
    )
    assert receipt.source_canonical_sha256 == (
        receipt.import_receipt.source_canonical_sha256
    )

    for field_name, bad_value in (
        ("stale_on_return", False),
        ("snapshot_authoritative", True),
        ("coherent_snapshot_observed", True),
        ("source_provenance_authenticated", True),
        ("target_observations_authenticated", True),
        ("database_window_enforced", True),
        ("account_opening_authorized", True),
        ("account_provisioning_authorized", True),
        ("runtime_activation_authorized", True),
    ):
        with pytest.raises(ValueError):
            replace(receipt, **{field_name: bad_value})


def test_decision_recomputes_candidate_hashes_and_requires_provenance() -> None:
    provenance = _finding(
        LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN
    )
    mismatch = _finding(LegacySnapshotReconciliationFindingKind.CANDIDATE_MISMATCH)
    diverged = _evidence(
        imported_balances=(PaperAccountBalance("USDT", Decimal("999")),)
    )
    decision = _receipt(
        findings=(provenance, mismatch),
        evidence=diverged,
    )
    assert decision.evidence.candidates[0].opening_payload_sha256 != (
        decision.evidence.candidates[1].opening_payload_sha256
    )

    forged_hash = _evidence(imported_digest=_SHA_A)
    with pytest.raises(ValueError):
        _receipt(evidence=forged_hash)

    with pytest.raises(ValueError):
        _receipt(findings=(), evidence=_evidence())
    with pytest.raises(ValueError):
        _receipt(findings=(provenance,), evidence=diverged)
    with pytest.raises(ValueError):
        _receipt(findings=(provenance, provenance))

    missing_collateral = _evidence(
        imported_balances=(
            PaperAccountBalance("BNB", Decimal("0")),
            PaperAccountBalance("BTC", Decimal("0")),
        )
    )
    with pytest.raises(ValueError, match="contain the collateral asset"):
        _receipt(
            findings=(provenance, mismatch),
            evidence=missing_collateral,
        )

    contradictory_hypothesis = _evidence(
        hypothesis_balances=(
            PaperAccountBalance("BNB", Decimal("0")),
            PaperAccountBalance("BTC", Decimal("0")),
            PaperAccountBalance("USDT", Decimal("999")),
        )
    )
    with pytest.raises(ValueError, match="operator hypothesis"):
        _receipt(
            findings=(provenance, mismatch),
            evidence=contradictory_hypothesis,
        )


def test_decision_rejects_false_positive_and_negative_quantization_findings() -> None:
    provenance = _finding(
        LegacySnapshotReconciliationFindingKind.RUNTIME_PROVENANCE_UNPROVEN
    )
    quantization = _finding(
        LegacySnapshotReconciliationFindingKind.QUANTIZATION_REQUIRED
    )
    exact_evidence = _evidence()
    assert (
        legacy_opening_quantization_required(_context(), exact_evidence.candidates)
        is False
    )
    with pytest.raises(ValueError, match="quantization finding"):
        _receipt(findings=(provenance, quantization), evidence=exact_evidence)

    quantized_context = replace(
        _context(),
        hypothesis_starting_collateral=Decimal("0.1"),
    )
    quantized_evidence = _evidence(context=quantized_context)
    assert (
        legacy_opening_quantization_required(
            quantized_context, quantized_evidence.candidates
        )
        is True
    )
    with pytest.raises(ValueError, match="quantization finding"):
        _receipt(
            context=quantized_context,
            findings=(provenance,),
            evidence=quantized_evidence,
        )


def test_blocked_receipt_rejects_partial_or_forged_opening_evidence() -> None:
    blocked_kind = _finding(
        LegacySnapshotReconciliationFindingKind.TARGET_CATALOG_DRIFT
    )
    blocked = _receipt(
        disposition=LegacySnapshotReconciliationDisposition.BLOCKED,
        findings=(blocked_kind,),
        evidence=_blocked_evidence(),
    )
    assert blocked.disposition is LegacySnapshotReconciliationDisposition.BLOCKED

    for forged_evidence in (
        _evidence(),
        replace(_blocked_evidence(), reset_timestamp="2026-08-13T00:00:00.000000"),
        replace(_blocked_evidence(), hypothesis_realised_pnl=Decimal("1")),
        replace(_blocked_evidence(), hypothesis_trade_fees=Decimal("1")),
        replace(_blocked_evidence(), hypothesis_liquidation_fees=Decimal("1")),
    ):
        with pytest.raises(ValueError):
            _receipt(
                disposition=LegacySnapshotReconciliationDisposition.BLOCKED,
                findings=(blocked_kind,),
                evidence=forged_evidence,
            )

    decision_only = _finding(
        LegacySnapshotReconciliationFindingKind.QUANTIZATION_REQUIRED
    )
    with pytest.raises(ValueError):
        _receipt(
            disposition=LegacySnapshotReconciliationDisposition.BLOCKED,
            findings=(decision_only,),
            evidence=_blocked_evidence(),
        )


def test_receipt_rejects_config_import_relation_and_identity_substitution() -> None:
    receipt = _receipt()

    with pytest.raises(ValueError):
        replace(
            receipt,
            context=replace(
                _context(),
                import_context=replace(_import_context(), batch_size=1),
            ),
        )
    with pytest.raises(ValueError):
        replace(receipt, target_system_identifier=23)
    with pytest.raises(ValueError):
        replace(receipt, source_canonical_sha256=_SHA_B)
    with pytest.raises(ValueError):
        replace(receipt, config_document_sha256=_SHA_B)
    with pytest.raises(ValueError):
        replace(receipt, import_receipt_sha256=_SHA_B)

    relation = replace(receipt.import_receipt.relations[0], sha256=_SHA_A)
    tampered_import = replace(
        receipt.import_receipt,
        relations=(relation, *receipt.import_receipt.relations[1:]),
    )
    tampered_context = replace(
        receipt.context,
        import_receipt_sha256=legacy_snapshot_import_receipt_sha256(tampered_import),
    )
    with pytest.raises(ValueError, match="relation evidence"):
        _receipt(context=tampered_context, import_receipt=tampered_import)


def test_contracts_are_frozen_and_validate_pickle_restore() -> None:
    values = (
        _context(),
        _candidate(LegacyOpeningCandidateSource.IMPORTED_ACCOUNT_BALANCES),
        _finding(LegacySnapshotReconciliationFindingKind.CANDIDATE_MISMATCH),
        _evidence(),
        _receipt(),
    )
    for value in values:
        first_field = fields(value)[0].name
        with pytest.raises((FrozenInstanceError, TypeError)):
            setattr(value, first_field, object())
        assert pickle.loads(pickle.dumps(value)) == value


def test_reconciliation_is_not_wired_to_runtime_or_deployment() -> None:
    root = Path(__file__).parents[1]
    allowed = {
        Path("scripts/postgres_legacy_snapshot_reconciliation.py"),
        Path("trading/persistence/postgres_legacy_snapshot_reconciliation.py"),
    }
    consumers = set()
    for source_path in sorted(root.rglob("*.py")):
        if "tests" in source_path.parts or ".venv" in source_path.parts:
            continue
        if (
            source_path
            == root / "trading/application/legacy_snapshot_reconciliation.py"
        ):
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports_reconciliation = any(
            (
                isinstance(node, ast.Import)
                and any(
                    alias.name.endswith("legacy_snapshot_reconciliation")
                    for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.endswith("legacy_snapshot_reconciliation")
            )
            for node in ast.walk(tree)
        )
        if imports_reconciliation:
            consumers.add(source_path.relative_to(root))

    assert consumers == {path for path in allowed if (root / path).is_file()}


@pytest.mark.parametrize(
    ("result", "expected_exit", "expected_status"),
    (
        (_receipt(), 10, "DECISION_REQUIRED"),
        (
            _receipt(
                disposition=LegacySnapshotReconciliationDisposition.BLOCKED,
                findings=(
                    _finding(
                        LegacySnapshotReconciliationFindingKind.TARGET_CATALOG_DRIFT
                    ),
                ),
                evidence=replace(
                    _blocked_evidence(),
                ),
            ),
            21,
            "BLOCKED",
        ),
    ),
)
def test_cli_resolves_two_services_reconciles_once_and_emits_compact_receipt(
    tmp_path: Path,
    capsys,
    monkeypatch,
    result: LegacySnapshotReconciliationReceipt,
    expected_exit: int,
    expected_status: str,
) -> None:
    config, receipt = _cli_paths(tmp_path)
    factories = {
        "elvis_target_admin": MagicMock(name="admin_factory"),
        "elvis_target_readiness": MagicMock(name="readiness_factory"),
    }
    resolver = MagicMock(side_effect=factories.__getitem__)
    adapter = MagicMock()
    adapter.reconcile.return_value = result
    constructor = MagicMock(return_value=adapter)
    monkeypatch.setattr(cli, "PostgresLegacySnapshotReconciliation", constructor)

    assert (
        cli.main(
            _cli_arguments(config, receipt),
            service_connection_factory=resolver,
        )
        == expected_exit
    )

    resolver.assert_any_call("elvis_target_admin")
    resolver.assert_any_call("elvis_target_readiness")
    assert resolver.call_count == 2
    constructor.assert_called_once_with(
        factories["elvis_target_admin"],
        factories["elvis_target_readiness"],
    )
    called_context, called_import_receipt = adapter.reconcile.call_args.args
    assert called_context.import_context == _context().import_context
    assert called_context.execution_scope == _context().execution_scope
    assert called_context.account_key == _context().account_key
    assert called_context.owner_generation == _context().owner_generation
    assert called_context.collateral_asset == _context().collateral_asset
    assert called_context.margin_quantum == _context().margin_quantum
    assert called_context.hypothesis_starting_collateral == (
        _context().hypothesis_starting_collateral
    )
    assert called_context.config_document_sha256 == cli._canonical_document_sha256(
        _config_document()
    )
    assert called_context.import_receipt_sha256 == (
        legacy_snapshot_import_receipt_sha256(called_import_receipt)
    )
    assert called_import_receipt == _import_receipt()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    assert captured.out.count("\n") == 1
    assert ": " not in captured.out
    output = json.loads(captured.out)
    assert output["status"] == expected_status
    assert output["import_disposition"] == "IMPORTED"
    assert output["declared_source_system_identifier"] == "11"
    assert output["account_opening_authorized"] is False
    assert output["account_provisioning_authorized"] is False
    assert output["runtime_activation_authorized"] is False
    assert output["stale_on_return"] is True
    assert output["snapshot_authoritative"] is False
    assert output["coherent_snapshot_observed"] is False
    assert output["source_provenance_authenticated"] is False
    assert output["target_observations_authenticated"] is False
    assert output["database_window_enforced"] is False
    assert output["evidence"]["hypothesis_trade_fees_decimal"] == str(
        result.evidence.hypothesis_trade_fees
    )
    assert output["evidence"]["hypothesis_liquidation_fees_decimal"] == str(
        result.evidence.hypothesis_liquidation_fees
    )


@pytest.mark.parametrize(
    "missing",
    (
        "--assess",
        "--confirm-reviewed-database-window",
        "--confirm-disposable-target",
    ),
)
def test_cli_requires_confirmations_before_file_or_service_access(
    tmp_path: Path,
    capsys,
    missing: str,
) -> None:
    config, receipt = _cli_paths(tmp_path)
    config.unlink()
    resolver = MagicMock(side_effect=AssertionError("must remain offline"))

    arguments = [value for value in _cli_arguments(config, receipt) if value != missing]
    assert cli.main(arguments, service_connection_factory=resolver) == 2
    resolver.assert_not_called()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "ERROR", "code": "INPUT"}
    assert captured.err == ""


@pytest.mark.parametrize(
    ("mutate", "raw_payload"),
    (
        (lambda document: document.update(extra=True), None),
        (
            lambda document: document["target"].update(
                readiness_service=document["target"]["admin_service"]
            ),
            None,
        ),
        (
            lambda document: document["opening"].update(margin_quantum_decimal="+0.01"),
            None,
        ),
        (
            lambda document: document["opening"].update(
                hypothesis_starting_collateral_decimal="NaN"
            ),
            None,
        ),
        (
            lambda _document: None,
            '{"schema_version":1,"schema_version":1}',
        ),
    ),
)
def test_cli_rejects_noncanonical_or_open_config_before_service_resolution(
    tmp_path: Path,
    capsys,
    mutate,
    raw_payload: str | None,
) -> None:
    config, receipt = _cli_paths(tmp_path)
    document = _config_document()
    mutate(document)
    if raw_payload is None:
        _write_json(config, document)
    else:
        config.write_text(raw_payload, encoding="utf-8")
    resolver = MagicMock(side_effect=AssertionError("must remain offline"))

    assert (
        cli.main(
            _cli_arguments(config, receipt),
            service_connection_factory=resolver,
        )
        == 2
    )
    resolver.assert_not_called()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "ERROR", "code": "INPUT"}
    assert captured.err == ""


@pytest.mark.parametrize(
    "mutation",
    (
        lambda document: document.update(status="READY"),
        lambda document: document.update(target_exact=False),
        lambda document: document.update(runtime_activation_authorized=True),
        lambda document: document.update(stale_on_return=False),
        lambda document: document.update(snapshot_authoritative=True),
        lambda document: document.update(source_system_identifier=11),
        lambda document: document["relations"].reverse(),
        lambda document: document["relations"][0].update(sha256="A" * 64),
        lambda document: document.update(extra="forbidden"),
    ),
)
def test_cli_rejects_nonexact_import_receipt_before_service_resolution(
    tmp_path: Path,
    capsys,
    mutation,
) -> None:
    config, receipt = _cli_paths(tmp_path)
    document = _import_receipt_document()
    mutation(document)
    _write_json(receipt, document)
    resolver = MagicMock(side_effect=AssertionError("must remain offline"))

    assert (
        cli.main(
            _cli_arguments(config, receipt),
            service_connection_factory=resolver,
        )
        == 2
    )
    resolver.assert_not_called()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "ERROR", "code": "INPUT"}
    assert captured.err == ""


def test_cli_rejects_symlinks_oversize_files_and_same_resolved_factory(
    tmp_path: Path,
    capsys,
) -> None:
    config, receipt = _cli_paths(tmp_path)
    link = tmp_path / "config-link.json"
    link.symlink_to(config)
    resolver = MagicMock(side_effect=AssertionError("must remain offline"))
    arguments = _cli_arguments(link, receipt)
    assert cli.main(arguments, service_connection_factory=resolver) == 2
    resolver.assert_not_called()
    assert json.loads(capsys.readouterr().out)["code"] == "INPUT"

    fifo = tmp_path / "receipt.fifo"
    os.mkfifo(fifo)
    assert (
        cli.main(
            _cli_arguments(config, fifo),
            service_connection_factory=resolver,
        )
        == 2
    )
    resolver.assert_not_called()
    assert json.loads(capsys.readouterr().out)["code"] == "INPUT"

    config.write_bytes(b" " * 65_537)
    assert (
        cli.main(
            _cli_arguments(config, receipt),
            service_connection_factory=resolver,
        )
        == 2
    )
    resolver.assert_not_called()
    assert json.loads(capsys.readouterr().out)["code"] == "INPUT"

    _write_json(config, _config_document())
    same_factory = MagicMock()
    assert (
        cli.main(
            _cli_arguments(config, receipt),
            service_connection_factory=lambda _service: same_factory,
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["code"] == "INPUT"


def test_cli_fails_closed_without_nofollow_support(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    config, receipt = _cli_paths(tmp_path)
    resolver = MagicMock(side_effect=AssertionError("must remain offline"))
    monkeypatch.delattr(cli.os, "O_NOFOLLOW", raising=False)

    assert (
        cli.main(
            _cli_arguments(config, receipt),
            service_connection_factory=resolver,
        )
        == 2
    )
    resolver.assert_not_called()
    assert json.loads(capsys.readouterr().out) == {
        "status": "ERROR",
        "code": "INPUT",
    }


@pytest.mark.parametrize(
    ("error", "expected_exit", "expected_code"),
    (
        (PostgresLegacySnapshotReconciliationStorageError("private"), 20, "STORAGE"),
        (PostgresLegacySnapshotReconciliationInputError("private"), 2, "INPUT"),
        (PostgresLegacySnapshotReconciliationConflict("private"), 23, "CONFLICT"),
        (RuntimeError("private"), 70, "INTERNAL"),
    ),
)
def test_cli_maps_errors_without_leaking_exception_or_service_data(
    tmp_path: Path,
    capsys,
    monkeypatch,
    error: BaseException,
    expected_exit: int,
    expected_code: str,
) -> None:
    config, receipt = _cli_paths(tmp_path)
    adapter = MagicMock()
    adapter.reconcile.side_effect = error
    monkeypatch.setattr(
        cli,
        "PostgresLegacySnapshotReconciliation",
        MagicMock(return_value=adapter),
    )
    secret = "postgresql://admin:never-print@example.invalid/elvis"
    factories = {
        "elvis_target_admin": MagicMock(name=secret),
        "elvis_target_readiness": MagicMock(name="readiness"),
    }

    assert (
        cli.main(
            _cli_arguments(config, receipt),
            service_connection_factory=factories.__getitem__,
        )
        == expected_exit
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "ERROR",
        "code": expected_code,
    }
    assert captured.err == ""
    assert "private" not in captured.out
    assert secret not in captured.out


def test_deploy_example_is_closed_secret_free_and_uses_distinct_services() -> None:
    path = Path("deploy/v2/legacy-snapshot-reconciliation-v1.example.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document) == {
        "schema_version",
        "batch_size",
        "source",
        "target",
        "opening",
    }
    assert document["schema_version"] == 1
    assert type(document["batch_size"]) is int
    assert 1 <= document["batch_size"] <= 512
    assert set(document["source"]) == {"expected_database", "expected_role"}
    assert set(document["target"]) == {
        "admin_service",
        "readiness_service",
        "bootstrap_context",
    }
    assert document["target"]["admin_service"] != (
        document["target"]["readiness_service"]
    )
    assert set(document["opening"]) == {
        "execution_scope",
        "account_key",
        "owner_generation",
        "collateral_asset",
        "margin_quantum_decimal",
        "hypothesis_starting_collateral_decimal",
    }
    serialized = json.dumps(document, sort_keys=True).lower()
    for forbidden in (
        "password",
        "passfile",
        "postgresql://",
        '"dsn"',
        '"host"',
        '"port"',
    ):
        assert forbidden not in serialized
