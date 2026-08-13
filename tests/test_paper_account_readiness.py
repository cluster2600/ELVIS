"""Adversarial contracts for pure paper-account readiness evidence."""

import ast
import copy
import importlib.util
import inspect
import pickle
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from trading.application import (
    LegacyRelationWatermark,
    MigrationIdentity,
    PaperAccountReadinessAssessment,
    PaperAccountReadinessContext,
    PaperAccountReadinessDisposition,
    PaperAccountReadinessFinding,
    PaperAccountReadinessFindingKind,
    PaperAccountReadinessPort,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
BIGINT_MAX = (1 << 63) - 1
INTEGER_MIN = -(1 << 31)
INTEGER_MAX = (1 << 31) - 1
LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)
PUBLIC_EXPORTS = {
    "LegacyRelationWatermark",
    "MigrationIdentity",
    "PaperAccountReadinessAssessment",
    "PaperAccountReadinessContext",
    "PaperAccountReadinessDisposition",
    "PaperAccountReadinessFinding",
    "PaperAccountReadinessFindingKind",
    "PaperAccountReadinessPort",
}


def context(**overrides: object) -> PaperAccountReadinessContext:
    values = {
        "execution_scope": "paper:test",
        "account_key": "account-1",
        "owner_generation": 7,
        "opening_payload_sha256": SHA_A,
    }
    values.update(overrides)
    return PaperAccountReadinessContext(**values)


def migrations() -> tuple[MigrationIdentity, ...]:
    return (
        MigrationIdentity(1, "legacy_baseline", SHA_A),
        MigrationIdentity(2, "order_position_journal", SHA_B),
        MigrationIdentity(3, "paper_account_ledger", "c" * 64),
    )


def watermarks(
    *, open_positions: int = 0, reverse: bool = False
) -> tuple[LegacyRelationWatermark, ...]:
    values = tuple(
        LegacyRelationWatermark(
            relation,
            open_positions if relation == "np.open_positions" else 0,
            19 if relation == "np.open_positions" and open_positions else None,
        )
        for relation in LEGACY_RELATIONS
    )
    return tuple(reversed(values)) if reverse else values


def finding(
    kind: PaperAccountReadinessFindingKind,
    subject_id: str = "account-1",
    *,
    subject_kind: str = "paper_account",
) -> PaperAccountReadinessFinding:
    return PaperAccountReadinessFinding(kind, subject_kind, subject_id)


def assessment(**overrides: object) -> PaperAccountReadinessAssessment:
    expected = migrations()
    values = {
        "context": context(),
        "expected_migrations": expected,
        "applied_migrations": expected,
        "account_version": 0,
        "legacy_watermarks": watermarks(reverse=True),
        "findings": (),
    }
    values.update(overrides)
    return PaperAccountReadinessAssessment(**values)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("execution_scope", object(), TypeError),
        ("execution_scope", "", ValueError),
        ("execution_scope", " paper:test", ValueError),
        ("execution_scope", "x" * 129, ValueError),
        ("execution_scope", "bad\x00scope", ValueError),
        ("execution_scope", "bad\ud800scope", ValueError),
        ("account_key", object(), TypeError),
        ("account_key", "", ValueError),
        ("account_key", "account-1 ", ValueError),
        ("account_key", "x" * 256, ValueError),
        ("account_key", "bad\x00key", ValueError),
        ("owner_generation", True, TypeError),
        ("owner_generation", 1.0, TypeError),
        ("owner_generation", 0, ValueError),
        ("owner_generation", BIGINT_MAX + 1, ValueError),
        ("opening_payload_sha256", object(), TypeError),
        ("opening_payload_sha256", "A" * 64, ValueError),
        ("opening_payload_sha256", "a" * 63, ValueError),
    ),
)
def test_context_rejects_invalid_or_unrepresentable_provenance(
    field, value, error
) -> None:
    with pytest.raises(error):
        context(**{field: value})


def test_context_accepts_exact_storage_boundaries() -> None:
    value = context(
        execution_scope="s" * 128,
        account_key="a" * 255,
        owner_generation=BIGINT_MAX,
        opening_payload_sha256="f" * 64,
    )

    assert value.owner_generation == BIGINT_MAX


@pytest.mark.parametrize(
    ("values", "error"),
    (
        ({"version": True}, TypeError),
        ({"version": 1.0}, TypeError),
        ({"version": 0}, ValueError),
        ({"version": INTEGER_MAX + 1}, ValueError),
        ({"name": object()}, TypeError),
        ({"name": ""}, ValueError),
        ({"name": "Uppercase"}, ValueError),
        ({"name": "bad-name"}, ValueError),
        ({"name": "1_starts_with_digit"}, ValueError),
        ({"checksum": object()}, TypeError),
        ({"checksum": "F" * 64}, ValueError),
        ({"checksum": "f" * 65}, ValueError),
    ),
)
def test_migration_identity_rejects_invalid_storage_values(values, error) -> None:
    arguments = {"version": 1, "name": "legacy_baseline", "checksum": SHA_A}
    arguments.update(values)
    with pytest.raises(error):
        MigrationIdentity(**arguments)


def test_migration_identity_accepts_postgres_integer_maximum() -> None:
    assert MigrationIdentity(INTEGER_MAX, "migration_1", SHA_A).version == INTEGER_MAX


@pytest.mark.parametrize(
    ("values", "error"),
    (
        ({"relation": object()}, TypeError),
        ({"relation": "np.unknown"}, ValueError),
        ({"row_count": True}, TypeError),
        ({"row_count": -1}, ValueError),
        ({"row_count": BIGINT_MAX + 1}, ValueError),
        ({"row_count": 1, "max_id": None}, ValueError),
        ({"row_count": 0, "max_id": 1}, ValueError),
        ({"row_count": 1, "max_id": True}, TypeError),
        ({"row_count": 1, "max_id": INTEGER_MIN - 1}, ValueError),
        ({"row_count": 1, "max_id": INTEGER_MAX + 1}, ValueError),
    ),
)
def test_legacy_watermark_rejects_incoherent_inventory(values, error) -> None:
    arguments = {
        "relation": "np.trades",
        "row_count": 0,
        "max_id": None,
    }
    arguments.update(values)
    with pytest.raises(error):
        LegacyRelationWatermark(**arguments)


@pytest.mark.parametrize("max_id", (INTEGER_MIN, INTEGER_MAX))
def test_legacy_watermark_accepts_postgres_integer_boundaries(max_id) -> None:
    value = LegacyRelationWatermark("np.trades", BIGINT_MAX, max_id)
    assert value.row_count == BIGINT_MAX
    assert value.max_id == max_id


@pytest.mark.parametrize(
    ("values", "error"),
    (
        ({"kind": "ACCOUNT_INSOLVENT"}, TypeError),
        ({"subject_kind": object()}, TypeError),
        ({"subject_kind": ""}, ValueError),
        ({"subject_kind": "x" * 65}, ValueError),
        ({"subject_id": object()}, TypeError),
        ({"subject_id": " id"}, ValueError),
        ({"subject_id": "x" * 256}, ValueError),
        ({"subject_id": "bad\x00id"}, ValueError),
    ),
)
def test_finding_requires_typed_kind_and_bounded_subject(values, error) -> None:
    arguments = {
        "kind": PaperAccountReadinessFindingKind.ACCOUNT_INSOLVENT,
        "subject_kind": "paper_account",
        "subject_id": "account-1",
    }
    arguments.update(values)
    with pytest.raises(error):
        PaperAccountReadinessFinding(**arguments)


@pytest.mark.parametrize(
    "kind",
    (
        PaperAccountReadinessFindingKind.ACCOUNT_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.POSITION_REPLAY_FAILED,
        PaperAccountReadinessFindingKind.UNRESOLVED_SUBMISSION,
        PaperAccountReadinessFindingKind.UNACCOUNTED_ORDER,
    ),
)
def test_replay_or_ownership_findings_require_reconciliation(kind) -> None:
    result = assessment(findings=(finding(kind),))
    assert (
        result.disposition is PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
    )


@pytest.mark.parametrize(
    "kind",
    (
        PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED,
        PaperAccountReadinessFindingKind.ACCOUNT_PROVENANCE_MISMATCH,
        PaperAccountReadinessFindingKind.UNEXPECTED_ACCOUNT,
        PaperAccountReadinessFindingKind.ACCOUNT_INSOLVENT,
        PaperAccountReadinessFindingKind.MARGIN_RESERVATION_PRESENT,
        PaperAccountReadinessFindingKind.DURABLE_OPEN_POSITION,
        PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION,
        PaperAccountReadinessFindingKind.RUNTIME_CONTROL_NOT_LEGACY,
    ),
)
def test_non_reconciliation_findings_are_blocking(kind) -> None:
    overrides = {"findings": (finding(kind),)}
    if kind is PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED:
        overrides["account_version"] = None
    elif kind is PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION:
        overrides = {"legacy_watermarks": watermarks(open_positions=1)}
    result = assessment(**overrides)
    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED


def test_reconciliation_takes_priority_over_blocking_findings() -> None:
    result = assessment(
        findings=(
            finding(PaperAccountReadinessFindingKind.ACCOUNT_INSOLVENT),
            finding(
                PaperAccountReadinessFindingKind.UNRESOLVED_SUBMISSION,
                "order-1",
                subject_kind="client_order",
            ),
        )
    )

    assert (
        result.disposition is PaperAccountReadinessDisposition.RECONCILIATION_REQUIRED
    )


def test_empty_exact_evidence_is_only_prepared_for_a_separate_fence() -> None:
    result = assessment()

    assert result.disposition is PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
    assert result.snapshot_authoritative is False


@pytest.mark.parametrize(
    ("applied", "kind"),
    (
        ((), PaperAccountReadinessFindingKind.MIGRATION_LEDGER_ABSENT),
        (
            migrations()[:2],
            PaperAccountReadinessFindingKind.MIGRATION_PENDING,
        ),
        (
            (
                MigrationIdentity(1, "legacy_baseline", SHA_A),
                MigrationIdentity(2, "order_position_journal", "d" * 64),
                MigrationIdentity(3, "paper_account_ledger", "c" * 64),
            ),
            PaperAccountReadinessFindingKind.MIGRATION_DRIFT,
        ),
    ),
)
def test_migration_evidence_derives_one_canonical_blocker(applied, kind) -> None:
    result = assessment(applied_migrations=applied, legacy_watermarks=())

    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert result.findings == (
        PaperAccountReadinessFinding(
            kind,
            "migration_ledger",
            "np.schema_migrations",
        ),
    )


def test_matching_supplied_migration_finding_is_replaced_by_canonical_subject() -> None:
    result = assessment(
        applied_migrations=(),
        findings=(
            finding(
                PaperAccountReadinessFindingKind.MIGRATION_LEDGER_ABSENT,
                "caller-selected-subject",
                subject_kind="caller-selected-kind",
            ),
        ),
    )

    assert result.findings == (
        PaperAccountReadinessFinding(
            PaperAccountReadinessFindingKind.MIGRATION_LEDGER_ABSENT,
            "migration_ledger",
            "np.schema_migrations",
        ),
    )


def test_explicit_raw_migration_drift_overrides_a_decodable_prefix() -> None:
    result = assessment(
        applied_migrations=(),
        findings=(
            finding(
                PaperAccountReadinessFindingKind.MIGRATION_DRIFT,
                "raw-malformed-row",
                subject_kind="raw_migration_row",
            ),
        ),
    )

    assert result.findings == (
        PaperAccountReadinessFinding(
            PaperAccountReadinessFindingKind.MIGRATION_DRIFT,
            "migration_ledger",
            "np.schema_migrations",
        ),
    )


def test_explicit_raw_migration_drift_survives_complete_decodable_prefix() -> None:
    result = assessment(
        findings=(
            finding(
                PaperAccountReadinessFindingKind.MIGRATION_DRIFT,
                "malformed-trailing-row",
                subject_kind="raw_migration_row",
            ),
        ),
    )

    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert result.findings == (
        PaperAccountReadinessFinding(
            PaperAccountReadinessFindingKind.MIGRATION_DRIFT,
            "migration_ledger",
            "np.schema_migrations",
        ),
    )


def test_conflicting_supplied_migration_finding_is_rejected() -> None:

    with pytest.raises(ValueError, match="migration"):
        assessment(
            findings=(
                finding(
                    PaperAccountReadinessFindingKind.MIGRATION_PENDING,
                    "np.schema_migrations",
                    subject_kind="migration_ledger",
                ),
            )
        )


def test_legacy_open_position_watermark_derives_one_canonical_blocker() -> None:
    result = assessment(legacy_watermarks=watermarks(open_positions=2, reverse=True))

    assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    assert result.findings == (
        PaperAccountReadinessFinding(
            PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION,
            "legacy_relation",
            "np.open_positions",
        ),
    )


def test_legacy_open_finding_cannot_contradict_an_empty_watermark() -> None:
    with pytest.raises(ValueError, match="legacy open-position"):
        assessment(
            findings=(
                finding(
                    PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION,
                    "np.open_positions",
                    subject_kind="legacy_relation",
                ),
            )
        )


def test_assessment_canonicalizes_watermarks_and_unique_findings() -> None:
    unresolved = finding(
        PaperAccountReadinessFindingKind.UNRESOLVED_SUBMISSION,
        "order-2",
        subject_kind="client_order",
    )
    insolvent = finding(PaperAccountReadinessFindingKind.ACCOUNT_INSOLVENT)
    result = assessment(
        legacy_watermarks=watermarks(reverse=True),
        findings=(unresolved, insolvent, unresolved),
    )

    assert tuple(value.relation for value in result.legacy_watermarks) == (
        LEGACY_RELATIONS
    )
    assert result.findings == tuple(
        sorted(
            {unresolved, insolvent},
            key=lambda value: (
                value.kind.value,
                value.subject_kind,
                value.subject_id,
            ),
        )
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("context", object(), TypeError),
        ("expected_migrations", [], TypeError),
        ("expected_migrations", (), ValueError),
        ("expected_migrations", (object(),), TypeError),
        ("applied_migrations", [], TypeError),
        ("applied_migrations", (object(),), TypeError),
        ("account_version", True, TypeError),
        ("account_version", -1, ValueError),
        ("account_version", BIGINT_MAX + 1, ValueError),
        ("legacy_watermarks", [], TypeError),
        ("legacy_watermarks", (object(),), TypeError),
        ("findings", [], TypeError),
        ("findings", (object(),), TypeError),
    ),
)
def test_assessment_rejects_invalid_top_level_evidence(field, value, error) -> None:
    with pytest.raises(error):
        assessment(**{field: value})


def test_migrations_must_be_unique_and_strictly_increasing() -> None:
    first, second, third = migrations()
    for field in ("expected_migrations", "applied_migrations"):
        with pytest.raises(ValueError, match="unique increasing exact contiguous"):
            assessment(**{field: (second, first, third)})
        with pytest.raises(ValueError, match="unique increasing exact contiguous"):
            assessment(**{field: (first, first)})


def test_watermarks_must_cover_each_legacy_relation_exactly_once() -> None:
    complete = watermarks()
    with pytest.raises(ValueError, match="every legacy relation once"):
        assessment(legacy_watermarks=complete[:-1])
    with pytest.raises(ValueError, match="repeat a relation"):
        assessment(legacy_watermarks=complete[:-1] + (complete[0],))


def test_incomplete_migration_evidence_still_rejects_duplicate_watermarks() -> None:
    duplicate = (
        LegacyRelationWatermark("np.open_positions", 0, None),
        LegacyRelationWatermark("np.open_positions", 1, 4),
    )

    with pytest.raises(ValueError, match="repeat a relation"):
        assessment(
            applied_migrations=(),
            account_version=None,
            legacy_watermarks=duplicate,
        )


@pytest.mark.parametrize(
    "account_version",
    (None, True, -1, BIGINT_MAX + 1),
)
def test_missing_or_invalid_account_version_requires_coherent_evidence(
    account_version,
) -> None:
    if account_version is None:
        with pytest.raises(ValueError, match="account finding"):
            assessment(account_version=None)
        result = assessment(
            account_version=None,
            findings=(
                finding(PaperAccountReadinessFindingKind.ACCOUNT_NOT_PROVISIONED),
            ),
        )
        assert result.disposition is PaperAccountReadinessDisposition.BLOCKED
    else:
        with pytest.raises((TypeError, ValueError)):
            assessment(account_version=account_version)


def test_all_contract_values_are_frozen_slotted_copyable_and_pickleable() -> None:
    values = (
        context(),
        migrations()[0],
        watermarks()[0],
        finding(PaperAccountReadinessFindingKind.ACCOUNT_INSOLVENT),
        assessment(),
    )

    for value in values:
        assert not hasattr(value, "__dict__")
        assert copy.copy(value) == value
        assert copy.deepcopy(value) == value
        assert hash(value) == hash(copy.copy(value))
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, next(iter(value.__dataclass_fields__)), None)
        with pytest.raises(TypeError):
            value.__setstate__((None,))
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            assert pickle.loads(pickle.dumps(value, protocol=protocol)) == value


def test_replace_rederives_but_never_makes_assessment_authoritative() -> None:
    blocked = assessment(
        findings=(finding(PaperAccountReadinessFindingKind.ACCOUNT_INSOLVENT),)
    )

    prepared = replace(blocked, findings=())

    assert prepared.disposition is PaperAccountReadinessDisposition.PREPARED_FOR_FENCE
    assert prepared.snapshot_authoritative is False


def test_disposition_policy_collections_are_immutable() -> None:
    from trading.application import paper_account_readiness as module

    assert isinstance(module._RECONCILIATION_FINDINGS, frozenset)
    assert isinstance(module._MIGRATION_FINDINGS, frozenset)


def test_readiness_port_is_positional_only() -> None:
    parameters = inspect.signature(PaperAccountReadinessPort.assess).parameters
    assert tuple(parameters) == ("self", "context")
    assert parameters["context"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert parameters["context"].annotation in {
        PaperAccountReadinessContext,
        "PaperAccountReadinessContext",
    }
    assert inspect.signature(PaperAccountReadinessPort.assess).return_annotation in {
        PaperAccountReadinessAssessment,
        "PaperAccountReadinessAssessment",
    }


def test_readiness_contract_is_exported_by_application_facade() -> None:
    import trading.application as application

    assert PUBLIC_EXPORTS <= set(application.__all__)
    assert all(getattr(application, name) is globals()[name] for name in PUBLIC_EXPORTS)


def _literal_import_target(
    node: ast.Call,
    *,
    builtins_aliases: set[str],
    builtin_import_aliases: set[str],
    importlib_aliases: set[str],
    import_module_aliases: set[str],
) -> str | None:
    target = (
        node.args[0].value
        if node.args and isinstance(node.args[0], ast.Constant)
        else next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
            ),
            None,
        )
    )
    if not isinstance(target, str):
        return None
    built_in = (
        isinstance(node.func, ast.Name) and node.func.id in builtin_import_aliases
    ) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "__import__"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in builtins_aliases
    )
    import_module = (
        isinstance(node.func, ast.Name) and node.func.id in import_module_aliases
    ) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in importlib_aliases
    )
    if not built_in and not import_module:
        return None
    if built_in and target.startswith("trading."):
        return "trading"
    if not target.startswith("."):
        return target
    package = (
        node.args[1].value
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
        else next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "package" and isinstance(keyword.value, ast.Constant)
            ),
            None,
        )
    )
    if not isinstance(package, str):
        return None
    try:
        return importlib.util.resolve_name(target, package)
    except (ImportError, ValueError):
        return None


def _uses_paper_account_readiness(source: str) -> bool:
    """Detect direct, facade, relative, aliased, and literal-dynamic use."""
    tree = ast.parse(source)
    module = "trading.application.paper_account_readiness"
    builtins_aliases = {"builtins"}
    builtin_import_aliases = {"__import__"}
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "trading":
                    return True
                if alias.name == module or alias.name.startswith(f"{module}."):
                    return True
                if alias.name == "trading.application":
                    return True
                if alias.name.startswith("trading.application.") and not alias.asname:
                    return True
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
                if alias.name == "builtins":
                    builtins_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            imported_module = node.module or ""
            if imported_module == "importlib" and "import_module" in imported:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if imported_module == "builtins" and "__import__" in imported:
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )
            if imported_module == module:
                return True
            if node.level and imported_module.endswith("paper_account_readiness"):
                return True
            if node.level and "paper_account_readiness" in imported:
                return True
            if node.level and imported & PUBLIC_EXPORTS:
                return True
            if imported_module == "trading" and "application" in imported:
                return True
            if imported_module == "trading.application" and imported & (
                PUBLIC_EXPORTS | {"paper_account_readiness", "*"}
            ):
                return True

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            built_in = (
                isinstance(value, ast.Name) and value.id in builtin_import_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "__import__"
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            )
            import_module = (
                isinstance(value, ast.Name) and value.id in import_module_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "import_module"
                and isinstance(value.value, ast.Name)
                and value.value.id in importlib_aliases
            )
            if not built_in and not import_module:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                aliases = builtin_import_aliases if built_in else import_module_aliases
                if target.id not in aliases:
                    aliases.add(target.id)
                    changed = True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _literal_import_target(
            node,
            builtins_aliases=builtins_aliases,
            builtin_import_aliases=builtin_import_aliases,
            importlib_aliases=importlib_aliases,
            import_module_aliases=import_module_aliases,
        )
        if target == module or (target and target.startswith(f"{module}.")):
            return True
        if target in {"trading", "trading.application"}:
            return True
    return False


@pytest.mark.parametrize(
    "source",
    (
        "from trading.application.paper_account_readiness "
        "import PaperAccountReadinessAssessment",
        "import trading.application.paper_account_readiness as readiness",
        "from trading.application import PaperAccountReadinessContext",
        "from trading.application import paper_account_readiness",
        "import trading.application as app\napp.PaperAccountReadinessPort",
        "import trading as root\nroot.application.PaperAccountReadinessAssessment",
        "from trading import application as app\napp.PaperAccountReadinessContext",
        "from .paper_account_readiness import PaperAccountReadinessAssessment",
        "from . import paper_account_readiness",
        "from ..application import PaperAccountReadinessPort",
        "from importlib import import_module as load\n"
        "load('trading.application.paper_account_readiness')",
        "import importlib as loader\n"
        "loader.import_module('trading.application').PaperAccountReadinessPort",
        "root = __import__('trading')\n"
        "root.application.PaperAccountReadinessAssessment",
        "from builtins import __import__ as load\n"
        "load('trading.application.paper_account_readiness')",
        "load = __import__\nload('trading.application.paper_account_readiness')",
        "import importlib\nload = importlib.import_module\n"
        "load('trading.application.paper_account_readiness')",
        "from importlib import import_module\n"
        "import_module('.paper_account_readiness', package='trading.application')",
    ),
)
def test_readiness_consumer_detector_catches_supported_forms(source) -> None:
    assert _uses_paper_account_readiness(source)


@pytest.mark.parametrize(
    "source",
    (
        "from trading.application.order_service import OrderService",
        "from trading.application import OrderService",
        "from trading.domain.paper_accounting import PaperAccount",
        "name = 'trading.application.paper_account_readiness'",
    ),
)
def test_readiness_consumer_detector_allows_unrelated_forms(source) -> None:
    assert not _uses_paper_account_readiness(source)


def test_readiness_contract_is_pure_and_has_no_runtime_consumer() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "application" / "paper_account_readiness.py"
    facade_path = root / "trading" / "application" / "__init__.py"
    consumers = []
    scanned = []
    for source_path in sorted(root.rglob("*.py")):
        if (
            source_path in {module_path, facade_path}
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        scanned.append(source_path.relative_to(root))
        if _uses_paper_account_readiness(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))

    assert consumers == [
        Path("trading/application/paper_runtime_activation.py"),
        Path("trading/persistence/paper_account_readiness.py"),
        Path("trading/persistence/paper_runtime_activation.py"),
    ]
    assert {Path("main.py"), Path("core/bootstrap.py")} <= set(scanned)

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            if node.module:
                imported_roots.add(node.module.split(".")[0])
                if node.module.startswith("trading."):
                    assert node.module == "trading.domain._validation"

    assert imported_roots <= set(sys.stdlib_module_names) | {"trading"}
