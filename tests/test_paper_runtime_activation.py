"""Pure contract tests for the dormant paper-runtime activation boundary."""

import ast
import copy
import inspect
import pickle
import traceback
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trading.application import (
    PaperRuntimeActivationBlocked,
    PaperRuntimeActivationBusy,
    PaperRuntimeActivationCommitUnknown,
    PaperRuntimeActivationConflict,
    PaperRuntimeActivationContext,
    PaperRuntimeActivationDisposition,
    PaperRuntimeActivationPort,
    PaperRuntimeActivationReceipt,
    PaperRuntimeActivationResult,
    PaperRuntimeActivationSource,
)
from trading.application.paper_account_readiness import (
    LegacyRelationWatermark,
    MigrationIdentity,
    PaperAccountReadinessAssessment,
    PaperAccountReadinessContext,
    PaperAccountReadinessFinding,
    PaperAccountReadinessFindingKind,
)

LEGACY_RELATIONS = (
    "np.account_balances",
    "np.liquidations",
    "np.margin_history",
    "np.model_predictions",
    "np.open_positions",
    "np.trades",
    "np.trading_session_resets",
)


def _readiness() -> PaperAccountReadinessContext:
    return PaperAccountReadinessContext(
        "paper:test",
        "paper-main",
        7,
        "a" * 64,
    )


def _context(
    *,
    readiness: PaperAccountReadinessContext | None = None,
    activation_id: str = "activate-paper-1",
    source: PaperRuntimeActivationSource = PaperRuntimeActivationSource.LEGACY,
    expected_runtime_generation: int = 0,
) -> PaperRuntimeActivationContext:
    return PaperRuntimeActivationContext(
        readiness or _readiness(),
        activation_id,
        source,
        expected_runtime_generation,
    )


def _assessment(
    readiness: PaperAccountReadinessContext,
    *,
    blocked: bool,
) -> PaperAccountReadinessAssessment:
    migration = MigrationIdentity(1, "paper_runtime", "b" * 64)
    findings = (
        (
            PaperAccountReadinessFinding(
                PaperAccountReadinessFindingKind.LEGACY_OPEN_POSITION,
                "legacy_relation",
                "np.open_positions",
            ),
        )
        if blocked
        else ()
    )
    watermarks = tuple(
        LegacyRelationWatermark(
            relation,
            1 if blocked and relation == "np.open_positions" else 0,
            1 if blocked and relation == "np.open_positions" else None,
        )
        for relation in LEGACY_RELATIONS
    )
    return PaperAccountReadinessAssessment(
        readiness,
        (migration,),
        (migration,),
        0,
        watermarks,
        findings,
    )


def _blocked() -> PaperRuntimeActivationBlocked:
    context = _context()
    return PaperRuntimeActivationBlocked(
        context,
        _assessment(context.readiness, blocked=True),
    )


def test_context_derives_exact_next_generation_for_each_supported_source() -> None:
    legacy = _context()
    paused = _context(
        activation_id="resume-paper-42",
        source=PaperRuntimeActivationSource.PAUSED,
        expected_runtime_generation=42,
    )

    assert legacy.target_runtime_generation == 1
    assert paused.target_runtime_generation == 43
    assert legacy.readiness.execution_scope == "paper:test"
    assert not hasattr(legacy, "__dict__")


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"readiness": None}, TypeError),
        ({"activation_id": None}, TypeError),
        ({"activation_id": ""}, ValueError),
        ({"activation_id": " padded "}, ValueError),
        ({"activation_id": "x" * 256}, ValueError),
        ({"activation_id": "nul\x00id"}, ValueError),
        ({"source": "LEGACY"}, TypeError),
        ({"expected_runtime_generation": True}, TypeError),
        ({"expected_runtime_generation": -1}, ValueError),
        ({"expected_runtime_generation": 1 << 63}, ValueError),
        ({"expected_runtime_generation": 1.0}, TypeError),
        (
            {
                "source": PaperRuntimeActivationSource.LEGACY,
                "expected_runtime_generation": 1,
            },
            ValueError,
        ),
        (
            {
                "source": PaperRuntimeActivationSource.PAUSED,
                "expected_runtime_generation": 0,
            },
            ValueError,
        ),
    ),
)
def test_context_rejects_ambiguous_or_unrepresentable_activation_identity(
    changes,
    error,
) -> None:
    values = {
        "readiness": _readiness(),
        "activation_id": "activate-paper-1",
        "source": PaperRuntimeActivationSource.LEGACY,
        "expected_runtime_generation": 0,
    }
    values.update(changes)

    with pytest.raises(error):
        PaperRuntimeActivationContext(**values)


@pytest.mark.parametrize(
    "disposition",
    (
        PaperRuntimeActivationDisposition.ACTIVATED,
        PaperRuntimeActivationDisposition.REPLAYED,
    ),
)
def test_receipt_retains_exact_context_and_target_generation(disposition) -> None:
    context = _context(
        source=PaperRuntimeActivationSource.PAUSED,
        expected_runtime_generation=8,
    )
    receipt = PaperRuntimeActivationReceipt(context, 9, disposition)

    assert receipt.context is context
    assert receipt.runtime_generation == context.target_runtime_generation
    assert receipt.disposition is disposition


@pytest.mark.parametrize(
    ("values", "error"),
    (
        ((None, 1, PaperRuntimeActivationDisposition.ACTIVATED), TypeError),
        ((_context(), True, PaperRuntimeActivationDisposition.ACTIVATED), TypeError),
        ((_context(), 2, PaperRuntimeActivationDisposition.ACTIVATED), ValueError),
        ((_context(), 1, "ACTIVATED"), TypeError),
    ),
)
def test_receipt_rejects_non_exact_context_generation_or_disposition(
    values,
    error,
) -> None:
    with pytest.raises(error):
        PaperRuntimeActivationReceipt(*values)


def test_blocked_requires_authoritative_failure_for_the_exact_context() -> None:
    context = _context()
    blocked = PaperRuntimeActivationBlocked(
        context,
        _assessment(context.readiness, blocked=True),
    )

    assert blocked.context is context
    assert blocked.assessment.context is context.readiness
    assert blocked.assessment.findings

    with pytest.raises(ValueError, match="prepared"):
        PaperRuntimeActivationBlocked(
            context,
            _assessment(context.readiness, blocked=False),
        )
    with pytest.raises(ValueError, match="exact readiness context"):
        PaperRuntimeActivationBlocked(
            context,
            _assessment(_readiness(), blocked=True),
        )


@pytest.mark.parametrize(
    ("error_type", "message"),
    (
        (PaperRuntimeActivationBusy, "paper runtime activation is busy"),
        (PaperRuntimeActivationConflict, "paper runtime activation conflicts"),
        (
            PaperRuntimeActivationCommitUnknown,
            "paper runtime activation commit is unknown",
        ),
    ),
)
def test_typed_failures_are_frozen_pickle_safe_and_preserve_full_context(
    error_type,
    message,
) -> None:
    context = _context(
        activation_id="resume-paper-8",
        source=PaperRuntimeActivationSource.PAUSED,
        expected_runtime_generation=8,
    )
    error = error_type(context)

    assert str(error) == message
    assert error.context is context
    assert error.activation_id == context.activation_id
    assert error.requires_reconciliation is (
        error_type is not PaperRuntimeActivationBusy
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        error.context = _context()
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(error, protocol=protocol))
        assert type(restored) is error_type
        assert restored.context == context


@pytest.mark.parametrize(
    "error_type",
    (
        PaperRuntimeActivationBusy,
        PaperRuntimeActivationConflict,
        PaperRuntimeActivationCommitUnknown,
    ),
)
def test_typed_failures_allow_only_base_exception_runtime_state(error_type) -> None:
    context = _context()
    error = error_type(context)

    try:
        raise error
    except error_type as raised:
        assert raised is error
        assert raised.__traceback__ is not None
        rendered = "".join(traceback.format_exception(raised))
        assert str(raised) in rendered

    @contextmanager
    def propagate():
        yield

    with pytest.raises(error_type) as propagated:
        with propagate():
            raise error_type(context)
    assert propagated.value.context is context
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        propagated.value.unexpected = "mutation"


@pytest.mark.parametrize(
    "value",
    (
        _context(),
        PaperRuntimeActivationReceipt(
            _context(),
            1,
            PaperRuntimeActivationDisposition.ACTIVATED,
        ),
        _blocked(),
    ),
)
def test_contracts_are_frozen_and_copy_protocols_cannot_alias_mutable_state(
    value,
) -> None:
    assert copy.copy(value) == value
    assert copy.deepcopy(value) == value
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        value.unexpected = "mutation"


def test_activation_port_is_positional_only_and_result_union_is_public() -> None:
    parameters = inspect.signature(PaperRuntimeActivationPort.activate).parameters
    assert tuple(parameters) == ("self", "context")
    assert parameters["context"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert PaperRuntimeActivationResult == (
        PaperRuntimeActivationReceipt | PaperRuntimeActivationBlocked
    )


def test_activation_contracts_are_exported_by_application_facade() -> None:
    import trading.application as application

    expected = {
        "PaperRuntimeActivationBlocked",
        "PaperRuntimeActivationBusy",
        "PaperRuntimeActivationCommitUnknown",
        "PaperRuntimeActivationConflict",
        "PaperRuntimeActivationContext",
        "PaperRuntimeActivationDisposition",
        "PaperRuntimeActivationPort",
        "PaperRuntimeActivationReceipt",
        "PaperRuntimeActivationResult",
        "PaperRuntimeActivationSource",
    }
    assert expected <= set(application.__all__)
    assert all(getattr(application, name) is globals()[name] for name in expected)


_ACTIVATION_MODULE = "trading.application.paper_runtime_activation"
_ACTIVATION_EXPORTS = {
    "PaperRuntimeActivationBlocked",
    "PaperRuntimeActivationBusy",
    "PaperRuntimeActivationCommitUnknown",
    "PaperRuntimeActivationConflict",
    "PaperRuntimeActivationContext",
    "PaperRuntimeActivationDisposition",
    "PaperRuntimeActivationPort",
    "PaperRuntimeActivationReceipt",
    "PaperRuntimeActivationResult",
    "PaperRuntimeActivationSource",
}


def _uses_paper_runtime_activation(source: str) -> bool:
    """Detect direct, facade, relative, aliased, and literal dynamic use."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {
                    "trading",
                    "trading.application",
                    _ACTIVATION_MODULE,
                }:
                    return True
                if alias.name.startswith(f"{_ACTIVATION_MODULE}."):
                    return True
                if (
                    alias.name.startswith("trading.application.")
                    and alias.asname is None
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            module = node.module or ""
            if module == _ACTIVATION_MODULE:
                return True
            if node.level and (
                module.endswith("paper_runtime_activation")
                or "paper_runtime_activation" in imported
                or bool(imported & _ACTIVATION_EXPORTS)
            ):
                return True
            if module == "trading" and "application" in imported:
                return True
            if module == "trading.application" and imported & (
                _ACTIVATION_EXPORTS | {"paper_runtime_activation", "*"}
            ):
                return True
        elif isinstance(node, ast.Constant) and node.value in {
            _ACTIVATION_MODULE,
            "trading.application",
        }:
            return True
    return False


@pytest.mark.parametrize(
    "source",
    (
        "from trading.application.paper_runtime_activation "
        "import PaperRuntimeActivationContext",
        "import trading.application.paper_runtime_activation as activation",
        "from trading.application import PaperRuntimeActivationPort",
        "from .paper_runtime_activation import PaperRuntimeActivationReceipt",
        "from importlib import import_module as load\n"
        "load('trading.application.paper_runtime_activation')",
        "load = __import__\n" "load('trading.application.paper_runtime_activation')",
    ),
)
def test_activation_consumer_detector_catches_supported_forms(source) -> None:
    assert _uses_paper_runtime_activation(source)


def test_activation_contract_has_only_its_dormant_persistence_adapter_consumer() -> (
    None
):
    root = Path(__file__).parents[1]
    contract_path = root / "trading" / "application" / "paper_runtime_activation.py"
    facade_path = root / "trading" / "application" / "__init__.py"
    adapter_path = root / "trading" / "persistence" / "paper_runtime_activation.py"
    consumers = []
    for source_path in root.rglob("*.py"):
        if (
            source_path in {contract_path, facade_path, adapter_path}
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        if _uses_paper_runtime_activation(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))

    assert consumers == []
