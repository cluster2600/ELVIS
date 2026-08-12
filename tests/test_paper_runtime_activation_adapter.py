"""Fast gates for the unwired PostgreSQL paper-runtime activation adapter."""

import ast
import inspect
from pathlib import Path

import pytest

from tests.test_paper_runtime_activation import _context
from trading.persistence.paper_runtime_activation import (
    _ACQUIRE_ACTIVATION_FENCE_SQL,
    _ACTIVATE_RUNTIME_GENERATION_SQL,
    _ACTIVATION_TRANSACTION_SQL,
    _CHECK_CONSTRAINTS_SQL,
    _SELECT_ACTIVATION_ID_SQL,
    _SELECT_RUNTIME_CONTROL_SQL,
    _SET_LOCK_TIMEOUT_SQL,
    PaperRuntimeActivationStorageError,
    PostgresPaperRuntimeActivation,
)


def test_constructor_requires_a_callable_connection_factory() -> None:
    for factory in (None, object(), 1):
        with pytest.raises(TypeError, match="connection_factory"):
            PostgresPaperRuntimeActivation(factory)


def test_activate_is_positional_only_and_rejects_input_before_connection_io() -> None:
    parameters = inspect.signature(PostgresPaperRuntimeActivation.activate).parameters
    assert tuple(parameters) == ("self", "context")
    assert parameters["context"].kind is inspect.Parameter.POSITIONAL_ONLY

    calls = []

    def connect():
        calls.append("connect")
        raise AssertionError("invalid input must not open a connection")

    with pytest.raises(TypeError, match="PaperRuntimeActivationContext"):
        PostgresPaperRuntimeActivation(connect).activate(None)
    assert calls == []


def test_connection_failure_uses_activation_storage_boundary() -> None:
    def fail():
        raise RuntimeError("connect failed")

    with pytest.raises(PaperRuntimeActivationStorageError) as failure:
        PostgresPaperRuntimeActivation(fail).activate(_context())

    assert isinstance(failure.value.__cause__, Exception)


def test_activation_sql_delegates_fence_and_mutation_to_exact_capabilities() -> None:
    assert (
        _ACTIVATION_TRANSACTION_SQL == "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
    )
    assert _SET_LOCK_TIMEOUT_SQL == "SET LOCAL lock_timeout = '1s'"
    assert _ACQUIRE_ACTIVATION_FENCE_SQL == (
        "SELECT np.acquire_paper_runtime_activation_fence()"
    )
    assert " ".join(_SELECT_RUNTIME_CONTROL_SQL.split()) == (
        "SELECT mode, runtime_generation FROM np.paper_runtime_control "
        "WHERE control_key = TRUE"
    )
    assert " ".join(_SELECT_ACTIVATION_ID_SQL.split()) == (
        "SELECT runtime_generation, activation_id, execution_scope, account_key, "
        "owner_generation, opening_version, opening_payload_sha256 FROM "
        "np.paper_runtime_generations WHERE activation_id = %s"
    )
    assert " ".join(_ACTIVATE_RUNTIME_GENERATION_SQL.split()) == (
        "SELECT mode, runtime_generation FROM "
        "np.activate_paper_runtime_generation(%s, %s, %s, %s, %s, %s, %s, %s)"
    )
    assert _CHECK_CONSTRAINTS_SQL == "SET CONSTRAINTS ALL IMMEDIATE"


_ADAPTER_MODULE = "trading.persistence.paper_runtime_activation"
_ADAPTER_EXPORTS = {
    "PaperRuntimeActivationStorageError",
    "PostgresPaperRuntimeActivation",
}


def _uses_activation_adapter(source: str) -> bool:
    """Detect direct, facade, relative, aliased, and literal dynamic use."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"trading", "trading.persistence", _ADAPTER_MODULE}:
                    return True
                if alias.name.startswith(f"{_ADAPTER_MODULE}."):
                    return True
                if (
                    alias.name.startswith("trading.persistence.")
                    and alias.asname is None
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            module = node.module or ""
            if module == _ADAPTER_MODULE:
                return True
            if node.level and (
                module.endswith("paper_runtime_activation")
                or "paper_runtime_activation" in imported
                or bool(imported & _ADAPTER_EXPORTS)
            ):
                return True
            if module == "trading" and "persistence" in imported:
                return True
            if module == "trading.persistence" and imported & (
                _ADAPTER_EXPORTS | {"paper_runtime_activation", "*"}
            ):
                return True
        elif isinstance(node, ast.Constant) and node.value in {
            _ADAPTER_MODULE,
            "trading.persistence",
        }:
            return True
    return False


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence.paper_runtime_activation "
        "import PostgresPaperRuntimeActivation",
        "import trading.persistence.paper_runtime_activation as activation",
        "from trading.persistence import PostgresPaperRuntimeActivation",
        "from .paper_runtime_activation import PaperRuntimeActivationStorageError",
        "from importlib import import_module as load\n"
        "load('trading.persistence.paper_runtime_activation')",
        "load = __import__\nload('trading.persistence.paper_runtime_activation')",
    ),
)
def test_adapter_consumer_detector_catches_supported_forms(source) -> None:
    assert _uses_activation_adapter(source)


def test_activation_adapter_is_unwired_and_not_facade_exported() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "persistence" / "paper_runtime_activation.py"
    facade_path = root / "trading" / "persistence" / "__init__.py"
    consumers = []
    for source_path in root.rglob("*.py"):
        if (
            source_path == module_path
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        if _uses_activation_adapter(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))

    assert consumers == []
    assert not _uses_activation_adapter(facade_path.read_text(encoding="utf-8"))

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert imports <= {
        "collections.abc",
        "psycopg2",
        "trading.application.paper_account_readiness",
        "trading.application.paper_runtime_activation",
        "trading.persistence.order_position_journal",
        "trading.persistence.paper_account_readiness",
    }
