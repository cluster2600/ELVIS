"""Fast gates for the unwired atomic PostgreSQL paper-account owner."""

import ast
import inspect
from pathlib import Path

import pytest

from tests.test_paper_account_submission import _context
from trading.persistence.atomic_paper_account_owner import (
    PostgresAtomicPaperAccountOwner,
)
from trading.persistence.paper_account_journal import PaperAccountStorageError


class _Planner:
    def plan(self, attempt, /):
        raise AssertionError("the constructor must not execute the planner")


@pytest.mark.parametrize("planner", (None, object(), lambda: None))
def test_constructor_requires_a_planner_protocol(planner) -> None:
    with pytest.raises(TypeError, match="planner"):
        PostgresAtomicPaperAccountOwner(lambda: None, planner)


def test_execute_contract_is_positional_only() -> None:
    parameters = inspect.signature(PostgresAtomicPaperAccountOwner.execute).parameters
    assert tuple(parameters) == ("self", "context")
    assert parameters["context"].kind is inspect.Parameter.POSITIONAL_ONLY


def test_connection_failure_uses_the_account_storage_boundary() -> None:
    def fail():
        raise RuntimeError("connect failed")

    with pytest.raises(PaperAccountStorageError) as failure:
        PostgresAtomicPaperAccountOwner(fail, _Planner()).execute(_context())

    assert isinstance(failure.value.__cause__, Exception)


_OWNER_MODULE = "trading.persistence.atomic_paper_account_owner"
_OWNER_EXPORTS = {"PostgresAtomicPaperAccountOwner"}


def _uses_atomic_account_owner(source: str) -> bool:
    """Conservatively detect static and literal-dynamic owner consumers."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"trading", "trading.persistence", _OWNER_MODULE}:
                    return True
                if alias.name.startswith(f"{_OWNER_MODULE}."):
                    return True
                if (
                    alias.name.startswith("trading.persistence.")
                    and alias.asname is None
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            module = node.module or ""
            if module == _OWNER_MODULE:
                return True
            if node.level and (
                module.endswith("atomic_paper_account_owner")
                or "atomic_paper_account_owner" in imported
                or imported & _OWNER_EXPORTS
            ):
                return True
            if module == "trading" and "persistence" in imported:
                return True
            if module == "trading.persistence" and imported & (
                _OWNER_EXPORTS | {"atomic_paper_account_owner", "*"}
            ):
                return True
        elif isinstance(node, ast.Constant) and node.value in {
            _OWNER_MODULE,
            "trading.persistence",
        }:
            return True
    return False


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence.atomic_paper_account_owner "
        "import PostgresAtomicPaperAccountOwner",
        "import trading.persistence.atomic_paper_account_owner as owner",
        "from trading.persistence import atomic_paper_account_owner",
        "from .atomic_paper_account_owner import PostgresAtomicPaperAccountOwner",
        "from importlib import import_module as load\n"
        "load('trading.persistence.atomic_paper_account_owner')",
        "load = __import__\n" "load('trading.persistence.atomic_paper_account_owner')",
    ),
)
def test_owner_consumer_detector_catches_supported_forms(source) -> None:
    assert _uses_atomic_account_owner(source)


def test_atomic_account_owner_is_unwired_and_imports_only_owned_boundaries() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "persistence" / "atomic_paper_account_owner.py"
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
        if _uses_atomic_account_owner(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))
    assert consumers == []

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    allowed = {
        "dataclasses",
        "psycopg2",
        "typing",
        "trading.application.durable_submission",
        "trading.domain.order_lifecycle",
        "trading.domain.paper_accounting",
        "trading.domain.paper_economics",
        "trading.domain.paper_settlement",
        "trading.domain.positions",
        "trading.persistence.journal_codec",
        "trading.persistence.atomic_paper_submission_owner",
        "trading.persistence.order_position_journal",
        "trading.persistence.paper_account_journal",
        "trading.persistence.paper_account_journal_codec",
    }
    assert imports <= allowed

    source = module_path.read_text(encoding="utf-8")
    assert ".reserve_instruction(" not in source
    assert ".append_event(" not in source
    assert ".provision_account(" not in source
    assert ".replay_account(" not in source
    assert "PostgresAtomicPaperSubmissionOwner" not in source
    assert "trading.execution" not in source
