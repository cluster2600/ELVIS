"""Static trust-boundary checks for durable fresh-opening provisioning."""

import ast
from pathlib import Path

MODULE = (
    Path(__file__).parents[1]
    / "trading"
    / "persistence"
    / "postgres_fresh_opening_provisioning.py"
)


def test_adapter_calls_only_the_three_bounded_database_capabilities():
    source = MODULE.read_text(encoding="utf-8")

    assert "np.acquire_paper_fresh_opening_fence" in source
    assert "np.commit_paper_fresh_opening" in source
    assert "np.read_paper_fresh_opening" in source
    for forbidden in (
        "INSERT INTO",
        "UPDATE np.",
        "DELETE FROM",
        "CREATE ROLE",
        "GRANT ",
        "ALTER ROLE",
    ):
        assert forbidden not in source


def test_adapter_has_no_ambient_configuration_or_secret_output_path():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert imported_roots.isdisjoint({"logging", "os", "subprocess"})
    assert called_names.isdisjoint({"print", "open", "input"})


def test_adapter_keeps_bootstrap_activation_and_runtime_unwired():
    source = MODULE.read_text(encoding="utf-8")

    for forbidden in (
        "PostgresBootstrap",
        "activate_paper_runtime_generation",
        "main.py",
        "docker",
        "ansible",
        "runtime_activation_authorized=True",
        "trading_authorized=True",
    ):
        assert forbidden not in source
