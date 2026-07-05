"""
Security regression tests for utils/api_connection_tester.py Vault handling.

Ensures the hardcoded 'trading-bot-token' literal is gone and that a missing
VAULT_TOKEN is handled gracefully (warn + skip) instead of injecting a default.
"""

import importlib
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "utils" / "api_connection_tester.py"
)


def test_no_hardcoded_vault_token_literal():
    """The 'trading-bot-token' literal must not appear in the source."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "trading-bot-token" not in source


def test_module_imports():
    """The module must import cleanly in a minimal environment."""
    module = importlib.import_module("utils.api_connection_tester")
    assert hasattr(module, "APIConnectionTester")


def test_vault_skipped_when_token_unset(monkeypatch):
    """With VAULT_TOKEN unset, test_vault warns and reports DISCONNECTED."""
    from utils.api_connection_tester import APIConnectionTester, ConnectionStatus

    monkeypatch.delenv("VAULT_TOKEN", raising=False)

    tester = APIConnectionTester()
    status = tester.test_vault()

    assert status.status == ConnectionStatus.DISCONNECTED
    assert status.error_message == "VAULT_TOKEN not set"
    # The env var must not have been injected with a default value.
    import os

    assert os.getenv("VAULT_TOKEN") is None


def test_vault_uses_env_token_when_set(monkeypatch):
    """When VAULT_TOKEN is set, it is passed through to the hvac client."""
    hvac = pytest.importorskip("hvac")

    monkeypatch.setenv("VAULT_TOKEN", "env-provided-token")

    captured = {}

    class _FakeClient:
        def __init__(self, url=None, token=None):
            captured["url"] = url
            captured["token"] = token

        def is_authenticated(self):
            return False

    monkeypatch.setattr(hvac, "Client", _FakeClient)

    from utils.api_connection_tester import APIConnectionTester

    tester = APIConnectionTester()
    tester.test_vault()

    assert captured["token"] == "env-provided-token"
