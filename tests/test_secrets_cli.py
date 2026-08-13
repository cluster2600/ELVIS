"""Tests for the utils/secrets_manager.py command-line entry point.

The command ``python utils/secrets_manager.py`` runs an interactive secrets
setup. These tests cover the argparse-driven
``main()`` with a mocked EnhancedSecretsManager, plus a real subprocess smoke
test, and assert that:

  * ``--list`` prints secret names grouped by category
  * ``--get`` reports presence ONLY and never prints the secret value
  * ``--set`` reads the value via a hidden getpass prompt (not argv/stdout)

The tests run in a minimal environment: no live Vault/DB/Redis is required.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

secrets_manager = pytest.importorskip("utils.secrets_manager")

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET_VALUE = "super-secret-value-should-never-print"


def _mock_manager(**kwargs):
    manager = MagicMock()
    manager.list_secrets.return_value = kwargs.get(
        "list_secrets",
        {"api_keys": ["BINANCE_API_KEY"], "database": ["POSTGRES_PASSWORD"]},
    )
    manager.get_secret.return_value = kwargs.get("get_secret", None)
    return manager


def test_list_prints_names_by_category(capsys):
    manager = _mock_manager()
    with patch.object(
        secrets_manager, "get_enhanced_secrets_manager", return_value=manager
    ):
        rc = secrets_manager.main(["--list"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "api_keys:" in out
    assert "BINANCE_API_KEY" in out
    assert "POSTGRES_PASSWORD" in out
    manager.list_secrets.assert_called_once()


def test_get_reports_presence_without_printing_value(capsys):
    manager = _mock_manager(get_secret=SECRET_VALUE)
    with patch.object(
        secrets_manager, "get_enhanced_secrets_manager", return_value=manager
    ):
        rc = secrets_manager.main(
            ["--get", "BINANCE_API_KEY", "--category", "api_keys"]
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "PRESENT" in out
    # The actual secret value must never be printed.
    assert SECRET_VALUE not in out
    manager.get_secret.assert_called_once_with(
        "BINANCE_API_KEY", "api_keys", warn_if_missing=False
    )


def test_get_missing_reports_missing_and_nonzero(capsys):
    manager = _mock_manager(get_secret=None)
    with patch.object(
        secrets_manager, "get_enhanced_secrets_manager", return_value=manager
    ):
        rc = secrets_manager.main(["--get", "NOPE"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "MISSING" in out


def test_set_prompts_hidden_and_stores(capsys):
    manager = _mock_manager()
    with (
        patch.object(
            secrets_manager, "get_enhanced_secrets_manager", return_value=manager
        ),
        patch.object(
            secrets_manager.getpass, "getpass", return_value=SECRET_VALUE
        ) as gp,
    ):
        rc = secrets_manager.main(
            ["--set", "BINANCE_API_KEY", "--category", "api_keys"]
        )

    out = capsys.readouterr().out
    assert rc == 0
    # Value was read via the hidden prompt, not printed.
    gp.assert_called_once()
    assert SECRET_VALUE not in out
    manager.set_secret.assert_called_once_with(
        "BINANCE_API_KEY", SECRET_VALUE, "api_keys"
    )


def test_requires_a_mode():
    # No --set/--get/--list -> argparse errors out (SystemExit 2).
    with pytest.raises(SystemExit):
        secrets_manager.main([])


def test_subprocess_list_smoke():
    # End-to-end: run as a module so the documented CLI actually works.
    # VAULT_ENABLED=false keeps it off any live service.
    env = dict(os.environ, VAULT_ENABLED="false")
    result = subprocess.run(
        [sys.executable, "-m", "utils.secrets_manager", "--list"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
