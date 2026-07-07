"""Tests that secrets listing covers the real flat Vault layout.

``EnhancedSecretsManager.list_secrets()`` (used by ``vault_admin.py --list``)
must enumerate the ``_VAULT_KEY_MAP`` flat paths the bot actually reads
(``secrets/binance``, ``secrets/binance_testnet``) — reporting field NAMES
only, never secret values.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("cryptography")

from utils.secrets_manager import EnhancedSecretsManager  # noqa: E402

SECRET_VALUE = "super-secret-key-value-123456"


def _manager_with_mock_vault(monkeypatch):
    mgr = EnhancedSecretsManager.__new__(EnhancedSecretsManager)
    mgr._secrets_cache = {}
    mgr._cipher = None
    vault = MagicMock()

    def get_secret(path, key=None):
        if path in ("binance", "binance_testnet"):
            return {"api_key": SECRET_VALUE, "secret_key": SECRET_VALUE}
        return None  # legacy category paths empty

    vault.get_secret.side_effect = get_secret
    mgr.vault_client = vault
    monkeypatch.setattr(mgr, "_load_secrets", lambda: {}, raising=False)
    return mgr


def test_list_includes_flat_key_map_paths(monkeypatch):
    mgr = _manager_with_mock_vault(monkeypatch)
    listing = mgr.list_secrets()
    assert "vault_binance" in listing
    assert "vault_binance_testnet" in listing
    assert sorted(listing["vault_binance"]) == ["api_key", "secret_key"]


def test_list_never_exposes_values(monkeypatch):
    mgr = _manager_with_mock_vault(monkeypatch)
    listing = mgr.list_secrets()
    assert SECRET_VALUE not in repr(listing)
