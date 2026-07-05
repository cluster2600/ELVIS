#!/usr/bin/env python3
"""
Focused tests for scripts/vault_admin.py::backup_vault_secrets.

These run in a minimal environment: the Vault client is mocked to return fake
secrets, so no live Vault/DB/Redis service is required.
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# cryptography ships with the project (used by the secrets manager), but guard
# the import so the suite still collects if it is ever stripped from CI.
Fernet = pytest.importorskip("cryptography.fernet").Fernet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_vault_admin():
    """Import scripts/vault_admin.py as a module (it is not a package)."""
    module_path = PROJECT_ROOT / "scripts" / "vault_admin.py"
    spec = importlib.util.spec_from_file_location("vault_admin_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_SECRETS = {
    "binance": {"api_key": "LIVE-KEY-123", "secret_key": "LIVE-SECRET-456"},
    "binance_testnet": {"api_key": "TEST-KEY-789", "secret_key": "TEST-SECRET-000"},
}

# Every plaintext value that must NOT appear verbatim in the encrypted file.
PLAINTEXT_VALUES = [v for fields in FAKE_SECRETS.values() for v in fields.values()]


def _make_manager(monkeypatch, vault_admin):
    """Patch get_enhanced_secrets_manager to yield a mock backed by fake data."""
    vault_client = MagicMock()
    vault_client.health_check.return_value = True
    vault_client.get_secret.side_effect = lambda path: FAKE_SECRETS.get(path)

    manager = MagicMock()
    manager.vault_client = vault_client

    monkeypatch.setattr(
        vault_admin, "get_enhanced_secrets_manager", lambda logger=None: manager
    )
    return manager


def test_backup_writes_encrypted_file_and_round_trips(tmp_path, monkeypatch):
    """Backup writes ciphertext (no plaintext) that decrypts back to the secrets."""
    vault_admin = _load_vault_admin()
    _make_manager(monkeypatch, vault_admin)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("BACKUP_KEY", key)

    out = tmp_path / "backup.enc"
    ok = vault_admin.backup_vault_secrets(str(out))

    assert ok is True
    assert out.exists()

    raw = out.read_bytes()

    # File must be encrypted: no plaintext secret value may appear in it.
    for value in PLAINTEXT_VALUES:
        assert value.encode() not in raw, f"plaintext {value!r} leaked into backup"
    # It must also not be a plaintext JSON dump.
    with pytest.raises(Exception):
        json.loads(raw.decode())

    # Round-trip: decrypting with the same key recovers the original secrets.
    decrypted = json.loads(Fernet(key.encode()).decrypt(raw).decode())
    assert decrypted == FAKE_SECRETS


def test_backup_generates_key_when_env_absent(tmp_path, monkeypatch, capsys):
    """Without BACKUP_KEY a key is generated and printed once for the operator."""
    vault_admin = _load_vault_admin()
    _make_manager(monkeypatch, vault_admin)

    monkeypatch.delenv("BACKUP_KEY", raising=False)

    out = tmp_path / "backup.enc"
    ok = vault_admin.backup_vault_secrets(str(out))

    assert ok is True
    assert out.exists()

    printed = capsys.readouterr().out
    assert "BACKUP_KEY=" in printed

    # Extract the printed key and confirm it decrypts the file (still encrypted).
    key_line = next(
        line for line in printed.splitlines() if line.strip().startswith("BACKUP_KEY=")
    )
    key = key_line.strip().split("BACKUP_KEY=", 1)[1]

    raw = out.read_bytes()
    for value in PLAINTEXT_VALUES:
        assert value.encode() not in raw

    decrypted = json.loads(Fernet(key.encode()).decrypt(raw).decode())
    assert decrypted == FAKE_SECRETS


def test_backup_returns_false_when_vault_unavailable(tmp_path, monkeypatch):
    """An unhealthy Vault yields a failed backup and writes no file."""
    vault_admin = _load_vault_admin()
    manager = _make_manager(monkeypatch, vault_admin)
    manager.vault_client.health_check.return_value = False

    monkeypatch.setenv("BACKUP_KEY", Fernet.generate_key().decode())

    out = tmp_path / "backup.enc"
    ok = vault_admin.backup_vault_secrets(str(out))

    assert ok is False
    assert not out.exists()
