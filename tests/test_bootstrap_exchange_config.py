"""Tests that Kraken/Coinbase credentials are actually configurable.

APIConfig now exposes KRAKEN_*/COINBASE_* properties (Vault first, env
fallback, None when unset) plus a dict-style .get(), so
core/bootstrap.create_exchange_manager registers those exchanges when creds
are present and only Binance otherwise. These test the APIConfig contract
directly (no DI container / live exchanges needed).
"""

import importlib

import pytest


@pytest.fixture
def api_config():
    # Reload so property reads reflect the current environment.
    import config.config as cfg

    importlib.reload(cfg)
    return cfg.APIConfig()


def test_kraken_coinbase_none_when_unset(api_config, monkeypatch):
    for var in (
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "COINBASE_PASSPHRASE",
    ):
        monkeypatch.delenv(var, raising=False)
    # No secrets manager value + no env -> None (so bootstrap skips them)
    monkeypatch.setattr(api_config._secrets, "get_secret", lambda *a, **k: None)
    assert api_config.KRAKEN_API_KEY is None
    assert api_config.COINBASE_API_SECRET is None


def test_kraken_coinbase_read_from_env(api_config, monkeypatch):
    monkeypatch.setattr(api_config._secrets, "get_secret", lambda *a, **k: None)
    monkeypatch.setenv("KRAKEN_API_KEY", "k-key")
    monkeypatch.setenv("KRAKEN_API_SECRET", "k-sec")
    monkeypatch.setenv("COINBASE_API_KEY", "c-key")
    assert api_config.KRAKEN_API_KEY == "k-key"
    assert api_config.KRAKEN_API_SECRET == "k-sec"
    assert api_config.COINBASE_API_KEY == "c-key"


def test_get_accessor_does_not_raise(api_config, monkeypatch):
    # Regression: bootstrap uses api_config.get(...); APIConfig had no .get(),
    # which raised AttributeError on the Kraken/Coinbase path.
    monkeypatch.setattr(api_config._secrets, "get_secret", lambda *a, **k: None)
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    assert api_config.get("KRAKEN_API_KEY") is None
    assert api_config.get("NONEXISTENT", "dflt") == "dflt"


def test_secrets_manager_value_wins_over_env(api_config, monkeypatch):
    monkeypatch.setenv("KRAKEN_API_KEY", "env-key")
    monkeypatch.setattr(
        api_config._secrets,
        "get_secret",
        lambda name, *a, **k: "vault-key" if name == "KRAKEN_API_KEY" else None,
    )
    assert api_config.KRAKEN_API_KEY == "vault-key"
