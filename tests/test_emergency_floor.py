"""balanced_starter emergency floor (review on #47): the THIRD $1000-era
hardcoded threshold. Guards against re-introducing absolute-dollar floors."""

import pytest

from config.config import PAPER_TRADING_CONFIG
from trading.strategies.balanced_starter import emergency_close_floor

DEPOSIT = float(PAPER_TRADING_CONFIG["INITIAL_USDT_BALANCE"])


def test_default_floor_is_20pct_of_deposit(monkeypatch):
    monkeypatch.delenv("ELVIS_EMERGENCY_CLOSE_PCT", raising=False)
    assert emergency_close_floor() == pytest.approx(DEPOSIT * 0.2)


def test_deposit_clears_its_own_floor(monkeypatch):
    """The regression: at the $100 deposit the old $200 floor fired every
    cycle. The configured deposit must always sit above the floor."""
    monkeypatch.delenv("ELVIS_EMERGENCY_CLOSE_PCT", raising=False)
    assert DEPOSIT > emergency_close_floor()


def test_env_override(monkeypatch):
    monkeypatch.setenv("ELVIS_EMERGENCY_CLOSE_PCT", "0.5")
    assert emergency_close_floor() == pytest.approx(DEPOSIT * 0.5)


def test_malformed_env_falls_back_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("ELVIS_EMERGENCY_CLOSE_PCT", "banana")
    assert emergency_close_floor() == pytest.approx(DEPOSIT * 0.2)
