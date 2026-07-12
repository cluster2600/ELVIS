"""Portfolio-protection floor (review findings on #45).

The old hardcoded `balance < 700` floor emergency-stopped every paper run
once the deposit became $100. The floor must be mode-aware: paper measures
against the configured deposit, live against the session's first observed
real balance — never the fictional paper deposit.
"""

import pytest

from config.config import PAPER_TRADING_CONFIG
from main import _protection_floor

DEPOSIT = float(PAPER_TRADING_CONFIG["INITIAL_USDT_BALANCE"])


def test_paper_floor_is_pct_of_configured_deposit(monkeypatch):
    monkeypatch.delenv("ELVIS_PROTECT_FLOOR_PCT", raising=False)
    floor, baseline, pct = _protection_floor(
        "paper", current_balance=55.0, baseline_state={}
    )
    assert baseline == DEPOSIT
    assert pct == pytest.approx(0.7)
    assert floor == pytest.approx(DEPOSIT * 0.7)


def test_paper_deposit_passes_its_own_floor(monkeypatch):
    """The exact regression: the starting deposit must clear the floor."""
    monkeypatch.delenv("ELVIS_PROTECT_FLOOR_PCT", raising=False)
    floor, _, _ = _protection_floor("paper", DEPOSIT, {})
    assert DEPOSIT >= floor


def test_env_pct_override(monkeypatch):
    monkeypatch.setenv("ELVIS_PROTECT_FLOOR_PCT", "0.5")
    floor, _, pct = _protection_floor("paper", 55.0, {})
    assert pct == pytest.approx(0.5)
    assert floor == pytest.approx(DEPOSIT * 0.5)


def test_live_uses_first_observed_balance_not_paper_deposit(monkeypatch):
    monkeypatch.delenv("ELVIS_PROTECT_FLOOR_PCT", raising=False)
    state = {}
    floor, baseline, _ = _protection_floor("live", 5000.0, state)
    assert baseline == 5000.0  # real account, not the $100 paper deposit
    assert floor == pytest.approx(3500.0)


def test_live_baseline_sticks_across_iterations(monkeypatch):
    """A later dip must be measured against the SESSION-START balance,
    otherwise the floor would chase the balance down and never trigger."""
    monkeypatch.delenv("ELVIS_PROTECT_FLOOR_PCT", raising=False)
    state = {}
    _protection_floor("live", 5000.0, state)
    floor, baseline, _ = _protection_floor("live", 4000.0, state)
    assert baseline == 5000.0
    assert floor == pytest.approx(3500.0)
    assert 4000.0 >= floor  # -20% survives
    assert 3400.0 < floor  # -32% trips


def test_paper_ignores_live_baseline_state(monkeypatch):
    monkeypatch.delenv("ELVIS_PROTECT_FLOOR_PCT", raising=False)
    state = {"baseline": 9999.0}
    _, baseline, _ = _protection_floor("paper", 50.0, state)
    assert baseline == DEPOSIT
