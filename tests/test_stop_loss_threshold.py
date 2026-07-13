"""Stop-loss threshold (review on #47): the FOURTH $1000-era absolute.
Guards against reintroducing fixed-dollar exits that micro-notional
positions can never hit."""

import pytest

from main import _stop_loss_threshold


def test_default_is_half_percent_of_notional(monkeypatch):
    monkeypatch.delenv("ELVIS_SL_PCT", raising=False)
    assert _stop_loss_threshold(64000.0, 0.001) == pytest.approx(-0.32)


def test_micro_notional_threshold_is_reachable(monkeypatch):
    """The regression: at ~$1 notional the old -$15 could never trigger."""
    monkeypatch.delenv("ELVIS_SL_PCT", raising=False)
    threshold = _stop_loss_threshold(64000.0, 0.000016)
    assert -1.0 < threshold < 0  # small, hittable, never a fixed -15


def test_env_override(monkeypatch):
    monkeypatch.setenv("ELVIS_SL_PCT", "0.01")
    assert _stop_loss_threshold(100.0, 1.0) == pytest.approx(-1.0)


def test_malformed_env_falls_back(monkeypatch):
    monkeypatch.setenv("ELVIS_SL_PCT", "not-a-number")
    assert _stop_loss_threshold(64000.0, 0.001) == pytest.approx(-0.32)


def test_always_negative(monkeypatch):
    monkeypatch.setenv("ELVIS_SL_PCT", "0.02")
    assert _stop_loss_threshold(50000.0, 0.5) < 0
