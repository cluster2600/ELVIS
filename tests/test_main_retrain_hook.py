"""Tests for the trading-loop pacing and retrain hook wired into ``main.py``.

These cover the two helpers that back the trading loop:

* ``_strategy_loop_sleep_seconds`` — paces the loop from the strategy's
  ``trading_frequency_minutes`` (documented 5-minute frequency) with a sane
  default.
* ``_retrain_strategy_if_due`` — once-per-iteration, non-fatal retrain hook
  that calls ``strategy.should_retrain()`` when present.

The helpers are pure (they only need ``logging``), so we load just their
definitions out of ``main.py`` rather than importing the whole module. That
keeps the test free of ``main``'s import-time side effects (Vault/Binance/DI
bootstrap) and free of any heavy, CI-absent dependencies (torch, talib,
optuna, shap, pytrends, tweepy).
"""

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"
_HELPER_NAMES = ("_strategy_loop_sleep_seconds", "_retrain_strategy_if_due")


def _load_helpers():
    """Exec only the two helper functions from ``main.py`` in isolation."""
    source = MAIN_PATH.read_text()
    module = ast.parse(source)
    wanted = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in _HELPER_NAMES
    ]
    assert {n.name for n in wanted} == set(
        _HELPER_NAMES
    ), "main.py must define the pacing + retrain helper functions"
    namespace = {"logging": logging}
    code = compile(ast.Module(body=wanted, type_ignores=[]), str(MAIN_PATH), "exec")
    exec(code, namespace)
    return namespace


@pytest.fixture(scope="module")
def helpers():
    return _load_helpers()


# --- pacing -------------------------------------------------------------------


def test_sleep_uses_trading_frequency_minutes(helpers):
    strategy = MagicMock()
    strategy.trading_frequency_minutes = 5
    seconds = helpers["_strategy_loop_sleep_seconds"](strategy, default_seconds=1.0)
    assert seconds == 5 * 60


def test_sleep_falls_back_to_default_when_unset(helpers):
    # A plain object without the attribute -> use the default.
    class Strat:
        pass

    seconds = helpers["_strategy_loop_sleep_seconds"](Strat(), default_seconds=1.0)
    assert seconds == 1.0


def test_sleep_ignores_nonpositive_or_bad_frequency(helpers):
    for bad in (0, -3, None, "nope"):
        strategy = MagicMock()
        strategy.trading_frequency_minutes = bad
        seconds = helpers["_strategy_loop_sleep_seconds"](strategy, default_seconds=2.5)
        assert seconds == 2.5


# --- retrain hook -------------------------------------------------------------


def test_retrain_noop_when_should_retrain_missing(helpers):
    class Strat:
        # no should_retrain attribute at all
        train_model = MagicMock()

    strategy = Strat()
    price_fetcher = MagicMock()
    result = helpers["_retrain_strategy_if_due"](
        strategy, price_fetcher, logging.getLogger("t")
    )
    assert result is False
    strategy.train_model.assert_not_called()
    price_fetcher.get_historical_klines.assert_not_called()


def test_retrain_skipped_when_not_due(helpers):
    strategy = MagicMock()
    strategy.should_retrain.return_value = False
    price_fetcher = MagicMock()
    result = helpers["_retrain_strategy_if_due"](
        strategy, price_fetcher, logging.getLogger("t")
    )
    assert result is False
    strategy.should_retrain.assert_called_once_with()
    strategy.train_model.assert_not_called()


def test_retrain_runs_when_due(helpers):
    strategy = MagicMock()
    strategy.should_retrain.return_value = True

    # Enough non-empty data (>200 rows) to pass the guard.
    data = MagicMock()
    data.empty = False
    data.__len__ = lambda self: 500
    price_fetcher = MagicMock()
    price_fetcher.get_historical_klines.return_value = data

    result = helpers["_retrain_strategy_if_due"](
        strategy, price_fetcher, logging.getLogger("t")
    )
    assert result is True
    price_fetcher.get_historical_klines.assert_called_once()
    strategy.train_model.assert_called_once_with(data)


def test_retrain_is_non_fatal_on_error(helpers):
    strategy = MagicMock()
    strategy.should_retrain.side_effect = RuntimeError("boom")
    price_fetcher = MagicMock()
    # Must swallow the exception and report False, never raise.
    result = helpers["_retrain_strategy_if_due"](
        strategy, price_fetcher, logging.getLogger("t")
    )
    assert result is False
