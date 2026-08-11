"""
Focused tests for ResearchBasedStrategy behavior that must match
docs/research_strategy_guide.md.

These tests run in a minimal environment: they do NOT require talib, torch,
tweepy, or pytrends. The strategy is constructed with social data disabled and
its Random Forest model is mocked, so no real training/prediction is needed.

Covered guarantees:
- generate_signals() never returns a 'HOLD' signal (binary BUY/SELL only),
  across the untrained fallback, the trained-prediction path, the
  prediction-failure fallback, and the empty-data guard.
- train_model() uses a time-series-aware split (TimeSeriesSplit, no shuffle)
  for its 10-fold cross-validation instead of the default (Stratified)KFold.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_strategy(tmp_path, social_data_enabled=False):
    """Build a ResearchBasedStrategy without touching disk models."""
    from trading.strategies.research_based_strategy import ResearchBasedStrategy

    logger = logging.getLogger("test_research_strategy_features")

    # Avoid loading any pre-existing on-disk model during construction.
    with patch.object(ResearchBasedStrategy, "load_model", return_value=False):
        strategy = ResearchBasedStrategy(
            logger=logger,
            social_data_enabled=social_data_enabled,
            enable_rolling_training=False,
            model_save_path=str(tmp_path / "models"),
        )
    return strategy


def _make_ohlcv(periods=80, base_price=100000.0, seed=7):
    """Create a small OHLCV DataFrame sufficient for indicator calculation."""
    rng = np.random.default_rng(seed)
    changes = rng.normal(0, 0.005, periods)
    closes = [base_price]
    for change in changes:
        closes.append(closes[-1] * (1 + change))
    closes = closes[1:]
    highs = [c * 1.001 for c in closes]
    lows = [c * 0.999 for c in closes]
    volumes = [rng.uniform(500, 2000) for _ in range(periods)]
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def _financial_features(rsi=50.0):
    from trading.models.feature_schemas import RESEARCH_FINANCIAL_9_V1

    values = {
        name: float(index + 1)
        for index, name in enumerate(RESEARCH_FINANCIAL_9_V1.names)
    }
    values["RSI"] = rsi
    return values


def test_generate_signals_untrained_never_hold(tmp_path):
    """Untrained model must fall back to a binary BUY/SELL, never HOLD."""
    strategy = _make_strategy(tmp_path)
    strategy.is_trained = False

    df = _make_ohlcv()
    signals = strategy.generate_signals({"BTCUSDT": df})

    assert signals["BTCUSDT"]["signal"] in ("BUY", "SELL")
    assert signals["BTCUSDT"]["signal"] != "HOLD"


def test_generate_signals_untrained_neutral_rsi_picks_side(tmp_path):
    """A neutral RSI (35..65) must still resolve to BUY or SELL, not HOLD."""
    strategy = _make_strategy(tmp_path)
    strategy.is_trained = False

    df = _make_ohlcv()
    # Force a neutral RSI so the previously-HOLD branch is exercised.
    with patch.object(
        strategy,
        "calculate_financial_indicators",
        return_value=_financial_features(),
    ):
        signals = strategy.generate_signals({"BTCUSDT": df})

    assert signals["BTCUSDT"]["signal"] in ("BUY", "SELL")


def test_generate_signals_trained_prediction_never_hold(tmp_path):
    """A trained model prediction path yields BUY/SELL, never HOLD."""
    strategy = _make_strategy(tmp_path)
    strategy.is_trained = True

    # Mock the RF model so no real prediction/training is needed.
    mock_model = MagicMock()
    # probabilities[0]=SELL prob, probabilities[1]=BUY prob
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    strategy.rf_model = mock_model
    strategy.feature_scaler = strategy.feature_scaler.fit(np.ones((2, 9)))

    df = _make_ohlcv()
    signals = strategy.generate_signals({"BTCUSDT": df})

    assert signals["BTCUSDT"]["signal"] in ("BUY", "SELL")
    assert signals["BTCUSDT"]["signal"] != "HOLD"
    mock_model.predict_proba.assert_called_once()


def test_generate_signals_prediction_failure_fallback_never_hold(tmp_path):
    """If prediction raises, the RSI fallback must still be binary."""
    strategy = _make_strategy(tmp_path)
    strategy.is_trained = True

    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = RuntimeError("boom")
    strategy.rf_model = mock_model
    strategy.feature_scaler = strategy.feature_scaler.fit(np.ones((2, 9)))

    df = _make_ohlcv()
    # Neutral RSI to hit the previously-HOLD branch inside the fallback.
    with patch.object(
        strategy,
        "calculate_financial_indicators",
        return_value=_financial_features(),
    ):
        signals = strategy.generate_signals({"BTCUSDT": df})

    assert signals["BTCUSDT"]["signal"] in ("BUY", "SELL")
    assert signals["BTCUSDT"]["signal"] != "HOLD"
    mock_model.predict_proba.assert_called_once()


def test_generate_signals_empty_data_never_hold(tmp_path):
    """Missing/empty market data must not produce a HOLD signal."""
    strategy = _make_strategy(tmp_path)

    signals = strategy.generate_signals({"BTCUSDT": pd.DataFrame()})
    assert signals["BTCUSDT"]["signal"] in ("BUY", "SELL")
    assert signals["BTCUSDT"]["signal"] != "HOLD"


def test_train_model_uses_time_series_split(tmp_path):
    """train_model must pass a TimeSeriesSplit(n_splits=10) to cross_val_score."""
    from sklearn.model_selection import TimeSeriesSplit

    import trading.strategies.research_based_strategy as mod

    strategy = _make_strategy(tmp_path)

    # Enough rows to pass the len>=100 guard and yield >=50 valid samples.
    training_data = _make_ohlcv(periods=160)

    captured = {}

    def fake_cross_val_score(estimator, X, y, cv=None, **kwargs):
        captured["cv"] = cv
        return np.array([0.5] * getattr(cv, "n_splits", 1))

    with patch.object(mod, "cross_val_score", side_effect=fake_cross_val_score):
        # Avoid disk writes from the real save_model.
        with patch.object(strategy, "save_model", return_value=None):
            strategy.train_model(training_data)

    assert isinstance(captured.get("cv"), TimeSeriesSplit)
    assert captured["cv"].n_splits == strategy.cv_folds == 10
    # TimeSeriesSplit never shuffles; assert it is a genuine time-series splitter.
    assert not hasattr(captured["cv"], "shuffle") or captured["cv"].shuffle is False


def test_social_collectors_are_import_guarded(tmp_path):
    """Collectors return the documented neutral constants when libs are absent."""
    import trading.strategies.research_based_strategy as mod
    from trading.strategies.research_based_strategy import (
        GoogleTrendsCollector,
        TwitterSentimentCollector,
    )

    logger = logging.getLogger("test_research_strategy_features")

    # Simulate the libraries being unavailable (as in the minimal CI env).
    with patch.object(mod, "tweepy", None), patch.object(mod, "TrendReq", None):
        twitter = TwitterSentimentCollector(logger)
        trends = GoogleTrendsCollector(logger)
        assert twitter.get_lagged_price_sentiment() == 0.0
        assert trends.get_bitcoin_trends_interpolated() == 50.0
