"""Focused behavior tests for the Bonenkamp HFT strategy."""

import logging
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from trading.models.feature_schemas import BONENKAMP_FINANCIAL_9_V1
from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy
from trading.strategies.ensemble_strategy import EnsembleStrategy


def create_test_data(periods: int = 120) -> pd.DataFrame:
    """Create deterministic OHLCV data with both upward and downward labels."""
    rng = np.random.default_rng(42)
    base_price = 107_000.0
    changes = rng.normal(0, 0.005, periods)
    prices = [base_price]
    for change in changes:
        prices.append(prices[-1] * (1 + change))

    opens = prices[:-1]
    closes = prices[1:]
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=5 * periods),
        periods=periods,
        freq="5min",
    )
    return pd.DataFrame(
        {
            "open": opens,
            "high": [price * 1.002 for price in opens],
            "low": [price * 0.998 for price in opens],
            "close": closes,
            "volume": rng.uniform(500, 2_000, periods),
        },
        index=timestamps,
    )


def make_strategy(tmp_path, *, social: bool = False) -> BonenkampHFTStrategy:
    with patch.object(BonenkampHFTStrategy, "load_model", return_value=False):
        return BonenkampHFTStrategy(
            logging.getLogger("test-bonenkamp-strategy"),
            use_social_features=social,
            model_save_path=str(tmp_path),
        )


def financial_feature_values() -> dict[str, float]:
    return {
        name: float(index + 1)
        for index, name in enumerate(BONENKAMP_FINANCIAL_9_V1.names)
    }


def test_financial_indicators_are_finite(tmp_path) -> None:
    strategy = make_strategy(tmp_path)

    indicators = strategy.calculate_financial_indicators(create_test_data())

    assert tuple(indicators) == BONENKAMP_FINANCIAL_9_V1.names
    assert all(np.isfinite(value) for value in indicators.values())


def test_social_features_are_finite(tmp_path) -> None:
    strategy = make_strategy(tmp_path, social=True)

    social_features = strategy.collect_social_features()

    assert set(social_features) == {"TWITTER_PRICE_LAG", "GOOGLE_TRENDS"}
    assert all(np.isfinite(value) for value in social_features.values())


@pytest.mark.parametrize("social, size", [(False, 9), (True, 11)])
def test_feature_preparation_has_declared_size(
    tmp_path, social: bool, size: int
) -> None:
    strategy = make_strategy(tmp_path, social=social)

    features = strategy.prepare_feature_vector(create_test_data())

    assert features.shape == (1, size)
    assert np.isfinite(features).all()


def test_model_training_activates_candidate_after_validation(tmp_path) -> None:
    strategy = make_strategy(tmp_path)
    strategy.rf_model = RandomForestClassifier(n_estimators=2, random_state=3)

    with patch(
        "trading.strategies.bonenkamp_hft_strategy.cross_val_score",
        return_value=np.full(10, 0.5),
    ):
        score = strategy.train_model(create_test_data())

    assert score == 0.5
    assert strategy.is_trained is True
    assert tuple(strategy.rf_model.classes_) == (0, 1)
    assert (tmp_path / "feature_manifest.json").is_file()


def test_trained_model_generates_declared_signal(tmp_path) -> None:
    strategy = make_strategy(tmp_path)
    training = np.arange(180, dtype=float).reshape(20, 9)
    labels = np.array([0, 1] * 10)
    strategy.feature_scaler = StandardScaler().fit(training)
    strategy.rf_model = RandomForestClassifier(n_estimators=2, random_state=3).fit(
        strategy.feature_scaler.transform(training), labels
    )
    strategy.is_trained = True

    with (
        patch.object(
            strategy,
            "calculate_financial_indicators",
            return_value=financial_feature_values(),
        ),
        patch.object(
            strategy.rf_model,
            "predict_proba",
            return_value=np.array([[0.3, 0.7]]),
        ) as predict,
    ):
        result = strategy.generate_signals(
            {"BTCUSDT": pd.DataFrame({"close": [107_500.0]})}
        )["BTCUSDT"]

    predict.assert_called_once()
    assert result["signal"] == "BUY"
    assert result["confidence"] == 0.7


def test_performance_metrics_have_expected_fields(tmp_path) -> None:
    strategy = make_strategy(tmp_path, social=True)
    strategy.daily_returns = [0.01, -0.005, 0.015, 0.02, -0.01, 0.008, 0.012]
    strategy.f1_scores = [0.65, 0.58, 0.62, 0.59, 0.61]

    metrics = strategy.calculate_performance_metrics()

    assert set(metrics) == {
        "sharpe_ratio",
        "annual_return",
        "f1_score",
        "cumulative_return",
    }
    assert all(np.isfinite(value) for value in metrics.values())


def test_ensemble_keeps_bonenkamp_integration_available() -> None:
    logger = logging.getLogger("test-bonenkamp-ensemble")
    with patch.object(BonenkampHFTStrategy, "load_model", return_value=False):
        ensemble = EnsembleStrategy(
            logger=logger,
            symbols=["BTCUSDT"],
            enable_research_strategy=False,
            enable_rl_strategy=False,
            enable_bonenkamp_hft=True,
            bonenkamp_use_social=True,
        )

    assert ensemble.enable_bonenkamp_hft is True
    assert ensemble.bonenkamp_strategy is not None
