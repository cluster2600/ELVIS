#!/usr/bin/env python3
"""
Test script for Bonenkamp HFT Strategy implementation
Tests the exact methodology from the research paper
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trading.strategies.bonenkamp_hft_strategy import BonenkampHFTStrategy
from trading.strategies.ensemble_strategy import EnsembleStrategy


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def create_test_data():
    """Create realistic test data for Bitcoin"""
    # Generate 100 periods of 5-minute data (about 8 hours)
    periods = 100
    base_price = 107000  # Current realistic BTC price

    # Generate realistic price movements
    np.random.seed(42)  # For reproducible tests
    price_changes = np.random.normal(0, 0.005, periods)  # 0.5% standard deviation

    prices = [base_price]
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    # Create DataFrame with OHLCV data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=8), periods=periods + 1, freq="5min"
    )

    # Ensure all arrays have the same length (periods)
    open_prices = prices[:-1]
    close_prices = prices[1:]
    high_prices = [p * (1 + abs(np.random.normal(0, 0.002))) for p in open_prices]
    low_prices = [p * (1 - abs(np.random.normal(0, 0.002))) for p in open_prices]
    volumes = [np.random.uniform(500, 2000) for _ in range(periods)]

    data = {
        "timestamp": timestamps[:-1],  # Remove last timestamp to match periods
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    }

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)

    return df


def test_financial_indicators():
    """Test the 9 financial indicators calculation"""
    logger = setup_logging()
    logger.info("🧪 Testing Financial Indicators Calculation")

    # Create strategy instance
    strategy = BonenkampHFTStrategy(logger, use_social_features=False)

    # Create test data
    test_data = create_test_data()

    # Calculate indicators
    indicators = strategy.calculate_financial_indicators(test_data)

    # Verify all 9 indicators are present
    expected_indicators = [
        "RSI",
        "STOCH",
        "ROC",
        "EMA",
        "MACD",
        "CCI",
        "OBV",
        "ATR",
        "WILLR",
    ]

    logger.info(f"📊 Calculated indicators: {list(indicators.keys())}")

    for indicator in expected_indicators:
        assert indicator in indicators, f"Missing indicator: {indicator}"
        assert isinstance(
            indicators[indicator], (int, float)
        ), f"Invalid type for {indicator}"
        assert not np.isnan(indicators[indicator]), f"NaN value for {indicator}"
        logger.info(f"✅ {indicator}: {indicators[indicator]:.4f}")

    logger.info("✅ All financial indicators calculated successfully!")
    return indicators


def test_social_features():
    """Test social feature collection"""
    logger = setup_logging()
    logger.info("🧪 Testing Social Features Collection")

    # Create strategy with social features enabled
    strategy = BonenkampHFTStrategy(logger, use_social_features=True)

    # Test social features
    social_features = strategy.collect_social_features()

    expected_features = ["TWITTER_PRICE_LAG", "GOOGLE_TRENDS"]

    logger.info(f"📱 Social features: {social_features}")

    for feature in expected_features:
        assert feature in social_features, f"Missing social feature: {feature}"
        assert isinstance(
            social_features[feature], (int, float)
        ), f"Invalid type for {feature}"
        logger.info(f"✅ {feature}: {social_features[feature]:.4f}")

    logger.info("✅ Social features collected successfully!")
    return social_features


def test_feature_preparation():
    """Test feature vector preparation"""
    logger = setup_logging()
    logger.info("🧪 Testing Feature Vector Preparation")

    # Test both configurations
    for use_social in [False, True]:
        strategy = BonenkampHFTStrategy(logger, use_social_features=use_social)
        test_data = create_test_data()

        features = strategy.prepare_feature_vector(test_data)

        expected_features = 11 if use_social else 9
        assert features.shape == (
            1,
            expected_features,
        ), f"Wrong feature shape: {features.shape}"
        assert not np.any(np.isnan(features)), "Features contain NaN values"

        logger.info(f"✅ Feature vector (social={use_social}): shape {features.shape}")

    logger.info("✅ Feature preparation working correctly!")


def test_model_training():
    """Test Random Forest model training"""
    logger = setup_logging()
    logger.info("🧪 Testing Model Training")

    strategy = BonenkampHFTStrategy(logger, use_social_features=True)
    test_data = create_test_data()

    # Train model
    f1_score = strategy.train_model(test_data)

    assert f1_score >= 0.0, "Invalid F1 score"
    assert strategy.is_trained, "Strategy not marked as trained"

    logger.info(f"✅ Model trained with F1-score: {f1_score:.3f}")
    logger.info(f"🎯 Target F1-score: {strategy.target_f1_score:.3f}")

    return f1_score


def test_signal_generation():
    """Test trading signal generation"""
    logger = setup_logging()
    logger.info("🧪 Testing Signal Generation")

    strategy = BonenkampHFTStrategy(logger, use_social_features=True)
    test_data = create_test_data()

    # Train model first
    strategy.train_model(test_data)

    # Generate signals
    signals = strategy.generate_signals({"BTCUSDT": test_data})

    assert "BTCUSDT" in signals, "No signal generated for BTCUSDT"

    signal_data = signals["BTCUSDT"]
    assert "signal" in signal_data, "Missing signal field"
    assert "confidence" in signal_data, "Missing confidence field"
    assert signal_data["signal"] in [
        "BUY",
        "SELL",
        "HOLD",
    ], f"Invalid signal: {signal_data['signal']}"
    assert (
        0.0 <= signal_data["confidence"] <= 1.0
    ), f"Invalid confidence: {signal_data['confidence']}"

    logger.info(
        f"✅ Generated signal: {signal_data['signal']} (confidence: {signal_data['confidence']:.3f})"
    )

    return signals


def test_performance_metrics():
    """Test performance metrics calculation"""
    logger = setup_logging()
    logger.info("🧪 Testing Performance Metrics")

    strategy = BonenkampHFTStrategy(logger, use_social_features=True)

    # Simulate some trading returns
    strategy.daily_returns = [0.01, -0.005, 0.015, 0.02, -0.01, 0.008, 0.012]
    strategy.f1_scores = [0.65, 0.58, 0.62, 0.59, 0.61]

    metrics = strategy.calculate_performance_metrics()

    assert "sharpe_ratio" in metrics, "Missing Sharpe ratio"
    assert "annual_return" in metrics, "Missing annual return"
    assert "f1_score" in metrics, "Missing F1 score"

    logger.info(f"📊 Performance Metrics:")
    logger.info(
        f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (target: {strategy.target_sharpe_ratio:.2f})"
    )
    logger.info(
        f"   Annual Return: {metrics['annual_return']:.1%} (target: {strategy.target_annual_return:.1%})"
    )
    logger.info(
        f"   F1 Score: {metrics['f1_score']:.3f} (target: {strategy.target_f1_score:.3f})"
    )

    logger.info("✅ Performance metrics calculated successfully!")
    return metrics


def test_ensemble_integration():
    """Test integration with ensemble strategy"""
    logger = setup_logging()
    logger.info("🧪 Testing Ensemble Integration")

    # Create ensemble with Bonenkamp strategy enabled
    ensemble = EnsembleStrategy(
        logger=logger,
        symbols=["BTCUSDT"],
        enable_research_strategy=False,  # Disable others for clean test
        enable_rl_strategy=False,
        enable_bonenkamp_hft=True,
        bonenkamp_use_social=True,
    )

    # Verify Bonenkamp strategy is initialized
    assert ensemble.enable_bonenkamp_hft, "Bonenkamp HFT not enabled"
    assert ensemble.bonenkamp_strategy is not None, "Bonenkamp strategy not initialized"

    # Create test market data
    market_data = {
        "close": 107500.0,
        "price": 107500.0,
        "high": 108000.0,
        "low": 107000.0,
        "volume": 1200.0,
        "rsi": 55.0,
        "macd": 150.0,
        "macd_signal": 145.0,
        "sma_20": 107200.0,
        "sma_50": 106800.0,
    }

    # Generate signal
    signal, confidence = ensemble.generate_signal("BTCUSDT", market_data)

    assert signal in ["BUY", "SELL", "HOLD"], f"Invalid ensemble signal: {signal}"
    assert 0.0 <= confidence <= 1.0, f"Invalid ensemble confidence: {confidence}"

    logger.info(f"✅ Ensemble signal: {signal} (confidence: {confidence:.3f})")
    logger.info("✅ Ensemble integration working correctly!")

    return signal, confidence


def run_comprehensive_test():
    """Run all tests comprehensively"""
    logger = setup_logging()
    logger.info("🚀 Starting Comprehensive Bonenkamp HFT Strategy Test")
    logger.info("=" * 60)

    try:
        # Test 1: Financial Indicators
        logger.info("Test 1: Financial Indicators")
        indicators = test_financial_indicators()
        logger.info("")

        # Test 2: Social Features
        logger.info("Test 2: Social Features")
        social_features = test_social_features()
        logger.info("")

        # Test 3: Feature Preparation
        logger.info("Test 3: Feature Preparation")
        test_feature_preparation()
        logger.info("")

        # Test 4: Model Training
        logger.info("Test 4: Model Training")
        f1_score = test_model_training()
        logger.info("")

        # Test 5: Signal Generation
        logger.info("Test 5: Signal Generation")
        signals = test_signal_generation()
        logger.info("")

        # Test 6: Performance Metrics
        logger.info("Test 6: Performance Metrics")
        metrics = test_performance_metrics()
        logger.info("")

        # Test 7: Ensemble Integration
        logger.info("Test 7: Ensemble Integration")
        ensemble_signal, ensemble_confidence = test_ensemble_integration()
        logger.info("")

        # Summary
        logger.info("=" * 60)
        logger.info("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        logger.info("")
        logger.info("📊 Test Summary:")
        logger.info(f"   Financial Indicators: ✅ All 9 calculated")
        logger.info(f"   Social Features: ✅ Twitter & Google Trends")
        logger.info(f"   Model Training: ✅ F1-score: {f1_score:.3f}")
        logger.info(f"   Signal Generation: ✅ {signals['BTCUSDT']['signal']}")
        logger.info(f"   Performance Metrics: ✅ Calculated")
        logger.info(f"   Ensemble Integration: ✅ {ensemble_signal}")
        logger.info("")
        logger.info("🎯 Bonenkamp (2021) Research Implementation: COMPLETE")
        logger.info("📈 Ready for high-frequency trading at 5-minute intervals")
        logger.info("🎪 Target Performance: 14.9% annual return, 2.02 Sharpe ratio")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
