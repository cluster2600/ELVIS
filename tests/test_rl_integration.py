#!/usr/bin/env python3
"""
Test script to verify RL integration with the ensemble strategy
"""

import logging
import os
import sys
from datetime import datetime

import pandas as pd
import pytest

pytest.importorskip(
    "torch", reason="RL model/strategy require torch, unavailable in this env"
)

from core.models.trading_rl_model import TradingRLModel
from trading.strategies.ensemble_strategy import EnsembleStrategy
from trading.strategies.rl_strategy import RLStrategy
from utils.paper_trade_db import get_all_trades

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_rl_model_standalone():
    """Test the RL model standalone"""
    logger.info("Testing RL model standalone...")

    try:
        # Initialize RL model
        rl_model = TradingRLModel(logger, "models/trading_rl_model.pth")

        # Test with sample market data
        test_market_data = {
            "price": 119000.0,
            "volume": 1000.0,
            "rsi": 45.0,
            "macd": 0.1,
            "bb_upper": 120000.0,
            "bb_lower": 118000.0,
            "atr": 500.0,
            "adx": 25.0,
        }

        # Get prediction
        signal, confidence = rl_model.predict_action(test_market_data)
        logger.info(
            f"RL Model standalone prediction: {signal} (confidence: {confidence:.3f})"
        )

        # Test with different market conditions
        test_cases = [
            {"price": 119000.0, "volume": 1000.0, "rsi": 25.0, "macd": 0.5},  # Oversold
            {
                "price": 119000.0,
                "volume": 1000.0,
                "rsi": 75.0,
                "macd": -0.3,
            },  # Overbought
            {"price": 119000.0, "volume": 1000.0, "rsi": 50.0, "macd": 0.0},  # Neutral
        ]

        for i, test_case in enumerate(test_cases):
            signal, confidence = rl_model.predict_action(test_case)
            logger.info(
                f"Test case {i+1}: {signal} ({confidence:.3f}) - RSI: {test_case['rsi']}, MACD: {test_case['macd']}"
            )

        return True

    except Exception as e:
        logger.error(f"RL model standalone test failed: {e}")
        return False


def test_rl_strategy():
    """Test the RL strategy"""
    logger.info("Testing RL strategy...")

    try:
        # Initialize RL strategy
        rl_strategy = RLStrategy(logger, ["BTCUSDT"])

        # Test with sample market data
        test_market_data = {
            "price": 119000.0,
            "volume": 1000.0,
            "rsi": 45.0,
            "macd": 0.1,
            "bb_upper": 120000.0,
            "bb_lower": 118000.0,
            "atr": 500.0,
            "adx": 25.0,
        }

        # Get prediction
        signal, confidence = rl_strategy.generate_signal("BTCUSDT", test_market_data)
        logger.info(f"RL Strategy prediction: {signal} (confidence: {confidence:.3f})")

        # Test performance metrics
        metrics = rl_strategy.get_performance_metrics()
        logger.info(f"RL Strategy metrics: {metrics}")

        return True

    except Exception as e:
        logger.error(f"RL strategy test failed: {e}")
        return False


def test_ensemble_integration():
    """Test the ensemble strategy with RL integration"""
    logger.info("Testing ensemble strategy with RL integration...")

    try:
        # Initialize ensemble strategy with RL enabled
        ensemble_strategy = EnsembleStrategy(
            logger=logger,
            symbols=["BTCUSDT"],
            enable_rl_strategy=True,
            enable_research_strategy=False,  # Disable research strategy for simpler testing
        )

        # Test with sample market data
        test_market_data = {
            "price": 119000.0,
            "close": 119000.0,
            "volume": 1000.0,
            "rsi": 45.0,
            "macd": 0.1,
            "macd_signal": 0.05,
            "bb_upper": 120000.0,
            "bb_lower": 118000.0,
            "bb_middle": 119000.0,
            "atr": 500.0,
            "adx": 25.0,
            "sma_20": 119200.0,
            "sma_50": 119500.0,
        }

        # Get ensemble prediction
        signal, confidence = ensemble_strategy.generate_signal(
            "BTCUSDT", test_market_data
        )
        logger.info(
            f"Ensemble Strategy prediction: {signal} (confidence: {confidence:.3f})"
        )

        # Test with different market conditions
        test_cases = [
            {
                "price": 119000.0,
                "close": 119000.0,
                "volume": 1000.0,
                "rsi": 25.0,
                "macd": 0.5,
                "macd_signal": 0.3,
            },  # Oversold
            {
                "price": 119000.0,
                "close": 119000.0,
                "volume": 1000.0,
                "rsi": 75.0,
                "macd": -0.3,
                "macd_signal": -0.1,
            },  # Overbought
            {
                "price": 119000.0,
                "close": 119000.0,
                "volume": 1000.0,
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
            },  # Neutral
        ]

        for i, test_case in enumerate(test_cases):
            # Fill in missing fields with defaults
            complete_test_case = {**test_market_data, **test_case}
            signal, confidence = ensemble_strategy.generate_signal(
                "BTCUSDT", complete_test_case
            )
            logger.info(
                f"Ensemble test case {i+1}: {signal} ({confidence:.3f}) - RSI: {test_case['rsi']}, MACD: {test_case['macd']}"
            )

        # Test RL performance metrics
        rl_metrics = ensemble_strategy.get_rl_performance_metrics()
        logger.info(f"Ensemble RL metrics: {rl_metrics}")

        return True

    except Exception as e:
        logger.error(f"Ensemble integration test failed: {e}")
        return False


def test_trade_result_update():
    """Test updating RL model with trade results"""
    logger.info("Testing trade result updates...")

    try:
        # Initialize ensemble strategy with RL enabled
        ensemble_strategy = EnsembleStrategy(
            logger=logger,
            symbols=["BTCUSDT"],
            enable_rl_strategy=True,
            enable_research_strategy=False,
        )

        # Simulate trade results
        trade_results = [
            {
                "price": 119000.0,
                "pnl": 1.50,
                "fees": 0.48,
                "side": "BUY",
                "quantity": 0.001,
            },
            {
                "price": 119200.0,
                "pnl": -0.80,
                "fees": 0.48,
                "side": "SELL",
                "quantity": 0.001,
            },
            {
                "price": 119100.0,
                "pnl": 2.10,
                "fees": 0.48,
                "side": "BUY",
                "quantity": 0.001,
            },
        ]

        # Update RL model with trade results
        for i, trade_result in enumerate(trade_results):
            ensemble_strategy.update_rl_with_trade_result(trade_result)
            logger.info(
                f"Updated RL model with trade result {i+1}: PnL=${trade_result['pnl']:.2f}"
            )

        # Test metrics after updates
        rl_metrics = ensemble_strategy.get_rl_performance_metrics()
        logger.info(f"RL metrics after updates: {rl_metrics}")

        return True

    except Exception as e:
        logger.error(f"Trade result update test failed: {e}")
        return False


def main():
    """Main test function"""
    logger.info("Starting RL integration tests...")

    # Check if trained model exists
    if not os.path.exists("models/trading_rl_model.pth"):
        logger.error("RL model not found. Please run train_rl_model.py first.")
        sys.exit(1)

    # Run tests
    tests = [
        ("RL Model Standalone", test_rl_model_standalone),
        ("RL Strategy", test_rl_strategy),
        ("Ensemble Integration", test_ensemble_integration),
        ("Trade Result Updates", test_trade_result_update),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*60}")

        try:
            result = test_func()
            results.append((test_name, result))

            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")

        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! RL integration is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
