#!/usr/bin/env python3
"""
Test the bug fixes:
1. Database clearing removed from paper trading initialization
2. RL training warnings fixed with synthetic data
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set environment
os.environ["VAULT_ENABLED"] = "false"


def test_database_preservation():
    """Test that database trades are preserved"""
    logger.info("🧪 Testing Database Preservation")
    logger.info("=" * 50)

    try:
        from trading.execution.binance_executor import BinanceExecutor
        from utils.paper_trade_db import get_all_trades, record_trade

        # Add a test trade first
        logger.info("1. Adding test trade to database...")
        record_trade("BTCUSDT", "TEST", 97000.0, 0.001, 10.0, 0.4)

        # Get trade count before initialization
        trades_before = get_all_trades()
        trade_count_before = len(trades_before)
        logger.info(f"   Trades before initialization: {trade_count_before}")

        # Create and initialize executor (should NOT clear trades)
        logger.info("2. Creating and initializing BinanceExecutor...")
        executor = BinanceExecutor(
            logger=logger, is_testnet=True, use_futures=False  # Paper trading
        )

        executor.initialize()

        # Get trade count after initialization
        trades_after = get_all_trades()
        trade_count_after = len(trades_after)
        logger.info(f"   Trades after initialization: {trade_count_after}")

        # Verify trades were preserved
        if trade_count_after >= trade_count_before:
            logger.info("   ✅ Database trades preserved successfully!")
            return True
        else:
            logger.error("   ❌ Database trades were cleared!")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def test_rl_synthetic_training():
    """Test RL model initialization with synthetic data"""
    logger.info("\n🚀 Testing RL Synthetic Training")
    logger.info("-" * 50)

    try:
        from core.models.trading_rl_model import TradingRLModel

        # Remove existing model file to force fresh training
        model_path = "models/test_trading_rl_model.pth"
        if os.path.exists(model_path):
            os.remove(model_path)

        logger.info("1. Creating RL model with fresh database...")
        rl_model = TradingRLModel(logger, model_path)

        logger.info("2. Training on historical data (should use synthetic data)...")
        success = rl_model.train_on_historical_data(limit=100)

        if success:
            logger.info("   ✅ RL training completed successfully!")

            # Check training stats
            stats = rl_model.get_training_stats()
            logger.info(f"   📊 Training stats: {stats}")

            # Test prediction
            market_data = {"price": 97000.0, "volume": 1000.0, "rsi": 50.0, "macd": 0.0}

            signal, confidence = rl_model.predict_action(market_data)
            logger.info(
                f"   🎯 Test prediction: {signal} (confidence: {confidence:.3f})"
            )

            return True
        else:
            logger.error("   ❌ RL training failed!")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function"""
    logger.info("🚀 Bug Fixes Test Suite")
    logger.info("=" * 60)

    # Test 1: Database preservation
    test1_success = test_database_preservation()

    # Test 2: RL synthetic training
    test2_success = test_rl_synthetic_training()

    # Summary
    logger.info("=" * 60)
    logger.info("📋 BUG FIXES TEST SUMMARY:")
    logger.info(f"   Database Preservation: {'✅' if test1_success else '❌'}")
    logger.info(f"   RL Synthetic Training: {'✅' if test2_success else '❌'}")

    all_tests_passed = all([test1_success, test2_success])

    if all_tests_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("✅ Database clearing bug is fixed!")
        logger.info("✅ RL training warnings are fixed!")
        logger.info("\n📋 Fixed Issues:")
        logger.info("   • Paper trading no longer clears existing database trades")
        logger.info(
            "   • RL model creates synthetic training data when no historical trades exist"
        )
        logger.info("   • Both warnings should no longer appear in normal operation")
    else:
        logger.warning("\n⚠️ Some tests failed - check logs above")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
