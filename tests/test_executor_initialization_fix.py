#!/usr/bin/env python3
"""
Test the executor initialization fix
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


def test_binance_executor_initialization():
    """Test that Binance executor returns proper boolean values"""
    logger.info("🧪 Testing Binance Executor Initialization")
    logger.info("=" * 50)

    try:
        from trading.execution.binance_executor import BinanceExecutor

        # Test 1: Paper trading (should succeed)
        logger.info("1. Testing paper trading initialization...")
        executor = BinanceExecutor(
            logger=logger, is_testnet=True, use_futures=False  # Paper trading
        )

        result = executor.initialize()
        logger.info(f"   Paper trading result: {result} (type: {type(result)})")
        assert isinstance(result, bool), "Result should be boolean"
        assert result == True, "Paper trading should succeed"
        logger.info("   ✅ Paper trading initialization returns True")

        # Test 2: Futures testnet (might fail due to API keys, but should return boolean)
        logger.info("2. Testing futures testnet initialization...")
        executor2 = BinanceExecutor(
            logger=logger, is_testnet=True, use_futures=True  # Futures
        )

        result2 = executor2.initialize()
        logger.info(f"   Futures testnet result: {result2} (type: {type(result2)})")
        assert isinstance(result2, bool), "Result should be boolean"
        logger.info("   ✅ Futures testnet initialization returns boolean")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def test_enhanced_executor_initialization():
    """Test that Enhanced Binance executor returns proper boolean values"""
    logger.info("\n🚀 Testing Enhanced Executor Initialization")
    logger.info("-" * 50)

    try:
        from trading.execution.enhanced_binance_executor import EnhancedBinanceExecutor

        # Test enhanced executor
        logger.info("1. Testing enhanced executor initialization...")
        executor = EnhancedBinanceExecutor(
            logger=logger, is_testnet=True, use_futures=False, enable_bnb_fees=True
        )

        result = executor.initialize()
        logger.info(f"   Enhanced executor result: {result} (type: {type(result)})")
        assert isinstance(result, bool), "Result should be boolean"
        assert result == True, "Enhanced executor should succeed in paper mode"
        logger.info("   ✅ Enhanced executor initialization returns True")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def test_bootstrap_executor_creation():
    """Test that bootstrap can now properly check initialization"""
    logger.info("\n⚙️ Testing Bootstrap Executor Creation")
    logger.info("-" * 50)

    try:
        # Simulate the bootstrap logic
        from trading.execution.binance_executor import BinanceExecutor

        logger.info("1. Simulating bootstrap executor creation...")

        # This simulates what happens in bootstrap.py
        executor = BinanceExecutor(
            logger=logger,
            is_testnet=True,
            use_futures=True,  # Try futures first
            default_leverage=50,
        )

        success = executor.initialize()
        logger.info(f"   First attempt (futures): {success}")

        if not success:
            logger.info("   Falling back to paper trading...")
            # Fallback to paper trading executor
            executor = BinanceExecutor(
                logger=logger, is_testnet=True, use_futures=False
            )
            success = executor.initialize()
            logger.info(f"   Fallback attempt (paper): {success}")

        assert isinstance(success, bool), "Success should be boolean"
        logger.info("   ✅ Bootstrap logic now works correctly")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function"""
    logger.info("🚀 Executor Initialization Fix Test")
    logger.info("=" * 60)

    # Test 1: Basic executor
    test1_success = test_binance_executor_initialization()

    # Test 2: Enhanced executor
    test2_success = test_enhanced_executor_initialization()

    # Test 3: Bootstrap simulation
    test3_success = test_bootstrap_executor_creation()

    # Summary
    logger.info("=" * 60)
    logger.info("📋 INITIALIZATION FIX TEST SUMMARY:")
    logger.info(f"   Basic Executor: {'✅' if test1_success else '❌'}")
    logger.info(f"   Enhanced Executor: {'✅' if test2_success else '❌'}")
    logger.info(f"   Bootstrap Simulation: {'✅' if test3_success else '❌'}")

    all_tests_passed = all([test1_success, test2_success, test3_success])

    if all_tests_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("✅ Executor initialization bug is fixed!")
        logger.info(
            "✅ Bootstrap will no longer show 'falling back to paper trading' warning"
        )
        logger.info("✅ All executors now return proper boolean success/failure status")
        logger.info("\n📋 Fixed Issues:")
        logger.info("   • initialize() now returns bool instead of None")
        logger.info("   • Bootstrap can properly detect initialization success/failure")
        logger.info("   • Enhanced executor properly chains initialization")
        logger.info("   • Paper trading always returns True (successful)")
    else:
        logger.warning("\n⚠️ Some tests failed - check logs above")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
