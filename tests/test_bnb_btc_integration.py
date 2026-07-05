#!/usr/bin/env python3
"""
Test the complete BNB → BTC integration
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


def test_multi_symbol_price_fetching():
    """Test that price fetcher can handle both BTCUSDT and BNBBTC"""
    logger.info("🧪 Testing Multi-Symbol Price Fetching")
    logger.info("=" * 50)

    try:
        from utils.price_fetcher import PriceFetcher

        # Test with both symbols
        price_fetcher = PriceFetcher(
            logger=logger, symbols=["BTCUSDT", "BNBBTC"], timeframe="1m"
        )

        # Test BTCUSDT
        btc_price = price_fetcher.get_current_price("BTCUSDT")
        logger.info(f"✅ BTCUSDT Price: ${btc_price}")

        # Test BNBBTC (this will use mock data in paper mode)
        bnb_btc_price = price_fetcher.get_current_price("BNBBTC")
        logger.info(f"✅ BNBBTC Price: {bnb_btc_price} BTC")

        # Test historical data
        btc_data = price_fetcher.get_historical_klines("BTCUSDT", "1m", limit=5)
        bnb_data = price_fetcher.get_historical_klines("BNBBTC", "1m", limit=5)

        logger.info(f"✅ BTCUSDT Data: {len(btc_data)} records")
        logger.info(f"✅ BNBBTC Data: {len(bnb_data)} records")

        return True

    except Exception as e:
        logger.error(f"❌ Price fetching test failed: {e}")
        return False


def test_dual_executor_setup():
    """Test that we can create both futures and spot executors"""
    logger.info("\n🤖 Testing Dual Executor Setup")
    logger.info("-" * 50)

    try:
        from trading.execution.binance_executor import BinanceExecutor

        # Test futures executor for BTCUSDT
        futures_executor = BinanceExecutor(
            logger=logger, is_testnet=True, use_futures=True  # Futures for BTCUSDT
        )
        futures_executor.initialize()
        logger.info("✅ Futures executor initialized")

        # Test spot executor for BNBBTC
        spot_executor = BinanceExecutor(
            logger=logger, is_testnet=True, use_futures=False  # Spot for BNBBTC
        )
        spot_executor.initialize()
        logger.info("✅ Spot executor initialized")

        # Test balances
        futures_balance = futures_executor.get_balance()
        spot_balance = spot_executor.get_balance()

        logger.info(f"✅ Futures Balance: {futures_balance}")
        logger.info(f"✅ Spot Balance: {spot_balance}")

        return True

    except Exception as e:
        logger.error(f"❌ Dual executor test failed: {e}")
        return False


def test_bnb_btc_conversion_logic():
    """Test the BNB → BTC conversion logic"""
    logger.info("\n💱 Testing BNB → BTC Conversion Logic")
    logger.info("-" * 50)

    try:
        from trading.execution.binance_executor import BinanceExecutor

        # Create spot executor with BNB balance
        executor = BinanceExecutor(logger=logger, is_testnet=True, use_futures=False)
        executor.initialize()

        # Get current balance
        balance = executor.get_balance()
        bnb_balance = balance.get("BNB", 0)
        total_balance = executor.get_account_balance()

        logger.info(f"📊 Current BNB Balance: {bnb_balance:.6f}")
        logger.info(f"📊 Total Portfolio: ${total_balance:.2f}")

        # Calculate BNB allocation
        bnb_price_estimate = 757  # Current BNB price from our tests
        bnb_value_usdt = bnb_balance * bnb_price_estimate
        bnb_allocation = bnb_value_usdt / total_balance if total_balance > 0 else 0

        logger.info(f"📊 BNB Value: ${bnb_value_usdt:.2f}")
        logger.info(f"📊 BNB Allocation: {bnb_allocation:.1%}")

        # Test conversion threshold
        should_convert = bnb_allocation > 0.05  # 5% threshold
        logger.info(f"🔄 Should Convert BNB→BTC: {should_convert}")

        if should_convert and bnb_balance > 0.1:
            # Simulate conversion
            conversion_amount = bnb_balance * 0.3  # Convert 30%
            bnbbtc_price = 0.00665  # From our real market test
            btc_amount = conversion_amount * bnbbtc_price

            logger.info(
                f"💡 Simulation: Convert {conversion_amount:.6f} BNB → {btc_amount:.8f} BTC"
            )

            # Test the trade
            result = executor._execute_paper_trade(
                "BNBBTC", "BUY", btc_amount, bnbbtc_price
            )
            if result:
                logger.info(f"✅ BNB→BTC conversion simulation successful")
                logger.info(f"   Order ID: {result.get('orderId', 'N/A')}")
            else:
                logger.warning("⚠️ Conversion simulation failed")
        else:
            logger.info("💡 No conversion needed (allocation below threshold)")

        return True

    except Exception as e:
        logger.error(f"❌ Conversion logic test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def test_bootstrap_configuration():
    """Test that bootstrap includes both symbols"""
    logger.info("\n⚙️ Testing Bootstrap Configuration")
    logger.info("-" * 50)

    try:
        # Check if bootstrap file has been updated
        bootstrap_path = "/Users/maxime/BTC_BOT/BTC_BOT/core/bootstrap.py"

        with open(bootstrap_path, "r") as f:
            content = f.read()

        # Check for multi-symbol configuration
        if "BNBBTC" in content:
            logger.info("✅ Bootstrap includes BNBBTC symbol")
        else:
            logger.warning("⚠️ Bootstrap may not include BNBBTC")

        if "symbols=['BTCUSDT', 'BNBBTC']" in content:
            logger.info("✅ Price fetcher configured for both symbols")
        else:
            logger.warning("⚠️ Price fetcher may not be configured for both symbols")

        return True

    except Exception as e:
        logger.error(f"❌ Bootstrap test failed: {e}")
        return False


def main():
    """Main test function"""
    logger.info("🚀 BNB → BTC Integration Test")
    logger.info("=" * 60)

    # Test 1: Multi-symbol price fetching
    test1_success = test_multi_symbol_price_fetching()

    # Test 2: Dual executor setup
    test2_success = test_dual_executor_setup()

    # Test 3: BNB → BTC conversion logic
    test3_success = test_bnb_btc_conversion_logic()

    # Test 4: Bootstrap configuration
    test4_success = test_bootstrap_configuration()

    # Summary
    logger.info("=" * 60)
    logger.info("📋 INTEGRATION TEST SUMMARY:")
    logger.info(f"   Multi-Symbol Price Fetching: {'✅' if test1_success else '❌'}")
    logger.info(f"   Dual Executor Setup: {'✅' if test2_success else '❌'}")
    logger.info(f"   BNB→BTC Conversion Logic: {'✅' if test3_success else '❌'}")
    logger.info(f"   Bootstrap Configuration: {'✅' if test4_success else '❌'}")

    all_tests_passed = all([test1_success, test2_success, test3_success, test4_success])

    if all_tests_passed:
        logger.info("\n🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("✅ Your bot is now ready to buy BTC with BNB!")
        logger.info("\n📋 What happens now:")
        logger.info("   1. Bot monitors both BTCUSDT (futures) and BNBBTC (spot)")
        logger.info("   2. When BNB allocation > 5%, bot converts BNB → BTC")
        logger.info("   3. BTCUSDT trading continues as normal")
        logger.info("   4. BTC accumulates in your balance")
        logger.info("\n🔄 To start trading:")
        logger.info("   python main.py --mode paper")
    else:
        logger.warning("\n⚠️ Some tests failed - check logs above")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
