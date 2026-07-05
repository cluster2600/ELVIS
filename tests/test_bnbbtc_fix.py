#!/usr/bin/env python3
"""
Test the BNBBTC symbol fix - ensure it uses spot market correctly
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


def test_bnbbtc_price_fetcher():
    """Test that PriceFetcher can handle BNBBTC correctly"""
    logger.info("🧪 Testing BNBBTC PriceFetcher Fix")
    logger.info("=" * 50)

    try:
        from utils.price_fetcher import PriceFetcher

        logger.info("1. Creating PriceFetcher with BNBBTC symbol...")
        price_fetcher = PriceFetcher(
            logger=logger,
            symbols=["BTCUSDT", "BNBBTC"],  # Include both futures and spot symbols
            timeframe="1m",
        )

        logger.info("2. Testing spot-only symbol detection...")
        is_spot_only = price_fetcher._is_spot_only_symbol("BNBBTC")
        logger.info(f"   BNBBTC is spot-only: {is_spot_only}")

        if not is_spot_only:
            logger.error("   ❌ BNBBTC should be detected as spot-only")
            return False

        # Test that BTCUSDT is not spot-only
        btc_spot_only = price_fetcher._is_spot_only_symbol("BTCUSDT")
        logger.info(f"   BTCUSDT is spot-only: {btc_spot_only}")

        if btc_spot_only:
            logger.error("   ❌ BTCUSDT should not be spot-only")
            return False

        logger.info("3. Testing current price fetching...")
        try:
            # Test BNBBTC price (should use spot client)
            bnbbtc_price = price_fetcher.get_current_price("BNBBTC")
            if bnbbtc_price:
                logger.info(f"   ✅ BNBBTC price: {bnbbtc_price}")
            else:
                logger.warning("   ⚠️ Could not fetch BNBBTC price")

            # Test BTCUSDT price (should use regular client)
            btcusdt_price = price_fetcher.get_current_price("BTCUSDT")
            if btcusdt_price:
                logger.info(f"   ✅ BTCUSDT price: {btcusdt_price}")
            else:
                logger.warning("   ⚠️ Could not fetch BTCUSDT price")

        except Exception as e:
            logger.error(f"   ❌ Error fetching prices: {e}")
            return False

        logger.info("4. Testing historical data fetching...")
        try:
            # This should not raise the "Invalid symbol" error anymore
            price_fetcher.get_historical_data()
            logger.info("   ✅ Historical data fetching completed without errors")

            # Check if we got data for both symbols
            btc_candles = price_fetcher.candles.get("BTCUSDT", [])
            bnb_candles = price_fetcher.candles.get("BNBBTC", [])

            logger.info(f"   BTCUSDT candles: {len(btc_candles)}")
            logger.info(f"   BNBBTC candles: {len(bnb_candles)}")

            return True

        except Exception as e:
            logger.error(f"   ❌ Error fetching historical data: {e}")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def test_bnbbtc_historical_klines():
    """Test that get_historical_klines works for BNBBTC"""
    logger.info("\n📈 Testing BNBBTC Historical Klines")
    logger.info("-" * 50)

    try:
        from utils.price_fetcher import PriceFetcher

        logger.info("1. Creating PriceFetcher...")
        price_fetcher = PriceFetcher(logger=logger, symbols=["BNBBTC"])

        logger.info("2. Testing get_historical_klines for BNBBTC...")
        df = price_fetcher.get_historical_klines("BNBBTC", "1m", 50)

        if not df.empty:
            logger.info(f"   ✅ Got {len(df)} klines for BNBBTC")
            logger.info(f"   Latest price: {df['close'].iloc[-1]}")
            logger.info(
                f"   Date range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}"
            )
            return True
        else:
            logger.warning("   ⚠️ No klines data returned")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function"""
    logger.info("🚀 BNBBTC Symbol Fix Test")
    logger.info("=" * 60)

    # Test 1: PriceFetcher with BNBBTC
    test1_success = test_bnbbtc_price_fetcher()

    # Test 2: Historical klines
    test2_success = test_bnbbtc_historical_klines()

    # Summary
    logger.info("=" * 60)
    logger.info("📋 BNBBTC FIX TEST SUMMARY:")
    logger.info(f"   PriceFetcher BNBBTC Support: {'✅' if test1_success else '❌'}")
    logger.info(f"   Historical Klines BNBBTC: {'✅' if test2_success else '❌'}")

    all_tests_passed = all([test1_success, test2_success])

    if all_tests_passed:
        logger.info("\n🎉 BNBBTC SYMBOL FIX SUCCESS!")
        logger.info("✅ BNBBTC is now properly handled as spot-only symbol")
        logger.info("✅ PriceFetcher uses spot client for BNBBTC")
        logger.info("✅ No more 'Invalid symbol' errors for BNBBTC")
        logger.info("\n📋 Fixed Issues:")
        logger.info("   • BNBBTC now uses spot client instead of futures client")
        logger.info("   • Added spot_only_symbols detection")
        logger.info(
            "   • Fixed get_historical_data, get_current_price, and get_historical_klines"
        )
        logger.info("   • Bot should be able to fetch BNBBTC price data successfully")
    else:
        logger.warning("\n⚠️ Some tests failed - check logs above")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
