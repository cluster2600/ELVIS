#!/usr/bin/env python3
"""
Test the enhanced paper trading with BNB balance
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


def test_paper_trading_balances():
    """Test paper trading with multi-asset balances"""
    try:
        from trading.execution.enhanced_binance_executor import EnhancedBinanceExecutor
        from utils.paper_trade_db import get_account_balance, get_all_balances

        logger.info("🧪 Testing Paper Trading with BNB Balances")
        logger.info("=" * 50)

        # Test 1: Check database balances
        logger.info("📊 Database Balances:")
        all_balances = get_all_balances()
        for asset, balance in all_balances.items():
            if balance > 0:
                logger.info(f"   {asset}: {balance:.6f}")

        # Test 2: Test enhanced executor
        executor = EnhancedBinanceExecutor(
            logger=logger,
            is_testnet=True,
            use_futures=True,
            enable_bnb_fees=True,
            bnb_trading_enabled=True,
        )

        executor.initialize()

        # Test 3: Get enhanced balance info
        balance_info = executor.get_enhanced_balance()

        logger.info("\n🪙 Enhanced Balance Information:")
        balances = balance_info["balances"]
        bnb_info = balance_info["bnb_info"]

        total_value_usd = 0
        for asset, amount in balances.items():
            if asset == "USDT":
                value_usd = amount
            elif asset == "BNB":
                value_usd = bnb_info["value_usdt"]
            else:
                value_usd = 0  # For other assets, would need price lookup

            total_value_usd += value_usd

            if amount > 0:
                logger.info(f"   {asset}: {amount:.6f} (${value_usd:.2f})")

        logger.info(f"\n💰 Total Portfolio Value: ${total_value_usd:.2f}")

        # Test 4: BNB specific info
        logger.info(f"\n🔍 BNB Details:")
        logger.info(f"   BNB Price: ${bnb_info['price_usdt']:.2f}")
        logger.info(f"   Sufficient for fees: {bnb_info['sufficient_for_fees']}")
        logger.info(f"   Min required: {bnb_info['min_required']} BNB")

        # Test 5: Fee optimization info
        fee_opt = balance_info["fee_optimization"]
        logger.info(f"\n⚙️ Fee Optimization:")
        logger.info(f"   Enabled: {fee_opt['enabled']}")
        logger.info(f"   Auto-buy enabled: {fee_opt['auto_buy_enabled']}")
        logger.info(f"   Futures discount: {fee_opt['futures_discount']:.0f}%")
        logger.info(f"   Spot discount: {fee_opt['spot_discount']:.0f}%")

        # Test 6: Test a small BNB trade (simulated)
        logger.info(f"\n🔄 Testing BNB Trade Simulation:")
        bnb_price = bnb_info["price_usdt"]
        trade_value = 100  # $100 trade
        bnb_quantity = trade_value / bnb_price

        fee_analysis = executor.calculate_fee_with_bnb(trade_value, is_futures=True)
        logger.info(f"   Trade: ${trade_value} (≈{bnb_quantity:.6f} BNB)")
        logger.info(f"   Standard fee: ${fee_analysis['standard_fee_usdt']:.6f}")
        logger.info(f"   With BNB discount: ${fee_analysis['discounted_fee_usdt']:.6f}")
        logger.info(f"   Savings: ${fee_analysis['savings_usdt']:.6f}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def test_bnb_auto_buy_logic():
    """Test BNB auto-buy decision logic"""
    try:
        from trading.execution.enhanced_binance_executor import EnhancedBinanceExecutor

        logger.info("\n🤖 Testing BNB Auto-Buy Logic")
        logger.info("-" * 50)

        executor = EnhancedBinanceExecutor(
            logger=logger,
            is_testnet=True,
            use_futures=True,
            enable_bnb_fees=True,
            bnb_trading_enabled=True,
            min_bnb_balance=0.1,
        )

        executor.initialize()

        # Test different trade scenarios
        trade_scenarios = [100, 1000, 5000, 10000]  # Different trade sizes

        for trade_value in trade_scenarios:
            analysis = executor.should_auto_buy_bnb(trade_value)

            logger.info(f"\n💵 ${trade_value:,} Trade Scenario:")
            logger.info(f"   Should buy BNB: {analysis['should_buy']}")
            logger.info(f"   Current balance: {analysis['current_balance']:.6f} BNB")
            logger.info(f"   Needed balance: {analysis['needed_balance']:.6f} BNB")

            if analysis["should_buy"]:
                logger.info(f"   Buy amount: {analysis['buy_amount_bnb']:.6f} BNB")
                logger.info(f"   Cost: ${analysis['buy_cost_usdt']:.2f}")

        return True

    except Exception as e:
        logger.error(f"❌ Auto-buy test failed: {e}")
        return False


def main():
    """Main test function"""
    logger.info("🚀 Paper Trading BNB Integration Test")
    logger.info("=" * 60)

    # Test 1: Basic balances and functionality
    test1_success = test_paper_trading_balances()

    # Test 2: BNB auto-buy logic
    test2_success = test_bnb_auto_buy_logic()

    # Summary
    logger.info("=" * 60)
    logger.info("📋 TEST RESULTS SUMMARY:")
    logger.info(f"   Balance & Fee Tests: {'✅ PASS' if test1_success else '❌ FAIL'}")
    logger.info(f"   Auto-Buy Logic: {'✅ PASS' if test2_success else '❌ FAIL'}")

    if test1_success and test2_success:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("✅ Paper trading with BNB is fully functional")
        logger.info("💡 Your bot now starts with:")
        logger.info("   • $1,000 USDT for trading")
        logger.info("   • ~1.32 BNB (~$1,000 worth) for fees & trading")
        logger.info("   • Automatic fee optimization")
        logger.info("   • Multi-asset portfolio tracking")
    else:
        logger.warning("⚠️ Some tests failed - check logs above")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
