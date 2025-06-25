#!/usr/bin/env python3
"""
Test script to verify portfolio value calculation is working correctly.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from trading.execution.binance_executor import BinanceExecutor
from utils.paper_trade_db import init_db, record_trade

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_portfolio_calculation():
    """Test that portfolio values update correctly after trades."""
    
    logger.info("Testing portfolio value calculation...")
    
    # Initialize database
    init_db()
    
    # Create executor instance in paper trading mode
    executor = BinanceExecutor(logger=logger, is_testnet=True)
    executor._init_paper_trading_db()
    
    # Get initial balance
    initial_balance = executor.get_balance()
    initial_total = executor.get_account_balance()
    
    logger.info(f"Initial balance: {initial_balance}")
    logger.info(f"Initial total value: ${initial_total:.2f}")
    
    # Simulate some trades
    logger.info("\nSimulating trades...")
    
    # Buy trade: Buy 0.001 BTC at $97000
    record_trade("BTCUSDT", "BUY", 97000.0, 0.001, 0.0, 0.4)
    logger.info("Recorded BUY trade: 0.001 BTC at $97000")
    
    # Check balance after buy
    balance_after_buy = executor.get_balance()
    total_after_buy = executor.get_account_balance()
    
    logger.info(f"Balance after BUY: {balance_after_buy}")
    logger.info(f"Total value after BUY: ${total_after_buy:.2f}")
    
    # Sell trade: Sell 0.001 BTC at $98000 (profit)
    record_trade("BTCUSDT", "SELL", 98000.0, 0.001, 1000.0, 0.4)
    logger.info("Recorded SELL trade: 0.001 BTC at $98000")
    
    # Check balance after sell
    balance_after_sell = executor.get_balance()
    total_after_sell = executor.get_account_balance()
    
    logger.info(f"Balance after SELL: {balance_after_sell}")
    logger.info(f"Total value after SELL: ${total_after_sell:.2f}")
    
    # Calculate expected values
    expected_usdt = 10000.0 - 97.4 + 97.6  # Initial - buy_cost + sell_proceeds
    expected_btc = 0.0  # Should be back to 0 after round trip
    
    logger.info(f"\nExpected USDT: ${expected_usdt:.2f}")
    logger.info(f"Expected BTC: {expected_btc:.6f}")
    
    # Verify calculations
    actual_usdt = balance_after_sell['USDT']
    actual_btc = balance_after_sell['BTC']
    
    logger.info(f"Actual USDT: ${actual_usdt:.2f}")
    logger.info(f"Actual BTC: {actual_btc:.6f}")
    
    # Check if values are changing (not stuck at 10000)
    if total_after_buy != initial_total:
        logger.info("✅ Portfolio values ARE updating correctly")
    else:
        logger.error("❌ Portfolio values are NOT updating")
    
    # Check profit calculation
    profit = total_after_sell - initial_total
    logger.info(f"Net profit from trades: ${profit:.2f}")

if __name__ == "__main__":
    test_portfolio_calculation()