#!/usr/bin/env python3
"""
Test futures trading with leverage
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import pandas as pd
from trading.execution.binance_executor import BinanceExecutor
from trading.strategies.ensemble_strategy import EnsembleStrategy
from config.config import TRADING_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_futures_executor():
    """Test futures executor initialization"""
    logger.info("=== Testing Futures Executor ===")
    
    # Create futures executor with leverage
    executor = BinanceExecutor(
        logger=logger, 
        is_testnet=True,
        use_futures=True,
        default_leverage=10
    )
    
    success = executor.initialize()
    logger.info(f"Futures executor initialization: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        # Test balance retrieval
        balance = executor.get_balance()
        total_balance = executor.get_account_balance()
        logger.info(f"Futures Balance: {balance}")
        logger.info(f"Total Account Balance: ${total_balance:.2f}")
        
        # Test position information
        position = executor.get_position('BTCUSDT')
        logger.info(f"Current Position: {position}")
    
    return executor

def test_leverage_position_sizing():
    """Test position sizing with leverage"""
    logger.info("=== Testing Leverage Position Sizing ===")
    
    # Initialize strategy
    strategy = EnsembleStrategy(logger=logger)
    
    # Create mock data
    mock_data = pd.DataFrame({
        'open': [97000] * 50,
        'high': [97500] * 50,
        'low': [96500] * 50,
        'close': [97000] * 50,
        'volume': [1000] * 50
    })
    
    current_price = 97000.0
    available_capital = 10000.0
    
    # Test different leverage levels
    leverage_levels = [1, 5, 10, 20, 50]
    
    logger.info(f"Testing position sizing with different leverage levels:")
    logger.info(f"Available Capital: ${available_capital:.2f}")
    logger.info(f"Current Price: ${current_price:.2f}")
    logger.info(f"Risk per trade: 2%")
    logger.info("")
    
    for leverage in leverage_levels:
        position_size = strategy.calculate_position_size(
            mock_data, current_price, available_capital, leverage
        )
        position_value = position_size * current_price
        leverage_exposure = position_value
        actual_capital_used = position_value / leverage
        
        logger.info(f"Leverage {leverage}x:")
        logger.info(f"  Position Size: {position_size:.6f} BTC")
        logger.info(f"  Position Value: ${position_value:.2f}")
        logger.info(f"  Capital Used: ${actual_capital_used:.2f}")
        logger.info(f"  Leverage Exposure: ${leverage_exposure:.2f}")
        logger.info(f"  % of Capital: {(actual_capital_used / available_capital) * 100:.2f}%")
        logger.info("")

def test_leverage_trades():
    """Test placing leveraged trades"""
    logger.info("=== Testing Leveraged Trades ===")
    
    # Create futures executor
    executor = BinanceExecutor(
        logger=logger, 
        is_testnet=True,
        use_futures=False,  # Use paper trading for safety
        default_leverage=10
    )
    executor.initialize()
    
    current_price = 97000.0
    leverage = 10
    
    # Test leveraged BUY order
    logger.info(f"Placing leveraged BUY order with {leverage}x leverage")
    position_size = 0.01  # Small test size
    
    buy_result = executor.execute_buy('BTCUSDT', position_size, current_price)
    logger.info(f"BUY Result: {buy_result}")
    
    # Test leveraged SELL order
    logger.info(f"Placing leveraged SELL order with {leverage}x leverage")
    sell_result = executor.execute_sell('BTCUSDT', position_size, current_price + 100)  # Small profit
    logger.info(f"SELL Result: {sell_result}")

def test_configuration():
    """Test futures configuration"""
    logger.info("=== Testing Futures Configuration ===")
    
    logger.info(f"Default Mode: {TRADING_CONFIG['DEFAULT_MODE']}")
    logger.info(f"Default Leverage: {TRADING_CONFIG['DEFAULT_LEVERAGE']}")
    logger.info(f"Leverage Range: {TRADING_CONFIG['LEVERAGE_MIN']}x - {TRADING_CONFIG['LEVERAGE_MAX']}x")
    logger.info(f"Risk Per Trade: {TRADING_CONFIG['RISK_PER_TRADE'] * 100}%")
    
    # Show how leverage affects risk
    capital = 10000
    risk_per_trade = TRADING_CONFIG['RISK_PER_TRADE']
    leverage = TRADING_CONFIG['DEFAULT_LEVERAGE']
    
    logger.info(f"\nLeverage Risk Analysis:")
    logger.info(f"  Capital: ${capital}")
    logger.info(f"  Risk per trade: {risk_per_trade * 100}% = ${capital * risk_per_trade}")
    logger.info(f"  With {leverage}x leverage: ${capital * risk_per_trade * leverage} exposure")
    logger.info(f"  Actual capital at risk: ${capital * risk_per_trade} (same as without leverage)")

if __name__ == "__main__":
    logger.info("Starting Futures Trading with Leverage Tests...")
    
    # Test 1: Configuration
    test_configuration()
    print()
    
    # Test 2: Futures executor
    executor = test_futures_executor()
    print()
    
    # Test 3: Leverage position sizing
    test_leverage_position_sizing()
    print()
    
    # Test 4: Leverage trades
    test_leverage_trades()
    
    logger.info("Futures trading tests completed!")