#!/usr/bin/env python3
"""
Debug the main bot execution to understand why trades aren't completing
"""

import logging
import time
import pandas as pd
import numpy as np
from trading.execution.binance_executor import BinanceExecutor
from trading.strategies.ensemble_strategy import EnsembleStrategy

def debug_main_bot():
    """Debug the main bot execution flow"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== DEBUGGING MAIN BOT EXECUTION ===")
    
    # Step 1: Initialize executor (same as main bot)
    logger.info("1. Initializing BinanceExecutor...")
    executor = BinanceExecutor(logger=logger, is_testnet=True)
    executor.initialize()
    
    # Step 2: Initialize strategy (same as main bot)
    logger.info("2. Initializing EnsembleStrategy...")
    strategy = EnsembleStrategy(logger=logger)
    
    # Step 3: Create market data (same as main bot mock data)
    logger.info("3. Creating market data...")
    np.random.seed(42)
    mock_data = {
        'open': np.random.normal(97000, 200, 50),
        'high': np.random.normal(97200, 200, 50),
        'low': np.random.normal(96800, 200, 50),
        'close': np.random.normal(97000, 200, 50),
        'volume': np.random.normal(1000, 100, 50),
    }
    data = pd.DataFrame(mock_data)
    
    # Add technical indicators (same as main bot)
    data['rsi'] = 35  # Oversold - should trigger BUY
    data['macd'] = 5.0
    data['signal_line'] = 2.0  # MACD > signal - should trigger BUY
    data['sma_20'] = 96900
    data['adx'] = 40
    data['atr'] = 100
    
    logger.info(f"Data shape: {data.shape}, Close price: {data.iloc[-1]['close']:.2f}")
    
    # Step 4: Generate signals (same as main bot)
    logger.info("4. Generating signals...")
    signals = strategy.generate_signals({"BTCUSDT": data})
    logger.info(f"Generated signals: {signals}")
    
    # Step 5: Process signals (same as main bot)
    logger.info("5. Processing signals...")
    for symbol, signal_info in signals.items():
        signal = signal_info.get('signal', 'HOLD')
        confidence = signal_info.get('confidence', 0.0)
        
        logger.info(f"Signal: {signal}, Confidence: {confidence}")
        
        if signal in ['BUY', 'SELL'] and confidence > 0.1:
            current_price = data.iloc[-1]['close']
            available_balance = executor.get_account_balance()
            
            # Calculate position size (same as main bot)
            position_size = strategy.calculate_position_size(data, current_price, available_balance)
            
            logger.info(f"Executing {signal} order - Price: ${current_price:.2f}, Size: {position_size:.6f}")
            
            # Execute trade (same as main bot)
            if signal == 'BUY':
                logger.info("Calling executor.place_order for BUY...")
                order_result = executor.place_order(symbol, 'buy', position_size, current_price)
                logger.info(f"Order result: {order_result}")
                if order_result:
                    logger.info(f"✅ BUY order executed successfully")
                else:
                    logger.error(f"❌ Failed to execute BUY order")
            elif signal == 'SELL':
                logger.info("Calling executor.place_order for SELL...")
                order_result = executor.place_order(symbol, 'sell', position_size, current_price)
                logger.info(f"Order result: {order_result}")
                if order_result:
                    logger.info(f"✅ SELL order executed successfully")
                else:
                    logger.error(f"❌ Failed to execute SELL order")
        else:
            logger.info(f"No trade executed - conditions not met")
    
    logger.info("=== DEBUGGING COMPLETE ===")
    return True

if __name__ == "__main__":
    debug_main_bot()