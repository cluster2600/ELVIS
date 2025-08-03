#!/usr/bin/env python3
"""
Simple test for balanced strategy configuration.
"""

import os
import logging
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def test_balanced_strategy_simple():
    """Simple test of balanced strategy"""
    print("=== TESTING BALANCED STRATEGY SIMPLE ===")
    
    # Test environment variable
    strategy_mode = os.getenv('STRATEGY_MODE', 'ensemble')
    print(f"STRATEGY_MODE: {strategy_mode}")
    
    # Test strategy directly
    try:
        from trading.strategies.balanced_starter import BalancedStarterStrategy
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Create strategy
        strategy = BalancedStarterStrategy(logger)
        
        print(f"Strategy type: {type(strategy).__name__}")
        print(f"Target profit per trade: ${strategy.target_profit_per_trade}")
        print(f"Daily trade target: {strategy.daily_trade_target}")
        print(f"Min position hold time: {strategy.min_position_hold_time}s")
        print(f"Adaptation interval: {strategy.adaptation_interval}s")
        
        # Test signal generation
        market_data = {
            'price': 97000.0,
            'close': 97000.0,
            'rsi': 55.0,
            'macd': 0.001,
            'volume': 1000.0
        }
        
        signal, confidence = strategy.generate_signal('BTCUSDT', market_data)
        print(f"Test signal: {signal} (confidence: {confidence:.3f})")
        
        print("✅ Balanced strategy configuration looks good!")
        
    except Exception as e:
        print(f"❌ Error testing strategy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_balanced_strategy_simple()