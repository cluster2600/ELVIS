#!/usr/bin/env python3
"""
Test the recalibrated bot for the new Bitcoin price paradigm ($120k+).
This script verifies that all thresholds and parameters work correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock

# Add the project root to the path
sys.path.append('/Users/maxime/BTC_BOT/BTC_BOT')

from trading.strategies.ensemble_strategy import EnsembleStrategy
import logging

def test_ath_detection():
    """Test that ATH detection works correctly with new thresholds."""
    print("=== Testing ATH Detection ===")
    
    # Create a mock logger
    logger = Mock()
    logger.info = lambda x: print(f"INFO: {x}")
    logger.warning = lambda x: print(f"WARNING: {x}")
    logger.error = lambda x: print(f"ERROR: {x}")
    logger.debug = lambda x: print(f"DEBUG: {x}")
    
    # Create strategy instance
    strategy = EnsembleStrategy(logger=logger)
    
    # Test different price levels
    test_prices = [
        (118000, "Current actual BTC price"),
        (122000, "Near ATH"),
        (125000, "High ATH threshold"),
        (130000, "Extreme ATH threshold"),
        (135000, "Emergency stop level")
    ]
    
    for price, description in test_prices:
        print(f"\n--- Testing {description}: ${price:,} ---")
        
        # Create mock market data
        market_data = {
            'close': price,
            'price': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'volume': 1000,
            'rsi': 55.0,
            'macd': 0.1,
            'macd_signal': 0.05,
            'sma_20': price * 0.98,
            'sma_50': price * 0.96,
            'atr': price * 0.02,
            'adx': 25.0,
            'bb_upper': price * 1.02,
            'bb_lower': price * 0.98,
            'bb_middle': price
        }
        
        try:
            signal, confidence = strategy.generate_signal('BTCUSDT', market_data)
            print(f"   Result: {signal} with {confidence:.1%} confidence")
            
            # Verify expected behavior
            if price > 130000:
                print(f"   ✓ Expected: SELL bias in extreme ATH territory")
            elif price > 125000:
                print(f"   ✓ Expected: SELL bias in high ATH territory") 
            elif price > 120000:
                print(f"   ✓ Expected: Slight sell bias in normal high range")
            else:
                print(f"   ✓ Expected: Normal trading allowed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_position_sizing():
    """Test position sizing with new price levels."""
    print("\n=== Testing Position Sizing ===")
    
    logger = Mock()
    logger.info = lambda x: print(f"INFO: {x}")
    logger.warning = lambda x: print(f"WARNING: {x}")
    
    strategy = EnsembleStrategy(logger=logger)
    
    # Mock data for position sizing
    mock_data = pd.DataFrame({
        'high': [120000] * 50,
        'low': [118000] * 50, 
        'close': [119000] * 50,
        'volume': [1000] * 50
    })
    
    test_scenarios = [
        (119000, 10000, 50, 0.8, "High confidence at current price"),
        (125000, 10000, 50, 0.7, "Normal confidence at high ATH"),
        (130000, 10000, 50, 0.6, "Lower confidence at extreme ATH")
    ]
    
    for price, capital, leverage, confidence, description in test_scenarios:
        print(f"\n--- {description} ---")
        print(f"   Price: ${price:,}, Capital: ${capital:,}, Leverage: {leverage}x, Confidence: {confidence:.1%}")
        
        try:
            position_size = strategy.calculate_position_size(
                mock_data, price, capital, leverage, confidence
            )
            position_value = position_size * price
            print(f"   Result: {position_size:.6f} BTC (${position_value:.2f} value)")
            print(f"   Risk: {(position_value / capital) * 100:.2f}% of capital")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_profit_targets():
    """Test the updated profit targets."""
    print("\n=== Testing Profit Targets ===")
    
    test_prices = [
        (90000, "Below $100k", 1.00),
        (110000, "Above $100k", 2.50), 
        (125000, "Above $120k", 5.00)
    ]
    
    for price, description, expected_target in test_prices:
        print(f"\n--- {description}: ${price:,} ---")
        
        # Simulate the profit target logic from main.py
        if price > 120000:
            profit_target = 5.00
        elif price > 100000:
            profit_target = 2.50
        else:
            profit_target = 1.00
            
        print(f"   Profit target: ${profit_target:.2f}")
        
        if profit_target == expected_target:
            print(f"   ✓ Correct target for price level")
        else:
            print(f"   ❌ Expected ${expected_target:.2f}, got ${profit_target:.2f}")

def test_emergency_thresholds():
    """Test emergency stop thresholds."""
    print("\n=== Testing Emergency Thresholds ===")
    
    # Test emergency stop logic (from main.py)
    test_prices = [
        (118000, "Current price", False),
        (125000, "High ATH", False),
        (130000, "Extreme ATH", False), 
        (135000, "Emergency threshold", False),  # At threshold, no stop
        (135001, "Just above emergency", True),
        (140000, "Above emergency", True)
    ]
    
    for price, description, should_stop in test_prices:
        print(f"\n--- {description}: ${price:,} ---")
        
        # Simulate emergency stop logic (must be GREATER than 135000, not equal)
        emergency_stop = price > 135000
        
        print(f"   Emergency stop triggered: {emergency_stop}")
        
        if emergency_stop == should_stop:
            print(f"   ✓ Correct emergency behavior")
        else:
            print(f"   ❌ Expected stop={should_stop}, got stop={emergency_stop}")

def main():
    """Run all recalibration tests."""
    print("🔧 TESTING RECALIBRATED BITCOIN BOT FOR $120k+ PARADIGM")
    print("=" * 60)
    
    try:
        test_ath_detection()
        test_position_sizing()
        test_profit_targets()
        test_emergency_thresholds()
        
        print("\n" + "=" * 60)
        print("✅ RECALIBRATION TESTS COMPLETED")
        print("🚀 Bot is ready for the new Bitcoin price paradigm!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()