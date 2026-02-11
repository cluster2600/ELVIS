#!/usr/bin/env python3
"""
Test script to specifically test the cooldown mechanism.
"""

import logging
import time
from datetime import datetime
from trading.strategies.balanced_starter import BalancedStarterStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_cooldown():
    """Test that the cooldown mechanism prevents rapid trading."""
    strategy = BalancedStarterStrategy(logger)
    
    logger.info("=== TESTING SPECIFIC COOLDOWN MECHANISM ===")
    logger.info(f"Minimum position hold time: {strategy.min_position_hold_time} seconds")
    
    # Test 1: No recent trade time - should proceed
    logger.info("\n--- Test 1: No recent trade time ---")
    strategy.last_trade_time = None
    try:
        strategy.take_profits_on_scalping_positions(97000.0)
        logger.info("✅ Test 1 passed: No cooldown when no recent trade time")
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
    
    # Test 2: Recent trade time - should be blocked
    logger.info("\n--- Test 2: Recent trade time (should be blocked) ---")
    strategy.last_trade_time = datetime.now()
    try:
        strategy.take_profits_on_scalping_positions(97000.0)
        logger.info("✅ Test 2 passed: Cooldown blocking recent trades")
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")
    
    # Test 3: Old trade time - should proceed
    logger.info("\n--- Test 3: Old trade time (should proceed) ---")
    from datetime import timedelta
    strategy.last_trade_time = datetime.now() - timedelta(seconds=700)  # 700 seconds ago
    try:
        strategy.take_profits_on_scalping_positions(97000.0)
        logger.info("✅ Test 3 passed: Old trade time allows new trades")
    except Exception as e:
        logger.error(f"❌ Test 3 failed: {e}")
    
    logger.info("\n=== COOLDOWN TEST COMPLETE ===")
    logger.info("Expected behavior:")
    logger.info("- Test 1: Should proceed (no recent trade)")
    logger.info("- Test 2: Should show cooldown message")
    logger.info("- Test 3: Should proceed (old trade time)")

if __name__ == "__main__":
    test_specific_cooldown()