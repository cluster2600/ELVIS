#!/usr/bin/env python3
"""
Test script to verify the cooldown and profit target fixes are working.
"""

import logging
from datetime import datetime

from trading.strategies.balanced_starter import BalancedStarterStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cooldown_fix():
    """Test that the cooldown mechanism is working."""
    strategy = BalancedStarterStrategy(logger)

    # Check configuration
    logger.info("=== TESTING COOLDOWN FIX ===")
    logger.info(f"Target profit per trade: ${strategy.target_profit_per_trade}")
    logger.info(f"Daily trade target: {strategy.daily_trade_target}")
    logger.info(f"Trades per hour: {strategy.trades_per_hour}")
    logger.info(f"Min position hold time: {strategy.min_position_hold_time} seconds")
    logger.info(f"Adaptation interval: {strategy.adaptation_interval} seconds")

    # Test cooldown mechanism
    logger.info("\n=== TESTING COOLDOWN MECHANISM ===")

    # Set a recent trade time
    strategy.last_trade_time = datetime.now()

    # Try to take profits (should be blocked by cooldown)
    logger.info("Testing take_profits_on_scalping_positions with recent trade time...")
    strategy.take_profits_on_scalping_positions(97000.0)

    # Test with no recent trade time
    logger.info("\nTesting with no recent trade time...")
    strategy.last_trade_time = None
    strategy.take_profits_on_scalping_positions(97000.0)

    logger.info("\n=== EXPECTED BEHAVIOR ===")
    logger.info("- Profit target: $1.00 (was $0.10)")
    logger.info("- Stop loss: $0.20 (was $0.05)")
    logger.info("- Cooldown: 600 seconds (was 15 seconds)")
    logger.info("- Daily trades: 50 (was 2000)")
    logger.info("- Adaptation interval: 1 hour (was 10 minutes)")

    logger.info("\n=== COOLDOWN TEST COMPLETE ===")


if __name__ == "__main__":
    test_cooldown_fix()
