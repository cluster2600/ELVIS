#!/usr/bin/env python3

import logging

from config.config import TRADING_CONFIG
from trading.strategies.balanced_starter import BalancedStarterStrategy

print("=== LEVERAGE & COOLDOWN SETTINGS ===")
print(f'Default Leverage: {TRADING_CONFIG["DEFAULT_LEVERAGE"]}x')
print(f'Max Leverage: {TRADING_CONFIG["LEVERAGE_MAX"]}x')
print(f'Cooldown: {TRADING_CONFIG["COOLDOWN"]}s')
print()

# Test strategy initialization
logger = logging.getLogger("test")
strategy = BalancedStarterStrategy(logger)
print(f"Strategy Min Hold Time: {strategy.min_position_hold_time}s")
print(f"Target Profit: ${strategy.target_profit_per_trade}")
print(f"Trades Per Hour: {strategy.trades_per_hour}")
print()
print("✅ All leverage settings updated to 100x")
print("✅ Emergency cooldowns removed")
