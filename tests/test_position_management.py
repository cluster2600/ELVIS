#!/usr/bin/env python3
"""
Test position management and sizing
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

import numpy as np
import pandas as pd

from trading.execution.binance_executor import BinanceExecutor
from trading.strategies.ensemble_strategy import EnsembleStrategy
from utils.paper_trade_db import (
    add_open_position,
    get_open_positions,
    init_db,
    record_trade,
)
from utils.price_fetcher import PriceFetcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_position_sizing():
    """Test position sizing calculation"""
    logger.info("=== Testing Position Sizing ===")

    # Initialize strategy
    strategy = EnsembleStrategy(logger=logger)

    # Create mock data
    mock_data = pd.DataFrame(
        {
            "open": [97000] * 50,
            "high": [97500] * 50,
            "low": [96500] * 50,
            "close": [97000] * 50,
            "volume": [1000] * 50,
        }
    )

    current_price = 97000.0
    available_capital = 10000.0

    # Test position sizing
    position_size = strategy.calculate_position_size(
        mock_data, current_price, available_capital
    )

    logger.info(f"Position Size Calculation:")
    logger.info(f"  Current Price: ${current_price:.2f}")
    logger.info(f"  Available Capital: ${available_capital:.2f}")
    logger.info(f"  Calculated Position Size: {position_size:.6f} BTC")
    logger.info(f"  Position Value: ${position_size * current_price:.2f}")
    logger.info(
        f"  % of Capital: {(position_size * current_price / available_capital) * 100:.2f}%"
    )

    return position_size


def test_manual_positions():
    """Manually create some test positions"""
    logger.info("=== Creating Test Positions ===")

    # Initialize database
    init_db()

    # Add some test positions
    test_positions = [
        ("BTCUSDT", 97000.0, 0.001, 1.0),
        ("BTCUSDT", 96800.0, 0.0015, 1.0),
        ("BTCUSDT", 97200.0, 0.0008, 1.0),
    ]

    for symbol, entry_price, quantity, leverage in test_positions:
        try:
            add_open_position(symbol, entry_price, quantity, leverage)
            logger.info(f"Added position: {symbol} {quantity:.6f} @ ${entry_price:.2f}")
        except Exception as e:
            logger.error(f"Failed to add position: {e}")

    # Check positions
    positions = get_open_positions()
    logger.info(f"Current open positions: {len(positions)}")
    for pos in positions:
        logger.info(f"  {pos[1]} | Entry: ${pos[2]:.2f} | Qty: {pos[3]:.6f}")


def test_position_display():
    """Test how positions are displayed in dashboard format"""
    logger.info("=== Testing Position Display Format ===")

    positions_raw = get_open_positions()

    # Convert to dashboard format (similar to main.py)
    open_positions = []
    for pos in positions_raw:
        if len(pos) >= 4:
            # Mock current price (in real scenario this comes from price fetcher)
            current_price = 97100.0  # Slight profit
            pnl = (current_price - float(pos[2])) * float(pos[3])

            open_positions.append(
                {
                    "symbol": pos[1],
                    "size": float(pos[3]),
                    "entry_price": float(pos[2]),
                    "pnl": pnl,
                    "entry_time": pos[5] if len(pos) > 5 else "N/A",
                }
            )

    logger.info("Dashboard Format Positions:")
    for i, pos in enumerate(open_positions):
        symbol = pos["symbol"]
        size = pos["size"]
        entry_price = pos["entry_price"]
        pnl = pos["pnl"]
        side = "LONG" if size > 0 else "SHORT"

        logger.info(
            f"  {i+1}. {symbol} {side} {abs(size):.6f} @ ${entry_price:.2f} | PnL: ${pnl:.2f}"
        )

    return open_positions


def test_executor_positions():
    """Test executor position management"""
    logger.info("=== Testing Executor Position Management ===")

    executor = BinanceExecutor(logger=logger, is_testnet=True)
    executor._init_paper_trading_db()

    # Test balance
    balance = executor.get_balance()
    total_balance = executor.get_account_balance()

    logger.info(f"Current Balance: {balance}")
    logger.info(f"Total Account Value: ${total_balance:.2f}")

    # Test position size calculation
    current_price = 97000.0

    # Mock a trade execution (just to see position sizing in action)
    logger.info(f"Testing position sizing with price ${current_price:.2f}")

    # Create some test data for strategy
    mock_data = pd.DataFrame(
        {
            "open": [97000] * 20,
            "high": [97500] * 20,
            "low": [96500] * 20,
            "close": [97000] * 20,
            "volume": [1000] * 20,
        }
    )

    strategy = EnsembleStrategy(logger=logger)
    position_size = strategy.calculate_position_size(
        mock_data, current_price, total_balance
    )

    logger.info(f"Recommended position size: {position_size:.6f} BTC")
    logger.info(f"Position value: ${position_size * current_price:.2f}")


if __name__ == "__main__":
    logger.info("Starting Position Management Tests...")

    # Test 1: Position sizing calculation
    test_position_sizing()
    print()

    # Test 2: Manual position creation
    test_manual_positions()
    print()

    # Test 3: Position display format
    test_position_display()
    print()

    # Test 4: Executor position management
    test_executor_positions()

    logger.info("Position management tests completed!")
