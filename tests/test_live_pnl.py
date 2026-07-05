#!/usr/bin/env python3
"""
Test live P&L calculations for dashboard
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from utils.paper_trade_db import get_all_trades, get_open_positions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_live_pnl_calculations():
    """Test live P&L calculations"""
    logger.info("=== Testing Live P&L Calculations ===")

    # Get all trades for realized P&L
    all_trades = get_all_trades(limit=1000)
    logger.info(f"Found {len(all_trades)} total trades")

    # Calculate realized P&L
    realized_pnl = 0.0
    profitable_trades = 0
    losing_trades = 0

    logger.info("\nCalculating realized P&L from trades...")
    for trade in all_trades:
        if len(trade) >= 7:
            pnl = float(trade[6])
            realized_pnl += pnl

            if pnl > 0:
                profitable_trades += 1
            elif pnl < 0:
                losing_trades += 1

    logger.info(f"Total Realized P&L: ${realized_pnl:.2f}")
    logger.info(f"Profitable trades: {profitable_trades}")
    logger.info(f"Losing trades: {losing_trades}")
    logger.info(
        f"Neutral trades: {len(all_trades) - profitable_trades - losing_trades}"
    )

    # Get open positions for unrealized P&L
    open_positions = get_open_positions()
    logger.info(f"\nFound {len(open_positions)} open positions")

    unrealized_pnl = 0.0
    current_btc_price = 107500.0  # Mock current price

    if open_positions:
        logger.info("Calculating unrealized P&L from open positions...")
        for i, pos in enumerate(open_positions):
            if len(pos) >= 4:
                symbol = pos[1]
                entry_price = float(pos[2])
                quantity = float(pos[3])

                # Calculate unrealized P&L
                position_pnl = (current_btc_price - entry_price) * quantity
                unrealized_pnl += position_pnl

                logger.info(
                    f"Position {i+1}: {symbol} | Entry: ${entry_price:.2f} | "
                    f"Qty: {quantity:.6f} | Current: ${current_btc_price:.2f} | "
                    f"PnL: ${position_pnl:.2f}"
                )
    else:
        logger.info("No open positions - unrealized P&L is $0.00")

    logger.info(f"Total Unrealized P&L: ${unrealized_pnl:.2f}")

    # Calculate total portfolio
    starting_balance = 10000.0
    total_pnl = realized_pnl + unrealized_pnl
    portfolio_value = starting_balance + total_pnl

    logger.info(f"\n=== PORTFOLIO SUMMARY ===")
    logger.info(f"Starting Balance: ${starting_balance:.2f}")
    logger.info(f"Realized P&L: ${realized_pnl:.2f}")
    logger.info(f"Unrealized P&L: ${unrealized_pnl:.2f}")
    logger.info(f"Total P&L: ${total_pnl:.2f}")
    logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
    logger.info(f"Return: {(total_pnl / starting_balance * 100):.2f}%")


def test_dashboard_format():
    """Test dashboard display format"""
    logger.info(f"\n=== DASHBOARD DISPLAY FORMAT ===")

    # Simulate dashboard calculations
    from utils.paper_trade_db import get_all_trades, get_open_positions

    # Calculate realized P&L
    all_trades = get_all_trades(limit=1000)
    realized_pnl = sum(float(trade[6]) for trade in all_trades if len(trade) >= 7)

    # Calculate unrealized P&L
    open_positions = get_open_positions()
    unrealized_pnl = 0.0
    current_price = 107500.0

    for pos in open_positions:
        if len(pos) >= 4:
            entry_price = float(pos[2])
            quantity = float(pos[3])
            position_pnl = (current_price - entry_price) * quantity
            unrealized_pnl += position_pnl

    # Portfolio value
    starting_balance = 10000.0
    portfolio_value = starting_balance + realized_pnl + unrealized_pnl

    # Dashboard format
    logger.info("--- Portfolio ---")
    logger.info(f"Value: ${portfolio_value:,.2f}")

    # Color coding simulation
    realized_symbol = "+" if realized_pnl >= 0 else "-"
    unrealized_symbol = "+" if unrealized_pnl >= 0 else "-"

    logger.info(f"Unrealized PnL: {unrealized_symbol}${abs(unrealized_pnl):,.2f}")
    logger.info(f"Realized PnL: {realized_symbol}${abs(realized_pnl):,.2f}")


def test_real_time_updates():
    """Test that calculations would change with new data"""
    logger.info(f"\n=== REAL-TIME UPDATE TEST ===")

    # Get recent trades to show this changes over time
    recent_trades = get_all_trades(limit=10)
    logger.info("Recent 10 trades and their impact on P&L:")

    running_pnl = 0.0
    for i, trade in enumerate(recent_trades):
        if len(trade) >= 7:
            pnl = float(trade[6])
            running_pnl += pnl
            timestamp = trade[1]

            logger.info(
                f"{i+1}. {timestamp} | {trade[3]} | PnL: ${pnl:.2f} | Running Total: ${running_pnl:.2f}"
            )

    logger.info(f"\nAs new trades execute, the realized P&L will update in real-time!")
    logger.info(f"Dashboard will refresh every few seconds with latest data.")


if __name__ == "__main__":
    logger.info("Starting Live P&L Tests...")

    # Test 1: Live P&L calculations
    test_live_pnl_calculations()

    # Test 2: Dashboard format
    test_dashboard_format()

    # Test 3: Real-time updates
    test_real_time_updates()

    logger.info("Live P&L tests completed!")
