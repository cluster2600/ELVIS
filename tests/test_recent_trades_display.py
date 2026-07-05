#!/usr/bin/env python3
"""
Test recent trades display for dashboard
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from utils.paper_trade_db import get_all_trades

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_recent_trades_display():
    """Test recent trades display logic"""
    logger.info("=== Testing Recent Trades Display ===")

    # Get recent trades from database
    all_recent_trades = get_all_trades(limit=50)
    logger.info(f"Found {len(all_recent_trades)} recent trades")

    # Separate profitable trades and recent activity
    profitable_trades = []
    recent_activity = []

    for trade in all_recent_trades:
        if len(trade) >= 7:
            trade_data = {
                "side": trade[3],
                "symbol": trade[2],
                "price": float(trade[4]),
                "quantity": float(trade[5]),
                "pnl": float(trade[6]),
                "timestamp": trade[1],
            }

            # Collect trades with actual P&L
            if trade_data["pnl"] != 0.0:
                profitable_trades.append(trade_data)

            # Also collect recent activity (last 10 trades)
            if len(recent_activity) < 10:
                recent_activity.append(trade_data)

    logger.info(f"Found {len(profitable_trades)} trades with actual P&L")
    logger.info(f"Found {len(recent_activity)} recent trades")

    # Show what would be displayed
    display_trades = profitable_trades[:3] if profitable_trades else recent_activity[:3]

    logger.info(f"\n=== TRADES WITH ACTUAL P&L ===")
    if profitable_trades:
        for i, trade in enumerate(profitable_trades[:5]):  # Show top 5
            side = trade["side"]
            symbol = trade["symbol"]
            price = trade["price"]
            quantity = trade["quantity"]
            pnl = trade["pnl"]
            timestamp = trade["timestamp"]

            pnl_symbol = "+" if pnl >= 0 else ""
            logger.info(
                f"{i+1}. {timestamp} | {side} {quantity:.4f} {symbol} @ ${price:.2f} | PnL: {pnl_symbol}${pnl:.2f}"
            )
    else:
        logger.info("No trades with P&L found")

    logger.info(f"\n=== RECENT ACTIVITY (fallback) ===")
    for i, trade in enumerate(recent_activity[:5]):
        side = trade["side"]
        symbol = trade["symbol"]
        price = trade["price"]
        quantity = trade["quantity"]
        pnl = trade["pnl"]
        trade_value = price * quantity
        timestamp = trade["timestamp"]

        logger.info(
            f"{i+1}. {timestamp} | {side} {quantity:.4f} {symbol} @ ${price:.2f} | Val: ${trade_value:.0f}"
        )


def test_dashboard_format():
    """Test dashboard display format"""
    logger.info(f"\n=== DASHBOARD DISPLAY FORMAT ===")

    # Simulate dashboard logic
    all_recent_trades = get_all_trades(limit=50)

    profitable_trades = []
    recent_activity = []

    for trade in all_recent_trades:
        if len(trade) >= 7:
            trade_data = {
                "side": trade[3],
                "symbol": trade[2],
                "price": float(trade[4]),
                "quantity": float(trade[5]),
                "pnl": float(trade[6]),
                "timestamp": trade[1],
            }

            if trade_data["pnl"] != 0.0:
                profitable_trades.append(trade_data)

            if len(recent_activity) < 10:
                recent_activity.append(trade_data)

    # Show what dashboard would display
    display_trades = profitable_trades[:3] if profitable_trades else recent_activity[:3]

    logger.info("--- Recent Trades ---")

    if display_trades:
        for i, trade in enumerate(display_trades):
            side = trade["side"]
            symbol = trade["symbol"]
            price = trade["price"]
            quantity = trade["quantity"]
            pnl = trade["pnl"]

            # Simulate dashboard display
            trade_info = f"{side} {quantity:.4f} {symbol} @ ${price:.2f}"

            if pnl != 0.0:
                pnl_display = f"PnL: ${pnl:.2f}"
                logger.info(f"{trade_info:<30} {pnl_display}")
            else:
                trade_value = price * quantity
                val_display = f"Val: ${trade_value:.0f}"
                logger.info(f"{trade_info:<30} {val_display}")
    else:
        logger.info("No recent trades")


def analyze_trade_patterns():
    """Analyze trade patterns for better display"""
    logger.info(f"\n=== TRADE PATTERN ANALYSIS ===")

    all_trades = get_all_trades(limit=100)

    buy_trades = 0
    sell_trades = 0
    profitable_count = 0
    losing_count = 0
    neutral_count = 0

    for trade in all_trades:
        if len(trade) >= 7:
            side = trade[3]
            pnl = float(trade[6])

            if side == "BUY":
                buy_trades += 1
            elif side == "SELL":
                sell_trades += 1

            if pnl > 0:
                profitable_count += 1
            elif pnl < 0:
                losing_count += 1
            else:
                neutral_count += 1

    logger.info(f"Recent 100 trades breakdown:")
    logger.info(f"  BUY trades: {buy_trades}")
    logger.info(f"  SELL trades: {sell_trades}")
    logger.info(f"  Profitable trades: {profitable_count}")
    logger.info(f"  Losing trades: {losing_count}")
    logger.info(f"  Neutral trades (0.00 P&L): {neutral_count}")

    logger.info(f"\nDashboard Strategy:")
    if profitable_count > 0 or losing_count > 0:
        logger.info("✅ Will show trades with actual P&L (more meaningful)")
    else:
        logger.info("⚡ Will show recent activity with trade values (fallback)")


if __name__ == "__main__":
    logger.info("Starting Recent Trades Display Tests...")

    # Test 1: Recent trades logic
    test_recent_trades_display()

    # Test 2: Dashboard format
    test_dashboard_format()

    # Test 3: Trade pattern analysis
    analyze_trade_patterns()

    logger.info("Recent trades display tests completed!")
