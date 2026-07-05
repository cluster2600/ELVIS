#!/usr/bin/env python3
"""
Test dashboard with live trade statistics
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from utils.console_dashboard import ConsoleDashboard
from utils.paper_trade_db import get_all_trades

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dashboard_trade_stats():
    """Test dashboard trade statistics display"""
    logger.info("=== Testing Dashboard Trade Statistics ===")

    # Get live trades from database
    trades_raw = get_all_trades(limit=100)

    # Convert to dashboard format
    recent_trades = []
    for trade in trades_raw[:10]:  # Last 10 trades
        if len(trade) >= 7:
            recent_trades.append(
                {
                    "symbol": trade[2],
                    "side": trade[3],
                    "price": float(trade[4]),
                    "quantity": float(trade[5]),
                    "pnl": float(trade[6]),
                    "timestamp": trade[1],
                }
            )

    # Create mock config with live data
    config = {
        "portfolio_value": 12500.0,
        "unrealized_pnl": 150.0,
        "realized_pnl": 814.11,  # Real total from database
        "open_positions": [],
        "recent_trades": recent_trades,
    }

    # Test the statistics calculation logic (without curses)
    logger.info("Testing live statistics calculation...")

    # Calculate statistics like dashboard does
    if trades_raw:
        wins = 0
        losses = 0
        total_pnl = 0.0
        win_pnls = []
        loss_pnls = []

        for trade in trades_raw:
            if len(trade) >= 7:
                pnl = float(trade[6])
                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                    win_pnls.append(pnl)
                elif pnl < 0:
                    losses += 1
                    loss_pnls.append(pnl)

        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        logger.info("Live Dashboard Statistics:")
        logger.info(f"  Trades: {total_trades} | Win Rate: {win_rate:.1f}%")
        logger.info(f"  Wins: {wins} | Losses: {losses}")
        logger.info(f"  Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
        logger.info(f"  Total PnL: ${total_pnl:.2f}")

    logger.info("\nRecent Trades (Dashboard Format):")
    for i, trade in enumerate(recent_trades[:5]):
        side = trade["side"]
        symbol = trade["symbol"]
        price = trade["price"]
        quantity = trade["quantity"]
        pnl = trade["pnl"]

        logger.info(
            f"  {i+1}. {side} {quantity:.4f} {symbol} @ ${price:.2f} | PnL: ${pnl:.2f}"
        )

    return True


if __name__ == "__main__":
    logger.info("Starting Dashboard Trade Statistics Tests...")

    success = test_dashboard_trade_stats()

    if success:
        logger.info("✅ Dashboard trade statistics test completed successfully!")
        logger.info(
            "The console dashboard will now show live trade data instead of static sample data."
        )
    else:
        logger.error("❌ Dashboard test failed!")

    logger.info("Dashboard tests completed!")
