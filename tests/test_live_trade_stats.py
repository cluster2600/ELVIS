#!/usr/bin/env python3
"""
Test live trade statistics from database
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from utils.paper_trade_db import get_all_trades, init_db

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_live_trade_statistics():
    """Test live trade statistics calculation"""
    logger.info("=== Testing Live Trade Statistics ===")

    # Initialize database
    init_db()

    # Get recent trades
    trades_raw = get_all_trades(limit=100)
    logger.info(f"Found {len(trades_raw)} trades in database")

    if not trades_raw:
        logger.warning("No trades found in database")
        return

    # Calculate statistics
    wins = 0
    losses = 0
    total_pnl = 0.0
    win_pnls = []
    loss_pnls = []
    buy_trades = 0
    sell_trades = 0

    logger.info("Processing trades...")
    for i, trade in enumerate(trades_raw[:10]):  # Show first 10 trades
        if len(trade) >= 7:
            trade_id, timestamp, symbol, side, price, quantity, pnl, fee = trade[:8]
            total_pnl += float(pnl)

            logger.info(
                f"Trade {i+1}: {side} {quantity} {symbol} @ ${price} | PnL: ${pnl} | {timestamp}"
            )

            if side.upper() == "BUY":
                buy_trades += 1
            elif side.upper() == "SELL":
                sell_trades += 1

            pnl_val = float(pnl)
            if pnl_val > 0:
                wins += 1
                win_pnls.append(pnl_val)
            elif pnl_val < 0:
                losses += 1
                loss_pnls.append(pnl_val)

    # Calculate final statistics
    total_trades = len(trades_raw)
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    logger.info("\n=== TRADE STATISTICS SUMMARY ===")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"BUY Trades: {buy_trades}")
    logger.info(f"SELL Trades: {sell_trades}")
    logger.info(f"Winning Trades: {wins}")
    logger.info(f"Losing Trades: {losses}")
    logger.info(f"Neutral Trades: {total_trades - wins - losses}")
    logger.info(f"Win Rate: {win_rate:.1f}%")
    logger.info(f"Average Win: ${avg_win:.2f}")
    logger.info(f"Average Loss: ${avg_loss:.2f}")
    logger.info(f"Total P&L: ${total_pnl:.2f}")

    # Test what would be displayed in dashboard
    logger.info("\n=== DASHBOARD DISPLAY FORMAT ===")
    logger.info(f"Trades: {total_trades} | Win Rate: {win_rate:.1f}%")
    logger.info(f"Wins: {wins} | Losses: {losses}")
    logger.info(f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
    logger.info(f"Total PnL: ${total_pnl:.2f}")


def test_trade_analyzer_comparison():
    """Test comparison with TradeAnalyzer"""
    logger.info("\n=== Testing TradeAnalyzer Comparison ===")

    try:
        from utils.paper_trade_db import get_all_trades
        from utils.trade_analyzer import TradeAnalyzer

        # Get live trades
        trades_raw = get_all_trades(limit=100)

        # Convert to TradeAnalyzer format
        live_trades = []
        for trade in trades_raw:
            if len(trade) >= 7:
                live_trades.append(
                    {
                        "symbol": trade[2],
                        "side": trade[3].lower(),
                        "quantity": float(trade[5]),
                        "price": float(trade[4]),
                        "pnl": float(trade[6]),
                        "timestamp": trade[1],
                    }
                )

        # Create analyzer
        analyzer = TradeAnalyzer(live_trades)

        # Get statistics
        win_loss = analyzer.get_win_loss_distribution()
        avg_pnl = analyzer.get_average_pnl()

        logger.info("TradeAnalyzer Results:")
        logger.info(f"Win/Loss Distribution: {win_loss}")
        logger.info(f"Average PnL: {avg_pnl}")

    except Exception as e:
        logger.error(f"TradeAnalyzer test failed: {e}")


if __name__ == "__main__":
    logger.info("Starting Live Trade Statistics Tests...")

    # Test 1: Live trade statistics
    test_live_trade_statistics()

    # Test 2: TradeAnalyzer comparison
    test_trade_analyzer_comparison()

    logger.info("Live trade statistics tests completed!")
