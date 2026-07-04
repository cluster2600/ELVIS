#!/usr/bin/env python3
"""
Test the candlestick dashboard with real data
"""

import curses
import time

import numpy as np
import pandas as pd

from utils.console_dashboard import ConsoleDashboard
from utils.logger_config import setup_logging
from utils.price_fetcher import PriceFetcher


def create_mock_ohlc_data():
    """Create realistic OHLC data for testing"""
    np.random.seed(42)
    base_price = 105000.0
    data = []

    for i in range(50):
        open_price = base_price
        change_pct = np.random.normal(0, 0.003)  # 0.3% volatility
        close_price = open_price * (1 + change_pct)

        # High and low based on volatility
        volatility = abs(change_pct) + np.random.uniform(0.002, 0.008)
        high_price = max(open_price, close_price) * (1 + volatility / 2)
        low_price = min(open_price, close_price) * (1 - volatility / 2)

        volume = np.random.uniform(50, 200)

        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )
        base_price = close_price

    return pd.DataFrame(data)


def test_dashboard(stdscr):
    """Test dashboard with candlestick chart"""

    # Setup logging
    logger = setup_logging("CandlestickTest", log_level="INFO")

    # Create mock OHLC data
    ohlc_data = create_mock_ohlc_data()
    current_price = ohlc_data["close"].iloc[-1]

    # Create price fetcher (optional - for real data)
    try:
        price_fetcher = PriceFetcher(logger)
    except:
        price_fetcher = None

    # Create dashboard config with OHLC data
    config = {
        "portfolio_value": 10520.30,
        "unrealized_pnl": 150.10,
        "realized_pnl": 450.20,
        "current_price": current_price,
        "ohlc_data": ohlc_data,
        "indicators": {"rsi": 45.6, "macd": -6.74, "sma_20": 105191.87},
        "open_positions": [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "size": 0.001902,
                "entry_price": 105000,
                "current_price": current_price,
                "pnl": 15.98,
            }
        ],
        "recent_trades": [
            {
                "timestamp": "14:03:45",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "price": 105173.66,
                "quantity": 0.001902,
            }
        ],
    }

    # Create dashboard
    dashboard = ConsoleDashboard(
        config=config, logger=logger, price_fetcher=price_fetcher
    )

    # Add some live trading messages
    dashboard.add_log_message("🚀 ELVIS Trading Bot Started")
    dashboard.add_log_message("💰 Portfolio value: $10,520.30")
    dashboard.add_log_message("📊 Fetching market data for BTCUSDT...")
    dashboard.add_log_message("📈 BTC Price: $105,173.66 (+0.2%)")
    dashboard.add_log_message("🎯 Signal generated: SELL with confidence 0.70")
    dashboard.add_log_message(
        "💸 [PAPER TRADE] Executing SELL: 0.001902 BTCUSDT at $105,173.66"
    )
    dashboard.add_log_message("✅ [PAPER TRADE] SELL order completed successfully")
    dashboard.add_log_message("📝 Trade recorded in database with PnL: $15.98")
    dashboard.add_log_message("🔄 Waiting for next trading opportunity...")

    # Run dashboard
    try:
        dashboard.run(stdscr)
    except KeyboardInterrupt:
        logger.info("Dashboard test stopped by user")


if __name__ == "__main__":
    print("🔥 Testing Candlestick Dashboard")
    print("Press 'q' to quit the dashboard")
    print("Launching in 3 seconds...")
    time.sleep(3)

    try:
        curses.wrapper(test_dashboard)
    except Exception as e:
        print(f"Error running dashboard: {e}")
        print("Make sure you're running in a proper terminal with color support")
