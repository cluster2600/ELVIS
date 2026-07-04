import logging
import time

from binance.client import Client

from config import API_CONFIG, TRADING_CONFIG
from core.metrics.performance_monitor import PerformanceMonitor
from utils.paper_trade_db import add_open_position, close_open_position, record_trade


class PaperBot:
    def __init__(self):
        self.client = Client(API_CONFIG["API_KEY"], API_CONFIG["API_SECRET"])
        self.pm = PerformanceMonitor()
        self.BINANCE_TAKER_FEE = 0.0004  # Taker fee rate

    def execute_trade(self, symbol, side, quantity, price):
        fee = quantity * price * self.BINANCE_TAKER_FEE  # Calculate fee
        pnl = 0.0  # PNL calculation would go here if needed
        record_trade(symbol, side, price, quantity, pnl, fee)  # Pass fee
        self.pm.add_trade(symbol, side, price, quantity, pnl, fee)
        if side == "BUY":
            add_open_position(symbol, price, quantity)
        elif side == "SELL":
            close_open_position(symbol)

    # Other methods remain unchanged
    def run(self):
        # Example run loop
        while True:
            # Simulate trades
            self.execute_trade("BTCUSDT", "BUY", 1.0, 50000.0)
            time.sleep(1)


if __name__ == "__main__":
    bot = PaperBot()
    bot.run()
