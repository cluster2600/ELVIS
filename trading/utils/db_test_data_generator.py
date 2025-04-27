from utils.paper_trade_db import init_db, record_trade, add_open_position
from datetime import datetime
import random

def generate_test_trades(n=10):
    symbols = ["BTCUSDT", "ETHUSDT"]
    sides = ["buy", "sell"]

    for _ in range(n):
        symbol = random.choice(symbols)
        side = random.choice(sides)
        price = round(random.uniform(29000, 31000), 2)
        quantity = round(random.uniform(0.001, 0.01), 6)
        pnl = round(random.uniform(-10, 10), 2)

        record_trade(symbol, side, price, quantity, pnl)

def generate_test_open_positions(n=3):
    symbols = ["BTCUSDT", "ETHUSDT"]

    for _ in range(n):
        symbol = random.choice(symbols)
        entry_price = round(random.uniform(29000, 31000), 2)
        quantity = round(random.uniform(0.001, 0.02), 6)
        leverage = random.choice([1, 2, 5])

        add_open_position(symbol, entry_price, quantity, leverage)

if __name__ == "__main__":
    print("Initializing DB...")
    init_db()

    print("Generating test trades...")
    generate_test_trades(n=25)

    print("Generating test open positions...")
    generate_test_open_positions(n=5)

    print("Test data generation complete!")