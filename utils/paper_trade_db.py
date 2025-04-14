import sqlite3
from datetime import datetime

DB_PATH = "paper_trades.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            price REAL,
            quantity REAL,
            pnl REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS open_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            entry_price REAL,
            quantity REAL,
            entry_time TEXT
        )
    """)
    conn.commit()
    conn.close()

def record_trade(symbol, side, price, quantity, pnl=0.0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO trades (timestamp, symbol, side, price, quantity, pnl) VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.utcnow().isoformat(), symbol, side, price, quantity, pnl))
    conn.commit()
    conn.close()

def add_open_position(symbol, entry_price, quantity):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO open_positions (symbol, entry_price, quantity, entry_time) VALUES (?, ?, ?, ?)",
              (symbol, entry_price, quantity, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def close_open_position(symbol):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))
    conn.commit()
    conn.close()

def get_open_positions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT symbol, entry_price, quantity, entry_time FROM open_positions")
    positions = c.fetchall()
    conn.close()
    return positions

def get_all_trades():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, symbol, side, price, quantity, pnl FROM trades")
    trades = c.fetchall()
    conn.close()
    return trades

def calculate_unrealized_pnl(symbol, current_price):
    positions = get_open_positions()
    for pos in positions:
        if pos[0] == symbol:
            entry_price = pos[1]
            quantity = pos[2]
            return (current_price - entry_price) * quantity
    return 0.0
