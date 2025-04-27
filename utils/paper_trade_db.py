import psycopg2
import os
from datetime import datetime
from dotenv import load_dotenv

# Load .env from parent folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def get_conn():
    try:
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            port=os.environ.get('DB_PORT', 5432),
            user=os.environ.get('DB_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD', ''),
            dbname=os.environ.get('DB_NAME', 'trading_bot')
        )
        return conn
    except psycopg2.Error as e:
        print(f"[ERROR] Database connection failed: {e}")
        raise

def init_db():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP,
                        symbol TEXT,
                        side TEXT,
                        price REAL,
                        quantity REAL,
                        pnl REAL,
                        fee REAL
                    )
                """)
                c.execute("""
                    CREATE TABLE IF NOT EXISTS open_positions (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT,
                        entry_price REAL,
                        quantity REAL,
                        leverage REAL,
                        entry_time TIMESTAMP
                    )
                """)
            conn.commit()
            print("Tables initialized successfully.")
    except psycopg2.Error as e:
        print(f"[ERROR] Error initializing database: {e}")

def record_trade(symbol, side, price, quantity, pnl=0.0, fee=None, entry_time=None, leverage=1.0):
    try:
        if fee is None:
            fee_rate = 0.00075 if os.environ.get('PAY_FEES_WITH_BNB', 'false').lower() == 'true' else 0.001
            trading_fee = price * quantity * fee_rate

            funding_rate = float(os.environ.get('DEFAULT_FUNDING_RATE', 0.0001))
            if entry_time:
                holding_seconds = (datetime.utcnow() - entry_time).total_seconds()
                holding_hours = holding_seconds / 3600
                position_value = price * quantity
                leverage_fee = position_value * funding_rate * (holding_hours / 8) * leverage
            else:
                leverage_fee = 0.0

            fee = trading_fee + leverage_fee

        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO trades (timestamp, symbol, side, price, quantity, pnl, fee) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (datetime.utcnow(), symbol, side, price, quantity, pnl, fee)
                )
            conn.commit()
            print(f"Trade recorded: {symbol} {side} {quantity} @ {price} | Trading Fee: {trading_fee:.4f} | Leverage Fee: {leverage_fee:.4f}")
    except psycopg2.Error as e:
        print(f"[ERROR] Error recording trade: {e}")

def add_open_position(symbol, entry_price, quantity, leverage=1.0):
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO open_positions (symbol, entry_price, quantity, leverage, entry_time) VALUES (%s, %s, %s, %s, %s)",
                    (symbol, entry_price, quantity, leverage, datetime.utcnow())
                )
            conn.commit()
            print(f"Open position added: {symbol} {quantity} @ {entry_price}")
    except psycopg2.Error as e:
        print(f"[ERROR] Error adding open position: {e}")

def close_open_position(symbol):
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("DELETE FROM open_positions WHERE symbol = %s", (symbol,))
            conn.commit()
            print(f"Closed positions for symbol: {symbol}")
    except psycopg2.Error as e:
        print(f"[ERROR] Error closing position: {e}")

def get_open_positions():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("SELECT symbol, entry_price, quantity, leverage, entry_time FROM open_positions")
                positions = c.fetchall()
            return positions
    except psycopg2.Error as e:
        print(f"[ERROR] Error fetching open positions: {e}")
        return []

def get_all_trades(limit=100):
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("""
                    SELECT timestamp, symbol, side, price, quantity, pnl, fee
                    FROM trades
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
                trades = c.fetchall()
            return trades[::-1]
    except psycopg2.Error as e:
        print(f"[ERROR] Error fetching trades: {e}")
        return []

def get_trade_count():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("SELECT COUNT(*) FROM trades")
                count = c.fetchone()[0]
            return count
    except psycopg2.Error as e:
        print(f"[ERROR] Error counting trades: {e}")
        return 0

def get_total_fees():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("SELECT COALESCE(SUM(fee), 0) FROM trades")
                total_fees = c.fetchone()[0]
            return total_fees
    except psycopg2.Error as e:
        print(f"[ERROR] Error fetching total fees: {e}")
        return 0

def check_liquidations(current_price_data):
    margin_threshold = float(os.environ.get('LIQUIDATION_MARGIN', 0.9))
    positions = get_open_positions()
    for p in positions:
        symbol = p[0]
        entry_price = p[1]
        quantity = p[2]
        leverage = p[3]
        entry_time = p[4]

        current_price = current_price_data.get(symbol, entry_price)
        pnl = (current_price - entry_price) * quantity * leverage
        if pnl <= -abs(entry_price * quantity * margin_threshold):
            print(f"[LIQUIDATION] {symbol} position liquidated.")
            record_trade(symbol, "LIQUIDATION", current_price, quantity, pnl, entry_time=entry_time, leverage=leverage)
            close_open_position(symbol)

def init():
    init_db()