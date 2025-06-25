import psycopg2
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def get_conn():
    try:
        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', os.environ.get('DB_HOST', 'localhost')),
            port=os.environ.get('POSTGRES_PORT', os.environ.get('DB_PORT', 5432)),
            user=os.environ.get('POSTGRES_USER', os.environ.get('DB_USER', 'postgres')),
            password=os.environ.get('POSTGRES_PASSWORD', os.environ.get('DB_PASSWORD', '')),
            dbname=os.environ.get('POSTGRES_DBNAME', os.environ.get('DB_NAME', 'trading_bot')),
            connect_timeout=5  # 5 second timeout
        )
        # Set the search path to use the 'np' schema
        with conn.cursor() as c:
            c.execute("SET search_path TO np, public;")
        conn.commit()
        return conn
    except psycopg2.OperationalError as e:
        print(f"[WARNING] Could not connect to PostgreSQL: {e}")
        print("[INFO] Paper trading will continue without database persistence")
        return None

def init_db():
    conn = get_conn()
    if conn is None:
        return
    try:
        with conn.cursor() as c:
            # Try to grant permissions but don't fail if it doesn't work
            try:
                c.execute("GRANT USAGE, CREATE ON SCHEMA np TO CURRENT_USER;")
                conn.commit()
            except Exception as e:
                print(f"[INFO] Could not grant permissions, trying to continue: {e}")
                conn.rollback()  # Rollback the failed transaction
            
            # Create tables in the np schema
            c.execute("""
                CREATE TABLE IF NOT EXISTS np.trades (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    quantity REAL,
                    pnl REAL,
                    fee REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS np.open_positions (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT,
                    entry_price REAL,
                    quantity REAL,
                    leverage REAL,
                    entry_time TIMESTAMP DEFAULT NOW()
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS np.liquidations (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    symbol TEXT,
                    entry_price REAL,
                    liquidation_price REAL,
                    quantity REAL,
                    leverage REAL,
                    liquidation_fee REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS np.margin_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    balance REAL,
                    used_margin REAL,
                    open_positions INT
                )
            """)
        conn.commit()
    finally:
        conn.close()

def record_trade(symbol, side, price, quantity, pnl=0.0, fee=0.0, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot record trade - database not available")
        return
    try:
        # Convert all numeric values to Python native types to avoid numpy type issues
        price = float(price) if price is not None else 0.0
        quantity = float(quantity) if quantity is not None else 0.0
        pnl = float(pnl) if pnl is not None else 0.0
        fee = float(fee) if fee is not None else 0.0
        
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO trades (timestamp, symbol, side, price, quantity, pnl, fee)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (timestamp, str(symbol), str(side), price, quantity, pnl, fee))
        conn.commit()
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to record trade: {e}")
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        print(f"[ERROR] Error args: {e.args}")
        conn.rollback()
    finally:
        conn.close()

def add_open_position(symbol, entry_price, quantity, leverage=1.0):
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot add open position - database not available")
        return
    try:
        # Convert all numeric values to Python native types
        entry_price = float(entry_price) if entry_price is not None else 0.0
        quantity = float(quantity) if quantity is not None else 0.0
        leverage = float(leverage) if leverage is not None else 1.0
        
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO open_positions (symbol, entry_price, quantity, leverage)
                VALUES (%s, %s, %s, %s)
            """, (str(symbol), entry_price, quantity, leverage))
        conn.commit()
    except Exception as e:
        print(f"[ERROR] Failed to add open position: {e}")
        conn.rollback()
    finally:
        conn.close()

def close_open_position(symbol):
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot close open position - database not available")
        return
    try:
        with conn.cursor() as c:
            c.execute("DELETE FROM open_positions WHERE symbol = %s", (symbol,))
        conn.commit()
    except Exception as e:
        print(f"[ERROR] Failed to close open position: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_open_positions():
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot get open positions - database not available")
        return []
    try:
        with conn.cursor() as c:
            c.execute("SELECT id, symbol, entry_price, quantity, leverage, entry_time FROM open_positions")
            result = c.fetchall()
        return result
    except Exception as e:
        print(f"[ERROR] Failed to get open positions: {e}")
        return []
    finally:
        conn.close()

def get_all_trades(limit=100):
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("""
                    SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
                    FROM trades
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
                trades = c.fetchall()
        return trades
    except Exception as e:
        print(f"[ERROR] Fetching trades failed: {e}")
        return []

def get_trade_count():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("SELECT COUNT(*) FROM trades")
                count = c.fetchone()[0]
                return count
    except Exception as e:
        print(f"[ERROR] Fetching trade count failed: {e}")
        return 0

def get_total_fees():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("SELECT SUM(fee) FROM trades")
                total = c.fetchone()[0]
                return total if total is not None else 0.0
    except Exception as e:
        print(f"[ERROR] Fetching total fees failed: {e}")
        return 0.0

def get_pnl_breakdown():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("""
                    SELECT symbol, SUM(pnl) as total_pnl, COUNT(*) as trade_count
                    FROM trades
                    GROUP BY symbol
                """)
                breakdown = c.fetchall()
                return {row[0]: {'total_pnl': row[1], 'trade_count': row[2]} for row in breakdown}
    except Exception as e:
        print(f"[ERROR] Fetching PnL breakdown failed: {e}")
        return {}

def get_rolling_stats(window=24):
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("""
                    SELECT timestamp, pnl, fee
                    FROM trades
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (window,))
                stats = c.fetchall()
                rolling_pnl = sum(row[1] for row in stats)
                rolling_fees = sum(row[2] for row in stats)
                return {'rolling_pnl': rolling_pnl, 'rolling_fees': rolling_fees, 'window': window}
    except Exception as e:
        print(f"[ERROR] Fetching rolling stats failed: {e}")
        return {'rolling_pnl': 0.0, 'rolling_fees': 0.0, 'window': window}

def get_trade_distribution():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("""
                    SELECT side, COUNT(*) as count
                    FROM trades
                    GROUP BY side
                """)
                distribution = c.fetchall()
                return {row[0]: row[1] for row in distribution}
    except Exception as e:
        print(f"[ERROR] Fetching trade distribution failed: {e}")
        return {}

def record_liquidation(symbol, entry_price, liquidation_price, quantity, leverage, fee):
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot record liquidation - database not available")
        return
    try:
        # Convert all numeric values to Python native types
        entry_price = float(entry_price) if entry_price is not None else 0.0
        liquidation_price = float(liquidation_price) if liquidation_price is not None else 0.0
        quantity = float(quantity) if quantity is not None else 0.0
        leverage = float(leverage) if leverage is not None else 1.0
        fee = float(fee) if fee is not None else 0.0
        
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO liquidations (symbol, entry_price, liquidation_price, quantity, leverage, liquidation_fee)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (str(symbol), entry_price, liquidation_price, quantity, leverage, fee))
        conn.commit()
    except Exception as e:
        print(f"[ERROR] Failed to record liquidation: {e}")
        conn.rollback()
    finally:
        conn.close()

def record_margin_snapshot(balance, used_margin, open_positions):
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot record margin snapshot - database not available")
        return
    try:
        # Convert all numeric values to Python native types
        balance = float(balance) if balance is not None else 0.0
        used_margin = float(used_margin) if used_margin is not None else 0.0
        open_positions = int(open_positions) if open_positions is not None else 0
        
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO margin_history (balance, used_margin, open_positions)
                VALUES (%s, %s, %s)
            """, (balance, used_margin, open_positions))
        conn.commit()
    except Exception as e:
        print(f"[ERROR] Failed to record margin snapshot: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_volume_profile():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("""
                    SELECT price, SUM(quantity) as total_quantity
                    FROM trades
                    GROUP BY price
                    ORDER BY price ASC
                """)
                results = c.fetchall()
                return [{'price': row[0], 'volume': row[1]} for row in results]
    except Exception as e:
        print(f"[ERROR] Fetching volume profile failed: {e}")
        return []

def get_market_depth():
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("""
                    SELECT price, SUM(quantity) as total_quantity, side
                    FROM trades
                    GROUP BY price, side
                    ORDER BY price ASC
                """)
                results = c.fetchall()
                depth = {'bids': [], 'asks': []}
                for price, quantity, side in results:
                    if side.lower() == 'buy':
                        depth['bids'].append({'price': price, 'quantity': quantity})
                    else:
                        depth['asks'].append({'price': price, 'quantity': quantity})
                return depth
    except Exception as e:
        print(f"[ERROR] Fetching market depth failed: {e}")
        return {'bids': [], 'asks': []}
