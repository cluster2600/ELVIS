import psycopg2

PSYCOPG2_AVAILABLE = True

import os
from datetime import datetime

from dotenv import load_dotenv

from .secrets_manager import get_enhanced_secrets_manager

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
)

# Initialize secrets manager
secrets_manager = get_enhanced_secrets_manager()


def get_conn():
    """Get PostgreSQL connection using enhanced secrets manager."""
    try:
        # Get database credentials from secrets manager (with Vault fallback)
        db_creds = secrets_manager.get_database_credentials()

        conn = psycopg2.connect(
            host=db_creds.get("host", "localhost"),
            port=int(db_creds.get("port", 5432)),
            user=db_creds.get("user", "postgres"),
            password=db_creds.get("password", ""),
            dbname=db_creds.get("dbname", "trading_bot"),
            connect_timeout=5,  # 5 second timeout
        )
        # Set the search path to use the 'np' schema
        with conn.cursor() as c:
            c.execute("SET search_path TO np, public;")
        conn.commit()
        return conn
    except psycopg2.OperationalError as e:
        print(f"[ERROR] Could not connect to PostgreSQL: {e}")
        print("[ERROR] PostgreSQL connection is required - no fallback available")
        return None


def init_db():
    conn = get_conn()
    if conn is None:
        print("[ERROR] Cannot initialize database - PostgreSQL connection required")
        return
    try:
        c = conn.cursor()

        # PostgreSQL syntax - create tables in the np schema
        try:
            c.execute("GRANT USAGE, CREATE ON SCHEMA np TO CURRENT_USER;")
            conn.commit()
        except Exception as e:
            print(f"[INFO] Could not grant permissions, trying to continue: {e}")
            conn.rollback()

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
        c.execute("DROP TABLE IF EXISTS np.open_positions CASCADE;")
        c.execute("""
            CREATE TABLE IF NOT EXISTS np.open_positions (
                id SERIAL PRIMARY KEY,
                symbol TEXT,
                side TEXT,
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
                open_positions INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS np.trading_session_resets (
                id SERIAL PRIMARY KEY,
                reset_timestamp TIMESTAMP DEFAULT NOW(),
                reason TEXT
            )
        """)
        conn.commit()
        print("[INFO] Database tables initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        conn.rollback()
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

        c = conn.cursor()

        # PostgreSQL syntax with %s placeholders
        c.execute(
            """
            INSERT INTO trades (timestamp, symbol, side, price, quantity, pnl, fee)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
            (timestamp, str(symbol), str(side), price, quantity, pnl, fee),
        )

        conn.commit()
        # Reduced logging to prevent dashboard spam
        pass
    except Exception as e:
        import traceback

        print(f"[ERROR] Failed to record trade: {e}")
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        print(f"[ERROR] Error args: {e.args}")
        conn.rollback()
    finally:
        conn.close()


def add_open_position(symbol, side, entry_price, quantity, leverage=1.0):
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot add open position - database not available")
        return
    try:
        # Convert all numeric values to Python native types
        entry_price = float(entry_price) if entry_price is not None else 0.0
        quantity = float(quantity) if quantity is not None else 0.0
        leverage = float(leverage) if leverage is not None else 1.0

        c = conn.cursor()

        # PostgreSQL syntax with %s placeholders
        c.execute(
            """
            INSERT INTO open_positions (symbol, side, entry_price, quantity, leverage)
            VALUES (%s, %s, %s, %s, %s)
        """,
            (str(symbol), str(side), entry_price, quantity, leverage),
        )

        conn.commit()
        # Reduced logging to prevent dashboard spam
        pass
    except Exception as e:
        print(f"[ERROR] Failed to add open position: {e}")
        conn.rollback()
    finally:
        conn.close()


def close_open_position(symbol, side):
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot close open position - database not available")
        return
    try:
        c = conn.cursor()

        # PostgreSQL syntax with %s placeholders
        c.execute(
            "DELETE FROM open_positions WHERE symbol = %s AND side = %s", (symbol, side)
        )

        conn.commit()
        # Reduced logging to prevent dashboard spam
        pass
    except Exception as e:
        print(f"[ERROR] Failed to close open position: {e}")
        conn.rollback()
    finally:
        conn.close()


def close_position(position_id, exit_price, pnl, fee=0.0):
    """Close a specific position by ID and record the trade"""
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot close position - database not available")
        return
    try:
        c = conn.cursor()

        # Get position details before closing
        c.execute(
            "SELECT symbol, side, entry_price, quantity, leverage FROM open_positions WHERE id = %s",
            (position_id,),
        )
        position = c.fetchone()

        if position:
            symbol, side, entry_price, quantity, leverage = position

            # Record the closing trade
            record_trade(symbol, side, exit_price, quantity, pnl, fee)

            # Remove from open positions
            c.execute("DELETE FROM open_positions WHERE id = %s", (position_id,))

            conn.commit()
            # Only log significant trades to reduce spam
            if abs(pnl) >= 5.0:  # Only log profits/losses >= $5
                print(
                    f"[INFO] Closed position {position_id}: {side} {symbol} P&L: ${pnl:.2f}"
                )
        else:
            print(f"[WARNING] Position {position_id} not found")

    except Exception as e:
        print(f"[ERROR] Failed to close position {position_id}: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_open_positions():
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot get open positions - database not available")
        return []
    try:
        c = conn.cursor()
        c.execute(
            "SELECT id, symbol, side, entry_price, quantity, leverage, entry_time FROM open_positions"
        )
        result = c.fetchall()
        return result
    except Exception as e:
        print(f"[ERROR] Failed to get open positions: {e}")
        return []
    finally:
        conn.close()


def get_all_trades(limit=100, exclude_test=False):
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot get trades - database not available")
        return []
    try:
        c = conn.cursor()

        # PostgreSQL syntax with %s placeholders
        if exclude_test:
            c.execute(
                """
                SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                WHERE side != %s
                ORDER BY timestamp DESC
                LIMIT %s
            """,
                ("TEST", limit),
            )
        else:
            c.execute(
                """
                SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                ORDER BY timestamp DESC
                LIMIT %s
            """,
                (limit,),
            )

        trades = c.fetchall()
        return trades
    except Exception as e:
        print(f"[ERROR] Fetching trades failed: {e}")
        return []
    finally:
        conn.close()


def get_trade_count(exclude_test=False):
    conn = get_conn()
    if conn is None:
        print(f"[WARNING] Cannot get trade count - database not available")
        return 0
    try:
        c = conn.cursor()
        if exclude_test:
            c.execute("SELECT COUNT(*) FROM trades WHERE side != 'TEST'")
        else:
            c.execute("SELECT COUNT(*) FROM trades")
        count = c.fetchone()[0]
        return count
    except Exception as e:
        print(f"[ERROR] Fetching trade count failed: {e}")
        return 0
    finally:
        conn.close()


def get_total_fees(exclude_test=False):
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                if exclude_test:
                    c.execute("SELECT SUM(fee) FROM trades WHERE side != 'TEST'")
                else:
                    c.execute("SELECT SUM(fee) FROM trades")
                total = c.fetchone()[0]
                return total if total is not None else 0.0
    except Exception as e:
        print(f"[ERROR] Fetching total fees failed: {e}")
        return 0.0


def get_pnl_breakdown(exclude_test=False):
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                if exclude_test:
                    c.execute("""
                        SELECT symbol, SUM(pnl) as total_pnl, COUNT(*) as trade_count
                        FROM trades
                        WHERE side != 'TEST'
                        GROUP BY symbol
                    """)
                else:
                    c.execute("""
                        SELECT symbol, SUM(pnl) as total_pnl, COUNT(*) as trade_count
                        FROM trades
                        GROUP BY symbol
                    """)
                breakdown = c.fetchall()
                return {
                    row[0]: {"total_pnl": row[1], "trade_count": row[2]}
                    for row in breakdown
                }
    except Exception as e:
        print(f"[ERROR] Fetching PnL breakdown failed: {e}")
        return {}


def get_rolling_stats(window=24):
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute(
                    """
                    SELECT timestamp, pnl, fee
                    FROM trades
                    ORDER BY timestamp DESC
                    LIMIT %s
                """,
                    (window,),
                )
                stats = c.fetchall()
                rolling_pnl = sum(row[1] for row in stats)
                rolling_fees = sum(row[2] for row in stats)
                return {
                    "rolling_pnl": rolling_pnl,
                    "rolling_fees": rolling_fees,
                    "window": window,
                }
    except Exception as e:
        print(f"[ERROR] Fetching rolling stats failed: {e}")
        return {"rolling_pnl": 0.0, "rolling_fees": 0.0, "window": window}


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
        liquidation_price = (
            float(liquidation_price) if liquidation_price is not None else 0.0
        )
        quantity = float(quantity) if quantity is not None else 0.0
        leverage = float(leverage) if leverage is not None else 1.0
        fee = float(fee) if fee is not None else 0.0

        with conn.cursor() as c:
            c.execute(
                """
                INSERT INTO liquidations (symbol, entry_price, liquidation_price, quantity, leverage, liquidation_fee)
                VALUES (%s, %s, %s, %s, %s, %s)
            """,
                (str(symbol), entry_price, liquidation_price, quantity, leverage, fee),
            )
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
            c.execute(
                """
                INSERT INTO margin_history (balance, used_margin, open_positions)
                VALUES (%s, %s, %s)
            """,
                (balance, used_margin, open_positions),
            )
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
                return [{"price": row[0], "volume": row[1]} for row in results]
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
                depth = {"bids": [], "asks": []}
                for price, quantity, side in results:
                    if side.lower() == "buy":
                        depth["bids"].append({"price": price, "quantity": quantity})
                    else:
                        depth["asks"].append({"price": price, "quantity": quantity})
                return depth
    except Exception as e:
        print(f"[ERROR] Fetching market depth failed: {e}")
        return {"bids": [], "asks": []}


def clear_all_trades():
    """Clear all trades from the database (for fresh start)."""
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("DELETE FROM trades")
                conn.commit()
                print("[INFO] All trades cleared from database")
    except Exception as e:
        print(f"[ERROR] Failed to clear trades: {e}")


def clear_open_positions():
    """Clear all open positions from the database (for fresh start)."""
    try:
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute("DELETE FROM open_positions")
                conn.commit()
                print("[INFO] All open positions cleared from database")
    except Exception as e:
        print(f"[ERROR] Failed to clear open positions: {e}")


def reset_paper_trading_balances():
    """Reset paper trading balances to initial $1000 each in USDT and BNB."""
    try:
        conn = get_conn()
        if conn is None:
            print("[ERROR] Cannot reset balances - database not available")
            return False

        with conn.cursor() as c:
            # Reset USDT balance to $1000
            c.execute("""
                INSERT INTO np.account_balances (asset, balance) 
                VALUES ('USDT', 1000.0) 
                ON CONFLICT (asset) 
                DO UPDATE SET balance = 1000.0, last_updated = NOW()
            """)

            # Reset BNB balance to $1000
            c.execute("""
                INSERT INTO np.account_balances (asset, balance) 
                VALUES ('BNB', 1000.0) 
                ON CONFLICT (asset) 
                DO UPDATE SET balance = 1000.0, last_updated = NOW()
            """)

            # Clear any other asset balances to 0
            c.execute("""
                UPDATE np.account_balances 
                SET balance = 0.0, last_updated = NOW() 
                WHERE asset NOT IN ('USDT', 'BNB')
            """)

        conn.commit()
        print("[INFO] Paper trading balances reset to $1000 each in USDT and BNB")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to reset balances: {e}")
        return False


def initialize_paper_trading():
    """Complete paper trading initialization - reset all data to starting state."""
    print("[INFO] Initializing paper trading mode...")

    # Initialize database structure
    init_db_with_balances()

    # Clear all existing data
    clear_all_trades()
    clear_open_positions()

    # Reset balances to initial state
    reset_paper_trading_balances()

    print("[INFO] Paper trading initialized: $1000 USDT + $1000 BNB starting balances")
    return True


def init_db_with_balances():
    """Initialize database with multi-asset balance support"""
    conn = get_conn()
    if conn is None:
        print("[ERROR] Cannot initialize database - PostgreSQL connection required")
        return
    try:
        c = conn.cursor()

        # Grant permissions
        try:
            c.execute("GRANT USAGE, CREATE ON SCHEMA np TO CURRENT_USER;")
            conn.commit()
        except Exception as e:
            print(f"[INFO] Could not grant permissions, trying to continue: {e}")
            conn.rollback()

        # Create enhanced balance table
        c.execute("""
            CREATE TABLE IF NOT EXISTS np.account_balances (
                id SERIAL PRIMARY KEY,
                asset TEXT UNIQUE NOT NULL,
                balance REAL NOT NULL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT NOW()
            )
        """)

        # Insert initial balances - $1000 each in USDT and BNB
        c.execute("""
            INSERT INTO np.account_balances (asset, balance) 
            VALUES ('USDT', 1000.0) 
            ON CONFLICT (asset) DO NOTHING
        """)

        # Start with $1000 in BNB as well for paper trading
        c.execute("""
            INSERT INTO np.account_balances (asset, balance)
            VALUES ('BNB', 1000.0)
            ON CONFLICT (asset) DO NOTHING
        """)

        # Track trading-session resets so P&L can start fresh per session
        c.execute("""
            CREATE TABLE IF NOT EXISTS np.trading_session_resets (
                id SERIAL PRIMARY KEY,
                reset_timestamp TIMESTAMP DEFAULT NOW(),
                reason TEXT
            )
        """)

        conn.commit()
        print("[INFO] Enhanced database with multi-asset balances initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize enhanced database: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_account_balance(asset="USDT"):
    """Get balance for a specific asset"""
    try:
        conn = get_conn()
        if conn is None:
            return 1000.0 if asset in ["USDT", "BNB"] else 0.0

        with conn.cursor() as c:
            c.execute(
                "SELECT balance FROM np.account_balances WHERE asset = %s", (asset,)
            )
            result = c.fetchone()
            return (
                float(result[0])
                if result
                else (1000.0 if asset in ["USDT", "BNB"] else 0.0)
            )
    except Exception as e:
        print(f"[ERROR] Error getting {asset} balance: {e}")
        return 1000.0 if asset in ["USDT", "BNB"] else 0.0


def update_account_balance(asset, new_balance):
    """Update balance for a specific asset"""
    try:
        conn = get_conn()
        if conn is None:
            return False

        with conn.cursor() as c:
            c.execute(
                """
                INSERT INTO np.account_balances (asset, balance) 
                VALUES (%s, %s) 
                ON CONFLICT (asset) 
                DO UPDATE SET balance = %s, last_updated = NOW()
            """,
                (asset, new_balance, new_balance),
            )
        conn.commit()
        return True
    except Exception as e:
        print(f"[ERROR] Error updating {asset} balance: {e}")
        return False


def get_all_balances():
    """Get all asset balances"""
    try:
        conn = get_conn()
        if conn is None:
            return {"USDT": 1000.0, "BNB": 1000.0}

        with conn.cursor() as c:
            c.execute("SELECT asset, balance FROM np.account_balances")
            results = c.fetchall()
            return {row[0]: float(row[1]) for row in results}
    except Exception as e:
        print(f"[ERROR] Error getting all balances: {e}")
        return {"USDT": 1000.0, "BNB": 1000.0}


def reset_trading_session():
    """
    Reset the trading session by creating a new reset timestamp.
    This makes the bot start with 0 realized P&L for the next session
    while preserving all historical trade data.
    """
    try:
        conn = get_conn()
        if conn is None:
            print("[ERROR] Cannot reset trading session - database not available")
            return False

        with conn.cursor() as c:
            # Insert new reset timestamp
            c.execute("""
                INSERT INTO trading_session_resets (reset_timestamp, reason) 
                VALUES (NOW(), 'Manual bot shutdown')
            """)

            # Get the count of resets for logging
            c.execute("SELECT COUNT(*) FROM trading_session_resets")
            reset_count = c.fetchone()[0]

        conn.commit()
        print(
            f"[INFO] Trading session reset #{reset_count} - P&L will start fresh next session"
        )
        return True

    except Exception as e:
        print(f"[ERROR] Error resetting trading session: {e}")
        return False


class PaperTradeDB:
    """Thin object wrapper over this module's functions.

    trading/api/app.py expects a PaperTradeDB() with method access; the storage
    layer is otherwise function-based. This delegates the common operations so
    the Flask API can import and run.
    """

    def get_all_trades(self, limit=100, exclude_test=False):
        return get_all_trades(limit=limit, exclude_test=exclude_test)

    def get_open_positions(self):
        return get_open_positions()

    def get_trade_count(self, exclude_test=False):
        return get_trade_count(exclude_test=exclude_test)

    def get_total_fees(self, exclude_test=False):
        return get_total_fees(exclude_test=exclude_test)

    def record_trade(self, symbol, side, price, quantity, pnl=0.0, fee=0.0):
        return record_trade(symbol, side, price, quantity, pnl, fee)
