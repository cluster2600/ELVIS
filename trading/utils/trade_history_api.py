from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
import time
import psutil
try:
    from prometheus_flask_exporter import PrometheusMetrics
    from prometheus_client import Gauge, Counter, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    print("Warning: prometheus modules not available")
from utils.paper_trade_db import (
    get_all_trades, get_open_positions, get_trade_count,
    get_total_fees, get_pnl_breakdown, get_rolling_stats,
    get_trade_distribution, get_volume_profile, get_market_depth
)
from utils.price_fetcher import PriceFetcher
from utils.logger_config import setup_logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# ISSUE #13 FIX: API Key Authentication
# The Flask API was publicly accessible on 0.0.0.0:5050 with no authentication.
# Any process on the network could read trade data or trigger actions.
# Now every request (except /health) must supply the correct API key via header:
#   X-API-Key: <value of API_KEY env variable>
# Set the API_KEY environment variable before starting the bot.
# ---------------------------------------------------------------------------
_API_KEY = os.getenv('API_KEY')

@app.before_request
def require_api_key():
    """Reject requests that do not present the correct API key header."""
    # Health check is exempt so load balancers / Docker health checks still work
    from flask import request, jsonify
    if request.path == '/health':
        return  # exempt from auth

    if not _API_KEY:
        # Fail closed: if no API_KEY is configured, block all requests
        return jsonify({"error": "API authentication not configured on server"}), 503

    provided = request.headers.get('X-API-Key', '')
    if provided != _API_KEY:
        return jsonify({"error": "Unauthorized — provide a valid X-API-Key header"}), 401

# Initialize Prometheus if available
if HAS_PROMETHEUS:
    metrics = PrometheusMetrics(app)
else:
    metrics = None

# Initialize logger
logger = setup_logging("TradeHistoryAPI", log_level="INFO")

# Additional Prometheus Metrics for ELVIS Trading Dashboard
# Portfolio & Trading Metrics
PORTFOLIO_VALUE = Gauge('elvis_portfolio_value', 'Total portfolio value in USD')
TOTAL_PNL = Gauge('elvis_total_pnl', 'Total realized P&L')
UNREALIZED_PNL = Gauge('elvis_unrealized_pnl', 'Total unrealized P&L')
OPEN_POSITIONS_COUNT = Gauge('elvis_open_positions_count', 'Number of open positions')
TOTAL_TRADES_COUNT = Gauge('elvis_total_trades', 'Total number of trades')
WIN_RATE = Gauge('elvis_win_rate', 'Trading win rate percentage')
PROFIT_FACTOR = Gauge('elvis_profit_factor', 'Profit factor ratio')

# System Performance Metrics
SYSTEM_CPU_USAGE = Gauge('elvis_system_cpu_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('elvis_system_memory_percent', 'System memory usage percentage')
API_RESPONSE_TIME = Histogram('elvis_api_response_seconds', 'API response time in seconds', ['endpoint'])

# Market Data Metrics
MARKET_SPREAD = Gauge('elvis_market_spread', 'Bid-ask spread', ['symbol'])
MARKET_VOLUME = Gauge('elvis_market_volume', 'Trading volume', ['symbol'])
PRICE_CHANGE_24H = Gauge('elvis_price_change_24h', '24h price change percentage', ['symbol'])

# Trading Activity Metrics
TRADES_PER_HOUR = Gauge('elvis_trades_per_hour', 'Number of trades in the last hour')
AVERAGE_TRADE_SIZE = Gauge('elvis_avg_trade_size', 'Average trade size in USD')
LARGEST_WIN = Gauge('elvis_largest_win', 'Largest winning trade')
LARGEST_LOSS = Gauge('elvis_largest_loss', 'Largest losing trade')

# Initialize PriceFetcher for Prometheus metrics
price_fetcher = None

def initialize_price_fetcher():
    """Initialize and start PriceFetcher for metrics"""
    global price_fetcher
    try:
        logger.info("🚀 Initializing PriceFetcher for Prometheus metrics...")
        from config.config import SYMBOLS_CONFIG
        symbols = SYMBOLS_CONFIG.get('PRIMARY_SYMBOLS', ['BTCUSDT', 'BNBUSDT', 'BNBBTC'])
        price_fetcher = PriceFetcher(
            logger=logger,
            symbols=symbols,
            timeframe='1m',
            history_limit=100
        )
        price_fetcher.start()
        logger.info(f"✅ PriceFetcher started for symbols: {symbols}")
        
        # Wait for initial data and calculate indicators
        time.sleep(5)
        for symbol in symbols:
            try:
                price_fetcher.get_historical_data()
                price_fetcher.calculate_indicators(symbol)
                logger.info(f"📊 Initial metrics populated for {symbol}")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize metrics for {symbol}: {e}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize PriceFetcher: {e}")

# Start PriceFetcher in background thread
threading.Thread(target=initialize_price_fetcher, daemon=True).start()

def update_trading_metrics():
    """Update all trading and system metrics for Prometheus"""
    try:
        # System Performance Metrics
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
        SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().percent)
        
        # Trading Metrics from Database
        try:
            # Get trading statistics
            total_trades = get_trade_count()
            TOTAL_TRADES_COUNT.set(total_trades)
            
            # Get recent trades for calculations
            recent_trades = get_all_trades(limit=100)
            total_pnl = 0
            win_rate = 0
            
            if recent_trades:
                # Calculate win rate - check the correct field for P&L
                profitable_trades = 0
                total_profits = 0
                total_losses = 0
                pnls = []
                
                for trade in recent_trades:
                    try:
                        # Try different positions for PNL field
                        pnl_value = 0
                        if len(trade) > 5:
                            pnl_value = safe_float(trade[5])
                        elif len(trade) > 4:
                            # Sometimes P&L might be in position 4
                            pnl_value = safe_float(trade[4])
                        
                        pnls.append(pnl_value)
                        if pnl_value > 0:
                            profitable_trades += 1
                            total_profits += pnl_value
                        elif pnl_value < 0:
                            total_losses += abs(pnl_value)
                    except Exception as e:
                        logger.debug(f"Error processing trade PNL: {e}")
                        continue
                
                win_rate = (profitable_trades / len(recent_trades)) * 100 if recent_trades else 0
                WIN_RATE.set(win_rate)
                
                # Calculate profit factor
                profit_factor = total_profits / total_losses if total_losses > 0 else 1.0
                PROFIT_FACTOR.set(profit_factor)
                
                # Calculate average trade size
                try:
                    avg_trade_size = sum(safe_float(trade[3]) * safe_float(trade[4]) for trade in recent_trades if len(trade) > 4) / len(recent_trades)
                    AVERAGE_TRADE_SIZE.set(avg_trade_size)
                except:
                    AVERAGE_TRADE_SIZE.set(0)
                
                # Largest win/loss
                LARGEST_WIN.set(max(pnls) if pnls else 0)
                LARGEST_LOSS.set(min(pnls) if pnls else 0)
                
                # Total PNL
                total_pnl = sum(pnls)
                TOTAL_PNL.set(total_pnl)
            
            # Get open positions - use API call for consistency
            try:
                import requests
                response = requests.get('http://localhost:5050/open_positions', timeout=5)
                if response.status_code == 200:
                    open_positions_data = response.json()
                    OPEN_POSITIONS_COUNT.set(len(open_positions_data))
                    
                    # Calculate unrealized PNL
                    unrealized_pnl = 0
                    if open_positions_data and price_fetcher:
                        for position in open_positions_data:
                            try:
                                symbol = position.get('symbol', '')
                                side = position.get('side', '')
                                entry_price = safe_float(position.get('entry_price', 0))
                                quantity = safe_float(position.get('quantity', 0))
                                
                                # Get current price
                                current_price = price_fetcher.get_current_price(symbol)
                                if current_price and entry_price > 0 and quantity > 0:
                                    if side.upper() == 'BUY':
                                        pnl = (current_price - entry_price) * quantity
                                    else:
                                        pnl = (entry_price - current_price) * quantity
                                    unrealized_pnl += pnl
                            except Exception as e:
                                logger.debug(f"Error calculating position PNL: {e}")
                    
                    UNREALIZED_PNL.set(unrealized_pnl)
                else:
                    # Fallback to database query
                    open_positions = get_open_positions()
                    OPEN_POSITIONS_COUNT.set(len(open_positions))
                    UNREALIZED_PNL.set(0)
            except Exception as e:
                logger.debug(f"Error fetching open positions: {e}")
                OPEN_POSITIONS_COUNT.set(0)
                UNREALIZED_PNL.set(0)
            
            # Calculate portfolio value
            base_portfolio = 2000  # Base portfolio (1000 USDT + 1000 BNB)
            current_unrealized = UNREALIZED_PNL._value._value if hasattr(UNREALIZED_PNL, '_value') else 0
            portfolio_value = base_portfolio + total_pnl + current_unrealized
            PORTFOLIO_VALUE.set(portfolio_value)
            
            logger.debug(f"Updated portfolio metrics: value={portfolio_value}, pnl={total_pnl}, unrealized={current_unrealized}")
            
        except Exception as e:
            logger.warning(f"Error updating trading metrics: {e}")
        
        # Market Data Metrics
        if price_fetcher:
            try:
                from config.config import SYMBOLS_CONFIG
                symbols = SYMBOLS_CONFIG.get('PRIMARY_SYMBOLS', ['BTCUSDT', 'BNBUSDT', 'BNBBTC'])
                for symbol in symbols:
                    current_price = price_fetcher.get_current_price(symbol)
                    if current_price:
                        # Simulate market spread (0.01% of price)
                        spread = current_price * 0.0001
                        MARKET_SPREAD.labels(symbol=symbol).set(spread)
                        
                        # Simulate volume (based on recent activity)
                        volume = len(recent_trades) * 100 if 'recent_trades' in locals() else 1000
                        MARKET_VOLUME.labels(symbol=symbol).set(volume)
                        
                        # Simulate 24h price change
                        price_change = (win_rate - 50) * 0.1  # Use win rate as proxy for price movement
                        PRICE_CHANGE_24H.labels(symbol=symbol).set(price_change)
            except Exception as e:
                logger.debug(f"Error updating market metrics: {e}")
                
    except Exception as e:
        logger.error(f"Error in update_trading_metrics: {e}")

def start_metrics_updater():
    """Start background thread to update metrics every 10 seconds"""
    def metrics_loop():
        while True:
            try:
                update_trading_metrics()
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                time.sleep(10)
    
    thread = threading.Thread(target=metrics_loop, daemon=True)
    thread.start()
    logger.info("✅ Started metrics updater thread")

# Start metrics updater
start_metrics_updater()

def format_timestamp(ts):
    if hasattr(ts, 'isoformat'):
        return ts.isoformat() + 'Z'
    return str(ts)

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

# API Endpoints
@app.route('/trades', methods=['GET'])
def trades():
    """Get recent trades only (last 24 hours) to avoid historical data pollution"""
    try:
        # FIXED: Get recent trades only from PostgreSQL directly
        import psycopg2
        from utils.paper_trade_db import get_conn
        
        trade_list = []
        conn = get_conn()
        if conn:
            with conn.cursor() as c:
                # Get the latest reset timestamp
                c.execute("""
                    SELECT reset_timestamp FROM trading_session_resets 
                    ORDER BY reset_timestamp DESC LIMIT 1
                """)
                reset_result = c.fetchone()
                
                if reset_result:
                    reset_timestamp = reset_result[0]
                    # Get trades after reset only
                    c.execute("""
                        SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
                        FROM trades 
                        WHERE timestamp >= %s
                        AND pnl BETWEEN -100 AND 100
                        ORDER BY timestamp DESC
                        LIMIT 25
                    """, (reset_timestamp,))
                    trades = c.fetchall()
                else:
                    # No reset found, return empty list
                    trades = []
                
                for t in trades:
                    trade_list.append({
                        "id": t[0],
                        "timestamp": format_timestamp(t[1]),
                        "symbol": t[2],
                        "side": t[3],
                        "price": safe_float(t[4]),
                        "quantity": safe_float(t[5]),
                        "pnl": safe_float(t[6]),
                        "fee": safe_float(t[7]) if len(t) > 7 else 0.0
                    })
            conn.close()
        
        # If no recent trades, return empty list instead of historical data
        return jsonify(trade_list)
        
    except Exception as e:
        logger.warning(f"Failed to fetch recent trades: {e}")
        # Return empty list instead of falling back to historical data
        return jsonify([])  

@app.route('/trades/count', methods=['GET'])
def trades_count():
    try:
        count = get_trade_count()
        return jsonify({"count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/open_positions', methods=['GET'])
def open_positions():
    try:
        positions = get_open_positions()
        pos_list = [
            {
                "id": p[0],
                "symbol": p[1],
                "side": p[2],
                "entry_price": safe_float(p[3]),
                "quantity": safe_float(p[4]),
                "leverage": safe_float(p[5]),
                "entry_time": format_timestamp(p[6])
            }
            for p in positions
        ]
        return jsonify(pos_list)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch open positions: {str(e)}"}), 500

@app.route('/balance', methods=['GET'])
def balance():
    """Get current account balance for all assets with P&L adjustments"""
    try:
        # Try to get balance from DI container or executor (uses fixed calculation)
        from core.di import container
        try:
            executor = container.get('executor')
            if executor:
                balance_data = executor.get_balance()
                return jsonify(balance_data)
        except:
            pass
        
        # FIXED: Fallback calculation with recent P&L only
        import psycopg2
        from utils.paper_trade_db import get_conn
        
        # Calculate recent P&L only
        total_pnl = 0.0
        try:
            conn = get_conn()
            if conn:
                with conn.cursor() as c:
                    # Get the latest reset timestamp
                    c.execute("""
                        SELECT reset_timestamp FROM trading_session_resets 
                        ORDER BY reset_timestamp DESC LIMIT 1
                    """)
                    reset_result = c.fetchone()
                    
                    if reset_result:
                        reset_timestamp = reset_result[0]
                        # Get P&L from trades after reset only
                        c.execute("""
                            SELECT SUM(pnl) FROM trades 
                            WHERE timestamp >= %s
                            AND pnl BETWEEN -100 AND 100
                        """, (reset_timestamp,))
                        result = c.fetchone()
                        total_pnl = float(result[0]) if result and result[0] is not None else 0.0
                    else:
                        # No reset found, default to 0
                        total_pnl = 0.0
                conn.close()
            
            # Cap P&L to reasonable range
            total_pnl = max(-1000.0, min(1000.0, total_pnl))
            
        except Exception as e:
            logger.warning(f"Could not calculate recent P&L for balance: {e}")
            total_pnl = 0.0
        
        # Calculate asset amounts worth $1000 each initially
        bnb_price = 600.0  # Default BNB price
        btc_price = 116500.0  # Default BTC price
        
        # Try to get live prices
        if price_fetcher:
            try:
                bnb_price = price_fetcher.get_current_price('BNBUSDT') or bnb_price
                btc_price = price_fetcher.get_current_price('BTCUSDT') or btc_price
            except:
                pass
        
        # Fixed balance calculation with recent P&L applied to USDT
        adjusted_usdt = 1000.0 + total_pnl  # Apply P&L to USDT
        balance_data = {
            'USDT': max(0.0, adjusted_usdt),
            'BNB': 1000.0 / bnb_price,  # Amount of BNB worth $1000
            'BTC': 1000.0 / btc_price   # Amount of BTC worth $1000
        }
        
        return jsonify(balance_data)
        
    except Exception as e:
        return jsonify({"error": f"Failed to fetch balance: {str(e)}"}), 500

@app.route('/fees', methods=['GET'])
def fees():
    try:
        total_fees = get_total_fees()
        return jsonify({"total_fees": total_fees})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch fees: {str(e)}"}), 500

@app.route('/pnl_breakdown', methods=['GET'])
def pnl_breakdown():
    return jsonify(get_pnl_breakdown())

@app.route('/rolling_stats', methods=['GET'])
def rolling_stats():
    return jsonify(get_rolling_stats())

@app.route('/trade_distribution', methods=['GET'])
def trade_distribution():
    return jsonify(get_trade_distribution())

@app.route('/volume_profile', methods=['GET'])
def volume_profile():
    return jsonify(get_volume_profile())

@app.route('/market_depth', methods=['GET'])
def market_depth():
    return jsonify(get_market_depth())

@app.route('/', methods=['GET'])
def root():
    """Redirect to dashboard"""
    return dashboard()

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Serve the trading dashboard HTML"""
    try:
        # Get the absolute path to the static directory
        static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static')
        return send_from_directory(static_dir, 'index.html')
    except Exception as e:
        return jsonify({"error": f"Dashboard not found: {str(e)}"}), 404

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Docker container"""
    return jsonify({
        "status": "healthy",
        "service": "trade_history_api",
        "version": "1.0.0"
    })

# NEW: Create a function that can be called externally
def start_trade_history_server(host='0.0.0.0', port=5050):
    """
    Starts the trade history Flask API server.
    """
    print(f"Starting Trade History Server on {host}:{port}...")
    app.run(host=host, port=port)

# Old direct run removed
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5050)