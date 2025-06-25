"""
REST API for ELVIS Trading Bot
Provides endpoints for monitoring, control, and data access
"""

from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import os
from functools import wraps
import jwt

# Import trading components
from utils.redis_cache import get_cache
from utils.paper_trade_db import PaperTradeDB
from utils.logger_config import get_logger

# Import Swagger components
from .swagger import swaggerui_blueprint, get_swagger_spec

# Import WebSocket handler
from .websocket_handler import init_websocket, broadcast_alert, get_connected_clients_info

logger = get_logger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Register Swagger UI blueprint
app.register_blueprint(swaggerui_blueprint)

# Initialize WebSocket support
socketio = init_websocket(app, os.getenv('API_SECRET_KEY', 'your-secret-key-here'))

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# Configuration
app.config['SECRET_KEY'] = os.getenv('API_SECRET_KEY', 'your-secret-key-here')
app.config['JSON_SORT_KEYS'] = False

# Global variables for bot state
bot_state = {
    'running': False,
    'mode': 'paper',
    'start_time': None,
    'strategy': None,
    'balance': 0,
    'open_positions': 0
}


def require_auth(f):
    """Decorator to require authentication for endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Verify token
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user = payload
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


# Health check endpoint
@app.route('/health', methods=['GET'])
@limiter.limit("1000 per hour")
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


# Authentication endpoints
@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per hour")
def login():
    """Login endpoint to get JWT token"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    username = data.get('username')
    password = data.get('password')
    
    # Simple authentication (replace with proper auth in production)
    if username == os.getenv('API_USERNAME', 'admin') and \
       password == os.getenv('API_PASSWORD', 'admin'):
        
        # Generate token
        payload = {
            'user': username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'token': token,
            'expires_in': 86400  # 24 hours
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401


# Bot status endpoints
@app.route('/api/bot/status', methods=['GET'])
@require_auth
def get_bot_status():
    """Get current bot status"""
    cache = get_cache()
    
    # Try to get real-time data from cache
    cached_status = cache.get('bot:status')
    if cached_status:
        return jsonify(cached_status)
    
    # Return current state
    return jsonify({
        'running': bot_state['running'],
        'mode': bot_state['mode'],
        'start_time': bot_state['start_time'],
        'strategy': bot_state['strategy'],
        'uptime': _calculate_uptime()
    })


@app.route('/api/bot/start', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def start_bot():
    """Start the trading bot"""
    if bot_state['running']:
        return jsonify({'error': 'Bot is already running'}), 400
    
    data = request.get_json() or {}
    mode = data.get('mode', 'paper')
    strategy = data.get('strategy', 'ensemble')
    
    # In a real implementation, this would start the actual bot process
    bot_state['running'] = True
    bot_state['mode'] = mode
    bot_state['strategy'] = strategy
    bot_state['start_time'] = datetime.now().isoformat()
    
    logger.info(f"Bot started in {mode} mode with {strategy} strategy")
    
    return jsonify({
        'message': 'Bot started successfully',
        'mode': mode,
        'strategy': strategy,
        'start_time': bot_state['start_time']
    })


@app.route('/api/bot/stop', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def stop_bot():
    """Stop the trading bot"""
    if not bot_state['running']:
        return jsonify({'error': 'Bot is not running'}), 400
    
    # In a real implementation, this would stop the actual bot process
    bot_state['running'] = False
    bot_state['start_time'] = None
    
    logger.info("Bot stopped")
    
    return jsonify({'message': 'Bot stopped successfully'})


# Trading data endpoints
@app.route('/api/account/balance', methods=['GET'])
@require_auth
def get_balance():
    """Get account balance"""
    cache = get_cache()
    
    # Try cache first
    cached_balance = cache.get('account:balance')
    if cached_balance:
        return jsonify(cached_balance)
    
    # Mock data for demonstration
    balance_data = {
        'total_balance': 10000,
        'available_balance': 8500,
        'in_position': 1500,
        'currency': 'USDT',
        'timestamp': datetime.now().isoformat()
    }
    
    # Cache for 60 seconds
    cache.set('account:balance', balance_data, ttl=60)
    
    return jsonify(balance_data)


@app.route('/api/positions', methods=['GET'])
@require_auth
def get_positions():
    """Get open positions"""
    cache = get_cache()
    
    # Try cache first
    cached_positions = cache.get('trading:positions')
    if cached_positions:
        return jsonify(cached_positions)
    
    # Mock data for demonstration
    positions = [
        {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'quantity': 0.05,
            'entry_price': 50000,
            'current_price': 51000,
            'pnl': 50,
            'pnl_percentage': 2.0,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    # Cache for 30 seconds
    cache.set('trading:positions', positions, ttl=30)
    
    return jsonify(positions)


@app.route('/api/trades/history', methods=['GET'])
@require_auth
def get_trade_history():
    """Get trade history"""
    # Get query parameters
    limit = request.args.get('limit', 50, type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        # Use paper trade database
        db = PaperTradeDB()
        trades = db.get_all_trades()
        
        # Filter by date if provided
        if start_date:
            start = datetime.fromisoformat(start_date)
            trades = [t for t in trades if datetime.fromisoformat(t['timestamp']) >= start]
        
        if end_date:
            end = datetime.fromisoformat(end_date)
            trades = [t for t in trades if datetime.fromisoformat(t['timestamp']) <= end]
        
        # Sort by timestamp descending
        trades.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Limit results
        trades = trades[:limit]
        
        return jsonify({
            'trades': trades,
            'count': len(trades),
            'total': len(db.get_all_trades())
        })
        
    except Exception as e:
        logger.error(f"Error fetching trade history: {e}")
        return jsonify({'error': 'Failed to fetch trade history'}), 500


@app.route('/api/performance/stats', methods=['GET'])
@require_auth
def get_performance_stats():
    """Get performance statistics"""
    cache = get_cache()
    
    # Try cache first
    cached_stats = cache.get('performance:stats')
    if cached_stats:
        return jsonify(cached_stats)
    
    try:
        db = PaperTradeDB()
        trades = db.get_all_trades()
        
        # Calculate statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in trades if t.get('pnl', 0) <= 0])
        
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'best_trade': max((t.get('pnl', 0) for t in trades), default=0),
            'worst_trade': min((t.get('pnl', 0) for t in trades), default=0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache for 5 minutes
        cache.set('performance:stats', stats, ttl=300)
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error calculating performance stats: {e}")
        return jsonify({'error': 'Failed to calculate statistics'}), 500


# Market data endpoints
@app.route('/api/market/price/<symbol>', methods=['GET'])
@require_auth
def get_market_price(symbol: str):
    """Get current market price for a symbol"""
    cache = get_cache()
    
    # Try cache first
    cache_key = f'price:{symbol}'
    cached_price = cache.get(cache_key)
    
    if cached_price:
        return jsonify({
            'symbol': symbol,
            'price': cached_price,
            'source': 'cache',
            'timestamp': datetime.now().isoformat()
        })
    
    # Mock price for demonstration
    mock_prices = {
        'BTCUSDT': 50000,
        'ETHUSDT': 3000,
        'BNBUSDT': 400
    }
    
    price = mock_prices.get(symbol, 0)
    
    if price:
        # Cache for 60 seconds
        cache.set(cache_key, price, ttl=60)
        
        return jsonify({
            'symbol': symbol,
            'price': price,
            'source': 'mock',
            'timestamp': datetime.now().isoformat()
        })
    
    return jsonify({'error': f'Symbol {symbol} not found'}), 404


@app.route('/api/market/indicators/<symbol>', methods=['GET'])
@require_auth
def get_market_indicators(symbol: str):
    """Get technical indicators for a symbol"""
    cache = get_cache()
    
    # Try cache first
    cache_key = f'indicators:{symbol}:all'
    cached_indicators = cache.get(cache_key)
    
    if cached_indicators:
        return jsonify(cached_indicators)
    
    # Mock indicators for demonstration
    indicators = {
        'symbol': symbol,
        'rsi': 55.5,
        'macd': 100.0,
        'signal': 95.0,
        'sma_20': 49500,
        'sma_50': 49000,
        'ema_12': 50100,
        'ema_26': 49900,
        'timestamp': datetime.now().isoformat()
    }
    
    # Cache for 30 seconds
    cache.set(cache_key, indicators, ttl=30)
    
    return jsonify(indicators)


# Configuration endpoints
@app.route('/api/config', methods=['GET'])
@require_auth
def get_config():
    """Get current bot configuration"""
    # Mock configuration
    config = {
        'trading': {
            'symbol': 'BTCUSDT',
            'max_position_size': 0.1,
            'risk_per_trade': 0.02,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        },
        'strategy': {
            'type': 'ensemble',
            'models': ['random_forest', 'neural_network', 'transformer'],
            'weights': [0.4, 0.3, 0.3]
        },
        'risk_management': {
            'max_daily_trades': 5,
            'max_drawdown': 0.2,
            'position_sizing': 'kelly_criterion'
        }
    }
    
    return jsonify(config)


@app.route('/api/config', methods=['PUT'])
@require_auth
@limiter.limit("10 per hour")
def update_config():
    """Update bot configuration"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # In a real implementation, this would validate and update configuration
    logger.info(f"Configuration update requested: {data}")
    
    return jsonify({
        'message': 'Configuration updated successfully',
        'updated_fields': list(data.keys())
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded', 'message': str(error.description)}), 429


# Swagger specification endpoint
@app.route('/api/swagger.json', methods=['GET'])
def swagger_spec():
    """Get OpenAPI specification"""
    return jsonify(get_swagger_spec())


# Dashboard-specific endpoints
@app.route('/api/dashboard/overview', methods=['GET'])
@require_auth
def get_dashboard_overview():
    """Get dashboard overview data"""
    cache = get_cache()
    
    # Try cache first
    cached_overview = cache.get('dashboard:overview')
    if cached_overview:
        return jsonify(cached_overview)
    
    try:
        # Get performance stats
        db = PaperTradeDB()
        trades = db.get_all_trades()
        
        total_trades = len(trades)
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        
        overview = {
            'bot_status': {
                'running': bot_state['running'],
                'mode': bot_state['mode'],
                'uptime': _calculate_uptime(),
                'health': 'healthy'
            },
            'performance': {
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'today_pnl': total_pnl * 0.1,  # Mock today's PnL
                'week_pnl': total_pnl * 0.7    # Mock week's PnL
            },
            'positions': {
                'open_positions': 1,  # Mock data
                'total_value': 2500,
                'unrealized_pnl': 75.50
            },
            'market': {
                'btc_price': 50100,
                'eth_price': 3050,
                'market_trend': 'bullish'
            },
            'alerts': {
                'critical': 0,
                'warnings': 2,
                'info': 5
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache for 60 seconds
        cache.set('dashboard:overview', overview, ttl=60)
        return jsonify(overview)
        
    except Exception as e:
        logger.error(f"Error generating dashboard overview: {e}")
        return jsonify({'error': 'Failed to generate overview'}), 500


@app.route('/api/dashboard/charts/performance', methods=['GET'])
@require_auth
def get_performance_chart_data():
    """Get performance chart data"""
    days = request.args.get('days', 7, type=int)
    
    try:
        db = PaperTradeDB()
        trades = db.get_all_trades()
        
        # Generate mock time series data for the chart
        from datetime import timedelta
        import random
        
        chart_data = []
        cumulative_pnl = 0
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i-1)
            
            # Mock daily PnL (some randomness but generally trending up)
            daily_pnl = random.uniform(-50, 100) if i > 0 else 0
            cumulative_pnl += daily_pnl
            
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'daily_pnl': round(daily_pnl, 2),
                'cumulative_pnl': round(cumulative_pnl, 2),
                'trades_count': random.randint(0, 5)
            })
        
        return jsonify({
            'data': chart_data,
            'period': f'{days} days',
            'total_return': round(cumulative_pnl, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating performance chart: {e}")
        return jsonify({'error': 'Failed to generate chart data'}), 500


@app.route('/api/dashboard/alerts', methods=['GET'])
@require_auth
def get_dashboard_alerts():
    """Get dashboard alerts and notifications"""
    # Mock alert data
    alerts = [
        {
            'id': 1,
            'type': 'warning',
            'title': 'High Market Volatility',
            'message': 'BTC volatility is above 5% in the last hour',
            'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat(),
            'read': False
        },
        {
            'id': 2,
            'type': 'info',
            'title': 'Strategy Performance',
            'message': 'Ensemble strategy showing positive returns today',
            'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
            'read': False
        },
        {
            'id': 3,
            'type': 'success',
            'title': 'Trade Executed',
            'message': 'Successfully executed BUY order for 0.05 BTC',
            'timestamp': (datetime.now() - timedelta(hours=4)).isoformat(),
            'read': True
        }
    ]
    
    return jsonify({
        'alerts': alerts,
        'unread_count': len([a for a in alerts if not a['read']]),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/dashboard/alerts/<int:alert_id>/read', methods=['POST'])
@require_auth
def mark_alert_read(alert_id: int):
    """Mark an alert as read"""
    # In a real implementation, this would update the alert status in database
    return jsonify({
        'message': f'Alert {alert_id} marked as read',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/dashboard/websocket/clients', methods=['GET'])
@require_auth
def get_websocket_clients():
    """Get information about connected WebSocket clients"""
    clients_info = get_connected_clients_info()
    return jsonify(clients_info)


@app.route('/api/dashboard/broadcast/alert', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def broadcast_dashboard_alert():
    """Broadcast alert to all connected dashboard clients"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    alert_type = data.get('type', 'info')
    message = data.get('message', '')
    severity = data.get('severity', 'info')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    # Broadcast to all connected clients
    broadcast_alert(alert_type, message, severity)
    
    return jsonify({
        'message': 'Alert broadcasted successfully',
        'type': alert_type,
        'severity': severity
    })


# Multi-exchange endpoints
@app.route('/api/exchanges', methods=['GET'])
@require_auth
def get_exchanges():
    """Get information about all configured exchanges"""
    try:
        # This would normally get from the exchange manager
        # For now, return mock data
        exchanges = {
            'binance': {
                'name': 'Binance',
                'status': 'healthy',
                'supported_symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
                'fees': {'maker': 0.001, 'taker': 0.001},
                'last_check': datetime.now().isoformat()
            },
            'kraken': {
                'name': 'Kraken',
                'status': 'healthy',
                'supported_symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
                'fees': {'maker': 0.0016, 'taker': 0.0026},
                'last_check': datetime.now().isoformat()
            },
            'coinbase': {
                'name': 'Coinbase',
                'status': 'healthy',
                'supported_symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
                'fees': {'maker': 0.005, 'taker': 0.005},
                'last_check': datetime.now().isoformat()
            }
        }
        
        return jsonify({
            'exchanges': exchanges,
            'total_exchanges': len(exchanges),
            'healthy_exchanges': len([e for e in exchanges.values() if e['status'] == 'healthy']),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting exchanges info: {e}")
        return jsonify({'error': 'Failed to get exchanges info'}), 500


@app.route('/api/exchanges/prices/<symbol>', methods=['GET'])
@require_auth
def get_multi_exchange_prices(symbol: str):
    """Get prices from all exchanges for a symbol"""
    try:
        # Mock multi-exchange prices
        import random
        base_price = 50000 if symbol == 'BTCUSDT' else 3000
        
        prices = {}
        for exchange in ['binance', 'kraken', 'coinbase']:
            # Add some variation to simulate real price differences
            variation = random.uniform(-0.01, 0.01)  # ±1%
            price = base_price * (1 + variation)
            prices[exchange] = round(price, 2)
        
        # Calculate spread
        min_price = min(prices.values())
        max_price = max(prices.values())
        spread = max_price - min_price
        spread_pct = (spread / min_price) * 100
        
        return jsonify({
            'symbol': symbol,
            'prices': prices,
            'min_price': min_price,
            'max_price': max_price,
            'avg_price': round(sum(prices.values()) / len(prices), 2),
            'spread': round(spread, 2),
            'spread_percentage': round(spread_pct, 4),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting multi-exchange prices: {e}")
        return jsonify({'error': 'Failed to get prices'}), 500


@app.route('/api/arbitrage/opportunities', methods=['GET'])
@require_auth
def get_arbitrage_opportunities():
    """Get current arbitrage opportunities"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        
        # Mock arbitrage opportunities
        import random
        opportunities = []
        
        # Simulate some arbitrage opportunities
        if random.random() > 0.7:  # 30% chance of having opportunities
            base_price = 50000 if symbol == 'BTCUSDT' else 3000
            
            for i in range(random.randint(1, 3)):
                buy_price = base_price * (1 + random.uniform(-0.005, 0.005))
                sell_price = buy_price * (1 + random.uniform(0.006, 0.015))  # Profitable spread
                
                exchanges = ['binance', 'kraken', 'coinbase']
                buy_exchange = random.choice(exchanges)
                sell_exchange = random.choice([e for e in exchanges if e != buy_exchange])
                
                opportunities.append({
                    'symbol': symbol,
                    'buy_exchange': buy_exchange,
                    'sell_exchange': sell_exchange,
                    'buy_price': round(buy_price, 2),
                    'sell_price': round(sell_price, 2),
                    'profit_percentage': round(((sell_price - buy_price) / buy_price) * 100, 3),
                    'profit_absolute': round(sell_price - buy_price, 2),
                    'timestamp': datetime.now().isoformat()
                })
        
        return jsonify({
            'symbol': symbol,
            'opportunities': opportunities,
            'count': len(opportunities),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting arbitrage opportunities: {e}")
        return jsonify({'error': 'Failed to get arbitrage opportunities'}), 500


@app.route('/api/portfolio/consolidated', methods=['GET'])
@require_auth
def get_consolidated_portfolio():
    """Get consolidated portfolio across all exchanges"""
    try:
        # Mock consolidated portfolio
        portfolio = {
            'total_value_usd': 12500.75,
            'balances': {
                'BTC': {
                    'total_balance': 0.25,
                    'total_free': 0.23,
                    'total_locked': 0.02,
                    'exchanges': {
                        'binance': {'free': 0.15, 'locked': 0.01, 'total': 0.16},
                        'kraken': {'free': 0.08, 'locked': 0.01, 'total': 0.09}
                    }
                },
                'ETH': {
                    'total_balance': 2.5,
                    'total_free': 2.3,
                    'total_locked': 0.2,
                    'exchanges': {
                        'binance': {'free': 1.5, 'locked': 0.1, 'total': 1.6},
                        'coinbase': {'free': 0.8, 'locked': 0.1, 'total': 0.9}
                    }
                },
                'USDT': {
                    'total_balance': 5000.0,
                    'total_free': 4800.0,
                    'total_locked': 200.0,
                    'exchanges': {
                        'binance': {'free': 3000.0, 'locked': 100.0, 'total': 3100.0},
                        'kraken': {'free': 1800.0, 'locked': 100.0, 'total': 1900.0}
                    }
                }
            },
            'exchange_count': 3,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(portfolio)
        
    except Exception as e:
        logger.error(f"Error getting consolidated portfolio: {e}")
        return jsonify({'error': 'Failed to get consolidated portfolio'}), 500


@app.route('/api/exchanges/health', methods=['GET'])
@require_auth
def get_exchanges_health():
    """Get health status of all exchanges"""
    try:
        # Mock exchange health data
        health = {
            'binance': {
                'status': 'healthy',
                'last_check': datetime.now().isoformat(),
                'error_count': 0,
                'response_time_ms': 120,
                'api_calls_remaining': 1200
            },
            'kraken': {
                'status': 'healthy',
                'last_check': datetime.now().isoformat(),
                'error_count': 1,
                'response_time_ms': 250,
                'api_calls_remaining': 800
            },
            'coinbase': {
                'status': 'healthy',
                'last_check': datetime.now().isoformat(),
                'error_count': 0,
                'response_time_ms': 180,
                'api_calls_remaining': 950
            }
        }
        
        return jsonify({
            'health': health,
            'summary': {
                'total_exchanges': len(health),
                'healthy_count': len([h for h in health.values() if h['status'] == 'healthy']),
                'avg_response_time': round(sum(h['response_time_ms'] for h in health.values()) / len(health), 1)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting exchanges health: {e}")
        return jsonify({'error': 'Failed to get exchanges health'}), 500


# Advanced Orders API Endpoints
@app.route('/api/orders/advanced', methods=['POST'])
@require_auth
@limiter.limit("50 per hour")
def submit_advanced_order():
    """Submit an advanced order (OCO, Iceberg, or TWAP)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        order_type = data.get('order_type')
        if not order_type:
            return jsonify({'error': 'order_type is required'}), 400
        
        # Get the advanced order manager from bootstrap
        from core.bootstrap import get_component
        advanced_order_manager = get_component('advanced_order_manager')
        
        if not advanced_order_manager:
            return jsonify({'error': 'Advanced order manager not available'}), 503
        
        # Route to appropriate order creation method
        if order_type.lower() == 'oco':
            result = _create_oco_order(advanced_order_manager, data)
        elif order_type.lower() == 'iceberg':
            result = _create_iceberg_order(advanced_order_manager, data)
        elif order_type.lower() == 'twap':
            result = _create_twap_order(advanced_order_manager, data)
        else:
            return jsonify({'error': f'Unsupported order type: {order_type}'}), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error submitting advanced order: {e}")
        return jsonify({'error': 'Failed to submit advanced order'}), 500


@app.route('/api/orders/advanced/<order_id>', methods=['GET'])
@require_auth
def get_advanced_order_status(order_id):
    """Get status of an advanced order"""
    try:
        from core.bootstrap import get_component
        advanced_order_manager = get_component('advanced_order_manager')
        
        if not advanced_order_manager:
            return jsonify({'error': 'Advanced order manager not available'}), 503
        
        result = advanced_order_manager.get_order_status(order_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting order status: {e}")
        return jsonify({'error': 'Failed to get order status'}), 500


@app.route('/api/orders/advanced/<order_id>', methods=['DELETE'])
@require_auth
@limiter.limit("100 per hour")
def cancel_advanced_order(order_id):
    """Cancel an advanced order"""
    try:
        from core.bootstrap import get_component
        advanced_order_manager = get_component('advanced_order_manager')
        
        if not advanced_order_manager:
            return jsonify({'error': 'Advanced order manager not available'}), 503
        
        # This would need to be made async in a real implementation
        import asyncio
        if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop():
            # If we're already in an async context
            result = advanced_order_manager.cancel_order(order_id)
        else:
            # Create new event loop
            result = asyncio.run(advanced_order_manager.cancel_order(order_id))
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error cancelling advanced order: {e}")
        return jsonify({'error': 'Failed to cancel order'}), 500


@app.route('/api/orders/advanced', methods=['GET'])
@require_auth
def get_all_advanced_orders():
    """Get all active advanced orders"""
    try:
        from core.bootstrap import get_component
        advanced_order_manager = get_component('advanced_order_manager')
        
        if not advanced_order_manager:
            return jsonify({'error': 'Advanced order manager not available'}), 503
        
        # Optional filters
        order_type = request.args.get('type')
        symbol = request.args.get('symbol')
        
        if order_type:
            orders = advanced_order_manager.get_orders_by_type(order_type)
        elif symbol:
            orders = advanced_order_manager.get_orders_by_symbol(symbol.upper())
        else:
            orders = advanced_order_manager.get_all_active_orders()
        
        return jsonify({
            'orders': orders,
            'count': len(orders),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting advanced orders: {e}")
        return jsonify({'error': 'Failed to get orders'}), 500


@app.route('/api/orders/advanced/statistics', methods=['GET'])
@require_auth
def get_advanced_order_statistics():
    """Get advanced order execution statistics"""
    try:
        from core.bootstrap import get_component
        advanced_order_manager = get_component('advanced_order_manager')
        
        if not advanced_order_manager:
            return jsonify({'error': 'Advanced order manager not available'}), 503
        
        stats = advanced_order_manager.get_statistics()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting order statistics: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500


def _create_oco_order(manager, data):
    """Create OCO order from request data"""
    required_fields = ['symbol', 'side', 'quantity', 'limit_price', 'stop_price']
    
    for field in required_fields:
        if field not in data:
            return {'success': False, 'error': f'Missing required field: {field}'}
    
    import asyncio
    try:
        if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop():
            result = manager.create_oco_order(
                symbol=data['symbol'].upper(),
                side=data['side'].upper(),
                quantity=float(data['quantity']),
                limit_price=float(data['limit_price']),
                stop_price=float(data['stop_price']),
                stop_limit_price=data.get('stop_limit_price'),
                exchange=data.get('exchange', 'binance')
            )
        else:
            result = asyncio.run(manager.create_oco_order(
                symbol=data['symbol'].upper(),
                side=data['side'].upper(),
                quantity=float(data['quantity']),
                limit_price=float(data['limit_price']),
                stop_price=float(data['stop_price']),
                stop_limit_price=data.get('stop_limit_price'),
                exchange=data.get('exchange', 'binance')
            ))
        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _create_iceberg_order(manager, data):
    """Create Iceberg order from request data"""
    required_fields = ['symbol', 'side', 'quantity', 'price', 'chunk_size']
    
    for field in required_fields:
        if field not in data:
            return {'success': False, 'error': f'Missing required field: {field}'}
    
    import asyncio
    try:
        if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop():
            result = manager.create_iceberg_order(
                symbol=data['symbol'].upper(),
                side=data['side'].upper(),
                quantity=float(data['quantity']),
                price=float(data['price']),
                chunk_size=float(data['chunk_size']),
                chunk_variance=data.get('chunk_variance', 0.0),
                delay_between_chunks=data.get('delay_between_chunks', 0),
                exchange=data.get('exchange', 'binance')
            )
        else:
            result = asyncio.run(manager.create_iceberg_order(
                symbol=data['symbol'].upper(),
                side=data['side'].upper(),
                quantity=float(data['quantity']),
                price=float(data['price']),
                chunk_size=float(data['chunk_size']),
                chunk_variance=data.get('chunk_variance', 0.0),
                delay_between_chunks=data.get('delay_between_chunks', 0),
                exchange=data.get('exchange', 'binance')
            ))
        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _create_twap_order(manager, data):
    """Create TWAP order from request data"""
    required_fields = ['symbol', 'side', 'quantity', 'duration_minutes']
    
    for field in required_fields:
        if field not in data:
            return {'success': False, 'error': f'Missing required field: {field}'}
    
    import asyncio
    try:
        if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop():
            result = manager.create_twap_order(
                symbol=data['symbol'].upper(),
                side=data['side'].upper(),
                quantity=float(data['quantity']),
                duration_minutes=int(data['duration_minutes']),
                interval_minutes=data.get('interval_minutes', 5),
                price_limit=data.get('price_limit'),
                participation_rate=data.get('participation_rate', 0.1),
                exchange=data.get('exchange', 'binance')
            )
        else:
            result = asyncio.run(manager.create_twap_order(
                symbol=data['symbol'].upper(),
                side=data['side'].upper(),
                quantity=float(data['quantity']),
                duration_minutes=int(data['duration_minutes']),
                interval_minutes=data.get('interval_minutes', 5),
                price_limit=data.get('price_limit'),
                participation_rate=data.get('participation_rate', 0.1),
                exchange=data.get('exchange', 'binance')
            ))
        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}


# Helper functions
def _calculate_uptime() -> Optional[str]:
    """Calculate bot uptime"""
    if not bot_state['start_time']:
        return None
    
    start = datetime.fromisoformat(bot_state['start_time'])
    uptime = datetime.now() - start
    
    days = uptime.days
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{days}d {hours}h {minutes}m {seconds}s"


if __name__ == '__main__':
    # Development server with WebSocket support
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
