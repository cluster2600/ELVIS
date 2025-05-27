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

logger = get_logger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Register Swagger UI blueprint
app.register_blueprint(swaggerui_blueprint)

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
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)
