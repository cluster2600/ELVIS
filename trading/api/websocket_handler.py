"""
WebSocket handler for real-time dashboard updates
Provides real-time streaming of trading data, market prices, and bot status
"""

from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import logging
import threading
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import jwt
from functools import wraps

from utils.redis_cache import get_cache
from utils.paper_trade_db import PaperTradeDB
from utils.logger_config import get_logger

logger = get_logger(__name__)

# Global SocketIO instance
socketio = None
background_task = None
connected_clients = {}


def init_websocket(app: Flask, secret_key: str) -> SocketIO:
    """Initialize WebSocket support for the Flask app"""
    global socketio
    
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=False,
        engineio_logger=False
    )
    
    # Set up event handlers
    setup_event_handlers(secret_key)
    
    # Start background task for real-time updates
    start_background_task()
    
    logger.info("WebSocket handler initialized")
    return socketio


def setup_event_handlers(secret_key: str):
    """Set up WebSocket event handlers"""
    
    def require_ws_auth(f):
        """Decorator to require authentication for WebSocket events"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = None
            
            # Try to get token from different sources
            if len(args) > 0 and isinstance(args[0], dict):
                token = args[0].get('token')
            
            if not token:
                emit('error', {'message': 'Authentication required'})
                return
            
            try:
                # Verify token
                payload = jwt.decode(token, secret_key, algorithms=['HS256'])
                # Add user info to the function call
                return f(payload, *args, **kwargs)
                
            except jwt.ExpiredSignatureError:
                emit('error', {'message': 'Token has expired'})
                return
            except jwt.InvalidTokenError:
                emit('error', {'message': 'Invalid token'})
                return
        
        return decorated_function
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        client_id = request.sid if hasattr(request, 'sid') else 'unknown'
        logger.info(f"Client connected: {client_id}")
        emit('connected', {'message': 'Connected to ELVIS dashboard'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        client_id = request.sid if hasattr(request, 'sid') else 'unknown'
        if client_id in connected_clients:
            del connected_clients[client_id]
        logger.info(f"Client disconnected: {client_id}")
    
    @socketio.on('authenticate')
    @require_ws_auth
    def handle_authenticate(user_payload, data):
        """Authenticate client and join rooms"""
        client_id = request.sid if hasattr(request, 'sid') else 'unknown'
        connected_clients[client_id] = {
            'user': user_payload.get('user'),
            'connected_at': datetime.now().isoformat(),
            'subscriptions': []
        }
        
        emit('authenticated', {
            'user': user_payload.get('user'),
            'message': 'Authentication successful'
        })
        logger.info(f"Client authenticated: {client_id} - {user_payload.get('user')}")
    
    @socketio.on('subscribe')
    @require_ws_auth
    def handle_subscribe(user_payload, data):
        """Subscribe to real-time data streams"""
        channels = data.get('channels', [])
        client_id = request.sid if hasattr(request, 'sid') else 'unknown'
        
        valid_channels = [
            'market_data', 'bot_status', 'trades', 'positions', 
            'performance', 'alerts', 'orders'
        ]
        
        subscribed = []
        for channel in channels:
            if channel in valid_channels:
                join_room(channel)
                subscribed.append(channel)
        
        if client_id in connected_clients:
            connected_clients[client_id]['subscriptions'] = subscribed
        
        emit('subscribed', {
            'channels': subscribed,
            'message': f'Subscribed to {len(subscribed)} channels'
        })
        logger.info(f"Client {client_id} subscribed to: {subscribed}")
    
    @socketio.on('unsubscribe')
    @require_ws_auth
    def handle_unsubscribe(user_payload, data):
        """Unsubscribe from data streams"""
        channels = data.get('channels', [])
        client_id = request.sid if hasattr(request, 'sid') else 'unknown'
        
        unsubscribed = []
        for channel in channels:
            leave_room(channel)
            unsubscribed.append(channel)
        
        if client_id in connected_clients:
            current_subs = connected_clients[client_id].get('subscriptions', [])
            connected_clients[client_id]['subscriptions'] = [
                sub for sub in current_subs if sub not in unsubscribed
            ]
        
        emit('unsubscribed', {
            'channels': unsubscribed,
            'message': f'Unsubscribed from {len(unsubscribed)} channels'
        })
        logger.info(f"Client {client_id} unsubscribed from: {unsubscribed}")
    
    @socketio.on('get_status')
    @require_ws_auth
    def handle_get_status(user_payload, data):
        """Get current bot status"""
        status_data = get_real_time_status()
        emit('bot_status', status_data)
    
    @socketio.on('get_market_data')
    @require_ws_auth
    def handle_get_market_data(user_payload, data):
        """Get current market data"""
        symbol = data.get('symbol', 'BTCUSDT')
        market_data = get_real_time_market_data(symbol)
        emit('market_data', market_data)


def start_background_task():
    """Start background task for real-time data broadcasting"""
    global background_task
    
    if background_task is None:
        background_task = threading.Thread(target=background_data_stream)
        background_task.daemon = True
        background_task.start()
        logger.info("Background data streaming task started")


def background_data_stream():
    """Background task to stream real-time data to connected clients"""
    while True:
        try:
            if socketio and len(connected_clients) > 0:
                # Broadcast bot status
                status_data = get_real_time_status()
                socketio.emit('bot_status_update', status_data, room='bot_status')
                
                # Broadcast market data for BTC only (user requested BTC-only trading)
                symbols = ['BTCUSDT']
                for symbol in symbols:
                    market_data = get_real_time_market_data(symbol)
                    socketio.emit('market_data_update', market_data, room='market_data')
                
                # Broadcast trade updates
                trade_updates = get_recent_trades()
                if trade_updates:
                    socketio.emit('trades_update', trade_updates, room='trades')
                
                # Broadcast position updates
                position_updates = get_position_updates()
                socketio.emit('positions_update', position_updates, room='positions')
                
                # Broadcast performance updates
                performance_data = get_performance_updates()
                socketio.emit('performance_update', performance_data, room='performance')
            
            time.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logger.error(f"Error in background data stream: {e}")
            time.sleep(5)  # Wait longer on error


def get_real_time_status() -> Dict[str, Any]:
    """Get real-time bot status"""
    cache = get_cache()
    
    # Try to get from cache first
    cached_status = cache.get('dashboard:bot_status')
    if cached_status:
        return cached_status
    
    # Generate current status
    status = {
        'running': True,  # Mock data - replace with actual bot status
        'mode': 'paper',
        'strategy': 'ensemble',
        'uptime': '2h 15m 32s',
        'last_update': datetime.now().isoformat(),
        'health': 'healthy',
        'errors': 0,
        'warnings': 2,
        'memory_usage': 45.6,
        'cpu_usage': 12.3,
        'api_calls_remaining': 1850
    }
    
    # Cache for 30 seconds
    cache.set('dashboard:bot_status', status, ttl=30)
    return status


def get_real_time_market_data(symbol: str) -> Dict[str, Any]:
    """Get real-time market data for a symbol"""
    cache = get_cache()
    
    cache_key = f'dashboard:market:{symbol}'
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data
    
    # Mock real-time market data
    import random
    base_prices = {'BTCUSDT': 50000}
    base_price = base_prices.get(symbol, 1000)
    
    # Add some random fluctuation
    price_change = random.uniform(-0.02, 0.02)  # ±2%
    current_price = base_price * (1 + price_change)
    
    market_data = {
        'symbol': symbol,
        'price': round(current_price, 2),
        'price_change_24h': round(price_change * 100, 2),
        'volume_24h': random.randint(10000, 100000),
        'high_24h': round(current_price * 1.05, 2),
        'low_24h': round(current_price * 0.95, 2),
        'bid': round(current_price * 0.999, 2),
        'ask': round(current_price * 1.001, 2),
        'timestamp': datetime.now().isoformat(),
        'indicators': {
            'rsi': round(random.uniform(30, 70), 1),
            'macd': round(random.uniform(-50, 50), 2),
            'sma_20': round(current_price * 0.98, 2),
            'sma_50': round(current_price * 0.96, 2)
        }
    }
    
    # Cache for 10 seconds
    cache.set(cache_key, market_data, ttl=10)
    return market_data


def get_recent_trades() -> List[Dict[str, Any]]:
    """Get recent trade updates"""
    try:
        db = PaperTradeDB()
        trades = db.get_all_trades()
        
        # Return only the most recent trades (last 5)
        recent_trades = sorted(trades, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
        
        return {
            'trades': recent_trades,
            'count': len(recent_trades),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching recent trades: {e}")
        return {'trades': [], 'count': 0, 'timestamp': datetime.now().isoformat()}


def get_position_updates() -> Dict[str, Any]:
    """Get current position updates"""
    # Mock position data
    positions = [
        {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'quantity': 0.05,
            'entry_price': 49800,
            'current_price': 50100,
            'pnl': 15.0,
            'pnl_percentage': 0.6,
            'unrealized_pnl': 15.0,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    return {
        'positions': positions,
        'total_pnl': sum(p.get('pnl', 0) for p in positions),
        'total_unrealized': sum(p.get('unrealized_pnl', 0) for p in positions),
        'position_count': len(positions),
        'timestamp': datetime.now().isoformat()
    }


def get_performance_updates() -> Dict[str, Any]:
    """Get performance metrics updates"""
    try:
        db = PaperTradeDB()
        trades = db.get_all_trades()
        
        # Calculate basic performance metrics
        total_trades = len(trades)
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        
        performance = {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'avg_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'best_trade': max((t.get('pnl', 0) for t in trades), default=0),
            'worst_trade': min((t.get('pnl', 0) for t in trades), default=0),
            'daily_pnl': total_pnl,  # Simplified - should be today's PnL
            'weekly_pnl': total_pnl,  # Simplified - should be this week's PnL
            'timestamp': datetime.now().isoformat()
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'timestamp': datetime.now().isoformat()
        }


def broadcast_alert(alert_type: str, message: str, severity: str = 'info'):
    """Broadcast alert to all connected clients"""
    if socketio:
        alert_data = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('alert', alert_data, room='alerts')
        logger.info(f"Alert broadcasted: {alert_type} - {message}")


def broadcast_trade_execution(trade_data: Dict[str, Any]):
    """Broadcast trade execution to connected clients"""
    if socketio:
        socketio.emit('trade_executed', trade_data, room='trades')
        logger.info(f"Trade execution broadcasted: {trade_data.get('symbol')} - {trade_data.get('side')}")


def get_connected_clients_info() -> Dict[str, Any]:
    """Get information about connected clients"""
    return {
        'total_clients': len(connected_clients),
        'clients': connected_clients,
        'timestamp': datetime.now().isoformat()
    }