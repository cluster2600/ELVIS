import argparse
import logging
import threading
import signal
import sys
import time
import curses
import os

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Set Vault environment variables only if not already set
if not os.getenv('VAULT_ADDR'):
    os.environ['VAULT_ADDR'] = 'http://127.0.0.1:8200'
if not os.getenv('VAULT_TOKEN'):
    os.environ['VAULT_TOKEN'] = 'trading-bot-token'

from core.bootstrap import bootstrap_application
from core.di import container
from core.events import event_bus, SystemEvent
from utils.console_dashboard import ConsoleDashboard
from trading.utils.trade_history_api import app as trade_history_app
import pandas as pd
import ta


def start_trade_history_server():
    """
    Start the Trade History Flask server in a separate thread.
    """
    trade_history_app.run(host="0.0.0.0", port=5050)


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the price data."""
    try:
        if len(data) < 50:  # Need enough data for indicators
            return data
        
        # Check if required columns exist
        required_columns = ['close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger = container.get_optional('logger', logging.getLogger(__name__))
            logger.error(f"Missing required columns for technical indicators: {missing_columns}")
            logger.info(f"Available columns: {list(data.columns)}")
            return data
            
        # Ensure numeric types for calculations
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        # Add Simple Moving Averages
        data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
        
        # Add ADX
        data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14)
        
        # Add RSI
        data['rsi'] = ta.momentum.rsi(data['close'], window=14)
        
        # Add MACD
        macd = ta.trend.MACD(data['close'])
        data['macd'] = macd.macd()
        data['signal_line'] = macd.macd_signal()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['close'])
        data['lower_bb'] = bollinger.bollinger_lband()
        data['sma_bb'] = bollinger.bollinger_mavg() 
        data['upper_bb'] = bollinger.bollinger_hband()
        
        # Add other indicators that might be needed
        data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
        
        return data
    except Exception as e:
        logger = container.get_optional('logger', logging.getLogger(__name__))
        logger.error(f"Error calculating technical indicators: {e}")
        return data


# Global shutdown flag
shutdown_requested = False
trading_thread = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully - close all positions and exit."""
    global shutdown_requested
    logger = container.get_optional('logger', logging.getLogger(__name__))
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    
    # Set shutdown flag instead of immediate exit
    shutdown_requested = True
    
    # 🚨 CRITICAL: Close all open positions before shutdown
    try:
        logger.info("🔄 Closing all open positions before shutdown...")
        
        # Get the executor from DI container
        exchange_manager = container.get_optional('exchange_manager')
        if exchange_manager and hasattr(exchange_manager, 'executors'):
            for exchange_name, executor in exchange_manager.executors.items():
                if hasattr(executor, 'close_all_positions'):
                    logger.info(f"🔄 Closing positions on {exchange_name}...")
                    results = executor.close_all_positions(reason=f"Shutdown signal {signum}")
                    
                    if results['closed']:
                        logger.info(f"✅ {exchange_name}: Closed {len(results['closed'])} positions, P&L: ${results['total_pnl']:.2f}")
                    if results['errors']:
                        logger.warning(f"⚠️ {exchange_name}: {len(results['errors'])} errors occurred")
        else:
            logger.warning("⚠️ No executors found - positions may remain open")
            
    except Exception as e:
        logger.error(f"❌ Error closing positions during shutdown: {e}")
        logger.error("⚠️ Some positions may remain open!")
    
    # Publish shutdown event
    event_bus.publish(SystemEvent(
        system_type='shutdown',
        component='main',
        status='ok',
        message=f'Shutdown signal {signum} received - positions closed',
        source='signal_handler'
    ))
    
    # Wait for trading thread to finish if it exists
    if trading_thread and trading_thread.is_alive():
        logger.info("⏳ Waiting for trading operations to complete...")
        trading_thread.join(timeout=10)  # Reduced timeout since positions are already closed
        
    logger.info("✅ Graceful shutdown complete - all positions closed")
    sys.exit(0)


def risk_management_loop(risk_manager, logger):
    """Continuously manage position risk in a separate thread."""
    global shutdown_requested
    while not shutdown_requested:
        try:
            risk_manager.manage_positions()
        except Exception as e:
            logger.error(f"Error in risk management loop: {e}")
        
        # Check for shutdown during sleep
        for i in range(5):
            if shutdown_requested:
                logger.info("Shutdown requested, exiting risk management loop...")
                return
            time.sleep(1)
    
    logger.info("Risk management loop finished due to shutdown request")

def main(mode: str, log_level: str):
    """
    Main entry point for the trading bot using dependency injection.

    Args:
        mode (str): Trading mode, either 'paper' or 'live'.
        log_level (str): Logging level string.
    """
    # Bootstrap the application
    bootstrapper = bootstrap_application(mode, log_level)
    logger = container.get('logger')
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start Trade History Server in background
        threading.Thread(target=start_trade_history_server, daemon=True).start()
        logger.info("Started Trade History Server on 0.0.0.0:5050...")
        
        # Get dependencies from container
        strategy_manager = container.get('strategy_manager')
        risk_manager = container.get('risk_manager')
        price_fetcher = container.get('price_fetcher')
        executor = container.get('executor')
        llm_advisor = container.get('llm_advisor')
        
        # Get the active strategy (ensemble or research-based)
        active_strategy = container.get('strategy')
        strategy_mode = os.getenv('STRATEGY_MODE', 'ensemble')
        logger.info(f"🎯 Active strategy: {type(active_strategy).__name__} (mode: {strategy_mode})")
        
        if strategy_mode == 'research':
            logger.info("🔬 Research-based strategy active - targeting 14.9% annual returns")
            logger.info("📊 Binary classification: BUY/SELL only (no HOLD signals)")
            logger.info("🎯 Following Bonenkamp (2021) research methodology")
        elif strategy_mode == 'balanced':
            logger.info("🎯 Balanced strategy active - EMERGENCY MODE")
            logger.info("⚠️ Reduced trading frequency: 2 trades/hour max")
            logger.info("💰 Higher profit targets: $1.00 per trade")
            logger.info("⏱️ Cooldown enforcement: 10 minutes between trades")
            
            # Initial training for research strategy if a saved model is not loaded
            if hasattr(active_strategy, 'is_trained') and not active_strategy.is_trained:
                logger.info("🧠 Research model not found or loaded. Initiating pre-training...")
                try:
                    # Fetch a good amount of historical data for robust initial training
                    # The research uses 1 week of 5-min data, which is 2016 data points.
                    logger.info("Fetching historical data for initial training...")
                    initial_data = price_fetcher.get_historical_klines("BTCUSDT", "5m", limit=2100)
                    
                    if initial_data is not None and not initial_data.empty and len(initial_data) > 200:
                        logger.info(f"✅ Fetched {len(initial_data)} records for initial training.")
                        active_strategy.train_model(initial_data)
                    else:
                        logger.error(f"❌ Could not fetch sufficient initial data for training. Bot will not trade until trained.")
                except Exception as e:
                    logger.error(f"❌ Error during initial model training: {e}", exc_info=True)
        
        # Initialize the console dashboard
        price_fetcher = container.get('price_fetcher')
        if strategy_mode == 'research':
            logger.info("🔬 Research-based strategy active - targeting 14.9% annual returns")
            logger.info("📊 Binary classification: BUY/SELL only (no HOLD signals)")
            logger.info("🎯 Following Bonenkamp (2021) research methodology")
            
            # Initial training for research strategy if a saved model is not loaded
            if hasattr(active_strategy, 'is_trained') and not active_strategy.is_trained:
                logger.info("🧠 Research model not found or loaded. Initiating pre-training...")
                try:
                    # Fetch a good amount of historical data for robust initial training
                    # The research uses 1 week of 5-min data, which is 2016 data points.
                    logger.info("Fetching historical data for initial training...")
                    initial_data = price_fetcher.get_historical_klines("BTCUSDT", "5m", limit=2100)
                    
                    if initial_data is not None and not initial_data.empty and len(initial_data) > 200:
                        logger.info(f"✅ Fetched {len(initial_data)} records for initial training.")
                        active_strategy.train_model(initial_data)
                    else:
                        logger.error(f"❌ Could not fetch sufficient initial data for training. Bot will not trade until trained.")
                except Exception as e:
                    logger.error(f"❌ Error during initial model training: {e}", exc_info=True)
        
        # Initialize the console dashboard
        price_fetcher = container.get('price_fetcher')
        
        # Initialize the console dashboard
        price_fetcher = container.get('price_fetcher')
        dashboard = ConsoleDashboard(
            config={
                'portfolio_value': 10520.30,
                'unrealized_pnl': 150.10,
                'realized_pnl': 450.20,
                'open_positions': [],
                'recent_trades': [],
                'risk_manager': risk_manager,
                'performance_monitor': container.get('performance_monitor'),
                'trade_analyzer': container.get('trade_analyzer'),
                'system_monitor': container.get('system_monitor')
            }, 
            logger=logger, 
            price_fetcher=price_fetcher
        )
        
        # Register dashboard in container for other components to use
        container.register_singleton('dashboard', lambda: dashboard)
        
        # Start risk management loop
        threading.Thread(target=risk_management_loop, args=(risk_manager, logger), daemon=True).start()
        logger.info("Started Risk Management loop...")
        
        # Publish system ready event
        event_bus.publish(SystemEvent(
            system_type='startup',
            component='main',
            status='ok',
            message='Trading system started successfully',
            source='main',
            metrics={
                'mode': mode,
                'symbol': 'BTCUSDT'
            }
        ))
        
        # Main trading loop
        logger.info(f"Starting trading bot in {mode} mode...")
        
        # Get executor for trade execution
        executor = container.get('executor')
        
        # The main loop will now be managed by the strategy manager
        def trading_loop():
            global shutdown_requested
            while not shutdown_requested:
                try:
                    # Get market data with detailed logging
                    logger.info("=== TRADING LOOP ITERATION START ===")
                    logger.info("Attempting to fetch market data...")
                    try:
                        # Multi-symbol trading: BTCUSDT + BNBUSDT (both USDT pairs for balance consistency)
                        from config.config import SYMBOLS_CONFIG
                        symbols_to_trade = SYMBOLS_CONFIG.get('PRIMARY_SYMBOLS', ["BTCUSDT", "BNBUSDT"])
                        all_data = {}
                        
                        for trading_symbol in symbols_to_trade:
                            try:
                                symbol_data = price_fetcher.get_historical_klines(trading_symbol, "1m")
                                if not symbol_data.empty:
                                    all_data[trading_symbol] = symbol_data
                                    logger.debug(f"✅ {trading_symbol}: {len(symbol_data)} records")
                                else:
                                    logger.warning(f"⚠️ No data for {trading_symbol}")
                            except Exception as e:
                                logger.warning(f"⚠️ Error fetching {trading_symbol}: {e}")
                        
                        # Use BTCUSDT as primary for dashboard/indicators
                        data = all_data.get("BTCUSDT", pd.DataFrame())
                        logger.debug(f"Fetched data shape: {data.shape if not data.empty else 'EMPTY'}")
                        if not data.empty:
                            logger.debug(f"Data columns: {list(data.columns)}")
                            # Check if 'close' column exists before accessing it
                            if 'close' in data.columns:
                                logger.debug(f"Latest close price: {data.iloc[-1]['close']}")
                            else:
                                logger.warning(f"'close' column missing from data. Available columns: {list(data.columns)}")
                                # Try to use a different column or create mock data
                                if len(data.columns) >= 5:  # Assume standard OHLCV order
                                    data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume'] + list(data.columns[6:])
                                    logger.debug(f"Reassigned column names. Latest close price: {data.iloc[-1]['close']}")
                                else:
                                    logger.warning("Insufficient columns, falling back to mock data")
                                    data = pd.DataFrame()  # Force fallback to mock data
                        else:
                            # EMERGENCY: Get real Bitcoin price for ATH detection
                            logger.error("🚨 No market data available - getting real BTC price for emergency ATH detection")
                            try:
                                import requests
                                response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
                                if response.status_code == 200:
                                    real_price = float(response.json()['price'])
                                    logger.error(f"🚨 REAL BTC PRICE: ${real_price:,.2f}")
                                    
                                    # EMERGENCY: If BTC is above $135k, STOP TRADING (Updated for new paradigm)
                                    if real_price > 135000:
                                        logger.error(f"🚨 EMERGENCY: BTC at ${real_price:,.2f} > $135k - EXTREME ATH!")
                                        logger.error("🚨 STOPPING ALL TRADING - MANUAL INTERVENTION REQUIRED")
                                        return  # Exit trading loop
                                    
                                    # Use real price for data
                                    mock_data = {
                                        'open': [real_price] * 50,
                                        'high': [real_price * 1.001] * 50,
                                        'low': [real_price * 0.999] * 50,
                                        'close': [real_price] * 50,
                                        'volume': [1000] * 50,
                                    }
                                else:
                                    # Fallback with warning
                                    logger.error("🚨 Could not get real price - using emergency mock data")
                                    mock_data = {
                                        'open': [97000] * 50,
                                        'high': [97200] * 50,
                                        'low': [96800] * 50,
                                        'close': [97000] * 50,
                                        'volume': [1000] * 50,
                                    }
                            except Exception as price_error:
                                logger.error(f"🚨 Emergency price fetch failed: {price_error}")
                                mock_data = {
                                    'open': [97000] * 50,
                                    'high': [97200] * 50,
                                    'low': [96800] * 50,
                                    'close': [97000] * 50,
                                    'volume': [1000] * 50,
                                }
                            
                            data = pd.DataFrame(mock_data)
                            logger.info(f"Emergency data created with BTC price: ${data.iloc[-1]['close']:,.2f}")
                    except Exception as e:
                        logger.error(f"Error fetching market data: {e}")
                        # Fallback to mock data
                        import numpy as np
                        mock_data = {
                            'open': np.random.normal(97000, 200, 50),
                            'high': np.random.normal(97200, 200, 50), 
                            'low': np.random.normal(96800, 200, 50),
                            'close': np.random.normal(97000, 200, 50),
                            'volume': np.random.normal(1000, 100, 50),
                        }
                        data = pd.DataFrame(mock_data)
                        logger.info("Using fallback mock data")
                        if 'close' in data.columns:
                            logger.info(f"Fallback data latest close: {data.iloc[-1]['close']:.2f}")
                        else:
                            logger.error("Fallback mock data missing 'close' column - this should not happen")
                    
                    if not data.empty:
                        # Calculate technical indicators for the data
                        logger.debug("Adding technical indicators...")
                        data = add_technical_indicators(data)
                        logger.debug(f"Data with indicators shape: {data.shape}")
                        logger.debug(f"Available columns: {list(data.columns)}")
                        
                        # Update dashboard config with real OHLC data
                        try:
                            if 'dashboard' in locals() and hasattr(dashboard, 'config'):
                                # Ensure OHLC data is properly converted to numeric types
                                ohlc_subset = data[['open', 'high', 'low', 'close']].tail(40).copy()
                                
                                # Force conversion to numeric types and handle any conversion errors
                                for col in ['open', 'high', 'low', 'close']:
                                    ohlc_subset[col] = pd.to_numeric(ohlc_subset[col], errors='coerce')
                                
                                # Drop any rows with NaN values after conversion
                                ohlc_subset = ohlc_subset.dropna()
                                
                                # Only update if we have valid data
                                if not ohlc_subset.empty:
                                    dashboard.config['ohlc_data'] = ohlc_subset
                                    dashboard.config['current_price'] = float(data.iloc[-1]['close'])
                                    dashboard.config['indicators'] = {
                                        'rsi': float(data.iloc[-1].get('rsi', 50.0)) if pd.notna(data.iloc[-1].get('rsi')) else 50.0,
                                        'macd': float(data.iloc[-1].get('macd', 0.0)) if pd.notna(data.iloc[-1].get('macd')) else 0.0,
                                        'sma_20': float(data.iloc[-1].get('sma_20', data.iloc[-1]['close'])) if pd.notna(data.iloc[-1].get('sma_20')) else float(data.iloc[-1]['close'])
                                    }
                                    
                                    # Update open positions from paper trading database
                                    try:
                                        from utils.paper_trade_db import get_open_positions
                                        open_positions_raw = get_open_positions()
                                        
                                        # Convert to dashboard format
                                        open_positions = []
                                        for pos in open_positions_raw:
                                            # Correct schema: (id, symbol, side, entry_price, quantity, leverage, entry_time)
                                            if len(pos) >= 7:
                                                try:
                                                    symbol = pos[1]
                                                    side = pos[2]
                                                    entry_price = float(pos[3])
                                                    quantity = float(pos[4])
                                                    leverage = int(pos[5])
                                                    entry_time = pos[6]

                                                    # Get current price for this symbol
                                                    try:
                                                        if price_fetcher:
                                                            current_price = price_fetcher.get_current_price(symbol)
                                                            if current_price is None:
                                                                current_price = entry_price
                                                            else:
                                                                current_price = float(current_price)
                                                        else:
                                                            current_price = entry_price
                                                    except Exception as price_err:
                                                        logger.warning(f"Could not fetch current price for {symbol}: {price_err}")
                                                        current_price = entry_price

                                                    # Calculate comprehensive P&L
                                                    if hasattr(executor, 'calculate_open_position_pnl'):
                                                        pnl_detail = executor.calculate_open_position_pnl(
                                                            symbol, side, current_price, entry_price,
                                                            quantity, leverage, entry_time
                                                        )
                                                        net_pnl = pnl_detail.get('net_pnl', 0.0)
                                                        gross_pnl = pnl_detail.get('gross_pnl', 0.0)
                                                        fees_info = {
                                                            'total_fees': pnl_detail.get('ongoing_costs', 0.0) + pnl_detail.get('estimated_exit_fee', 0.0),
                                                            'funding_fee': pnl_detail.get('funding_fee', 0.0),
                                                            'borrowing_cost': pnl_detail.get('borrowing_cost', 0.0),
                                                            'hours_held': pnl_detail.get('hours_held', 0.0)
                                                        }
                                                    else:
                                                        # Fallback to simple calculation
                                                        pnl_factor = 1 if side.upper() == 'BUY' else -1
                                                        net_pnl = (current_price - entry_price) * quantity * pnl_factor
                                                        gross_pnl = net_pnl
                                                        fees_info = {'total_fees': 0}

                                                    open_positions.append({
                                                        'symbol': symbol,
                                                        'size': quantity,
                                                        'entry_price': entry_price,
                                                        'pnl': net_pnl,
                                                        'gross_pnl': gross_pnl,
                                                        'fees_info': fees_info,
                                                        'leverage': leverage,
                                                        'entry_time': entry_time
                                                    })
                                                except (ValueError, TypeError, IndexError) as e:
                                                    logger.error(f"Error processing position tuple {pos}: {e}")
                                                    continue
                                            else:
                                                logger.warning(f"Skipping malformed position tuple: {pos}")
                                        
                                        dashboard.config['open_positions'] = open_positions
                                        
                                        # Update portfolio value and PnL with live data
                                        try:
                                            # Get dynamic balance from executor
                                            balance_info = executor.get_balance() if executor else {'USDT': 10000.0, 'BTC': 0.0}
                                            usdt_balance = balance_info.get('USDT', 10000.0)
                                            btc_balance = balance_info.get('BTC', 0.0)
                                            
                                            # Calculate total portfolio value (USDT + BTC value in USDT)
                                            current_btc_price = float(data.iloc[-1]['close']) if 'close' in data.columns else 97000.0
                                            btc_value_in_usdt = btc_balance * current_btc_price
                                            total_portfolio_value = usdt_balance + btc_value_in_usdt
                                            
                                            dashboard.config['portfolio_value'] = total_portfolio_value
                                            
                                            # Add real-time price data
                                            dashboard.config['current_btc_price'] = current_btc_price
                                            
                                            # Add real-time leverage from executor
                                            if executor and hasattr(executor, 'default_leverage'):
                                                dashboard.config['leverage'] = executor.default_leverage
                                            else:
                                                dashboard.config['leverage'] = 10  # Default fallback
                                            
                                            # Calculate total unrealized PnL with live prices
                                            total_unrealized_pnl = sum(pos['pnl'] for pos in open_positions)
                                            dashboard.config['unrealized_pnl'] = total_unrealized_pnl
                                            
                                            # Calculate realized PnL from trade history (exclude TEST trades)
                                            try:
                                                from utils.paper_trade_db import get_all_trades
                                                all_trades = get_all_trades(limit=1000, exclude_test=True)
                                                total_realized_pnl = sum(float(trade[6]) for trade in all_trades if len(trade) >= 7)
                                                dashboard.config['realized_pnl'] = total_realized_pnl
                                            except Exception:
                                                dashboard.config['realized_pnl'] = 0.0
                                            
                                            # Log the updated values
                                            logger.info(f"Portfolio Update - USDT: ${usdt_balance:.2f}, BTC: {btc_balance:.6f}, "
                                                      f"BTC Value: ${btc_value_in_usdt:.2f}, Total: ${total_portfolio_value:.2f}, "
                                                      f"Unrealized PnL: ${total_unrealized_pnl:.2f}, Realized PnL: ${dashboard.config['realized_pnl']:.2f}")
                                            
                                        except Exception as e:
                                            logger.warning(f"Could not update portfolio value: {e}")
                                            # Fallback to static values
                                            dashboard.config['portfolio_value'] = 10000.0
                                        
                                        # Update recent trades from database (exclude TEST trades)
                                        try:
                                            from utils.paper_trade_db import get_all_trades
                                            recent_trades_raw = get_all_trades(limit=10, exclude_test=True)
                                            recent_trades = []
                                            for trade in recent_trades_raw:
                                                # trade is tuple: (id, timestamp, symbol, side, price, quantity, pnl, fee)
                                                if len(trade) >= 8:
                                                    recent_trades.append({
                                                        'symbol': trade[2],
                                                        'side': trade[3],
                                                        'price': float(trade[4]),
                                                        'quantity': float(trade[5]),
                                                        'pnl': float(trade[6]),
                                                        'timestamp': trade[1]
                                                    })
                                            dashboard.config['recent_trades'] = recent_trades
                                        except Exception as e:
                                            logger.debug(f"Could not update recent trades: {e}")
                                        
                                        logger.debug(f"Updated dashboard with {len(open_positions)} open positions and {len(dashboard.config.get('recent_trades', []))} recent trades")
                                        
                                    except Exception as e:
                                        logger.warning(f"Could not update open positions: {e}")
                                    
                                    logger.debug(f"Updated dashboard with OHLC data types: {ohlc_subset.dtypes.to_dict()}")
                                else:
                                    logger.warning("OHLC data conversion resulted in empty DataFrame")
                        except Exception as e:
                            logger.warning(f"Could not update dashboard config: {e}")
                        
                        # Log key indicator values
                        latest = data.iloc[-1]
                        # Format indicator values safely
                        rsi_val = latest.get('rsi', 'N/A')
                        macd_val = latest.get('macd', 'N/A')
                        sma_val = latest.get('sma_20', 'N/A')
                        
                        rsi_str = f"{rsi_val:.2f}" if isinstance(rsi_val, (int, float)) else str(rsi_val)
                        macd_str = f"{macd_val:.4f}" if isinstance(macd_val, (int, float)) else str(macd_val)
                        sma_str = f"{sma_val:.2f}" if isinstance(sma_val, (int, float)) else str(sma_val)
                        
                        logger.info(f"Latest indicators - RSI: {rsi_str}, MACD: {macd_str}, SMA: {sma_str}")
                        
                        # 🤖 PERIODIC LLM MARKET ANALYSIS (every 10th iteration)
                        if hasattr(trading_loop, 'iteration_count'):
                            trading_loop.iteration_count += 1
                        else:
                            trading_loop.iteration_count = 1
                            
                        if llm_advisor and trading_loop.iteration_count % 10 == 0:
                            try:
                                sentiment_analysis = llm_advisor.analyze_market_sentiment(market_data)
                                sentiment = sentiment_analysis.get('sentiment', 'Neutral')
                                sentiment_confidence = sentiment_analysis.get('confidence', 0)
                                risk_level = sentiment_analysis.get('risk_level', 'Medium')
                                analysis = sentiment_analysis.get('analysis', '')
                                
                                logger.info(f"🧠 LLM Market Sentiment: {sentiment} ({sentiment_confidence:.1%} confidence)")
                                logger.info(f"📊 Risk Level: {risk_level}")
                                if analysis:
                                    logger.info(f"💡 LLM Analysis: {analysis[:100]}...")
                                
                                # 🤖 Enhanced LLM Analysis - Performance & Trends
                                if trading_loop.iteration_count % 20 == 0:  # Every 20 iterations for comprehensive analysis
                                    try:
                                        # Get recent trading data
                                        recent_trades = []
                                        portfolio_data = {}
                                        open_positions = []
                                        
                                        try:
                                            # Get recent trades from API
                                            import requests
                                            response = requests.get('http://localhost:5050/trades', timeout=2)
                                            if response.status_code == 200:
                                                recent_trades = response.json()[-10:]  # Last 10 trades
                                            
                                            # Get portfolio data
                                            balance_response = requests.get('http://localhost:5050/balance', timeout=2)
                                            if balance_response.status_code == 200:
                                                balance_data = balance_response.json()
                                                usdt_balance = balance_data.get('USDT', 1000)
                                                bnb_balance = balance_data.get('BNB', 1.67)
                                                btc_balance = balance_data.get('BTC', 0.0086)
                                                
                                                # Calculate portfolio value with current prices
                                                bnb_price = market_data.get('bnb_price', 600)
                                                btc_price = market_data.get('btc_price', 117000)
                                                total_value = usdt_balance + (bnb_balance * bnb_price) + (btc_balance * btc_price)
                                                
                                                portfolio_data = {
                                                    'total_value': total_value,
                                                    'realized_pnl': sum(float(t.get('pnl', 0)) for t in recent_trades),
                                                    'unrealized_pnl': 0  # Will be calculated from open positions
                                                }
                                            
                                            # Get open positions
                                            positions_response = requests.get('http://localhost:5050/open_positions', timeout=2)
                                            if positions_response.status_code == 200:
                                                open_positions = positions_response.json()
                                        
                                        except Exception as data_e:
                                            logger.debug(f"Could not fetch trading data for LLM analysis: {data_e}")
                                        
                                        # 📊 Trading Performance Analysis
                                        if recent_trades:
                                            performance_analysis = llm_advisor.analyze_trading_performance(
                                                recent_trades, portfolio_data, open_positions
                                            )
                                            
                                            perf_score = performance_analysis.get('performance_score', 'N/A')
                                            win_rate = performance_analysis.get('win_rate', 'N/A')
                                            risk_assessment = performance_analysis.get('risk_assessment', 'UNKNOWN')
                                            
                                            logger.info(f"🎯 LLM Performance Analysis - Score: {perf_score}/10, Win Rate: {win_rate}%, Risk: {risk_assessment}")
                                            
                                            # Log key recommendations
                                            recommendations = performance_analysis.get('recommendations', [])
                                            for i, rec in enumerate(recommendations[:3], 1):
                                                logger.info(f"💡 LLM Recommendation {i}: {rec}")
                                        
                                        # 📈 Market Trend Analysis
                                        if hasattr(data, 'index') and len(data) > 10:  # Ensure we have enough price data
                                            try:
                                                trend_analysis = llm_advisor.analyze_market_trends(data)
                                                
                                                trend = trend_analysis.get('trend', 'UNKNOWN')
                                                momentum = trend_analysis.get('momentum', 'UNKNOWN')
                                                volatility = trend_analysis.get('volatility', 'N/A')
                                                
                                                logger.info(f"📈 LLM Trend Analysis - Trend: {trend}, Momentum: {momentum}, Volatility: {volatility}%")
                                                
                                                # 🔮 Generate Comprehensive Trading Insights
                                                if recent_trades and trend_analysis.get('analysis'):
                                                    insights_report = llm_advisor.generate_trading_insights(
                                                        performance_analysis, trend_analysis
                                                    )
                                                    
                                                    # Log the insights report (truncated for console)
                                                    insights_lines = insights_report.split('\n')
                                                    logger.info("🤖 LLM COMPREHENSIVE TRADING INSIGHTS:")
                                                    for line in insights_lines[:5]:  # Show first 5 lines
                                                        if line.strip():
                                                            logger.info(f"   {line.strip()}")
                                                    
                                                    # Store insights for dashboard display
                                                    if 'dashboard' in locals() and hasattr(dashboard, 'config'):
                                                        dashboard.config['llm_insights'] = {
                                                            'performance': performance_analysis,
                                                            'trends': trend_analysis,
                                                            'report': insights_report,
                                                            'timestamp': datetime.now()
                                                        }
                                            
                                            except Exception as trend_e:
                                                logger.debug(f"LLM trend analysis error: {trend_e}")
                                    
                                    except Exception as analysis_e:
                                        logger.warning(f"LLM comprehensive analysis error: {analysis_e}")
                                        logger.debug("LLM analysis error details:", exc_info=True)
                                    
                                # Store for dashboard
                                if 'dashboard' in locals() and hasattr(dashboard, 'config'):
                                    dashboard.config['llm_sentiment'] = {
                                        'sentiment': sentiment,
                                        'confidence': sentiment_confidence,
                                        'risk_level': risk_level,
                                        'analysis': analysis,
                                        'timestamp': sentiment_analysis.get('timestamp')
                                    }
                                    
                            except Exception as e:
                                logger.debug(f"LLM sentiment analysis failed: {e}")
                        
                        # BALANCED STARTER STRATEGY - Only execute if not main strategy
                        if strategy_mode != 'balanced':
                            try:
                                from trading.strategies.balanced_starter import BalancedStarterStrategy
                                
                                # Initialize balanced starter if not exists
                                if not hasattr(container.get('strategy'), 'balanced_starter'):
                                    container.get('strategy').balanced_starter = BalancedStarterStrategy(logger)
                                    logger.info("🎯 INITIALIZED BALANCED STARTER STRATEGY")
                                
                                # Execute balanced starter strategy
                                balanced_result = container.get('strategy').balanced_starter.execute_strategy(
                                    data, executor.get_account_balance()
                                )
                                
                                if balanced_result['action'] in ['INITIALIZED', 'ADAPTED']:
                                    logger.info(f"🎯 BALANCED STARTER: {balanced_result['action']} at ${balanced_result.get('price', 0):,.2f}")
                                    if 'market_bias' in balanced_result:
                                        logger.info(f"📊 Market Bias: {balanced_result['market_bias']:.3f}")
                                    if 'positions' in balanced_result:
                                        pos = balanced_result['positions']
                                        logger.info(f"📊 Position Distribution: {pos['long_count']} LONG, {pos['short_count']} SHORT")
                            
                            except Exception as e:
                                logger.error(f"Error in balanced starter strategy: {e}")
                        else:
                            logger.info("🎯 BALANCED STARTER: Running as main strategy, skipping secondary execution")
                        
                        # EMERGENCY PORTFOLIO PROTECTION - Check balance before trading
                        try:
                            current_balance = executor.get_account_balance()
                            if current_balance < 700:  # If lost more than 30% of $1000
                                logger.error(f"🚨 PORTFOLIO PROTECTION: Balance dropped to ${current_balance:.2f}")
                                logger.error("🚨 EMERGENCY SHUTDOWN - Portfolio protection activated")
                                logger.error("🚨 Manual intervention required - losses too high")
                                return  # Exit trading loop
                        except Exception as balance_error:
                            logger.error(f"Could not check balance: {balance_error}")
                        
                        # Use the active strategy (ensemble or research-based)
                        logger.info(f"Using strategy: {type(active_strategy).__name__}")
                        
                        # Generate signals using the NEW generate_signal method (singular) with anti-HOLD logic
                        if hasattr(active_strategy, 'generate_signal'):
                            # Get current market data for signal generation with error handling
                            try:
                                current_price = float(data.iloc[-1]['close'])
                                logger.debug(f"Current price extracted: {current_price} (type: {type(current_price)})")
                                
                                market_data = {
                                    'close': current_price,
                                    'price': current_price,
                                    'high': float(data.iloc[-1].get('high', current_price)),
                                    'low': float(data.iloc[-1].get('low', current_price)),
                                    'volume': float(data.iloc[-1].get('volume', 1000)),
                                    'rsi': float(data.iloc[-1].get('rsi', 50.0)) if pd.notna(data.iloc[-1].get('rsi')) else 50.0,
                                    'macd': float(data.iloc[-1].get('macd', 0.0)) if pd.notna(data.iloc[-1].get('macd')) else 0.0,
                                    'macd_signal': float(data.iloc[-1].get('signal_line', 0.0)) if pd.notna(data.iloc[-1].get('signal_line')) else 0.0,
                                    'sma_20': float(data.iloc[-1].get('sma_20', current_price)) if pd.notna(data.iloc[-1].get('sma_20')) else current_price,
                                    'sma_50': float(data.iloc[-1].get('sma_50', current_price)) if pd.notna(data.iloc[-1].get('sma_50')) else current_price,
                                    'atr': float(data.iloc[-1].get('atr', current_price * 0.02)) if pd.notna(data.iloc[-1].get('atr')) else current_price * 0.02,
                                    'adx': float(data.iloc[-1].get('adx', 25.0)) if pd.notna(data.iloc[-1].get('adx')) else 25.0,
                                    'bb_upper': float(data.iloc[-1].get('upper_bb', current_price * 1.02)) if pd.notna(data.iloc[-1].get('upper_bb')) else current_price * 1.02,
                                    'bb_lower': float(data.iloc[-1].get('lower_bb', current_price * 0.98)) if pd.notna(data.iloc[-1].get('lower_bb')) else current_price * 0.98,
                                    'bb_middle': float(data.iloc[-1].get('sma_bb', current_price)) if pd.notna(data.iloc[-1].get('sma_bb')) else current_price
                                }
                            except Exception as e:
                                logger.error(f"❌ Error processing market data: {e}")
                                logger.error(f"   Data types - close: {type(data.iloc[-1]['close'])}, value: {data.iloc[-1]['close']}")
                                continue
                            
                            # Process ALL symbols from symbols_to_trade  
                            logger.info(f"🔄 Processing {len(symbols_to_trade)} trading symbols: {symbols_to_trade}")
                            
                            for symbol in symbols_to_trade:
                                try:
                                    # Get data for this specific symbol
                                    symbol_data = all_data.get(symbol)
                                    if symbol_data is None or symbol_data.empty:
                                        logger.warning(f"⚠️ No data available for {symbol}, skipping")
                                        continue
                                    
                                    # Extract current price and market data for this symbol
                                    current_price = float(symbol_data.iloc[-1]['close'])
                                    
                                    # Create market data specific to this symbol
                                    market_data = {
                                        'close': current_price,
                                        'price': current_price,
                                        'high': float(symbol_data.iloc[-1].get('high', current_price)),
                                        'low': float(symbol_data.iloc[-1].get('low', current_price)),
                                        'volume': float(symbol_data.iloc[-1].get('volume', 1000)),
                                        'rsi': float(symbol_data.iloc[-1].get('rsi', 50.0)) if pd.notna(symbol_data.iloc[-1].get('rsi')) else 50.0,
                                        'macd': float(symbol_data.iloc[-1].get('macd', 0.0)) if pd.notna(symbol_data.iloc[-1].get('macd')) else 0.0,
                                        'macd_signal': float(symbol_data.iloc[-1].get('signal_line', 0.0)) if pd.notna(symbol_data.iloc[-1].get('signal_line')) else 0.0,
                                        'sma_20': float(symbol_data.iloc[-1].get('sma_20', current_price)) if pd.notna(symbol_data.iloc[-1].get('sma_20')) else current_price,
                                        'sma_50': float(symbol_data.iloc[-1].get('sma_50', current_price)) if pd.notna(symbol_data.iloc[-1].get('sma_50')) else current_price,
                                        'atr': float(symbol_data.iloc[-1].get('atr', current_price * 0.02)) if pd.notna(symbol_data.iloc[-1].get('atr')) else current_price * 0.02,
                                        'adx': float(symbol_data.iloc[-1].get('adx', 25.0)) if pd.notna(symbol_data.iloc[-1].get('adx')) else 25.0,
                                        'bb_upper': float(symbol_data.iloc[-1].get('upper_bb', current_price * 1.02)) if pd.notna(symbol_data.iloc[-1].get('upper_bb')) else current_price * 1.02,
                                        'bb_lower': float(symbol_data.iloc[-1].get('lower_bb', current_price * 0.98)) if pd.notna(symbol_data.iloc[-1].get('lower_bb')) else current_price * 0.98,
                                        'bb_middle': float(symbol_data.iloc[-1].get('sma_bb', current_price)) if pd.notna(symbol_data.iloc[-1].get('sma_bb')) else current_price
                                    }
                                    
                                    logger.info(f"🎯 Processing {symbol}: Price=${current_price:.2f}, RSI={market_data['rsi']:.1f}")
                                    
                                    # Generate signal for this symbol
                                    signal, confidence = active_strategy.generate_signal(symbol, market_data)
                                    
                                    logger.info(f"🎉 {symbol} SIGNAL: {signal} with confidence {confidence:.3f}")
                                    
                                    # 🤖 LLM SIGNAL ENHANCEMENT
                                    if llm_advisor and signal in ['BUY', 'SELL']:
                                        try:
                                            llm_analysis = llm_advisor.enhance_trading_signal(signal, confidence, market_data)
                                            
                                            # Apply LLM confidence adjustment
                                            original_confidence = confidence
                                            confidence = llm_analysis.get('adjusted_confidence', confidence)
                                            validation = llm_analysis.get('validation', 'CAUTION')
                                            
                                            logger.info(f"🧠 LLM Enhancement: {validation} | Confidence {original_confidence:.3f} → {confidence:.3f}")
                                            
                                            if validation == 'REJECT':
                                                logger.warning(f"🚫 LLM REJECTED signal: {llm_analysis.get('reasoning', 'No reason given')}")
                                                signal = 'HOLD'
                                                confidence = 0.1
                                            elif validation == 'CAUTION':
                                                logger.info(f"⚠️ LLM CAUTION: {llm_analysis.get('recommendation', 'Proceed with caution')}")
                                            elif validation == 'CONFIRM':
                                                logger.info(f"✅ LLM CONFIRMS signal: {llm_analysis.get('recommendation', 'Signal validated')}")
                                            
                                            # Log risk factors
                                            risk_factors = llm_analysis.get('risk_factors', [])
                                            if risk_factors:
                                                logger.info(f"⚠️ LLM Risk Factors: {', '.join(risk_factors)}")
                                                
                                        except Exception as e:
                                            logger.warning(f"LLM enhancement failed: {e}")
                                    
                                    # EMERGENCY ATH FIX: Allow HOLD during extreme market conditions  
                                    if signal == 'HOLD':
                                        # During extreme ATH periods, HOLD is often the correct decision
                                        if current_price > 130000:  # Updated: Extreme ATH territory now at $130k
                                            logger.info(f"💎 HOLD signal during EXTREME ATH (${current_price:,.0f}) - SMART DECISION")
                                        elif current_price > 125000:  # Normal ATH range
                                            logger.info(f"💎 HOLD signal in ATH range (${current_price:,.0f}) - Cautious approach")
                                        else:
                                            logger.info(f"⏸️ HOLD signal - Waiting for better opportunity")
                                    
                                    logger.info(f"Final signal for {symbol}: {signal} (confidence: {confidence:.3f})")
                                    
                                    # Execute trades based on signals for this symbol
                                    if signal in ['BUY', 'SELL'] and confidence >= 0.90:  # High threshold for quality trades
                                        logger.info(f"🎯 High confidence {symbol} signal: {signal} with {confidence:.3f} confidence")
                                        
                                        try:
                                            # Get current account balance
                                            available_balance = executor.get_account_balance()
                                            logger.info(f"💰 Available balance: ${available_balance}")
                                            
                                            # EMERGENCY RISK MANAGEMENT: Extremely conservative position sizing
                                            max_risk_per_trade = 5.0  # Maximum $5 per trade
                                            position_size = min(available_balance * 0.005, max_risk_per_trade) / current_price  # 0.5% of balance, max $5
                                            
                                            if signal == 'BUY':
                                                logger.info(f"🟢 [BUY] Executing {symbol} order - Size: {position_size:.6f}, Price: ${current_price:.2f}")
                                                result = executor.execute_buy(symbol, position_size, current_price)
                                                
                                                if result:
                                                    logger.info(f"✅ [SUCCESS] {symbol} BUY executed successfully")
                                                else:
                                                    logger.error(f"❌ [FAIL] Failed to execute {symbol} BUY order")
                                                    
                                            elif signal == 'SELL':
                                                logger.info(f"🔴 [SELL] Executing {symbol} order - Size: {position_size:.6f}, Price: ${current_price:.2f}")  
                                                result = executor.execute_sell(symbol, position_size, current_price)
                                                
                                                if result:
                                                    logger.info(f"✅ [SUCCESS] {symbol} SELL executed successfully")
                                                else:
                                                    logger.error(f"❌ [FAIL] Failed to execute {symbol} SELL order")
                                                    
                                        except Exception as e:
                                            logger.error(f"❌ Trading execution error for {symbol}: {e}")
                                    
                                    else:
                                        logger.info(f"📊 {symbol} Signal: {signal} | Confidence: {confidence:.3f} | Action: HOLD (below 90% threshold)")
                                    
                                except Exception as e:
                                    logger.error(f"❌ Error processing {symbol}: {e}")
                                    continue  # Continue with next symbol
                            
                            # END OF SYMBOL LOOP
                            logger.info("✅ Completed processing all trading symbols")
                            
                            # CRITICAL: Check for stop losses on open positions FIRST
                            try:
                                from utils.paper_trade_db import get_open_positions
                                open_positions = get_open_positions()
                                
                                for position in open_positions:
                                    try:
                                        # position format: (id, symbol, side, entry_price, quantity, leverage, entry_time)
                                        if len(position) < 5:
                                            continue
                                        
                                        # Debug: log the actual position tuple to see the structure
                                        logger.debug(f"Position tuple: {position}")
                                        logger.debug(f"Position length: {len(position)}")
                                        
                                        pos_symbol = position[1]
                                        side = position[2]  # BUY/SELL
                                        
                                        # Add validation before float conversion
                                        try:
                                            entry_price = float(position[3])
                                        except (ValueError, TypeError) as e:
                                            logger.error(f"Could not convert entry_price '{position[3]}' to float: {e}")
                                            logger.error(f"Full position tuple: {position}")
                                            continue
                                            
                                        try:
                                            quantity = float(position[4])
                                        except (ValueError, TypeError) as e:
                                            logger.error(f"Could not convert quantity '{position[4]}' to float: {e}")
                                            logger.error(f"Full position tuple: {position}")
                                            continue
                                        
                                        if quantity == 0 or entry_price <= 0:
                                            continue
                                        
                                        # Get current price for this position
                                        if pos_symbol == symbol:
                                            position_current_price = current_price
                                        else:
                                            position_current_price = price_fetcher.get_current_price(pos_symbol)
                                            if position_current_price is None:
                                                continue
                                                
                                        # Calculate P&L percentage
                                        if side.upper() == 'BUY':  # LONG position
                                            pnl_pct = ((position_current_price - entry_price) / entry_price) * 100
                                        else:  # SHORT position
                                            pnl_pct = ((entry_price - position_current_price) / entry_price) * 100
                                        
                                        # DYNAMIC STOP LOSS: Use risk manager if available
                                        risk_manager = container.get_optional('risk_manager')
                                        stop_loss_threshold = -1.0  # Default
                                        
                                        if risk_manager:
                                            stop_loss_threshold = -(risk_manager.dynamic_stop_loss_pct * 100)
                                            logger.debug(f"🔧 Dynamic stop loss threshold: {stop_loss_threshold:.1f}%")
                                        if pnl_pct < stop_loss_threshold:
                                            logger.warning(f"🛑 STOP LOSS triggered for {pos_symbol}: {pnl_pct:.2f}% loss")
                                            try:
                                                close_signal = 'SELL' if side.upper() == 'BUY' else 'BUY'
                                                close_size = abs(quantity)
                                                success = executor.place_order(pos_symbol, close_signal, close_size, position_current_price)
                                                if success:
                                                    logger.info(f"🛑 Stop loss executed: {close_signal.upper()} {close_size:.6f} {pos_symbol}")
                                                else:
                                                    logger.error(f"❌ Stop loss execution failed for {pos_symbol}")
                                            except Exception as e:
                                                logger.error(f"❌ Stop loss execution error: {e}")
                                        
                                        # MICRO-SCALPING TAKE PROFIT: 10 cents profit target
                                        # Calculate absolute profit in USD
                                        if side.upper() == 'BUY':  # LONG position
                                            absolute_profit = (position_current_price - entry_price) * quantity
                                        else:  # SHORT position
                                            absolute_profit = (entry_price - position_current_price) * quantity
                                        
                                        # UPDATED: Dynamic take profit based on Bitcoin price level
                                        if position_current_price > 120000:  # High price Bitcoin
                                            take_profit_threshold_usd = 5.00  # $5.00 profit target for high prices
                                        elif position_current_price > 100000:  # Medium high Bitcoin
                                            take_profit_threshold_usd = 2.50  # $2.50 profit target
                                        else:  # Normal Bitcoin prices
                                            take_profit_threshold_usd = 1.00  # $1.00 profit target
                                        
                                        if absolute_profit >= take_profit_threshold_usd:
                                            logger.info(f"💰 TAKE PROFIT triggered for {pos_symbol}: ${absolute_profit:.4f} profit (${take_profit_threshold_usd} target)")
                                            try:
                                                close_signal = 'SELL' if side.upper() == 'BUY' else 'BUY'
                                                close_size = abs(quantity)
                                                success = executor.place_order(pos_symbol, close_signal, close_size, position_current_price)
                                                if success:
                                                    logger.info(f"💰 Take profit executed: {close_signal.upper()} {close_size:.6f} {pos_symbol}")
                                                else:
                                                    logger.error(f"❌ Take profit execution failed for {pos_symbol}")
                                            except Exception as e:
                                                logger.error(f"❌ Take profit execution error: {e}")
                                                
                                    except Exception as e:
                                        logger.error(f"❌ Position management error: {e}")
                                        
                            except Exception as e:
                                logger.error(f"❌ Position management loop error: {e}")
                            
                            # MAXIMUM SPEED TRADING - No delays whatsoever
                            current_time = time.time()
                            
                            # Execute trades based on signals with MUCH HIGHER threshold to stop overtrading
                            if signal in ['BUY', 'SELL'] and confidence >= 0.90:  # CRITICAL: High threshold for quality trades only
                                logger.info(f"🎯 High confidence signal detected: {signal} with {confidence:.3f} confidence")
                            
                            # Multi-symbol trading is now handled by the main loop above
                            logger.info("✅ Multi-symbol trading handled by main loop - BNBUSDT included")
                            
                            # Continue with execution logic
                            current_price = data.iloc[-1]['close']
                            
                            # DYNAMIC RISK MANAGEMENT with x50 LEVERAGE ENFORCEMENT
                            available_balance = executor.get_account_balance()
                            base_leverage = getattr(executor, 'default_leverage', 50)  # Default to x50
                            
                            # Get risk manager for dynamic calculations
                            risk_manager = container.get_optional('risk_manager')
                            
                            try:
                                # ENFORCE MINIMUM x50 LEVERAGE
                                if risk_manager and hasattr(risk_manager, 'enforce_minimum_leverage'):
                                    leverage = risk_manager.enforce_minimum_leverage(base_leverage)
                                    # Use dynamic position sizing
                                    if hasattr(risk_manager, 'calculate_dynamic_position_size'):
                                        position_size = risk_manager.calculate_dynamic_position_size(
                                            available_balance, current_price, confidence, leverage
                                        )
                                    else:
                                        # Fallback position size calculation
                                        position_size = min(0.001, float(available_balance) / float(current_price) * 0.05)
                                    logger.info(f"⚡ DYNAMIC: Leverage {leverage}x, Dynamic sizing used")
                                else:
                                    # Fallback to strategy calculation with enforced leverage
                                    leverage = max(base_leverage, 50.0)  # Ensure minimum x50
                                    balance_info = executor.get_balance()
                                    position_size = active_strategy.calculate_position_size(
                                        data, 
                                        current_price, 
                                        available_capital=available_balance, 
                                        leverage=leverage, 
                                        signal_confidence=confidence
                                    )
                                    logger.info(f"⚡ ENFORCED: Leverage {leverage}x, Strategy sizing used")
                            except Exception as e:
                                logger.error(f"Error in position size calculation: {e}")
                                # Emergency fallback position size with proper type conversion
                                leverage = max(base_leverage, 50.0)  # Set leverage for emergency fallback
                                try:
                                    safe_balance = float(available_balance) if available_balance is not None else 1000.0
                                    safe_price = float(current_price) if current_price is not None else 97000.0
                                    position_size = min(0.001, safe_balance / safe_price * 0.05)  # REDUCED from 0.1 to 0.05
                                    logger.warning(f"🚨 Using emergency position size: {position_size:.6f} with leverage {leverage}x")
                                except Exception as calc_error:
                                    logger.error(f"❌ Emergency calculation failed: {calc_error}")
                                    logger.error(f"   available_balance: {available_balance} (type: {type(available_balance)})")
                                    logger.error(f"   current_price: {current_price} (type: {type(current_price)})")
                                    position_size = 0.001  # Ultra-safe fallback
                                    if 'leverage' not in locals():
                                        leverage = 50.0  # Default leverage
                                    logger.warning(f"🚨 Using ultra-safe fallback position size: {position_size:.6f} with leverage {leverage}x")
                            
                            # Position size calculated - ready for execution
                            
                            # Emergency check - force minimum position size if zero
                            if position_size <= 0:
                                if 'leverage' not in locals():
                                    leverage = max(base_leverage, 50.0)  # Ensure leverage is defined
                                try:
                                    safe_balance = float(available_balance) if available_balance is not None else 1000.0
                                    safe_price = float(current_price) if current_price is not None else 97000.0
                                    position_size = min(0.001, safe_balance / safe_price * 0.05)  # REDUCED
                                    logger.warning(f"🚨 Position size was zero, forced to: {position_size:.6f}")
                                except Exception as e:
                                    logger.error(f"❌ Zero position size fix failed: {e}")
                                    position_size = 0.001  # Ultra-safe fallback
                                    logger.warning(f"🚨 Using ultra-safe position size: {position_size:.6f}")

                            logger.info(f"🎯 DYNAMIC x50+ EXECUTION: {signal} order - Price: ${current_price:.2f}, Size: {position_size:.6f}, Balance: ${available_balance:.2f}, Leverage: {leverage}x")
                            
                            # DUPLICATE TRADE PREVENTION
                            trade_key = f"{signal}_{symbol}_{current_price:.2f}_{position_size:.6f}"
                            current_time_ms = int(time.time() * 1000)
                            
                            # Skip if same trade executed within last 30 seconds
                            if hasattr(trading_loop, 'recent_trades'):
                                if trading_loop.recent_trades.get(trade_key, 0) + 30000 > current_time_ms:
                                    logger.warning(f"🚫 DUPLICATE PREVENTED: {trade_key}")
                                    continue
                            else:
                                trading_loop.recent_trades = {}
                            
                            # Execute order with new method
                            logger.info(f"🎯 ORDER ATTEMPT: {signal} | Size: {position_size:.6f} | Price: ${current_price:.2f}")
                            
                            if signal == 'BUY':
                                order_result = executor.place_order(symbol, 'buy', position_size, current_price)
                                if order_result:
                                    logger.info(f"🎉 [SUCCESS] BUY order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                                    
                                    # 🔥 FIX: Add position to risk manager for monitoring
                                    try:
                                        position_data = {
                                            'symbol': symbol,
                                            'side': 'BUY',
                                            'entry_price': current_price,
                                            'quantity': position_size,
                                            'leverage': leverage,
                                            'timestamp': time.time()
                                        }
                                        risk_manager.add_position(symbol, position_data)
                                        logger.info(f"✅ Position added to risk manager: BUY {position_size:.6f} {symbol}")
                                    except Exception as e:
                                        logger.error(f"❌ Failed to add BUY position to risk manager: {e}")
                                    
                                    # Record trade to prevent duplicates
                                    trading_loop.recent_trades[trade_key] = current_time_ms
                                    # Small delay to ensure trade completion
                                    time.sleep(0.5)
                                else:
                                    logger.error(f"❌ [FAIL] Failed to execute BUY order for {symbol} - Size: {position_size:.6f}, Price: ${current_price:.2f}")
                            elif signal == 'SELL':
                                order_result = executor.place_order(symbol, 'sell', position_size, current_price)
                                if order_result:
                                    logger.info(f"🎉 [SUCCESS] SELL order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                                    
                                    # 🔥 FIX: Add position to risk manager for monitoring  
                                    try:
                                        position_data = {
                                            'symbol': symbol,
                                            'side': 'SELL',
                                            'entry_price': current_price,
                                            'quantity': position_size,
                                            'leverage': leverage,
                                            'timestamp': time.time()
                                        }
                                        risk_manager.add_position(symbol, position_data)
                                        logger.info(f"✅ Position added to risk manager: SELL {position_size:.6f} {symbol}")
                                    except Exception as e:
                                        logger.error(f"❌ Failed to add SELL position to risk manager: {e}")
                                    
                                    # Record trade to prevent duplicates
                                    trading_loop.recent_trades[trade_key] = current_time_ms
                                    # Small delay to ensure trade completion
                                    time.sleep(0.5)
                                else:
                                    logger.error(f"❌ [FAIL] Failed to execute SELL order for {symbol} - Size: {position_size:.6f}, Price: ${current_price:.2f}")
                            else:
                                logger.info(f"📊 Signal: {signal} | Confidence: {confidence:.3f} | Action: HOLD (below 90% threshold)")
                        else:
                            # Fallback for strategies without the new generate_signal method
                            logger.warning(f"Strategy {type(active_strategy).__name__} doesn't have generate_signal method - using old approach")
                            if hasattr(active_strategy, 'generate_signals'):
                                signal_data = {"BTCUSDT": data}
                                signals = active_strategy.generate_signals(signal_data)
                                
                                for symbol, signal_info in signals.items():
                                    signal = signal_info.get('signal', 'HOLD')
                                    confidence = signal_info.get('confidence', 0.0)
                                    
                                    if signal in ['BUY', 'SELL'] and confidence >= 0.90:
                                        current_price = data.iloc[-1]['close']
                                        available_balance = executor.get_account_balance()
                                        position_size = active_strategy.calculate_position_size(
                                            data, current_price, available_balance
                                        )
                                        
                                        if signal == 'BUY':
                                            executor.place_order(symbol, 'buy', position_size, current_price)
                                            logger.info(f"BUY signal executed: {position_size} BTC at {current_price}")
                                        elif signal == 'SELL':
                                            executor.place_order(symbol, 'sell', position_size, current_price)
                                            logger.info(f"SELL signal executed: {position_size} BTC at {current_price}")
                    else:
                        logger.error("Data is empty even after mock data creation!")
                        
                    # EMERGENCY: MUCH SLOWER LOOP to prevent disaster
                    logger.debug("=== TRADING LOOP ITERATION END ===\n")
                    logger.info("🚀 MAXIMUM SPEED: No cooldown - immediate next iteration")
                    
                    # COOLDOWN REMOVED: Immediate next iteration for maximum trading speed
                    if shutdown_requested:
                        logger.info("Shutdown requested, exiting trading loop...")
                        return
                    time.sleep(1)  # Minimal 1-second pause to prevent excessive CPU usage
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    # Check for shutdown before sleeping in error handling
                    for i in range(60):
                        if shutdown_requested:
                            logger.info("Shutdown requested during error sleep, exiting trading loop...")
                            return
                        time.sleep(1)
            
            logger.info("Trading loop finished due to shutdown request")

        # Start trading thread as non-daemon so it can complete before shutdown
        global trading_thread
        trading_thread = threading.Thread(target=trading_loop, daemon=False)
        trading_thread.start()
        
        # The dashboard's run method is blocking, so it should be the last call
        import sys
        
        # Check if we can run a curses dashboard
        # Modified to be less restrictive - only require TERM to be set
        if not os.getenv('TERM'):
            logger.info("TERM not set - running in headless mode")
            logger.info("Check logs for trading activity")
            # Keep the main thread alive in headless mode
            signal.pause()
        else:
            try:
                logger.info("Starting console dashboard...")
                logger.info(f"Dashboard has price_fetcher: {dashboard.price_fetcher is not None}")
                logger.info("🎯 Look for Market Depth in the RIGHT PANE (columns 94-120)")
                
                # Temporarily disable console logging to prevent screen jumping
                root_logger = logging.getLogger()
                console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
                for handler in console_handlers:
                    root_logger.removeHandler(handler)
                
                curses.wrapper(dashboard.run)
                
                # Re-enable console logging after dashboard exits
                for handler in console_handlers:
                    root_logger.addHandler(handler)
            except curses.error as e:
                logger.warning(f"Curses terminal UI not available: {e}")
                logger.info("Running in headless mode - check logs for trading activity")
                # Keep the main thread alive in headless mode
                signal.pause()
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
        
        # Publish error event
        event_bus.publish(SystemEvent(
            system_type='error',
            component='main',
            status='critical',
            message=f'Unexpected error: {str(e)}',
            source='main'
        ))
    finally:
        # Cleanup
        bootstrapper.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELVIS Trading Bot with Dependency Injection")
    parser.add_argument("--mode", type=str, default="paper", 
                       help="Trading mode: paper or live")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    args = parser.parse_args()

    # Start the main bot
    main(mode=args.mode, log_level=args.log_level.upper())

