import argparse
import logging
import threading
import signal
import sys
import time
import curses
import os

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
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger = container.get_optional('logger', logging.getLogger(__name__))
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    
    # Set shutdown flag instead of immediate exit
    shutdown_requested = True
    
    # Publish shutdown event
    event_bus.publish(SystemEvent(
        system_type='shutdown',
        component='main',
        status='ok',
        message=f'Shutdown signal {signum} received',
        source='signal_handler'
    ))
    
    # Wait for trading thread to finish if it exists
    if trading_thread and trading_thread.is_alive():
        logger.info("Waiting for trading operations to complete...")
        trading_thread.join(timeout=30)  # Wait up to 30 seconds
        
    logger.info("Graceful shutdown complete")
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
        
        # Get the active strategy (ensemble or research-based)
        active_strategy = container.get('strategy')
        strategy_mode = os.getenv('STRATEGY_MODE', 'ensemble')
        logger.info(f"🎯 Active strategy: {type(active_strategy).__name__} (mode: {strategy_mode})")
        
        if strategy_mode == 'research':
            logger.info("🔬 Research-based strategy active - targeting 14.9% annual returns")
            logger.info("📊 Binary classification: BUY/SELL only (no HOLD signals)")
            logger.info("🎯 Following Bonenkamp (2021) research methodology")
            
            # Initial training for research strategy if a saved model is not loaded
            if not active_strategy.is_trained:
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
            if not active_strategy.is_trained:
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
                        data = price_fetcher.get_historical_klines("BTCUSDT", "1m")
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
                            logger.warning("No real market data available - creating mock data for testing")
                            # Create mock data to test the trading logic
                            import numpy as np
                            np.random.seed(int(time.time()) % 1000)  # Different seed each time
                            mock_data = {
                                'open': np.random.normal(97000, 200, 50),
                                'high': np.random.normal(97200, 200, 50),
                                'low': np.random.normal(96800, 200, 50),
                                'close': np.random.normal(97000, 200, 50),
                                'volume': np.random.normal(1000, 100, 50),
                            }
                            data = pd.DataFrame(mock_data)
                            logger.info(f"Created mock data with shape: {data.shape}")
                            if 'close' in data.columns:
                                logger.info(f"Mock data latest close: {data.iloc[-1]['close']:.2f}")
                            else:
                                logger.error("Mock data missing 'close' column - this should not happen")
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
                                            # pos is a tuple: (id, symbol, entry_price, quantity, leverage, entry_time)
                                            if len(pos) >= 4:
                                                # Get current price for this symbol
                                                try:
                                                    if price_fetcher:
                                                        current_price = price_fetcher.get_current_price(pos[1])
                                                        if current_price is None:
                                                            current_price = float(pos[2])
                                                        else:
                                                            current_price = float(current_price)
                                                    else:
                                                        current_price = float(pos[2])  # Use entry price as fallback
                                                except:
                                                    current_price = float(pos[2])  # Use entry price as fallback
                                                
                                                # Calculate comprehensive P&L including all Binance fees
                                                try:
                                                    leverage = int(pos[4]) if len(pos) > 4 else 10
                                                    entry_time = pos[5] if len(pos) > 5 else None
                                                    
                                                    if hasattr(executor, 'calculate_open_position_pnl'):
                                                        pnl_detail = executor.calculate_open_position_pnl(
                                                            pos[1], current_price, float(pos[2]), 
                                                            float(pos[3]), leverage, entry_time
                                                        )
                                                        net_pnl = pnl_detail['net_pnl']
                                                        gross_pnl = pnl_detail['gross_pnl']
                                                        fees_info = {
                                                            'total_fees': pnl_detail['ongoing_costs'] + pnl_detail['estimated_exit_fee'],
                                                            'funding_fee': pnl_detail['funding_fee'],
                                                            'borrowing_cost': pnl_detail['borrowing_cost'],
                                                            'hours_held': pnl_detail['hours_held']
                                                        }
                                                    else:
                                                        # Fallback to simple calculation
                                                        net_pnl = (current_price - float(pos[2])) * float(pos[3]) if current_price and current_price > 0 else 0
                                                        gross_pnl = net_pnl
                                                        fees_info = {'total_fees': 0}
                                                except Exception as e:
                                                    logger.warning(f"Could not calculate comprehensive P&L: {e}")
                                                    net_pnl = (current_price - float(pos[2])) * float(pos[3]) if current_price and current_price > 0 else 0
                                                    gross_pnl = net_pnl
                                                    fees_info = {'total_fees': 0}
                                                
                                                open_positions.append({
                                                    'symbol': pos[1],
                                                    'size': float(pos[3]),
                                                    'entry_price': float(pos[2]),
                                                    'pnl': net_pnl,  # Net P&L after all fees
                                                    'gross_pnl': gross_pnl,  # Gross P&L before fees
                                                    'fees_info': fees_info,
                                                    'leverage': leverage,
                                                    'entry_time': pos[5] if len(pos) > 5 else 'N/A'
                                                })
                                        
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
                        
                        # Use the active strategy (ensemble or research-based)
                        logger.info(f"Using strategy: {type(active_strategy).__name__}")
                        
                        # Generate signals using the NEW generate_signal method (singular) with anti-HOLD logic
                        if hasattr(active_strategy, 'generate_signal'):
                            # Get current market data for signal generation
                            current_price = float(data.iloc[-1]['close'])
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
                            
                            symbol = "BTCUSDT"
                            logger.info(f"🎯 Calling NEW generate_signal method with market data for {symbol}")
                            logger.info(f"📊 Market data: Price=${current_price:.2f}, RSI={market_data['rsi']:.1f}, MACD={market_data['macd']:.3f}")
                            
                            # Call the NEW method with anti-HOLD logic and research strategy integration
                            signal, confidence = active_strategy.generate_signal(symbol, market_data)
                            
                            logger.info(f"🎉 NEW METHOD RESULT: {signal} with confidence {confidence:.3f}")
                            
                            # The new method should NEVER return HOLD - verify this
                            if signal == 'HOLD':
                                logger.error(f"🚨 CRITICAL ERROR: New generate_signal method returned HOLD! This should never happen!")
                                logger.error(f"🚨 Signal: {signal}, Confidence: {confidence}")
                                # Force it to BUY as emergency fallback
                                signal = 'BUY'
                                confidence = 0.65
                                logger.warning(f"🚨 EMERGENCY OVERRIDE: Forced to {signal} with {confidence:.3f} confidence")
                            
                            logger.info(f"Strategy signal for {symbol}: {signal} (confidence: {confidence:.3f})")
                            
                            # CRITICAL: Check for stop losses on open positions FIRST
                            try:
                                from utils.paper_trade_db import get_open_positions
                                open_positions = get_open_positions()
                                
                                for position in open_positions:
                                    try:
                                        # position format: (id, symbol, entry_price, quantity, leverage, entry_time)
                                        if len(position) < 4:
                                            continue
                                            
                                        pos_symbol = position[1]
                                        entry_price = float(position[2])
                                        quantity = float(position[3])
                                        
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
                                        if quantity > 0:  # LONG position
                                            pnl_pct = ((position_current_price - entry_price) / entry_price) * 100
                                        else:  # SHORT position
                                            pnl_pct = ((entry_price - position_current_price) / entry_price) * 100
                                        
                                        # STOP LOSS: Close position if loss > 2%
                                        if pnl_pct < -2.0:
                                            logger.warning(f"🛑 STOP LOSS triggered for {pos_symbol}: {pnl_pct:.2f}% loss")
                                            try:
                                                close_signal = 'sell' if quantity > 0 else 'buy'
                                                close_size = abs(quantity)
                                                success = executor.place_order(pos_symbol, close_signal, close_size, position_current_price)
                                                if success:
                                                    logger.info(f"🛑 Stop loss executed: {close_signal.upper()} {close_size:.6f} {pos_symbol}")
                                                else:
                                                    logger.error(f"❌ Stop loss execution failed for {pos_symbol}")
                                            except Exception as e:
                                                logger.error(f"❌ Stop loss execution error: {e}")
                                        
                                        # TAKE PROFIT: Close position if profit > 3%
                                        elif pnl_pct > 3.0:
                                            logger.info(f"💰 TAKE PROFIT triggered for {pos_symbol}: {pnl_pct:.2f}% profit")
                                            try:
                                                close_signal = 'sell' if quantity > 0 else 'buy'
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
                            
                            # Execute trades based on signals with CONSERVATIVE threshold
                            if signal in ['BUY', 'SELL'] and confidence >= 0.75:  # INCREASED threshold from 60% to 75%
                                current_price = data.iloc[-1]['close']
                                
                                # Check if we have too many open positions (limit to 2)
                                try:
                                    from utils.paper_trade_db import get_open_positions
                                    open_positions = get_open_positions()
                                    if len(open_positions) >= 2:
                                        logger.warning(f"⚠️ Too many open positions ({len(open_positions)}), skipping new trade")
                                        continue
                                except Exception as e:
                                    logger.error(f"Error checking open positions: {e}")
                                
                                # Calculate position size with EMERGENCY FALLBACKS
                                available_balance = executor.get_account_balance()
                                leverage = getattr(executor, 'default_leverage', 10)  # Get leverage from executor
                                
                                try:
                                    # Pass balances to the position size calculation
                                    balance_info = executor.get_balance()
                                    position_size = active_strategy.calculate_position_size(
                                        data, 
                                        current_price, 
                                        available_capital=available_balance, 
                                        leverage=leverage, 
                                        signal_confidence=confidence
                                    )
                                except Exception as e:
                                    logger.error(f"Error in position size calculation: {e}")
                                    # Emergency fallback position size
                                    position_size = min(0.001, available_balance / current_price * 0.05)  # REDUCED from 0.1 to 0.05
                                    logger.warning(f"🚨 Using emergency position size: {position_size:.6f}")
                                
                                # REDUCE position size to be more conservative
                                position_size = position_size * 0.5  # Use only half the calculated size
                                
                                # Emergency check - force minimum position size if zero
                                if position_size <= 0:
                                    position_size = min(0.001, available_balance / current_price * 0.05)  # REDUCED
                                    logger.warning(f"🚨 Position size was zero, forced to: {position_size:.6f}")

                                logger.info(f"🎯 CONSERVATIVE EXECUTION: {signal} order - Price: ${current_price:.2f}, Size: {position_size:.6f}, Balance: ${available_balance:.2f}, Leverage: {leverage}x")
                                
                                # Execute order with new method
                                logger.info(f"🎯 ORDER ATTEMPT: {signal} | Size: {position_size:.6f} | Price: ${current_price:.2f}")
                                
                                if signal == 'BUY':
                                    order_result = executor.place_order(symbol, 'buy', position_size, current_price)
                                    if order_result:
                                        logger.info(f"🎉 [SUCCESS] BUY order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                                        # Small delay to ensure trade completion
                                        time.sleep(0.5)
                                    else:
                                        logger.error(f"❌ [FAIL] Failed to execute BUY order for {symbol} - Size: {position_size:.6f}, Price: ${current_price:.2f}")
                                elif signal == 'SELL':
                                    order_result = executor.place_order(symbol, 'sell', position_size, current_price)
                                    if order_result:
                                        logger.info(f"🎉 [SUCCESS] SELL order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                                        # Small delay to ensure trade completion
                                        time.sleep(0.5)
                                    else:
                                        logger.error(f"❌ [FAIL] Failed to execute SELL order for {symbol} - Size: {position_size:.6f}, Price: ${current_price:.2f}")
                            else:
                                logger.info(f"📊 Signal: {signal} | Confidence: {confidence:.3f} | Action: HOLD (below 75% threshold)")
                        else:
                            # Fallback for strategies without the new generate_signal method
                            logger.warning(f"Strategy {type(active_strategy).__name__} doesn't have generate_signal method - using old approach")
                            if hasattr(active_strategy, 'generate_signals'):
                                signal_data = {"BTCUSDT": data}
                                signals = active_strategy.generate_signals(signal_data)
                                
                                for symbol, signal_info in signals.items():
                                    signal = signal_info.get('signal', 'HOLD')
                                    confidence = signal_info.get('confidence', 0.0)
                                    
                                    if signal in ['BUY', 'SELL'] and confidence >= 0.6:
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
                        
                    # Wait before next iteration - faster for live dashboard updates
                    logger.debug("=== TRADING LOOP ITERATION END ===\n")
                    logger.debug("Waiting 3 seconds before next iteration (faster for live updates)...")
                    
                    # Check for shutdown during sleep to enable faster shutdown
                    for i in range(3):  # Reduced to 3 seconds for much more frequent updates
                        if shutdown_requested:
                            logger.info("Shutdown requested during sleep, exiting trading loop...")
                            return
                        time.sleep(1)
                    
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
