import argparse
import logging
import threading
import signal
import sys
import time
import curses

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


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger = container.get_optional('logger', logging.getLogger(__name__))
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    
    # Publish shutdown event
    event_bus.publish(SystemEvent(
        system_type='shutdown',
        component='main',
        status='ok',
        message=f'Shutdown signal {signum} received',
        source='signal_handler'
    ))
    
    sys.exit(0)


def risk_management_loop(risk_manager, logger):
    """Continuously manage position risk in a separate thread."""
    while True:
        try:
            risk_manager.manage_positions()
        except Exception as e:
            logger.error(f"Error in risk management loop: {e}")
        time.sleep(5)  # Check every 5 seconds

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
        
        # Let's correct the instantiation of the dashboard
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
            while True:
                try:
                    # Get market data
                    data = price_fetcher.get_historical_klines("BTCUSDT", "1m")
                    if not data.empty:
                        # Calculate technical indicators for the data
                        data = add_technical_indicators(data)
                        
                        # Get active strategy
                        active_strategy = strategy_manager.get_active_strategy(data)
                        
                        # Generate signals using the strategy
                        if hasattr(active_strategy, 'generate_signals'):
                            # For ensemble strategy, pass data as dict
                            if hasattr(active_strategy, 'symbols'):
                                signal_data = {"BTCUSDT": data}
                                signals = active_strategy.generate_signals(signal_data)
                                
                                # Process signals for each symbol
                                for symbol, signal_info in signals.items():
                                    signal = signal_info.get('signal', 'HOLD')
                                    confidence = signal_info.get('confidence', 0.0)
                                    
                                    logger.info(f"Strategy signal for {symbol}: {signal} (confidence: {confidence:.3f})")
                                    
                                    # Execute trades based on signals
                                    if signal in ['BUY', 'SELL'] and confidence > 0.6:  # Only trade with high confidence
                                        current_price = data.iloc[-1]['close']
                                        
                                        # Calculate position size
                                        available_balance = executor.get_account_balance()
                                        position_size = active_strategy.calculate_position_size(
                                            data, current_price, available_balance
                                        )
                                        
                                        # Place order
                                        if signal == 'BUY':
                                            order_result = executor.place_order(symbol, 'buy', position_size, current_price)
                                            if order_result:
                                                logger.info(f"[PAPER TRADE] BUY order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                                        elif signal == 'SELL':
                                            order_result = executor.place_order(symbol, 'sell', position_size, current_price)
                                            if order_result:
                                                logger.info(f"[PAPER TRADE] SELL order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                            else:
                                # For other strategies, use traditional approach
                                buy_signal, sell_signal = active_strategy.generate_signals(data)
                                
                                if buy_signal or sell_signal:
                                    current_price = data.iloc[-1]['close']
                                    available_balance = executor.get_account_balance()
                                    position_size = active_strategy.calculate_position_size(
                                        data, current_price, available_balance
                                    )
                                    
                                    if buy_signal:
                                        executor.place_order("BTCUSDT", 'buy', position_size, current_price)
                                        logger.info(f"BUY signal executed: {position_size} BTC at {current_price}")
                                    elif sell_signal:
                                        executor.place_order("BTCUSDT", 'sell', position_size, current_price)
                                        logger.info(f"SELL signal executed: {position_size} BTC at {current_price}")
                        
                    # Wait before next iteration
                    time.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(60)

        threading.Thread(target=trading_loop, daemon=True).start()
        
        # The dashboard's run method is blocking, so it should be the last call
        try:
            curses.wrapper(dashboard.run)
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
