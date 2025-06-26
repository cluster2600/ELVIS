#!/usr/bin/env python3
"""
Modified main.py that forces dashboard to show regardless of TTY detection
"""
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

def run_with_forced_dashboard():
    """Run the trading bot with forced dashboard display"""
    
    print("🚀 STARTING ELVIS TRADING BOT WITH FORCED DASHBOARD")
    print("=" * 60)
    
    # Parse arguments like main.py
    parser = argparse.ArgumentParser(description='ELVIS Trading Bot')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper', help='Trading mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Log level')
    args = parser.parse_args()
    
    try:
        # Bootstrap like main.py
        bootstrapper = bootstrap_application(args.mode, args.log_level)
        logger = container.get('logger')
        
        logger.info(f"🎯 ELVIS Trading Bot starting in {args.mode} mode")
        
        # Get components
        price_fetcher = container.get('price_fetcher')
        risk_manager = container.get('risk_manager')
        
        # Create dashboard with proper price_fetcher
        dashboard = ConsoleDashboard(
            config={
                'portfolio_value': 10817.22,
                'unrealized_pnl': 0.0,
                'realized_pnl': 817.22,
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
        
        logger.info("✅ Dashboard created with live price fetcher")
        logger.info(f"Price fetcher available: {dashboard.price_fetcher is not None}")
        
        # Start background trading (simplified version)
        def background_trading():
            """Simplified background trading loop"""
            logger.info("🔄 Starting background trading loop...")
            count = 0
            while True:
                try:
                    count += 1
                    logger.debug(f"Trading loop iteration {count}")
                    
                    # Update dashboard with live data periodically
                    if count % 10 == 0:  # Every 10 iterations
                        try:
                            # Get live portfolio data (simplified)
                            from utils.paper_trade_db import get_all_trades
                            all_trades = get_all_trades(limit=1000)
                            realized_pnl = sum(float(trade[6]) for trade in all_trades if len(trade) >= 7)
                            
                            dashboard.config['realized_pnl'] = realized_pnl
                            dashboard.config['portfolio_value'] = 10000.0 + realized_pnl
                            
                            logger.debug(f"Updated dashboard - Portfolio: ${dashboard.config['portfolio_value']:.2f}")
                        except Exception as e:
                            logger.warning(f"Dashboard update error: {e}")
                    
                    time.sleep(2)  # 2 second intervals
                    
                except Exception as e:
                    logger.error(f"Background trading error: {e}")
                    time.sleep(5)
        
        # Start background thread
        trading_thread = threading.Thread(target=background_trading, daemon=True)
        trading_thread.start()
        
        logger.info("📊 Starting console dashboard (FORCED MODE)...")
        print("\n" + "=" * 60)
        print("🎯 DASHBOARD STARTING - Look for Market Depth in RIGHT PANE")
        print("📍 Market Depth should appear at columns 94-120")
        print("💡 If terminal is too narrow, you might not see it")
        print("🔧 Resize terminal to at least 120 columns wide")
        print("=" * 60)
        
        # FORCE dashboard to run regardless of TTY detection
        try:
            curses.wrapper(dashboard.run)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            print(f"\n❌ Dashboard failed to start: {e}")
            print("🔧 This might be a terminal compatibility issue")
            print("💡 Try running in a different terminal (iTerm2, Terminal.app)")
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
    finally:
        if 'bootstrapper' in locals():
            bootstrapper.cleanup()

if __name__ == "__main__":
    run_with_forced_dashboard()