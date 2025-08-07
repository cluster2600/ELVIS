#!/usr/bin/env python3
"""
Start PriceFetcher service to populate Prometheus metrics for Grafana monitoring.
This script runs the PriceFetcher in background to continuously update trading metrics.
"""

import logging
import time
import signal
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils.price_fetcher import PriceFetcher
from utils.logger_config import setup_logging

# Setup logging
logger = setup_logging("MetricsServer", log_level="INFO")

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info(f"Received signal {signum}, shutting down...")
    running = False

def main():
    """Main function to start price fetcher for metrics."""
    global running
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting ELVIS Metrics Server...")
    logger.info("📊 This will populate Prometheus metrics for Grafana dashboards")
    
    # Initialize price fetcher with BTC symbols
    symbols = ['BTCUSDT', 'BNBBTC']
    price_fetcher = PriceFetcher(
        logger=logger, 
        symbols=symbols,
        timeframe='1m',  # More frequent updates
        history_limit=100
    )
    
    try:
        # Start the price fetcher WebSocket stream
        logger.info(f"📈 Starting price fetching for symbols: {symbols}")
        price_fetcher.start()
        
        # Wait a moment for connection
        time.sleep(5)
        
        # Force initial calculation of indicators for all symbols
        for symbol in symbols:
            try:
                price_fetcher.get_historical_data()
                price_fetcher.calculate_indicators(symbol)
                logger.info(f"✅ Initial metrics populated for {symbol}")
            except Exception as e:
                logger.error(f"❌ Error initializing metrics for {symbol}: {e}")
        
        logger.info("🎯 Metrics server running! Check http://localhost:5050/metrics")
        logger.info("📊 Grafana dashboards: http://localhost:3001")
        logger.info("⚡ Press Ctrl+C to stop")
        
        # Keep running until shutdown signal
        while running:
            time.sleep(10)  # Check every 10 seconds
            
            # Optionally log current metrics status
            if time.time() % 60 < 10:  # Every minute
                try:
                    current_price = price_fetcher.get_current_price('BTCUSDT')
                    if current_price:
                        logger.info(f"📈 BTCUSDT: ${current_price:,.2f}")
                except Exception as e:
                    logger.debug(f"Price check error: {e}")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Error in metrics server: {e}")
    finally:
        logger.info("🛑 Shutting down metrics server...")
        running = False
        sys.exit(0)

if __name__ == "__main__":
    main()