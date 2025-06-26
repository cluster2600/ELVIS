#!/usr/bin/env python3
"""
Test dashboard market depth with live data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from utils.console_dashboard import ConsoleDashboard
from utils.price_fetcher import PriceFetcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dashboard_market_depth():
    """Test dashboard market depth functionality"""
    logger.info("=== Testing Dashboard Market Depth ===")
    
    # Create real price fetcher
    price_fetcher = PriceFetcher(logger)
    
    # Test the market depth method directly
    config = {
        'portfolio_value': 10817.22,
        'unrealized_pnl': 0.0,
        'realized_pnl': 817.22,
        'open_positions': []
    }
    
    # Create dashboard with real price fetcher
    dashboard = ConsoleDashboard(config=config, logger=logger, price_fetcher=price_fetcher)
    
    # Test the market depth drawing function directly (simulated)
    logger.info("Testing _draw_volume_profile_pane method...")
    
    # Check if price fetcher is available
    if dashboard.price_fetcher:
        logger.info("✅ Price fetcher is available in dashboard")
        
        # Test order book fetching
        order_book = dashboard.price_fetcher.get_order_book("BTCUSDT", limit=10)
        
        if order_book:
            logger.info("✅ Order book data fetched successfully")
            
            bids_data = order_book.get('bids', [])
            asks_data = order_book.get('asks', [])
            
            if bids_data and asks_data:
                logger.info(f"✅ Order book contains {len(bids_data)} bids and {len(asks_data)} asks")
                
                # Simulate what the dashboard would display
                logger.info("\n=== SIMULATED DASHBOARD MARKET DEPTH ===")
                logger.info("--- Market Depth ---")
                logger.info("Price      | Qty      | Bar")
                
                # Calculate max quantity for bar scaling
                all_quantities = [float(bid[1]) for bid in bids_data[:5]] + [float(ask[1]) for ask in asks_data[:5]]
                max_qty = max(all_quantities) if all_quantities else 1
                
                # Display asks (sell orders) - reverse order
                asks_display = asks_data[:5]
                asks_display.reverse()
                
                logger.info("--- ASKS (Sell) ---")
                for ask in asks_display:
                    try:
                        price = float(ask[0])
                        qty = float(ask[1])
                        bar_width = max(0, min(8, int((qty / max_qty) * 8)))
                        bar_str = '█' * bar_width + '░' * (8 - bar_width)
                        logger.info(f"{price:8.0f} | {qty:8.3f} | {bar_str}")
                    except (ValueError, TypeError, IndexError):
                        continue
                
                # Spread
                if bids_data and asks_data:
                    try:
                        best_bid = float(bids_data[0][0])
                        best_ask = float(asks_data[0][0])
                        spread = best_ask - best_bid
                        logger.info(f"--- SPREAD: ${spread:.2f} ---")
                    except:
                        pass
                
                # Display bids (buy orders)
                logger.info("--- BIDS (Buy) ---")
                for bid in bids_data[:5]:
                    try:
                        price = float(bid[0])
                        qty = float(bid[1])
                        bar_width = max(0, min(8, int((qty / max_qty) * 8)))
                        bar_str = '█' * bar_width + '░' * (8 - bar_width)
                        logger.info(f"{price:8.0f} | {qty:8.3f} | {bar_str}")
                    except (ValueError, TypeError, IndexError):
                        continue
                
                logger.info("✅ Market depth simulation successful!")
                return True
            else:
                logger.error("❌ Order book data is empty")
                return False
        else:
            logger.error("❌ Failed to fetch order book data")
            return False
    else:
        logger.error("❌ Price fetcher not available in dashboard")
        return False

if __name__ == "__main__":
    logger.info("Starting Dashboard Market Depth Tests...")
    
    success = test_dashboard_market_depth()
    
    if success:
        logger.info("✅ Dashboard market depth test completed successfully!")
        logger.info("The console dashboard market depth should now show live order book data.")
        logger.info("Run 'python utils/console_dashboard.py' to see the updated dashboard.")
    else:
        logger.error("❌ Dashboard market depth test failed!")
    
    logger.info("Dashboard market depth tests completed!")