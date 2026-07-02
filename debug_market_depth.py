#!/usr/bin/env python3
"""
Debug market depth display in console dashboard
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from utils.console_dashboard import ConsoleDashboard
from utils.price_fetcher import PriceFetcher

# Setup logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_market_depth():
    """Debug market depth display"""
    logger.info("=== Debugging Market Depth Display ===")
    
    # Create dashboard with real price fetcher
    price_fetcher = PriceFetcher(logger)
    config = {'portfolio_value': 10000}
    dashboard = ConsoleDashboard(config=config, logger=logger, price_fetcher=price_fetcher)
    
    # Simulate terminal dimensions
    max_y, max_x = 40, 120  # Minimum required size
    
    # Calculate pane dimensions (same as in _draw_frame)
    left_pane_width = 35
    chart_pane_width = max(10, max_x - left_pane_width - 30)
    right_pane_width = 28
    
    chart_pane_x = left_pane_width + 1
    right_pane_x = chart_pane_x + chart_pane_width + 1
    
    # Market depth pane dimensions (from line 112)
    start_y = 9
    start_x = right_pane_x + 2
    height = max_y - 20  # This is the potential issue!
    width = right_pane_width - 2
    
    logger.info(f"Terminal dimensions: {max_x}x{max_y}")
    logger.info(f"Left pane width: {left_pane_width}")
    logger.info(f"Chart pane width: {chart_pane_width}")
    logger.info(f"Right pane width: {right_pane_width}")
    logger.info(f"Chart pane X: {chart_pane_x}")
    logger.info(f"Right pane X: {right_pane_x}")
    
    logger.info(f"\nMarket depth pane:")
    logger.info(f"  start_y: {start_y}")
    logger.info(f"  start_x: {start_x}")
    logger.info(f"  height: {height}")  # This could be too small!
    logger.info(f"  width: {width}")
    
    if height < 15:
        logger.error(f"❌ Height {height} is too small for market depth display!")
        logger.error("Market depth needs at least 15 lines to display properly")
        logger.info("💡 Solution: Increase terminal height or adjust layout")
        return False
    
    # Test if we can fetch order book
    if dashboard.price_fetcher:
        order_book = dashboard.price_fetcher.get_order_book("BTCUSDT", limit=10)
        if order_book:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            logger.info(f"✅ Order book: {len(bids)} bids, {len(asks)} asks")
            
            # Simulate what the pane would display
            logger.info(f"\n=== SIMULATED MARKET DEPTH PANE (height={height}) ===")
            line_count = 0
            
            logger.info("--- Market Depth ---")
            line_count += 1
            
            if line_count < height:
                logger.info("Price      | Qty      | Bar")
                line_count += 1
            
            if line_count < height:
                logger.info("--- ASKS (Sell) ---")
                line_count += 1
            
            # Display 5 asks
            asks_display = asks[:5]
            asks_display.reverse()
            
            for ask in asks_display:
                if line_count >= height:
                    logger.warning(f"⚠️  Ran out of space at line {line_count}")
                    break
                price = float(ask[0])
                qty = float(ask[1])
                logger.info(f"{price:8.0f} | {qty:8.3f} | ████░░░░")
                line_count += 1
            
            if line_count < height:
                logger.info("--- SPREAD: $0.01 ---")
                line_count += 1
            
            if line_count < height:
                logger.info("--- BIDS (Buy) ---")
                line_count += 1
            
            # Display 5 bids
            for bid in bids[:5]:
                if line_count >= height:
                    logger.warning(f"⚠️  Ran out of space at line {line_count}")
                    break
                price = float(bid[0])
                qty = float(bid[1])
                logger.info(f"{price:8.0f} | {qty:8.3f} | ████░░░░")
                line_count += 1
            
            logger.info(f"\nTotal lines used: {line_count} / {height}")
            
            if line_count <= height:
                logger.info("✅ Market depth fits in allocated space")
                return True
            else:
                logger.error("❌ Market depth doesn't fit in allocated space")
                return False
        else:
            logger.error("❌ Failed to fetch order book")
            return False
    else:
        logger.error("❌ No price fetcher available")
        return False

if __name__ == "__main__":
    logger.info("Starting Market Depth Debug...")
    
    success = debug_market_depth()
    
    if success:
        logger.info("✅ Market depth should display correctly")
    else:
        logger.error("❌ Market depth has display issues")
    
    logger.info("Debug completed!")