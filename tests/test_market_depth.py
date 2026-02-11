#!/usr/bin/env python3
"""
Test market depth / order book functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from utils.price_fetcher import PriceFetcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_market_depth():
    """Test market depth / order book data fetching"""
    logger.info("=== Testing Market Depth / Order Book ===")
    
    # Create price fetcher
    price_fetcher = PriceFetcher(logger)
    
    # Test order book fetching
    logger.info("Fetching order book for BTCUSDT...")
    order_book = price_fetcher.get_order_book("BTCUSDT", limit=10)
    
    if order_book:
        logger.info("✅ Order book data received successfully!")
        
        # Check structure
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        logger.info(f"Bids count: {len(bids)}")
        logger.info(f"Asks count: {len(asks)}")
        
        if bids and asks:
            logger.info("\n=== TOP 5 ASKS (Sell Orders) ===")
            for i, ask in enumerate(asks[:5]):
                price, qty = float(ask[0]), float(ask[1])
                logger.info(f"{i+1}. ${price:,.2f} | {qty:.6f} BTC")
            
            # Calculate spread
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            
            logger.info(f"\n=== SPREAD ===")
            logger.info(f"Best Bid: ${best_bid:,.2f}")
            logger.info(f"Best Ask: ${best_ask:,.2f}")
            logger.info(f"Spread: ${spread:.2f}")
            
            logger.info("\n=== TOP 5 BIDS (Buy Orders) ===")
            for i, bid in enumerate(bids[:5]):
                price, qty = float(bid[0]), float(bid[1])
                logger.info(f"{i+1}. ${price:,.2f} | {qty:.6f} BTC")
                
        else:
            logger.warning("❌ Order book data is empty!")
            
    else:
        logger.error("❌ Failed to fetch order book data!")
        
    # Test if this works with dashboard format
    logger.info("\n=== DASHBOARD FORMAT TEST ===")
    if order_book and order_book.get('bids') and order_book.get('asks'):
        bids_data = order_book['bids']
        asks_data = order_book['asks']
        
        # Calculate max quantity for bar scaling (like in dashboard)
        all_quantities = [float(bid[1]) for bid in bids_data[:5]] + [float(ask[1]) for ask in asks_data[:5]]
        max_qty = max(all_quantities) if all_quantities else 1
        
        logger.info("Market Depth Display Preview:")
        logger.info("Price      | Qty      | Bar")
        logger.info("--- ASKS (Sell) ---")
        
        asks_display = asks_data[:5]
        asks_display.reverse()  # Show highest price first
        
        for ask in asks_display:
            try:
                price = float(ask[0])
                qty = float(ask[1])
                bar_width = max(0, min(8, int((qty / max_qty) * 8)))
                bar_str = '█' * bar_width + '░' * (8 - bar_width)
                logger.info(f"{price:8.0f} | {qty:8.3f} | {bar_str}")
            except (ValueError, TypeError, IndexError):
                continue
        
        logger.info(f"--- SPREAD: ${spread:.2f} ---")
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
    
    return order_book is not None

if __name__ == "__main__":
    logger.info("Starting Market Depth Tests...")
    
    success = test_market_depth()
    
    if success:
        logger.info("✅ Market depth test completed successfully!")
        logger.info("The console dashboard should display live order book data.")
    else:
        logger.error("❌ Market depth test failed!")
    
    logger.info("Market depth tests completed!")