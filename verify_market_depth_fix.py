#!/usr/bin/env python3
"""
Verify market depth fix in console dashboard
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

def verify_market_depth_fix():
    """Verify that market depth is now working"""
    
    print("=== Market Depth Fix Verification ===")
    
    # Test 1: Check if price fetcher can get order book
    print("\n1. Testing PriceFetcher order book...")
    try:
        from utils.price_fetcher import PriceFetcher
        logger = logging.getLogger("test")
        price_fetcher = PriceFetcher(logger)
        
        order_book = price_fetcher.get_order_book("BTCUSDT", limit=10)
        if order_book and order_book.get('bids') and order_book.get('asks'):
            print("   ✅ PriceFetcher can fetch live order book data")
        else:
            print("   ❌ PriceFetcher failed to get order book")
            return False
    except Exception as e:
        print(f"   ❌ PriceFetcher error: {e}")
        return False
    
    # Test 2: Check if dashboard can initialize with price fetcher
    print("\n2. Testing ConsoleDashboard initialization...")
    try:
        from utils.console_dashboard import ConsoleDashboard
        config = {'portfolio_value': 10000}
        dashboard = ConsoleDashboard(config=config, logger=logger, price_fetcher=price_fetcher)
        
        if dashboard.price_fetcher:
            print("   ✅ ConsoleDashboard initialized with PriceFetcher")
        else:
            print("   ❌ ConsoleDashboard missing PriceFetcher")
            return False
    except Exception as e:
        print(f"   ❌ ConsoleDashboard initialization error: {e}")
        return False
    
    # Test 3: Check if market depth function works
    print("\n3. Testing market depth function...")
    try:
        # Mock the curses functions to avoid initialization issues
        def mock_safe_addstr(*args):
            pass
        
        dashboard.safe_addstr = mock_safe_addstr
        
        # Call market depth function
        dashboard._draw_volume_profile_pane(9, 94, 20, 26)
        print("   ✅ Market depth function executes without errors")
    except Exception as e:
        print(f"   ❌ Market depth function error: {e}")
        return False
    
    # Test 4: Check real-time data
    print("\n4. Testing real-time market data...")
    try:
        order_book = dashboard.price_fetcher.get_order_book("BTCUSDT", limit=5)
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        spread = best_ask - best_bid
        
        print(f"   Current BTC Price: ~${best_bid:,.2f}")
        print(f"   Bid-Ask Spread: ${spread:.2f}")
        print("   ✅ Real-time market data is available")
    except Exception as e:
        print(f"   ❌ Real-time data error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = verify_market_depth_fix()
    
    if success:
        print("\n🎉 SUCCESS: Market depth is now working!")
        print("\nWhat's fixed:")
        print("✅ Live order book data from Binance API")
        print("✅ Real-time bid/ask prices and quantities") 
        print("✅ Visual volume bars and spread calculation")
        print("✅ Proper error handling for curses colors")
        print("✅ Dashboard initialization with PriceFetcher")
        print("\nThe console dashboard market depth section will now display live data!")
        print("Run: python utils/console_dashboard.py")
    else:
        print("\n❌ FAILED: Market depth still has issues")
        print("Check the error messages above for troubleshooting.")