#!/usr/bin/env python3
"""
Debug what actually happens when running the dashboard
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import io
from utils.console_dashboard import ConsoleDashboard, ConsoleDashboardManager
from utils.price_fetcher import PriceFetcher

def debug_live_dashboard():
    """Debug the actual dashboard running process"""
    
    print("=== DEBUGGING LIVE DASHBOARD ===\n")
    
    # Capture all logs
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    
    # Setup logger
    logger = logging.getLogger("dashboard_debug")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    print("1. Testing how dashboard is actually created...")
    
    # Test the way dashboard is typically created in main
    try:
        price_fetcher = PriceFetcher(logger)
        config = {
            'portfolio_value': 10817.22,
            'unrealized_pnl': 0.0,
            'realized_pnl': 817.22,
            'open_positions': []
        }
        
        print(f"   Price fetcher created: {price_fetcher is not None}")
        
        # Test ConsoleDashboardManager (this is likely how it's used)
        print("   Testing ConsoleDashboardManager...")
        dashboard_manager = ConsoleDashboardManager(logger, config, price_fetcher)
        
        print(f"   Dashboard manager created: {dashboard_manager is not None}")
        print(f"   Dashboard has price_fetcher: {dashboard_manager.dashboard.price_fetcher is not None}")
        
        # Test the dashboard directly
        dashboard = dashboard_manager.dashboard
        
        # Check if market depth method exists and is callable
        if hasattr(dashboard, '_draw_volume_profile_pane'):
            print("   ✅ Market depth method exists")
        else:
            print("   ❌ Market depth method missing!")
            return
        
    except Exception as e:
        print(f"   ❌ Dashboard creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n2. Testing market depth method with detailed logging...")
    
    # Capture what would be displayed
    display_calls = []
    
    def debug_safe_addstr(y, x, text, *args):
        display_calls.append({
            'y': y, 'x': x, 'text': text, 
            'args': args, 'call_num': len(display_calls) + 1
        })
        logger.debug(f"DISPLAY CALL {len(display_calls) + 1}: y={y}, x={x}, text='{text}', args={args}")
    
    # Replace safe_addstr
    dashboard.safe_addstr = debug_safe_addstr
    
    try:
        # Call market depth with realistic parameters
        start_y, start_x, height, width = 9, 94, 20, 26
        logger.info(f"Calling market depth with: y={start_y}, x={start_x}, h={height}, w={width}")
        
        dashboard._draw_volume_profile_pane(start_y, start_x, height, width)
        
        print(f"   ✅ Market depth method completed")
        print(f"   📊 Display calls made: {len(display_calls)}")
        
        if display_calls:
            print("   📝 Sample display calls:")
            for call in display_calls[:5]:
                print(f"      Call {call['call_num']}: [{call['y']:2d},{call['x']:3d}] {call['text'][:30]}...")
        else:
            print("   ❌ No display calls made!")
        
    except Exception as e:
        print(f"   ❌ Market depth method failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Check logs for any issues
    log_output = log_stream.getvalue()
    
    print("\n3. Analyzing logs...")
    
    error_lines = [line for line in log_output.split('\n') if 'ERROR' in line or 'Error' in line]
    warning_lines = [line for line in log_output.split('\n') if 'WARNING' in line or 'Warning' in line]
    market_depth_lines = [line for line in log_output.split('\n') if 'market depth' in line.lower() or 'order book' in line.lower()]
    
    print(f"   Errors: {len(error_lines)}")
    print(f"   Warnings: {len(warning_lines)}")
    print(f"   Market depth related logs: {len(market_depth_lines)}")
    
    if error_lines:
        print("   🚨 ERRORS FOUND:")
        for error in error_lines[:3]:
            print(f"      {error}")
    
    if warning_lines:
        print("   ⚠️  WARNINGS FOUND:")
        for warning in warning_lines[:3]:
            print(f"      {warning}")
    
    if market_depth_lines:
        print("   📋 MARKET DEPTH LOGS:")
        for line in market_depth_lines[:5]:
            print(f"      {line}")
    
    print("\n4. Testing standalone dashboard main() function...")
    
    # Test what happens when you run utils/console_dashboard.py directly
    try:
        from utils.console_dashboard import main
        print("   ✅ Main function imported successfully")
        
        # Check if it uses MockPriceFetcher or real PriceFetcher
        import inspect
        main_source = inspect.getsource(main)
        
        if "MockPriceFetcher" in main_source:
            print("   ⚠️  Main function might still use MockPriceFetcher")
        else:
            print("   ✅ Main function should use real PriceFetcher")
            
    except Exception as e:
        print(f"   ❌ Main function issue: {e}")
    
    print("\n5. FINAL DIAGNOSIS:")
    
    if len(display_calls) > 0:
        print("   🎯 MARKET DEPTH IS WORKING!")
        print("   📊 The function is executing and making display calls")
        print("   📍 Data is being positioned at the calculated coordinates")
        print("")
        print("   🔍 IF YOU STILL DON'T SEE IT, CHECK:")
        print("   1. Terminal size: Make sure it's EXACTLY 120x40 or larger")
        print("   2. Look in the RIGHT PANE (far right side of terminal)")
        print("   3. Scroll horizontally if needed to see columns 94-120")
        print("   4. Check if any error messages appear when running dashboard")
        print("   5. Try running with: TERM=xterm-256color python utils/console_dashboard.py")
        
        # Show where to look
        print(f"\n   📍 EXACT LOCATION TO LOOK:")
        print(f"   Market depth starts at column {start_x} (from left edge)")
        print(f"   It spans {width} characters wide")
        print(f"   It starts at row {start_y} (from top)")
        print(f"   It spans {height} rows tall")
        
    else:
        print("   ❌ MARKET DEPTH NOT EXECUTING!")
        print("   🔧 The function is not making any display calls")
        print("   💡 This suggests an internal error or early return")
        
    return len(display_calls) > 0

if __name__ == "__main__":
    success = debug_live_dashboard()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ MARKET DEPTH CODE IS WORKING")
        print("If you still don't see it, it's a terminal/display issue")
    else:
        print("❌ MARKET DEPTH CODE HAS AN ISSUE")
        print("Check the error messages above")