#!/usr/bin/env python3
"""
Debug why main.py might not show the dashboard
"""
import sys
import os
import logging

def debug_main_dashboard():
    """Debug the main.py dashboard startup conditions"""
    
    print("=== DEBUGGING MAIN.PY DASHBOARD STARTUP ===\n")
    
    # Check the conditions that main.py uses to decide whether to show dashboard
    print("1. Checking terminal conditions...")
    
    is_tty = sys.stdout.isatty()
    term_env = os.getenv('TERM')
    
    print(f"   sys.stdout.isatty(): {is_tty}")
    print(f"   TERM environment variable: {term_env}")
    print(f"   Terminal size: {os.get_terminal_size() if hasattr(os, 'get_terminal_size') else 'Unknown'}")
    
    # This is the exact condition from main.py line 513
    will_show_dashboard = is_tty and term_env
    
    print(f"\n   🎯 WILL SHOW DASHBOARD: {will_show_dashboard}")
    
    if not will_show_dashboard:
        print("\n   ❌ DASHBOARD WILL NOT SHOW!")
        print("   Reason:")
        if not is_tty:
            print("      - stdout is not a TTY (terminal)")
        if not term_env:
            print("      - TERM environment variable not set")
        
        print("\n   💡 SOLUTIONS:")
        if not is_tty:
            print("      - Run directly in terminal (not through script/IDE)")
            print("      - Ensure you're running interactively")
        if not term_env:
            print("      - Set TERM variable: export TERM=xterm-256color")
            print("      - Or run: TERM=xterm-256color python main.py --mode paper --log-level INFO")
    else:
        print("\n   ✅ Terminal conditions OK for dashboard")
    
    print("\n2. Testing dashboard creation...")
    
    try:
        # Test if we can create the dashboard components like main.py does
        from core.bootstrap import bootstrap_application
        from core.di import container
        
        print("   Bootstrapping application...")
        bootstrapper = bootstrap_application('paper', 'INFO')
        
        # Get components like main.py does
        logger = container.get('logger')
        price_fetcher = container.get('price_fetcher')
        risk_manager = container.get('risk_manager')
        
        print(f"   Logger: {logger is not None}")
        print(f"   Price fetcher: {price_fetcher is not None}")
        print(f"   Risk manager: {risk_manager is not None}")
        
        # Test dashboard creation
        from utils.console_dashboard import ConsoleDashboard
        
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
        
        print(f"   ✅ Dashboard created successfully")
        print(f"   Dashboard has price_fetcher: {dashboard.price_fetcher is not None}")
        
        # Test market depth function
        display_calls = []
        def capture_calls(y, x, text, *args):
            display_calls.append(f"[{y:2d},{x:3d}] {text}")
        
        dashboard.safe_addstr = capture_calls
        
        # Test market depth with realistic main.py positioning
        dashboard._draw_volume_profile_pane(9, 94, 20, 26)
        
        print(f"   Market depth display calls: {len(display_calls)}")
        if display_calls:
            print("   Sample market depth output:")
            for call in display_calls[:3]:
                print(f"      {call}")
        
        bootstrapper.cleanup()
        
    except Exception as e:
        print(f"   ❌ Dashboard creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Recommendation...")
    
    if will_show_dashboard:
        print("   ✅ Your terminal should show the dashboard when running main.py")
        print("   🔍 If market depth still appears empty:")
        print("      - Look in the RIGHT PANE (columns 94-120)")
        print("      - Ensure terminal is at least 120 columns wide")
        print("      - Check that no errors appear in the terminal")
        print("      - Try: TERM=xterm-256color python main.py --mode paper --log-level DEBUG")
    else:
        print("   ❌ Your terminal setup prevents dashboard from showing")
        print("   🔧 Try these commands:")
        print("      export TERM=xterm-256color")
        print("      python main.py --mode paper --log-level INFO")
        print("   Or:")
        print("      TERM=xterm-256color python main.py --mode paper --log-level INFO")

if __name__ == "__main__":
    debug_main_dashboard()