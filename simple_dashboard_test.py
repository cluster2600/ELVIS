#!/usr/bin/env python3
"""
Simple test to verify dashboard market depth visibility
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import curses
import logging
import time
from utils.console_dashboard import ConsoleDashboard
from utils.price_fetcher import PriceFetcher

def simple_dashboard_test(stdscr):
    """Test dashboard with minimal setup"""
    
    # Setup
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(6, curses.COLOR_WHITE, -1)
    
    # Clear screen
    stdscr.clear()
    
    # Create dashboard
    logger = logging.getLogger("simple_test")
    price_fetcher = PriceFetcher(logger)
    config = {'portfolio_value': 10000}
    dashboard = ConsoleDashboard(config=config, logger=logger, price_fetcher=price_fetcher)
    dashboard.stdscr = stdscr
    
    # Get terminal size
    max_y, max_x = stdscr.getmaxyx()
    
    # Display basic info
    stdscr.addstr(0, 0, f"Terminal Size: {max_x}x{max_y}")
    stdscr.addstr(1, 0, "Testing Market Depth Visibility...")
    
    if max_y < 40 or max_x < 120:
        stdscr.addstr(3, 0, "ERROR: Terminal too small!")
        stdscr.addstr(4, 0, f"Need: 120x40, Have: {max_x}x{max_y}")
        stdscr.addstr(5, 0, "Press any key to exit...")
        stdscr.refresh()
        stdscr.getch()
        return
    
    # Calculate market depth position (same as dashboard)
    left_pane_width = 35
    chart_pane_width = max(10, max_x - left_pane_width - 30)
    right_pane_width = 28
    chart_pane_x = left_pane_width + 1
    right_pane_x = chart_pane_x + chart_pane_width + 1
    
    start_y = 9
    start_x = right_pane_x + 2
    height = max_y - 20
    width = right_pane_width - 2
    
    stdscr.addstr(3, 0, f"Market Depth Position: y={start_y}, x={start_x}, h={height}, w={width}")
    
    # Draw a simple box where market depth should be
    try:
        # Draw border
        for i in range(width):
            stdscr.addch(start_y, start_x + i, '-')
            stdscr.addch(start_y + height - 1, start_x + i, '-')
        
        for i in range(height):
            stdscr.addch(start_y + i, start_x, '|')
            stdscr.addch(start_y + i, start_x + width - 1, '|')
        
        # Add title
        stdscr.addstr(start_y + 1, start_x + 2, "MARKET DEPTH HERE")
        stdscr.addstr(start_y + 2, start_x + 2, "Position looks OK")
        
        stdscr.addstr(5, 0, "✅ Market depth position is visible!")
        
    except Exception as e:
        stdscr.addstr(5, 0, f"❌ Error drawing at market depth position: {e}")
    
    # Test the actual market depth function
    try:
        stdscr.addstr(7, 0, "Testing actual market depth function...")
        dashboard._draw_volume_profile_pane(start_y, start_x, height, width)
        stdscr.addstr(8, 0, "✅ Market depth function completed!")
    except Exception as e:
        stdscr.addstr(8, 0, f"❌ Market depth function error: {str(e)[:50]}")
    
    stdscr.addstr(max_y - 2, 0, "Press any key to exit...")
    stdscr.refresh()
    stdscr.getch()

if __name__ == "__main__":
    print("Starting simple dashboard test...")
    print("This will show where the market depth should appear.")
    print("Make sure your terminal is at least 120x40.")
    
    try:
        curses.wrapper(simple_dashboard_test)
    except KeyboardInterrupt:
        print("Test interrupted.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()