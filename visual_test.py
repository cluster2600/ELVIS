#!/usr/bin/env python3
"""
Visual test to show exactly where market depth appears
"""
import curses
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.price_fetcher import PriceFetcher
import logging

def visual_test(stdscr):
    """Visual test showing where market depth should appear"""
    
    # Setup colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    
    stdscr.clear()
    
    # Get terminal size
    max_y, max_x = stdscr.getmaxyx()
    
    # Show terminal info
    stdscr.addstr(0, 0, f"Terminal Size: {max_x} cols x {max_y} rows", curses.color_pair(4))
    stdscr.addstr(1, 0, f"Minimum needed: 120 cols x 40 rows", curses.color_pair(4))
    
    if max_x < 120 or max_y < 40:
        stdscr.addstr(3, 0, "WARNING: Terminal too small!", curses.color_pair(2))
        stdscr.addstr(4, 0, f"Resize to at least 120x40", curses.color_pair(2))
    else:
        stdscr.addstr(3, 0, "Terminal size OK!", curses.color_pair(1))
    
    # Calculate dashboard layout
    left_pane_width = 35
    chart_pane_width = max(10, max_x - left_pane_width - 30)
    right_pane_width = 28
    chart_pane_x = left_pane_width + 1
    right_pane_x = chart_pane_x + chart_pane_width + 1
    
    # Show layout
    stdscr.addstr(5, 0, "Dashboard Layout:", curses.color_pair(3))
    stdscr.addstr(6, 0, f"Left pane: cols 0-{left_pane_width}")
    stdscr.addstr(7, 0, f"Chart pane: cols {chart_pane_x}-{chart_pane_x + chart_pane_width}")
    stdscr.addstr(8, 0, f"Right pane: cols {right_pane_x}-{max_x-1}", curses.color_pair(1))
    
    # Draw visual indicators for panes
    if max_y > 15:
        # Left pane border
        for y in range(10, min(max_y-5, 25)):
            if left_pane_width < max_x:
                stdscr.addch(y, left_pane_width, '|', curses.color_pair(3))
        
        # Chart pane border  
        for y in range(10, min(max_y-5, 25)):
            if chart_pane_x + chart_pane_width < max_x:
                stdscr.addch(y, chart_pane_x + chart_pane_width, '|', curses.color_pair(3))
        
        # Right pane area (where market depth should be)
        if right_pane_x < max_x:
            for y in range(10, min(max_y-5, 25)):
                if right_pane_x < max_x:
                    stdscr.addch(y, right_pane_x, '|', curses.color_pair(1))
            
            # Mark market depth area
            market_depth_y = 12
            market_depth_x = right_pane_x + 2
            
            if market_depth_x < max_x and market_depth_y < max_y:
                stdscr.addstr(market_depth_y, market_depth_x, "MARKET DEPTH HERE", curses.color_pair(2))
                
                # Get actual live data to show
                try:
                    logger = logging.getLogger("visual_test")
                    price_fetcher = PriceFetcher(logger)
                    order_book = price_fetcher.get_order_book("BTCUSDT", limit=3)
                    
                    if order_book and order_book.get('bids') and order_book.get('asks'):
                        # Show sample live data
                        y_offset = market_depth_y + 2
                        if y_offset < max_y and market_depth_x < max_x:
                            stdscr.addstr(y_offset, market_depth_x, "Live BTC Data:", curses.color_pair(1))
                            
                            # Show best bid/ask
                            best_bid = float(order_book['bids'][0][0])
                            best_ask = float(order_book['asks'][0][0])
                            
                            if y_offset + 1 < max_y:
                                stdscr.addstr(y_offset + 1, market_depth_x, f"Bid: ${best_bid:,.2f}", curses.color_pair(1))
                            if y_offset + 2 < max_y:
                                stdscr.addstr(y_offset + 2, market_depth_x, f"Ask: ${best_ask:,.2f}", curses.color_pair(2))
                        
                except Exception as e:
                    if market_depth_y + 2 < max_y and market_depth_x < max_x:
                        stdscr.addstr(market_depth_y + 2, market_depth_x, f"Data error: {str(e)[:20]}")
    
    # Instructions
    instruction_y = max_y - 8
    if instruction_y > 0:
        stdscr.addstr(instruction_y, 0, "INSTRUCTIONS:", curses.color_pair(4))
        stdscr.addstr(instruction_y + 1, 0, "1. Look for GREEN vertical line - that's the right pane")
        stdscr.addstr(instruction_y + 2, 0, "2. Market depth should appear just right of that line")
        stdscr.addstr(instruction_y + 3, 0, "3. If you see 'MARKET DEPTH HERE', the positioning is correct")
        stdscr.addstr(instruction_y + 4, 0, "4. In real dashboard, live order book data appears there")
        stdscr.addstr(instruction_y + 5, 0, "")
        stdscr.addstr(instruction_y + 6, 0, "Press any key to exit...", curses.color_pair(3))
    
    stdscr.refresh()
    stdscr.getch()

if __name__ == "__main__":
    print("Starting visual test...")
    print("This will show you exactly where market depth should appear.")
    print("Make sure your terminal is at least 120 columns wide.")
    print()
    
    try:
        curses.wrapper(visual_test)
        print("\nNow run the actual dashboard:")
        print("python utils/console_dashboard.py")
        print()
        print("Look in the same location you saw 'MARKET DEPTH HERE'")
        print("You should see live BTC order book data there!")
        
    except Exception as e:
        print(f"Visual test failed: {e}")
        print("This might indicate a terminal compatibility issue.")
        print("Try using a different terminal application.")