#!/usr/bin/env python3
"""
Test what appears in the right pane of the dashboard
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from utils.console_dashboard import ConsoleDashboard
from utils.price_fetcher import PriceFetcher

def test_right_pane():
    """Test all content that should appear in the right pane"""
    
    print("=== TESTING RIGHT PANE CONTENT ===\n")
    
    # Setup
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    
    price_fetcher = PriceFetcher(logger)
    config = {'portfolio_value': 10000}
    dashboard = ConsoleDashboard(config=config, logger=logger, price_fetcher=price_fetcher)
    
    # Simulate 120x40 terminal (minimum size)
    max_x, max_y = 120, 40
    
    # Calculate right pane positioning (same as dashboard)
    left_pane_width = 35
    chart_pane_width = max(10, max_x - left_pane_width - 30)
    right_pane_width = 28
    chart_pane_x = left_pane_width + 1
    right_pane_x = chart_pane_x + chart_pane_width + 1
    
    print(f"Terminal size: {max_x}x{max_y}")
    print(f"Right pane starts at column: {right_pane_x}")
    print(f"Right pane width: {right_pane_width}")
    print(f"Right pane spans columns: {right_pane_x} to {right_pane_x + right_pane_width}")
    print()
    
    # Check what gets drawn in the right pane
    right_pane_content = []
    
    def capture_right_pane(y, x, text, *args):
        # Only capture content in the right pane area
        if x >= right_pane_x and x < right_pane_x + right_pane_width:
            right_pane_content.append({
                'y': y, 'x': x, 'text': text,
                'relative_x': x - right_pane_x
            })
    
    dashboard.safe_addstr = capture_right_pane
    
    print("🔍 Testing what gets drawn in right pane...")
    
    # Test market depth (should be in right pane)
    print("\n1. Testing Market Depth...")
    market_depth_y = 9
    market_depth_x = right_pane_x + 2
    market_depth_height = max_y - 20
    market_depth_width = right_pane_width - 2
    
    print(f"   Market depth position: y={market_depth_y}, x={market_depth_x}")
    print(f"   Market depth size: h={market_depth_height}, w={market_depth_width}")
    
    try:
        dashboard._draw_volume_profile_pane(market_depth_y, market_depth_x, market_depth_height, market_depth_width)
        market_depth_items = [item for item in right_pane_content if 'Market Depth' in item['text'] or 'ASKS' in item['text'] or 'BIDS' in item['text']]
        print(f"   ✅ Market depth items captured: {len(market_depth_items)}")
        
        if market_depth_items:
            print("   📋 Market depth content:")
            for item in market_depth_items[:5]:
                print(f"      Row {item['y']:2d}, Col {item['relative_x']:2d}: {item['text']}")
        else:
            print("   ❌ No market depth content captured in right pane!")
            
    except Exception as e:
        print(f"   ❌ Market depth error: {e}")
    
    # Test position sizing (also in right pane, below market depth)
    print("\n2. Testing Position Sizing...")
    position_sizing_y = max_y - 15
    position_sizing_x = right_pane_x + 2
    position_sizing_height = 8
    position_sizing_width = right_pane_width - 2
    
    print(f"   Position sizing position: y={position_sizing_y}, x={position_sizing_x}")
    
    try:
        initial_count = len(right_pane_content)
        dashboard._draw_position_sizing_pane(position_sizing_y, position_sizing_x, position_sizing_height, position_sizing_width)
        position_items = right_pane_content[initial_count:]
        print(f"   ✅ Position sizing items captured: {len(position_items)}")
        
        if position_items:
            print("   📋 Position sizing content:")
            for item in position_items[:3]:
                print(f"      Row {item['y']:2d}, Col {item['relative_x']:2d}: {item['text']}")
                
    except Exception as e:
        print(f"   ❌ Position sizing error: {e}")
    
    print(f"\n📊 TOTAL RIGHT PANE CONTENT: {len(right_pane_content)} items")
    
    if len(right_pane_content) > 0:
        print("\n✅ RIGHT PANE HAS CONTENT!")
        print(f"   Content spans rows {min(item['y'] for item in right_pane_content)} to {max(item['y'] for item in right_pane_content)}")
        print(f"   Content spans columns {right_pane_x + min(item['relative_x'] for item in right_pane_content)} to {right_pane_x + max(item['relative_x'] for item in right_pane_content)}")
        
        print("\n🎯 IN THE DASHBOARD, LOOK FOR:")
        print(f"   - Right side of terminal (columns {right_pane_x}-{right_pane_x + right_pane_width})")
        print(f"   - Market Depth starting around row {market_depth_y}")
        print(f"   - Position info around row {position_sizing_y}")
        
        # Show sample content
        print("\n📝 SAMPLE CONTENT YOU SHOULD SEE:")
        market_samples = [item for item in right_pane_content if 'Market' in item['text'] or 'ASKS' in item['text'] or 'Price' in item['text']][:3]
        for item in market_samples:
            print(f"   Row {item['y']:2d}: {item['text']}")
    else:
        print("\n❌ NO CONTENT IN RIGHT PANE!")
        print("   This indicates a positioning or drawing issue.")
    
    return len(right_pane_content) > 0

if __name__ == "__main__":
    success = test_right_pane()
    
    if success:
        print("\n✅ Right pane content is being generated correctly")
        print("If you still don't see it, check:")
        print("1. Terminal width is at least 120 columns")
        print("2. Look at the far right side of the dashboard")
        print("3. Check if terminal is cutting off the right edge")
    else:
        print("\n❌ Right pane content is not being generated")
        print("There's an issue with the drawing functions")