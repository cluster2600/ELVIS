#!/usr/bin/env python3
"""
Test the dashboard layout to show where market depth appears
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_dashboard_layout():
    """Test where market depth appears in different terminal sizes"""

    print("=== DASHBOARD LAYOUT TEST ===\n")

    # Test different terminal sizes
    terminal_sizes = [
        (120, 40, "Minimum required"),
        (130, 45, "Comfortable size"),
        (140, 50, "Large terminal"),
        (80, 24, "Too small (your current)"),
    ]

    for width, height, description in terminal_sizes:
        print(f"📏 Terminal: {width}x{height} ({description})")

        # Calculate layout (same as dashboard)
        left_pane_width = 35
        chart_pane_width = max(10, width - left_pane_width - 30)
        right_pane_width = 28

        chart_pane_x = left_pane_width + 1
        right_pane_x = chart_pane_x + chart_pane_width + 1

        # Market depth positioning (with the new adaptive logic)
        if width < 130:
            market_depth_x = chart_pane_x + 2
            market_depth_width = min(30, chart_pane_width - 4)
            location = "CENTER PANE"
        else:
            market_depth_x = right_pane_x + 2
            market_depth_width = right_pane_width - 2
            location = "RIGHT PANE"

        market_depth_y = 9
        market_depth_height = height - 20

        print(f"   📍 Market Depth Location: {location}")
        print(
            f"   📊 Position: columns {market_depth_x}-{market_depth_x + market_depth_width}, rows {market_depth_y}-{market_depth_y + market_depth_height}"
        )

        # Check if visible
        if market_depth_x + market_depth_width <= width and market_depth_height > 0:
            print(f"   ✅ VISIBLE - Market depth fits in terminal")
        else:
            print(f"   ❌ HIDDEN - Market depth extends beyond terminal")
            if market_depth_x + market_depth_width > width:
                print(
                    f"      💡 Need {market_depth_x + market_depth_width - width} more columns"
                )
            if market_depth_height <= 0:
                print(f"      💡 Need {20 - height + market_depth_height} more rows")

        print()

    print("🎯 SOLUTION:")
    print("With the updated code, market depth will now appear in the CENTER")
    print("for terminals smaller than 130 columns wide, making it more visible!")
    print()
    print("🔍 TO SEE MARKET DEPTH:")
    print("1. Resize terminal to at least 120x40")
    print("2. Run: python utils/console_dashboard.py")
    print("3. Look for 'Market Depth' in the center or right section")
    print("4. You should see live BTC prices and order book data")


if __name__ == "__main__":
    test_dashboard_layout()
