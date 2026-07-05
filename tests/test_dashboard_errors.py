#!/usr/bin/env python3
"""
Test for dashboard errors when running market depth
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import io
import logging

from utils.console_dashboard import ConsoleDashboard
from utils.price_fetcher import PriceFetcher


def test_dashboard_errors():
    """Test if any errors occur during dashboard operations"""

    # Capture all log output
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)

    # Set up logger with capture
    logger = logging.getLogger("test_dashboard")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    print("=== Testing Dashboard for Errors ===")

    try:
        # Create dashboard
        price_fetcher = PriceFetcher(logger)
        config = {"portfolio_value": 10000}
        dashboard = ConsoleDashboard(
            config=config, logger=logger, price_fetcher=price_fetcher
        )

        # Mock the safe_addstr to avoid curses issues
        displayed_content = []

        def mock_safe_addstr(y, x, text, *args):
            displayed_content.append(f"[{y:2d},{x:3d}] {text}")

        dashboard.safe_addstr = mock_safe_addstr

        print("Testing _draw_frame method...")

        # Simulate the _draw_frame method call that includes market depth
        # This is similar to what happens in the real dashboard
        try:
            # Simulate terminal size
            class MockStdscr:
                def getmaxyx(self):
                    return 40, 120

                def clear(self):
                    pass

                def refresh(self):
                    pass

                def addstr(self, *args):
                    pass

            dashboard.stdscr = MockStdscr()

            # Call _draw_frame which should call _draw_volume_profile_pane
            dashboard._draw_frame()

            print(f"✅ _draw_frame completed, {len(displayed_content)} items displayed")

            # Check if market depth content was displayed
            market_depth_items = [
                item
                for item in displayed_content
                if "Market Depth" in item
                or "ASKS" in item
                or "BIDS" in item
                or "SPREAD" in item
            ]

            print(f"Market depth items found: {len(market_depth_items)}")

            if market_depth_items:
                print("✅ Market depth content is being displayed:")
                for item in market_depth_items[:5]:  # Show first 5
                    print(f"  {item}")
            else:
                print("❌ No market depth content found in display")
                print("First 10 displayed items:")
                for item in displayed_content[:10]:
                    print(f"  {item}")

        except Exception as e:
            print(f"❌ Error in _draw_frame: {e}")
            import traceback

            traceback.print_exc()

        # Check captured logs for errors
        log_output = log_capture.getvalue()
        error_lines = [line for line in log_output.split("\n") if "ERROR" in line]
        warning_lines = [line for line in log_output.split("\n") if "WARNING" in line]

        print(f"\nLog analysis:")
        print(f"  Errors: {len(error_lines)}")
        print(f"  Warnings: {len(warning_lines)}")

        if error_lines:
            print("❌ Errors found:")
            for error in error_lines[:3]:  # Show first 3 errors
                print(f"  {error}")

        if warning_lines:
            print("⚠️  Warnings found:")
            for warning in warning_lines[:3]:  # Show first 3 warnings
                print(f"  {warning}")

        return len(market_depth_items) > 0

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dashboard_errors()

    if success:
        print("\n✅ Market depth is working in dashboard context")
    else:
        print("\n❌ Market depth is not working in dashboard context")
        print("Check error messages above for details.")
