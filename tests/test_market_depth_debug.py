#!/usr/bin/env python3
"""
Test market depth with detailed debugging
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import curses
import logging

from utils.console_dashboard import ConsoleDashboard
from utils.price_fetcher import PriceFetcher


def test_market_depth_with_logging():
    """Test market depth with full debug logging"""

    # Setup detailed logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("=== Testing Market Depth with Debug Logging ===")

    # Create dashboard with real price fetcher
    price_fetcher = PriceFetcher(logger)
    config = {"portfolio_value": 10000}
    dashboard = ConsoleDashboard(
        config=config, logger=logger, price_fetcher=price_fetcher
    )

    # Mock safe_addstr to see what would be displayed
    display_lines = []

    def mock_safe_addstr(y, x, text, *args):
        display_lines.append(f"Line {y:2d}: {text}")
        logger.info(f"DISPLAY: Line {y:2d} at x={x}: {text}")

    # Replace safe_addstr with our mock
    dashboard.safe_addstr = mock_safe_addstr

    logger.info("Calling _draw_volume_profile_pane...")

    # Call the market depth function directly
    start_y, start_x, height, width = 9, 94, 20, 26
    dashboard._draw_volume_profile_pane(start_y, start_x, height, width)

    logger.info(
        f"Market depth function completed. Total lines displayed: {len(display_lines)}"
    )

    if display_lines:
        logger.info("\n=== WHAT WOULD BE DISPLAYED ===")
        for line in display_lines:
            logger.info(line)
        return True
    else:
        logger.error("❌ No lines were displayed!")
        return False


if __name__ == "__main__":
    success = test_market_depth_with_logging()

    if success:
        print("✅ Market depth function is working and displaying data")
    else:
        print("❌ Market depth function is not displaying data")
