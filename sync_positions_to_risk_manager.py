#!/usr/bin/env python3
"""
Sync existing open positions to risk manager
This fixes the 6 positions that are already open but not managed
"""

import json
import os
import sys

import requests

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from core.bootstrap import bootstrap_application
from core.di import container


def sync_existing_positions():
    """Load existing positions and add them to risk manager"""
    try:
        # Bootstrap the application to get container
        bootstrap_application()

        # Get dependencies
        risk_manager = container.get("risk_manager")
        logger = container.get("logger")

        logger.info("🔄 Syncing existing open positions to risk manager...")

        # Get current open positions from API
        try:
            response = requests.get("http://localhost:5050/open_positions", timeout=5)
            if response.status_code == 200:
                positions = response.json()
                logger.info(f"📊 Found {len(positions)} open positions to sync")

                for pos in positions:
                    try:
                        # Convert API format to risk manager format
                        position_data = {
                            "symbol": pos["symbol"],
                            "side": pos["side"],
                            "entry_price": pos["entry_price"],
                            "quantity": pos["quantity"],
                            "leverage": pos.get("leverage", 100),
                            "timestamp": pos["entry_time"],
                        }

                        # Add to risk manager
                        risk_manager.add_position(pos["symbol"], position_data)
                        logger.info(
                            f"✅ Synced position: {pos['side']} {pos['quantity']:.6f} {pos['symbol']} @ ${pos['entry_price']}"
                        )

                    except Exception as e:
                        logger.error(f"❌ Failed to sync position {pos.get('id')}: {e}")

                logger.info(
                    f"🎉 Sync complete! {len(positions)} positions now managed by risk manager"
                )
                logger.info(
                    "🔥 These positions will now be monitored for stop-loss and take-profit"
                )

            else:
                logger.error(
                    f"❌ Failed to fetch positions: HTTP {response.status_code}"
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to trading API: {e}")
            logger.error("💡 Make sure the trading bot is running first")

    except Exception as e:
        logger.error(f"❌ Position sync failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = sync_existing_positions()
    sys.exit(0 if success else 1)
