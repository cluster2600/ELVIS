#!/usr/bin/env python3
"""
Start the complete ELVIS trading bot with all services
"""

import os
import sys
import threading
import time

# Add project root to path
sys.path.insert(0, "/app")

print("🚀 ELVIS Trading Bot - Complete System Starting...")

# Start the trading bot main loop
if __name__ == "__main__":
    try:
        # Set environment for headless operation
        os.environ["TERM"] = ""  # Force headless mode

        # Import and run main
        from main import main

        print("✅ Starting ELVIS trading bot in paper mode...")
        main(mode="paper", log_level="INFO")

    except Exception as e:
        print(f"❌ Error starting trading bot: {e}")
        import traceback

        traceback.print_exc()
