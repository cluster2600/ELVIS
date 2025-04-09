"""
Script to run the trading dashboard.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trading.scripts.dashboard import TradingDashboard

def main():
    """Run the trading dashboard."""
    try:
        # Set up logging
        logger = logging.getLogger("dashboard")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Create and run the dashboard
        dashboard = TradingDashboard(logger)
        dashboard.run()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {str(e)}")

if __name__ == "__main__":
    main() 