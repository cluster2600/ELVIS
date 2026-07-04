"""
Test script for the trading dashboard with mock data.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trading.scripts.dashboard import TradingDashboard


class MockTradingDashboard(TradingDashboard):
    """Trading dashboard with mock data for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize the mock dashboard."""
        super().__init__(*args, **kwargs)

        # Initialize state with mock values
        self.portfolio_value = 100000  # Initial capital
        self.position_size = 0.1  # Mock position size
        self.entry_price = 50000  # Mock entry price
        self.current_price = 50000  # Initial price

        # Generate mock trades
        self.trades = []
        start_time = datetime.now() - timedelta(hours=1)

        for i in range(10):
            trade_time = start_time + timedelta(minutes=i * 6)
            side = "BUY" if i % 2 == 0 else "SELL"
            price = 50000 + np.random.normal(0, 100)
            size = np.random.uniform(0.001, 0.01)
            pnl = np.random.normal(0, 50)

            self.trades.append(
                {
                    "time": trade_time.strftime("%H:%M:%S"),
                    "side": side,
                    "price": price,
                    "size": size,
                    "pnl": pnl,
                }
            )

        # Initialize metrics
        self.metrics = {
            "total_trades": len(self.trades),
            "winning_trades": len([t for t in self.trades if t["pnl"] > 0]),
            "losing_trades": len([t for t in self.trades if t["pnl"] < 0]),
            "win_rate": 0.5,
            "profit_factor": 1.5,
            "sharpe_ratio": 1.2,
        }

    def update_market_data(self):
        """Update market data with mock values."""
        try:
            # Simulate price movement with mean reversion
            price_change = np.random.normal(0, 10)
            self.current_price = max(
                40000, min(60000, self.current_price + price_change)
            )

            # Update portfolio value
            if self.position_size != 0:
                pnl = (self.current_price - self.entry_price) * self.position_size
                self.portfolio_value = 100000 + pnl  # Base value + PnL

        except Exception as e:
            self.console.print(f"[red]Error updating market data: {str(e)}[/red]")

    def create_market_panel(self):
        """Create mock market data panel."""
        try:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Price")
            table.add_column("Size")

            # Generate mock order book
            mid_price = self.current_price
            for i in range(5):
                ask_price = mid_price + (i + 1) * 10
                ask_size = np.random.uniform(0.1, 1.0)
                table.add_row(f"${ask_price:,.2f}", f"{ask_size:.4f}")

            table.add_row("─" * 20, "─" * 10)

            for i in range(5):
                bid_price = mid_price - (i + 1) * 10
                bid_size = np.random.uniform(0.1, 1.0)
                table.add_row(f"${bid_price:,.2f}", f"{bid_size:.4f}")

            return Panel(table, title="Order Book")

        except Exception as e:
            return Panel(f"Error creating market panel: {str(e)}", title="Order Book")


def main():
    """Main function."""
    try:
        dashboard = MockTradingDashboard()
        dashboard.run()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {str(e)}")


if __name__ == "__main__":
    main()
