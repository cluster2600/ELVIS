#!/usr/bin/env python3
"""
Create a dashboard screenshot for documentation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import os

# Create figure
fig = plt.figure(figsize=(12, 8))
fig.suptitle('ELVIS Dashboard', fontsize=16)

# Create grid
gs = GridSpec(3, 2, figure=fig)

# Portfolio value chart
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('Portfolio Value')
dates = pd.date_range(start='2025-03-01', periods=30)
portfolio_values = 1000 + np.cumsum(np.random.normal(5, 10, 30))
ax1.plot(dates, portfolio_values)
ax1.set_ylabel('USD')
ax1.grid(True)

# Performance metrics
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('Performance Metrics')
ax2.axis('off')
metrics = {
    'Total Trades': 42,
    'Win Rate': '68.5%',
    'Profit Factor': 2.34,
    'Sharpe Ratio': 1.87,
    'Max Drawdown': '12.3%',
    'Total PnL': '$342.56'
}
y_pos = 0.9
for metric, value in metrics.items():
    ax2.text(0.1, y_pos, f"{metric}:", fontweight='bold')
    ax2.text(0.5, y_pos, f"{value}")
    y_pos -= 0.15

# Recent trades table
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title('Recent Trades')
ax3.axis('off')
trades = [
    ['12:30:45', 'BTCUSDT', 'BUY', '35,420.50', '0.0025', '+$12.45'],
    ['11:15:22', 'BTCUSDT', 'SELL', '35,380.75', '0.0025', '-$8.20'],
    ['09:45:10', 'BTCUSDT', 'BUY', '35,310.25', '0.0030', '+$21.35'],
    ['08:20:05', 'BTCUSDT', 'SELL', '35,290.50', '0.0020', '+$5.75'],
    ['07:05:30', 'BTCUSDT', 'BUY', '35,250.75', '0.0025', '+$15.60']
]
columns = ['Time', 'Symbol', 'Side', 'Price', 'Quantity', 'PnL']
cell_text = trades

# Add a table at the bottom of the axes
table = ax3.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Strategy signals
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title('Strategy Signals')
ax4.axis('off')
signals = {
    'EMA-RSI Strategy': 'BUY',
    'Technical Strategy': 'HOLD',
    'Mean Reversion Strategy': 'SELL',
    'Trend Following Strategy': 'HOLD'
}
y_pos = 0.9
for strategy, signal in signals.items():
    ax4.text(0.1, y_pos, f"{strategy}:", fontweight='bold')
    color = 'green' if signal == 'BUY' else 'red' if signal == 'SELL' else 'black'
    ax4.text(0.7, y_pos, f"{signal}", color=color, fontweight='bold')
    y_pos -= 0.2

# Market data chart
ax5 = fig.add_subplot(gs[2, :])
ax5.set_title('Market Data - BTCUSDT')
dates = pd.date_range(start='2025-04-01', periods=24, freq='h')
close_prices = 35000 + np.cumsum(np.random.normal(0, 50, 24))
ax5.plot(dates, close_prices, label='Price')
ax5.set_ylabel('USD')
ax5.grid(True)
ax5.legend()

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
os.makedirs('images', exist_ok=True)
plt.savefig('images/dashboard.png', dpi=150)
print("Dashboard image created: images/dashboard.png")
