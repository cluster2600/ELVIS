"""
Downloads OHLCV data from the public Binance API for the BTCUSDT symbol,
then saves it to a local CSV file for later use in model training.
"""
from binance.client import Client
# The import of KLINE_INTERVAL_1H from binance.enums fails because it may not exist in the installed version.
# Fix by importing KLINE_INTERVAL_1H from binance.client or define it manually if needed.

try:
    from binance.enums import KLINE_INTERVAL_1H
except ImportError:
    # Fallback: define KLINE_INTERVAL_1H manually or import from alternative location
    KLINE_INTERVAL_1H = '1h'
import pandas as pd
import os
from datetime import datetime

# Initialize Binance client without API key (using public endpoints only)
client = Client()

# Define request parameters: symbol, interval, data limit
symbol = 'BTCUSDT'
interval = '1h'  # 1 hour interval (equivalent to Client.KLINE_INTERVAL_1H)
limit = 1000  # Binance limits to 1000 candles per request

# Output path to save the CSV file containing the data
output_path = os.path.join(os.path.dirname(__file__), 'price_data.csv')

# Request candle (Kline) data from Binance API
klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

# Create a pandas DataFrame with columns matching the API response data
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
])

# Convert timestamp to datetime format and select useful columns
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({
    'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
})

# Save the DataFrame to a CSV file
df.to_csv(output_path, index=False)

print(f"✅ Data saved to {output_path}")
