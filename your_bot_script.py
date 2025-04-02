import ccxt
import pandas as pd
from datetime import datetime
import time
import numpy as np
import talib
import logging

# API Configuration (replace with your keys)
API_KEY = 'YOUR_API_KEY'
SECRET_KEY = 'YOUR_SECRET_KEY'

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True
})

SYMBOL = 'BTC/USDT'  # Trading pair
TIMEFRAME = '5m'     # Timeframe for data collection

def get_order_book_depth(symbol, price):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        bids_total = sum(entry[1] for entry in orderbook['bids'][:5])
        asks_total = sum(entry[1] for entry in orderbook['asks'][:5])
        return (bids_total - asks_total) / (bids_total + asks_total + 1e-9)
    except Exception as e:
        logging.error(f"Order book fetch failed: {str(e)}", exc_info=True)
        return 0.0

def get_processed_data():
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=150)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Technical indicators
    df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
    df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # Volatility calculation
    df['volatility'] = df['high'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # Order book balance feature
    current_price = exchange.fetch_ticker(SYMBOL)['last']
    df['order_book_balance'] = get_order_book_depth(SYMBOL, current_price)
    
    return df.dropna()

def trading_strategy(df):
    buy_signal = (df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1]) and \
                 (df['rsi'].iloc[-1] < 30) and \
                 (df['volatility'].iloc[-1] > 0.005)
    
    sell_signal = (df['ema_9'].iloc[-1] < df['ema_21'].iloc[-1]) or \
                  (df['rsi'].iloc[-1] > 70) or \
                  (df['order_book_balance'].iloc[-1] < -0.3)
    
    return buy_signal, sell_signal

def execute_trade(action, volatility):
    current_price = exchange.fetch_ticker(SYMBOL)['last']
    balance = exchange.fetch_free_balance()
    
    if action == 'buy':
        usdt_available = float(balance.get('USDT', 0))
        amount = (usdt_available * 0.98) / current_price
        stop_loss_level = 1.5
        stop_loss_price = current_price * (1 - volatility * stop_loss_level)
        
        # Create orders with proper parameters
        exchange.create_order(
            SYMBOL,
            'limit',
            'buy',
            amount,
            current_price,
            {
                'stopPrice': stop_loss_price,
                'takeProfit': current_price * 1.02
            }
        )
        logging.info(f"Bought {amount:.5f} BTC at {current_price}")
        
    elif action == 'sell':
        btc_available = float(balance.get('BTC', 0))
        if btc_available > 0.001:
            exchange.create_market_sell_order(SYMBOL, btc_available * 0.995)
            logging.info(f"Sold {btc_available:.5f} BTC at market price")

def main():
    try:
        df = get_processed_data()
        buy_signal, sell_signal = trading_strategy(df)
        
        if buy_signal and not sell_signal:
            execute_trade('buy', df['volatility'].iloc[-1])
        elif sell_signal and not buy_signal:
            execute_trade('sell', 0)  # Pass placeholder volatility
        
    except Exception as e:
        logging.error(f"Critical error: {str(e)}", exc_info=True)

if __name__ == '__main__':
    logging.basicConfig(
        filename='trading.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    while True:
        main()
        time.sleep(60)  # Run every minute
