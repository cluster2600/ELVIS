"""
Trading Summary - Clean dashboard display for scalping bot
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import requests
from utils.paper_trade_db import get_open_positions, get_all_trades
from datetime import datetime

def print_trading_summary():
    """Print clean trading summary without spam"""
    try:
        # Get current price
        response = requests.get('https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT', timeout=5)
        current_price = float(response.json()['price'])
        
        # Get positions
        positions = get_open_positions()
        long_count = sum(1 for pos in positions if pos[2].upper() == 'BUY')
        short_count = sum(1 for pos in positions if pos[2].upper() == 'SELL')
        
        # Calculate P&L
        total_pnl = 0
        profit_ready = 0
        
        for pos in positions:
            pos_id, symbol, side, entry_price, quantity, leverage, entry_time = pos
            
            if side.upper() == 'BUY':
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            total_pnl += pnl
            if pnl >= 0.50:
                profit_ready += 1
        
        # Get recent trades
        trades = get_all_trades(limit=50, exclude_test=True)
        recent_pnl = sum(float(trade[6]) for trade in trades[-10:] if len(trade) >= 7)
        
        # Print clean summary
        print(f"\n🚀 SCALPING BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 BTC: ${current_price:,.2f} | Positions: {len(positions)} ({long_count}L/{short_count}S)")
        print(f"💰 Unrealized P&L: ${total_pnl:.2f} | Ready for profit: {profit_ready}")
        print(f"📈 Last 10 trades P&L: ${recent_pnl:.2f}")
        print("="*60)
        
    except Exception as e:
        print(f"Error in trading summary: {e}")

if __name__ == "__main__":
    print_trading_summary()