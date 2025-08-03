#!/usr/bin/env python3
"""
Analyze paper trading logs to understand trading behavior and identify issues.
"""

import sys
import os
import logging
import time
import pandas as pd
from datetime import datetime

from utils.price_fetcher import PriceFetcher
from utils.paper_trade_db import (
    get_all_trades, get_trade_count, get_total_fees, 
    get_pnl_breakdown, get_trade_distribution, 
    get_open_positions, get_conn
)

def analyze_trades_continuously():
    """Continuously analyze and display paper trading logs."""
    logger = logging.getLogger(__name__)
    price_fetcher = PriceFetcher(logger=logger)
    
    try:
        price_fetcher.start()
        print("Connecting to price stream for real-time PnL...")
        time.sleep(5)

        while True:
            # Clear console for continuous refresh
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=== LIVE PAPER TRADING ANALYSIS ===")
            print(f"Last updated: {datetime.now()}")
            
            # --- Database and Trade Analysis ---
            total_trades = get_trade_count(exclude_test=True)
            print(f"\nTotal trades (excluding test): {total_trades}")

            if total_trades == 0:
                print("❌ No trades found!")
                time.sleep(5)
                continue

            trades = get_all_trades(limit=20, exclude_test=True)
            df = pd.DataFrame(trades, columns=['id', 'timestamp', 'symbol', 'side', 'price', 'quantity', 'pnl', 'fee'])
            
            # --- Open Positions with Real-Time PnL ---
            print("\n--- OPEN POSITIONS (Real-Time PnL) ---")
            open_positions = get_open_positions()
            print(f"Current open positions: {len(open_positions)}")
            
            total_unrealized_pnl = 0.0
            if open_positions:
                for pos in open_positions:
                    symbol, entry_price, quantity, leverage = pos[1], pos[2], pos[3], pos[4]
                    current_price = price_fetcher.get_current_price(symbol)
                    
                    if current_price is not None:
                        unrealized_pnl = (current_price - entry_price) * quantity
                        total_unrealized_pnl += unrealized_pnl
                        pnl_str = f"| PnL: ${unrealized_pnl:8.2f}"
                    else:
                        pnl_str = "| PnL: (price unavailable)"
                        
                    print(f"  {symbol} | Entry: ${entry_price:8.2f} | Qty: {quantity:.6f} | Leverage: {leverage:.1f}x {pnl_str}")
                
                print(f"\nTotal Unrealized PnL: ${total_unrealized_pnl:.2f}")

            # --- Recent Trades ---
            print("\n--- RECENT TRADES (Last 20) ---")
            for _, trade in df.iterrows():
                print(f"{trade['timestamp']} | {trade['side']:4s} | ${trade['price']:8.2f} | {trade['quantity']:8.6f} BTC | PnL: ${trade['pnl']:8.2f}")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nStopping analysis.")
    finally:
        if price_fetcher.running:
            price_fetcher.ws.close()
            print("Price stream disconnected.")

def check_database_connection():
    """Check if database connection is working."""
    try:
        conn = get_conn()
        if conn is None:
            return False
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        conn.close()
        return True
    except Exception:
        return False
