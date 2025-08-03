#!/usr/bin/env python3

import sys
sys.path.append('.')
from utils.paper_trade_db import get_conn
from datetime import datetime, timedelta
import psycopg2

def analyze_trading_performance():
    """Analyze trading performance over the last 24 hours"""
    
    print("=== BITCOIN ATH TRADING PERFORMANCE ANALYSIS ===")
    print("Current BTC Price: $120,423.15 - NEAR ATH TERRITORY!")
    print()
    
    try:
        conn = get_conn()
        if not conn:
            print("Could not connect to database")
            return
            
        with conn.cursor() as c:
            # Get trades from last 24 hours
            c.execute("""
                SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                AND side != 'TEST'
                ORDER BY timestamp DESC
            """)
            trades_24h = c.fetchall()
            
            print("=== TRADES LAST 24 HOURS ===")
            print(f"Total trades: {len(trades_24h)}")
            print()
            
            if not trades_24h:
                print("No trades found in last 24 hours")
                return
                
            # Calculate basic metrics
            total_pnl = sum(trade[6] for trade in trades_24h)
            total_fees = sum(trade[7] for trade in trades_24h)
            winning_trades = [t for t in trades_24h if t[6] > 0]
            losing_trades = [t for t in trades_24h if t[6] < 0]
            
            print("=== FINANCIAL PERFORMANCE ===")
            print(f"Total P&L: ${total_pnl:.2f}")
            print(f"Total Fees: ${total_fees:.2f}")
            print(f"Net P&L (after fees): ${total_pnl - total_fees:.2f}")
            print()
            
            # Win/Loss Analysis
            win_rate = len(winning_trades)/len(trades_24h)*100 if trades_24h else 0
            print("=== WIN/LOSS ANALYSIS ===")
            print(f"Winning trades: {len(winning_trades)} ({win_rate:.1f}%)")
            print(f"Losing trades: {len(losing_trades)} ({(100-win_rate):.1f}%)")
            
            if winning_trades:
                avg_win = sum(t[6] for t in winning_trades) / len(winning_trades)
                max_win = max(t[6] for t in winning_trades)
                print(f"Average win: ${avg_win:.2f}")
                print(f"Max win: ${max_win:.2f}")
                
            if losing_trades:
                avg_loss = sum(t[6] for t in losing_trades) / len(losing_trades)
                max_loss = min(t[6] for t in losing_trades)
                print(f"Average loss: ${avg_loss:.2f}")
                print(f"Max loss: ${max_loss:.2f}")
            print()
            
            # Analyze trading sides - THIS IS CRITICAL FOR ATH ANALYSIS
            buy_trades = [t for t in trades_24h if t[3] == 'BUY']
            sell_trades = [t for t in trades_24h if t[3] == 'SELL']
            short_trades = [t for t in trades_24h if t[3] == 'SHORT']
            long_trades = [t for t in trades_24h if t[3] == 'LONG']
            
            print("=== TRADE DISTRIBUTION (CRITICAL FOR ATH) ===")
            print(f"BUY trades: {len(buy_trades)}")
            print(f"SELL trades: {len(sell_trades)}")
            print(f"SHORT trades: {len(short_trades)}")
            print(f"LONG trades: {len(long_trades)}")
            
            # Check for shorts during ATH - MAJOR PROBLEM!
            if short_trades:
                short_pnl = sum(t[6] for t in short_trades)
                print(f"*** SHORT P&L: ${short_pnl:.2f} ***")
                print("*** WARNING: SHORTING DURING ATH IS THE MAIN PROBLEM! ***")
                
                # Show short trade details
                print()
                print("=== SHORT TRADES DURING ATH (MAJOR ISSUE) ===")
                for i, trade in enumerate(short_trades):
                    id, timestamp, symbol, side, price, quantity, pnl, fee = trade
                    print(f"{i+1}. {timestamp} | SHORT at ${price:.2f} | P&L: ${pnl:.2f}")
            print()
            
            # Position sizing analysis
            print("=== POSITION SIZING ANALYSIS ===")
            avg_quantity = sum(t[5] for t in trades_24h) / len(trades_24h)
            print(f"Average position size: {avg_quantity:.4f} BTC")
            
            # Check if $1 profit targets are being hit
            small_wins = [t for t in winning_trades if 0 < t[6] < 2]  # $0-$2 wins
            print(f"Small wins ($0-$2): {len(small_wins)} trades")
            if small_wins:
                avg_small_win = sum(t[6] for t in small_wins) / len(small_wins)
                print(f"Average small win: ${avg_small_win:.2f}")
            print()
            
            # Fee analysis
            print("=== FEE IMPACT ANALYSIS ===")
            print(f"Total fees paid: ${total_fees:.2f}")
            print(f"Fee to profit ratio: {(total_fees/abs(total_pnl)*100):.1f}%" if total_pnl != 0 else "N/A")
            
            # Check if fees are eating profits
            if total_fees > abs(total_pnl):
                print("*** CRITICAL: Fees are higher than total profits! ***")
            print()
            
            # Trading frequency analysis
            print("=== TRADING FREQUENCY ANALYSIS ===")
            if len(trades_24h) > 1:
                # Calculate time between trades
                trade_times = [t[1] for t in trades_24h]
                time_diffs = []
                for i in range(1, len(trade_times)):
                    diff = (trade_times[i-1] - trade_times[i]).total_seconds() / 60
                    time_diffs.append(diff)
                
                avg_interval = sum(time_diffs) / len(time_diffs)
                min_interval = min(time_diffs)
                print(f"Average time between trades: {avg_interval:.1f} minutes")
                print(f"Minimum time between trades: {min_interval:.1f} minutes")
                
                # Check if emergency fixes are working
                if min_interval < 5:
                    print("*** WARNING: Trades happening too frequently! Emergency cooldown not working ***")
                elif avg_interval > 30:
                    print("*** Good: Cooldown periods seem to be working ***")
            print()
            
            # Show recent trades
            print("=== RECENT TRADES (Last 15) ===")
            for i, trade in enumerate(trades_24h[:15]):
                id, timestamp, symbol, side, price, quantity, pnl, fee = trade
                # Highlight problematic trades
                warning = ""
                if side == "SHORT" and price > 115000:
                    warning = " ← PROBLEM: SHORT DURING ATH!"
                elif pnl < -10:
                    warning = " ← BIG LOSS!"
                elif fee > abs(pnl):
                    warning = " ← FEE > PROFIT!"
                    
                print(f"{i+1:2}. {timestamp} | {side:5} | ${price:8.2f} | Qty: {quantity:6.4f} | P&L: ${pnl:6.2f} | Fee: ${fee:4.2f}{warning}")
            
            print()
            
            # Get older trades for comparison
            c.execute("""
                SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                AND timestamp < NOW() - INTERVAL '24 hours'
                AND side != 'TEST'
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            older_trades = c.fetchall()
            
            if older_trades:
                older_pnl = sum(trade[6] for trade in older_trades)
                older_fees = sum(trade[7] for trade in older_trades)
                print("=== COMPARISON WITH PREVIOUS PERIOD ===")
                print(f"Previous period P&L: ${older_pnl:.2f}")
                print(f"Previous period fees: ${older_fees:.2f}")
                print(f"Performance change: ${(total_pnl - older_pnl):.2f}")
            
        conn.close()
        
    except Exception as e:
        print(f"Error analyzing trades: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_trading_performance()