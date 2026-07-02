#!/usr/bin/env python3
"""
Analyze recent trades to check if the trading performance fixes are working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.paper_trade_db import get_conn, get_all_trades, get_trade_count, get_total_fees, get_pnl_breakdown
from datetime import datetime, timedelta
import statistics

def analyze_trading_frequency():
    """Analyze trading frequency to check if it's been reduced from 77 trades/minute."""
    conn = get_conn()
    if conn is None:
        print("[ERROR] Cannot connect to database")
        return
    
    try:
        with conn.cursor() as c:
            # Get trades from the last hour
            c.execute("""
                SELECT timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC
            """)
            recent_trades = c.fetchall()
            
            if not recent_trades:
                print("[INFO] No trades found in the last hour")
                return
            
            print(f"[INFO] Found {len(recent_trades)} trades in the last hour")
            
            # Calculate trades per minute for different time windows
            time_windows = [5, 10, 15, 30, 60]  # minutes
            
            for window in time_windows:
                cutoff_time = datetime.now() - timedelta(minutes=window)
                trades_in_window = [t for t in recent_trades if t[0] > cutoff_time]
                trades_per_minute = len(trades_in_window) / window if window > 0 else 0
                
                print(f"[ANALYSIS] Last {window} minutes: {len(trades_in_window)} trades = {trades_per_minute:.2f} trades/minute")
            
            # Show most recent trades
            print("\n[RECENT TRADES] Last 10 trades:")
            for i, trade in enumerate(recent_trades[:10]):
                timestamp, symbol, side, price, quantity, pnl, fee = trade
                print(f"  {i+1}. {timestamp} | {symbol} {side} | Price: ${price:.2f} | Qty: {quantity:.4f} | P&L: ${pnl:.2f} | Fee: ${fee:.2f}")
            
    except Exception as e:
        print(f"[ERROR] Failed to analyze trading frequency: {e}")
    finally:
        conn.close()

def analyze_profit_levels():
    """Check if profits are now 10-cent levels instead of 1-cent levels."""
    conn = get_conn()
    if conn is None:
        print("[ERROR] Cannot connect to database")
        return
    
    try:
        with conn.cursor() as c:
            # Get trades from the last 24 hours
            c.execute("""
                SELECT timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                AND pnl != 0
                ORDER BY timestamp DESC
            """)
            profitable_trades = c.fetchall()
            
            if not profitable_trades:
                print("[INFO] No profitable trades found in the last 24 hours")
                return
            
            print(f"[INFO] Found {len(profitable_trades)} trades with P&L in the last 24 hours")
            
            # Analyze profit distribution
            pnl_values = [abs(float(t[5])) for t in profitable_trades]  # P&L column
            
            if pnl_values:
                avg_pnl = statistics.mean(pnl_values)
                median_pnl = statistics.median(pnl_values)
                min_pnl = min(pnl_values)
                max_pnl = max(pnl_values)
                
                print(f"[ANALYSIS] P&L Statistics:")
                print(f"  Average P&L: ${avg_pnl:.4f}")
                print(f"  Median P&L: ${median_pnl:.4f}")
                print(f"  Min P&L: ${min_pnl:.4f}")
                print(f"  Max P&L: ${max_pnl:.4f}")
                
                # Check if most trades are now in 10-cent range vs 1-cent range
                cent_trades = sum(1 for p in pnl_values if p < 0.05)  # < 5 cents
                ten_cent_trades = sum(1 for p in pnl_values if 0.05 <= p < 0.15)  # 5-15 cents
                high_profit_trades = sum(1 for p in pnl_values if p >= 0.15)  # > 15 cents
                
                print(f"  Small profits (< 5 cents): {cent_trades} trades ({cent_trades/len(pnl_values)*100:.1f}%)")
                print(f"  Medium profits (5-15 cents): {ten_cent_trades} trades ({ten_cent_trades/len(pnl_values)*100:.1f}%)")
                print(f"  High profits (> 15 cents): {high_profit_trades} trades ({high_profit_trades/len(pnl_values)*100:.1f}%)")
                
    except Exception as e:
        print(f"[ERROR] Failed to analyze profit levels: {e}")
    finally:
        conn.close()

def calculate_win_rate_and_fees():
    """Calculate current win rate and fee impact."""
    conn = get_conn()
    if conn is None:
        print("[ERROR] Cannot connect to database")
        return
    
    try:
        with conn.cursor() as c:
            # Get all trades from the last 24 hours
            c.execute("""
                SELECT timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                ORDER BY timestamp DESC
            """)
            trades = c.fetchall()
            
            if not trades:
                print("[INFO] No trades found in the last 24 hours")
                return
            
            print(f"[INFO] Analyzing {len(trades)} trades from the last 24 hours")
            
            # Calculate win rate
            profitable_trades = [t for t in trades if float(t[5]) > 0]  # P&L > 0
            losing_trades = [t for t in trades if float(t[5]) < 0]  # P&L < 0
            break_even_trades = [t for t in trades if float(t[5]) == 0]  # P&L = 0
            
            total_trades = len(trades)
            win_rate = (len(profitable_trades) / total_trades * 100) if total_trades > 0 else 0
            
            print(f"[ANALYSIS] Win Rate Analysis:")
            print(f"  Total trades: {total_trades}")
            print(f"  Profitable trades: {len(profitable_trades)} ({len(profitable_trades)/total_trades*100:.1f}%)")
            print(f"  Losing trades: {len(losing_trades)} ({len(losing_trades)/total_trades*100:.1f}%)")
            print(f"  Break-even trades: {len(break_even_trades)} ({len(break_even_trades)/total_trades*100:.1f}%)")
            print(f"  Win rate: {win_rate:.2f}%")
            
            # Calculate fee impact
            total_fees = sum(float(t[6]) for t in trades)  # Fee column
            total_pnl = sum(float(t[5]) for t in trades)  # P&L column
            
            print(f"[ANALYSIS] Fee Impact Analysis:")
            print(f"  Total fees paid: ${total_fees:.2f}")
            print(f"  Total P&L: ${total_pnl:.2f}")
            print(f"  Net P&L after fees: ${total_pnl - total_fees:.2f}")
            
            if total_pnl != 0:
                fee_percentage = (total_fees / abs(total_pnl)) * 100
                print(f"  Fees as % of gross P&L: {fee_percentage:.2f}%")
            
    except Exception as e:
        print(f"[ERROR] Failed to calculate win rate and fees: {e}")
    finally:
        conn.close()

def check_cooldown_and_confidence():
    """Check if cooldown periods and higher confidence thresholds are working."""
    conn = get_conn()
    if conn is None:
        print("[ERROR] Cannot connect to database")
        return
    
    try:
        with conn.cursor() as c:
            # Get recent trades to check time gaps
            c.execute("""
                SELECT timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                WHERE timestamp > NOW() - INTERVAL '2 hours'
                ORDER BY timestamp DESC
            """)
            trades = c.fetchall()
            
            if len(trades) < 2:
                print("[INFO] Not enough trades to analyze cooldown periods")
                return
            
            print(f"[INFO] Analyzing {len(trades)} trades for cooldown periods")
            
            # Calculate time gaps between trades
            time_gaps = []
            for i in range(len(trades) - 1):
                current_time = trades[i][0]
                next_time = trades[i + 1][0]
                gap = (current_time - next_time).total_seconds()
                time_gaps.append(gap)
            
            if time_gaps:
                avg_gap = statistics.mean(time_gaps)
                median_gap = statistics.median(time_gaps)
                min_gap = min(time_gaps)
                max_gap = max(time_gaps)
                
                print(f"[ANALYSIS] Time Gap Analysis:")
                print(f"  Average gap between trades: {avg_gap:.1f} seconds")
                print(f"  Median gap between trades: {median_gap:.1f} seconds")
                print(f"  Min gap between trades: {min_gap:.1f} seconds")
                print(f"  Max gap between trades: {max_gap:.1f} seconds")
                
                # Check if gaps are reasonable (cooldown working)
                short_gaps = sum(1 for g in time_gaps if g < 30)  # < 30 seconds
                medium_gaps = sum(1 for g in time_gaps if 30 <= g < 120)  # 30-120 seconds
                long_gaps = sum(1 for g in time_gaps if g >= 120)  # > 2 minutes
                
                print(f"  Short gaps (< 30s): {short_gaps} ({short_gaps/len(time_gaps)*100:.1f}%)")
                print(f"  Medium gaps (30-120s): {medium_gaps} ({medium_gaps/len(time_gaps)*100:.1f}%)")
                print(f"  Long gaps (> 2min): {long_gaps} ({long_gaps/len(time_gaps)*100:.1f}%)")
                
    except Exception as e:
        print(f"[ERROR] Failed to check cooldown and confidence: {e}")
    finally:
        conn.close()

def calculate_net_pnl():
    """Calculate net P&L from recent trades."""
    conn = get_conn()
    if conn is None:
        print("[ERROR] Cannot connect to database")
        return
    
    try:
        with conn.cursor() as c:
            # Get net P&L for different time periods
            time_periods = [
                ('1 hour', '1 hour'),
                ('6 hours', '6 hours'),
                ('24 hours', '1 day'),
                ('7 days', '1 week')
            ]
            
            print("[ANALYSIS] Net P&L Analysis:")
            
            for period_name, sql_interval in time_periods:
                c.execute(f"""
                    SELECT SUM(pnl) as total_pnl, SUM(fee) as total_fees, COUNT(*) as trade_count
                    FROM trades
                    WHERE timestamp > NOW() - INTERVAL '{sql_interval}'
                """)
                result = c.fetchone()
                
                if result and result[0] is not None:
                    total_pnl, total_fees, trade_count = result
                    net_pnl = total_pnl - total_fees
                    
                    print(f"  Last {period_name}:")
                    print(f"    Trades: {trade_count}")
                    print(f"    Gross P&L: ${total_pnl:.2f}")
                    print(f"    Total fees: ${total_fees:.2f}")
                    print(f"    Net P&L: ${net_pnl:.2f}")
                    print(f"    Per trade: ${net_pnl/trade_count:.4f}" if trade_count > 0 else "    Per trade: N/A")
                else:
                    print(f"  Last {period_name}: No trades")
                
    except Exception as e:
        print(f"[ERROR] Failed to calculate net P&L: {e}")
    finally:
        conn.close()

def main():
    print("=" * 80)
    print("TRADING PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("1. TRADING FREQUENCY ANALYSIS")
    print("-" * 50)
    analyze_trading_frequency()
    print()
    
    print("2. PROFIT LEVELS ANALYSIS")
    print("-" * 50)
    analyze_profit_levels()
    print()
    
    print("3. WIN RATE AND FEE IMPACT")
    print("-" * 50)
    calculate_win_rate_and_fees()
    print()
    
    print("4. COOLDOWN AND CONFIDENCE CHECK")
    print("-" * 50)
    check_cooldown_and_confidence()
    print()
    
    print("5. NET P&L CALCULATION")
    print("-" * 50)
    calculate_net_pnl()
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()