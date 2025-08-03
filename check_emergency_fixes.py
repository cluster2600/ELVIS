#!/usr/bin/env python3
"""
Emergency Fix Analysis Script
Check if the recent fixes are working to prevent high-frequency micro-trading.
"""

import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.paper_trade_db import get_conn

def analyze_recent_trades(hours=2):
    """Analyze trades from the last N hours to assess emergency fixes."""
    
    print(f"🔍 ANALYZING TRADES FROM LAST {hours} HOURS")
    print("=" * 60)
    
    conn = get_conn()
    if conn is None:
        print("❌ ERROR: Cannot connect to database")
        return
    
    try:
        c = conn.cursor()
        
        # Get trades from the last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        c.execute("""
            SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
            FROM trades
            WHERE timestamp > %s AND side != 'TEST'
            ORDER BY timestamp DESC
        """, (cutoff_time,))
        
        trades = c.fetchall()
        
        if not trades:
            print("❌ NO TRADES FOUND in the last {} hours".format(hours))
            print("This could mean:")
            print("  - Bot is not running")
            print("  - Fixes are working TOO well (no valid signals)")
            print("  - Database connectivity issues")
            return
        
        print(f"📊 Found {len(trades)} trades in the last {hours} hours")
        print()
        
        # 1. Trading Frequency Analysis
        print("🕐 TRADING FREQUENCY ANALYSIS")
        print("-" * 40)
        
        # Group trades by minute to check frequency
        trades_by_minute = defaultdict(list)
        for trade in trades:
            timestamp = trade[1]
            minute_key = timestamp.strftime("%Y-%m-%d %H:%M")
            trades_by_minute[minute_key].append(trade)
        
        # Calculate trades per minute
        trades_per_minute = [len(trades_list) for trades_list in trades_by_minute.values()]
        
        if trades_per_minute:
            max_trades_per_minute = max(trades_per_minute)
            avg_trades_per_minute = sum(trades_per_minute) / len(trades_per_minute)
            
            print(f"   Max trades per minute: {max_trades_per_minute}")
            print(f"   Average trades per minute: {avg_trades_per_minute:.2f}")
            
            if max_trades_per_minute > 10:
                print("   ⚠️  WARNING: High frequency trading detected!")
            elif max_trades_per_minute > 5:
                print("   ⚠️  CAUTION: Moderate frequency trading")
            else:
                print("   ✅ Trading frequency looks reasonable")
        
        # 2. Profit Level Analysis
        print("\n💰 PROFIT LEVEL ANALYSIS")
        print("-" * 40)
        
        profitable_trades = [trade for trade in trades if trade[6] > 0]  # pnl > 0
        losing_trades = [trade for trade in trades if trade[6] < 0]
        
        if profitable_trades:
            profit_amounts = [trade[6] for trade in profitable_trades]
            avg_profit = statistics.mean(profit_amounts)
            max_profit = max(profit_amounts)
            min_profit = min(profit_amounts)
            
            print(f"   Profitable trades: {len(profitable_trades)}")
            print(f"   Average profit: ${avg_profit:.4f}")
            print(f"   Max profit: ${max_profit:.4f}")
            print(f"   Min profit: ${min_profit:.4f}")
            
            # Check if profits are targeting $1.00 instead of cents
            micro_profits = [p for p in profit_amounts if 0 < p < 0.05]  # Less than 5 cents
            
            if len(micro_profits) > len(profitable_trades) * 0.5:
                print("   ⚠️  WARNING: Many trades still targeting micro-profits!")
            elif avg_profit > 0.5:
                print("   ✅ Profits targeting reasonable levels")
            else:
                print("   ⚠️  CAUTION: Average profit still quite low")
        
        # 3. Cooldown Period Analysis
        print("\n❄️  COOLDOWN PERIOD ANALYSIS")
        print("-" * 40)
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x[1])
        
        if len(sorted_trades) > 1:
            time_gaps = []
            for i in range(1, len(sorted_trades)):
                prev_time = sorted_trades[i-1][1]
                curr_time = sorted_trades[i][1]
                gap_seconds = (curr_time - prev_time).total_seconds()
                time_gaps.append(gap_seconds)
            
            if time_gaps:
                avg_gap = statistics.mean(time_gaps)
                min_gap = min(time_gaps)
                max_gap = max(time_gaps)
                
                print(f"   Average gap between trades: {avg_gap:.1f} seconds")
                print(f"   Minimum gap: {min_gap:.1f} seconds")
                print(f"   Maximum gap: {max_gap:.1f} seconds")
                
                rapid_trades = [gap for gap in time_gaps if gap < 60]  # Less than 1 minute
                
                if len(rapid_trades) > len(time_gaps) * 0.3:
                    print("   ⚠️  WARNING: Many trades occurring rapidly!")
                elif avg_gap > 300:  # 5 minutes
                    print("   ✅ Good cooldown periods between trades")
                else:
                    print("   ⚠️  CAUTION: Some trades occurring quickly")
        
        # 4. Win Rate and Fee Impact
        print("\n📈 WIN RATE AND FEE ANALYSIS")
        print("-" * 40)
        
        total_trades = len(trades)
        winning_trades = len(profitable_trades)
        losing_trades_count = len(losing_trades)
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(trade[6] for trade in trades)
        total_fees = sum(trade[7] for trade in trades)
        net_result = total_pnl - total_fees
        
        print(f"   Total trades: {total_trades}")
        print(f"   Winning trades: {winning_trades}")
        print(f"   Losing trades: {losing_trades_count}")
        print(f"   Win rate: {win_rate:.1f}%")
        print(f"   Total P&L: ${total_pnl:.4f}")
        print(f"   Total fees: ${total_fees:.4f}")
        print(f"   Net result: ${net_result:.4f}")
        
        fee_ratio = (total_fees / abs(total_pnl)) if total_pnl != 0 else float('inf')
        
        if fee_ratio > 0.5:
            print("   ⚠️  WARNING: Fees are eating into profits significantly!")
        elif fee_ratio > 0.2:
            print("   ⚠️  CAUTION: Fees are having moderate impact")
        else:
            print("   ✅ Fee impact is reasonable")
        
        # 5. Recent Activity Check
        print("\n🔄 RECENT ACTIVITY CHECK")
        print("-" * 40)
        
        # Check if bot is still active (trades in last 30 minutes)
        recent_cutoff = datetime.now() - timedelta(minutes=30)
        c.execute("""
            SELECT COUNT(*) FROM trades
            WHERE timestamp > %s AND side != 'TEST'
        """, (recent_cutoff,))
        
        recent_count = c.fetchone()[0]
        
        if recent_count > 0:
            print(f"   ✅ Bot is active: {recent_count} trades in last 30 minutes")
        else:
            print("   ⚠️  Bot may be stopped: No trades in last 30 minutes")
        
        # Overall Assessment
        print("\n🎯 OVERALL ASSESSMENT")
        print("=" * 60)
        
        issues = []
        
        if max_trades_per_minute > 10:
            issues.append("High frequency trading still occurring")
        
        if profitable_trades and statistics.mean([trade[6] for trade in profitable_trades]) < 0.1:
            issues.append("Still targeting micro-profits")
        
        if time_gaps and statistics.mean(time_gaps) < 120:
            issues.append("Insufficient cooldown periods")
        
        if fee_ratio > 0.3:
            issues.append("Fees eating into profits")
        
        if recent_count == 0:
            issues.append("Bot appears to be stopped")
        
        if not issues:
            print("✅ EMERGENCY FIXES APPEAR TO BE WORKING!")
            print("   - Trading frequency is controlled")
            print("   - Profit targets are reasonable")
            print("   - Cooldown periods are effective")
            print("   - Fee impact is manageable")
        else:
            print("⚠️  ISSUES STILL DETECTED:")
            for issue in issues:
                print(f"   - {issue}")
        
        # Recent trades sample
        print("\n📋 RECENT TRADES SAMPLE (Last 10)")
        print("-" * 60)
        
        for i, trade in enumerate(trades[:10]):
            timestamp = trade[1].strftime("%Y-%m-%d %H:%M:%S")
            symbol = trade[2]
            side = trade[3]
            price = trade[4]
            quantity = trade[5]
            pnl = trade[6]
            fee = trade[7]
            
            print(f"{i+1:2d}. {timestamp} | {symbol} {side} | ${price:.2f} | Q:{quantity:.4f} | P&L:${pnl:.4f} | Fee:${fee:.4f}")
        
    except Exception as e:
        print(f"❌ ERROR analyzing trades: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_recent_trades(hours=2)