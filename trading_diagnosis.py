#!/usr/bin/env python3
"""
Comprehensive trading diagnosis script to identify all issues causing losses
"""

import sys
import os
from datetime import datetime, timedelta
import statistics
from collections import defaultdict

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.paper_trade_db import (
    get_all_trades, get_open_positions, get_trade_count, 
    get_total_fees, get_pnl_breakdown, get_rolling_stats
)

def diagnose_all_issues():
    """Comprehensive diagnosis of all trading issues"""
    print("=" * 100)
    print("🔍 COMPREHENSIVE TRADING DIAGNOSIS")
    print("=" * 100)
    
    # Get all trades
    trades = get_all_trades(limit=500, exclude_test=True)
    
    if not trades:
        print("❌ NO TRADES FOUND - Bot may not be trading at all!")
        return
    
    print(f"📊 Analyzing {len(trades)} trades from the database")
    print(f"🕐 Date range: {trades[-1][1]} to {trades[0][1]}")
    
    # 1. CORE PROBLEM ANALYSIS
    print("\n🚨 CORE PROBLEM ANALYSIS:")
    print("=" * 50)
    
    # Check for micro-profits
    winning_trades = [t for t in trades if t[6] > 0]
    losing_trades = [t for t in trades if t[6] < 0]
    
    if winning_trades:
        avg_win = statistics.mean(t[6] for t in winning_trades)
        max_win = max(t[6] for t in winning_trades)
        min_win = min(t[6] for t in winning_trades)
        print(f"💰 WIN ANALYSIS:")
        print(f"   Average win: ${avg_win:.4f}")
        print(f"   Maximum win: ${max_win:.4f}")
        print(f"   Minimum win: ${min_win:.4f}")
        
        # Check for micro-profits
        micro_profits = [t for t in winning_trades if t[6] < 0.10]
        print(f"   Micro-profits (<$0.10): {len(micro_profits)}/{len(winning_trades)} ({len(micro_profits)/len(winning_trades)*100:.1f}%)")
        
        if len(micro_profits) > len(winning_trades) * 0.8:
            print("   ⚠️  CRITICAL: 80%+ of wins are micro-profits!")
    
    # 2. FEE ANALYSIS
    print(f"\n💸 FEE ANALYSIS:")
    print("=" * 50)
    
    total_fees = sum(t[7] for t in trades)
    total_pnl = sum(t[6] for t in trades)
    
    print(f"   Total fees paid: ${total_fees:.2f}")
    print(f"   Total P&L: ${total_pnl:.2f}")
    print(f"   Net P&L: ${total_pnl - total_fees:.2f}")
    
    if total_pnl != 0:
        fee_ratio = total_fees / abs(total_pnl)
        print(f"   Fee to P&L ratio: {fee_ratio:.2f}x")
        
        if fee_ratio > 1.0:
            print("   🚨 CRITICAL: Fees exceed P&L!")
    
    # Check trades where fees > profits
    profitable_trades = [t for t in trades if t[6] > 0]
    fee_exceeds_profit = [t for t in profitable_trades if t[7] > t[6]]
    
    print(f"   Trades where fees > profits: {len(fee_exceeds_profit)}/{len(profitable_trades)} ({len(fee_exceeds_profit)/len(profitable_trades)*100:.1f}%)")
    
    # 3. TRADING FREQUENCY ANALYSIS
    print(f"\n⏱️ TRADING FREQUENCY ANALYSIS:")
    print("=" * 50)
    
    # Time between trades
    if len(trades) > 1:
        time_diffs = []
        for i in range(len(trades) - 1):
            diff = (trades[i][1] - trades[i+1][1]).total_seconds()
            time_diffs.append(diff)
        
        avg_time_between = statistics.mean(time_diffs)
        print(f"   Average time between trades: {avg_time_between:.1f} seconds")
        
        # Check for over-trading
        trades_per_minute = len(trades) / ((trades[0][1] - trades[-1][1]).total_seconds() / 60)
        print(f"   Trading frequency: {trades_per_minute:.2f} trades/minute")
        
        if trades_per_minute > 1.0:
            print("   ⚠️  WARNING: Over-trading detected (>1 trade/minute)")
    
    # 4. POSITION ANALYSIS  
    print(f"\n📈 POSITION ANALYSIS:")
    print("=" * 50)
    
    buy_trades = [t for t in trades if t[3] == 'BUY']
    sell_trades = [t for t in trades if t[3] == 'SELL']
    
    print(f"   BUY trades: {len(buy_trades)}")
    print(f"   SELL trades: {len(sell_trades)}")
    print(f"   BUY/SELL ratio: {len(buy_trades)/len(sell_trades):.2f}" if sell_trades else "∞ (no sells)")
    
    # Check for position imbalance
    if len(buy_trades) > len(sell_trades) * 2:
        print("   ⚠️  WARNING: Significant position imbalance - too many BUYs")
    
    # 5. OPEN POSITIONS ANALYSIS
    print(f"\n🔄 OPEN POSITIONS ANALYSIS:")
    print("=" * 50)
    
    open_positions = get_open_positions()
    print(f"   Current open positions: {len(open_positions)}")
    
    if open_positions:
        for pos in open_positions:
            pos_id, symbol, side, entry_price, quantity, leverage, entry_time = pos
            print(f"   Position {pos_id}: {symbol} {side} @ ${entry_price:.2f} | Size: {quantity} | Leverage: {leverage}x")
    
    # 6. PROFIT TARGET ANALYSIS
    print(f"\n🎯 PROFIT TARGET ANALYSIS:")
    print("=" * 50)
    
    # Analyze profit distribution
    profit_ranges = {
        'micro ($0.00-$0.10)': [t for t in winning_trades if 0 <= t[6] <= 0.10],
        'small ($0.10-$1.00)': [t for t in winning_trades if 0.10 < t[6] <= 1.00],
        'medium ($1.00-$5.00)': [t for t in winning_trades if 1.00 < t[6] <= 5.00],
        'large (>$5.00)': [t for t in winning_trades if t[6] > 5.00]
    }
    
    for range_name, trades_in_range in profit_ranges.items():
        count = len(trades_in_range)
        pct = (count / len(winning_trades) * 100) if winning_trades else 0
        print(f"   {range_name}: {count} trades ({pct:.1f}%)")
    
    # 7. LOSS ANALYSIS
    print(f"\n📉 LOSS ANALYSIS:")
    print("=" * 50)
    
    if losing_trades:
        avg_loss = statistics.mean(t[6] for t in losing_trades)
        max_loss = min(t[6] for t in losing_trades)  # min because losses are negative
        print(f"   Average loss: ${avg_loss:.4f}")
        print(f"   Maximum loss: ${max_loss:.4f}")
        
        # Check if stop losses are working
        large_losses = [t for t in losing_trades if t[6] < -1.0]
        print(f"   Large losses (>$1): {len(large_losses)}")
        
        if large_losses:
            print("   ⚠️  WARNING: Stop losses may not be working properly")
    
    # 8. RECOMMENDATIONS
    print(f"\n💡 CRITICAL RECOMMENDATIONS:")
    print("=" * 50)
    
    recommendations = []
    
    # Check for micro-trading
    if winning_trades:
        micro_profit_ratio = len([t for t in winning_trades if t[6] < 0.10]) / len(winning_trades)
        if micro_profit_ratio > 0.5:
            recommendations.append("🔴 CRITICAL: Increase minimum profit target from $0.10 to $1.00+")
    
    # Check fee impact
    if total_pnl != 0 and total_fees / abs(total_pnl) > 0.5:
        recommendations.append("🔴 CRITICAL: Reduce trading frequency - fees are eating profits")
    
    # Check position imbalance
    if len(buy_trades) > len(sell_trades) * 1.5:
        recommendations.append("🔴 CRITICAL: Fix position closing logic - too many unclosed positions")
    
    # Check win rate
    win_rate = len(winning_trades) / len(trades) * 100
    if win_rate < 20:
        recommendations.append("🔴 CRITICAL: Strategy is fundamentally flawed - win rate too low")
    
    # Check over-trading
    if len(trades) > 100:
        total_time = (trades[0][1] - trades[-1][1]).total_seconds() / 3600  # hours
        if total_time > 0 and len(trades) / total_time > 10:  # >10 trades/hour
            recommendations.append("🔴 CRITICAL: Over-trading detected - implement cooldown periods")
    
    if not recommendations:
        recommendations.append("✅ No critical issues detected in current sample")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    # 9. SPECIFIC FIXES NEEDED
    print(f"\n🔧 SPECIFIC FIXES NEEDED:")
    print("=" * 50)
    
    print("   1. INCREASE PROFIT TARGETS:")
    print("      - Current: $0.10 (too small)")
    print("      - Recommended: $1.00 minimum")
    print("      - Justification: Fees are $0.07-$0.40 per trade")
    
    print("   2. REDUCE TRADING FREQUENCY:")
    print("      - Current: High frequency micro-trading")
    print("      - Recommended: Quality over quantity")
    print("      - Implementation: Increase confidence threshold")
    
    print("   3. FIX POSITION MANAGEMENT:")
    print("      - Issue: Positions not closing properly")
    print("      - Fix: Check take profit/stop loss logic")
    print("      - Verify: SELL signal generation")
    
    print("   4. IMPLEMENT PROPER STOP LOSSES:")
    print("      - Current: Very small losses suggest no real stop loss")
    print("      - Recommended: 1-2% stop loss minimum")
    
    print("   5. BALANCE POSITION SIZING:")
    print("      - Issue: Position sizes too small for meaningful profits")
    print("      - Fix: Increase minimum position size")
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_fees': total_fees,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    try:
        diagnose_all_issues()
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()