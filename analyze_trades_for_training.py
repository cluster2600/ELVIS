#!/usr/bin/env python3
"""
Analyze trades from database to prepare for model training
"""

import sys
sys.path.append('/Users/maxime/BTC_BOT/BTC_BOT')

from utils.paper_trade_db import get_all_trades
from datetime import datetime, timedelta
import pandas as pd

def analyze_trade_data():
    """Analyze trade data structure and outcomes"""
    print("🔍 TRADING HISTORY ANALYSIS FOR MODEL TRAINING")
    print("=" * 55)
    
    # Get all trades
    all_trades = get_all_trades(limit=1000)
    print(f"Total trades in database: {len(all_trades)}")
    
    # Separate by type
    buy_trades = []
    sell_trades = []
    test_trades = []
    
    for trade in all_trades:
        if len(trade) >= 8:
            if trade[3] == 'TEST':
                test_trades.append(trade)
            elif trade[3] == 'BUY':
                buy_trades.append(trade)
            elif trade[3] == 'SELL':
                sell_trades.append(trade)
    
    print(f"BUY trades: {len(buy_trades)}")
    print(f"SELL trades: {len(sell_trades)}")
    print(f"TEST trades: {len(test_trades)}")
    
    # Analyze time range for real trades
    real_trades = buy_trades + sell_trades
    if real_trades:
        timestamps = [trade[1] for trade in real_trades]
        earliest = min(timestamps)
        latest = max(timestamps)
        duration = latest - earliest
        
        print(f"\nTrading period: {earliest} to {latest}")
        print(f"Duration: {duration}")
    
    # Analyze trade outcomes for learning
    print("\n📈 TRADE OUTCOME ANALYSIS:")
    
    profitable_sells = []
    losing_sells = []
    breakeven_sells = []
    
    for trade in sell_trades:
        pnl = float(trade[6])
        fee = float(trade[7])
        net_pnl = pnl - fee
        
        if net_pnl > 1.0:  # Significant profit after fees
            profitable_sells.append({
                'timestamp': trade[1],
                'symbol': trade[2],
                'price': float(trade[4]),
                'quantity': float(trade[5]),
                'gross_pnl': pnl,
                'fee': fee,
                'net_pnl': net_pnl
            })
        elif net_pnl < -1.0:  # Significant loss after fees
            losing_sells.append({
                'timestamp': trade[1],
                'symbol': trade[2],
                'price': float(trade[4]),
                'quantity': float(trade[5]),
                'gross_pnl': pnl,
                'fee': fee,
                'net_pnl': net_pnl
            })
        else:
            breakeven_sells.append(trade)
    
    print(f"Profitable trades (net >$1): {len(profitable_sells)}")
    print(f"Losing trades (net <-$1): {len(losing_sells)}")
    print(f"Breakeven trades: {len(breakeven_sells)}")
    
    if profitable_sells:
        print(f"\nTop profitable trade:")
        best = max(profitable_sells, key=lambda x: x['net_pnl'])
        print(f"  Time: {best['timestamp']}")
        print(f"  Price: ${best['price']:.2f}")
        print(f"  Net P&L: ${best['net_pnl']:.2f}")
    
    if losing_sells:
        print(f"\nWorst losing trade:")
        worst = min(losing_sells, key=lambda x: x['net_pnl'])
        print(f"  Time: {worst['timestamp']}")
        print(f"  Price: ${worst['price']:.2f}")
        print(f"  Net P&L: ${worst['net_pnl']:.2f}")
    
    # Check for paired trades (BUY -> SELL)
    print(f"\n🔄 TRADE PAIRING ANALYSIS:")
    
    # Group trades by approximate time windows
    paired_trades = []
    
    # Sort trades by timestamp
    buy_trades_sorted = sorted(buy_trades, key=lambda x: x[1])
    sell_trades_sorted = sorted(sell_trades, key=lambda x: x[1])
    
    print(f"Analyzing {len(buy_trades_sorted)} BUY and {len(sell_trades_sorted)} SELL trades for pairs...")
    
    # Simple pairing: match BUY trades with subsequent SELL trades
    for buy_trade in buy_trades_sorted[-50:]:  # Recent 50 BUY trades
        buy_time = buy_trade[1]
        buy_price = float(buy_trade[4])
        buy_qty = float(buy_trade[5])
        
        # Find corresponding SELL trade within reasonable time window
        for sell_trade in sell_trades_sorted:
            sell_time = sell_trade[1]
            if sell_time > buy_time and (sell_time - buy_time).total_seconds() < 3600:  # Within 1 hour
                sell_price = float(sell_trade[4])
                sell_qty = float(sell_trade[5])
                sell_pnl = float(sell_trade[6])
                
                # Check if quantities are similar (allowing for partial fills)
                if abs(buy_qty - sell_qty) < buy_qty * 0.1:  # Within 10%
                    paired_trades.append({
                        'buy_time': buy_time,
                        'sell_time': sell_time,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'quantity': buy_qty,
                        'duration_minutes': (sell_time - buy_time).total_seconds() / 60,
                        'gross_pnl': sell_pnl,
                        'price_change': sell_price - buy_price,
                        'price_change_pct': (sell_price - buy_price) / buy_price * 100
                    })
                    break
    
    print(f"Found {len(paired_trades)} paired BUY->SELL trades")
    
    if paired_trades:
        # Analyze successful vs unsuccessful trades
        successful = [t for t in paired_trades if t['gross_pnl'] > 0]
        unsuccessful = [t for t in paired_trades if t['gross_pnl'] < 0]
        
        print(f"Successful pairs: {len(successful)}")
        print(f"Unsuccessful pairs: {len(unsuccessful)}")
        
        if successful:
            avg_success_duration = sum(t['duration_minutes'] for t in successful) / len(successful)
            avg_success_pct = sum(t['price_change_pct'] for t in successful) / len(successful)
            print(f"Avg successful trade duration: {avg_success_duration:.1f} minutes")
            print(f"Avg successful price change: {avg_success_pct:.2f}%")
        
        if unsuccessful:
            avg_fail_duration = sum(t['duration_minutes'] for t in unsuccessful) / len(unsuccessful)
            avg_fail_pct = sum(t['price_change_pct'] for t in unsuccessful) / len(unsuccessful)
            print(f"Avg unsuccessful trade duration: {avg_fail_duration:.1f} minutes")
            print(f"Avg unsuccessful price change: {avg_fail_pct:.2f}%")
    
    return {
        'total_trades': len(all_trades),
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades),
        'profitable_sells': profitable_sells,
        'losing_sells': losing_sells,
        'paired_trades': paired_trades
    }

if __name__ == "__main__":
    results = analyze_trade_data()
    
    print(f"\n🎯 TRAINING DATA POTENTIAL:")
    print(f"Total trade records: {results['total_trades']}")
    print(f"Profitable outcomes: {len(results['profitable_sells'])}")
    print(f"Loss outcomes: {len(results['losing_sells'])}")
    print(f"Paired trades: {len(results['paired_trades'])}")
    
    # Assess if we have enough data for training
    total_outcomes = len(results['profitable_sells']) + len(results['losing_sells'])
    if total_outcomes > 50:
        print(f"\n✅ Sufficient data for model training ({total_outcomes} labeled outcomes)")
    elif total_outcomes > 20:
        print(f"\n⚠️ Limited data for training ({total_outcomes} labeled outcomes)")
    else:
        print(f"\n❌ Insufficient data for training ({total_outcomes} labeled outcomes)")
        print("Consider running the bot longer to collect more trade data")