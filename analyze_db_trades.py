#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/maxime/BTC_BOT/BTC_BOT')

from utils.paper_trade_db import get_all_trades, get_open_positions, get_pnl_breakdown, get_trade_count, get_total_fees, get_rolling_stats
from datetime import datetime, timedelta

def analyze_trading_session():
    print('=== TRADING SESSION ANALYSIS ===')
    print()

    # Get all trades from last 24 hours
    trades = get_all_trades(limit=1000)
    print(f'Total trades found: {len(trades)}')

    # Filter trades from last 12-24 hours
    now = datetime.now()
    twelve_hours_ago = now - timedelta(hours=12)
    twenty_four_hours_ago = now - timedelta(hours=24)

    recent_trades = []
    for trade in trades:
        trade_time = trade[1]  # timestamp is at index 1
        if isinstance(trade_time, str):
            trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
        if trade_time >= twelve_hours_ago:
            recent_trades.append(trade)

    print(f'Trades in last 12 hours: {len(recent_trades)}')
    print()

    # Show trade details
    print('=== RECENT TRADES (Last 12 Hours) ===')
    total_pnl = 0
    total_fees = 0
    buy_count = 0
    sell_count = 0

    for trade in recent_trades[:20]:  # Show first 20
        trade_id, timestamp, symbol, side, price, quantity, pnl, fee = trade
        total_pnl += (pnl or 0)
        total_fees += (fee or 0)
        
        if side.upper() == 'BUY':
            buy_count += 1
        elif side.upper() == 'SELL':
            sell_count += 1
        
        print(f'{timestamp} | {side:4} {symbol:8} | Price: ${price:8.2f} | Qty: {quantity:8.6f} | PnL: ${pnl:6.2f} | Fee: ${fee:6.2f}')

    print()
    print('=== SUMMARY ===')
    print(f'Total P&L: ${total_pnl:.2f}')
    print(f'Total Fees: ${total_fees:.2f}')
    print(f'Net P&L: ${total_pnl - total_fees:.2f}')
    print(f'Buy trades: {buy_count}')
    print(f'Sell trades: {sell_count}')
    print()

    # Get open positions
    positions = get_open_positions()
    print(f'=== OPEN POSITIONS ({len(positions)}) ===')
    for pos in positions:
        pos_id, symbol, side, entry_price, quantity, leverage, entry_time = pos
        print(f'ID: {pos_id} | {side:4} {symbol:8} | Entry: ${entry_price:8.2f} | Qty: {quantity:8.6f} | Leverage: {leverage}x | Time: {entry_time}')

    print()

    # PnL breakdown
    pnl_breakdown = get_pnl_breakdown()
    print('=== PNL BREAKDOWN BY SYMBOL ===')
    for symbol, data in pnl_breakdown.items():
        print(f'{symbol:8} | Total PnL: ${data["total_pnl"]:8.2f} | Trades: {data["trade_count"]:3}')

    print()

    # Rolling stats
    rolling = get_rolling_stats(24)
    print('=== ROLLING 24H STATS ===')
    print(f'Rolling PnL: ${rolling["rolling_pnl"]:.2f}')
    print(f'Rolling Fees: ${rolling["rolling_fees"]:.2f}')
    print(f'Net Rolling: ${rolling["rolling_pnl"] - rolling["rolling_fees"]:.2f}')

    # Additional analysis
    print()
    print('=== TRADING PATTERNS ANALYSIS ===')
    
    # Analyze trade frequency
    if recent_trades:
        trade_times = [trade[1] for trade in recent_trades]
        earliest = min(trade_times)
        latest = max(trade_times)
        duration = latest - earliest
        if duration.total_seconds() > 0:
            trade_frequency = len(recent_trades) / (duration.total_seconds() / 3600)  # trades per hour
            print(f'Trade frequency: {trade_frequency:.2f} trades/hour')
        
        # Analyze trade sizes
        trade_sizes = [abs(trade[5] * trade[4]) for trade in recent_trades if trade[5] and trade[4]]  # quantity * price
        if trade_sizes:
            avg_trade_size = sum(trade_sizes) / len(trade_sizes)
            print(f'Average trade size: ${avg_trade_size:.2f}')
            print(f'Largest trade: ${max(trade_sizes):.2f}')
            print(f'Smallest trade: ${min(trade_sizes):.2f}')
        
        # Win/Loss ratio
        winning_trades = sum(1 for trade in recent_trades if (trade[6] or 0) > 0)
        losing_trades = sum(1 for trade in recent_trades if (trade[6] or 0) < 0)
        neutral_trades = len(recent_trades) - winning_trades - losing_trades
        
        print(f'Winning trades: {winning_trades}')
        print(f'Losing trades: {losing_trades}') 
        print(f'Neutral trades: {neutral_trades}')
        
        if winning_trades + losing_trades > 0:
            win_rate = winning_trades / (winning_trades + losing_trades) * 100
            print(f'Win rate: {win_rate:.1f}%')

if __name__ == "__main__":
    analyze_trading_session()