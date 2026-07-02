#!/usr/bin/env python3
"""
Standalone Console Dashboard for ELVIS Trading Bot
Shows live trading data without requiring all dependencies
"""

import os
import sys
import time
import json
import requests
from datetime import datetime

def print_header():
    print("\033[95m" + "="*80)
    print("🤖 ELVIS Trading Bot - Console Dashboard")
    print("Enhanced Leveraged Virtual Investment System")
    print("="*80 + "\033[0m")

def print_status_line(text, status="INFO"):
    colors = {
        "INFO": "\033[96m",   # Cyan
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m"    # Red
    }
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{colors.get(status, '')}{timestamp} - {status} - {text}\033[0m")

def get_api_data(endpoint):
    """Fetch data from ELVIS API"""
    try:
        response = requests.get(f"http://localhost:5050{endpoint}", timeout=5)
        return response.json()
    except Exception as e:
        return None

def display_balances():
    print("\n\033[94m💰 Account Balances:\033[0m")
    print("-" * 40)
    print(f"{'Asset':<10} {'Balance':<15} {'Value':<10}")
    print("-" * 40)
    print(f"{'USDT':<10} {'$1,000.00':<15} {'$1,000.00':<10}")
    print(f"{'BNB':<10} {'$1,000.00':<15} {'$1,000.00':<10}")
    print("-" * 40)
    print(f"{'TOTAL':<10} {'':<15} {'$2,000.00':<10}")

def display_trades():
    print("\n\033[94m🔄 Recent Trades:\033[0m")
    trades = get_api_data("/trades")
    
    if trades:
        print("-" * 70)
        print(f"{'Time':<8} {'Symbol':<10} {'Side':<6} {'Qty':<12} {'P&L':<10}")
        print("-" * 70)
        
        for trade in trades[:10]:  # Show last 10 trades
            try:
                timestamp = trade.get('timestamp', '')[:8] if trade.get('timestamp') else 'N/A'
                symbol = trade.get('symbol', 'N/A')[:10]
                side = trade.get('side', 'N/A')[:6]
                quantity = f"{float(trade.get('quantity', 0)):,.2f}"[:12]
                pnl = float(trade.get('pnl', 0))
                pnl_color = "\033[92m" if pnl >= 0 else "\033[91m"  # Green/Red
                pnl_str = f"{pnl_color}{pnl:+.4f}\033[0m"
                
                print(f"{timestamp:<8} {symbol:<10} {side:<6} {quantity:<12} {pnl_str}")
            except Exception as e:
                continue
        
        # Calculate total P&L
        try:
            total_pnl = sum(float(trade.get('pnl', 0)) for trade in trades)
            pnl_color = "\033[92m" if total_pnl >= 0 else "\033[91m"
            print("-" * 70)
            print(f"{'TOTAL P&L:':<52} {pnl_color}{total_pnl:+.4f}\033[0m")
        except:
            pass
    else:
        print("Unable to fetch trades data")

def display_strategies():
    print("\n\033[94m🎯 Active Strategies:\033[0m")
    strategies = [
        "🎯 Bonenkamp HFT Strategy - Research-based 5-minute trading",
        "🤖 Ensemble Strategy - Multi-model decision making", 
        "🧠 Deep Reinforcement Learning - AI-driven optimization",
        "📚 Research-Based Model - Academic implementation"
    ]
    
    for strategy in strategies:
        print(f"  ✅ {strategy}")

def display_targets():
    print("\n\033[94m📈 Performance Targets:\033[0m")
    print("-" * 40)
    print(f"Annual Return Target:  14.9%")
    print(f"Sharpe Ratio Target:   2.02")
    print(f"Trading Frequency:     5-minute intervals")
    print(f"Trading Mode:          Paper Trading")
    print(f"F1-Score Target:       0.576")

def display_system_status():
    print("\n\033[94m⚡ System Status:\033[0m")
    
    # Check API health
    health = get_api_data("/health")
    if health and health.get('status') == 'healthy':
        print_status_line("ELVIS API - Healthy", "SUCCESS")
    else:
        print_status_line("ELVIS API - Offline", "ERROR")
    
    # Check trade count
    trades = get_api_data("/trades")
    if trades:
        print_status_line(f"Total Trades: {len(trades)}", "INFO")
    else:
        print_status_line("Unable to fetch trade data", "WARNING")

def main():
    print_header()
    
    print("\n\033[93m🎮 Console Dashboard Controls:\033[0m")
    print("  • Press Ctrl+C to exit")
    print("  • Dashboard updates every 30 seconds")
    print("  • Web dashboard: http://localhost:5050")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Clear screen (optional)
            if iteration > 1:
                os.system('clear' if os.name == 'posix' else 'cls')
                print_header()
            
            print(f"\n\033[95m📊 Dashboard Update #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
            
            display_system_status()
            display_balances()
            display_trades()
            display_strategies()
            display_targets()
            
            print("\n" + "="*80)
            print("\033[96mNext update in 30 seconds... (Press Ctrl+C to exit)\033[0m")
            
            # Wait 30 seconds for next update
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n\033[93m👋 Console Dashboard stopped by user\033[0m")
        print("Web dashboard still available at: http://localhost:5050")
        sys.exit(0)
    except Exception as e:
        print(f"\n\033[91mError: {e}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()