#!/bin/bash

# Script to run the trading bot with proper terminal support for curses dashboard

# Set proper terminal environment
export TERM=xterm-256color

# Ensure we're running in a TTY
if [ ! -t 0 ]; then
    echo "No TTY detected. Please run this script in a terminal."
    echo ""
    echo "Options to run with curses dashboard:"
    echo "1. Run directly in terminal:"
    echo "   python3 main.py --mode paper"
    echo ""
    echo "2. Run in screen session:"
    echo "   screen -S trading_bot python3 main.py --mode paper"
    echo ""
    echo "3. Run in tmux session:"
    echo "   tmux new-session -d -s trading_bot 'python3 main.py --mode paper'"
    echo "   tmux attach -t trading_bot"
    echo ""
    echo "4. Run with script utility (maintains TTY):"
    echo "   script -q /dev/null python3 main.py --mode paper"
    exit 1
fi

echo "🚀 Starting ELVIS Trading Bot with Candlestick Dashboard..."
echo "Terminal: $TERM"
echo "TTY: $(tty)"
echo ""
echo "🕯️  NEW: Real-time candlestick chart with OHLC data!"
echo "📊 Features:"
echo "   - Live BTC/USDT candlestick chart"
echo "   - Real price data from Binance" 
echo "   - Technical indicators (RSI, MACD, SMA)"
echo "   - Live trading logs"
echo "   - Paper trading execution"
echo ""
echo "Controls:"
echo "- Press 'q' to quit dashboard"
echo "- Press '1-6' to change timeframes"
echo "- Press 'd' for drawing mode"
echo ""

# Run the bot
python3 main.py --mode paper --log-level INFO