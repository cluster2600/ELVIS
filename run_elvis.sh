#!/bin/bash

# ASCII Art for ELVIS
echo "
 _______ __     ____  __ ____ _____
|  ____| \ \   / /  \/  |___ \_   _|
| |__     \ \_/ /| \  / | __) || |
|  __|     \   / | |\/| ||__ < | |
| |____     | |  | |  | |___) || |_
|______|    |_|  |_|  |_|____/_____|

Enhanced Leveraged Virtual Investment System
"

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv310" ]; then
    echo "Activating virtual environment..."
    source venv310/bin/activate
fi

# Check for command line arguments
MODE="paper"  # Default fallback if config can't be read
# Try to read default mode from config.py
if [ -f "config/config.py" ]; then
    DEFAULT_MODE=$(grep "DEFAULT_MODE" config/config.py | grep -o "'[^']*'" | sed "s/'//g")
    if [ ! -z "$DEFAULT_MODE" ]; then
        MODE="$DEFAULT_MODE"
    fi
fi
SYMBOL="BTCUSDT"
TIMEFRAME="1h"
LEVERAGE=75
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --timeframe)
            TIMEFRAME="$2"
            shift 2
            ;;
        --leverage)
            LEVERAGE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if trying to run in live mode
if [ "$MODE" == "live" ]; then
    # Try to read PRODUCTION_MODE from config.py
    PRODUCTION_MODE="false"
    if [ -f "config/config.py" ]; then
        PRODUCTION_MODE_VALUE=$(grep "PRODUCTION_MODE" config/config.py | grep -o "True\|False")
        if [ "$PRODUCTION_MODE_VALUE" == "False" ]; then
            echo "⚠️  WARNING: PRODUCTION_MODE is disabled in config. Cannot run in live mode."
            echo "To enable live trading, set PRODUCTION_MODE to True in config/config.py"
            echo "Exiting..."
            exit 1
        fi
    fi
    
    echo "⚠️  WARNING: Starting ELVIS in LIVE mode. Real trading will occur!"
    echo "You have 5 seconds to cancel (Ctrl+C)..."
    sleep 5
fi

echo "Starting ELVIS in $MODE mode for $SYMBOL on $TIMEFRAME timeframe with $LEVERAGE leverage..."
python main.py --mode $MODE --symbol $SYMBOL --timeframe $TIMEFRAME --leverage $LEVERAGE --log-level $LOG_LEVEL
