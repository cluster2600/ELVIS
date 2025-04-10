#!/bin/bash

# ASCII Art for ELVIS
echo "
 _______  _        __      __  _____   _____ 
|  ____| | |       \ \    / / |_   _| / ____|
| |__    | |        \ \  / /    | |  | (___  
|  __|   | |         \ \/ /     | |   \___ \ 
| |____  | |____      \  /     _| |_  ____) |
|______| |______|      \/     |_____||_____/ 
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
SYMBOL="BTCUSDT"
TIMEFRAME="1h"
LEVERAGE=125
STRATEGY="ensemble"
LOG_LEVEL="INFO"
DASHBOARD="console"
ENVIRONMENT="testnet"

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
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --dashboard)
            DASHBOARD="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if trying to run in live mode
if [ "$MODE" == "live" ] && [ "$ENVIRONMENT" == "production" ]; then
    echo "⚠️  WARNING: Starting ELVIS in LIVE mode on PRODUCTION environment. Real trading will occur!"
    echo "You have 5 seconds to cancel (Ctrl+C)..."
    sleep 5
elif [ "$MODE" == "live" ] && [ "$ENVIRONMENT" == "testnet" ]; then
    echo "⚠️  WARNING: Starting ELVIS in LIVE mode on TESTNET environment. Paper trading will occur!"
    echo "You have 5 seconds to cancel (Ctrl+C)..."
    sleep 5
fi

echo "Starting ELVIS in $MODE mode for $SYMBOL on $TIMEFRAME timeframe with $LEVERAGE leverage using $STRATEGY strategy with $DASHBOARD dashboard on $ENVIRONMENT environment..."
python main.py --mode $MODE --symbol $SYMBOL --timeframe $TIMEFRAME --leverage $LEVERAGE --strategy $STRATEGY --log-level $LOG_LEVEL --dashboard $DASHBOARD --environment $ENVIRONMENT
