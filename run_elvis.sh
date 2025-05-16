#!/bin/bash

echo "
 _______  _        __      __  _____   _____ 
|  ____| | |       \ \    / / |_   _| / ____|
| |__    | |        \ \  / /    | |  | (___  
|  __|   | |         \ \/ /     | |   \___ \ 
| |____  | |____      \  /     _| |_  ____) |
|______| |______|      \/     |_____||_____/ 
"

# --- Load .env ---
if [ -f .env ]; then
    echo "🔧 Loading environment variables..."
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ .env file not found."
    exit 1
fi

# --- env-coreml setup ---
if [ ! -d "env-coreml" ]; then
    echo "📦 Creating env-coreml..."
    python3.11 -m venv env-coreml
fi

echo "📄 Installing coreml requirements..."
source env-coreml/bin/activate
pip install --upgrade pip
if [ -f requirements_coreml.txt ]; then
    pip install -r requirements_coreml.txt
else
    echo "❌ requirements_coreml.txt not found."
    deactivate
    exit 1
fi
deactivate

# --- env-ydf setup ---
if [ ! -d "env-ydf" ]; then
    echo "📦 Creating env-ydf..."
    python3.11 -m venv env-ydf
fi

echo "📄 Installing ydf requirements..."
source env-ydf/bin/activate
pip install --upgrade pip
if [ -f requirements_ydf.txt ]; then
    pip install -r requirements_ydf.txt
else
    echo "❌ requirements_ydf.txt not found."
    deactivate
    exit 1
fi
deactivate

# --- Activate CoreML environment and run bot ---
echo "🚀 Activating env-coreml..."
source env-coreml/bin/activate
echo "[DEBUG] Python: $(which python)"
python --version

# --- Default values ---
MODE="paper"
SYMBOL="BTCUSDT"
TIMEFRAME="1h"
LEVERAGE=125
STRATEGY="ensemble"
LOG_LEVEL="INFO"
DASHBOARD="console"
ENVIRONMENT="testnet"

# --- Parse CLI arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --symbol) SYMBOL="$2"; shift 2 ;;
        --timeframe) TIMEFRAME="$2"; shift 2 ;;
        --leverage) LEVERAGE="$2"; shift 2 ;;
        --strategy) STRATEGY="$2"; shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --dashboard) DASHBOARD="$2"; shift 2 ;;
        --environment) ENVIRONMENT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Safety prompt for live mode ---
if [ "$MODE" == "live" ] && [ "$ENVIRONMENT" == "production" ]; then
    echo "⚠️ LIVE trading on PRODUCTION! Ctrl+C to abort!"
    sleep 5
elif [ "$MODE" == "live" ]; then
    echo "⚠️ LIVE trading on TESTNET. Ctrl+C to cancel..."
    sleep 5
fi

# --- Launch the bot ---
echo "🧠 Starting ELVIS..."
python main.py \
    --mode "$MODE" \
    --symbol "$SYMBOL" \
    --timeframe "$TIMEFRAME" \
    --leverage "$LEVERAGE" \
    --strategy "$STRATEGY" \
    --log-level "$LOG_LEVEL" \
    --dashboard "$DASHBOARD" \
    --environment "$ENVIRONMENT"