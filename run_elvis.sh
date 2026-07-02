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

# --- Validate critical env vars ---
REQUIRED_VARS=("BINANCE_API_KEY" "BINANCE_API_SECRET")
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo "❌ Missing required environment variable: $var"
        exit 1
    fi
done

# --- venv314 setup ---
if [ ! -d "venv314" ]; then
    echo "📦 Creating venv314..."
    python3.14 -m venv venv314
fi

echo "📄 Installing requirements..."
source venv314/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found."
    deactivate
    exit 1
fi

# --- Activate venv314 ---
echo "🚀 Activating venv314..."
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

# --- Ensure logs dir ---
mkdir -p logs

# --- Launch ELVIS ---
echo "🧠 Starting ELVIS..."
python main.py \
    --mode "$MODE" \
    --log-level "$LOG_LEVEL" \
    >> logs/elvis_$(date +'%Y%m%d_%H%M%S').log 2>&1
