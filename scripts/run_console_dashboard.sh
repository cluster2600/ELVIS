#!/bin/bash

# ELVIS Console Dashboard Runner
# This script runs ELVIS in a container with the console dashboard


# Resolve repo root so this script works from any cwd (moved into scripts/).
cd "$(dirname "$0")/.." || exit 1
echo "🚀 Starting ELVIS Console Dashboard in Container..."

# Stop any existing elvis containers
docker stop elvis-console 2>/dev/null || true
docker rm elvis-console 2>/dev/null || true

# Run ELVIS with console dashboard in container
# First install missing dependencies, then run ELVIS
docker run -it --name elvis-console --network host \
  -e TRADING_MODE=paper \
  -e STRATEGY_MODE=ensemble \
  -e HIGH_FREQUENCY_TRADING=true \
  -e POSTGRES_HOST=localhost \
  -e POSTGRES_PORT=5432 \
  -e POSTGRES_USER=elvis_user \
  -e POSTGRES_PASSWORD=elvis_password \
  -e POSTGRES_DBNAME=elvis_trading \
  -e REDIS_HOST=localhost \
  -e REDIS_PORT=6379 \
  -e INITIAL_USDT_BALANCE=1000.0 \
  -e INITIAL_BNB_BALANCE=1000.0 \
  elvis-trading-bot:simple bash -c "
    echo '📦 Installing missing dependencies...'
    pip install --quiet ta numpy pandas scipy scikit-learn matplotlib statsmodels
    echo '✅ Dependencies installed'
    echo '🎯 Starting ELVIS Console Dashboard...'
    echo '📊 Look for Market Depth on the RIGHT side (columns 94-120)'
    echo '💰 Account: \$1000 USDT + \$1000 BNB paper trading'
    echo '🔄 Press Ctrl+C to exit'
    echo ''
    python main.py --mode paper --log-level INFO
  "

echo "Console dashboard stopped."