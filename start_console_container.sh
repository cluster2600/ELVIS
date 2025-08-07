#!/bin/bash

# Start ELVIS in container and show console dashboard logs
echo "🚀 Starting ELVIS Console Dashboard in Container (Non-Interactive Mode)"

# Clean up any existing containers
docker stop elvis-console 2>/dev/null || true
docker rm elvis-console 2>/dev/null || true

# Start ELVIS in container with console output
docker run --name elvis-console --network host \
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
    pip install --quiet ta numpy pandas scipy scikit-learn matplotlib statsmodels protobuf
    echo '✅ Dependencies installed'
    echo '🎯 Starting ELVIS Console Dashboard...'
    echo '📊 Console dashboard will show live trading data'
    echo '💰 Paper trading: \$1000 USDT + \$1000 BNB'
    echo '🔄 Use docker logs elvis-console -f to follow logs'
    echo ''
    python main.py --mode paper --log-level INFO
  " &

echo "✅ ELVIS container started in background"
echo ""
echo "📊 To view the console dashboard output:"
echo "   docker logs elvis-console -f"
echo ""
echo "🛑 To stop the container:"
echo "   docker stop elvis-console"
echo ""
echo "🔄 To restart:"
echo "   ./start_console_container.sh"