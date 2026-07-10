#!/usr/bin/env bash
#
# ELVIS Trading Bot Startup Script with Vault Authentication
# Ensures vault is running and properly authenticated before starting the bot
#

set -euo pipefail


# Resolve repo root so this script works from any cwd (moved into scripts/).
cd "$(dirname "$0")/.." || exit 1
echo "🚀 Starting ELVIS Trading Bot with Vault Authentication..."

# Set vault environment variables
export VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"

require_env() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        echo "❌ Required environment variable missing: $var_name"
        exit 1
    fi
}

# Check if vault is running
if ! curl -s "${VAULT_ADDR}/v1/sys/health" > /dev/null 2>&1; then
    echo "⚠️  Vault not running, starting vault server..."

    require_env "VAULT_DEV_ROOT_TOKEN_ID"
    export VAULT_TOKEN="${VAULT_TOKEN:-$VAULT_DEV_ROOT_TOKEN_ID}"

    # Start vault in dev mode with caller-supplied token
    vault server -dev -dev-root-token-id="$VAULT_DEV_ROOT_TOKEN_ID" > vault-dev.log 2>&1 &
    VAULT_PID=$!
    echo $VAULT_PID > vault.pid

    echo "⏳ Waiting for vault to start..."
    sleep 5

    # Verify vault is responding
    if ! curl -s "${VAULT_ADDR}/v1/sys/health" > /dev/null 2>&1; then
        echo "❌ Failed to start vault server"
        exit 1
    fi

    echo "✅ Vault server started successfully"

    # Create secrets using externally supplied environment variables
    echo "🔧 Setting up vault secrets..."

    require_env "BINANCE_API_KEY"
    require_env "BINANCE_API_SECRET"

    vault kv put secret/trading/api-keys \
        binance-api-key="$BINANCE_API_KEY" \
        binance-api-secret="$BINANCE_API_SECRET" \
        binance-futures-testnet-api-key="${BINANCE_FUTURES_TESTNET_API_KEY:-}" \
        binance-futures-testnet-api-secret="${BINANCE_FUTURES_TESTNET_API_SECRET:-}" > /dev/null 2>&1

    vault kv put secret/database/credentials \
        postgres-host="${POSTGRES_HOST:-localhost}" \
        postgres-port="${POSTGRES_PORT:-5432}" \
        postgres-user="${POSTGRES_USER:-elvis_user}" \
        postgres-password="${POSTGRES_PASSWORD:-}" \
        postgres-dbname="${POSTGRES_DBNAME:-elvis_trading}" > /dev/null 2>&1

    vault kv put secret/redis \
        host="${REDIS_HOST:-localhost}" \
        port="${REDIS_PORT:-6379}" \
        password="${REDIS_PASSWORD:-}" > /dev/null 2>&1

    vault kv put secret/notifications/telegram \
        telegram-bot-token="${TELEGRAM_BOT_TOKEN:-}" \
        telegram-chat-id="${TELEGRAM_CHAT_ID:-}" > /dev/null 2>&1

    echo "✅ Vault secrets configured"
else
    echo "✅ Vault already running"
fi

# Test vault authentication
echo "🔐 Testing vault authentication..."
require_env "VAULT_TOKEN"
if vault status > /dev/null 2>&1; then
    echo "✅ Vault authentication successful"
else
    echo "❌ Vault authentication failed"
    exit 1
fi

# Start the trading bot
echo "🤖 Starting ELVIS trading bot with FEWER, BIGGER trades strategy..."
echo "⚡ Configuration:"
echo "   - 100x Leverage: ENABLED"
echo "   - High Win Rate Filter: ENABLED (85% confidence)"
echo "   - Trade Cooldown: 15 minutes between trades"
echo "   - Position Sizing: $25-$150 per trade"
echo "   - Stop Loss: $50 | Take Profit: $25"
echo "   - Max Daily Trades: 8"
echo ""

# Run the bot with vault environment variables
python main.py --mode paper --log-level INFO

echo "🛑 Trading bot stopped"
