#!/bin/bash

# ELVIS Trading Bot - Service Startup Script
# This script starts all required external services for the trading bot

echo "🚀 Starting ELVIS Trading Bot Services..."
echo "=" * 45

# Set environment variables
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='trading-bot-token'
export PROMETHEUS_PUSHGATEWAY_URL='http://localhost:9091'

# Function to check if a service is running on a port
check_port() {
    lsof -i :$1 > /dev/null 2>&1
    return $?
}

# Function to wait for service to start
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=10
    local attempt=1

    echo "⏳ Waiting for $service_name to start on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if check_port $port; then
            echo "✅ $service_name is running"
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start"
    return 1
}

# 1. Start Vault (if not running)
echo "🔐 Starting HashiCorp Vault..."
if check_port 8200; then
    echo "✅ Vault already running on port 8200"
else
    vault server -dev -dev-root-token-id=trading-bot-token > vault-dev.log 2>&1 &
    VAULT_PID=$!
    echo $VAULT_PID > vault.pid
    
    if wait_for_service 8200 "Vault"; then
        # Configure KV engine and secrets
        echo "📝 Configuring Vault secrets..."
        vault secrets list | grep secret > /dev/null || vault secrets enable -path=secret kv-v2
        
        # Create secrets at paths expected by the secrets manager
        vault kv put secret/trading/api-keys \
            binance-api-key=test-api-key \
            binance-api-secret=test-api-secret \
            telegram-bot-token=test-bot-token > /dev/null
            
        vault kv put secret/database/credentials \
            postgres-host=localhost \
            postgres-port=5432 \
            postgres-user=elvis_user \
            postgres-password=elvis_password \
            redis-host=localhost \
            redis-port=6379 > /dev/null
            
        echo "✅ Vault configured with secrets"
    fi
fi

# 2. Start Prometheus Pushgateway (if not running)
echo "📊 Starting Prometheus Pushgateway..."
if check_port 9091; then
    echo "✅ Pushgateway already running on port 9091"
else
    if [ ! -d "tools/pushgateway-1.6.2.darwin-amd64" ]; then
        echo "📥 Downloading Pushgateway..."
        mkdir -p tools
        cd tools
        curl -L -s -o pushgateway.tar.gz https://github.com/prometheus/pushgateway/releases/download/v1.6.2/pushgateway-1.6.2.darwin-amd64.tar.gz
        tar -xzf pushgateway.tar.gz
        cd ..
    fi
    
    nohup ./tools/pushgateway-1.6.2.darwin-amd64/pushgateway --web.listen-address=:9091 > pushgateway.log 2>&1 &
    PUSH_PID=$!
    echo $PUSH_PID > pushgateway.pid
    
    wait_for_service 9091 "Pushgateway"
fi

# 3. Test all connections
echo "🔧 Testing all API connections..."
python3 -c "
from utils.api_connection_tester import get_api_tester
from utils.logging_utils import setup_logger
import os

# Set environment variables for Python
os.environ['VAULT_ADDR'] = 'http://127.0.0.1:8200'
os.environ['VAULT_TOKEN'] = 'trading-bot-token'
os.environ['PROMETHEUS_PUSHGATEWAY_URL'] = 'http://localhost:9091'

logger = setup_logger('startup_test')
tester = get_api_tester(logger)

statuses = tester.test_all_apis()
health = tester.get_overall_health()

critical_services = 0
for name in ['binance_spot', 'binance_futures', 'postgres', 'vault']:
    if statuses[name].status.value == 'connected':
        critical_services += 1

print(f'🏥 System Health: {health[\"overall_status\"].upper()} ({health[\"health_percentage\"]:.0f}%)')
print(f'📈 Critical Services: {critical_services}/4 online')

if critical_services == 4:
    print('🎉 All critical services operational - Ready for trading!')
else:
    print('⚠️  Some critical services offline - Check configuration')

tester.stop_monitoring()
"

# Set environment variables for the session
echo ""
echo "🔧 Setting environment variables..."
echo "export VAULT_ADDR='http://127.0.0.1:8200'" >> ~/.bashrc
echo "export VAULT_TOKEN='trading-bot-token'" >> ~/.bashrc  
echo "export PROMETHEUS_PUSHGATEWAY_URL='http://localhost:9091'" >> ~/.bashrc

echo ""
echo "✅ Service startup complete!"
echo "🚀 ELVIS Trading Bot services are ready"
echo ""
echo "💡 To set environment variables in current shell, run:"
echo "   export VAULT_ADDR='http://127.0.0.1:8200'"
echo "   export VAULT_TOKEN='trading-bot-token'"
echo "   export PROMETHEUS_PUSHGATEWAY_URL='http://localhost:9091'"