#!/bin/bash

# Quick fix for Apple Container build issues
# This script provides multiple approaches to get ELVIS running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_header "🛠️ ELVIS Apple Container Build Fix"
echo ""

# Option 1: Try minimal Docker build first
print_header "Option 1: Building with Docker Desktop (Recommended)"
if command -v docker &> /dev/null && docker info &> /dev/null; then
    print_success "Docker Desktop is available and running"
    echo "Building ELVIS with minimal dependencies..."
    
    if docker build -f Dockerfile.minimal -t elvis-bot:latest .; then
        print_success "✅ Docker build successful!"
        echo ""
        echo "You can now use either:"
        echo "• Docker Compose: ./apple_container_elvis.sh start"
        echo "• Apple Container: Import the image and use native CLI"
        exit 0
    else
        print_warning "Docker build failed, trying alternative approaches..."
    fi
else
    print_warning "Docker Desktop not available or not running"
fi

echo ""

# Option 2: Use pre-built Python environment
print_header "Option 2: Direct Python Execution (No Container)"
print_success "Running ELVIS directly with your Python environment..."

echo "Installing required packages..."
pip install --quiet requests pyyaml python-dotenv colorlog Flask redis ccxt python-binance websocket-client prometheus-client SQLAlchemy tqdm psutil

echo "Starting ELVIS in paper trading mode..."
export TRADING_MODE=paper
export STRATEGY_MODE=ensemble
export HIGH_FREQUENCY_TRADING=true
export INITIAL_USDT_BALANCE=1000.0
export INITIAL_BNB_BALANCE=1000.0

print_success "✅ Starting ELVIS without containers..."
echo ""
echo "📊 Dashboard will be available at: http://localhost:5050"
echo "💰 Paper trading with $1000 USDT + $1000 BNB"
echo ""
echo "To stop: Press Ctrl+C"
echo ""

python main.py --mode paper --log-level INFO

echo ""
print_header "🎯 Alternative Solutions"
echo ""
echo "If you prefer containerized deployment:"
echo ""
echo "1. Use Docker Desktop:"
echo "   docker build -f Dockerfile.minimal -t elvis-bot:latest ."
echo "   docker-compose up"
echo ""
echo "2. Fix Apple Container networking:"
echo "   • Check Apple Container network settings"
echo "   • Try container system restart: container system stop && container system start"
echo "   • Use VPN or different network connection"
echo ""
echo "3. Use the Docker Compose workflow:"
echo "   ./apple_container_elvis.sh setup"
echo "   ./apple_container_elvis.sh start"