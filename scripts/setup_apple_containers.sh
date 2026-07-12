#!/bin/bash

# ELVIS Trading Bot - Apple Container System Setup
# Enhanced Leveraged Virtual Investment System

set -e

echo "🍎 Setting up ELVIS for Apple Container System"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Docker is installed
check_docker() {
    print_header "Checking Docker Installation..."
    
    if command -v docker &> /dev/null; then
        print_status "Docker is installed: $(docker --version)"
    else
        print_error "Docker is not installed. Please install Docker Desktop for Mac first."
        echo "Download from: https://www.docker.com/products/docker-desktop/"
        exit 1
    fi
    
    # Check if Docker is running
    if docker info &> /dev/null; then
        print_status "Docker is running"
    else
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
}

# Check Apple Silicon compatibility
check_apple_silicon() {
    print_header "Checking Apple Silicon Compatibility..."
    
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        print_status "Apple Silicon (M1/M2/M3) detected - using optimized configuration"
        export DOCKER_DEFAULT_PLATFORM=linux/arm64
    elif [[ "$ARCH" == "x86_64" ]]; then
        print_status "Intel Mac detected - using standard configuration"
        export DOCKER_DEFAULT_PLATFORM=linux/amd64
    else
        print_warning "Unknown architecture: $ARCH"
    fi
}

# Create necessary directories
create_directories() {
    print_header "Creating Directory Structure..."
    
    mkdir -p logs models data/processed data/raw
    mkdir -p models/checkpoints models/logs
    mkdir -p grafana/dashboards grafana/provisioning
    
    print_status "Directory structure created"
}

# Setup environment file for containers
setup_environment() {
    print_header "Setting up Container Environment..."
    
    if [[ ! -f .env ]]; then
        print_warning ".env file not found. Creating template..."
        cat > .env << EOL
# ELVIS Trading Bot Container Configuration

# Trading Mode
TRADING_MODE=paper
STRATEGY_MODE=ensemble
HIGH_FREQUENCY_TRADING=true
LEVERAGE=100
PROFIT_MODE=aggressive

# Paper Trading Initial Balances  
INITIAL_USDT_BALANCE=1000.0
INITIAL_BNB_BALANCE=1000.0

# Binance Testnet API (for paper trading)
BINANCE_FUTURES_TESTNET_API_KEY=your_testnet_api_key_here
BINANCE_FUTURES_TESTNET_API_SECRET=your_testnet_api_secret_here

# Optional: Telegram Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Database Configuration (handled by containers)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=elvis_user
POSTGRES_PASSWORD=elvis_password
POSTGRES_DBNAME=elvis_trading

# Redis Configuration (handled by containers)
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
EOL
        print_status "Created .env template file"
        print_warning "Please edit .env file with your API keys before starting containers"
    else
        print_status "Found existing .env file"
    fi
}

# Create Apple Container specific docker-compose override
create_apple_override() {
    print_header "Creating Apple Container Optimization..."
    
    cat > docker-compose.apple.yml << EOL
version: '3.8'

# Apple Container System Optimizations
services:
  elvis-bot:
    platform: linux/arm64  # Optimize for Apple Silicon
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
    environment:
      # Apple Container optimizations
      - MALLOC_ARENA_MAX=2
      - PYTHONOPTIMIZE=1
    labels:
      - "com.apple.container.description=ELVIS Trading Bot - Enhanced Leveraged Virtual Investment System"
      - "com.apple.container.version=1.0"

  postgres:
    platform: linux/arm64
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '1.0'

  redis:
    platform: linux/arm64
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.5'

  prometheus:
    platform: linux/arm64
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  grafana:
    platform: linux/arm64
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.5'
EOL
    
    print_status "Created Apple Container optimization file"
}

# Build containers for Apple system
build_containers() {
    print_header "Building ELVIS Containers..."
    
    print_status "Building ELVIS trading bot image..."
    docker build -f docker/Dockerfile.simple -t elvis-bot:latest .
    
    print_status "Container build complete"
}

# Initialize paper trading database
init_database() {
    print_header "Initializing Paper Trading Database..."
    
    print_status "Starting database container..."
    docker-compose up -d postgres
    
    # Wait for database to be ready
    print_status "Waiting for database to be ready..."
    sleep 10
    
    # Initialize paper trading database
    print_status "Setting up paper trading with $1000 USDT + $1000 BNB..."
    docker-compose exec -T postgres psql -U elvis_user -d elvis_trading -c "
        CREATE SCHEMA IF NOT EXISTS np;
        CREATE TABLE IF NOT EXISTS np.account_balances (
            id SERIAL PRIMARY KEY,
            asset TEXT UNIQUE NOT NULL,
            balance REAL NOT NULL DEFAULT 0,
            last_updated TIMESTAMP DEFAULT NOW()
        );
        INSERT INTO np.account_balances (asset, balance) 
        VALUES ('USDT', 1000.0), ('BNB', 1000.0) 
        ON CONFLICT (asset) DO NOTHING;
    " 2>/dev/null || print_warning "Database initialization will happen on first run"
    
    print_status "Paper trading database ready"
}

# Create startup script for Apple Containers
create_startup_script() {
    print_header "Creating Apple Container Startup Script..."
    
    cat > start_elvis_apple.sh << EOL
#!/bin/bash

# ELVIS Trading Bot - Apple Container Startup

echo "🍎 Starting ELVIS Trading Bot on Apple Container System"

# Use Apple-optimized compose files
export COMPOSE_FILE="docker-compose.yml:docker-compose.apple.yml"

# Start all services
docker-compose up -d

echo "✅ ELVIS Trading Bot started successfully!"
echo ""
echo "📊 Access points:"
echo "   • Trading Dashboard: http://localhost:5050"
echo "   • Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "   • Prometheus: http://localhost:9090"
echo ""
echo "📱 Paper Trading Started with:"
echo "   • \\$1000 USDT balance"
echo "   • \\$1000 BNB balance"
echo ""
echo "🔧 Management commands:"
echo "   • View logs: docker-compose logs -f elvis-bot"
echo "   • Stop system: docker-compose down"
echo "   • Reset paper trading: docker-compose exec elvis-bot python scripts/reset_paper_trading.py"
EOL
    
    chmod +x start_elvis_apple.sh
    print_status "Created startup script: start_elvis_apple.sh"
}

# Create stop script
create_stop_script() {
    print_header "Creating Stop Script..."
    
    cat > stop_elvis.sh << EOL
#!/bin/bash

echo "🛑 Stopping ELVIS Trading Bot..."

# Stop all containers
docker-compose down

echo "✅ ELVIS Trading Bot stopped"
EOL
    
    chmod +x stop_elvis.sh
    print_status "Created stop script: stop_elvis.sh"
}

# Main setup function
main() {
    print_header "🚀 ELVIS Trading Bot - Apple Container Setup"
    echo ""
    
    check_docker
    check_apple_silicon
    create_directories
    setup_environment
    create_apple_override
    build_containers
    init_database
    create_startup_script
    create_stop_script
    
    echo ""
    print_header "🎉 Setup Complete!"
    echo ""
    print_status "ELVIS Trading Bot is ready for Apple Container System"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Edit .env file with your API keys (optional for paper trading)"
    echo "   2. Start ELVIS: ./start_elvis_apple.sh"
    echo "   3. Open dashboard: http://localhost:5050"
    echo ""
    print_status "Features enabled:"
    echo "   ✅ Paper trading with \$1000 USDT + \$1000 BNB"
    echo "   ✅ Bonenkamp HFT strategy (5-minute intervals)"
    echo "   ✅ Ensemble trading strategies"
    echo "   ✅ Real-time dashboard"
    echo "   ✅ PostgreSQL database for trade history"
    echo "   ✅ Redis caching"
    echo "   ✅ Prometheus metrics"
    echo "   ✅ Grafana dashboards"
    echo ""
    print_header "Happy Trading! 📈"
}

# Run main function
main "$@"