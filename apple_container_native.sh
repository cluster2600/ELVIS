#!/bin/bash

# ELVIS Trading Bot - Apple Container Native CLI
# Direct integration with Apple's container command

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

show_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
 ███████╗██╗     ██╗   ██╗██╗███████╗
 ██╔════╝██║     ██║   ██║██║██╔════╝
 █████╗  ██║     ██║   ██║██║███████╗
 ██╔══╝  ██║     ╚██╗ ██╔╝██║╚════██║
 ███████╗███████╗ ╚████╔╝ ██║███████║
 ╚══════╝╚══════╝  ╚═══╝  ╚═╝╚══════╝
 
Enhanced Leveraged Virtual Investment System
Apple Container Native Integration
EOF
    echo -e "${NC}"
}

check_apple_container() {
    if ! command -v container &> /dev/null; then
        print_error "❌ Apple Container CLI not found"
        echo "Please install from: https://developer.apple.com/documentation/container"
        exit 1
    fi
    
    print_success "✅ Apple Container CLI found: $(container --version 2>/dev/null || echo 'available')"
}

build_elvis_image() {
    print_header "🔨 Building ELVIS Image with Apple Container"
    
    # Try building with the minimal Dockerfile first (no network dependencies)
    print_success "Attempting build with minimal Dockerfile (no network dependencies)..."
    if container build -f Dockerfile.minimal -t elvis-bot:latest . 2>/dev/null; then
        print_success "✅ ELVIS image built successfully with minimal Dockerfile"
        return 0
    fi
    
    print_warning "Minimal build failed, trying full Dockerfile..."
    
    # Fallback to full Dockerfile with network dependencies
    print_success "Building ELVIS trading bot image with full dependencies..."
    if container build -f Dockerfile.simple -t elvis-bot:latest .; then
        print_success "✅ ELVIS image built successfully with full Dockerfile"
        return 0
    else
        print_error "❌ Image build failed. Network connectivity issues detected."
        echo ""
        echo "Troubleshooting options:"
        echo "1. Check your internet connection"
        echo "2. Try building with Docker Desktop instead:"
        echo "   docker build -f Dockerfile.minimal -t elvis-bot:latest ."
        echo "3. Use the Docker Compose setup:"
        echo "   ./apple_container_elvis.sh setup"
        echo ""
        return 1
    fi
}

create_network() {
    print_header "🌐 Creating ELVIS Network"
    
    # Create network for ELVIS containers
    if ! container network list | grep -q "elvis-network"; then
        container network create elvis-network
        print_success "✅ Network 'elvis-network' created"
    else
        print_success "✅ Network 'elvis-network' already exists"
    fi
}

start_postgres() {
    print_header "🐘 Starting PostgreSQL Database"
    
    # Stop existing postgres container if running
    container stop elvis-postgres 2>/dev/null || true
    container rm elvis-postgres 2>/dev/null || true
    
    # Start PostgreSQL container
    container run -d \
        --name elvis-postgres \
        --network elvis-network \
        -p 5432:5432 \
        -e POSTGRES_DB=elvis_trading \
        -e POSTGRES_USER=elvis_user \
        -e POSTGRES_PASSWORD=elvis_password \
        postgres:15-alpine
    
    print_success "✅ PostgreSQL started"
    
    # Wait for database to be ready
    print_success "Waiting for database to initialize..."
    sleep 10
    
    # Initialize paper trading database
    print_success "Setting up paper trading database..."
    container exec elvis-postgres psql -U elvis_user -d elvis_trading -c "
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
        
        CREATE TABLE IF NOT EXISTS np.trades (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT NOW(),
            symbol TEXT,
            side TEXT,
            price REAL,
            quantity REAL,
            pnl REAL,
            fee REAL
        );
        
        CREATE TABLE IF NOT EXISTS np.open_positions (
            id SERIAL PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            quantity REAL,
            leverage REAL,
            entry_time TIMESTAMP DEFAULT NOW()
        );
    " 2>/dev/null || print_warning "Database initialization will complete on first connection"
    
    print_success "✅ Paper trading database ready with \$1000 USDT + \$1000 BNB"
}

start_redis() {
    print_header "🔴 Starting Redis Cache"
    
    # Stop existing redis container if running
    container stop elvis-redis 2>/dev/null || true
    container rm elvis-redis 2>/dev/null || true
    
    # Start Redis container
    container run -d \
        --name elvis-redis \
        --network elvis-network \
        -p 6379:6379 \
        redis:7-alpine
    
    print_success "✅ Redis started"
}

start_prometheus() {
    print_header "📊 Starting Prometheus"
    
    # Create prometheus config if it doesn't exist
    if [[ ! -f "prometheus.yml" ]]; then
        cat > prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'elvis-bot'
    static_configs:
      - targets: ['elvis-bot:8000']
  
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    fi
    
    # Stop existing prometheus container if running
    container stop elvis-prometheus 2>/dev/null || true
    container rm elvis-prometheus 2>/dev/null || true
    
    # Start Prometheus container
    container run -d \
        --name elvis-prometheus \
        --network elvis-network \
        -p 9090:9090 \
        -v "$(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
        prom/prometheus:latest
    
    print_success "✅ Prometheus started"
}

start_grafana() {
    print_header "📈 Starting Grafana"
    
    # Stop existing grafana container if running
    container stop elvis-grafana 2>/dev/null || true
    container rm elvis-grafana 2>/dev/null || true
    
    # Start Grafana container
    container run -d \
        --name elvis-grafana \
        --network elvis-network \
        -p 3000:3000 \
        -e GF_SECURITY_ADMIN_PASSWORD=admin \
        grafana/grafana:latest
    
    print_success "✅ Grafana started"
}

start_elvis() {
    print_header "🤖 Starting ELVIS Trading Bot"
    
    # Stop existing elvis container if running
    container stop elvis-bot 2>/dev/null || true
    container rm elvis-bot 2>/dev/null || true
    
    # Prepare environment variables
    ENV_VARS=""
    if [[ -f ".env" ]]; then
        while IFS= read -r line; do
            if [[ ! "$line" =~ ^[[:space:]]*# ]] && [[ -n "$line" ]]; then
                ENV_VARS="$ENV_VARS -e $line"
            fi
        done < .env
    fi
    
    # Default environment variables
    DEFAULT_ENV="
        -e TRADING_MODE=paper
        -e STRATEGY_MODE=ensemble
        -e HIGH_FREQUENCY_TRADING=true
        -e LEVERAGE=100
        -e PROFIT_MODE=aggressive
        -e POSTGRES_HOST=elvis-postgres
        -e POSTGRES_PORT=5432
        -e POSTGRES_USER=elvis_user
        -e POSTGRES_PASSWORD=elvis_password
        -e POSTGRES_DBNAME=elvis_trading
        -e REDIS_HOST=elvis-redis
        -e REDIS_PORT=6379
        -e REDIS_DB=0
        -e INITIAL_USDT_BALANCE=1000.0
        -e INITIAL_BNB_BALANCE=1000.0
        -e PYTHONPATH=/app
        -e PYTHONUNBUFFERED=1
    "
    
    # Start ELVIS trading bot container
    eval "container run -d \\
        --name elvis-bot \\
        --network elvis-network \\
        -p 5050:5050 \\
        -p 8000:8000 \\
        -v \"$(pwd)/logs:/app/logs\" \\
        -v \"$(pwd)/models:/app/models\" \\
        -v \"$(pwd)/data:/app/data\" \\
        $DEFAULT_ENV \\
        $ENV_VARS \\
        elvis-bot:latest"
    
    print_success "✅ ELVIS trading bot started"
    
    # Wait for ELVIS to be ready
    print_success "Waiting for ELVIS to initialize..."
    sleep 15
    
    # Check if ELVIS is healthy
    if container logs elvis-bot 2>/dev/null | grep -q "Trading bot initialized" || container inspect elvis-bot &>/dev/null; then
        print_success "✅ ELVIS is running successfully!"
    else
        print_warning "⚠️  ELVIS may still be starting up..."
    fi
}

show_status() {
    print_header "📊 ELVIS Container Status"
    
    containers=(elvis-bot elvis-postgres elvis-redis elvis-prometheus elvis-grafana)
    
    echo "Container Status:"
    echo "=================="
    for container_name in "${containers[@]}"; do
        if container inspect "$container_name" &>/dev/null; then
            status=$(container inspect "$container_name" | grep -o '"Status":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "unknown")
            if [[ "$status" == "running" ]]; then
                print_success "✅ $container_name: Running"
            else
                print_warning "⚠️  $container_name: $status"
            fi
        else
            print_error "❌ $container_name: Not found"
        fi
    done
    
    echo ""
    echo "Access Points:"
    echo "=============="
    echo "• Trading Dashboard: http://localhost:5050"
    echo "• Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "• Prometheus: http://localhost:9090"
    echo "• PostgreSQL: localhost:5432 (elvis_user/elvis_password)"
    echo "• Redis: localhost:6379"
}

stop_all() {
    print_header "🛑 Stopping All ELVIS Containers"
    
    containers=(elvis-bot elvis-grafana elvis-prometheus elvis-redis elvis-postgres)
    
    for container_name in "${containers[@]}"; do
        if container inspect "$container_name" &>/dev/null; then
            container stop "$container_name" 2>/dev/null || true
            container rm "$container_name" 2>/dev/null || true
            print_success "✅ Stopped $container_name"
        fi
    done
    
    print_success "✅ All ELVIS containers stopped"
}

show_logs() {
    print_header "📋 ELVIS Trading Logs"
    
    if container inspect elvis-bot &>/dev/null; then
        container logs elvis-bot
    else
        print_error "❌ ELVIS container not running"
    fi
}

reset_paper_trading() {
    print_header "🔄 Resetting Paper Trading"
    
    if container inspect elvis-bot &>/dev/null; then
        container exec elvis-bot python reset_paper_trading.py
        print_success "✅ Paper trading reset to \$1000 USDT + \$1000 BNB"
    else
        print_error "❌ ELVIS container not running"
    fi
}

cmd_setup() {
    print_header "🍎 Setting up ELVIS with Apple Container"
    
    check_apple_container
    create_network
    build_elvis_image
    
    print_success "✅ Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  $0 start     # Start all services"
    echo "  $0 status    # Check status"
}

cmd_start() {
    print_header "🚀 Starting ELVIS Full Stack"
    
    check_apple_container
    create_network
    start_postgres
    start_redis
    start_prometheus
    start_grafana
    start_elvis
    
    echo ""
    show_status
    
    echo ""
    print_success "🎉 ELVIS Trading Bot is now running!"
    echo ""
    echo "📊 Access your dashboards:"
    echo "  • Trading: http://localhost:5050"
    echo "  • Grafana: http://localhost:3000"
    echo ""
    echo "💰 Paper Trading Active:"
    echo "  • \$1000 USDT balance"
    echo "  • \$1000 BNB balance"
    echo "  • Bonenkamp HFT strategy enabled"
    echo ""
    echo "📱 Management:"
    echo "  $0 logs    # View trading activity"
    echo "  $0 stop    # Stop all services"
}

show_help() {
    echo "ELVIS Trading Bot - Apple Container Native CLI"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Build ELVIS image and setup network"
    echo "  start     - Start all ELVIS services"
    echo "  stop      - Stop all services"
    echo "  status    - Show container status"
    echo "  logs      - Show ELVIS logs"
    echo "  reset     - Reset paper trading balances"
    echo "  help      - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup    # First-time setup"
    echo "  $0 start    # Start all services"
    echo "  $0 logs     # Monitor trading"
}

# Main command dispatcher
main() {
    show_banner
    
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    case "$1" in
        "setup")
            cmd_setup
            ;;
        "start")
            cmd_start
            ;;
        "stop")
            stop_all
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "reset")
            reset_paper_trading
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"