#!/bin/bash

# ELVIS Trading Bot - Apple Container System Integration
# One-click setup and management for Apple Container app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1  # repo root (scripts moved into scripts/)

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
Apple Container Integration
EOF
    echo -e "${NC}"
}

show_help() {
    echo "ELVIS Trading Bot - Apple Container Management"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Complete setup for Apple Container system"
    echo "  start     - Start ELVIS trading bot"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Show service status"
    echo "  logs      - Show ELVIS logs"
    echo "  reset     - Reset paper trading to initial state"
    echo "  dashboard - Open trading dashboard"
    echo "  test      - Run container setup tests"
    echo "  update    - Update containers"
    echo "  backup    - Backup trading data"
    echo "  help      - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup    # First-time setup"
    echo "  $0 start    # Start trading"
    echo "  $0 logs     # Monitor activity"
}

check_requirements() {
    if ! command -v docker &> /dev/null; then
        print_error "❌ Docker is not installed"
        echo "Please install Docker Desktop for Mac:"
        echo "https://www.docker.com/products/docker-desktop/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "❌ Docker is not running"
        echo "Please start Docker Desktop"
        exit 1
    fi
}

cmd_setup() {
    print_header "🍎 Setting up ELVIS for Apple Container System"
    
    check_requirements
    
    if [[ -f "scripts/setup_apple_containers.sh" ]]; then
        scripts/setup_apple_containers.sh
    else
        print_error "Setup script not found!"
        exit 1
    fi
    
    print_success "✅ Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  $0 start     # Start trading"
    echo "  $0 dashboard # Open dashboard"
}

cmd_start() {
    print_header "🚀 Starting ELVIS Trading Bot"
    
    check_requirements
    
    # Use Apple-optimized configuration if available
    if [[ -f "docker-compose.apple.yml" ]]; then
        export COMPOSE_FILE="docker-compose.yml:docker-compose.apple.yml"
        print_success "Using Apple Container optimizations"
    fi
    
    docker-compose up -d
    
    # Wait for services to be ready
    print_success "Waiting for services to start..."
    sleep 10
    
    # Check if services are healthy
    if docker-compose ps | grep -q "Up"; then
        print_success "✅ ELVIS Trading Bot started successfully!"
        echo ""
        echo "📊 Access points:"
        echo "  • Trading Dashboard: http://localhost:5050"
        echo "  • Grafana Metrics:   http://localhost:3000 (admin/admin)" 
        echo "  • Prometheus:        http://localhost:9090"
        echo ""
        echo "💰 Paper Trading:"
        echo "  • Starting with \$1000 USDT + \$1000 BNB"
        echo "  • Bonenkamp HFT strategy active"
        echo "  • 5-minute trading intervals"
        echo ""
        echo "📱 Management:"
        echo "  $0 logs    # View trading activity"
        echo "  $0 status  # Check system status"
        echo "  $0 stop    # Stop trading"
    else
        print_error "❌ Some services failed to start"
        echo "Check logs: $0 logs"
        exit 1
    fi
}

cmd_stop() {
    print_header "🛑 Stopping ELVIS Trading Bot"
    
    docker-compose down
    print_success "✅ ELVIS stopped"
}

cmd_restart() {
    print_header "🔄 Restarting ELVIS Trading Bot"
    
    cmd_stop
    sleep 2
    cmd_start
}

cmd_status() {
    print_header "📊 ELVIS Service Status"
    
    echo "Container Status:"
    docker-compose ps
    
    echo ""
    echo "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>/dev/null || echo "Docker stats unavailable"
    
    echo ""
    echo "Health Checks:"
    for service in elvis-bot postgres redis; do
        if docker-compose ps "$service" | grep -q "Up"; then
            print_success "✅ $service: Running"
        else
            print_error "❌ $service: Stopped"
        fi
    done
}

cmd_logs() {
    print_header "📋 ELVIS Trading Logs"
    
    if [[ $# -gt 1 ]]; then
        # Follow logs with tail
        docker-compose logs -f --tail=100 elvis-bot
    else
        # Show recent logs
        docker-compose logs --tail=50 elvis-bot
    fi
}

cmd_reset() {
    print_header "🔄 Resetting Paper Trading"
    
    print_warning "This will reset all trading data to initial state:"
    print_warning "• Clear all trades"
    print_warning "• Clear all positions"  
    print_warning "• Reset to \$1000 USDT + \$1000 BNB"
    echo ""
    read -p "Continue? (y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        docker-compose exec elvis-bot python scripts/reset_paper_trading.py
        print_success "✅ Paper trading reset complete"
    else
        echo "Reset cancelled"
    fi
}

cmd_dashboard() {
    print_header "📊 Opening Trading Dashboard"
    
    if command -v open &> /dev/null; then
        open http://localhost:5050
        print_success "Dashboard opened in browser"
    else
        echo "Open manually: http://localhost:5050"
    fi
}

cmd_test() {
    print_header "🧪 Running Container Tests"
    
    if [[ -f "scripts/test_container_setup.sh" ]]; then
        scripts/test_container_setup.sh
    else
        print_error "Test script not found!"
        exit 1
    fi
}

cmd_update() {
    print_header "⬆️ Updating ELVIS Containers"
    
    print_success "Pulling latest images..."
    docker-compose pull
    
    print_success "Rebuilding ELVIS..."
    docker-compose build elvis-bot
    
    print_success "Restarting services..."
    docker-compose up -d
    
    print_success "✅ Update complete"
}

cmd_backup() {
    print_header "💾 Backing up ELVIS Data"
    
    backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    print_success "Backing up database..."
    docker-compose exec -T postgres pg_dump -U elvis_user elvis_trading > "$backup_dir/database.sql"
    
    print_success "Backing up models..."
    tar -czf "$backup_dir/models.tar.gz" models/ 2>/dev/null || true
    
    print_success "Backing up logs..."
    tar -czf "$backup_dir/logs.tar.gz" logs/ 2>/dev/null || true
    
    print_success "✅ Backup created: $backup_dir"
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
            cmd_setup "$@"
            ;;
        "start")
            cmd_start "$@"
            ;;
        "stop")
            cmd_stop "$@"
            ;;
        "restart")
            cmd_restart "$@"
            ;;
        "status")
            cmd_status "$@"
            ;;
        "logs")
            cmd_logs "$@"
            ;;
        "reset")
            cmd_reset "$@"
            ;;
        "dashboard")
            cmd_dashboard "$@"
            ;;
        "test")
            cmd_test "$@"
            ;;
        "update")
            cmd_update "$@"
            ;;
        "backup")
            cmd_backup "$@"
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