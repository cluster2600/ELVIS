# ELVIS Trading Bot - Apple Container System Setup

## Overview

This guide helps you run the **ELVIS Trading Bot** (Enhanced Leveraged Virtual Investment System) using Apple's Container application, which uses Docker images optimized for Apple Silicon (M1/M2/M3) and Intel Macs.

## Features

### 🎯 **Trading Capabilities**
- **Paper Trading**: Start with $1000 USDT + $1000 BNB
- **Bonenkamp HFT Strategy**: 5-minute high-frequency trading
- **Ensemble Strategies**: Multiple trading algorithms
- **Multi-Asset Trading**: Support for BTCUSDT, BNBUSDT, and more

### 🏗️ **Container Architecture**
- **PostgreSQL**: Database for trade history and positions
- **Redis**: High-performance caching
- **Prometheus**: Metrics collection
- **Grafana**: Trading dashboards
- **ELVIS Bot**: Main trading engine

### 🍎 **Apple Optimizations**
- ARM64 architecture support (Apple Silicon)
- Optimized memory usage
- Fast container startup
- Apple Container app integration

## Prerequisites

1. **Docker Desktop for Mac**
   - Download: https://www.docker.com/products/docker-desktop/
   - Ensure Docker is running

2. **Apple Container App**
   - Install from the Mac App Store
   - Or use Docker directly

## Quick Setup

### 1. **Run Automated Setup**
```bash
# Make setup script executable and run
chmod +x setup_apple_containers.sh
./setup_apple_containers.sh
```

### 2. **Configure API Keys (Optional)**
Edit the `.env` file with your Binance testnet API keys:
```bash
# For paper trading, you can use testnet keys or leave empty
BINANCE_FUTURES_TESTNET_API_KEY=your_testnet_key_here
BINANCE_FUTURES_TESTNET_API_SECRET=your_testnet_secret_here
```

### 3. **Start ELVIS**
```bash
./start_elvis_apple.sh
```

### 4. **Access Dashboards**
- **Trading Dashboard**: http://localhost:5050
- **Grafana Metrics**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Manual Setup

If you prefer manual setup:

### 1. **Build Containers**
```bash
# Build ELVIS trading bot
docker build -f Dockerfile.simple -t elvis-bot:latest .
```

### 2. **Start Services**
```bash
# Start all services
docker-compose up -d

# Or with Apple optimizations
export COMPOSE_FILE="docker-compose.yml:docker-compose.apple.yml"
docker-compose up -d
```

### 3. **Initialize Paper Trading**
```bash
# Reset to initial balances
docker-compose exec elvis-bot python reset_paper_trading.py

# Check balances
docker-compose exec elvis-bot python check_paper_balances.py
```

## Container Configuration

### **Services Overview**

| Service | Port | Purpose |
|---------|------|---------|
| elvis-bot | 5050, 8000 | Main trading engine and API |
| postgres | 5432 | Trade history database |
| redis | 6379 | Caching and data storage |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Dashboards and visualization |

### **Volume Mounts**
- `./logs` → Container logs
- `./models` → Trading models
- `./data` → Trading data
- `./.env` → Environment configuration

### **Resource Limits (Apple Silicon)**
```yaml
elvis-bot:
  memory: 2GB
  cpu: 2 cores

postgres:
  memory: 512MB
  cpu: 1 core

redis:
  memory: 256MB
  cpu: 0.5 cores
```

## Trading Configuration

### **Paper Trading Settings**
```bash
TRADING_MODE=paper
INITIAL_USDT_BALANCE=1000.0
INITIAL_BNB_BALANCE=1000.0
```

### **Strategy Configuration**
```bash
STRATEGY_MODE=ensemble
HIGH_FREQUENCY_TRADING=true
LEVERAGE=100
PROFIT_MODE=aggressive
```

### **Bonenkamp HFT Strategy**
- **Frequency**: 5-minute intervals
- **Features**: 9 financial indicators + 2 social features
- **Model**: Random Forest with 600 trees
- **Target**: 14.9% annual return, 2.02 Sharpe ratio

## Management Commands

### **Container Management**
```bash
# View logs
docker-compose logs -f elvis-bot

# Stop all services
docker-compose down

# Restart services
docker-compose restart

# Update containers
docker-compose pull && docker-compose up -d
```

### **Trading Management**
```bash
# Reset paper trading
docker-compose exec elvis-bot python reset_paper_trading.py

# Check balances
docker-compose exec elvis-bot python check_paper_balances.py

# View positions
docker-compose exec elvis-bot python test_positions_display.py

# Test Bonenkamp strategy
docker-compose exec elvis-bot python test_bonenkamp_strategy.py
```

### **Database Management**
```bash
# Connect to database
docker-compose exec postgres psql -U elvis_user -d elvis_trading

# Backup database
docker-compose exec postgres pg_dump -U elvis_user elvis_trading > backup.sql

# View trade history
docker-compose exec postgres psql -U elvis_user -d elvis_trading -c "SELECT * FROM np.trades ORDER BY timestamp DESC LIMIT 10;"
```

## Apple Container App Integration

### **Using Apple Container App**

1. **Import Docker Compose**
   - Open Apple Container app
   - Import `docker-compose.yml`
   - Configure resource limits

2. **Set Environment**
   - Load `.env` file
   - Verify API keys
   - Set memory limits

3. **Start Services**
   - Start all containers
   - Monitor resource usage
   - Check health status

### **Performance Optimization**

For Apple Silicon (M1/M2/M3):
- Use `linux/arm64` platform
- Enable BuildKit for faster builds
- Set appropriate memory limits
- Use local volumes for better performance

## Monitoring and Metrics

### **Grafana Dashboards**
Access: http://localhost:3000 (admin/admin)

Available dashboards:
- Trading Performance
- Position Metrics
- Strategy Analysis
- System Health

### **Prometheus Metrics**
Access: http://localhost:9090

Key metrics:
- Trading signals generated
- Position P&L
- API response times
- System resource usage

### **Real-time Dashboard**
Access: http://localhost:5050

Features:
- Live position monitoring
- Real-time P&L
- Trade execution logs
- Strategy performance

## Troubleshooting

### **Common Issues**

1. **Docker not starting**
   - Ensure Docker Desktop is running
   - Check Docker permissions
   - Restart Docker service

2. **Database connection errors**
   - Wait for PostgreSQL to initialize
   - Check container health status
   - Verify network connectivity

3. **Memory issues on Apple Silicon**
   - Increase Docker Desktop memory
   - Adjust container limits
   - Monitor resource usage

4. **API key errors**
   - Verify testnet API keys
   - Check .env file format
   - Ensure keys have proper permissions

### **Debug Commands**
```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs elvis-bot

# Connect to container
docker-compose exec elvis-bot /bin/bash

# Test database connection
docker-compose exec elvis-bot python -c "from utils.paper_trade_db import get_conn; print('DB OK' if get_conn() else 'DB Error')"
```

## Security Considerations

### **Container Security**
- Non-root user execution
- Read-only file systems where possible
- Network isolation
- Secret management through environment variables

### **API Key Safety**
- Use testnet keys for paper trading
- Never commit real API keys
- Use Docker secrets in production
- Regularly rotate keys

## Updates and Maintenance

### **Updating ELVIS**
```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose build

# Restart with new images
docker-compose up -d
```

### **Backup Strategy**
```bash
# Backup database
docker-compose exec postgres pg_dump -U elvis_user elvis_trading > backup_$(date +%Y%m%d).sql

# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

## Support

### **Getting Help**
- Check logs: `docker-compose logs -f elvis-bot`
- Review configuration: `.env` file
- Test components individually
- Check Apple Container app documentation

### **Performance Tuning**
- Adjust memory limits based on usage
- Monitor CPU utilization
- Optimize container startup order
- Use Apple Silicon-optimized images

---

## 🚀 Quick Start Summary

1. **Setup**: `./setup_apple_containers.sh`
2. **Configure**: Edit `.env` file (optional)
3. **Start**: `./start_elvis_apple.sh`
4. **Access**: http://localhost:5050
5. **Trade**: Paper trading with $1000 USDT + $1000 BNB

**Happy Trading!** 📈