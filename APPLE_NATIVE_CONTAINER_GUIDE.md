# 🍎 ELVIS Trading Bot - Apple Native Container Guide

## Using Apple's Container CLI Tool

You have Apple's native `container` CLI installed! This guide shows you how to run ELVIS using Apple's container system directly instead of Docker.

## ⚡ Quick Start

### **1. Setup ELVIS (One Time)**
```bash
./scripts/apple_container_native.sh setup
```

### **2. Start Trading**
```bash
./scripts/apple_container_native.sh start
```

### **3. Access Dashboard**
```bash
open http://localhost:5050
```

> **Note**: The trade-history API binds `127.0.0.1` by default (see `SECURITY.md`). For the dashboard to be reachable through the container port mapping, set `TRADE_API_HOST=0.0.0.0` in your `.env` file (the script passes `.env` entries into the container).

---

## 🎯 **What's Different**

### **Apple Container vs Docker**
- ✅ **Native Integration**: Built specifically for macOS
- ✅ **Better Performance**: Optimized for Apple Silicon
- ✅ **System Integration**: Works with macOS security
- ✅ **Resource Efficiency**: Lower overhead than Docker

### **Same ELVIS Features**
- 💰 **Paper Trading**: $1000 USDT + $1000 BNB
- 🎯 **Bonenkamp HFT**: 5-minute trading strategy
- 📊 **Real-time Dashboard**: Live monitoring
- 🗄️ **PostgreSQL Database**: Trade history
- 📈 **Grafana Analytics**: Performance metrics

---

## 🛠️ **Apple Container Commands**

### **ELVIS Management**
```bash
# Complete setup
./scripts/apple_container_native.sh setup

# Start all services
./scripts/apple_container_native.sh start

# Check status
./scripts/apple_container_native.sh status

# View logs
./scripts/apple_container_native.sh logs

# Stop services
./scripts/apple_container_native.sh stop

# Reset paper trading
./scripts/apple_container_native.sh reset
```

### **Direct Apple Container Commands**
```bash
# List all containers
container list

# Check ELVIS status
container inspect elvis-bot

# View ELVIS logs
container logs elvis-bot

# Execute commands in ELVIS
container exec elvis-bot python check_paper_balances.py

# Stop specific container
container stop elvis-bot

# Remove container
container rm elvis-bot
```

---

## 🏗️ **Container Architecture**

### **Services Started**
1. **elvis-postgres** - PostgreSQL database (port 5432)
2. **elvis-redis** - Redis cache (port 6379)  
3. **elvis-prometheus** - Metrics collection (port 9090)
4. **elvis-grafana** - Dashboards (port 3000; note: the Docker Compose stack maps Grafana to host port 3001 instead)
5. **elvis-bot** - Trading engine (ports 5050, 8000)

### **Network Configuration**
- **Network**: `elvis-network`
- **Internal Communication**: Container-to-container
- **External Access**: Via port mapping

### **Volume Mounts**
```bash
./logs     → /app/logs      # Trading logs
./models   → /app/models    # AI models
./data     → /app/data      # Market data
```

---

## 📊 **Monitoring with Apple Container**

### **Container Status**
```bash
# List all ELVIS containers
container list | grep elvis

# Detailed container info (JSON output)
container inspect elvis-bot
```

### **Health Checks**
```bash
# Check if ELVIS is responsive
curl http://localhost:5050/health

# Test database connection
container exec elvis-postgres pg_isready -U elvis_user

# Test Redis connection
container exec elvis-redis redis-cli ping
```

### **Log Monitoring**
```bash
# Follow ELVIS logs in real-time
container logs elvis-bot --follow

# Get last 100 log lines
container logs elvis-bot --tail 100

# View specific service logs
container logs elvis-postgres
container logs elvis-redis
```

---

## 🔧 **Configuration**

### **Environment Variables**
The script automatically sets up these environment variables:

```bash
# Trading Configuration
TRADING_MODE=paper
STRATEGY_MODE=ensemble
HIGH_FREQUENCY_TRADING=true
LEVERAGE=100
PROFIT_MODE=aggressive

# Database Configuration
POSTGRES_HOST=elvis-postgres
POSTGRES_USER=elvis_user
POSTGRES_PASSWORD=elvis_password
POSTGRES_DBNAME=elvis_trading

# Paper Trading Balances
INITIAL_USDT_BALANCE=1000.0
INITIAL_BNB_BALANCE=1000.0
```

### **Custom Configuration**
Create a `.env` file to override defaults:
```bash
# Your custom settings
BINANCE_FUTURES_TESTNET_API_KEY=your_key_here
BINANCE_FUTURES_TESTNET_API_SECRET=your_secret_here
LEVERAGE=50
PROFIT_MODE=conservative

# Required for the dashboard/API to be reachable from the host
# (the trade-history API binds 127.0.0.1 by default)
TRADE_API_HOST=0.0.0.0
```

---

## 🚀 **Advanced Usage**

### **Manual Container Management**

#### **Build ELVIS Image**
The setup script tries `Dockerfile.minimal` first (fewer network dependencies) and falls back to `Dockerfile.simple`:
```bash
container build -f Dockerfile.minimal -t elvis-bot:latest .
# or, with full dependencies:
container build -f Dockerfile.simple -t elvis-bot:latest .
```

#### **Create Network**
```bash
container network create elvis-network
```

#### **Start PostgreSQL**
```bash
container run -d \
    --name elvis-postgres \
    --network elvis-network \
    -p 5432:5432 \
    -e POSTGRES_DB=elvis_trading \
    -e POSTGRES_USER=elvis_user \
    -e POSTGRES_PASSWORD=elvis_password \
    postgres:15-alpine
```

#### **Start ELVIS Bot**
```bash
container run -d \
    --name elvis-bot \
    --network elvis-network \
    -p 5050:5050 \
    -p 8000:8000 \
    -e TRADING_MODE=paper \
    -e POSTGRES_HOST=elvis-postgres \
    -v "$(pwd)/logs:/app/logs" \
    elvis-bot:latest
```

### **Database Operations**
```bash
# Connect to PostgreSQL
container exec -it elvis-postgres psql -U elvis_user -d elvis_trading

# View account balances
container exec elvis-postgres psql -U elvis_user -d elvis_trading -c \
    "SELECT * FROM np.account_balances;"

# View recent trades
container exec elvis-postgres psql -U elvis_user -d elvis_trading -c \
    "SELECT * FROM np.trades ORDER BY timestamp DESC LIMIT 10;"

# View open positions
container exec elvis-postgres psql -U elvis_user -d elvis_trading -c \
    "SELECT * FROM np.open_positions;"
```

### **Trading Operations**
```bash
# Reset paper trading balances
container exec elvis-bot python reset_paper_trading.py

# Check current balances
container exec elvis-bot python check_paper_balances.py

# Test Bonenkamp strategy
container exec elvis-bot python tests/test_bonenkamp_strategy.py

# View position display
container exec elvis-bot python tests/test_positions_display.py
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Container Won't Start**
```bash
# Check container status
container inspect elvis-bot

# View startup logs
container logs elvis-bot

# Remove and recreate
container stop elvis-bot
container rm elvis-bot
./scripts/apple_container_native.sh start
```

#### **Database Connection Error**
```bash
# Check PostgreSQL status
container logs elvis-postgres

# Test connection manually
container exec elvis-postgres pg_isready -U elvis_user

# Restart database
container stop elvis-postgres
container rm elvis-postgres
./scripts/apple_container_native.sh start
```

#### **Port Conflicts**
```bash
# Check what's using ports
lsof -i :5050 -i :3000 -i :9090 -i :5432 -i :6379

# Stop conflicting services
sudo lsof -ti:5050 | xargs kill -9
```

#### **Network Issues**
```bash
# List networks
container network list

# Recreate network
container network rm elvis-network
container network create elvis-network
```

### **Performance Issues**
```bash
# Inspect container details (JSON output)
container inspect elvis-bot

# Restart with more resources (if container supports it)
container stop elvis-bot
container rm elvis-bot
# Edit script to add resource limits and restart
```

---

## 📈 **Trading Features**

### **Paper Trading**
- **Initial Balances**: $1000 USDT + $1000 BNB
- **Reset Command**: `./scripts/apple_container_native.sh reset`
- **Balance Check**: `container exec elvis-bot python check_paper_balances.py`

### **Bonenkamp HFT Strategy**
- **Frequency**: 5-minute intervals
- **Research-Based**: Academic paper implementation
- **Features**: 9 financial + 2 social indicators
- **Target Performance**: 14.9% annual return

### **Real-time Monitoring**
- **Dashboard**: http://localhost:5050
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

---

## 🔐 **Security**

### **Container Security**
- Non-root user execution in containers
- Network isolation between services
- Read-only configuration files
- Environment variable secrets

### **Apple Container Benefits**
- Native macOS security integration
- Sandboxed container execution
- System-level resource controls
- Secure container networking

---

## 🎯 **Next Steps**

1. **Setup**: Run `./scripts/apple_container_native.sh setup`
2. **Start**: Run `./scripts/apple_container_native.sh start`
3. **Monitor**: Open http://localhost:5050
4. **Customize**: Edit `.env` file for your preferences

---

## ✅ **Ready to Trade with Apple Containers!**

ELVIS is now optimized for Apple's native container system, providing:

- 🍎 **Native macOS Integration**
- ⚡ **Better Performance** 
- 🔒 **Enhanced Security**
- 💰 **Paper Trading** with $1000 USDT + $1000 BNB
- 📈 **Professional Trading Strategies**

### **Happy Trading!** 🚀📊