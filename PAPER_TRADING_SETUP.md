# Paper Trading Setup Documentation

## Overview

The ELVIS trading bot supports paper trading mode with multi-asset balances. In paper trading mode, the bot starts with **$1000 USDT** and **$1000 BNB** for a total portfolio value of **$2000**.

## Initial Balances

- **USDT**: $1,000.00 (for general cryptocurrency trading)
- **BNB**: $1,000.00 (for Binance ecosystem trading and fee optimization)
- **Total Portfolio**: $2,000.00

## Configuration

### Environment Variables
```bash
TRADING_MODE=paper  # Enable paper trading mode
```

### Database Configuration
The paper trading system uses PostgreSQL to track:
- Account balances for multiple assets
- Trade history
- Open positions
- Performance metrics

## Scripts

### Reset Paper Trading
```bash
python reset_paper_trading.py
```
This script:
- Clears all trade history
- Clears all open positions
- Resets USDT balance to $1000
- Resets BNB balance to $1000
- Initializes database structure

### Check Current Balances
```bash
python check_paper_balances.py
```
This script displays:
- Current balance for each asset
- Total portfolio value
- Trading status

## Features

### Multi-Asset Support
- Trade multiple cryptocurrency pairs simultaneously
- Track individual asset balances
- Support for USDT and BNB trading pairs

### BNB Fee Optimization
- Use BNB to pay trading fees for discounts
- Automatic BNB balance management
- Fee optimization for better profitability

### Performance Tracking
- Real-time P&L calculation
- Trade history and statistics
- Portfolio performance metrics

## Trading Pairs

### Primary Pairs
- BTCUSDT (Bitcoin trading)
- BNBUSDT (BNB trading)

### Secondary Pairs
- ETHUSDT (Ethereum trading)
- Other major cryptocurrencies

## Benefits of Starting with BNB

1. **Fee Discounts**: BNB provides trading fee discounts on Binance
2. **Diversification**: Start with exposure to both USDT and BNB
3. **Ecosystem Benefits**: Access to Binance ecosystem features
4. **Fee Optimization**: Automatic use of BNB for fee payments

## Usage

1. **Initialize Paper Trading**:
   ```bash
   python reset_paper_trading.py
   ```

2. **Check Status**:
   ```bash
   python check_paper_balances.py
   ```

3. **Start Trading**:
   ```bash
   # Set environment to paper mode
   export TRADING_MODE=paper
   
   # Start the bot
   python main.py
   ```

## Database Schema

### Account Balances Table
```sql
CREATE TABLE np.account_balances (
    id SERIAL PRIMARY KEY,
    asset TEXT UNIQUE NOT NULL,
    balance REAL NOT NULL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT NOW()
);
```

### Initial Data
```sql
INSERT INTO np.account_balances (asset, balance) VALUES 
('USDT', 1000.0),
('BNB', 1000.0);
```

## Monitoring

The paper trading system provides real-time monitoring of:
- Current balances
- Open positions
- Trade performance
- Fee costs
- Portfolio growth

## Reset Instructions

To start fresh paper trading:

1. Stop the trading bot if running
2. Run the reset script: `python reset_paper_trading.py`
3. Verify balances: `python check_paper_balances.py`
4. Restart the trading bot in paper mode

## Notes

- Paper trading uses simulated trades, no real money is involved
- All trading logic is identical to live trading
- Performance metrics accurately reflect trading strategy effectiveness
- Database persistence ensures trade history survives restarts