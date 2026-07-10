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
TRADING_MODE=paper  # paper -> Binance testnet endpoint; anything else -> live api.binance.com
```

`TRADING_MODE` (read in `trading/config/api_config.py`) only selects which
Binance REST endpoint the bot talks to: `paper` points `API_CONFIG` at
`https://testnet.binance.vision` (using `TESTNET_API_SPOT_KEY` /
`TESTNET_API_SPOT_SECRET`), any other value points it at
`https://api.binance.com`.

The bot's own paper/live behaviour is chosen separately by the `--mode`
argument of `main.py` (`--mode paper|live`, default `paper`), passed through to
`main(mode=...)`. For a normal paper-trading session leave both at their paper
defaults.

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
- Trades the primary pairs concurrently (BTCUSDT + BNBUSDT), capped at
  `MAX_CONCURRENT_PAIRS = 2` (`config/config.py`, `SYMBOLS_CONFIG`)
- Tracks individual asset balances in `np.account_balances`
- Support for USDT and BNB trading pairs

### BNB Fee Optimization
Controlled by the BNB flags in `config/config.py` and implemented in
`trading/execution/enhanced_binance_executor.py`:
- `ENABLE_BNB_FEES` — pay trading fees in BNB for a discount
  (10% on futures, 25% on spot)
- `AUTO_BUY_BNB` / `MIN_BNB_BALANCE` — automatically top up BNB
  (`buy_bnb_for_fees()`) when the balance drops below the minimum
- `MAX_BNB_BUY_PERCENT`, `BNB_REBALANCE_THRESHOLD` — cap and threshold for the
  auto-buy / rebalance behaviour

### Performance Tracking
- Real-time P&L calculation
- Trade history and statistics
- Portfolio performance metrics

## Trading Pairs

Configured in `config/config.py` under `SYMBOLS_CONFIG`.

### Primary Pairs (`PRIMARY_SYMBOLS`)
- BTCUSDT (Bitcoin trading)
- BNBUSDT (BNB trading)

### Secondary Pairs (`SECONDARY_SYMBOLS`, optional)
- ETHUSDT (Ethereum trading)
- ADAUSDT (Cardano trading)

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
   # main.py already defaults to --mode paper; TRADING_MODE=paper keeps
   # REST calls on the Binance testnet endpoint.
   export TRADING_MODE=paper

   # Start the bot (equivalent to: python main.py --mode paper)
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

### Session resets table

```sql
CREATE TABLE np.trading_session_resets (
    id SERIAL PRIMARY KEY,
    reset_timestamp TIMESTAMP DEFAULT NOW(),
    reason TEXT
);
```

**How it works.** When the bot is stopped, `reset_trading_session()`
(in `utils/paper_trade_db.py`) inserts a row into
`np.trading_session_resets` with the current timestamp. The dashboards
and trade-history APIs (`native_console_dashboard.py`,
`utils/console_dashboard.py`, `trading/utils/trade_history_api.py`,
`trading/execution/binance_executor.py`) then read the most recent
`reset_timestamp` and only count trades *after* it, so realized P&L
starts fresh for the next session while all historical trades stay in
`np.trades`.

**How to use.** The table is created automatically by `init_db()` and
`init_db_with_balances()` (`CREATE TABLE IF NOT EXISTS`), so it exists on
fresh databases before the first reset — no manual migration is needed.
On a database created before this table existed, simply run either init
function (or `python reset_paper_trading.py`, which calls
`init_db_with_balances()`) once to create it. To start a fresh P&L
session programmatically:

```python
from utils.paper_trade_db import reset_trading_session
reset_trading_session()  # inserts a new reset marker; historical trades kept
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