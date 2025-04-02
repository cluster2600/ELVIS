# ELVIS: Enhanced Leveraged Virtual Investment System

![ELVIS Logo](images/elvis.png)

## Overview  
ELVIS integrates **Deep Reinforcement Learning (DRL)** and **high-frequency trading (HFT)** for Binance Futures, targeting **BTC/USDT**. It tackles overfitting in financial RL and employs a **Random Forest model** with technical indicators (**SMA, RSI, MACD, Bollinger Bands**), enriched by insights on combining financial and social features. **Tested across multiple cryptocurrencies and market crashes**, it aims to surpass traditional strategies.  

> **⚠ WARNING: NON-PRODUCTION MODE ONLY ⚠**  
> ELVIS is currently configured to run in non-production mode by default. Live trading is disabled for safety.  
> This project is for educational purposes only and is not production-ready without extensive validation.  
> Leveraged trading carries high risk—use simulation or Binance Testnet first.  
> To enable live trading (at your own risk), set `PRODUCTION_MODE: True` in `config/config.py`.

## Features  
- **Multiple Trading Strategies**:
  - **Technical Strategy**: Uses RSI, MACD, DX, and OBV for trend-based trading
  - **Mean Reversion Strategy**: Leverages Bollinger Bands and RSI for mean reversion trading
  - **Trend Following Strategy**: Employs Moving Averages and ADX for trend identification and following
  
- **Advanced Model Types**:
  - **Random Forest Model**: Uses TensorFlow Decision Forests for robust prediction
  - **Neural Network Model**: Implements LSTM networks for time series forecasting
  - **Ensemble Model**: Combines multiple models for improved prediction accuracy
  
- **Comprehensive Risk Management**:
  - **Position Sizing**: Dynamic position sizing based on volatility and available capital
  - **Leverage Control**: Adjusts leverage based on trend strength and market conditions
  - **Stop Loss & Take Profit**: Calculates optimal levels using ATR and price action
  - **Trade Limits**: Enforces daily trade limits, profit targets, and loss thresholds
  
- **Performance Monitoring & Reporting**:
  - **Trade Tracking**: Records all trades with timestamps and performance metrics
  - **Metrics Calculation**: Computes win rate, profit factor, Sharpe ratio, and drawdown
  - **Visualization**: Generates equity curves, daily returns, and win/loss distributions
  - **HTML Reports**: Creates comprehensive performance reports with metrics and charts
  
- **Robust Testing Framework**:
  - **Unit Tests**: Comprehensive test coverage for all components
  - **Integration Tests**: End-to-end testing of the entire trading system
  - **Mocking**: Simulates exchange API for reliable testing
  
- **Core Infrastructure**:
  - **Binance Futures Integration**: Uses `ccxt` for trading with proper error handling
  - **Technical Indicators**: Calculates SMA, RSI, MACD, Bollinger Bands, ATR, ADX, and more
  - **Data Processing**: Efficient data downloading, cleaning, and indicator calculation
  - **Telegram Notifications**: Real-time alerts for trades, errors, and system status
  - **Data Caching**: Reduces API calls through intelligent caching
  - **Modular Architecture**: Clear separation of concerns with standardized interfaces

## Papers  
- **Deep Reinforcement Learning for Cryptocurrency Trading** by Berend Jelmer Dirk Gort et al.  
- **"High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features"** by Annelotte Bonenkamp (Bachelor Thesis, University of Amsterdam, June 2021).  

## Prerequisites  
- **Python 3.10** (tested with virtual environment `venv310`).  
- **Modules**: `ccxt`, `numpy`, `pandas`, `joblib`, `binance`, `talib`, `requests`, `yfinance` (optional), `telebot`, `websocket-client`, `ta`, `python-dotenv`, `torch`, `optuna`, plus `ElegantRL` dependencies.  
- **Binance Futures API access** (`API key` and `secret`).  
- **Configuration**: `.env` file with:  
  ```plaintext
  BINANCE_API_KEY=your_binance_api_key  
  BINANCE_API_SECRET=your_binance_api_secret  
  TELEGRAM_TOKEN=your_telegram_token  
  TELEGRAM_CHAT_ID=your_telegram_chat_id  
  ```

## Installation  
### Clone Repository:  
```bash
git clone https://github.com/cluster2600/ELVIS.git && cd ELVIS  
```

### Virtual Environment:  
#### Linux/macOS:  
```bash
python3.10 -m venv venv310 && source venv310/bin/activate  
```
#### Windows:  
```bash
python -m venv venv310 && venv310\Scripts\activate  
```

### Install Dependencies:  
```bash
pip install -r requirements.txt  
```
Set up `.env` with credentials (see Configuration below).  

## How to Use  
### Configuration  
Edit `config/config.py` for:  
- API credentials (loaded from `.env`)
- Trading parameters (symbol, timeframe, leverage, etc.)
- Technical indicators
- Backtesting configuration
- File paths
- Logging settings

### Folder Structure  
- `config/`: Configuration files for all settings.
- `core/`: Core functionality modules.
  - `data/`: Data handling modules.
  - `models/`: Model definitions.
  - `metrics/`: Performance metrics.
- `trading/`: Trading execution modules.
  - `strategies/`: Trading strategies.
  - `execution/`: Order execution.
  - `risk/`: Risk management.
- `utils/`: Utility functions.
- `data/`: Training/validation data.
- `logs/`: Trading and backtesting logs.

### Running the Bot  
To run the bot, use the provided script:
```bash
./run_elvis.sh --mode paper --symbol BTCUSDT --timeframe 1h --leverage 75
```

To run the bot with the console dashboard enabled:
```bash
./run_dashboard.sh
```

Available modes:
- `live`: Live trading with real money
- `paper`: Paper trading (simulated)
- `backtest`: Backtesting with historical data

### Console Dashboard
ELVIS includes a real-time console dashboard for monitoring trading performance directly in your terminal:

- **Portfolio Information**: Track your portfolio value, position size, and unrealized PnL
- **Performance Metrics**: View key metrics like win rate, profit factor, and Sharpe ratio
- **Recent Trades**: See your most recent trades with PnL information
- **Strategy Signals**: Monitor buy/sell signals from your active strategy
- **Real-time Candlestick Charts**: View live candlestick data with price movement visualization
- **Market Statistics**: Monitor daily high/low, volume, and market sentiment
- **System Statistics**: Track CPU/memory usage, uptime, and API calls

The console dashboard features:
- **Multiple Views**: Standard, Detailed, and Candlestick Chart views
- **Real-time WebSocket Data**: Direct connection to Binance for live price updates
- **Color-coded Information**: Green for positive values, red for negative
- **Real-time Updates**: Instant updates as trades are executed and prices change
- **Compact Display**: Fits in a standard terminal window
- **Low Resource Usage**: Efficient compared to web-based dashboards
- **Error Handling**: Robust error handling for terminal resizing and display issues

To use the console dashboard:
1. Run the bot with the dashboard enabled using `./run_dashboard.sh`
2. The dashboard will appear directly in your terminal
3. Use number keys to switch between views:
   - `1`: Standard view (portfolio, metrics, trades, signals)
   - `2`: Detailed view (portfolio, market stats, system stats, candle info)
   - `3`: Candlestick Chart view (real-time price chart)
4. Press 'q' to quit the dashboard and stop the bot

**Note**: The dashboard requires a terminal with support for Unicode box-drawing characters and a minimum size of 50x10 characters. For best results, use a terminal with a dark background.

```
┌─────────────────────── ELVIS Console Dashboard ───────────────────────┐
│                                                                       │
│ Portfolio Information                                                 │
│   Portfolio Value: $1,250.45                                          │
│   Position Size: 0.00125000 BTC                                       │
│   Entry Price: $75,420.50                                             │
│   Current Price: $75,655.24                                           │
│                                                                       │
│ Performance Metrics                                                   │
│   Total Trades: 42          Win Rate: 68.5%        Profit Factor: 2.34│
│   Winning Trades: 28        Losing Trades: 14      Sharpe Ratio: 1.87 │
│                                                                       │
│ Recent Trades                                                         │
│   Time     Symbol   Side    Price       Quantity        PnL           │
│   12:30:45 BTCUSDT  BUY     $75,420.50  0.00250000      +$58.69       │
│   11:15:22 BTCUSDT  SELL    $75,380.75  0.00250000      -$24.75       │
│   09:45:10 BTCUSDT  BUY     $75,310.25  0.00300000      +$103.50      │
│                                                                       │
│ Strategy Signals                                                      │
│   EMA-RSI Strategy: BUY                                               │
│   Technical Strategy: HOLD                                            │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Recent Updates  
- **Project Renaming**: Renamed to ELVIS (Enhanced Leveraged Virtual Investment System).
- **Modular Architecture**: Implemented a new architecture with clear separation of concerns.
- **Centralized Configuration**: Created a configuration module for all settings.
- **Improved Logging**: Enhanced logging system with colored console output and file rotation.
- **Standardized Interfaces**: Defined base interfaces for models, processors, strategies, and execution.
- **Enhanced Metrics**: Created comprehensive metrics utilities for performance analysis.
- **Command-line Interface**: Added a main entry point with command-line argument parsing.
- **Multiple Trading Strategies**: 
  - **Technical Strategy**: Uses RSI, MACD, DX, and OBV indicators
  - **Mean Reversion Strategy**: Uses Bollinger Bands and RSI
  - **Trend Following Strategy**: Uses Moving Averages and ADX
  - **EMA-RSI Strategy**: Uses EMA crossovers and RSI
  - **Sentiment Strategy**: Incorporates news and social media sentiment
  - **Grid Strategy**: Implements dynamic grid trading based on volatility
- **Advanced Model Types**: 
  - **Random Forest Model**: Using TensorFlow Decision Forests
  - **Neural Network Model**: Using LSTM and Dense layers
  - **Ensemble Model**: Combining multiple models
  - **Transformer Model**: Using self-attention mechanism
  - **Reinforcement Learning Model**: Using PPO algorithm
- **Advanced Risk Management**:
  - **Risk Manager**: Basic risk management with position sizing and trade limits
  - **Advanced Risk Manager**: Implements Kelly Criterion, circuit breakers, and drawdown protection
- **Performance Monitoring**: 
  - **Trade Tracking**: Records all trades with timestamps and metrics
  - **Performance Metrics**: Calculates win rate, profit factor, Sharpe ratio
  - **Visualization**: Generates equity curves, daily returns, win/loss distributions
  - **HTML Reports**: Creates comprehensive performance reports
- **Real-time Dashboard**: Displays performance metrics and trading signals
- **Monte Carlo Simulation**: Tests strategy robustness through simulations
- **Mock Data Generation**: Creates realistic market data for testing when API calls fail
- **Comprehensive Testing**: Unit tests and integration tests for all components

## Future Improvements
See [Future Improvements](docs/future_improvements.md) for a detailed roadmap of planned enhancements, including:
- Advanced trading strategies (sentiment analysis, grid trading)
- Enhanced machine learning models (transformers, reinforcement learning)
- Infrastructure improvements (real-time dashboard, distributed computing)
- Risk management enhancements
- Data enhancements
- And more...

## Risk Disclaimer  
This bot is experimental and not financial advice. Use at your own risk. Consider using Binance Testnet before live trading.
