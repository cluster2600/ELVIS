# ELVIS: Enhanced Leveraged Virtual Investment System

![ELVIS Logo](images/elvis.png)

## Overview
ELVIS (Enhanced Leveraged Virtual Investment System) is a modular framework for developing and deploying cryptocurrency trading bots on Binance Futures, specifically targeting BTC/USDT. It integrates various trading strategies, machine learning models (including Random Forest, Neural Networks, Transformers, and Reinforcement Learning), risk management techniques, and performance monitoring tools.

> **⚠ WARNING: NON-PRODUCTION MODE ONLY ⚠**  
> ELVIS is currently configured to run in non-production mode by default. Live trading is disabled for safety.  
> This project is for educational purposes only and is not production-ready without extensive validation.  
> Leveraged trading carries high risk—use simulation or Binance Testnet first.  
> To enable live trading (at your own risk), set `PRODUCTION_MODE: True` in `config/config.py`.

## Features  
- **Multiple Trading Strategies**:
- **Multiple Trading Strategies**: Technical, Mean Reversion, Trend Following, EMA/RSI, Sentiment, Grid.
- **Advanced Model Types**: Random Forest (TFDF), Neural Network (LSTM), Transformer, Reinforcement Learning (PPO), Ensemble.
- **Comprehensive Risk Management**: Basic and Advanced (Kelly Criterion, Drawdown Protection) options.
- **Performance Monitoring**: Tracks trades, calculates key metrics (Sharpe, PnL, Win Rate, etc.), generates plots and HTML reports.
- **Real-time Console Dashboard**: For monitoring paper trading sessions.
- **Modular Architecture**: Core components (data, models, metrics), Trading components (strategies, execution, risk), Utilities.
- **Testing**: Includes unit and integration tests using pytest.
- **Configuration**: Centralized settings in `config/config.py` and API keys via `.env`.

## Papers  
- **Deep Reinforcement Learning for Cryptocurrency Trading** by Berend Jelmer Dirk Gort et al.  
- **"High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features"** by Annelotte Bonenkamp (Bachelor Thesis, University of Amsterdam, June 2021).  

## Prerequisites  
- **Python 3.10+** (developed with 3.11).
- **Dependencies**: Listed in `requirements.txt`. Install using `pip install -r requirements.txt`.
- **Binance API Keys**: Required for live trading and potentially for data fetching in paper/backtest modes if not using mock data.
- **Telegram Bot Token/Chat ID**: Optional, for notifications.
- **Configuration**: Create a `.env` file in the root directory with your API keys and Telegram details:
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

## How to Use

### Configuration
1.  **Create `.env` file**: Add your API keys and Telegram info (see Prerequisites).
2.  **Edit `config/config.py`**: Adjust trading parameters (symbol, timeframe, leverage, strategy defaults), file paths, logging level, and **importantly**, set `PRODUCTION_MODE` to `True` **only** if you intend to live trade (default is `False` for safety).

### Running the Bot
Use `main.py` as the entry point. Select the mode and strategy via command-line arguments:

```bash
python main.py --mode <mode> --strategy <strategy_name> [other_options]
```

**Available Modes (`--mode`):**
-   `live`: Live trading on Binance Futures (**Requires `PRODUCTION_MODE = True` in config and valid API keys**). High risk!
-   `paper`: Simulated trading using real-time data via a console dashboard. Ideal for testing strategies without risk.
-   `backtest`: Backtesting strategies using historical data (Implementation details might vary).

**Available Strategies (`--strategy`):**
-   `technical` (Default)
-   `mean_reversion`
-   `trend_following`
-   `ema_rsi`
-   `sentiment` (May require additional setup/data sources)
-   `grid`

**Example (Paper Trading with Console Dashboard):**
```bash
python main.py --mode paper --strategy ema_rsi --symbol BTCUSDT --timeframe 1h
```
This will start the paper trading bot using the EMA/RSI strategy and display the real-time console dashboard.

### Console Dashboard (Paper Trading Mode)
When running in `paper` mode, ELVIS displays a real-time console dashboard:

- **Real-time Updates**: Shows portfolio value, position, PnL, metrics, recent trades, strategy signals, market data, and system stats.
- **Live Candlestick Chart**: Visualizes price action.
- **Interactive Views**: Use number keys (`1`, `2`, `3`) to switch between Standard, Detailed, and Chart views.
- **Quit**: Press `q` to exit the dashboard and stop the paper trading session.

**Note**: The dashboard requires a terminal supporting Unicode and color, with a minimum size (e.g., 80x24). It uses a WebSocket connection via `utils/price_fetcher.py` for live data.

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
*(Summary of features moved to the main Features section)*

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
