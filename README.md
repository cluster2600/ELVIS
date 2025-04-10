# ELVIS: Enhanced Leveraged Virtual Investment System

![ELVIS Logo](images/elvis.png)

## Overview
ELVIS (Enhanced Leveraged Virtual Investment System) is a modular framework for developing and deploying cryptocurrency trading bots on Binance Futures, specifically targeting BTC/USDT. It integrates various trading strategies, machine learning models (including Random Forest, Neural Networks, Transformers, and Reinforcement Learning), risk management techniques, and performance monitoring tools.

## Documentation
- [Training Documentation](docs/training.md) - Comprehensive guide to training RL agents
- [Random Forest Documentation](docs/random_forest.md) - Guide to the Random Forest model implementation
- [CHANGELOG.md](CHANGELOG.md) - Version history and changes
- [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) - Planned enhancements

## Sources

This project is inspired by and builds upon several academic papers and research:

- **Deep Reinforcement Learning for Cryptocurrency Trading** by Berend Jelmer Dirk Gort et al.
- **High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features** by Annelotte Bonenkamp (Bachelor Thesis, University of Amsterdam, June 2021)
- **Attention Is All You Need** by Vaswani et al. (Transformer architecture)
- **Proximal Policy Optimization Algorithms** by Schulman et al. (PPO implementation)
- **A Comprehensive Guide to Machine Learning for Trading** by Marcos Lopez de Prado

> **⚠ WARNING: NON-PRODUCTION MODE ONLY ⚠**  
> ELVIS is currently configured to run in non-production mode by default. Live trading is disabled for safety.  
> This project is for educational purposes only and is not production-ready without extensive validation.  
> Leveraged trading carries high risk—use simulation or Binance Testnet first.  
> To enable live trading (at your own risk), set `PRODUCTION_MODE: True` in `config/config.py`.

## Features

- **Machine Learning Models**
  - Transformer-based time series forecasting
  - Reinforcement learning agents
  - Explainable AI components
  - Automated feature engineering

- **Risk Management**
  - Advanced position sizing using Kelly Criterion
  - Dynamic risk allocation based on market regimes
  - Drawdown protection with circuit breakers
  - Correlation analysis and portfolio optimization

- **Data Processing**
  - On-chain data integration
  - Order book analysis
  - Funding rate monitoring
  - Technical indicator calculation
  - Market regime detection

- **Strategy Validation**
  - Monte Carlo simulations
  - Walk-forward analysis
  - Statistical validation
  - Stress testing
  - Performance metrics

- **Monitoring & Visualization**
  - Real-time Prometheus metrics
  - Grafana dashboards for trading analytics
  - System resource monitoring
  - Order book depth visualization
  - Performance tracking and alerts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/elvis-trading.git
cd elvis-trading
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Set up monitoring (optional):
```bash
# Install Prometheus
brew install prometheus  # macOS
# or
sudo apt-get install prometheus  # Ubuntu/Debian

# Install Grafana
brew install grafana  # macOS
# or
sudo apt-get install grafana  # Ubuntu/Debian
```

## Configuration

The system uses YAML configuration files for different components:

- `trading/config/model_config.yaml`: Machine learning model settings
- `trading/config/risk_config.yaml`: Risk management parameters
- `trading/config/data_config.yaml`: Data processing settings
- `trading/config/validation_config.yaml`: Strategy validation parameters
- `grafana/provisioning/datasources/datasources.yml`: Prometheus data source configuration
- `grafana/provisioning/dashboards/dashboards.yml`: Grafana dashboard provisioning

## Usage

### Training Models

```bash
./run_training.sh
```

This script will:
1. Activate the virtual environment
2. Install required packages
3. Run the model training pipeline
4. Save trained models and optimized parameters

### Validating Strategies

```bash
python trading/scripts/validate_strategy.py \
    --strategy examples/simple_strategy.py \
    --data your_data.csv \
    --mode all
```

The validation script supports:
- Monte Carlo simulations
- Walk-forward analysis
- Statistical tests
- Stress testing

### Monitoring with Grafana

ELVIS includes a comprehensive Grafana dashboard for real-time monitoring:

1. Start Prometheus:
```bash
prometheus --config.file=prometheus.yml
```

2. Start Grafana:
```bash
grafana-server
```

3. Access the dashboard at `http://localhost:3000` (default Grafana port)

The dashboard includes:
- Real-time BTC/USDT price tracking
- Portfolio value and performance metrics
- Technical indicators (EMA, RSI, MACD, Bollinger Bands)
- Order book depth and volume analysis
- System resource monitoring
- Pending orders tracking

### Example Strategy

A simple moving average crossover strategy is provided in `examples/simple_strategy.py`:

```python
def strategy(data: pd.DataFrame, initial_capital: float = 100000, params: dict = None) -> dict:
    # Calculate moving averages
    data['short_ma'] = data['close'].rolling(window=params['short_window']).mean()
    data['long_ma'] = data['close'].rolling(window=params['long_window']).mean()
    
    # Generate signals and calculate returns
    ...
    
    return {
        'returns': returns,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }
```

## Project Structure and File Relationships

### Core Components

#### Main Application Files
- `main.py`: The main entry point of the application, orchestrating all components
- `run_elvis.sh`: Shell script to run the main application
- `elvis_testnet.sh`: Shell script to run the application in testnet mode

#### Core Module (`/core`)
- `__init__.py`: Package initialization
- `/metrics`: Performance and trading metrics calculations
- `/models`: ML model implementations and management
- `/data`: Data handling and processing
- `/validation`: Input validation and data verification

#### Utils Module (`/utils`)
- `console_dashboard.py`: Terminal-based UI for monitoring
- `trading_dashboard.py`: Web-based trading interface
- `message_queue.py`: Inter-process communication
- `price_fetcher.py`: Real-time price data retrieval
- `logging_utils.py`: Logging configuration and utilities
- `notification_utils.py`: Alert and notification system

#### Services Module (`/services`)
- `/telegram`: Telegram bot integration
- `/ml_engine`: Machine learning model management
- `/risk_management`: Risk assessment and management
- `/data_pipeline`: Data processing and ETL
- `/strategy_engine`: Trading strategy implementation

#### Monitoring Module (`/grafana`)
- `/provisioning/datasources`: Prometheus data source configuration
- `/provisioning/dashboards`: Grafana dashboard provisioning
- `/dashboards`: JSON dashboard definitions

### Testing and Development
- `test_elvis.py`: Main test suite
- `test_console_dashboard.py`: Dashboard testing
- `test_binance_api.py`: Binance API integration testing
- `test_symbols.py`: Symbol validation testing
- `test_env.py`: Environment configuration testing

### Model Training and Management
- `ensemble_models.py`: ML model ensemble management
- `create_coreml_model.py`: CoreML model conversion
- `training.py`: Model training orchestration
- `function_train_test.py`: Training/testing utilities
- `function_CPCV.py`: Cross-validation utilities
- `function_PBO.py`: Portfolio optimization
- `function_finance_metrics.py`: Financial metrics calculation

### Configuration and Setup
- `setup.py`: Package installation configuration
- `requirements.txt`: Python dependencies
- `.env`: Environment variables
- `setup_secure_config.sh`: Security configuration

### Documentation
- `/docs`: Additional documentation

### File Relationships
1. **Core Application Flow**:
   - `main.py` → `core/` → `services/` → `utils/`
   - Main orchestrates core components, which use services and utilities

2. **Trading Dashboard Flow**:
   - `utils/trading_dashboard.py` → `utils/price_fetcher.py` → `services/strategy_engine/`
   - Dashboard displays data from price fetcher and strategy engine

3. **ML Pipeline Flow**:
   - `services/ml_engine/` → `core/models/` → `ensemble_models.py`
   - ML engine uses core models and ensemble management

4. **Risk Management Flow**:
   - `services/risk_management/` → `utils/price_fetcher.py` → `function_finance_metrics.py`
   - Risk management uses price data and financial metrics

5. **Notification System Flow**:
   - `utils/notification_utils.py` → `services/telegram/`
   - Notifications are sent through Telegram service

6. **Monitoring Flow**:
   - `main.py` → Prometheus metrics → Grafana dashboards
   - Real-time metrics are collected and visualized

### Unconnected Components
1. **Standalone Testing Files**:
   - `test_symbols.py`
   - `test_binance_api.py`
   - `test_env.py`
   These are independent test files not integrated into the main application flow.

2. **Legacy Files**:
   - `your_bot_script.py.bak`
   - `1. Resume summary section`
   These files appear to be backups or documentation not actively used.

3. **Data Files**:
   - `export_trades.xlsx`
   - `test_results.json`
   These are output files generated by the system but not part of the codebase.

### Dependencies
- Python 3.8+
- Required packages listed in `requirements.txt`
- Binance API access
- Telegram Bot Token (for notifications)
- CoreML (for model deployment)
- Prometheus (for metrics collection)
- Grafana (for visualization)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors
- Inspired by various trading systems and research papers
- Built with ❤️ for the crypto trading community
