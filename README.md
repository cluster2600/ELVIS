# ELVIS: Enhanced Leveraged Virtual Investment System

![ELVIS Logo](images/elvis.png)

## Overview
ELVIS (Enhanced Leveraged Virtual Investment System) is a modular framework for developing and deploying cryptocurrency trading bots on Binance Futures, specifically targeting BTC/USDT. It integrates various trading strategies, machine learning models (including Random Forest, Neural Networks, Transformers, and Reinforcement Learning), risk management techniques, and performance monitoring tools.

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

## Configuration

The system uses YAML configuration files for different components:

- `trading/config/model_config.yaml`: Machine learning model settings
- `trading/config/risk_config.yaml`: Risk management parameters
- `trading/config/data_config.yaml`: Data processing settings
- `trading/config/validation_config.yaml`: Strategy validation parameters

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

## Project Structure

The ELVIS project is structured into several key directories and files. Below is an overview of the main components and their interactions.

### Main Components:

1. **`main.py`**: The entry point of the application. It initializes and runs the appropriate trading bot based on command-line arguments.
2. **`utils`**: Contains utility functions and classes.
   - `console_dashboard.py`: Provides a console-based dashboard for monitoring trading activity.
   - `trading_dashboard.py`: Provides a trading dashboard with real-time data and performance metrics.
3. **`config`**: Contains configuration settings for the application.
4. **`trading`**: Contains modules related to trading strategies and execution.
   - `strategies`: Contains various trading strategies.
     - `base_strategy.py`: Defines a base class for trading strategies.
     - `technical_strategy.py`, `mean_reversion_strategy.py`, `trend_following_strategy.py`, `ema_rsi_strategy.py`, `ensemble_strategy.py`: Implement specific trading strategies.
   - `execution`: Contains execution strategies.
     - `base_executor.py`: Defines a base class for execution strategies.
     - `binance_executor.py`: Implements a Binance execution strategy.
5. **`core`**: Contains core functionality.
   - `data/processors`: Contains data processing modules.
     - `base_processor.py`: Defines a base class for data processors.
     - `binance_processor.py`: Implements a Binance data processor.
   - `models`: Contains machine learning models.
     - `base_model.py`: Defines a base class for machine learning models.
     - `ensemble_model.py`: Implements an ensemble model.

### Dependency Map:

- `main.py`:
  - Imports from `utils`: `setup_logger`, `console_dashboard`, `trading_dashboard`
  - Imports from `config`: Configuration settings
  - Imports from `trading.strategies`: Various trading strategies
  - Initializes and runs the trading bot based on command-line arguments

## Dependencies

### Core Dependencies
- `Python >= 3.10`
- `numpy`: For numerical computations
- `pandas`: For data manipulation and analysis
- `scipy`: For scientific computing and statistical analysis
- `scikit-learn`: For machine learning algorithms
- `matplotlib` and `seaborn`: For data visualization
- `tqdm`: For progress bars
- `rich`: For rich text and beautiful formatting in the terminal
- `pyyaml`: For parsing YAML configuration files
- `ccxt`: For interacting with cryptocurrency exchanges
- `ta`: For technical analysis indicators

### Database Dependencies
- `psycopg2`: For PostgreSQL database interaction

### System Monitoring
- `psutil`: For system monitoring (CPU, memory usage)

### Development Dependencies
- List any additional dependencies required for development, testing, or deployment

### Environment Variables
The project uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_PORT`
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`

### Installation
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
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

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
