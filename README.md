# ELVIS Trading System

![ELVIS Logo](images/elvis.png)

ELVIS (Enhanced Leveraged Virtual Investment System) is a comprehensive trading system that combines machine learning, risk management, and advanced analytics for cryptocurrency trading.

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

```
elvis-trading/
├── trading/
│   ├── models/
│   │   ├── transformer_models.py
│   │   ├── rl_agents.py
│   │   └── explainable_ai.py
│   ├── risk_management/
│   │   └── risk_manager.py
│   ├── data/
│   │   └── data_processor.py
│   ├── testing/
│   │   └── strategy_validator.py
│   ├── config/
│   │   ├── model_config.yaml
│   │   ├── risk_config.yaml
│   │   ├── data_config.yaml
│   │   └── validation_config.yaml
│   └── scripts/
│       ├── train_models.py
│       └── validate_strategy.py
├── examples/
│   └── simple_strategy.py
├── setup.py
└── README.md
```

## Dependencies

- Python >= 3.10
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- rich
- pyyaml
- ccxt
- ta

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
