# ELVIS Trading Bot - Comprehensive Project Overview

![Project Image](./images/elvis.png)

## Table of Contents

- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
  - [Models](#models)
  - [Trading System](#trading-system)
  - [Data Processing](#data-processing)
  - [Training Pipeline](#training-pipeline)
  - [Utilities & Monitoring](#utilities--monitoring)
- [Configuration](#configuration)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## Introduction

The ELVIS Trading Bot is a sophisticated, modular algorithmic trading system that leverages machine learning models for automated cryptocurrency trading. The system integrates multiple ML architectures, real-time data processing, risk management, and execution modules to facilitate intelligent trading strategies with comprehensive monitoring and visualization capabilities.

---

## System Architecture

```mermaid
graph TB
    subgraph "Entry Points"
        Main[main.py]
        Training[training/train_models.py]
        Scripts[run_*.sh]
    end
    
    subgraph "Core Models"
        BaseModel[BaseModel Interface]
        RF[RandomForestModel]
        NN[NeuralNetworkModel]
        Trans[TransformerModel]
        Ensemble[EnsembleModel]
        RL[RL Agents]
    end
    
    subgraph "Trading System"
        BaseStrategy[BaseStrategy]
        EnsStrategy[EnsembleStrategy]
        BaseExecutor[BaseExecutor]
        BinanceExec[BinanceExecutor]
        RiskMgr[AdvancedRiskManager]
    end
    
    subgraph "Data Pipeline"
        BaseProcessor[BaseProcessor]
        BinanceProcessor[BinanceProcessor]
        PriceFetcher[PriceFetcher]
        DataDownloader[DataDownloader]
    end
    
    subgraph "Training Infrastructure"
        TrainPipeline[TrainingPipeline]
        ModelTrainer[ModelTrainer]
        Evaluator[Evaluator]
        ReplayBuffer[ReplayBuffer]
    end
    
    subgraph "Utilities & Monitoring"
        Dashboard[ConsoleDashboard]
        TelegramBot[TelegramNotifier]
        TradeAPI[TradeHistoryAPI]
        Monitoring[Monitoring]
        Grafana[Grafana Dashboards]
    end
    
    subgraph "Configuration"
        Config[config.py]
        ModelConfig[model_config.yaml]
        APIConfig[API Configuration]
    end
    
    Main --> EnsStrategy
    Main --> BinanceExec
    Main --> Dashboard
    Main --> RiskMgr
    Main --> TelegramBot
    Main --> TradeAPI
    
    Training --> TrainPipeline
    TrainPipeline --> ModelTrainer
    TrainPipeline --> Evaluator
    
    EnsStrategy --> RF
    EnsStrategy --> NN
    EnsStrategy --> Ensemble
    
    RF --|> BaseModel
    NN --|> BaseModel
    Trans --|> BaseModel
    Ensemble --|> BaseModel
    
    EnsStrategy --|> BaseStrategy
    BinanceExec --|> BaseExecutor
    BinanceProcessor --|> BaseProcessor
    
    EnsStrategy --> PriceFetcher
    EnsStrategy --> RiskMgr
    BinanceExec --> PriceFetcher
    
    TrainPipeline --> BinanceProcessor
    ModelTrainer --> RL
    
    Dashboard --> Monitoring
    Monitoring --> Grafana
    
    Config --> Main
    Config --> Training
    ModelConfig --> TrainPipeline
```

---

## Core Components

### Models

The system implements a hierarchical model architecture with a common interface:

```mermaid
classDiagram
    class BaseModel {
        <<abstract>>
        +train(X, y)
        +predict(X) ndarray
        +save(path)
        +load(path) BaseModel
        +get_params() Dict
        +set_params(**params)
    }
    
    class RandomForestModel {
        -model: tfdf.RandomForestModel
        -optuna_trial: Optional[Trial]
        +train(X, y)
        +predict(X) ndarray
        +cross_validate(X, y, cv) Dict
        +get_feature_importance() Dict
        +explain_predictions(X) Dict
        +push_cv_metrics_to_prometheus()
    }
    
    class NeuralNetworkModel {
        -model: tf.keras.Model
        -sequence_length: int
        +create_sequences(data) Tuple
        +train(X, y)
        +predict(X) ndarray
        +evaluate(X, y) Dict
        +get_feature_importance() Dict
    }
    
    class TransformerModel {
        -model: torch.nn.Module
        -d_model: int
        -nhead: int
        -num_layers: int
        +train(X, y)
        +predict(X) ndarray
        +save_model(path)
        +load_model(path)
        +get_attention_weights() ndarray
    }
    
    class EnsembleModel {
        -models: List[BaseModel]
        -weights: List[float]
        -voting_type: str
        +add_model(model, weight)
        +train(X, y)
        +predict(X) ndarray
        +get_feature_importance() Dict
    }
    
    RandomForestModel --|> BaseModel
    NeuralNetworkModel --|> BaseModel
    TransformerModel --|> BaseModel
    EnsembleModel --|> BaseModel
    
    EnsembleModel --> BaseModel : contains
```

### Trading System

The trading system follows a strategy pattern with pluggable execution backends:

```mermaid
classDiagram
    class BaseStrategy {
        <<abstract>>
        +generate_signals(data) Tuple[bool, bool]
        +calculate_position_size(data, price, capital) float
        +calculate_stop_loss(data, entry_price) float
        +calculate_take_profit(data, entry_price) float
    }
    
    class EnsembleStrategy {
        -ydf_model: RandomForestModel
        -coreml_model: NeuralNetworkModel
        -mlx_model: Optional[LLMModel]
        -executor: BaseExecutor
        -risk_manager: RiskManager
        +generate_signals(data) Tuple[bool, bool]
        +run()
        +_consensus_signal() bool
    }
    
    class BaseExecutor {
        <<abstract>>
        +initialize()
        +get_balance() Dict[str, float]
        +get_position(symbol) Dict
        +execute_buy(symbol, quantity, price) Dict
        +execute_sell(symbol, quantity, price) Dict
        +set_leverage(symbol, leverage)
    }
    
    class BinanceExecutor {
        -client: binance.Client
        -is_testnet: bool
        +initialize()
        +get_balance() Dict[str, float]
        +get_funding_rate(symbol) float
        +get_order_book(symbol) Dict
        +execute_buy(symbol, quantity, price) Dict
        +execute_sell(symbol, quantity, price) Dict
    }
    
    class AdvancedRiskManager {
        -max_position_size: float
        -max_daily_trades: int
        -max_drawdown: float
        +manage_risk(signal, current_position) bool
        +calculate_position_size(signal_strength) float
        +check_daily_limits() bool
    }
    
    EnsembleStrategy --|> BaseStrategy
    BinanceExecutor --|> BaseExecutor
    
    EnsembleStrategy --> BaseExecutor
    EnsembleStrategy --> AdvancedRiskManager
    EnsembleStrategy --> RandomForestModel
    EnsembleStrategy --> NeuralNetworkModel
```

### Data Processing

Data processing follows a pipeline pattern for modularity and extensibility:

```mermaid
classDiagram
    class BaseProcessor {
        <<abstract>>
        -data_source: str
        -start_date: str
        -end_date: str
        -time_interval: str
        +download_data(ticker_list) DataFrame
        +clean_data() DataFrame
        +add_technical_indicator(indicators) DataFrame
        +df_to_array(indicators, if_vix) tuple
        +run(tickers, indicators, if_vix) tuple
    }
    
    class BinanceProcessor {
        -client: binance.Client
        +download_data(ticker_list) DataFrame
        +clean_data() DataFrame
        +add_technical_indicator(indicators) DataFrame
        +calculate_rsi(data) Series
        +calculate_macd(data) DataFrame
        +calculate_bollinger_bands(data) DataFrame
    }
    
    class PriceFetcher {
        -api_config: APIConfig
        -prometheus_metrics: Dict
        +fetch_historical_data(symbol, interval, limit) DataFrame
        +fetch_current_price(symbol) float
        +calculate_technical_indicators(data) DataFrame
        +update_prometheus_metrics(data)
    }
    
    class DataDownloader {
        +download_binance_data(symbol, interval, start, end) DataFrame
        +save_to_csv(data, filename)
        +load_from_csv(filename) DataFrame
    }
    
    BinanceProcessor --|> BaseProcessor
    BinanceProcessor --> PriceFetcher
    PriceFetcher --> DataDownloader
```

### Training Pipeline

The training system supports multiple model types and distributed training:

```mermaid
flowchart TD
    Start([Training Start]) --> LoadConfig[Load Configuration]
    LoadConfig --> SetupLogging[Setup Logging & Monitoring]
    SetupLogging --> LoadData[Load Training Data]
    LoadData --> PrepareFeatures[Prepare Features & Targets]
    PrepareFeatures --> CreateLoaders[Create Data Loaders]
    
    CreateLoaders --> TrainModels{Train Models}
    TrainModels --> |ML Models| TrainML[Train ML Models]
    TrainModels --> |RL Agents| TrainRL[Train RL Agents]
    
    TrainML --> EvaluateML[Evaluate ML Models]
    TrainRL --> EvaluateRL[Evaluate RL Agents]
    
    EvaluateML --> ExplainML[Generate ML Explanations]
    EvaluateRL --> SkipExplain[Skip RL Explanations]
    
    ExplainML --> SaveModels[Save Models & Metrics]
    SkipExplain --> SaveModels
    
    SaveModels --> End([Training Complete])
    
    subgraph "Model Training"
        TrainML --> RF_Train[Random Forest]
        TrainML --> NN_Train[Neural Network]
        TrainML --> Trans_Train[Transformer]
        TrainML --> Ensemble_Train[Ensemble]
    end
    
    subgraph "RL Training"
        TrainRL --> DQN_Train[DQN Agent]
        TrainRL --> PPO_Train[PPO Agent]
        TrainRL --> A3C_Train[A3C Agent]
    end
```

### Utilities & Monitoring

The system includes comprehensive monitoring and utility components:

```mermaid
classDiagram
    class ConsoleDashboard {
        -strategy: EnsembleStrategy
        -risk_manager: RiskManager
        -running: bool
        +start()
        +stop()
        +_draw_frame()
        +_update_metrics()
        +_handle_input()
    }
    
    class TelegramNotifier {
        -bot_token: str
        -chat_id: str
        +send_message(message)
        +send_trade_alert(trade_info)
        +send_error_alert(error)
    }
    
    class TradeHistoryAPI {
        -app: Flask
        +get_trades() List[Dict]
        +get_performance_metrics() Dict
        +get_balance_history() List[Dict]
    }
    
    class Monitoring {
        -prometheus_client: PrometheusClient
        -grafana_config: Dict
        +push_metrics(metrics)
        +create_dashboard(config)
        +setup_alerts(rules)
    }
    
    class PerformanceMonitor {
        -metrics_history: List[Dict]
        +track_trade(trade_info)
        +calculate_sharpe_ratio() float
        +calculate_max_drawdown() float
        +generate_report() Dict
    }
    
    ConsoleDashboard --> PerformanceMonitor
    TelegramNotifier --> TradeHistoryAPI
    Monitoring --> PerformanceMonitor
```

---

## Core Models

### BaseModel Interface

Defines the abstract interface all models must implement, including methods for training, prediction, saving/loading, and parameter management.

### RandomForestModel

Implements a Random Forest classifier using TensorFlow Decision Forests. Supports training, evaluation, prediction, cross-validation with k-folds, and SHAP-based explainability. Includes robust error handling and logging.

### NeuralNetworkModel

A TensorFlow/Keras-based LSTM neural network model for time series forecasting. Supports sequence creation, training with early stopping, prediction, evaluation, and model persistence. Feature importance is approximated via sensitivity analysis.

### TransformerModel

Implements a transformer architecture for time series forecasting using PyTorch. Includes positional encoding, multi-head attention, and feed-forward layers. Supports training, evaluation, prediction, and saving/loading model state. Attention weights extraction is planned for interpretability.

### EnsembleModel

Combines multiple sub-models (Random Forest, Neural Network, etc.) using weighted soft or hard voting. Supports training orchestration, prediction aggregation, evaluation, feature importance aggregation, and configuration persistence.

---

## Training Pipeline

The training pipeline (`training/train_models.py`) manages the end-to-end process:

- Loads configuration and data.
- Prepares features and targets.
- Creates data loaders with time-series splits.
- Supports distributed training.
- Trains models with checkpointing and early stopping.
- Trains reinforcement learning agents.
- Evaluates models and saves metrics.
- Generates explanations using SHAP or LIME.
- Logs training progress and metrics.

---

## Data Processing

The `BaseProcessor` interface defines methods for downloading, cleaning, and feature engineering on market data. Implementations handle technical indicator calculation and data transformation for model consumption.

---

## Trading Strategies

### BaseStrategy

Abstract base class defining methods for signal generation, position sizing, stop loss, and take profit calculations.

### EnsembleStrategy

Combines predictions from multiple models including YDF Random Forest, CoreML Neural Network, and optionally MLX LLM. Generates consensus trading signals and calculates position sizes based on risk.

---

## Execution Modules

### BaseExecutor

Abstract interface for trading executors, defining methods for initialization, balance retrieval, order execution, and order management.

### BinanceExecutor

Concrete implementation interfacing with Binance API. Handles client initialization, balance queries, funding rates, order book retrieval, and order execution with error handling.

---

## Utilities

### PriceFetcher

Fetches historical and real-time Binance price data, calculates technical indicators (RSI, MACD, SMA, EMA), and updates Prometheus metrics for monitoring.

### ConsoleDashboard

Curses-based terminal UI displaying trading system metrics, system resource usage, and recent trades. Supports extensibility for multi-timeframe views and technical indicators.

### TrainingMonitor

Tracks training and validation metrics, supports early stopping, and displays progress during model training.

---

## Testing

Unit tests for the RandomForestModel validate training, prediction, evaluation metrics, feature importance, and cross-validation functionality, ensuring model robustness.

---

## Configuration

Configuration files in YAML and Python manage model parameters, training settings, data paths, and environment variables. The training pipeline reads these configurations to orchestrate the workflow.

---

## Monitoring and Metrics

Prometheus metrics integration allows pushing cross-validation metrics to a Pushgateway. The system tracks real-time price and indicator metrics, enabling observability and alerting.

---

## Future Improvements

- Enhanced visualization dashboards with multi-timeframe and technical indicator overlays.
- Advanced trading strategies with dynamic position sizing and regime detection.
- Expanded risk management including VaR and drawdown protection.
- Online and incremental learning capabilities.
- Improved model interpretability and explanation tools.
- Continuous integration of new data sources and market features.

---

## References

- `core/models/`
- `training/`
- `trading/strategies/`
- `trading/execution/`
- `utils/`
- `docs/`

### Documentation Files

- [Architecture Links Part 1](docs/architecture_links_part1.mmd)
- [Architecture Links](docs/architecture_links.mmd)
- [Bot Architecture Mermaid](docs/bot_architecture_mermaid.md)
- [Future Improvements](docs/future_improvements.md)
- [Random Forest Model Documentation](docs/random_forest.md)
- [Training Pipeline Documentation](docs/training.md)

---

This README will be maintained and expanded as the project evolves to provide clear guidance and documentation for developers and stakeholders.
