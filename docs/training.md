# ELVIS Trading System - Model Training Documentation

> ⚠️ **Partially outdated (audited 2026-07-02).** A large share of this document's concrete claims — file paths, class/param names, config files, and library choices (e.g. TFDF/Optuna/SHAP, `trading_config.yaml`) — no longer match the code. Treat the source under `core/`, `trading/`, and `training/` as the authority until this doc is rewritten.


## Overview

This document provides comprehensive documentation of the model training pipeline for the ELVIS trading system. It covers the architecture, components, data flow, training processes, evaluation methods, and configuration management for both traditional ML models and reinforcement learning agents.

---

## Training Architecture

```mermaid
graph TB
    subgraph "Training Entry Points"
        CLI[CLI Arguments]
        Config[Configuration Files]
        Scripts[Training Scripts]
    end
    
    subgraph "Training Pipeline"
        Pipeline[TrainingPipeline]
        Setup[Setup & Initialization]
        DataLoader[Data Loading]
        Preparation[Data Preparation]
        Training[Model Training]
        Evaluation[Model Evaluation]
        Explanation[Model Explanation]
        Persistence[Model Persistence]
    end
    
    subgraph "Model Training Components"
        Trainer[ModelTrainer]
        Monitor[TrainingMonitor]
        Evaluator[Evaluator]
        Checkpoint[CheckpointManager]
        TensorBoard[TensorBoard Writer]
    end
    
    subgraph "Model Types"
        ML[ML Models]
        RL[RL Agents]
        Ensemble[Ensemble Models]
    end
    
    subgraph "Data Sources"
        BinanceAPI[Binance API]
        CSV[CSV Files]
        Parquet[Parquet Files]
        Processor[Data Processor]
    end
    
    subgraph "Outputs"
        Models[Trained Models]
        Metrics[Training Metrics]
        Explanations[Model Explanations]
        Checkpoints[Model Checkpoints]
        Logs[Training Logs]
    end
    
    CLI --> Pipeline
    Config --> Pipeline
    Scripts --> Pipeline
    
    Pipeline --> Setup
    Setup --> DataLoader
    DataLoader --> Preparation
    Preparation --> Training
    Training --> Evaluation
    Evaluation --> Explanation
    Explanation --> Persistence
    
    Training --> Trainer
    Training --> Monitor
    Training --> Evaluator
    Training --> Checkpoint
    Training --> TensorBoard
    
    Trainer --> ML
    Trainer --> RL
    Trainer --> Ensemble
    
    BinanceAPI --> Processor
    CSV --> Processor
    Parquet --> Processor
    Processor --> DataLoader
    
    Persistence --> Models
    Persistence --> Metrics
    Persistence --> Explanations
    Persistence --> Checkpoints
    Persistence --> Logs
```

---

## Core Components

### 1. Training Pipeline (`training/train_models.py`)

The main orchestrator for the entire training process:

```mermaid
classDiagram
    class TrainingPipeline {
        -config: Dict
        -logger: Logger
        -data_processor: BaseProcessor
        -model_trainer: ModelTrainer
        -evaluator: Evaluator
        -checkpoint_manager: CheckpointManager
        +setup_environment()
        +load_configuration()
        +initialize_components()
        +prepare_data()
        +train_models()
        +evaluate_models()
        +generate_explanations()
        +save_artifacts()
    }
    
    class CheckpointManager {
        -checkpoint_dir: str
        -save_frequency: int
        +save_checkpoint(model, epoch, metrics)
        +load_checkpoint(path) Dict
        +cleanup_old_checkpoints()
        +get_best_checkpoint() str
    }
    
    class TrainingMonitor {
        -metrics_history: List[Dict]
        -early_stopping: EarlyStopping
        +log_metrics(epoch, metrics)
        +check_early_stopping() bool
        +plot_learning_curves()
        +save_metrics_history()
    }
    
    TrainingPipeline --> CheckpointManager
    TrainingPipeline --> TrainingMonitor
```

**Key Features:**
- Signal handlers for graceful interruption
- Distributed training support
- Comprehensive logging and monitoring
- Automatic directory creation for outputs
- Configuration validation and loading

### 2. Model Trainer (`training/models/model_trainer.py`)

Handles model-specific training logic:

```mermaid
classDiagram
    class ModelTrainer {
        -config: Dict
        -device: torch.device
        -models: Dict[str, BaseModel]
        +prepare_data(data) Tuple
        +train_ml_models(data_loaders)
        +train_rl_agents(env)
        +train_ensemble_models(models)
        +evaluate_model(model, data) Dict
        +explain_model(model, data) Dict
        +save_model(model, path)
        +load_model(path) BaseModel
    }
    
    class EnsembleTrainer {
        -base_models: List[BaseModel]
        -ensemble_type: str
        +train_stacking_ensemble(X, y)
        +train_weighted_ensemble(X, y)
        +train_neural_ensemble(X, y)
        +optimize_weights(predictions, targets)
    }
    
    class RLTrainer {
        -agents: Dict[str, RLAgent]
        -environment: TradingEnvironment
        +train_dqn_agent(episodes)
        +train_ppo_agent(episodes)
        +train_a3c_agent(episodes)
        +evaluate_agent(agent, episodes) Dict
    }
    
    ModelTrainer --> EnsembleTrainer
    ModelTrainer --> RLTrainer
```

**Supported Models:**
- **Traditional ML:** Random Forest, Neural Networks, Transformers
- **Ensemble Methods:** Stacking, Weighted Voting, Neural Ensembles
- **Reinforcement Learning:** DQN, PPO, A3C agents

### 3. Evaluator (`training/models/evaluator.py`)

Monitors and evaluates model performance:

```mermaid
classDiagram
    class Evaluator {
        -metrics_history: List[Dict]
        -best_metrics: Dict
        -save_path: str
        +record_metrics(epoch, metrics)
        +evaluate_performance(model, data) Dict
        +save_best_model(model, metrics)
        +plot_learning_curves()
        +generate_performance_report() Dict
        +calculate_statistical_significance() Dict
    }
    
    class MetricsCalculator {
        +calculate_classification_metrics(y_true, y_pred) Dict
        +calculate_regression_metrics(y_true, y_pred) Dict
        +calculate_trading_metrics(returns) Dict
        +calculate_risk_metrics(returns) Dict
    }
    
    class PerformancePlotter {
        +plot_training_curves(metrics)
        +plot_confusion_matrix(y_true, y_pred)
        +plot_feature_importance(importance)
        +plot_prediction_distribution(predictions)
    }
    
    Evaluator --> MetricsCalculator
    Evaluator --> PerformancePlotter
```

---

## Data Processing Pipeline

```mermaid
flowchart TD
    Start([Data Processing Start]) --> Source{Data Source}
    
    Source --> |Binance API| API[Fetch from Binance]
    Source --> |CSV Files| CSV[Load CSV Data]
    Source --> |Parquet Files| Parquet[Load Parquet Data]
    
    API --> Validate[Validate Data Quality]
    CSV --> Validate
    Parquet --> Validate
    
    Validate --> Clean[Clean & Preprocess]
    Clean --> Features[Feature Engineering]
    Features --> Technical[Technical Indicators]
    Technical --> Normalize[Normalize Features]
    Normalize --> Split[Train/Validation Split]
    Split --> Loaders[Create Data Loaders]
    Loaders --> Ready([Data Ready for Training])
    
    subgraph "Feature Engineering"
        Features --> OHLCV[OHLCV Features]
        Features --> Volume[Volume Features]
        Features --> Price[Price-based Features]
        Features --> Time[Time-based Features]
    end
    
    subgraph "Technical Indicators"
        Technical --> RSI[RSI]
        Technical --> MACD[MACD]
        Technical --> BB[Bollinger Bands]
        Technical --> SMA[Simple Moving Average]
        Technical --> EMA[Exponential Moving Average]
    end
```

### Data Sources and Formats

- **Binance API:** Real-time and historical OHLCV data
- **CSV Files:** Processed training data with features and targets
- **Parquet Files:** Compressed columnar data format for large datasets

### Feature Engineering

```mermaid
classDiagram
    class FeatureEngineer {
        +create_price_features(data) DataFrame
        +create_volume_features(data) DataFrame
        +create_technical_indicators(data) DataFrame
        +create_time_features(data) DataFrame
        +create_lag_features(data, lags) DataFrame
        +normalize_features(data) DataFrame
    }
    
    class TechnicalIndicators {
        +calculate_rsi(prices, period) Series
        +calculate_macd(prices) DataFrame
        +calculate_bollinger_bands(prices, period) DataFrame
        +calculate_moving_averages(prices, windows) DataFrame
        +calculate_momentum_indicators(data) DataFrame
    }
    
    FeatureEngineer --> TechnicalIndicators
```

---

## Training Workflow

```mermaid
sequenceDiagram
    participant CLI as CLI/Script
    participant Pipeline as TrainingPipeline
    participant Config as Configuration
    participant Data as DataProcessor
    participant Trainer as ModelTrainer
    participant Eval as Evaluator
    participant Save as Persistence
    
    CLI->>Pipeline: Initialize with arguments
    Pipeline->>Config: Load configuration
    Config-->>Pipeline: Return config dict
    
    Pipeline->>Data: Initialize data processor
    Data->>Data: Download/load data
    Data->>Data: Clean and preprocess
    Data->>Data: Engineer features
    Data-->>Pipeline: Return processed data
    
    Pipeline->>Trainer: Initialize model trainer
    
    loop For each model type
        Pipeline->>Trainer: Train model
        Trainer->>Trainer: Setup model architecture
        Trainer->>Trainer: Train on data
        Trainer->>Eval: Evaluate performance
        Eval-->>Trainer: Return metrics
        Trainer-->>Pipeline: Return trained model
    end
    
    Pipeline->>Trainer: Train ensemble models
    Trainer-->>Pipeline: Return ensemble models
    
    Pipeline->>Eval: Generate explanations
    Eval-->>Pipeline: Return explanations
    
    Pipeline->>Save: Save models and artifacts
    Save-->>Pipeline: Confirm saved
    
    Pipeline-->>CLI: Training complete
```

---

## Configuration Management

### Model Configuration (`training/config/model_config.yaml`)

```yaml
# Feature Configuration
features:
  feature_columns: ['feature1', 'feature2', 'feature3']
  target_column: 'price'
  normalization: 'standard'  # standard, minmax, robust

# Model Parameters
models:
  transformer:
    d_model: 512
    nhead: 8
    num_layers: 6
    dropout: 0.1
    max_seq_length: 100
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
  
  neural_network:
    hidden_layers: [128, 64, 32]
    activation: 'relu'
    dropout: 0.2

# Training Parameters
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10
  checkpoint_frequency: 5

# RL Agent Settings
reinforcement_learning:
  agents: ['dqn', 'ppo', 'a3c']
  episodes: 1000
  environment: 'trading_env'
  reward_function: 'sharpe_ratio'

# Output Paths
paths:
  models: './models'
  logs: './logs'
  checkpoints: './checkpoints'
  explanations: './explanations'
```

### Configuration Classes

```mermaid
classDiagram
    class ConfigManager {
        -config_path: str
        -config: Dict
        +load_config(path) Dict
        +validate_config(config) bool
        +get_model_config(model_name) Dict
        +get_training_config() Dict
        +get_data_config() Dict
        +save_config(config, path)
    }
    
    class ModelConfig {
        -model_type: str
        -hyperparameters: Dict
        +get_hyperparameters() Dict
        +set_hyperparameter(key, value)
        +validate_hyperparameters() bool
    }
    
    class TrainingConfig {
        -batch_size: int
        -epochs: int
        -learning_rate: float
        +get_optimizer_config() Dict
        +get_scheduler_config() Dict
        +get_early_stopping_config() Dict
    }
    
    ConfigManager --> ModelConfig
    ConfigManager --> TrainingConfig
```

---

## Model Training Processes

### Traditional ML Models

```mermaid
flowchart TD
    Start([ML Training Start]) --> LoadData[Load Training Data]
    LoadData --> PrepareFeatures[Prepare Features]
    PrepareFeatures --> SplitData[Split Train/Validation]
    SplitData --> InitModel[Initialize Model]
    InitModel --> TrainModel[Train Model]
    TrainModel --> ValidateModel[Validate Model]
    ValidateModel --> CheckEarlyStopping{Early Stopping?}
    CheckEarlyStopping --> |No| TrainModel
    CheckEarlyStopping --> |Yes| EvaluateModel[Final Evaluation]
    EvaluateModel --> SaveModel[Save Best Model]
    SaveModel --> End([Training Complete])
```

### Reinforcement Learning Agents

```mermaid
flowchart TD
    Start([RL Training Start]) --> InitEnv[Initialize Trading Environment]
    InitEnv --> InitAgent[Initialize RL Agent]
    InitAgent --> Episode[Start Episode]
    Episode --> State[Get Current State]
    State --> Action[Select Action]
    Action --> Execute[Execute Action in Environment]
    Execute --> Reward[Receive Reward]
    Reward --> NextState[Get Next State]
    NextState --> UpdateAgent[Update Agent]
    UpdateAgent --> CheckDone{Episode Done?}
    CheckDone --> |No| State
    CheckDone --> |Yes| EvaluateEpisode[Evaluate Episode]
    EvaluateEpisode --> CheckMaxEpisodes{Max Episodes?}
    CheckMaxEpisodes --> |No| Episode
    CheckMaxEpisodes --> |Yes| FinalEval[Final Evaluation]
    FinalEval --> SaveAgent[Save Best Agent]
    SaveAgent --> End([Training Complete])
```

---

## Evaluation and Metrics

### Performance Metrics

```mermaid
classDiagram
    class PerformanceMetrics {
        +accuracy: float
        +precision: float
        +recall: float
        +f1_score: float
        +auc_roc: float
        +sharpe_ratio: float
        +max_drawdown: float
        +total_return: float
        +volatility: float
        +win_rate: float
    }
    
    class TradingMetrics {
        +calculate_sharpe_ratio(returns) float
        +calculate_max_drawdown(equity_curve) float
        +calculate_calmar_ratio(returns, drawdown) float
        +calculate_sortino_ratio(returns) float
        +calculate_win_rate(trades) float
        +calculate_profit_factor(trades) float
    }
    
    class StatisticalMetrics {
        +calculate_information_ratio(returns, benchmark) float
        +calculate_beta(returns, market) float
        +calculate_alpha(returns, market, risk_free) float
        +calculate_var(returns, confidence) float
        +calculate_cvar(returns, confidence) float
    }
    
    PerformanceMetrics --> TradingMetrics
    PerformanceMetrics --> StatisticalMetrics
```

---

## Model Explanation and Interpretability

### Explanation Methods

```mermaid
graph TB
    subgraph "Explanation Techniques"
        SHAP[SHAP Values]
        LIME[LIME Explanations]
        FeatureImp[Feature Importance]
        Attention[Attention Weights]
        Permutation[Permutation Importance]
    end
    
    subgraph "Model Types"
        RF[Random Forest]
        NN[Neural Network]
        Trans[Transformer]
        Ensemble[Ensemble]
    end
    
    subgraph "Outputs"
        Plots[Explanation Plots]
        Reports[Explanation Reports]
        JSON[JSON Explanations]
        CSV[CSV Metrics]
    end
    
    RF --> SHAP
    RF --> FeatureImp
    NN --> LIME
    NN --> Permutation
    Trans --> Attention
    Trans --> SHAP
    Ensemble --> SHAP
    Ensemble --> FeatureImp
    
    SHAP --> Plots
    LIME --> Reports
    FeatureImp --> JSON
    Attention --> Plots
    Permutation --> CSV
```

---

## Known Issues and Limitations

### Current Limitations
- **RL Agent Explanations:** Currently unsupported due to incompatibility with SHAP/LIME
- **TensorFlow Warnings:** Version compatibility issues with SHAP explainers
- **Device Mismatch:** Handled by ensuring model and data are on the same device
- **Memory Usage:** Large models may require distributed training for optimal performance

### Planned Improvements
- Enhanced RL agent explanation support using specialized techniques
- Improved memory management for large-scale training
- Advanced hyperparameter optimization with Optuna integration
- Real-time training monitoring with MLflow integration

---

## Usage Examples

### Basic Training Command
```bash
python training/train_models.py --config training/config/model_config.yaml --data data/processed/training_data.csv
```

### Advanced Training with Custom Parameters
```bash
python training/train_models.py \
    --config training/config/model_config.yaml \
    --data data/processed/training_data.csv \
    --models transformer,random_forest,ensemble \
    --epochs 200 \
    --batch-size 64 \
    --distributed
```

---

## References

### Core Files
- `training/train_models.py` - Main training pipeline
- `training/models/model_trainer.py` - Model training logic
- `training/models/evaluator.py` - Performance evaluation
- `training/config/model_config.yaml` - Configuration file
- `training/data/data_downloader.py` - Data acquisition
- `core/data/processors/base_processor.py` - Data processing interface

### Related Documentation
- [Random Forest Model Documentation](random_forest.md)
- [Future Improvements](future_improvements.md)
- [Architecture Overview](../README.md)

---

This documentation will be continuously updated as the training pipeline evolves and new features are added.
