# ELVIS Trading Bot - Comprehensive Project Overview

![Project Image](./images/elvis.png)

## Table of Contents

- [Introduction](#introduction)
- [Architecture Overview](#architecture-overview)
- [Core Models](#core-models)
  - [BaseModel Interface](#basemodel-interface)
  - [RandomForestModel](#randomforestmodel)
  - [NeuralNetworkModel](#neuralnetworkmodel)
  - [TransformerModel](#transformermodel)
  - [EnsembleModel](#ensemblemodel)
- [Training Pipeline](#training-pipeline)
- [Data Processing](#data-processing)
- [Trading Strategies](#trading-strategies)
  - [BaseStrategy](#basestrategy)
  - [EnsembleStrategy](#ensemblestrategy)
- [Execution Modules](#execution-modules)
  - [BaseExecutor](#baseexecutor)
  - [BinanceExecutor](#binanceexecutor)
- [Utilities](#utilities)
  - [PriceFetcher](#pricefetcher)
  - [ConsoleDashboard](#consoledashboard)
  - [TrainingMonitor](#trainingmonitor)
- [Testing](#testing)
- [Configuration](#configuration)
- [Monitoring and Metrics](#monitoring-and-metrics)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## Introduction

The ELVIS Trading Bot is a modular, extensible trading system designed to leverage machine learning models for algorithmic trading. It integrates multiple model architectures, a robust training pipeline, real-time data processing, and execution modules to facilitate automated trading strategies. The system includes monitoring and visualization tools to provide insights into trading performance and system health.

---

## Architecture Overview

The system is organized into several key components:

- **Core Models:** Implementations of machine learning models including Random Forest, Neural Networks, Transformers, and Ensembles.
- **Training Pipeline:** Orchestrates data loading, model training, evaluation, and explanation generation.
- **Data Processing:** Handles data acquisition, cleaning, feature engineering, and transformation.
- **Trading Strategies:** Define signal generation and position sizing logic.
- **Execution Modules:** Interface with trading platforms to execute orders.
- **Utilities:** Support monitoring, logging, price fetching, and dashboard visualization.
- **Testing:** Unit tests ensure model correctness and robustness.
- **Configuration:** YAML and Python config files manage parameters and environment settings.
- **Monitoring:** Prometheus metrics integration and console dashboards provide real-time insights.

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
