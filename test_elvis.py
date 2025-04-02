#!/usr/bin/env python
"""
Test script for ELVIS (Enhanced Leveraged Virtual Investment System).
This script tests all components of the system and reports any issues.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import json
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_elvis.log')
    ]
)
logger = logging.getLogger('ELVIS_TEST')

# Import ELVIS components
try:
    # Core components
    from core.data.processors.binance_processor import BinanceProcessor
    from core.models.random_forest_model import RandomForestModel
    from core.models.neural_network_model import NeuralNetworkModel
    from core.models.ensemble_model import EnsembleModel
    from core.models.transformer_model import TransformerModel
    from core.models.reinforcement_learning_model import ReinforcementLearningModel
    from core.metrics.performance_monitor import PerformanceMonitor
    from core.metrics.monte_carlo import MonteCarloSimulator
    
    # Trading components
    from trading.strategies.technical_strategy import TechnicalStrategy
    from trading.strategies.mean_reversion_strategy import MeanReversionStrategy
    from trading.strategies.trend_following_strategy import TrendFollowingStrategy
    from trading.strategies.sentiment_strategy import SentimentStrategy
    from trading.strategies.grid_strategy import GridStrategy
    from trading.execution.binance_executor import BinanceExecutor
    from trading.risk.risk_manager import RiskManager
    from trading.risk.advanced_risk_manager import AdvancedRiskManager
    
    # Utilities
    from utils.logging_utils import setup_logger
    from utils.notification_utils import send_notification
    from utils.dashboard_utils import DashboardManager
    
    # Configuration
    from config import TRADING_CONFIG, FILE_PATHS
    
    logger.info("Successfully imported all ELVIS components")
except ImportError as e:
    logger.error(f"Failed to import ELVIS components: {e}")
    traceback.print_exc()
    sys.exit(1)

class ElvisTester:
    """
    Tester for ELVIS components.
    """
    
    def __init__(self):
        """
        Initialize the tester.
        """
        self.logger = logger
        self.test_results = {
            'data_processors': {},
            'models': {},
            'strategies': {},
            'risk_managers': {},
            'utilities': {},
            'integration': {}
        }
        self.success_count = 0
        self.failure_count = 0
        
        # Create test data directory if it doesn't exist
        os.makedirs('test_data', exist_ok=True)
    
    def run_all_tests(self):
        """
        Run all tests.
        """
        self.logger.info("Starting ELVIS tests")
        
        # Test data processors
        self.test_data_processors()
        
        # Test models
        self.test_models()
        
        # Test strategies
        self.test_strategies()
        
        # Test risk managers
        self.test_risk_managers()
        
        # Test utilities
        self.test_utilities()
        
        # Test integration
        self.test_integration()
        
        # Report results
        self.report_results()
    
    def test_data_processors(self):
        """
        Test data processors.
        """
        self.logger.info("Testing data processors")
        
        # Test BinanceProcessor
        self.run_test(
            "BinanceProcessor",
            self._test_binance_processor,
            "data_processors"
        )
    
    def _test_binance_processor(self):
        """
        Test BinanceProcessor.
        """
        try:
            # Create processor
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_interval = "1h"
            
            processor = BinanceProcessor(
                start_date=start_date,
                end_date=end_date,
                time_interval=time_interval,
                logger=self.logger
            )
            
            # Create mock data instead of downloading from Binance
            self.logger.info("Creating mock data for testing")
            
            # Create a date range
            date_range = pd.date_range(
                start=datetime.now() - timedelta(days=5),
                end=datetime.now(),
                freq='1H'
            )
            
            # Create mock data
            data = pd.DataFrame({
                'date': date_range,
                'open': np.random.normal(35000, 1000, len(date_range)),
                'high': np.random.normal(36000, 1000, len(date_range)),
                'low': np.random.normal(34000, 1000, len(date_range)),
                'close': np.random.normal(35500, 1000, len(date_range)),
                'volume': np.random.normal(100, 20, len(date_range)),
                'ticker': 'BTC/USDT'
            })
            
            # Set date as index
            data.set_index('date', inplace=True)
            
            # Check if data is valid
            assert isinstance(data, pd.DataFrame), "Downloaded data is not a DataFrame"
            assert not data.empty, "Downloaded data is empty"
            assert 'open' in data.columns, "Downloaded data does not have 'open' column"
            assert 'high' in data.columns, "Downloaded data does not have 'high' column"
            assert 'low' in data.columns, "Downloaded data does not have 'low' column"
            assert 'close' in data.columns, "Downloaded data does not have 'close' column"
            assert 'volume' in data.columns, "Downloaded data does not have 'volume' column"
            
            # Save test data for other tests
            data.to_csv('test_data/binance_data.csv')
            
            return True, "BinanceProcessor test passed"
        except Exception as e:
            return False, f"BinanceProcessor test failed: {e}"
    
    def test_models(self):
        """
        Test models.
        """
        self.logger.info("Testing models")
        
        # Load test data
        try:
            data = pd.read_csv('test_data/binance_data.csv', index_col=0)
            data.index = pd.to_datetime(data.index)
        except FileNotFoundError:
            self.logger.error("Test data not found. Run data processor tests first.")
            return
        
        # Prepare features and target
        X = data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add some technical indicators as features
        X['sma_10'] = X['close'].rolling(10).mean()
        X['sma_20'] = X['close'].rolling(20).mean()
        X['rsi'] = self._calculate_rsi(X['close'])
        
        # Create a simple target (up or down next period)
        y = (X['close'].shift(-1) > X['close']).astype(int)
        
        # Drop NaN values
        X = X.dropna()
        y = y.loc[X.index]
        
        # Save processed data for other tests
        X.to_csv('test_data/features.csv')
        y.to_csv('test_data/target.csv')
        
        # Test RandomForestModel
        self.run_test(
            "RandomForestModel",
            lambda: self._test_model(RandomForestModel, X, y),
            "models"
        )
        
        # Test NeuralNetworkModel
        self.run_test(
            "NeuralNetworkModel",
            lambda: self._test_model(NeuralNetworkModel, X, y),
            "models"
        )
        
        # Test EnsembleModel
        self.run_test(
            "EnsembleModel",
            lambda: self._test_ensemble_model(X, y),
            "models"
        )
        
        # Test TransformerModel
        self.run_test(
            "TransformerModel",
            lambda: self._test_model(TransformerModel, X, y, input_dim=X.shape[1]),
            "models"
        )
        
        # Test ReinforcementLearningModel
        self.run_test(
            "ReinforcementLearningModel",
            lambda: self._test_reinforcement_learning_model(X),
            "models"
        )
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate RSI.
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _test_model(self, model_class, X, y, **kwargs):
        """
        Test a model.
        """
        try:
            # Create model
            model = model_class(self.logger, **kwargs)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            # Train model (with a small subset for quick testing)
            model.train(X_train.iloc[:100], y_train.iloc[:100])
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Check if predictions are valid
            assert len(predictions) == len(X_test), f"Predictions length ({len(predictions)}) does not match test data length ({len(X_test)})"
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Check if metrics are valid
            assert isinstance(metrics, dict), "Evaluation metrics is not a dictionary"
            
            return True, f"{model_class.__name__} test passed"
        except Exception as e:
            return False, f"{model_class.__name__} test failed: {e}"
    
    def _test_ensemble_model(self, X, y):
        """
        Test EnsembleModel.
        """
        try:
            # Create component models
            rf_model = RandomForestModel(self.logger)
            nn_model = NeuralNetworkModel(self.logger)
            
            # Create ensemble model
            model = EnsembleModel(
                self.logger,
                models=[
                    ('random_forest', rf_model),
                    ('neural_network', nn_model)
                ]
            )
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            # Train component models (with a small subset for quick testing)
            rf_model.train(X_train.iloc[:100], y_train.iloc[:100])
            nn_model.train(X_train.iloc[:100], y_train.iloc[:100])
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Check if predictions are valid
            assert len(predictions) == len(X_test), f"Predictions length ({len(predictions)}) does not match test data length ({len(X_test)})"
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Check if metrics are valid
            assert isinstance(metrics, dict), "Evaluation metrics is not a dictionary"
            
            return True, "EnsembleModel test passed"
        except Exception as e:
            return False, f"EnsembleModel test failed: {e}"
    
    def _test_reinforcement_learning_model(self, X):
        """
        Test ReinforcementLearningModel.
        """
        try:
            # Create model
            model = ReinforcementLearningModel(
                self.logger,
                state_dim=X.shape[1],
                action_dim=3,  # buy, sell, hold
                max_episodes=2,  # Use a small number for testing
                max_timesteps=100  # Use a small number for testing
            )
            
            # Make predictions (without training for quick testing)
            predictions = model.predict(X)
            
            # Check if predictions are valid
            assert len(predictions) == len(X), f"Predictions length ({len(predictions)}) does not match data length ({len(X)})"
            
            return True, "ReinforcementLearningModel test passed"
        except Exception as e:
            return False, f"ReinforcementLearningModel test failed: {e}"
    
    def test_strategies(self):
        """
        Test strategies.
        """
        self.logger.info("Testing strategies")
        
        # Load test data
        try:
            data = pd.read_csv('test_data/binance_data.csv', index_col=0)
            data.index = pd.to_datetime(data.index)
        except FileNotFoundError:
            self.logger.error("Test data not found. Run data processor tests first.")
            return
        
        # Add technical indicators
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['rsi'] = self._calculate_rsi(data['close'])
        data['macd'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
        data['macdsignal'] = data['macd'].ewm(span=9).mean()
        data['atr'] = self._calculate_atr(data)
        
        # Drop NaN values
        data = data.dropna()
        
        # Test TechnicalStrategy
        self.run_test(
            "TechnicalStrategy",
            lambda: self._test_strategy(TechnicalStrategy, data),
            "strategies"
        )
        
        # Test MeanReversionStrategy
        self.run_test(
            "MeanReversionStrategy",
            lambda: self._test_strategy(MeanReversionStrategy, data),
            "strategies"
        )
        
        # Test TrendFollowingStrategy
        self.run_test(
            "TrendFollowingStrategy",
            lambda: self._test_strategy(TrendFollowingStrategy, data),
            "strategies"
        )
        
        # Test SentimentStrategy
        self.run_test(
            "SentimentStrategy",
            lambda: self._test_strategy(SentimentStrategy, data),
            "strategies"
        )
        
        # Test GridStrategy
        self.run_test(
            "GridStrategy",
            lambda: self._test_strategy(GridStrategy, data),
            "strategies"
        )
    
    def _calculate_atr(self, data, period=14):
        """
        Calculate ATR.
        """
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _test_strategy(self, strategy_class, data):
        """
        Test a strategy.
        """
        try:
            # Create strategy
            strategy = strategy_class(self.logger)
            
            # Generate signals
            buy_signals = []
            sell_signals = []
            
            for i in range(len(data) - 1):
                # Use a window of data
                window = data.iloc[max(0, i-30):i+1]
                
                # Generate signals
                buy_signal, sell_signal = strategy.generate_signals(window)
                
                buy_signals.append(buy_signal)
                sell_signals.append(sell_signal)
            
            # Check if signals are valid
            assert len(buy_signals) == len(data) - 1, f"Buy signals length ({len(buy_signals)}) does not match data length ({len(data) - 1})"
            assert len(sell_signals) == len(data) - 1, f"Sell signals length ({len(sell_signals)}) does not match data length ({len(data) - 1})"
            
            # Test position size calculation
            position_size = strategy.calculate_position_size(
                data.iloc[-10:],
                data['close'].iloc[-1],
                10000.0
            )
            
            # Check if position size is valid
            assert position_size >= 0, f"Position size ({position_size}) is negative"
            
            # Test stop loss calculation
            stop_loss = strategy.calculate_stop_loss(
                data.iloc[-10:],
                data['close'].iloc[-1]
            )
            
            # Check if stop loss is valid
            assert stop_loss < data['close'].iloc[-1], f"Stop loss ({stop_loss}) is not below entry price ({data['close'].iloc[-1]})"
            
            # Test take profit calculation
            take_profit = strategy.calculate_take_profit(
                data.iloc[-10:],
                data['close'].iloc[-1]
            )
            
            # Check if take profit is valid
            assert take_profit > data['close'].iloc[-1], f"Take profit ({take_profit}) is not above entry price ({data['close'].iloc[-1]})"
            
            return True, f"{strategy_class.__name__} test passed"
        except Exception as e:
            return False, f"{strategy_class.__name__} test failed: {e}"
    
    def test_risk_managers(self):
        """
        Test risk managers.
        """
        self.logger.info("Testing risk managers")
        
        # Test RiskManager
        self.run_test(
            "RiskManager",
            self._test_risk_manager,
            "risk_managers"
        )
        
        # Test AdvancedRiskManager
        self.run_test(
            "AdvancedRiskManager",
            self._test_advanced_risk_manager,
            "risk_managers"
        )
    
    def _test_risk_manager(self):
        """
        Test RiskManager.
        """
        try:
            # Create risk manager
            risk_manager = RiskManager(self.logger)
            
            # Load test data
            try:
                data = pd.read_csv('test_data/binance_data.csv', index_col=0)
                data.index = pd.to_datetime(data.index)
            except FileNotFoundError:
                self.logger.error("Test data not found. Run data processor tests first.")
                return False, "RiskManager test failed: Test data not found"
            
            # Test position size calculation
            position_size = risk_manager.calculate_position_size(
                data.iloc[-10:],
                data['close'].iloc[-1],
                10000.0
            )
            
            # Check if position size is valid
            assert position_size >= 0, f"Position size ({position_size}) is negative"
            
            # Test trade limits
            assert risk_manager.check_trade_limits(), "Trade limits check failed"
            
            # Test updating trade stats
            risk_manager.update_trade_stats(100.0)
            risk_manager.update_trade_stats(-50.0)
            
            # Check if trade stats are updated
            assert risk_manager.trades_today == 2, f"Trades today ({risk_manager.trades_today}) is not 2"
            assert risk_manager.daily_pnl == 50.0, f"Daily PnL ({risk_manager.daily_pnl}) is not 50.0"
            
            return True, "RiskManager test passed"
        except Exception as e:
            return False, f"RiskManager test failed: {e}"
    
    def _test_advanced_risk_manager(self):
        """
        Test AdvancedRiskManager.
        """
        try:
            # Create advanced risk manager
            risk_manager = AdvancedRiskManager(self.logger)
            
            # Load test data
            try:
                data = pd.read_csv('test_data/binance_data.csv', index_col=0)
                data.index = pd.to_datetime(data.index)
            except FileNotFoundError:
                self.logger.error("Test data not found. Run data processor tests first.")
                return False, "AdvancedRiskManager test failed: Test data not found"
            
            # Add technical indicators
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['atr'] = self._calculate_atr(data)
            
            # Drop NaN values
            data = data.dropna()
            
            # Test position size calculation
            position_size = risk_manager.calculate_position_size(
                data.iloc[-10:],
                data['close'].iloc[-1],
                10000.0
            )
            
            # Check if position size is valid
            assert position_size >= 0, f"Position size ({position_size}) is negative"
            
            # Test market regime detection
            market_regime = risk_manager.detect_market_regime(data.iloc[-30:])
            
            # Check if market regime is valid
            assert market_regime in ['bullish', 'bearish', 'neutral'], f"Market regime ({market_regime}) is not valid"
            
            # Test volatility adjustment
            volatility_adjustment = risk_manager.calculate_volatility_adjustment(data.iloc[-30:])
            
            # Check if volatility adjustment is valid
            assert volatility_adjustment > 0, f"Volatility adjustment ({volatility_adjustment}) is not positive"
            
            # Test updating equity
            risk_manager.update_equity(10000.0)
            risk_manager.update_equity(9500.0)
            
            # Check if drawdown is calculated
            assert len(risk_manager.drawdown_history) > 0, "Drawdown history is empty"
            assert risk_manager.drawdown_history[-1] == 0.05, f"Drawdown ({risk_manager.drawdown_history[-1]}) is not 0.05"
            
            # Test circuit breaker
            assert not risk_manager.check_circuit_breaker(), "Circuit breaker is active"
            
            # Test updating win/loss history
            risk_manager.update_win_loss_history(100.0)
            risk_manager.update_win_loss_history(-50.0)
            
            # Check if win/loss history is updated
            assert len(risk_manager.win_history) == 1, f"Win history length ({len(risk_manager.win_history)}) is not 1"
            assert len(risk_manager.loss_history) == 1, f"Loss history length ({len(risk_manager.loss_history)}) is not 1"
            
            # Test win rate and win/loss ratio
            win_rate = risk_manager.get_win_rate()
            win_loss_ratio = risk_manager.get_win_loss_ratio()
            
            # Check if win rate and win/loss ratio are valid
            assert 0 <= win_rate <= 1, f"Win rate ({win_rate}) is not between 0 and 1"
            assert win_loss_ratio > 0, f"Win/loss ratio ({win_loss_ratio}) is not positive"
            
            # Test Kelly criterion
            kelly = risk_manager.calculate_kelly_criterion(win_rate, win_loss_ratio)
            
            # Check if Kelly criterion is valid
            assert 0 <= kelly <= 1, f"Kelly criterion ({kelly}) is not between 0 and 1"
            
            # Test risk metrics
            metrics = risk_manager.get_risk_metrics()
            
            # Check if metrics are valid
            assert isinstance(metrics, dict), "Risk metrics is not a dictionary"
            assert 'win_rate' in metrics, "Win rate is not in risk metrics"
            assert 'win_loss_ratio' in metrics, "Win/loss ratio is not in risk metrics"
            
            return True, "AdvancedRiskManager test passed"
        except Exception as e:
            return False, f"AdvancedRiskManager test failed: {e}"
    
    def test_utilities(self):
        """
        Test utilities.
        """
        self.logger.info("Testing utilities")
        
        # Test PerformanceMonitor
        self.run_test(
            "PerformanceMonitor",
            self._test_performance_monitor,
            "utilities"
        )
        
        # Test MonteCarloSimulator
        self.run_test(
            "MonteCarloSimulator",
            self._test_monte_carlo_simulator,
            "utilities"
        )
        
        # Test DashboardManager
        self.run_test(
            "DashboardManager",
            self._test_dashboard_manager,
            "utilities"
        )
    
    def _test_performance_monitor(self):
        """
        Test PerformanceMonitor.
        """
        try:
            # Create performance monitor
            monitor = PerformanceMonitor(self.logger)
            
            # Add trades
            for i in range(10):
                monitor.add_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'BTC/USDT',
                    'side': 'buy' if i % 2 == 0 else 'sell',
                    'price': 36000.0 + i * 100,
                    'quantity': 0.1,
                    'pnl': 100.0 if i % 3 == 0 else -50.0
                })
            
            # Calculate metrics
            metrics = monitor.calculate_metrics()
            
            # Check if metrics are valid
            assert isinstance(metrics, dict), "Performance metrics is not a dictionary"
            assert 'total_trades' in metrics, "Total trades is not in performance metrics"
            assert 'win_rate' in metrics, "Win rate is not in performance metrics"
            assert 'profit_factor' in metrics, "Profit factor is not in performance metrics"
            assert 'sharpe_ratio' in metrics, "Sharpe ratio is not in performance metrics"
            
            # Generate report
            report_path = monitor.generate_report()
            
            # Check if report is generated
            assert os.path.exists(report_path), f"Report not found at {report_path}"
            
            return True, "PerformanceMonitor test passed"
        except Exception as e:
            return False, f"PerformanceMonitor test failed: {e}"
    
    def _test_monte_carlo_simulator(self):
        """
        Test MonteCarloSimulator.
        """
        try:
            # Create Monte Carlo simulator
            simulator = MonteCarloSimulator(
                self.logger,
                num_simulations=10,  # Use a small number for testing
                parallel=False  # Disable parallel processing for testing
            )
            
            # Generate random returns
            returns = np.random.normal(0.001, 0.01, 100)
            
            # Run simulation
            results = simulator.run_simulation(returns)
            
            # Check if results are valid
            assert isinstance(results, dict), "Simulation results is not a dictionary"
            assert 'statistics' in results, "Statistics is not in simulation results"
            assert 'profit_probability' in results, "Profit probability is not in simulation results"
            
            return True, "MonteCarloSimulator test passed"
        except Exception as e:
            return False, f"MonteCarloSimulator test failed: {e}"
    
    def _test_dashboard_manager(self):
        """
        Test DashboardManager.
        """
        try:
            # Create dashboard manager
            dashboard = DashboardManager(self.logger)
            
            # Add trade
            dashboard.add_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 36000.0,
                'quantity': 0.1,
                'pnl': 100.0
            })
            
            # Update portfolio value
            dashboard.update_portfolio_value(10000.0)
            
            # Update metrics
            dashboard.update_metrics({
                'total_trades': 1,
                'win_rate': 1.0,
                'profit_factor': 2.0,
                'sharpe_ratio': 1.5
            })
            
            # Update model performance
            dashboard.update_model_performance({
                'random_forest': {
                    'accuracy': 0.8,
                    'precision': 0.7,
                    'recall': 0.6
                },
                'neural_network': {
                    'accuracy': 0.7,
                    'precision': 0.6,
                    'recall': 0.5
                }
            })
            
            # Update strategy signals
            dashboard.update_strategy_signals({
                'technical': {
                    'buy': True,
                    'sell': False
                },
                'mean_reversion': {
                    'buy': False,
                    'sell': True
                }
            })
            
            # Update market data
            dashboard.update_market_data({
                'prices': [
                    {'time': '09:00', 'price': 36000.0},
                    {'time': '10:00', 'price': 36100.0},
                    {'time': '11:00', 'price': 36200.0}
                ]
            })
            
            return True, "DashboardManager test passed"
        except Exception as e:
            return False, f"DashboardManager test failed: {e}"
    
    def test_integration(self):
        """
        Test integration.
        """
        self.logger.info("Testing integration")
        
        # Test end-to-end workflow
        self.run_test(
            "End-to-end workflow",
            self._test_end_to_end_workflow,
            "integration"
        )
    
    def _test_end_to_end_workflow(self):
        """
        Test end-to-end workflow.
        """
        try:
            # Load test data
            try:
                data = pd.read_csv('test_data/binance_data.csv', index_col=0)
                data.index = pd.to_datetime(data.index)
            except FileNotFoundError:
                self.logger.error("Test data not found. Run data processor tests first.")
                return False, "End-to-end workflow test failed: Test data not found"
            
            # Add technical indicators
            data['sma_10'] = data['close'].rolling(10).mean()
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['rsi'] = self._calculate_rsi(data['close'])
            data['macd'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
            data['macdsignal'] = data['macd'].ewm(span=9).mean()
            data['atr'] = self._calculate_atr(data)
            
            # Drop NaN values
            data = data.dropna()
            
            # Create strategy
            strategy = TechnicalStrategy(self.logger)
            
            # Create risk manager
            risk_manager = AdvancedRiskManager(self.logger)
            
            # Create model
            model = RandomForestModel(self.logger)
            
            # Create performance monitor
            monitor = PerformanceMonitor(self.logger)
            
            # Simulate trading
            portfolio_value = 10000.0
            position = 0.0
            
            for i in range(len(data) - 1):
                # Use a window of data
                window = data.iloc[max(0, i-30):i+1]
                
                # Generate signals
                buy_signal, sell_signal = strategy.generate_signals(window)
                
                # Current price
                current_price = data['close'].iloc[i]
                
                # Execute trades
                if buy_signal and position == 0:
                    # Calculate position size
                    position_size = risk_manager.calculate_position_size(
                        window,
                        current_price,
                        portfolio_value
                    )
                    
                    # Calculate entry price (with slippage)
                    entry_price = current_price * 1.001
                    
                    # Update position
                    position = position_size
                    portfolio_value -= position * entry_price
                    
                    # Record trade
                    monitor.add_trade({
                        'timestamp': data.index[i].isoformat(),
                        'symbol': 'BTC/USDT',
                        'side': 'buy',
                        'price': entry_price,
                        'quantity': position,
                        'pnl': 0.0
                    })
                
                elif sell_signal and position > 0:
                    # Calculate exit price (with slippage)
                    exit_price = current_price * 0.999
                    
                    # Calculate PnL
                    pnl = (exit_price - entry_price) * position
                    
                    # Update portfolio value
                    portfolio_value += position * exit_price
                    
                    # Update risk manager
                    risk_manager.update_trade_stats(pnl)
                    
                    # Record trade
                    monitor.add_trade({
                        'timestamp': data.index[i].isoformat(),
                        'symbol': 'BTC/USDT',
                        'side': 'sell',
                        'price': exit_price,
                        'quantity': position,
                        'pnl': pnl
                    })
                    
                    # Reset position
                    position = 0.0
            
            # Calculate final metrics
            metrics = monitor.calculate_metrics()
            
            # Check if metrics are valid
            assert isinstance(metrics, dict), "Performance metrics is not a dictionary"
            assert 'total_trades' in metrics, "Total trades is not in performance metrics"
            
            return True, "End-to-end workflow test passed"
        except Exception as e:
            return False, f"End-to-end workflow test failed: {e}"
    
    def run_test(self, test_name, test_func, category):
        """
        Run a test and record the result.
        
        Args:
            test_name (str): The name of the test.
            test_func (callable): The test function.
            category (str): The category of the test.
        """
        self.logger.info(f"Running test: {test_name}")
        
        try:
            # Run test
            success, message = test_func()
            
            # Record result
            self.test_results[category][test_name] = {
                'success': success,
                'message': message
            }
            
            # Update counters
            if success:
                self.success_count += 1
                self.logger.info(f"Test passed: {test_name}")
            else:
                self.failure_count += 1
                self.logger.error(f"Test failed: {test_name} - {message}")
        except Exception as e:
            # Record error
            self.test_results[category][test_name] = {
                'success': False,
                'message': f"Error running test: {e}"
            }
            
            # Update counter
            self.failure_count += 1
            self.logger.error(f"Error running test: {test_name} - {e}")
            traceback.print_exc()
    
    def report_results(self):
        """
        Report test results.
        """
        self.logger.info("Test results:")
        self.logger.info(f"Total tests: {self.success_count + self.failure_count}")
        self.logger.info(f"Passed: {self.success_count}")
        self.logger.info(f"Failed: {self.failure_count}")
        
        # Print results by category
        for category, tests in self.test_results.items():
            self.logger.info(f"\n{category.upper()}:")
            
            for test_name, result in tests.items():
                status = "PASS" if result['success'] else "FAIL"
                self.logger.info(f"  {status}: {test_name}")
                
                if not result['success']:
                    self.logger.info(f"    {result['message']}")
        
        # Save results to file
        with open('test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        self.logger.info(f"Test results saved to test_results.json")


if __name__ == "__main__":
    tester = ElvisTester()
    tester.run_all_tests()
