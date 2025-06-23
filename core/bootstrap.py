"""
Application bootstrapper for dependency injection and event setup.
"""

import os
import logging
from typing import Optional

from core.di import container
from core.events import event_bus, SystemEvent
from config import TRADING_CONFIG, API_CONFIG
from utils.logger_config import setup_logging, TradingLogger
from utils.redis_cache import RedisCache
from utils.secrets_manager import SecretsManager
from utils.async_utils import AsyncTaskManager


class ApplicationBootstrapper:
    """
    Bootstraps the application with dependency injection and event handling.
    """
    
    def __init__(self, mode: str = 'paper', log_level: str = 'INFO'):
        """
        Initialize the bootstrapper.
        
        Args:
            mode: Trading mode ('paper' or 'live')
            log_level: Logging level
        """
        self.mode = mode
        self.log_level = log_level
        self.logger = None
    
    def bootstrap(self) -> None:
        """Bootstrap the entire application."""
        # Setup logging first
        self._setup_logging()
        
        # Register configurations
        self._register_configurations()
        
        # Register core services
        self._register_core_services()
        
        # Register data services
        self._register_data_services()
        
        # Register trading services
        self._register_trading_services()
        
        # Register models
        self._register_models()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Publish startup event
        event_bus.publish(SystemEvent(
            system_type='startup',
            component='bootstrapper',
            status='ok',
            message='Application bootstrapped successfully',
            source='ApplicationBootstrapper'
        ))
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        trading_context = {
            "mode": self.mode,
            "symbol": TRADING_CONFIG.get('SYMBOL', 'BTCUSDT'),
        }
        
        self.logger = setup_logging(
            app_name="ELVIS",
            log_level=self.log_level,
            enable_file_logging=True,
            enable_json_logging=os.getenv('ENABLE_JSON_LOGS', 'false').lower() == 'true',
            enable_remote_logging=bool(os.getenv('REMOTE_LOG_HOST')),
            remote_host=os.getenv('REMOTE_LOG_HOST'),
            remote_port=int(os.getenv('REMOTE_LOG_PORT', 514)),
            trading_context=trading_context
        )
        
        # Register logger
        container.register_singleton('logger', lambda: self.logger)
        container.register_singleton('trading_logger', lambda: TradingLogger(
            "main",
            symbol=TRADING_CONFIG.get('SYMBOL', 'BTCUSDT'),
            strategy="ensemble"
        ))
    
    def _register_configurations(self) -> None:
        """Register configuration dependencies."""
        container.register_configuration('trading_config', TRADING_CONFIG)
        container.register_configuration('api_config', API_CONFIG)
        container.register_configuration('app_config', {
            'mode': self.mode,
            'log_level': self.log_level,
            'symbol': TRADING_CONFIG.get('SYMBOL', 'BTCUSDT'),
            'starting_balance': TRADING_CONFIG.get('STARTING_BALANCE', 1000.0),
        })
    
    def _register_core_services(self) -> None:
        """Register core service dependencies."""
        # Secrets Manager
        container.register_singleton('secrets_manager', lambda: SecretsManager())
        
        # Redis Cache
        def create_redis_cache():
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            return RedisCache(host=redis_host, port=redis_port)
        
        container.register_singleton('redis_cache', create_redis_cache)
        
        # Async Task Manager
        container.register_singleton('async_task_manager', 
                                   lambda: AsyncTaskManager(max_concurrent=10))
        
        # Event Bus (already global, but register for consistency)
        container.register_singleton('event_bus', lambda: event_bus)

        # Performance Monitor
        from core.metrics.performance_monitor import PerformanceMonitor
        def create_performance_monitor():
            pm = PerformanceMonitor(window=50)  # Smaller window for demo
            # Add sample percentage returns for demonstration (in decimal format)
            sample_returns = [0.005, 0.012, -0.008, 0.015, -0.003, 0.007, 0.018, -0.002, 0.021, 0.004] * 10  # 100 returns
            for ret in sample_returns:
                pm.add_return(ret)
            return pm
        container.register_singleton('performance_monitor', create_performance_monitor)

        # Trade Analyzer
        from utils.trade_analyzer import TradeAnalyzer
        def create_trade_analyzer():
            # Initialize with sample trades for demonstration (replace with real trades in production)
            sample_trades = [
                {'symbol': 'BTCUSDT', 'side': 'buy', 'quantity': 0.001, 'price': 95000, 'pnl': 50.0, 'timestamp': '2024-06-23 10:00:00'},
                {'symbol': 'BTCUSDT', 'side': 'sell', 'quantity': 0.001, 'price': 96000, 'pnl': 100.0, 'timestamp': '2024-06-23 11:00:00'},
                {'symbol': 'BTCUSDT', 'side': 'buy', 'quantity': 0.002, 'price': 94000, 'pnl': -25.0, 'timestamp': '2024-06-23 12:00:00'},
                {'symbol': 'BTCUSDT', 'side': 'sell', 'quantity': 0.0015, 'price': 97000, 'pnl': 150.0, 'timestamp': '2024-06-23 13:00:00'},
                {'symbol': 'BTCUSDT', 'side': 'buy', 'quantity': 0.0005, 'price': 93000, 'pnl': -10.0, 'timestamp': '2024-06-23 14:00:00'},
            ]
            return TradeAnalyzer(sample_trades)
        container.register_singleton('trade_analyzer', create_trade_analyzer)

        # System Monitor
        from utils.system_monitor import SystemMonitor
        container.register_singleton('system_monitor', lambda: SystemMonitor(api_url='https://api.binance.com'))

        # Market Regime Detector
        from trading.market_regime_detector import MarketRegimeDetector
        container.register_singleton('market_regime_detector', lambda: MarketRegimeDetector())

        # Order Flow Analyzer
        from trading.order_flow_analyzer import OrderFlowAnalyzer
        container.register_singleton('order_flow_analyzer', lambda: OrderFlowAnalyzer())

        # Strategy Manager
        from trading.strategy_manager import StrategyManager
        def create_strategy_manager():
            from trading.strategies.mean_reversion_strategy import MeanReversionStrategy
            from trading.strategies.trend_following_strategy import TrendFollowingStrategy
            
            strategies = {
                'trending': container.get('trend_following_strategy'),
                'mean-reverting': container.get('mean_reversion_strategy'),
                'default': container.get('strategy') # EnsembleStrategy
            }
            return StrategyManager(
                market_regime_detector=container.get('market_regime_detector'),
                strategies=strategies
            )
        container.register_singleton('strategy_manager', create_strategy_manager)
    
    def _register_data_services(self) -> None:
        """Register data-related service dependencies."""
        from utils.price_fetcher import PriceFetcher
        from core.data.processors.binance_processor import BinanceProcessor
        
        # Price Fetcher
        def create_price_fetcher():
            logger = container.get('logger')
            return PriceFetcher(logger)
        
        container.register_singleton('price_fetcher', create_price_fetcher)
        
        # Data Processor
        def create_data_processor():
            return BinanceProcessor(
                data_source='binance',
                start_date='2023-01-01',
                end_date='2024-01-01',
                time_interval='5m'
            )
        
        container.register_singleton('data_processor', create_data_processor)
    
    def _register_trading_services(self) -> None:
        """Register trading-related service dependencies."""
        from trading.execution.binance_executor import BinanceExecutor
        from trading.risk_management import RiskManager
        from trading.utils.telegram_notifier import TelegramNotifier
        from trading.strategies.ensemble_strategy import EnsembleStrategy
        
        # Telegram Notifier
        def create_notifier():
            logger = container.get('logger')
            trading_config = container.get('trading_config')
            return TelegramNotifier(logger, trading_config)
        
        container.register_singleton('notifier', create_notifier)
        
        # Risk Manager
        def create_risk_manager():
            logger = container.get('logger')
            executor = container.get('executor')
            performance_monitor = container.get('performance_monitor')
            rm = RiskManager(executor, logger, performance_monitor)
            # Add sample PnL data for demonstration
            rm.realized_pnl = 265.0  # Sum of sample trade PnLs
            rm.unrealized_pnl = 0.0  # No open positions yet
            return rm
        
        container.register_singleton('risk_manager', create_risk_manager)
        
        # Binance Executor
        def create_executor():
            logger = container.get('logger')
            api_config = container.get('api_config')
            app_config = container.get('app_config')
            
            executor = BinanceExecutor(
                logger=logger,
                api_key=api_config['BINANCE_API_KEY'],
                api_secret=api_config['BINANCE_API_SECRET'],
                is_testnet=(app_config['mode'] == 'paper')
            )
            executor.initialize()
            return executor
        
        container.register_singleton('executor', create_executor)
        
        # Ensemble Strategy
        def create_strategy():
            logger = container.get('logger')
            
            return EnsembleStrategy(
                logger=logger
            )
        
        container.register_singleton('strategy', create_strategy)

        # Grid Strategy
        from trading.strategies.grid_strategy import GridStrategy
        container.register_factory('grid_strategy', 
                                 lambda: GridStrategy(logger=container.get('logger')))
                                 
        # Mean Reversion Strategy
        from trading.strategies.mean_reversion_strategy import MeanReversionStrategy
        container.register_factory('mean_reversion_strategy',
                                 lambda: MeanReversionStrategy(logger=container.get('logger')))
                                 
        # Trend Following Strategy
        from trading.strategies.trend_following_strategy import TrendFollowingStrategy
        container.register_factory('trend_following_strategy',
                                 lambda: TrendFollowingStrategy(logger=container.get('logger')))
    
    def _register_models(self) -> None:
        """Register ML model dependencies."""
        from core.models.random_forest_model import RandomForestModel
        from core.models.neural_network_model import NeuralNetworkModel
        from core.models.transformer_model import TransformerModel
        from core.models.ensemble_model import EnsembleModel
        
        # Random Forest Model
        container.register_factory('random_forest_model', 
                                 lambda: RandomForestModel())
        
        # Neural Network Model
        container.register_factory('neural_network_model', 
                                 lambda: NeuralNetworkModel(sequence_length=60))
        
        # Transformer Model
        container.register_factory('transformer_model',
                                 lambda: TransformerModel(
                                     input_dim=10,
                                     d_model=128,
                                     nhead=8,
                                     num_layers=4
                                 ))
        
        # Ensemble Model
        def create_ensemble_model():
            ensemble = EnsembleModel(voting_type='soft')
            # Add sub-models as needed
            return ensemble
        
        container.register_factory('ensemble_model', create_ensemble_model)
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for the application."""
        # Import event handlers
        from trading.event_handlers import (
            market_data_handlers,
            trading_signal_handlers,
            risk_handlers,
            system_handlers
        )
        
        # Auto-register all handlers
        from core.events.decorators import auto_register_handlers
        
        # Register handlers from each module
        for module in [market_data_handlers, trading_signal_handlers, 
                      risk_handlers, system_handlers]:
            try:
                auto_register_handlers(module)
                self.logger.info(f"Registered event handlers from {module.__name__}")
            except ImportError:
                # Handlers might not exist yet, that's okay
                pass
    
    def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        # Publish shutdown event
        event_bus.publish(SystemEvent(
            system_type='shutdown',
            component='bootstrapper',
            status='ok',
            message='Application shutting down',
            source='ApplicationBootstrapper'
        ))
        
        # Stop event bus
        event_bus.stop()
        
        # Clear container
        container.clear()
        
        self.logger.info("Application cleanup completed")


def bootstrap_application(mode: str = 'paper', log_level: str = 'INFO') -> ApplicationBootstrapper:
    """
    Bootstrap the application and return the bootstrapper instance.
    
    Args:
        mode: Trading mode ('paper' or 'live')
        log_level: Logging level
        
    Returns:
        ApplicationBootstrapper instance
    """
    bootstrapper = ApplicationBootstrapper(mode, log_level)
    bootstrapper.bootstrap()
    return bootstrapper
