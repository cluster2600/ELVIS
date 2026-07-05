"""
Centralized logging configuration for ELVIS Trading Bot
Provides structured logging with file rotation, console output, and remote logging capabilities
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Log format patterns
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
SIMPLE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
JSON_FORMAT = "%(message)s"  # For JSON formatter


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add extra fields if any
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_obj[key] = value

        return json.dumps(log_obj)


class TradingContextFilter(logging.Filter):
    """Add trading-specific context to log records"""

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        # Add trading context to the record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logging(
    app_name: str = "ELVIS",
    log_level: str = "INFO",
    enable_file_logging: bool = True,
    enable_json_logging: bool = False,
    enable_remote_logging: bool = False,
    remote_host: Optional[str] = None,
    remote_port: Optional[int] = None,
    trading_context: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """
    Set up comprehensive logging configuration

    Args:
        app_name: Application name for log identification
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to log to files
        enable_json_logging: Whether to use JSON format for logs
        enable_remote_logging: Whether to send logs to remote server
        remote_host: Remote logging server host
        remote_port: Remote logging server port
        trading_context: Trading-specific context to add to logs

    Returns:
        Configured logger instance
    """

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if enable_json_logging:
        console_handler.setFormatter(JSONFormatter())
    else:
        # Use colorlog if available
        try:
            import colorlog

            console_handler.setFormatter(
                colorlog.ColoredFormatter(
                    "%(log_color)s" + SIMPLE_FORMAT,
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                )
            )
        except ImportError:
            console_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT))

    root_logger.addHandler(console_handler)

    # File handlers
    if enable_file_logging:
        # Main log file with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / f"{app_name.lower()}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            JSONFormatter()
            if enable_json_logging
            else logging.Formatter(DETAILED_FORMAT)
        )
        root_logger.addHandler(file_handler)

        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / f"{app_name.lower()}_errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
        root_logger.addHandler(error_handler)

        # Trading-specific log file
        trading_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / f"{app_name.lower()}_trading.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        trading_handler.setLevel(logging.INFO)
        trading_handler.setFormatter(
            JSONFormatter()
            if enable_json_logging
            else logging.Formatter(DETAILED_FORMAT)
        )
        # Add filter for trading-specific logs
        trading_handler.addFilter(lambda record: record.name.startswith("trading"))
        root_logger.addHandler(trading_handler)

    # Remote logging handler (e.g., for Logstash, Fluentd)
    if enable_remote_logging and remote_host and remote_port:
        remote_handler = logging.handlers.SocketHandler(remote_host, remote_port)
        remote_handler.setLevel(logging.INFO)
        remote_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(remote_handler)

    # Add trading context filter if provided
    if trading_context:
        context_filter = TradingContextFilter(trading_context)
        for handler in root_logger.handlers:
            handler.addFilter(context_filter)

    # Configure specific loggers
    configure_module_loggers(log_level)

    # Get app-specific logger
    app_logger = logging.getLogger(app_name)
    app_logger.info(f"Logging initialized for {app_name} at level {log_level}")

    return app_logger


def configure_module_loggers(default_level: str = "INFO"):
    """Configure logging levels for specific modules"""

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("binance").setLevel(logging.INFO)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Set specific levels for our modules
    logging.getLogger("trading").setLevel(logging.DEBUG)
    logging.getLogger("core.models").setLevel(logging.INFO)
    logging.getLogger("utils").setLevel(logging.INFO)
    logging.getLogger("training").setLevel(logging.INFO)


def get_logger(
    name: str, trading_context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Get a logger instance with optional trading context

    Args:
        name: Logger name (usually __name__)
        trading_context: Optional trading-specific context

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if trading_context:
        # Create a logger adapter with context
        logger = logging.LoggerAdapter(logger, trading_context)

    return logger


class TradingLogger:
    """Specialized logger for trading operations with structured logging"""

    def __init__(self, name: str, symbol: str = None, strategy: str = None):
        self.logger = logging.getLogger(f"trading.{name}")
        self.symbol = symbol
        self.strategy = strategy
        self.context = {
            "symbol": symbol,
            "strategy": strategy,
        }

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with context"""
        extra = {**self.context, **kwargs}
        getattr(self.logger, level)(message, extra=extra)

    def trade_signal(self, signal_type: str, price: float, confidence: float, **kwargs):
        """Log trading signal"""
        self._log(
            "info",
            f"Trading signal generated: {signal_type}",
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            **kwargs,
        )

    def order_placed(self, order_type: str, quantity: float, price: float, **kwargs):
        """Log order placement"""
        self._log(
            "info",
            f"Order placed: {order_type}",
            order_type=order_type,
            quantity=quantity,
            price=price,
            **kwargs,
        )

    def order_filled(self, order_id: str, fill_price: float, **kwargs):
        """Log order execution"""
        self._log(
            "info",
            f"Order filled: {order_id}",
            order_id=order_id,
            fill_price=fill_price,
            **kwargs,
        )

    def error(self, error_type: str, message: str, **kwargs):
        """Log trading error"""
        self._log(
            "error",
            f"Trading error: {error_type} - {message}",
            error_type=error_type,
            **kwargs,
        )

    def risk_alert(self, alert_type: str, message: str, **kwargs):
        """Log risk management alert"""
        self._log(
            "warning",
            f"Risk alert: {alert_type} - {message}",
            alert_type=alert_type,
            **kwargs,
        )
