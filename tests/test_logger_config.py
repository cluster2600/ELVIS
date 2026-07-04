"""
Tests for centralized logger configuration
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from utils.logger_config import (
    JSONFormatter,
    TradingContextFilter,
    TradingLogger,
    configure_module_loggers,
    get_logger,
    setup_logging,
)


class TestJSONFormatter:
    """Test JSON log formatter"""

    def test_json_formatter_basic(self):
        """Test basic JSON formatting"""
        formatter = JSONFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        log_obj = json.loads(formatted)

        assert log_obj["level"] == "INFO"
        assert log_obj["logger"] == "test.logger"
        assert log_obj["message"] == "Test message"
        assert log_obj["line"] == 10
        assert "timestamp" in log_obj

    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception info"""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        formatted = formatter.format(record)
        log_obj = json.loads(formatted)

        assert "exception" in log_obj
        assert "ValueError: Test error" in log_obj["exception"]

    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatting with extra fields"""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=30,
            msg="Trade executed",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.symbol = "BTCUSDT"
        record.price = 50000
        record.quantity = 0.1

        formatted = formatter.format(record)
        log_obj = json.loads(formatted)

        assert log_obj["symbol"] == "BTCUSDT"
        assert log_obj["price"] == 50000
        assert log_obj["quantity"] == 0.1


class TestTradingContextFilter:
    """Test trading context filter"""

    def test_trading_context_filter(self):
        """Test adding trading context to log records"""
        context = {"symbol": "BTCUSDT", "strategy": "ensemble", "mode": "paper"}

        filter_obj = TradingContextFilter(context)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = filter_obj.filter(record)

        assert result is True
        assert record.symbol == "BTCUSDT"
        assert record.strategy == "ensemble"
        assert record.mode == "paper"


class TestSetupLogging:
    """Test logging setup function"""

    @patch("pathlib.Path.mkdir")
    def test_setup_logging_basic(self, mock_mkdir):
        """Test basic logging setup"""
        logger = setup_logging(
            app_name="TEST",
            log_level="INFO",
            enable_file_logging=False,
            enable_json_logging=False,
        )

        assert logger.name == "TEST"
        assert logger.getEffectiveLevel() == logging.INFO

        # Should have at least console handler
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) >= 1

    @patch("pathlib.Path.mkdir")
    @patch("logging.handlers.RotatingFileHandler")
    def test_setup_logging_with_files(self, mock_file_handler, mock_mkdir):
        """Test logging setup with file handlers"""
        mock_handler = MagicMock()
        mock_file_handler.return_value = mock_handler

        logger = setup_logging(
            app_name="TEST",
            log_level="DEBUG",
            enable_file_logging=True,
            enable_json_logging=False,
        )

        # Should create multiple file handlers
        assert mock_file_handler.call_count >= 3  # main, error, trading

    @patch("pathlib.Path.mkdir")
    def test_setup_logging_with_json(self, mock_mkdir):
        """Test logging setup with JSON formatting"""
        logger = setup_logging(
            app_name="TEST",
            log_level="INFO",
            enable_file_logging=False,
            enable_json_logging=True,
        )

        # Console handler should have JSON formatter
        root_logger = logging.getLogger()
        console_handler = root_logger.handlers[0]
        assert isinstance(console_handler.formatter, JSONFormatter)

    @patch("pathlib.Path.mkdir")
    @patch("logging.handlers.SocketHandler")
    def test_setup_logging_with_remote(self, mock_socket_handler, mock_mkdir):
        """Test logging setup with remote logging"""
        mock_handler = MagicMock()
        mock_socket_handler.return_value = mock_handler

        logger = setup_logging(
            app_name="TEST",
            log_level="INFO",
            enable_remote_logging=True,
            remote_host="logserver.com",
            remote_port=514,
        )

        mock_socket_handler.assert_called_once_with("logserver.com", 514)

    @patch("pathlib.Path.mkdir")
    def test_setup_logging_with_context(self, mock_mkdir):
        """Test logging setup with trading context"""
        # Temporarily re-enable logging for this test
        original_level = logging.root.disabled
        logging.disable(logging.NOTSET)

        try:
            context = {"symbol": "BTCUSDT", "mode": "live"}

            logger = setup_logging(
                app_name="TEST",
                log_level="INFO",
                enable_file_logging=False,
                trading_context=context,
            )

            # Add test handler after setup since setup_logging clears handlers
            root_logger = logging.getLogger()
            test_handler = MagicMock()
            test_handler.level = (
                logging.INFO
            )  # Set level attribute for logging comparison

            # Add the context filter to our test handler too
            from utils.logger_config import TradingContextFilter

            context_filter = TradingContextFilter(context)
            test_handler.addFilter(context_filter)

            root_logger.addHandler(test_handler)

            logger.info("Test message")

            # Handler should receive call and have context
            test_handler.handle.assert_called()
            # Verify the log record has context
            call_args = test_handler.handle.call_args
            if call_args:
                record = call_args[0][0]
                assert hasattr(record, "symbol")
                assert record.symbol == "BTCUSDT"
                assert record.mode == "live"
        finally:
            # Restore original logging state
            logging.disable(original_level)


class TestConfigureModuleLoggers:
    """Test module logger configuration"""

    def test_configure_module_loggers(self):
        """Test configuring specific module loggers"""
        configure_module_loggers("INFO")

        # Third-party loggers should be set to WARNING or higher
        assert logging.getLogger("urllib3").level >= logging.WARNING
        assert logging.getLogger("websocket").level >= logging.WARNING
        assert logging.getLogger("requests").level >= logging.WARNING

        # Our modules should have appropriate levels
        assert logging.getLogger("trading").level == logging.DEBUG
        assert logging.getLogger("core.models").level == logging.INFO


class TestGetLogger:
    """Test get_logger function"""

    def test_get_logger_basic(self):
        """Test getting basic logger"""
        logger = get_logger("test.module")

        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_context(self):
        """Test getting logger with context"""
        context = {"symbol": "BTCUSDT"}
        logger = get_logger("test.module", context)

        assert isinstance(logger, logging.LoggerAdapter)
        assert logger.extra == context


class TestTradingLogger:
    """Test specialized trading logger"""

    def test_trading_logger_initialization(self):
        """Test TradingLogger initialization"""
        logger = TradingLogger("test", symbol="BTCUSDT", strategy="ensemble")

        assert logger.symbol == "BTCUSDT"
        assert logger.strategy == "ensemble"
        assert logger.logger.name == "trading.test"

    @patch("logging.Logger.info")
    def test_trade_signal_logging(self, mock_info):
        """Test logging trade signals"""
        logger = TradingLogger("test", symbol="BTCUSDT")

        logger.trade_signal(
            signal_type="BUY",
            price=50000,
            confidence=0.85,
            indicators={"rsi": 30, "macd": "bullish"},
        )

        mock_info.assert_called_once()
        call_args = mock_info.call_args
        assert "Trading signal generated: BUY" in call_args[0][0]

        extra = call_args[1]["extra"]
        assert extra["signal_type"] == "BUY"
        assert extra["price"] == 50000
        assert extra["confidence"] == 0.85
        assert extra["indicators"]["rsi"] == 30

    @patch("logging.Logger.info")
    def test_order_placed_logging(self, mock_info):
        """Test logging order placement"""
        logger = TradingLogger("test", symbol="BTCUSDT")

        logger.order_placed(
            order_type="LIMIT_BUY", quantity=0.1, price=49500, order_id="12345"
        )

        mock_info.assert_called_once()
        extra = mock_info.call_args[1]["extra"]
        assert extra["order_type"] == "LIMIT_BUY"
        assert extra["quantity"] == 0.1
        assert extra["order_id"] == "12345"

    @patch("logging.Logger.info")
    def test_order_filled_logging(self, mock_info):
        """Test logging order execution"""
        logger = TradingLogger("test", symbol="BTCUSDT")

        logger.order_filled(
            order_id="12345", fill_price=49480, fill_time="2024-01-01T10:00:00"
        )

        mock_info.assert_called_once()
        extra = mock_info.call_args[1]["extra"]
        assert extra["order_id"] == "12345"
        assert extra["fill_price"] == 49480

    @patch("logging.Logger.error")
    def test_error_logging(self, mock_error):
        """Test logging trading errors"""
        logger = TradingLogger("test", symbol="BTCUSDT")

        logger.error(
            error_type="INSUFFICIENT_BALANCE",
            message="Not enough USDT",
            required=1000,
            available=500,
        )

        mock_error.assert_called_once()
        assert "Trading error: INSUFFICIENT_BALANCE" in mock_error.call_args[0][0]

    @patch("logging.Logger.warning")
    def test_risk_alert_logging(self, mock_warning):
        """Test logging risk alerts"""
        logger = TradingLogger("test", symbol="BTCUSDT")

        logger.risk_alert(
            alert_type="MAX_DRAWDOWN",
            message="Approaching maximum drawdown limit",
            current_drawdown=0.18,
            max_allowed=0.20,
        )

        mock_warning.assert_called_once()
        assert "Risk alert: MAX_DRAWDOWN" in mock_warning.call_args[0][0]
