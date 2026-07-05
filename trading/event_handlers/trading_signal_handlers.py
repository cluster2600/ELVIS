"""
Trading signal event handlers for the ELVIS Trading Bot.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from core.di import container
from core.events import event_bus
from core.events.decorators import (
    async_event_handler,
    conditional_event_handler,
    event_handler,
    priority_event_handler,
)
from core.events.event_types import (
    NotificationEvent,
    OrderEvent,
    RiskEvent,
    TradingSignalEvent,
)

logger = logging.getLogger(__name__)


@priority_event_handler("trading_signal", priority=10)
def validate_trading_signal(event: TradingSignalEvent):
    """Validate trading signals before processing (high priority)."""
    try:
        # Basic validation
        if event.strength < 0 or event.strength > 1:
            logger.error(f"Invalid signal strength: {event.strength}")
            return

        if event.signal_type not in ["buy", "sell", "hold"]:
            logger.error(f"Invalid signal type: {event.signal_type}")
            return

        if event.price <= 0:
            logger.error(f"Invalid price: {event.price}")
            return

        logger.info(
            f"Valid {event.signal_type} signal for {event.symbol} "
            f"from {event.strategy_name} with strength {event.strength:.2f}"
        )

    except Exception as e:
        logger.error(f"Failed to validate trading signal: {e}")


@conditional_event_handler(
    "trading_signal", lambda e: e.signal_type != "hold" and e.strength > 0.7
)
def execute_strong_signals(event: TradingSignalEvent):
    """Execute strong buy/sell signals (strength > 0.7)."""
    try:
        executor = container.get("executor")
        risk_manager = container.get("risk_manager")

        # Check with risk manager
        if not risk_manager.check_signal_allowed(event):
            logger.warning(f"Risk manager rejected signal for {event.symbol}")
            return

        # Calculate position size
        position_size = risk_manager.calculate_position_size(
            event.symbol, event.price, event.strength
        )

        if position_size <= 0:
            logger.warning(f"Position size too small for {event.symbol}")
            return

        # Create order event
        order_event = OrderEvent(
            order_id=f"{event.symbol}_{datetime.now().timestamp()}",
            symbol=event.symbol,
            order_type="market",
            side=event.signal_type,
            quantity=position_size,
            price=event.price,
            source="trading_signal_handler",
            metadata={
                "signal_strength": event.strength,
                "strategy": event.strategy_name,
                "reasons": event.reasons,
            },
        )

        # Publish order event
        event_bus.publish(order_event)

        logger.info(
            f"Created {event.signal_type} order for {position_size} "
            f"{event.symbol} @ ${event.price:.2f}"
        )

    except Exception as e:
        logger.error(f"Failed to execute trading signal: {e}")


@async_event_handler("trading_signal")
async def notify_trading_signals(event: TradingSignalEvent):
    """Send notifications for trading signals."""
    try:
        notifier = container.get("notifier")

        # Only notify for actual trades (not hold signals)
        if event.signal_type == "hold":
            return

        # Create notification message
        message = (
            f"🚨 Trading Signal Alert 🚨\n"
            f"Symbol: {event.symbol}\n"
            f"Signal: {event.signal_type.upper()}\n"
            f"Strength: {event.strength:.2%}\n"
            f"Price: ${event.price:.2f}\n"
            f"Strategy: {event.strategy_name}\n"
        )

        if event.reasons:
            message += f"Reasons: {', '.join(event.reasons)}\n"

        if event.stop_loss:
            message += f"Stop Loss: ${event.stop_loss:.2f}\n"

        if event.take_profit:
            message += f"Take Profit: ${event.take_profit:.2f}\n"

        # Send notification
        await notifier.send_async(message)

        # Also create notification event for logging
        notification_event = NotificationEvent(
            notification_type="trade",
            title="Trading Signal",
            message=message,
            channels=["telegram"],
            source="trading_signal_handler",
        )
        event_bus.publish(notification_event)

    except Exception as e:
        logger.error(f"Failed to send trading signal notification: {e}")


@event_handler("trading_signal")
def log_trading_signals(event: TradingSignalEvent):
    """Log all trading signals for analysis."""
    try:
        trading_logger = container.get("trading_logger")

        log_data = {
            "timestamp": event.timestamp,
            "symbol": event.symbol,
            "signal_type": event.signal_type,
            "strength": event.strength,
            "strategy": event.strategy_name,
            "price": event.price,
            "quantity": event.quantity,
            "stop_loss": event.stop_loss,
            "take_profit": event.take_profit,
            "reasons": event.reasons,
        }

        trading_logger.log_signal(log_data)

    except Exception as e:
        logger.error(f"Failed to log trading signal: {e}")


@event_handler("trading_signal")
def update_strategy_metrics(event: TradingSignalEvent):
    """Update strategy performance metrics."""
    try:
        # Get metrics from container
        metrics = container.get_optional("strategy_metrics", {})

        # Initialize strategy metrics if needed
        if event.strategy_name not in metrics:
            metrics[event.strategy_name] = {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "hold_signals": 0,
                "avg_strength": 0.0,
                "last_signal_time": None,
            }

        strategy_metrics = metrics[event.strategy_name]

        # Update counters
        strategy_metrics["total_signals"] += 1
        strategy_metrics[f"{event.signal_type}_signals"] += 1

        # Update average strength
        current_avg = strategy_metrics["avg_strength"]
        total = strategy_metrics["total_signals"]
        strategy_metrics["avg_strength"] = (
            current_avg * (total - 1) + event.strength
        ) / total

        strategy_metrics["last_signal_time"] = event.timestamp

        # Push metrics to Prometheus
        from utils.monitoring import push_metric_to_prometheus

        push_metric_to_prometheus(
            f"strategy_signals_total",
            strategy_metrics["total_signals"],
            labels={"strategy": event.strategy_name, "signal_type": event.signal_type},
        )

        push_metric_to_prometheus(
            f"strategy_signal_strength_avg",
            strategy_metrics["avg_strength"],
            labels={"strategy": event.strategy_name},
        )

    except Exception as e:
        logger.error(f"Failed to update strategy metrics: {e}")


@conditional_event_handler("trading_signal", lambda e: e.signal_type in ["buy", "sell"])
def check_risk_limits(event: TradingSignalEvent):
    """Check risk limits before executing trades."""
    try:
        risk_manager = container.get("risk_manager")

        # Check various risk limits
        checks = {
            "position_limit": risk_manager.check_position_limit(event.symbol),
            "exposure_limit": risk_manager.check_exposure_limit(),
            "daily_loss_limit": risk_manager.check_daily_loss_limit(),
            "correlation_limit": risk_manager.check_correlation_limit(event.symbol),
        }

        # If any check fails, publish risk event
        for check_name, passed in checks.items():
            if not passed:
                risk_event = RiskEvent(
                    risk_type=check_name,
                    severity="warning",
                    current_value=0.0,  # Would be filled with actual values
                    threshold_value=0.0,  # Would be filled with actual limits
                    action_taken=f"Blocked {event.signal_type} signal",
                    affected_positions=[event.symbol],
                    source="trading_signal_handler",
                )
                event_bus.publish(risk_event)

                logger.warning(f"Risk check '{check_name}' failed for {event.symbol}")
                return

        logger.info(f"All risk checks passed for {event.symbol}")

    except Exception as e:
        logger.error(f"Failed to check risk limits: {e}")
