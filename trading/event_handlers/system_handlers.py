"""
System event handlers for the ELVIS Trading Bot.
"""

import logging
import os
from datetime import datetime

import psutil

from core.di import container
from core.events import event_bus
from core.events.decorators import (
    async_event_handler,
    event_handler,
    throttled_event_handler,
)
from core.events.event_types import NotificationEvent, SystemEvent

logger = logging.getLogger(__name__)


@event_handler("system")
def log_system_events(event: SystemEvent):
    """Log all system events."""
    try:
        log_method = {
            "ok": logger.info,
            "warning": logger.warning,
            "error": logger.error,
            "critical": logger.critical,
        }.get(event.status, logger.info)

        log_method(f"[{event.component}] {event.message}")

        # Log metrics if available
        if event.metrics:
            logger.debug(f"System metrics: {event.metrics}")

    except Exception as e:
        logger.error(f"Failed to log system event: {e}")


@event_handler("system")
def handle_startup_events(event: SystemEvent):
    """Handle system startup events."""
    if event.system_type != "startup":
        return

    try:
        logger.info("=" * 50)
        logger.info(f"System component started: {event.component}")
        logger.info(f"Status: {event.status}")
        logger.info("=" * 50)

        # Initialize monitoring for this component
        from utils.monitoring import push_metric_to_prometheus

        push_metric_to_prometheus(
            "component_startup_total",
            1,
            labels={"component": event.component, "status": event.status},
        )

    except Exception as e:
        logger.error(f"Failed to handle startup event: {e}")


@event_handler("system")
def handle_shutdown_events(event: SystemEvent):
    """Handle system shutdown events."""
    if event.system_type != "shutdown":
        return

    try:
        logger.info(f"Shutting down component: {event.component}")

        # Perform cleanup actions based on component
        if event.component == "strategy":
            # Cancel any pending orders
            executor = container.get_optional("executor")
            if executor:
                executor.cancel_all_orders()

        elif event.component == "data_feed":
            # Close data connections
            price_fetcher = container.get_optional("price_fetcher")
            if price_fetcher and hasattr(price_fetcher, "close"):
                price_fetcher.close()

    except Exception as e:
        logger.error(f"Failed to handle shutdown event: {e}")


@throttled_event_handler("system", min_interval_ms=60000)  # Once per minute
def monitor_system_health(event: SystemEvent):
    """Monitor system health and resources."""
    if event.system_type != "health":
        return

    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Check thresholds
        health_status = "ok"
        warnings = []

        if cpu_percent > 80:
            health_status = "warning"
            warnings.append(f"High CPU usage: {cpu_percent}%")

        if memory.percent > 85:
            health_status = "warning"
            warnings.append(f"High memory usage: {memory.percent}%")

        if disk.percent > 90:
            health_status = "critical"
            warnings.append(f"Low disk space: {disk.percent}% used")

        # Log health status
        if warnings:
            logger.warning(f"System health issues: {', '.join(warnings)}")

        # Update metrics
        from utils.monitoring import push_metric_to_prometheus

        push_metric_to_prometheus("system_cpu_usage_percent", cpu_percent)
        push_metric_to_prometheus("system_memory_usage_percent", memory.percent)
        push_metric_to_prometheus("system_disk_usage_percent", disk.percent)

        # Send alert if critical
        if health_status == "critical":
            notification_event = NotificationEvent(
                notification_type="system",
                title="System Health Alert",
                message=f"System health critical: {', '.join(warnings)}",
                channels=["telegram", "email"],
                source="system_handler",
            )
            event_bus.publish(notification_event)

    except Exception as e:
        logger.error(f"Failed to monitor system health: {e}")


@async_event_handler("system")
async def handle_error_events(event: SystemEvent):
    """Handle system error events."""
    if event.system_type != "error":
        return

    try:
        # Log error details
        logger.error(f"System error in {event.component}: {event.message}")

        # Send notification for critical errors
        if event.status == "critical":
            notifier = container.get("notifier")

            message = (
                f"🚨 CRITICAL SYSTEM ERROR 🚨\n"
                f"Component: {event.component}\n"
                f"Error: {event.message}\n"
                f"Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

            await notifier.send_async(message, priority="critical")

        # Store error in error log
        error_log = {
            "timestamp": event.timestamp,
            "component": event.component,
            "status": event.status,
            "message": event.message,
            "metrics": event.metrics,
        }

        # In production, this would write to an error database
        error_logger = logging.getLogger("error_log")
        error_logger.error(f"SYSTEM_ERROR: {error_log}")

    except Exception as e:
        logger.error(f"Failed to handle error event: {e}")


@event_handler("system")
def update_system_metrics(event: SystemEvent):
    """Update system metrics for monitoring."""
    try:
        from utils.monitoring import push_metric_to_prometheus

        # Count system events by type
        push_metric_to_prometheus(
            "system_events_total",
            1,
            labels={
                "type": event.system_type,
                "component": event.component,
                "status": event.status,
            },
        )

        # Track component status
        status_value = {"ok": 1, "warning": 0.5, "error": 0.25, "critical": 0}.get(
            event.status, 0.5
        )

        push_metric_to_prometheus(
            f"component_status", status_value, labels={"component": event.component}
        )

    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")


@event_handler("system")
def backup_system_state(event: SystemEvent):
    """Backup system state periodically."""
    if event.system_type != "backup":
        return

    try:
        logger.info("Starting system state backup...")

        # Get components to backup
        strategy = container.get_optional("strategy")
        risk_manager = container.get_optional("risk_manager")

        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": container.get_optional("app_config", {}).get("mode", "unknown"),
            "positions": {},
            "risk_state": {},
            "strategy_state": {},
        }

        # Backup positions
        if strategy and hasattr(strategy, "get_positions"):
            backup_data["positions"] = strategy.get_positions()

        # Backup risk state
        if risk_manager and hasattr(risk_manager, "get_state"):
            backup_data["risk_state"] = risk_manager.get_state()

        # Backup strategy state
        if strategy and hasattr(strategy, "get_state"):
            backup_data["strategy_state"] = strategy.get_state()

        # Save backup (in production, this would save to S3 or similar)
        import json

        backup_dir = "backups"
        os.makedirs(backup_dir, exist_ok=True)

        backup_file = os.path.join(
            backup_dir, f"system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)

        logger.info(f"System state backed up to {backup_file}")

    except Exception as e:
        logger.error(f"Failed to backup system state: {e}")
