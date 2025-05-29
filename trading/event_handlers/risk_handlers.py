"""
Risk event handlers for the ELVIS Trading Bot.
"""

import logging
from datetime import datetime

from core.events.decorators import (
    event_handler,
    priority_event_handler,
    async_event_handler
)
from core.events.event_types import (
    RiskEvent,
    OrderEvent,
    NotificationEvent,
    SystemEvent
)
from core.events import event_bus
from core.di import container


logger = logging.getLogger(__name__)


@priority_event_handler('risk', priority=20)
def handle_critical_risk(event: RiskEvent):
    """Handle critical risk events with highest priority."""
    if event.severity != 'critical':
        return
        
    try:
        logger.critical(f"CRITICAL RISK: {event.risk_type} - {event.action_taken}")
        
        # Take immediate action based on risk type
        if event.risk_type == 'drawdown':
            # Stop all trading
            strategy = container.get('strategy')
            strategy.stop_trading()
            
            # Cancel all open orders
            executor = container.get('executor')
            executor.cancel_all_orders()
            
            # Publish system event
            system_event = SystemEvent(
                system_type='error',
                component='risk_manager',
                status='critical',
                message=f"Trading halted due to critical drawdown",
                source='risk_handler'
            )
            event_bus.publish(system_event)
            
        elif event.risk_type == 'position_size':
            # Close oversized positions
            for symbol in event.affected_positions:
                logger.warning(f"Reducing position size for {symbol}")
                # Implementation would reduce position
                
    except Exception as e:
        logger.error(f"Failed to handle critical risk: {e}")


@event_handler('risk')
def log_risk_events(event: RiskEvent):
    """Log all risk events for compliance and analysis."""
    try:
        risk_log = {
            'timestamp': event.timestamp,
            'risk_type': event.risk_type,
            'severity': event.severity,
            'current_value': event.current_value,
            'threshold_value': event.threshold_value,
            'action_taken': event.action_taken,
            'affected_positions': event.affected_positions
        }
        
        # Log to specialized risk log file
        risk_logger = logging.getLogger('risk_audit')
        risk_logger.info(f"RISK_EVENT: {risk_log}")
        
        # Store in database for compliance
        # In production, this would write to a database
        
    except Exception as e:
        logger.error(f"Failed to log risk event: {e}")


@async_event_handler('risk')
async def notify_risk_alerts(event: RiskEvent):
    """Send notifications for risk alerts."""
    # Only notify for warning and critical events
    if event.severity not in ['warning', 'critical']:
        return
        
    try:
        notifier = container.get('notifier')
        
        # Create notification message
        emoji = '⚠️' if event.severity == 'warning' else '🚨'
        message = (
            f"{emoji} Risk Alert: {event.severity.upper()} {emoji}\n"
            f"Type: {event.risk_type}\n"
            f"Current: {event.current_value:.2f}\n"
            f"Threshold: {event.threshold_value:.2f}\n"
            f"Action: {event.action_taken}\n"
        )
        
        if event.affected_positions:
            message += f"Affected: {', '.join(event.affected_positions)}\n"
            
        # Send notification
        await notifier.send_async(message, priority='high')
        
        # Create notification event
        notification_event = NotificationEvent(
            notification_type='alert',
            title=f'{event.severity.upper()} Risk Alert',
            message=message,
            channels=['telegram', 'email'] if event.severity == 'critical' else ['telegram'],
            source='risk_handler'
        )
        event_bus.publish(notification_event)
        
    except Exception as e:
        logger.error(f"Failed to send risk notification: {e}")


@event_handler('risk')
def update_risk_metrics(event: RiskEvent):
    """Update risk metrics for monitoring."""
    try:
        from utils.monitoring import push_metric_to_prometheus
        
        # Push risk event counter
        push_metric_to_prometheus(
            'risk_events_total',
            1,
            labels={
                'risk_type': event.risk_type,
                'severity': event.severity
            }
        )
        
        # Push current risk values
        push_metric_to_prometheus(
            f'risk_{event.risk_type}_current',
            event.current_value,
            labels={'severity': event.severity}
        )
        
        # Push threshold values
        push_metric_to_prometheus(
            f'risk_{event.risk_type}_threshold',
            event.threshold_value,
            labels={'severity': event.severity}
        )
        
    except Exception as e:
        logger.error(f"Failed to update risk metrics: {e}")


@event_handler('risk')
def adjust_trading_parameters(event: RiskEvent):
    """Adjust trading parameters based on risk events."""
    try:
        risk_manager = container.get('risk_manager')
        
        # Adjust parameters based on risk type and severity
        if event.risk_type == 'volatility':
            if event.severity == 'warning':
                # Reduce position sizes by 25%
                risk_manager.adjust_position_multiplier(0.75)
                logger.warning("Reduced position sizes due to high volatility")
                
            elif event.severity == 'critical':
                # Reduce position sizes by 50%
                risk_manager.adjust_position_multiplier(0.5)
                # Widen stop losses
                risk_manager.adjust_stop_loss_multiplier(1.5)
                logger.warning("Significantly reduced risk due to extreme volatility")
                
        elif event.risk_type == 'drawdown':
            if event.current_value > 0.1:  # 10% drawdown
                # Reduce risk taking
                risk_manager.adjust_max_risk_per_trade(0.5)
                logger.warning("Reduced risk per trade due to drawdown")
                
        elif event.risk_type == 'exposure':
            if event.severity in ['warning', 'critical']:
                # Prevent new positions
                risk_manager.set_new_positions_allowed(False)
                logger.warning("New positions disabled due to high exposure")
                
    except Exception as e:
        logger.error(f"Failed to adjust trading parameters: {e}")


@priority_event_handler('risk', priority=15)
def emergency_stop_loss(event: RiskEvent):
    """Execute emergency stop loss for critical position risks."""
    if event.risk_type != 'position_size' or event.severity != 'critical':
        return
        
    try:
        executor = container.get('executor')
        
        for symbol in event.affected_positions:
            # Get current position
            position = executor.get_position(symbol)
            if not position or position['quantity'] == 0:
                continue
                
            # Create emergency sell order
            order_event = OrderEvent(
                order_id=f"EMERGENCY_{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                order_type='market',
                side='sell',
                quantity=abs(position['quantity']),
                source='risk_handler',
                metadata={'reason': 'emergency_stop_loss', 'risk_event': event.event_id}
            )
            
            event_bus.publish(order_event)
            logger.critical(f"Emergency stop loss triggered for {symbol}")
            
    except Exception as e:
        logger.error(f"Failed to execute emergency stop loss: {e}")
