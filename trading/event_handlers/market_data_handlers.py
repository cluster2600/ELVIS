"""
Market data event handlers for the ELVIS Trading Bot.
"""

import logging
from typing import List

from core.events.decorators import (
    event_handler,
    async_event_handler,
    throttled_event_handler,
    buffered_event_handler
)
from core.events.event_types import MarketDataEvent, PerformanceEvent
from core.di import container
from utils.monitoring import push_metric_to_prometheus


logger = logging.getLogger(__name__)


@event_handler('market_data')
def log_price_update(event: MarketDataEvent):
    """Log price updates for debugging."""
    logger.debug(f"Price update for {event.symbol}: ${event.price:.2f}")


@throttled_event_handler('market_data', min_interval_ms=5000)
def update_dashboard_price(event: MarketDataEvent):
    """Update dashboard with price data (throttled to every 5 seconds)."""
    try:
        dashboard = container.get_optional('dashboard')
        if dashboard:
            # Update dashboard price display
            dashboard.update_price(event.symbol, event.price, event.volume)
    except Exception as e:
        logger.error(f"Failed to update dashboard: {e}")


@async_event_handler('market_data')
async def cache_market_data(event: MarketDataEvent):
    """Cache market data in Redis for historical analysis."""
    try:
        redis_cache = container.get('redis_cache')
        cache_key = f"market_data:{event.symbol}:{event.timestamp.timestamp()}"
        
        await redis_cache.set_async(cache_key, {
            'price': event.price,
            'volume': event.volume,
            'bid': event.bid,
            'ask': event.ask,
            'indicators': event.indicators
        }, ttl=3600)  # 1 hour TTL
        
    except Exception as e:
        logger.error(f"Failed to cache market data: {e}")


@buffered_event_handler('market_data', buffer_size=100, flush_interval_ms=10000)
def store_market_data_batch(events: List[MarketDataEvent]):
    """Store batches of market data for efficiency."""
    logger.info(f"Storing batch of {len(events)} market data events")
    
    try:
        # In a real implementation, this would bulk insert to a database
        # For now, we'll just log the batch
        for event in events:
            logger.debug(f"Batch item: {event.symbol} @ ${event.price:.2f}")
            
        # Push metrics
        push_metric_to_prometheus(
            'market_data_batch_size',
            len(events),
            labels={'handler': 'store_market_data_batch'}
        )
        
    except Exception as e:
        logger.error(f"Failed to store market data batch: {e}")


@event_handler('market_data')
def calculate_technical_indicators(event: MarketDataEvent):
    """Calculate and update technical indicators based on market data."""
    try:
        # Check if we have indicator data
        if not event.indicators:
            return
            
        # Extract key indicators
        rsi = event.indicators.get('rsi')
        macd = event.indicators.get('macd')
        sma_20 = event.indicators.get('sma_20')
        
        # Log significant indicator values
        if rsi and (rsi < 30 or rsi > 70):
            logger.info(f"{event.symbol} RSI extreme: {rsi:.2f}")
            
        if macd and macd.get('histogram'):
            histogram = macd['histogram']
            if abs(histogram) > 10:  # Significant MACD divergence
                logger.info(f"{event.symbol} MACD histogram: {histogram:.2f}")
                
    except Exception as e:
        logger.error(f"Failed to process technical indicators: {e}")


@event_handler('market_data')
def monitor_price_volatility(event: MarketDataEvent):
    """Monitor price volatility and alert on high volatility."""
    try:
        redis_cache = container.get('redis_cache')
        
        # Get recent prices from cache
        cache_key_pattern = f"market_data:{event.symbol}:*"
        recent_prices = []
        
        # In a real implementation, we'd query recent prices from cache
        # For now, we'll use the current price
        
        # Calculate volatility metrics
        if event.indicators and 'volatility' in event.indicators:
            volatility = event.indicators['volatility']
            
            if volatility > 0.05:  # 5% volatility threshold
                logger.warning(f"High volatility detected for {event.symbol}: {volatility:.2%}")
                
                # Publish performance event
                from core.events import event_bus
                perf_event = PerformanceEvent(
                    metric_type='volatility',
                    value=volatility,
                    period='1h',
                    source='market_data_handler',
                    details={'symbol': event.symbol, 'price': event.price}
                )
                event_bus.publish(perf_event)
                
    except Exception as e:
        logger.error(f"Failed to monitor volatility: {e}")


@async_event_handler('market_data')
async def update_price_metrics(event: MarketDataEvent):
    """Update Prometheus metrics with price data."""
    try:
        # Push price metric
        push_metric_to_prometheus(
            f'{event.symbol.lower()}_price',
            event.price,
            labels={'exchange': 'binance', 'pair': event.symbol}
        )
        
        # Push volume metric
        push_metric_to_prometheus(
            f'{event.symbol.lower()}_volume',
            event.volume,
            labels={'exchange': 'binance', 'pair': event.symbol}
        )
        
        # Push spread metric if available
        if event.bid and event.ask:
            spread = event.ask - event.bid
            spread_percentage = (spread / event.price) * 100
            
            push_metric_to_prometheus(
                f'{event.symbol.lower()}_spread_percentage',
                spread_percentage,
                labels={'exchange': 'binance', 'pair': event.symbol}
            )
            
    except Exception as e:
        logger.error(f"Failed to update price metrics: {e}")
