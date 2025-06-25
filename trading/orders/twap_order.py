"""
TWAP (Time-Weighted Average Price) Order Implementation
Executes large orders over a specified time period to achieve average market price
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
import logging
import math

from .base_order import BaseOrder, OrderStatus, OrderSide, OrderType, TimeInForce


class TWAPOrder(BaseOrder):
    """
    Time-Weighted Average Price (TWAP) Order
    
    Executes a large order by breaking it into smaller time-based intervals,
    aiming to achieve a price close to the time-weighted average market price.
    """
    
    def __init__(self, symbol: str, side: OrderSide, quantity: float,
                 duration_minutes: int, interval_minutes: int = 5,
                 price_limit: Optional[float] = None,
                 participation_rate: float = 0.1,
                 aggressive_on_close: bool = True,
                 start_time: Optional[datetime] = None,
                 time_in_force: TimeInForce = TimeInForce.GTC,
                 exchange: str = None, logger: logging.Logger = None):
        """
        Initialize TWAP order
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Total order quantity
            duration_minutes: Total execution duration in minutes
            interval_minutes: Interval between slices in minutes
            price_limit: Maximum/minimum price limit (None for no limit)
            participation_rate: Market participation rate (0.0-1.0)
            aggressive_on_close: Be more aggressive on final slices
            start_time: When to start execution (None for immediate)
            time_in_force: Time in force
            exchange: Target exchange
            logger: Logger instance
        """
        super().__init__(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.TWAP,
            price=price_limit,
            time_in_force=time_in_force,
            exchange=exchange,
            logger=logger
        )
        
        self.duration_minutes = duration_minutes
        self.interval_minutes = interval_minutes
        self.price_limit = price_limit
        self.participation_rate = max(0.01, min(1.0, participation_rate))
        self.aggressive_on_close = aggressive_on_close
        self.start_time = start_time or datetime.now()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)
        
        # TWAP state
        self.slices_executed = 0
        self.active_slice_id = None
        self.slice_schedule = self._generate_slice_schedule()
        self.completed_slices = []
        self.market_data_history = []
        self.vwap_tracking = []
        
        # Pricing strategy
        self.price_improvement_factor = 0.0002  # 0.02% improvement
        self.max_price_deviation = 0.005  # 0.5% maximum deviation from market
    
    def _generate_slice_schedule(self) -> List[Dict[str, Any]]:
        """Generate schedule of TWAP slices"""
        total_slices = max(1, self.duration_minutes // self.interval_minutes)
        base_slice_quantity = self.quantity / total_slices
        
        schedule = []
        current_time = self.start_time
        
        for i in range(total_slices):
            # Adjust slice size for final slice to account for rounding
            if i == total_slices - 1:
                slice_quantity = self.quantity - sum(s['quantity'] for s in schedule)
            else:
                slice_quantity = base_slice_quantity
            
            # More aggressive sizing towards the end if enabled
            if self.aggressive_on_close and i >= total_slices * 0.8:
                urgency_factor = 1 + (i - total_slices * 0.8) / (total_slices * 0.2) * 0.5
                slice_quantity *= urgency_factor
                slice_quantity = min(slice_quantity, self.quantity - sum(s['quantity'] for s in schedule))
            
            schedule.append({
                'slice_number': i + 1,
                'quantity': round(slice_quantity, 8),
                'scheduled_time': current_time,
                'urgency_factor': 1.0 if not self.aggressive_on_close or i < total_slices * 0.8 
                                else 1 + (i - total_slices * 0.8) / (total_slices * 0.2) * 0.5
            })
            
            current_time += timedelta(minutes=self.interval_minutes)
        
        return schedule
    
    def validate(self) -> bool:
        """Validate TWAP order parameters"""
        if self.quantity <= 0:
            return False
        
        if self.duration_minutes <= 0:
            return False
        
        if self.interval_minutes <= 0 or self.interval_minutes > self.duration_minutes:
            return False
        
        if self.participation_rate <= 0 or self.participation_rate > 1:
            return False
        
        if self.price_limit and self.price_limit <= 0:
            return False
        
        return True
    
    def to_exchange_format(self) -> Dict[str, Any]:
        """Convert to exchange format (for current slice)"""
        current_slice = self.get_current_slice()
        if not current_slice:
            return {}
        
        slice_price = self._calculate_slice_price(current_slice)
        
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'type': 'limit' if slice_price else 'market',
            'quantity': current_slice['quantity'],
            'price': slice_price,
            'timeInForce': 'IOC',  # Immediate or Cancel for TWAP slices
            'twap': True,
            'slice_number': current_slice['slice_number']
        }
    
    def get_current_slice(self) -> Optional[Dict[str, Any]]:
        """Get the current slice that should be executed"""
        now = datetime.now()
        
        # Check if we've started
        if now < self.start_time:
            return None
        
        # Check if we've finished
        if now > self.end_time or self.slices_executed >= len(self.slice_schedule):
            return None
        
        # Find the slice that should be executed now
        for i, slice_info in enumerate(self.slice_schedule[self.slices_executed:], self.slices_executed):
            if now >= slice_info['scheduled_time']:
                return slice_info
        
        return None
    
    def _calculate_slice_price(self, slice_info: Dict[str, Any]) -> Optional[float]:
        """Calculate the price for a TWAP slice"""
        # Use market orders if no price limit
        if not self.price_limit:
            return None
        
        # Get current market price (simplified - would use real market data)
        current_market_price = self._get_current_market_price()
        if not current_market_price:
            return self.price_limit
        
        # Apply participation rate and urgency
        urgency_factor = slice_info.get('urgency_factor', 1.0)
        improvement_factor = self.price_improvement_factor * urgency_factor
        
        if self.side == OrderSide.BUY:
            # For buy orders, slightly increase price for better fill probability
            target_price = current_market_price * (1 + improvement_factor)
            # Respect upper price limit
            slice_price = min(target_price, self.price_limit)
        else:
            # For sell orders, slightly decrease price
            target_price = current_market_price * (1 - improvement_factor)
            # Respect lower price limit
            slice_price = max(target_price, self.price_limit)
        
        return round(slice_price, 8)
    
    def _get_current_market_price(self) -> Optional[float]:
        """Get current market price (placeholder - would integrate with market data)"""
        # This would integrate with real market data feeds
        # For now, return the price limit as a fallback
        return self.price_limit
    
    def update_market_data(self, market_data: Dict[str, Any]):
        """Update market data for TWAP calculations"""
        self.market_data_history.append({
            'timestamp': datetime.now(),
            'price': market_data.get('price'),
            'volume': market_data.get('volume'),
            'bid': market_data.get('bid'),
            'ask': market_data.get('ask')
        })
        
        # Keep only recent data (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.market_data_history = [
            data for data in self.market_data_history 
            if data['timestamp'] > cutoff_time
        ]
    
    def get_next_slice_params(self) -> Optional[Dict[str, Any]]:
        """Get parameters for the next slice to execute"""
        current_slice = self.get_current_slice()
        if not current_slice:
            return None
        
        slice_price = self._calculate_slice_price(current_slice)
        
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'type': 'limit' if slice_price else 'market',
            'quantity': current_slice['quantity'],
            'price': slice_price,
            'timeInForce': 'IOC',
            'metadata': {
                'twap_order_id': self.id,
                'slice_number': current_slice['slice_number'],
                'total_slices': len(self.slice_schedule),
                'urgency_factor': current_slice.get('urgency_factor', 1.0)
            }
        }
    
    def on_slice_submitted(self, exchange_order_id: str, slice_info: Dict[str, Any]):
        """Handle slice submission"""
        self.active_slice_id = exchange_order_id
        
        if self.logger:
            self.logger.info(
                f"TWAP {self.id}: Slice {slice_info['slice_number']}/{len(self.slice_schedule)} "
                f"submitted ({slice_info['quantity']} @ {slice_info.get('price', 'MARKET')})"
            )
    
    def on_slice_filled(self, exchange_order_id: str, fill_data: Dict[str, Any]):
        """Handle slice being filled"""
        if exchange_order_id != self.active_slice_id:
            return
        
        # Record the fill
        fill_quantity = fill_data.get('quantity', 0)
        fill_price = fill_data.get('price', 0)
        
        self.add_fill(
            quantity=fill_quantity,
            price=fill_price,
            fee=fill_data.get('fee', 0.0),
            fill_id=fill_data.get('fill_id'),
            timestamp=fill_data.get('timestamp')
        )
        
        # Record completed slice
        self.completed_slices.append({
            'slice_number': self.slices_executed + 1,
            'order_id': exchange_order_id,
            'quantity': fill_quantity,
            'price': fill_price,
            'timestamp': fill_data.get('timestamp', datetime.now()),
            'execution_time': datetime.now()
        })
        
        # Update VWAP tracking
        self._update_vwap_tracking(fill_quantity, fill_price)
        
        self.slices_executed += 1
        self.active_slice_id = None
        
        if self.logger:
            self.logger.info(
                f"TWAP {self.id}: Slice {self.slices_executed}/{len(self.slice_schedule)} "
                f"filled at {fill_price} (VWAP: {self.get_current_vwap():.6f})"
            )
    
    def on_slice_partial_fill(self, exchange_order_id: str, fill_data: Dict[str, Any]):
        """Handle partial slice fill"""
        if exchange_order_id != self.active_slice_id:
            return
        
        # For TWAP, we typically accept partial fills and move to next slice
        self.on_slice_filled(exchange_order_id, fill_data)
    
    def _update_vwap_tracking(self, quantity: float, price: float):
        """Update VWAP tracking"""
        self.vwap_tracking.append({
            'timestamp': datetime.now(),
            'quantity': quantity,
            'price': price,
            'cumulative_quantity': self.filled_quantity,
            'cumulative_value': sum(t['quantity'] * t['price'] for t in self.vwap_tracking) + quantity * price
        })
    
    def get_current_vwap(self) -> float:
        """Get current volume-weighted average price"""
        if not self.vwap_tracking or self.filled_quantity == 0:
            return 0.0
        
        total_value = sum(t['quantity'] * t['price'] for t in self.vwap_tracking)
        return total_value / self.filled_quantity
    
    def can_execute_slice(self) -> bool:
        """Check if a slice can be executed now"""
        now = datetime.now()
        
        # Check timing
        if now < self.start_time or now > self.end_time:
            return False
        
        # Check if there's an active slice
        if self.active_slice_id:
            return False
        
        # Check if we have more slices
        if self.slices_executed >= len(self.slice_schedule):
            return False
        
        # Check if it's time for the next slice
        current_slice = self.get_current_slice()
        return current_slice is not None
    
    def get_twap_progress(self) -> Dict[str, Any]:
        """Get TWAP execution progress"""
        now = datetime.now()
        time_elapsed = (now - self.start_time).total_seconds() / 60  # minutes
        time_progress = min(100, (time_elapsed / self.duration_minutes) * 100)
        
        return {
            'total_quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'total_slices': len(self.slice_schedule),
            'slices_executed': self.slices_executed,
            'time_elapsed_minutes': time_elapsed,
            'time_remaining_minutes': max(0, self.duration_minutes - time_elapsed),
            'time_progress_percentage': time_progress,
            'quantity_progress_percentage': (self.filled_quantity / self.quantity) * 100,
            'current_vwap': self.get_current_vwap(),
            'estimated_completion': self.end_time.isoformat(),
            'is_behind_schedule': self.slices_executed < (time_progress / 100) * len(self.slice_schedule),
            'next_slice_time': self._get_next_slice_time()
        }
    
    def _get_next_slice_time(self) -> Optional[str]:
        """Get next slice execution time"""
        if self.slices_executed >= len(self.slice_schedule):
            return None
        
        next_slice = self.slice_schedule[self.slices_executed]
        return next_slice['scheduled_time'].isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TWAP order to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            'duration_minutes': self.duration_minutes,
            'interval_minutes': self.interval_minutes,
            'price_limit': self.price_limit,
            'participation_rate': self.participation_rate,
            'aggressive_on_close': self.aggressive_on_close,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'slices_executed': self.slices_executed,
            'completed_slices': self.completed_slices,
            'current_vwap': self.get_current_vwap(),
            'progress': self.get_twap_progress()
        })
        return base_dict
    
    def __str__(self) -> str:
        """String representation of TWAP order"""
        return (f"TWAP {self.side.value.upper()} {self.quantity} {self.symbol} "
                f"over {self.duration_minutes}min (slices: {self.slices_executed}/{len(self.slice_schedule)}, "
                f"VWAP: {self.get_current_vwap():.6f}) ({self.status.value})")


class TWAPOrderManager:
    """
    Manager for TWAP orders
    Handles slice execution and timing coordination
    """
    
    def __init__(self, exchange_manager, logger: logging.Logger = None):
        """
        Initialize TWAP order manager
        
        Args:
            exchange_manager: Exchange manager for order execution
            logger: Logger instance
        """
        self.exchange_manager = exchange_manager
        self.logger = logger
        self.active_twap_orders: Dict[str, TWAPOrder] = {}
        self.slice_order_mapping: Dict[str, str] = {}  # slice_order_id -> twap_id
        self.monitoring_task = None
    
    async def submit_twap_order(self, twap_order: TWAPOrder) -> Dict[str, Any]:
        """
        Submit TWAP order and start slice management
        
        Args:
            twap_order: TWAP order to submit
            
        Returns:
            Dict with submission result
        """
        try:
            if not twap_order.validate():
                return {'success': False, 'error': 'Invalid TWAP order parameters'}
            
            exchange = self.exchange_manager.get_exchange(twap_order.exchange)
            if not exchange:
                return {'success': False, 'error': f'Exchange {twap_order.exchange} not available'}
            
            # Register the TWAP order
            self.active_twap_orders[twap_order.id] = twap_order
            twap_order.update_status(OrderStatus.SUBMITTED)
            
            # Start monitoring if not already running
            if not self.monitoring_task:
                self.monitoring_task = asyncio.create_task(self._monitor_twap_orders())
            
            return {
                'success': True,
                'twap_id': twap_order.id,
                'start_time': twap_order.start_time.isoformat(),
                'end_time': twap_order.end_time.isoformat(),
                'total_slices': len(twap_order.slice_schedule)
            }
            
        except Exception as e:
            self.logger.error(f"Error submitting TWAP order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _submit_slice(self, twap_order: TWAPOrder, exchange) -> Dict[str, Any]:
        """Submit the next slice for a TWAP order"""
        try:
            slice_params = twap_order.get_next_slice_params()
            if not slice_params:
                return {'success': False, 'error': 'No slice to submit'}
            
            # Submit slice order
            result = exchange.create_order(**slice_params)
            slice_order_id = result.get('id')
            
            if slice_order_id:
                # Track slice
                self.slice_order_mapping[slice_order_id] = twap_order.id
                twap_order.on_slice_submitted(slice_order_id, slice_params)
                
                return {
                    'success': True,
                    'twap_id': twap_order.id,
                    'slice_order_id': slice_order_id,
                    'slice_quantity': slice_params['quantity']
                }
            else:
                return {'success': False, 'error': 'Failed to get slice order ID'}
                
        except Exception as e:
            return {'success': False, 'error': f'Slice submission failed: {e}'}
    
    async def _monitor_twap_orders(self):
        """Background task to monitor and execute TWAP orders"""
        while self.active_twap_orders:
            try:
                for twap_id, twap_order in list(self.active_twap_orders.items()):
                    await self._process_twap_order(twap_order)
                
                # Check every 10 seconds
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in TWAP monitoring: {e}")
                await asyncio.sleep(30)
        
        self.monitoring_task = None
    
    async def _process_twap_order(self, twap_order: TWAPOrder):
        """Process a single TWAP order"""
        try:
            # Check if order is complete or expired
            now = datetime.now()
            if (twap_order.status == OrderStatus.FILLED or 
                now > twap_order.end_time or
                twap_order.slices_executed >= len(twap_order.slice_schedule)):
                
                if twap_order.status != OrderStatus.FILLED:
                    twap_order.update_status(OrderStatus.FILLED)
                
                del self.active_twap_orders[twap_order.id]
                return
            
            # Check if we can execute next slice
            if twap_order.can_execute_slice():
                exchange = self.exchange_manager.get_exchange(twap_order.exchange)
                if exchange:
                    result = await self._submit_slice(twap_order, exchange)
                    if not result['success']:
                        self.logger.warning(f"Failed to submit slice for {twap_order.id}: {result['error']}")
            
        except Exception as e:
            self.logger.error(f"Error processing TWAP order {twap_order.id}: {e}")
    
    def handle_order_update(self, exchange_order_id: str, update_data: Dict[str, Any]):
        """Handle order updates from exchange"""
        twap_id = self.slice_order_mapping.get(exchange_order_id)
        if not twap_id:
            return
        
        twap_order = self.active_twap_orders.get(twap_id)
        if not twap_order:
            return
        
        # Handle slice updates
        status = update_data.get('status')
        if status == 'filled':
            twap_order.on_slice_filled(exchange_order_id, update_data)
        elif status == 'partially_filled':
            twap_order.on_slice_partial_fill(exchange_order_id, update_data)
        
        # Remove slice mapping when done
        if status in ['filled', 'cancelled', 'rejected']:
            del self.slice_order_mapping[exchange_order_id]
    
    async def cancel_twap_order(self, twap_id: str) -> Dict[str, Any]:
        """Cancel an active TWAP order"""
        try:
            twap_order = self.active_twap_orders.get(twap_id)
            if not twap_order:
                return {'success': False, 'error': 'TWAP order not found'}
            
            exchange = self.exchange_manager.get_exchange(twap_order.exchange)
            if not exchange:
                return {'success': False, 'error': 'Exchange not available'}
            
            # Cancel active slice
            if twap_order.active_slice_id:
                try:
                    exchange.cancel_order(twap_order.active_slice_id)
                except Exception as e:
                    self.logger.warning(f"Failed to cancel active slice: {e}")
            
            twap_order.cancel("User cancelled")
            del self.active_twap_orders[twap_id]
            
            return {
                'success': True,
                'twap_id': twap_id,
                'slices_completed': twap_order.slices_executed,
                'quantity_filled': twap_order.filled_quantity
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling TWAP order: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_active_twap_orders(self) -> List[Dict[str, Any]]:
        """Get list of active TWAP orders"""
        return [order.to_dict() for order in self.active_twap_orders.values()]