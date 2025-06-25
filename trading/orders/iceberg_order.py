"""
Iceberg Order Implementation
Breaks large orders into smaller chunks to hide order size and reduce market impact
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
import logging
import random

from .base_order import BaseOrder, OrderStatus, OrderSide, OrderType, TimeInForce


class IcebergOrder(BaseOrder):
    """
    Iceberg Order
    
    Breaks a large order into smaller visible chunks, only showing one chunk
    at a time to the market. When a chunk is filled, the next chunk is submitted.
    """
    
    def __init__(self, symbol: str, side: OrderSide, quantity: float,
                 price: float, chunk_size: float,
                 chunk_variance: float = 0.0,
                 delay_between_chunks: int = 0,
                 price_improvement: bool = True,
                 time_in_force: TimeInForce = TimeInForce.GTC,
                 exchange: str = None, logger: logging.Logger = None):
        """
        Initialize Iceberg order
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Total order quantity
            price: Base price for limit orders
            chunk_size: Size of each visible chunk
            chunk_variance: Random variance in chunk size (0.0-1.0)
            delay_between_chunks: Delay between chunks in seconds
            price_improvement: Whether to improve price between chunks
            time_in_force: Time in force
            exchange: Target exchange
            logger: Logger instance
        """
        super().__init__(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.ICEBERG,
            price=price,
            time_in_force=time_in_force,
            exchange=exchange,
            logger=logger
        )
        
        self.chunk_size = chunk_size
        self.chunk_variance = max(0.0, min(1.0, chunk_variance))
        self.delay_between_chunks = delay_between_chunks
        self.price_improvement = price_improvement
        
        # Iceberg state
        self.chunks_submitted = 0
        self.chunks_filled = 0
        self.active_chunk_id = None
        self.active_chunk_quantity = 0.0
        self.completed_chunks = []
        self.next_chunk_time = None
        
        # Calculate number of chunks
        self.total_chunks = self._calculate_total_chunks()
        self.chunk_schedule = self._generate_chunk_schedule()
        
        # Price tracking for improvement
        self.last_market_price = price
        self.price_improvement_factor = 0.0001  # 0.01% improvement
    
    def _calculate_total_chunks(self) -> int:
        """Calculate total number of chunks needed"""
        return max(1, int(self.quantity / self.chunk_size)) + (1 if self.quantity % self.chunk_size > 0 else 0)
    
    def _generate_chunk_schedule(self) -> List[Dict[str, Any]]:
        """Generate schedule of chunk sizes and timings"""
        schedule = []
        remaining_quantity = self.quantity
        
        for i in range(self.total_chunks):
            # Calculate chunk size with variance
            base_chunk = min(self.chunk_size, remaining_quantity)
            
            if self.chunk_variance > 0 and i < self.total_chunks - 1:
                # Add random variance (but ensure we don't exceed remaining)
                variance = random.uniform(-self.chunk_variance, self.chunk_variance)
                chunk_size = base_chunk * (1 + variance)
                chunk_size = max(0.001, min(chunk_size, remaining_quantity))
            else:
                chunk_size = base_chunk
            
            schedule.append({
                'chunk_number': i + 1,
                'quantity': round(chunk_size, 8),
                'delay': self.delay_between_chunks if i > 0 else 0
            })
            
            remaining_quantity -= chunk_size
            
            if remaining_quantity <= 0.001:
                break
        
        return schedule
    
    def validate(self) -> bool:
        """Validate Iceberg order parameters"""
        if self.quantity <= 0:
            return False
        
        if self.chunk_size <= 0 or self.chunk_size > self.quantity:
            return False
        
        if self.price <= 0:
            return False
        
        if self.chunk_variance < 0 or self.chunk_variance > 1:
            return False
        
        return True
    
    def to_exchange_format(self) -> Dict[str, Any]:
        """Convert to exchange format (for current chunk)"""
        if not self.chunk_schedule or self.chunks_submitted >= len(self.chunk_schedule):
            return {}
        
        current_chunk = self.chunk_schedule[self.chunks_submitted]
        current_price = self._get_current_chunk_price()
        
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'type': 'limit',
            'quantity': current_chunk['quantity'],
            'price': current_price,
            'timeInForce': self.time_in_force.value.upper(),
            'iceberg': True,
            'icebergQty': current_chunk['quantity']
        }
    
    def _get_current_chunk_price(self) -> float:
        """Get price for current chunk with potential improvement"""
        if not self.price_improvement:
            return self.price
        
        # Simple price improvement logic
        if self.side == OrderSide.BUY:
            # For buy orders, slightly increase price to improve fill probability
            improved_price = self.price * (1 + self.price_improvement_factor)
        else:
            # For sell orders, slightly decrease price
            improved_price = self.price * (1 - self.price_improvement_factor)
        
        return round(improved_price, 8)
    
    def get_next_chunk_params(self) -> Optional[Dict[str, Any]]:
        """Get parameters for the next chunk to submit"""
        if self.chunks_submitted >= len(self.chunk_schedule):
            return None
        
        chunk_info = self.chunk_schedule[self.chunks_submitted]
        
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'type': 'limit',
            'quantity': chunk_info['quantity'],
            'price': self._get_current_chunk_price(),
            'timeInForce': self.time_in_force.value.upper(),
            'metadata': {
                'iceberg_order_id': self.id,
                'chunk_number': chunk_info['chunk_number'],
                'total_chunks': self.total_chunks
            }
        }
    
    def on_chunk_submitted(self, exchange_order_id: str, chunk_quantity: float):
        """Handle chunk submission"""
        self.active_chunk_id = exchange_order_id
        self.active_chunk_quantity = chunk_quantity
        self.chunks_submitted += 1
        
        # Set next chunk time if there's a delay
        if self.delay_between_chunks > 0:
            self.next_chunk_time = datetime.now() + timedelta(seconds=self.delay_between_chunks)
        
        if self.logger:
            self.logger.info(
                f"Iceberg {self.id}: Chunk {self.chunks_submitted}/{self.total_chunks} "
                f"submitted ({chunk_quantity} @ {self._get_current_chunk_price()})"
            )
    
    def on_chunk_filled(self, exchange_order_id: str, fill_data: Dict[str, Any]):
        """Handle chunk being filled"""
        if exchange_order_id != self.active_chunk_id:
            return
        
        # Record the fill
        self.add_fill(
            quantity=fill_data.get('quantity', self.active_chunk_quantity),
            price=fill_data.get('price', self.price),
            fee=fill_data.get('fee', 0.0),
            fill_id=fill_data.get('fill_id'),
            timestamp=fill_data.get('timestamp')
        )
        
        # Mark chunk as completed
        self.completed_chunks.append({
            'chunk_number': self.chunks_filled + 1,
            'order_id': exchange_order_id,
            'quantity': fill_data.get('quantity', self.active_chunk_quantity),
            'price': fill_data.get('price', self.price),
            'timestamp': fill_data.get('timestamp', datetime.now())
        })
        
        self.chunks_filled += 1
        self.active_chunk_id = None
        self.active_chunk_quantity = 0.0
        
        if self.logger:
            self.logger.info(
                f"Iceberg {self.id}: Chunk {self.chunks_filled}/{self.total_chunks} "
                f"filled at {fill_data.get('price', self.price)}"
            )
        
        # Update last market price for price improvement
        self.last_market_price = fill_data.get('price', self.price)
    
    def can_submit_next_chunk(self) -> bool:
        """Check if next chunk can be submitted"""
        # Check if there are more chunks
        if self.chunks_submitted >= len(self.chunk_schedule):
            return False
        
        # Check if there's an active chunk
        if self.active_chunk_id:
            return False
        
        # Check delay
        if self.next_chunk_time and datetime.now() < self.next_chunk_time:
            return False
        
        # Check if order is still active
        if not self.is_active():
            return False
        
        return True
    
    def get_iceberg_progress(self) -> Dict[str, Any]:
        """Get progress information"""
        return {
            'total_quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'total_chunks': self.total_chunks,
            'chunks_submitted': self.chunks_submitted,
            'chunks_filled': self.chunks_filled,
            'active_chunk_id': self.active_chunk_id,
            'active_chunk_quantity': self.active_chunk_quantity,
            'progress_percentage': (self.filled_quantity / self.quantity) * 100,
            'estimated_completion': self._estimate_completion_time()
        }
    
    def _estimate_completion_time(self) -> Optional[str]:
        """Estimate completion time based on current progress"""
        if self.chunks_filled == 0:
            return None
        
        # Simple estimation based on average chunk fill time
        avg_chunk_time = (datetime.now() - self.created_at).total_seconds() / self.chunks_filled
        remaining_chunks = self.total_chunks - self.chunks_filled
        estimated_seconds = remaining_chunks * avg_chunk_time
        
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        return estimated_completion.isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Iceberg order to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            'chunk_size': self.chunk_size,
            'chunk_variance': self.chunk_variance,
            'delay_between_chunks': self.delay_between_chunks,
            'price_improvement': self.price_improvement,
            'total_chunks': self.total_chunks,
            'chunks_submitted': self.chunks_submitted,
            'chunks_filled': self.chunks_filled,
            'active_chunk_id': self.active_chunk_id,
            'active_chunk_quantity': self.active_chunk_quantity,
            'completed_chunks': self.completed_chunks,
            'progress': self.get_iceberg_progress()
        })
        return base_dict
    
    def __str__(self) -> str:
        """String representation of Iceberg order"""
        return (f"ICEBERG {self.side.value.upper()} {self.quantity} {self.symbol} "
                f"@ {self.price} (chunks: {self.chunk_size}, "
                f"progress: {self.chunks_filled}/{self.total_chunks}) "
                f"({self.status.value})")


class IcebergOrderManager:
    """
    Manager for Iceberg orders
    Handles chunk submission and execution coordination
    """
    
    def __init__(self, exchange_manager, logger: logging.Logger = None):
        """
        Initialize Iceberg order manager
        
        Args:
            exchange_manager: Exchange manager for order execution
            logger: Logger instance
        """
        self.exchange_manager = exchange_manager
        self.logger = logger
        self.active_iceberg_orders: Dict[str, IcebergOrder] = {}
        self.chunk_order_mapping: Dict[str, str] = {}  # chunk_order_id -> iceberg_id
        self.monitoring_task = None
    
    async def submit_iceberg_order(self, iceberg_order: IcebergOrder) -> Dict[str, Any]:
        """
        Submit Iceberg order and start chunk management
        
        Args:
            iceberg_order: Iceberg order to submit
            
        Returns:
            Dict with submission result
        """
        try:
            if not iceberg_order.validate():
                return {'success': False, 'error': 'Invalid Iceberg order parameters'}
            
            exchange = self.exchange_manager.get_exchange(iceberg_order.exchange)
            if not exchange:
                return {'success': False, 'error': f'Exchange {iceberg_order.exchange} not available'}
            
            # Submit first chunk
            first_chunk_result = await self._submit_next_chunk(iceberg_order, exchange)
            
            if first_chunk_result['success']:
                self.active_iceberg_orders[iceberg_order.id] = iceberg_order
                iceberg_order.update_status(OrderStatus.SUBMITTED)
                
                # Start monitoring if not already running
                if not self.monitoring_task:
                    self.monitoring_task = asyncio.create_task(self._monitor_iceberg_orders())
            
            return first_chunk_result
            
        except Exception as e:
            self.logger.error(f"Error submitting Iceberg order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _submit_next_chunk(self, iceberg_order: IcebergOrder, exchange) -> Dict[str, Any]:
        """Submit the next chunk for an iceberg order"""
        try:
            chunk_params = iceberg_order.get_next_chunk_params()
            if not chunk_params:
                return {'success': False, 'error': 'No more chunks to submit'}
            
            # Submit chunk order
            result = exchange.create_order(**chunk_params)
            chunk_order_id = result.get('id')
            
            if chunk_order_id:
                # Track chunk
                self.chunk_order_mapping[chunk_order_id] = iceberg_order.id
                iceberg_order.on_chunk_submitted(chunk_order_id, chunk_params['quantity'])
                
                return {
                    'success': True,
                    'iceberg_id': iceberg_order.id,
                    'chunk_order_id': chunk_order_id,
                    'chunk_quantity': chunk_params['quantity']
                }
            else:
                return {'success': False, 'error': 'Failed to get chunk order ID'}
                
        except Exception as e:
            return {'success': False, 'error': f'Chunk submission failed: {e}'}
    
    async def _monitor_iceberg_orders(self):
        """Background task to monitor and manage iceberg orders"""
        while self.active_iceberg_orders:
            try:
                for iceberg_id, iceberg_order in list(self.active_iceberg_orders.items()):
                    await self._process_iceberg_order(iceberg_order)
                
                # Check every second
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in iceberg monitoring: {e}")
                await asyncio.sleep(5)
        
        self.monitoring_task = None
    
    async def _process_iceberg_order(self, iceberg_order: IcebergOrder):
        """Process a single iceberg order"""
        try:
            # Check if order is complete
            if iceberg_order.status == OrderStatus.FILLED:
                del self.active_iceberg_orders[iceberg_order.id]
                return
            
            # Check if we can submit next chunk
            if iceberg_order.can_submit_next_chunk():
                exchange = self.exchange_manager.get_exchange(iceberg_order.exchange)
                if exchange:
                    result = await self._submit_next_chunk(iceberg_order, exchange)
                    if not result['success']:
                        self.logger.warning(f"Failed to submit next chunk for {iceberg_order.id}: {result['error']}")
            
        except Exception as e:
            self.logger.error(f"Error processing iceberg order {iceberg_order.id}: {e}")
    
    def handle_order_update(self, exchange_order_id: str, update_data: Dict[str, Any]):
        """Handle order updates from exchange"""
        iceberg_id = self.chunk_order_mapping.get(exchange_order_id)
        if not iceberg_id:
            return
        
        iceberg_order = self.active_iceberg_orders.get(iceberg_id)
        if not iceberg_order:
            return
        
        # Handle chunk fill
        if update_data.get('status') == 'filled':
            iceberg_order.on_chunk_filled(exchange_order_id, update_data)
            
            # Remove chunk mapping
            del self.chunk_order_mapping[exchange_order_id]
    
    async def cancel_iceberg_order(self, iceberg_id: str) -> Dict[str, Any]:
        """Cancel an active iceberg order"""
        try:
            iceberg_order = self.active_iceberg_orders.get(iceberg_id)
            if not iceberg_order:
                return {'success': False, 'error': 'Iceberg order not found'}
            
            exchange = self.exchange_manager.get_exchange(iceberg_order.exchange)
            if not exchange:
                return {'success': False, 'error': 'Exchange not available'}
            
            # Cancel active chunk
            if iceberg_order.active_chunk_id:
                try:
                    exchange.cancel_order(iceberg_order.active_chunk_id)
                except Exception as e:
                    self.logger.warning(f"Failed to cancel active chunk: {e}")
            
            iceberg_order.cancel("User cancelled")
            del self.active_iceberg_orders[iceberg_id]
            
            return {'success': True, 'iceberg_id': iceberg_id}
            
        except Exception as e:
            self.logger.error(f"Error cancelling iceberg order: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_active_iceberg_orders(self) -> List[Dict[str, Any]]:
        """Get list of active iceberg orders"""
        return [order.to_dict() for order in self.active_iceberg_orders.values()]