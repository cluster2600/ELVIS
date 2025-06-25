"""
Advanced Order Manager
Unified manager for all advanced order types (OCO, Iceberg, TWAP)
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import asyncio
import logging

from .base_order import BaseOrder, OrderStatus, OrderSide, OrderType
from .oco_order import OCOOrder, OCOOrderManager
from .iceberg_order import IcebergOrder, IcebergOrderManager
from .twap_order import TWAPOrder, TWAPOrderManager


class AdvancedOrderManager:
    """
    Unified manager for all advanced order types
    Coordinates OCO, Iceberg, and TWAP orders
    """
    
    def __init__(self, exchange_manager, logger: logging.Logger = None):
        """
        Initialize advanced order manager
        
        Args:
            exchange_manager: Exchange manager for order execution
            logger: Logger instance
        """
        self.exchange_manager = exchange_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Individual order managers
        self.oco_manager = OCOOrderManager(exchange_manager, logger)
        self.iceberg_manager = IcebergOrderManager(exchange_manager, logger)
        self.twap_manager = TWAPOrderManager(exchange_manager, logger)
        
        # All active orders tracking
        self.all_active_orders: Dict[str, BaseOrder] = {}
        self.order_type_mapping: Dict[str, str] = {}  # order_id -> order_type
        
        # Statistics
        self.stats = {
            'total_submitted': 0,
            'total_filled': 0,
            'total_cancelled': 0,
            'by_type': {
                'oco': {'submitted': 0, 'filled': 0, 'cancelled': 0},
                'iceberg': {'submitted': 0, 'filled': 0, 'cancelled': 0},
                'twap': {'submitted': 0, 'filled': 0, 'cancelled': 0}
            }
        }
    
    async def submit_order(self, order: BaseOrder) -> Dict[str, Any]:
        """
        Submit an advanced order
        
        Args:
            order: Advanced order (OCO, Iceberg, or TWAP)
            
        Returns:
            Dict with submission result
        """
        try:
            if not isinstance(order, (OCOOrder, IcebergOrder, TWAPOrder)):
                return {'success': False, 'error': 'Unsupported order type'}
            
            # Route to appropriate manager
            if isinstance(order, OCOOrder):
                result = await self.oco_manager.submit_oco_order(order)
                order_type = 'oco'
            elif isinstance(order, IcebergOrder):
                result = await self.iceberg_manager.submit_iceberg_order(order)
                order_type = 'iceberg'
            elif isinstance(order, TWAPOrder):
                result = await self.twap_manager.submit_twap_order(order)
                order_type = 'twap'
            else:
                return {'success': False, 'error': 'Unknown order type'}
            
            if result['success']:
                # Track in unified system
                self.all_active_orders[order.id] = order
                self.order_type_mapping[order.id] = order_type
                
                # Update statistics
                self.stats['total_submitted'] += 1
                self.stats['by_type'][order_type]['submitted'] += 1
                
                self.logger.info(f"Advanced order submitted: {order_type.upper()} {order.id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error submitting advanced order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an advanced order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict with cancellation result
        """
        try:
            order_type = self.order_type_mapping.get(order_id)
            if not order_type:
                return {'success': False, 'error': 'Order not found'}
            
            # Route to appropriate manager
            if order_type == 'oco':
                result = await self.oco_manager.cancel_oco_order(order_id)
            elif order_type == 'iceberg':
                result = await self.iceberg_manager.cancel_iceberg_order(order_id)
            elif order_type == 'twap':
                result = await self.twap_manager.cancel_twap_order(order_id)
            else:
                return {'success': False, 'error': 'Unknown order type'}
            
            if result['success']:
                # Remove from tracking
                if order_id in self.all_active_orders:
                    del self.all_active_orders[order_id]
                if order_id in self.order_type_mapping:
                    del self.order_type_mapping[order_id]
                
                # Update statistics
                self.stats['total_cancelled'] += 1
                self.stats['by_type'][order_type]['cancelled'] += 1
                
                self.logger.info(f"Advanced order cancelled: {order_type.upper()} {order_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error cancelling advanced order: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of an advanced order
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict with order status and details
        """
        try:
            order = self.all_active_orders.get(order_id)
            if not order:
                return {'success': False, 'error': 'Order not found'}
            
            order_type = self.order_type_mapping.get(order_id)
            
            result = {
                'success': True,
                'order_id': order_id,
                'order_type': order_type,
                'status': order.status.value,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'remaining_quantity': order.remaining_quantity,
                'created_at': order.created_at.isoformat(),
                'fills': order.fills
            }
            
            # Add type-specific details
            if isinstance(order, OCOOrder):
                result.update({
                    'limit_price': order.limit_price,
                    'stop_price': order.stop_price,
                    'stop_limit_price': order.stop_limit_price,
                    'filled_order_type': order.filled_order_type
                })
            elif isinstance(order, IcebergOrder):
                result.update({
                    'price': order.price,
                    'chunk_size': order.chunk_size,
                    'progress': order.get_iceberg_progress()
                })
            elif isinstance(order, TWAPOrder):
                result.update({
                    'duration_minutes': order.duration_minutes,
                    'interval_minutes': order.interval_minutes,
                    'progress': order.get_twap_progress()
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_all_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active advanced orders"""
        try:
            orders = []
            
            for order_id, order in self.all_active_orders.items():
                order_type = self.order_type_mapping.get(order_id)
                order_dict = order.to_dict()
                order_dict['order_type_class'] = order_type
                orders.append(order_dict)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error getting active orders: {e}")
            return []
    
    def get_orders_by_type(self, order_type: str) -> List[Dict[str, Any]]:
        """
        Get active orders by type
        
        Args:
            order_type: 'oco', 'iceberg', or 'twap'
            
        Returns:
            List of orders of specified type
        """
        try:
            if order_type == 'oco':
                return self.oco_manager.get_active_oco_orders()
            elif order_type == 'iceberg':
                return self.iceberg_manager.get_active_iceberg_orders()
            elif order_type == 'twap':
                return self.twap_manager.get_active_twap_orders()
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting orders by type: {e}")
            return []
    
    def get_orders_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get active orders for a specific symbol"""
        try:
            symbol_orders = []
            
            for order in self.all_active_orders.values():
                if order.symbol == symbol:
                    order_type = self.order_type_mapping.get(order.id)
                    order_dict = order.to_dict()
                    order_dict['order_type_class'] = order_type
                    symbol_orders.append(order_dict)
            
            return symbol_orders
            
        except Exception as e:
            self.logger.error(f"Error getting orders by symbol: {e}")
            return []
    
    def handle_order_update(self, exchange_order_id: str, update_data: Dict[str, Any]):
        """
        Handle order updates from exchanges
        Routes to appropriate order manager
        
        Args:
            exchange_order_id: Exchange order ID
            update_data: Update data from exchange
        """
        try:
            # Try each manager to handle the update
            self.oco_manager.handle_order_update(exchange_order_id, update_data)
            self.iceberg_manager.handle_order_update(exchange_order_id, update_data)
            self.twap_manager.handle_order_update(exchange_order_id, update_data)
            
            # Update our tracking if order is filled/cancelled
            status = update_data.get('status')
            if status in ['filled', 'cancelled']:
                # Find and update our order
                for order_id, order in list(self.all_active_orders.items()):
                    if (hasattr(order, 'exchange_order_id') and order.exchange_order_id == exchange_order_id) or \
                       (hasattr(order, 'limit_order_id') and order.limit_order_id == exchange_order_id) or \
                       (hasattr(order, 'stop_order_id') and order.stop_order_id == exchange_order_id) or \
                       (hasattr(order, 'active_chunk_id') and order.active_chunk_id == exchange_order_id) or \
                       (hasattr(order, 'active_slice_id') and order.active_slice_id == exchange_order_id):
                        
                        order_type = self.order_type_mapping.get(order_id)
                        
                        if status == 'filled' and order.status == OrderStatus.FILLED:
                            # Order completely filled
                            self.stats['total_filled'] += 1
                            if order_type:
                                self.stats['by_type'][order_type]['filled'] += 1
                            
                            # Remove from active tracking
                            del self.all_active_orders[order_id]
                            if order_id in self.order_type_mapping:
                                del self.order_type_mapping[order_id]
                        
                        break
            
        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order execution statistics"""
        try:
            # Add current active counts
            current_stats = self.stats.copy()
            current_stats['active_orders'] = {
                'total': len(self.all_active_orders),
                'oco': len(self.oco_manager.active_oco_orders),
                'iceberg': len(self.iceberg_manager.active_iceberg_orders),
                'twap': len(self.twap_manager.active_twap_orders)
            }
            
            return current_stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return self.stats
    
    async def create_oco_order(self, symbol: str, side: OrderSide, quantity: float,
                              limit_price: float, stop_price: float,
                              stop_limit_price: Optional[float] = None,
                              exchange: str = None) -> Dict[str, Any]:
        """
        Convenience method to create and submit OCO order
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            limit_price: Limit order price
            stop_price: Stop trigger price
            stop_limit_price: Stop limit price (optional)
            exchange: Target exchange
            
        Returns:
            Dict with submission result
        """
        try:
            oco_order = OCOOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                stop_limit_price=stop_limit_price,
                exchange=exchange,
                logger=self.logger
            )
            
            return await self.submit_order(oco_order)
            
        except Exception as e:
            self.logger.error(f"Error creating OCO order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def create_iceberg_order(self, symbol: str, side: OrderSide, quantity: float,
                                  price: float, chunk_size: float,
                                  chunk_variance: float = 0.0,
                                  delay_between_chunks: int = 0,
                                  exchange: str = None) -> Dict[str, Any]:
        """
        Convenience method to create and submit Iceberg order
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Total order quantity
            price: Order price
            chunk_size: Size of each chunk
            chunk_variance: Random variance in chunk size
            delay_between_chunks: Delay between chunks in seconds
            exchange: Target exchange
            
        Returns:
            Dict with submission result
        """
        try:
            iceberg_order = IcebergOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                chunk_size=chunk_size,
                chunk_variance=chunk_variance,
                delay_between_chunks=delay_between_chunks,
                exchange=exchange,
                logger=self.logger
            )
            
            return await self.submit_order(iceberg_order)
            
        except Exception as e:
            self.logger.error(f"Error creating Iceberg order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def create_twap_order(self, symbol: str, side: OrderSide, quantity: float,
                               duration_minutes: int, interval_minutes: int = 5,
                               price_limit: Optional[float] = None,
                               participation_rate: float = 0.1,
                               exchange: str = None) -> Dict[str, Any]:
        """
        Convenience method to create and submit TWAP order
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Total order quantity
            duration_minutes: Total execution duration
            interval_minutes: Interval between slices
            price_limit: Maximum/minimum price limit
            participation_rate: Market participation rate
            exchange: Target exchange
            
        Returns:
            Dict with submission result
        """
        try:
            twap_order = TWAPOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                duration_minutes=duration_minutes,
                interval_minutes=interval_minutes,
                price_limit=price_limit,
                participation_rate=participation_rate,
                exchange=exchange,
                logger=self.logger
            )
            
            return await self.submit_order(twap_order)
            
        except Exception as e:
            self.logger.error(f"Error creating TWAP order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cleanup_completed_orders(self):
        """Clean up completed orders from tracking"""
        try:
            completed_orders = []
            
            for order_id, order in list(self.all_active_orders.items()):
                if not order.is_active():
                    completed_orders.append(order_id)
            
            for order_id in completed_orders:
                if order_id in self.all_active_orders:
                    del self.all_active_orders[order_id]
                if order_id in self.order_type_mapping:
                    del self.order_type_mapping[order_id]
            
            if completed_orders:
                self.logger.info(f"Cleaned up {len(completed_orders)} completed orders")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up completed orders: {e}")