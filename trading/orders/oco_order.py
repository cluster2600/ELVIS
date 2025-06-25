"""
OCO (One-Cancels-Other) Order Implementation
Combines a limit order with a stop order - when one fills, the other is cancelled
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .base_order import BaseOrder, OrderStatus, OrderSide, OrderType, TimeInForce


class OCOOrder(BaseOrder):
    """
    One-Cancels-Other Order
    
    Consists of two orders:
    1. Limit order (take profit)
    2. Stop order (stop loss)
    
    When one order is filled, the other is automatically cancelled
    """
    
    def __init__(self, symbol: str, side: OrderSide, quantity: float,
                 limit_price: float, stop_price: float,
                 stop_limit_price: Optional[float] = None,
                 time_in_force: TimeInForce = TimeInForce.GTC,
                 exchange: str = None, logger: logging.Logger = None):
        """
        Initialize OCO order
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            limit_price: Limit order price (take profit)
            stop_price: Stop trigger price
            stop_limit_price: Stop limit price (if None, uses stop market)
            time_in_force: Time in force
            exchange: Target exchange
            logger: Logger instance
        """
        super().__init__(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.OCO,
            price=None,  # OCO doesn't have single price
            time_in_force=time_in_force,
            exchange=exchange,
            logger=logger
        )
        
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.stop_limit_price = stop_limit_price
        
        # Child order tracking
        self.limit_order_id = None
        self.stop_order_id = None
        self.filled_order_type = None  # 'limit' or 'stop'
        
        # Validation
        self._validate_prices()
    
    def _validate_prices(self):
        """Validate OCO price relationships"""
        if self.side == OrderSide.BUY:
            # For buy orders: limit_price should be higher than stop_price
            if self.limit_price <= self.stop_price:
                raise ValueError("For BUY OCO: limit_price must be > stop_price")
        else:
            # For sell orders: limit_price should be higher than stop_price
            if self.limit_price <= self.stop_price:
                raise ValueError("For SELL OCO: limit_price must be > stop_price")
    
    def validate(self) -> bool:
        """Validate OCO order parameters"""
        if self.quantity <= 0:
            return False
        
        if self.limit_price <= 0 or self.stop_price <= 0:
            return False
        
        if self.stop_limit_price and self.stop_limit_price <= 0:
            return False
        
        try:
            self._validate_prices()
            return True
        except ValueError as e:
            if self.logger:
                self.logger.error(f"OCO validation failed: {e}")
            return False
    
    def to_exchange_format(self) -> Dict[str, Any]:
        """Convert OCO order to exchange format"""
        # Different exchanges have different OCO implementations
        # This is a generic format that can be adapted
        
        order_data = {
            'symbol': self.symbol,
            'side': self.side.value,
            'type': 'oco',
            'quantity': self.quantity,
            'timeInForce': self.time_in_force.value.upper(),
            'limitPrice': self.limit_price,
            'stopPrice': self.stop_price
        }
        
        if self.stop_limit_price:
            order_data['stopLimitPrice'] = self.stop_limit_price
            order_data['stopType'] = 'stop_limit'
        else:
            order_data['stopType'] = 'stop_market'
        
        return order_data
    
    def get_limit_order_params(self) -> Dict[str, Any]:
        """Get parameters for the limit order component"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'type': 'limit',
            'quantity': self.quantity,
            'price': self.limit_price,
            'timeInForce': self.time_in_force.value.upper()
        }
    
    def get_stop_order_params(self) -> Dict[str, Any]:
        """Get parameters for the stop order component"""
        order_params = {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'stopPrice': self.stop_price,
            'timeInForce': self.time_in_force.value.upper()
        }
        
        if self.stop_limit_price:
            order_params['type'] = 'stop_limit'
            order_params['price'] = self.stop_limit_price
        else:
            order_params['type'] = 'stop_market'
        
        return order_params
    
    def on_limit_filled(self, fill_data: Dict[str, Any]):
        """Handle limit order being filled"""
        self.filled_order_type = 'limit'
        self.add_fill(
            quantity=fill_data.get('quantity', self.quantity),
            price=fill_data.get('price', self.limit_price),
            fee=fill_data.get('fee', 0.0),
            fill_id=fill_data.get('fill_id'),
            timestamp=fill_data.get('timestamp')
        )
        
        if self.logger:
            self.logger.info(f"OCO {self.id}: Limit order filled at {self.limit_price}")
    
    def on_stop_filled(self, fill_data: Dict[str, Any]):
        """Handle stop order being filled"""
        self.filled_order_type = 'stop'
        self.add_fill(
            quantity=fill_data.get('quantity', self.quantity),
            price=fill_data.get('price', self.stop_limit_price or self.stop_price),
            fee=fill_data.get('fee', 0.0),
            fill_id=fill_data.get('fill_id'),
            timestamp=fill_data.get('timestamp')
        )
        
        if self.logger:
            self.logger.info(f"OCO {self.id}: Stop order filled at {fill_data.get('price')}")
    
    def cancel_remaining_order(self):
        """Cancel the unfilled order after one side is filled"""
        if self.status == OrderStatus.FILLED:
            if self.logger:
                self.logger.info(f"OCO {self.id}: Cancelling remaining order")
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert OCO order to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'stop_limit_price': self.stop_limit_price,
            'limit_order_id': self.limit_order_id,
            'stop_order_id': self.stop_order_id,
            'filled_order_type': self.filled_order_type
        })
        return base_dict
    
    def __str__(self) -> str:
        """String representation of OCO order"""
        stop_type = "STOP_LIMIT" if self.stop_limit_price else "STOP_MARKET"
        return (f"OCO {self.side.value.upper()} {self.quantity} {self.symbol} "
                f"LIMIT@{self.limit_price} {stop_type}@{self.stop_price} "
                f"({self.status.value})")


class OCOOrderManager:
    """
    Manager for OCO orders
    Handles the coordination between limit and stop order components
    """
    
    def __init__(self, exchange_manager, logger: logging.Logger = None):
        """
        Initialize OCO order manager
        
        Args:
            exchange_manager: Exchange manager for order execution
            logger: Logger instance
        """
        self.exchange_manager = exchange_manager
        self.logger = logger
        self.active_oco_orders: Dict[str, OCOOrder] = {}
        self.order_id_mapping: Dict[str, str] = {}  # exchange_order_id -> oco_id
    
    async def submit_oco_order(self, oco_order: OCOOrder) -> Dict[str, Any]:
        """
        Submit OCO order to exchange
        
        Args:
            oco_order: OCO order to submit
            
        Returns:
            Dict with submission result
        """
        try:
            if not oco_order.validate():
                return {'success': False, 'error': 'Invalid OCO order parameters'}
            
            exchange = self.exchange_manager.get_exchange(oco_order.exchange)
            if not exchange:
                return {'success': False, 'error': f'Exchange {oco_order.exchange} not available'}
            
            # Check if exchange supports native OCO
            if hasattr(exchange, 'create_oco_order'):
                # Native OCO support
                result = await self._submit_native_oco(exchange, oco_order)
            else:
                # Simulate OCO with separate orders
                result = await self._submit_simulated_oco(exchange, oco_order)
            
            if result['success']:
                self.active_oco_orders[oco_order.id] = oco_order
                oco_order.update_status(OrderStatus.SUBMITTED)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error submitting OCO order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _submit_native_oco(self, exchange, oco_order: OCOOrder) -> Dict[str, Any]:
        """Submit OCO using native exchange support"""
        try:
            order_params = oco_order.to_exchange_format()
            result = exchange.create_oco_order(**order_params)
            
            oco_order.exchange_order_id = result.get('id')
            oco_order.limit_order_id = result.get('limit_order_id')
            oco_order.stop_order_id = result.get('stop_order_id')
            
            return {'success': True, 'oco_id': oco_order.id, 'exchange_result': result}
            
        except Exception as e:
            return {'success': False, 'error': f'Native OCO submission failed: {e}'}
    
    async def _submit_simulated_oco(self, exchange, oco_order: OCOOrder) -> Dict[str, Any]:
        """Simulate OCO by submitting separate orders and managing them"""
        try:
            # Submit limit order first
            limit_params = oco_order.get_limit_order_params()
            limit_result = exchange.create_order(**limit_params)
            oco_order.limit_order_id = limit_result.get('id')
            
            # Submit stop order
            stop_params = oco_order.get_stop_order_params()
            stop_result = exchange.create_order(**stop_params)
            oco_order.stop_order_id = stop_result.get('id')
            
            # Track order IDs for monitoring
            self.order_id_mapping[oco_order.limit_order_id] = oco_order.id
            self.order_id_mapping[oco_order.stop_order_id] = oco_order.id
            
            return {
                'success': True,
                'oco_id': oco_order.id,
                'limit_order_id': oco_order.limit_order_id,
                'stop_order_id': oco_order.stop_order_id
            }
            
        except Exception as e:
            # Cleanup if partial submission
            await self._cleanup_partial_oco(exchange, oco_order)
            return {'success': False, 'error': f'Simulated OCO submission failed: {e}'}
    
    async def _cleanup_partial_oco(self, exchange, oco_order: OCOOrder):
        """Cleanup partially submitted OCO orders"""
        try:
            if oco_order.limit_order_id:
                exchange.cancel_order(oco_order.limit_order_id)
            if oco_order.stop_order_id:
                exchange.cancel_order(oco_order.stop_order_id)
        except Exception as e:
            self.logger.error(f"Error cleaning up partial OCO: {e}")
    
    async def cancel_oco_order(self, oco_id: str) -> Dict[str, Any]:
        """Cancel an active OCO order"""
        try:
            oco_order = self.active_oco_orders.get(oco_id)
            if not oco_order:
                return {'success': False, 'error': 'OCO order not found'}
            
            exchange = self.exchange_manager.get_exchange(oco_order.exchange)
            if not exchange:
                return {'success': False, 'error': 'Exchange not available'}
            
            # Cancel both orders
            cancelled_orders = []
            if oco_order.limit_order_id:
                try:
                    exchange.cancel_order(oco_order.limit_order_id)
                    cancelled_orders.append('limit')
                except Exception as e:
                    self.logger.warning(f"Failed to cancel limit order: {e}")
            
            if oco_order.stop_order_id:
                try:
                    exchange.cancel_order(oco_order.stop_order_id)
                    cancelled_orders.append('stop')
                except Exception as e:
                    self.logger.warning(f"Failed to cancel stop order: {e}")
            
            oco_order.cancel("User cancelled")
            del self.active_oco_orders[oco_id]
            
            return {
                'success': True,
                'oco_id': oco_id,
                'cancelled_orders': cancelled_orders
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling OCO order: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_order_update(self, exchange_order_id: str, update_data: Dict[str, Any]):
        """Handle order updates from exchange"""
        oco_id = self.order_id_mapping.get(exchange_order_id)
        if not oco_id:
            return
        
        oco_order = self.active_oco_orders.get(oco_id)
        if not oco_order:
            return
        
        # Determine which order was updated
        if exchange_order_id == oco_order.limit_order_id:
            if update_data.get('status') == 'filled':
                oco_order.on_limit_filled(update_data)
                self._cancel_remaining_order(oco_order)
        elif exchange_order_id == oco_order.stop_order_id:
            if update_data.get('status') == 'filled':
                oco_order.on_stop_filled(update_data)
                self._cancel_remaining_order(oco_order)
    
    def _cancel_remaining_order(self, oco_order: OCOOrder):
        """Cancel the remaining order after one is filled"""
        try:
            exchange = self.exchange_manager.get_exchange(oco_order.exchange)
            
            if oco_order.filled_order_type == 'limit' and oco_order.stop_order_id:
                exchange.cancel_order(oco_order.stop_order_id)
            elif oco_order.filled_order_type == 'stop' and oco_order.limit_order_id:
                exchange.cancel_order(oco_order.limit_order_id)
            
            # Remove from active orders
            if oco_order.id in self.active_oco_orders:
                del self.active_oco_orders[oco_order.id]
                
        except Exception as e:
            self.logger.error(f"Error cancelling remaining OCO order: {e}")
    
    def get_active_oco_orders(self) -> List[Dict[str, Any]]:
        """Get list of active OCO orders"""
        return [order.to_dict() for order in self.active_oco_orders.values()]