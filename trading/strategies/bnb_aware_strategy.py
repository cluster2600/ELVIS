#!/usr/bin/env python3
"""
BNB-aware trading strategy that optimizes fees and includes BNB trading
"""

import logging
from typing import Dict, Any, List, Optional
from trading.strategies.base_strategy import BaseStrategy

class BNBAwareStrategy(BaseStrategy):
    """
    Trading strategy that includes BNB fee optimization and BNB trading
    """
    
    def __init__(self, logger: logging.Logger, symbols: List[str] = None, **kwargs):
        # Include BNB symbols by default
        default_symbols = ['BTCUSDT', 'BNBUSDT']
        symbols = symbols or default_symbols
        super().__init__(logger, symbols, **kwargs)
        
        # BNB-specific settings
        self.enable_bnb_optimization = kwargs.get('enable_bnb_optimization', True)
        self.bnb_allocation_percent = kwargs.get('bnb_allocation_percent', 5.0)  # 5% for BNB
        self.min_bnb_balance = kwargs.get('min_bnb_balance', 0.1)
        
        self.logger.info(f"🪙 BNB-aware strategy initialized")
        self.logger.info(f"   BNB optimization: {self.enable_bnb_optimization}")
        self.logger.info(f"   BNB allocation: {self.bnb_allocation_percent}%")
    
    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal with BNB considerations
        """
        try:
            # Get base signal from parent class
            signal = super().generate_signal(symbol, data)
            
            # Add BNB-specific logic
            if symbol == 'BNBUSDT':
                signal = self._generate_bnb_signal(data)
            else:
                # For other symbols, check if we need BNB for fees
                signal = self._optimize_signal_for_bnb_fees(signal, symbol, data)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    def _generate_bnb_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate specific signals for BNB trading
        """
        try:
            current_price = data.get('price', 0)
            
            # Simple BNB accumulation strategy
            # Buy BNB when:
            # 1. We're low on BNB balance for fees
            # 2. BNB is showing bullish momentum
            # 3. We haven't reached our BNB allocation limit
            
            balance_info = data.get('balance_info', {})
            bnb_balance = balance_info.get('BNB', 0)
            total_balance_usdt = balance_info.get('total_usdt', 1000)
            
            # Check if we need more BNB
            bnb_value_usdt = bnb_balance * current_price
            bnb_allocation_current = (bnb_value_usdt / total_balance_usdt) * 100
            
            should_accumulate_bnb = (
                bnb_balance < self.min_bnb_balance or 
                bnb_allocation_current < self.bnb_allocation_percent
            )
            
            if should_accumulate_bnb:
                # Calculate how much BNB to buy
                target_bnb_value = total_balance_usdt * (self.bnb_allocation_percent / 100)
                needed_bnb_value = target_bnb_value - bnb_value_usdt
                
                if needed_bnb_value > 10:  # Only if we need more than $10 worth
                    return {
                        'action': 'BUY',
                        'confidence': 0.8,
                        'position_size': min(needed_bnb_value / current_price, 0.05),  # Max 5% position
                        'reason': f'BNB accumulation for fees (current: {bnb_allocation_current:.1f}%, target: {self.bnb_allocation_percent}%)'
                    }
            
            # Check technical indicators for BNB trading opportunities
            indicators = data.get('indicators', {})
            rsi = indicators.get('rsi', 50)
            
            if rsi < 30 and bnb_allocation_current < self.bnb_allocation_percent * 1.5:
                return {
                    'action': 'BUY',
                    'confidence': 0.6,
                    'position_size': 0.02,  # Small position
                    'reason': f'BNB oversold (RSI: {rsi:.1f}) + allocation opportunity'
                }
            elif rsi > 70 and bnb_allocation_current > self.bnb_allocation_percent * 2:
                return {
                    'action': 'SELL',
                    'confidence': 0.6,
                    'position_size': 0.02,
                    'reason': f'BNB overbought (RSI: {rsi:.1f}) + excess allocation'
                }
            
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reason': f'BNB balanced (allocation: {bnb_allocation_current:.1f}%, RSI: {rsi:.1f})'
            }
            
        except Exception as e:
            self.logger.error(f"Error in BNB signal generation: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': f'BNB signal error: {e}'}
    
    def _optimize_signal_for_bnb_fees(self, signal: Dict[str, Any], symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize trading signal considering BNB fee benefits
        """
        if not self.enable_bnb_optimization:
            return signal
        
        try:
            # Get balance info
            balance_info = data.get('balance_info', {})
            bnb_balance = balance_info.get('BNB', 0)
            
            # If we have sufficient BNB, slightly increase confidence due to lower fees
            if bnb_balance >= self.min_bnb_balance:
                original_confidence = signal.get('confidence', 0)
                fee_bonus = 0.05  # 5% confidence bonus for having BNB
                signal['confidence'] = min(1.0, original_confidence + fee_bonus)
                
                if 'reason' in signal:
                    signal['reason'] += ' (BNB fee optimization active)'
            else:
                # If we're low on BNB, slightly reduce confidence
                original_confidence = signal.get('confidence', 0)
                fee_penalty = 0.02  # 2% confidence penalty for higher fees
                signal['confidence'] = max(0.0, original_confidence - fee_penalty)
                
                if 'reason' in signal:
                    signal['reason'] += ' (BNB balance low - higher fees)'
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error optimizing signal for BNB fees: {e}")
            return signal
    
    def get_position_size(self, signal: Dict[str, Any], symbol: str, balance: Dict[str, float]) -> float:
        """
        Calculate position size with BNB considerations
        """
        base_size = super().get_position_size(signal, symbol, balance)
        
        # For BNB trades, use smaller position sizes
        if symbol == 'BNBUSDT':
            return min(base_size, 0.05)  # Max 5% for BNB
        
        return base_size
    
    def validate_trade(self, signal: Dict[str, Any], symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate trade with BNB fee considerations
        """
        validation = super().validate_trade(signal, symbol, data)
        
        # Add BNB-specific validations
        if symbol == 'BNBUSDT':
            # Don't sell BNB if we're below minimum balance
            balance_info = data.get('balance_info', {})
            bnb_balance = balance_info.get('BNB', 0)
            
            if signal.get('action') == 'SELL' and bnb_balance <= self.min_bnb_balance:
                validation['valid'] = False
                validation['reason'] = f'Cannot sell BNB: balance ({bnb_balance:.6f}) at or below minimum ({self.min_bnb_balance})'
        
        return validation
