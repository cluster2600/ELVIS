#!/usr/bin/env python3
"""
Script to enable BNB trading and fee optimization in the trading bot
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set environment
os.environ["VAULT_ENABLED"] = "false"


def update_trading_config():
    """Update trading configuration to support BNB"""

    config_path = Path(__file__).parent / "config" / "config.py"

    # Read current config
    with open(config_path, "r") as f:
        content = f.read()

    # Add BNB-specific configuration
    bnb_config = """
# BNB Trading and Fee Optimization Configuration
BNB_CONFIG = {
    'ENABLE_BNB_FEES': True,           # Use BNB to pay trading fees (10% discount on futures, 25% on spot)
    'BNB_TRADING_ENABLED': True,       # Allow trading BNB pairs
    'MIN_BNB_BALANCE': 0.1,           # Minimum BNB balance to maintain for fees
    'AUTO_BUY_BNB': True,             # Automatically buy BNB when balance is low
    'MAX_BNB_BUY_PERCENT': 5.0,       # Max % of portfolio to spend on BNB auto-buy
    'BNB_SYMBOLS': ['BNBUSDT', 'BNBBTC'],  # Available BNB trading pairs
    'BNB_REBALANCE_THRESHOLD': 0.05,  # Rebalance when BNB balance drops below this
}
"""

    # Add symbols configuration
    symbols_config = """
# Multi-Asset Trading Configuration  
SYMBOLS_CONFIG = {
    'PRIMARY_SYMBOLS': ['BTCUSDT', 'BNBUSDT'],     # Primary trading pairs
    'SECONDARY_SYMBOLS': ['ETHUSDT', 'ADAUSDT'],    # Secondary pairs (optional)
    'STABLE_PAIRS': ['BTCUSDT', 'ETHUSDT'],        # Stable, high-liquidity pairs
    'FEE_OPTIMIZATION_PAIRS': ['BNBUSDT'],         # Pairs for fee optimization
    'MAX_CONCURRENT_PAIRS': 3,                     # Maximum pairs to trade simultaneously
}
"""

    # Check if BNB config already exists
    if "BNB_CONFIG" not in content:
        # Add BNB config before the last line
        lines = content.split("\n")
        insert_pos = len(lines) - 1  # Before last line

        lines.insert(insert_pos, "")
        lines.insert(insert_pos + 1, bnb_config)
        lines.insert(insert_pos + 2, "")
        lines.insert(insert_pos + 3, symbols_config)

        new_content = "\n".join(lines)

        # Write updated config
        with open(config_path, "w") as f:
            f.write(new_content)

        logger.info("✅ Updated config.py with BNB trading configuration")
        return True
    else:
        logger.info("ℹ️ BNB configuration already exists in config.py")
        return False


def create_bnb_strategy():
    """Create a simple BNB-aware trading strategy"""

    strategy_content = '''#!/usr/bin/env python3
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
'''

    strategy_path = (
        Path(__file__).parent / "trading" / "strategies" / "bnb_aware_strategy.py"
    )

    # Create strategy file
    with open(strategy_path, "w") as f:
        f.write(strategy_content)

    logger.info("✅ Created BNB-aware trading strategy")


def test_bnb_integration():
    """Test the BNB integration with current setup"""
    try:
        from trading.execution.enhanced_binance_executor import EnhancedBinanceExecutor

        logger.info("🧪 Testing BNB integration...")

        # Test enhanced executor
        executor = EnhancedBinanceExecutor(
            logger=logger,
            is_testnet=True,
            use_futures=True,
            enable_bnb_fees=True,
            bnb_trading_enabled=True,
        )

        executor.initialize()

        # Test fee calculation
        trade_value = 5000  # $5000 trade
        fee_analysis = executor.calculate_fee_with_bnb(trade_value, is_futures=True)

        logger.info(f"💰 Fee Analysis for ${trade_value:,} trade:")
        logger.info(f"   Standard fee: ${fee_analysis['standard_fee_usdt']:.2f}")
        logger.info(f"   With BNB: ${fee_analysis['discounted_fee_usdt']:.2f}")
        logger.info(
            f"   Savings: ${fee_analysis['savings_usdt']:.2f} ({fee_analysis['discount_percent']:.0f}%)"
        )

        # Test BNB auto-buy logic
        bnb_analysis = executor.should_auto_buy_bnb(trade_value)
        logger.info(f"🔄 BNB Analysis:")
        logger.info(f"   Should auto-buy: {bnb_analysis['should_buy']}")

        if bnb_analysis["should_buy"]:
            logger.info(
                f"   Recommended BNB purchase: {bnb_analysis['buy_amount_bnb']:.6f} BNB"
            )
            logger.info(f"   Cost: ${bnb_analysis['buy_cost_usdt']:.2f}")

        return True

    except Exception as e:
        logger.error(f"❌ BNB integration test failed: {e}")
        return False


def main():
    """Main function to enable BNB trading"""
    logger.info("🪙 Enabling BNB Trading and Fee Optimization")
    logger.info("=" * 60)

    # 1. Update configuration
    config_updated = update_trading_config()

    # 2. Create BNB strategy
    create_bnb_strategy()

    # 3. Test integration
    test_success = test_bnb_integration()

    # Summary
    logger.info("=" * 60)
    logger.info("📋 BNB INTEGRATION SUMMARY:")
    logger.info(f"   Config updated: {'✅' if config_updated else 'ℹ️ Already exists'}")
    logger.info(f"   Strategy created: ✅")
    logger.info(f"   Integration test: {'✅' if test_success else '❌'}")

    if test_success:
        logger.info("\n🎉 BNB TRADING SUCCESSFULLY ENABLED!")
        logger.info("\n💡 Benefits now available:")
        logger.info("   • 10% fee discount on futures trading")
        logger.info("   • 25% fee discount on spot trading")
        logger.info("   • Automatic BNB balance management")
        logger.info("   • BNB trading opportunities")
        logger.info("   • Multi-asset portfolio optimization")

        logger.info("\n🚀 Next steps:")
        logger.info(
            "   1. Update your main trading script to use EnhancedBinanceExecutor"
        )
        logger.info("   2. Consider adding BNBUSDT to your trading symbols")
        logger.info("   3. Monitor BNB balance for optimal fee savings")
    else:
        logger.warning("\n⚠️ Some issues found - please check the errors above")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
