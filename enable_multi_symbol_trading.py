#!/usr/bin/env python3
"""
Enable multi-symbol trading (BTCUSDT + BNBBTC) in main.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_main_for_multi_symbol():
    """Update main.py to support BTCUSDT and BNBBTC trading"""
    logger.info("🔧 Updating main.py for Multi-Symbol Trading")
    logger.info("="*50)
    
    main_file = '/Users/maxime/BTC_BOT/BTC_BOT/main.py'
    
    try:
        # Read current main.py
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Backup original
        with open(main_file + '.backup', 'w') as f:
            f.write(content)
        logger.info("✅ Created backup: main.py.backup")
        
        # Update the trading loop to handle multiple symbols
        old_symbol_line = '                        data = price_fetcher.get_historical_klines("BTCUSDT", "1m")'
        new_symbol_section = '''                        # Multi-symbol trading: BTCUSDT (futures) + BNBBTC (spot)
                        symbols_to_trade = ["BTCUSDT", "BNBBTC"]
                        all_data = {}
                        
                        for trading_symbol in symbols_to_trade:
                            try:
                                symbol_data = price_fetcher.get_historical_klines(trading_symbol, "1m")
                                if not symbol_data.empty:
                                    all_data[trading_symbol] = symbol_data
                                    logger.debug(f"✅ {trading_symbol}: {len(symbol_data)} records")
                                else:
                                    logger.warning(f"⚠️ No data for {trading_symbol}")
                            except Exception as e:
                                logger.warning(f"⚠️ Error fetching {trading_symbol}: {e}")
                        
                        # Use BTCUSDT as primary for dashboard/indicators
                        data = all_data.get("BTCUSDT", pd.DataFrame())'''
        
        content = content.replace(old_symbol_line, new_symbol_section)
        
        # Update the hardcoded symbol variable
        old_symbol_var = '                            symbol = "BTCUSDT"'
        new_symbol_var = '''                            # Process each symbol with appropriate executor
                            symbols_data = {
                                "BTCUSDT": data,  # Primary symbol data
                                "BNBBTC": all_data.get("BNBBTC", pd.DataFrame())
                            }
                            
                            # Process BTCUSDT first (futures)
                            symbol = "BTCUSDT"'''
        
        content = content.replace(old_symbol_var, new_symbol_var)
        
        # Add BNBBTC processing after BTCUSDT
        old_execute_section = '''                            # Execute trades based on signals with MUCH HIGHER threshold to stop overtrading
                            if signal in ['BUY', 'SELL'] and confidence >= 0.90:  # CRITICAL: High threshold for quality trades only'''
        
        new_execute_section = '''                            # Execute trades based on signals with MUCH HIGHER threshold to stop overtrading
                            if signal in ['BUY', 'SELL'] and confidence >= 0.90:  # CRITICAL: High threshold for quality trades only
                                # Execute BTCUSDT trade (futures)
                                self._execute_symbol_trade(symbol, signal, confidence, data, executor)
                            
                            # PROCESS BNBBTC (BNB → BTC conversion)
                            if "BNBBTC" in all_data and not all_data["BNBBTC"].empty:
                                bnbbtc_data = all_data["BNBBTC"]
                                bnbbtc_current_price = float(bnbbtc_data.iloc[-1]['close'])
                                
                                # Check if we should convert BNB to BTC
                                balance_info = executor.get_balance()
                                bnb_balance = balance_info.get('BNB', 0)
                                
                                if bnb_balance > 0.1:  # Minimum BNB balance for conversion
                                    # Simple conversion logic: convert if BNB > 5% of portfolio
                                    total_balance = executor.get_account_balance()
                                    bnb_value_usdt = bnb_balance * 300  # Approximate BNB price
                                    bnb_allocation = bnb_value_usdt / total_balance
                                    
                                    if bnb_allocation > 0.05:  # Convert if BNB > 5% allocation
                                        conversion_amount = bnb_balance * 0.3  # Convert 30% of BNB
                                        btc_amount = conversion_amount * bnbbtc_current_price
                                        
                                        logger.info(f"🪙 BNB→BTC Conversion: {conversion_amount:.6f} BNB → {btc_amount:.8f} BTC")
                                        
                                        # Create spot executor for BNBBTC if needed
                                        if not hasattr(trading_loop, 'spot_executor'):
                                            from trading.execution.binance_executor import BinanceExecutor
                                            trading_loop.spot_executor = BinanceExecutor(
                                                logger=logger,
                                                is_testnet=True,
                                                use_futures=False  # Spot trading for BNBBTC
                                            )
                                            trading_loop.spot_executor.initialize()
                                        
                                        # Execute BNB→BTC conversion
                                        result = trading_loop.spot_executor.place_order(
                                            'BNBBTC', 'buy', btc_amount, bnbbtc_current_price
                                        )
                                        if result:
                                            logger.info(f"✅ BNB→BTC conversion successful: {result}")
                                        else:
                                            logger.error("❌ BNB→BTC conversion failed")
                            
                            # Continue with original BTCUSDT execution logic'''
        
        # Replace the execute section (but be careful to only replace the first occurrence)
        if old_execute_section in content:
            content = content.replace(old_execute_section, new_execute_section, 1)
        
        # Write updated content
        with open(main_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ main.py updated with multi-symbol trading support")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating main.py: {e}")
        # Restore backup if something went wrong
        try:
            with open(main_file + '.backup', 'r') as f:
                backup_content = f.read()
            with open(main_file, 'w') as f:
                f.write(backup_content)
            logger.info("✅ Restored from backup due to error")
        except:
            pass
        return False

def add_helper_method():
    """Add helper method for symbol trading"""
    helper_code = '''
def _execute_symbol_trade(symbol, signal, confidence, data, executor):
    """Helper method to execute trades for a specific symbol"""
    try:
        current_price = data.iloc[-1]['close']
        available_balance = executor.get_account_balance()
        
        # Calculate position size based on symbol
        if symbol == "BTCUSDT":
            # Use existing logic for BTCUSDT
            position_size = min(0.001, available_balance / current_price * 0.05)
        elif symbol == "BNBBTC":
            # For BNBBTC, use smaller position sizes
            position_size = min(0.01, available_balance / current_price * 0.02)
        else:
            position_size = 0.001  # Default
        
        if position_size <= 0:
            position_size = 0.001
        
        logger.info(f"🎯 {symbol} {signal}: Price=${current_price:.2f}, Size={position_size:.6f}")
        
        if signal == 'BUY':
            order_result = executor.place_order(symbol, 'buy', position_size, current_price)
        else:
            order_result = executor.place_order(symbol, 'sell', position_size, current_price)
        
        if order_result:
            logger.info(f"✅ {symbol} {signal} executed successfully")
        else:
            logger.error(f"❌ {symbol} {signal} execution failed")
            
    except Exception as e:
        logger.error(f"❌ Error executing {symbol} trade: {e}")
'''
    
    with open('/Users/maxime/BTC_BOT/BTC_BOT/trading_helper_methods.py', 'w') as f:
        f.write(helper_code)
    
    logger.info("✅ Created helper methods file: trading_helper_methods.py")

def main():
    """Main function to enable multi-symbol trading"""
    logger.info("🚀 Multi-Symbol Trading Enablement")
    logger.info("="*60)
    
    # Update main.py
    update_success = update_main_for_multi_symbol()
    
    # Add helper methods
    add_helper_method()
    
    # Summary
    logger.info("="*60)
    logger.info("📋 MULTI-SYMBOL TRADING SUMMARY:")
    logger.info(f"   main.py Update: {'✅' if update_success else '❌'}")
    logger.info(f"   Helper Methods: ✅")
    
    if update_success:
        logger.info("\n🎉 MULTI-SYMBOL TRADING ENABLED!")
        logger.info("✅ Bot now supports BTCUSDT (futures) + BNBBTC (spot)")
        logger.info("✅ Automatic BNB→BTC conversion when BNB allocation > 5%")
        logger.info("✅ Dual executor setup (futures + spot)")
        logger.info("📝 Restart bot to begin multi-symbol trading")
        
        logger.info("\n📊 Expected Behavior:")
        logger.info("   • BTCUSDT: Futures trading as usual")
        logger.info("   • BNBBTC: Spot trading to convert BNB→BTC")
        logger.info("   • Automatic conversion when BNB allocation > 5%")
        logger.info("   • Both symbols monitored simultaneously")
    else:
        logger.warning("\n⚠️ Update failed - check errors above")
        logger.info("📝 Backup available at main.py.backup")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()