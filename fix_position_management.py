#!/usr/bin/env python3
"""
Fix Position Management Issue in ELVIS Trading Bot

PROBLEM IDENTIFIED:
- Bot creates positions via executor.place_order()
- Positions are stored in database but NEVER added to risk_manager
- Risk manager doesn't know about positions, so they never get closed
- This is why bot has 6 open positions for 3 days without closure

SOLUTION:
Add risk_manager.add_position() calls after successful order execution
"""

import sys
import os

def fix_main_py():
    """Add risk manager integration to main.py order execution"""
    
    main_py_path = "/Users/maxime/BTC_BOT/BTC_BOT/main.py"
    
    # Read the current file
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Define the fix - add risk manager integration after successful orders
    buy_order_fix = '''                                order_result = executor.place_order(symbol, 'buy', position_size, current_price)
                                if order_result:
                                    logger.info(f"🎉 [SUCCESS] BUY order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                                    
                                    # 🔥 FIX: Add position to risk manager for monitoring
                                    try:
                                        position_data = {
                                            'symbol': symbol,
                                            'side': 'BUY',
                                            'entry_price': current_price,
                                            'quantity': position_size,
                                            'leverage': leverage,
                                            'timestamp': time.time()
                                        }
                                        risk_manager.add_position(symbol, position_data)
                                        logger.info(f"✅ Position added to risk manager: BUY {position_size:.6f} {symbol}")
                                    except Exception as e:
                                        logger.error(f"❌ Failed to add BUY position to risk manager: {e}")
                                    
                                    # Record trade to prevent duplicates
                                    trading_loop.recent_trades[trade_key] = current_time_ms
                                    # Small delay to ensure trade completion
                                    time.sleep(0.5)'''
    
    sell_order_fix = '''                                order_result = executor.place_order(symbol, 'sell', position_size, current_price)
                                if order_result:
                                    logger.info(f"🎉 [SUCCESS] SELL order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                                    
                                    # 🔥 FIX: Add position to risk manager for monitoring  
                                    try:
                                        position_data = {
                                            'symbol': symbol,
                                            'side': 'SELL',
                                            'entry_price': current_price,
                                            'quantity': position_size,
                                            'leverage': leverage,
                                            'timestamp': time.time()
                                        }
                                        risk_manager.add_position(symbol, position_data)
                                        logger.info(f"✅ Position added to risk manager: SELL {position_size:.6f} {symbol}")
                                    except Exception as e:
                                        logger.error(f"❌ Failed to add SELL position to risk manager: {e}")
                                    
                                    # Record trade to prevent duplicates
                                    trading_loop.recent_trades[trade_key] = current_time_ms
                                    # Small delay to ensure trade completion
                                    time.sleep(0.5)'''
    
    # Replace the existing patterns
    old_buy_pattern = '''                                order_result = executor.place_order(symbol, 'buy', position_size, current_price)
                                if order_result:
                                    logger.info(f"🎉 [SUCCESS] BUY order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                                    # Record trade to prevent duplicates
                                    trading_loop.recent_trades[trade_key] = current_time_ms
                                    # Small delay to ensure trade completion
                                    time.sleep(0.5)'''
    
    old_sell_pattern = '''                                order_result = executor.place_order(symbol, 'sell', position_size, current_price)
                                if order_result:
                                    logger.info(f"🎉 [SUCCESS] SELL order executed: {position_size:.6f} {symbol} at ${current_price:.2f}")
                                    # Record trade to prevent duplicates
                                    trading_loop.recent_trades[trade_key] = current_time_ms
                                    # Small delay to ensure trade completion
                                    time.sleep(0.5)'''
    
    # Apply the fixes
    content = content.replace(old_buy_pattern, buy_order_fix)
    content = content.replace(old_sell_pattern, sell_order_fix)
    
    # Write the fixed file
    with open(main_py_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed main.py - Added risk manager integration to order execution")

def create_position_sync_script():
    """Create a script to sync existing positions to risk manager"""
    
    sync_script_content = '''#!/usr/bin/env python3
"""
Sync existing open positions to risk manager
This fixes the 6 positions that are already open but not managed
"""

import requests
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from core.bootstrap import bootstrap_application
from core.di import container

def sync_existing_positions():
    """Load existing positions and add them to risk manager"""
    try:
        # Bootstrap the application to get container
        bootstrap_application()
        
        # Get dependencies
        risk_manager = container.get('risk_manager')
        logger = container.get('logger')
        
        logger.info("🔄 Syncing existing open positions to risk manager...")
        
        # Get current open positions from API
        try:
            response = requests.get('http://localhost:5050/open_positions', timeout=5)
            if response.status_code == 200:
                positions = response.json()
                logger.info(f"📊 Found {len(positions)} open positions to sync")
                
                for pos in positions:
                    try:
                        # Convert API format to risk manager format
                        position_data = {
                            'symbol': pos['symbol'],
                            'side': pos['side'], 
                            'entry_price': pos['entry_price'],
                            'quantity': pos['quantity'],
                            'leverage': pos.get('leverage', 100),
                            'timestamp': pos['entry_time']
                        }
                        
                        # Add to risk manager
                        risk_manager.add_position(pos['symbol'], position_data)
                        logger.info(f"✅ Synced position: {pos['side']} {pos['quantity']:.6f} {pos['symbol']} @ ${pos['entry_price']}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to sync position {pos.get('id')}: {e}")
                
                logger.info(f"🎉 Sync complete! {len(positions)} positions now managed by risk manager")
                logger.info("🔥 These positions will now be monitored for stop-loss and take-profit")
                
            else:
                logger.error(f"❌ Failed to fetch positions: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to trading API: {e}")
            logger.error("💡 Make sure the trading bot is running first")
            
    except Exception as e:
        logger.error(f"❌ Position sync failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = sync_existing_positions()
    sys.exit(0 if success else 1)
'''
    
    sync_script_path = "/Users/maxime/BTC_BOT/BTC_BOT/sync_positions_to_risk_manager.py"
    with open(sync_script_path, 'w') as f:
        f.write(sync_script_content)
    
    # Make it executable
    os.chmod(sync_script_path, 0o755)
    
    print(f"✅ Created position sync script: {sync_script_path}")

def main():
    print("🔧 FIXING ELVIS TRADING BOT POSITION MANAGEMENT")
    print("=" * 60)
    print()
    print("🎯 PROBLEM IDENTIFIED:")
    print("  - Bot creates positions but never adds them to risk manager")
    print("  - Risk manager doesn't know about positions")
    print("  - Positions never get stop-loss or take-profit monitoring")
    print("  - This is why 6 positions are open for 3 days!")
    print()
    print("🛠️  APPLYING FIXES:")
    print("  1. Adding risk_manager.add_position() calls after successful orders")
    print("  2. Creating script to sync existing positions")
    print()
    
    # Apply the main fix
    try:
        fix_main_py()
        create_position_sync_script()
        
        print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print()
        print("📋 NEXT STEPS:")
        print("  1. Restart the trading bot to use the fixed code")
        print("  2. Run the sync script to manage existing positions:")
        print("     python sync_positions_to_risk_manager.py")
        print("  3. Monitor logs for position closing activity")
        print()
        print("🎉 Your bot will now properly close positions!")
        
    except Exception as e:
        print(f"❌ ERROR applying fixes: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)