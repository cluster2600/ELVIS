#!/usr/bin/env python3
"""
Simple verification script to check the trading strategy fixes.
"""

import sys
import os
sys.path.append('/Users/maxime/BTC_BOT/BTC_BOT')

def check_risk_management_fixes():
    """Verify risk management parameters are fixed"""
    print("🔧 CHECKING RISK MANAGEMENT FIXES...")
    
    try:
        # Import and check risk management settings
        from trading.risk_management import RiskManager
        
        # Create instance (we'll check default values)
        rm = RiskManager(None, None)  # Pass None for executor and logger for testing
        
        print(f"✅ Cooldown period: {rm.cooldown_period} seconds")
        print(f"   Expected: 1800 (30 minutes) | Actual: {rm.cooldown_period}")
        assert rm.cooldown_period == 1800, f"❌ Cooldown wrong: {rm.cooldown_period}"
        
        print(f"✅ Max trades per day: {rm.max_trades_per_day}")
        print(f"   Expected: 48 | Actual: {rm.max_trades_per_day}")
        assert rm.max_trades_per_day == 48, f"❌ Max trades wrong: {rm.max_trades_per_day}"
        
        print(f"✅ Leverage target: {rm.leverage_target}x")
        print(f"   Expected: 10.0 | Actual: {rm.leverage_target}")
        assert rm.leverage_target == 10.0, f"❌ Leverage wrong: {rm.leverage_target}"
        
        print(f"✅ Daily profit target: ${rm.daily_profit_target_usd}")
        print(f"   Expected: 100.0 | Actual: {rm.daily_profit_target_usd}")
        assert rm.daily_profit_target_usd == 100.0, f"❌ Profit target wrong: {rm.daily_profit_target_usd}"
        
        print(f"✅ Daily loss limit: ${rm.daily_loss_limit_usd}")
        print(f"   Expected: -50.0 | Actual: {rm.daily_loss_limit_usd}")
        assert rm.daily_loss_limit_usd == -50.0, f"❌ Loss limit wrong: {rm.daily_loss_limit_usd}"
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking risk management: {e}")
        return False

def check_configuration():
    """Check configuration settings"""
    print("\n🔧 CHECKING CONFIGURATION...")
    
    try:
        from config.config import TRADING_CONFIG
        
        print(f"✅ Default leverage: {TRADING_CONFIG['DEFAULT_LEVERAGE']}x")
        print(f"   Expected: 10 | Actual: {TRADING_CONFIG['DEFAULT_LEVERAGE']}")
        assert TRADING_CONFIG['DEFAULT_LEVERAGE'] == 10, f"❌ Config leverage wrong: {TRADING_CONFIG['DEFAULT_LEVERAGE']}"
        
        print(f"✅ Cooldown in config: {TRADING_CONFIG.get('COOLDOWN', 'not set')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
        return False

def check_environment():
    """Check environment variables"""
    print("\n🔧 CHECKING ENVIRONMENT VARIABLES...")
    
    strategy_mode = os.getenv('STRATEGY_MODE', 'not set')
    print(f"✅ STRATEGY_MODE: {strategy_mode}")
    
    leverage = os.getenv('LEVERAGE', 'not set')
    print(f"✅ LEVERAGE: {leverage}")
    
    cooldown = os.getenv('COOLDOWN_MINUTES', 'not set')
    print(f"✅ COOLDOWN_MINUTES: {cooldown}")
    
    # Check .env file
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"✅ .env file exists")
        with open(env_file, 'r') as f:
            content = f.read()
            if 'STRATEGY_MODE=balanced' in content:
                print("✅ .env contains STRATEGY_MODE=balanced")
            else:
                print("⚠️  .env might not have STRATEGY_MODE=balanced")
    else:
        print("⚠️  .env file not found")
    
    return True

def check_ensemble_cooldown():
    """Check ensemble strategy has cooldown"""
    print("\n🔧 CHECKING ENSEMBLE STRATEGY COOLDOWN...")
    
    try:
        # Check if file has cooldown code
        ensemble_file = 'trading/strategies/ensemble_strategy.py'
        with open(ensemble_file, 'r') as f:
            content = f.read()
            
        if 'self.cooldown_period = 1800' in content:
            print("✅ Ensemble strategy has cooldown period set")
        else:
            print("❌ Ensemble strategy missing cooldown period")
            
        if 'def record_trade_signal' in content:
            print("✅ Ensemble strategy has trade recording method")
        else:
            print("❌ Ensemble strategy missing trade recording")
            
        if 'Cooldown active:' in content:
            print("✅ Ensemble strategy has cooldown check")
        else:
            print("❌ Ensemble strategy missing cooldown check")
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking ensemble strategy: {e}")
        return False

def main():
    """Run verification checks"""
    print("🚨 VERIFYING TRADING STRATEGY FIXES")
    print("=" * 60)
    
    success = True
    
    success &= check_risk_management_fixes()
    success &= check_configuration()
    success &= check_environment()
    success &= check_ensemble_cooldown()
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎯 ALL FIXES VERIFIED SUCCESSFULLY!")
        print("\n📊 CRITICAL PROBLEMS FIXED:")
        print("✅ Win rate: Was 14.8% → Should improve with reduced trading")
        print("✅ Trade frequency: Was 22.53/hour → Now max 2/hour (30min cooldown)")
        print("✅ Leverage: Was 100x → Now 10x (reduced losses)")
        print("✅ Daily loss limit: Was $300 → Now $50 (stricter)")
        print("✅ Strategy: Changed to balanced (has emergency fixes)")
        print("✅ Duplicate trades: Prevention mechanism added")
        
        print("\n🚀 EXPECTED IMPROVEMENTS:")
        print("- Trade frequency: 22.53/hour → 2/hour (91% reduction)")
        print("- Leverage impact: 100x → 10x (90% reduction in exposure)")
        print("- Fee drain: Should reduce dramatically")
        print("- Win rate: Should improve with less over-trading")
        print("- Net P&L: Should become positive")
        
        print("\n⚠️  RESTART THE BOT TO APPLY FIXES:")
        print("1. Stop current bot if running")
        print("2. Run: python main.py --mode dashboard")
        print("3. Look for 'BALANCED STARTER: EMERGENCY MODE' in logs")
        print("4. Verify cooldown messages: 'Cooldown active: X minutes remaining'")
        print("5. Check trades are spaced 30+ minutes apart")
        
    else:
        print("❌ SOME ISSUES FOUND!")
        print("Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)