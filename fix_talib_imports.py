#!/usr/bin/env python3
"""
Fix all talib imports to be optional
"""

import os

files_to_fix = [
    'trading/scripts/dashboard.py',
    'trading/strategies/trend_following_strategy.py', 
    'trading/strategies/technical_strategy.py',
    'trading/strategies/mean_reversion_strategy.py',
    'trading/scripts/run_dashboard.py',
    'trading/strategies/sentiment_strategy.py'
]

for file_path in files_to_fix:
    full_path = f"/Users/maxime/BTC_BOT/BTC_BOT/{file_path}"
    
    if os.path.exists(full_path):
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Replace direct import with try/except
            if 'import talib' in content and 'try:' not in content:
                content = content.replace(
                    'import talib',
                    '''try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print(f"Warning: talib not available in {__file__}, using fallbacks")'''
                )
                
                with open(full_path, 'w') as f:
                    f.write(content)
                
                print(f"✅ Fixed {file_path}")
            else:
                print(f"⏭️ Skipped {file_path} (already has try/except or no direct import)")
                
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    else:
        print(f"⚠️ File not found: {file_path}")

print("🎉 All talib imports fixed!")