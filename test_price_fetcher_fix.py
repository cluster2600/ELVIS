#!/usr/bin/env python3
"""
Test and fix the price fetcher to return proper market data
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

# Set environment for testing
os.environ['VAULT_ENABLED'] = 'false'
os.environ['VAULT_ADDR'] = 'http://127.0.0.1:8200'

def test_price_fetcher_methods():
    """Test different price fetcher methods to find working ones"""
    try:
        from utils.price_fetcher import PriceFetcher
        
        logger.info("🔧 Testing price fetcher methods...")
        
        # Initialize price fetcher
        price_fetcher = PriceFetcher(logger=logger, symbols=['BTCUSDT'])
        
        # Test Method 1: get_historical_klines (should work)
        logger.info("Testing get_historical_klines method...")
        try:
            df = price_fetcher.get_historical_klines('BTCUSDT', '5m', 10)
            if df is not None and not df.empty:
                logger.info(f"✅ get_historical_klines: Got {len(df)} candles")
                logger.info(f"Latest BTC price: ${df.iloc[-1]['close']:.2f}")
                logger.info(f"DataFrame columns: {list(df.columns)}")
                return True, df
            else:
                logger.warning("⚠️ get_historical_klines returned empty DataFrame")
        except Exception as e:
            logger.error(f"❌ get_historical_klines failed: {e}")
        
        # Test Method 2: get_candle_history (after calling get_historical_data)
        logger.info("Testing get_historical_data + get_candle_history...")
        try:
            price_fetcher.get_historical_data()  # Populate internal cache
            candles = price_fetcher.get_candle_history('BTCUSDT')
            if candles and len(candles) > 0:
                logger.info(f"✅ get_candle_history: Got {len(candles)} candles")
                # Parse the last candle (should be list format)
                last_candle = candles[-1]
                if len(last_candle) >= 5:
                    latest_price = float(last_candle[4])  # Close price
                    logger.info(f"Latest BTC price: ${latest_price:.2f}")
                return True, candles
            else:
                logger.warning("⚠️ get_candle_history returned empty data")
        except Exception as e:
            logger.error(f"❌ get_candle_history failed: {e}")
        
        # Test Method 3: Direct client access
        logger.info("Testing direct client access...")
        try:
            if hasattr(price_fetcher, 'client') and price_fetcher.client:
                klines = price_fetcher.client.klines(symbol='BTCUSDT', interval='5m', limit=10)
                if klines and len(klines) > 0:
                    logger.info(f"✅ Direct client: Got {len(klines)} candles")
                    latest_price = float(klines[-1][4])  # Close price
                    logger.info(f"Latest BTC price: ${latest_price:.2f}")
                    return True, klines
                else:
                    logger.warning("⚠️ Direct client returned empty data")
        except Exception as e:
            logger.error(f"❌ Direct client failed: {e}")
        
        return False, None
        
    except Exception as e:
        logger.error(f"❌ Price fetcher initialization failed: {e}")
        return False, None

def create_improved_price_fetcher():
    """Create an improved price fetcher with better data access"""
    logger.info("🔧 Creating improved price fetcher...")
    
    # Read the original file
    with open('/Users/maxime/BTC_BOT/BTC_BOT/utils/price_fetcher.py', 'r') as f:
        content = f.read()
    
    # Add a new method to return data properly
    new_method = '''
    def get_market_data(self, symbol: str = None, interval: str = None, limit: int = 200):
        """
        Get market data for a symbol with proper DataFrame format.
        This is a unified method that works reliably.
        """
        symbol = symbol or (self.symbols[0] if self.symbols else 'BTCUSDT')
        interval = interval or self.timeframe
        
        try:
            # Try the reliable get_historical_klines method first
            df = self.get_historical_klines(symbol, interval, limit)
            if df is not None and not df.empty:
                self.logger.info(f"Retrieved {len(df)} candles for {symbol}")
                return df
            
            # Fallback: try to populate internal cache and convert to DataFrame
            self.get_historical_data()
            candles = self.get_candle_history(symbol)
            
            if candles and len(candles) > 0:
                # Convert raw candle data to DataFrame
                import pandas as pd
                df = pd.DataFrame(candles, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert to proper data types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Convert timestamps
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                
                self.logger.info(f"Converted {len(df)} candles to DataFrame for {symbol}")
                return df
            
            # Last resort: return empty DataFrame with proper structure
            self.logger.warning(f"No data available for {symbol}, returning empty DataFrame")
            import pandas as pd
            return pd.DataFrame(columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            import pandas as pd
            return pd.DataFrame(columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
'''
    
    # Add the new method before the _cache_get method
    insert_position = content.find('    def _cache_get(self, key):')
    if insert_position != -1:
        new_content = content[:insert_position] + new_method + '\n' + content[insert_position:]
        
        # Write the improved file
        with open('/Users/maxime/BTC_BOT/BTC_BOT/utils/price_fetcher.py', 'w') as f:
            f.write(new_content)
        
        logger.info("✅ Added get_market_data method to price fetcher")
        return True
    else:
        logger.error("❌ Could not find insertion point in price fetcher")
        return False

def test_improved_price_fetcher():
    """Test the improved price fetcher"""
    try:
        # Reload the module to get the new method
        import importlib
        import utils.price_fetcher
        importlib.reload(utils.price_fetcher)
        
        from utils.price_fetcher import PriceFetcher
        
        logger.info("🔧 Testing improved price fetcher...")
        
        # Initialize price fetcher
        price_fetcher = PriceFetcher(logger=logger, symbols=['BTCUSDT'])
        
        # Test the new get_market_data method
        df = price_fetcher.get_market_data('BTCUSDT', '5m', 10)
        
        if df is not None and not df.empty:
            logger.info(f"✅ get_market_data: Got {len(df)} candles")
            logger.info(f"Latest BTC price: ${df.iloc[-1]['close']:.2f}")
            logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            return True
        else:
            logger.warning("⚠️ get_market_data returned empty DataFrame")
            return False
            
    except Exception as e:
        logger.error(f"❌ Testing improved price fetcher failed: {e}")
        return False

def main():
    """Main function to test and fix price fetcher"""
    logger.info("🔧 Starting price fetcher diagnostic and fix...")
    
    # Test current methods
    works, data = test_price_fetcher_methods()
    
    if works:
        logger.info("✅ Price fetcher is already working with existing methods")
    else:
        logger.info("🔧 Adding improved method to price fetcher...")
        
        # Add improved method
        if create_improved_price_fetcher():
            # Test the improved version
            if test_improved_price_fetcher():
                logger.info("✅ Price fetcher fixed and working!")
            else:
                logger.error("❌ Improved price fetcher still not working")
        else:
            logger.error("❌ Could not add improved method")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()