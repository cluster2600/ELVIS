#!/usr/bin/env python3
"""
Check available BNB trading pairs on Binance
"""

import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_binance_symbols():
    """Check available BNB trading pairs on Binance"""
    logger.info("🔍 Checking Binance Trading Pairs")
    logger.info("=" * 50)
    
    try:
        # Get exchange info from Binance API
        logger.info("1. Fetching exchange info from Binance...")
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo")
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch exchange info: {response.status_code}")
            return
        
        data = response.json()
        symbols = data.get('symbols', [])
        
        logger.info(f"2. Found {len(symbols)} total trading pairs")
        
        # Filter BNB-related pairs
        logger.info("3. Searching for BNB trading pairs...")
        bnb_pairs = []
        
        for symbol_info in symbols:
            symbol = symbol_info['symbol']
            base_asset = symbol_info['baseAsset']
            quote_asset = symbol_info['quoteAsset']
            status = symbol_info['status']
            
            # Check if BNB is involved
            if 'BNB' in symbol and status == 'TRADING':
                bnb_pairs.append({
                    'symbol': symbol,
                    'base': base_asset,
                    'quote': quote_asset,
                    'status': status
                })
        
        logger.info(f"4. Found {len(bnb_pairs)} active BNB trading pairs:")
        
        # Group by quote asset
        by_quote = {}
        for pair in bnb_pairs:
            quote = pair['quote']
            if quote not in by_quote:
                by_quote[quote] = []
            by_quote[quote].append(pair)
        
        # Display grouped results
        for quote_asset in sorted(by_quote.keys()):
            pairs = by_quote[quote_asset]
            logger.info(f"\n   {quote_asset} pairs:")
            for pair in pairs:
                logger.info(f"     {pair['symbol']} = {pair['base']}/{pair['quote']}")
        
        # Specifically check for BTC pairs
        logger.info("\n🔍 BTC-related BNB pairs:")
        btc_pairs = [p for p in bnb_pairs if 'BTC' in p['symbol']]
        
        if btc_pairs:
            for pair in btc_pairs:
                logger.info(f"   ✅ {pair['symbol']} = {pair['base']}/{pair['quote']}")
        else:
            logger.warning("   ❌ No direct BNB/BTC pairs found")
        
        # Check the specific symbols we're interested in
        logger.info("\n🎯 Checking specific symbols:")
        test_symbols = ['BNBBTC', 'BTCBNB', 'BNBUSDT', 'BTCUSDT']
        
        for test_symbol in test_symbols:
            found = any(p['symbol'] == test_symbol for p in bnb_pairs + 
                       [{'symbol': s['symbol']} for s in symbols if s['symbol'] == test_symbol])
            
            if found:
                symbol_info = next((s for s in symbols if s['symbol'] == test_symbol), None)
                if symbol_info:
                    logger.info(f"   ✅ {test_symbol} = {symbol_info['baseAsset']}/{symbol_info['quoteAsset']} (status: {symbol_info['status']})")
                else:
                    logger.info(f"   ✅ {test_symbol} exists")
            else:
                logger.warning(f"   ❌ {test_symbol} not found")
        
        return bnb_pairs
        
    except Exception as e:
        logger.error(f"Error checking Binance symbols: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def check_symbol_details(symbol):
    """Check detailed info for a specific symbol"""
    logger.info(f"\n🔬 Detailed check for {symbol}:")
    
    try:
        # Try to get 24hr ticker
        response = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"   ✅ {symbol} is active")
            logger.info(f"   Price: {data['lastPrice']}")
            logger.info(f"   Volume: {data['volume']}")
            logger.info(f"   24h Change: {data['priceChangePercent']}%")
            return True
        else:
            logger.warning(f"   ❌ {symbol} error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ Error checking {symbol}: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 BNB Symbol Checker")
    logger.info("=" * 60)
    
    # Check all BNB pairs
    bnb_pairs = check_binance_symbols()
    
    # Test specific symbols
    test_symbols = ['BNBBTC', 'BTCBNB', 'BNBUSDT']
    for symbol in test_symbols:
        check_symbol_details(symbol)
    
    logger.info("\n" + "=" * 60)
    logger.info("📋 RECOMMENDATIONS:")
    
    # Based on results, provide recommendations
    if any(p['symbol'] == 'BNBUSDT' for p in bnb_pairs):
        logger.info("✅ Use BNBUSDT for BNB trading (most liquid)")
    
    if any(p['symbol'] == 'BNBBTC' for p in bnb_pairs):
        logger.info("✅ BNBBTC exists - check if it's active")
    else:
        logger.info("❌ BNBBTC not available - use BNBUSDT + BTCUSDT conversion")
    
    logger.info("\n💡 Alternative approach:")
    logger.info("   1. Trade BNB → USDT (using BNBUSDT)")
    logger.info("   2. Trade USDT → BTC (using BTCUSDT)")
    logger.info("   This achieves BNB → BTC conversion indirectly")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()