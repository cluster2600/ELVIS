#!/usr/bin/env python3
"""
Simple script to test Binance API connection with futures testnet keys.
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_futures_connection():
    """Test futures API connection"""
    try:
        from binance.um_futures import UMFutures

        api_key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET")

        if not api_key or not api_secret:
            print("❌ Futures testnet API keys not found in .env file")
            return False

        print(f"🔑 Testing futures testnet connection...")
        print(f"API Key: {api_key[:8]}...{api_key[-8:]}")

        # Initialize futures client
        client = UMFutures(
            key=api_key, secret=api_secret, base_url="https://testnet.binancefuture.com"
        )

        # Test connection
        account_info = client.account()
        print("✅ Futures testnet connection successful!")
        print(f"📊 Account info: {len(account_info.get('assets', []))} assets found")

        # Test getting price
        ticker = client.ticker_price(symbol="BTCUSDT")
        print(f"💰 Current BTC price: ${float(ticker['price']):,.2f}")

        return True

    except Exception as e:
        print(f"❌ Futures connection failed: {e}")
        return False


def test_spot_connection():
    """Test spot API connection as fallback"""
    try:
        from binance.client import Client

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            print("❌ Spot API keys not found in .env file")
            return False

        print(f"🔑 Testing spot connection...")
        print(f"API Key: {api_key[:8]}...{api_key[-8:]}")

        # Initialize spot client
        client = Client(api_key, api_secret, testnet=True)

        # Test connection
        account_info = client.get_account()
        print("✅ Spot testnet connection successful!")
        print(
            f"📊 Account info: {len(account_info.get('balances', []))} balances found"
        )

        # Test getting price
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        print(f"💰 Current BTC price: ${float(ticker['price']):,.2f}")

        return True

    except Exception as e:
        print(f"❌ Spot connection failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing Binance API connections...\n")

    futures_ok = test_futures_connection()
    print()
    spot_ok = test_spot_connection()

    print("\n" + "=" * 50)
    if futures_ok:
        print("✅ Futures testnet API working - ready for trading!")
    elif spot_ok:
        print("✅ Spot testnet API working - fallback available!")
    else:
        print("❌ No working API connections found")
        print("Please check your API keys in the .env file")
    print("=" * 50)
