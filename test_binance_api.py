from binance.um_futures import UMFutures
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('TESTNET_FUTURES_API')
api_secret = os.getenv('TESTNET_FUTURES_SECRET')

print(f"Using API Key: {api_key[:5]}...")
print(f"Using API Secret: {api_secret[:5]}...")

client = UMFutures(
    key=api_key,
    secret=api_secret,
    base_url='https://testnet.binancefuture.com'
)

try:
    server_time = client.time()
    print(f"Server Time: {server_time}")
    account = client.account()
    print(f"Account Info: {account}")
    price = client.mark_price('BTCUSDT')
    print(f"BTCUSDT Mark Price: {price}")
except Exception as e:
    print(f"Error: {e}")